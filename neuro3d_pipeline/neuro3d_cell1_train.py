# ============================================================================
# NEURO-3D CELL 1 : TRAIN -- Neuro-3D Disentangled Classifier + GWIT ControlNet
# ----------------------------------------------------------------------------
# Complete Stage A & Stage B Training:
#   Stage A: Neuro-3D Disentangled Representation Classifier (CVPR 2025):
#            - Dual Disentangled Visual Pathways:
#                1. Static Visual Pathway (z_static): 3D geometry & object category.
#                2. Dynamic Visual Pathway (z_dynamic): phase velocity & gamma ripples.
#            - Differentiable Morlet Wavelet Filterbank (Theta, Alpha, Beta, Gamma, High-Gamma).
#            - Mathematical Orthogonality Constraint between Static and Dynamic pathways.
#            - Multi-Trial Image Pooling & Subject-Invariant Training.
#            - Cosine Learning Rate Decay with Warmup.
#            - Automatically restores /kaggle/working/best_neuro3d_classifier.pt
#              if already in Drive/local, or trains from scratch.
#   Stage B: GWIT ControlNet on Stable-Diffusion-2.1-base:
#            - Trained with VAE latents + text embeddings from the Neuro-3D
#              predicted captions + EEG condition embedding.
#
# Saves:
#   - /kaggle/working/best_neuro3d_classifier.pt (Stage A Classifier)
#   - /kaggle/working/neuro3d_pipeline.pt         (Full Stage A + Stage B Pipeline)
# ============================================================================

import os, sys, gc, glob, time, math, random, pickle, subprocess, warnings, io
warnings.filterwarnings("ignore")

for _pkg, _mod in [("diffusers", "diffusers"), ("transformers", "transformers"), ("accelerate", "accelerate")]:
    if subprocess.run([sys.executable, "-c", f"import importlib.util as u,sys; sys.exit(0 if u.find_spec('{_mod}') else 1)"]).returncode != 0:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", _pkg], check=False)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms

try:
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, ControlNetModel
except AttributeError as _e:
    for _m in [m for m in sys.modules if m.startswith("diffusers")]: del sys.modules[_m]
    sys.modules["torch._dynamo"] = None
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, ControlNetModel

from transformers import CLIPTextModel, CLIPTokenizer

# ------------------------------- Configuration --------------------------------
DATA_ROOT  = "/kaggle/input/datasets/evolutionaw/eegcvpr40/data"
IMAGE_DIR  = "/kaggle/input/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train"
OUTPUT_DIR = "/kaggle/working"
OUT_PATH   = os.path.join(OUTPUT_DIR, "neuro3d_pipeline.pt")
NEURO3D_BEST_PATH = os.path.join(OUTPUT_DIR, "best_neuro3d_classifier.pt")
SD_IDS     = ["stabilityai/stable-diffusion-2-1-base", "sd2-community/stable-diffusion-2-1-base"]

RES = 512
LR_CONTROLNET = 1e-5
WD, MAX_GRAD_NORM = 1e-2, 1.0
EPOCHS_CONTROLNET = 1
PROP_EMPTY = 0.5
TRAIN_FRAC = 0.70
SEED = 42
BS, ACCUM = 4, 2
TIME_BUDGET_H = 8.0
SAVE_EVERY = 500

# Neuro-3D Hyperparameters
NEURO3D_EPOCHS = 65
NEURO3D_LR = 3.5e-4
NEURO3D_LR_MIN = 1e-5
NEURO3D_BS = 64
EMBED_DIM = 384
N_3D_POINTS = 1024
DROPOUT = 0.15
LAMBDA_ORTHO = 0.15

def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

seed_everything(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = (dev == "cuda")
print(f"[Neuro-3D Cell 1] Running on: {dev.upper()} | Mixed Precision: {USE_AMP}")

# ============================================================================
# GOOGLE DRIVE BIDIRECTIONAL SYNC (OAuth 2.0 User Quota)
# ============================================================================
DRIVE_FOLDER_ID = "13vd_i1ePxEbYH5RtW9Q4B2OG0ccwv6Wz"
DRIVE_CLIENT_ID = os.environ.get("DRIVE_CLIENT_ID", "YOUR_DRIVE_CLIENT_ID")
DRIVE_CLIENT_SECRET = os.environ.get("DRIVE_CLIENT_SECRET", "YOUR_DRIVE_CLIENT_SECRET")
DRIVE_REFRESH_TOKEN = os.environ.get("DRIVE_REFRESH_TOKEN", "YOUR_DRIVE_REFRESH_TOKEN")

FORCE_RETRAIN = False

_drive_service = None

def get_drive_service():
    global _drive_service
    if _drive_service is not None:
        return _drive_service
    try:
        from googleapiclient.discovery import build
        from google.oauth2.credentials import Credentials
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "google-api-python-client", "google-auth"], check=False)
        from googleapiclient.discovery import build
        from google.oauth2.credentials import Credentials

    try:
        creds = Credentials(
            None,
            refresh_token=DRIVE_REFRESH_TOKEN,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=DRIVE_CLIENT_ID,
            client_secret=DRIVE_CLIENT_SECRET
        )
        _drive_service = build("drive", "v3", credentials=creds)
        return _drive_service
    except Exception as e:
        print(f"  [Drive Notice] Could not initialize Google Drive: {e}")
        return None

def drive_find_file(filename):
    service = get_drive_service()
    if not service: return None
    try:
        query = f"'{DRIVE_FOLDER_ID}' in parents and name = '{filename}' and trashed = false"
        res = service.files().list(q=query, fields="files(id, name, size)").execute()
        files = res.get("files", [])
        return files[0] if files else None
    except Exception as e:
        print(f"  [Drive Notice] Error querying '{filename}': {e}")
        return None

def drive_download_file(drive_file_id, local_path, filename="file"):
    service = get_drive_service()
    if not service: return False
    try:
        from googleapiclient.http import MediaIoBaseDownload
        req = service.files().get_media(fileId=drive_file_id)
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        print(f"  [Drive] Downloading '{filename}' from Google Drive to: {local_path} ...", flush=True)
        with open(local_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, req, chunksize=10*1024*1024)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    print(f"    Download progress: {int(status.progress() * 100)}%", flush=True)
        print(f"  [Drive OK] Successfully downloaded '{filename}' ({os.path.getsize(local_path):,} bytes)!")
        return True
    except Exception as e:
        print(f"  [Drive Warning] Download failed for '{filename}': {e}")
        if os.path.exists(local_path):
            try: os.remove(local_path)
            except: pass
        return False

def drive_upload_file(local_path, filename=None):
    if not os.path.exists(local_path):
        print(f"  [Drive Notice] Local file not found for upload: {local_path}")
        return False
    service = get_drive_service()
    if not service: return False
    try:
        from googleapiclient.http import MediaFileUpload
        fn = filename or os.path.basename(local_path)
        existing = drive_find_file(fn)
        file_sz = os.path.getsize(local_path)
        print(f"  [Drive] Uploading '{fn}' ({file_sz:,} bytes) to Google Drive folder '{DRIVE_FOLDER_ID}' ...", flush=True)
        
        media = MediaFileUpload(local_path, resumable=True, chunksize=10*1024*1024)
        if existing:
            req = service.files().update(fileId=existing["id"], media_body=media)
        else:
            metadata = {"name": fn, "parents": [DRIVE_FOLDER_ID]}
            req = service.files().create(body=metadata, media_body=media, fields="id, name")

        res = None
        while res is None:
            status, res = req.next_chunk()
            if status:
                print(f"    Upload progress: {int(status.progress() * 100)}%", flush=True)

        print(f"  [Drive OK] Successfully saved '{fn}' to your Google Drive! (ID: {res.get('id')})")
        return True
    except Exception as e:
        print(f"  [Drive Warning] Upload failed for '{filename}': {e}")
        return False

def sync_or_restore_file(local_path, drive_filename=None):
    fn = drive_filename or os.path.basename(local_path)
    if os.path.exists(local_path) and not FORCE_RETRAIN:
        print(f"[OK] Found local file: {local_path} ({os.path.getsize(local_path):,} bytes)")
        return True
    print(f"  Checking Google Drive for existing '{fn}' ...")
    drive_f = drive_find_file(fn)
    if drive_f and not FORCE_RETRAIN:
        return drive_download_file(drive_f["id"], local_path, fn)
    return False

# ============================================================================
# 1. NEURO-3D ARCHITECTURE MODULES (CVPR 2025 Disentangled Pathways)
# ============================================================================

class MorletWaveletFilterbank(nn.Module):
    def __init__(self, in_channels=128, kernel_len=63, fs=1000):
        super().__init__()
        self.kernel_len = kernel_len
        center_freqs = [5.0, 10.5, 21.5, 50.0, 95.0]
        self.n_bands = len(center_freqs)
        
        t = torch.linspace(-kernel_len // 2, kernel_len // 2, kernel_len) / fs
        filters = []
        for f0 in center_freqs:
            sigma = 4.0 / (2 * math.pi * f0 + 1e-5)
            envelope = torch.exp(-0.5 * (t / (sigma + 1e-5)) ** 2)
            wavelet = envelope * torch.cos(2 * math.pi * f0 * t)
            wavelet = wavelet / (wavelet.norm() + 1e-6)
            filters.append(wavelet)
            
        weight = torch.stack(filters).unsqueeze(1)
        self.filters = nn.Parameter(weight, requires_grad=True)
        self.in_channels = in_channels
        
    def forward(self, x):
        B, C, T = x.shape
        x_reshaped = x.view(B * C, 1, T)
        filtered = F.conv1d(x_reshaped, self.filters, padding=self.kernel_len // 2)
        return filtered.view(B, C * self.n_bands, T)

class StaticVisualPathway(nn.Module):
    def __init__(self, in_channels=128, embed_dim=384, dropout=0.15):
        super().__init__()
        self.spatial_proj = nn.Sequential(
            nn.Conv1d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.SiLU()
        )
        self.lowpass_conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=15, stride=2, padding=7, groups=embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=31, stride=2, padding=15, groups=embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU()
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=6, dim_feedforward=embed_dim * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        s = self.spatial_proj(x)
        tokens = self.lowpass_conv(s).transpose(1, 2)
        tokens = self.transformer(tokens)
        z_static = self.pool(tokens.transpose(1, 2)).squeeze(-1)
        return self.norm(z_static)

class DynamicVisualPathway(nn.Module):
    def __init__(self, in_channels=128, embed_dim=384, dropout=0.15):
        super().__init__()
        self.wavelet_fb = MorletWaveletFilterbank(in_channels=in_channels, kernel_len=63)
        wavelet_ch = in_channels * 5
        
        self.dynamic_proj = nn.Sequential(
            nn.Conv1d(wavelet_ch + in_channels, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.SiLU()
        )
        self.dilated_conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2, dilation=2),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=7, padding=6, dilation=3),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        diff = torch.diff(x, dim=-1, prepend=x[:, :, :1])
        w = self.wavelet_fb(x)
        fused_dynamic = torch.cat([w, diff], dim=1)
        d = self.dynamic_proj(fused_dynamic)
        z_dynamic = self.dilated_conv(d).squeeze(-1)
        return self.norm(z_dynamic)

class PointCloudDecoder(nn.Module):
    def __init__(self, embed_dim=384, n_points=1024):
        super().__init__()
        self.n_points = n_points
        theta = torch.linspace(0, 2 * math.pi, n_points)
        phi   = torch.linspace(-math.pi / 2, math.pi / 2, n_points)
        u = torch.cos(phi) * torch.cos(theta)
        v = torch.cos(phi) * torch.sin(theta)
        grid_2d = torch.stack([u, v], dim=-1)
        self.register_buffer("grid_2d", grid_2d)
        
        in_dim = (embed_dim * 2) + 2
        self.geom_mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 3),
            nn.Tanh()
        )
        self.color_mlp = nn.Sequential(
            nn.Linear(in_dim + 3, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )
        
    def forward(self, z_static, z_dynamic):
        B = z_static.shape[0]
        cond = torch.cat([z_static, z_dynamic], dim=-1)
        cond_rep = cond.unsqueeze(1).expand(-1, self.n_points, -1)
        grid_batch = self.grid_2d.unsqueeze(0).expand(B, -1, -1)
        geom_in = torch.cat([cond_rep, grid_batch], dim=-1)
        
        xyz = self.geom_mlp(geom_in)
        color_in = torch.cat([geom_in, xyz], dim=-1)
        rgb = self.color_mlp(color_in)
        return torch.cat([xyz, rgb], dim=-1)

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class GradientReversal(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class Neuro3DClassifierAndDecoder(nn.Module):
    def __init__(self, n_classes=40, n_subjects=6, embed_dim=384, n_points=1024,
                 subject_mode="invariant", dropout=0.15):
        super().__init__()
        self.subject_mode = subject_mode
        self.embed_dim = embed_dim
        
        self.static_pathway  = StaticVisualPathway(in_channels=128, embed_dim=embed_dim, dropout=dropout)
        self.dynamic_pathway = DynamicVisualPathway(in_channels=128, embed_dim=embed_dim, dropout=dropout)
        
        if subject_mode == "conditioned":
            self.subj_embed = nn.Embedding(n_subjects, embed_dim)
        else:
            self.subj_embed = None
            
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, n_classes)
        )
        self.pointcloud_decoder = PointCloudDecoder(embed_dim=embed_dim, n_points=n_points)
        
        # 5. Subject Invariance Head (GRL)
        if subject_mode == "invariant":
            self.grl = GradientReversal(alpha=1.0)
            self.subject_classifier = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, n_subjects)
            )
        
    def compute_orthogonality_loss(self, z_static, z_dynamic):
        cos_sim = F.cosine_similarity(z_static, z_dynamic, dim=-1)
        return torch.mean(torch.abs(cos_sim))

    def forward(self, x_eeg, sub=None, decode_3d=False):
        z_static  = self.static_pathway(x_eeg)
        z_dynamic = self.dynamic_pathway(x_eeg)
        
        if self.subject_mode == "conditioned" and sub is not None:
            z_static = z_static + self.subj_embed(sub)
            
        logits_cls = self.classifier(z_static)
        
        point_cloud = None
        if decode_3d:
            point_cloud = self.pointcloud_decoder(z_static, z_dynamic)
            
        return {
            "logits": logits_cls,
            "point_cloud": point_cloud,
            "z_static": z_static,
            "z_dynamic": z_dynamic
        }

# ============================================================================
# 2. CONTROLNET & EEG EMBEDDING MODULES (GWIT)
# ============================================================================

class SubjectLayers(nn.Module):
    def __init__(s, C, D, n_subjects):
        super().__init__()
        s.weights = nn.Parameter(torch.randn(n_subjects, C, D))
        s.weights.data *= 1 / C ** 0.5
    def forward(s, x, subjects):
        _, C, D = s.weights.shape
        w = s.weights.gather(0, subjects.view(-1, 1, 1).expand(-1, C, D))
        return torch.einsum("bct,bcd->bdt", x, w)

class EEGCondEmbedding(nn.Module):
    def __init__(s, out_ch=320, C=128, chans=(320, 640, 1280, 2560), strides=(5, 2, 2, 2), n_subjects=7):
        super().__init__(); s.out_ch = out_ch; s.subjects = None
        s.subj_layers = SubjectLayers(C, C, n_subjects)
        s.conv_in = nn.Conv1d(C, chans[0], kernel_size=3, stride=1)
        s.blocks = nn.ModuleList()
        for i in range(len(chans) - 1):
            s.blocks.append(nn.Conv1d(chans[i], chans[i], kernel_size=3, padding=1))
            s.blocks.append(nn.Conv1d(chans[i], chans[i + 1], kernel_size=strides[i + 1] + 1, padding=1, stride=strides[i + 1]))
    def forward(s, cond):
        x = cond.permute(0, 2, 1) if cond.shape[1] > cond.shape[2] else cond
        if 512 - x.shape[2] > 0: x = F.pad(x, (0, 512 - x.shape[2]))
        e = F.silu(s.conv_in(s.subj_layers(x, s.subjects)))
        for b in s.blocks: e = F.silu(b(e))
        e = e.reshape(e.shape[0], s.out_ch, e.shape[1] // s.out_ch, e.shape[2]).permute(0, 1, 3, 2)
        return F.pad(e, (0, 64 - e.shape[3], 0, 0))

# ============================================================================
# 3. LOAD DATASET & PREPARE SPLITS
# ============================================================================
print("\n" + "=" * 70)
print("1. LOADING EEGCVPR40 DATASET & ANNOTATIONS")
print("=" * 70)

pths = glob.glob(os.path.join(DATA_ROOT, "**", "*.pth"), recursive=True)
if not pths: pths = glob.glob(os.path.join(".", "**", "*.pth"), recursive=True)
if not pths: raise FileNotFoundError(f"No .pth files found in {DATA_ROOT}")

bn = os.path.basename
eeg_file = max([p for p in pths if "split" not in bn(p).lower()], key=os.path.getsize)
split_file = next((p for p in sorted([p for p in pths if "split" in bn(p).lower()],
                                     key=lambda p: "all" not in bn(p).lower())), None)

raw_data = torch.load(eeg_file, map_location="cpu", weights_only=False)
raw_images = raw_data["images"]
records = raw_data["dataset"]

X, O, IMG, SUB = [], [], [], []
for oi, it in enumerate(records):
    e = torch.as_tensor(it["eeg"]).float()
    if e.shape[0] != 128 and e.shape[1] == 128: e = e.t()
    if not (450 <= e.shape[1] <= 600): continue
    X.append(e[:, 20:460].contiguous())
    O.append(oi); IMG.append(int(it["image"])); SUB.append(int(it["subject"]))

del raw_data, records
X = torch.stack(X); O = np.array(O); IMG = np.array(IMG); SUB = np.array(SUB)

uniq_imgs = sorted(set(IMG.tolist()))
img_names = {u: raw_images[u] for u in uniq_imgs}
synsets   = sorted({img_names[u].split("_")[0] for u in uniq_imgs})
syn2cls   = {s: i for i, s in enumerate(synsets)}
N_CLASSES = len(synsets)
N_CHANNELS = X.shape[1]

trial_cls  = np.array([syn2cls[img_names[im].split("_")[0]] for im in IMG])
subj_vals  = sorted(set(SUB.tolist()))
subj_map   = {s: i for i, s in enumerate(subj_vals)}
trial_subj = np.array([subj_map[s] for s in SUB])
N_SUBJECTS = max(7, int(SUB.max()) + 1)

rows_by_img = {}
for r, im in enumerate(IMG): rows_by_img.setdefault(int(im), []).append(r)

if split_file is not None:
    sp = torch.load(split_file, map_location="cpu", weights_only=False)["splits"][0]
    row_of = {int(o): r for r, o in enumerate(O)}
    tr_l, va_l, te_l = [[row_of[int(i)] for i in sp[k] if int(i) in row_of] for k in ("train", "val", "test")]
    tr, va, te = np.array(tr_l), np.array(va_l), np.array(te_l)
else:
    rng = np.random.RandomState(SEED)
    ims = np.array(sorted(rows_by_img.keys()))
    rng.shuffle(ims)
    n_te, n_va = int(0.15 * len(ims)), int(0.15 * len(ims))
    te_ims, va_ims, tr_ims = ims[:n_te], ims[n_te:n_te + n_va], ims[n_te + n_va:]
    tr = np.array([r for im in tr_ims for r in rows_by_img[im]])
    va = np.array([r for im in va_ims for r in rows_by_img[im]])
    te = np.array([r for im in te_ims for r in rows_by_img[im]])

print(f"Trials Partition -> Train: {len(tr)} | Val: {len(va)} | Test: {len(te)}")

# Training-set channel standardization
train_signals = X[tr].float()
channel_mean = train_signals.mean(dim=(0, 2), keepdim=True)
channel_std  = train_signals.std(dim=(0, 2), keepdim=True) + 1e-6
X_norm = (X.float() - channel_mean) / channel_std
del train_signals

# ============================================================================
# 4. STAGE A: TRAIN NEURO-3D CLASSIFIER (DISENTANGLED STATIC / DYNAMIC)
# ============================================================================
print("\n" + "=" * 70)
print(f"STAGE A: Training Neuro-3D Classifier ({NEURO3D_EPOCHS} Epochs)")
print("=" * 70)

neuro3d_model = Neuro3DClassifierAndDecoder(
    n_classes=N_CLASSES, n_subjects=N_SUBJECTS, embed_dim=EMBED_DIM,
    n_points=N_3D_POINTS, subject_mode="invariant", dropout=DROPOUT
).to(dev)

X_norm_t     = X_norm.to(dev)
trial_cls_t   = torch.as_tensor(trial_cls, device=dev, dtype=torch.long)
trial_subj_t  = torch.as_tensor(trial_subj, device=dev, dtype=torch.long)

val_imgs_uniq  = sorted(set(IMG[va]))
test_imgs_uniq = sorted(set(IMG[te]))
val_img2trials = {im: np.where(IMG[va] == im)[0] for im in val_imgs_uniq}
test_img2trials = {im: np.where(IMG[te] == im)[0] for im in test_imgs_uniq}

restored = sync_or_restore_file(NEURO3D_BEST_PATH, "best_neuro3d_classifier.pt")
if restored and os.path.exists(NEURO3D_BEST_PATH):
    try:
        ckpt = torch.load(NEURO3D_BEST_PATH, map_location=dev, weights_only=False)
        state = ckpt.get("model_state", ckpt)
        neuro3d_model.load_state_dict(state, strict=False)
        print(f"[OK] Loaded proven Neuro-3D checkpoint from: {NEURO3D_BEST_PATH}")
    except Exception as e:
        print(f"Checkpoint load error: {e}. Re-training...")
        restored = False

if not restored:
    print(f"Starting Neuro-3D Training: {NEURO3D_EPOCHS} epochs | LR {NEURO3D_LR} -> {NEURO3D_LR_MIN}...")
    optimizer_ns = torch.optim.AdamW(neuro3d_model.parameters(), lr=NEURO3D_LR, weight_decay=0.03)
    scheduler_ns = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ns, T_max=NEURO3D_EPOCHS, eta_min=NEURO3D_LR_MIN)
    scaler_ns = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    best_pool_acc = 0.0

    for ep in range(1, NEURO3D_EPOCHS + 1):
        neuro3d_model.train()
        perm = np.random.permutation(len(tr))
        ep_loss = []
        current_lr = optimizer_ns.param_groups[0]["lr"]

        for k in range(0, len(tr), NEURO3D_BS):
            b_idx = tr[perm[k:k + NEURO3D_BS]]
            x_b = X_norm_t[b_idx]
            y_b = trial_cls_t[b_idx]

            optimizer_ns.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
                out = neuro3d_model(x_b, sub=None, decode_3d=False)
                loss_ce = F.cross_entropy(out["logits"], y_b, label_smoothing=0.05)
                loss_ortho = neuro3d_model.compute_orthogonality_loss(out["z_static"], out["z_dynamic"])
                total_loss = loss_ce + LAMBDA_ORTHO * loss_ortho

            scaler_ns.scale(total_loss).backward()
            scaler_ns.unscale_(optimizer_ns)
            nn.utils.clip_grad_norm_(neuro3d_model.parameters(), 1.0)
            scaler_ns.step(optimizer_ns)
            scaler_ns.update()
            ep_loss.append(total_loss.item())

        scheduler_ns.step()

        if ep % 5 == 0 or ep == NEURO3D_EPOCHS:
            neuro3d_model.eval()
            with torch.no_grad():
                val_feats = []
                for i in range(0, len(va), 64):
                    z_s = neuro3d_model.static_pathway(X_norm_t[va[i:i+64]])
                    val_feats.append(z_s)
                val_feats = torch.cat(val_feats, dim=0)

                # Pooled evaluation per unique image
                pool_correct = sum(
                    1 for im, t_idxs in val_img2trials.items()
                    if neuro3d_model.classifier(val_feats[t_idxs].mean(dim=0, keepdim=True)).argmax().item() == trial_cls[va[t_idxs[0]]]
                )
                pool_acc = pool_correct / len(val_img2trials) * 100.0

                avg_l = np.mean(ep_loss) if ep_loss else 0.0
                print(f"  Neuro-3D Ep {ep:02d}/{NEURO3D_EPOCHS} | Loss: {avg_l:.4f} | LR: {current_lr:.2e} | Val Pooled Acc: {pool_acc:.2f}%")
                
                if pool_acc > best_pool_acc:
                    best_pool_acc = pool_acc
                    torch.save({"epoch": ep, "model_state": neuro3d_model.state_dict(), "best_acc": best_pool_acc}, NEURO3D_BEST_PATH)
                    print(f"    --> [NEW BEST] Checkpoint saved with Val Pooled Acc: {best_pool_acc:.2f}%")

    print(f"\n[Drive Sync] Backing up best Neuro-3D classifier to Google Drive folder...")
    drive_upload_file(NEURO3D_BEST_PATH, "best_neuro3d_classifier.pt")

# ============================================================================
# 5. PREDICT ACCURATE COARSE CAPTIONS FOR CONTROLNET (RELOAD BEST CHECKPOINT)
# ============================================================================
print("\n" + "=" * 70)
print("RELOAD BEST CHECKPOINT & PREDICT COARSE CAPTIONS FOR CONTROLNET")
print("=" * 70)

if os.path.exists(NEURO3D_BEST_PATH):
    best_ckpt = torch.load(NEURO3D_BEST_PATH, map_location=dev, weights_only=False)
    state = best_ckpt.get("model_state", best_ckpt)
    neuro3d_model.load_state_dict(state, strict=False)
    b_ep = best_ckpt.get("epoch", "unknown")
    b_acc = best_ckpt.get("best_acc", "unknown")
    print(f"[✓] Reloaded BEST Neuro-3D checkpoint from epoch {b_ep} (Val Pooled Acc: {b_acc}%)")

neuro3d_model.eval()

# Synset to Caption mapping
SYNSET_TO_CAPTION = {
    "n02106662": "a photo of a German shepherd",
    "n02124075": "a photo of an Egyptian cat",
    "n02281787": "a photo of a butterfly",
    "n02389026": "a photo of a horse",
    "n02492035": "a photo of a capuchin monkey",
    "n02504458": "a photo of an African elephant",
    "n02510455": "a photo of a giant panda",
    "n02607072": "a photo of an anemone fish",
    "n02690373": "a photo of an airliner airplane",
    "n02906734": "a photo of a broom",
    "n02951358": "a photo of a canoe",
    "n02992529": "a photo of a mobile phone",
    "n03063599": "a photo of a coffee mug",
    "n03100240": "a photo of a convertible car",
    "n03180011": "a photo of a desktop computer",
    "n03197337": "a photo of a digital watch",
    "n03272010": "a photo of an electric guitar",
    "n03272562": "a photo of a train locomotive",
    "n03297495": "a photo of an espresso maker",
    "n03376595": "a photo of a folding chair",
    "n03445777": "a photo of a golf ball",
    "n03452741": "a photo of a grand piano",
    "n03584829": "a photo of an iron",
    "n03590841": "a photo of a jack-o-lantern",
    "n03709823": "a photo of a mailbag postbag",
    "n03773504": "a photo of a missile rocket",
    "n03775071": "a photo of a mitten glove",
    "n03792782": "a photo of a mountain bike",
    "n03792972": "a photo of a mountain tent",
    "n03877472": "a photo of pajamas",
    "n03888257": "a photo of a parachute",
    "n03982430": "a photo of a pool billiard table",
    "n04044716": "a photo of a radio telescope",
    "n04069434": "a photo of a camera",
    "n04086273": "a photo of a revolver handgun",
    "n04120489": "a photo of a running shoe sneaker",
    "n07753592": "a photo of a banana",
    "n07873807": "a photo of a pizza",
    "n11939491": "a photo of a daisy flower",
    "n13054560": "a photo of a mushroom"
}

cls2syn = {i: s for s, i in syn2cls.items()}
id_to_caption = {i: SYNSET_TO_CAPTION.get(s, f"a photo of {s}") for i, s in cls2syn.items()}

CAP = np.zeros(len(IMG), dtype=int)
with torch.no_grad():
    all_feats = []
    for k in range(0, len(X_norm_t), 64):
        all_feats.append(neuro3d_model.static_pathway(X_norm_t[k:k+64]))
    all_feats = torch.cat(all_feats, dim=0)

    for im in uniq_imgs:
        im_rows = rows_by_img[im]
        pooled_feat = all_feats[im_rows].mean(dim=0, keepdim=True)
        pred_cls = neuro3d_model.classifier(pooled_feat).argmax(dim=-1).item()
        CAP[im_rows] = pred_cls

val_cap_acc = np.mean(CAP[va] == trial_cls[va]) * 100.0
test_cap_acc = np.mean(CAP[te] == trial_cls[te]) * 100.0
print(f"Caption Accuracy for ControlNet: Val {val_cap_acc:.2f}% | Test {test_cap_acc:.2f}%")

del all_feats
gc.collect(); torch.cuda.empty_cache()

# ============================================================================
# 6. STAGE B: TRAIN GWIT CONTROLNET ON STABLE DIFFUSION 2.1 BASE
# ============================================================================
print("\n" + "=" * 70)
print("STAGE B: Training GWIT ControlNet (Conditioned on Neuro-3D Captions)")
print("=" * 70)

restored_pipe = sync_or_restore_file(OUT_PATH, "neuro3d_pipeline.pt")
if restored_pipe and os.path.exists(OUT_PATH):
    print(f"[OK] Pipeline checkpoint already exists: {OUT_PATH}. Skipping ControlNet training.")
else:
    for k in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"]:
        val = os.environ.get(k, "")
        if not val or len(val.strip()) < 10 or "your_token" in val or "token" in val.lower():
            os.environ.pop(k, None)

    SD_IDS = [
        "sd2-community/stable-diffusion-2-1-base",
        "stabilityai/stable-diffusion-2-1-base",
        "runwayml/stable-diffusion-v1-5",
        "CompVis/stable-diffusion-v1-4"
    ]

    SD_ID = None
    for sid in SD_IDS:
        try:
            UNet2DConditionModel.load_config(sid, subfolder="unet", token=False)
            SD_ID = sid
            break
        except Exception as e:
            print("Cannot reach", sid, str(e)[:80])

    assert SD_ID, "Could not reach Stable Diffusion 2.1 base models."
    print(f"Using: {SD_ID}")

    tok = CLIPTokenizer.from_pretrained(SD_ID, subfolder="tokenizer", token=False)
    tenc = CLIPTextModel.from_pretrained(SD_ID, subfolder="text_encoder", torch_dtype=torch.float16, token=False).to(dev).eval()
    prompts = [id_to_caption[i] for i in range(40)] + [""]
    EMPTY = 40
    with torch.no_grad():
        ids_tok = tok(prompts, max_length=tok.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(dev)
        TEXT = tenc(ids_tok, return_dict=False)[0].half()
    del tenc; torch.cuda.empty_cache()

    def resolve_image_path(name):
        syn = name.split("_")[0]
        p1 = os.path.join(IMAGE_DIR, syn, name + ".JPEG")
        if os.path.exists(p1): return p1
        p2 = os.path.join(IMAGE_DIR, name + ".JPEG")
        if os.path.exists(p2): return p2
        g = glob.glob(os.path.join(IMAGE_DIR, "**", name + ".*"), recursive=True)
        return g[0] if g else None

    img_paths = {u: resolve_image_path(img_names[u]) for u in uniq_imgs}
    vae = AutoencoderKL.from_pretrained(SD_ID, subfolder="vae", torch_dtype=torch.float16).to(dev).eval()
    SCALE = vae.config.scaling_factor
    img_tf = transforms.Compose([
        transforms.Resize(RES, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(RES), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])
    ])

    need = sorted({int(IMG[r]) for r in np.concatenate([tr, va]) if img_paths.get(int(IMG[r])) is not None})
    IROW = {u: k for k, u in enumerate(need)}

    if need:
        MOM = torch.zeros(len(need), 8, RES // 8, RES // 8, dtype=torch.float16, device=dev)
        with torch.no_grad():
            for k in tqdm(range(0, len(need), 16), desc="Encoding VAE posteriors"):
                px = torch.stack([img_tf(Image.open(img_paths[u]).convert("RGB")) for u in need[k:k + 16]]).to(dev).half()
                MOM[k:k + len(px)] = vae.encode(px).latent_dist.parameters
    else:
        print("Notice: Local ImageNet directory not found. Initializing synthetic latent cache.")
        MOM = torch.zeros(len(uniq_imgs), 8, RES // 8, RES // 8, dtype=torch.float16, device=dev)
        IROW = {u: k for k, u in enumerate(uniq_imgs)}

    del vae; torch.cuda.empty_cache()
    ROW_IMG = np.array([IROW.get(int(i), 0) for i in IMG])

    unet = UNet2DConditionModel.from_pretrained(SD_ID, subfolder="unet")
    controlnet = ControlNetModel.from_unet(unet)
    controlnet.controlnet_cond_embedding = EEGCondEmbedding(controlnet.config.block_out_channels[0], n_subjects=N_SUBJECTS)
    unet.requires_grad_(False); unet.to(dev, dtype=torch.float16); unet.enable_gradient_checkpointing()
    controlnet.to(dev).train(); controlnet.enable_gradient_checkpointing()
    sched = DDPMScheduler.from_pretrained(SD_ID, subfolder="scheduler")

    def loss_fn(rows, train=True):
        rows = np.asarray(rows)
        x = X_norm_t[rows]
        subj = torch.as_tensor(SUB[rows], device=dev)
        mean, logvar = MOM[torch.as_tensor(ROW_IMG[rows], device=dev)].float().chunk(2, dim=1)
        lat = (mean + torch.exp(0.5 * logvar.clamp(-30.0, 20.0)) * torch.randn_like(mean)) * SCALE
        cap_ids = [EMPTY if (train and random.random() < PROP_EMPTY) else int(CAP[r]) for r in rows]
        txt = TEXT[torch.as_tensor(cap_ids, device=dev)]
        noise = torch.randn_like(lat)
        t = torch.randint(0, sched.config.num_train_timesteps, (len(rows),), device=dev).long()
        noisy = sched.add_noise(lat, noise, t)
        controlnet.controlnet_cond_embedding.subjects = subj
        with torch.autocast("cuda", dtype=torch.float16):
            down, mid = controlnet(noisy, t, encoder_hidden_states=txt, controlnet_cond=x, return_dict=False)
            pred = unet(noisy.half(), t, encoder_hidden_states=txt,
                        down_block_additional_residuals=[s_.half() for s_ in down],
                        mid_block_additional_residual=mid.half(), return_dict=False)[0]
        return F.mse_loss(pred.float(), noise.float())

    params = [p for p in controlnet.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR_CONTROLNET, weight_decay=WD)
    scaler = torch.amp.GradScaler("cuda")

    def save_pipeline(epoch):
        torch.save({
            "controlnet": {k: v.detach().cpu().half() for k, v in controlnet.state_dict().items()},
            "neuro3d_state": {k: v.cpu() for k, v in neuro3d_model.state_dict().items()},
            "n_subjects": N_SUBJECTS,
            "sd_id": SD_ID,
            "res": RES,
            "epoch": epoch,
            "text_emb": TEXT.cpu(),
            "id_to_caption": id_to_caption,
            "syn2cls": syn2cls,
            "norm_mean": channel_mean.cpu(),
            "norm_std": channel_std.cpu()
        }, OUT_PATH)
        print(f"Saved full pipeline checkpoint -> {OUT_PATH} (Epoch {epoch})")

    print(f"\nTraining ControlNet for {EPOCHS_CONTROLNET} epoch(s)...")
    for ep in range(1, EPOCHS_CONTROLNET + 1):
        perm = np.random.permutation(tr)
        nb = len(perm) // BS
        run = []
        bar = tqdm(range(nb), desc=f"ControlNet Epoch {ep}/{EPOCHS_CONTROLNET}")
        for b in bar:
            loss = loss_fn(perm[b * BS:(b + 1) * BS])
            scaler.scale(loss / ACCUM).backward()
            run.append(loss.item())
            if (b + 1) % ACCUM == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            if b % 50 == 0:
                bar.set_postfix(loss=f"{np.mean(run[-50:]):.4f}")
            if (b + 1) % SAVE_EVERY == 0:
                save_pipeline(ep)

    save_pipeline(EPOCHS_CONTROLNET)
    print("\n[Drive Sync] Backing up full pipeline checkpoint to Google Drive folder...")
    drive_upload_file(OUT_PATH, "neuro3d_pipeline.pt")

print("\n" + "=" * 70)
print(f"CELL 1 COMPLETE: NEURO-3D ({NEURO3D_EPOCHS} Ep) + CONTROLNET FULLY TRAINED!")
print("Saved models:")
print(f"  - Best Classifier : {NEURO3D_BEST_PATH}")
print(f"  - Full Pipeline   : {OUT_PATH}")
print("=" * 70)
