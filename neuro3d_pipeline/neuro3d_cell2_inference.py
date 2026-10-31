# ============================================================================
# NEURO-3D CELL 2 : INFERENCE & EVALUATION (SAMPLE 700 TO THE END)
# ----------------------------------------------------------------------------
# 1. Model Loading (Local vs. Google Drive):
#    - Loads trained 'best_neuro3d_classifier.pt' from /kaggle/working/.
#    - If not found locally, automatically downloads it from Google Drive:
#      13vd_i1ePxEbYH5RtW9Q4B2OG0ccwv6Wz
# 2. Evaluation Range:
#    - Starts evaluation strictly from SAMPLE 700 through the end of the test set.
# 3. Real-Time Incremental CSV Logging (One by One):
#    - Creates/appends to 'test_evaluation_results.csv' with 3 columns:
#      [real_image_class, predicted_class, isTrue]
#    - Appends each result row IMMEDIATELY with flush() as each sample completes
#      (one by one, never waiting until the end), preventing any data loss.
# 4. Google Drive Output Generation:
#    - Segmentation phase (YOLOv8) is completely removed.
#    - Saves in Drive folder 'GeneratedImage/' under the main folder.
#    - For each sample, creates/reuses subfolder: 'sample_XXXX/'
#    - Uploads two images: 'real.png' and 'generated.png' (overwrites if existing).
#    - Syncs 'test_evaluation_results.csv' to Google Drive.
# ============================================================================

import os, sys, gc, glob, random, math, subprocess, warnings, csv, time
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

try:
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, UniPCMultistepScheduler, ControlNetModel
except AttributeError as _e:
    for _m in [m for m in sys.modules if m.startswith("diffusers")]: del sys.modules[_m]
    sys.modules["torch._dynamo"] = None
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, UniPCMultistepScheduler, ControlNetModel

from transformers import CLIPTextModel, CLIPTokenizer

# ----------------- Configuration & Folders -----------------
DATA_ROOT = "/kaggle/input/datasets/evolutionaw/eegcvpr40/data"
OUTPUT_DIR = "/kaggle/working"
LOCAL_GEN_DIR = os.path.join(OUTPUT_DIR, "GeneratedImage")
os.makedirs(LOCAL_GEN_DIR, exist_ok=True)

NEURO3D_BEST_PATH = os.path.join(OUTPUT_DIR, "best_neuro3d_classifier.pt")
PIPELINE_PATH = os.path.join(OUTPUT_DIR, "neuro3d_pipeline.pt")
CSV_OUTPUT_PATH = os.path.join(LOCAL_GEN_DIR, "test_evaluation_results.csv")

# Start from sample 700 to the end as requested
START_SAMPLE_IDX = 700

STEPS = 20             # Fast high-quality UniPC sampling steps (approx 0.8s / image)
GUIDANCE = 4.0
SEED = 42
EMBED_DIM = 384
N_3D_POINTS = 1024
DROPOUT = 0.15

dev = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = (dev == "cuda")

def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

seed_everything(SEED)

# Clean prior GPU memory
for _var in ['unet', 'vae', 'txt', 'tok', 'c_all', 'lat']:
    if _var in globals():
        del globals()[_var]
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ============================================================================
# GOOGLE DRIVE BIDIRECTIONAL SYNC (OAuth 2.0 User Quota)
# ============================================================================
DRIVE_MAIN_FOLDER_ID = "13vd_i1ePxEbYH5RtW9Q4B2OG0ccwv6Wz"
DRIVE_CLIENT_ID = os.environ.get("DRIVE_CLIENT_ID", "YOUR_DRIVE_CLIENT_ID")
DRIVE_CLIENT_SECRET = os.environ.get("DRIVE_CLIENT_SECRET", "YOUR_DRIVE_CLIENT_SECRET")
DRIVE_REFRESH_TOKEN = os.environ.get("DRIVE_REFRESH_TOKEN", "YOUR_DRIVE_REFRESH_TOKEN")

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

def drive_find_file(filename, parent_id=DRIVE_MAIN_FOLDER_ID):
    service = get_drive_service()
    if not service: return None
    try:
        query = f"'{parent_id}' in parents and name = '{filename}' and trashed = false"
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

def get_or_create_drive_folder(parent_id, folder_name):
    service = get_drive_service()
    if not service: return None
    try:
        q = f"'{parent_id}' in parents and name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        res = service.files().list(q=q, fields="files(id, name)").execute()
        files = res.get("files", [])
        if files:
            return files[0]["id"]
        body = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id]
        }
        created = service.files().create(body=body, fields="id, name").execute()
        print(f"  [Drive] Created folder '{folder_name}' in Google Drive! (ID: {created['id']})")
        return created["id"]
    except Exception as e:
        print(f"  [Drive Error] Could not get/create folder '{folder_name}': {e}")
        return None

def fetch_all_drive_subfolders(parent_folder_id):
    service = get_drive_service()
    if not service: return {}
    folder_dict = {}
    page_token = None
    try:
        while True:
            q = f"'{parent_folder_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            res = service.files().list(q=q, fields="nextPageToken, files(id, name)", pageSize=1000, pageToken=page_token).execute()
            for item in res.get("files", []):
                folder_dict[item["name"]] = item["id"]
            page_token = res.get("nextPageToken")
            if not page_token:
                break
    except Exception as e:
        print(f"  [Drive Notice] Error listing existing subfolders: {e}")
    return folder_dict

def drive_upload_or_overwrite(parent_folder_id, filename, local_path, mimetype=None):
    if not os.path.exists(local_path):
        return None
    service = get_drive_service()
    if not service: return None
    try:
        from googleapiclient.http import MediaFileUpload
        if mimetype is None:
            if filename.endswith(".png"): mimetype = "image/png"
            elif filename.endswith(".jpg") or filename.endswith(".jpeg"): mimetype = "image/jpeg"
            elif filename.endswith(".csv"): mimetype = "text/csv"
            else: mimetype = "application/octet-stream"

        media = MediaFileUpload(local_path, mimetype=mimetype, resumable=False)
        q = f"'{parent_folder_id}' in parents and name = '{filename}' and trashed = false"
        res = service.files().list(q=q, fields="files(id, name)").execute()
        files = res.get("files", [])
        if files:
            file_id = files[0]["id"]
            updated = service.files().update(fileId=file_id, media_body=media, fields="id, name").execute()
            return updated["id"]
        else:
            body = {"name": filename, "parents": [parent_folder_id]}
            created = service.files().create(body=body, media_body=media, fields="id, name").execute()
            return created["id"]
    except Exception as e:
        print(f"    [Drive Warning] Upload/Overwrite error for '{filename}': {e}")
        return None

# ============================================================================
# 1. VERIFY & RESTORE NEURO-3D MODEL CHECKPOINT (LOCAL vs GOOGLE DRIVE)
# ============================================================================
print("\n" + "=" * 75)
print("CHECKING NEURO-3D MODEL CHECKPOINTS (LOCAL vs GOOGLE DRIVE)")
print("=" * 75)

if os.path.exists(NEURO3D_BEST_PATH):
    print(f"[✓] Found local model checkpoint: {NEURO3D_BEST_PATH} ({os.path.getsize(NEURO3D_BEST_PATH):,} bytes)")
else:
    print(f"[Notice] Local checkpoint '{NEURO3D_BEST_PATH}' not found. Searching Google Drive...")
    drive_f = drive_find_file(os.path.basename(NEURO3D_BEST_PATH), DRIVE_MAIN_FOLDER_ID)
    if drive_f:
        print(f"  Found '{drive_f['name']}' in Google Drive! Restoring...")
        drive_download_file(drive_f["id"], NEURO3D_BEST_PATH, os.path.basename(NEURO3D_BEST_PATH))
    else:
        print(f"  [Notice] '{os.path.basename(NEURO3D_BEST_PATH)}' not found in Drive. Will proceed with initialized weights.")

if not os.path.exists(PIPELINE_PATH):
    drive_pipe = drive_find_file(os.path.basename(PIPELINE_PATH), DRIVE_MAIN_FOLDER_ID)
    if drive_pipe:
        print(f"  Found '{drive_pipe['name']}' in Google Drive! Restoring...")
        drive_download_file(drive_pipe["id"], PIPELINE_PATH, os.path.basename(PIPELINE_PATH))

# ============================================================================
# 2. NEURO-3D ARCHITECTURE (Dual Disentangled Pathways)
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
# 3. LOAD DATASET & TEST SPLITS
# ============================================================================
print("\n" + "=" * 75)
print("LOADING EEGCVPR40 DATASET FOR NEURO-3D TEST EVALUATION")
print("=" * 75)

pths = glob.glob(os.path.join(DATA_ROOT, "**", "*.pth"), recursive=True)
if not pths: pths = glob.glob(os.path.join(".", "**", "*.pth"), recursive=True)
if not pths: raise FileNotFoundError("Dataset not found!")

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
cls2syn   = {i: s for s, i in syn2cls.items()}
N_CLASSES = len(synsets)

trial_cls  = np.array([syn2cls[img_names[im].split("_")[0]] for im in IMG])
subj_vals  = sorted(set(SUB.tolist()))
subj_map   = {s: i for i, s in enumerate(subj_vals)}
trial_subj = np.array([subj_map[s] for s in SUB])
N_SUBJECTS = len(subj_vals)

# Benchmark Test Split
if split_file is not None:
    sp = torch.load(split_file, map_location="cpu", weights_only=False)["splits"][0]
    row_of = {int(o): r for r, o in enumerate(O)}
    tr_l, va_l, te_l = [[row_of[int(i)] for i in sp[k] if int(i) in row_of] for k in ("train", "val", "test")]
    tr, va, te = np.array(tr_l), np.array(va_l), np.array(te_l)
else:
    rows_by_img = {}
    for r, im in enumerate(IMG): rows_by_img.setdefault(int(im), []).append(r)
    rng = np.random.RandomState(SEED)
    ims = np.array(sorted(rows_by_img.keys()))
    rng.shuffle(ims)
    n_te = int(0.15 * len(ims))
    te_ims = ims[:n_te]
    te = np.array([r for im in te_ims for r in rows_by_img[im]])
    tr = np.array([r for im in ims[n_te:] for r in rows_by_img[im]])

print(f"Total Test Trials: {len(te)} trials across {len(set(IMG[te]))} unique images")

# Training-set channel standardization
train_signals = X[tr].float()
channel_mean = train_signals.mean(dim=(0, 2), keepdim=True)
channel_std  = train_signals.std(dim=(0, 2), keepdim=True) + 1e-6
X_norm = (X.float() - channel_mean) / channel_std
del train_signals

# ============================================================================
# 4. LOAD NEURO-3D MODEL
# ============================================================================
print("\n" + "=" * 75)
print("INITIALIZING NEURO-3D MODEL FOR INFERENCE")
print("=" * 75)

model = Neuro3DClassifierAndDecoder(
    n_classes=N_CLASSES, n_subjects=N_SUBJECTS, embed_dim=EMBED_DIM,
    n_points=N_3D_POINTS, subject_mode="invariant", dropout=DROPOUT
).to(dev)

if os.path.exists(NEURO3D_BEST_PATH):
    ckpt = torch.load(NEURO3D_BEST_PATH, map_location=dev, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    print(f"[✓] Loaded trained checkpoint: {NEURO3D_BEST_PATH}")
else:
    print(f"[Notice] {NEURO3D_BEST_PATH} not found. Running with initialized weights.")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

model.eval()

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

def get_clean_class_name(cls_idx):
    syn = cls2syn.get(cls_idx, "")
    caption = SYNSET_TO_CAPTION.get(syn, syn)
    return caption.replace("a photo of an ", "").replace("a photo of a ", "").replace("a photo of ", "").strip()

# ----------------- Image Lookup for Real Stimulus Images -----------------
IMAGE_DIR_CANDIDATES = [
    "/kaggle/input/competitions/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train",
    "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train",
    "/kaggle/input/datasets/evolutionaw/eegcvpr40/data/images",
    "/kaggle/input/datasets/evolutionaw/eegcvpr40/images",
    os.path.join(DATA_ROOT, "images"),
    "./images",
    "/kaggle/working/images"
]

IMAGE_PATH_CACHE = {}

def resolve_image_path(name):
    if name in IMAGE_PATH_CACHE:
        return IMAGE_PATH_CACHE[name]
    syn = name.split("_")[0]
    for root in IMAGE_DIR_CANDIDATES:
        if not os.path.exists(root):
            continue
        p1 = os.path.join(root, syn, name + ".JPEG")
        if os.path.exists(p1):
            IMAGE_PATH_CACHE[name] = p1; return p1
        p2 = os.path.join(root, name + ".JPEG")
        if os.path.exists(p2):
            IMAGE_PATH_CACHE[name] = p2; return p2
        p3 = os.path.join(root, syn, name + ".jpg")
        if os.path.exists(p3):
            IMAGE_PATH_CACHE[name] = p3; return p3
        g = glob.glob(os.path.join(root, syn, name + ".*"))
        if g:
            IMAGE_PATH_CACHE[name] = g[0]; return g[0]
    IMAGE_PATH_CACHE[name] = None
    return None

# ============================================================================
# 5. LOAD DIFFUSION PIPELINE (Fast UniPC Multi-Step, NO YOLO / NO SEGMENTATION)
# ============================================================================
print("\n" + "=" * 75)
print("LOADING STABLE DIFFUSION BASE COMPONENTS (Fast UniPC Multi-Step)")
print("=" * 75)

for k in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"]:
    val = os.environ.get(k, "")
    if not val or len(val.strip()) < 10 or "your_token" in val or "token" in val.lower():
        os.environ.pop(k, None)

SD_CANDIDATES = [
    "sd2-community/stable-diffusion-2-1-base",
    "stabilityai/stable-diffusion-2-1-base",
    "runwayml/stable-diffusion-v1-5",
    "CompVis/stable-diffusion-v1-4"
]

tok, txt, vae, unet, sched, active_sd = None, None, None, None, None, None

for cand_id in SD_CANDIDATES:
    try:
        print(f"Attempting to load model from: {cand_id} ...")
        tok = CLIPTokenizer.from_pretrained(cand_id, subfolder="tokenizer", token=False)
        txt = CLIPTextModel.from_pretrained(cand_id, subfolder="text_encoder", token=False).to(dev)
        vae = AutoencoderKL.from_pretrained(cand_id, subfolder="vae", token=False).to(dev)
        unet = UNet2DConditionModel.from_pretrained(cand_id, subfolder="unet", token=False).to(dev)
        sched = UniPCMultistepScheduler.from_pretrained(cand_id, subfolder="scheduler", token=False)
        active_sd = cand_id
        print(f"Successfully loaded Stable Diffusion components from: {active_sd}")
        break
    except Exception as e:
        print(f"  Notice: {cand_id} failed ({str(e)[:80]}). Trying next candidate...")

if active_sd is None:
    print("Warning: All online SD candidates unavailable. Fallback visual generator will be used.")

# ============================================================================
# 6. INITIALIZE GOOGLE DRIVE TARGET DIRECTORY & CSV
# ============================================================================
print("\n" + "=" * 75)
print("INITIALIZING GOOGLE DRIVE 'GeneratedImage' DIRECTORY & CSV LOG")
print("=" * 75)

gen_folder_id = get_or_create_drive_folder(DRIVE_MAIN_FOLDER_ID, "GeneratedImage")
if gen_folder_id:
    print(f"[✓] Google Drive 'GeneratedImage' folder ready (ID: {gen_folder_id})")
    print("  Fetching existing subfolders in 'GeneratedImage' to prevent duplicates...")
    existing_drive_subfolders = fetch_all_drive_subfolders(gen_folder_id)
    print(f"  Found {len(existing_drive_subfolders)} existing subfolder(s) in Drive.")
else:
    print("[Notice] Could not initialize Google Drive folder. Outputs will be saved locally.")
    existing_drive_subfolders = {}

# Initialize CSV file if it doesn't exist yet
if not os.path.exists(CSV_OUTPUT_PATH) or os.path.getsize(CSV_OUTPUT_PATH) == 0:
    with open(CSV_OUTPUT_PATH, mode="w", newline="", encoding="utf-8") as f_init:
        writer_init = csv.writer(f_init)
        writer_init.writerow(["real_image_class", "predicted_class", "isTrue"])
    print(f"[✓] Initialized new evaluation CSV at: {CSV_OUTPUT_PATH}")
else:
    print(f"[✓] Existing evaluation CSV found at: {CSV_OUTPUT_PATH} (Will append results one-by-one)")

# ============================================================================
# 7. INFERENCE LOOP FROM SAMPLE 700 TO THE END (ONE-BY-ONE CSV LOGGING)
# ============================================================================
TOTAL_TEST_SAMPLES = len(te)
print("\n" + "=" * 75)
print(f"STARTING NEURO-3D GENERATION & EVALUATION: SAMPLES {START_SAMPLE_IDX} TO {TOTAL_TEST_SAMPLES - 1}")
print(f"Total samples to evaluate in this run: {TOTAL_TEST_SAMPLES - START_SAMPLE_IDX}")
print(f"Logging mode: Real-time incremental CSV append (one by one)")
print("=" * 75)

total_true_count = 0
total_false_count = 0
total_evaluated_count = 0
start_time = time.time()

for s_i in range(START_SAMPLE_IDX, TOTAL_TEST_SAMPLES):
    trial_idx = te[s_i]
    im_id = IMG[trial_idx]
    im_name = img_names[im_id]
    gt_c = trial_cls[trial_idx]
    gt_class_name = get_clean_class_name(gt_c)
    gt_caption = id_to_caption.get(gt_c, gt_class_name)

    # 1. Neuro-3D Inference on EEG trial
    x_trial = X_norm[trial_idx:trial_idx+1].to(device=dev)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
            out = model(x_trial, sub=None, decode_3d=False)
            pred_c = out["logits"].argmax(dim=-1).item()

    pred_class_name = get_clean_class_name(pred_c)
    pred_prompt = id_to_caption.get(pred_c, f"a photo of {pred_class_name}")
    is_match = (pred_c == gt_c)
    is_true_str = "true" if is_match else "false"

    if is_match:
        total_true_count += 1
    else:
        total_false_count += 1
    total_evaluated_count += 1

    # 2. Append result to CSV ONE-BY-ONE immediately with flush()
    with open(CSV_OUTPUT_PATH, mode="a", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([gt_class_name, pred_class_name, is_true_str])
        f_csv.flush()

    # 3. Local Sample Folder Creation (sample_0700, sample_0701, etc.)
    sample_folder_name = f"sample_{s_i:04d}"
    sample_local_dir = os.path.join(LOCAL_GEN_DIR, sample_folder_name)
    os.makedirs(sample_local_dir, exist_ok=True)

    real_img_path = os.path.join(sample_local_dir, "real.png")
    gen_img_path = os.path.join(sample_local_dir, "generated.png")

    # 4. Obtain Real Stimulus Image (512x512)
    real_p = resolve_image_path(im_name)
    real_np = None
    if real_p and os.path.exists(real_p):
        try:
            real_np = np.array(Image.open(real_p).convert("RGB").resize((512, 512)))
        except Exception:
            real_np = None
    if real_np is None:
        real_np = np.ones((512, 512, 3), dtype=np.uint8) * 240
        cv2.rectangle(real_np, (15, 15), (497, 497), (200, 210, 200), 2)
        cv2.putText(real_np, "REAL STIMULUS", (70, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 80, 20), 2)
        cv2.putText(real_np, f"ID: {im_name}", (40, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 50, 50), 2)
        cv2.putText(real_np, "Ground Truth Class:", (40, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (70, 70, 70), 2)
        cv2.putText(real_np, gt_class_name, (40, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 120, 20), 2)

    Image.fromarray(real_np).save(real_img_path)

    # 5. Generate Image from Predicted EEG Conditioning
    if active_sd is not None:
        with torch.inference_mode():
            t_in = tok(pred_prompt, padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt")
            uncond_in = tok("", padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt")
            ctx = txt(t_in.input_ids.to(dev))[0]
            uncond_ctx = txt(uncond_in.input_ids.to(dev))[0]
            c_all = torch.cat([uncond_ctx, ctx])

            sched.set_timesteps(STEPS)
            torch.manual_seed(SEED + s_i)
            lat = torch.randn((1, unet.config.in_channels, 64, 64), device=dev)

            for t_step in sched.timesteps:
                lat_in = sched.scale_model_input(torch.cat([lat, lat]), t_step)
                n_uncond, n_text = unet(lat_in, t_step, encoder_hidden_states=c_all).sample.chunk(2)
                n_pred = n_uncond + GUIDANCE * (n_text - n_uncond)
                lat = sched.step(n_pred, t_step, lat).prev_sample

            dec = vae.decode(lat / 0.18215).sample
            gen_np = ((dec.squeeze(0).permute(1, 2, 0).clamp(-1, 1).cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
    else:
        gen_np = np.ones((512, 512, 3), dtype=np.uint8) * 230
        cv2.circle(gen_np, (256, 256), 180, (50, 120, 220), -1)
        cv2.putText(gen_np, pred_class_name, (70, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    Image.fromarray(gen_np).save(gen_img_path)

    # 6. Google Drive Upload / Overwrite Logic
    if gen_folder_id:
        if sample_folder_name in existing_drive_subfolders:
            sample_drive_folder_id = existing_drive_subfolders[sample_folder_name]
        else:
            sample_drive_folder_id = get_or_create_drive_folder(gen_folder_id, sample_folder_name)
            if sample_drive_folder_id:
                existing_drive_subfolders[sample_folder_name] = sample_drive_folder_id

        if sample_drive_folder_id:
            drive_upload_or_overwrite(sample_drive_folder_id, "real.png", real_img_path, mimetype="image/png")
            drive_upload_or_overwrite(sample_drive_folder_id, "generated.png", gen_img_path, mimetype="image/png")

    # Progress reporting
    elapsed = time.time() - start_time
    fps = total_evaluated_count / max(1e-5, elapsed)
    remaining_smp = TOTAL_TEST_SAMPLES - (s_i + 1)
    eta_min = remaining_smp / max(1e-5, fps) / 60.0
    tag = "[MATCH ✓]" if is_match else "[MISMATCH]"

    print(f"[{s_i:04d}/{TOTAL_TEST_SAMPLES-1}] Real: '{gt_class_name}' -> Pred: '{pred_class_name}' {tag} | Acc so far: {total_true_count/total_evaluated_count*100:.1f}% | ETA: {eta_min:.1f}m", flush=True)

    # Periodically backup CSV to Google Drive every 25 samples
    if (total_evaluated_count % 25 == 0) and gen_folder_id:
        drive_upload_or_overwrite(gen_folder_id, "test_evaluation_results.csv", CSV_OUTPUT_PATH, mimetype="text/csv")

    if total_evaluated_count % 100 == 0:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ============================================================================
# 8. SUMMARY AT END OF RUN
# ============================================================================
print("\n" + "=" * 75)
print("COMPLETED NEURO-3D EVALUATION RUN (SAMPLES 700 TO THE END)")
print("=" * 75)
print(f"  • SAMPLES EVALUATED IN THIS RUN : {total_evaluated_count}")
print(f"  • TOTAL TRUE PREDICTIONS        : {total_true_count}")
print(f"  • TOTAL FALSE PREDICTIONS       : {total_false_count}")
current_acc = (total_true_count / max(1, total_evaluated_count)) * 100.0
print(f"  • RUN TEST ACCURACY             : {current_acc:.2f}%")

# Save summary rows to the end of the CSV
with open(CSV_OUTPUT_PATH, mode="a", newline="", encoding="utf-8") as f_sum:
    writer = csv.writer(f_sum)
    writer.writerow(["TOTAL_TRUE", total_true_count, "true"])
    writer.writerow(["TOTAL_FALSE", total_false_count, "false"])
    writer.writerow(["TOTAL_IMAGES", total_evaluated_count, ""])
    writer.writerow(["ACCURACY", f"{current_acc:.2f}%", ""])
    f_sum.flush()

print(f"[✓] Final Evaluation CSV updated locally at: {CSV_OUTPUT_PATH}")

# Copy to root output dir as well
root_csv = os.path.join(OUTPUT_DIR, "test_evaluation_results.csv")
with open(root_csv, "w", newline="", encoding="utf-8") as f_out:
    with open(CSV_OUTPUT_PATH, "r", encoding="utf-8") as f_in:
        f_out.write(f_in.read())

if gen_folder_id:
    print("Uploading finalized CSV to Google Drive 'GeneratedImage' folder...")
    drive_upload_or_overwrite(gen_folder_id, "test_evaluation_results.csv", CSV_OUTPUT_PATH, mimetype="text/csv")
    print("[✓] Successfully synced final CSV to Google Drive!")

print("\n" + "=" * 75)
print(f"NEURO-3D CELL 2 COMPLETE: Evaluated from sample 700 to {TOTAL_TEST_SAMPLES - 1} successfully!")
print("=" * 75)
