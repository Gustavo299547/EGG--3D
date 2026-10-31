# ============================================================================
# NEURO-3D: FULL TEST EVALUATION ACROSS ALL SAMPLES (>1,000 TRIALS)
# ----------------------------------------------------------------------------
# 1. Model Loading (Local vs. Google Drive):
#    - Loads trained 'best_neuro3d.pt' (or 'best_neuro3d_classifier.pt') from
#      /kaggle/working/ or local directory.
#    - If not found locally, automatically downloads it from Google Drive:
#      13vd_i1ePxEbYH5RtW9Q4B2OG0ccwv6Wz
# 2. Comprehensive Test Evaluation across ALL 1,987 Test Samples:
#    - Evaluates every single test trial without truncation.
#    - Computes Single-Trial Accuracy & Balanced Accuracy (Macro Recall).
#    - Computes Image-Level Multi-Trial Pooled Accuracy.
#    - Computes Top-1 and Top-5 Accuracies.
#    - Generates 40-Class True Positive, False Positive, Precision, Recall & F1 table.
#    - Generates Subject-Level Accuracy breakdown (Subjects 1 to 6).
# 3. CSV Report & Cloud Sync:
#    - Generates 'test_evaluation_results_neuro3d.csv' with 3 columns:
#      [real_image_class, predicted_class, isTrue]
#    - Records every test trial prediction row-by-row.
#    - Counts total true, total false, total images, and overall accuracy.
#    - Uploads/syncs the final evaluation CSV to Google Drive.
# ============================================================================

import os, sys, gc, glob, math, random, csv, time, subprocess, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ============================================================================
# 1. CONFIGURATION & GOOGLE DRIVE SETTINGS
# ============================================================================
DATA_ROOT = "/kaggle/input/datasets/evolutionaw/eegcvpr40/data"
OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "test_evaluation_results_neuro3d.csv")

# Model Candidates to look for (local and Drive)
MODEL_CANDIDATES = [
    "best_neuro3d.pt",
    "best_neuro3d_classifier.pt"
]

# Google Drive OAuth Credentials (personal 15 GB quota)
DRIVE_MAIN_FOLDER_ID = "13vd_i1ePxEbYH5RtW9Q4B2OG0ccwv6Wz"
DRIVE_CLIENT_ID = os.environ.get("DRIVE_CLIENT_ID", "YOUR_DRIVE_CLIENT_ID")
DRIVE_CLIENT_SECRET = os.environ.get("DRIVE_CLIENT_SECRET", "YOUR_DRIVE_CLIENT_SECRET")
DRIVE_REFRESH_TOKEN = os.environ.get("DRIVE_REFRESH_TOKEN", "YOUR_DRIVE_REFRESH_TOKEN")

SEED = 42
EMBED_DIM = 384
N_3D_POINTS = 1024
DROPOUT = 0.15

dev = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = (dev == "cuda")

def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

seed_everything(SEED)

# ============================================================================
# 2. GOOGLE DRIVE SYNC HELPERS
# ============================================================================
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

def drive_upload_or_overwrite(parent_folder_id, filename, local_path, mimetype=None):
    if not os.path.exists(local_path):
        return None
    service = get_drive_service()
    if not service: return None
    try:
        from googleapiclient.http import MediaFileUpload
        if mimetype is None:
            if filename.endswith(".csv"): mimetype = "text/csv"
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
# 3. NEURO-3D ARCHITECTURE (CVPR 2025)
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
# 4. LOAD EEGCVPR40 DATASET & BENCHMARK TEST SPLIT
# ============================================================================
print("\n" + "=" * 75)
print("1. LOADING EEGCVPR40 BENCHMARK DATASET")
print("=" * 75)

pths = glob.glob(os.path.join(DATA_ROOT, "**", "*.pth"), recursive=True)
if not pths: pths = glob.glob(os.path.join(".", "**", "*.pth"), recursive=True)
if not pths: raise FileNotFoundError(f"Dataset not found in {DATA_ROOT} or local directory.")

bn = os.path.basename
eeg_file = max([p for p in pths if "split" not in bn(p).lower()], key=os.path.getsize)
split_file = next((p for p in sorted([p for p in pths if "split" in bn(p).lower()],
                                     key=lambda p: "all" not in bn(p).lower())), None)

print(f"EEG Data File   : {eeg_file}")
print(f"Split File      : {split_file}")

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

print(f"Total Test Trials Loaded : {len(te)} trials across {len(set(IMG[te]))} unique images")

# Training-set channel standardization (preserving true visual amplitude variation)
train_signals = X[tr].float()
channel_mean = train_signals.mean(dim=(0, 2), keepdim=True)
channel_std  = train_signals.std(dim=(0, 2), keepdim=True) + 1e-6
X_norm = (X.float() - channel_mean) / channel_std
del train_signals

# Synset to clean human-readable class name mapping
SYNSET_TO_NAME = {
    "n02106662": "German shepherd", "n02124075": "Egyptian cat", "n02281787": "butterfly",
    "n02389026": "horse", "n02492035": "capuchin monkey", "n02504458": "African elephant",
    "n02510455": "giant panda", "n02607072": "anemone fish", "n02690373": "airliner airplane",
    "n02906734": "broom", "n02951358": "canoe", "n02992529": "mobile phone",
    "n03063599": "coffee mug", "n03100240": "convertible car", "n03180011": "desktop computer",
    "n03197337": "digital watch", "n03272010": "electric guitar", "n03272562": "train locomotive",
    "n03297495": "espresso maker", "n03376595": "folding chair", "n03445777": "golf ball",
    "n03452741": "grand piano", "n03584829": "iron", "n03590841": "jack-o-lantern",
    "n03709823": "mailbag postbag", "n03773504": "missile rocket", "n03775071": "mitten glove",
    "n03792782": "mountain bike", "n03792972": "mountain tent", "n03877472": "pajamas",
    "n03888257": "parachute", "n03982430": "pool billiard table", "n04044716": "radio telescope",
    "n04069434": "camera", "n04086273": "revolver handgun", "n04120489": "running shoe sneaker",
    "n07753592": "banana", "n07873807": "pizza", "n11939491": "daisy flower", "n13054560": "mushroom"
}

def get_clean_name(cls_idx):
    syn = cls2syn.get(cls_idx, "")
    return SYNSET_TO_NAME.get(syn, syn)

# ============================================================================
# 5. FIND & LOAD TRAINED NEURO-3D MODEL CHECKPOINT
# ============================================================================
print("\n" + "=" * 75)
print("2. LOCATING & LOADING TRAINED NEURO-3D MODEL CHECKPOINT")
print("=" * 75)

model = Neuro3DClassifierAndDecoder(
    n_classes=N_CLASSES, n_subjects=N_SUBJECTS, embed_dim=EMBED_DIM,
    n_points=N_3D_POINTS, subject_mode="invariant", dropout=DROPOUT
).to(dev)

loaded_path = None
for cand in MODEL_CANDIDATES:
    cand_local = os.path.join(OUTPUT_DIR, cand)
    if os.path.exists(cand_local):
        loaded_path = cand_local
        print(f"[✓] Found local model checkpoint: {cand_local} ({os.path.getsize(cand_local):,} bytes)")
        break
    elif os.path.exists(cand):
        loaded_path = cand
        print(f"[✓] Found local model checkpoint: {cand} ({os.path.getsize(cand):,} bytes)")
        break
    else:
        print(f"  Searching Google Drive for '{cand}' ...")
        drive_f = drive_find_file(cand, DRIVE_MAIN_FOLDER_ID)
        if drive_f:
            print(f"  Found '{drive_f['name']}' in Google Drive! Downloading...")
            if drive_download_file(drive_f["id"], cand_local, cand):
                loaded_path = cand_local
                break

if loaded_path and os.path.exists(loaded_path):
    ckpt = torch.load(loaded_path, map_location=dev, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    b_ep = ckpt.get("epoch", "unknown")
    print(f"[✓] Successfully restored Neuro-3D weights from: {loaded_path} (Trained Epoch: {b_ep})")
else:
    print("[Notice] Trained checkpoint not found locally or in Drive. Proceeding with initialized weights.")

model.eval()

# ============================================================================
# 6. EVALUATE ACROSS ALL TEST TRIALS (>1,000 SAMPLES)
# ============================================================================
print("\n" + "=" * 75)
print(f"3. EVALUATING NEURO-3D ON ALL {len(te)} TEST SAMPLES")
print("=" * 75)

test_imgs_uniq = sorted(set(IMG[te]))
test_img2trials = {im: np.where(IMG[te] == im)[0] for im in test_imgs_uniq}

all_preds, all_trues, all_top5 = [], [], []
embeddings = []

bs = 64
start_eval_time = time.time()

with torch.no_grad():
    for k in range(0, len(te), bs):
        b_idx = te[k:k + bs]
        x_b = X_norm[b_idx].to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
            z_s = model.static_pathway(x_b)
            logits = model.classifier(z_s)
            
        embeddings.append(z_s.float().cpu())
        top1 = logits.argmax(dim=-1).cpu().numpy()
        top5 = torch.topk(logits, k=min(5, N_CLASSES), dim=-1).indices.cpu().numpy()
        trues = trial_cls[b_idx]
        
        all_preds.extend(top1)
        all_trues.extend(trues)
        for t_val, t5 in zip(trues, top5):
            all_top5.append(t_val in t5)

embeddings = torch.cat(embeddings, dim=0)
all_preds = np.array(all_preds)
all_trues = np.array(all_trues)

total_test_samples = len(all_preds)
total_true_single = int(np.sum(all_preds == all_trues))
total_false_single = total_test_samples - total_true_single
single_trial_acc = (total_true_single / total_test_samples) * 100.0
single_trial_top5 = np.mean(all_top5) * 100.0

# Multi-Trial Image-Level Pooling
pool_preds, pool_trues, pool_top5 = [], [], []
for im, tr_idxs in test_img2trials.items():
    pooled = embeddings[tr_idxs].mean(dim=0, keepdim=True).to(dev)
    true_c = trial_cls[te[tr_idxs[0]]]
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
            p_logits = model.classifier(pooled)
    p_pred = p_logits.argmax(dim=-1).item()
    p_top5 = torch.topk(p_logits, k=min(5, N_CLASSES), dim=-1).indices.cpu().numpy()[0]
    
    pool_preds.append(p_pred)
    pool_trues.append(true_c)
    pool_top5.append(true_c in p_top5)

pool_preds = np.array(pool_preds)
pool_trues = np.array(pool_trues)
total_true_pool = int(np.sum(pool_preds == pool_trues))
total_false_pool = len(pool_trues) - total_true_pool
pool_acc = (total_true_pool / len(pool_trues)) * 100.0
pool_top5_acc = np.mean(pool_top5) * 100.0

# Balanced accuracy
recalls_list = [np.mean(all_preds[all_trues == c] == c) for c in range(N_CLASSES) if np.sum(all_trues == c) > 0]
balanced_acc = np.mean(recalls_list) * 100.0

eval_duration = time.time() - start_eval_time

# ============================================================================
# 7. PRINT MASTER SCIENTIFIC EVALUATION REPORT
# ============================================================================
print("\n" + "=" * 80)
print("NEURO-3D SCIENTIFIC BENCHMARK REPORT: ALL TEST SAMPLES")
print("=" * 80)
print(f"  • TOTAL TEST TRIALS EVALUATED       : {total_test_samples} trials across {len(test_imgs_uniq)} unique images")
print(f"  • INFERENCE SPEED                   : {total_test_samples / max(1e-5, eval_duration):.1f} trials/second ({eval_duration:.2f}s total)")
print(f"  --------------------------------------------------------------------------")
print(f"  • IMAGE-LEVEL POOLED TEST ACCURACY  : {total_true_pool} / {len(pool_trues)} ({pool_acc:.2f}%)")
print(f"  • Image-Level Top-5 Accuracy       : {int(np.sum(pool_top5))} / {len(pool_top5)} ({pool_top5_acc:.2f}%)")
print(f"  --------------------------------------------------------------------------")
print(f"  • SINGLE-TRIAL TEST ACCURACY       : {total_true_single} / {total_test_samples} ({single_trial_acc:.2f}%)")
print(f"  • Single-Trial Balanced Accuracy   : {balanced_acc:.2f}%")
print(f"  • Single-Trial Top-5 Accuracy      : {single_trial_top5:.2f}%")
print(f"  --------------------------------------------------------------------------")
print(f"  • Random Chance Baseline (40 Cls)  : 2.50%")
print("=" * 80)

# Full 40-Class Breakdown Table
print("\n[CLASS-LEVEL TRUE POSITIVE & RECALL BREAKDOWN (Image-Level)]")
print(f"{'Class Name':<22} | {'Total':<6} | {'TP (Right)':<10} | {'FP (Wrong)':<10} | {'Recall':<10} | {'Precision':<10} | {'F1-Score'}")
print("-" * 85)

for c in range(N_CLASSES):
    c_name = get_clean_name(c)
    tp = int(np.sum((pool_preds == c) & (pool_trues == c)))
    fp = int(np.sum((pool_preds == c) & (pool_trues != c)))
    fn = int(np.sum((pool_preds != c) & (pool_trues == c)))
    total_c = tp + fn
    rec = tp / max(1, total_c) * 100.0
    prec = tp / max(1, tp + fp) * 100.0
    f1 = 2 * prec * rec / max(1e-5, prec + rec)
    print(f"{c_name:<22} | {total_c:<6} | {tp:<10} | {fp:<10} | {rec:>8.1f}% | {prec:>8.1f}% | {f1:>6.1f}%")

# Subject-Level Breakdown Table
print("\n[SUBJECT-LEVEL ACCURACY BREAKDOWN (Correctly Labeled)]")
print(f"{'Subject ID':<15} | {'Test Trials':<15} | {'Single-Trial Accuracy'}")
print("-" * 55)
for s_idx in range(N_SUBJECTS):
    mask = (trial_subj[te] == s_idx)
    if np.sum(mask) > 0:
        s_acc = np.mean(all_preds[mask] == all_trues[mask]) * 100.0
        print(f"Subject {subj_vals[s_idx]:<8} | {np.sum(mask):<15} | {s_acc:.2f}%")
print("-" * 55)

# ============================================================================
# 8. WRITE COMPLETE 3-COLUMN CSV & UPLOAD TO GOOGLE DRIVE
# ============================================================================
print("\n" + "=" * 75)
print("COMPILING FULL 3-COLUMN EVALUATION CSV REPORT")
print("=" * 75)

with open(CSV_OUTPUT_PATH, mode="w", newline="", encoding="utf-8") as f_csv:
    writer = csv.writer(f_csv)
    # Header
    writer.writerow(["real_image_class", "predicted_class", "isTrue"])
    
    # Write row for each of the test trials
    for i in range(total_test_samples):
        r_name = get_clean_name(all_trues[i])
        p_name = get_clean_name(all_preds[i])
        match_str = "true" if all_trues[i] == all_preds[i] else "false"
        writer.writerow([r_name, p_name, match_str])
        
    # Summary rows strictly adhering to 3 columns
    writer.writerow(["TOTAL_TRUE", total_true_single, "true"])
    writer.writerow(["TOTAL_FALSE", total_false_single, "false"])
    writer.writerow(["TOTAL_IMAGES", total_test_samples, ""])
    writer.writerow(["SINGLE_TRIAL_ACCURACY", f"{single_trial_acc:.2f}%", ""])
    writer.writerow(["POOLED_ACCURACY", f"{pool_acc:.2f}%", ""])
    writer.writerow(["TOP5_ACCURACY", f"{single_trial_top5:.2f}%", ""])

print(f"[✓] Complete evaluation CSV saved locally to: {CSV_OUTPUT_PATH}")

# Upload to Google Drive
print(f"Uploading evaluation CSV to Google Drive folder '{DRIVE_MAIN_FOLDER_ID}'...")
drive_id = drive_upload_or_overwrite(DRIVE_MAIN_FOLDER_ID, "test_evaluation_results_neuro3d.csv", CSV_OUTPUT_PATH, mimetype="text/csv")
if drive_id:
    print(f"[✓] Successfully uploaded 'test_evaluation_results_neuro3d.csv' to Google Drive! (ID: {drive_id})")
else:
    print("[Notice] Could not upload CSV to Google Drive. Local copy is preserved.")

print("\n" + "=" * 75)
print("NEURO-3D ALL-SAMPLES TEST EVALUATION COMPLETE!")
print("=" * 75)
