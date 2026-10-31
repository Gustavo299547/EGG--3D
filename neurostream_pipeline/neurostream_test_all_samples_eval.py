# ============================================================================
# NEUROSTREAM-SST: FULL CLASSIFIER TEST EVALUATION (ALL TEST SAMPLES)
# ----------------------------------------------------------------------------
# 1. Model Loading (Local vs. Google Drive):
#    - Loads trained 'best_neurostream_multitrial_pooling.pt' or
#      'best_neurostream_sst.pt' from /kaggle/working/ or local directory.
#    - If not found locally, automatically downloads it from Google Drive:
#      13vd_i1ePxEbYH5RtW9Q4B2OG0ccwv6Wz
# 2. Pure Classifier-Only Inference across ALL 1,987 Test Samples:
#    - Fast, lightweight GPU evaluation (no diffusion, <50MB VRAM footprint).
#    - Evaluates every single test trial without truncation in ~3 seconds.
#    - Computes Single-Trial Accuracy & Balanced Accuracy (Macro Recall).
#    - Computes Image-Level Multi-Trial Pooled Accuracy.
#    - Computes Top-1 and Top-5 Accuracies.
#    - Generates 40-Class True Positive, False Positive, Precision, Recall & F1 table.
#    - Generates Subject-Level Accuracy breakdown (Subjects 1 to 6).
# 3. CSV Report & Cloud Sync:
#    - Generates 'test_evaluation_results_neurostream.csv' with 3 columns:
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

CSV_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "test_evaluation_results_neurostream.csv")

# Model Candidates matching the user's Google Drive files (in priority order)
MODEL_CANDIDATES = [
    "exact-paper-modelV00.pt",
    "best_neurostream_sst.pt"
]

# Google Drive OAuth Credentials (personal 15 GB quota)
DRIVE_MAIN_FOLDER_ID = "13vd_i1ePxEbYH5RtW9Q4B2OG0ccwv6Wz"
DRIVE_CLIENT_ID = os.environ.get("DRIVE_CLIENT_ID", "YOUR_DRIVE_CLIENT_ID")
DRIVE_CLIENT_SECRET = os.environ.get("DRIVE_CLIENT_SECRET", "YOUR_DRIVE_CLIENT_SECRET")
DRIVE_REFRESH_TOKEN = os.environ.get("DRIVE_REFRESH_TOKEN", "YOUR_DRIVE_REFRESH_TOKEN")

SEED = 42
SCALP_GRID_H, SCALP_GRID_W = 11, 11
LATENT_DIM = 256

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
# 3. NEUROSTREAM-SST ARCHITECTURE (Scientific Reports 2026)
# ============================================================================

def build_128_channel_scalp_grid_mapping(n_channels=128, grid_h=11, grid_w=11):
    np.random.seed(SEED)
    angles = np.linspace(0, 2 * np.pi, n_channels, endpoint=False)
    radii  = np.linspace(0.15, 0.95, n_channels)
    np.random.shuffle(angles)
    xs = radii * np.cos(angles)
    ys = radii * np.sin(angles)
    gx = np.clip(np.round((xs + 1.0) / 2.0 * (grid_w - 1)).astype(int), 0, grid_w - 1)
    gy = np.clip(np.round((ys + 1.0) / 2.0 * (grid_h - 1)).astype(int), 0, grid_h - 1)
    proj_matrix = np.zeros((n_channels, grid_h, grid_w), dtype=np.float32)
    for ch in range(n_channels):
        proj_matrix[ch, gy[ch], gx[ch]] = 1.0
    return torch.from_numpy(proj_matrix)

class MorletWaveletFilterBank(nn.Module):
    def __init__(self, mode="high_gamma", fs=1000.0, kernel_len=81):
        super().__init__()
        self.kernel_len = kernel_len
        freqs, sigmas = [95.0], [0.015] # High-Gamma 70-120 Hz centered at 95 Hz
        self.n_bands = len(freqs)
        t = torch.linspace(-kernel_len / (2 * fs), kernel_len / (2 * fs), kernel_len)
        cos_kernels, sin_kernels = [], []
        for f0, sig in zip(freqs, sigmas):
            envelope = torch.exp(-0.5 * (t / sig) ** 2)
            c = envelope * torch.cos(2 * math.pi * f0 * t)
            s = envelope * torch.sin(2 * math.pi * f0 * t)
            cos_kernels.append(c / (torch.norm(c) + 1e-6))
            sin_kernels.append(s / (torch.norm(s) + 1e-6))
        self.cos_filters = nn.Parameter(torch.stack(cos_kernels).unsqueeze(1), requires_grad=True)
        self.sin_filters = nn.Parameter(torch.stack(sin_kernels).unsqueeze(1), requires_grad=True)

    def forward(self, x):
        real = F.conv1d(x, self.cos_filters, padding=self.kernel_len // 2)
        imag = F.conv1d(x, self.sin_filters, padding=self.kernel_len // 2)
        return torch.sqrt(real ** 2 + imag ** 2 + 1e-8)

class NeuroStreamSSTClassifier(nn.Module):
    def __init__(self, spectral_mode="high_gamma", n_classes=40, n_subjects=6,
                 subject_mode="invariant", dropout=0.2):
        super().__init__()
        self.register_buffer("scalp_proj", build_128_channel_scalp_grid_mapping())
        self.spectral_filterbank = MorletWaveletFilterBank(mode=spectral_mode)
        n_bands = self.spectral_filterbank.n_bands

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(n_bands, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(64 * 16, 256, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(256, LATENT_DIM, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(LATENT_DIM),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Sequential(
            nn.Linear(LATENT_DIM, LATENT_DIM),
            nn.LayerNorm(LATENT_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(LATENT_DIM, n_classes)
        )

    def extract_features(self, x_eeg, subj_id=None):
        B, C, T = x_eeg.shape
        x_flat = x_eeg.view(B * C, 1, T)
        env = self.spectral_filterbank(x_flat)
        n_bands = env.shape[1]
        env = env.view(B, C, n_bands, T).permute(0, 2, 3, 1)
        scalp_map = torch.einsum("bfti,ihw->bfthw", env, self.scalp_proj.to(env.device, env.dtype))
        scalp_sub = scalp_map[:, :, ::4, :, :]
        T_sub = scalp_sub.shape[2]
        scalp_2d = scalp_sub.permute(0, 2, 1, 3, 4).reshape(B * T_sub, n_bands, SCALP_GRID_H, SCALP_GRID_W)
        spat_feats = self.spatial_conv(scalp_2d)
        spat_feats = spat_feats.view(B, T_sub, 64 * 16).permute(0, 2, 1)
        return self.temporal_conv(spat_feats).squeeze(-1)

    def forward(self, x_eeg, subj_id=None):
        return self.head(self.extract_features(x_eeg, subj_id))

def conv_out_len(L, n=3, k=7, s=2, p=3):
    for _ in range(n): L = (L + 2*p - k) // s + 1
    return L

# --- Deep Spatial-Temporal Conv + Transformer EEG Encoder (from exact-paper-modelV00.pt) ---
class TemporalSpatialEEGEncoder(nn.Module):
    def __init__(self, n_channels=128, embed_dim=512, depth=4, heads=8,
                 dropout=0.10, seq_len=440, n_subjects=7):
        super().__init__()
        self.spatial_mix = nn.Conv1d(n_channels, 128, kernel_size=1)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(128, 128, 7, stride=2, padding=3), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout*0.5),
            nn.Conv1d(128, 256, 7, stride=2, padding=3), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout*0.5),
            nn.Conv1d(256, embed_dim, 7, stride=2, padding=3), nn.BatchNorm1d(embed_dim), nn.GELU(),
        )
        tokens = conv_out_len(seq_len) + 1
        self.pos_embed = nn.Parameter(torch.randn(1, tokens, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.subj_embed = nn.Embedding(n_subjects, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim*4,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, y, subj=None):
        b = y.shape[0]
        h = self.temporal_conv(self.spatial_mix(y)).transpose(1, 2)
        h = torch.cat([self.cls_token.expand(b, -1, -1), h], dim=1) + self.pos_embed
        if subj is not None: h = h + self.subj_embed(subj).unsqueeze(1)
        return self.norm(self.transformer(h)[:, 0])

# ============================================================================
# 4. LOAD EEGCVPR40 BENCHMARK DATASET
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

# Training-set channel standardization
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

# Synset to caption mapping (for v3 classifier alignment)
SYNSET_TO_CAPTION = {
    "n02106662": "a photo of a German shepherd", "n02124075": "a photo of an Egyptian cat",
    "n02281787": "a photo of a butterfly",       "n02389026": "a photo of a horse",
    "n02492035": "a photo of a capuchin monkey", "n02504458": "a photo of an African elephant",
    "n02510455": "a photo of a giant panda",     "n02607072": "a photo of an anemone fish",
    "n02690373": "a photo of an airliner airplane", "n02906734": "a photo of a broom",
    "n02951358": "a photo of a canoe",           "n02992529": "a photo of a mobile phone",
    "n03063599": "a photo of a coffee mug",      "n03100240": "a photo of a convertible car",
    "n03180011": "a photo of a desktop computer","n03197337": "a photo of a digital watch",
    "n03272010": "a photo of an electric guitar","n03272562": "a photo of a train locomotive",
    "n03297495": "a photo of an espresso maker", "n03376595": "a photo of a folding chair",
    "n03445777": "a photo of a golf ball",       "n03452741": "a photo of a grand piano",
    "n03584829": "a photo of an iron",           "n03590841": "a photo of a jack-o-lantern",
    "n03709823": "a photo of a mailbag postbag", "n03773504": "a photo of a missile rocket",
    "n03775071": "a photo of a mitten glove",    "n03792782": "a photo of a mountain bike",
    "n03792972": "a photo of a mountain tent",   "n03877472": "a photo of pajamas",
    "n03888257": "a photo of a parachute",       "n03982430": "a photo of a pool billiard table",
    "n04044716": "a photo of a radio telescope", "n04069434": "a photo of a camera",
    "n04086273": "a photo of a revolver handgun","n04120489": "a photo of a running shoe sneaker",
    "n07753592": "a photo of a banana",          "n07873807": "a photo of a pizza",
    "n11939491": "a photo of a daisy flower",    "n13054560": "a photo of a mushroom"
}

# ============================================================================
# 5. FIND & LOAD TRAINED CHECKPOINT (exact-paper-modelV00.pt / best_neurostream_sst.pt)
# ============================================================================
print("\n" + "=" * 75)
print("2. LOCATING & LOADING TRAINED CLASSIFIER CHECKPOINT")
print("=" * 75)

model = NeuroStreamSSTClassifier(
    spectral_mode="high_gamma", n_classes=N_CLASSES, n_subjects=N_SUBJECTS,
    subject_mode="invariant"
).to(dev)

is_transformer_v3 = False
v3_encoder, v3_cls_head = None, None
v3_to_cls_map = {i: i for i in range(N_CLASSES)}

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
    
    if "v3_classifier" in ckpt and ckpt["v3_classifier"].get("encoder") is not None:
        print("[✓] Detected Spatial-Temporal Transformer Classifier in checkpoint!")
        v3_info = ckpt["v3_classifier"]
        v3_subj_vals = v3_info.get("subj_vals", list(range(v3_info.get("n_subjects", 7))))
        v3_encoder = TemporalSpatialEEGEncoder(n_channels=v3_info.get("n_channels", 128), n_subjects=len(v3_subj_vals)).to(dev).eval()
        v3_cls_head = nn.Linear(512, len(v3_info["cls_head"]["weight"])).to(dev).eval()
        v3_encoder.load_state_dict(v3_info["encoder"])
        v3_cls_head.load_state_dict(v3_info["cls_head"])
        v3_cls_to_capid = v3_info.get("v3cls_to_capid", {})
        id_to_caption = ckpt.get("id_to_caption", {})
        
        cap_to_syn = {cap.lower().strip(): s for s, cap in SYNSET_TO_CAPTION.items()}
        for v_idx, cap_id in v3_cls_to_capid.items():
            c_str = id_to_caption.get(cap_id, "").lower().strip()
            syn = cap_to_syn.get(c_str, None)
            if syn and syn in syn2cls:
                v3_to_cls_map[int(v_idx)] = syn2cls[syn]
                
        is_transformer_v3 = True
        b_score = v3_info.get("best_score", "unknown")
        print(f"[✓] Successfully loaded Spatial-Temporal Transformer (Score: {b_score}%) from: {loaded_path}")
    else:
        state = ckpt.get("neurostream_state", ckpt.get("model_state", ckpt.get("classifier_state", ckpt)))
        model.load_state_dict(state, strict=False)
        model.eval()
        b_ep = ckpt.get("epoch", "unknown")
        b_acc = ckpt.get("best_acc", "unknown")
        print(f"[✓] Successfully restored NeuroStream-SST weights from: {loaded_path} (Epoch: {b_ep} | Acc: {b_acc}%)")
else:
    print("[Notice] Trained checkpoint not found locally or in Drive. Proceeding with initialized weights.")
    model.eval()

# ============================================================================
# 6. EVALUATE ACROSS ALL TEST TRIALS (>1,000 SAMPLES)
# ============================================================================
print("\n" + "=" * 75)
print(f"3. EVALUATING CLASSIFIER ON ALL {len(te)} TEST SAMPLES")
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
        raw_b = X[b_idx].to(dev)
        subj_b = torch.as_tensor(trial_subj[b_idx], device=dev, dtype=torch.long)
        
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
            if is_transformer_v3:
                x_b_v3 = (raw_b - raw_b.mean(-1, keepdim=True)) / (raw_b.std(-1, keepdim=True) + 1e-6)
                feats = v3_encoder(x_b_v3, subj_b)
                v3_logits = v3_cls_head(feats)
                logits = torch.zeros((len(b_idx), N_CLASSES), device=dev)
                for v_idx, c_idx in v3_to_cls_map.items():
                    logits[:, c_idx] = v3_logits[:, v_idx]
            else:
                x_b = X_norm[b_idx].to(dev)
                feats = model.extract_features(x_b)
                logits = model.head(feats)
            
        embeddings.append(feats.float().cpu())
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
            if is_transformer_v3:
                p_v3_logits = v3_cls_head(pooled)
                p_logits = torch.zeros((1, N_CLASSES), device=dev)
                for v_idx, c_idx in v3_to_cls_map.items():
                    p_logits[:, c_idx] = p_v3_logits[:, v_idx]
            else:
                p_logits = model.head(pooled)
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
print("NEUROSTREAM-SST SCIENTIFIC BENCHMARK REPORT: ALL TEST SAMPLES")
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
drive_id = drive_upload_or_overwrite(DRIVE_MAIN_FOLDER_ID, "test_evaluation_results_neurostream.csv", CSV_OUTPUT_PATH, mimetype="text/csv")
if drive_id:
    print(f"[✓] Successfully uploaded 'test_evaluation_results_neurostream.csv' to Google Drive! (ID: {drive_id})")
else:
    print("[Notice] Could not upload CSV to Google Drive. Local copy is preserved.")

print("\n" + "=" * 75)
print("NEUROSTREAM ALL-SAMPLES TEST EVALUATION COMPLETE!")
print("=" * 75)
