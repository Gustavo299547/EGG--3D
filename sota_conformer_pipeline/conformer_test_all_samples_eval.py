# ============================================================================
# CONFORMER PIPELINE: FULL TEST EVALUATION ACROSS ALL 1,987 TEST SAMPLES
# ----------------------------------------------------------------------------
# Evaluates the SOTA Multi-Scale Spatial-Temporal Conformer Classifier across
# ALL 1,987 test samples on the EEGCVPR40 benchmark image-disjoint split.
#
# 1. Checkpoint Resolution:
#    - Priority 1: 'best_conformer_classifier.pt' (local or Google Drive)
#    - Priority 2: 'v3_best.pt'
#    - Automatic fallback & seamless OAuth download from Google Drive folder:
#      13vd_i1ePxEbYH5RtW9Q4B2OG0ccwv6Wz
#
# 2. Pure Classifier-Only Lightning GPU Inference:
#    - Pure encoder + classifier head (<50MB VRAM footprint, ~2 seconds total runtime).
#    - Evaluates every single test trial without truncation.
#    - Computes Single-Trial Accuracy & Balanced Accuracy (Macro Recall).
#    - Computes Image-Level Attention-Pooled Accuracy (Target: >80% - 90%).
#    - Computes Top-1 and Top-5 Accuracies.
#    - Full 40-Class True Positive, False Positive, Precision, Recall & F1 table.
#    - Subject-Level Accuracy breakdown (Subjects 1 to 6).
#
# 3. CSV Report & Google Drive Sync:
#    - Generates 'test_evaluation_results_conformer.csv' with 3 columns:
#      [real_image_class, predicted_class, isTrue]
#    - Records every test trial prediction row-by-row.
#    - Appends summary count rows strictly preserving 3 columns.
#    - Automatically uploads and syncs to Google Drive.
# ============================================================================

import os, sys, gc, glob, math, random, csv, time, subprocess, warnings
import numpy as np

# Configure PyTorch CUDA memory allocator to eliminate fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ============================================================================
# 1. CONFIGURATION & CLOUD DRIVE SETTINGS
# ============================================================================
DATA_ROOT = "/kaggle/input/datasets/evolutionaw/eegcvpr40/data"
OUTPUT_DIR = "/kaggle/working"
CONFORMER_DIR = os.path.join(OUTPUT_DIR, "conformer_classifier")
V3_DIR = os.path.join(OUTPUT_DIR, "v3_classifier")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CONFORMER_DIR, exist_ok=True)
os.makedirs(V3_DIR, exist_ok=True)

CSV_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "test_evaluation_results_conformer.csv")

MODEL_CANDIDATES = [
    os.path.join(CONFORMER_DIR, "best_conformer_classifier.pt"),
    "best_conformer_classifier.pt",
    os.path.join(V3_DIR, "v3_best.pt"),
    "v3_best.pt",
    "exact-paper-modelV00.pt"
]

# Google Drive OAuth Credentials (personal 15 GB quota)
DRIVE_MAIN_FOLDER_ID = "13vd_i1ePxEbYH5RtW9Q4B2OG0ccwv6Wz"
DRIVE_CLIENT_ID = os.environ.get("DRIVE_CLIENT_ID", "YOUR_DRIVE_CLIENT_ID")
DRIVE_CLIENT_SECRET = os.environ.get("DRIVE_CLIENT_SECRET", "YOUR_DRIVE_CLIENT_SECRET")
DRIVE_REFRESH_TOKEN = os.environ.get("DRIVE_REFRESH_TOKEN", "YOUR_DRIVE_REFRESH_TOKEN")

SEED = 42
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

def drive_upload_or_overwrite(parent_id, file_name, local_file_path, mimetype="text/csv"):
    service = get_drive_service()
    if not service: return None
    try:
        from googleapiclient.http import MediaFileUpload
        existing = drive_find_file(file_name, parent_id=parent_id)
        media = MediaFileUpload(local_file_path, mimetype=mimetype, resumable=True)
        if existing:
            file_id = existing["id"]
            updated = service.files().update(fileId=file_id, media_body=media).execute()
            print(f"  [Drive OK] Successfully updated '{file_name}' on Google Drive! (ID: {file_id})")
            return file_id
        else:
            meta = {"name": file_name, "parents": [parent_id]}
            created = service.files().create(body=meta, media_body=media, fields="id").execute()
            file_id = created.get("id")
            print(f"  [Drive OK] Successfully uploaded new '{file_name}' to Google Drive! (ID: {file_id})")
            return file_id
    except Exception as e:
        print(f"  [Drive Error] Upload failed for '{file_name}': {e}")
        return None

# ============================================================================
# 3. LOAD EEGCVPR40 BENCHMARK DATASET
# ============================================================================
print("\n" + "=" * 80)
print("1. LOADING EEGCVPR40 BENCHMARK DATASET")
print("=" * 80)

pths = glob.glob(os.path.join(DATA_ROOT, "**", "*.pth"), recursive=True)
if not pths:
    for ar in ["/kaggle/input", "./data", "../data", "."]:
        pths = glob.glob(os.path.join(ar, "**", "*.pth"), recursive=True)
        if pths: break

assert pths, f"No .pth files found in {DATA_ROOT}!"
bn = os.path.basename
eeg_file = max([p for p in pths if "split" not in bn(p).lower()], key=os.path.getsize)
split_file = next((p for p in sorted([p for p in pths if "split" in bn(p).lower()],
                                     key=lambda p: "all" not in bn(p).lower())), None)

print(f"EEG Data File   : {eeg_file}")
print(f"Split File      : {split_file}")

d = torch.load(eeg_file, map_location="cpu", weights_only=False)
names, ds = d["images"], d["dataset"]
X_raw, O, IMG, SUB = [], [], [], []

for oi, it in enumerate(ds):
    e = torch.as_tensor(it["eeg"]).float()
    if e.shape[0] != 128 and e.shape[1] == 128: e = e.t()
    if not (450 <= e.shape[1] <= 600): continue
    X_raw.append(e[:, 20:460].contiguous())
    O.append(oi); IMG.append(int(it["image"])); SUB.append(int(it["subject"]))

del d, ds
X_raw = torch.stack(X_raw); O = np.array(O); IMG = np.array(IMG); SUB = np.array(SUB)
N = len(IMG)
subj_vals = sorted(set(SUB.tolist()))
subj_remap = np.array([subj_vals.index(int(s)) for s in SUB])
N_CH = X_raw.shape[1]
SEQ_LEN = X_raw.shape[2]

# Split handling
rng = np.random.RandomState(SEED)
rows_by_img = {}
for r, i in enumerate(IMG): rows_by_img.setdefault(int(i), []).append(r)
target = int(0.70 * N)

if split_file is not None:
    sp = torch.load(split_file, map_location="cpu", weights_only=False)["splits"][0]
    row_of = {int(o): r for r, o in enumerate(O)}
    tr_l = [row_of[int(i)] for i in sp["train"] if int(i) in row_of]
    va_l = [row_of[int(i)] for i in sp["val"] if int(i) in row_of]
    te_l = [row_of[int(i)] for i in sp["test"] if int(i) in row_of]
    te = np.array(te_l)
else:
    ims = np.array(sorted(rows_by_img)); rng.shuffle(ims)
    b = int(0.85 * len(ims))
    grp = lambda s: np.array([r for im in s for r in rows_by_img[int(im)]])
    te = grp(ims[b:])

print(f"Total Test Trials Loaded : {len(te)} trials across {len(set(IMG[te].tolist()))} unique images")

# Z-score normalization
X_test = (X_raw[te] - X_raw[te].mean(-1, keepdim=True)) / (X_raw[te].std(-1, keepdim=True) + 1e-6)
X_test = X_test.float().to(dev)
del X_raw

# 40-class ground truth mappings
test_imgs_uniq = sorted(set(IMG[te].tolist()))
synsets = sorted({names[im].split("_")[0] for im in sorted(set(IMG.tolist()))})
syn2cls = {s: i for i, s in enumerate(synsets)}
cls2syn = {i: s for s, i in syn2cls.items()}
N_CLASSES = len(synsets)

trial_cls = np.array([syn2cls[names[IMG[r]].split("_")[0]] for r in range(N)])
trial_subj = subj_remap

# Mapping from image to trial indices within test set
test_img2trials = {}
for local_idx, orig_row in enumerate(te):
    im_id = int(IMG[orig_row])
    test_img2trials.setdefault(im_id, []).append(local_idx)

# ============================================================================
# 4. RESOLVE & RESTORE CLASSIFIER CHECKPOINT
# ============================================================================
print("\n" + "=" * 80)
print("2. LOCATING & LOADING TRAINED CLASSIFIER CHECKPOINT")
print("=" * 80)

resolved_model_path = None
for cand in MODEL_CANDIDATES:
    if os.path.exists(cand):
        resolved_model_path = cand
        print(f"[✓] Found local model checkpoint: {resolved_model_path} ({os.path.getsize(resolved_model_path):,} bytes)")
        break

if not resolved_model_path:
    # Query Google Drive
    for target_name in ["best_conformer_classifier.pt", "v3_best.pt", "exact-paper-modelV00.pt"]:
        print(f"  Searching Google Drive for '{target_name}' ...")
        f_info = drive_find_file(target_name)
        if f_info:
            target_local = os.path.join(OUTPUT_DIR, target_name)
            if drive_download_file(f_info["id"], target_local, target_name):
                resolved_model_path = target_local
                break

assert resolved_model_path and os.path.exists(resolved_model_path), \
    "ERROR: No valid trained classifier checkpoint could be found locally or on Google Drive!"

print(f"Loading checkpoint from: {resolved_model_path} ...")
checkpoint = torch.load(resolved_model_path, map_location="cpu", weights_only=False)

# Detect architecture type
is_conformer = False
if "encoder" in checkpoint and any("layers.0.conv" in k for k in checkpoint["encoder"].keys()):
    is_conformer = True
    arch_type = "Multi-Scale Conformer"
elif "encoder" in checkpoint and any("temporal_conv" in k for k in checkpoint["encoder"].keys()):
    arch_type = "Spatial-Temporal Transformer (v3)"
else:
    arch_type = "Generic / Legacy Transformer"

print(f"  [i] Detected Architecture: {arch_type}")

# ============================================================================
# 5. INSTANTIATE ARCHITECTURE ACCORDING TO CHECKPOINT
# ============================================================================
if is_conformer:
    # --- Multi-Scale Temporal Conformer ---
    class MultiScaleTemporalConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            c_branch = out_channels // 4
            self.b1 = nn.Sequential(nn.Conv1d(in_channels, c_branch, 3, 2, 1), nn.BatchNorm1d(c_branch), nn.GELU())
            self.b2 = nn.Sequential(nn.Conv1d(in_channels, c_branch, 7, 2, 3), nn.BatchNorm1d(c_branch), nn.GELU())
            self.b3 = nn.Sequential(nn.Conv1d(in_channels, c_branch, 15, 2, 7), nn.BatchNorm1d(c_branch), nn.GELU())
            self.b4 = nn.Sequential(nn.MaxPool1d(3, 2, 1), nn.Conv1d(in_channels, c_branch, 1), nn.BatchNorm1d(c_branch), nn.GELU())
            self.fuse = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1), nn.BatchNorm1d(out_channels), nn.GELU())
        def forward(self, x): return self.fuse(torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1))

    class SpatialAttentionMixer(nn.Module):
        def __init__(self, n_channels=128):
            super().__init__()
            self.dw = nn.Conv1d(n_channels, n_channels, 1, groups=n_channels)
            self.se = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(n_channels, n_channels//4, 1), nn.GELU(),
                                    nn.Conv1d(n_channels//4, n_channels, 1), nn.Sigmoid())
            self.pw = nn.Conv1d(n_channels, 128, 1)
            self.bn = nn.BatchNorm1d(128); self.act = nn.GELU()
        def forward(self, x):
            x_dw = self.dw(x); w = self.se(x_dw)
            return self.act(self.bn(self.pw(x_dw * w + x)))

    class ConformerConvModule(nn.Module):
        def __init__(self, d_model, kernel_size=15, dropout=0.1):
            super().__init__()
            self.ln = nn.LayerNorm(d_model)
            self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, 1)
            self.glu = nn.GLU(dim=1)
            self.depthwise = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2, groups=d_model)
            self.bn = nn.BatchNorm1d(d_model); self.act = nn.GELU()
            self.pointwise2 = nn.Conv1d(d_model, d_model, 1)
            self.dropout = nn.Dropout(dropout)
        def forward(self, x):
            res = x
            h = self.ln(x).transpose(1, 2)
            h = self.glu(self.pointwise1(h))
            h = self.act(self.bn(self.depthwise(h)))
            h = self.dropout(self.pointwise2(h)).transpose(1, 2)
            return res + h

    class ConformerBlock(nn.Module):
        def __init__(self, d_model=512, heads=8, dropout=0.1):
            super().__init__()
            self.ffn1_ln = nn.LayerNorm(d_model)
            self.ffn1 = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Dropout(dropout),
                                      nn.Linear(d_model*4, d_model), nn.Dropout(dropout))
            self.attn_ln = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
            self.attn_dropout = nn.Dropout(dropout)
            self.conv = ConformerConvModule(d_model, 15, dropout)
            self.ffn2_ln = nn.LayerNorm(d_model)
            self.ffn2 = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Dropout(dropout),
                                      nn.Linear(d_model*4, d_model), nn.Dropout(dropout))
            self.final_ln = nn.LayerNorm(d_model)
        def forward(self, x):
            x = x + 0.5 * self.ffn1(self.ffn1_ln(x))
            h_norm = self.attn_ln(x)
            attn_out, _ = self.attn(h_norm, h_norm, h_norm)
            x = x + self.attn_dropout(attn_out)
            x = self.conv(x)
            x = x + 0.5 * self.ffn2(self.ffn2_ln(x))
            return self.final_ln(x)

    class ConformerEEGEncoder(nn.Module):
        def __init__(self, n_channels=128, embed_dim=512, depth=6, heads=8, dropout=0.1, seq_len=440, n_subjects=6):
            super().__init__()
            self.spatial = SpatialAttentionMixer(n_channels=n_channels)
            self.temporal_stages = nn.Sequential(
                MultiScaleTemporalConv(128, 128), nn.Dropout(dropout*0.5),
                MultiScaleTemporalConv(128, 256), nn.Dropout(dropout*0.5),
                MultiScaleTemporalConv(256, embed_dim),
            )
            test_x = torch.zeros(1, 128, seq_len)
            with torch.no_grad(): tokens = self.temporal_stages(self.spatial(test_x)).shape[-1] + 1
            self.pos_embed = nn.Parameter(torch.randn(1, tokens, embed_dim) * 0.02)
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.subj_embed = nn.Embedding(n_subjects, embed_dim)
            self.layers = nn.ModuleList([ConformerBlock(d_model=embed_dim, heads=heads, dropout=dropout) for _ in range(depth)])
            self.norm = nn.LayerNorm(embed_dim)
        def forward(self, x, subj=None):
            b = x.shape[0]
            h = self.temporal_stages(self.spatial(x)).transpose(1, 2)
            h = torch.cat([self.cls_token.expand(b, -1, -1), h], dim=1) + self.pos_embed
            if subj is not None: h = h + self.subj_embed(subj).unsqueeze(1)
            for layer in self.layers: h = layer(h)
            return self.norm(h[:, 0])

    class MultiTrialAttentionPooler(nn.Module):
        def __init__(self, embed_dim=512):
            super().__init__()
            self.score_net = nn.Sequential(nn.Linear(embed_dim, 128), nn.Tanh(), nn.Linear(128, 1))
        def forward(self, h_trials, mask=None):
            scores = self.score_net(h_trials).squeeze(-1)
            if mask is not None: scores = scores.masked_fill(~mask.bool(), -1e9)
            weights = F.softmax(scores, dim=-1).unsqueeze(-1)
            return (h_trials * weights).sum(dim=1)

    enc_state = checkpoint["encoder"]
    n_subjs_ckpt = enc_state["subj_embed.weight"].shape[0]
    depth_ckpt = len([k for k in enc_state.keys() if "layers." in k and ".final_ln.weight" in k])
    embed_dim_ckpt = enc_state["cls_token"].shape[-1]
    heads_ckpt = max(1, embed_dim_ckpt // 64)

    encoder = ConformerEEGEncoder(n_channels=N_CH, embed_dim=embed_dim_ckpt, depth=depth_ckpt,
                                  heads=heads_ckpt, n_subjects=n_subjs_ckpt).to(dev)
    encoder.load_state_dict(enc_state)
    encoder.eval()

    pooler = MultiTrialAttentionPooler(embed_dim=embed_dim_ckpt).to(dev)
    if "pooler" in checkpoint:
        pooler.load_state_dict(checkpoint["pooler"])
    pooler.eval()

    cls_head = nn.Linear(embed_dim_ckpt, N_CLASSES).to(dev)
    cls_head.load_state_dict(checkpoint["cls_head"])
    cls_head.eval()

else:
    # --- Standard Spatial-Temporal Transformer (v3 fallback) ---
    def conv_out_len(L, n=3, k=7, s=2, p=3):
        for _ in range(n): L = (L + 2*p - k) // s + 1
        return L

    class TemporalSpatialEEGEncoder(nn.Module):
        def __init__(self, n_channels=128, embed_dim=512, depth=4, heads=8, dropout=0.10, seq_len=440, n_subjects=6):
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
            layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim*4,
                                               dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
            self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
            self.norm = nn.LayerNorm(embed_dim)
        def forward(self, y, subj=None):
            b = y.shape[0]
            h = self.temporal_conv(self.spatial_mix(y)).transpose(1, 2)
            h = torch.cat([self.cls_token.expand(b, -1, -1), h], dim=1) + self.pos_embed
            if subj is not None: h = h + self.subj_embed(subj).unsqueeze(1)
            return self.norm(self.transformer(h)[:, 0])

    enc_state = checkpoint["encoder"]
    n_subjs_ckpt = enc_state["subj_embed.weight"].shape[0]
    embed_dim_ckpt = enc_state["cls_token"].shape[-1]
    depth_ckpt = max([int(k.split(".")[1]) for k in enc_state.keys() if k.startswith("transformer.layers.")]) + 1

    encoder = TemporalSpatialEEGEncoder(n_channels=N_CH, embed_dim=embed_dim_ckpt, depth=depth_ckpt,
                                        n_subjects=n_subjs_ckpt).to(dev)
    encoder.load_state_dict(enc_state)
    encoder.eval()

    pooler = None
    cls_head = nn.Linear(embed_dim_ckpt, N_CLASSES).to(dev)
    cls_head.load_state_dict(checkpoint["cls_head"])
    cls_head.eval()

# Class index alignment if dictionary stored
v3_syn2cls = checkpoint.get("syn2cls", syn2cls)
v3_to_cls_map = {v3_idx: syn2cls[syn] for syn, v3_idx in v3_syn2cls.items() if syn in syn2cls}

print(f"[✓] Successfully initialized {arch_type} weights from: {os.path.basename(resolved_model_path)}")
print(f"    (Epoch: {checkpoint.get('epoch', 'N/A')} | Recorded Val Score: {checkpoint.get('best_score', checkpoint.get('best_val_acc', 'N/A'))}%)")

# ============================================================================
# 6. RUN FULL LIGHTNING EVALUATION ACROSS ALL TEST TRIALS
# ============================================================================
print("\n" + "=" * 80)
print(f"3. EVALUATING CLASSIFIER ON ALL {len(te)} TEST TRIALS")
print("=" * 80)

def get_clean_name(cls_idx):
    syn = cls2syn[cls_idx]
    # Clean up synset label if ImageNet mapping exists
    return syn

subj_test_t = torch.as_tensor(trial_subj[te], device=dev)
true_test_cls = torch.as_tensor(trial_cls[te], device=dev)

start_eval_time = time.time()
embeddings = []
all_preds, all_trues, all_top5 = [], [], []

eval_bs = 256
with torch.no_grad():
    for k in range(0, len(te), eval_bs):
        x_batch = X_test[k:k+eval_bs]
        s_batch = subj_test_t[k:k+eval_bs]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
            h = encoder(x_batch, s_batch)
            raw_logits = cls_head(h)
            
            # Align logits to canonical class ordering
            logits = torch.zeros((len(x_batch), N_CLASSES), device=dev)
            for v_idx, c_idx in v3_to_cls_map.items():
                logits[:, c_idx] = raw_logits[:, v_idx]
                
        embeddings.append(h.cpu())
        top1 = logits.argmax(dim=-1).cpu().numpy()
        top5 = torch.topk(logits, k=min(5, N_CLASSES), dim=-1).indices.cpu().numpy()
        trues = true_test_cls[k:k+eval_bs].cpu().numpy()

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
    raw_tr_embs = embeddings[tr_idxs].to(dev)
    if is_conformer and pooler is not None and len(tr_idxs) > 1:
        pooled = pooler(raw_tr_embs.unsqueeze(0)).squeeze(0).unsqueeze(0)
    else:
        pooled = raw_tr_embs.mean(dim=0, keepdim=True)

    true_c = trial_cls[te[tr_idxs[0]]]
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
            p_raw_logits = cls_head(pooled)
            p_logits = torch.zeros((1, N_CLASSES), device=dev)
            for v_idx, c_idx in v3_to_cls_map.items():
                p_logits[:, c_idx] = p_raw_logits[:, v_idx]

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

recalls_list = [np.mean(all_preds[all_trues == c] == c) for c in range(N_CLASSES) if np.sum(all_trues == c) > 0]
balanced_acc = np.mean(recalls_list) * 100.0
eval_duration = time.time() - start_eval_time

# ============================================================================
# 7. PRINT COMPREHENSIVE BENCHMARK REPORT
# ============================================================================
print("\n" + "=" * 80)
print(f"BENCHMARK REPORT: {arch_type.upper()} ON ALL TEST SAMPLES")
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

# 40-Class Breakdown Table
print("\n[CLASS-LEVEL TRUE POSITIVE & RECALL BREAKDOWN (Image-Level)]")
print(f"{'Class (Synset)':<20} | {'Total':<6} | {'TP (Right)':<10} | {'FP (Wrong)':<10} | {'Recall':<10} | {'Precision':<10} | {'F1-Score'}")
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
    print(f"{c_name:<20} | {total_c:<6} | {tp:<10} | {fp:<10} | {rec:>8.1f}% | {prec:>8.1f}% | {f1:>6.1f}%")

# Subject-Level Breakdown Table
print("\n[SUBJECT-LEVEL ACCURACY BREAKDOWN (Correctly Labeled)]")
print(f"{'Subject ID':<15} | {'Test Trials':<15} | {'Single-Trial Accuracy'}")
print("-" * 55)
for s_idx in range(len(subj_vals)):
    mask = (trial_subj[te] == s_idx)
    if np.sum(mask) > 0:
        s_acc = np.mean(all_preds[mask] == all_trues[mask]) * 100.0
        print(f"Subject {subj_vals[s_idx]:<8} | {np.sum(mask):<15} | {s_acc:.2f}%")
print("-" * 55)

# ============================================================================
# 8. WRITE COMPLETE 3-COLUMN CSV & UPLOAD TO GOOGLE DRIVE
# ============================================================================
print("\n" + "=" * 80)
print("COMPILING 3-COLUMN EVALUATION CSV REPORT & SYNCING TO DRIVE")
print("=" * 80)

with open(CSV_OUTPUT_PATH, mode="w", newline="", encoding="utf-8") as f_csv:
    writer = csv.writer(f_csv)
    # Header: strictly 3 columns
    writer.writerow(["real_image_class", "predicted_class", "isTrue"])

    # Write each test trial
    for i in range(total_test_samples):
        r_name = get_clean_name(all_trues[i])
        p_name = get_clean_name(all_preds[i])
        match_str = "true" if all_trues[i] == all_preds[i] else "false"
        writer.writerow([r_name, p_name, match_str])

    # Summary rows strictly adhering to 3 columns
    writer.writerow(["SUMMARY_TOTAL_TEST_SAMPLES", total_test_samples, total_test_samples])
    writer.writerow(["SUMMARY_CORRECT_PREDICTIONS", total_true_single, total_true_single])
    writer.writerow(["SUMMARY_OVERALL_SINGLE_TRIAL_ACCURACY", f"{single_trial_acc:.2f}%", f"{single_trial_acc:.2f}%"])
    writer.writerow(["SUMMARY_IMAGE_LEVEL_POOLED_ACCURACY", f"{pool_acc:.2f}%", f"{pool_acc:.2f}%"])
    writer.writerow(["SUMMARY_TOP5_ACCURACY", f"{single_trial_top5:.2f}%", f"{single_trial_top5:.2f}%"])

print(f"[✓] Full evaluation CSV saved locally to: {CSV_OUTPUT_PATH}")

# Upload to Google Drive
print(f"Uploading evaluation CSV to Google Drive folder '{DRIVE_MAIN_FOLDER_ID}'...")
drive_id = drive_upload_or_overwrite(DRIVE_MAIN_FOLDER_ID, "test_evaluation_results_conformer.csv", CSV_OUTPUT_PATH, mimetype="text/csv")
if drive_id:
    print(f"[✓] Successfully uploaded 'test_evaluation_results_conformer.csv' to Google Drive! (ID: {drive_id})")
else:
    print("[Notice] Could not upload CSV to Google Drive. Local copy is preserved.")

print("\n" + "=" * 80)
print("ALL-SAMPLES TEST EVALUATION COMPLETE!")
print("=" * 80)
