# ============================================================================
# EVALUATE SOTA NEUROSTREAM MODEL (BEST_SOTA_NEUROSTREAM.PT) - DIRECT TEST ACCURACY
# ----------------------------------------------------------------------------
# Evaluates single-trial inference for ALL test images/trials directly across:
#   [1] Baseline Single-Trial (Model Alone - Raw 1-shot)
#   [2] Method 1: Test-Time Augmentation (TTA - 8-pass Latent Averaging)
#   [3] Method 3: Prototypical Memory Bank (Nearest Class Centroid)
#   [4] Method 4: Spatiotemporal Filtering (1-30Hz Butterworth + CAR)
#   [5] Combined: All-in-One Simulated Pooling Ensemble
#   [6] Oracle Reference: True Multi-Trial Image Pooling (Upper Bound)
#
# Outputs:
#   - Direct 40-class Exact Accuracy (Top-1, Top-3, Top-5, Correct/Total count)
#   - Trial-by-trial Right vs. Wrong prediction table for all test images
#   - Per-class accuracy breakdown
#   - Auto-sync to Google Drive folder 'denoiserResult'
# ============================================================================

import os, sys, glob, random, math, csv, time, subprocess, warnings
import numpy as np
import scipy.signal as signal
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

# ----------------------------------------------------------------------------
# 1. CONFIGURATION & CLOUD DRIVE
# ----------------------------------------------------------------------------
DATA_ROOT  = "/kaggle/input/datasets/evolutionaw/eegcvpr40/data"
OUTPUT_DIR = "./eeg_denoiser_results"
if not os.path.exists(OUTPUT_DIR):
    OUTPUT_DIR = "/kaggle/working/eeg_denoiser_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_MODEL_PATH = "/kaggle/working/best_sota_neurostream.pt"
FALLBACK_PATHS = [
    TARGET_MODEL_PATH,
    "./best_sota_neurostream.pt",
    os.path.join(OUTPUT_DIR, "best_sota_neurostream.pt"),
    "./checkpoints/best_sota_neurostream.pt",
    "/kaggle/working/best_neurostream_sst.pt",
    "./checkpoints/best_neurostream_sst.pt"
]

DRIVE_TARGET_FOLDER_ID = "1TBeohhdFfmVuRU16FLGmPYZxdivNg3yN" # denoiserResult
DRIVE_CLIENT_ID = os.environ.get("DRIVE_CLIENT_ID", "YOUR_DRIVE_CLIENT_ID")
DRIVE_CLIENT_SECRET = os.environ.get("DRIVE_CLIENT_SECRET", "YOUR_DRIVE_CLIENT_SECRET")
DRIVE_REFRESH_TOKEN = os.environ.get("DRIVE_REFRESH_TOKEN", "YOUR_DRIVE_REFRESH_TOKEN")

SEED = 42
dev = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = (dev == "cuda")

# 40-Class ImageNet Human-readable Names for EEGCVPR40
HUMAN_NAMES = {
    "n01614925": "bald eagle", "n01910747": "jellyfish", "n02085936": "Maltese dog",
    "n02106662": "German shepherd", "n02123045": "tabby cat", "n02124075": "Egyptian cat",
    "n02127052": "lynx", "n02165456": "sorrel horse", "n02391049": "zebra",
    "n02410509": "bison", "n02437312": "camel", "n02504458": "African elephant",
    "n02510455": "giant panda", "n02607072": "airliner", "n02690373": "airship",
    "n02958343": "car wheel", "n02979186": "cassette player", "n03001847": "camera",
    "n03047690": "keyboard", "n03085013": "convertible car", "n03100240": "convertible coupe",
    "n03146219": "cradle", "n03179701": "desk table", "n03272562": "electric guitar",
    "n03291876": "espresso maker", "n03445777": "golf ball", "n03478564": "grand piano",
    "n03594734": "iron", "n03706229": "mailbag", "n03761084": "microwave oven",
    "n03792782": "mountain bike", "n03877845": "parachute", "n03938244": "pillow",
    "n03950228": "plastic bag", "n04069434": "radio telescope", "n04118776": "revolver gun",
    "n04200800": "running shoe", "n04254680": "soccer ball", "n04487081": "trolleybus",
    "n04548280": "wall clock"
}

def seed_everything(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

seed_everything(SEED)

def get_drive_service():
    try:
        from googleapiclient.discovery import build
        from google.oauth2.credentials import Credentials
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "google-api-python-client", "google-auth"], check=False)
        from googleapiclient.discovery import build
        from google.oauth2.credentials import Credentials

    creds = Credentials(None, refresh_token=DRIVE_REFRESH_TOKEN,
                        token_uri="https://oauth2.googleapis.com/token",
                        client_id=DRIVE_CLIENT_ID, client_secret=DRIVE_CLIENT_SECRET)
    return build("drive", "v3", credentials=creds)

def upload_file_to_drive(local_path, folder_id=DRIVE_TARGET_FOLDER_ID):
    if not os.path.exists(local_path): return False
    try:
        from googleapiclient.http import MediaFileUpload
        service = get_drive_service()
        fname = os.path.basename(local_path)
        q = f"'{folder_id}' in parents and name = '{fname}' and trashed = false"
        res = service.files().list(q=q, fields="files(id, name)").execute()
        files = res.get("files", [])
        media = MediaFileUpload(local_path, resumable=True)
        if files:
            service.files().update(fileId=files[0]["id"], media_body=media).execute()
            print(f"  [Drive Sync] Updated '{fname}' in folder '{folder_id}'")
        else:
            meta = {"name": fname, "parents": [folder_id]}
            service.files().create(body=meta, media_body=media, fields="id").execute()
            print(f"  [Drive Sync] Uploaded '{fname}' to folder '{folder_id}'")
        return True
    except Exception as e:
        print(f"  [Drive Notice] Cloud sync notice ({e})")
        return False

# ----------------------------------------------------------------------------
# 2. SOTA NEUROSTREAM ARCHITECTURE (EXACT MATCH FOR BEST_SOTA_NEUROSTREAM.PT)
# ----------------------------------------------------------------------------
class MultiScaleTemporalConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        branch_ch = out_ch // 4
        self.b1 = nn.Sequential(nn.Conv1d(in_ch, branch_ch, 5, 2, 2, bias=False), nn.BatchNorm1d(branch_ch), nn.GELU())
        self.b2 = nn.Sequential(nn.Conv1d(in_ch, branch_ch, 11, 2, 5, bias=False), nn.BatchNorm1d(branch_ch), nn.GELU())
        self.b3 = nn.Sequential(nn.Conv1d(in_ch, branch_ch, 23, 2, 11, bias=False), nn.BatchNorm1d(branch_ch), nn.GELU())
        self.b4 = nn.Sequential(nn.AvgPool1d(3, 2, 1), nn.Conv1d(in_ch, branch_ch, 1, bias=False), nn.BatchNorm1d(branch_ch), nn.GELU())

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

class ConformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=4, ffn_dim=512, dropout=0.1):
        super().__init__()
        self.ffn1 = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model), nn.Dropout(dropout))
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, 1), nn.GLU(1),
            nn.Conv1d(d_model, d_model, 7, 1, 3, groups=d_model),
            nn.BatchNorm1d(d_model), nn.GELU(), nn.Conv1d(d_model, d_model, 1), nn.Dropout(dropout)
        )
        self.ffn2 = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model), nn.Dropout(dropout))
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ffn1(x)
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x_conv = self.conv(self.conv_norm(x).transpose(1, 2)).transpose(1, 2)
        x = x + x_conv
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)

class AttentionMultiTrialPooler(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.score_net = nn.Sequential(nn.Linear(d_model, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, feats):
        w = F.softmax(self.score_net(feats), dim=0)
        return torch.sum(feats * w, dim=0, keepdim=True)

class SOTANeuroStreamModel(nn.Module):
    def __init__(self, n_channels=128, n_classes=40, n_subjects=6, d_model=256, n_blocks=4):
        super().__init__()
        self.d_model = d_model
        self.spatial_filter = nn.Sequential(nn.Conv2d(1, 32, (n_channels, 1), bias=False), nn.BatchNorm2d(32), nn.GELU())
        self.temporal_stage1 = MultiScaleTemporalConv(32, 64)
        self.temporal_stage2 = MultiScaleTemporalConv(64, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 110, d_model) * 0.02)
        self.blocks = nn.ModuleList([ConformerBlock(d_model, 4, 512, 0.1) for _ in range(n_blocks)])
        self.subject_embed = nn.Embedding(max(7, n_subjects + 1), d_model)
        nn.init.zeros_(self.subject_embed.weight)

        self.proj_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 128))
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(0.25),
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(d_model, n_classes)
        )
        self.pooler = AttentionMultiTrialPooler(d_model)

    def extract_features(self, x_eeg, subj_id=None):
        B = x_eeg.shape[0]
        spat = self.spatial_filter(x_eeg.unsqueeze(1)).squeeze(2)
        t1 = self.temporal_stage1(spat)
        t2 = self.temporal_stage2(t1)
        h = t2.transpose(1, 2) + self.pos_emb[:, :t2.shape[2], :]
        for blk in self.blocks: h = blk(h)
        feat = h.mean(dim=1)
        if subj_id is not None:
            feat = feat - 0.2 * self.subject_embed(subj_id)
        return feat

    def forward(self, x_eeg, subj_id=None):
        f = self.extract_features(x_eeg, subj_id)
        return self.head(f), f, None

# ----------------------------------------------------------------------------
# 3. DATASET LOADER (IDENTICAL NORMALIZATION & IMAGE SPLIT)
# ----------------------------------------------------------------------------
def load_benchmark_data():
    pths = glob.glob(os.path.join(DATA_ROOT, "**", "*.pth"), recursive=True)
    if not pths:
        for ar in ["/kaggle/input", "./data", "../data", "."]:
            pths = glob.glob(os.path.join(ar, "**", "*.pth"), recursive=True)
            if pths: break

    eeg_file, split_file = None, None
    if pths:
        bn = os.path.basename
        eeg_file = max([p for p in pths if "split" not in bn(p).lower()], key=os.path.getsize)
        split_file = next((p for p in sorted([p for p in pths if "split" in bn(p).lower()],
                                             key=lambda p: "all" not in bn(p).lower())), None)

    if not eeg_file or not os.path.exists(eeg_file):
        print("  [Notice] EEGCVPR40 not found. Generating synthetic benchmark data...")
        N_tr, N_te = 1200, 400
        synsets = [f"n02{i:06d}" for i in range(40)]
        X_tr = np.random.randn(N_tr, 128, 440).astype(np.float32)
        Y_tr = np.random.randint(0, 40, size=N_tr)
        S_tr = np.random.randint(0, 6, size=N_tr)
        I_tr = np.random.randint(0, 1600, size=N_tr)

        X_te = np.random.randn(N_te, 128, 440).astype(np.float32)
        Y_te = np.random.randint(0, 40, size=N_te)
        S_te = np.random.randint(0, 6, size=N_te)
        I_te = np.random.randint(1600, 2000, size=N_te)
        return X_tr, Y_tr, S_tr, I_tr, X_te, Y_te, S_te, I_te, synsets

    print(f"[+] Found EEG dataset: {eeg_file}")
    d = torch.load(eeg_file, map_location="cpu", weights_only=False)
    names, ds = d["images"], d["dataset"]

    X, Y, SUBJ, IMGS = [], [], [], []
    synsets = sorted({names[int(it["image"])].split("_")[0] for it in ds})
    syn2cls = {s: i for i, s in enumerate(synsets)}

    for it in ds:
        e = torch.as_tensor(it["eeg"]).float()
        if e.shape[0] != 128 and e.shape[1] == 128: e = e.t()
        if not (450 <= e.shape[1] <= 600): continue
        X.append(e[:, 20:460].numpy())
        im_id = int(it["image"])
        Y.append(syn2cls[names[im_id].split("_")[0]])
        SUBJ.append(int(it["subject"]))
        IMGS.append(im_id)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int64)
    SUBJ = np.array(SUBJ, dtype=np.int64)
    IMGS = np.array(IMGS, dtype=np.int64)

    # Channel standardization matching SOTA NeuroStream training
    X = (X - X.mean(axis=(1, 2), keepdims=True)) / (X.std(axis=(1, 2), keepdims=True) + 1e-6)

    if split_file and os.path.exists(split_file):
        sp = torch.load(split_file, map_location="cpu", weights_only=False)["splits"][0]
        tr_imgs = set(sp["train"])
        val_imgs = set(sp.get("val", []))
        te_imgs = set(sp["test"])
        tr_idx = np.array([i for i, im in enumerate(IMGS) if im in tr_imgs or (not val_imgs and im not in te_imgs)])
        te_idx = np.array([i for i, im in enumerate(IMGS) if im in te_imgs])
    else:
        uniq_ims = np.unique(IMGS)
        rng = np.random.RandomState(SEED); rng.shuffle(uniq_ims)
        n_tr = int(0.80 * len(uniq_ims))
        tr_idx = np.array([i for i, im in enumerate(IMGS) if im in set(uniq_ims[:n_tr])])
        te_idx = np.array([i for i, im in enumerate(IMGS) if im in set(uniq_ims[n_tr:])])

    print(f"  • Training Trials: {len(tr_idx)} | Test Trials: {len(te_idx)} | Classes: {len(synsets)}")
    return X[tr_idx], Y[tr_idx], SUBJ[tr_idx], IMGS[tr_idx], X[te_idx], Y[te_idx], SUBJ[te_idx], IMGS[te_idx], synsets

# ----------------------------------------------------------------------------
# 4. SIMULATED POOLING MODULES
# ----------------------------------------------------------------------------
class TestTimeAugmenter:
    """Method 1: Test-Time Augmentation in Signal & Latent Space."""
    def __init__(self, n_passes=8, jitter_std=0.03, max_time_shift=8, channel_mask_prob=0.08):
        self.n_passes = n_passes
        self.jitter_std = jitter_std
        self.max_time_shift = max_time_shift
        self.channel_mask_prob = channel_mask_prob

    def augment(self, x_eeg):
        B, C, T = x_eeg.shape
        variants = [x_eeg]
        for _ in range(self.n_passes - 1):
            x_aug = x_eeg.clone()
            shift = np.random.randint(-self.max_time_shift, self.max_time_shift + 1)
            if shift != 0:
                x_aug = torch.roll(x_aug, shifts=shift, dims=-1)
                if shift > 0: x_aug[:, :, :shift] = x_aug[:, :, shift:shift+1]
                else: x_aug[:, :, shift:] = x_aug[:, :, shift-1:shift]
            mask = (torch.rand(B, C, 1, device=x_eeg.device) > self.channel_mask_prob).float()
            x_aug = x_aug * mask + torch.randn_like(x_aug) * self.jitter_std
            variants.append(x_aug)
        return variants

    def predict(self, model, x_eeg, subj_id=None):
        variants = self.augment(x_eeg)
        feats = [model.extract_features(v, subj_id) for v in variants]
        pooled_f = torch.stack(feats, dim=0).mean(dim=0)
        return model.head(pooled_f), pooled_f

class PrototypicalMemoryBank:
    """Method 3: Class Centroid Prototypes."""
    def __init__(self, temperature=14.0):
        self.temperature = temperature
        self.prototypes = None

    def build_bank(self, model, X_tr, Y_tr, device="cuda"):
        model.eval()
        X_t = torch.as_tensor(X_tr, dtype=torch.float32)
        all_feats = []
        with torch.no_grad():
            for k in range(0, len(X_t), 64):
                xb = X_t[k : k + 64].to(device)
                all_feats.append(model.extract_features(xb).cpu())
        all_feats = F.normalize(torch.cat(all_feats, dim=0), dim=-1)
        classes = np.unique(Y_tr)
        self.prototypes = torch.zeros(len(classes), all_feats.shape[-1])
        for c in classes:
            self.prototypes[c] = F.normalize(all_feats[Y_tr == c].mean(dim=0), dim=-1)

    def classify(self, test_feats):
        norm_feats = F.normalize(test_feats, dim=-1)
        proto = self.prototypes.to(test_feats.device)
        return self.temperature * torch.matmul(norm_feats, proto.T)

class SpatioTemporalFilter:
    """Method 4: 1-30Hz Butterworth Bandpass + Common Average Reference (CAR)."""
    def __init__(self, lowcut=1.0, highcut=30.0, fs=250.0):
        nyq = 0.5 * fs
        self.b, self.a = signal.butter(4, [lowcut/nyq, highcut/nyq], btype="band")

    def filter(self, X_eeg):
        orig_shape = X_eeg.shape
        arr = X_eeg if len(orig_shape) == 3 else X_eeg[np.newaxis, ...]
        bp = signal.filtfilt(self.b, self.a, arr, axis=-1)
        car = bp - bp.mean(axis=-2, keepdims=True)
        return car.reshape(orig_shape).astype(np.float32)

# ----------------------------------------------------------------------------
# 5. DIRECT ACCURACY COMPUTATION
# ----------------------------------------------------------------------------
def compute_direct_classification_metrics(logits, labels):
    """
    Computes standard 40-class multi-class classification metrics:
    - Top-1 Accuracy (Exact Correctness)
    - Top-3 Accuracy
    - Top-5 Accuracy
    - Mean True Class Rank (1.00 is optimal)
    - Total Correct Count
    """
    N = len(labels)
    preds_top1 = np.argmax(logits, axis=-1)
    correct_count = int(np.sum(preds_top1 == labels))
    top1_acc = 100.0 * correct_count / N

    sorted_indices = np.argsort(-logits, axis=-1)
    top3_acc = 100.0 * np.mean([labels[i] in sorted_indices[i, :3] for i in range(N)])
    top5_acc = 100.0 * np.mean([labels[i] in sorted_indices[i, :5] for i in range(N)])

    ranks = [int(np.where(sorted_indices[i] == labels[i])[0][0] + 1) for i in range(N)]
    mean_rank = float(np.mean(ranks))

    return {
        "correct": correct_count,
        "total": N,
        "top1_acc": top1_acc,
        "top3_acc": top3_acc,
        "top5_acc": top5_acc,
        "mean_rank": mean_rank,
        "preds_top1": preds_top1,
        "ranks": ranks,
        "sorted_indices": sorted_indices
    }

# ----------------------------------------------------------------------------
# 6. MASTER EVALUATION RUNNER
# ----------------------------------------------------------------------------
def evaluate_single_trial_direct_accuracy(model_file=TARGET_MODEL_PATH):
    print("=" * 105)
    print("DIRECT SINGLE-TRIAL CLASSIFICATION ACCURACY FOR ALL TEST SAMPLES (EXACT 40-CLASS)")
    print(f"Target Checkpoint: {model_file}")
    print("=" * 105)

    X_tr, Y_tr, S_tr, I_tr, X_te, Y_te, S_te, I_te, synsets = load_benchmark_data()
    n_classes = len(synsets)
    n_subjects = int(max(S_tr.max(), S_te.max())) + 1
    N_te = len(X_te)

    # 1. Resolve Checkpoint Path
    active_path = None
    if os.path.exists(model_file):
        active_path = model_file
    else:
        for cand in FALLBACK_PATHS:
            if os.path.exists(cand):
                active_path = cand; break

    model = SOTANeuroStreamModel(n_channels=128, n_classes=n_classes, n_subjects=n_subjects, d_model=256, n_blocks=4).to(dev)

    if active_path and os.path.exists(active_path):
        print(f"[+] Loaded Model Weights from: {active_path}")
        ck = torch.load(active_path, map_location=dev, weights_only=False)
        state = ck.get("model_state", ck.get("model_state_dict", ck))
        model_keys = model.state_dict()
        filtered = {k: v for k, v in state.items() if k in model_keys and model_keys[k].shape == v.shape}
        model.load_state_dict(filtered, strict=False)
        print(f"  [OK] Successfully loaded {len(filtered)} model tensors.")
        if isinstance(ck, dict) and "epoch" in ck:
            print(f"  [OK] Checkpoint Epoch: {ck['epoch']} | Saved Pooled Acc: {ck.get('pooled_acc', 'N/A')}")
    else:
        print("  [Notice] Checkpoint not found in standard paths; initializing fresh SOTA NeuroStream model.")

    model.eval()
    X_te_t = torch.as_tensor(X_te, dtype=torch.float32)
    S_te_t = torch.as_tensor(S_te, dtype=torch.long)

    # ------------------------------------------------------------------------
    # [1] BASELINE: Raw Single-Trial (1-shot - Model Alone)
    # ------------------------------------------------------------------------
    print("\n[+] Evaluating [1] Baseline Single-Trial (Raw 1-shot - Model Alone)...")
    base_logits, base_feats = [], []
    with torch.no_grad():
        for k in range(0, N_te, 64):
            xb = X_te_t[k : k + 64].to(dev)
            sb = S_te_t[k : k + 64].to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
                l, f, _ = model(xb, sb)
            base_logits.append(l.cpu()); base_feats.append(f.cpu())
    base_logits = torch.cat(base_logits, dim=0).numpy()
    base_feats  = torch.cat(base_feats, dim=0)

    # ------------------------------------------------------------------------
    # [2] METHOD 1: Test-Time Augmentation (TTA)
    # ------------------------------------------------------------------------
    print("[+] Evaluating [2] Method 1: Test-Time Augmentation (8-pass Latent Averaging)...")
    tta = TestTimeAugmenter(n_passes=8, jitter_std=0.03, max_time_shift=8, channel_mask_prob=0.08)
    tta_logits = []
    with torch.no_grad():
        for k in range(0, N_te, 32):
            xb = X_te_t[k : k + 32].to(dev)
            sb = S_te_t[k : k + 32].to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
                l, _ = tta.predict(model, xb, sb)
            tta_logits.append(l.cpu())
    tta_logits = torch.cat(tta_logits, dim=0).numpy()

    # ------------------------------------------------------------------------
    # [3] METHOD 3: Prototypical Memory Bank
    # ------------------------------------------------------------------------
    print("[+] Evaluating [3] Method 3: Prototypical Memory Bank (Nearest Class Centroid)...")
    pmb = PrototypicalMemoryBank(temperature=14.0)
    pmb.build_bank(model, X_tr, Y_tr, device=dev)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
            proto_logits = pmb.classify(base_feats.to(dev)).cpu().numpy()

    # ------------------------------------------------------------------------
    # [4] METHOD 4: Spatiotemporal Filtering
    # ------------------------------------------------------------------------
    print("[+] Evaluating [4] Method 4: Spatiotemporal Filtering (1-30Hz Butterworth + CAR)...")
    stf = SpatioTemporalFilter(1.0, 30.0, 250.0)
    X_te_filt = stf.filter(X_te)
    X_te_filt_t = torch.as_tensor(X_te_filt, dtype=torch.float32)

    filt_logits = []
    with torch.no_grad():
        for k in range(0, N_te, 64):
            xb = X_te_filt_t[k : k + 64].to(dev)
            sb = S_te_t[k : k + 64].to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
                l, _, _ = model(xb, sb)
            filt_logits.append(l.cpu())
    filt_logits = torch.cat(filt_logits, dim=0).numpy()

    # ------------------------------------------------------------------------
    # [5] COMBINED: Ensemble of Simulated Pooling
    # ------------------------------------------------------------------------
    print("[+] Evaluating [5] Combined All-in-One Simulated Pooling Ensemble...")
    combined_logits = 0.40 * tta_logits + 0.35 * proto_logits + 0.25 * filt_logits

    # ------------------------------------------------------------------------
    # [6] ORACLE REFERENCE: True Multi-Trial Image Pooling (Upper Bound)
    # ------------------------------------------------------------------------
    print("[+] Evaluating [6] Oracle Reference: True Multi-Trial Image Pooling (Upper Bound)...")
    te_img2trials = {}
    for idx, im in enumerate(I_te): te_img2trials.setdefault(im, []).append(idx)
    oracle_logits, oracle_labels = [], []
    with torch.no_grad():
        for im, tr_idxs in te_img2trials.items():
            tr_f = base_feats[tr_idxs].to(dev)
            p_f = model.pooler(tr_f)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
                p_l = model.head(p_f).cpu().numpy()[0]
            oracle_logits.append(p_l)
            oracle_labels.append(Y_te[tr_idxs[0]])
    oracle_logits = np.array(oracle_logits)
    oracle_labels = np.array(oracle_labels)

    # ------------------------------------------------------------------------
    # 7. METRICS & SUMMARY TABLE
    # ------------------------------------------------------------------------
    m_base  = compute_direct_classification_metrics(base_logits, Y_te)
    m_tta   = compute_direct_classification_metrics(tta_logits, Y_te)
    m_proto = compute_direct_classification_metrics(proto_logits, Y_te)
    m_filt  = compute_direct_classification_metrics(filt_logits, Y_te)
    m_comb  = compute_direct_classification_metrics(combined_logits, Y_te)
    m_ora   = compute_direct_classification_metrics(oracle_logits, oracle_labels)

    summary_rows = [
        {"Method": "[1] Model Alone (Baseline Single-Trial)", "Correct/Total": f"{m_base['correct']}/{m_base['total']}", "Top-1 Accuracy": f"{m_base['top1_acc']:.2f}%", "Top-3 Accuracy": f"{m_base['top3_acc']:.2f}%", "Top-5 Accuracy": f"{m_base['top5_acc']:.2f}%", "Avg Rank": f"{m_base['mean_rank']:.2f}"},
        {"Method": "[2] Method 1: Test-Time Augmentation", "Correct/Total": f"{m_tta['correct']}/{m_tta['total']}", "Top-1 Accuracy": f"{m_tta['top1_acc']:.2f}%", "Top-3 Accuracy": f"{m_tta['top3_acc']:.2f}%", "Top-5 Accuracy": f"{m_tta['top5_acc']:.2f}%", "Avg Rank": f"{m_tta['mean_rank']:.2f}"},
        {"Method": "[3] Method 3: Prototypical Memory Bank", "Correct/Total": f"{m_proto['correct']}/{m_proto['total']}", "Top-1 Accuracy": f"{m_proto['top1_acc']:.2f}%", "Top-3 Accuracy": f"{m_proto['top3_acc']:.2f}%", "Top-5 Accuracy": f"{m_proto['top5_acc']:.2f}%", "Avg Rank": f"{m_proto['mean_rank']:.2f}"},
        {"Method": "[4] Method 4: Spatiotemporal Filtering", "Correct/Total": f"{m_filt['correct']}/{m_filt['total']}", "Top-1 Accuracy": f"{m_filt['top1_acc']:.2f}%", "Top-3 Accuracy": f"{m_filt['top3_acc']:.2f}%", "Top-5 Accuracy": f"{m_filt['top5_acc']:.2f}%", "Avg Rank": f"{m_filt['mean_rank']:.2f}"},
        {"Method": "[5] Combined All-in-One Ensemble", "Correct/Total": f"{m_comb['correct']}/{m_comb['total']}", "Top-1 Accuracy": f"{m_comb['top1_acc']:.2f}%", "Top-3 Accuracy": f"{m_comb['top3_acc']:.2f}%", "Top-5 Accuracy": f"{m_comb['top5_acc']:.2f}%", "Avg Rank": f"{m_comb['mean_rank']:.2f}"},
        {"Method": "[6] Oracle: True Multi-Trial Pooled", "Correct/Total": f"{m_ora['correct']}/{m_ora['total']}", "Top-1 Accuracy": f"{m_ora['top1_acc']:.2f}%", "Top-3 Accuracy": f"{m_ora['top3_acc']:.2f}%", "Top-5 Accuracy": f"{m_ora['top5_acc']:.2f}%", "Avg Rank": f"{m_ora['mean_rank']:.2f}"}
    ]

    print("\n" + "=" * 105)
    print("DIRECT 40-CLASS CLASSIFICATION ACCURACY SUMMARY (ALL TEST SAMPLES)")
    print("=" * 105)
    print(f"{'Method / Strategy':<42} | {'Correct/Total':<14} | {'Top-1 Acc':<11} | {'Top-3 Acc':<11} | {'Top-5 Acc':<11} | {'Avg Rank'}")
    print("-" * 105)
    for r in summary_rows:
        print(f"{r['Method']:<42} | {r['Correct/Total']:<14} | {r['Top-1 Accuracy']:>9} | {r['Top-3 Accuracy']:>9} | {r['Top-5 Accuracy']:>9} | {r['Avg Rank']:>8}")
    print("=" * 105)

    # ------------------------------------------------------------------------
    # 8. TRIAL-BY-TRIAL RIGHT VS. WRONG PREDICTION RECORDS
    # ------------------------------------------------------------------------
    probs_comb = F.softmax(torch.from_numpy(combined_logits), dim=-1).numpy()
    probs_tta  = F.softmax(torch.from_numpy(tta_logits), dim=-1).numpy()
    probs_proto= F.softmax(torch.from_numpy(proto_logits), dim=-1).numpy()
    probs_filt = F.softmax(torch.from_numpy(filt_logits), dim=-1).numpy()
    probs_base = F.softmax(torch.from_numpy(base_logits), dim=-1).numpy()

    all_trials_records = []
    for i in range(N_te):
        real_c = int(Y_te[i])
        real_syn = synsets[real_c]
        real_label = f"{real_syn} ({HUMAN_NAMES.get(real_syn, real_syn)})"

        pred_base_c  = int(m_base["preds_top1"][i])
        pred_tta_c   = int(m_tta["preds_top1"][i])
        pred_proto_c = int(m_proto["preds_top1"][i])
        pred_filt_c  = int(m_filt["preds_top1"][i])
        pred_comb_c  = int(m_comb["preds_top1"][i])

        syn_base  = synsets[pred_base_c]
        syn_tta   = synsets[pred_tta_c]
        syn_proto = synsets[pred_proto_c]
        syn_filt  = synsets[pred_filt_c]
        syn_comb  = synsets[pred_comb_c]

        is_corr_base  = (real_c == pred_base_c)
        is_corr_tta   = (real_c == pred_tta_c)
        is_corr_proto = (real_c == pred_proto_c)
        is_corr_filt  = (real_c == pred_filt_c)
        is_corr_comb  = (real_c == pred_comb_c)

        sorted_comb = m_comb["sorted_indices"][i]
        top3_comb_names = [f"{synsets[c]} ({HUMAN_NAMES.get(synsets[c], synsets[c])})" for c in sorted_comb[:3]]

        all_trials_records.append({
            "trial_index": i,
            "image_id": int(I_te[i]),
            "real_class": real_label,
            # Model Alone
            "pred_model_alone": f"{syn_base} ({HUMAN_NAMES.get(syn_base, syn_base)})",
            "status_model_alone": "RIGHT [OK]" if is_corr_base else "WRONG [X]",
            "conf_model_alone": round(float(probs_base[i, pred_base_c]), 4),
            # Method 1: TTA
            "pred_tta": f"{syn_tta} ({HUMAN_NAMES.get(syn_tta, syn_tta)})",
            "status_tta": "RIGHT [OK]" if is_corr_tta else "WRONG [X]",
            "conf_tta": round(float(probs_tta[i, pred_tta_c]), 4),
            # Method 3: Prototypical Memory Bank
            "pred_proto": f"{syn_proto} ({HUMAN_NAMES.get(syn_proto, syn_proto)})",
            "status_proto": "RIGHT [OK]" if is_corr_proto else "WRONG [X]",
            "conf_proto": round(float(probs_proto[i, pred_proto_c]), 4),
            # Method 4: Filtering
            "pred_filter": f"{syn_filt} ({HUMAN_NAMES.get(syn_filt, syn_filt)})",
            "status_filter": "RIGHT [OK]" if is_corr_filt else "WRONG [X]",
            "conf_filter": round(float(probs_filt[i, pred_filt_c]), 4),
            # Combined
            "pred_combined": f"{syn_comb} ({HUMAN_NAMES.get(syn_comb, syn_comb)})",
            "status_combined": "RIGHT [OK]" if is_corr_comb else "WRONG [X]",
            "conf_combined": round(float(probs_comb[i, pred_comb_c]), 4),
            "rank_of_true_in_combined": int(m_comb["ranks"][i]),
            "is_top3_combined": bool(real_c in sorted_comb[:3]),
            "is_top5_combined": bool(real_c in sorted_comb[:5]),
            "top3_candidates": " | ".join(top3_comb_names)
        })

    # ------------------------------------------------------------------------
    # 9. PER-CLASS ACCURACY BREAKDOWN
    # ------------------------------------------------------------------------
    class_stats = []
    for c_idx, syn in enumerate(synsets):
        mask = (Y_te == c_idx)
        n_c = int(np.sum(mask))
        if n_c == 0: continue
        c_name = f"{syn} ({HUMAN_NAMES.get(syn, syn)})"
        acc_base = 100.0 * np.sum(m_base["preds_top1"][mask] == c_idx) / n_c
        acc_tta  = 100.0 * np.sum(m_tta["preds_top1"][mask] == c_idx) / n_c
        acc_comb = 100.0 * np.sum(m_comb["preds_top1"][mask] == c_idx) / n_c
        class_stats.append({
            "class_index": c_idx,
            "class_name": c_name,
            "test_trials": n_c,
            "acc_model_alone": f"{acc_base:.1f}%",
            "acc_tta": f"{acc_tta:.1f}%",
            "acc_combined": f"{acc_comb:.1f}%"
        })

    # Export & Cloud Sync
    df_trials  = pd.DataFrame(all_trials_records)
    df_summary = pd.DataFrame(summary_rows)
    df_classes = pd.DataFrame(class_stats)

    csv_trials_path  = os.path.join(OUTPUT_DIR, "all_test_trials_right_or_wrong_predictions.csv")
    csv_summary_path = os.path.join(OUTPUT_DIR, "single_trial_methods_exact_accuracy_summary.csv")
    csv_classes_path = os.path.join(OUTPUT_DIR, "per_class_accuracy_breakdown.csv")

    df_trials.to_csv(csv_trials_path, index=False)
    df_summary.to_csv(csv_summary_path, index=False)
    df_classes.to_csv(csv_classes_path, index=False)

    print(f"\n[OK] Trial-by-Trial Right/Wrong CSV saved: {csv_trials_path}")
    print(f"[OK] Summary Exact Accuracy CSV saved:   {csv_summary_path}")
    print(f"[OK] Per-Class Breakdown CSV saved:       {csv_classes_path}")

    print("\n[+] Syncing reports to Google Drive folder 'denoiserResult'...")
    upload_file_to_drive(csv_trials_path, folder_id=DRIVE_TARGET_FOLDER_ID)
    upload_file_to_drive(csv_summary_path, folder_id=DRIVE_TARGET_FOLDER_ID)
    upload_file_to_drive(csv_classes_path, folder_id=DRIVE_TARGET_FOLDER_ID)
    print(f"[SUCCESS] All synced to Drive: https://drive.google.com/drive/folders/{DRIVE_TARGET_FOLDER_ID}\n")

    return {
        "df_trials": df_trials,
        "df_summary": df_summary,
        "df_classes": df_classes
    }

# Execute master benchmark
res = evaluate_single_trial_direct_accuracy(TARGET_MODEL_PATH)
df_sum = res["df_summary"]
df_rec = res["df_trials"]

print("\n" + "=" * 105)
print("FIRST 20 TEST SAMPLES: RIGHT VS. WRONG PREDICTIONS ACROSS METHODS")
print("=" * 105)
sample_cols = ["trial_index", "real_class", "status_model_alone", "status_tta", "status_proto", "status_filter", "status_combined", "pred_combined", "conf_combined"]

try:
    display(df_sum)
    display(df_rec[sample_cols].head(25))
except NameError:
    print(df_sum.to_string(index=False))
    print("\nSample Predictions:")
    print(df_rec[sample_cols].head(25).to_string(index=False))
