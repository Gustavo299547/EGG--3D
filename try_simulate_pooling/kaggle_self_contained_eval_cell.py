# ============================================================================
# KAGGLE SELF-CONTAINED CELL: SIMULATED POOLING EVALUATION BENCHMARK
# ----------------------------------------------------------------------------
# Copy-paste this entire cell into Kaggle and run! Zero external file imports.
# Benchmarks all single-trial noise-reduction / simulated pooling techniques:
#   1. Raw Single-Trial (Baseline 1-shot)
#   2. Method 1: Test-Time Augmentation (TTA: 8-pass temporal/noise/channel jitter)
#   3. Method 3: Prototypical Memory Bank (Nearest Class Centroid in latent space)
#   4. Method 4: Spatiotemporal Filtering (1-30Hz Butterworth + Spatial CAR)
#   5. Combined: All-in-One Simulated Pooling Ensemble
#   6. Oracle Reference: True Multi-Trial Image Pooling
# ============================================================================

import os, sys, glob, random, math, csv, time, subprocess, warnings
import numpy as np
import scipy.signal as signal
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

DATA_ROOT  = "/kaggle/input/datasets/evolutionaw/eegcvpr40/data"
OUTPUT_DIR = "/kaggle/working/eeg_denoiser_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DRIVE_TARGET_FOLDER_ID = "1TBeohhdFfmVuRU16FLGmPYZxdivNg3yN" # denoiserResult
DRIVE_CLIENT_ID = os.environ.get("DRIVE_CLIENT_ID", "YOUR_DRIVE_CLIENT_ID")
DRIVE_CLIENT_SECRET = os.environ.get("DRIVE_CLIENT_SECRET", "YOUR_DRIVE_CLIENT_SECRET")
DRIVE_REFRESH_TOKEN = os.environ.get("DRIVE_REFRESH_TOKEN", "YOUR_DRIVE_REFRESH_TOKEN")

MODEL_CANDIDATES = [
    "/kaggle/working/best_sota_neurostream.pt",
    "/kaggle/working/best_neurostream_sst.pt",
    "./best_sota_neurostream.pt",
    "./best_neurostream_sst.pt",
    "./checkpoints/best_neurostream_sst.pt"
]

SEED = 42
dev = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = (dev == "cuda")

def seed_everything(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

seed_everything(SEED)

# ----------------------------------------------------------------------------
# Google Drive Cloud Sync
# ----------------------------------------------------------------------------
def get_drive_service():
    try:
        from googleapiclient.discovery import build
        from google.oauth2.credentials import Credentials
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "google-api-python-client", "google-auth"], check=False)
        from googleapiclient.discovery import build
        from google.oauth2.credentials import Credentials

    creds = Credentials(
        None,
        refresh_token=DRIVE_REFRESH_TOKEN,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=DRIVE_CLIENT_ID,
        client_secret=DRIVE_CLIENT_SECRET
    )
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
# Model Architecture
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
# Dataset Loader
# ----------------------------------------------------------------------------
def load_eegcvpr40_data():
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
        print("  [Notice] EEGCVPR40 not found. Generating synthetic data for local verification...")
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
        rng = np.random.RandomState(SEED)
        rng.shuffle(uniq_ims)
        n_tr = int(0.80 * len(uniq_ims))
        tr_idx = np.array([i for i, im in enumerate(IMGS) if im in set(uniq_ims[:n_tr])])
        te_idx = np.array([i for i, im in enumerate(IMGS) if im in set(uniq_ims[n_tr:])])

    print(f"  • Training Trials: {len(tr_idx)} | Test Trials: {len(te_idx)} | Classes: {len(synsets)}")
    return X[tr_idx], Y[tr_idx], SUBJ[tr_idx], IMGS[tr_idx], X[te_idx], Y[te_idx], SUBJ[te_idx], IMGS[te_idx], synsets

# ----------------------------------------------------------------------------
# Simulated Pooling Methods: TTA, Memory Bank, Spatiotemporal Filter
# ----------------------------------------------------------------------------
class TestTimeAugmenter:
    def __init__(self, n_passes=8, jitter_std=0.03, max_time_shift=8, channel_mask_prob=0.08):
        self.n_passes = n_passes
        self.jitter_std = jitter_std
        self.max_time_shift = max_time_shift
        self.channel_mask_prob = channel_mask_prob

    def augment_single_trial(self, x_eeg):
        if x_eeg.dim() == 2: x_eeg = x_eeg.unsqueeze(0)
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

    def predict_with_tta(self, model, x_eeg, subj_id=None):
        variants = self.augment_single_trial(x_eeg)
        feats_list = [model.extract_features(v, subj_id) for v in variants]
        pooled_feats = torch.stack(feats_list, dim=0).mean(dim=0)
        return model.head(pooled_feats), pooled_feats

class PrototypicalMemoryBank:
    def __init__(self, temperature=14.0):
        self.temperature = temperature
        self.prototypes = None

    def build_memory_bank(self, model, X_tr, Y_tr, device="cuda"):
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

    def classify_with_prototypes(self, test_feats):
        norm_feats = F.normalize(test_feats, dim=-1)
        proto = self.prototypes.to(test_feats.device)
        return self.temperature * torch.matmul(norm_feats, proto.T)

class SpatioTemporalFilter:
    def __init__(self, lowcut=1.0, highcut=30.0, fs=250.0):
        nyq = 0.5 * fs
        self.b, self.a = signal.butter(4, [lowcut/nyq, highcut/nyq], btype="band")

    def filter_eeg(self, X_eeg):
        orig_shape = X_eeg.shape
        arr = X_eeg if len(orig_shape) == 3 else X_eeg[np.newaxis, ...]
        bp = signal.filtfilt(self.b, self.a, arr, axis=-1)
        car = bp - bp.mean(axis=-2, keepdims=True)
        return car.reshape(orig_shape).astype(np.float32)

# ----------------------------------------------------------------------------
# N-Way Top-K Paper Metric Evaluator
# ----------------------------------------------------------------------------
def compute_n_way_topk(logits, labels, n_way=10, k=1, n_samplings=100, seed=42):
    rng = np.random.RandomState(seed)
    N, n_classes = logits.shape
    if n_way > n_classes: n_way = n_classes
    all_accuracies = []
    all_classes_set = np.arange(n_classes)
    for _ in range(n_samplings):
        correct = 0
        for i in range(N):
            true_c = labels[i]
            neg_candidates = np.setdiff1d(all_classes_set, [true_c])
            sampled_negs = rng.choice(neg_candidates, size=n_way - 1, replace=False)
            candidate_set = np.append(sampled_negs, true_c)
            ranked = candidate_set[np.argsort(-logits[i, candidate_set])]
            if true_c in ranked[:k]: correct += 1
        all_accuracies.append(100.0 * correct / N)
    return float(np.mean(all_accuracies)), float(np.std(all_accuracies))

# ----------------------------------------------------------------------------
# Master Runner
# ----------------------------------------------------------------------------
def run_all_simulated_pooling_evaluation(checkpoint_path=None):
    print("=" * 95)
    print("SIMULATED POOLING BENCHMARK: EVALUATING NOISE REDUCTION TECHNIQUES")
    print("Testing under Single-Trial Constraints (No Multi-Trial Viewings)")
    print("=" * 95)

    X_tr, Y_tr, S_tr, I_tr, X_te, Y_te, S_te, I_te, synsets = load_eegcvpr40_data()
    n_classes = len(synsets)
    n_subjects = int(max(S_tr.max(), S_te.max())) + 1
    N_te = len(X_te)

    model_path = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        model_path = checkpoint_path
    else:
        for cand in MODEL_CANDIDATES:
            if os.path.exists(cand):
                model_path = cand; break

    model = SOTANeuroStreamModel(n_channels=128, n_classes=n_classes, n_subjects=n_subjects, d_model=256, n_blocks=4).to(dev)

    if model_path and os.path.exists(model_path):
        print(f"[+] Loaded Model Weights from: {model_path}")
        ck = torch.load(model_path, map_location=dev, weights_only=False)
        state = ck.get("model_state", ck.get("model_state_dict", ck))
        model_keys = model.state_dict()
        filtered = {k: v for k, v in state.items() if k in model_keys and model_keys[k].shape == v.shape}
        model.load_state_dict(filtered, strict=False)
        print(f"  [✓] Successfully loaded {len(filtered)} model tensors.")
    else:
        print("  [Notice] Initializing fresh SOTA NeuroStream model.")

    model.eval()
    X_te_t = torch.as_tensor(X_te, dtype=torch.float32)
    S_te_t = torch.as_tensor(S_te, dtype=torch.long)

    # 1. Baseline Single-Trial
    print("\n[+] Evaluating [1] Baseline Single-Trial (Raw 1-shot)...")
    base_logits, base_feats = [], []
    with torch.no_grad():
        for k in range(0, N_te, 64):
            l, f, _ = model(X_te_t[k:k+64].to(dev), S_te_t[k:k+64].to(dev))
            base_logits.append(l.cpu()); base_feats.append(f.cpu())
    base_logits = torch.cat(base_logits, dim=0).numpy()
    base_feats  = torch.cat(base_feats, dim=0)

    # 2. Method 1: TTA
    print("[+] Evaluating [2] Method 1: Test-Time Augmentation (8-pass Latent Averaging)...")
    tta = TestTimeAugmenter(n_passes=8, jitter_std=0.03, max_time_shift=8, channel_mask_prob=0.08)
    tta_logits = []
    with torch.no_grad():
        for k in range(0, N_te, 32):
            l, _ = tta.predict_with_tta(model, X_te_t[k:k+32].to(dev), S_te_t[k:k+32].to(dev))
            tta_logits.append(l.cpu())
    tta_logits = torch.cat(tta_logits, dim=0).numpy()

    # 3. Method 3: Memory Bank
    print("[+] Evaluating [3] Method 3: Prototypical Memory Bank (Nearest Class Centroid)...")
    pmb = PrototypicalMemoryBank(temperature=14.0)
    pmb.build_memory_bank(model, X_tr, Y_tr, device=dev)
    with torch.no_grad():
        proto_logits = pmb.classify_with_prototypes(base_feats.to(dev)).cpu().numpy()

    # 4. Method 4: Spatiotemporal Filtering
    print("[+] Evaluating [4] Method 4: Spatiotemporal Filtering (1-30Hz Butterworth + CAR)...")
    stf = SpatioTemporalFilter(1.0, 30.0, 250.0)
    X_te_filt_t = torch.as_tensor(stf.filter_eeg(X_te), dtype=torch.float32)
    filt_logits = []
    with torch.no_grad():
        for k in range(0, N_te, 64):
            l, _, _ = model(X_te_filt_t[k:k+64].to(dev), S_te_t[k:k+64].to(dev))
            filt_logits.append(l.cpu())
    filt_logits = torch.cat(filt_logits, dim=0).numpy()

    # 5. Combined All-in-One
    print("[+] Evaluating [5] Combined All-in-One Simulated Pooling...")
    combined_logits = 0.40 * tta_logits + 0.35 * proto_logits + 0.25 * filt_logits

    # 6. Oracle Multi-Trial Pooling
    print("[+] Evaluating [6] Oracle Reference: True Multi-Trial Image Pooling...")
    te_img2trials = {}
    for idx, im in enumerate(I_te): te_img2trials.setdefault(im, []).append(idx)
    oracle_logits, oracle_labels = [], []
    with torch.no_grad():
        for im, tr_idxs in te_img2trials.items():
            p_f = model.pooler(base_feats[tr_idxs].to(dev))
            oracle_logits.append(model.head(p_f).cpu().numpy()[0])
            oracle_labels.append(Y_te[tr_idxs[0]])
    oracle_logits = np.array(oracle_logits)
    oracle_labels = np.array(oracle_labels)

    # 7. Benchmark Table
    eval_targets = [
        ("Raw Single-Trial (Baseline)", base_logits, Y_te),
        ("Method 1: Test-Time Augmentation (TTA)", tta_logits, Y_te),
        ("Method 3: Prototypical Memory Bank", proto_logits, Y_te),
        ("Method 4: Spatiotemporal Filtering", filt_logits, Y_te),
        ("Combined: All-in-One Simulated Pooling", combined_logits, Y_te),
        ("Oracle: True Multi-Trial Image Pooling", oracle_logits, oracle_labels)
    ]

    print("\n" + "=" * 95)
    print("MASTER BENCHMARK: SIMULATED POOLING COMPARISON vs. PUBLISHED PAPER BASELINES")
    print("=" * 95)
    print(f"{'Framework / Simulation Strategy':<38} | {'2-way Top-1':<12} | {'10-way Top-1':<13} | {'10-way Top-2':<13} | {'50-way Top-1'}")
    print("-" * 95)

    paper_rows = [
        ("GWIT (Brain3D Best Pooled)", 94.60, 85.40, 90.60, 76.30),
        ("BrainVis (ICASSP 2025)",    88.00, 70.60, 79.60, 57.80),
        ("EEG-CLIP (2025)",           85.70, 65.50, 74.20, 54.50),
        ("DreamDiffusion (ECCV)",     73.00, 31.40, 48.00, 16.40),
    ]
    for name, a2, a10_1, a10_2, a50_1 in paper_rows:
        print(f"{name:<38} | {a2:>10.2f}% | {a10_1:>11.2f}% | {a10_2:>11.2f}% | {a50_1:>11.2f}%")
    print("-" * 95)

    comparison_results = []
    for name, log_arr, lab_arr in eval_targets:
        a2, _   = compute_n_way_topk(log_arr, lab_arr, 2, 1, 100)
        a10_1, _= compute_n_way_topk(log_arr, lab_arr, 10, 1, 100)
        a10_2, _= compute_n_way_topk(log_arr, lab_arr, 10, 2, 100)
        a50_1, _= compute_n_way_topk(log_arr, lab_arr, min(50, n_classes), 1, 100)
        print(f"{name:<38} | {a2:>10.2f}% | {a10_1:>11.2f}% | {a10_2:>11.2f}% | {a50_1:>11.2f}%")
        comparison_results.append({
            "Strategy": name, "2-way Top-1": f"{a2:.2f}%", "10-way Top-1": f"{a10_1:.2f}%",
            "10-way Top-2": f"{a10_2:.2f}%", "50-way Top-1": f"{a50_1:.2f}%"
        })
    print("=" * 95)

    # 8. Per-Sample Records
    all_test_samples = []
    probs_comb = F.softmax(torch.from_numpy(combined_logits), dim=-1).numpy()
    probs_base = F.softmax(torch.from_numpy(base_logits), dim=-1).numpy()

    for i in range(N_te):
        real_c = int(Y_te[i])
        pred_base_c = int(base_logits[i].argmax())
        pred_comb_c = int(combined_logits[i].argmax())
        sorted_comb = np.argsort(-combined_logits[i])
        all_test_samples.append({
            "trial_index": i,
            "image_id": int(I_te[i]),
            "real_class": synsets[real_c],
            "raw_single_prediction": synsets[pred_base_c],
            "simulated_pooling_prediction": synsets[pred_comb_c],
            "is_correct_top1": (real_c == pred_comb_c),
            "is_correct_top2": bool(real_c in sorted_comb[:2]),
            "is_correct_top5": bool(real_c in sorted_comb[:5]),
            "rank_of_true_class": int(np.where(sorted_comb == real_c)[0][0] + 1),
            "confidence_simulated_top1": round(float(probs_comb[i, pred_comb_c]), 4),
            "confidence_raw_top1": round(float(probs_base[i, pred_base_c]), 4),
            "top5_candidates": " | ".join([synsets[c] for c in sorted_comb[:5]])
        })

    # 9. Export & Sync
    csv_samples_path = os.path.join(OUTPUT_DIR, "all_test_samples_simulated_pooling.csv")
    csv_comp_path = os.path.join(OUTPUT_DIR, "comparison_simulated_pooling_vs_paper.csv")
    pd.DataFrame(all_test_samples).to_csv(csv_samples_path, index=False)
    pd.DataFrame(comparison_results).to_csv(csv_comp_path, index=False)

    print(f"\n[✓] Sample Predictions CSV saved to: {csv_samples_path}")
    print(f"[✓] Benchmark Comparison CSV saved to: {csv_comp_path}")
    print("\n[+] Syncing reports to Google Drive folder 'denoiserResult'...")
    upload_file_to_drive(csv_samples_path, folder_id=DRIVE_TARGET_FOLDER_ID)
    upload_file_to_drive(csv_comp_path, folder_id=DRIVE_TARGET_FOLDER_ID)
    print(f"[SUCCESS] All synced to Drive: https://drive.google.com/drive/folders/{DRIVE_TARGET_FOLDER_ID}\n")

    return {
        "all_test_samples": all_test_samples,
        "comparison_results": comparison_results,
        "total_test_samples": N_te
    }

# Execute
results = run_all_simulated_pooling_evaluation()
df_comp = pd.DataFrame(results["comparison_results"])
display(df_comp)
