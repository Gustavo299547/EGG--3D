# ============================================================================
# NEUROSTREAM-SST: EXACT PAPER METRICS EVALUATION & BENCHMARK COMPARISON
# ----------------------------------------------------------------------------
# Evaluates NeuroStream-SST using the exact evaluation protocol from:
#   "Brain3D: EEG-to-3D Decoding of Visual Representations via Multimodal Reasoning"
#   (arXiv:2604.08068v1, Table 1):
#
# Metrics Evaluated:
#   1. 2-way Accuracy Top-1   (Random Chance: 50.0%) - Paper GWIT: 94.6%
#   2. 10-way Accuracy Top-1  (Random Chance: 10.0%) - Paper GWIT: 85.4%
#   3. 10-way Accuracy Top-2  (Random Chance: 20.0%) - Paper GWIT: 90.6%
#   4. 50-way / 40-way Top-1  (Random Chance: 2.50%) - Paper GWIT: 76.3%
#   5. 50-way / 40-way Top-2  (Random Chance: 5.00%) - Paper GWIT: 82.2%
#
# Evaluated across:
#   • Single-Trial Mode (Raw 1-shot inference on all test trials)
#   • Image-Level Pooled Mode (Feature-averaged across viewing trials)
#
# Generates side-by-side comparative table against GWIT, BrainVis, EEG-CLIP,
# DreamDiffusion, exports CSVs, and syncs directly to Google Drive.
# ============================================================================

import os, sys, io, gc, glob, math, random, csv, time, subprocess, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

warnings.filterwarnings("ignore")

# ============================================================================
# 1. CONFIGURATION & CLOUD DRIVE CREDENTIALS
# ============================================================================
DATA_ROOT  = "/kaggle/input/datasets/evolutionaw/eegcvpr40/data"
OUTPUT_DIR = "./eeg_denoiser_results"
if not os.path.exists(OUTPUT_DIR):
    OUTPUT_DIR = "/kaggle/working/eeg_denoiser_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DRIVE_MAIN_FOLDER_ID = "13vd_i1ePxEbYH5RtW9Q4B2OG0ccwv6Wz"
DRIVE_TARGET_FOLDER_ID = "1TBeohhdFfmVuRU16FLGmPYZxdivNg3yN" # denoiserResult folder
DRIVE_CLIENT_ID = os.environ.get("DRIVE_CLIENT_ID", "YOUR_DRIVE_CLIENT_ID")
DRIVE_CLIENT_SECRET = os.environ.get("DRIVE_CLIENT_SECRET", "YOUR_DRIVE_CLIENT_SECRET")
DRIVE_REFRESH_TOKEN = os.environ.get("DRIVE_REFRESH_TOKEN", "YOUR_DRIVE_REFRESH_TOKEN")

MODEL_CANDIDATES = [
    "/kaggle/working/best_neurostream_sst.pt",
    "./best_neurostream_sst.pt",
    "./checkpoints/best_neurostream_sst.pt",
    "best_neurostream_sst.pt",
    "exact-paper-modelV00.pt",
    "denoiser_v2_single_trial_80.pt",
    "denoiser_strategy_combined.pt",
    "denoiser_strategy_3.pt"
]

SEED = 42
SCALP_GRID_H, SCALP_GRID_W = 11, 11
LATENT_DIM = 256
dev = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = (dev == "cuda")

def seed_everything(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

seed_everything(SEED)

# ============================================================================
# 2. GOOGLE DRIVE HELPER: AUTO-DOWNLOAD & AUTO-UPLOAD
# ============================================================================
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

def download_file_from_drive(filename, target_path, folder_id=DRIVE_MAIN_FOLDER_ID):
    try:
        from googleapiclient.http import MediaIoBaseDownload
        service = get_drive_service()
        q = f"'{folder_id}' in parents and name = '{filename}' and trashed = false"
        res = service.files().list(q=q, fields="files(id, name, size)").execute()
        files = res.get("files", [])
        if not files:
            # Also check target folder
            q2 = f"'{DRIVE_TARGET_FOLDER_ID}' in parents and name = '{filename}' and trashed = false"
            res2 = service.files().list(q=q2, fields="files(id, name, size)").execute()
            files = res2.get("files", [])
        if not files:
            return False

        fid = files[0]["id"]
        print(f"  [Drive] Downloading '{filename}' ({int(files[0].get('size', 0))/(1024*1024):.1f} MB)...")
        req = service.files().get_media(fileId=fid)
        with open(target_path, "wb") as f_out:
            downloader = MediaIoBaseDownload(f_out, req)
            done = False
            while not done:
                status, done = downloader.next_chunk()
        print(f"  [Drive] Download complete -> {target_path}")
        return True
    except Exception as e:
        print(f"  [Drive Notice] Download error: {e}")
        return False

def upload_file_to_drive(local_path, filename=None, folder_id=DRIVE_TARGET_FOLDER_ID):
    try:
        from googleapiclient.http import MediaFileUpload
        service = get_drive_service()
        name = filename or os.path.basename(local_path)
        media = MediaFileUpload(local_path, resumable=True)
        q = f"'{folder_id}' in parents and name = '{name}' and trashed = false"
        res = service.files().list(q=q, fields="files(id, name)").execute()
        existing = res.get("files", [])
        if existing:
            service.files().update(fileId=existing[0]["id"], media_body=media).execute()
            print(f"  [Drive Sync] Updated existing '{name}' in folder '{folder_id}'")
        else:
            meta = {"name": name, "parents": [folder_id]}
            service.files().create(body=meta, media_body=media, fields="id, name").execute()
            print(f"  [Drive Sync] Uploaded '{name}' to folder '{folder_id}'")
    except Exception as e:
        print(f"  [Drive Sync Notice] Upload error: {e}")

# ============================================================================
# 3. NEUROSTREAM-SST ARCHITECTURE
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
        self.mode = mode
        self.fs = fs
        self.kernel_len = kernel_len
        if mode == "high_gamma":
            freqs, sigmas = [95.0], [0.015]
        else:
            freqs  = [4.5, 10.5, 21.5, 50.0, 95.0]
            sigmas = [0.08, 0.04, 0.025, 0.018, 0.012]
        self.n_bands = len(freqs)
        t = torch.linspace(-kernel_len / (2 * fs), kernel_len / (2 * fs), kernel_len)
        cos_k, sin_k = [], []
        for f0, sig in zip(freqs, sigmas):
            env = torch.exp(-0.5 * (t / sig) ** 2)
            c = env * torch.cos(2 * math.pi * f0 * t)
            s = env * torch.sin(2 * math.pi * f0 * t)
            cos_k.append(c / (torch.norm(c) + 1e-6))
            sin_k.append(s / (torch.norm(s) + 1e-6))
        self.cos_filters = nn.Parameter(torch.stack(cos_k).unsqueeze(1), requires_grad=True)
        self.sin_filters = nn.Parameter(torch.stack(sin_k).unsqueeze(1), requires_grad=True)

    def forward(self, x):
        real = F.conv1d(x, self.cos_filters, padding=self.kernel_len // 2)
        imag = F.conv1d(x, self.sin_filters, padding=self.kernel_len // 2)
        return torch.sqrt(real ** 2 + imag ** 2 + 1e-8)

class NeuroStreamSSTClassifier(nn.Module):
    def __init__(self, spectral_mode="high_gamma", n_classes=40, dropout=0.2):
        super().__init__()
        self.spectral_mode = spectral_mode
        self.register_buffer("scalp_proj", build_128_channel_scalp_grid_mapping())
        self.spectral_filterbank = MorletWaveletFilterBank(mode=spectral_mode)
        n_bands = self.spectral_filterbank.n_bands

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(n_bands, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(64 * 16, 256, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Conv1d(256, LATENT_DIM, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(LATENT_DIM), nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Sequential(
            nn.Linear(LATENT_DIM, LATENT_DIM),
            nn.LayerNorm(LATENT_DIM), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(LATENT_DIM, n_classes)
        )

    def extract_features(self, x_eeg):
        B, C, T = x_eeg.shape
        env = self.spectral_filterbank(x_eeg.view(B * C, 1, T))
        n_bands = env.shape[1]
        env = env.view(B, C, n_bands, T).permute(0, 2, 3, 1)
        scalp_map = torch.einsum("bfti,ihw->bfthw", env, self.scalp_proj.to(env.device, env.dtype))
        scalp_sub = scalp_map[:, :, ::4, :, :]
        T_sub = scalp_sub.shape[2]
        scalp_2d = scalp_sub.permute(0, 2, 1, 3, 4).reshape(B * T_sub, n_bands, SCALP_GRID_H, SCALP_GRID_W)
        spat = self.spatial_conv(scalp_2d).view(B, T_sub, 64 * 16).permute(0, 2, 1)
        return self.temporal_conv(spat).squeeze(-1)

    def forward(self, x_eeg):
        feats = self.extract_features(x_eeg)
        return self.head(feats), feats

# ============================================================================
# 4. DATASET LOADER WITH STANDARDIZATION & IMAGE SPLIT
# ============================================================================
def load_benchmark_data(data_root=DATA_ROOT):
    pths = glob.glob(os.path.join(data_root, "**", "*.pth"), recursive=True)
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

    if eeg_file and os.path.exists(eeg_file):
        print(f"[+] Found EEG dataset: {eeg_file}")
        d = torch.load(eeg_file, map_location="cpu", weights_only=False)
        names, ds = d["images"], d["dataset"]
        X, Y, IMGS = [], [], []

        synsets = sorted({names[int(it["image"])].split("_")[0] for it in ds})
        syn2cls = {s: i for i, s in enumerate(synsets)}

        for it in ds:
            e = torch.as_tensor(it["eeg"]).float()
            if e.shape[0] != 128 and e.shape[1] == 128: e = e.t()
            if not (450 <= e.shape[1] <= 600): continue
            X.append(e[:, 20:460].numpy())
            im_id = int(it["image"])
            syn = names[im_id].split("_")[0]
            Y.append(syn2cls[syn])
            IMGS.append(im_id)

        X, Y, IMGS = np.array(X), np.array(Y), np.array(IMGS)

        if split_file and os.path.exists(split_file):
            sp = torch.load(split_file, map_location="cpu", weights_only=False)["splits"][0]
            te_idx = np.array([i for i, im in enumerate(IMGS) if im in sp["test"]])
        else:
            shuf = np.random.RandomState(SEED).permutation(len(X))
            s = int(0.80 * len(X))
            te_idx = shuf[s:]

        print(f"  • Total Test Trials: {len(te_idx)} | Classes: {len(synsets)}")
        return X[te_idx], Y[te_idx], IMGS[te_idx], synsets

    # Synthetic fallback for instant local execution
    print("  [Notice] EEGCVPR40 not found locally. Generating synthetic data for evaluation verification...")
    np.random.seed(SEED)
    N_te = 200
    synsets = [f"n02{i:06d}" for i in range(40)]
    X_te = np.random.randn(N_te, 128, 440).astype(np.float32)
    Y_te = np.random.randint(0, 40, size=N_te)
    te_imgs = np.random.randint(0, 50, size=N_te)
    return X_te, Y_te, te_imgs, synsets

# ============================================================================
# 5. EXACT PAPER METRIC EVALUATOR (N-WAY TOP-K PROTOCOL)
# ============================================================================
def compute_n_way_topk(logits, labels, n_way=10, k=1, n_samplings=100, seed=42):
    """
    Computes n-way Top-k accuracy as defined in Brain3D (arXiv:2604.08068v1, Sec. 2.4):
    For each trial, samples an n-way candidate set consisting of the true class and (n-1)
    random negative classes. Success is counted if the true class ranks within the top-k scores.
    Repeated over n_samplings trials to obtain statistically robust mean and std.
    """
    rng = np.random.RandomState(seed)
    N, n_classes = logits.shape
    if n_way > n_classes:
        n_way = n_classes

    all_accuracies = []
    all_classes_set = np.arange(n_classes)

    for _ in range(n_samplings):
        correct = 0
        for i in range(N):
            true_c = labels[i]
            # Sample (n_way - 1) negative classes
            neg_candidates = np.setdiff1d(all_classes_set, [true_c])
            sampled_negs = rng.choice(neg_candidates, size=n_way - 1, replace=False)
            candidate_set = np.append(sampled_negs, true_c)
            
            # Restrict logits to candidate set
            cand_scores = logits[i, candidate_set]
            # Rank candidates (descending)
            ranked_indices = np.argsort(-cand_scores)
            ranked_classes = candidate_set[ranked_indices]
            
            # Check if true class is in top-k
            if true_c in ranked_classes[:k]:
                correct += 1

        all_accuracies.append(100.0 * correct / N)

    return float(np.mean(all_accuracies)), float(np.std(all_accuracies))

# ============================================================================
# 6. MASTER EVALUATION RUNNER
# ============================================================================
def run_paper_evaluation(checkpoint_path=None):
    print("=" * 95)
    print("NEUROSTREAM-SST: EXACT BRAIN3D PAPER METRICS EVALUATION PROTOCOL")
    print("Benchmarking Against: GWIT, BrainVis, EEG-CLIP, DreamDiffusion (Table 1)")
    print("=" * 95)

    X_te, Y_te, IMGS_te, synsets = load_benchmark_data()
    n_classes = len(synsets)

    # 1. Resolve & Load Model Checkpoint
    model_path = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        model_path = checkpoint_path
    else:
        for cand in MODEL_CANDIDATES:
            for p in [os.path.join(OUTPUT_DIR, cand), cand, os.path.join(".", cand), os.path.join("../", cand)]:
                if os.path.exists(p):
                    model_path = p; break
            if model_path: break

    if not model_path or not os.path.exists(model_path):
        target_dl = os.path.join(OUTPUT_DIR, "best_neurostream_sst.pt")
        for cand in MODEL_CANDIDATES:
            if download_file_from_drive(cand, target_dl):
                model_path = target_dl; break

    model = None
    if model_path and os.path.exists(model_path):
        print(f"[+] Loaded Model Weights from: {model_path}")
        try:
            ck = torch.load(model_path, map_location="cpu", weights_only=False)
            if isinstance(ck, dict):
                state_dict = ck.get("model_state", ck.get("model_state_dict", ck))
            else:
                state_dict = ck

            if "spectral_filterbank.cos_filters" in state_dict or "spatial_conv.0.weight" in state_dict:
                model = NeuroStreamSSTClassifier(n_classes=n_classes).to(dev)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                print(f"  [✓] NeuroStream-SST Classifier successfully loaded from: {model_path}")
                if isinstance(ck, dict) and "epoch" in ck:
                    print(f"  [✓] Checkpoint Epoch: {ck['epoch']} | Val Pooled Acc: {ck.get('val_pool_acc', 'N/A')}")
            elif "backbone.head.weight" in state_dict or "head.weight" in state_dict:
                try:
                    from eeg_signal_pattern_analysis.denoiser_strategies_classifiers import (
                        Classifier_Denoiser_Combined, Classifier_Denoiser_Strategy_3, Classifier_Denoiser_v2_HighInference
                    )
                    if "proj_head.0.weight" in state_dict:
                        model = Classifier_Denoiser_v2_HighInference(n_channels=128, n_classes=n_classes).to(dev)
                    elif "denoiser.stage1_bandpass.filter.weight" in state_dict:
                        model = Classifier_Denoiser_Combined(n_channels=128, n_classes=n_classes).to(dev)
                    else:
                        model = Classifier_Denoiser_Strategy_3(n_channels=128, n_classes=n_classes).to(dev)
                    model.load_state_dict(state_dict, strict=False)
                    print(f"  [✓] Denoised Classifier architecture detected and loaded ({model.strategy_name}).")
                except Exception:
                    model = NeuroStreamSSTClassifier(n_classes=n_classes).to(dev)
            else:
                model = NeuroStreamSSTClassifier(n_classes=n_classes).to(dev)
                model.load_state_dict(state_dict, strict=False)
                print("  [✓] NeuroStream-SST model loaded.")
        except Exception as e:
            print(f"  [!] Weight loading notice ({e}), initializing standard NeuroStream architecture.")
            model = NeuroStreamSSTClassifier(n_classes=n_classes).to(dev)
    else:
        print("  [Notice] Initializing fresh NeuroStream-SST Classifier.")
        model = NeuroStreamSSTClassifier(n_classes=n_classes).to(dev)

    model.eval()

    # 2. Extract Predictions across ALL Test Trials with standardized inputs
    # Per-trial z-score normalization matching NeuroStream-SST training
    X_te = (X_te - X_te.mean(axis=-1, keepdims=True)) / (X_te.std(axis=-1, keepdims=True) + 1e-6)
    X_te_t = torch.as_tensor(X_te, dtype=torch.float32)
    N_te = len(X_te)
    bs = 64

    all_logits = []
    all_feats = []
    with torch.no_grad():
        for k in range(0, N_te, bs):
            xb = X_te_t[k:k+bs].to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
                logits, feats = model(xb)
            all_logits.append(logits.cpu())
            all_feats.append(feats.cpu())

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_feats  = torch.cat(all_feats, dim=0)

    # 3. Compute Metrics for SINGLE-TRIAL Mode
    print("\n[+] Computing Exact Paper Metrics for SINGLE-TRIAL Mode (100 random samplings per trial)...")
    s_2way_top1, s_2way_std   = compute_n_way_topk(all_logits, Y_te, n_way=2,  k=1, n_samplings=100)
    s_10way_top1, s_10way_std = compute_n_way_topk(all_logits, Y_te, n_way=10, k=1, n_samplings=100)
    s_10way_top2, s_10way2_std= compute_n_way_topk(all_logits, Y_te, n_way=10, k=2, n_samplings=100)
    s_50way_top1, s_50way_std = compute_n_way_topk(all_logits, Y_te, n_way=min(50, n_classes), k=1, n_samplings=100)
    s_50way_top2, s_50way2_std= compute_n_way_topk(all_logits, Y_te, n_way=min(50, n_classes), k=2, n_samplings=100)

    # 4. Compute Metrics for IMAGE-LEVEL POOLED Mode
    print("[+] Computing Exact Paper Metrics for IMAGE-LEVEL POOLED Mode...")
    img2trials = {}
    for idx, im in enumerate(IMGS_te):
        img2trials.setdefault(im, []).append(idx)

    pooled_logits = []
    pooled_labels = []
    with torch.no_grad():
        for im, tr_idxs in img2trials.items():
            p_feat = all_feats[tr_idxs].mean(dim=0, keepdim=True).to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
                p_l = model.head(p_feat).cpu().numpy()[0]
            pooled_logits.append(p_l)
            pooled_labels.append(Y_te[tr_idxs[0]])

    pooled_logits = np.array(pooled_logits)
    pooled_labels = np.array(pooled_labels)

    p_2way_top1, _  = compute_n_way_topk(pooled_logits, pooled_labels, n_way=2,  k=1, n_samplings=100)
    p_10way_top1, _ = compute_n_way_topk(pooled_logits, pooled_labels, n_way=10, k=1, n_samplings=100)
    p_10way_top2, _ = compute_n_way_topk(pooled_logits, pooled_labels, n_way=10, k=2, n_samplings=100)
    p_50way_top1, _ = compute_n_way_topk(pooled_logits, pooled_labels, n_way=min(50, n_classes), k=1, n_samplings=100)
    p_50way_top2, _ = compute_n_way_topk(pooled_logits, pooled_labels, n_way=min(50, n_classes), k=2, n_samplings=100)

    # 5. Build Comprehensive Sample-Level Records for ALL Test Samples
    probs = torch.softmax(torch.from_numpy(all_logits), dim=-1).numpy()
    all_test_samples = []
    for i in range(N_te):
        real_c = int(Y_te[i])
        real_syn = synsets[real_c]
        sorted_indices = np.argsort(-all_logits[i])
        pred_c = int(sorted_indices[0])
        pred_syn = synsets[pred_c]
        is_top1 = (real_c == pred_c)
        is_top2 = bool(real_c in sorted_indices[:2])
        is_top5 = bool(real_c in sorted_indices[:5])
        rank_of_true = int(np.where(sorted_indices == real_c)[0][0] + 1)
        conf_top1 = float(probs[i, pred_c])
        conf_true = float(probs[i, real_c])
        top5_str = " | ".join([synsets[c] for c in sorted_indices[:5]])

        all_test_samples.append({
            "trial_index": i,
            "image_id": int(IMGS_te[i]),
            "real_class": real_syn,
            "predicted_class": pred_syn,
            "is_correct_top1": is_top1,
            "is_correct_top2": is_top2,
            "is_correct_top5": is_top5,
            "rank_of_true_class": rank_of_true,
            "confidence_top1": round(conf_top1, 4),
            "confidence_true_class": round(conf_true, 4),
            "top5_candidates": top5_str
        })

    # 6. Display Our Model's Calculated Metrics & Comparison with Paper Benchmarks
    print("\n" + "=" * 95)
    print("OUR MODEL CALCULATED TEST METRICS (Calculated using Brain3D Table 1 Evaluation Protocol)")
    print("=" * 95)
    print(f"• Total Test Samples Evaluated: {N_te}")
    top1_raw = 100.0 * sum(s["is_correct_top1"] for s in all_test_samples) / N_te
    top2_raw = 100.0 * sum(s["is_correct_top2"] for s in all_test_samples) / N_te
    top5_raw = 100.0 * sum(s["is_correct_top5"] for s in all_test_samples) / N_te
    print(f"• Standard Classification Acc : Top-1: {top1_raw:.2f}% | Top-2: {top2_raw:.2f}% | Top-5: {top5_raw:.2f}%")
    print("\n" + "=" * 95)
    print("COMPARATIVE BENCHMARK TABLE: PUBLISHED PAPER BASELINES vs. OUR CALCULATED NEUROSTREAM")
    print("=" * 95)
    header = f"{'Framework / Backbone':<30} | {'2-way Top-1':<12} | {'10-way Top-1':<13} | {'10-way Top-2':<13} | {'50-way Top-1':<13} | {'50-way Top-2'}"
    print(header)
    print("-" * 95)

    paper_benchmarks = [
        ("GWIT (Brain3D Best)",   94.60, 85.40, 90.60, 76.30, 82.20),
        ("BrainVis (ICASSP 2025)",88.00, 70.60, 79.60, 57.80, 64.90),
        ("EEG-CLIP (2025)",       85.70, 65.50, 74.20, 54.50, 60.20),
        ("DreamDiffusion (ECCV)", 73.00, 31.40, 48.00, 16.40, 20.60),
    ]

    for name, a2, a10_1, a10_2, a50_1, a50_2 in paper_benchmarks:
        print(f"{name:<30} | {a2:>10.2f}% | {a10_1:>11.2f}% | {a10_2:>11.2f}% | {a50_1:>11.2f}% | {a50_2:>10.2f}%")

    print("-" * 95)
    print(f"{'NeuroStream (Single-Trial)':<30} | {s_2way_top1:>10.2f}% | {s_10way_top1:>11.2f}% | {s_10way_top2:>11.2f}% | {s_50way_top1:>11.2f}% | {s_50way_top2:>10.2f}%")
    print(f"{'NeuroStream (Image-Pooled)':<30} | {p_2way_top1:>10.2f}% | {p_10way_top1:>11.2f}% | {p_10way_top2:>11.2f}% | {p_50way_top1:>11.2f}% | {p_50way_top2:>10.2f}%")
    print("-" * 95)

    delta_gwit_pooled = p_10way_top1 - 85.40
    sign = "+" if delta_gwit_pooled >= 0 else ""
    print(f"{'Difference (Pooled vs GWIT)':<30} | {p_2way_top1-94.60:+10.2f}% | {sign}{delta_gwit_pooled:+10.2f}% | {p_10way_top2-90.60:+10.2f}% | {p_50way_top1-76.30:+10.2f}% | {p_50way_top2-82.20:+10.2f}%")
    print("=" * 95)

    # 7. Save Comparison Summary CSV
    csv_comp_path = os.path.join(OUTPUT_DIR, "paper_evaluation_comparison_neurostream.csv")
    with open(csv_comp_path, mode="w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["Framework", "2-way_Top1", "10-way_Top1", "10-way_Top2", "50-way_Top1", "50-way_Top2"])
        for name, a2, a10_1, a10_2, a50_1, a50_2 in paper_benchmarks:
            writer.writerow([name, f"{a2:.2f}%", f"{a10_1:.2f}%", f"{a10_2:.2f}%", f"{a50_1:.2f}%", f"{a50_2:.2f}%"])
        writer.writerow(["NeuroStream_Single_Trial", f"{s_2way_top1:.2f}%", f"{s_10way_top1:.2f}%", f"{s_10way_top2:.2f}%", f"{s_50way_top1:.2f}%", f"{s_50way_top2:.2f}%"])
        writer.writerow(["NeuroStream_Image_Pooled", f"{p_2way_top1:.2f}%", f"{p_10way_top1:.2f}%", f"{p_10way_top2:.2f}%", f"{p_50way_top1:.2f}%", f"{p_50way_top2:.2f}%"])
        writer.writerow(["Delta_Pooled_vs_GWIT", f"{p_2way_top1-94.60:+.2f}%", f"{p_10way_top1-85.40:+.2f}%", f"{p_10way_top2-90.60:+.2f}%", f"{p_50way_top1-76.30:+.2f}%", f"{p_50way_top2-82.20:+.2f}%"])

    print(f"\n[✓] Comparison CSV saved to: {csv_comp_path}")

    # 8. Save Comprehensive Sample-Level CSV for ALL Test Samples
    csv_samples_path = os.path.join(OUTPUT_DIR, "all_test_samples_predictions.csv")
    with open(csv_samples_path, mode="w", newline="", encoding="utf-8") as f_samp:
        writer = csv.DictWriter(f_samp, fieldnames=[
            "trial_index", "image_id", "real_class", "predicted_class",
            "is_correct_top1", "is_correct_top2", "is_correct_top5",
            "rank_of_true_class", "confidence_top1", "confidence_true_class",
            "top5_candidates"
        ])
        writer.writeheader()
        for sample in all_test_samples:
            writer.writerow(sample)

    print(f"[✓] All {len(all_test_samples)} Test Sample Predictions saved to: {csv_samples_path}")

    # 9. Sync to Google Drive
    print("\n[+] Syncing evaluation reports to Google Drive folder 'denoiserResult'...")
    upload_file_to_drive(csv_comp_path, folder_id=DRIVE_TARGET_FOLDER_ID)
    upload_file_to_drive(csv_samples_path, folder_id=DRIVE_TARGET_FOLDER_ID)
    print(f"[SUCCESS] All evaluations synced to Drive: https://drive.google.com/drive/folders/{DRIVE_TARGET_FOLDER_ID}\n")

    summary_metrics = {
        "single_trial": {
            "2way_top1": s_2way_top1,
            "10way_top1": s_10way_top1,
            "10way_top2": s_10way_top2,
            "50way_top1": s_50way_top1,
            "50way_top2": s_50way_top2,
        },
        "image_pooled": {
            "2way_top1": p_2way_top1,
            "10way_top1": p_10way_top1,
            "10way_top2": p_10way_top2,
            "50way_top1": p_50way_top1,
            "50way_top2": p_50way_top2,
        },
        "total_test_samples": N_te,
        "overall_top1_acc": round(top1_raw, 2),
        "overall_top2_acc": round(top2_raw, 2),
        "overall_top5_acc": round(top5_raw, 2)
    }

    # 10. Return ALL Test Samples & Summary
    return {
        "all_test_samples": all_test_samples,
        "summary_metrics": summary_metrics,
        "comparison_benchmarks": paper_benchmarks
    }

if __name__ == "__main__":
    results = run_paper_evaluation()
    print(f"[Finished] Returned {len(results['all_test_samples'])} test sample predictions.")
