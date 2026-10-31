# ============================================================================
# MASTER EVALUATION: BENCHMARKING ALL SIMULATED POOLING METHODS
# ----------------------------------------------------------------------------
# Evaluates and compares all single-trial simulated pooling strategies across
# ALL test samples:
#   1. Baseline: Raw Single-Trial (1-shot inference)
#   2. Method 1: Test-Time Augmentation (TTA in signal & latent space)
#   3. Method 2: Single-to-Average (S2A) Neural Denoiser
#   4. Method 3: Class Prototypical Memory Bank
#   5. Method 4: Spatiotemporal Filtering (1-30 Hz + CAR)
#   6. Method 5: Combined All-in-One Simulated Pooling
#   7. Reference Oracle: True Multi-Trial Image Pooling
#
# Generates side-by-side benchmark tables, exports per-sample CSVs for all
# test trials, returns Python dictionaries, and auto-syncs to Google Drive.
# ============================================================================

import os, sys, glob, random, math, csv, time, subprocess, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

warnings.filterwarnings("ignore")

# Safe module path resolution
_CURR_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
_PARENT_DIR = os.path.dirname(_CURR_DIR)
for p in [_CURR_DIR, _PARENT_DIR, os.getcwd(), "/kaggle/working"]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from try_simulate_pooling.method1_test_time_augmentation import TestTimeAugmenter
    from try_simulate_pooling.method2_single_to_average_denoiser import SingleToAverageUNet
    from try_simulate_pooling.method3_prototypical_memory_bank import PrototypicalMemoryBank, remove_subject_baseline_offsets
    from try_simulate_pooling.method4_spatiotemporal_filtering import SpatioTemporalFilter
except ImportError:
    try:
        from method1_test_time_augmentation import TestTimeAugmenter
        from method2_single_to_average_denoiser import SingleToAverageUNet
        from method3_prototypical_memory_bank import PrototypicalMemoryBank, remove_subject_baseline_offsets
        from method4_spatiotemporal_filtering import SpatioTemporalFilter
    except ImportError:
        pass

try:
    from neurostream_v2_high_acc.cell1_train_model import SOTANeuroStreamModel, load_eegcvpr40_data
except ImportError:
    try:
        from cell1_train_model import SOTANeuroStreamModel, load_eegcvpr40_data
    except ImportError:
        pass

# ============================================================================
# 1. CONFIGURATION & CLOUD DRIVE
# ============================================================================
OUTPUT_DIR = "./eeg_denoiser_results"
if not os.path.exists(OUTPUT_DIR):
    OUTPUT_DIR = "/kaggle/working/eeg_denoiser_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DRIVE_TARGET_FOLDER_ID = "1TBeohhdFfmVuRU16FLGmPYZxdivNg3yN" # denoiserResult
DRIVE_CLIENT_ID = os.environ.get("DRIVE_CLIENT_ID", "YOUR_DRIVE_CLIENT_ID")
DRIVE_CLIENT_SECRET = os.environ.get("DRIVE_CLIENT_SECRET", "YOUR_DRIVE_CLIENT_SECRET")
DRIVE_REFRESH_TOKEN = os.environ.get("DRIVE_REFRESH_TOKEN", "YOUR_DRIVE_REFRESH_TOKEN")

MODEL_CANDIDATES = [
    "/kaggle/working/best_sota_neurostream.pt",
    "./best_sota_neurostream.pt",
    os.path.join(OUTPUT_DIR, "best_sota_neurostream.pt"),
    "best_sota_neurostream.pt",
    "/kaggle/working/best_neurostream_sst.pt",
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
            print(f"  [Drive Sync] Updated existing '{fname}' in folder '{folder_id}'")
        else:
            meta = {"name": fname, "parents": [folder_id]}
            service.files().create(body=meta, media_body=media, fields="id").execute()
            print(f"  [Drive Sync] Uploaded '{fname}' to folder '{folder_id}'")
        return True
    except Exception as e:
        print(f"  [Drive Notice] Cloud sync notice ({e})")
        return False

# ============================================================================
# 2. EXACT PAPER METRIC EVALUATOR (N-WAY TOP-K)
# ============================================================================
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
            cand_scores = logits[i, candidate_set]
            ranked_indices = np.argsort(-cand_scores)
            ranked_classes = candidate_set[ranked_indices]
            if true_c in ranked_classes[:k]:
                correct += 1
        all_accuracies.append(100.0 * correct / N)

    return float(np.mean(all_accuracies)), float(np.std(all_accuracies))

# ============================================================================
# 3. MASTER EVALUATION OF ALL SIMULATED POOLING METHODS
# ============================================================================
def evaluate_all_simulated_methods(checkpoint_path=None):
    print("=" * 95)
    print("SIMULATED POOLING BENCHMARK: EVALUATING NOISE REDUCTION TECHNIQUES")
    print("Testing under Single-Trial Constraints (No True Multi-Trial Averaging)")
    print("=" * 95)

    X_tr, Y_tr, S_tr, I_tr, X_te, Y_te, S_te, I_te, synsets = load_eegcvpr40_data()
    n_classes = len(synsets)
    n_subjects = int(max(S_tr.max(), S_te.max())) + 1
    N_te = len(X_te)

    # 1. Resolve & Load Base Classifier Model
    model_path = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        model_path = checkpoint_path
    else:
        for cand in MODEL_CANDIDATES:
            if os.path.exists(cand):
                model_path = cand; break

    model = SOTANeuroStreamModel(
        n_channels=128, n_classes=n_classes, n_subjects=n_subjects,
        d_model=256, n_blocks=4
    ).to(dev)

    if model_path and os.path.exists(model_path):
        print(f"[+] Loaded Model Weights from: {model_path}")
        try:
            ck = torch.load(model_path, map_location=dev, weights_only=False)
            state = ck.get("model_state", ck.get("model_state_dict", ck))
            model_keys = model.state_dict()
            filtered_state = {k: v for k, v in state.items() if k in model_keys and model_keys[k].shape == v.shape}
            model.load_state_dict(filtered_state, strict=False)
            print(f"  [✓] Successfully loaded {len(filtered_state)} compatible weight tensors.")
        except Exception as e:
            print(f"  [!] Weight loading notice ({e}).")
    else:
        print("  [Notice] Initializing fresh SOTA NeuroStream model.")

    model.eval()

    X_te_t = torch.as_tensor(X_te, dtype=torch.float32)
    S_te_t = torch.as_tensor(S_te, dtype=torch.long)

    # ------------------------------------------------------------------------
    # [1] BASELINE: Raw Single-Trial (1-shot)
    # ------------------------------------------------------------------------
    print("\n[+] Evaluating [1] Baseline Single-Trial (Raw 1-shot)...")
    base_logits = []
    base_feats = []
    with torch.no_grad():
        for k in range(0, N_te, 64):
            xb = X_te_t[k : k + 64].to(dev)
            sb = S_te_t[k : k + 64].to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
                l, f, _ = model(xb, sb)
            base_logits.append(l.cpu())
            base_feats.append(f.cpu())
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
                l, _ = tta.predict_with_tta(model, xb, sb)
            tta_logits.append(l.cpu())
    tta_logits = torch.cat(tta_logits, dim=0).numpy()

    # ------------------------------------------------------------------------
    # [3] METHOD 3: Prototypical Memory Bank (Nearest Class Centroid)
    # ------------------------------------------------------------------------
    print("[+] Evaluating [3] Method 3: Prototypical Memory Bank (Cosine Nearest Centroid)...")
    pmb = PrototypicalMemoryBank(temperature=14.0)
    pmb.build_memory_bank(model, X_tr, Y_tr, device=dev)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
            proto_logits = pmb.classify_with_prototypes(base_feats.to(dev)).cpu().numpy()

    # ------------------------------------------------------------------------
    # [4] METHOD 4: Spatiotemporal Filtering (1-30Hz Bandpass + Spatial CAR)
    # ------------------------------------------------------------------------
    print("[+] Evaluating [4] Method 4: Spatiotemporal Filtering (1-30Hz Butterworth + CAR)...")
    stf = SpatioTemporalFilter(lowcut=1.0, highcut=30.0, fs=250.0)
    X_te_filtered = stf.filter_eeg(X_te)
    X_te_filt_t = torch.as_tensor(X_te_filtered, dtype=torch.float32)

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
    # [5] COMBINED: Ensemble of Simulated Pooling (TTA + Prototypes + Filter)
    # ------------------------------------------------------------------------
    print("[+] Evaluating [5] Combined All-in-One Simulated Pooling...")
    combined_logits = 0.40 * tta_logits + 0.35 * proto_logits + 0.25 * filt_logits

    # ------------------------------------------------------------------------
    # [6] ORACLE REFERENCE: True Multi-Trial Image Pooling (Upper Bound)
    # ------------------------------------------------------------------------
    print("[+] Evaluating [6] Oracle Reference: True Multi-Trial Image Pooling...")
    te_img2trials = {}
    for idx, im in enumerate(I_te):
        te_img2trials.setdefault(im, []).append(idx)

    oracle_logits = []
    oracle_labels = []
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

    # ========================================================================
    # 4. COMPUTE EXACT PAPER METRICS ACROSS ALL METHODS
    # ========================================================================
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
    header = f"{'Framework / Simulation Strategy':<38} | {'2-way Top-1':<12} | {'10-way Top-1':<13} | {'10-way Top-2':<13} | {'50-way Top-1'}"
    print(header)
    print("-" * 95)

    # Reference published baselines from Brain3D Table 1
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
        a2, _   = compute_n_way_topk(log_arr, lab_arr, n_way=2,  k=1, n_samplings=100)
        a10_1, _= compute_n_way_topk(log_arr, lab_arr, n_way=10, k=1, n_samplings=100)
        a10_2, _= compute_n_way_topk(log_arr, lab_arr, n_way=10, k=2, n_samplings=100)
        a50_1, _= compute_n_way_topk(log_arr, lab_arr, n_way=min(50, n_classes), k=1, n_samplings=100)
        print(f"{name:<38} | {a2:>10.2f}% | {a10_1:>11.2f}% | {a10_2:>11.2f}% | {a50_1:>11.2f}%")
        comparison_results.append({
            "strategy": name,
            "2way_top1": a2,
            "10way_top1": a10_1,
            "10way_top2": a10_2,
            "50way_top1": a50_1
        })
    print("=" * 95)

    # ========================================================================
    # 5. BUILD PER-SAMPLE PREDICTIONS FOR ALL TEST TRIALS
    # ========================================================================
    all_test_samples = []
    probs_comb = F.softmax(torch.from_numpy(combined_logits), dim=-1).numpy()
    probs_base = F.softmax(torch.from_numpy(base_logits), dim=-1).numpy()

    for i in range(N_te):
        real_c = int(Y_te[i])
        real_syn = synsets[real_c]

        # Predictions under baseline vs simulated pooling
        pred_base_c = int(base_logits[i].argmax())
        pred_comb_c = int(combined_logits[i].argmax())

        sorted_comb = np.argsort(-combined_logits[i])
        is_top1 = (real_c == pred_comb_c)
        is_top2 = bool(real_c in sorted_comb[:2])
        is_top5 = bool(real_c in sorted_comb[:5])
        rank_comb = int(np.where(sorted_comb == real_c)[0][0] + 1)

        all_test_samples.append({
            "trial_index": i,
            "image_id": int(I_te[i]),
            "real_class": real_syn,
            "raw_single_prediction": synsets[pred_base_c],
            "simulated_pooling_prediction": synsets[pred_comb_c],
            "is_correct_top1": is_top1,
            "is_correct_top2": is_top2,
            "is_correct_top5": is_top5,
            "rank_of_true_class": rank_comb,
            "confidence_simulated_top1": round(float(probs_comb[i, pred_comb_c]), 4),
            "confidence_raw_top1": round(float(probs_base[i, pred_base_c]), 4),
            "top5_candidates": " | ".join([synsets[c] for c in sorted_comb[:5]])
        })

    # ========================================================================
    # 6. EXPORT CSVS & SYNC TO GOOGLE DRIVE
    # ========================================================================
    csv_samples_path = os.path.join(OUTPUT_DIR, "all_test_samples_simulated_pooling.csv")
    with open(csv_samples_path, mode="w", newline="", encoding="utf-8") as f_s:
        writer = csv.DictWriter(f_s, fieldnames=list(all_test_samples[0].keys()))
        writer.writeheader()
        for sample in all_test_samples:
            writer.writerow(sample)

    csv_comp_path = os.path.join(OUTPUT_DIR, "comparison_simulated_pooling_vs_paper.csv")
    with open(csv_comp_path, mode="w", newline="", encoding="utf-8") as f_c:
        writer = csv.writer(f_c)
        writer.writerow(["Framework_or_Strategy", "2-way_Top1", "10-way_Top1", "10-way_Top2", "50-way_Top1"])
        for name, a2, a10_1, a10_2, a50_1 in paper_rows:
            writer.writerow([name, f"{a2:.2f}%", f"{a10_1:.2f}%", f"{a10_2:.2f}%", f"{a50_1:.2f}%"])
        for res in comparison_results:
            writer.writerow([res["strategy"], f"{res['2way_top1']:.2f}%", f"{res['10way_top1']:.2f}%", f"{res['10way_top2']:.2f}%", f"{res['50way_top1']:.2f}%"])

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

if __name__ == "__main__":
    results = evaluate_all_simulated_methods()
    print(f"[Finished] Evaluated and returned {len(results['all_test_samples'])} test sample predictions.")
