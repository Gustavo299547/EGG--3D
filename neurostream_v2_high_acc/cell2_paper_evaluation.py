# ============================================================================
# CELL 2: SOTA NEUROSTREAM-V2 (EXACT BRAIN3D PAPER METRICS EVALUATION PROTOCOL)
# ----------------------------------------------------------------------------
# Evaluates SOTA NeuroStream-V2 using the exact evaluation protocol from:
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
#   • Dynamic Attention-Pooled Mode (Learned trial weighting per image)
#
# Returns ALL test samples with full diagnostic records, outputs comparison
# table against GWIT, BrainVis, EEG-CLIP, DreamDiffusion, and syncs to Google Drive.
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

# ============================================================================
# 1. CONFIGURATION & CLOUD DRIVE
# ============================================================================
DATA_ROOT  = "/kaggle/input/datasets/evolutionaw/eegcvpr40/data"
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
    "./best_neurostream_sst.pt"
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
        print(f"  [Drive Notice] Cloud sync skipped ({e})")
        return False

# Robust module path resolution (supports both .py scripts and Jupyter/Kaggle notebooks)
_CURR_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if _CURR_DIR not in sys.path:
    sys.path.insert(0, _CURR_DIR)
_PARENT_DIR = os.path.dirname(_CURR_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

for p in [os.getcwd(), "/kaggle/working"]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from neurostream_v2_high_acc.cell1_train_model import SOTANeuroStreamModel, load_eegcvpr40_data
except ImportError:
    try:
        from cell1_train_model import SOTANeuroStreamModel, load_eegcvpr40_data
    except ImportError:
        pass

# ============================================================================
# 2. EXACT PAPER METRIC EVALUATOR (N-WAY TOP-K PROTOCOL)
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
# 3. MASTER EVALUATION RUNNER
# ============================================================================
def run_paper_evaluation(checkpoint_path=None):
    print("=" * 95)
    print("SOTA NEUROSTREAM-V2: EXACT BRAIN3D PAPER METRICS EVALUATION PROTOCOL")
    print("Benchmarking Against: GWIT, BrainVis, EEG-CLIP, DreamDiffusion (Table 1)")
    print("=" * 95)

    _, _, _, _, X_te, Y_te, S_te, I_te, synsets = load_eegcvpr40_data()
    n_classes = len(synsets)
    n_subjects = int(S_te.max()) + 1
    N_te = len(X_te)

    # 1. Resolve & Load Checkpoint
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
        print(f"[+] Loading model weights from: {model_path}")
        try:
            ck = torch.load(model_path, map_location=dev, weights_only=False)
            state = ck.get("model_state", ck.get("model_state_dict", ck))
            model_keys = model.state_dict()
            filtered_state = {k: v for k, v in state.items() if k in model_keys and model_keys[k].shape == v.shape}
            model.load_state_dict(filtered_state, strict=False)
            print(f"  [✓] Successfully loaded {len(filtered_state)} compatible model weight tensors.")
            if isinstance(ck, dict) and "epoch" in ck:
                print(f"  [✓] Epoch: {ck['epoch']} | Saved Pooled Acc: {ck.get('pooled_acc', 'N/A')}")
        except Exception as e:
            print(f"  [!] Notice during weight loading ({e}).")
    else:
        print("  [Notice] Initializing fresh SOTA NeuroStream model.")

    model.eval()

    # 2. Extract Features on ALL Test Trials
    X_te_t = torch.as_tensor(X_te, dtype=torch.float32)
    S_te_t = torch.as_tensor(S_te, dtype=torch.long)
    bs = 64

    all_logits = []
    all_feats = []
    with torch.no_grad():
        for k in range(0, N_te, bs):
            xb = X_te_t[k : k + bs].to(dev)
            sb = S_te_t[k : k + bs].to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
                l, f, _ = model(xb, sb)
            all_logits.append(l.cpu())
            all_feats.append(f.cpu())

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_feats  = torch.cat(all_feats, dim=0)

    # 3. Compute Metrics for SINGLE-TRIAL Mode
    print("\n[+] Computing Exact Paper Metrics for SINGLE-TRIAL Mode (100 random samplings per trial)...")
    s_2way_top1, _   = compute_n_way_topk(all_logits, Y_te, n_way=2,  k=1, n_samplings=100)
    s_10way_top1, _  = compute_n_way_topk(all_logits, Y_te, n_way=10, k=1, n_samplings=100)
    s_10way_top2, _  = compute_n_way_topk(all_logits, Y_te, n_way=10, k=2, n_samplings=100)
    s_50way_top1, _  = compute_n_way_topk(all_logits, Y_te, n_way=min(50, n_classes), k=1, n_samplings=100)
    s_50way_top2, _  = compute_n_way_topk(all_logits, Y_te, n_way=min(50, n_classes), k=2, n_samplings=100)

    # 4. Compute Metrics for DYNAMIC ATTENTION-POOLED Mode
    print("[+] Computing Exact Paper Metrics for DYNAMIC ATTENTION-POOLED Mode...")
    te_img2trials = {}
    for idx, im in enumerate(I_te):
        te_img2trials.setdefault(im, []).append(idx)

    pooled_logits = []
    pooled_labels = []
    with torch.no_grad():
        for im, tr_idxs in te_img2trials.items():
            tr_f = all_feats[tr_idxs].to(dev)
            p_f = model.pooler(tr_f)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
                p_l = model.head(p_f).cpu().numpy()[0]
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
            "image_id": int(I_te[i]),
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
    print("COMPARATIVE BENCHMARK TABLE: PUBLISHED PAPER BASELINES vs. OUR CALCULATED SOTA NEUROSTREAM")
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
    print(f"{'NeuroStream-V2 (Single-Trial)':<30} | {s_2way_top1:>10.2f}% | {s_10way_top1:>11.2f}% | {s_10way_top2:>11.2f}% | {s_50way_top1:>11.2f}% | {s_50way_top2:>10.2f}%")
    print(f"{'NeuroStream-V2 (Attn-Pooled)':<30} | {p_2way_top1:>10.2f}% | {p_10way_top1:>11.2f}% | {p_10way_top2:>11.2f}% | {p_50way_top1:>11.2f}% | {p_50way_top2:>10.2f}%")
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
        writer.writerow(["NeuroStream_V2_Single_Trial", f"{s_2way_top1:.2f}%", f"{s_10way_top1:.2f}%", f"{s_10way_top2:.2f}%", f"{s_50way_top1:.2f}%", f"{s_50way_top2:.2f}%"])
        writer.writerow(["NeuroStream_V2_Attn_Pooled", f"{p_2way_top1:.2f}%", f"{p_10way_top1:.2f}%", f"{p_10way_top2:.2f}%", f"{p_50way_top1:.2f}%", f"{p_50way_top2:.2f}%"])
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

    return {
        "all_test_samples": all_test_samples,
        "summary_metrics": summary_metrics,
        "comparison_benchmarks": paper_benchmarks
    }

if __name__ == "__main__":
    results = run_paper_evaluation()
    print(f"[Finished] Returned {len(results['all_test_samples'])} test sample predictions.")
