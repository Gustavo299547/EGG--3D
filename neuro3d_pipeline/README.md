# Neuro-3D Pipeline: Towards 3D Visual Decoding from EEG Signals (CVPR 2025)

This directory contains the complete implementation of the **Neuro-3D** neural decoding framework based on the CVPR 2025 paradigm.

---

## 🌟 Architectural Overview

Neuro-3D studies neural responses to visual stimuli by disentangling brain dynamics into two complementary visual pathways:

1. **Static Visual Pathway ($z_{\text{static}}$)**:
   - Captures invariant 3D object geometry, category identity, structural form, and sustained evoked potentials ($P_{100}$, $N_{170}$).
   - Processes spatial projections through low-frequency temporal convolutions and self-attention transformer layers.
2. **Dynamic Visual Pathway ($z_{\text{dynamic}}$)**:
   - Captures temporal phase velocity ($\frac{dX}{dt}$), motion flux, high-frequency gamma ripples (30–120 Hz), and viewpoint transitions.
   - Utilizes a differentiable Morlet wavelet filterbank and dilated temporal convolutions.
3. **Disentanglement Orthogonality Constraint**:
   - Enforces mathematical orthogonality ($\mathcal{L}_{\text{ortho}} = |\cos(z_{\text{static}}, z_{\text{dynamic}})|$) so the two pathways capture complementary, non-redundant information.
4. **Continuous 3D Point Cloud Folding Decoder**:
   - Generates colored 3D point clouds $\mathcal{P} \in \mathbb{R}^{1024 \times 6}$ ($x, y, z, r, g, b$) bounded within a $[-1, 1]^3$ canonical volume, exportable to `.ply`.

---

## 📁 Files in this Directory

| File | Purpose | Outputs |
| :--- | :--- | :--- |
| **`neuro3d_classifier_and_decoder.py`** | Standalone scientific reference model. Implements the full Neuro-3D architecture, 5 BCI methodology fixes, training engine, and ASCII `.ply` 3D point cloud export. | `best_neuro3d.pt`, `.ply` point clouds |
| **`neuro3d_cell1_train.py`** | **Stage 1 (Cell 1) Training**: Trains the Neuro-3D classifier and the GWIT ControlNet on Stable Diffusion 2.1-base. Automatically syncs and backs up models to Google Drive. | `best_neuro3d_classifier.pt`, `neuro3d_pipeline.pt` |
| **`neuro3d_cell2_inference.py`** | **Stage 2 (Cell 2) Inference**: Evaluates unseen test trials starting from **sample 700 to the end**. Generates paired real and predicted images (no YOLO segmentation) and logs results **row-by-row into a CSV file with immediate flush**. | `/GeneratedImage/sample_XXXX/`, `test_evaluation_results.csv` |
| **`neuro3d_test_all_samples_eval.py`** | **Full Evaluation Across ALL Test Samples (>1,000 trials)**: Evaluates trained `best_neuro3d.pt` across all 1,987 test trials, outputs Single-Trial & Pooled Accuracies, Top-5, 40-Class and Subject tables, and syncs a 3-column CSV report to Google Drive. | `test_evaluation_results_neuro3d.csv` |

---

## 🚀 How to Run in Kaggle

### Step 1: Run Cell 1 (`neuro3d_cell1_train.py`)
1. Paste `neuro3d_cell1_train.py` into a Kaggle GPU notebook cell.
2. The script checks your Google Drive (`13vd_i1ePxEbYH5RtW9Q4B2OG0ccwv6Wz`) for `best_neuro3d_classifier.pt`. If found, it skips to ControlNet training; otherwise, it trains for 65 epochs.
3. Backs up the trained weights to Google Drive automatically.

### Step 2: Run Cell 2 (`neuro3d_cell2_inference.py`)
1. Paste `neuro3d_cell2_inference.py` into the next notebook cell.
2. The script loads `best_neuro3d_classifier.pt` directly from `/kaggle/working/` or downloads it from Drive.
3. It evaluates samples from **index 700 through 1986**.
4. Every single sample is immediately written to `test_evaluation_results.csv` on disk and synced to Google Drive in real-time.
