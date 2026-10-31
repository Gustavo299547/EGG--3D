# NeuroStream-SST Pipeline: Spectral-Spatio-Temporal Deep Learning (2026)

This directory contains the complete end-to-end implementation of the **NeuroStream-SST** pipeline (Scientific Reports 2026), connecting EEG neural classification directly to 2D image diffusion and 3D TRELLIS reconstruction.

---

## 🌟 Core Innovations

1. **Morlet Wavelet Spectral Filterbank (High-Gamma 70–120 Hz)**:
   - Captures true physiological waveband envelopes centered at 95 Hz, immune to low-frequency recording block drift.
2. **11×11 Scalp Topographical Projection**:
   - Geodesic 2D scalp coordinate projection mapping 128 electrodes onto an 11×11 spatial matrix, processed with spatial CNNs.
3. **Multi-Trial Image Pooling**:
   - Cancels stochastic brain noise by averaging trial repetitions ($1 \le k \le 6$) during training and inference.
4. **End-to-End Pipeline**:
   - Direct integration from EEG conditioning $\rightarrow$ SD 2.1-base generation $\rightarrow$ TRELLIS 3D representations.

---

## 📁 Files in this Directory

| File | Purpose | Outputs |
| :--- | :--- | :--- |
| **`neurostream_cell1_train.py`** | Baseline Stage 1 training script with NeuroStream-SST classifier + GWIT ControlNet + Google Drive sync. | `best_neurostream_sst.pt`, `exact-paper-modelV00.pt` |
| **`neurostream_WithmultitryialPooling_cell1_train.py`** | **Upgraded Stage 1 training** with Multi-Trial Pooling during training + Cosine LR Decay (75 epochs). Checkpoint reloading prevents late-epoch degradation. | `best_neurostream_multitrial_pooling.pt`, `neurostream_multitrial_pipeline.pt` |
| **`neurostream_cell2_inference.py`** | Full test evaluation and image generation across **all test samples (>1,000 images)** without segmentation. Generates `test_evaluation_results.csv` and uploads to Drive. | `GeneratedImage/sample_XXXX/`, `test_evaluation_results.csv` |
| **`neurostream_test_all_samples_eval.py`** | **Fast pure-classifier evaluation** across **ALL 1,987 test samples** in ~3 seconds. Computes Single-Trial, Pooled, Top-5, 40-Class, and Subject breakdowns + uploads 3-column CSV (`real_image_class,predicted_class,isTrue`) to Google Drive. | `test_evaluation_results_neurostream.csv` (Synced to Drive) |
| **`neurostream_cell3_setup.py`** | Environment and dependency setup for TRELLIS 3D generation (Open3D, Kaolin, Spconv, CUDA extensions). | Configured 3D environment |
| **`neurostream_cell4_3d_gen.py`** | Generates 3D Gaussian Splats (`.ply`), Radiance Fields, and textured 3D meshes (`.glb`) directly from generated images. | `.glb`, `.ply`, interactive 3D viewers |
| **`neurostream_sst_classifier.py`** | Standalone scientific classifier benchmark addressing all 5 methodology problems on EEGCVPR40. | `best_neurostream_sst.pt` |
| **`neurostream_view_all_100_samples.py`** | Visualizer script rendering paginated 3-panel comparative grids of 100 test samples. | `triplet_page_*.png` |
| **`neurostream_view_all_150_samples.py`** | Visualizer script rendering paginated grids of 150 test samples. | `triplet_page_*.png` |

---

## 🚀 Execution Workflow in Kaggle

1. **Cell 1**: Run `neurostream_WithmultitryialPooling_cell1_train.py` (or `neurostream_cell1_train.py`).
2. **Cell 2**: Run `neurostream_cell2_inference.py` to evaluate all test trials and produce the 3-column evaluation CSV.
3. **Cell 3**: Run `neurostream_cell3_setup.py` to install and patch 3D environment dependencies.
4. **Cell 4**: Run `neurostream_cell4_3d_gen.py` to reconstruct 3D meshes and Gaussian Splats.
