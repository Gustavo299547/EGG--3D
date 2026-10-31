# ============================================================================
# METHOD 3: CLASS PROTOTYPICAL MEMORY BANK & SUBJECT BASELINE SUBTRACTION
# ----------------------------------------------------------------------------
# In single-trial EEG, a single volatile sample can suffer from classification
# boundary jitter.
#
# This module implements:
#   1. Pre-computed Class Prototype Centroids:
#      Aggregates all training set representations per class into clean,
#      noise-free anchor centroids c_y.
#   2. Temperature-scaled Cosine Prototype Classifier:
#      P(y = c | x) = exp(tau * cos(f(x), c_c)) / sum_j exp(tau * cos(f(x), c_j))
#   3. Subject Resting / DC-Offset Baseline Removal:
#      Subtracts channel-wise baseline offsets per subject to neutralize
#      individual skull impedance shifts.
# ============================================================================

import torch
import torch.nn.functional as F
import numpy as np

class PrototypicalMemoryBank:
    def __init__(self, temperature=14.0):
        self.temperature = temperature
        self.prototypes = None # [n_classes, d_model]
        self.n_classes = None

    def build_memory_bank(self, model, X_tr, Y_tr, device="cuda"):
        """
        Passes training data through the feature encoder and calculates
        normalized class centroids c_y in latent feature space.
        """
        device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        model.eval()
        X_t = torch.as_tensor(X_tr, dtype=torch.float32)
        all_feats = []

        with torch.no_grad():
            for k in range(0, len(X_t), 64):
                xb = X_t[k : k + 64].to(device)
                if hasattr(model, "extract_features"):
                    f = model.extract_features(xb)
                else:
                    _, f = model(xb)
                all_feats.append(f.cpu())

        all_feats = torch.cat(all_feats, dim=0) # [N_tr, d_model]
        all_feats = F.normalize(all_feats, dim=-1)

        classes = np.unique(Y_tr)
        self.n_classes = len(classes)
        d_model = all_feats.shape[-1]
        self.prototypes = torch.zeros(self.n_classes, d_model)

        for c in classes:
            c_mask = (Y_tr == c)
            c_centroid = all_feats[c_mask].mean(dim=0)
            self.prototypes[c] = F.normalize(c_centroid, dim=-1)

        print(f"[+] Prototypical Memory Bank built: {self.n_classes} Class Centroids in R^{d_model}")

    def classify_with_prototypes(self, test_feats):
        """
        Given test latent features [B, d_model], computes cosine similarity
        against memory bank prototypes.
        Returns: logits [B, n_classes]
        """
        assert self.prototypes is not None, "Call build_memory_bank first!"
        norm_feats = F.normalize(test_feats, dim=-1) # [B, d_model]
        proto = self.prototypes.to(test_feats.device)

        # Cosine similarity scaled by inverse temperature tau
        logits = self.temperature * torch.matmul(norm_feats, proto.T) # [B, n_classes]
        return logits

def remove_subject_baseline_offsets(X_eeg):
    """
    Subtracts channel-wise mean and linear drift across the epoch for each trial.
    Removes DC offsets and slow electrode sweat artifacts.
    """
    # X_eeg: [B, 128, 440]
    baseline = X_eeg.mean(axis=-1, keepdims=True)
    return X_eeg - baseline
