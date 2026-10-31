# ============================================================================
# METHOD 1: TEST-TIME AUGMENTATION (TTA) IN SIGNAL & FEATURE SPACE
# ----------------------------------------------------------------------------
# Simulates multi-trial pooling from a single trial at inference time by:
#   1. Slight temporal shifting (+/- 4 to 8 time steps ~ 8-16ms)
#   2. Low-amplitude Gaussian jitter (epsilon ~ N(0, 0.03^2))
#   3. Random channel dropout / electrode masking (5-10% channels)
#   4. Multi-pass forward inference & latent feature averaging:
#        z_pooled = (1/K) * sum_{i=1}^K f_theta(x_perturbed_i)
# ============================================================================

import torch
import torch.nn.functional as F
import numpy as np

class TestTimeAugmenter:
    def __init__(self, n_passes=8, jitter_std=0.03, max_time_shift=8, channel_mask_prob=0.08):
        self.n_passes = n_passes
        self.jitter_std = jitter_std
        self.max_time_shift = max_time_shift
        self.channel_mask_prob = channel_mask_prob

    def augment_single_trial(self, x_eeg):
        """
        Input: x_eeg tensor of shape [128, 440] or [B, 128, 440]
        Returns: list of K augmented variations
        """
        if x_eeg.dim() == 2:
            x_eeg = x_eeg.unsqueeze(0) # [1, 128, 440]

        B, C, T = x_eeg.shape
        variants = [x_eeg] # Always include the pristine original

        for _ in range(self.n_passes - 1):
            x_aug = x_eeg.clone()

            # 1. Temporal shifting (circular or zero-padded roll)
            shift = np.random.randint(-self.max_time_shift, self.max_time_shift + 1)
            if shift != 0:
                x_aug = torch.roll(x_aug, shifts=shift, dims=-1)
                if shift > 0:
                    x_aug[:, :, :shift] = x_aug[:, :, shift:shift+1]
                else:
                    x_aug[:, :, shift:] = x_aug[:, :, shift-1:shift]

            # 2. Channel Masking (simulate noisy / loose electrodes)
            mask = (torch.rand(B, C, 1, device=x_eeg.device) > self.channel_mask_prob).float()
            x_aug = x_aug * mask

            # 3. Additive Gaussian Jitter (thermal / amplifier noise smoothing)
            noise = torch.randn_like(x_aug) * self.jitter_std
            x_aug = x_aug + noise

            variants.append(x_aug)

        return variants

    def predict_with_tta(self, model, x_eeg, subj_id=None):
        """
        Averages model latent representations across K synthetic variations.
        Returns: pooled_logits [B, n_classes], pooled_feats [B, d_model]
        """
        variants = self.augment_single_trial(x_eeg)
        feats_list = []

        model.eval()
        with torch.no_grad():
            for v in variants:
                if hasattr(model, "extract_features"):
                    f = model.extract_features(v, subj_id)
                else:
                    _, f = model(v)
                feats_list.append(f)

        # Average latent representations (Smooth manifold pooling)
        stacked_feats = torch.stack(feats_list, dim=0) # [K, B, d_model]
        pooled_feats = stacked_feats.mean(dim=0)       # [B, d_model]

        # Classify pooled features
        if hasattr(model, "head"):
            pooled_logits = model.head(pooled_feats)
        else:
            pooled_logits = model(pooled_feats)

        return pooled_logits, pooled_feats
