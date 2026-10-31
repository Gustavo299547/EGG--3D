# ============================================================================
# METHOD 2: SINGLE-TO-AVERAGE (S2A) NEURAL DENOISER
# ----------------------------------------------------------------------------
# Trains a 1D Temporal Residual U-Net purely to reconstruct trial-averaged
# clean EEG waveforms from single-trial noisy inputs:
#     Loss = || Denoiser(x_single) - x_averaged ||_2^2 + 0.1 * (1 - PearsonCorr)
#
# At inference time, feed the single noisy trial into the denoiser first.
# The network projects the single trial toward the averaged manifold before
# classification or 3D visual generation.
# ============================================================================

import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        p = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=p, bias=False)
        self.bn1   = nn.BatchNorm1d(channels)
        self.act1  = nn.GELU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=p, bias=False)
        self.bn2   = nn.BatchNorm1d(channels)

    def forward(self, x):
        return x + self.bn2(self.conv2(self.act1(self.bn1(self.conv1(x)))))

class SingleToAverageUNet(nn.Module):
    """Temporal 1D U-Net mapping single-trial [B, 128, 440] to trial-averaged clean EEG."""
    def __init__(self, in_channels=128, hidden_dim=64):
        super().__init__()
        # Encoder
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden_dim), nn.GELU()
        )
        self.res1 = ResBlock1D(hidden_dim, kernel_size=7)
        self.down1 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim * 2), nn.GELU()
        )
        self.res2 = ResBlock1D(hidden_dim * 2, kernel_size=7)
        self.down2 = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim * 4), nn.GELU()
        )
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock1D(hidden_dim * 4, kernel_size=7),
            ResBlock1D(hidden_dim * 4, kernel_size=7)
        )
        # Decoder
        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim * 2), nn.GELU()
        )
        self.dec_res2 = ResBlock1D(hidden_dim * 2, kernel_size=7)
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim), nn.GELU()
        )
        self.dec_res1 = ResBlock1D(hidden_dim, kernel_size=7)
        # Output clean residual
        self.out_conv = nn.Conv1d(hidden_dim, in_channels, kernel_size=7, padding=3)

    def forward(self, x):
        # Global residual connection: output clean EEG = x + Delta
        h0 = self.in_conv(x)
        h1 = self.res1(h0)
        d1 = self.down1(h1)
        h2 = self.res2(d1)
        d2 = self.down2(h2)
        b  = self.bottleneck(d2)
        u2 = self.up2(b) + h2
        dec2 = self.dec_res2(u2)
        u1 = self.up1(dec2) + h1
        dec1 = self.dec_res1(u1)
        delta = self.out_conv(dec1)
        return x + delta

def compute_s2a_loss(pred_avg, true_avg):
    """MSE + Negative Pearson Correlation Loss."""
    mse = F.mse_loss(pred_avg, true_avg)
    
    # Pearson Correlation along temporal dimension
    p_c = pred_avg - pred_avg.mean(dim=-1, keepdim=True)
    t_c = true_avg - true_avg.mean(dim=-1, keepdim=True)
    p_norm = p_c / (p_c.std(dim=-1, keepdim=True) + 1e-6)
    t_norm = t_c / (t_c.std(dim=-1, keepdim=True) + 1e-6)
    corr = (p_norm * t_norm).mean()
    
    return mse + 0.15 * (1.0 - corr)

def train_s2a_denoiser(X_tr, IMGS_tr, epochs=25, batch_size=32, lr=1e-3, device="cuda"):
    """
    Pairs each single trial with its image-averaged ground truth counterpart for training.
    """
    device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"[+] Training Single-to-Average (S2A) Denoiser on {device.upper()}...")

    # 1. Precompute True Image Averages
    img2trials = {}
    for idx, im in enumerate(IMGS_tr):
        img2trials.setdefault(im, []).append(idx)

    img2avg = {}
    for im, tr_idxs in img2trials.items():
        img2avg[im] = X_tr[tr_idxs].mean(axis=0) # [128, 440]

    # Target pairs: (X_tr[i], img2avg[IMGS_tr[i]])
    X_single = torch.as_tensor(X_tr, dtype=torch.float32)
    X_target = torch.as_tensor(np.array([img2avg[im] for im in IMGS_tr]), dtype=torch.float32)

    model = SingleToAverageUNet(in_channels=128, hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    N = len(X_single)
    for ep in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(N)
        total_loss = 0.0

        for k in range(0, N, batch_size):
            idx = perm[k : k + batch_size]
            xb = X_single[idx].to(device)
            yb = X_target[idx].to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = compute_s2a_loss(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(idx)

        scheduler.step()
        if ep % 5 == 0 or ep == epochs:
            print(f"  S2A Denoiser Ep {ep:02d}/{epochs} | Loss: {total_loss / N:.4f}")

    return model
