# ============================================================================
# METHOD 4: SPATIOTEMPORAL FILTERING (1-30Hz BANDPASS & COMMON AVERAGE REFERENCE)
# ----------------------------------------------------------------------------
# Removes volume conduction noise, eye-blink artifacts, and line noise before
# the signal ever reaches the neural network:
#   1. Zero-Phase Butterworth Bandpass Filter (1-30 Hz):
#      Restricts input to visual evoked frequency bands (theta, alpha, beta),
#      eliminating 50/60Hz AC line noise and >40Hz facial muscle tremors.
#   2. Common Average Reference (CAR) / Spatial Laplacian:
#      Subtracts the instantaneous scalp mean across all 128 channels:
#          x_car(c, t) = x(c, t) - (1/C) * sum_{j=1}^C x(j, t)
#      Removes global non-cortical artifacts (respiration, cardiac pulse).
# ============================================================================

import numpy as np
import torch
import scipy.signal as signal

class SpatioTemporalFilter:
    def __init__(self, lowcut=1.0, highcut=30.0, fs=250.0, order=4):
        self.fs = fs
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        self.b, self.a = signal.butter(order, [low, high], btype="band")

    def apply_bandpass_numpy(self, X_eeg):
        """
        X_eeg: numpy array [N, 128, 440] or [128, 440]
        Zero-phase forward-backward Butterworth filtering along the time axis.
        """
        orig_shape = X_eeg.shape
        if len(orig_shape) == 2:
            X_eeg = X_eeg[np.newaxis, ...]

        # Filter along axis=-1 (temporal samples)
        filtered = signal.filtfilt(self.b, self.a, X_eeg, axis=-1)
        return filtered.reshape(orig_shape).astype(np.float32)

    def apply_spatial_car(self, X_eeg):
        """
        Common Average Reference (CAR) spatial filter:
        x_car(c, t) = x(c, t) - mean_{all_channels}(x(t))
        """
        if isinstance(X_eeg, torch.Tensor):
            car_mean = X_eeg.mean(dim=-2, keepdim=True)
            return X_eeg - car_mean
        else:
            car_mean = X_eeg.mean(axis=-2, keepdims=True)
            return X_eeg - car_mean

    def filter_eeg(self, X_eeg):
        """
        Complete spatiotemporal filter pipeline:
        1. Bandpass filter 1-30 Hz
        2. Common Average Reference spatial filter
        """
        is_torch = isinstance(X_eeg, torch.Tensor)
        device = X_eeg.device if is_torch else None
        
        arr = X_eeg.cpu().numpy() if is_torch else X_eeg
        # Step 1: Bandpass
        bp = self.apply_bandpass_numpy(arr)
        # Step 2: Spatial CAR
        car = self.apply_spatial_car(bp)

        if is_torch:
            return torch.as_tensor(car, dtype=torch.float32, device=device)
        return car
