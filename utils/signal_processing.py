"""
Signal Processing utilities for the MindLink AI EEG pipeline.
"""

import numpy as np
from scipy.signal import butter, filtfilt, welch
from typing import Tuple

SAMPLING_RATE = 256   # Hz

# ─── Band definitions ─────────────────────────────────────────────────────────
BANDS = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 50.0),
}


def bandpass_filter(signal: np.ndarray, low: float, high: float, fs: int = SAMPLING_RATE, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def notch_filter(signal: np.ndarray, notch_freq: float = 50.0, fs: int = SAMPLING_RATE, quality: float = 30.0) -> np.ndarray:
    from scipy.signal import iirnotch
    b, a = iirnotch(notch_freq, quality, fs)
    return filtfilt(b, a, signal)


def compute_band_power(signal: np.ndarray, band: str, fs: int = SAMPLING_RATE) -> float:
    """Compute relative band power for a single-channel signal."""
    low, high = BANDS[band]
    filtered = bandpass_filter(signal, low, high, fs)
    return float(np.mean(filtered ** 2))


def compute_all_band_powers(eeg: np.ndarray, fs: int = SAMPLING_RATE) -> dict:
    """Returns mean band power across all channels for each band."""
    powers = {}
    for band in BANDS:
        ch_powers = [compute_band_power(eeg[ch], band, fs) for ch in range(eeg.shape[0])]
        powers[band] = float(np.mean(ch_powers))
    return powers


def compute_spectrogram(signal: np.ndarray, fs: int = SAMPLING_RATE) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Short-time Fourier transform spectrogram.
    Returns (times, freqs, power_matrix).
    """
    from scipy.signal import spectrogram as scipy_spectrogram
    f, t, Sxx = scipy_spectrogram(signal, fs=fs, nperseg=64, noverlap=48)
    return t, f, 10 * np.log10(Sxx + 1e-10)


def normalize_eeg(eeg: np.ndarray) -> np.ndarray:
    """Z-score normalise each channel independently."""
    mean = eeg.mean(axis=1, keepdims=True)
    std  = eeg.std(axis=1, keepdims=True) + 1e-8
    return (eeg - mean) / std


def eeg_to_tensor(eeg: np.ndarray) -> "torch.Tensor":
    import torch
    normed = normalize_eeg(eeg)
    return torch.FloatTensor(normed).unsqueeze(0)   # (1, channels, samples)
