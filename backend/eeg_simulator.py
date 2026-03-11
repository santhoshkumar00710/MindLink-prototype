"""
EEG Signal Simulator
Generates realistic 8-channel EEG signals from thought commands using
alpha, beta, and gamma wave synthesis plus band-limited noise.
"""

import numpy as np
from typing import Tuple, Dict

# ─── Constants ──────────────────────────────────────────────────────────────
N_CHANNELS = 8
SAMPLING_RATE = 256          # Hz
DURATION = 1.0               # seconds per signal window
N_SAMPLES = int(SAMPLING_RATE * DURATION)

# Channel labels (10-20 system inspired)
CHANNEL_LABELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"]

# ─── Thought → EEG Pattern Mapping ──────────────────────────────────────────
THOUGHT_PATTERNS: Dict[str, Dict] = {
    "open_browser": {
        "alpha_amp": 0.3,
        "beta_amp": 1.2,
        "gamma_amp": 0.6,
        "theta_amp": 0.2,
        "noise_amp": 0.15,
        "dominant_channels": [0, 1, 2, 3],   # frontal emphasis
    },
    "scroll_down": {
        "alpha_amp": 0.5,
        "beta_amp": 0.8,
        "gamma_amp": 0.4,
        "theta_amp": 0.3,
        "noise_amp": 0.12,
        "dominant_channels": [4, 5],           # motor cortex (C3/C4)
    },
    "type_hello": {
        "alpha_amp": 0.2,
        "beta_amp": 1.5,
        "gamma_amp": 0.8,
        "theta_amp": 0.1,
        "noise_amp": 0.18,
        "dominant_channels": [2, 3, 4, 5],    # frontocentral
    },
    "play_music": {
        "alpha_amp": 0.8,
        "beta_amp": 0.4,
        "gamma_amp": 0.3,
        "theta_amp": 0.6,
        "noise_amp": 0.10,
        "dominant_channels": [1, 3, 7],        # right hemisphere
    },
    "stop_action": {
        "alpha_amp": 1.0,
        "beta_amp": 0.2,
        "gamma_amp": 0.1,
        "theta_amp": 0.4,
        "noise_amp": 0.08,
        "dominant_channels": [0, 1, 6, 7],    # occipitofrontal
    },
    "default": {
        "alpha_amp": 0.5,
        "beta_amp": 0.5,
        "gamma_amp": 0.3,
        "theta_amp": 0.3,
        "noise_amp": 0.20,
        "dominant_channels": list(range(8)),
    },
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _sine_wave(freq: float, amplitude: float, t: np.ndarray, phase: float = 0.0) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * freq * t + phase)


def _band_limited_noise(low: float, high: float, samples: int, amplitude: float) -> np.ndarray:
    """Generate noise filtered to [low, high] Hz band."""
    from scipy.signal import butter, filtfilt
    noise = np.random.randn(samples)
    nyq = SAMPLING_RATE / 2.0
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    filtered = filtfilt(b, a, noise)
    # normalise to unit variance then scale
    std = filtered.std() or 1.0
    return amplitude * filtered / std


def _build_channel_signal(pattern: Dict, ch_idx: int, t: np.ndarray) -> np.ndarray:
    """Generate a single-channel EEG timeseries for the given thought pattern."""
    # Amplitude boost if this is a dominant channel for the thought
    boost = 1.5 if ch_idx in pattern["dominant_channels"] else 0.6

    # Each channel gets slightly different phase for spatial realism
    phase_offset = ch_idx * np.pi / 4

    signal = (
        _sine_wave(10.0, pattern["alpha_amp"] * boost, t, phase_offset)          # alpha  8-13 Hz
        + _sine_wave(20.0, pattern["beta_amp"] * boost, t, phase_offset * 0.7)   # beta  13-30 Hz
        + _sine_wave(40.0, pattern["gamma_amp"] * boost, t, phase_offset * 0.3)  # gamma 30-50 Hz
        + _sine_wave(6.0,  pattern["theta_amp"] * boost, t, phase_offset * 1.2)  # theta  4-8  Hz
        + _band_limited_noise(1.0, 50.0, len(t), pattern["noise_amp"])
    )
    return signal


# ─── Public API ──────────────────────────────────────────────────────────────

def map_thought_to_key(thought: str) -> str:
    """Map raw user text to the closest internal pattern key."""
    thought_lower = thought.lower().strip()
    mapping = {
        "open browser":  "open_browser",
        "open_browser":  "open_browser",
        "scroll down":   "scroll_down",
        "scroll_down":   "scroll_down",
        "type hello":    "type_hello",
        "type_hello":    "type_hello",
        "play music":    "play_music",
        "play_music":    "play_music",
        "stop":          "stop_action",
        "stop action":   "stop_action",
        "stop_action":   "stop_action",
    }
    for key, val in mapping.items():
        if key in thought_lower:
            return val
    return "default"


def simulate_eeg(thought: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Generate a multi-channel EEG array for the given thought string.

    Returns
    -------
    eeg : ndarray, shape (N_CHANNELS, N_SAMPLES)
    time : ndarray, shape (N_SAMPLES,)
    pattern_key : str
    """
    np.random.seed(abs(hash(thought)) % (2**31))   # reproducible per thought
    pattern_key = map_thought_to_key(thought)
    pattern = THOUGHT_PATTERNS.get(pattern_key, THOUGHT_PATTERNS["default"])

    t = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)
    eeg = np.stack(
        [_build_channel_signal(pattern, ch, t) for ch in range(N_CHANNELS)],
        axis=0,
    )
    return eeg, t, pattern_key


def compute_power_spectrum(eeg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute average power spectral density across channels.

    Returns
    -------
    freqs : ndarray
    psd   : ndarray
    """
    from scipy.signal import welch
    freqs, psds = [], []
    for ch in range(eeg.shape[0]):
        f, p = welch(eeg[ch], fs=SAMPLING_RATE, nperseg=128)
        freqs.append(f)
        psds.append(p)
    freqs = np.array(freqs[0])
    psd   = np.mean(psds, axis=0)
    return freqs, psd
