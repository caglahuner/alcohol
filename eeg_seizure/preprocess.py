"""EEG preprocessing: band-pass filtering and EEG band decomposition."""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt

# Standard clinical EEG sub-bands (Hz)
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "gamma": (30.0, 45.0),
}


def bandpass(x, fs, low, high, order=4):
    """Zero-phase Butterworth band-pass filter."""
    ny = fs / 2.0
    high = min(high, ny * 0.99)
    b, a = butter(order, [low / ny, high / ny], btype="bandpass")
    return filtfilt(b, a, x)


def decompose_bands(x, fs, bands=BANDS, order=4):
    """Return a dict {band_name: filtered_signal} for each EEG sub-band."""
    return {name: bandpass(x, fs, lo, hi, order) for name, (lo, hi) in bands.items()}
