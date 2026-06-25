"""Feature extraction from single-channel EEG segments.

Combines time-domain statistics, Hjorth parameters, non-linear measures and
frequency-band powers into one feature vector per segment - the standard kind
of representation used for the Bonn epilepsy classification benchmark.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import skew, kurtosis
from .preprocess import BANDS


def _hjorth(x):
    dx = np.diff(x)
    ddx = np.diff(dx)
    var0 = np.var(x) + 1e-12
    var1 = np.var(dx) + 1e-12
    var2 = np.var(ddx) + 1e-12
    activity = var0
    mobility = np.sqrt(var1 / var0)
    complexity = np.sqrt(var2 / var1) / (mobility + 1e-12)
    return activity, mobility, complexity


def _line_length(x):
    return np.sum(np.abs(np.diff(x)))


def _zero_crossings(x):
    return np.sum(np.diff(np.signbit(x - np.mean(x))) != 0)


def _band_powers(x, fs):
    freqs = np.fft.rfftfreq(x.size, 1 / fs)
    psd = np.abs(np.fft.rfft(x)) ** 2
    total = np.sum(psd) + 1e-12
    out = []
    for lo, hi in BANDS.values():
        mask = (freqs >= lo) & (freqs < hi)
        out.append(np.sum(psd[mask]) / total)  # relative band power
    return out


def _spectral_entropy(x, fs):
    psd = np.abs(np.fft.rfft(x)) ** 2
    p = psd / (np.sum(psd) + 1e-12)
    p = p[p > 0]
    return -np.sum(p * np.log2(p)) / np.log2(p.size + 1e-12)


FEATURE_NAMES = [
    "mean", "std", "var", "skew", "kurtosis", "rms", "ptp",
    "line_length", "zero_crossings",
    "hjorth_activity", "hjorth_mobility", "hjorth_complexity",
    "spectral_entropy",
    "bp_delta", "bp_theta", "bp_alpha", "bp_beta", "bp_gamma",
]


def extract_features(x, fs=173.61):
    """Return a 1-D feature vector for one EEG segment."""
    x = np.asarray(x, float).ravel()
    act, mob, comp = _hjorth(x)
    feats = [
        np.mean(x), np.std(x), np.var(x), skew(x), kurtosis(x),
        np.sqrt(np.mean(x ** 2)), np.ptp(x),
        _line_length(x), _zero_crossings(x),
        act, mob, comp,
        _spectral_entropy(x, fs),
    ]
    feats += _band_powers(x, fs)
    return np.array(feats, float)


def feature_matrix(segments, fs=173.61):
    """Stack feature vectors for a list/array of segments -> (n_segments, n_feats)."""
    return np.vstack([extract_features(s, fs) for s in segments])
