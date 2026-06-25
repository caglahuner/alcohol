"""Load the Bonn EEG epilepsy dataset, with a synthetic fallback.

Bonn dataset (Andrzejak et al., 2001): five sets Z, O, N, F, S, each with 100
single-channel surface-EEG segments of 4097 samples at fs = 173.61 Hz (~23.6 s).

    Z, O : healthy volunteers (eyes open / eyes closed)
    N, F : interictal (non-seizure) intracranial recordings
    S    : ictal (seizure) recordings

Place the unzipped folders (each containing *.txt files) under ``data/`` -
see ``download_data.py``. If the data is absent, ``load_synthetic`` produces a
separable stand-in so the pipeline still runs and the tests pass offline.
"""
from __future__ import annotations
import os
import glob
import numpy as np

FS = 173.61
SET_FOLDERS = {"Z": "Z", "O": "O", "N": "N", "F": "F", "S": "S"}


def load_set(data_dir, set_name):
    """Load all segments of one Bonn set as an array of shape (n_segments, n)."""
    folder = os.path.join(data_dir, SET_FOLDERS[set_name])
    files = sorted(glob.glob(os.path.join(folder, "*.txt")) +
                   glob.glob(os.path.join(folder, "*.TXT")))
    if not files:
        raise FileNotFoundError(
            f"No .txt files in {folder}. Run download_data.py or see the README."
        )
    return np.array([np.loadtxt(f) for f in files])


def load_bonn(data_dir, sets=("Z", "S")):
    """Load several sets and return (X_segments, y_labels, label_names).

    Labels are the index of each set in ``sets`` (e.g. Z->0, S->1).
    """
    segs, labels = [], []
    for i, s in enumerate(sets):
        data = load_set(data_dir, s)
        segs.append(data)
        labels.append(np.full(data.shape[0], i))
    X = np.vstack(segs)
    y = np.concatenate(labels)
    return X, y, list(sets)


def load_synthetic(n_per_class=100, n=4097, fs=FS, seed=0):
    """Synthetic two-class EEG stand-in (healthy-like vs seizure-like).

    Class 0: low-amplitude mixed alpha/beta with noise.
    Class 1: higher-amplitude, more rhythmic, spikier (seizure-like).
    Returns (X, y, ["healthy", "seizure"]).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    X, y = [], []
    for cls in (0, 1):
        for _ in range(n_per_class):
            base = rng.uniform(6, 9)            # overlapping base rhythm
            sig = (0.8 * np.sin(2 * np.pi * base * t + rng.uniform(0, 6))
                   + 2.0 * rng.standard_normal(n))     # strong shared noise
            if cls == 1:                                 # seizure: subtle extras
                sig += 0.5 * np.sin(2 * np.pi * rng.uniform(3, 5) * t)
                spikes = rng.random(n) > 0.999
                sig[spikes] += rng.uniform(2, 4, spikes.sum())
            X.append(sig); y.append(cls)
    return np.array(X), np.array(y), ["healthy", "seizure"]
