"""Offline tests on synthetic EEG (no dataset required)."""
import numpy as np
from eeg_seizure import (load_synthetic, feature_matrix, extract_features,
                         evaluate, FEATURE_NAMES, decompose_bands, FS)


def test_feature_length():
    x = np.random.randn(4097)
    assert extract_features(x, FS).shape[0] == len(FEATURE_NAMES)


def test_band_decomposition():
    x = np.random.randn(4097)
    bands = decompose_bands(x, FS)
    assert set(bands) == {"delta", "theta", "alpha", "beta", "gamma"}
    assert all(v.shape == x.shape for v in bands.values())


def test_classification_separates_synthetic():
    X_raw, y, _ = load_synthetic(n_per_class=40)
    X = feature_matrix(X_raw, FS)
    res = evaluate(X, y, n_splits=5)
    # synthetic classes are clearly separable; expect strong accuracy
    assert max(r["accuracy"] for r in res.values()) > 0.9
