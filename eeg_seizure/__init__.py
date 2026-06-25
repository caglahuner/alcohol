"""EEG seizure classification on the Bonn dataset (or a synthetic fallback)."""
from .datasets import load_bonn, load_set, load_synthetic, FS
from .preprocess import bandpass, decompose_bands, BANDS
from .features import extract_features, feature_matrix, FEATURE_NAMES
from .pipeline import make_classifiers, evaluate

__all__ = ["load_bonn", "load_set", "load_synthetic", "FS", "bandpass",
           "decompose_bands", "BANDS", "extract_features", "feature_matrix",
           "FEATURE_NAMES", "make_classifiers", "evaluate"]
