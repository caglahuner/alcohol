"""Run the EEG seizure-classification benchmark.

Default task is binary healthy-vs-seizure. With no dataset present it uses a
synthetic fallback so it always runs:

    python run_classification.py                 # synthetic if data/ is empty
    python run_classification.py --sets Z S      # Bonn healthy (Z) vs ictal (S)
    python run_classification.py --sets Z O N F S # 5-class

Outputs a CV-accuracy table and saves a confusion matrix to examples/.
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from eeg_seizure import (load_bonn, load_synthetic, feature_matrix, evaluate, FS)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=None,
                    help="Bonn sets to classify, e.g. Z S. Omit to use synthetic data.")
    ap.add_argument("--folds", type=int, default=10)
    args = ap.parse_args()

    if args.sets:
        X_raw, y, names = load_bonn(DATA, sets=tuple(args.sets))
        print(f"Loaded Bonn sets {args.sets}: {X_raw.shape[0]} segments")
    else:
        X_raw, y, names = load_synthetic()
        print("No --sets given: using synthetic EEG fallback.")

    print("Extracting features ...")
    X = feature_matrix(X_raw, fs=FS)
    print(f"Feature matrix: {X.shape}")

    results = evaluate(X, y, n_splits=args.folds)
    print(f"\n{'Classifier':<16}{'CV accuracy':>14}")
    print("-" * 30)
    best = max(results, key=lambda k: results[k]["accuracy"])
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["accuracy"]):
        print(f"{name:<16}{r['accuracy']*100:>13.2f}%")

    os.makedirs(os.path.join(HERE, "examples"), exist_ok=True)
    cm = confusion_matrix(y, results[best]["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=names)
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(f"{best}  (CV acc = {results[best]['accuracy']*100:.1f}%)")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "examples", "confusion_matrix.png"), dpi=130)
    print(f"\nBest: {best}. Confusion matrix saved to examples/.")


if __name__ == "__main__":
    main()
