import os
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

# Dataset locations
DATA_DIR = Path("Train")

CLASSES = {
    "non_alcoholic": 0,
    "alcoholic": 1
}


def extract_features(image_path):
    """Extract numerical features from one spectrogram image."""

    with Image.open(image_path) as image:
        img = np.asarray(image.convert("L"), dtype=np.float32)

    # Normalise pixel intensities
    img = img / 255.0

    features = {}

    # Basic intensity statistics
    features["mean_intensity"] = np.mean(img)
    features["std_intensity"] = np.std(img)
    features["min_intensity"] = np.min(img)
    features["max_intensity"] = np.max(img)
    features["p25"] = np.percentile(img, 25)
    features["p50"] = np.percentile(img, 50)
    features["p75"] = np.percentile(img, 75)

    # Image contrast
    features["contrast"] = np.std(img)

    # Pixel intensity histogram
    hist, _ = np.histogram(img, bins=16, range=(0, 1))
    hist = hist / hist.sum()

    for i, value in enumerate(hist):
        features[f"histogram_{i}"] = value

    # Horizontal and vertical intensity profiles
    horizontal = np.mean(img, axis=1)
    vertical = np.mean(img, axis=0)

    features["horizontal_mean"] = np.mean(horizontal)
    features["horizontal_std"] = np.std(horizontal)
    features["vertical_mean"] = np.mean(vertical)
    features["vertical_std"] = np.std(vertical)

    # Image gradients
    gradient_y, gradient_x = np.gradient(img)

    features["gradient_x"] = np.mean(np.abs(gradient_x))
    features["gradient_y"] = np.mean(np.abs(gradient_y))
    features["gradient_std"] = np.std(
        np.sqrt(gradient_x**2 + gradient_y**2)
    )

    return features


def main():
    rows = []

    for class_name, label in CLASSES.items():

        folder = DATA_DIR / class_name

        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")

        image_paths = sorted(
            path for path in folder.rglob("*")
            if path.suffix.lower() in [".png", ".jpg", ".jpeg"]
        )

        print(f"{class_name}: {len(image_paths)} images")

        for i, image_path in enumerate(image_paths):

            try:
                features = extract_features(image_path)

                features["status"] = label
                features["class_name"] = class_name
                features["File_Name"] = image_path.name

                rows.append(features)

            except Exception as error:
                print(f"Could not process {image_path}: {error}")

            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(image_paths)}")

    df = pd.DataFrame(rows)

    output_path = DATA_DIR / "cleaned_dataset.csv"
    df.to_csv(output_path, index=False)

    print("\n===== DATASET SUMMARY =====")
    print(f"Total images processed: {len(df)}")
    print(f"Dataset shape: {df.shape}")
    print("\nClass distribution:")
    print(df["class_name"].value_counts())
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
