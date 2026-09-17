"""
tools/build_dataset.py

Walks the sample dataset, runs the full Module 1-3 pipeline on every
image, fits PCA once across all histograms, and saves:
  - models/pca.joblib      (fitted PCA, reusable at inference time)
  - data/features.joblib   (X feature matrix, y labels, fruit types)

Usage:
    python tools/build_dataset.py --data data/sample
"""

import os
import sys
import argparse

import numpy as np
import joblib

# Ensure project root is on sys.path so 'config' and 'src' resolve
# regardless of the current working directory this script is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.preprocessing import preprocess_image
from src.segmentation import segment_foreground, detect_blemishes
from src.features import extract_raw_features, fit_histogram_pca, build_feature_vector


def collect_image_paths(data_dir):
    """
    Walk data_dir/<fruit>/<condition>/*.jpg and return a list of
    (path, fruit, condition) tuples.
    """
    records = []
    for fruit in sorted(os.listdir(data_dir)):
        fruit_dir = os.path.join(data_dir, fruit)
        if not os.path.isdir(fruit_dir):
            continue
        for condition in sorted(os.listdir(fruit_dir)):
            cond_dir = os.path.join(fruit_dir, condition)
            if not os.path.isdir(cond_dir):
                continue
            for fname in sorted(os.listdir(cond_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    records.append((os.path.join(cond_dir, fname), fruit, condition))
    return records


def main():
    parser = argparse.ArgumentParser(description="Build feature dataset from sample images")
    parser.add_argument("--data", default="data/sample", help="Root folder of labelled images")
    args = parser.parse_args()

    records = collect_image_paths(args.data)
    print(f"Found {len(records)} images under '{args.data}'")

    raw_features_list = []
    labels = []
    fruit_types = []
    skipped = 0

    for path, fruit, condition in records:
        try:
            img = preprocess_image(path)
            mask = segment_foreground(img)
            _, defect_ratio, _ = detect_blemishes(img, mask)
            raw = extract_raw_features(img, mask, defect_ratio)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            skipped += 1
            continue

        raw_features_list.append(raw)
        labels.append(condition)   # "fresh" / "rotten" -- the grading target
        fruit_types.append(fruit)  # "apple" / "banana" / "orange" -- kept for reference

    print(f"Processed {len(raw_features_list)} images ({skipped} skipped)")

    # Fit PCA once across every histogram in the dataset
    pca = fit_histogram_pca([r["histogram"] for r in raw_features_list])

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    pca_path = os.path.join(config.MODELS_DIR, "pca.joblib")
    joblib.dump(pca, pca_path)
    print(f"Saved fitted PCA to {pca_path}")

    # Build final feature vectors using the fitted PCA
    X = np.vstack([build_feature_vector(r, pca) for r in raw_features_list])
    y = np.array(labels)
    fruit_arr = np.array(fruit_types)

    features_path = os.path.join(config.DATA_DIR, "features.joblib")
    joblib.dump({"X": X, "y": y, "fruit_types": fruit_arr}, features_path)
    print(f"Saved feature matrix {X.shape} and labels {y.shape} to {features_path}")


if __name__ == "__main__":
    main()
