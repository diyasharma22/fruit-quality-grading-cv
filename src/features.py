"""
src/features.py
Module 3: Feature extraction.

Two layers:
  1. Per-image raw feature extraction (extract_raw_features) -- independent
     of other images in the dataset.
  2. Dataset-level PCA fit/transform for the color histogram, since PCA
     needs variance across many samples to be meaningful. The fitted PCA
     model is persisted with joblib so a single new image can be
     transformed at inference time without refitting.
"""

import cv2
import numpy as np
import joblib
from sklearn.decomposition import PCA

import config
from src.segmentation import kmeans_color_clusters


def compute_color_histogram(img, mask):
    """
    Raw HSV color histogram restricted to the fruit mask.
    16 bins per channel (config.HIST_BINS_PER_CHANNEL) -> 16*16*16 = 4096-d.
    Normalized so histograms are comparable regardless of fruit mask size.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bins = config.HIST_BINS_PER_CHANNEL

    hist = cv2.calcHist(
        [hsv], [0, 1, 2], mask, [bins, bins, bins],
        [0, 180, 0, 256, 0, 256]
    )
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def compute_edge_density(img, mask):
    """
    Canny edge density inside the fruit mask: (edge pixels) / (fruit area).
    Rotten/bruised fruit tends to have more irregular surface texture,
    producing more edge pixels than a smooth, fresh surface.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, config.CANNY_LOW, config.CANNY_HIGH)
    edges_in_fruit = cv2.bitwise_and(edges, edges, mask=mask)

    fruit_area = int(np.count_nonzero(mask))
    edge_pixels = int(np.count_nonzero(edges_in_fruit))

    return (edge_pixels / fruit_area) if fruit_area > 0 else 0.0


def compute_corner_density(img, mask):
    """
    Harris corner density inside the fruit mask: (corner pixels) / (fruit area).
    Blemishes and irregular textures produce more corner-like responses
    than a smooth fruit surface.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_f = np.float32(gray)

    harris = cv2.cornerHarris(
        gray_f, config.HARRIS_BLOCK_SIZE, config.HARRIS_KSIZE, config.HARRIS_K
    )
    harris = cv2.dilate(harris, None)

    corner_mask = (harris > 0.01 * harris.max()).astype(np.uint8) * 255
    corner_mask = cv2.bitwise_and(corner_mask, corner_mask, mask=mask)

    fruit_area = int(np.count_nonzero(mask))
    corner_pixels = int(np.count_nonzero(corner_mask))

    return (corner_pixels / fruit_area) if fruit_area > 0 else 0.0


def extract_raw_features(img, fruit_mask, defect_ratio):
    """
    Extract all per-image raw features that don't require dataset-level
    fitting. Returns a dict so callers can pick apart pieces (e.g. the
    histogram needs PCA later; the rest are used as-is).
    """
    histogram = compute_color_histogram(img, fruit_mask)
    edge_density = compute_edge_density(img, fruit_mask)
    corner_density = compute_corner_density(img, fruit_mask)
    centers, _ = kmeans_color_clusters(img, fruit_mask)
    dominant_colors = centers.flatten()  # k*3-d vector

    return {
        "histogram": histogram,
        "edge_density": edge_density,
        "corner_density": corner_density,
        "dominant_colors": dominant_colors,
        "defect_ratio": defect_ratio,
    }


def fit_histogram_pca(histograms, n_components=config.PCA_COMPONENTS):
    """
    Fit PCA on a list/array of raw histograms collected across the dataset.
    Must be called once over many samples -- fitting PCA on a single
    histogram is meaningless, since PCA finds directions of variance
    *across* samples.
    """
    X = np.vstack(histograms)
    pca = PCA(n_components=n_components, random_state=config.RANDOM_STATE)
    pca.fit(X)
    return pca


def save_pca(pca, path):
    joblib.dump(pca, path)


def load_pca(path):
    return joblib.load(path)


def build_feature_vector(raw_features, pca):
    """
    Combine a raw_features dict into the final fixed-length feature vector
    used by the classifier:
      [pca(histogram) | edge_density | corner_density | dominant_colors | defect_ratio]
    """
    hist_reduced = pca.transform(raw_features["histogram"].reshape(1, -1)).flatten()

    vector = np.concatenate([
        hist_reduced,
        [raw_features["edge_density"]],
        [raw_features["corner_density"]],
        raw_features["dominant_colors"],
        [raw_features["defect_ratio"]],
    ])
    return vector
