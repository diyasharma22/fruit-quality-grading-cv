"""
config.py
Central configuration for the fruit quality grading pipeline.
Keeping all tunables here means preprocessing, segmentation, and
classification thresholds can be adjusted without touching pipeline code.
"""

import os

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLE_DATA_DIR = os.path.join(DATA_DIR, "sample")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODEL_BUNDLE_PATH = os.path.join(MODELS_DIR, "grading_model.joblib")

# ---------- Preprocessing ----------
MAX_IMAGE_DIM = 512
GAUSSIAN_KERNEL = (5, 5)
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# ---------- Segmentation ----------
MORPH_KERNEL_SIZE = (5, 5)
MIN_BLEMISH_AREA = 15

BLEMISH_HSV_LOWER = (0, 0, 0)
BLEMISH_HSV_UPPER = (180, 255, 80)

# ---------- Feature extraction ----------
KMEANS_CLUSTERS = 3
HIST_BINS_PER_CHANNEL = 16
PCA_COMPONENTS = 6
CANNY_LOW = 50
CANNY_HIGH = 150
HARRIS_BLOCK_SIZE = 2
HARRIS_KSIZE = 3
HARRIS_K = 0.04

# ---------- Grading thresholds ----------
GRADE_THRESHOLDS = {
    "Good": 0.03,
    "Medium": 0.10,
}

# ---------- Classifier ----------
KNN_NEIGHBORS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------- Logging ----------
LOG_FILE = os.path.join(LOGS_DIR, "pipeline.log")
