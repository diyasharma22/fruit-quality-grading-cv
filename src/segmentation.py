"""
src/segmentation.py
Module 2 (part 1): Segmentation.

Provides:
  - segment_foreground(): Otsu thresholding on HSV saturation channel
    to isolate the fruit from its background.
  - kmeans_color_clusters(): K-Means clustering of pixel colors.
  - detect_blemishes(): HSV-based dark/brown blemish detection inside
    the fruit mask.
"""

import cv2
import numpy as np

import config
from src.preprocessing import morphological_cleanup


def segment_foreground(img):
    """
    Segment the fruit from its background using Otsu's automatic
    thresholding on the HSV saturation channel.

    Saturation is used (not grayscale) because fruit is typically more
    colorful/saturated than a plain background, regardless of the
    fruit's actual hue -- so this generalizes across apples, bananas,
    oranges, etc. without per-fruit tuning.

    Returns a binary mask (0/255) of the same height/width as img,
    containing only the largest connected foreground region.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    # Otsu's method picks the threshold automatically from the histogram
    _, mask = cv2.threshold(
        saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    mask = morphological_cleanup(mask)
    mask = _keep_largest_component(mask)

    return mask


def _keep_largest_component(mask):
    """
    Keep only the largest connected white region in a binary mask.
    Removes stray noise blobs that survive morphological cleanup.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if num_labels <= 1:
        return mask  # no foreground found

    # stats[0] is background; find largest among the rest by area
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    cleaned = np.zeros_like(mask)
    cleaned[labels == largest_label] = 255
    return cleaned


def kmeans_color_clusters(img, mask=None, k=config.KMEANS_CLUSTERS):
    """
    Cluster pixel colors into k groups using K-Means.
    If a mask is provided, only foreground pixels are clustered.

    Returns:
      centers: (k, 3) array of cluster center colors (BGR)
      labels: cluster assignment per pixel considered
    """
    pixels = img.reshape(-1, 3).astype(np.float32)

    if mask is not None:
        flat_mask = mask.reshape(-1) > 0
        pixels = pixels[flat_mask]

    if len(pixels) < k:
        # not enough pixels to cluster meaningfully
        return np.zeros((k, 3), dtype=np.float32), np.array([])

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, attempts=5,
        flags=cv2.KMEANS_RANDOM_CENTERS
    )

    return centers, labels


def detect_blemishes(img, fruit_mask):
    """
    Detect dark/brown blemish regions inside the fruit mask using HSV
    thresholding on the Value channel (blemishes are darker than
    healthy fruit surface, regardless of the fruit's hue).

    Small contours below MIN_BLEMISH_AREA are discarded as noise.

    Returns:
      blemish_mask: binary mask of blemish pixels (0/255)
      defect_ratio: blemish_area / fruit_area (float, 0.0 - 1.0)
      contours: list of blemish contours (for annotation/drawing)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    raw_blemish_mask = cv2.inRange(
        hsv, np.array(config.BLEMISH_HSV_LOWER), np.array(config.BLEMISH_HSV_UPPER)
    )

    # restrict to inside the fruit only
    raw_blemish_mask = cv2.bitwise_and(raw_blemish_mask, raw_blemish_mask, mask=fruit_mask)

    contours, _ = cv2.findContours(
        raw_blemish_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    filtered_contours = [c for c in contours if cv2.contourArea(c) >= config.MIN_BLEMISH_AREA]

    blemish_mask = np.zeros_like(raw_blemish_mask)
    cv2.drawContours(blemish_mask, filtered_contours, -1, 255, thickness=cv2.FILLED)

    fruit_area = int(np.count_nonzero(fruit_mask))
    blemish_area = int(np.count_nonzero(blemish_mask))

    defect_ratio = (blemish_area / fruit_area) if fruit_area > 0 else 0.0

    return blemish_mask, defect_ratio, filtered_contours
