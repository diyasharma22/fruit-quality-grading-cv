"""
src/preprocessing.py
Module 1: Preprocessing & Enhancement.

Pipeline: load -> resize -> denoise (Gaussian + bilateral) -> CLAHE contrast
enhancement -> morphological clean-up (applied later on masks, not raw image).
"""

import os
import cv2
import numpy as np

import config


class ImageLoadError(Exception):
    """Raised when an image file cannot be read or is invalid."""
    pass


def safe_imread(path):
    """
    Safely load an image from disk.
    Raises ImageLoadError instead of returning None, so callers can
    catch one exception type and skip the file without crashing a batch run.
    """
    if not os.path.isfile(path):
        raise ImageLoadError(f"File not found: {path}")

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageLoadError(f"Could not decode image (corrupt or unsupported): {path}")

    return img


def resize_image(img, max_dim=config.MAX_IMAGE_DIM):
    """
    Resize image so its longest side is at most max_dim, preserving aspect ratio.
    Keeps runtime predictable regardless of input photo resolution.
    """
    h, w = img.shape[:2]
    longest = max(h, w)

    if longest <= max_dim:
        return img

    scale = max_dim / float(longest)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def denoise_image(img):
    """
    Two-stage denoising:
      1. Light Gaussian blur to remove fine sensor noise.
      2. Bilateral filter to smooth flat regions while preserving edges,
         which matters for later contour/blemish detection.
    """
    blurred = cv2.GaussianBlur(img, config.GAUSSIAN_KERNEL, 0)
    denoised = cv2.bilateralFilter(
        blurred,
        d=config.BILATERAL_D,
        sigmaColor=config.BILATERAL_SIGMA_COLOR,
        sigmaSpace=config.BILATERAL_SIGMA_SPACE,
    )
    return denoised


def enhance_contrast(img):
    """
    Apply CLAHE on the L channel of LAB color space.
    LAB separates lightness from color, so contrast is enhanced without
    distorting the fruit's actual hue (important since color feeds later
    into blemish detection and histogram features).
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID,
    )
    l_enhanced = clahe.apply(l_channel)

    merged = cv2.merge((l_enhanced, a_channel, b_channel))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced


def morphological_cleanup(mask):
    """
    Clean up a binary mask using morphological opening (remove small noise
    specks) followed by closing (fill small holes inside the foreground).
    Used on segmentation masks, not the raw color image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.MORPH_KERNEL_SIZE)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def preprocess_image(path):
    """
    Full Module 1 pipeline for a single image path.
    Returns the enhanced BGR image ready for segmentation.
    """
    img = safe_imread(path)
    img = resize_image(img)
    img = denoise_image(img)
    img = enhance_contrast(img)
    return img
