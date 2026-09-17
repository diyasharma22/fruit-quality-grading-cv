"""
src/report.py
Module 5: Reporting.

Draws an annotated visualization (fruit outline + blemish contours +
grade text) and saves it to the outputs directory, so each graded
image produces a self-contained, shareable result.
"""

import os
import cv2
import numpy as np

import config


def draw_annotations(img, fruit_mask, blemish_contours, grade_text_lines):
    """
    Draw the fruit boundary (green) and blemish contours (red) on a copy
    of img, then overlay grade_text_lines (list of strings) in the
    top-left corner.

    Returns the annotated image (does not modify img in place).
    """
    annotated = img.copy()

    # Fruit boundary: outer contour of the mask
    fruit_contours, _ = cv2.findContours(
        fruit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(annotated, fruit_contours, -1, (0, 255, 0), 2)

    # Blemish contours
    cv2.drawContours(annotated, blemish_contours, -1, (0, 0, 255), 2)

    # Text overlay
    y = 25
    for line in grade_text_lines:
        cv2.putText(
            annotated, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            annotated, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA
        )
        y += 25

    return annotated


def save_report_image(annotated_img, source_path, out_dir=config.OUTPUTS_DIR):
    """
    Save annotated_img to out_dir, named after the source image, and
    return the saved path.
    """
    os.makedirs(out_dir, exist_ok=True)
    # Include fruit/condition folder names in the filename to avoid
    # collisions between e.g. apple/rotten/img_000.jpg and
    # banana/fresh/img_000.jpg both producing img_000_graded.jpg.
    norm_path = source_path.replace("\\", "/")
    parts = [p for p in norm_path.split("/") if p]
    base = os.path.splitext(parts[-1])[0]
    prefix_parts = parts[-3:-1] if len(parts) >= 3 else []
    prefix = "_".join(prefix_parts)
    fname = f"{prefix}_{base}_graded.jpg" if prefix else f"{base}_graded.jpg"
    out_path = os.path.join(out_dir, fname)
    cv2.imwrite(out_path, annotated_img)
    return out_path
