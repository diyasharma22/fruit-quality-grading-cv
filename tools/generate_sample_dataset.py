"""
tools/generate_sample_dataset.py

Generates a synthetic fruit-image dataset for developing and testing the
pipeline without needing internet access to a real labelled dataset.

Folder layout matches what the real Mendeley "Fruits Dataset for
Classification" uses, so real data can be substituted later without
changing the data loader:

    data/sample/<fruit_type>/<condition>/img_XXX.jpg

Usage:
    python tools/generate_sample_dataset.py --out data/sample --per-class 40
"""

import os
import argparse
import random

import cv2
import numpy as np

FRUITS = {
    "apple":  (60, 60, 200),   # BGR
    "banana": (60, 220, 230),
    "orange": (30, 140, 250),
}

IMG_SIZE = 300


def make_background(size):
    """Plain, slightly textured background (simulates a table/white surface)."""
    shade = random.randint(200, 235)
    bg = np.full((size, size, 3), shade, dtype=np.uint8)
    noise = np.random.randint(-5, 5, bg.shape, dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return bg


def draw_fruit(canvas, color, rotten=False):
    """Draw a circular fruit shape, optionally with dark blemish blobs."""
    h, w = canvas.shape[:2]
    center = (w // 2 + random.randint(-10, 10), h // 2 + random.randint(-10, 10))
    radius = random.randint(90, 110)

    # base fruit body with slight per-pixel color jitter for realism
    overlay = canvas.copy()
    cv2.circle(overlay, center, radius, color, -1, lineType=cv2.LINE_AA)
    # soft shading: darker gradient near edge
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    canvas[mask > 0] = overlay[mask > 0]

    if rotten:
        num_blemishes = random.randint(3, 7)
        for _ in range(num_blemishes):
            angle = random.uniform(0, 2 * np.pi)
            dist = random.uniform(0, radius * 0.7)
            bx = int(center[0] + dist * np.cos(angle))
            by = int(center[1] + dist * np.sin(angle))
            b_radius = random.randint(8, 22)
            dark_color = (
                random.randint(10, 40),
                random.randint(10, 40),
                random.randint(10, 40),
            )
            cv2.circle(canvas, (bx, by), b_radius, dark_color, -1, lineType=cv2.LINE_AA)

    return canvas


def generate_image(fruit_color, rotten):
    canvas = make_background(IMG_SIZE)
    canvas = draw_fruit(canvas, fruit_color, rotten=rotten)
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic fruit dataset")
    parser.add_argument("--out", default="data/sample", help="Output root folder")
    parser.add_argument("--per-class", type=int, default=40,
                         help="Images per (fruit, condition) combination")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    total = 0
    for fruit_name, color in FRUITS.items():
        for condition, rotten in (("fresh", False), ("rotten", True)):
            out_dir = os.path.join(args.out, fruit_name, condition)
            os.makedirs(out_dir, exist_ok=True)

            for i in range(args.per_class):
                img = generate_image(color, rotten)
                filename = os.path.join(out_dir, f"img_{i:03d}.jpg")
                cv2.imwrite(filename, img)
                total += 1

    print(f"Generated {total} synthetic images under '{args.out}'")


if __name__ == "__main__":
    main()
