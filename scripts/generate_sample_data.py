"""Generate synthetic 'machined part' images for demoing the pipeline.

Good parts:   ring-shaped part with mild noise/lighting variation.
Defect parts: same part with scratches, holes, or missing chunks.

Usage: python scripts/generate_sample_data.py --good 200 --defect 30
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def base_part(size=256):
    img = np.full((size, size), 40, np.uint8)
    c = size // 2
    cv2.circle(img, (c, c), 90, 180, -1)
    cv2.circle(img, (c, c), 45, 60, -1)
    for ang in range(0, 360, 60):  # bolt holes
        x = int(c + 68 * np.cos(np.radians(ang)))
        y = int(c + 68 * np.sin(np.radians(ang)))
        cv2.circle(img, (x, y), 8, 50, -1)
    return img


def augment(img):
    noise = np.random.normal(0, 6, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    alpha = random.uniform(0.9, 1.1)  # lighting drift
    return np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def add_defect(img):
    img = img.copy()
    kind = random.choice(["scratch", "hole", "chip"])
    if kind == "scratch":
        p1 = (random.randint(60, 200), random.randint(60, 200))
        p2 = (p1[0] + random.randint(-60, 60), p1[1] + random.randint(-60, 60))
        cv2.line(img, p1, p2, 255, random.randint(2, 4))
    elif kind == "hole":
        cv2.circle(img, (random.randint(80, 176), random.randint(80, 176)),
                   random.randint(6, 14), 0, -1)
    else:  # chip on outer edge
        ang = random.uniform(0, 2 * np.pi)
        x, y = int(128 + 88 * np.cos(ang)), int(128 + 88 * np.sin(ang))
        cv2.circle(img, (x, y), random.randint(10, 18), 40, -1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--good", type=int, default=200)
    ap.add_argument("--defect", type=int, default=30)
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()

    good_dir = Path(args.out) / "good"
    defect_dir = Path(args.out) / "defect"
    good_dir.mkdir(parents=True, exist_ok=True)
    defect_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.good):
        cv2.imwrite(str(good_dir / f"good_{i:04d}.png"), augment(base_part()))
    for i in range(args.defect):
        cv2.imwrite(str(defect_dir / f"defect_{i:04d}.png"),
                    augment(add_defect(base_part())))
    print(f"wrote {args.good} good + {args.defect} defect images to {args.out}")


if __name__ == "__main__":
    main()
