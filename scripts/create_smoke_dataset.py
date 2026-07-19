from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny synthetic segmentation dataset for code checks.")
    parser.add_argument("--output", default="data/Kvasir-SEG")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output)
    image_dir = root / "images"
    mask_dir = root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for i in range(args.samples):
        h = w = args.size
        image = rng.normal(80, 18, size=(h, w, 3)).clip(0, 255).astype(np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (int(rng.integers(w // 3, 2 * w // 3)), int(rng.integers(h // 3, 2 * h // 3)))
        axes = (int(rng.integers(w // 8, w // 4)), int(rng.integers(h // 8, h // 4)))
        angle = float(rng.integers(0, 180))
        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
        image[mask > 0] = np.clip(image[mask > 0].astype(np.int16) + np.array([80, 25, 45]), 0, 255)
        name = f"synthetic_{i:03d}.jpg"
        cv2.imwrite(str(image_dir / name), image)
        cv2.imwrite(str(mask_dir / name), mask)
    print(f"Created {args.samples} synthetic pairs under {root}")


if __name__ == "__main__":
    main()
