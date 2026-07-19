from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from datasets.kvasir_dataset import MEAN, STD
from models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a mask for one colonoscopy image.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    image_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(args.image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_h, original_w = image_rgb.shape[:2]
    size = int(config["data"]["image_size"])
    resized = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    normalized = (resized - MEAN) / STD
    tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).float().unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    mask = (prob >= float(config["train"].get("threshold", 0.5))).astype(np.uint8) * 255
    mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(output_path)
    overlay = image_rgb.copy()
    overlay[mask > 0] = (0.55 * overlay[mask > 0] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
    Image.fromarray(overlay).save(output_path.with_name(output_path.stem + "_overlay.png"))


if __name__ == "__main__":
    main()
