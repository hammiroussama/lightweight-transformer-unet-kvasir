from __future__ import annotations

import argparse
import csv
import json
import time
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from datasets import build_loaders
from models import build_model
from utils.metrics import binary_metrics_from_logits, hd95
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one trained experiment.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timed-iterations", type=int, default=50)
    return parser.parse_args()


def measure_latency_ms(model, sample, device, warmup: int, iterations: int) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                _ = model(sample)
            end.record()
            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
        else:
            start_t = time.perf_counter()
            for _ in range(iterations):
                _ = model(sample)
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
    return elapsed_ms / iterations / sample.shape[0]


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    loader = build_loaders(config)["test"]
    threshold = float(config["train"].get("threshold", 0.5))
    rows: list[dict[str, float | str]] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="test", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = probs >= threshold
            for i, name in enumerate(batch["name"]):
                one_logits = logits[i : i + 1]
                one_mask = masks[i : i + 1]
                metrics = binary_metrics_from_logits(one_logits, one_mask, threshold)
                pred_np = preds[i, 0].cpu().numpy()
                mask_np = (one_mask[0, 0].cpu().numpy() >= 0.5)
                rows.append({"name": name, **metrics, "hd95_pixels": hd95(pred_np, mask_np)})

    output_dir = checkpoint_path.parent.parent
    per_image_path = output_dir / "per_image_metrics.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    sample_batch = next(iter(loader))["image"].to(device)
    latency = measure_latency_ms(model, sample_batch, device, args.warmup, args.timed_iterations)
    summary: dict[str, float | int | str] = {
        "experiment_name": str(config["experiment_name"]),
        "checkpoint": str(checkpoint_path),
        "test_samples": len(rows),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "inference_ms_per_image": latency,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "not available",
        "python_version": platform.python_version(),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_size": int(config["data"]["image_size"]),
        "batch_size": int(config["train"]["batch_size"]),
        "seed": int(config.get("seed", 42)),
    }
    for key in ("dice", "iou", "precision", "recall", "hd95_pixels"):
        values = np.array([float(row[key]) for row in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
