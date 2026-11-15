from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

CONFIGS = [
    "configs/unet.yaml",
    "configs/attention_unet.yaml",
    "configs/transformer_unet.yaml",
    "configs/proposed_model.yaml",
]


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all four controlled experiments.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--final-test", action="store_true", help="Confirm model selection is complete before running all held-out evaluations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.final_test:
        raise SystemExit("Use train.py and evaluate.py --split val during development. Add --final-test only after model selection.")
    python = sys.executable
    summaries = []
    for config_path in CONFIGS:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        output_dir = Path(config["output_dir"])
        checkpoint = output_dir / "checkpoints" / "best.pt"
        if not args.skip_training:
            command = [python, "train.py", "--config", config_path]
            if args.epochs is not None:
                command += ["--epochs", str(args.epochs)]
            if args.device:
                command += ["--device", args.device]
            run(command)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        command = [python, "evaluate.py", "--checkpoint", str(checkpoint), "--split", "test"]
        if args.device:
            command += ["--device", args.device]
        run(command)
        with (output_dir / "metrics.json").open("r", encoding="utf-8") as f:
            summaries.append(json.load(f))
    Path("results").mkdir(exist_ok=True)
    df = pd.DataFrame(summaries)
    df.to_csv("results/summary.csv", index=False)
    df[[
        "experiment_name",
        "dice_mean",
        "iou_mean",
        "precision_mean",
        "recall_mean",
        "hd95_pixels_mean",
        "parameters",
        "inference_ms_per_image",
    ]].to_csv("results/ablation.csv", index=False)
    print("Completed all experiments. See results/ for comparison tables and each run directory for evaluation details.")


if __name__ == "__main__":
    main()
