from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from datasets import build_loaders
from models import build_model
from utils import BCEDiceLoss, load_config, set_seed
from utils.engine import run_epoch
from utils.io import ensure_dir, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one Kvasir-SEG experiment.")
    parser.add_argument("--config", required=True, help="Path to a YAML configuration file.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override.")
    parser.add_argument("--device", default=None, help="cuda, cpu, or cuda:0. Auto-detected by default.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["train"]["epochs"] = int(args.epochs)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = ensure_dir(config["output_dir"])
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")

    loaders = build_loaders(config)
    model = build_model(config["model"]).to(device)
    criterion = BCEDiceLoss(
        bce_weight=float(config["train"].get("bce_weight", 0.5)),
        dice_weight=float(config["train"].get("dice_weight", 0.5)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"].get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(config["train"]["epochs"]), 1)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    threshold = float(config["train"].get("threshold", 0.5))
    patience = int(config["train"].get("early_stopping_patience", 7))
    best_val_dice = -1.0
    epochs_without_improvement = 0
    history_path = output_dir / "history.csv"
    fieldnames = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_dice",
        "train_iou",
        "train_precision",
        "train_recall",
        "val_loss",
        "val_dice",
        "val_iou",
        "val_precision",
        "val_recall",
    ]
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_metrics = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            threshold=threshold,
            description=f"train {epoch}",
        )
        val_metrics = run_epoch(
            model,
            loaders["val"],
            criterion,
            device,
            optimizer=None,
            scaler=None,
            threshold=threshold,
            description=f"val {epoch}",
        )
        lr = optimizer.param_groups[0]["lr"]
        row = {"epoch": epoch, "learning_rate": lr}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        with history_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                epoch,
                config,
                best_val_dice,
            )
        else:
            epochs_without_improvement += 1
        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            epoch,
            config,
            best_val_dice,
        )
        scheduler.step()
        print(
            f"Epoch {epoch:03d} | train Dice={train_metrics['dice']:.4f} | "
            f"val Dice={val_metrics['dice']:.4f} | best={best_val_dice:.4f}"
        )
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch} epochs.")
            break


if __name__ == "__main__":
    main()
