from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_histories(history_files: dict[str, Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    for name, path in history_files.items():
        df = pd.read_csv(path)
        plt.plot(df["epoch"], df["val_dice"], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Dice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "validation_dice_curves.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    for name, path in history_files.items():
        df = pd.read_csv(path)
        plt.plot(df["epoch"], df["val_loss"], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "validation_loss_curves.png", dpi=220)
    plt.close()


def denormalize(image: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std = np.array([0.229, 0.224, 0.225])[:, None, None]
    return np.clip(image * std + mean, 0, 1)
