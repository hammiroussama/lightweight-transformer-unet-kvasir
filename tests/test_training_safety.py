import csv

import pytest
import torch

from datasets import build_loaders
from utils.losses import DiceLoss


@pytest.mark.parametrize("size", [256, 384])
def test_half_precision_dice_matches_float32(size):
    logits = torch.full((1, 1, size, size), 12.0, dtype=torch.float16, requires_grad=True)
    targets = torch.ones_like(logits, dtype=torch.float32)
    loss = DiceLoss()(logits, targets)
    reference = DiceLoss()(logits.detach().float(), targets)
    assert torch.isfinite(loss)
    assert 0 <= loss.item() < 1e-5
    torch.testing.assert_close(loss, reference)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_evaluation_does_not_generate_a_new_split(tmp_path):
    config = {"data": {"root": str(tmp_path), "split_csv": str(tmp_path / "missing.csv")}, "train": {}}
    with pytest.raises(FileNotFoundError, match="Saved split"):
        build_loaders(config, splits=("val",), require_split=True)
    assert not (tmp_path / "missing.csv").exists()


def test_duplicate_images_across_splits_are_rejected(tmp_path):
    split = tmp_path / "split.csv"
    with split.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "mask", "split"])
        writer.writerow(["images/a.jpg", "masks/a.jpg", "train"])
        writer.writerow(["images/a.jpg", "masks/a.jpg", "val"])
    config = {"data": {"root": str(tmp_path), "split_csv": str(split), "image_size": 32}, "train": {}}
    with pytest.raises(ValueError, match="Duplicate"):
        build_loaders(config)
