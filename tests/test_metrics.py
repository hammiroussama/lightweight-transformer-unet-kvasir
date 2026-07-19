import numpy as np
import torch

from utils.metrics import binary_metrics_from_logits, hd95


def test_perfect_metrics() -> None:
    target = torch.ones(1, 1, 8, 8)
    logits = torch.full_like(target, 20.0)
    metrics = binary_metrics_from_logits(logits, target)
    assert metrics["dice"] > 0.999
    assert metrics["iou"] > 0.999


def test_hd95_identical() -> None:
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 5:13] = True
    assert hd95(mask, mask) == 0.0
