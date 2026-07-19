from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

EPS = 1e-7


def binary_metrics_from_logits(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = probs >= threshold
    targets_b = targets >= 0.5
    dims = tuple(range(1, preds.ndim))
    tp = (preds & targets_b).sum(dims).float()
    fp = (preds & ~targets_b).sum(dims).float()
    fn = (~preds & targets_b).sum(dims).float()
    dice = (2 * tp + EPS) / (2 * tp + fp + fn + EPS)
    iou = (tp + EPS) / (tp + fp + fn + EPS)
    precision = (tp + EPS) / (tp + fp + EPS)
    recall = (tp + EPS) / (tp + fn + EPS)
    return {
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
    }


def hd95(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return float(math.hypot(*pred.shape[-2:]))
    pred_surface = pred ^ binary_erosion(pred)
    target_surface = target ^ binary_erosion(target)
    dt_target = distance_transform_edt(~target_surface)
    dt_pred = distance_transform_edt(~pred_surface)
    distances = np.concatenate([dt_target[pred_surface], dt_pred[target_surface]])
    return float(np.percentile(distances, 95)) if distances.size else 0.0


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def average(self) -> float:
        return self.total / max(self.count, 1)
