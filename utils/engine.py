from __future__ import annotations

from contextlib import nullcontext

import torch
from tqdm import tqdm

from .metrics import AverageMeter, binary_metrics_from_logits


def run_epoch(
    model: torch.nn.Module,
    loader,
    criterion,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    threshold: float = 0.5,
    description: str = "epoch",
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    meters = {name: AverageMeter() for name in ("loss", "dice", "iou", "precision", "recall")}
    iterator = tqdm(loader, desc=description, leave=False)
    for batch in iterator:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        amp_enabled = device.type == "cuda"
        context = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled) if amp_enabled else nullcontext()
        with torch.set_grad_enabled(training), context:
            logits = model(images)
            loss = criterion(logits, masks)
        if training:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        metrics = binary_metrics_from_logits(logits.detach(), masks, threshold)
        batch_size = images.shape[0]
        meters["loss"].update(float(loss.item()), batch_size)
        for key, value in metrics.items():
            meters[key].update(value, batch_size)
        iterator.set_postfix(loss=f"{meters['loss'].average:.4f}", dice=f"{meters['dice'].average:.4f}")
    return {key: meter.average for key, meter in meters.items()}
