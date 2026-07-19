from __future__ import annotations

import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _find_pairs(root: Path) -> list[tuple[Path, Path]]:
    image_dir = root / "images"
    mask_dir = root / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {image_dir} and {mask_dir}. See README.md for dataset setup."
        )
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    mask_by_stem = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
    pairs = [(img, mask_by_stem[img.stem]) for img in images if img.stem in mask_by_stem]
    if not pairs:
        raise RuntimeError("No image-mask pairs were found.")
    return pairs


def create_split_csv(
    root: str | Path,
    output_csv: str | Path,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Path:
    root = Path(root)
    output_csv = Path(output_csv)
    pairs = _find_pairs(root)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n = len(pairs)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(max(n_train, 1), n - 2)
    n_val = min(max(n_val, 1), n - n_train - 1)
    labels = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "mask", "split"])
        for (image, mask), split in zip(pairs, labels):
            writer.writerow([image.relative_to(root), mask.relative_to(root), split])
    return output_csv


def _transform(image: np.ndarray, mask: np.ndarray, image_size: int, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    if training:
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        if random.random() < 0.2:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        if random.random() < 0.4:
            angle = random.uniform(-20.0, 20.0)
            matrix = cv2.getRotationMatrix2D((image_size / 2, image_size / 2), angle, 1.0)
            image = cv2.warpAffine(
                image,
                matrix,
                (image_size, image_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            mask = cv2.warpAffine(
                mask,
                matrix,
                (image_size, image_size),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )
        if random.random() < 0.3:
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-18.0, 18.0)
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    image_f = image.astype(np.float32) / 255.0
    image_f = (image_f - MEAN) / STD
    mask_f = (mask > 127).astype(np.float32)
    image_t = torch.from_numpy(np.ascontiguousarray(image_f.transpose(2, 0, 1))).float()
    mask_t = torch.from_numpy(np.ascontiguousarray(mask_f[None])).float()
    return image_t, mask_t


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class KvasirSegDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split_csv: str | Path,
        split: str,
        image_size: int = 256,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        with Path(split_csv).open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.items = [row for row in rows if row["split"] == split]
        if not self.items:
            raise RuntimeError(f"No samples found for split={split!r} in {split_csv}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.items[index]
        image_path = self.root / row["image"]
        mask_path = self.root / row["mask"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(f"Failed to read {image_path} or {mask_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_t, mask_t = _transform(image, mask, self.image_size, training=(self.split == "train"))
        return {"image": image_t, "mask": mask_t, "name": image_path.name}


def build_loaders(config: dict) -> dict[str, DataLoader]:
    data_cfg = config["data"]
    train_cfg = config["train"]
    root = Path(data_cfg["root"])
    split_csv = Path(data_cfg["split_csv"])
    if not split_csv.exists():
        create_split_csv(
            root,
            split_csv,
            seed=int(config.get("seed", 42)),
            train_ratio=float(data_cfg.get("train_ratio", 0.70)),
            val_ratio=float(data_cfg.get("val_ratio", 0.15)),
        )
    common = dict(root=root, split_csv=split_csv, image_size=int(data_cfg["image_size"]))
    datasets = {split: KvasirSegDataset(split=split, **common) for split in ("train", "val", "test")}
    workers = int(data_cfg.get("num_workers", 4))
    pin_memory = bool(torch.cuda.is_available())
    generator = torch.Generator()
    generator.manual_seed(int(config.get("seed", 42)))
    return {
        split: DataLoader(
            ds,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=(split == "train"),
            num_workers=workers,
            pin_memory=pin_memory,
            persistent_workers=(workers > 0),
            worker_init_fn=_seed_worker,
            generator=generator,
            drop_last=False,
        )
        for split, ds in datasets.items()
    }
