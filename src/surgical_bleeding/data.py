from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def collect_files(root: str | Path, recursive: bool = False) -> list[Path]:
    root = Path(root)
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def pair_by_stem(images_dir: str | Path, masks_dir: str | Path, mask_suffix: str = "", recursive: bool = False) -> list[tuple[Path, Path]]:
    images = {p.stem: p for p in collect_files(images_dir, recursive)}
    masks = {(p.stem[:-len(mask_suffix)] if mask_suffix and p.stem.endswith(mask_suffix) else p.stem): p for p in collect_files(masks_dir, recursive)}
    keys = sorted(images.keys() & masks.keys())
    return [(images[key], masks[key]) for key in keys]


def read_pairs_csv(csv_path: str | Path) -> list[tuple[Path, Path]]:
    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"image_path", "mask_path"}.issubset(reader.fieldnames):
            raise ValueError("Pairs CSV must contain image_path and mask_path columns")
        return [(Path(row["image_path"]), Path(row["mask_path"])) for row in reader]


class BloodSegmentationDataset(Dataset):
    def __init__(self, pairs: Sequence[tuple[str | Path, str | Path]], img_size: int = 512, augment: bool = False, mask_threshold: int = 0):
        self.pairs = [(Path(image), Path(mask)) for image, mask in pairs]
        if not self.pairs:
            raise ValueError("Dataset is empty")
        self.img_size = int(img_size)
        self.augment = bool(augment)
        self.mask_threshold = int(mask_threshold)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        image_path, mask_path = self.pairs[idx]
        image = Image.open(image_path).convert("RGB").resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize((self.img_size, self.img_size), Image.Resampling.NEAREST)
        target = (np.asarray(mask, dtype=np.uint8) > self.mask_threshold).astype(np.uint8)
        if self.augment:
            image, target = self._augment(image, target)
        return {"image": TF.to_tensor(image), "mask": torch.from_numpy(target).float(), "image_path": str(image_path), "mask_path": str(mask_path)}

    @staticmethod
    def _augment(image: Image.Image, target: np.ndarray) -> tuple[Image.Image, np.ndarray]:
        if random.random() < 0.5:
            image, target = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), np.fliplr(target).copy()
        if random.random() < 0.5:
            angle = random.uniform(-10, 10)
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR)
            target = np.asarray(Image.fromarray(target).rotate(angle, resample=Image.Resampling.NEAREST), dtype=np.uint8)
        if random.random() < 0.3:
            arr = np.asarray(image, dtype=np.float32) * random.uniform(0.85, 1.15)
            mean = arr.mean(axis=(0, 1), keepdims=True)
            arr = (arr - mean) * random.uniform(0.85, 1.15) + mean
            image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        if random.random() < 0.2:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.6)))
        return image, target

