"""
LoveDA Dataset + augmentation pipeline.

Layout expected:
    <root>/
        Urban/
            images_png/*.png
            masks_png/*.png
        Rural/
            images_png/*.png
            masks_png/*.png

Where <root> is typically data/loveda/Train or data/loveda/Val.

Class convention (LoveDA):
    0 = no-data  (ignored in loss via ignore_index=0)
    1 = background
    2 = building
    3 = road
    4 = water
    5 = barren
    6 = forest
    7 = agriculture

Mask PNGs store class indices directly as pixel values 0-7.
"""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

# ImageNet stats — SegFormer was pretrained on ImageNet-1k, so inputs
# must be normalized to these statistics or the backbone sees garbage.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_SIZE = 1024


def build_transforms(split: str) -> A.Compose:
    """
    Build Albumentations pipeline for a given split.

    Train: geometric augs (flip, rotate90) + photometric (color jitter)
           + resize to 1024 + normalize.
    Val:   resize to 1024 + normalize only.

    Resize uses bilinear for images, nearest for masks — masks must stay
    as integer class indices (2.7 "road" pixels would be meaningless).
    """
    if split == "train":
        return A.Compose(
            [
                # Geometric first. Resize before flips is fine either way,
                # but resizing first keeps downstream op costs predictable.
                A.Resize(
                    height=IMG_SIZE,
                    width=IMG_SIZE,
                    interpolation=1,           # cv2.INTER_LINEAR for image
                    mask_interpolation=0,      # cv2.INTER_NEAREST for mask
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),       # 0/90/180/270 — safe for aerial
                # Photometric. Mild — aerial color carries real class signal
                # (green channel distinguishes forest vs agri vs barren).
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                    p=0.5,
                ),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    elif split == "val":
        return A.Compose(
            [
                A.Resize(
                    height=IMG_SIZE,
                    width=IMG_SIZE,
                    interpolation=1,
                    mask_interpolation=0,
                ),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")


class LoveDADataset(Dataset):
    """
    LoveDA semantic segmentation dataset. Combines Urban + Rural under <root>.

    Args:
        root: path to Train/ or Val/ directory (contains Urban/ and Rural/)
        split: 'train' or 'val' — controls augmentation pipeline
        transform: optional override; if None, build_transforms(split) is used

    Returns per __getitem__:
        image: float tensor, shape (3, 1024, 1024), ImageNet-normalized
        mask:  long tensor,  shape (1024, 1024), values in {0..7}
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: A.Compose | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform if transform is not None else build_transforms(split)

        # Collect (image_path, mask_path) pairs from both Urban and Rural.
        self.samples: list[tuple[Path, Path]] = []
        for domain in ("Urban", "Rural"):
            img_dir = self.root / domain / "images_png"
            msk_dir = self.root / domain / "masks_png"
            if not img_dir.is_dir() or not msk_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing {img_dir} or {msk_dir}. "
                    f"Expected LoveDA layout under {self.root}."
                )
            for img_path in sorted(img_dir.glob("*.png")):
                msk_path = msk_dir / img_path.name
                if not msk_path.exists():
                    raise FileNotFoundError(
                        f"Image {img_path} has no matching mask at {msk_path}"
                    )
                self.samples.append((img_path, msk_path))

        if not self.samples:
            raise RuntimeError(f"No samples found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, msk_path = self.samples[idx]

        # PIL -> np.uint8 RGB for the image
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        # Mask is a single-channel PNG with class indices 0..7 as raw pixel values.
        # 'L' mode ensures we read it as a 2D uint8 array, not RGBA.
        mask = np.array(Image.open(msk_path).convert("L"), dtype=np.uint8)

        out = self.transform(image=image, mask=mask)
        image_t: torch.Tensor = out["image"]        # float, (3, H, W), normalized
        mask_t: torch.Tensor = out["mask"].long()   # long, (H, W)

        return image_t, mask_t


if __name__ == "__main__":
    # Minimal self-check: instantiate both splits, grab one sample each,
    # print shapes. Doesn't exercise DataLoader — that's test_loader.py's job.
    import sys

    root = Path("data/loveda")
    if not root.exists():
        print(f"[warn] {root} not found. Run this from the project root.", file=sys.stderr)
        sys.exit(1)

    for split_name, subdir in [("train", "Train"), ("val", "Val")]:
        ds = LoveDADataset(root / subdir, split=split_name)
        img, msk = ds[0]
        print(
            f"{split_name}: {len(ds)} samples | "
            f"image {tuple(img.shape)} {img.dtype} "
            f"[{img.min():.3f}, {img.max():.3f}] | "
            f"mask {tuple(msk.shape)} {msk.dtype} "
            f"values {sorted(torch.unique(msk).tolist())}"
        )
