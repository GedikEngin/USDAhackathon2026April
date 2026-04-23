"""
Phase 2 deliverable: smoke-test the LoveDA DataLoader pipeline.

Builds train and val DataLoaders, pulls 20 batches from each, asserts:
  - image shape (B, 3, 1024, 1024), float
  - mask shape  (B, 1024, 1024), long, values in {0..7}
  - normalization is actually applied (values NOT in [0, 1])

Prints timing. Exits 0 if everything is clean, 1 otherwise.

Run from project root:
    python scripts/test_loader.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Make `scripts/` importable when run as `python scripts/test_loader.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import LoveDADataset  # noqa: E402

N_ITERS = 20
BATCH_SIZE = 2           # matches Phase 3 locked config
NUM_WORKERS = 2          # exercise multiprocessing now to catch pickling bugs
EXPECTED_H = EXPECTED_W = 1024
VALID_CLASSES = set(range(8))  # 0..7 inclusive


def check_batch(images: torch.Tensor, masks: torch.Tensor, batch_idx: int, split: str) -> None:
    """Raise AssertionError with a descriptive message if anything is off."""
    assert images.ndim == 4, f"[{split} b{batch_idx}] image ndim={images.ndim}, want 4"
    assert images.shape[1:] == (3, EXPECTED_H, EXPECTED_W), (
        f"[{split} b{batch_idx}] image shape {tuple(images.shape)}, "
        f"want (_, 3, {EXPECTED_H}, {EXPECTED_W})"
    )
    assert images.dtype == torch.float32, (
        f"[{split} b{batch_idx}] image dtype {images.dtype}, want float32"
    )

    assert masks.ndim == 3, f"[{split} b{batch_idx}] mask ndim={masks.ndim}, want 3"
    assert masks.shape[1:] == (EXPECTED_H, EXPECTED_W), (
        f"[{split} b{batch_idx}] mask shape {tuple(masks.shape)}, "
        f"want (_, {EXPECTED_H}, {EXPECTED_W})"
    )
    assert masks.dtype == torch.long, (
        f"[{split} b{batch_idx}] mask dtype {masks.dtype}, want long"
    )

    # Mask values must all be valid class indices
    unique = set(torch.unique(masks).tolist())
    bad = unique - VALID_CLASSES
    assert not bad, f"[{split} b{batch_idx}] mask has invalid class values: {bad}"

    # Normalization sanity: ImageNet-normalized data will have values well
    # outside [0, 1] — roughly [-2.1, 2.6]. If we see values stuck in [0, 1]
    # we forgot to normalize.
    img_min, img_max = images.min().item(), images.max().item()
    assert img_min < 0.0, (
        f"[{split} b{batch_idx}] image min={img_min:.3f} >= 0; normalization likely missing"
    )
    assert img_max > 1.0 or img_min < -0.01, (
        f"[{split} b{batch_idx}] image range [{img_min:.3f}, {img_max:.3f}] "
        f"looks un-normalized"
    )


def run_split(root: Path, split: str, subdir: str) -> None:
    print(f"\n=== {split.upper()} ({subdir}) ===")
    t0 = time.time()
    ds = LoveDADataset(root / subdir, split=split)
    print(f"  dataset built: {len(ds)} samples in {time.time() - t0:.2f}s")

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=(split == "train"),
        persistent_workers=False,
    )

    seen_classes: set[int] = set()
    t_start = time.time()
    t_first_batch = None
    for i, (images, masks) in enumerate(loader):
        if i == 0:
            t_first_batch = time.time() - t_start
        check_batch(images, masks, batch_idx=i, split=split)
        seen_classes.update(torch.unique(masks).tolist())
        if i + 1 >= N_ITERS:
            break
    elapsed = time.time() - t_start

    print(f"  first batch: {t_first_batch:.2f}s (includes worker spawn)")
    print(f"  {N_ITERS} batches total: {elapsed:.2f}s ({elapsed / N_ITERS:.2f}s/batch avg)")
    print(f"  classes seen across {N_ITERS} batches: {sorted(seen_classes)}")
    # Report one sample's stats to confirm normalization visually
    print(
        f"  sample batch stats: image range "
        f"[{images.min().item():.3f}, {images.max().item():.3f}], "
        f"mask classes {sorted(torch.unique(masks).tolist())}"
    )


def main() -> int:
    root = Path("data/loveda")
    if not root.exists():
        print(f"ERROR: {root} not found. Run from project root.", file=sys.stderr)
        return 1

    try:
        run_split(root, split="train", subdir="Train")
        run_split(root, split="val", subdir="Val")
    except AssertionError as e:
        print(f"\nFAIL: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print("\nPhase 2 smoke test: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
