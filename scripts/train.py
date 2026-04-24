"""
SegFormer-B1 fine-tune on LoveDA (Urban + Rural combined).

Phase 3 first pass. Locked decisions (see DECISIONS_LOG.md Phase 1 & 2):
  - CE loss, ignore_index=0, MFB class weights for classes 1..7
  - AdamW + cosine schedule w/ linear warmup
  - Batch 2, grad-accum 4 (effective batch 8)
  - Mixed precision (fp16) via torch.amp
  - ImageNet normalization in dataset.py (do not re-normalize here)

Usage:
    python scripts/train.py \\
        --data-root data/loveda \\
        --output-dir model/segformer-b1-run1 \\
        --epochs 15

Fallback for slow epochs (see Phase 3 gate in PHASE_PLAN.md):
    python scripts/train.py --crop-size 768 ...
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

# Make "from dataset import ..." work regardless of where train.py is launched
# from, as long as scripts/ is on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import LoveDADataset, build_transforms  # noqa: E402


# ---------------------------------------------------------------------------
# Constants locked in Phase 1. Do NOT tune these here — change the decisions
# log first, then come back.
# ---------------------------------------------------------------------------
NUM_CLASSES = 8          # 0..7 inclusive. Class 0 ignored in loss.
IGNORE_INDEX = 0
CLASS_NAMES = [
    "no_data",       # 0 — ignored
    "background",    # 1
    "building",      # 2
    "road",          # 3
    "water",         # 4
    "barren",        # 5
    "forest",        # 6
    "agriculture",   # 7
]
# MFB weights for classes 1..7, computed in Phase 1. Class 0 gets weight 0
# because it's ignored — but we pass the full 8-length tensor to CE for index
# alignment.
MFB_WEIGHTS_1_TO_7 = [0.255, 0.824, 1.730, 1.426, 1.748, 0.567, 0.451]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PRETRAINED_MODEL = "nvidia/segformer-b1-finetuned-ade-512-512"


# ---------------------------------------------------------------------------
# Transforms — only used when --crop-size != 1024. Otherwise we fall through
# to dataset.py's build_transforms() which is locked at 1024.
# ---------------------------------------------------------------------------
def build_transforms_custom_size(split: str, size: int) -> A.Compose:
    """
    Same pipeline as dataset.build_transforms() but at a custom resize size.
    Used for the --crop-size 768 fallback path.
    """
    if split == "train":
        return A.Compose([
            A.Resize(size, size, interpolation=1, mask_interpolation=0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2,
                          saturation=0.2, hue=0.05, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(size, size, interpolation=1, mask_interpolation=0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Keep cudnn benchmark ON — much faster for fixed input sizes like ours.
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# LR schedule: linear warmup then cosine decay to 0.
# ---------------------------------------------------------------------------
def make_lr_schedule(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Confusion matrix → per-class IoU. The numerically correct way to compute
# mIoU: accumulate a confusion matrix across the whole val set, then compute
# IoU from it. Averaging per-batch IoUs is WRONG — classes that don't appear
# in a batch skew the average.
# ---------------------------------------------------------------------------
class ConfusionMatrix:
    def __init__(self, num_classes: int, ignore_index: int):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.mat = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """
        pred, target: (N,) int64 tensors on CPU, flattened.
        """
        mask = target != self.ignore_index
        pred = pred[mask]
        target = target[mask]
        # bincount on (target * C + pred) gives a flat confusion matrix.
        k = target * self.num_classes + pred
        bincount = torch.bincount(k, minlength=self.num_classes ** 2)
        self.mat += bincount.view(self.num_classes, self.num_classes)

    def compute_iou(self) -> dict:
        """
        Returns dict with per-class IoU (including ignore class, which will
        be NaN since it's never counted) and mIoU stats.
        """
        mat = self.mat.float()
        tp = mat.diag()
        fp = mat.sum(dim=0) - tp   # predicted as class c but wasn't
        fn = mat.sum(dim=1) - tp   # was class c but predicted something else
        denom = tp + fp + fn
        iou = torch.where(denom > 0, tp / denom, torch.full_like(tp, float("nan")))
        # mIoU over classes 1..7 (skip ignore class 0).
        valid = iou[1:]
        valid = valid[~torch.isnan(valid)]
        miou = valid.mean().item() if len(valid) > 0 else float("nan")
        # mIoU excluding background (class 1) for the "no-bg" variant noted
        # in DECISIONS_LOG.md Phase 1 open questions.
        valid_nobg = iou[2:]
        valid_nobg = valid_nobg[~torch.isnan(valid_nobg)]
        miou_nobg = valid_nobg.mean().item() if len(valid_nobg) > 0 else float("nan")
        return {
            "per_class": iou.tolist(),   # length num_classes
            "mIoU": miou,
            "mIoU_no_bg": miou_nobg,
        }


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------
@dataclass
class EpochMetrics:
    train_loss: float
    val_loss: float
    val_miou: float
    val_miou_no_bg: float
    per_class_iou: list  # length NUM_CLASSES
    epoch_seconds: float
    lr_end: float


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scheduler,
    scaler: GradScaler,
    loss_fn,
    device,
    grad_accum: int,
    log_every: int,
    epoch: int,
    total_epochs: int,
) -> float:
    """
    One epoch of training. Returns mean train loss (over optimizer steps).
    Note: `loss.backward()` is called every micro-step; optimizer.step() only
    every `grad_accum` micro-steps. Loss is divided by grad_accum for correct
    gradient magnitude.
    """
    model.train()
    running_loss = 0.0
    running_count = 0
    micro_step = 0
    t0 = time.time()

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast(device_type="cuda", dtype=torch.float16):
            # SegFormer returns logits at input_resolution/4. We upsample
            # to the mask resolution before computing the loss — this is
            # the standard pattern for SegFormer fine-tuning.
            outputs = model(pixel_values=images)
            logits = outputs.logits  # (B, C, H/4, W/4)
            logits = nn.functional.interpolate(
                logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )
            loss = loss_fn(logits, masks) / grad_accum

        scaler.scale(loss).backward()
        micro_step += 1

        if micro_step % grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * grad_accum  # undo the division for logging
            running_count += 1

            if running_count % log_every == 0:
                elapsed = time.time() - t0
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"  [epoch {epoch}/{total_epochs}] step {running_count} "
                    f"loss={running_loss / running_count:.4f} "
                    f"lr={lr_now:.2e} elapsed={elapsed:.1f}s",
                    flush=True,
                )

    # Flush any final partial accumulation (if len(loader) not divisible
    # by grad_accum). Safer than silently dropping gradient.
    if micro_step % grad_accum != 0:
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    return running_loss / max(1, running_count)


@torch.no_grad()
def validate(model, loader: DataLoader, loss_fn, device, confmat: ConfusionMatrix):
    model.eval()
    total_loss = 0.0
    total_count = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(pixel_values=images)
            logits = outputs.logits
            logits = nn.functional.interpolate(
                logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )
            loss = loss_fn(logits, masks)

        total_loss += loss.item()
        total_count += 1

        preds = logits.argmax(dim=1)  # (B, H, W)
        confmat.update(preds.flatten().cpu(), masks.flatten().cpu())

    return total_loss / max(1, total_count)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(path: Path, model, optimizer, scheduler, scaler,
                    epoch: int, best_miou: float, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_miou": best_miou,
        "args": vars(args),
    }, path)


def prune_old_checkpoints(output_dir: Path, keep_last_n: int) -> None:
    """Keep only the N most recent epoch_*.pt files; always keep best.pt."""
    ckpts = sorted(
        output_dir.glob("epoch_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    for old in ckpts[:-keep_last_n]:
        try:
            old.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path("data/loveda"))
    p.add_argument("--output-dir", type=Path, default=Path("model/segformer-b1-run1"))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--crop-size", type=int, default=1024,
                   help="Resize size. 1024 uses dataset.build_transforms. "
                        "Anything else uses the custom pipeline in train.py.")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=50,
                   help="Log train loss every N optimizer steps.")
    p.add_argument("--keep-last-n", type=int, default=3,
                   help="Keep only the last N epoch checkpoints on disk.")
    p.add_argument("--resume", type=Path, default=None,
                   help="Path to a checkpoint to resume from.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Record run config for later reference.
    with open(args.output_dir / "run_config.json", "w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v
                   for k, v in vars(args).items()}, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA not available. Training will be unusably slow.", flush=True)
    print(f"Device: {device} "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})",
          flush=True)

    # Datasets + loaders ------------------------------------------------------
    if args.crop_size == 1024:
        train_tf = build_transforms("train")
        val_tf = build_transforms("val")
    else:
        print(f"Using custom crop size: {args.crop_size} (fallback path)", flush=True)
        train_tf = build_transforms_custom_size("train", args.crop_size)
        val_tf = build_transforms_custom_size("val", args.crop_size)

    train_ds = LoveDADataset(args.data_root / "Train", split="train", transform=train_tf)
    val_ds = LoveDADataset(args.data_root / "Val", split="val", transform=val_tf)
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    # Model -------------------------------------------------------------------
    # num_labels=8 because class 0 is a real index slot (we just ignore it in
    # the loss). Simpler than remapping labels at load time.
    print(f"Loading {PRETRAINED_MODEL} with num_labels={NUM_CLASSES} "
          f"(ignore_mismatched_sizes=True — decode head reinitialized)", flush=True)
    model = SegformerForSemanticSegmentation.from_pretrained(
        PRETRAINED_MODEL,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    ).to(device)

    # Loss --------------------------------------------------------------------
    # 8-length weight tensor: class 0 gets 0 (it's ignored anyway), 1..7 get MFB.
    weights = torch.tensor([0.0] + MFB_WEIGHTS_1_TO_7, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE_INDEX)

    # Optimizer + scheduler ---------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    print(f"Optimizer steps per epoch: {steps_per_epoch} | total: {total_steps}", flush=True)
    scheduler = make_lr_schedule(optimizer, args.warmup_steps, total_steps)

    scaler = GradScaler("cuda")

    # Optional resume ---------------------------------------------------------
    start_epoch = 1
    best_miou = -1.0
    if args.resume is not None:
        print(f"Resuming from {args.resume}", flush=True)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt.get("best_miou", -1.0)

    # CSV log -----------------------------------------------------------------
    csv_path = args.output_dir / "train_log.csv"
    write_header = not csv_path.exists()
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if write_header:
        csv_writer.writerow([
            "epoch", "train_loss", "val_loss",
            "val_mIoU", "val_mIoU_no_bg",
            "IoU_background", "IoU_building", "IoU_road", "IoU_water",
            "IoU_barren", "IoU_forest", "IoU_agriculture",
            "epoch_seconds", "lr_end",
        ])
        csv_file.flush()

    # Training loop -----------------------------------------------------------
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_t0 = time.time()
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, scaler, loss_fn,
                device, args.grad_accum, args.log_every, epoch, args.epochs,
            )

            confmat = ConfusionMatrix(NUM_CLASSES, IGNORE_INDEX)
            val_loss = validate(model, val_loader, loss_fn, device, confmat)
            iou_stats = confmat.compute_iou()
            per_class = iou_stats["per_class"]  # length 8
            epoch_seconds = time.time() - epoch_t0
            lr_end = scheduler.get_last_lr()[0]

            # Stdout summary — this is what you'll grep at the Saturday gate.
            print("", flush=True)
            print(f"=== Epoch {epoch}/{args.epochs} complete "
                  f"({epoch_seconds / 60:.1f} min) ===", flush=True)
            print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}", flush=True)
            print(f"  val_mIoU={iou_stats['mIoU']:.4f}  "
                  f"val_mIoU_no_bg={iou_stats['mIoU_no_bg']:.4f}", flush=True)
            for cls_idx in range(1, NUM_CLASSES):
                val = per_class[cls_idx]
                val_str = f"{val:.4f}" if not math.isnan(val) else "  nan "
                print(f"    IoU[{CLASS_NAMES[cls_idx]:>12}] = {val_str}", flush=True)
            print("", flush=True)

            # 45-min warning after epoch 1.
            if epoch == 1 and epoch_seconds > 45 * 60:
                print("#" * 70, flush=True)
                print(f"# WARNING: Epoch 1 took {epoch_seconds / 60:.1f} min (>45 min).", flush=True)
                print("# Per PHASE_PLAN.md, consider Ctrl+C and restart with", flush=True)
                print("#   --crop-size 768", flush=True)
                print("#" * 70, flush=True)

            # CSV row
            csv_writer.writerow([
                epoch,
                f"{train_loss:.6f}", f"{val_loss:.6f}",
                f"{iou_stats['mIoU']:.6f}", f"{iou_stats['mIoU_no_bg']:.6f}",
                *[f"{per_class[i]:.6f}" if not math.isnan(per_class[i]) else "nan"
                  for i in range(1, NUM_CLASSES)],
                f"{epoch_seconds:.1f}", f"{lr_end:.3e}",
            ])
            csv_file.flush()

            # Save this epoch's checkpoint.
            epoch_ckpt = args.output_dir / f"epoch_{epoch}.pt"
            save_checkpoint(epoch_ckpt, model, optimizer, scheduler, scaler,
                            epoch, best_miou, args)

            # Track best by val_mIoU.
            if iou_stats["mIoU"] > best_miou:
                best_miou = iou_stats["mIoU"]
                best_ckpt = args.output_dir / "best.pt"
                save_checkpoint(best_ckpt, model, optimizer, scheduler, scaler,
                                epoch, best_miou, args)
                print(f"  -> new best val_mIoU={best_miou:.4f}, saved best.pt", flush=True)

            prune_old_checkpoints(args.output_dir, args.keep_last_n)

    finally:
        csv_file.close()

    print("Training done.", flush=True)
    print(f"Best val_mIoU: {best_miou:.4f}", flush=True)
    print(f"Logs: {csv_path}", flush=True)
    print(f"Best checkpoint: {args.output_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()