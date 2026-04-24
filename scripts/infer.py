#!/usr/bin/env python3
"""
infer.py — Run the trained SegFormer-B1 on an image and produce:
    (a) a colored segmentation mask PNG,
    (b) per-class pixel breakdown,
    (c) a CO2-equivalent emissions estimate (annual + embodied) with citations.

Design notes
------------

1. Checkpoint loading requires weights_only=False because train.py stored
   vars(args) inside the checkpoint, which contains pathlib.PosixPath objects.
   Torch 2.6+ defaults weights_only=True and cannot unpickle those. This
   gotcha is documented in DECISIONS_LOG.md Phase 3 and CURRENT_STATE.md.

2. The model expects 1024x1024 ImageNet-normalized input. Non-1024 inputs are
   resized with a warning. Nearest-neighbor is used on the OUTPUT mask when
   resizing back, never on the input (bilinear on input, nearest on output).

3. TTA (test-time augmentation) averages logits over the original + horizontal
   flip + vertical flip (3 forward passes). Per PHASE_PLAN.md and Phase 3
   learnings, this is a cheap +0.01 to +0.03 mIoU freebie. On by default;
   --no-tta disables.

4. The colored mask uses an intuitive palette. LoveDA has no official palette
   in the upstream repo or in torchgeo — both use matplotlib's default colormap
   for visualization. Confirmed via manual check of both sources.

5. SegFormer outputs logits at 1/4 resolution. transformers' model adds no
   upsampling in the forward pass — we upsample here with bilinear.

Usage
-----
    python scripts/infer.py --image demo.png
    python scripts/infer.py --image demo.png --checkpoint model/segformer-b1-run1/best.pt
    python scripts/infer.py --image demo.png --no-tta --out-dir results/
    python scripts/infer.py --image demo.png --overlay  # save image+mask composite too
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation

# emissions.py sits next to this file in scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from emissions import (  # noqa: E402
    CLASS_NAMES,
    EMISSIONS_CLASSES,
    LAND_USE_EMISSIONS,
    SOURCES,
    compute_emissions,
)


# -----------------------------------------------------------------------------
# CONSTANTS (locked in earlier phases — do not change without a decision log)
# -----------------------------------------------------------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TRAIN_INPUT_SIZE = 1024          # Phase 2 locked
LOVEDA_PIXEL_SIZE_M = 0.3        # LoveDA GSD; 0.09 m²/pixel
NUM_CLASSES = 8                  # 0=no-data, 1=background, 2-7=real classes

# Intuitive RGB palette (no official LoveDA palette exists — verified Phase 5).
# Chosen to match common land-cover conventions: water=blue, forest=green,
# road=dark-grey, building=red, agri=yellow, barren=tan, background=light-grey.
PALETTE: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),          # no_data       → black (should be near-zero in practice)
    1: (200, 200, 200),    # background    → light grey (heterogeneous / excluded)
    2: (220, 20, 60),      # building      → crimson red
    3: (64, 64, 64),       # road          → dark grey
    4: (30, 144, 255),     # water         → dodger blue
    5: (210, 180, 140),    # barren        → tan
    6: (34, 139, 34),      # forest        → forest green
    7: (255, 215, 0),      # agriculture   → gold / yellow
}


# -----------------------------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------------------------
def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> SegformerForSemanticSegmentation:
    """
    Load SegFormer-B1 with the same architecture used in train.py and restore
    our trained weights from best.pt.

    weights_only=False is REQUIRED because train.py's save_checkpoint stored
    vars(args) which contains pathlib.PosixPath. Torch 2.6+ cannot unpickle
    those under the default weights_only=True. See DECISIONS_LOG Phase 3.
    """
    print(f"[load] reading checkpoint: {checkpoint_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    # Build the same architecture as training: ADE20K-pretrained backbone,
    # decode head reinitialized for 8 classes.
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b1-finetuned-ade-512-512",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )

    # Load our trained weights. weights_only=False per the comment above.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Our checkpoint format (from train.py) is a dict with keys including
    # 'model_state_dict' (the nn.Module weights) and 'args' (training CLI).
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        epoch = ckpt.get("epoch", "?")
        best_miou = ckpt.get("best_val_miou", ckpt.get("val_miou", "?"))
        print(f"[load] checkpoint from epoch {epoch}, val_mIoU={best_miou}")
    if "model" in ckpt:
       state = ckpt["model"]
    else:
        # Fallback: bare state dict.
        state = ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load] WARNING: {len(missing)} missing keys (showing first 3): "
              f"{missing[:3]}")
    if unexpected:
        print(f"[load] WARNING: {len(unexpected)} unexpected keys (showing first 3): "
              f"{unexpected[:3]}")

    model.eval()
    model.to(device)
    return model


# -----------------------------------------------------------------------------
# PREPROCESSING
# -----------------------------------------------------------------------------
def preprocess_image(image_path: Path) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Load an image, resize to 1024x1024 (bilinear), normalize with ImageNet stats,
    return as a CHW float tensor with batch dim added, plus the original (H, W)
    for later mask resizing.

    Matches the val-split transforms from dataset.py exactly.
    """
    img = Image.open(image_path).convert("RGB")
    original_size = img.size[::-1]  # PIL gives (W, H); we want (H, W)

    if original_size != (TRAIN_INPUT_SIZE, TRAIN_INPUT_SIZE):
        warnings.warn(
            f"Input image is {original_size}, model trained at "
            f"{TRAIN_INPUT_SIZE}x{TRAIN_INPUT_SIZE}. "
            f"Resizing. Accuracy may degrade on imagery with substantially "
            f"different ground sampling distance (LoveDA is 0.3m/px).",
            RuntimeWarning,
        )
        img = img.resize(
            (TRAIN_INPUT_SIZE, TRAIN_INPUT_SIZE), Image.Resampling.BILINEAR
        )

    arr = np.asarray(img, dtype=np.float32) / 255.0       # HWC, [0,1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD             # normalize
    tensor = torch.from_numpy(arr).permute(2, 0, 1)        # CHW
    tensor = tensor.unsqueeze(0)                           # add batch dim
    return tensor, original_size


# -----------------------------------------------------------------------------
# INFERENCE (with optional TTA)
# -----------------------------------------------------------------------------
@torch.inference_mode()
def predict_logits(
    model: SegformerForSemanticSegmentation,
    image: torch.Tensor,
    device: torch.device,
    use_tta: bool,
) -> torch.Tensor:
    """
    Run the model on `image` and return per-pixel logits at full resolution
    (B, NUM_CLASSES, H, W). If use_tta, averages logits over original +
    horizontal-flip + vertical-flip (3 forward passes).

    SegFormer outputs logits at 1/4 resolution; we upsample bilinearly to
    match input size.
    """
    image = image.to(device)
    H, W = image.shape[-2:]

    def _forward(x: torch.Tensor) -> torch.Tensor:
        out = model(pixel_values=x).logits              # (B, C, H/4, W/4)
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out

    logits = _forward(image)

    if use_tta:
        # hflip
        flipped_h = torch.flip(image, dims=[3])
        logits_h = _forward(flipped_h)
        logits_h = torch.flip(logits_h, dims=[3])       # unflip
        # vflip
        flipped_v = torch.flip(image, dims=[2])
        logits_v = _forward(flipped_v)
        logits_v = torch.flip(logits_v, dims=[2])
        logits = (logits + logits_h + logits_v) / 3.0

    return logits


# -----------------------------------------------------------------------------
# OUTPUT
# -----------------------------------------------------------------------------
def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Map an (H, W) uint8 class-index mask to an (H, W, 3) uint8 RGB image
    using PALETTE.
    """
    H, W = mask.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for class_id, color in PALETTE.items():
        rgb[mask == class_id] = color
    return rgb


def save_outputs(
    mask: np.ndarray,
    original_image_path: Path,
    out_dir: Path,
    make_overlay: bool,
) -> dict[str, Path]:
    """Save the raw mask PNG, the colored mask PNG, and optionally an overlay."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = original_image_path.stem
    paths: dict[str, Path] = {}

    # Raw class-index mask (single channel, values 0-7).
    raw_path = out_dir / f"{stem}_mask_raw.png"
    Image.fromarray(mask.astype(np.uint8)).save(raw_path)
    paths["raw_mask"] = raw_path

    # Colored mask.
    color_rgb = colorize_mask(mask)
    color_path = out_dir / f"{stem}_mask_colored.png"
    Image.fromarray(color_rgb).save(color_path)
    paths["colored_mask"] = color_path

    if make_overlay:
        # 50/50 blend of original (resized to match mask) and colored mask.
        orig = Image.open(original_image_path).convert("RGB")
        if orig.size != (mask.shape[1], mask.shape[0]):
            orig = orig.resize((mask.shape[1], mask.shape[0]), Image.Resampling.BILINEAR)
        overlay = Image.blend(orig, Image.fromarray(color_rgb), alpha=0.5)
        overlay_path = out_dir / f"{stem}_overlay.png"
        overlay.save(overlay_path)
        paths["overlay"] = overlay_path

    return paths


# -----------------------------------------------------------------------------
# REPORTING
# -----------------------------------------------------------------------------
def print_report(
    mask: np.ndarray,
    pixel_area_m2: float,
    tta_used: bool,
    inference_seconds: float,
) -> dict:
    """
    Print a human-readable report and return a dict suitable for json.dump.
    The dict is the same shape Weekend 2's agent tools will consume.
    """
    total_pixels = int(mask.size)
    unique, counts = np.unique(mask, return_counts=True)
    class_counts = {int(u): int(c) for u, c in zip(unique, counts)}

    emissions = compute_emissions(
        class_pixel_counts=class_counts,
        total_pixels=total_pixels,
        pixel_area_m2=pixel_area_m2,
    )

    # ----- stdout report -----
    print()
    print("=" * 70)
    print(f"  Land-cover classification")
    print(f"  inference: {inference_seconds:.2f}s"
          f"{' (TTA: 3x passes)' if tta_used else ''}")
    print(f"  total area: {emissions.total_area_ha:.3f} ha "
          f"({total_pixels:,} px @ {pixel_area_m2:.4f} m²/px)")
    print("=" * 70)
    print()
    print(f"{'class':<14} {'pixels %':>10} {'area (ha)':>12}")
    print("-" * 40)
    # sort classes by pixel count desc for readability
    ordered = sorted(class_counts.items(), key=lambda kv: -kv[1])
    for cid, n in ordered:
        name = CLASS_NAMES.get(cid, f"unknown_{cid}")
        pct = 100.0 * n / total_pixels
        area_ha = n * pixel_area_m2 / 10_000.0
        marker = "" if cid in EMISSIONS_CLASSES else "  (excluded)"
        print(f"{name:<14} {pct:>9.2f}% {area_ha:>12.4f}{marker}")
    print()

    if emissions.excluded_fraction > 0:
        print(
            f"  note: {emissions.excluded_fraction*100:.1f}% of pixels excluded "
            f"from emissions (no_data + background)"
        )
        print()

    print(f"{'class':<14} {'annual tCO2e/yr':>18} {'embodied tCO2e':>18}")
    print("-" * 56)
    for name, stats in emissions.per_class.items():
        f = LAND_USE_EMISSIONS[name]
        print(
            f"{name:<14} "
            f"{stats['annual_tco2e']:>+17.3f}  "
            f"{stats['embodied_tco2e']:>+17.1f}  "
            f"[{f.annual_source}]"
        )
    print("-" * 56)
    print(
        f"{'TOTAL':<14} "
        f"{emissions.total_annual_tco2e_per_yr:>+17.3f}  "
        f"{emissions.total_embodied_tco2e:>+17.1f}"
    )
    print()
    sign = "net source" if emissions.total_annual_tco2e_per_yr > 0 else "net sink"
    print(f"  annual: this parcel is a {sign} of "
          f"{abs(emissions.total_annual_tco2e_per_yr):.2f} tCO2e/yr")
    print(f"  embodied: this parcel stores / has consumed "
          f"{emissions.total_embodied_tco2e:.1f} tCO2e in standing stock")
    print()
    print("Sources:")
    # Only print sources actually used
    used_srcs = set()
    for name in emissions.per_class:
        f = LAND_USE_EMISSIONS[name]
        used_srcs.add(f.annual_source)
        used_srcs.add(f.embodied_source)
    for src in sorted(used_srcs):
        print(f"  [{src}] {SOURCES[src]}")
    print()

    # ----- machine-readable dict (for agent / JSON dump) -----
    return {
        "total_area_ha": emissions.total_area_ha,
        "assessed_area_ha": emissions.assessed_area_ha,
        "excluded_fraction": emissions.excluded_fraction,
        "excluded_breakdown": emissions.excluded_breakdown,
        "per_class": emissions.per_class,
        "total_annual_tco2e_per_yr": emissions.total_annual_tco2e_per_yr,
        "total_embodied_tco2e": emissions.total_embodied_tco2e,
        "tta_used": tta_used,
        "inference_seconds": inference_seconds,
        "pixel_area_m2": pixel_area_m2,
    }


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", type=Path, required=True, help="path to input image")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("model/segformer-b1-run1/best.pt"),
        help="path to trained checkpoint (default: model/segformer-b1-run1/best.pt)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("inference_outputs"),
        help="directory to save mask PNGs (default: inference_outputs/)",
    )
    p.add_argument(
        "--no-tta",
        dest="tta",
        action="store_false",
        help="disable test-time augmentation (default: TTA on)",
    )
    p.add_argument(
        "--overlay",
        action="store_true",
        help="also save a 50/50 overlay of image + colored mask",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="optional path to write the per-class + emissions breakdown as JSON",
    )
    p.add_argument(
        "--pixel-size-m",
        type=float,
        default=LOVEDA_PIXEL_SIZE_M,
        help=f"ground sample distance in meters (default: {LOVEDA_PIXEL_SIZE_M} for LoveDA)",
    )
    p.set_defaults(tta=True)
    args = p.parse_args()

    if not args.image.exists():
        print(f"ERROR: image not found: {args.image}", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device: {device}, TTA: {args.tta}")

    # --- load ---
    model = load_model(args.checkpoint, device)

    # --- preprocess ---
    image_tensor, original_size = preprocess_image(args.image)
    print(f"[preprocess] original size (H,W): {original_size}, "
          f"model input: {TRAIN_INPUT_SIZE}x{TRAIN_INPUT_SIZE}")

    # --- infer ---
    t0 = time.time()
    logits = predict_logits(model, image_tensor, device, use_tta=args.tta)
    mask_at_model_res = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    elapsed = time.time() - t0
    print(f"[infer] done in {elapsed:.2f}s")

    # --- resize mask back to original if needed (nearest-neighbor) ---
    if mask_at_model_res.shape != original_size:
        mask_img = Image.fromarray(mask_at_model_res)
        mask_img = mask_img.resize(
            (original_size[1], original_size[0]), Image.Resampling.NEAREST
        )
        mask = np.asarray(mask_img, dtype=np.uint8)
    else:
        mask = mask_at_model_res

    # --- save PNGs ---
    pixel_area_m2 = args.pixel_size_m ** 2
    output_paths = save_outputs(
        mask=mask,
        original_image_path=args.image,
        out_dir=args.out_dir,
        make_overlay=args.overlay,
    )
    for label, path in output_paths.items():
        print(f"[save] {label}: {path}")

    # --- report ---
    report = print_report(
        mask=mask,
        pixel_area_m2=pixel_area_m2,
        tta_used=args.tta,
        inference_seconds=elapsed,
    )

    # --- optional JSON dump for Weekend 2 integration ---
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[save] json report: {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())