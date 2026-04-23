#!/usr/bin/env python3
"""Compute per-class pixel and image-level frequency statistics for LoveDA,
broken down by split (Train/Val) and scene (Urban/Rural). Save sample
image+mask overlays for visual sanity check. Write dataset_stats.md.

Usage (from repo root, with `landuse` conda env active):
    python scripts/dataset_stats.py

Outputs:
    dataset_stats.md           # human-readable report
    data/loveda/stats.json     # raw numbers for downstream use (loss weights etc.)
    data/loveda/samples/*.png  # 8 sample image+mask overlays

Class values in LoveDA masks:
    0 = no-data      (ignored in loss; also excluded from all % calculations)
    1 = background   (real class)
    2 = building
    3 = road
    4 = water
    5 = barren
    6 = forest
    7 = agriculture
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants

CLASS_NAMES = {
    0: "no-data",
    1: "background",
    2: "building",
    3: "road",
    4: "water",
    5: "barren",
    6: "forest",
    7: "agriculture",
}

# 7 real classes (excluding no-data). Used for the histogram / weight calc.
REAL_CLASSES = list(range(1, 8))

# Fixed per-class palette for overlays. RGB, 0-255. Black reserved for no-data.
CLASS_COLORS = {
    0: (0, 0, 0),            # no-data: black
    1: (200, 200, 200),      # background: light grey
    2: (220, 20, 60),        # building: crimson
    3: (128, 128, 128),      # road: grey
    4: (30, 144, 255),       # water: dodger blue
    5: (160, 82, 45),        # barren: sienna
    6: (34, 139, 34),        # forest: forest green
    7: (255, 215, 0),        # agriculture: gold
}

# Directories we expect after unzip
SPLITS_SCENES = [
    ("Train", "Urban"),
    ("Train", "Rural"),
    ("Val", "Urban"),
    ("Val", "Rural"),
]


# ---------------------------------------------------------------------------
# Core stats

def scan_masks(mask_dir: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Walk every mask PNG in mask_dir, return:
      - pixel_counts: int64[8], total pixels per class (including 0=no-data)
      - image_presence: int64[8], # of images containing at least one pixel of class c
      - n_images: int, number of mask files scanned
    """
    pixel_counts = np.zeros(8, dtype=np.int64)
    image_presence = np.zeros(8, dtype=np.int64)
    mask_files = sorted(mask_dir.glob("*.png"))
    n_images = len(mask_files)

    for mf in tqdm(mask_files, desc=str(mask_dir.relative_to(mask_dir.parents[2])), leave=False):
        # LoveDA masks are single-channel PNGs with values 0..7. PIL opens
        # them as mode 'L' (8-bit grayscale). np.asarray gives uint8.
        m = np.asarray(Image.open(mf))
        # bincount with minlength=8 gives counts for classes 0..7
        counts = np.bincount(m.ravel(), minlength=8)
        if counts.shape[0] > 8:
            # Unexpected class value > 7. Should never happen on clean LoveDA.
            raise ValueError(f"Unexpected class values in {mf}: max={m.max()}")
        pixel_counts += counts
        # "present" = any pixel of that class
        image_presence += (counts > 0).astype(np.int64)

    return pixel_counts, image_presence, n_images


def compute_weights(pixel_counts_real: np.ndarray) -> dict[str, np.ndarray]:
    """Given pixel counts for the 7 real classes (not including no-data),
    compute candidate loss weights under two schemes:
      - inverse frequency, normalized to mean=1
      - median frequency balancing (MFB), normalized to mean=1

    Returns a dict of class_idx (1..7) -> weight. Output arrays are length 7.
    """
    freqs = pixel_counts_real / pixel_counts_real.sum()

    # Inverse frequency
    inv = 1.0 / freqs
    inv = inv / inv.mean()

    # Median frequency balancing (Eigen & Fergus 2015 / SegNet):
    #   w_c = median(freqs) / freq_c
    med = np.median(freqs)
    mfb = med / freqs
    mfb = mfb / mfb.mean()

    return {"inverse_freq": inv, "median_freq_balancing": mfb, "raw_freqs": freqs}


# ---------------------------------------------------------------------------
# Overlays

def make_overlay(image_path: Path, mask_path: Path, alpha: float = 0.5) -> np.ndarray:
    """Return an H×W×3 uint8 array: image with colored mask overlay."""
    img = np.asarray(Image.open(image_path).convert("RGB"))
    mask = np.asarray(Image.open(mask_path))
    colored = np.zeros_like(img)
    for cls, color in CLASS_COLORS.items():
        colored[mask == cls] = color
    blended = (img.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha)
    return blended.clip(0, 255).astype(np.uint8)


def save_samples(loveda_root: Path, out_dir: Path, n_per_scene: int = 2, seed: int = 42) -> list[Path]:
    """Save image+mask+overlay triptychs for 2 train urban, 2 train rural,
    2 val urban, 2 val rural = 8 total.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    saved = []

    for split, scene in SPLITS_SCENES:
        img_dir = loveda_root / split / scene / "images_png"
        mask_dir = loveda_root / split / scene / "masks_png"
        img_files = sorted(img_dir.glob("*.png"))
        picks = rng.sample(img_files, min(n_per_scene, len(img_files)))

        for img_path in picks:
            mask_path = mask_dir / img_path.name
            if not mask_path.exists():
                print(f"  WARN: no mask for {img_path.name}, skipping")
                continue

            img = np.asarray(Image.open(img_path).convert("RGB"))
            mask = np.asarray(Image.open(mask_path))
            overlay = make_overlay(img_path, mask_path)

            # Render colored mask too (not just the raw 0-7 grayscale)
            colored_mask = np.zeros_like(img)
            for cls, color in CLASS_COLORS.items():
                colored_mask[mask == cls] = color

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img)
            axes[0].set_title(f"{split}/{scene}: {img_path.name}")
            axes[0].axis("off")
            axes[1].imshow(colored_mask)
            axes[1].set_title("mask (colored by class)")
            axes[1].axis("off")
            axes[2].imshow(overlay)
            axes[2].set_title("overlay (alpha=0.5)")
            axes[2].axis("off")

            # Legend below
            from matplotlib.patches import Patch
            handles = [
                Patch(facecolor=np.array(CLASS_COLORS[c]) / 255.0, label=f"{c}: {CLASS_NAMES[c]}")
                for c in range(8)
            ]
            fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=8,
                       bbox_to_anchor=(0.5, -0.02), frameon=False)

            out_path = out_dir / f"{split.lower()}_{scene.lower()}_{img_path.stem}.png"
            fig.tight_layout()
            fig.savefig(out_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            saved.append(out_path)

    return saved


# ---------------------------------------------------------------------------
# Markdown report

def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def write_markdown(report_path: Path, results: dict, samples: list[Path], weights: dict) -> None:
    lines = []
    lines.append("# LoveDA Dataset Statistics")
    lines.append("")
    lines.append("_Auto-generated by `scripts/dataset_stats.py`._")
    lines.append("")
    lines.append("## Dataset overview")
    lines.append("")
    lines.append("LoveDA is a high-resolution (0.3 m/pixel) remote sensing semantic")
    lines.append("segmentation dataset covering Nanjing, Changzhou, and Wuhan. Images")
    lines.append("are 1024×1024 RGB PNGs; masks are single-channel PNGs with values 0–7.")
    lines.append("")
    lines.append("We use the official Train and Val splits only. The official Test split")
    lines.append("has withheld labels (LoveDA benchmark server), so we have no held-out")
    lines.append("test set; val doubles as our model-selection set.")
    lines.append("")
    lines.append("### Class values")
    lines.append("")
    lines.append("| Value | Name | Role |")
    lines.append("|-------|------|------|")
    lines.append("| 0 | no-data | **ignored** in loss and all stats below |")
    lines.append("| 1 | background | trained as a real class, excluded from emissions |")
    lines.append("| 2 | building | trained, emissions |")
    lines.append("| 3 | road | trained, emissions |")
    lines.append("| 4 | water | trained, emissions |")
    lines.append("| 5 | barren | trained, emissions |")
    lines.append("| 6 | forest | trained, emissions |")
    lines.append("| 7 | agriculture | trained, emissions |")
    lines.append("")

    # File counts
    lines.append("## Split sizes")
    lines.append("")
    lines.append("| Split | Scene | # images |")
    lines.append("|-------|-------|----------|")
    for (split, scene), r in results.items():
        lines.append(f"| {split} | {scene} | {r['n_images']} |")
    totals = {split: sum(r["n_images"] for (s, _), r in results.items() if s == split)
              for split in ("Train", "Val")}
    lines.append(f"| **Train total** |  | **{totals['Train']}** |")
    lines.append(f"| **Val total** |  | **{totals['Val']}** |")
    lines.append("")

    # Per-bucket pixel % (excluding no-data)
    lines.append("## Per-class pixel frequency")
    lines.append("")
    lines.append("Percentages exclude no-data pixels (class 0). Rows sum to 100%.")
    lines.append("")
    header = "| Split | Scene | " + " | ".join(CLASS_NAMES[c] for c in REAL_CLASSES) + " | no-data % of raw |"
    sep = "|" + "|".join(["---"] * (len(REAL_CLASSES) + 3)) + "|"
    lines.append(header)
    lines.append(sep)
    for (split, scene), r in results.items():
        pc = r["pixel_counts"]  # len 8
        real = pc[1:]
        total_labeled = real.sum()
        total_all = pc.sum()
        nodata_raw = pc[0] / total_all if total_all > 0 else 0.0
        row = f"| {split} | {scene} | " + " | ".join(pct(x / total_labeled) for x in real) \
              + f" | {pct(nodata_raw)} |"
        lines.append(row)

    # Combined train, combined val
    def combine(split: str) -> np.ndarray:
        return sum(r["pixel_counts"] for (s, _), r in results.items() if s == split)

    train_pc = combine("Train")
    val_pc = combine("Val")

    for label, pc in [("**Train (all)**", train_pc), ("**Val (all)**", val_pc)]:
        real = pc[1:]
        total_labeled = real.sum()
        total_all = pc.sum()
        nodata_raw = pc[0] / total_all if total_all > 0 else 0.0
        row = f"| {label} | | " + " | ".join(pct(x / total_labeled) for x in real) \
              + f" | {pct(nodata_raw)} |"
        lines.append(row)
    lines.append("")

    # Image-level presence
    lines.append("## Image-level class presence")
    lines.append("")
    lines.append("Fraction of images in the split/scene containing ≥1 pixel of the class.")
    lines.append("")
    header = "| Split | Scene | " + " | ".join(CLASS_NAMES[c] for c in REAL_CLASSES) + " |"
    sep = "|" + "|".join(["---"] * (len(REAL_CLASSES) + 2)) + "|"
    lines.append(header)
    lines.append(sep)
    for (split, scene), r in results.items():
        pres = r["image_presence"][1:]  # len 7
        n = r["n_images"]
        row = f"| {split} | {scene} | " + " | ".join(pct(p / n) for p in pres) + " |"
        lines.append(row)
    lines.append("")

    # Imbalance summary
    lines.append("## Class imbalance (Train, combined Urban+Rural)")
    lines.append("")
    real_train = train_pc[1:]
    freqs = real_train / real_train.sum()
    order = np.argsort(freqs)[::-1]  # most common first
    lines.append("| Rank | Class | Pixel freq | Ratio vs rarest |")
    lines.append("|------|-------|-----------|-----------------|")
    rarest = freqs.min()
    for rank, idx in enumerate(order, 1):
        cls = REAL_CLASSES[idx]
        lines.append(f"| {rank} | {CLASS_NAMES[cls]} | {pct(freqs[idx])} | {freqs[idx]/rarest:.1f}× |")
    lines.append("")
    lines.append(f"**Overall imbalance ratio (most common / rarest): "
                 f"{freqs.max() / freqs.min():.1f}×**")
    lines.append("")

    # Candidate loss weights
    lines.append("## Candidate loss weights")
    lines.append("")
    lines.append("Weights for 7 real classes (class 0 is ignored in loss). Both schemes")
    lines.append("are normalized so the mean weight = 1.0.")
    lines.append("")
    lines.append("| Class | Pixel freq | Inverse freq | Median freq balancing |")
    lines.append("|-------|-----------|--------------|----------------------|")
    for i, cls in enumerate(REAL_CLASSES):
        lines.append(f"| {CLASS_NAMES[cls]} | {pct(weights['raw_freqs'][i])} | "
                     f"{weights['inverse_freq'][i]:.3f} | "
                     f"{weights['median_freq_balancing'][i]:.3f} |")
    lines.append("")
    lines.append("**Recommendation:** median frequency balancing is the usual default for")
    lines.append("segmentation — inverse frequency overweights rare classes and can cause")
    lines.append("training instability. Start with MFB, tune from there if the rare-class")
    lines.append("IoU isn't budging.")
    lines.append("")

    # Urban vs Rural shift
    lines.append("## Domain shift: Urban vs Rural (Train)")
    lines.append("")
    urban = results[("Train", "Urban")]["pixel_counts"][1:]
    rural = results[("Train", "Rural")]["pixel_counts"][1:]
    ufreqs = urban / urban.sum()
    rfreqs = rural / rural.sum()
    lines.append("| Class | Urban freq | Rural freq | Ratio (U/R) |")
    lines.append("|-------|-----------|-----------|-------------|")
    for i, cls in enumerate(REAL_CLASSES):
        ratio = ufreqs[i] / rfreqs[i] if rfreqs[i] > 0 else float("inf")
        lines.append(f"| {CLASS_NAMES[cls]} | {pct(ufreqs[i])} | {pct(rfreqs[i])} | {ratio:.2f} |")
    lines.append("")

    # Samples
    lines.append("## Sample overlays")
    lines.append("")
    lines.append("Sanity-check renderings at `data/loveda/samples/`:")
    lines.append("")
    for p in samples:
        lines.append(f"- `{p.relative_to(report_path.parent)}`")
    lines.append("")

    # Meta
    lines.append("## Raw data")
    lines.append("")
    lines.append("Machine-readable stats at `data/loveda/stats.json`.")
    lines.append("")

    report_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loveda-root", type=Path,
                        default=Path(__file__).resolve().parent.parent / "data" / "loveda",
                        help="Path to unzipped LoveDA data (contains Train/ and Val/)")
    parser.add_argument("--out-md", type=Path,
                        default=Path(__file__).resolve().parent.parent / "dataset_stats.md",
                        help="Where to write the markdown report")
    parser.add_argument("--samples-dir", type=Path,
                        default=None,
                        help="Where to save sample overlays (default: <loveda-root>/samples)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.samples_dir is None:
        args.samples_dir = args.loveda_root / "samples"

    if not args.loveda_root.is_dir():
        raise SystemExit(f"LoveDA root not found: {args.loveda_root}\n"
                         f"Run scripts/download_loveda.sh first.")

    # Scan all mask directories
    results: dict[tuple[str, str], dict] = {}
    for split, scene in SPLITS_SCENES:
        mask_dir = args.loveda_root / split / scene / "masks_png"
        if not mask_dir.is_dir():
            raise SystemExit(f"Missing mask dir: {mask_dir}")
        print(f"Scanning {split}/{scene}...")
        pc, pres, n = scan_masks(mask_dir)
        results[(split, scene)] = {
            "pixel_counts": pc,
            "image_presence": pres,
            "n_images": n,
        }

    # Weights from combined Train set (Urban + Rural)
    train_pc = sum(r["pixel_counts"] for (s, _), r in results.items() if s == "Train")
    real_train = train_pc[1:]
    weights = compute_weights(real_train)

    # Dump raw JSON
    json_path = args.loveda_root / "stats.json"
    json_dump = {
        "buckets": {
            f"{split}_{scene}": {
                "n_images": r["n_images"],
                "pixel_counts": r["pixel_counts"].tolist(),
                "image_presence": r["image_presence"].tolist(),
            }
            for (split, scene), r in results.items()
        },
        "classes": CLASS_NAMES,
        "weights_train_combined": {
            "raw_freqs": weights["raw_freqs"].tolist(),
            "inverse_freq": weights["inverse_freq"].tolist(),
            "median_freq_balancing": weights["median_freq_balancing"].tolist(),
        },
    }
    json_path.write_text(json.dumps(json_dump, indent=2))
    print(f"Wrote {json_path}")

    # Samples
    print("Rendering sample overlays...")
    samples = save_samples(args.loveda_root, args.samples_dir, n_per_scene=2, seed=args.seed)
    print(f"Saved {len(samples)} overlays to {args.samples_dir}")

    # Markdown report
    write_markdown(args.out_md, results, samples, weights)
    print(f"Wrote {args.out_md}")

    # Terse stdout summary so you can eyeball it after the run
    print("\n=== Quick summary (Train, combined) ===")
    freqs = weights["raw_freqs"]
    for i, cls in enumerate(REAL_CLASSES):
        print(f"  {CLASS_NAMES[cls]:>12s}: {100*freqs[i]:5.2f}%  "
              f"(MFB weight {weights['median_freq_balancing'][i]:.2f})")
    print(f"  imbalance ratio (max/min): {freqs.max()/freqs.min():.1f}×")


if __name__ == "__main__":
    main()
