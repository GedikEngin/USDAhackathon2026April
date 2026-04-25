"""
Convert raw CDL categorical geotiffs into binary corn-mask geotiffs at a
locked 30 m / EPSG:5070 schema, ready for HLS chip masking in Phase 2-D.1.

Input  (from `download_cdl.py`):
    phase2/cdl/raw/cdl_<state_alpha>_<year>.tif    (categorical CDL, 1+ classes)

Output:
    phase2/cdl/cdl_corn_mask_<state_alpha>_<year>.tif   (uint8 binary, EPSG:5070, 30m)

The conversion does three things:

  1. Build a binary mask:  mask = (cdl == 1)   (corn class is value 1)
  2. Reproject to EPSG:5070 if the source is in a different CRS.
  3. Resample to 30 m if the source is finer (2024 native is 10 m;
     historical CDL is 30 m). Mode/majority resampling preserves the binary
     semantics; nearest-neighbor is the alternative but mode-on-binary is
     the same answer when downsampling 3:1, so we use nearest for speed.

Phase 2-D.1 sub-phase: D.1.a (CDL prep), step 2.

Usage:
    python scripts/cdl_to_corn_mask.py
    python scripts/cdl_to_corn_mask.py --in phase2/cdl/raw --out phase2/cdl

Notes
-----
- CDL "Corn" class value = 1. (Reference: USDA NASS CDL legend, stable across
  years.) Sweet corn is class 12, popcorn is class 13 — NOT included in the
  mask. Field-corn-for-grain only matches our NASS yield target.
- Output is uint8, single-band, LZW-compressed, tiled 256x256 — matches the
  conventions Phase A's gSSURGO outputs use for downstream zonal stats / chip
  extraction.
- After running, a QC table is printed: for each (state, year), corn pixel
  count, total in-state pixel count, and corn fraction. Drift across years
  is normal (rotation); a row showing 0% is a bug worth investigating.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CORN_CLASS_VALUE = 1  # USDA CDL "Corn"

TARGET_CRS = "EPSG:5070"  # CONUS Albers Equal Area; matches gSSURGO + HLS reproj
TARGET_RESOLUTION_M = 30.0  # Match historical CDL + HLS native resolution

# Filename pattern for raw CDL: cdl_<STATE>_<YEAR>.tif
RAW_FILENAME_RE = re.compile(r"^cdl_([A-Z]{2})_(\d{4})\.tif$")

DEFAULT_IN_DIR = "phase2/cdl/raw"
DEFAULT_OUT_DIR = "phase2/cdl"


# -----------------------------------------------------------------------------
# Argparse
# -----------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert raw CDL geotiffs to binary corn masks (D.1.a)."
    )
    p.add_argument("--in", dest="in_dir", default=DEFAULT_IN_DIR)
    p.add_argument("--out", dest="out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-build masks even if they already exist on disk.",
    )
    return p


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------


def _iter_raw_tiffs(in_dir: Path) -> Iterator[Tuple[Path, str, int]]:
    """Yield (path, state_alpha, year) for each well-named raw CDL geotiff."""
    for path in sorted(in_dir.glob("cdl_*.tif")):
        m = RAW_FILENAME_RE.match(path.name)
        if not m:
            print(f"  skip (unrecognized name): {path.name}", file=sys.stderr)
            continue
        state_alpha, year = m.group(1), int(m.group(2))
        yield path, state_alpha, year


# -----------------------------------------------------------------------------
# Core conversion
# -----------------------------------------------------------------------------


def _convert_one(
    raw_path: Path,
    out_path: Path,
) -> Tuple[int, int]:
    """Read raw CDL, threshold to corn-binary, reproject/resample to target,
    write uint8 LZW-compressed geotiff. Returns (corn_pixels, total_pixels).

    Reprojection-and-thresholding ordering:
      - Threshold FIRST in source CRS (cheap; one numpy op).
      - Then reproject the binary raster (cheap; one rasterio.warp call).
      Doing it the other way (reproject categorical, then threshold) would
      need NN resampling to preserve class codes during reprojection, which
      is correct but produces a transient categorical raster we don't need.
    """
    with rasterio.open(raw_path) as src:
        src_data = src.read(1)
        src_crs = src.crs
        src_transform = src.transform
        src_width = src.width
        src_height = src.height
        src_nodata = src.nodata

    # Build binary mask in source space.
    mask = (src_data == CORN_CLASS_VALUE).astype(np.uint8)

    # Compute target transform/dims for reproj+resample to 30m EPSG:5070.
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs,
        TARGET_CRS,
        src_width,
        src_height,
        *_bounds_from(src_transform, src_width, src_height),
        resolution=TARGET_RESOLUTION_M,
    )

    dst_data = np.zeros((dst_height, dst_width), dtype=np.uint8)
    reproject(
        source=mask,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=TARGET_CRS,
        resampling=Resampling.nearest,
        src_nodata=src_nodata,
        dst_nodata=0,
    )

    profile = {
        "driver": "GTiff",
        "height": dst_height,
        "width": dst_width,
        "count": 1,
        "dtype": "uint8",
        "crs": TARGET_CRS,
        "transform": dst_transform,
        "nodata": 0,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(dst_data, 1)

    corn_px = int(dst_data.sum())
    total_px = int(dst_data.size)
    return corn_px, total_px


def _bounds_from(transform, width: int, height: int) -> Tuple[float, float, float, float]:
    """Compute (left, bottom, right, top) in CRS units from transform + dims."""
    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)
    return left, bottom, right, top


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> int:
    args = _build_argparser().parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.is_dir():
        print(f"Input dir does not exist: {in_dir}", file=sys.stderr)
        return 1

    targets = list(_iter_raw_tiffs(in_dir))
    if not targets:
        print(f"No cdl_*.tif files found in {in_dir}", file=sys.stderr)
        return 1

    print(f"CDL → corn mask conversion — {len(targets)} files")
    print(f"  in:  {in_dir.resolve()}")
    print(f"  out: {out_dir.resolve()}")
    print()

    n_built = 0
    n_skipped = 0
    n_failed = 0
    qc_rows: list[Tuple[str, int, int, int, float]] = []

    for raw_path, state_alpha, year in targets:
        out_path = out_dir / f"cdl_corn_mask_{state_alpha}_{year}.tif"
        prefix = f"{state_alpha} {year}"

        if out_path.exists() and not args.force:
            with rasterio.open(out_path) as ds:
                arr = ds.read(1)
            corn_px = int(arr.sum())
            total_px = int(arr.size)
            qc_rows.append((state_alpha, year, corn_px, total_px,
                            100.0 * corn_px / max(total_px, 1)))
            n_skipped += 1
            print(f"  {prefix}  skip (exists)")
            continue

        try:
            corn_px, total_px = _convert_one(raw_path, out_path)
            n_built += 1
            qc_rows.append(
                (state_alpha, year, corn_px, total_px,
                 100.0 * corn_px / max(total_px, 1))
            )
            print(
                f"  {prefix}  built ({corn_px:,} corn px / "
                f"{total_px:,} total = "
                f"{100.0 * corn_px / max(total_px, 1):.2f}%)"
            )
        except Exception as e:
            n_failed += 1
            print(f"  {prefix}  FAIL: {e}", file=sys.stderr)

    print()
    print("Summary:")
    print(f"  built:   {n_built}")
    print(f"  skipped: {n_skipped}")
    print(f"  failed:  {n_failed}")

    # QC table — corn fraction by state-year. Sorted state-then-year so it's
    # easy to scan for missing rows or year-over-year anomalies.
    if qc_rows:
        print()
        print("QC table — corn pixel fraction by state and year:")
        print(f"  {'state':<6} {'year':<6} {'corn_px':>14} "
              f"{'total_px':>14} {'corn_pct':>9}")
        for st, yr, corn, total, pct in sorted(qc_rows):
            print(f"  {st:<6} {yr:<6} {corn:>14,} {total:>14,} {pct:>8.2f}%")

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
