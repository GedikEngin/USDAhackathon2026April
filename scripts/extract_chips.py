"""
scripts/extract_chips.py
Per-granule chip extraction for Phase 2-D.1.b/c.

Pure library, no CLI orchestration. Imported by scripts/download_hls.py
inside the per-granule loop. Can also be invoked standalone for
diagnostic re-runs against a cached granule directory.

Public API:
    extract_chips_for_granule(granule_meta, state_alpha, year,
                              granule_local_dir, chip_root,
                              counties_gdf=None, cdl_mask_path=None,
                              ...) -> list[ChipIndexRow]

For each (granule, county) intersection:
  1. Resolve which 5-state counties intersect the granule footprint.
  2. For each intersecting county:
     a. Window-read the 6 Prithvi bands + Fmask through the county's bbox
        in granule UTM, padded by a 224-pixel apron so the sliding-window
        scoring has room.
     b. Reproject the year-matched CDL corn mask onto the same grid via
        WarpedVRT (nearest-neighbor; CDL is categorical at 30 m, HLS is
        also at 30 m, so this is essentially a same-resolution warp).
     c. Build valid_pixel_mask = ~fmask_bad AND county_polygon_inside.
     d. Score every 224x224 footprint by sum(cdl_corn AND valid_pixel),
        via an integral image (O(H*W)).
     e. If best_score / 224^2 >= min_corn_frac: write the chip GeoTIFF
        (int16, 6 bands, DEFLATE+predictor=2) and emit a positive
        ChipIndexRow. Else emit a negative row with skip_reason.
  3. If zero counties intersect: emit one sentinel row.
  4. If granule download is unreadable / missing bands: emit one
     sentinel row with skip_reason="all_cloud" (best-fit reason from
     the existing enum; treat unreadable granules same as fully-clouded).

Locked design choices (decisions log entry 2-D.1.kickoff):
  - 224x224 chip size (Prithvi-EO-2.0 expected input)
  - 6 bands in PRITHVI_BAND_ORDER (blue, green, red, nir_narrow, swir1, swir2)
  - 5% min corn-pixel fraction
  - Output as int16 reflectance (raw HLS DN, scale factor 0.0001 NOT
    pre-applied; Prithvi inference applies its own normalization)
  - DEFLATE + predictor=2 compression for ~30-50% smaller files than LZW
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import glob
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import rasterio.windows
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

# Make `forecast/` importable when imported from scripts/
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from forecast.hls_common import (  # noqa: E402
    CHIP_ROOT_DEFAULT,
    ChipIndexRow,
    PRITHVI_BAND_ORDER,
    SENTINEL_GEOID_NO_INTERSECT,
    band_codes_for,
    chip_relpath,
    fmask_valid_mask,
    parse_granule_id,
)

log = logging.getLogger(__name__)


# =============================================================================
# Module-level config
# =============================================================================

# Chip pixel size, locked.
CHIP_SIZE_DEFAULT: int = 224

# Min fraction of CDL-corn pixels in the chip after Fmask masking, below
# which the chip is rejected. Locked at 5% in decisions log.
MIN_CORN_FRAC_DEFAULT: float = 0.05

# Default GeoPackage path for the 5-state county polygons (EPSG:5070).
# Phase A artifact; not regenerated.
COUNTIES_GPKG_DEFAULT: str = "phase2/data/tiger/tl_2018_us_county_5states_5070.gpkg"

# Default CDL corn mask directory (D.1.a artifacts).
CDL_MASK_DIR_DEFAULT: str = "phase2/cdl"

# GeoTIFF write profile for chips. DEFLATE + predictor=2 is materially
# smaller than LZW on int16 reflectance; tested standard.
CHIP_TIFF_PROFILE = {
    "driver": "GTiff",
    "dtype":  "int16",
    "count":  6,
    "height": CHIP_SIZE_DEFAULT,
    "width":  CHIP_SIZE_DEFAULT,
    "compress": "DEFLATE",
    "predictor": 2,
    "tiled": True,
    "blockxsize": 224,
    "blockysize": 224,
    "BIGTIFF": "IF_SAFER",
}


# =============================================================================
# Counties cache (read once per process, reused across granules)
# =============================================================================
# The county GeoPackage is small (~1 MB, 443 features) but reading it every
# granule is wasteful. Cache by path.

_COUNTIES_CACHE: dict[str, "object"] = {}


def load_counties(
    gpkg_path: str = COUNTIES_GPKG_DEFAULT,
    state_alpha: Optional[str] = None,
):
    """Return GeoDataFrame of TIGER 2018 5-state counties in EPSG:5070,
    optionally filtered to a single state. Cached across calls."""
    import geopandas as gpd

    cache_key = f"{gpkg_path}::{state_alpha or 'ALL'}"
    if cache_key in _COUNTIES_CACHE:
        return _COUNTIES_CACHE[cache_key]

    log.debug("loading county polygons from %s", gpkg_path)
    gdf = gpd.read_file(gpkg_path)
    # Sanity: project authoritative CRS
    if gdf.crs is None or gdf.crs.to_epsg() != 5070:
        log.warning("counties CRS unexpected: %s; reprojecting to EPSG:5070", gdf.crs)
        gdf = gdf.to_crs("EPSG:5070")

    # Normalize GEOID to 5-char zero-padded string
    if "GEOID" not in gdf.columns:
        raise KeyError(f"counties gpkg missing GEOID column; have {list(gdf.columns)}")
    gdf["GEOID"] = gdf["GEOID"].astype(str).str.zfill(5)

    # State filter via GEOID prefix (state FIPS = first 2 chars of GEOID).
    if state_alpha is not None:
        from forecast.hls_common import STATE_ALPHA_TO_FIPS
        fips = STATE_ALPHA_TO_FIPS[state_alpha]
        gdf = gdf[gdf["GEOID"].str.startswith(fips)].reset_index(drop=True)

    _COUNTIES_CACHE[cache_key] = gdf
    return gdf


# =============================================================================
# Granule local-file resolution
# =============================================================================

def _resolve_band_path(granule_dir: Path, granule_id: str, band_code: str) -> Optional[Path]:
    """Find the local GeoTIFF asset for a given band code (e.g. 'B04', 'Fmask')
    inside an earthaccess-downloaded granule directory.

    earthaccess.download writes each band as a separate .tif with a name like:
        HLS.L30.T15TVH.2018196T172224.v2.0.B04.tif
        HLS.S30.T14TQK.2018213T172049.v2.0.Fmask.tif

    Returns Path or None if the band file isn't present (which signals an
    incomplete download)."""
    # Try the canonical name first
    candidate = granule_dir / f"{granule_id}.{band_code}.tif"
    if candidate.exists():
        return candidate
    # Fallback: glob in case earthaccess put files in a subdirectory or the
    # name format shifts in a future release.
    matches = list(granule_dir.glob(f"**/{granule_id}.{band_code}.tif"))
    if matches:
        return matches[0]
    matches = list(granule_dir.glob(f"**/*.{band_code}.tif"))
    if matches:
        return matches[0]
    return None


def _list_granule_files(granule_dir: Path) -> list[Path]:
    """All .tif files under the granule directory. Used for debug logging
    when band resolution fails."""
    return sorted(granule_dir.rglob("*.tif"))


# =============================================================================
# Window math
# =============================================================================

def _window_for_polygon_with_apron(
    src: rasterio.io.DatasetReader,
    polygon_bounds_in_src_crs: tuple[float, float, float, float],
    apron_pixels: int,
) -> rasterio.windows.Window:
    """Compute a rasterio window covering the polygon's bbox plus an apron
    of `apron_pixels` on each side, clipped to the source's footprint.

    polygon_bounds_in_src_crs is (minx, miny, maxx, maxy) in src.crs.
    """
    minx, miny, maxx, maxy = polygon_bounds_in_src_crs
    win = rasterio.windows.from_bounds(
        minx, miny, maxx, maxy,
        transform=src.transform,
    )
    # Round outward and apply the apron
    col_off = int(np.floor(win.col_off)) - apron_pixels
    row_off = int(np.floor(win.row_off)) - apron_pixels
    width   = int(np.ceil(win.width))    + 2 * apron_pixels
    height  = int(np.ceil(win.height))   + 2 * apron_pixels

    # Clip to source footprint
    col_off = max(col_off, 0)
    row_off = max(row_off, 0)
    width   = min(width,  src.width  - col_off)
    height  = min(height, src.height - row_off)
    return rasterio.windows.Window(col_off, row_off, width, height)


# =============================================================================
# Sliding-window corn-richest scoring (integral image)
# =============================================================================

def _best_chip_position(
    scoreable: np.ndarray,
    chip_size: int,
) -> tuple[int, int, int]:
    """Find the (row, col) origin of the 224x224 footprint with the highest
    sum of `scoreable` (a 2D bool/int array). Returns (best_row, best_col,
    best_count).

    Uses an integral image so the cost is O(H*W) regardless of chip size.
    Returns (-1, -1, 0) if no valid position fits inside the array.
    """
    H, W = scoreable.shape
    if H < chip_size or W < chip_size:
        return -1, -1, 0

    # Integral image. cumsum along axis=0 then axis=1.
    arr = scoreable.astype(np.int32, copy=False)
    integ = arr.cumsum(axis=0).cumsum(axis=1)
    # Prepend a zero row/col so we can do the standard integral-image
    # rectangle-sum lookup without conditional bounds checks.
    pad = np.zeros((H + 1, W + 1), dtype=np.int32)
    pad[1:, 1:] = integ

    # For every valid top-left corner (r, c), the chip covers rows
    # [r, r+chip_size) and cols [c, c+chip_size).
    # rect_sum(r, c) = pad[r+ch, c+ch] - pad[r, c+ch] - pad[r+ch, c] + pad[r, c]
    ch = chip_size
    A = pad[ch:H + 1,    ch:W + 1]    # bottom-right
    B = pad[:H - ch + 1, ch:W + 1]    # top-right
    C = pad[ch:H + 1,    :W - ch + 1] # bottom-left
    D = pad[:H - ch + 1, :W - ch + 1] # top-left
    sums = A - B - C + D  # shape: (H-ch+1, W-ch+1)

    # argmax over the flattened sums grid; ties broken by lowest (row, col)
    flat_idx = int(np.argmax(sums))
    best_row, best_col = np.unravel_index(flat_idx, sums.shape)
    return int(best_row), int(best_col), int(sums[best_row, best_col])


# =============================================================================
# Single-county processor
# =============================================================================

def _process_one_county(
    *,
    county_row,           # GeoDataFrame row (Series) for this county, in EPSG:5070
    granule_meta: dict,
    band_paths: dict[str, Path],   # role -> local path (incl. 'fmask')
    cdl_mask_path: Path,
    chip_root: Path,
    chip_size: int,
    min_corn_frac: float,
) -> ChipIndexRow:
    """Run the full chip-extraction pipeline for one (granule, county) pair.

    Always returns exactly one ChipIndexRow -- positive (chip written) or
    negative (skip_reason set)."""
    geoid = str(county_row["GEOID"]).zfill(5)
    state_alpha = granule_meta.get("state_alpha")  # filled in by caller
    year = granule_meta["year"]
    parsed = parse_granule_id(granule_meta["granule_id"])

    # Common partial fields for any returned row
    base_fields = dict(
        GEOID=geoid,
        state_alpha=state_alpha,
        year=year,
        phase=granule_meta["phase"],
        scene_date=parsed.scene_date,
        granule_id=granule_meta["granule_id"],
        sensor=parsed.sensor,
        mgrs_tile=parsed.mgrs_tile,
        cmr_cloud_pct=granule_meta.get("cloud_pct"),
    )

    def _negative(reason: str) -> ChipIndexRow:
        return ChipIndexRow(**base_fields, chip_path=None, skip_reason=reason)

    # 1. Open one band to learn the granule's CRS + transform. We use red.
    red_path = band_paths["red"]
    with rasterio.open(red_path) as red_src:
        granule_crs = red_src.crs
        granule_transform = red_src.transform

        # Reproject the county polygon's bounds from EPSG:5070 to granule CRS
        county_bounds_5070 = county_row.geometry.bounds
        try:
            county_bounds_utm = transform_bounds(
                "EPSG:5070", granule_crs, *county_bounds_5070, densify_pts=21
            )
        except Exception as e:
            log.warning("transform_bounds failed for %s/%s: %s",
                        granule_meta["granule_id"], geoid, e)
            return _negative("county_not_in_granule")

        # Compute the read-window with apron
        apron = chip_size  # one full chip's worth on each side
        win = _window_for_polygon_with_apron(red_src, county_bounds_utm, apron)

        if win.width < chip_size or win.height < chip_size:
            # County's footprint inside this granule is smaller than one chip
            return _negative("county_not_in_granule")

        # The transform for the windowed region (used to align CDL via VRT)
        win_transform = rasterio.windows.transform(win, granule_transform)

    # 2. Read all 6 Prithvi bands through the same window. Each open-read pair
    #    is independent because earthaccess writes per-band files.
    #    HLS uses -9999 as the nodata sentinel (atmospheric correction failure
    #    + true off-edge). Track which pixels are genuinely-valid data, and
    #    zero out the sentinels so they don't poison downstream Prithvi
    #    normalization.
    HLS_NODATA = -9999
    bands = np.zeros((len(PRITHVI_BAND_ORDER), win.height, win.width), dtype=np.int16)
    data_valid = np.ones((win.height, win.width), dtype=bool)
    for i, role in enumerate(PRITHVI_BAND_ORDER):
        with rasterio.open(band_paths[role]) as src:
            arr = src.read(1, window=win, boundless=True, fill_value=HLS_NODATA)
        # A pixel is invalid in any band -> invalid for the chip
        data_valid &= (arr != HLS_NODATA)
        # Replace nodata with 0 in the saved chip so Prithvi doesn't see
        # extreme values. The data_valid mask is the source of truth for
        # "pixel has real data."
        arr_clean = np.where(arr == HLS_NODATA, np.int16(0), arr.astype(np.int16, copy=False))
        bands[i] = arr_clean

    # 3. Read Fmask, decode to valid-pixel mask
    with rasterio.open(band_paths["fmask"]) as fsrc:
        fmask_arr = fsrc.read(1, window=win, boundless=True, fill_value=255)
    fmask_valid = fmask_valid_mask(fmask_arr)

    # Combined per-pixel validity: real data AND clear-sky.
    # data_valid catches granule-edge nodata (the chief cause of
    # nodata leakage); fmask_valid catches clouds/shadows/snow/water.
    valid_pixel = data_valid & fmask_valid

    # 4. Reproject CDL mask onto the same windowed grid via WarpedVRT.
    #    CDL is at EPSG:5070/30m, granule bands are at granule CRS/30m -- so
    #    the warp is just a CRS reprojection (no resolution change). Using
    #    Resampling.nearest because CDL is categorical (binary corn mask).
    with rasterio.open(cdl_mask_path) as cdl_src:
        vrt_options = dict(
            crs=granule_crs,
            transform=win_transform,
            width=win.width,
            height=win.height,
            resampling=Resampling.nearest,
            src_nodata=cdl_src.nodata,
            nodata=0,  # outside-CDL areas treated as not-corn
        )
        with WarpedVRT(cdl_src, **vrt_options) as vrt:
            cdl_aligned = vrt.read(1)
    cdl_corn = cdl_aligned > 0  # binary mask: 1 = corn

    # 5. Build the scoreable mask: corn AND data-valid AND clear-sky
    scoreable = cdl_corn & valid_pixel  # bool

    # Distinguish the two failure modes that both produce scoreable.sum()==0:
    #   - no_cdl_overlap: CDL says no corn in this window at all
    #   - all_cloud: CDL has corn, but Fmask/data-validity masked it all out
    # The corn mask is what differentiates them. Decision log uses these
    # exact reasons; this is the first place they diverge in the code path.
    if int(cdl_corn.sum()) == 0:
        return _negative("no_cdl_overlap")
    if int(scoreable.sum()) == 0:
        return _negative("all_cloud")

    # 6. Slide the 224x224 footprint to find the corn-richest position.
    best_row, best_col, best_count = _best_chip_position(scoreable, chip_size)
    if best_row < 0:
        # Window was smaller than a chip after clipping -- shouldn't happen
        # given the check above, but defensive.
        return _negative("county_not_in_granule")

    corn_frac = best_count / float(chip_size * chip_size)
    if corn_frac < min_corn_frac:
        # Best position still doesn't meet the 5% threshold
        return _negative("below_corn_threshold")

    # Also compute valid_pixel fraction in the chosen chip (QC column)
    chip_valid = valid_pixel[best_row:best_row + chip_size,
                             best_col:best_col + chip_size]
    valid_pixel_frac = float(chip_valid.mean())

    # If 100% cloud at the best position, downgrade to all_cloud
    if valid_pixel_frac == 0.0:
        return _negative("all_cloud")

    # 7. Slice the 6-band stack at the chosen position
    chip_bands = bands[:, best_row:best_row + chip_size,
                          best_col:best_col + chip_size]

    # 8. Compute the chip's geo-transform and centroid lat/lon for QC
    chip_transform = win_transform * Affine.translation(best_col, best_row)
    # Centroid in granule CRS:
    cx = chip_transform.c + chip_transform.a * (chip_size / 2.0)
    cy = chip_transform.f + chip_transform.e * (chip_size / 2.0)
    # Convert (cx, cy) to lat/lon
    try:
        lon_arr, lat_arr = rasterio.warp.transform(
            granule_crs, "EPSG:4326", [cx], [cy]
        )
        chip_lon = float(lon_arr[0])
        chip_lat = float(lat_arr[0])
    except Exception:
        chip_lon = chip_lat = float("nan")

    # 9. Write the chip GeoTIFF
    out_relpath = chip_relpath(
        geoid=geoid,
        year=year,
        phase=granule_meta["phase"],
        scene_date=parsed.scene_date,
        sensor=parsed.sensor,
        mgrs_tile=parsed.mgrs_tile,
        chip_root=str(chip_root),
    )
    out_path = Path(out_relpath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    profile = dict(CHIP_TIFF_PROFILE)
    profile.update({
        "crs": granule_crs,
        "transform": chip_transform,
        "height": chip_size,
        "width": chip_size,
    })
    # Embed the band roles as descriptions for self-documenting GeoTIFFs
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(chip_bands)
        for i, role in enumerate(PRITHVI_BAND_ORDER, start=1):
            dst.set_band_description(i, role)
        dst.update_tags(
            granule_id=granule_meta["granule_id"],
            sensor=parsed.sensor,
            scene_date=parsed.scene_date.isoformat(),
            phase=granule_meta["phase"],
            chip_origin_row=str(best_row),
            chip_origin_col=str(best_col),
            corn_pixel_count=str(best_count),
            valid_pixel_frac=f"{valid_pixel_frac:.4f}",
            scaling_factor="0.0001",
            extractor_version=ChipIndexRow.__dataclass_fields__["extractor_version"].default,
        )

    return ChipIndexRow(
        **base_fields,
        chip_path=out_relpath,
        chip_pixel_h=chip_size,
        chip_pixel_w=chip_size,
        chip_origin_row=best_row,
        chip_origin_col=best_col,
        corn_pixel_count=best_count,
        corn_pixel_frac=corn_frac,
        valid_pixel_frac=valid_pixel_frac,
        chip_lat=chip_lat,
        chip_lon=chip_lon,
        skip_reason=None,
    )


# =============================================================================
# Public entry point
# =============================================================================

def extract_chips_for_granule(
    *,
    granule_meta: dict,
    state_alpha: str,
    year: int,
    granule_local_dir: Path,
    chip_root: Path,
    counties_gpkg: str = COUNTIES_GPKG_DEFAULT,
    cdl_mask_dir: str = CDL_MASK_DIR_DEFAULT,
    chip_size: int = CHIP_SIZE_DEFAULT,
    min_corn_frac: float = MIN_CORN_FRAC_DEFAULT,
) -> list[ChipIndexRow]:
    """Process one downloaded HLS granule. Returns one ChipIndexRow per
    (granule, county) intersection (mix of positive + negative), or one
    sentinel row if no county intersects.

    Parameters
    ----------
    granule_meta : dict
        From download_hls.py's CMR result. Must have at least:
          - granule_id (str)
          - phase (str)
          - cloud_pct (float, optional)
          - sensor (str, optional; recomputed from granule_id if absent)
          - scene_date / mgrs_tile (optional; recomputed from granule_id)
        We add 'state_alpha' and 'year' inside this function for downstream.
    state_alpha : str
        Two-letter USPS code (IA, NE, WI, MO, CO).
    year : int
        Year of the cell. Must match the year encoded in granule_id.
    granule_local_dir : Path
        Directory where earthaccess.download wrote the band GeoTIFFs.
    chip_root : Path
        Root directory under which county chip subdirs are written.
    """
    granule_meta = dict(granule_meta)  # don't mutate caller's dict
    granule_meta["state_alpha"] = state_alpha
    granule_meta["year"] = year

    parsed = parse_granule_id(granule_meta["granule_id"])
    sensor = parsed.sensor
    if parsed.year != year:
        log.warning(
            "granule_id year %d != cell year %d for %s; using cell year",
            parsed.year, year, granule_meta["granule_id"]
        )
    granule_meta.setdefault("phase", granule_meta.get("phase"))
    if not granule_meta.get("phase"):
        raise ValueError(
            f"granule_meta missing 'phase'; got {granule_meta!r}"
        )

    granule_local_dir = Path(granule_local_dir)
    chip_root = Path(chip_root)

    # 1. Resolve the local file paths for the 6 bands + Fmask.
    band_codes = band_codes_for(sensor)
    band_paths: dict[str, Path] = {}
    missing_bands: list[str] = []
    for role in (*PRITHVI_BAND_ORDER, "fmask"):
        code = band_codes[role]
        p = _resolve_band_path(granule_local_dir, granule_meta["granule_id"], code)
        if p is None:
            missing_bands.append(f"{role}={code}")
        else:
            band_paths[role] = p

    if missing_bands:
        log.warning(
            "granule %s missing band files (%s); files present: %s",
            granule_meta["granule_id"],
            ", ".join(missing_bands),
            [p.name for p in _list_granule_files(granule_local_dir)],
        )
        return [
            ChipIndexRow(
                GEOID=SENTINEL_GEOID_NO_INTERSECT,
                state_alpha=state_alpha,
                year=year,
                phase=granule_meta["phase"],
                scene_date=parsed.scene_date,
                granule_id=granule_meta["granule_id"],
                sensor=sensor,
                mgrs_tile=parsed.mgrs_tile,
                cmr_cloud_pct=granule_meta.get("cloud_pct"),
                chip_path=None,
                skip_reason="all_cloud",  # treat unreadable like fully clouded
            )
        ]

    # 2. Resolve the CDL mask for this (state, year)
    cdl_mask_path = Path(cdl_mask_dir) / f"cdl_corn_mask_{state_alpha}_{year}.tif"
    if not cdl_mask_path.exists():
        # Try a glob fallback in case the naming convention drifted
        candidates = list(Path(cdl_mask_dir).glob(f"*_{state_alpha}_{year}.tif"))
        if candidates:
            cdl_mask_path = candidates[0]
        else:
            raise FileNotFoundError(
                f"CDL corn mask not found for {state_alpha}-{year} at {cdl_mask_path} "
                f"(D.1.a artifact). Run download_cdl.py + cdl_to_corn_mask.py first."
            )

    # 3. Load (cached) county polygons for this state, in EPSG:5070
    counties = load_counties(counties_gpkg, state_alpha=state_alpha)
    if len(counties) == 0:
        log.warning("no counties loaded for state %s", state_alpha)
        return []

    # 4. Compute granule footprint in EPSG:5070 to filter counties to
    #    those that *might* intersect (cheap polygon test before the
    #    expensive band reads).
    with rasterio.open(band_paths["red"]) as red_src:
        granule_bounds = red_src.bounds   # in granule CRS
        granule_crs = red_src.crs
    granule_bounds_5070 = transform_bounds(
        granule_crs, "EPSG:5070", *granule_bounds, densify_pts=21
    )
    from shapely.geometry import box
    granule_box_5070 = box(*granule_bounds_5070)

    intersecting = counties[counties.intersects(granule_box_5070)].reset_index(drop=True)

    if len(intersecting) == 0:
        return [
            ChipIndexRow(
                GEOID=SENTINEL_GEOID_NO_INTERSECT,
                state_alpha=state_alpha,
                year=year,
                phase=granule_meta["phase"],
                scene_date=parsed.scene_date,
                granule_id=granule_meta["granule_id"],
                sensor=sensor,
                mgrs_tile=parsed.mgrs_tile,
                cmr_cloud_pct=granule_meta.get("cloud_pct"),
                chip_path=None,
                skip_reason="county_not_in_granule",
            )
        ]

    # 5. Process each intersecting county
    rows: list[ChipIndexRow] = []
    for _, county_row in intersecting.iterrows():
        try:
            row = _process_one_county(
                county_row=county_row,
                granule_meta=granule_meta,
                band_paths=band_paths,
                cdl_mask_path=cdl_mask_path,
                chip_root=chip_root,
                chip_size=chip_size,
                min_corn_frac=min_corn_frac,
            )
            rows.append(row)
        except Exception as e:
            log.error(
                "county %s in granule %s failed: %s",
                county_row["GEOID"], granule_meta["granule_id"], e,
                exc_info=True,
            )
            # Emit a negative row so we don't block resumability
            rows.append(ChipIndexRow(
                GEOID=str(county_row["GEOID"]).zfill(5),
                state_alpha=state_alpha,
                year=year,
                phase=granule_meta["phase"],
                scene_date=parsed.scene_date,
                granule_id=granule_meta["granule_id"],
                sensor=sensor,
                mgrs_tile=parsed.mgrs_tile,
                cmr_cloud_pct=granule_meta.get("cloud_pct"),
                chip_path=None,
                skip_reason="all_cloud",  # generic-failure fallback
            ))

    return rows


# =============================================================================
# Standalone diagnostic CLI
# =============================================================================
# Lets you re-run extraction against a cached granule directory without
# touching CMR. Useful when iterating on the chip logic.

def _cli():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--granule-dir", required=True, type=Path,
                    help="Path to a downloaded granule directory (with Bxx.tif files inside)")
    ap.add_argument("--state", required=True, choices=["IA", "NE", "WI", "MO", "CO"])
    ap.add_argument("--year", required=True, type=int)
    ap.add_argument("--phase", required=True, choices=["aug1", "sep1", "oct1", "final"])
    ap.add_argument("--cloud-pct", type=float, default=None,
                    help="cmr_cloud_pct to populate in the row (optional)")
    ap.add_argument("--chip-root", default=CHIP_ROOT_DEFAULT)
    ap.add_argument("--counties-gpkg", default=COUNTIES_GPKG_DEFAULT)
    ap.add_argument("--cdl-mask-dir", default=CDL_MASK_DIR_DEFAULT)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Infer granule_id from the directory name
    gid = args.granule_dir.name
    try:
        parse_granule_id(gid)
    except ValueError as e:
        raise SystemExit(f"can't parse granule_id from dir name {gid!r}: {e}")

    granule_meta = {
        "granule_id": gid,
        "phase": args.phase,
        "cloud_pct": args.cloud_pct,
    }
    rows = extract_chips_for_granule(
        granule_meta=granule_meta,
        state_alpha=args.state,
        year=args.year,
        granule_local_dir=args.granule_dir,
        chip_root=Path(args.chip_root),
        counties_gpkg=args.counties_gpkg,
        cdl_mask_dir=args.cdl_mask_dir,
    )
    print(f"Produced {len(rows)} rows:")
    n_pos = sum(1 for r in rows if r.chip_path is not None)
    n_neg = sum(1 for r in rows if r.chip_path is None)
    print(f"  positive: {n_pos}")
    print(f"  negative: {n_neg}")
    print()
    print("Sample rows:")
    for r in rows[:8]:
        d = dataclasses.asdict(r)
        # Trim long fields for display
        if d.get("chip_path"):
            print(f"  + {d['GEOID']}  {d['chip_path']}  corn_frac={d['corn_pixel_frac']:.3f}")
        else:
            print(f"  - {d['GEOID']}  skip_reason={d['skip_reason']}")


if __name__ == "__main__":
    _cli()
