"""
forecast/hls_common.py
Shared HLS helpers used by:
  - scripts/download_hls.py    (CMR query + granule orchestration)
  - scripts/extract_chips.py   (per-granule chip extraction)
  - forecast/<chip-picker>     (D.1.d, selects T=3 chips per query)

Everything in this module is pure (no I/O orchestration, no global state
beyond the optional GDAL config context manager). The constants and helpers
are lifted from the deprecated `scripts/hls_pull.py` per the Phase 2-D.1
decisions log entry 2-D.1.kickoff:
  - GDAL cloud-native config (verbatim)
  - Fmask bit decoder (verbatim, bits 1-5: cloud, cloud-adjacent, shadow,
    snow/ice, water)
  - L30/S30 band-name asymmetry handling

Lifecycle: replaced/deleted at end of Phase D.2 if Prithvi is fine-tuned;
otherwise survives into the production forecast path.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Avoid hard import at module top-level for the GDAL config -- it's only used
# inside the cloud-native config helpers, and importing osgeo.gdal at module
# load time triggers GDAL's plugin scan (slow, noisy).
# Imports get deferred into the functions that need them.


# =============================================================================
# Prithvi-EO-2.0 input bands and HLS band-map (L30 vs S30 asymmetry)
# =============================================================================

# Order of the 6 bands the Prithvi-EO-2.0 family expects. We write chip
# GeoTIFFs in this exact order so the inference loader reads them with no
# remapping.
PRITHVI_BAND_ORDER: tuple[str, ...] = (
    "blue",
    "green",
    "red",
    "nir_narrow",
    "swir1",
    "swir2",
)

# HLS Landsat (L30, OLI). Source: HLS v2.0 User Guide.
L30_BAND_CODES: dict[str, str] = {
    "blue":       "B02",
    "green":      "B03",
    "red":        "B04",
    "nir_narrow": "B05",  # Landsat narrow NIR
    "swir1":      "B06",
    "swir2":      "B07",
    "fmask":      "Fmask",
}

# HLS Sentinel-2 (S30, MSI). Note B8A (narrow NIR), NOT B08 (wide NIR).
# B08 is wide-NIR and is the L30->S30 footgun the decisions log calls out.
S30_BAND_CODES: dict[str, str] = {
    "blue":       "B02",
    "green":      "B03",
    "red":        "B04",
    "nir_narrow": "B8A",  # Sentinel-2 narrow NIR (matches L30 B05)
    "swir1":      "B11",
    "swir2":      "B12",
    "fmask":      "Fmask",
}

# HLS Surface Reflectance scale factor. int16 raw -> float reflectance.
HLS_REFLECTANCE_SCALE: float = 0.0001

# Sensor short-names as they appear in CMR. Used in CMR queries and for
# parsing granule_ids that look like 'HLS.L30.T15TVH.2018196T172224.v2.0'.
HLS_L30_SHORT_NAME = "HLSL30"
HLS_S30_SHORT_NAME = "HLSS30"


def band_codes_for(sensor: str) -> dict[str, str]:
    """sensor in {'L30', 'S30'} -> the {role: HLS-asset-code} map.

    Asymmetric on purpose: callers should always go through this rather than
    hardcoding. Wrong band-map asymmetry is the most-historically-common
    source of bugs in HLS pipelines (per decisions log).
    """
    if sensor == "L30":
        return L30_BAND_CODES
    if sensor == "S30":
        return S30_BAND_CODES
    raise ValueError(f"sensor must be 'L30' or 'S30', got {sensor!r}")


# =============================================================================
# Calendar phase windows (locked in decisions log; chip-time-of-year)
# =============================================================================
# Each entry is ((start_month, start_day), (end_month, end_day)), inclusive.
# These are the same windows the deprecated `hls_pull.py` used (HIST_WINDOWS),
# kept identical so any downstream scripts comparing eras are apples-to-apples.

PHASE_WINDOWS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "aug1":  ((7, 17),  (8, 15)),    # Jul 17 - Aug 15
    "sep1":  ((8, 17),  (9, 15)),    # Aug 17 - Sep 15
    "oct1":  ((9, 17),  (10, 15)),   # Sep 17 - Oct 15
    "final": ((10, 17), (11, 15)),   # Oct 17 - Nov 15
}

# Mapping to the canonical forecast_date strings used everywhere else in the
# pipeline (training_master.parquet, regressors, weather_features.csv, etc).
PHASE_TO_FORECAST_DATE: dict[str, str] = {
    "aug1":  "08-01",
    "sep1":  "09-01",
    "oct1":  "10-01",
    "final": "EOS",
}

# The full growing-season pull window. Per decisions log: pull-once per
# (state, year) over <year>-05-01 to <year>-11-15. The CMR query uses these
# bounds; phase labeling happens client-side.
GROWING_SEASON_START_MD: tuple[int, int] = (5, 1)   # May 1
GROWING_SEASON_END_MD:   tuple[int, int] = (11, 15) # Nov 15


def label_calendar_phase(scene_date: dt.date) -> Optional[str]:
    """Return one of {'aug1', 'sep1', 'oct1', 'final'} based on scene_date's
    month-day, or None if scene_date falls outside all four phase windows.

    Year-agnostic: only uses (month, day). Phase windows are locked.
    Used at chip-index-write time, not at CMR-query time.
    """
    md = (scene_date.month, scene_date.day)
    for phase, (start_md, end_md) in PHASE_WINDOWS.items():
        if start_md <= md <= end_md:
            return phase
    return None


# =============================================================================
# Granule ID parsing
# =============================================================================
# HLS granule_id format: HLS.<L30|S30>.T<MGRS>.<YYYYDOY>T<HHMMSS>.v2.0
# Example: HLS.L30.T15TVH.2018196T172224.v2.0
#          HLS.S30.T14TPP.2020134T170851.v2.0

_GRANULE_RE = re.compile(
    r"^HLS\."
    r"(?P<sensor>L30|S30)\."
    r"T(?P<tile>[0-9]{2}[A-Z]{3})\."
    r"(?P<year>[0-9]{4})(?P<doy>[0-9]{3})"
    r"T(?P<hms>[0-9]{6})"
    r"\.v(?P<version>[0-9.]+)$"
)


@dataclass(frozen=True)
class GranuleMeta:
    """Parsed components of an HLS v2.0 granule id."""
    granule_id:  str
    sensor:      str   # 'L30' or 'S30'
    mgrs_tile:   str   # e.g. '15TVH'
    scene_date:  dt.date
    scene_time:  dt.time
    version:     str

    @property
    def doy(self) -> int:
        return self.scene_date.timetuple().tm_yday

    @property
    def year(self) -> int:
        return self.scene_date.year


def parse_granule_id(granule_id: str) -> GranuleMeta:
    """Parse HLS v2.0 granule id. Raises ValueError on malformed input."""
    m = _GRANULE_RE.match(granule_id)
    if not m:
        raise ValueError(f"unrecognized HLS granule id format: {granule_id!r}")
    g = m.groupdict()
    year = int(g["year"])
    doy  = int(g["doy"])
    scene_date = dt.date(year, 1, 1) + dt.timedelta(days=doy - 1)
    hms = g["hms"]
    scene_time = dt.time(int(hms[0:2]), int(hms[2:4]), int(hms[4:6]))
    return GranuleMeta(
        granule_id=granule_id,
        sensor=g["sensor"],
        mgrs_tile=g["tile"],
        scene_date=scene_date,
        scene_time=scene_time,
        version=g["version"],
    )


# =============================================================================
# Fmask decoder (verbatim from hls_pull.py::create_quality_mask)
# =============================================================================
# HLS Fmask is a uint8 bit-packed QA layer:
#   bit 0: cirrus
#   bit 1: cloud
#   bit 2: cloud adjacent
#   bit 3: cloud shadow
#   bit 4: snow/ice
#   bit 5: water
#   bits 6-7: aerosol
# We treat bits {1,2,3,4,5} as "bad pixels" -- same as the deprecated
# hls_pull.py and the LP DAAC HLS_Tutorial.ipynb defaults. Cirrus (bit 0) is
# handled separately by the underlying HLS atmospheric correction; we don't
# double-mask it. Aerosol bits are intentionally NOT masked (low-aerosol
# pixels still useful for vegetation indices).

_FMASK_BAD_BITS: tuple[int, ...] = (1, 2, 3, 4, 5)


def fmask_bad_mask(
    fmask_array: np.ndarray,
    bit_nums: tuple[int, ...] = _FMASK_BAD_BITS,
) -> np.ndarray:
    """Return a boolean mask where True = BAD pixel (cloud/shadow/etc).

    Equivalent to (`hls_pull.py::create_quality_mask`) but with explicit
    sentinel handling for NaN-filled arrays produced by some rasterio reads.
    Squeezes a leading band axis if present.
    """
    q = np.nan_to_num(np.asarray(fmask_array), nan=255).astype(np.int16)
    if q.ndim == 3:
        q = q[0]
    bad = np.zeros(q.shape, dtype=bool)
    for bit in bit_nums:
        bad |= (q & (1 << bit)) > 0
    return bad


def fmask_valid_mask(fmask_array: np.ndarray) -> np.ndarray:
    """Convenience inverse: True where pixel is GOOD (clear-sky, non-water,
    non-snow). Use this when you want a positive-valid-pixel mask."""
    return ~fmask_bad_mask(fmask_array)


# =============================================================================
# Chip path layout
# =============================================================================
# Locked in decisions log:
#   data/v2/hls/chips/<GEOID>/<year>/<phase>_<scene_date>_<sensor>_<tile>_<doy>.tif
#
# Including sensor + tile + doy in the filename means that within a single
# (GEOID, year, phase) triple, multiple chips can coexist without collision
# (when two granules from different MGRS tiles cover the same county on
# different days within the phase window).

CHIP_ROOT_DEFAULT = "data/v2/hls/chips"
CHIP_INDEX_PATH_DEFAULT = "data/v2/hls/chip_index.parquet"
CHIP_INDEX_SHARD_DIR_DEFAULT = "data/v2/hls/chip_index_shards"
GRANULE_CACHE_DIR_DEFAULT = "data/v2/hls/raw"


def chip_relpath(
    *,
    geoid: str,
    year: int,
    phase: str,
    scene_date: dt.date,
    sensor: str,
    mgrs_tile: str,
    chip_root: str = CHIP_ROOT_DEFAULT,
) -> str:
    """Canonical chip-on-disk relative path. Used by both writer and reader.

    Returns paths like:
      data/v2/hls/chips/19153/2018/sep1_2018-09-04_S30_T15TVH_2018247.tif
    """
    if phase not in PHASE_WINDOWS:
        raise ValueError(f"unknown phase {phase!r}; expected one of {list(PHASE_WINDOWS)}")
    geoid = str(geoid).zfill(5)
    doy = scene_date.timetuple().tm_yday
    fname = (
        f"{phase}_{scene_date.isoformat()}_"
        f"{sensor}_T{mgrs_tile}_{scene_date.year}{doy:03d}.tif"
    )
    return os.path.join(chip_root, geoid, str(year), fname)


def chip_index_shard_path(
    state_alpha: str,
    year: int,
    shard_dir: str = CHIP_INDEX_SHARD_DIR_DEFAULT,
) -> str:
    """Per-(state, year) shard path. The orchestrator writes one shard per
    cell so multiple terminals can run concurrently without contention.

    Final merged index at `data/v2/hls/chip_index.parquet` is built by a
    separate one-shot script after all pulls complete.
    """
    return os.path.join(shard_dir, f"chip_index_{state_alpha}_{year}.parquet")


# =============================================================================
# ChipIndexRow dataclass -- the row written to chip_index.parquet
# =============================================================================
# Schema is documented in the spec written before code (D.1.b kickoff).
# Two row types coexist in the same table:
#   "positive" rows: chip_path is set, skip_reason is None
#   "negative" rows: chip_path is None, skip_reason explains why


@dataclass
class ChipIndexRow:
    # Spatial / temporal keys
    GEOID: str
    state_alpha: str
    year: int
    phase: str               # 'aug1' | 'sep1' | 'oct1' | 'final'
    scene_date: dt.date

    # Granule provenance
    granule_id: str
    sensor: str              # 'L30' | 'S30'
    mgrs_tile: str           # e.g. '15TVH'
    cmr_cloud_pct: Optional[float]  # CMR-reported eo:cloud_cover, 0..100

    # Chip provenance / QC -- None for negative rows
    chip_path: Optional[str] = None
    chip_pixel_h: Optional[int] = None
    chip_pixel_w: Optional[int] = None
    chip_origin_row: Optional[int] = None
    chip_origin_col: Optional[int] = None
    corn_pixel_count: Optional[int] = None
    corn_pixel_frac: Optional[float] = None
    valid_pixel_frac: Optional[float] = None
    chip_lat: Optional[float] = None
    chip_lon: Optional[float] = None

    # Skip reason (one of {county_not_in_granule, below_corn_threshold,
    # all_cloud, no_cdl_overlap}) when chip_path is None; else None.
    skip_reason: Optional[str] = None

    # Bookkeeping (auto-filled if not provided)
    extracted_at: dt.datetime = field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc)
    )
    extractor_version: str = "d1b-v1"


# Sentinel GEOID used for "granule was processed but didn't intersect any
# 5-state county" rows. Single row per granule, keeps resumability check
# (granule_id presence) consistent without sidecar files.
SENTINEL_GEOID_NO_INTERSECT = "00000"

# Allowed skip_reason values. Validated at row-write time.
SKIP_REASONS: frozenset[str] = frozenset({
    "county_not_in_granule",
    "below_corn_threshold",
    "all_cloud",
    "no_cdl_overlap",
})


# =============================================================================
# State <-> bbox / FIPS / alpha mapping (lifted verbatim from hls_pull.py)
# =============================================================================
# Bounding boxes used as the spatial filter on the CMR search. Mildly
# inflated to the state's outer extent so we don't miss granules whose
# footprints clip a county on the border. Granule->county intersection
# is re-tested precisely at chip-extraction time with the TIGER polygons.

STATE_BBOX: dict[str, tuple[float, float, float, float]] = {
    # state_alpha -> (lon_min, lat_min, lon_max, lat_max)
    "IA": (-96.7, 40.3, -90.1, 43.6),
    "CO": (-109.1, 36.9, -102.0, 41.1),
    "WI": (-92.9, 42.5, -86.8, 47.1),
    "MO": (-95.8, 35.9, -89.1, 40.6),
    "NE": (-104.1, 39.9, -95.3, 43.0),
}

STATE_FIPS_TO_ALPHA: dict[str, str] = {
    "08": "CO", "19": "IA", "29": "MO", "31": "NE", "55": "WI",
}
STATE_ALPHA_TO_FIPS: dict[str, str] = {v: k for k, v in STATE_FIPS_TO_ALPHA.items()}


# =============================================================================
# GDAL cloud-native config
# =============================================================================
# Originally lifted from hls_pull.py::configure_gdal_for_cloud_access, which
# used osgeo.gdal.SetConfigOption directly. The forecast-d1 env doesn't ship
# the osgeo Python bindings (rasterio bundles its own GDAL via the wheel),
# so we set these via os.environ. GDAL reads config options from env at
# library init time and rasterio surfaces them to the underlying GDAL too,
# so this is functionally equivalent.


def _hls_gdal_options() -> dict[str, str]:
    cookie_file = os.path.expanduser("~/cookies.txt")
    return {
        "GDAL_HTTP_COOKIEFILE": cookie_file,
        "GDAL_HTTP_COOKIEJAR":  cookie_file,
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": "TIF",
        "GDAL_HTTP_UNSAFESSL":  "YES",
        "GDAL_HTTP_MAX_RETRY":  "10",
        "GDAL_HTTP_RETRY_DELAY": "0.5",
    }


def _set_gdal_options(opts: dict[str, str]) -> None:
    """Set GDAL config options via os.environ AND rasterio.Env defaults.

    rasterio.Env() at the top of a process applies these GDAL options for
    all subsequent rasterio calls; setting os.environ as well ensures any
    GDAL invocation that reads from env (e.g. earthaccess's vsicurl reads
    via GDAL) also sees them.
    """
    import rasterio
    for k, v in opts.items():
        os.environ[k] = v
    # Push into rasterio's default Env. This is a no-op if rasterio is
    # already imported with a default env -- the values just live in
    # rasterio's internal options dict for the rest of the process.
    rasterio.Env(**opts).__enter__()
    # Note: we deliberately leak the Env (no __exit__) so it stays active
    # for the process lifetime. The same pattern hls_pull.py used with
    # gdal.SetConfigOption -- "set once, never restore."


def configure_gdal_for_cloud_native_hls() -> None:
    """Set GDAL config options globally for cloud-native HLS reads.

    Idempotent. Safe to call multiple times. This is the same configuration
    used by the deprecated hls_pull.py; preserved to keep auth/cookie
    behavior consistent across the two pulls.
    """
    _set_gdal_options(_hls_gdal_options())


@contextlib.contextmanager
def hls_gdal_config():
    """Scoped variant: applies HLS GDAL options for the duration of the
    `with` block, then restores prior os.environ values. Useful when other
    rasterio code in the same process should NOT see the HLS-specific
    cookie file (e.g. CDL or county-polygon reads).
    """
    import rasterio
    opts = _hls_gdal_options()
    saved_env = {k: os.environ.get(k) for k in opts}
    for k, v in opts.items():
        os.environ[k] = v
    try:
        with rasterio.Env(**opts):
            yield
    finally:
        for k in opts:
            if saved_env[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = saved_env[k]


# =============================================================================
# CMR / earthaccess helpers
# =============================================================================

def temporal_window_for_year(year: int) -> tuple[str, str]:
    """The full growing-season window for the given year, as ISO date
    strings ready to feed earthaccess.search_data(temporal=...)."""
    s_m, s_d = GROWING_SEASON_START_MD
    e_m, e_d = GROWING_SEASON_END_MD
    return (f"{year}-{s_m:02d}-{s_d:02d}", f"{year}-{e_m:02d}-{e_d:02d}")


def cmr_short_names_both() -> list[str]:
    """The two HLS short-names to query in one earthaccess.search_data call.
    earthaccess accepts a list and unions the results, so we get L30 and S30
    granules in a single CMR round-trip."""
    return [HLS_L30_SHORT_NAME, HLS_S30_SHORT_NAME]
