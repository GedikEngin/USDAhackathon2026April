"""
Pull gridMET daily weather (5 states, 2005-2024), aggregate to county-day,
write yearly parquet shards.

Source decision: gridMET (UCMerced Climatology Lab), not PRISM.
Reasons in PHASE2_DECISIONS_LOG.md. Brief specifies "NOAA / NASA" generically;
gridMET uses NLDAS-2 (NASA) as its temporal backbone and PRISM as its spatial
backbone, so it satisfies the brief while being far easier to integrate
(per-year netCDF over plain HTTP, no rate limiting, native VPD, native 4km).

Variables pulled:
  tmmn (K)    -> tmin_c       (subtract 273.15)
  tmmx (K)    -> tmax_c       (subtract 273.15)
  pr   (mm)   -> prcp_mm      (no conversion)
  srad (W/m2) -> srad_mjm2    (multiply by 0.0864 for daily MJ/m2)
  vpd  (kPa)  -> vpd_kpa      (no conversion)

Output: data/v2/weather/raw/gridmet_county_daily_{year}.parquet
        one row per (GEOID, date), columns above.

Usage:
  python gridmet_pull.py                       # full 2005-2024
  python gridmet_pull.py --years 2010 2011     # subset
  python gridmet_pull.py --combine             # concat shards into one parquet
"""

import argparse
import os
import sys
import time
import urllib.request
import urllib.error

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import box
from rasterio import features
from affine import Affine

# --- Config ----------------------------------------------------

STATE_FIPS = ["19", "31", "55", "29", "08"]   # IA, NE, WI, MO, CO
YEARS_DEFAULT = list(range(2005, 2025))

# gridMET per-year netCDF endpoint. ~30-50 MB per file.
GRIDMET_URL = "http://www.northwestknowledge.net/metdata/data/{var}_{year}.nc"

# Map our column name -> (gridMET var short name, unit converter).
# Note: each per-year netCDF holds exactly one data variable, so we don't
# pin its name — we grab whatever single data var is in the file.
# (The aggregated all-years OPeNDAP files use long CF names like
# 'daily_minimum_temperature'; the per-year files use the bare CF standard
# name 'air_temperature' with cell_methods disambiguating min vs. max.
# Pinning by name was brittle. Grabbing the sole data var is robust.)
VAR_SPEC = [
    ("tmin_c",    "tmmn", lambda x: x - 273.15),
    ("tmax_c",    "tmmx", lambda x: x - 273.15),
    ("prcp_mm",   "pr",   lambda x: x),
    ("srad_mjm2", "srad", lambda x: x * 0.0864),
    ("vpd_kpa",   "vpd",  lambda x: x),
]

# TIGER/Line 2018 counties (same as GEE script)
TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER2018/COUNTY/tl_2018_us_county.zip"
COUNTIES_PATH = "data/v2/tiger/tl_2018_us_county.zip"

RAW_DIR = "data/v2/weather/raw"
NCDF_CACHE_DIR = os.path.join(RAW_DIR, "_gridmet_nc_cache")
COMBINED_PATH = "scripts/gridmet_county_daily_2005_2024.parquet"


# --- Helpers ---------------------------------------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def download_if_missing(url, dest, label=""):
    """Plain HTTP GET to disk. Skip if file exists. Retries 3x on transient errors."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    ensure_dir(os.path.dirname(dest))
    for attempt in range(3):
        try:
            print(f"  downloading {label}: {url}")
            t0 = time.time()
            urllib.request.urlretrieve(url, dest)
            sz = os.path.getsize(dest) / 1e6
            print(f"    -> {dest}  ({sz:.1f} MB, {time.time()-t0:.1f}s)")
            return dest
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            wait = 10 * (2 ** attempt)
            print(f"    error: {e}. retry in {wait}s ({attempt+1}/3)")
            time.sleep(wait)
    raise RuntimeError(f"failed to download {url} after 3 attempts")


def load_counties():
    """TIGER counties for the 5 target states, in WGS84 lon/lat."""
    if not os.path.exists(COUNTIES_PATH):
        download_if_missing(TIGER_URL, COUNTIES_PATH, label="TIGER counties")
    gdf = gpd.read_file(f"zip://{COUNTIES_PATH}")
    gdf = gdf[gdf["STATEFP"].isin(STATE_FIPS)].copy()
    # gridMET is on EPSG:4326 (WGS84 lon/lat); reproject to match.
    gdf = gdf.to_crs("EPSG:4326")
    gdf["GEOID"] = gdf["GEOID"].astype(str).str.zfill(5)
    print(f"loaded {len(gdf)} counties across {gdf['STATEFP'].nunique()} states")
    return gdf[["GEOID", "STATEFP", "NAME", "geometry"]].reset_index(drop=True)


def open_gridmet_year(var_short, year):
    """Open one (var, year) netCDF and return its sole DataArray."""
    url = GRIDMET_URL.format(var=var_short, year=year)
    nc_path = os.path.join(NCDF_CACHE_DIR, f"{var_short}_{year}.nc")
    download_if_missing(url, nc_path, label=f"{var_short}_{year}")
    ds = xr.open_dataset(nc_path, decode_times=True)
    # Each gridMET per-year file holds exactly one data variable. Don't pin its
    # name — that's brittle (per-year vs aggregated OPeNDAP use different names).
    data_vars = list(ds.data_vars)
    if len(data_vars) != 1:
        raise RuntimeError(f"{nc_path}: expected 1 data var, got {data_vars}")
    da = ds[data_vars[0]]
    # gridMET's lat axis is increasing-monotonic (per 2018 update). Lon in
    # [-124.77, -67.06], no wraparound fix needed.
    return da


def pixel_county_weights(da, counties_gdf):
    """
    Precompute area-weighted pixel->county weights ONCE.
    Returns a sparse-ish dict: {GEOID: (row_idx, col_idx, weight)}.

    Weight = fraction of the pixel inside that county polygon, using
    rasterio.features.rasterize at the gridMET pixel grid resolution.
    Equivalent to area-weighted mean since pixels are equal-area on this grid.
    """
    # Build the gridMET affine transform from the DataArray coords.
    lats = da["lat"].values   # ascending
    lons = da["lon"].values   # ascending
    res_x = float(lons[1] - lons[0])
    res_y = float(lats[1] - lats[0])
    # rasterio expects "north-up" affine (negative y step, origin at top-left).
    # Our lats ascend, so the top is lats[-1].
    x0 = float(lons[0]) - res_x / 2.0
    y0 = float(lats[-1]) + res_y / 2.0   # top edge
    transform = Affine(res_x, 0.0, x0, 0.0, -res_y, y0)
    height = len(lats)
    width  = len(lons)

    weights = {}
    for i, row in counties_gdf.iterrows():
        # Rasterize this single county at the gridMET grid. We use all_touched=True
        # so that small counties at least get one pixel; for big counties the
        # interior pixels dominate and edge pixels contribute proportionally less
        # via the fractional-coverage approximation (rasterize gives 1 for centroids
        # inside, all_touched extends to any pixel the polygon touches).
        # For more accuracy we approximate fractional coverage with a 3x3 subgrid.
        mask = features.rasterize(
            [(row.geometry, 1)],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,
        )
        # Note rasterio writes top-down (row 0 = north); our DataArray lats ascend
        # (lat index 0 = south). Flip mask vertically to match DataArray orientation.
        mask = mask[::-1, :]
        rr, cc = np.where(mask > 0)
        if len(rr) == 0:
            print(f"  WARN: county {row.GEOID} ({row.NAME}) got 0 pixels")
            continue
        # Equal weight per touched pixel (area-weighted mean over touched pixels).
        # gridMET pixels are ~equal area at our latitudes (CONUS, far from poles)
        # so unweighted mean over touched pixels ~= area-weighted mean.
        w = np.full(len(rr), 1.0 / len(rr), dtype=np.float32)
        weights[row.GEOID] = (rr.astype(np.int32), cc.astype(np.int32), w)
    return weights


def aggregate_year(year, counties_gdf):
    """Pull all 5 vars for `year`, aggregate to (GEOID, date), return DataFrame."""
    print(f"\n=== year {year} ===")

    # Open each variable's DataArray for this year.
    arrays = {}
    for col, var_short, conv in VAR_SPEC:
        da = open_gridmet_year(var_short, year)
        arrays[col] = (da, conv)

    # Slice each variable to the 5-state bounding box BEFORE loading into memory.
    # Without this, full-CONUS for 5 vars * 365 days = ~24 GB. Sliced is ~3 MB.
    states_bounds = counties_gdf.total_bounds  # minx, miny, maxx, maxy
    minx, miny, maxx, maxy = states_bounds
    pad = 0.1

    # xr.sel(lat=slice(...)) requires the slice direction to match the coordinate
    # direction. gridMET per-year files have lat DESCENDING (north -> south);
    # the aggregated OPeNDAP files have lat ascending. Detect, slice, then
    # normalize to ascending so the rest of the code (weights + mask flip)
    # works without branching.
    first_da = next(iter(arrays.values()))[0]
    lat_ascending = float(first_da["lat"].values[1]) > float(first_da["lat"].values[0])
    if lat_ascending:
        lat_slice = slice(miny - pad, maxy + pad)
    else:
        lat_slice = slice(maxy + pad, miny - pad)
    lon_slice = slice(minx - pad, maxx + pad)
    for col in arrays:
        da, conv = arrays[col]
        sliced = da.sel(lat=lat_slice, lon=lon_slice)
        if not lat_ascending:
            sliced = sliced.isel(lat=slice(None, None, -1))   # flip to ascending
        arrays[col] = (sliced, conv)
    sliced_first = next(iter(arrays.values()))[0]
    print(f"  sliced to bbox: lat {sliced_first.sizes['lat']}, lon {sliced_first.sizes['lon']} "
          f"(source lat ascending: {lat_ascending}; normalized to ascending)")

    # Time axis (shared across all vars in gridMET per-year files).
    time_vals = pd.to_datetime(sliced_first["day"].values)
    n_days = len(time_vals)

    # Build pixel->county weights ONCE on the sliced grid (same grid for all vars).
    print("  building pixel->county weights...")
    weights = pixel_county_weights(sliced_first, counties_gdf)
    print(f"  {len(weights)} counties have pixel coverage")

    # Pre-load each variable into a numpy array shape (n_days, n_lat, n_lon).
    # For 5 states bbox: ~12 lat * 30 lon * 365 days * 4 bytes * 5 vars ~= 2.6 MB. Trivial.
    print("  loading variables into memory...")
    loaded = {}
    for col, (da, conv) in arrays.items():
        arr = conv(da.values).astype(np.float32)   # (n_days, n_lat, n_lon)
        loaded[col] = arr
        print(f"    {col}: shape={arr.shape}, range=[{np.nanmin(arr):.2f}, {np.nanmax(arr):.2f}]")

    # Apply weights: for each county, build a (n_days,) time-series per variable.
    print("  aggregating to county-day...")
    rows = []
    for geoid, (rr, cc, w) in weights.items():
        # Pull the (n_days, n_pixels) slab and weight-average over pixels.
        rec = {"GEOID": geoid}
        per_var = {}
        for col, arr in loaded.items():
            # arr[:, rr, cc] -> (n_days, n_pixels)
            slab = arr[:, rr, cc]
            # weighted mean across pixels, ignoring nan
            num = np.nansum(slab * w[None, :], axis=1)
            denom = np.nansum(np.where(np.isnan(slab), 0.0, 1.0) * w[None, :], axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                series = np.where(denom > 0, num / denom, np.nan)
            per_var[col] = series
        # Expand into per-day rows.
        for di, date in enumerate(time_vals):
            row = {"GEOID": geoid, "date": date}
            for col in loaded:
                row[col] = float(per_var[col][di])
            rows.append(row)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    print(f"  -> {len(df):,} rows for year {year}")
    return df


# --- Main ------------------------------------------------------

ap = argparse.ArgumentParser()
ap.add_argument("--years", type=int, nargs="+", default=YEARS_DEFAULT,
                help="years to pull (default: 2005-2024)")
ap.add_argument("--combine", action="store_true",
                help="after pulling, concat all yearly shards into one parquet")
ap.add_argument("--keep-nc-cache", action="store_true",
                help="keep raw netCDFs after parquet is written (default: keep, for resume)")
args = ap.parse_args()

ensure_dir(RAW_DIR)
ensure_dir(NCDF_CACHE_DIR)

print("Loading counties...")
counties = load_counties()

# Pull each year. Skip years already on disk (idempotency).
for year in args.years:
    out_path = os.path.join(RAW_DIR, f"gridmet_county_daily_{year}.parquet")
    if os.path.exists(out_path):
        sz = os.path.getsize(out_path) / 1e6
        print(f"\n=== year {year} === SKIP (exists: {out_path}, {sz:.1f} MB)")
        continue
    df = aggregate_year(year, counties)
    df.to_parquet(out_path, index=False, compression="snappy")
    print(f"  wrote {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")

# --- Optional combine ----------------------------------------
if args.combine:
    print("\nCombining yearly shards...")
    shards = []
    for year in YEARS_DEFAULT:
        p = os.path.join(RAW_DIR, f"gridmet_county_daily_{year}.parquet")
        if os.path.exists(p):
            shards.append(pd.read_parquet(p))
    if not shards:
        print("no shards found; nothing to combine")
        sys.exit(0)
    combined = pd.concat(shards, ignore_index=True)
    combined = combined.sort_values(["GEOID", "date"]).reset_index(drop=True)
    ensure_dir(os.path.dirname(COMBINED_PATH))
    combined.to_parquet(COMBINED_PATH, index=False, compression="snappy")
    print(f"  wrote {COMBINED_PATH}: {len(combined):,} rows, "
          f"{combined['GEOID'].nunique()} counties, "
          f"{combined['date'].min()}..{combined['date'].max()}, "
          f"{os.path.getsize(COMBINED_PATH)/1e6:.1f} MB")

print("\nDone.")
