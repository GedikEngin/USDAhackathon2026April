"""
Phase A.3 — gSSURGO soil features → per-county CSV.

Reads each state's gSSURGO .gdb, pulls the Valu1 table (USDA's pre-aggregated
MUKEY-level table), and computes per-county area-weighted means of soil
properties via per-county windowed reads of the MUKEY raster.

Output: scripts/gssurgo_county_features.csv  (keyed on GEOID)

Static across years — soil doesn't change with year — so this table joins
on GEOID alone in merge_all.py.

CRS: gSSURGO is EPSG:5070 (Albers Equal Area). Counties are reprojected to
5070; soil rasters are read native (categorical MUKEY codes can't be
resampled).

Memory strategy: per-county windowed reads. State rasters are too large
to materialize in memory (CO is ~13 GB int32). Each county window is at
most a few hundred MB; typical counties are 10–100 MB.
"""

import sys
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.features import geometry_mask
import pyogrio

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
GDB_DIR = REPO_ROOT / "phase2" / "data" / "gSSURGO"
TIGER_DIR = REPO_ROOT / "phase2" / "data" / "tiger"
OUT_CSV = REPO_ROOT / "scripts" / "gssurgo_county_features.csv"

TIGER_DIR.mkdir(parents=True, exist_ok=True)

STATES = {
    "CO": "08",
    "IA": "19",
    "MO": "29",
    "NE": "31",
    "WI": "55",
}

VALU_COLS = [
    "nccpi3corn",
    "nccpi3all",
    "aws0_100",
    "aws0_150",
    "soc0_30",
    "soc0_100",
    "rootznemc",
    "rootznaws",
    "droughty",
    "pctearthmc",
    "pwsl1pomu",
]

ALBERS = "EPSG:5070"
TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER2018/COUNTY/tl_2018_us_county.zip"


# ---------------------------------------------------------------------------
# Step 1 — TIGER counties
# ---------------------------------------------------------------------------

def load_counties():
    cache = TIGER_DIR / "tl_2018_us_county_5states_5070.gpkg"
    if cache.exists():
        print(f"[tiger] using cache: {cache}")
        return gpd.read_file(cache)

    zip_path = TIGER_DIR / "tl_2018_us_county.zip"
    if not zip_path.exists():
        print(f"[tiger] downloading {TIGER_URL}")
        urllib.request.urlretrieve(TIGER_URL, zip_path)

    extract_dir = TIGER_DIR / "tl_2018_us_county"
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

    shp = extract_dir / "tl_2018_us_county.shp"
    print(f"[tiger] reading {shp}")
    gdf = gpd.read_file(shp)

    keep = set(STATES.values())
    gdf = gdf[gdf["STATEFP"].isin(keep)].copy()
    gdf["GEOID"] = gdf["GEOID"].astype(str).str.zfill(5)
    gdf = gdf.to_crs(ALBERS)

    gdf[["GEOID", "STATEFP", "NAME", "geometry"]].to_file(cache, driver="GPKG")
    print(f"[tiger] cached {len(gdf)} counties → {cache}")
    return gdf


# ---------------------------------------------------------------------------
# Step 2 — per-state extraction
# ---------------------------------------------------------------------------

def open_muraster(gdb):
    """Open MURASTER_10m from a gSSURGO .gdb; fallback if URI form fails."""
    raster_path = f"OpenFileGDB:{gdb}:MURASTER_10m"
    try:
        return rasterio.open(raster_path)
    except rasterio.errors.RasterioIOError:
        with rasterio.open(str(gdb)) as parent:
            sds = [s for s in parent.subdatasets if "MURASTER_10m" in s]
        if not sds:
            raise RuntimeError(f"could not locate MURASTER_10m in {gdb}")
        return rasterio.open(sds[0])


def extract_state(state_alpha, state_fp, counties_5070):
    gdb = GDB_DIR / f"gSSURGO_{state_alpha}" / f"gSSURGO_{state_alpha}.gdb"
    print(f"\n[{state_alpha}] reading Valu1 from {gdb}")

    valu = pyogrio.read_dataframe(str(gdb), layer="Valu1", use_arrow=False)
    valu["mukey"] = pd.to_numeric(valu["mukey"], errors="coerce").astype("Int64")
    valu = valu.dropna(subset=["mukey"]).copy()
    valu["mukey"] = valu["mukey"].astype("int64")
    print(f"[{state_alpha}] Valu1 rows: {len(valu)}")

    # Per-property MUKEY → value dicts. Small (one entry per Valu1 row).
    lookups = {}
    for col in VALU_COLS:
        if col not in valu.columns:
            print(f"[{state_alpha}]   WARN: {col} missing from Valu1")
            lookups[col] = None
            continue
        s = pd.to_numeric(valu[col], errors="coerce")
        lookups[col] = dict(zip(valu["mukey"].to_numpy(), s.to_numpy()))

    src = open_muraster(gdb)
    print(f"[{state_alpha}] raster CRS: {src.crs}, shape: {src.shape}, "
          f"dtype: {src.dtypes[0]}")
    assert src.crs.to_string() == ALBERS, (
        f"[{state_alpha}] expected EPSG:5070, got {src.crs}"
    )
    raster_nodata = src.nodata
    print(f"[{state_alpha}] raster nodata: {raster_nodata}")

    counties_state = counties_5070[counties_5070["STATEFP"] == state_fp].copy()
    counties_state = counties_state.sort_values("GEOID").reset_index(drop=True)
    print(f"[{state_alpha}] {len(counties_state)} counties")

    results = {col: [] for col in VALU_COLS}
    geoids = []

    for idx, row in counties_state.iterrows():
        geom = row.geometry
        geoid = row["GEOID"]
        geoids.append(geoid)

        # County window in raster pixel coordinates.
        try:
            win = from_bounds(*geom.bounds, transform=src.transform)
        except Exception as e:
            print(f"[{state_alpha}]   {geoid} bbox failure: {e}")
            for col in VALU_COLS:
                results[col].append(np.nan)
            continue

        win = win.round_offsets().round_lengths()
        win = win.intersection(Window(0, 0, src.width, src.height))
        if win.width <= 0 or win.height <= 0:
            print(f"[{state_alpha}]   {geoid} empty window (out of raster)")
            for col in VALU_COLS:
                results[col].append(np.nan)
            continue

        mukey_window = src.read(1, window=win)
        win_transform = src.window_transform(win)

        # geometry_mask: invert=False → returns True for pixels NOT in geom.
        out_mask = geometry_mask(
            [geom],
            out_shape=mukey_window.shape,
            transform=win_transform,
            all_touched=False,
            invert=False,
        )
        in_county = ~out_mask

        if raster_nodata is not None:
            in_county &= (mukey_window != raster_nodata)

        county_mukeys = mukey_window[in_county]
        if county_mukeys.size == 0:
            for col in VALU_COLS:
                results[col].append(np.nan)
            continue

        # np.unique + inverse: lookup once per unique MUKEY, gather to pixels.
        # Massively faster than per-pixel dict lookup.
        unique_mukeys, inverse = np.unique(county_mukeys, return_inverse=True)

        for col in VALU_COLS:
            lut = lookups[col]
            if lut is None:
                results[col].append(np.nan)
                continue
            vals = np.array(
                [lut.get(int(m), np.nan) for m in unique_mukeys],
                dtype="float64",
            )
            pixel_vals = vals[inverse]
            if np.any(~np.isnan(pixel_vals)):
                mean_val = np.nanmean(pixel_vals)
            else:
                mean_val = np.nan
            results[col].append(mean_val)

        if (idx + 1) % 10 == 0 or idx + 1 == len(counties_state):
            print(f"[{state_alpha}]   processed {idx + 1}/{len(counties_state)} counties")

    src.close()

    out = pd.DataFrame({"GEOID": geoids})
    for col in VALU_COLS:
        out[col] = results[col]
    out["state_alpha"] = state_alpha

    print(f"[{state_alpha}] per-column QC:")
    for col in VALU_COLS:
        s = pd.Series(results[col])
        print(f"[{state_alpha}]   {col:12s}  mean={s.mean():.4g}  "
              f"NaN={s.isna().sum()}/{len(out)}")

    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    counties = load_counties()
    print(f"[main] {len(counties)} counties total across 5 states")

    parts = []
    for alpha, fp in STATES.items():
        df = extract_state(alpha, fp, counties)
        parts.append(df)
        # Incremental save in case a later state crashes.
        tmp = OUT_CSV.with_suffix(f".{alpha}.partial.csv")
        df.to_csv(tmp, index=False)
        print(f"[{alpha}] wrote partial → {tmp}")

    full = pd.concat(parts, ignore_index=True)
    full = full.sort_values("GEOID").reset_index(drop=True)

    cols = ["GEOID", "state_alpha"] + VALU_COLS
    full = full[cols]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(OUT_CSV, index=False)
    print(f"\n[main] wrote {len(full)} rows × {len(full.columns)} cols → {OUT_CSV}")

    print("\n[QC] per-state row counts:")
    print(full["state_alpha"].value_counts().sort_index().to_string())
    print("\n[QC] per-column NaN counts:")
    print(full[VALU_COLS].isna().sum().to_string())
    print("\n[QC] per-column means (pooled across all counties):")
    print(full[VALU_COLS].mean().to_string())


if __name__ == "__main__":
    main()
