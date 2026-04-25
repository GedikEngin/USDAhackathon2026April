"""
HLS Imagery → NDVI & EVI feature extraction.

Searches NASA's HLS archive (Harmonized Landsat-Sentinel) for scenes over each
of the 5 target states at each of the 4 forecast date windows. For each scene,
computes NDVI and EVI with Fmask cloud masking, and saves state-level mean
values to a single canonical CSV.

Coverage:
    - Historical: 2013 through 2017 inclusive (5 years).
      HLS v2.0 archive begins 2013 (Landsat-only era — Sentinel-2 component
      starts 2015), so 2013 is the practical earliest year. Cadence is lower
      in 2013–2014 than in 2015+.
    - Forecast: existing phase2/data/hls/hls_2025.csv is loaded if present and
      preserved in the combined output.

Output:
    phase2/data/hls/hls_vi_features.csv  (canonical, one row per state-year-date_window)
    Path is relative to the project root, so run this script FROM the project
    root (e.g. `python scripts/hls_pull.py`), not from inside scripts/.

Prerequisites:
    - NASA Earthdata account: https://urs.earthdata.nasa.gov/
    - First run will prompt for username/password and persist to ~/.netrc
    - pip install earthaccess rioxarray xarray geopandas gdal numpy pandas

Usage:
    python hls_pull.py                      # full 2013-2017 pull
    python hls_pull.py --years 2013 2014    # subset of years
    python hls_pull.py --resume             # skip (state, year, date) rows
                                            # already in the output CSV

Notes:
    - This is a slow, network-bound job. Expect 30-60+ min for the full range.
    - Progress is checkpointed after each year. Re-run with --resume to
      pick up where a prior run left off.
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
import earthaccess
from osgeo import gdal

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────

STATE_BBOX = {
    "Iowa":      (-96.7, 40.3, -90.1, 43.6),
    "Colorado":  (-109.1, 36.9, -102.0, 41.1),
    "Wisconsin": (-92.9, 42.5, -86.8, 47.1),
    "Missouri":  (-95.8, 35.9, -89.1, 40.6),
    "Nebraska":  (-104.1, 39.9, -95.3, 43.0),
}

# Month-day windows for each forecast date. Each window is the ~30 days
# preceding the forecast date — gives enough scenes to filter clouds.
HIST_WINDOWS = {
    "aug1":  ("07-17", "08-15"),
    "sep1":  ("08-17", "09-15"),
    "oct1":  ("09-17", "10-15"),
    "final": ("10-17", "11-15"),
}

DEFAULT_YEARS = list(range(2013, 2018))  # 2013, 2014, 2015, 2016, 2017
GRANULES_PER_WINDOW = 5

OUTPUT_DIR = "phase2/data/hls"
COMBINED_CSV = os.path.join(OUTPUT_DIR, "hls_vi_features.csv")
FORECAST_2025_CSV = os.path.join(OUTPUT_DIR, "hls_2025.csv")


# ── GDAL / Earthdata setup ───────────────────────────────────────────────────

def configure_gdal_for_cloud_access():
    """Set GDAL options for cloud-native HLS access (per the HLS tutorial)."""
    cookie_file = os.path.expanduser("~/cookies.txt")
    gdal.SetConfigOption("GDAL_HTTP_COOKIEFILE", cookie_file)
    gdal.SetConfigOption("GDAL_HTTP_COOKIEJAR",  cookie_file)
    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    gdal.SetConfigOption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", "TIF")
    gdal.SetConfigOption("GDAL_HTTP_UNSAFESSL", "YES")
    gdal.SetConfigOption("GDAL_HTTP_MAX_RETRY", "10")
    gdal.SetConfigOption("GDAL_HTTP_RETRY_DELAY", "0.5")
    os.environ["GDAL_HTTP_COOKIEFILE"] = cookie_file
    os.environ["GDAL_HTTP_COOKIEJAR"]  = cookie_file


# ── Helpers (adapted from HLS_Tutorial.ipynb) ────────────────────────────────

def create_quality_mask(quality_data, bit_nums=(1, 2, 3, 4, 5)):
    """Build a bad-pixel mask from the HLS Fmask layer."""
    mask = np.zeros(quality_data.shape[-2:], dtype=bool)
    q = np.nan_to_num(np.array(quality_data), 255).astype(np.int8)
    if q.ndim == 3:
        q = q[0]
    for bit in bit_nums:
        mask = np.logical_or(mask, (q & (1 << bit)) > 0)
    return mask


def scaling(band):
    """Apply HLS scale factor of 0.0001 (DN → reflectance)."""
    out = band.copy()
    out.data = band.data * 0.0001
    return out


def calc_ndvi(nir, red):
    return ((nir - red) / (nir + red + 1e-8)).clip(-1, 1)


def calc_evi(nir, red, blue):
    evi = 2.5 * ((nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + 1e-8))
    evi = xr.where(evi != np.inf, evi, np.nan, keep_attrs=True)
    return evi.clip(-1, 1)


def load_and_compute_vi(granule, bbox, product_type):
    """
    Load one HLS granule, compute NDVI and EVI clipped to the state bbox,
    and return mean / std stats.

    Uses earthaccess.open() to get authenticated fsspec file handles so we
    avoid /vsicurl/ + cookie auth issues. Each file handle is passed to
    rioxarray via the `opener=` argument, and rioxarray/GDAL treats it as a
    regular stream.

    Returns dict or None on failure.
    """
    chunk = dict(band=1, x=512, y=512)
    if "HLSS30" in product_type:
        # Sentinel-2 band names
        band_map = {"nir": "B8A", "red": "B04", "blue": "B02", "fmask": "Fmask"}
    else:
        # Landsat band names
        band_map = {"nir": "B05", "red": "B04", "blue": "B02", "fmask": "Fmask"}

    urls = granule.data_links()

    try:
        # Pick the URLs we want for this granule, then open them all in one
        # earthaccess.open() call (it batches HTTP session setup).
        wanted_urls = []
        for code in band_map.values():
            url = next((u for u in urls if f".{code}." in u), None)
            if url is None:
                return None
            wanted_urls.append(url)

        files = earthaccess.open(wanted_urls)
        if not files or len(files) != len(wanted_urls):
            return None

        bands = {}
        lon_min, lat_min, lon_max, lat_max = bbox
        for (role, _code), fh in zip(band_map.items(), files):
            da = rxr.open_rasterio(fh, chunks=chunk, masked=True).squeeze("band", drop=True)
            da = da.rio.reproject("EPSG:4326")
            da = da.rio.clip_box(minx=lon_min, miny=lat_min, maxx=lon_max, maxy=lat_max)
            bands[role] = da

        nir  = scaling(bands["nir"])
        red  = scaling(bands["red"])
        blue = scaling(bands["blue"])

        qmask = create_quality_mask(bands["fmask"].values)
        ndvi = calc_ndvi(nir, red).where(~qmask)
        evi  = calc_evi(nir, red, blue).where(~qmask)

        return {
            "ndvi_mean": float(ndvi.mean().compute()),
            "ndvi_std":  float(ndvi.std().compute()),
            "evi_mean":  float(evi.mean().compute()),
            "evi_std":   float(evi.std().compute()),
            "n_granules": 1,
        }
    except Exception as e:
        print(f"    ⚠️  Error processing granule: {e}")
        return None


def fetch_hls_vi_for_window(state, bbox, temporal_window, year, label):
    """
    Search HLS for granules in the given bbox + temporal window, compute VI
    stats per granule, and return the median across granules.
    """
    try:
        results = earthaccess.search_data(
            short_name=["HLSL30", "HLSS30"],
            bounding_box=bbox,
            temporal=temporal_window,
            count=GRANULES_PER_WINDOW,
        )
    except Exception as e:
        print(f"    ⚠️  Search error for {state} {label} {year}: {e}")
        return None

    if not results:
        print(f"    No granules: {state} {label} {year}")
        return None

    print(f"  {state} {label} {year}: {len(results)} granules")
    vi_list = []
    for g in results:
        urls = g.data_links()
        product = "HLSS30" if any("HLSS30" in u for u in urls) else "HLSL30"
        vi = load_and_compute_vi(g, bbox, product)
        if vi:
            vi_list.append(vi)

    if not vi_list:
        return None

    return {
        "ndvi_mean": float(np.median([v["ndvi_mean"] for v in vi_list])),
        "ndvi_std":  float(np.median([v["ndvi_std"]  for v in vi_list])),
        "evi_mean":  float(np.median([v["evi_mean"]  for v in vi_list])),
        "evi_std":   float(np.median([v["evi_std"]   for v in vi_list])),
        "n_granules": len(vi_list),
    }


# ── Main pull loop ───────────────────────────────────────────────────────────

def load_existing_records(path):
    """Read existing CSV (if any) and return list of dict records."""
    if not os.path.exists(path):
        return []
    return pd.read_csv(path).to_dict("records")


def already_done(records, state, year, date_label):
    """Check if a (state, year, date_label) row is already in records."""
    for r in records:
        if (r.get("state") == state
                and int(r.get("year", -1)) == year
                and r.get("forecast_date") == date_label
                # only count as done if VI was actually computed
                and not pd.isna(r.get("ndvi_mean", np.nan))):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS,
                        help=f"Years to pull (default: {DEFAULT_YEARS[0]}-{DEFAULT_YEARS[-1]})")
    parser.add_argument("--resume", action="store_true",
                        help="Skip (state, year, date_window) combinations already in output CSV")
    parser.add_argument("--output", type=str, default=COMBINED_CSV,
                        help=f"Output CSV path (default: {COMBINED_CSV}). "
                             "Use a unique path per process when running parallel.")
    args = parser.parse_args()

    output_csv = args.output
    output_dir = os.path.dirname(output_csv) or "."
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("HLS NDVI/EVI feature extraction")
    print("=" * 70)
    print(f"Years:    {args.years[0]}–{args.years[-1]} ({len(args.years)} years)")
    print(f"States:   {list(STATE_BBOX.keys())}")
    print(f"Windows:  {list(HIST_WINDOWS.keys())}")
    print(f"Output:   {output_csv}")
    print(f"Resume:   {args.resume}")
    print()

    # GDAL config + Earthdata auth
    configure_gdal_for_cloud_access()
    print("✓ GDAL configured for cloud access")

    auth = earthaccess.login(persist=True)
    if not auth or not auth.authenticated:
        raise SystemExit("✗ Earthdata authentication failed. Register at "
                         "https://urs.earthdata.nasa.gov/ and try again.")
    print("✓ Earthdata authenticated")
    print()

    # Seed records: prefer existing combined CSV; else seed from 2025 forecast
    # CSV if present (so it gets carried forward into the combined output).
    records = load_existing_records(output_csv)
    if records:
        print(f"✓ Loaded {len(records)} existing records from {output_csv}")
    elif os.path.exists(FORECAST_2025_CSV):
        records = load_existing_records(FORECAST_2025_CSV)
        print(f"✓ Seeded {len(records)} 2025 forecast records from {FORECAST_2025_CSV}")
    else:
        print("✓ Starting from empty record set")
    print()

    total_rows_attempted = 0
    total_rows_succeeded = 0

    for year in args.years:
        print(f"--- Year {year} ---")
        for state, bbox in STATE_BBOX.items():
            for date_label, (m_start, m_end) in HIST_WINDOWS.items():
                if args.resume and already_done(records, state, year, date_label):
                    print(f"  {state} {date_label} {year}: skipped (already in CSV)")
                    continue

                total_rows_attempted += 1
                t_start = f"{year}-{m_start}"
                t_end   = f"{year}-{m_end}"
                vi_stats = fetch_hls_vi_for_window(state, bbox, (t_start, t_end),
                                                   year, date_label)
                row = {
                    "state": state,
                    "year": year,
                    "forecast_date": date_label,
                    "is_forecast": False,
                }
                if vi_stats:
                    row.update(vi_stats)
                    total_rows_succeeded += 1
                records.append(row)

        # Checkpoint after each year
        df_out = pd.DataFrame(records)
        df_out.to_csv(output_csv, index=False)
        print(f"  → Checkpoint saved ({len(df_out)} total records in {output_csv})")
        print()

    # Final save (covers the case where args.years is empty after --resume)
    df_out = pd.DataFrame(records)
    df_out.to_csv(output_csv, index=False)

    print("=" * 70)
    print("✓ HLS fetch complete")
    print("=" * 70)
    print(f"Attempted this run: {total_rows_attempted}")
    print(f"Succeeded:          {total_rows_succeeded}")
    print(f"Total in CSV:       {len(df_out)}")
    print(f"Saved:              {output_csv}")
    print()
    print("First 10 rows:")
    print(df_out.head(10).to_string())


if __name__ == "__main__":
    main()
