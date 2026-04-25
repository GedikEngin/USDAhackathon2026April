"""
Build per-(GEOID, year, forecast_date) HLS vegetation-index features from
the raw HLS pull output.

The HLS pull (`scripts/hls_pull.py`) produces a state-level CSV keyed on
(state, year, forecast_date) where:
  - `state` is the FULL state name ("Iowa", "Colorado", ...)
  - `forecast_date` is one of the pull's labels: "aug1", "sep1", "oct1", "final"

This script does two normalizations so the output joins cleanly with the
rest of the pipeline:

  1. state (full name)  ->  state_alpha (IA, CO, WI, MO, NE)
  2. forecast_date label  ->  the project's canonical date strings:
        aug1   -> 08-01
        sep1   -> 09-01
        oct1   -> 10-01
        final  -> EOS

  3. Broadcast each state-level row to every GEOID in that state, using the
     county directory in `nass_corn_5states_features.csv`. This gives one
     row per (GEOID, year, forecast_date) ready for `merge_all.py`.

INPUT:
  phase2/data/hls/hls_vi_features.csv   (output of hls_pull.py; may be partial)

OUTPUT:
  scripts/hls_county_features.csv       (per-(GEOID, year, forecast_date))

Notes:
  - The 2025 forecast rows (is_forecast=True) are kept and broadcast just like
    historical rows. They will appear in the master table with NaN
    yield_target, which is correct.
  - Pre-2013 has no HLS at all (HLS v2.0 archive starts 2013). This script
    does NOT emit empty rows for 2005-2012 -- merge_all.py performs the outer
    join, so missing-HLS years naturally show up with NaN HLS columns there.
  - 2013-2014 are Landsat-only era; cadence is lower in those years and some
    (state, year, forecast_date) cells may be missing. Same handling: missing
    cells become NaN at merge time.

Schema OUT (per row):
  GEOID            5-char zero-padded county FIPS
  state_alpha      2-char USPS state code
  year             int
  forecast_date    str in {"08-01", "09-01", "10-01", "EOS"}
  hls_ndvi_mean    float    (median across granules, state-level)
  hls_ndvi_std     float    (median across granules, state-level)
  hls_evi_mean     float    (median across granules, state-level)
  hls_evi_std      float    (median across granules, state-level)
  hls_n_granules   int      (number of granules contributing)

Usage:
  python scripts/hls_features.py
  python scripts/hls_features.py --in phase2/data/hls/hls_vi_features.csv \\
                                 --out scripts/hls_county_features.csv \\
                                 --counties scripts/nass_corn_5states_features.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

# --- Config ----------------------------------------------------

DEFAULT_IN       = "phase2/data/hls/hls_vi_features.csv"
DEFAULT_OUT      = "scripts/hls_county_features.csv"
DEFAULT_COUNTIES = "scripts/nass_corn_5states_features.csv"

# State full name -> USPS code. Mirrors the 5 states in scripts/hls_pull.py.
STATE_NAME_TO_ALPHA = {
    "Iowa":      "IA",
    "Colorado":  "CO",
    "Wisconsin": "WI",
    "Missouri":  "MO",
    "Nebraska":  "NE",
}

# HLS pull label -> project-canonical forecast_date string used by
# weather_features.py and drought_features.py.
LABEL_TO_FORECAST_DATE = {
    "aug1":  "08-01",
    "sep1":  "09-01",
    "oct1":  "10-01",
    "final": "EOS",
}

# Column rename: HLS pull uses bare "ndvi_mean"; we prefix with "hls_" so the
# merged master table is unambiguous (NDVI and EVI from MODIS-via-Earth Engine
# already have ndvi_* names of their own).
COLUMN_RENAMES = {
    "ndvi_mean":  "hls_ndvi_mean",
    "ndvi_std":   "hls_ndvi_std",
    "evi_mean":   "hls_evi_mean",
    "evi_std":    "hls_evi_std",
    "n_granules": "hls_n_granules",
}

FEATURE_COLS = list(COLUMN_RENAMES.values())


# --- Main ------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in",  dest="in_path",       default=DEFAULT_IN,
                    help=f"HLS pull output CSV (default: {DEFAULT_IN})")
    ap.add_argument("--out", dest="out_path",      default=DEFAULT_OUT,
                    help=f"Output CSV (default: {DEFAULT_OUT})")
    ap.add_argument("--counties", dest="counties", default=DEFAULT_COUNTIES,
                    help=f"County directory CSV with GEOID + state_alpha "
                         f"(default: {DEFAULT_COUNTIES})")
    args = ap.parse_args()

    # ---- Load HLS pull output ----------------------------------------------
    print(f"Reading {args.in_path}...")
    if not os.path.exists(args.in_path):
        raise SystemExit(f"  ERROR: {args.in_path} does not exist. "
                         f"Run scripts/hls_pull.py first.")
    hls = pd.read_csv(args.in_path)
    print(f"  {len(hls):,} raw HLS rows; columns: {list(hls.columns)}")

    # Drop rows where the VI didn't compute (no granules or all errors).
    # These rows have NaN ndvi_mean.
    n_before = len(hls)
    hls = hls.dropna(subset=["ndvi_mean"]).reset_index(drop=True)
    n_dropped = n_before - len(hls)
    if n_dropped:
        print(f"  Dropped {n_dropped:,} rows with no successful VI compute "
              f"(no granules or all errors).")

    if len(hls) == 0:
        raise SystemExit("  ERROR: no usable HLS rows. Check the HLS pull.")

    # ---- Normalize state name -> state_alpha ------------------------------
    unknown_states = sorted(set(hls["state"]) - set(STATE_NAME_TO_ALPHA))
    if unknown_states:
        raise SystemExit(f"  ERROR: unknown state names in HLS pull: "
                         f"{unknown_states}. Update STATE_NAME_TO_ALPHA.")
    hls["state_alpha"] = hls["state"].map(STATE_NAME_TO_ALPHA)

    # ---- Normalize forecast_date label -> "08-01" / ... / "EOS" -----------
    unknown_labels = sorted(set(hls["forecast_date"]) - set(LABEL_TO_FORECAST_DATE))
    if unknown_labels:
        raise SystemExit(f"  ERROR: unknown forecast_date labels in HLS pull: "
                         f"{unknown_labels}. Update LABEL_TO_FORECAST_DATE.")
    hls["forecast_date"] = hls["forecast_date"].map(LABEL_TO_FORECAST_DATE)

    # ---- Rename feature columns to hls_* prefix ---------------------------
    hls = hls.rename(columns=COLUMN_RENAMES)

    # Keep only the columns we need going forward.
    keep = ["state_alpha", "year", "forecast_date"] + FEATURE_COLS
    hls = hls[keep].copy()

    print(f"  After cleanup: {len(hls):,} state-level rows, "
          f"{hls['state_alpha'].nunique()} states, "
          f"{hls['year'].nunique()} years, "
          f"{hls['forecast_date'].nunique()} forecast_dates")

    # ---- Load county directory --------------------------------------------
    print(f"\nReading county directory from {args.counties}...")
    counties = pd.read_csv(args.counties, usecols=["GEOID", "state_alpha"])
    counties["GEOID"] = counties["GEOID"].astype(str).str.zfill(5)
    counties = counties.drop_duplicates().reset_index(drop=True)
    print(f"  {len(counties):,} GEOIDs across {counties['state_alpha'].nunique()} states")

    # Quick sanity: every state in HLS must appear in the county directory.
    missing = sorted(set(hls["state_alpha"]) - set(counties["state_alpha"]))
    if missing:
        raise SystemExit(f"  ERROR: HLS has rows for states not in the county "
                         f"directory: {missing}. Inspect inputs.")

    # ---- Broadcast: state-level row -> one row per GEOID in that state ----
    print("\nBroadcasting state-level HLS to GEOIDs...")
    out = counties.merge(hls, on="state_alpha", how="inner")
    out = out[["GEOID", "state_alpha", "year", "forecast_date"] + FEATURE_COLS]
    out = out.sort_values(["GEOID", "year", "forecast_date"]).reset_index(drop=True)

    # Cast types for cleanliness.
    out["year"]            = out["year"].astype(int)
    out["hls_n_granules"]  = out["hls_n_granules"].astype("Int64")  # nullable int

    print(f"  Output: {len(out):,} rows")

    # ---- QC tail (matches weather_features / drought_features style) ------
    print(f"\nFeature rows: {len(out):,}")
    print(f"Columns: {list(out.columns)}")
    print("\nSample (head):")
    print(out.head(8).to_string(index=False))

    print("\nCoverage by year:")
    print(out.groupby("year").size().to_string())

    print("\nCoverage by forecast_date:")
    print(out.groupby("forecast_date").size().to_string())

    print("\nCoverage by (state_alpha, year):")
    print(out.groupby(["state_alpha", "year"]).size().unstack(fill_value=0).to_string())

    print("\nNull counts:")
    for c in FEATURE_COLS:
        n = out[c].isna().sum()
        print(f"  {c:20s} {n:>6,}  ({100*n/len(out):5.1f}%)")

    # ---- Write -------------------------------------------------------------
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    out.to_csv(args.out_path, index=False)
    print(f"\nWrote {args.out_path}  ({os.path.getsize(args.out_path)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
