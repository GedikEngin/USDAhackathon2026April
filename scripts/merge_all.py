"""
Phase A.6 — Master training table.

Outer-joins all per-source feature tables into a single canonical training
table at the (GEOID, year, forecast_date) grain. This is the input to Phases
B (analog retrieval), C (XGBoost), and D (Prithvi).

Sources (all already on disk, all keyed differently):

  scripts/nass_corn_5states_features.csv       (GEOID, year)
  phase2/data/ndvi/corn_ndvi_5states_<Y>.csv   (GEOID, year)  -- 21 files, concatenated
  scripts/gssurgo_county_features.csv          (GEOID,)       -- static, broadcast across years
  scripts/weather_county_features.csv          (GEOID, year, forecast_date)
  scripts/drought_county_features.csv          (GEOID, year, forecast_date)
  phase2/data/hls/hls_vi_features_*.csv        (state, year, forecast_date)
                                               -- multiple per-year-range slices,
                                                  concatenated and de-duped here;
                                                  state (full name) -> state_alpha
                                                  and forecast_date (aug1/sep1/oct1/final)
                                                  -> (08-01/09-01/10-01/EOS) normalized
                                                  to match the locked schema;
                                                  optional; broadcast to GEOID;
                                                  pre-2013 has no HLS rows by design

SKELETON: every (GEOID, year) in the NASS features file × the 4 forecast_date
labels = 4 rows per (GEOID, year). gSSURGO and NDVI broadcast across the 4
forecast dates. HLS (state-level) is broadcast to all GEOIDs in the state.

KEYS (locked schema from PHASE2_DATA_INVENTORY.md):
  GEOID         5-char zero-padded string  ("19153")
  state_alpha   2-char USPS                 ("IA")
  year          int
  forecast_date one of {"08-01", "09-01", "10-01", "EOS"}

TARGET:
  yield_target  bu/acre, combined-practice (yield_bu_acre_all from NASS)
                NaN for any (GEOID, year) absent from NASS — caller decides
                whether to filter for training or treat as forecast query.

AS-OF SAFETY: weather and drought features are pre-engineered to respect the
as-of rule (each row uses only data with timestamps strictly before its
forecast_date). The outer-join here is a pure key-aligned merge; nothing in
this script touches feature values, so as-of safety is preserved.

OUTER JOIN, NOT INNER: every (GEOID, year, forecast_date) in the skeleton is
preserved even if some sources have no row for it. Missing values become NaN.
This is intentional — sparseness is a real signal (e.g. NDVI gaps in CO 2005).

NO ROW FILTERING: train/val/holdout split is 2005–2022 / 2023 / 2024, applied
in the train script, not here. All years in the skeleton (which inherits from
NASS, currently 2004–2024) go into the master table.

Usage:
  python scripts/merge_all.py
  python scripts/merge_all.py --out scripts/training_master.parquet

Run from project root.
"""

import argparse
import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd

# --- Config ----------------------------------------------------

DEFAULT_NASS    = "scripts/nass_corn_5states_features.csv"
DEFAULT_GSSURGO = "scripts/gssurgo_county_features.csv"
DEFAULT_WEATHER = "scripts/weather_county_features.csv"
DEFAULT_DROUGHT = "scripts/drought_county_features.csv"
DEFAULT_NDVI_GLOB = "phase2/data/ndvi/corn_ndvi_5states_*.csv"
DEFAULT_HLS     = "phase2/data/hls/hls_vi_features_*.csv"  # glob: per-year-range splits, optional / in-flight
DEFAULT_OUT     = "scripts/training_master.parquet"

# Locked schema — match weather_features.py and drought_features.py exactly.
FORECAST_DATES = ["08-01", "09-01", "10-01", "EOS"]

# Source group tags for full-coverage QC at the end.
NASS_FEATURE_COLS    = ["irrigated_share", "harvest_ratio",
                        "acres_harvested_all", "acres_planted_all"]
NDVI_FEATURE_COLS    = ["ndvi_peak", "ndvi_gs_mean", "ndvi_gs_integral",
                        "ndvi_silking_mean", "ndvi_veg_mean"]
GSSURGO_FEATURE_COLS = ["nccpi3corn", "nccpi3all", "aws0_100", "aws0_150",
                        "soc0_30", "soc0_100", "rootznemc", "rootznaws",
                        "droughty", "pctearthmc", "pwsl1pomu"]
WEATHER_FEATURE_COLS = ["gdd_cum_f50_c86", "edd_hours_gt86f", "edd_hours_gt90f",
                        "vpd_kpa_veg", "vpd_kpa_silk", "vpd_kpa_grain",
                        "prcp_cum_mm", "dry_spell_max_days",
                        "srad_total_veg", "srad_total_silk", "srad_total_grain"]
DROUGHT_FEATURE_COLS = ["d0_pct", "d1_pct", "d2_pct", "d3_pct", "d4_pct", "d2plus"]


# --- Helpers ---------------------------------------------------

def fix_geoid(s):
    """Coerce GEOID to 5-char zero-padded string per locked schema."""
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)


def fix_state_fips(s):
    """STATEFP -> 2-char zero-padded string. Used for NDVI -> state_alpha lookup."""
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)


# state FIPS -> state_alpha. Keeps merge_all.py self-contained and avoids
# pulling in tigerlib/etc. just for a 5-element lookup.
STATE_FIPS_TO_ALPHA = {
    "08": "CO",
    "19": "IA",
    "29": "MO",
    "31": "NE",
    "55": "WI",
}

# Full state name -> state_alpha. HLS exports carry "Iowa", not "IA"; convert
# at load time so the merge key matches NASS/skeleton state_alpha exactly.
STATE_NAME_TO_ALPHA = {
    "Colorado":  "CO",
    "Iowa":      "IA",
    "Missouri":  "MO",
    "Nebraska":  "NE",
    "Wisconsin": "WI",
}

# HLS forecast_date label -> locked schema label. The HLS pipeline writes
# "aug1"/"sep1"/"oct1"/"final"; everything downstream uses the canonical
# "08-01"/"09-01"/"10-01"/"EOS" set defined in FORECAST_DATES.
HLS_FORECAST_DATE_MAP = {
    "aug1":  "08-01",
    "sep1":  "09-01",
    "oct1":  "10-01",
    "final": "EOS",
}


def load_nass(path):
    """Load NASS features. Returns DataFrame with normalized GEOID."""
    print(f"  reading NASS:    {path}")
    df = pd.read_csv(path)
    df["GEOID"] = fix_geoid(df["GEOID"])
    print(f"    {len(df):,} rows  ({df['year'].min()}..{df['year'].max()}, "
          f"{df['GEOID'].nunique()} GEOIDs)")
    return df


def load_gssurgo(path):
    """Load gSSURGO static county features. Drops state_alpha (NASS authoritative)."""
    print(f"  reading gSSURGO: {path}")
    df = pd.read_csv(path)
    df["GEOID"] = fix_geoid(df["GEOID"])
    # NASS is authoritative for state_alpha. Drop here to avoid suffixed cols on merge.
    df = df.drop(columns=[c for c in ("state_alpha",) if c in df.columns])
    print(f"    {len(df):,} rows  ({df['GEOID'].nunique()} GEOIDs, static)")
    return df


def load_ndvi(glob_pattern):
    """Concat all per-year NDVI CSVs. Schema:
       GEOID, NAME, STATEFP, year, ndvi_peak, ndvi_gs_mean, ndvi_gs_integral,
       ndvi_silking_mean, ndvi_veg_mean.
       Already pre-scaled (× 0.0001 server-side) — do NOT re-scale.
    """
    paths = sorted(glob.glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f"No NDVI CSVs matched glob: {glob_pattern}")
    print(f"  reading NDVI:    {len(paths)} files matching {glob_pattern}")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        # Some exports may carry stray columns; only keep what we declared.
        keep = [c for c in ["GEOID", "year"] + NDVI_FEATURE_COLS if c in df.columns]
        df = df[keep]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["GEOID"] = fix_geoid(df["GEOID"])
    print(f"    {len(df):,} rows  ({df['year'].min()}..{df['year'].max()}, "
          f"{df['GEOID'].nunique()} GEOIDs)")
    return df


def load_weather(path):
    """Per-(GEOID, year, forecast_date) weather features."""
    print(f"  reading weather: {path}")
    df = pd.read_csv(path)
    df["GEOID"] = fix_geoid(df["GEOID"])
    df["forecast_date"] = df["forecast_date"].astype(str)
    print(f"    {len(df):,} rows  ({df['year'].min()}..{df['year'].max()}, "
          f"forecast_dates={sorted(df['forecast_date'].unique())})")
    return df


def load_drought(path):
    """Per-(GEOID, year, forecast_date) drought features."""
    print(f"  reading drought: {path}")
    df = pd.read_csv(path)
    df["GEOID"] = fix_geoid(df["GEOID"])
    df["forecast_date"] = df["forecast_date"].astype(str)
    print(f"    {len(df):,} rows  ({df['year'].min()}..{df['year'].max()}, "
          f"forecast_dates={sorted(df['forecast_date'].unique())})")
    return df


def load_hls(glob_pattern):
    """Optional. Per-(state_alpha, year, forecast_date) HLS-derived VIs.

    The HLS pull writes one CSV per year-range slice (e.g. hls_vi_features_2013_2014.csv,
    hls_vi_features_2015.csv, ...), and slices can overlap on the boundary years from
    rerun batches. So this function:
      1. globs all matching files,
      2. concats them,
      3. drops exact-row duplicates from the overlap,
      4. normalizes `state` (full name, e.g. "Iowa") -> `state_alpha` ("IA") to match
         NASS/skeleton key,
      5. relabels forecast_date from the HLS pipeline's "aug1/sep1/oct1/final" to the
         locked schema's "08-01/09-01/10-01/EOS".

    Pre-2013 expected to have no rows (Landsat-only era cadence is too sparse).
    Returns DataFrame or None if no files match.
    """
    paths = sorted(glob.glob(glob_pattern))
    if not paths:
        print(f"  HLS files not found at {glob_pattern} — emitting NaN HLS columns. "
              f"(Expected during Phase A; HLS pull is a Phase D.1 prerequisite.)")
        return None

    print(f"  reading HLS:     {len(paths)} files matching {glob_pattern}")
    frames = []
    for p in paths:
        df_part = pd.read_csv(p)
        print(f"    {p}: {len(df_part):,} rows  "
              f"({df_part['year'].min()}..{df_part['year'].max()})")
        frames.append(df_part)
    df = pd.concat(frames, ignore_index=True)

    # Drop exact duplicate rows produced by overlapping year-range slices.
    # We dedupe on the full row, not just the key, so that if two slices
    # disagree on a value (shouldn't happen, but) we'd still see both rows
    # and the later assert on (state_alpha, year, forecast_date) uniqueness
    # via the merge will catch it.
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"    dropped {n_dropped:,} exact-duplicate rows from overlapping slices")

    # Normalize state column: HLS writes the full name, schema needs USPS alpha.
    if "state_alpha" not in df.columns:
        if "state" not in df.columns:
            raise KeyError(f"HLS files have neither 'state_alpha' nor 'state' column: "
                           f"got {list(df.columns)}")
        unknown = sorted(set(df["state"]) - set(STATE_NAME_TO_ALPHA))
        if unknown:
            raise ValueError(f"HLS state names not in STATE_NAME_TO_ALPHA: {unknown}")
        df["state_alpha"] = df["state"].map(STATE_NAME_TO_ALPHA)
        df = df.drop(columns=["state"])
    df["state_alpha"] = df["state_alpha"].astype(str)

    # Relabel forecast_date to the locked schema.
    df["forecast_date"] = df["forecast_date"].astype(str)
    unknown_fd = sorted(set(df["forecast_date"]) - set(HLS_FORECAST_DATE_MAP)
                        - set(FORECAST_DATES))
    if unknown_fd:
        raise ValueError(f"HLS forecast_date values not mappable: {unknown_fd}")
    df["forecast_date"] = df["forecast_date"].replace(HLS_FORECAST_DATE_MAP)

    # Final key-uniqueness check — at this point any remaining duplicate keys
    # would mean two slices disagreed on feature values, which is a real bug.
    dup_keys = df.duplicated(subset=["state_alpha", "year", "forecast_date"])
    if dup_keys.any():
        bad = df[df.duplicated(subset=["state_alpha", "year", "forecast_date"], keep=False)]
        raise ValueError(
            f"HLS slices disagree on values for {dup_keys.sum()} duplicated "
            f"(state_alpha, year, forecast_date) keys:\n{bad.sort_values(['state_alpha','year','forecast_date'])}"
        )

    print(f"    {len(df):,} rows total  (states={sorted(df['state_alpha'].unique())}, "
          f"{df['year'].min()}..{df['year'].max()}, "
          f"forecast_dates={sorted(df['forecast_date'].unique())})")
    return df


# --- Main merge ------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nass",     default=DEFAULT_NASS)
    ap.add_argument("--gssurgo",  default=DEFAULT_GSSURGO)
    ap.add_argument("--ndvi-glob", dest="ndvi_glob", default=DEFAULT_NDVI_GLOB)
    ap.add_argument("--weather",  default=DEFAULT_WEATHER)
    ap.add_argument("--drought",  default=DEFAULT_DROUGHT)
    ap.add_argument("--hls",      default=DEFAULT_HLS, dest="hls_glob",
                    help="Glob for HLS state-level feature CSVs (one per year-range slice). "
                         "Skipped if no files match.")
    ap.add_argument("--out",      default=DEFAULT_OUT)
    args = ap.parse_args()

    print("Loading sources...")
    nass    = load_nass(args.nass)
    gssurgo = load_gssurgo(args.gssurgo)
    ndvi    = load_ndvi(args.ndvi_glob)
    weather = load_weather(args.weather)
    drought = load_drought(args.drought)
    hls     = load_hls(args.hls_glob)

    # -- Build the skeleton ------------------------------------------
    # Every (GEOID, year) in NASS × 4 forecast_dates. NASS carries state_alpha
    # and county_name through to the output for downstream readability.
    print("\nBuilding skeleton (NASS keys × 4 forecast dates)...")
    nass_keys = nass[["GEOID", "year", "state_alpha", "county_name"]].drop_duplicates()
    fd_df = pd.DataFrame({"forecast_date": FORECAST_DATES})
    skeleton = nass_keys.merge(fd_df, how="cross")
    print(f"  skeleton: {len(skeleton):,} rows = "
          f"{len(nass_keys):,} (GEOID, year) × {len(FORECAST_DATES)} forecast_dates")

    # GEOID -> state_alpha lookup (used to broadcast HLS state-level features).
    geoid_to_state = nass_keys[["GEOID", "state_alpha"]].drop_duplicates()
    if geoid_to_state["GEOID"].duplicated().any():
        # Should not happen — every GEOID belongs to exactly one state. Guard anyway.
        raise ValueError("GEOID maps to multiple state_alpha values in NASS.")

    # -- Layer 1: NASS features (per-(GEOID, year), broadcast across forecast_dates) --
    nass_feats = nass.drop(columns=["state_alpha", "county_name"], errors="ignore")
    df = skeleton.merge(nass_feats, on=["GEOID", "year"], how="left")
    print(f"  + NASS features:    {len(df):,} rows")

    # -- Layer 2: NDVI (per-(GEOID, year), broadcast across forecast_dates) ----------
    # NDVI carries STATEFP / NAME we don't need; drop before merging to keep cols clean.
    ndvi_keep = ["GEOID", "year"] + [c for c in NDVI_FEATURE_COLS if c in ndvi.columns]
    df = df.merge(ndvi[ndvi_keep], on=["GEOID", "year"], how="left")
    print(f"  + NDVI:             {len(df):,} rows")

    # -- Layer 3: gSSURGO (per-GEOID, broadcast across all years and forecast_dates) -
    df = df.merge(gssurgo, on="GEOID", how="left")
    print(f"  + gSSURGO:          {len(df):,} rows")

    # -- Layer 4: weather (per-(GEOID, year, forecast_date)) -------------------------
    weather_keep = ["GEOID", "year", "forecast_date"] + WEATHER_FEATURE_COLS
    df = df.merge(weather[weather_keep], on=["GEOID", "year", "forecast_date"], how="left")
    print(f"  + weather:          {len(df):,} rows")

    # -- Layer 5: drought (per-(GEOID, year, forecast_date)) -------------------------
    drought_keep = ["GEOID", "year", "forecast_date"] + DROUGHT_FEATURE_COLS
    df = df.merge(drought[drought_keep], on=["GEOID", "year", "forecast_date"], how="left")
    print(f"  + drought:          {len(df):,} rows")

    # -- Layer 6: HLS (state-level → broadcast to GEOID) -----------------------------
    hls_feature_cols = []
    if hls is not None:
        hls_meta_cols = {"state_alpha", "year", "forecast_date"}
        hls_feature_cols = [c for c in hls.columns if c not in hls_meta_cols]
        # Broadcast: every GEOID in a state inherits the state's HLS reading for
        # that (year, forecast_date). Implementation: merge on state_alpha first,
        # then merge on (state_alpha, year, forecast_date).
        df = df.merge(
            hls[["state_alpha", "year", "forecast_date"] + hls_feature_cols],
            on=["state_alpha", "year", "forecast_date"], how="left",
        )
        print(f"  + HLS:              {len(df):,} rows  ({len(hls_feature_cols)} cols)")
    else:
        # No HLS file available — emit NaN columns so downstream code has stable schema.
        # Placeholder columns are documented in PHASE2_DATA_DICTIONARY.md.
        # We don't know the column names without the file, so this is left as a
        # no-op; downstream code that needs HLS columns must check existence.
        # (When the file lands, just rerun this script.)
        pass

    # -- Sort and finalize -------------------------------------------------------------
    # Stable forecast_date ordering (chronological, not lexicographic — "EOS" sorts
    # last alphabetically, but "10-01" < "EOS" lexicographically too, so this happens
    # to work; making it explicit anyway via Categorical for safety).
    df["forecast_date"] = pd.Categorical(df["forecast_date"],
                                         categories=FORECAST_DATES, ordered=True)
    df = df.sort_values(["GEOID", "year", "forecast_date"]).reset_index(drop=True)
    df["forecast_date"] = df["forecast_date"].astype(str)  # parquet doesn't like Categoricals as keys

    # Reorder columns: keys, target, then features grouped by source.
    key_cols    = ["GEOID", "state_alpha", "county_name", "year", "forecast_date"]
    target_cols = ["yield_target"]
    nass_aux    = [c for c in NASS_FEATURE_COLS + ["yield_bu_acre_irr"] if c in df.columns]
    ndvi_cols   = [c for c in NDVI_FEATURE_COLS    if c in df.columns]
    gss_cols    = [c for c in GSSURGO_FEATURE_COLS if c in df.columns]
    wx_cols     = [c for c in WEATHER_FEATURE_COLS if c in df.columns]
    dr_cols     = [c for c in DROUGHT_FEATURE_COLS if c in df.columns]
    hls_cols    = [c for c in hls_feature_cols     if c in df.columns]
    ordered = key_cols + target_cols + nass_aux + ndvi_cols + gss_cols + wx_cols + dr_cols + hls_cols
    leftover = [c for c in df.columns if c not in ordered]
    if leftover:
        print(f"  note: keeping {len(leftover)} extra column(s) at end: {leftover}")
    df = df[ordered + leftover]

    # -- Write -------------------------------------------------------------------------
    out_path = args.out
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, compression="snappy")
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nWrote {out_path}  ({size_mb:.2f} MB)")

    # -- QC tail -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("QC TAIL")
    print("=" * 70)

    print(f"\nShape: {df.shape}")
    print(f"\nColumns ({len(df.columns)}):")
    for c in df.columns:
        print(f"  {c}  ({df[c].dtype})")

    print("\nHead (first 6 rows, key cols + target + a feature from each source):")
    spot_cols = key_cols + ["yield_target"]
    if nass_aux: spot_cols.append(nass_aux[0])
    if ndvi_cols: spot_cols.append(ndvi_cols[0])
    if gss_cols:  spot_cols.append(gss_cols[0])
    if wx_cols:   spot_cols.append(wx_cols[0])
    if dr_cols:   spot_cols.append(dr_cols[0])
    if hls_cols:  spot_cols.append(hls_cols[0])
    print(df[spot_cols].head(6).to_string(index=False))

    print("\nNull counts per column:")
    nulls = df.isna().sum()
    for c in df.columns:
        n = int(nulls[c])
        pct = 100.0 * n / len(df) if len(df) else 0.0
        print(f"  {c:30s} {n:>8,}  ({pct:5.1f}%)")

    print("\nCoverage by year (row count, distinct GEOIDs, % yield_target non-null):")
    yr_grp = df.groupby("year").agg(
        rows=("GEOID", "size"),
        geoids=("GEOID", "nunique"),
        yield_pct=("yield_target", lambda s: 100.0 * s.notna().mean()),
    )
    print(yr_grp.to_string())

    print("\nCoverage by forecast_date (row count, % yield_target non-null):")
    fd_grp = df.groupby("forecast_date", observed=True).agg(
        rows=("GEOID", "size"),
        yield_pct=("yield_target", lambda s: 100.0 * s.notna().mean()),
    )
    print(fd_grp.to_string())

    print("\nCoverage by state (row count, distinct GEOIDs):")
    st_grp = df.groupby("state_alpha").agg(
        rows=("GEOID", "size"),
        geoids=("GEOID", "nunique"),
    )
    print(st_grp.to_string())

    # Full-coverage rows: non-null target + at least one non-null per source group.
    def has_any(cols):
        if not cols: return pd.Series(True, index=df.index)
        return df[cols].notna().any(axis=1)

    full_cov = (
        df["yield_target"].notna()
        & has_any(nass_aux)
        & has_any(ndvi_cols)
        & has_any(gss_cols)
        & has_any(wx_cols)
        & has_any(dr_cols)
    )
    print(f"\nRows with FULL feature coverage "
          f"(non-null yield_target + any-non-null in NASS/NDVI/gSSURGO/weather/drought): "
          f"{int(full_cov.sum()):,} / {len(df):,} ({100*full_cov.mean():.1f}%)")
    if hls_cols:
        full_cov_with_hls = full_cov & has_any(hls_cols)
        print(f"  with HLS too: {int(full_cov_with_hls.sum()):,} "
              f"({100*full_cov_with_hls.mean():.1f}%)")
    else:
        print("  (HLS not yet integrated — see Phase D.1.)")

    # Sanity asserts — fail loudly if the merge produced something wrong.
    assert df[["GEOID", "year", "forecast_date"]].duplicated().sum() == 0, \
        "Duplicate (GEOID, year, forecast_date) keys in output!"
    assert df["GEOID"].str.len().eq(5).all(), "GEOID is not 5-char zero-padded everywhere!"
    assert set(df["forecast_date"].unique()) <= set(FORECAST_DATES), \
        f"Unexpected forecast_date values: {set(df['forecast_date'].unique()) - set(FORECAST_DATES)}"
    print("\nAsserts passed: keys unique; GEOID padding correct; forecast_date values valid.")


if __name__ == "__main__":
    main()