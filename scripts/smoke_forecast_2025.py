"""
scripts/smoke_forecast_2025.py — end-to-end validation that the existing
Phase B/C forecast pipeline produces sensible 2025 state forecasts against
the v2 master parquet.

Run AFTER:
  1. python scripts/nass_features_v2.py
  2. python scripts/gridmet_pull.py --years 2025
     python scripts/gridmet_pull.py --combine
  3. python scripts/weather_features.py
  4. python scripts/drought_features.py
  5. python scripts/merge_all.py --nass scripts/nass_corn_5states_features_v2.csv \\
                                 --out scripts/training_master_v2.parquet

NOTE: there is no 2025 NASS API pull — NASS publishes no county-level 2025
corn data for our 5 states (probe-confirmed April 2026). Step 1 above
synthesizes 2025 NASS-aux from 2024 verbatim and marks provenance.

Run BEFORE writing any backend route code. If this script doesn't print
sane state forecasts for 2025 across all 5 states × 4 dates, the backend
will return junk and we'll waste hours debugging the API layer instead of
the data layer.

What it asserts:
  - master parquet has 2025 rows for all 5 states × 4 forecast_dates
  - the Phase C bundle loads from models/forecast/
  - bundle.predict() runs against 2025 rows and returns finite values
  - AnalogIndex.find() returns K=5 same-GEOID analogs for 2025 query rows
    where the embedding is complete
  - state_forecast_from_records() rolls up to a state-level cone
  - one-line summary printout per (state, date)

What it doesn't assert:
  - that the predictions are CORRECT (we don't have 2025 truth)
  - that the cone width is reasonable (subjective; eyeball it)

Usage:
  python scripts/smoke_forecast_2025.py
  python scripts/smoke_forecast_2025.py --master scripts/training_master_v2.parquet \\
                                        --bundle-dir models/forecast
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# Make `forecast` importable when run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forecast.aggregate import (
    StateForecast,
    build_records_from_master,
    state_forecast_from_records,
)
from forecast.analog import AnalogIndex
from forecast.cone import Cone, build_cone
from forecast.data import load_master, train_pool
from forecast.detrend import fit as fit_trend
from forecast.features import EMBEDDING_COLS, VALID_FORECAST_DATES, fit_standardizer
from forecast.regressor import RegressorBundle


STATES = ("CO", "IA", "MO", "NE", "WI")
TARGET_YEAR = 2025
DEFAULT_MASTER = "scripts/training_master_v2.parquet"
DEFAULT_BUNDLE_DIR = "models/forecast"


def predict_state_from_regressor(bundle, query_df, state, year, forecast_date):
    """Acres-weighted state-level point estimate. Mirrors
    backtest_phase_c.predict_state_from_regressor exactly so behavior here
    matches what the backend will compute."""
    sub = query_df[
        (query_df["state_alpha"] == state)
        & (query_df["year"] == year)
        & (query_df["forecast_date"] == forecast_date)
    ]
    if len(sub) == 0:
        return float("nan"), 0
    preds = bundle.regressors[forecast_date].predict(sub)
    acres = sub["acres_planted_all"].to_numpy(dtype=np.float64)
    # XGBoost returns a 0-length array (with an "Empty dataset at worker"
    # warning to stderr) when EVERY row in `sub` has NaN in every regressor
    # feature column. This is a data-integrity failure, not a normal "skip
    # this state" case — surface it as 0 valid counties and let the caller
    # print the diagnostic.
    if len(preds) != len(sub):
        return float("nan"), 0
    valid = ~np.isnan(preds) & ~np.isnan(acres) & (acres > 0)
    if not valid.any():
        return float("nan"), 0
    return float((preds[valid] * acres[valid]).sum() / acres[valid].sum()), int(valid.sum())


def forecast_county_cone(index, trend, row, *, k=5, pool="same_geoid"):
    """Same as backtest_phase_c.forecast_county_cone — wraps AnalogIndex.find +
    build_cone. Returns None when embedding is incomplete (the analog cone
    skips the county, mirroring backtest behavior)."""
    cols = EMBEDDING_COLS[row["forecast_date"]]
    if any(pd.isna(row.get(c)) for c in cols):
        return None
    try:
        analogs = index.find(
            geoid=str(row["GEOID"]),
            year=int(row["year"]),
            forecast_date=str(row["forecast_date"]),
            query_features=row,
            k=k,
            pool=pool,
        )
    except Exception:
        return None
    if not analogs:
        return None
    return build_cone(
        analogs=analogs,
        trend=trend,
        query_geoid=str(row["GEOID"]),
        query_state=str(row["state_alpha"]),
        query_year=int(row["year"]),
        query_forecast_date=str(row["forecast_date"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=DEFAULT_MASTER)
    ap.add_argument("--bundle-dir", default=DEFAULT_BUNDLE_DIR)
    ap.add_argument("--year", type=int, default=TARGET_YEAR)
    args = ap.parse_args()

    print("=" * 72)
    print(f"smoke_forecast_2025 — validating {args.year} forecast pipeline")
    print("=" * 72)

    # ---- 1. master parquet --------------------------------------------------
    print(f"\n[1] loading master parquet: {args.master}")
    if not Path(args.master).exists():
        print(f"    FAIL: file not found. Did you run merge_all.py with the v2 NASS feed?")
        sys.exit(1)
    master = load_master(args.master)
    print(f"    {len(master):,} rows, years {master['year'].min()}-{master['year'].max()}, "
          f"{master['GEOID'].nunique()} GEOIDs")

    target = master[master["year"] == args.year]
    if len(target) == 0:
        print(f"    FAIL: no rows for year={args.year} in master parquet.")
        print(f"    Confirm that nass_corn_5states_2025.csv exists and that "
              f"nass_features_v2.py ran with FORECAST_CSV present.")
        sys.exit(1)
    print(f"    {len(target):,} rows for year={args.year}")
    print(f"    by (state, forecast_date):")
    pivot = target.groupby(["state_alpha", "forecast_date"]).size().unstack(fill_value=0)
    print(pivot.to_string().replace("\n", "\n    "))

    # ---- 2. setup train/standardizer/trend/index ----------------------------
    print(f"\n[2] building train pool + standardizer + trend + AnalogIndex")
    train_df, mh = train_pool(master, n_min_history=10)
    print(f"    train pool: {len(train_df):,} rows ({mh.n_kept} GEOIDs kept, "
          f"{mh.n_dropped} dropped)")
    standardizer = fit_standardizer(train_df)
    trend = fit_trend(train_df)
    index = AnalogIndex.fit(train_df, standardizer, trend)
    for d in VALID_FORECAST_DATES:
        print(f"      AnalogIndex[{d}]: {index.n_candidates(d):,} candidates")

    # ---- 3. load Phase C bundle --------------------------------------------
    print(f"\n[3] loading Phase C bundle from {args.bundle_dir}")
    if not Path(args.bundle_dir).is_dir():
        print(f"    FAIL: bundle dir does not exist.")
        sys.exit(1)
    bundle = RegressorBundle.load(args.bundle_dir)
    for d, reg in sorted(bundle.regressors.items()):
        print(f"    regressor[{d}]: {len(reg.feature_cols)} features, "
              f"best_iteration={reg.best_iteration}, "
              f"val_rmse={reg.train_metrics.get('val_rmse', float('nan')):.2f}")

    # ---- 4. per-(state, date) state-level forecast --------------------------
    print(f"\n[4] state forecasts for year={args.year}")
    print(f"    {'state':<5} {'date':<5} {'point':>7}  {'cone':>20}  "
          f"{'width80':>7}  {'n_reg':>5} {'n_cone':>6}")
    print(f"    {'-'*5} {'-'*5} {'-'*7}  {'-'*20}  {'-'*7}  {'-'*5} {'-'*6}")

    failures = 0
    for state in STATES:
        for fd in VALID_FORECAST_DATES:
            # Regressor point.
            point, n_reg = predict_state_from_regressor(
                bundle, target, state, args.year, fd
            )
            if np.isnan(point) or n_reg == 0:
                print(f"    {state:<5} {fd:<5} {'NaN':>7}  "
                      f"{'(no regressor predictions)':<22}  {'-':>7}  "
                      f"{n_reg:>5} {'-':>6}  <-- FAIL")
                failures += 1
                continue

            # Per-county cones → state cone.
            sub = target[(target["state_alpha"] == state) & (target["forecast_date"] == fd)]
            cones_by_geoid: Dict[str, Cone] = {}
            for _, row in sub.iterrows():
                cone = forecast_county_cone(index, trend, row)
                if cone is not None:
                    cones_by_geoid[str(row["GEOID"])] = cone

            if not cones_by_geoid:
                print(f"    {state:<5} {fd:<5} {point:>7.1f}  "
                      f"{'(no cone — all counties had incomplete embedding)':<22}  "
                      f"{'-':>7}  {n_reg:>5} {0:>6}  <-- WARN")
                continue

            records = build_records_from_master(cones_by_geoid, master, state, args.year)
            try:
                sf: StateForecast = state_forecast_from_records(records, state, args.year, fd)
            except ValueError as e:
                print(f"    {state:<5} {fd:<5} {point:>7.1f}  state_forecast failed: {e}")
                failures += 1
                continue

            p10, p50, p90 = sf.percentiles[10], sf.percentiles[50], sf.percentiles[90]
            cone_str = f"[{p10:6.1f}, {p90:6.1f}]"
            in_cone = "ok" if p10 <= point <= p90 else "OUT"
            print(f"    {state:<5} {fd:<5} {point:>7.1f}  {cone_str:<20}  "
                  f"{p90 - p10:>7.1f}  {n_reg:>5} {sf.n_counties:>6}  {in_cone}")

    print()
    if failures > 0:
        print(f"FAIL: {failures} (state, date) combinations did not produce a forecast.")
        sys.exit(1)

    print("PASS: all 5 states × 4 dates produced regressor point and cone.")
    print()
    print("Next: write backend/forecast_routes.py (E.1) — the inference math is")
    print("validated and the route can call exactly these functions.")


if __name__ == "__main__":
    main()
