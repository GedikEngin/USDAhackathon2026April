"""
scripts/nass_features_v2.py — NASS engineered features, 2005-2025.

This file REPLACES the prior version. The prior version expected to read
`nass_corn_5states_2025.csv` produced by an API pull; that pull was
abandoned after a probe (April 25, 2026) confirmed NASS publishes no
county-level 2025 corn data for any of our 5 states (CO/IA/MO/NE/WI). The
state-level data NASS does publish for 2025 (weekly CONDITION/PROGRESS
ratings, prices) is not in our feature set.

What we do instead: SYNTHESIZE 2025 NASS-aux rows by carrying the 2024 NASS
feature values forward per county. The Phase C regressor's NASS-aux
features (acres_planted_all, irrigated_share, harvest_ratio) are
structural management priors — "this is how this county typically
operates." Last year's value is a defensible prior; the regressor's
predictive signal for 2025 comes from the IN-SEASON inputs (weather,
drought, soil), all of which ARE real for 2025.

Provenance column on every output row:
  nass_aux_provenance ∈ {"reported", "prior_year"}
    "reported"   = real NASS data for that (GEOID, year)
    "prior_year" = synthesized from the previous year's row;
                   used only for forecast years (2025+)

When NASS eventually publishes 2025 yield (~Jan 2027 in their normal
cycle), this file's logic flips: 2025 rows become "reported" and a new
forecast year (2026) becomes "prior_year". One-line CLI override of
`--forecast-years` covers that.

Inputs:
  scripts/nass_corn_5states_features.csv    canonical 2005-2024 features
  --forecast-years 2025                     comma-separated list of years
                                            to synthesize from previous year

Output:
  scripts/nass_corn_5states_features_v2.csv

Usage:
  python scripts/nass_features_v2.py
  python scripts/nass_features_v2.py --forecast-years 2025
  python scripts/nass_features_v2.py --forecast-years 2025,2026
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

DEFAULT_HISTORICAL_FEATURES = "scripts/nass_corn_5states_features.csv"
DEFAULT_OUT = "scripts/nass_corn_5states_features_v2.csv"
DEFAULT_FORECAST_YEARS = (2025,)


def _prior_year_for(forecast_year: int) -> int:
    """Default mapping forecast_year -> prior_year for the carry-forward.
    forecast_year=2025 -> 2024. Override at the call site if a year is
    missing for a particular county."""
    return forecast_year - 1


def synthesize_forecast_year(
    historical_feat: pd.DataFrame,
    forecast_year: int,
    prior_year: int = None,
) -> pd.DataFrame:
    """Create a 2025-style row per county by copying the prior_year row,
    flipping year + yield_target, marking provenance.

    Parameters
    ----------
    historical_feat : DataFrame
        Output of nass_features.py (the v1 file). Must have columns:
        GEOID, year, state_alpha, county_name, yield_target,
        irrigated_share, harvest_ratio, acres_harvested_all,
        acres_planted_all, yield_bu_acre_irr.
    forecast_year : int
        The year to synthesize (e.g. 2025).
    prior_year : int, optional
        The historical year to copy from. Defaults to forecast_year - 1.

    Returns
    -------
    DataFrame
        One row per county that had a prior_year row, with:
        - year set to forecast_year
        - yield_target set to NaN (we are FORECASTING this)
        - all other NASS-aux columns copied from prior_year
        - nass_aux_provenance = "prior_year"
    """
    if prior_year is None:
        prior_year = _prior_year_for(forecast_year)

    src = historical_feat[historical_feat["year"] == prior_year].copy()
    if len(src) == 0:
        raise ValueError(
            f"No rows in historical features for prior_year={prior_year}; "
            f"cannot synthesize forecast_year={forecast_year}. "
            f"Available years: {sorted(historical_feat['year'].unique())}"
        )

    # Build forecast-year rows. Copy everything, then mutate year + yield_target.
    fc = src.copy()
    fc["year"] = forecast_year
    fc["yield_target"] = np.nan
    fc["nass_aux_provenance"] = "prior_year"

    return fc


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--historical",
        default=DEFAULT_HISTORICAL_FEATURES,
        help=f"Path to v1 NASS features CSV (default: {DEFAULT_HISTORICAL_FEATURES})",
    )
    ap.add_argument(
        "--forecast-years",
        default=",".join(str(y) for y in DEFAULT_FORECAST_YEARS),
        help="Comma-separated forecast years to synthesize. Default: 2025",
    )
    ap.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"Output CSV path (default: {DEFAULT_OUT})",
    )
    args = ap.parse_args()

    forecast_years = tuple(
        int(y) for y in args.forecast_years.split(",") if y.strip()
    )

    # ---- load historical ---------------------------------------------------
    if not os.path.exists(args.historical):
        print(f"ERROR: historical features file not found: {args.historical}")
        print("Run scripts/nass_features.py first to produce the 2005-2024 file.")
        sys.exit(1)

    print(f"Reading historical features: {args.historical}")
    hist = pd.read_csv(args.historical)
    hist["GEOID"] = hist["GEOID"].astype(str).str.zfill(5)
    print(f"  {len(hist):,} rows  "
          f"({hist['year'].min()}-{hist['year'].max()}, "
          f"{hist['GEOID'].nunique()} GEOIDs)")

    # Every historical row is 'reported' provenance.
    hist["nass_aux_provenance"] = "reported"

    available = set(hist["year"].unique())
    print(f"  available years: {sorted(available)}")

    # ---- synthesize each forecast year -------------------------------------
    all_frames = [hist]
    for fy in forecast_years:
        py = _prior_year_for(fy)
        if py not in available:
            print(f"\nERROR: forecast_year={fy} requires prior_year={py}, "
                  f"but {py} is not in the historical file.")
            sys.exit(1)
        print(f"\nSynthesizing forecast_year={fy} from prior_year={py}...")
        fc = synthesize_forecast_year(hist, fy, prior_year=py)
        print(f"  {len(fc):,} forecast rows  "
              f"(yield_target=NaN by design; NASS-aux carried from {py})")
        all_frames.append(fc)

    # ---- combine -----------------------------------------------------------
    out = pd.concat(all_frames, ignore_index=True)
    out = out.sort_values(["GEOID", "year"]).reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f"OUT: {len(out):,} total rows  "
          f"({out['year'].min()}-{out['year'].max()})")
    print("  by year:")
    for y in sorted(out["year"].unique()):
        sub = out[out["year"] == y]
        n = len(sub)
        nn_yield = int(sub["yield_target"].notna().sum())
        prov = sub["nass_aux_provenance"].iloc[0] if n else "?"
        marker = " <- forecast (yield NaN)" if nn_yield == 0 else ""
        print(f"    {y}: {n:>4} rows, {nn_yield} with yield, "
              f"provenance={prov}{marker}")

    for fy in forecast_years:
        print(f"\n  {fy} county counts by state:")
        sub = out[out["year"] == fy]
        for st, n in sub["state_alpha"].value_counts().sort_index().items():
            print(f"    {st}: {n} counties")

    out.to_csv(args.out, index=False)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
