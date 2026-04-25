"""
scripts/weather_features.py — build per-(GEOID, year, forecast_date) corn-yield
weather features from the gridMET county-day parquet.

Forecast dates: "08-01", "09-01", "10-01", "EOS"  (EOS = 11-30 = Nov 30)

Features (all derived from May 1 -> cutoff_date_for_year, inclusive of May 1):

  gdd_cum_f50_c86       Cumulative growing degree days, base 50F cap 86F.
                        Both tmin and tmax clamped to [50, 86] before averaging
                        (McMaster & Wilhelm / NDAWN / Iowa State Extension
                        convention -- NOT the raw `(tavg - 50)` variant).
  edd_hours_gt86f       Sinusoidal-interpolated degree-hours above 86F. Heat
                        stress, computed via single-sine integral over each
                        daily arc (Allen 1976 / Baskerville-Emin 1969).
  edd_hours_gt90f       Same construction, threshold 90F.
  vpd_kpa_veg           Mean daily VPD over vegetative window (DOY 152-195),
                        clipped to cutoff_doy.
  vpd_kpa_silk          Mean daily VPD over silking window (DOY 196-227),
                        clipped to cutoff_doy.
  vpd_kpa_grain         Mean daily VPD over grain-fill window (DOY 228-273),
                        clipped to cutoff_doy. NaN at 08-01 (cutoff DOY 213
                        precedes window start at 228); populated at 09-01+.
  prcp_cum_mm           Cumulative precipitation, May 1 -> cutoff (inclusive).
  dry_spell_max_days    Longest run of consecutive days with prcp < 2 mm/day,
                        May 1 -> cutoff.
  srad_total_veg        Sum daily srad (MJ/m^2) over vegetative phase, clipped.
  srad_total_silk       Sum daily srad over silking, clipped.
  srad_total_grain      Sum daily srad over grain fill, clipped. NaN at 08-01;
                        populated at 09-01+.

AS-OF RULE: build_features_for_cutoff(df, year, cutoff_date) slices the daily
df ONCE at the top by `date <= cutoff_date`. Nothing downstream can see
post-cutoff data. Phase-window slices are additionally clipped to cutoff_doy
so e.g. on 08-01 the silking aggregate only includes DOYs <= 213, not the
full silking window.

Cutoff convention: data with timestamp <= cutoff is allowed (inclusive). The
"strictly before forecast_date" wording in the project spec is implemented
by treating the cutoff date itself as the morning-of (the day before the
forecast_date is the last "observable" day). Concretely, August 1 forecast =
data through July 31 inclusive, which is `cutoff_date = July 31`. We store
cutoffs as the forecast_date day itself with `<` semantics; for simplicity
the implementation uses `<= (cutoff_date - 1)` -- equivalent.

Usage:
  python scripts/weather_features.py
  python scripts/weather_features.py --in scripts/gridmet_county_daily_2005_2025.parquet
  python scripts/weather_features.py --in <parquet> --out <csv>
"""

from __future__ import annotations

import argparse
import datetime as dt
import os

import numpy as np
import pandas as pd


# --- Config ----------------------------------------------------------------

DEFAULT_IN  = "scripts/gridmet_county_daily_2005_2025.parquet"
DEFAULT_OUT = "scripts/weather_county_features.csv"

# Forecast date -> (month, day) of the *forecast issuance*. Data through the
# previous day (inclusive) is allowed. EOS is 11-30 (post-harvest reconciliation).
FORECAST_DATES = {
    "08-01": (8, 1),
    "09-01": (9, 1),
    "10-01": (10, 1),
    "EOS":   (11, 30),
}

# Corn phenology windows by day-of-year (Corn Belt convention).
PHASE_VEG  = (152, 195)   # vegetative
PHASE_SILK = (196, 227)   # silking
PHASE_GRAIN = (228, 273)  # grain fill

# GDD bounds, Fahrenheit.
GDD_BASE_F = 50.0
GDD_CAP_F  = 86.0

# Heat-stress thresholds, Fahrenheit.
EDD_THRESH_86F = 86.0
EDD_THRESH_90F = 90.0

# Dry-day cutoff for dry-spell run computation.
DRY_DAY_MM = 2.0

# Season window for cumulative aggregations (DOY May 1 = 121 in non-leap years;
# for leap years the day-of-year shift means May 1 is DOY 122. Pandas
# `.dt.dayofyear` already accounts for leap year, so May 1 == DOY 122 in
# leap years and DOY 121 otherwise. We use the calendar-date (May 1) to
# slice, not a fixed DOY, to be leap-year safe).
SEASON_START_MONTH = 5
SEASON_START_DAY = 1


# --- Conversion helpers ----------------------------------------------------

def c_to_f(t_c: pd.Series) -> pd.Series:
    return t_c * 9.0 / 5.0 + 32.0


# --- GDD F50/F86 (capped) --------------------------------------------------

def gdd_daily_capped(tmin_c: pd.Series, tmax_c: pd.Series) -> pd.Series:
    """Per-day GDD, base 50F cap 86F. Both endpoints clamped to [50, 86]
    BEFORE averaging (McMaster & Wilhelm / NDAWN convention).

    Returns a Series of per-day GDD values >= 0.
    """
    tmin_f = c_to_f(tmin_c).clip(GDD_BASE_F, GDD_CAP_F)
    tmax_f = c_to_f(tmax_c).clip(GDD_BASE_F, GDD_CAP_F)
    gdd = (tmin_f + tmax_f) / 2.0 - GDD_BASE_F
    return gdd.clip(lower=0.0)


# --- EDD (degree-hours above threshold via single-sine integral) -----------
#
# Single-sine model: temperature is approximated as a half-sine arc between
# tmin (sunrise) and tmax (solar noon), repeated symmetrically afternoon
# tmin -> tmax -> tmin (next sunrise). The closed-form integral of the
# excess area above threshold T in degrees-hours per day:
#
#   if tmax <= T:        0
#   elif tmin >= T:      24 * ((tmin + tmax)/2 - T)
#   else (T crossed):    24/pi * [ (mean - T)*(pi/2 - theta) + amplitude*cos(theta) ]
#       where mean      = (tmin + tmax)/2
#             amplitude = (tmax - tmin)/2
#             theta     = arcsin( (T - mean)/amplitude )

def degree_hours_above_threshold(
    tmin_c: pd.Series,
    tmax_c: pd.Series,
    threshold_f: float,
) -> pd.Series:
    """Daily degree-hours above `threshold_f` Fahrenheit. Vectorized.

    Returns a Series of per-day degree-hours (>= 0).
    """
    # Convert all to Fahrenheit; the model is unit-agnostic but we want the
    # integration to be in F-degree-hours.
    tmin_f = c_to_f(tmin_c).to_numpy(dtype=np.float64)
    tmax_f = c_to_f(tmax_c).to_numpy(dtype=np.float64)
    T = float(threshold_f)

    n = len(tmin_f)
    out = np.zeros(n, dtype=np.float64)

    # Case 1: entire day below threshold -> 0 (already)
    # Case 2: entire day above threshold
    case2 = tmin_f >= T
    out[case2] = 24.0 * ((tmin_f[case2] + tmax_f[case2]) / 2.0 - T)

    # Case 3: threshold crossed (tmin < T < tmax)
    case3 = (~case2) & (tmax_f > T)
    if case3.any():
        tmn = tmin_f[case3]
        tmx = tmax_f[case3]
        mean = (tmn + tmx) / 2.0
        amp  = (tmx - tmn) / 2.0
        # Numerical guard: if amp is tiny (< 1e-9), just set 0; otherwise
        # arcsin can blow up.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (T - mean) / amp
            ratio = np.clip(ratio, -1.0, 1.0)
            theta = np.arcsin(ratio)
        # Integrand: 24/pi * [ (mean - T)*(pi/2 - theta) + amp*cos(theta) ]
        dh = (24.0 / np.pi) * (
            (mean - T) * (np.pi / 2.0 - theta) + amp * np.cos(theta)
        )
        # Floor at 0 against any tiny negative numerical noise
        dh = np.maximum(dh, 0.0)
        out[case3] = dh

    return pd.Series(out, index=tmin_c.index)


# --- Dry-spell run --------------------------------------------------------

def longest_dry_run(prcp_mm: pd.Series, threshold_mm: float = DRY_DAY_MM) -> int:
    """Longest consecutive run of days with prcp < threshold_mm."""
    arr = prcp_mm.to_numpy(dtype=np.float64)
    if len(arr) == 0:
        return 0
    dry = (arr < threshold_mm).astype(np.int8)
    # Standard run-length max via cumsum-with-reset trick.
    max_run = 0
    cur = 0
    for v in dry:
        if v:
            cur += 1
            if cur > max_run:
                max_run = cur
        else:
            cur = 0
    return int(max_run)


# --- Phase-window slice (cutoff-clipped) ----------------------------------

def phase_slice(season_df: pd.DataFrame, doy_lo: int, doy_hi: int,
                cutoff_doy: int) -> pd.DataFrame:
    """Return rows of `season_df` whose `doy` is in [doy_lo, min(doy_hi, cutoff_doy)].

    `season_df` must have a `doy` column.
    """
    hi = min(doy_hi, cutoff_doy)
    if hi < doy_lo:
        return season_df.iloc[0:0]
    return season_df[(season_df["doy"] >= doy_lo) & (season_df["doy"] <= hi)]


# --- Per-(county, year, cutoff) feature builder ---------------------------

def build_features_for_cutoff(
    df_county: pd.DataFrame,
    year: int,
    cutoff_date: dt.date,
) -> dict:
    """Build feature dict for one (county, year, cutoff_date).

    df_county must have all years for the GEOID (will be filtered here).
    Columns required: date, tmin_c, tmax_c, prcp_mm, srad_mjm2, vpd_kpa.

    The single point of leakage control is the `mask = (date <= cutoff)`
    line below. Nothing downstream looks past `cutoff_date`.
    """
    # Coerce date to pandas Timestamp for safe comparison. Both object-dtype
    # (from python date) and datetime64 work after conversion.
    date_series = pd.to_datetime(df_county["date"])
    season_start = pd.Timestamp(year, SEASON_START_MONTH, SEASON_START_DAY)
    cutoff_ts = pd.Timestamp(cutoff_date) - pd.Timedelta(days=1)
    # cutoff_ts = the last observable day (data with timestamp == forecast_date
    # itself is excluded; "strictly before forecast_date" semantics).

    mask = (date_series.dt.year == year) & (date_series >= season_start) & (date_series <= cutoff_ts)
    season = df_county.loc[mask].copy()
    if len(season) == 0:
        # Could happen for cutoffs before May 1 (we never set such cutoffs)
        # or for years with no daily data.
        return {
            "gdd_cum_f50_c86":   np.nan,
            "edd_hours_gt86f":   np.nan,
            "edd_hours_gt90f":   np.nan,
            "vpd_kpa_veg":       np.nan,
            "vpd_kpa_silk":      np.nan,
            "vpd_kpa_grain":     np.nan,
            "prcp_cum_mm":       np.nan,
            "dry_spell_max_days": np.nan,
            "srad_total_veg":    np.nan,
            "srad_total_silk":   np.nan,
            "srad_total_grain":  np.nan,
        }

    # Add doy.
    season = season.assign(doy=pd.to_datetime(season["date"]).dt.dayofyear)
    cutoff_doy = (cutoff_ts.date() - dt.date(year, 1, 1)).days + 1

    # --- Cumulative aggregates (May 1 -> cutoff) --------------------------
    gdd_d = gdd_daily_capped(season["tmin_c"], season["tmax_c"])
    gdd_cum = float(gdd_d.sum())

    dh86 = degree_hours_above_threshold(season["tmin_c"], season["tmax_c"], EDD_THRESH_86F)
    dh90 = degree_hours_above_threshold(season["tmin_c"], season["tmax_c"], EDD_THRESH_90F)
    edd86_cum = float(dh86.sum())
    edd90_cum = float(dh90.sum())

    prcp_cum = float(season["prcp_mm"].sum())
    dry_spell = longest_dry_run(season["prcp_mm"])

    # --- Phase-window aggregates (clipped to cutoff_doy) ------------------
    veg = phase_slice(season, *PHASE_VEG, cutoff_doy)
    silk = phase_slice(season, *PHASE_SILK, cutoff_doy)
    grain = phase_slice(season, *PHASE_GRAIN, cutoff_doy)

    def _mean(s):
        return float(s.mean()) if len(s) else np.nan

    def _sum(s):
        return float(s.sum()) if len(s) else np.nan

    return {
        "gdd_cum_f50_c86":     gdd_cum,
        "edd_hours_gt86f":     edd86_cum,
        "edd_hours_gt90f":     edd90_cum,
        "vpd_kpa_veg":         _mean(veg["vpd_kpa"]),
        "vpd_kpa_silk":        _mean(silk["vpd_kpa"]),
        "vpd_kpa_grain":       _mean(grain["vpd_kpa"]),
        "prcp_cum_mm":         prcp_cum,
        "dry_spell_max_days":  float(dry_spell),
        "srad_total_veg":      _sum(veg["srad_mjm2"]),
        "srad_total_silk":     _sum(silk["srad_mjm2"]),
        "srad_total_grain":    _sum(grain["srad_mjm2"]),
    }


# --- Main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", default=DEFAULT_IN,
                    help=f"gridMET county-day parquet (default: {DEFAULT_IN})")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"Output CSV (default: {DEFAULT_OUT})")
    args = ap.parse_args()

    if not os.path.exists(args.in_path):
        raise SystemExit(f"ERROR: input not found: {args.in_path}")

    print(f"Reading gridMET parquet: {args.in_path}")
    df_all = pd.read_parquet(args.in_path)

    # Defensive normalizations.
    df_all["GEOID"] = df_all["GEOID"].astype(str).str.zfill(5)
    df_all["date"] = pd.to_datetime(df_all["date"])

    # Required columns.
    required = {"GEOID", "date", "tmin_c", "tmax_c", "prcp_mm", "srad_mjm2", "vpd_kpa"}
    missing = required - set(df_all.columns)
    if missing:
        raise SystemExit(f"ERROR: input missing columns: {sorted(missing)}")

    print(f"  {len(df_all):,} rows  "
          f"({df_all['date'].min().date()} .. {df_all['date'].max().date()})  "
          f"{df_all['GEOID'].nunique()} GEOIDs")

    geoids = sorted(df_all["GEOID"].unique())
    years = sorted(df_all["date"].dt.year.unique())
    print(f"  GEOIDs: {len(geoids)}, years: {len(years)} ({years[0]}..{years[-1]})")

    # Iterate (GEOID, year, forecast_date).
    total = len(geoids) * len(years) * len(FORECAST_DATES)
    print(f"\nDeriving features for {total:,} (GEOID, year, forecast_date) combos...")

    out_rows = []
    # Group by GEOID once for efficiency (avoid repeated full-table scans).
    by_geoid = df_all.groupby("GEOID", sort=False)
    n_done = 0
    for geoid, df_g in by_geoid:
        for year in years:
            for fd_label, (mm, dd) in FORECAST_DATES.items():
                cutoff = dt.date(int(year), mm, dd)
                feats = build_features_for_cutoff(df_g, int(year), cutoff)
                row = {"GEOID": str(geoid), "year": int(year), "forecast_date": fd_label}
                row.update(feats)
                out_rows.append(row)
                n_done += 1
        if (geoids.index(geoid) + 1) % 50 == 0:
            print(f"  {geoids.index(geoid) + 1}/{len(geoids)} GEOIDs done")

    out = pd.DataFrame(out_rows)
    # Order columns for readability.
    feature_cols = [
        "gdd_cum_f50_c86", "edd_hours_gt86f", "edd_hours_gt90f",
        "vpd_kpa_veg", "vpd_kpa_silk", "vpd_kpa_grain",
        "prcp_cum_mm", "dry_spell_max_days",
        "srad_total_veg", "srad_total_silk", "srad_total_grain",
    ]
    out = out[["GEOID", "year", "forecast_date"] + feature_cols]
    out = out.sort_values(["GEOID", "year", "forecast_date"]).reset_index(drop=True)

    # ---- QC tail ----------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"OUT: {len(out):,} rows  ({out['year'].min()}-{out['year'].max()})")
    print(f"\nCoverage by year:")
    print(out.groupby("year").size().to_string())
    print(f"\nCoverage by forecast_date:")
    print(out.groupby("forecast_date").size().to_string())
    print(f"\nNull counts per feature col:")
    for c in feature_cols:
        nn = int(out[c].isna().sum())
        pct = 100 * nn / len(out) if len(out) else 0
        print(f"  {c:25s} {nn:>6,} ({pct:5.1f}%)")
    print(f"\nExpected: vpd_kpa_grain and srad_total_grain ~25% null "
          f"(structural NaN at 08-01).")

    out.to_csv(args.out, index=False)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
