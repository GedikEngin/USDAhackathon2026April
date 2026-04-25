"""
Build per-(GEOID, year, forecast_date) corn-yield weather features from the
combined gridMET daily county parquet.

Forecast dates: "08-01", "09-01", "10-01", "EOS"  (EOS = 11-30 = Nov 30)

Features (all derived from May 1 -> cutoff_date_for_year, inclusive of May 1):
  gdd_cum_f50_c86       Growing degree days, base 50F cap 86F (Fahrenheit)
  edd_hours_gt86f       Sinusoidal-interpolated hours above 86F  (heat stress)
  edd_hours_gt90f       Sinusoidal-interpolated hours above 90F  (severe stress)
  vpd_kpa_veg           Mean daily VPD over vegetative phase (DOY 152-195)
  vpd_kpa_silk          Mean daily VPD over silking            (DOY 196-227)
  vpd_kpa_grain         Mean daily VPD over grain fill         (DOY 228-273)
  prcp_cum_mm           Cumulative precipitation, May 1 -> cutoff
  dry_spell_max_days    Longest run of <2mm/day, May 1 -> cutoff
  srad_total_veg        Sum daily srad (MJ/m2) over vegetative phase
  srad_total_silk       Sum daily srad (MJ/m2) over silking
  srad_total_grain      Sum daily srad (MJ/m2) over grain fill

AS-OF RULE: the function build_features_for_cutoff(df, year, cutoff_date) takes
the cutoff date as an argument and slices the daily df ONCE at the top before
deriving anything. By construction nothing downstream of that slice can see
post-cutoff data. Phase-window features are themselves clipped to the cutoff
(e.g., on 08-01 the silking-phase aggregate only includes DOYs <= 213).

Usage:
  python weather_features.py
  python weather_features.py --in scripts/gridmet_county_daily_2005_2024.parquet
                             --out scripts/weather_county_features.csv
"""

import argparse
import datetime as dt

import numpy as np
import pandas as pd

# --- Config ----------------------------------------------------

DEFAULT_IN  = "scripts/gridmet_county_daily_2005_2024.parquet"
DEFAULT_OUT = "scripts/weather_county_features.csv"

# Forecast-date suffixes (strings) -> (month, day) for the cutoff in each year Y.
# Cutoff is INCLUSIVE: data with timestamp <= cutoff is allowed.
# Strictly-before-D wording in the spec is implemented by adding 1 day; here we
# use cutoff = the morning of D, so the data slice is days < D, i.e. <= (D - 1).
FORECAST_DATES = {
    "08-01": (8,  1),
    "09-01": (9,  1),
    "10-01": (10, 1),
    "EOS":   (11, 30),
}

# Phenology windows (DOY ranges, inclusive)
DOY_PLANT_START = 121   # May 1
DOY_VEG     = (152, 195)  # Jun 1  - Jul 14
DOY_SILK    = (196, 227)  # Jul 15 - Aug 15
DOY_GRAIN   = (228, 273)  # Aug 16 - Sep 30

# Heat-stress thresholds (Fahrenheit -> Celsius)
F86 = (86 - 32) * 5.0 / 9.0   # 30.000 C
F90 = (90 - 32) * 5.0 / 9.0   # 32.222 C

# GDD (Fahrenheit) base/cap (industry standard for corn).
GDD_BASE_F = 50.0
GDD_CAP_F  = 86.0

# Dry-spell threshold
DRY_DAY_MM = 2.0


# --- Conversions -----------------------------------------------

def c_to_f(c):
    return c * 9.0 / 5.0 + 32.0


# --- GDD (Fahrenheit, base 50, cap 86, capped MIN as well) -----

def gdd_daily_capped(tmin_c, tmax_c):
    """
    Per-day GDD F50/F86. Both tmin and tmax are clamped to [50, 86] F before
    averaging. This is the standard corn GDD calculation (Mcmaster & Wilhelm,
    NDAWN, etc.), not the raw (tavg - 50) variant.
    """
    tmin_f = c_to_f(tmin_c).clip(GDD_BASE_F, GDD_CAP_F)
    tmax_f = c_to_f(tmax_c).clip(GDD_BASE_F, GDD_CAP_F)
    gdd = (tmin_f + tmax_f) / 2.0 - GDD_BASE_F
    return gdd.clip(lower=0.0)


# --- EDD/KDD via single-sine hourly interpolation --------------
#
# Allen 1976 / Baskerville-Emin 1969 method. For each day, we model temperature
# as a half-sine between tmin (at sunrise, treated as t=0) and tmax (at solar
# noon, t=pi). The fraction of the day above a threshold T is then computable
# in closed form. Returns degree-hours above the threshold.
#
# Hours above T over one day:
#   if tmax <= T:   0
#   elif tmin >= T: 24 * (mean - T)  where mean = (tmin + tmax) / 2 above T
#                                    -> approximate as 24*(mean - T) for hours,
#                                       but here we want hours, so the integral
#                                       reduces to: 24 hrs above, with mean
#                                       excess (tmin+tmax)/2 - T  -> degree-hours
#                                       = 24 * ((tmin+tmax)/2 - T)
#   else:           closed-form sine integral
#
# References: Snyder 1985 "Hand calculating degree days"; Roltsch et al 1999.

def degree_hours_above_threshold(tmin_c, tmax_c, threshold_c):
    """
    Return per-day degree-hours above `threshold_c`, vectorized.
    tmin_c, tmax_c: pandas Series (same index), Celsius.
    threshold_c: scalar, Celsius.
    """
    tmn = tmin_c.to_numpy(dtype=np.float64)
    tmx = tmax_c.to_numpy(dtype=np.float64)
    T   = float(threshold_c)
    out = np.zeros_like(tmn)

    # Case 1: entire day below threshold
    below = tmx <= T
    # already 0

    # Case 2: entire day above threshold (rare, but possible during heat waves)
    above = tmn >= T
    out[above] = 24.0 * ((tmn[above] + tmx[above]) / 2.0 - T)

    # Case 3: threshold crosses (the typical heat-stress case)
    cross = ~below & ~above
    if cross.any():
        a = (tmx[cross] - tmn[cross]) / 2.0           # amplitude
        m = (tmx[cross] + tmn[cross]) / 2.0           # mean
        # theta = arcsin((T - m) / a) in [-pi/2, pi/2] is the angle where
        # the sine equals T. Hours above T over the half-day arc 0..pi
        # (representing the 12-hour daylight-warming arc) is symmetric, but
        # the standard daily approximation distributes the warming over 24h
        # by assuming the temperature trace is one full sine cycle per day.
        # We integrate the positive part (sine_value - T) over [0, 2*pi].
        #
        # Closed form for degree-hours above threshold under single-sine:
        #   DH = (24 / pi) * [ a * cos(theta) - (T - m) * (pi/2 - theta) ]
        # where theta = arcsin((T - m) / a).
        ratio = (T - m) / a
        ratio = np.clip(ratio, -1.0, 1.0)
        theta = np.arcsin(ratio)
        dh = (24.0 / np.pi) * (a * np.cos(theta) - (T - m) * (np.pi / 2.0 - theta))
        # Numerical edge: if a is tiny (tmin ~ tmax), force 0 to avoid noise
        dh = np.where(a < 1e-6, 0.0, dh)
        dh = np.maximum(dh, 0.0)
        out[cross] = dh

    return out  # numpy array


# --- Dry spell -------------------------------------------------

def longest_dry_run(prcp_mm_series, threshold_mm=DRY_DAY_MM):
    """Longest consecutive run of days with prcp < threshold_mm."""
    if len(prcp_mm_series) == 0:
        return 0
    dry = (prcp_mm_series.to_numpy() < threshold_mm).astype(np.int8)
    if dry.sum() == 0:
        return 0
    # run lengths via diff trick
    # add sentinel zeros at both ends so every run is bounded
    padded = np.concatenate(([0], dry, [0]))
    diffs = np.diff(padded)
    run_starts = np.where(diffs == 1)[0]
    run_ends   = np.where(diffs == -1)[0]
    return int((run_ends - run_starts).max()) if len(run_starts) else 0


# --- Phase windows respecting cutoff ---------------------------

def phase_slice(df_year_county, doy_lo, doy_hi, cutoff_doy):
    """
    Return rows in df_year_county whose 'doy' is in [doy_lo, min(doy_hi, cutoff_doy)].
    Empty df if cutoff is before the phase start.
    """
    hi = min(doy_hi, cutoff_doy)
    if hi < doy_lo:
        return df_year_county.iloc[0:0]
    return df_year_county[(df_year_county["doy"] >= doy_lo)
                          & (df_year_county["doy"] <= hi)]


# --- Main feature builder --------------------------------------

def build_features_for_cutoff(df_county, year, cutoff_date):
    """
    df_county: DataFrame for a single GEOID, ALL years (will be filtered here).
    year: int, the target year.
    cutoff_date: datetime.date, the as-of date. Data with date <= cutoff_date is allowed.

    Returns a dict of feature values for this (GEOID, year, cutoff_date).
    AS-OF SAFETY: the slice happens on the very next line. Nothing below it
    has access to post-cutoff data.
    """
    # --- AS-OF SLICE (single point of leakage control) ---
    mask_year = df_county["date"].dt.year == year
    mask_cut  = df_county["date"].dt.date <= cutoff_date
    df = df_county.loc[mask_year & mask_cut].copy()
    if df.empty:
        return None
    df = df.sort_values("date")
    df["doy"] = df["date"].dt.dayofyear
    cutoff_doy = (cutoff_date - dt.date(year, 1, 1)).days + 1

    # Restrict the May-1-onward window
    season = df[df["doy"] >= DOY_PLANT_START]
    if season.empty:
        return None

    # GDD cumulative
    gdd_daily = gdd_daily_capped(season["tmin_c"], season["tmax_c"])
    gdd_cum = float(gdd_daily.sum())

    # EDD/KDD (degree-hours above 86F and 90F) - sum over season.
    # Use ALL season days for heat stress; this captures every hot day so far.
    dh86 = degree_hours_above_threshold(season["tmin_c"], season["tmax_c"], F86).sum()
    dh90 = degree_hours_above_threshold(season["tmin_c"], season["tmax_c"], F90).sum()

    # Phase windows for VPD and srad (clipped to cutoff_doy)
    veg   = phase_slice(df, *DOY_VEG,   cutoff_doy)
    silk  = phase_slice(df, *DOY_SILK,  cutoff_doy)
    grain = phase_slice(df, *DOY_GRAIN, cutoff_doy)

    def safe_mean(s):
        return float(s.mean()) if len(s) else np.nan

    def safe_sum(s):
        return float(s.sum()) if len(s) else np.nan

    feat = {
        "gdd_cum_f50_c86":   gdd_cum,
        "edd_hours_gt86f":   float(dh86),
        "edd_hours_gt90f":   float(dh90),
        "vpd_kpa_veg":       safe_mean(veg["vpd_kpa"]),
        "vpd_kpa_silk":      safe_mean(silk["vpd_kpa"]),
        "vpd_kpa_grain":     safe_mean(grain["vpd_kpa"]),
        "prcp_cum_mm":       float(season["prcp_mm"].sum()),
        "dry_spell_max_days": longest_dry_run(season["prcp_mm"]),
        "srad_total_veg":    safe_sum(veg["srad_mjm2"]),
        "srad_total_silk":   safe_sum(silk["srad_mjm2"]),
        "srad_total_grain":  safe_sum(grain["srad_mjm2"]),
    }
    return feat


# --- Main ------------------------------------------------------

ap = argparse.ArgumentParser()
ap.add_argument("--in",  dest="in_path",  default=DEFAULT_IN)
ap.add_argument("--out", dest="out_path", default=DEFAULT_OUT)
args = ap.parse_args()

print(f"Reading {args.in_path}...")
df_all = pd.read_parquet(args.in_path)
df_all["date"] = pd.to_datetime(df_all["date"])
df_all["GEOID"] = df_all["GEOID"].astype(str).str.zfill(5)
print(f"  {len(df_all):,} rows, {df_all['GEOID'].nunique()} counties, "
      f"{df_all['date'].min().date()}..{df_all['date'].max().date()}")

geoids = sorted(df_all["GEOID"].unique())
years  = sorted(df_all["date"].dt.year.unique())
print(f"  GEOIDs: {len(geoids)}, years: {len(years)}")

# Iterate over (GEOID, year, forecast_date)
out_rows = []
total = len(geoids) * len(years) * len(FORECAST_DATES)
done = 0
print(f"\nDeriving features ({total:,} (GEOID, year, forecast_date) combos)...")

# Group once by GEOID for speed (avoid re-filtering 500*20*4 times naively).
for geoid, df_g in df_all.groupby("GEOID", sort=False):
    df_g = df_g.set_index(pd.RangeIndex(len(df_g)))   # cheap reset
    for year in years:
        for fd_label, (mm, dd) in FORECAST_DATES.items():
            cutoff = dt.date(year, mm, dd)
            feat = build_features_for_cutoff(df_g, year, cutoff)
            done += 1
            if feat is None:
                continue
            row = {"GEOID": geoid, "year": int(year), "forecast_date": fd_label}
            row.update(feat)
            out_rows.append(row)
        if done % 5000 == 0:
            print(f"  {done:,} / {total:,}")

out = pd.DataFrame(out_rows)
out = out.sort_values(["GEOID", "year", "forecast_date"]).reset_index(drop=True)
print(f"\nFeature rows: {len(out):,}")
print(f"Columns: {list(out.columns)}")
print("\nSample (head):")
print(out.head(8).to_string(index=False))

print("\nCoverage by forecast_date:")
print(out.groupby("forecast_date").size().to_string())

# QC: any all-null feature columns?
feat_cols = [c for c in out.columns if c not in ("GEOID", "year", "forecast_date")]
print("\nNull counts:")
for c in feat_cols:
    n = out[c].isna().sum()
    print(f"  {c:24s} {n:>6,}  ({100*n/len(out):5.1f}%)")

import os
os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
out.to_csv(args.out_path, index=False)
print(f"\nWrote {args.out_path}  ({os.path.getsize(args.out_path)/1e6:.1f} MB)")
