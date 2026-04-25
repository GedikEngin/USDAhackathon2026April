"""
Build per-(GEOID, year, forecast_date) US Drought Monitor features from the
weekly USDM CSV.

Forecast dates: "08-01", "09-01", "10-01", "EOS"  (EOS = 11-30 = Nov 30)

Features (all measured AS-OF the last USDM reading strictly before the
forecast date):
  d0_pct          % of state at D0 or worse  (abnormally dry)
  d1_pct          % of state at D1 or worse  (moderate)
  d2_pct          % of state at D2 or worse  (severe)
  d3_pct          % of state at D3 or worse  (extreme)
  d4_pct          % of state at D4           (exceptional)
  d2plus          alias for d2_pct (severe-or-worse), exposed as a named
                  composite for downstream convenience

Note: USDM's Cumulative Percent Area columns are already nested
(D0 >= D1 >= D2 >= D3 >= D4), so "D2+" is exactly the published D2 column.
We expose it under both names so the modeling code can refer to a clearly-
named drought-stress signal without having to remember the cumulative
convention.

USDM is published WEEKLY at the state level (StateAbbreviation only; no
county FIPS in the source). To produce a per-(GEOID, year, forecast_date)
table we broadcast each state's reading to every GEOID in that state. The
GEOID directory is sourced from nass_corn_5states_features_v2.csv (which
already has GEOID + state_alpha + year for the modeling universe).

AS-OF RULE: for forecast date D in year Y, use the USDM row whose ValidEnd
is the maximum date strictly less than D. "Strictly before" (not <=)
prevents same-week leakage: USDM reports validity as a Tue->Mon span
(historically) or similar week window, and the map for week W can include
information from days that bracket the forecast date.

Usage:
  python drought_features.py
  python drought_features.py --in phase2/data/drought/drought_USDM-Colorado,Iowa,Missouri,Nebraska,Wisconsin.csv \
                             --geoid-source scripts/nass_corn_5states_features_v2.csv \
                             --out scripts/drought_county_features.csv
"""

import argparse
import datetime as dt

import numpy as np
import pandas as pd

# --- Config ----------------------------------------------------

DEFAULT_IN           = "phase2/data/drought/drought_USDM-Colorado,Iowa,Missouri,Nebraska,Wisconsin.csv"
DEFAULT_GEOID_SOURCE = "scripts/nass_corn_5states_features_v2.csv"
DEFAULT_OUT          = "scripts/drought_county_features.csv"

# Forecast-date suffixes (strings) -> (month, day). Match weather_features.py
# exactly so the merge_all join keys line up.
FORECAST_DATES = {
    "08-01": (8,  1),
    "09-01": (9,  1),
    "10-01": (10, 1),
    "EOS":   (11, 30),
}

# USDM D-level columns in source order (cumulative pct of area).
DLEVEL_COLS = ["D0", "D1", "D2", "D3", "D4"]

# Output column names (lowercased + suffix; matches the rest of the v2 schema)
OUT_COLS = {
    "D0": "d0_pct",
    "D1": "d1_pct",
    "D2": "d2_pct",
    "D3": "d3_pct",
    "D4": "d4_pct",
}


# --- As-of join ------------------------------------------------

def asof_state_reading(df_state, cutoff_date):
    """
    df_state: DataFrame for a single StateAbbreviation, sorted by ValidEnd ascending,
              with ValidEnd as datetime.date in a column named 'valid_end' and the
              D0..D4 percentage columns alongside.
    cutoff_date: datetime.date, the forecast date.

    Returns a 1-row Series of D0..D4 for the most recent reading whose
    valid_end is STRICTLY BEFORE cutoff_date. None if no such row exists.
    """
    mask = df_state["valid_end"] < cutoff_date
    if not mask.any():
        return None
    # df_state is pre-sorted ascending by valid_end, so the last True is the
    # closest-but-strictly-before reading.
    return df_state.loc[mask, DLEVEL_COLS].iloc[-1]


# --- Main feature builder --------------------------------------

def build_features_for_cutoff(df_state, year, cutoff_date):
    """
    df_state: DataFrame for a single state, ALL years, sorted ascending by valid_end.
    year: int, the target year (used for the output row only; the as-of slice
          is purely time-based and may pull a reading from year-1 if no
          in-year reading precedes the forecast date - rare but possible
          for the 08-01 cutoff if the data is sparse near the start).
    cutoff_date: datetime.date, the forecast date.

    Returns a dict of feature values for this (state, year, cutoff_date).
    AS-OF SAFETY: asof_state_reading() applies the strictly-before filter on
    valid_end. Nothing else in this function reads df_state.
    """
    row = asof_state_reading(df_state, cutoff_date)
    if row is None:
        return None

    feat = {OUT_COLS[c]: float(row[c]) for c in DLEVEL_COLS}
    # d2plus composite: USDM D-levels are cumulative, so "D2 or worse" is
    # already exactly the D2 percentage. Expose it under a stable, descriptive
    # name so downstream code doesn't have to know about the cumulative
    # convention.
    feat["d2plus"] = feat["d2_pct"]
    return feat


# --- Main ------------------------------------------------------

ap = argparse.ArgumentParser()
ap.add_argument("--in",  dest="in_path",  default=DEFAULT_IN)
ap.add_argument("--geoid-source", dest="geoid_source", default=DEFAULT_GEOID_SOURCE,
                help="CSV with GEOID + state_alpha + year columns; defines the "
                     "modeling universe to broadcast state-level USDM readings across.")
ap.add_argument("--out", dest="out_path", default=DEFAULT_OUT)
args = ap.parse_args()

print(f"Reading USDM CSV {args.in_path}...")
usdm = pd.read_csv(args.in_path)
usdm["valid_end"] = pd.to_datetime(usdm["ValidEnd"]).dt.date
usdm = usdm.rename(columns={"StateAbbreviation": "state_alpha"})
usdm = usdm[["state_alpha", "valid_end"] + DLEVEL_COLS].copy()
# Coerce D-levels to float (CSV reads them as float already, belt-and-braces).
for c in DLEVEL_COLS:
    usdm[c] = pd.to_numeric(usdm[c], errors="coerce")
usdm = usdm.sort_values(["state_alpha", "valid_end"]).reset_index(drop=True)
print(f"  {len(usdm):,} weekly rows, "
      f"states: {sorted(usdm['state_alpha'].unique())}, "
      f"{usdm['valid_end'].min()}..{usdm['valid_end'].max()}")

print(f"\nReading GEOID directory {args.geoid_source}...")
geo_dir = pd.read_csv(args.geoid_source)
geo_dir["GEOID"] = geo_dir["GEOID"].astype(str).str.zfill(5)
geo_dir = geo_dir[["GEOID", "state_alpha", "year"]].drop_duplicates()
geo_dir["year"] = geo_dir["year"].astype(int)
print(f"  {len(geo_dir):,} (GEOID, year) rows, "
      f"{geo_dir['GEOID'].nunique()} GEOIDs, "
      f"{geo_dir['year'].min()}..{geo_dir['year'].max()}, "
      f"states: {sorted(geo_dir['state_alpha'].unique())}")

# Pre-slice USDM by state for speed.
usdm_by_state = {st: g.reset_index(drop=True)
                 for st, g in usdm.groupby("state_alpha", sort=False)}

# 1) Build per-(state_alpha, year, forecast_date) feature rows. This is
#    cheap: 5 states * ~20 years * 4 dates = ~400 rows.
print("\nDeriving state-level as-of features...")
state_rows = []
years_in_geo = sorted(geo_dir["year"].unique())
for state in sorted(usdm_by_state):
    df_state = usdm_by_state[state]
    for year in years_in_geo:
        for fd_label, (mm, dd) in FORECAST_DATES.items():
            cutoff = dt.date(year, mm, dd)
            feat = build_features_for_cutoff(df_state, year, cutoff)
            if feat is None:
                continue
            row = {"state_alpha": state, "year": int(year), "forecast_date": fd_label}
            row.update(feat)
            state_rows.append(row)

state_feat = pd.DataFrame(state_rows)
print(f"  {len(state_feat):,} (state_alpha, year, forecast_date) rows")

# 2) Build the full (GEOID, year, forecast_date) skeleton, then left-join
#    the state-level features. This guarantees the output is always cleanly
#    keyed on all three columns, even for (year, forecast_date) combos that
#    have no USDM reading available (in which case the feature columns are
#    NaN, and merge_all.py decides downstream whether to drop or impute).
print("\nBroadcasting to GEOID...")
fd_labels = list(FORECAST_DATES.keys())
skeleton = (geo_dir.assign(_k=1)
                   .merge(pd.DataFrame({"forecast_date": fd_labels, "_k": 1}), on="_k")
                   .drop(columns="_k"))
out = skeleton.merge(state_feat, on=["state_alpha", "year", "forecast_date"], how="left")
out = out[["GEOID", "year", "forecast_date",
           "d0_pct", "d1_pct", "d2_pct", "d3_pct", "d4_pct", "d2plus"]]
out = out.sort_values(["GEOID", "year", "forecast_date"]).reset_index(drop=True)

print(f"\nFeature rows: {len(out):,}")
print(f"Columns: {list(out.columns)}")
print("\nSample (head):")
print(out.head(8).to_string(index=False))

print("\nCoverage by forecast_date:")
print(out.groupby("forecast_date").size().to_string())

# QC: any all-null feature columns? Sanity-check D0 >= D1 >= D2 >= D3 >= D4
# on the state-level intermediate (cheaper, and identical after broadcast).
feat_cols = ["d0_pct", "d1_pct", "d2_pct", "d3_pct", "d4_pct", "d2plus"]
print("\nNull counts:")
for c in feat_cols:
    n = out[c].isna().sum()
    print(f"  {c:10s} {n:>6,}  ({100*n/len(out):5.1f}%)")

print("\nMonotonicity check (D0 >= D1 >= D2 >= D3 >= D4) on state intermediate:")
sf = state_feat.dropna(subset=["d0_pct", "d1_pct", "d2_pct", "d3_pct", "d4_pct"])
viol = ((sf["d0_pct"] < sf["d1_pct"] - 1e-9) |
        (sf["d1_pct"] < sf["d2_pct"] - 1e-9) |
        (sf["d2_pct"] < sf["d3_pct"] - 1e-9) |
        (sf["d3_pct"] < sf["d4_pct"] - 1e-9))
print(f"  violations: {int(viol.sum())} / {len(sf):,}")

import os
os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
out.to_csv(args.out_path, index=False)
print(f"\nWrote {args.out_path}  ({os.path.getsize(args.out_path)/1e6:.1f} MB)")
