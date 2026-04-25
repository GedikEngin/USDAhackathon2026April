"""
Phase B - K-NN analog-year retrieval and cone-of-uncertainty construction.

Methodology centerpiece per the brief: instead of fitting a parametric model,
we retrieve the K most-similar historical (state, year) pairs and read off
their actual yields to build a (p10, p50, p90) cone.

Aggregation:
  - Master table is at (GEOID, year, forecast_date) granularity (county-level).
  - Retrieval is at (state_alpha, year, forecast_date) granularity, because the
    brief asks for state-level cones. We area-weight county features to state
    using acres_planted_all (falling back to acres_harvested_all if planted is
    not available; relevant for 2025 where harvested may not yet be reported).

Standardization:
  - Z-score per continuous feature column.
  - Scaler is FIT on training years only (2005-2022), then used to transform
    train, val (2023), holdout (2024), and forecast (2025). Never refit.

Embedding (configurable):
  - "climate"  (default): weather + drought + a couple of NDVI signals. The
                          retrieval question is "what year was this *like*?",
                          which is fundamentally a climate / phenology question.
  - "all":                every continuous feature column. Heavier embedding;
                          available for ablation.

Distance:
  - Euclidean in z-scored space (default). Configurable for future experiments.

K-NN:
  - For a query (state_q, year_q, fd_q), candidate pool is every
    (state, year, forecast_date) row with year < year_q and forecast_date == fd_q.
  - Top K by smallest Euclidean distance. Default K = 10.
  - As-of safety: same forecast_date for query and candidates; year-strict.

Cone:
  - p50 = median of K analog yield_targets   (point estimate this phase)
  - cone = (p10, p50, p90) of K analog yield_targets
  - returns the K analog (state, year, distance) tuples for interpretability

Usage:
  python scripts/retrieval.py
  python scripts/retrieval.py --master scripts/training_master.parquet \
                              --embedding climate --k 10
"""

import argparse
import os

import numpy as np
import pandas as pd

# --- Config ----------------------------------------------------

DEFAULT_MASTER = "scripts/training_master.parquet"

TRAIN_YEARS   = list(range(2005, 2023))   # 2005..2022 inclusive
VAL_YEAR      = 2023
HOLDOUT_YEAR  = 2024
FORECAST_YEAR = 2025

FORECAST_DATES = ["08-01", "09-01", "10-01", "EOS"]

# Keys / non-feature columns that must NEVER enter the standardized embedding.
NON_FEATURE = {
    "GEOID", "state_alpha", "year", "forecast_date",
    "county_name", "NAME", "STATEFP",
    "yield_target",
    "yield_bu_acre_all", "yield_bu_acre_irr", "yield_bu_acre_noirr",
    "production_bu_all", "production_bu_irr", "production_bu_noirr",
    # acres are used as weights, not retrieval dimensions:
    "acres_planted_all", "acres_planted_irr", "acres_planted_noirr",
    "acres_harvested_all", "acres_harvested_irr", "acres_harvested_noirr",
    "acres_harvested_noirr_derived",
}

# Climate-only retrieval embedding. Soil and acreage are covariates here;
# we are not asking "find counties with similar soil", we are asking
# "find years that played out similarly in weather and phenology".
CLIMATE_FEATURES = [
    "gdd_cum_f50_c86",
    "edd_hours_gt86f", "edd_hours_gt90f",
    "vpd_kpa_veg", "vpd_kpa_silk", "vpd_kpa_grain",
    "prcp_cum_mm", "dry_spell_max_days",
    "srad_total_veg", "srad_total_silk", "srad_total_grain",
    "d0_pct", "d1_pct", "d2_pct", "d3_pct", "d4_pct", "d2plus",
    "ndvi_silking_mean", "ndvi_gs_integral",
]

DEFAULT_K = 10
DEFAULT_PERCENTILES = (10, 50, 90)


# --- State aggregation ----------------------------------------

def aggregate_to_state(master_df, weight_col="acres_planted_all"):
    """
    Roll a county-level master table up to (state_alpha, year, forecast_date)
    via area-weighted means.

    yield_target, weather, drought, NDVI, gSSURGO are all area-weighted means.
    We use acres_planted_all by default; fall back to acres_harvested_all
    where planted is missing (rare; mainly 2025 forecast rows).

    Rows where the chosen weight is null are silently dropped from the
    aggregation for that (state, year, forecast_date). If every county in a
    state-year has a null weight, that state-year is dropped entirely.
    """
    df = master_df.copy()

    # Build effective weight: planted, falling back to harvested.
    if weight_col not in df.columns:
        raise ValueError(f"Weight column {weight_col!r} not in master table.")
    fallback = "acres_harvested_all"
    if fallback in df.columns:
        df["_w"] = df[weight_col].fillna(df[fallback]).fillna(0.0)
    else:
        df["_w"] = df[weight_col].fillna(0.0)

    # Drop rows with zero weight from the aggregation.
    df = df[df["_w"] > 0].copy()

    # Identify which columns to aggregate. Numeric, not a key, not the weight.
    keys = ["state_alpha", "year", "forecast_date"]
    numeric_cols = [
        c for c in df.columns
        if c not in keys + ["GEOID", "_w", "county_name", "NAME", "STATEFP"]
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    # Weighted mean for each group/column. NaN-safe: nulls get weight 0.
    def _wmean(group):
        w = group["_w"].to_numpy(dtype=float)
        out = {}
        for c in numeric_cols:
            v = group[c].to_numpy(dtype=float)
            mask = ~np.isnan(v)
            if not mask.any() or w[mask].sum() == 0:
                out[c] = np.nan
            else:
                out[c] = float((v[mask] * w[mask]).sum() / w[mask].sum())
        return pd.Series(out)

    state_df = df.groupby(keys, as_index=False).apply(_wmean, include_groups=False)
    # Newer pandas wraps the apply output; flatten if needed.
    if isinstance(state_df.index, pd.MultiIndex):
        state_df = state_df.reset_index()
    return state_df.reset_index(drop=True)


# --- Standardization ------------------------------------------

class ZScoreScaler:
    """
    Manual mean/std z-score scaler. Transparent and dependency-free; the
    spec specifically permits this in lieu of sklearn.StandardScaler.

    Fit on training-year rows only. Transform any subset of the same
    columns. NaN-safe: missing values pass through as NaN (caller decides
    how to handle them downstream; for retrieval we impute to zero, i.e.
    the trained mean, which is the standard "no information" choice).
    """
    def __init__(self):
        self.cols_ = None
        self.mean_ = None
        self.std_  = None

    def fit(self, df, cols):
        self.cols_ = list(cols)
        X = df[self.cols_].to_numpy(dtype=float)
        self.mean_ = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0, ddof=0)
        # Guard zero-variance: use 1.0 to avoid division by zero (zeroed col
        # contributes nothing to distance, which is the right behavior).
        self.std_ = np.where(std == 0, 1.0, std)
        return self

    def transform(self, df):
        if self.cols_ is None:
            raise RuntimeError("Scaler not fit yet.")
        X = df[self.cols_].to_numpy(dtype=float)
        Z = (X - self.mean_) / self.std_
        # Impute residual NaNs to 0 (= the trained mean in z-space).
        Z = np.where(np.isnan(Z), 0.0, Z)
        return Z


def select_embedding_columns(state_df, embedding="climate"):
    """
    Pick which numeric columns make up the retrieval embedding.

    "climate": curated climate / drought / NDVI subset.
    "all":     every numeric column that's not a key / target / acreage.
    """
    available = set(state_df.columns)
    if embedding == "climate":
        return [c for c in CLIMATE_FEATURES if c in available]
    if embedding == "all":
        return [
            c for c in state_df.columns
            if c not in NON_FEATURE
            and pd.api.types.is_numeric_dtype(state_df[c])
        ]
    raise ValueError(f"Unknown embedding {embedding!r}. Use 'climate' or 'all'.")


def build_scaler(state_df, embed_cols, train_years=TRAIN_YEARS):
    """Fit a ZScoreScaler on rows whose year is in train_years."""
    train_mask = state_df["year"].isin(train_years)
    if not train_mask.any():
        raise RuntimeError(
            f"No rows with year in {train_years[0]}..{train_years[-1]}; "
            "cannot fit scaler."
        )
    return ZScoreScaler().fit(state_df.loc[train_mask], embed_cols)


# --- K-NN retrieval -------------------------------------------

def find_analogs(query_row, candidates_df, candidates_z, embed_cols,
                 scaler, k=DEFAULT_K, metric="euclidean"):
    """
    Return the top-K analogs for one query row.

    query_row     : a single Series with at least state_alpha, year,
                    forecast_date and the embed_cols populated.
    candidates_df : DataFrame of all rows in the universe (unfiltered).
                    We will filter to year < query.year and same forecast_date.
    candidates_z  : already-z-scored matrix aligned row-for-row with
                    candidates_df.
    embed_cols    : the column list used to build candidates_z (used to
                    z-score the query consistently).
    scaler        : the fitted ZScoreScaler.
    k             : number of analogs to return (default 10).
    metric        : "euclidean" (default) or "cosine".

    Returns a list of dicts:
      [{"state_alpha", "year", "forecast_date", "distance", "yield_target"}, ...]
    sorted by distance ascending. Length <= k (may be fewer if the candidate
    pool is small).
    """
    qy   = int(query_row["year"])
    qfd  = query_row["forecast_date"]

    # Candidate filter: strictly earlier year, same forecast_date.
    mask = (candidates_df["year"].to_numpy() < qy) & \
           (candidates_df["forecast_date"].to_numpy() == qfd)
    if not mask.any():
        return []

    cand_df = candidates_df.loc[mask].reset_index(drop=True)
    cand_z  = candidates_z[mask]

    # z-score the query through the same scaler.
    q_df = pd.DataFrame([{c: query_row[c] for c in embed_cols}])
    q_z  = scaler.transform(q_df)[0]

    if metric == "euclidean":
        diffs = cand_z - q_z[None, :]
        dists = np.sqrt((diffs * diffs).sum(axis=1))
    elif metric == "cosine":
        # 1 - cosine_similarity. Falls back to euclidean if a vector is zero.
        a = cand_z
        b = q_z
        an = np.linalg.norm(a, axis=1)
        bn = np.linalg.norm(b)
        denom = an * bn
        denom = np.where(denom == 0, 1.0, denom)
        dists = 1.0 - (a @ b) / denom
    else:
        raise ValueError(f"Unknown metric {metric!r}.")

    # Exclude any candidate that is the query itself (defensive: shouldn't
    # happen given the year-strict filter, but cheap to be paranoid).
    self_mask = (
        (cand_df["state_alpha"].to_numpy() == query_row["state_alpha"]) &
        (cand_df["year"].to_numpy() == qy) &
        (cand_df["forecast_date"].to_numpy() == qfd)
    )
    if self_mask.any():
        dists[self_mask] = np.inf

    k_eff = min(k, (np.isfinite(dists)).sum())
    if k_eff == 0:
        return []

    order = np.argsort(dists, kind="stable")[:k_eff]
    out = []
    for idx in order:
        out.append({
            "state_alpha": cand_df.iloc[idx]["state_alpha"],
            "year": int(cand_df.iloc[idx]["year"]),
            "forecast_date": cand_df.iloc[idx]["forecast_date"],
            "distance": float(dists[idx]),
            "yield_target": float(cand_df.iloc[idx]["yield_target"])
                if not pd.isna(cand_df.iloc[idx]["yield_target"]) else np.nan,
        })
    return out


# --- Cone construction ----------------------------------------

def build_cone(analogs, percentiles=DEFAULT_PERCENTILES):
    """
    From a list of analog dicts (output of find_analogs), build a cone.

    Returns: dict with p10/p50/p90 (or whatever percentiles asked for),
             the analog yields used, and the analog (state, year, distance)
             list for narration. Returns NaN-filled cone if no analogs have
             a valid yield_target.
    """
    if not analogs:
        return {"p10": np.nan, "p50": np.nan, "p90": np.nan,
                "n_analogs": 0, "analogs": []}

    yields = np.array([a["yield_target"] for a in analogs], dtype=float)
    valid  = yields[~np.isnan(yields)]
    if len(valid) == 0:
        return {"p10": np.nan, "p50": np.nan, "p90": np.nan,
                "n_analogs": 0, "analogs": analogs}

    qs = np.percentile(valid, percentiles)
    cone = {f"p{p}": float(v) for p, v in zip(percentiles, qs)}
    cone["n_analogs"] = int(len(valid))
    cone["analogs"]   = analogs
    return cone


# --- Synthetic-data fallback ----------------------------------

# REMOVE-WHEN-REAL-MASTER-LANDS: this block fabricates a small state-year
# panel so the retrieval and backtest logic can be exercised before the
# real training_master.parquet exists. The shape mirrors what merge_all.py
# is going to emit; once that file lands, this synth path is no longer
# triggered (--master file is read instead).
def make_synthetic_master(n_states=5, years=range(2005, 2026), seed=0):
    """Plausible (state, year, forecast_date) panel with weather, drought,
    NDVI, soil, acreage, and yield_target. NOT a county panel - we emit one
    "county" per state to keep the synthetic dataset small while still
    exercising aggregate_to_state."""
    rng = np.random.default_rng(seed)
    states = ["IA", "NE", "MO", "WI", "CO"][:n_states]
    state_geoid = {"IA": 19001, "NE": 31001, "MO": 29001,
                   "WI": 55001, "CO": 8001}
    rows = []
    for s in states:
        # state-level "soil" baseline
        soil_base = float(rng.normal(0.7, 0.05))
        for y in years:
            # year shock - a global "good/bad year" signal that yield reacts to.
            year_shock = float(rng.normal(0.0, 1.0))
            # heat / drought differ by state
            for fd in FORECAST_DATES:
                fd_progress = {"08-01": 0.55, "09-01": 0.75,
                               "10-01": 0.92, "EOS": 1.0}[fd]
                gdd  = 1800 + 600 * fd_progress + 80 * year_shock + rng.normal(0, 30)
                edd86 = max(0, 50 + 70 * fd_progress
                            - 30 * year_shock + rng.normal(0, 8))
                edd90 = max(0, 0.4 * edd86 + rng.normal(0, 5))
                vpd_v = 1.4 + 0.3 * year_shock + rng.normal(0, 0.1)
                vpd_s = 1.7 + 0.4 * year_shock + rng.normal(0, 0.12)
                vpd_g = 1.5 + 0.3 * year_shock + rng.normal(0, 0.1)
                prcp  = max(0, 350 * fd_progress - 80 * year_shock
                            + rng.normal(0, 25))
                dry   = max(0, int(8 + 5 * year_shock + rng.normal(0, 2)))
                srad_v = 1100 + rng.normal(0, 30)
                srad_s = 950 * fd_progress + rng.normal(0, 25)
                srad_g = 800 * max(0, fd_progress - 0.4) + rng.normal(0, 25)
                d0 = float(np.clip(20 + 15 * year_shock + rng.normal(0, 4),
                                   0, 100))
                d1 = float(np.clip(d0 - 8 + rng.normal(0, 2), 0, 100))
                d2 = float(np.clip(d1 - 6 + rng.normal(0, 2), 0, 100))
                d3 = float(np.clip(d2 - 4 + rng.normal(0, 1.5), 0, 100))
                d4 = float(np.clip(d3 - 2 + rng.normal(0, 1), 0, 100))
                d2plus = d2
                ndvi_silk = 0.7 - 0.05 * year_shock + rng.normal(0, 0.02)
                ndvi_int  = 50 + 8 * fd_progress - 4 * year_shock + rng.normal(0, 1.5)
                ndvi_peak = 0.82 - 0.04 * year_shock + rng.normal(0, 0.02)

                # yield is a function of weather + state baseline + noise.
                # 2025 is the forecast year - leave yield_target as NaN.
                if y == FORECAST_YEAR:
                    yt = np.nan
                else:
                    base = {"IA": 185, "NE": 175, "MO": 160,
                            "WI": 165, "CO": 145}[s]
                    yt = (base
                          - 8.0 * year_shock          # bad year hurts
                          - 0.05 * edd86              # heat stress
                          + 0.03 * (prcp - 350)       # rain helps
                          + 30 * (soil_base - 0.7)    # soil baseline
                          + rng.normal(0, 6))

                rows.append({
                    "GEOID": state_geoid[s],
                    "state_alpha": s,
                    "year": int(y),
                    "forecast_date": fd,
                    "yield_target": float(yt) if not np.isnan(yt) else np.nan,
                    "acres_planted_all": 1_000_000.0,
                    "acres_harvested_all": 950_000.0,
                    "irrigated_share": float(rng.uniform(0, 0.3)),
                    "harvest_ratio": 0.95,
                    "gdd_cum_f50_c86": gdd,
                    "edd_hours_gt86f": edd86,
                    "edd_hours_gt90f": edd90,
                    "vpd_kpa_veg": vpd_v,
                    "vpd_kpa_silk": vpd_s,
                    "vpd_kpa_grain": vpd_g,
                    "prcp_cum_mm": prcp,
                    "dry_spell_max_days": dry,
                    "srad_total_veg": srad_v,
                    "srad_total_silk": srad_s,
                    "srad_total_grain": srad_g,
                    "d0_pct": d0, "d1_pct": d1, "d2_pct": d2,
                    "d3_pct": d3, "d4_pct": d4, "d2plus": d2plus,
                    "ndvi_silking_mean": ndvi_silk,
                    "ndvi_gs_integral": ndvi_int,
                    "ndvi_peak": ndvi_peak,
                    "nccpi3corn": soil_base,
                    "aws0_100": 0.18,
                    "soc0_30": 35.0,
                })
    return pd.DataFrame(rows)


# --- Main: smoke-test the retrieval pipeline ------------------

ap = argparse.ArgumentParser()
ap.add_argument("--master", default=DEFAULT_MASTER,
                help="Path to training_master.parquet (or .csv).")
ap.add_argument("--embedding", choices=["climate", "all"], default="climate")
ap.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean")
ap.add_argument("--k", type=int, default=DEFAULT_K)
ap.add_argument("--use-synthetic", action="store_true",
                help="Force the synthetic master table (testing only).")
args = ap.parse_args()

if args.use_synthetic or not os.path.exists(args.master):
    if not args.use_synthetic:
        print(f"[!] {args.master} not found; falling back to synthetic data.")
        print("    Once merge_all.py lands, rerun without --use-synthetic.")
    master = make_synthetic_master()
    src = "synthetic"
else:
    print(f"Reading {args.master}...")
    if args.master.endswith(".csv"):
        master = pd.read_csv(args.master)
    else:
        master = pd.read_parquet(args.master)
    src = args.master

print(f"  source: {src}")
print(f"  master rows: {len(master):,}")
print(f"  master cols: {len(master.columns)}")
print(f"  forecast_dates: {sorted(master['forecast_date'].unique())}")
print(f"  years: {master['year'].min()}..{master['year'].max()}")

# Aggregate to state-level.
print("\nAggregating county -> state (acres-weighted)...")
state_df = aggregate_to_state(master)
print(f"  state-year-date rows: {len(state_df):,}")
print(f"  states: {sorted(state_df['state_alpha'].unique())}")
print(f"  yield_target nulls: {state_df['yield_target'].isna().sum()} "
      f"(expected for {FORECAST_YEAR})")

# Build embedding + scaler.
embed_cols = select_embedding_columns(state_df, embedding=args.embedding)
print(f"\nEmbedding: {args.embedding!r} ({len(embed_cols)} columns)")
print(f"  cols: {embed_cols}")

scaler = build_scaler(state_df, embed_cols, train_years=TRAIN_YEARS)
print(f"  scaler fit on {len(TRAIN_YEARS)} training years "
      f"({TRAIN_YEARS[0]}..{TRAIN_YEARS[-1]})")

# Pre-z-score the entire panel (rows that need a query-time z-score will be
# z-scored individually inside find_analogs; this matrix is just the candidate
# pool's pre-computed embeddings).
all_z = scaler.transform(state_df)

# Smoke-test retrieval: run a query for one (state, year, date).
# Use the most recent year present that has a yield_target so we can sanity-
# check that the cone bracketed the actual yield.
test_year = HOLDOUT_YEAR if HOLDOUT_YEAR in state_df["year"].values else VAL_YEAR
test_state = sorted(state_df["state_alpha"].unique())[0]
test_fd = "EOS"
qrow = state_df[
    (state_df["state_alpha"] == test_state) &
    (state_df["year"] == test_year) &
    (state_df["forecast_date"] == test_fd)
]
if len(qrow) == 0:
    print(f"\n[smoke] no row for ({test_state}, {test_year}, {test_fd}); "
          f"skipping retrieval smoke test.")
else:
    qrow = qrow.iloc[0]
    print(f"\n[smoke] query: state={test_state} year={test_year} "
          f"forecast_date={test_fd}")
    analogs = find_analogs(
        qrow, state_df, all_z, embed_cols, scaler,
        k=args.k, metric=args.metric,
    )
    print(f"  retrieved {len(analogs)} analogs (K={args.k})")
    for a in analogs[:5]:
        print(f"    {a['state_alpha']} {a['year']} d={a['distance']:.3f} "
              f"yield={a['yield_target']:.1f}")
    cone = build_cone(analogs)
    actual = qrow["yield_target"]
    inside = (not np.isnan(actual)
              and cone["p10"] <= actual <= cone["p90"])
    print(f"  cone: p10={cone['p10']:.1f}  p50={cone['p50']:.1f}  "
          f"p90={cone['p90']:.1f}")
    print(f"  actual yield_target: {actual:.1f}")
    print(f"  inside 80% cone? {inside}")

# QC tail.
print("\nState-year panel head:")
print(state_df[["state_alpha", "year", "forecast_date", "yield_target"]
               + embed_cols[:4]].head(8).to_string(index=False))

print("\nNull counts in embedding columns (state-year level):")
for c in embed_cols:
    n = state_df[c].isna().sum()
    print(f"  {c:24s} {n:>6,}  ({100*n/len(state_df):5.1f}%)")

print(f"\nDone. Module exports: aggregate_to_state, ZScoreScaler, "
      f"select_embedding_columns, build_scaler, find_analogs, build_cone.")
