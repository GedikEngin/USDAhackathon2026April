"""
Phase B - Backtest the analog-year cone on val (2023) and holdout (2024).

For each (state, target_year, forecast_date) in the eval set, we:
  1. Build the state-year panel from the master table.
  2. Z-score the embedding using the train-years-only scaler.
  3. Retrieve K analogs from STRICTLY EARLIER years.
  4. Build the (p10, p50, p90) cone.
  5. Compare to the actual yield_target.

Metrics:
  empirical_coverage_80   fraction of (state, year, forecast_date) cases
                          where actual is in [p10, p90]. Target ~0.80.
  rmse_p50, mae_p50       error of the median-of-analogs point estimate.
  cone_width_mean/median  p90 - p10, in bu/acre.

Per-state and per-forecast_date breakdowns, plus the crossed (state x date)
table.

GO/NO-GO gate per PHASE2_PHASE_PLAN.md B.5: 80% cone empirical coverage on
val + holdout should be in [0.70, 0.90]. If outside that band: re-pick K
or embedding once, accept and move on.

Usage:
  python scripts/backtest_cone.py
  python scripts/backtest_cone.py --master scripts/training_master.parquet \
                                  --embedding climate --k 10
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Import the retrieval module from the same scripts/ directory.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from retrieval import (  # noqa: E402
    DEFAULT_K, DEFAULT_MASTER, FORECAST_DATES, HOLDOUT_YEAR, TRAIN_YEARS, VAL_YEAR,
    aggregate_to_state, build_cone, build_scaler, find_analogs,
    make_synthetic_master, select_embedding_columns,
)

# --- Config ----------------------------------------------------

DEFAULT_OUT = "phase2/results/phase_b_backtest.csv"
EVAL_YEARS  = [VAL_YEAR, HOLDOUT_YEAR]


# --- Backtest core --------------------------------------------

def run_backtest(state_df, embed_cols, scaler, candidates_z,
                 eval_years, k, metric):
    """
    For every (state, year, forecast_date) where year in eval_years and the
    row has a non-null yield_target, retrieve K analogs from strictly earlier
    years and record the cone alongside the actual.
    """
    rows = []
    eval_pool = state_df[
        state_df["year"].isin(eval_years) & state_df["yield_target"].notna()
    ].copy()
    for _, qrow in eval_pool.iterrows():
        analogs = find_analogs(
            qrow, state_df, candidates_z, embed_cols, scaler,
            k=k, metric=metric,
        )
        cone = build_cone(analogs)
        actual = float(qrow["yield_target"])
        p10, p50, p90 = cone.get("p10"), cone.get("p50"), cone.get("p90")
        if p10 is None or np.isnan(p10):
            inside = np.nan
            err    = np.nan
            width  = np.nan
        else:
            inside = float(p10 <= actual <= p90)
            err    = actual - p50
            width  = p90 - p10
        rows.append({
            "state_alpha":   qrow["state_alpha"],
            "year":          int(qrow["year"]),
            "forecast_date": qrow["forecast_date"],
            "n_analogs":     cone["n_analogs"],
            "actual":        actual,
            "p10":           p10,
            "p50":           p50,
            "p90":           p90,
            "in_80_cone":    inside,
            "error":         err,
            "abs_error":     abs(err) if not np.isnan(err) else np.nan,
            "cone_width":    width,
        })
    return pd.DataFrame(rows)


# --- Reporting helpers ----------------------------------------

def _agg_metrics(df):
    """Coverage / RMSE / MAE / mean+median width over a slice."""
    n = len(df)
    if n == 0:
        return {"n": 0, "coverage_80": np.nan, "rmse_p50": np.nan,
                "mae_p50": np.nan, "width_mean": np.nan,
                "width_median": np.nan}
    cov = float(df["in_80_cone"].mean()) if df["in_80_cone"].notna().any() \
        else np.nan
    rmse = float(np.sqrt(np.nanmean(df["error"].to_numpy() ** 2)))
    mae  = float(np.nanmean(df["abs_error"].to_numpy()))
    return {
        "n":            n,
        "coverage_80":  cov,
        "rmse_p50":     rmse,
        "mae_p50":      mae,
        "width_mean":   float(np.nanmean(df["cone_width"])),
        "width_median": float(np.nanmedian(df["cone_width"])),
    }


def print_table(title, rows, fmt_n="{:>4d}", float_fmt="{:>7.2f}"):
    """Pretty-print a list of (label, metrics_dict) tuples."""
    print(f"\n{title}")
    print("-" * len(title))
    print(f"{'slice':>20s}  {'n':>4s}  {'cov80':>6s}  {'rmse':>7s}  "
          f"{'mae':>7s}  {'width_mu':>9s}  {'width_med':>10s}")
    for label, m in rows:
        n_str   = fmt_n.format(m['n']) if isinstance(m['n'], int) else f"{m['n']:>4}"
        cov_str = "  n/a " if np.isnan(m['coverage_80']) \
            else f"{m['coverage_80']:>6.3f}"
        rmse_str = "    n/a" if np.isnan(m['rmse_p50']) \
            else float_fmt.format(m['rmse_p50'])
        mae_str  = "    n/a" if np.isnan(m['mae_p50']) \
            else float_fmt.format(m['mae_p50'])
        wmu_str  = "      n/a" if np.isnan(m['width_mean']) \
            else f"{m['width_mean']:>9.2f}"
        wmed_str = "       n/a" if np.isnan(m['width_median']) \
            else f"{m['width_median']:>10.2f}"
        print(f"{label:>20s}  {n_str}  {cov_str}  {rmse_str}  "
              f"{mae_str}  {wmu_str}  {wmed_str}")


def gate_verdict(coverage):
    """Phase B GO/NO-GO: 80% cone empirical coverage in [0.70, 0.90]."""
    if np.isnan(coverage):
        return "n/a"
    if 0.70 <= coverage <= 0.90:
        return "PASS"
    if coverage > 0.90:
        return "FAIL (cone too wide - empirical > 90%)"
    return "FAIL (cone too narrow - empirical < 70%)"


# --- Main ------------------------------------------------------

ap = argparse.ArgumentParser()
ap.add_argument("--master", default=DEFAULT_MASTER)
ap.add_argument("--out", default=DEFAULT_OUT)
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

print("\nAggregating county -> state...")
state_df = aggregate_to_state(master)
print(f"  state-year-date rows: {len(state_df):,}")

embed_cols = select_embedding_columns(state_df, embedding=args.embedding)
print(f"\nEmbedding: {args.embedding!r} ({len(embed_cols)} cols)")
print(f"  cols: {embed_cols}")

scaler = build_scaler(state_df, embed_cols, train_years=TRAIN_YEARS)
print(f"  scaler fit on years {TRAIN_YEARS[0]}..{TRAIN_YEARS[-1]}")

candidates_z = scaler.transform(state_df)
print(f"  z-scored matrix: {candidates_z.shape}")

print(f"\nRunning backtest:  K={args.k}  metric={args.metric}  "
      f"eval_years={EVAL_YEARS}")
results = run_backtest(state_df, embed_cols, scaler, candidates_z,
                       EVAL_YEARS, args.k, args.metric)
print(f"  evaluated {len(results)} (state, year, forecast_date) cases")

if len(results) == 0:
    print("\n[!] No eval cases (no rows with year in EVAL_YEARS and a "
          "non-null yield_target). Cannot report metrics.")
    sys.exit(1)

# Overall by year.
print("\n" + "=" * 70)
print("PHASE B BACKTEST RESULTS")
print("=" * 70)

per_year = []
for y in EVAL_YEARS:
    sub = results[results["year"] == y]
    per_year.append((str(y), _agg_metrics(sub)))
per_year.append(("val+holdout", _agg_metrics(results)))
print_table("Overall by year", per_year)

# By forecast_date (across both eval years).
per_fd = []
for fd in FORECAST_DATES:
    sub = results[results["forecast_date"] == fd]
    per_fd.append((fd, _agg_metrics(sub)))
print_table("By forecast_date (val+holdout combined)", per_fd)

# By state.
per_state = []
for s in sorted(results["state_alpha"].unique()):
    sub = results[results["state_alpha"] == s]
    per_state.append((s, _agg_metrics(sub)))
print_table("By state (val+holdout combined)", per_state)

# Crossed: state x forecast_date.
print("\nBy state x forecast_date (coverage_80 / rmse_p50)")
print("-" * 50)
print(f"{'state':>6s}  " + "  ".join(f"{fd:>14s}" for fd in FORECAST_DATES))
for s in sorted(results["state_alpha"].unique()):
    cells = []
    for fd in FORECAST_DATES:
        sub = results[(results["state_alpha"] == s) &
                      (results["forecast_date"] == fd)]
        m = _agg_metrics(sub)
        if m["n"] == 0:
            cells.append(f"{'n/a':>14s}")
        else:
            cov = "n/a" if np.isnan(m["coverage_80"]) else f"{m['coverage_80']:.2f}"
            rmse = "n/a" if np.isnan(m["rmse_p50"]) else f"{m['rmse_p50']:.1f}"
            cells.append(f"{cov:>5s}/{rmse:>7s}")
    print(f"{s:>6s}  " + "  ".join(cells))

# GO/NO-GO gate.
overall = _agg_metrics(results)
val_only = _agg_metrics(results[results["year"] == VAL_YEAR])
holdout_only = _agg_metrics(results[results["year"] == HOLDOUT_YEAR])

print("\n" + "=" * 70)
print("GATE CHECK (PHASE2_PHASE_PLAN.md B.5)")
print("=" * 70)
print(f"  80% cone coverage, val ({VAL_YEAR}):     "
      f"{val_only['coverage_80']:.3f}  -> {gate_verdict(val_only['coverage_80'])}")
print(f"  80% cone coverage, holdout ({HOLDOUT_YEAR}): "
      f"{holdout_only['coverage_80']:.3f}  -> "
      f"{gate_verdict(holdout_only['coverage_80'])}")
print(f"  80% cone coverage, combined:        "
      f"{overall['coverage_80']:.3f}  -> "
      f"{gate_verdict(overall['coverage_80'])}")
print(f"  RMSE on p50, combined:              {overall['rmse_p50']:.2f} bu/acre")
print(f"  Mean cone width, combined:          {overall['width_mean']:.2f} bu/acre")
print(f"  config: K={args.k}, metric={args.metric}, embedding={args.embedding}")

# Save per-(state, year, date) results.
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
results = results.sort_values(["year", "forecast_date", "state_alpha"]) \
                 .reset_index(drop=True)
results.to_csv(args.out, index=False)
print(f"\nWrote {args.out}  ({len(results)} rows, "
      f"{os.path.getsize(args.out)/1e3:.1f} KB)")

print("\nResults head:")
print(results.head(10).to_string(index=False))
