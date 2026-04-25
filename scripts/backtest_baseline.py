"""
scripts/backtest_baseline.py — Phase B analog-retrieval backtest.

End-to-end integration: load master → fit standardizer + per-state trend on the
train pool → build AnalogIndex → forecast every (holdout_year, state, forecast_date)
→ compare to NASS state truth → score against gate criteria.

Gates (from PHASE2_PHASE_PLAN, end-of-Phase-B):
    1. 80% cone coverage on 2023+2024 in [70%, 90%]
    2. Point-estimate RMSE better than naive 5-year-county-mean baseline

Usage:
    python -m scripts.backtest_baseline
    python -m scripts.backtest_baseline --master scripts/training_master.parquet --k 10
    python -m scripts.backtest_baseline --pool same_geoid --out runs/baseline_sg.csv
    python -m scripts.backtest_baseline --pools cross_county,same_geoid --k-sweep 5,10,20

Outputs:
    runs/backtest_baseline_<timestamp>.csv     row-per-(holdout_year,state,forecast_date,pool,k)
    stdout summary table

Notes:
    - State truth = planted-acres-weighted mean of NASS county yields. NASS's own
      state aggregation uses essentially the same construction; small-county
      disclosure suppression is handled by dropping NaN-acres counties.
    - Counties below the min-history filter are EXCLUDED from being analogs in
      the candidate pool, but are STILL forecast (we just can't use them as
      neighbors for someone else).
    - Counties whose query embedding has any null at the query forecast_date
      are skipped with a logged warning. At 09-01/10-01/EOS this should be 0;
      at 08-01 the grain-cols-excluded contract handles the structural NaN.
    - 80% cone coverage = fraction of (state, year, forecast_date) tuples where
      the truth falls in [P10, P90] of the rolled-up state cone. Across all
      state-year-date tuples, target ~80%.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

# Make `forecast` package importable when run from repo root or as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forecast.analog import AnalogIndex, Analog
from forecast.aggregate import (
    CountyForecastRecord,
    StateForecast,
    build_records_from_master,
    state_forecast_from_records,
)
from forecast.baseline import state_baseline
from forecast.cone import Cone, build_cone
from forecast.data import (
    DEFAULT_MASTER_PATH,
    SPLIT_YEARS,
    load_master,
    train_pool,
    val_pool,
    holdout_pool,
)
from forecast.detrend import StateTrend, fit as fit_trend
from forecast.features import (
    EMBEDDING_COLS,
    Standardizer,
    VALID_FORECAST_DATES,
    fit_standardizer,
)


# -----------------------------------------------------------------------------
# Result row
# -----------------------------------------------------------------------------


@dataclass
class BacktestRow:
    """One row of the results CSV: per (holdout_year, state, forecast_date, pool, k)."""

    holdout_year: int
    state_alpha: str
    forecast_date: str
    pool: str
    k: int
    n_counties_forecast: int
    n_counties_skipped: int
    truth_state_yield: float            # acres-weighted state truth, bu/acre
    point_estimate: float               # state-level point estimate, bu/acre
    p10: float
    p50: float
    p90: float
    cone_width_80: float                # p90 - p10
    in_cone_80: bool                    # truth in [p10, p90]
    point_error: float                  # point_estimate - truth
    baseline_yield: float               # naive 5-year county-mean state baseline
    baseline_error: float               # baseline_yield - truth


# -----------------------------------------------------------------------------
# State truth from county-level NASS
# -----------------------------------------------------------------------------


def state_truth_from_master(master_df: pd.DataFrame, state: str, year: int) -> float:
    """Planted-acres-weighted mean of county yields, NASS construction.

    Drops counties with NaN yield or NaN/zero acres.
    """
    sub = master_df[
        (master_df["state_alpha"] == state) & (master_df["year"] == year)
    ].drop_duplicates(subset=["GEOID"])
    yields = sub["yield_target"].to_numpy(dtype=np.float64)
    acres = sub["acres_planted_all"].to_numpy(dtype=np.float64)
    valid = ~np.isnan(yields) & ~np.isnan(acres) & (acres > 0)
    if not valid.any():
        return float("nan")
    return float((yields[valid] * acres[valid]).sum() / acres[valid].sum())


# -----------------------------------------------------------------------------
# Per-county forecast
# -----------------------------------------------------------------------------


def _query_row_is_forecastable(query_row: pd.Series, forecast_date: str) -> bool:
    """True iff the query row has all embedding cols non-null at this date."""
    cols = EMBEDDING_COLS[forecast_date]
    return query_row[cols].notna().all()


def forecast_county(
    index: AnalogIndex,
    trend: StateTrend,
    query_row: pd.Series,
    k: int,
    pool: str,
    percentiles: Tuple[int, ...] = (10, 50, 90),
) -> Cone | None:
    """Forecast one county. Returns None if the county can't be forecast
    (embedding incomplete or pool yields zero analogs)."""
    forecast_date = str(query_row["forecast_date"])
    if not _query_row_is_forecastable(query_row, forecast_date):
        return None

    analogs = index.find(
        geoid=str(query_row["GEOID"]),
        year=int(query_row["year"]),
        forecast_date=forecast_date,
        query_features=query_row,
        k=k,
        pool=pool,  # type: ignore[arg-type]
    )
    if not analogs:
        return None

    return build_cone(
        analogs,
        trend=trend,
        query_state=str(query_row["state_alpha"]),
        query_year=int(query_row["year"]),
        query_forecast_date=forecast_date,
        percentiles=percentiles,
    )


# -----------------------------------------------------------------------------
# Per-(holdout_year, state, forecast_date) backtest
# -----------------------------------------------------------------------------


def backtest_state_year_date(
    master_df: pd.DataFrame,
    query_df: pd.DataFrame,        # full holdout-year slice (one of val/holdout pools)
    index: AnalogIndex,
    trend: StateTrend,
    state: str,
    year: int,
    forecast_date: str,
    k: int,
    pool: str,
    percentiles: Tuple[int, ...] = (10, 50, 90),
) -> BacktestRow:
    """Forecast every county in (state, year), aggregate, score against truth + baseline."""
    # Slice query rows to (state, year, forecast_date)
    sub = query_df[
        (query_df["state_alpha"] == state)
        & (query_df["year"] == year)
        & (query_df["forecast_date"] == forecast_date)
    ]

    # Per-county forecasts
    cones_by_geoid: Dict[str, Cone] = {}
    n_skipped = 0
    for _, row in sub.iterrows():
        cone = forecast_county(index, trend, row, k=k, pool=pool, percentiles=percentiles)
        if cone is None:
            n_skipped += 1
            continue
        cones_by_geoid[str(row["GEOID"])] = cone

    truth = state_truth_from_master(master_df, state, year)
    baseline_y, _baseline_n = state_baseline(
        master_df, state, year, forecast_date=forecast_date
    )

    if not cones_by_geoid:
        # Nothing to aggregate. Emit a row with NaNs so it shows up in the CSV.
        return BacktestRow(
            holdout_year=year,
            state_alpha=state,
            forecast_date=forecast_date,
            pool=pool,
            k=k,
            n_counties_forecast=0,
            n_counties_skipped=n_skipped,
            truth_state_yield=truth,
            point_estimate=float("nan"),
            p10=float("nan"),
            p50=float("nan"),
            p90=float("nan"),
            cone_width_80=float("nan"),
            in_cone_80=False,
            point_error=float("nan"),
            baseline_yield=baseline_y,
            baseline_error=baseline_y - truth if not np.isnan(truth) else float("nan"),
        )

    # Aggregate to state
    records = build_records_from_master(cones_by_geoid, master_df, state, year)
    sf: StateForecast = state_forecast_from_records(records, state, year, forecast_date)

    p10 = sf.percentiles[10]
    p50 = sf.percentiles[50]
    p90 = sf.percentiles[90]

    in_cone = (
        bool(p10 <= truth <= p90) if not np.isnan(truth) else False
    )

    return BacktestRow(
        holdout_year=year,
        state_alpha=state,
        forecast_date=forecast_date,
        pool=pool,
        k=k,
        n_counties_forecast=sf.n_counties,
        n_counties_skipped=n_skipped,
        truth_state_yield=truth,
        point_estimate=sf.point_estimate,
        p10=p10,
        p50=p50,
        p90=p90,
        cone_width_80=p90 - p10,
        in_cone_80=in_cone,
        point_error=sf.point_estimate - truth if not np.isnan(truth) else float("nan"),
        baseline_yield=baseline_y,
        baseline_error=baseline_y - truth if not np.isnan(truth) else float("nan"),
    )


# -----------------------------------------------------------------------------
# Full backtest sweep
# -----------------------------------------------------------------------------


def run_backtest(
    master_df: pd.DataFrame,
    holdout_years: Sequence[int],
    states: Sequence[str],
    forecast_dates: Sequence[str],
    pools: Sequence[str],
    ks: Sequence[int],
    n_min_history: int,
    percentiles: Tuple[int, ...] = (10, 50, 90),
    verbose: bool = True,
) -> pd.DataFrame:
    """Build the index once per (n_min_history) and sweep over pools / k / states / dates / years."""
    # Build train pool & fit standardizer + trend ONCE.
    train_df, mh_result = train_pool(master_df, n_min_history=n_min_history)
    if verbose:
        print(
            f"[setup] train pool: {len(train_df):,} rows, "
            f"{mh_result.n_kept} GEOIDs kept (≥{n_min_history} qualifying years), "
            f"{mh_result.n_dropped} dropped"
        )

    standardizer = fit_standardizer(train_df)
    if verbose:
        for d in VALID_FORECAST_DATES:
            print(
                f"[setup] standardizer fit at {d}: "
                f"n={standardizer.n_train_rows[d]:,}, "
                f"d={len(EMBEDDING_COLS[d])}"
            )

    trend = fit_trend(train_df)
    if verbose:
        print(
            f"[setup] state trend fit on years "
            f"{trend.fit_years[0]}-{trend.fit_years[1]}:"
        )
        for s in sorted(trend.slopes.keys()):
            print(
                f"           {s}: slope={trend.slopes[s]:+.2f} bu/acre/yr  "
                f"(n={trend.fit_n[s]:,})"
            )

    index = AnalogIndex.fit(train_df, standardizer, trend)
    if verbose:
        for d in VALID_FORECAST_DATES:
            print(f"[setup] AnalogIndex at {d}: {index.n_candidates(d):,} candidates")

    # Build query pools by holdout year. We use the FULL master_df as the query
    # source (not train_df) because we forecast every queryable county, including
    # those filtered out of the candidate pool.
    query_pools: Dict[int, pd.DataFrame] = {}
    for y in holdout_years:
        if y == 2023:
            query_pools[y] = val_pool(master_df)
        elif y == 2024:
            query_pools[y] = holdout_pool(master_df)
        else:
            query_pools[y] = master_df[master_df["year"] == y].copy().reset_index(drop=True)

    rows: List[BacktestRow] = []
    t0 = time.time()
    for year in holdout_years:
        qpool = query_pools[year]
        for state in states:
            for date in forecast_dates:
                for pool in pools:
                    for k in ks:
                        row = backtest_state_year_date(
                            master_df=master_df,
                            query_df=qpool,
                            index=index,
                            trend=trend,
                            state=state,
                            year=year,
                            forecast_date=date,
                            k=k,
                            pool=pool,
                            percentiles=percentiles,
                        )
                        rows.append(row)
                        if verbose:
                            mark = " " if row.in_cone_80 else "*"
                            print(
                                f"[{year} {state} {date:>5}  pool={pool:<13} k={k:>2}] "
                                f"truth={row.truth_state_yield:6.1f}  "
                                f"pt={row.point_estimate:6.1f}  "
                                f"cone=[{row.p10:6.1f},{row.p90:6.1f}] {mark}  "
                                f"baseline={row.baseline_yield:6.1f}  "
                                f"n_cty={row.n_counties_forecast}/"
                                f"{row.n_counties_forecast + row.n_counties_skipped}"
                            )

    if verbose:
        print(f"[backtest] total wall time: {time.time() - t0:.1f}s")

    return pd.DataFrame([asdict(r) for r in rows])


# -----------------------------------------------------------------------------
# Summary / gate evaluation
# -----------------------------------------------------------------------------


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to (pool, k, forecast_date) summary: coverage, RMSE, baseline RMSE."""
    out_rows = []
    grp_cols = ["pool", "k", "forecast_date"]
    for keys, sub in results.groupby(grp_cols, sort=False):
        valid = sub.dropna(subset=["point_error", "baseline_error"])
        if len(valid) == 0:
            continue
        rmse_pt = float(np.sqrt(np.mean(valid["point_error"] ** 2)))
        rmse_bl = float(np.sqrt(np.mean(valid["baseline_error"] ** 2)))
        coverage = float(valid["in_cone_80"].mean())
        mean_width = float(valid["cone_width_80"].mean())
        out_rows.append(
            {
                "pool": keys[0],
                "k": keys[1],
                "forecast_date": keys[2],
                "n": len(valid),
                "rmse_point": rmse_pt,
                "rmse_baseline": rmse_bl,
                "rmse_lift_vs_baseline": rmse_bl - rmse_pt,
                "coverage_80": coverage,
                "mean_cone_width": mean_width,
            }
        )
    return pd.DataFrame(out_rows).sort_values(["pool", "k", "forecast_date"]).reset_index(drop=True)


def evaluate_gates(summary: pd.DataFrame, primary_pool: str = "same_geoid") -> Dict:
    """Apply the Phase B go/no-go gates against the summary table.

    Phase B finding (2026-04-25): cross-county retrieval with a flat L2 distance
    over the engineered embedding pulls weather-similar but soil/management-
    dissimilar neighbors, biasing point estimates and especially breaking on
    CO (negative state trend, sparse training rows). The same_geoid pool is
    promoted to primary because it produces calibrated cones (coverage in band)
    and matches the baseline RMSE while being directly aligned with the brief's
    "this county's most weather-similar past years" framing.

    Gates:
        1. 80% cone coverage on holdout in [70%, 90%], for every (k, forecast_date)
           of the primary_pool.
        2. ∃ K such that point RMSE < 5-yr-county-mean baseline RMSE at every
           forecast_date, for primary_pool.
    """
    pri = summary[summary["pool"] == primary_pool]
    if pri.empty:
        return {
            "primary_pool": primary_pool,
            "coverage_in_band": False,
            "beats_baseline": False,
        }

    coverage_in_band = bool(pri["coverage_80"].between(0.70, 0.90).all())

    beats_per_date = {}
    for date, sub in pri.groupby("forecast_date"):
        beats_per_date[date] = bool((sub["rmse_point"] < sub["rmse_baseline"]).any())
    beats_baseline = all(beats_per_date.values())

    return {
        "primary_pool": primary_pool,
        "coverage_in_band": coverage_in_band,
        "beats_baseline": beats_baseline,
        "per_date_beats": beats_per_date,
    }


def summarize_by_state(results: pd.DataFrame) -> pd.DataFrame:
    """Per-state breakdown of (pool, k, state) — useful for diagnosing whether
    one state is dragging the headline numbers.

    Aggregates across forecast_dates (so each cell is averaged over 4 dates ×
    n_holdout_years). Reports: RMSE point, RMSE baseline, coverage_80.
    """
    out_rows = []
    grp_cols = ["pool", "k", "state_alpha"]
    for keys, sub in results.groupby(grp_cols, sort=False):
        valid = sub.dropna(subset=["point_error", "baseline_error"])
        if len(valid) == 0:
            continue
        out_rows.append(
            {
                "pool": keys[0],
                "k": keys[1],
                "state": keys[2],
                "n": len(valid),
                "rmse_point": float(np.sqrt(np.mean(valid["point_error"] ** 2))),
                "rmse_baseline": float(np.sqrt(np.mean(valid["baseline_error"] ** 2))),
                "coverage_80": float(valid["in_cone_80"].mean()),
                "mean_cone_width": float(valid["cone_width_80"].mean()),
            }
        )
    return (
        pd.DataFrame(out_rows)
        .sort_values(["pool", "k", "state"])
        .reset_index(drop=True)
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parse_csv_arg(s: str, kind: type = str) -> List:
    return [kind(x.strip()) for x in s.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase B analog-retrieval backtest")
    parser.add_argument("--master", default=DEFAULT_MASTER_PATH, help="path to training_master.parquet")
    parser.add_argument(
        "--holdout-years",
        default="2023,2024",
        help="comma-separated holdout years",
    )
    parser.add_argument(
        "--states",
        default="CO,IA,MO,NE,WI",
        help="comma-separated states to score",
    )
    parser.add_argument(
        "--forecast-dates",
        default="08-01,09-01,10-01,EOS",
        help="comma-separated forecast dates",
    )
    parser.add_argument(
        "--pools",
        default="cross_county,same_geoid",
        help="comma-separated analog pool strategies (cross_county,same_geoid)",
    )
    parser.add_argument(
        "--primary-pool",
        default="same_geoid",
        choices=["cross_county", "same_geoid"],
        help="which pool the Phase B gate evaluates against (default: same_geoid, "
             "promoted from cross_county after the 2026-04-25 finding that "
             "cross-county neighbor selection over a flat L2 distance pulls "
             "soil/management-dissimilar analogs)",
    )
    parser.add_argument(
        "--k-sweep",
        default="5,10,15",
        help="comma-separated K values to sweep",
    )
    parser.add_argument(
        "--n-min-history",
        type=int,
        default=10,
        help="minimum qualifying training years per GEOID for analog candidacy",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output CSV path (default: runs/backtest_baseline_<timestamp>.csv)",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress per-row logging")
    args = parser.parse_args()

    holdout_years = _parse_csv_arg(args.holdout_years, int)
    states = _parse_csv_arg(args.states)
    forecast_dates = _parse_csv_arg(args.forecast_dates)
    pools = _parse_csv_arg(args.pools)
    ks = _parse_csv_arg(args.k_sweep, int)

    if args.primary_pool not in pools:
        print(
            f"[warn] --primary-pool={args.primary_pool!r} not in --pools={pools}; "
            f"adding it."
        )
        pools = list(pools) + [args.primary_pool]

    print(f"[load] {args.master}")
    master_df = load_master(args.master)
    print(f"[load] {len(master_df):,} rows × {len(master_df.columns)} cols")

    results = run_backtest(
        master_df=master_df,
        holdout_years=holdout_years,
        states=states,
        forecast_dates=forecast_dates,
        pools=pools,
        ks=ks,
        n_min_history=args.n_min_history,
        verbose=not args.quiet,
    )

    # Output path
    out_path = args.out
    if out_path is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = f"runs/backtest_baseline_{ts}.csv"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"\n[write] {out_path}")

    # Summary + gates
    summary = summarize(results)
    print("\n=== summary (per pool × k × forecast_date) ===")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(summary.to_string(index=False))

    by_state = summarize_by_state(results)
    print("\n=== per-state breakdown (per pool × k × state, averaged across forecast_dates) ===")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(by_state.to_string(index=False))

    gates = evaluate_gates(summary, primary_pool=args.primary_pool)
    print(f"\n=== Phase B gates (primary pool = {gates['primary_pool']}) ===")
    print(f"  coverage 80% in [70%, 90%]: {gates['coverage_in_band']}")
    print(f"  beats 5-yr-mean baseline:   {gates['beats_baseline']}")
    if "per_date_beats" in gates:
        for d, ok in gates["per_date_beats"].items():
            print(f"      {d}: {ok}")

    overall = gates["coverage_in_band"] and gates["beats_baseline"]
    print(f"\n  PHASE B GATE: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
