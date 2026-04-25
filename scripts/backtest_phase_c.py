"""
scripts/backtest_phase_c.py — Phase C trained-regressor + analog-cone backtest.

Mirrors scripts/backtest_baseline.py structurally. The point estimate is now
RegressorBundle.predict (county-level) acres-weighted to state level. The cone
is the Phase B same_geoid K=5 analog cone, unchanged. Both are scored against
state-level NASS truth on the val (2023) and holdout (2024) years.

Gate (from PHASE2_PHASE_PLAN, end-of-Phase-C):
    Trained-regressor state-level RMSE on 2023 val ≥ 15% better than the
    Phase B analog-median (PRE-recalibration) at end-of-season.

The harness computes both for every (year, state, date) and reports both,
but the gate verdict reads only EOS row of val=2023.

Usage
-----
    python -m scripts.backtest_phase_c
    python -m scripts.backtest_phase_c --bundle-dir models/forecast
    python -m scripts.backtest_phase_c --recal none    # default
    python -m scripts.backtest_phase_c --holdout-years 2023,2024

Outputs
-------
    runs/backtest_phase_c_<timestamp>.csv     row-per-(year, state, date)
    stdout summary tables + gate verdict

Notes
-----
- Phase B locked config inherited: pool=same_geoid, K=5, percentiles=(10,50,90),
  per-county trend with state-median fallback. Same standardizer, same
  AnalogIndex setup as backtest_baseline.py.
- Recalibration: defaults to 'none' per the Phase 2-C kickoff. The driver
  prints per-(state, date) signed residuals so the recal decision can be
  made post-hoc. Other modes ('regressor', 'phase_b', 'both') are wired but
  off-by-default.
- Counties whose query embedding has any null at the query forecast_date are
  skipped (analog cone) but the regressor still predicts (XGBoost handles
  NaN natively). State aggregation drops counties with NaN regressor weight
  (= NaN planted_acres), matching Phase B convention.
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

# Make `forecast` importable when run from repo root or as -m.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forecast.aggregate import (
    CountyForecastRecord,
    StateForecast,
    build_records_from_master,
    state_forecast_from_records,
)
from forecast.analog import AnalogIndex
from forecast.baseline import state_baseline
from forecast.cone import Cone, build_cone
from forecast.data import (
    DEFAULT_MASTER_PATH,
    SPLIT_YEARS,
    holdout_pool,
    load_master,
    train_pool,
    val_pool,
)
from forecast.detrend import CountyTrend, fit as fit_trend
from forecast.features import EMBEDDING_COLS, VALID_FORECAST_DATES, fit_standardizer
from forecast.regressor import FEATURE_COLS, RegressorBundle, _add_derived_columns


DEFAULT_BUNDLE_DIR = "models/forecast"
GATE_LIFT_THRESHOLD = 0.15  # 15% RMSE reduction at EOS, per PHASE2_PHASE_PLAN C gate


# -----------------------------------------------------------------------------
# Result row
# -----------------------------------------------------------------------------


@dataclass
class PhaseCRow:
    """One row of the Phase C results CSV: per (year, state, forecast_date).

    Carries both the regressor point and the analog-median point so the gate
    comparison is direct.
    """

    holdout_year: int
    state_alpha: str
    forecast_date: str
    n_counties_regressor: int           # how many counties got a regressor prediction
    n_counties_analog: int              # how many counties got an analog cone
    truth_state_yield: float            # NASS state truth, acres-weighted
    # Regressor point estimate
    point_regressor: float
    error_regressor: float              # regressor - truth
    # Phase B analog-median point estimate (PRE-recalibration; gate comparator)
    point_analog_median: float
    error_analog_median: float
    # Phase B cone (analog-derived) — kept for the regressor's headline cone
    p10: float
    p50: float
    p90: float
    cone_width_80: float
    in_cone_80_regressor: bool          # regressor point inside [p10, p90]?
    in_cone_80_truth: bool              # truth inside [p10, p90]?
    # Naive 5-yr-mean state baseline (from Phase B; reported for context)
    baseline_yield: float
    baseline_error: float


# -----------------------------------------------------------------------------
# State truth — same as Phase B (intentionally duplicated to keep this script
# standalone-readable; identical implementation)
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
# Per-county cone (Phase B; unchanged)
# -----------------------------------------------------------------------------


def _query_row_is_forecastable(query_row: pd.Series, forecast_date: str) -> bool:
    """True iff the query row has all embedding cols non-null at this date."""
    cols = EMBEDDING_COLS[forecast_date]
    return query_row[cols].notna().all()


def forecast_county_cone(
    index: AnalogIndex,
    trend: CountyTrend,
    query_row: pd.Series,
    k: int,
    pool: str,
    percentiles: Tuple[int, ...] = (10, 50, 90),
) -> Cone | None:
    """Phase B per-county cone. Returns None if the query embedding is incomplete
    or the analog pool yields zero candidates.
    """
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
        query_geoid=str(query_row["GEOID"]),
        query_state=str(query_row["state_alpha"]),
        query_year=int(query_row["year"]),
        query_forecast_date=forecast_date,
        percentiles=percentiles,
    )


# -----------------------------------------------------------------------------
# Per-county regressor prediction — feature-completeness check + predict
# -----------------------------------------------------------------------------


def _county_is_predictable_by_regressor(query_row: pd.Series, forecast_date: str) -> bool:
    """True iff the query row has all regressor feature cols non-null.

    XGBoost handles NaN natively, so we COULD predict on rows with some
    NaNs. But the Phase A.6 contract says the master table's listed feature
    columns are NaN-free in healthy data; if a NaN appears at predict time,
    that's a data-quality red flag we want to surface (skip + log) rather
    than silently let the booster route it to a default-direction split.
    """
    enriched = _add_derived_columns(query_row.to_frame().T)
    cols = FEATURE_COLS[forecast_date]
    return enriched[cols].notna().all().all()


def predict_state_from_regressor(
    bundle: RegressorBundle,
    query_df: pd.DataFrame,
    state: str,
    year: int,
    forecast_date: str,
) -> Tuple[float, int]:
    """Run the regressor on every queryable county in (state, year, forecast_date),
    acres-weight to state level.

    Returns (state_point_estimate, n_counties_predicted). NaN/0 if no county
    survived feature-completeness or has zero acres.
    """
    sub = query_df[
        (query_df["state_alpha"] == state)
        & (query_df["year"] == year)
        & (query_df["forecast_date"] == forecast_date)
    ]
    if len(sub) == 0:
        return float("nan"), 0

    # Feature-completeness mask (per-row).
    keep_mask = sub.apply(
        lambda r: _county_is_predictable_by_regressor(r, forecast_date), axis=1
    ).to_numpy()
    sub_kept = sub.iloc[keep_mask].reset_index(drop=True)
    if len(sub_kept) == 0:
        return float("nan"), 0

    # County-level predictions.
    preds = bundle.regressors[forecast_date].predict(sub_kept)
    acres = sub_kept["acres_planted_all"].to_numpy(dtype=np.float64)
    valid = ~np.isnan(preds) & ~np.isnan(acres) & (acres > 0)
    if not valid.any():
        return float("nan"), 0

    state_point = float((preds[valid] * acres[valid]).sum() / acres[valid].sum())
    return state_point, int(valid.sum())


# -----------------------------------------------------------------------------
# Per-(year, state, date) backtest row
# -----------------------------------------------------------------------------


def backtest_state_year_date(
    master_df: pd.DataFrame,
    query_df: pd.DataFrame,
    bundle: RegressorBundle,
    index: AnalogIndex,
    trend: CountyTrend,
    state: str,
    year: int,
    forecast_date: str,
    *,
    k: int,
    pool: str,
    percentiles: Tuple[int, ...] = (10, 50, 90),
) -> PhaseCRow:
    """Compute regressor point, analog cone, analog-median point, and 5-yr-mean
    baseline for one (year, state, date). Score against truth.
    """
    # ---- Truth + baseline (Phase B; unchanged) -----------------------------
    truth = state_truth_from_master(master_df, state, year)
    baseline_y, _baseline_n = state_baseline(
        master_df, state, year, forecast_date=forecast_date
    )

    # ---- Regressor state point --------------------------------------------
    point_reg, n_reg = predict_state_from_regressor(
        bundle, query_df, state, year, forecast_date
    )

    # ---- Phase B per-county cones → state aggregation ---------------------
    sub = query_df[
        (query_df["state_alpha"] == state)
        & (query_df["year"] == year)
        & (query_df["forecast_date"] == forecast_date)
    ]
    cones_by_geoid: Dict[str, Cone] = {}
    for _, row in sub.iterrows():
        cone = forecast_county_cone(index, trend, row, k=k, pool=pool, percentiles=percentiles)
        if cone is None:
            continue
        cones_by_geoid[str(row["GEOID"])] = cone

    if cones_by_geoid:
        records = build_records_from_master(cones_by_geoid, master_df, state, year)
        sf: StateForecast = state_forecast_from_records(records, state, year, forecast_date)
        p10, p50, p90 = sf.percentiles[10], sf.percentiles[50], sf.percentiles[90]
        # The Phase B per-county cone's point_estimate is the analog-median.
        # state_forecast_from_records acres-weights it to state level.
        point_analog = sf.point_estimate
        n_analog = sf.n_counties
    else:
        p10 = p50 = p90 = float("nan")
        point_analog = float("nan")
        n_analog = 0

    # ---- Score -------------------------------------------------------------
    err_reg = point_reg - truth if not (np.isnan(point_reg) or np.isnan(truth)) else float("nan")
    err_analog = (
        point_analog - truth
        if not (np.isnan(point_analog) or np.isnan(truth))
        else float("nan")
    )
    in_cone_truth = (
        bool(p10 <= truth <= p90)
        if not (np.isnan(p10) or np.isnan(p90) or np.isnan(truth))
        else False
    )
    in_cone_reg = (
        bool(p10 <= point_reg <= p90)
        if not (np.isnan(p10) or np.isnan(p90) or np.isnan(point_reg))
        else False
    )

    return PhaseCRow(
        holdout_year=year,
        state_alpha=state,
        forecast_date=forecast_date,
        n_counties_regressor=n_reg,
        n_counties_analog=n_analog,
        truth_state_yield=truth,
        point_regressor=point_reg,
        error_regressor=err_reg,
        point_analog_median=point_analog,
        error_analog_median=err_analog,
        p10=p10,
        p50=p50,
        p90=p90,
        cone_width_80=p90 - p10 if not (np.isnan(p10) or np.isnan(p90)) else float("nan"),
        in_cone_80_regressor=in_cone_reg,
        in_cone_80_truth=in_cone_truth,
        baseline_yield=baseline_y,
        baseline_error=baseline_y - truth if not np.isnan(truth) else float("nan"),
    )


# -----------------------------------------------------------------------------
# Full backtest sweep
# -----------------------------------------------------------------------------


def run_backtest(
    master_df: pd.DataFrame,
    bundle: RegressorBundle,
    holdout_years: Sequence[int],
    states: Sequence[str],
    forecast_dates: Sequence[str],
    *,
    k: int = 5,
    pool: str = "same_geoid",
    n_min_history: int = 10,
    percentiles: Tuple[int, ...] = (10, 50, 90),
    verbose: bool = True,
) -> pd.DataFrame:
    """Build train pool / standardizer / trend / AnalogIndex once, then sweep
    every (year, state, date) computing one PhaseCRow.

    Phase B locked config defaults: pool=same_geoid, K=5.
    """
    # Train-pool setup mirrors backtest_baseline.run_backtest exactly.
    train_df, mh_result = train_pool(master_df, n_min_history=n_min_history)
    if verbose:
        print(
            f"[setup] train pool: {len(train_df):,} rows, "
            f"{mh_result.n_kept} GEOIDs kept (≥{n_min_history} qualifying years), "
            f"{mh_result.n_dropped} dropped"
        )

    standardizer = fit_standardizer(train_df)
    trend = fit_trend(train_df)
    index = AnalogIndex.fit(train_df, standardizer, trend)
    if verbose:
        for d in VALID_FORECAST_DATES:
            print(
                f"[setup] index at {d}: "
                f"{index.n_candidates(d):,} candidates, "
                f"std n_train={standardizer.n_train_rows[d]:,}"
            )
        print(f"[setup] bundle dates fit: {sorted(bundle.regressors.keys())}")

    # Query pools by holdout year.
    query_pools: Dict[int, pd.DataFrame] = {}
    for y in holdout_years:
        if y == 2023:
            query_pools[y] = val_pool(master_df)
        elif y == 2024:
            query_pools[y] = holdout_pool(master_df)
        else:
            query_pools[y] = master_df[master_df["year"] == y].copy().reset_index(drop=True)

    rows: List[PhaseCRow] = []
    t0 = time.time()
    for year in holdout_years:
        qpool = query_pools[year]
        for state in states:
            for date in forecast_dates:
                row = backtest_state_year_date(
                    master_df=master_df,
                    query_df=qpool,
                    bundle=bundle,
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
                    cone_mark = " " if row.in_cone_80_truth else "*"
                    print(
                        f"[{year} {state} {date:>5}] "
                        f"truth={row.truth_state_yield:6.1f}  "
                        f"reg={row.point_regressor:6.1f}({row.error_regressor:+5.1f})  "
                        f"analog={row.point_analog_median:6.1f}({row.error_analog_median:+5.1f})  "
                        f"cone=[{row.p10:6.1f},{row.p90:6.1f}]{cone_mark}  "
                        f"baseline={row.baseline_yield:6.1f}  "
                        f"n_cty(reg/analog)={row.n_counties_regressor}/{row.n_counties_analog}"
                    )

    if verbose:
        print(f"[backtest] total wall time: {time.time() - t0:.1f}s")

    return pd.DataFrame([asdict(r) for r in rows])


# -----------------------------------------------------------------------------
# Summarize
# -----------------------------------------------------------------------------


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to (year, forecast_date): RMSE for regressor, analog-median, baseline.

    The gate cares about the EOS row of year=val.
    """
    out_rows = []
    for (year, date), sub in results.groupby(["holdout_year", "forecast_date"], sort=False):
        valid = sub.dropna(
            subset=["error_regressor", "error_analog_median", "baseline_error"]
        )
        if len(valid) == 0:
            continue
        rmse_reg = float(np.sqrt(np.mean(valid["error_regressor"] ** 2)))
        rmse_analog = float(np.sqrt(np.mean(valid["error_analog_median"] ** 2)))
        rmse_baseline = float(np.sqrt(np.mean(valid["baseline_error"] ** 2)))
        coverage_truth = float(valid["in_cone_80_truth"].astype(bool).mean())
        coverage_reg_in_cone = float(valid["in_cone_80_regressor"].astype(bool).mean())
        mean_width = float(valid["cone_width_80"].mean())
        # Gate metric: lift = (analog - regressor) / analog
        if rmse_analog > 0:
            lift = (rmse_analog - rmse_reg) / rmse_analog
        else:
            lift = float("nan")
        out_rows.append({
            "year": year,
            "forecast_date": date,
            "n": len(valid),
            "rmse_regressor": rmse_reg,
            "rmse_analog_median": rmse_analog,
            "rmse_baseline": rmse_baseline,
            "lift_vs_analog": lift,
            "coverage_80_truth": coverage_truth,
            "regressor_in_cone_80": coverage_reg_in_cone,
            "mean_cone_width": mean_width,
        })
    return pd.DataFrame(out_rows).sort_values(["year", "forecast_date"]).reset_index(drop=True)


def summarize_by_state(results: pd.DataFrame) -> pd.DataFrame:
    """Per-(year, state) breakdown — averaged across forecast_dates.

    Useful for spotting which state is dragging the headline numbers and for
    making the recalibration call.
    """
    out_rows = []
    for (year, state), sub in results.groupby(["holdout_year", "state_alpha"], sort=False):
        valid = sub.dropna(subset=["error_regressor", "error_analog_median"])
        if len(valid) == 0:
            continue
        out_rows.append({
            "year": year,
            "state": state,
            "n": len(valid),
            "rmse_regressor": float(np.sqrt(np.mean(valid["error_regressor"] ** 2))),
            "bias_regressor": float(valid["error_regressor"].mean()),
            "rmse_analog_median": float(np.sqrt(np.mean(valid["error_analog_median"] ** 2))),
            "bias_analog_median": float(valid["error_analog_median"].mean()),
            "rmse_baseline": float(np.sqrt(np.mean(valid["baseline_error"] ** 2))),
            "coverage_80_truth": float(valid["in_cone_80_truth"].astype(bool).mean()),
        })
    return pd.DataFrame(out_rows).sort_values(["year", "state"]).reset_index(drop=True)


def per_state_date_residuals(results: pd.DataFrame, year: int) -> pd.DataFrame:
    """Per-(state, forecast_date) signed residuals for one year.

    Used to inform the recalibration decision (Phase 2-C kickoff #8: "decide
    after first training run sees the residuals"). Positive = regressor over,
    negative = regressor under.
    """
    sub = results[results["holdout_year"] == year].copy()
    return (
        sub[["state_alpha", "forecast_date", "error_regressor", "error_analog_median"]]
        .rename(
            columns={
                "error_regressor": "resid_regressor",
                "error_analog_median": "resid_analog_median",
            }
        )
        .sort_values(["state_alpha", "forecast_date"])
        .reset_index(drop=True)
    )


# -----------------------------------------------------------------------------
# Gate evaluation
# -----------------------------------------------------------------------------


def evaluate_gate(summary: pd.DataFrame, val_year: int = 2023) -> Dict:
    """Phase C gate: trained-regressor state RMSE on val_year ≥ 15% better than
    Phase B analog-median (pre-recal) at end-of-season.

    The phase plan also says "earlier dates can be weaker; we just need to
    confirm the model adds value." We report lift at every date but the
    pass/fail bar is EOS only.
    """
    val_rows = summary[summary["year"] == val_year]
    if val_rows.empty:
        return {"val_year": val_year, "passed": False, "reason": "no val rows"}

    eos = val_rows[val_rows["forecast_date"] == "EOS"]
    if eos.empty:
        return {"val_year": val_year, "passed": False, "reason": "no EOS row"}
    if len(eos) > 1:
        return {"val_year": val_year, "passed": False, "reason": "duplicate EOS rows"}

    eos_lift = float(eos.iloc[0]["lift_vs_analog"])
    rmse_reg = float(eos.iloc[0]["rmse_regressor"])
    rmse_analog = float(eos.iloc[0]["rmse_analog_median"])
    passed = eos_lift >= GATE_LIFT_THRESHOLD

    per_date = {}
    for _, r in val_rows.iterrows():
        per_date[r["forecast_date"]] = {
            "rmse_regressor": float(r["rmse_regressor"]),
            "rmse_analog_median": float(r["rmse_analog_median"]),
            "lift": float(r["lift_vs_analog"]),
        }

    return {
        "val_year": val_year,
        "eos_rmse_regressor": rmse_reg,
        "eos_rmse_analog_median": rmse_analog,
        "eos_lift_vs_analog": eos_lift,
        "threshold": GATE_LIFT_THRESHOLD,
        "passed": passed,
        "per_date": per_date,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parse_csv_arg(s: str, kind: type = str) -> List:
    return [kind(x.strip()) for x in s.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase C trained-regressor backtest")
    parser.add_argument("--master", default=DEFAULT_MASTER_PATH)
    parser.add_argument(
        "--bundle-dir",
        default=DEFAULT_BUNDLE_DIR,
        help="directory containing regressor_<date>.json bundle",
    )
    parser.add_argument("--holdout-years", default="2023,2024")
    parser.add_argument("--states", default="CO,IA,MO,NE,WI")
    parser.add_argument("--forecast-dates", default="08-01,09-01,10-01,EOS")
    parser.add_argument("--k", type=int, default=5, help="K for analog cone (locked at 5)")
    parser.add_argument("--pool", default="same_geoid", help="Phase B pool (locked at same_geoid)")
    parser.add_argument("--n-min-history", type=int, default=10)
    parser.add_argument(
        "--recal",
        default="none",
        choices=["none", "regressor", "phase_b", "both"],
        help="Recalibration mode (default: none, per Phase 2-C kickoff). "
             "'regressor' refits a per-(state, date) shift on the regressor's val "
             "errors. 'phase_b' applies it to the analog-median path. 'both' does "
             "both. CURRENTLY only 'none' is implemented; flag is wired for next "
             "iteration after we see residuals.",
    )
    parser.add_argument("--out", default=None, help="output CSV path")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.recal != "none":
        print(
            f"[error] --recal {args.recal!r} is not implemented yet. The decision "
            f"on whether to recalibrate is deferred until residuals are seen "
            f"(per Phase 2-C kickoff decision #8). Re-run with --recal none "
            f"first; we'll wire 'regressor' / 'phase_b' / 'both' if needed."
        )
        return 2

    holdout_years = _parse_csv_arg(args.holdout_years, int)
    states = _parse_csv_arg(args.states)
    forecast_dates = _parse_csv_arg(args.forecast_dates)
    verbose = not args.quiet

    # ---- Load --------------------------------------------------------------
    print(f"[load] {args.master}")
    master_df = load_master(args.master)
    print(f"[load] {len(master_df):,} rows × {len(master_df.columns)} cols")

    print(f"[load] bundle ← {args.bundle_dir}")
    bundle = RegressorBundle.load(args.bundle_dir)
    for date in VALID_FORECAST_DATES:
        reg = bundle.regressors[date]
        p = reg.params
        print(
            f"          {date}: depth={p['max_depth']} lr={p['learning_rate']} "
            f"mcw={p['min_child_weight']} best_iter={reg.best_iteration} "
            f"val_rmse(county-level)={reg.train_metrics['val_rmse']:.2f}"
        )

    # ---- Backtest ----------------------------------------------------------
    results = run_backtest(
        master_df=master_df,
        bundle=bundle,
        holdout_years=holdout_years,
        states=states,
        forecast_dates=forecast_dates,
        k=args.k,
        pool=args.pool,
        n_min_history=args.n_min_history,
        verbose=verbose,
    )

    # ---- Persist -----------------------------------------------------------
    out_path = args.out
    if out_path is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = f"runs/backtest_phase_c_{ts}.csv"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"\n[write] {out_path}")

    # ---- Summary tables ----------------------------------------------------
    summary = summarize(results)
    print("\n=== summary (per year × forecast_date) ===")
    with pd.option_context("display.max_rows", None, "display.width", 160, "display.precision", 3):
        print(summary.to_string(index=False))

    by_state = summarize_by_state(results)
    print("\n=== per-state breakdown (per year × state, averaged over forecast_dates) ===")
    with pd.option_context("display.max_rows", None, "display.width", 160, "display.precision", 3):
        print(by_state.to_string(index=False))

    # ---- Per-(state, date) residuals on val (for the recal decision) ------
    val_year = min(holdout_years)
    print(f"\n=== per-(state, forecast_date) residuals on val={val_year} (for recal decision) ===")
    resids = per_state_date_residuals(results, val_year)
    with pd.option_context("display.max_rows", None, "display.width", 160, "display.precision", 2):
        print(resids.to_string(index=False))

    # ---- Gate verdict ------------------------------------------------------
    gate = evaluate_gate(summary, val_year=val_year)
    print(f"\n=== Phase C gate (val={val_year}) ===")
    if "reason" in gate:
        print(f"  cannot evaluate: {gate['reason']}")
        return 1
    print(f"  EOS rmse_regressor:     {gate['eos_rmse_regressor']:.3f}")
    print(f"  EOS rmse_analog_median: {gate['eos_rmse_analog_median']:.3f}")
    print(f"  EOS lift vs analog:     {gate['eos_lift_vs_analog']*100:+.1f}%   "
          f"(threshold ≥ {gate['threshold']*100:.0f}%)")
    print(f"  per-date lift:")
    for d in forecast_dates:
        if d in gate["per_date"]:
            entry = gate["per_date"][d]
            print(
                f"    {d:>5}: rmse_reg={entry['rmse_regressor']:6.2f}  "
                f"rmse_analog={entry['rmse_analog_median']:6.2f}  "
                f"lift={entry['lift']*100:+6.1f}%"
            )
    print(f"\n  PHASE C GATE: {'PASS' if gate['passed'] else 'FAIL'}")
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
