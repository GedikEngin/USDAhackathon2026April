"""
scripts/train_regressor.py — Phase C XGBoost training driver.

Sweeps a small hyperparameter grid per forecast_date, picks the best config
by val RMSE (early-stopped), assembles a RegressorBundle, and writes it to
disk along with a sweep-results CSV.

Usage
-----
    python -m scripts.train_regressor
    python -m scripts.train_regressor --out-dir models/forecast/v1
    python -m scripts.train_regressor --no-sweep              # fit once with defaults
    python -m scripts.train_regressor --max-depths 4,6 --learning-rates 0.05

Outputs
-------
    models/forecast/regressor_{08-01,09-01,10-01,EOS}.json    booster files
    models/forecast/regressor_*.json.meta.json                metadata sidecars
    runs/phase_c_sweep_<timestamp>.csv                        per-(date, config) row

The driver does NOT evaluate the Phase C gate — that's
scripts/backtest_phase_c.py, which compares state-rolled-up regressor RMSE on
2023 val against the Phase B analog-median baseline.

Sweep / selection notes
-----------------------
- Selection metric is **county-level val RMSE** (early-stopping metric).
  State-aggregated RMSE has 5 obs per date, too few to drive a sweep. The
  state-level gate eval uses the same trained models but rolls them up via
  forecast.aggregate.
- Once a config is picked, we keep the booster from that sweep run rather
  than refitting — xgb.train is deterministic given seed + tree_method=hist,
  so refit produces an identical booster.
- The bundle's val RMSE numbers are in-sample at the bundle level (val was
  used for both early stopping AND config selection). They reflect "did
  the model train sanely," not generalization. Generalization is the 2024
  holdout, touched once at the end of Phase G.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Make `forecast` importable when run from repo root or as -m.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forecast.data import DEFAULT_MASTER_PATH, load_master, train_pool, val_pool
from forecast.features import VALID_FORECAST_DATES
from forecast.regressor import (
    DEFAULT_EARLY_STOPPING_ROUNDS,
    DEFAULT_NUM_BOOST_ROUND,
    DEFAULT_PARAMS,
    Regressor,
    RegressorBundle,
    fit as fit_regressor,
)


# -----------------------------------------------------------------------------
# Sweep grid defaults (overridable via CLI)
# -----------------------------------------------------------------------------

DEFAULT_MAX_DEPTHS = (4, 6, 8)
DEFAULT_LEARNING_RATES = (0.05, 0.1)
DEFAULT_MIN_CHILD_WEIGHTS = (1, 5)


# -----------------------------------------------------------------------------
# Sweep over one forecast_date
# -----------------------------------------------------------------------------


def _build_grid(
    max_depths: Tuple[int, ...],
    learning_rates: Tuple[float, ...],
    min_child_weights: Tuple[int, ...],
) -> List[Dict[str, object]]:
    """Cartesian product of the three axes → list of param dicts.

    Each entry is a delta on top of DEFAULT_PARAMS. Order: depths outer,
    LR middle, MCW inner — same order as the printed sweep log so a reader
    can scan top-to-bottom and see depth changing slowest.
    """
    grid = []
    for d, lr, mcw in itertools.product(max_depths, learning_rates, min_child_weights):
        grid.append({
            "max_depth": d,
            "learning_rate": lr,
            "min_child_weight": mcw,
        })
    return grid


def sweep_one_date(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    forecast_date: str,
    grid: List[Dict[str, object]],
    *,
    num_boost_round: int,
    early_stopping_rounds: int,
    verbose: bool,
) -> Tuple[Regressor, List[Dict[str, object]]]:
    """Sweep `grid` for one forecast_date, return (best_regressor, sweep_rows).

    `sweep_rows` has one dict per config with the keys logged into the
    output CSV.
    """
    sweep_rows: List[Dict[str, object]] = []
    best_reg: Regressor | None = None
    best_val_rmse = float("inf")

    for i, delta in enumerate(grid, start=1):
        t0 = time.time()
        reg = fit_regressor(
            train_df,
            val_df,
            forecast_date=forecast_date,
            params=delta,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        elapsed = time.time() - t0

        train_rmse = reg.train_metrics["train_rmse"]
        val_rmse = reg.train_metrics["val_rmse"]
        sweep_rows.append({
            "forecast_date": forecast_date,
            "config_idx": i,
            "max_depth": delta["max_depth"],
            "learning_rate": delta["learning_rate"],
            "min_child_weight": delta["min_child_weight"],
            "best_iteration": reg.best_iteration,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "n_train": reg.n_train,
            "n_val": reg.n_val,
            "elapsed_sec": round(elapsed, 2),
        })

        if verbose:
            tag = ""
            if val_rmse < best_val_rmse:
                tag = " ←best"
            print(
                f"  [{forecast_date} cfg {i:>2}/{len(grid)}] "
                f"depth={delta['max_depth']} lr={delta['learning_rate']:<5} "
                f"mcw={delta['min_child_weight']} "
                f"→ best_iter={reg.best_iteration:>4}  "
                f"train_rmse={train_rmse:6.2f}  val_rmse={val_rmse:6.2f}  "
                f"({elapsed:4.1f}s){tag}"
            )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_reg = reg

    assert best_reg is not None, "Sweep produced no Regressor — empty grid?"
    return best_reg, sweep_rows


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


def _parse_csv_arg(s: str, kind: type = str) -> List:
    return [kind(x.strip()) for x in s.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase C XGBoost training driver — per-date sweep + bundle save"
    )
    parser.add_argument(
        "--master",
        default=DEFAULT_MASTER_PATH,
        help="path to training_master.parquet",
    )
    parser.add_argument(
        "--out-dir",
        default="models/forecast",
        help="directory to write the RegressorBundle (per-date booster files)",
    )
    parser.add_argument(
        "--sweep-csv",
        default=None,
        help="output CSV for sweep results (default: runs/phase_c_sweep_<ts>.csv)",
    )
    parser.add_argument(
        "--n-min-history",
        type=int,
        default=10,
        help="minimum qualifying training years per GEOID (matches Phase B)",
    )
    parser.add_argument(
        "--max-depths",
        default=",".join(str(d) for d in DEFAULT_MAX_DEPTHS),
        help="comma-separated max_depth values to sweep",
    )
    parser.add_argument(
        "--learning-rates",
        default=",".join(str(lr) for lr in DEFAULT_LEARNING_RATES),
        help="comma-separated learning_rate values to sweep",
    )
    parser.add_argument(
        "--min-child-weights",
        default=",".join(str(m) for m in DEFAULT_MIN_CHILD_WEIGHTS),
        help="comma-separated min_child_weight values to sweep",
    )
    parser.add_argument(
        "--num-boost-round",
        type=int,
        default=DEFAULT_NUM_BOOST_ROUND,
        help=f"max boosting rounds (default {DEFAULT_NUM_BOOST_ROUND})",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=DEFAULT_EARLY_STOPPING_ROUNDS,
        help=f"early-stopping patience (default {DEFAULT_EARLY_STOPPING_ROUNDS})",
    )
    parser.add_argument(
        "--no-sweep",
        action="store_true",
        help="skip the sweep — fit one model per date with DEFAULT_PARAMS",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress per-config logging")
    args = parser.parse_args()

    verbose = not args.quiet

    # ---- Load + slice ------------------------------------------------------

    print(f"[load] {args.master}")
    master_df = load_master(args.master)
    print(f"[load] {len(master_df):,} rows × {len(master_df.columns)} cols")

    train_df, mh_result = train_pool(master_df, n_min_history=args.n_min_history)
    val_df = val_pool(master_df)
    print(
        f"[setup] train pool: {len(train_df):,} rows, "
        f"{mh_result.n_kept} GEOIDs kept (≥{args.n_min_history} qualifying years), "
        f"{mh_result.n_dropped} dropped"
    )
    print(f"[setup] val pool:   {len(val_df):,} rows (year={sorted(val_df['year'].unique())})")

    # ---- Build grid --------------------------------------------------------

    if args.no_sweep:
        grid: List[Dict[str, object]] = [{
            "max_depth": int(DEFAULT_PARAMS["max_depth"]),
            "learning_rate": float(DEFAULT_PARAMS["learning_rate"]),
            "min_child_weight": int(DEFAULT_PARAMS["min_child_weight"]),
        }]
        print("[setup] --no-sweep: fitting once per date with DEFAULT_PARAMS")
    else:
        max_depths = tuple(_parse_csv_arg(args.max_depths, int))
        learning_rates = tuple(_parse_csv_arg(args.learning_rates, float))
        min_child_weights = tuple(_parse_csv_arg(args.min_child_weights, int))
        grid = _build_grid(max_depths, learning_rates, min_child_weights)
        print(
            f"[setup] sweep grid: "
            f"{len(max_depths)} max_depths × "
            f"{len(learning_rates)} lrs × "
            f"{len(min_child_weights)} mcws "
            f"= {len(grid)} configs/date × {len(VALID_FORECAST_DATES)} dates "
            f"= {len(grid) * len(VALID_FORECAST_DATES)} total fits"
        )

    # ---- Per-date sweep ----------------------------------------------------

    bundle = RegressorBundle()
    all_sweep_rows: List[Dict[str, object]] = []
    t_start = time.time()
    for date in VALID_FORECAST_DATES:
        if verbose:
            print(f"\n[sweep] forecast_date={date}")
        best_reg, rows = sweep_one_date(
            train_df,
            val_df,
            forecast_date=date,
            grid=grid,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
            verbose=verbose,
        )
        bundle.regressors[date] = best_reg
        all_sweep_rows.extend(rows)
    total_elapsed = time.time() - t_start
    print(f"\n[sweep] total wall time: {total_elapsed:.1f}s")

    # ---- Persist bundle ----------------------------------------------------

    out_dir = Path(args.out_dir)
    bundle.save(out_dir)
    print(f"\n[write] bundle → {out_dir}/")
    for date in VALID_FORECAST_DATES:
        safe = date.replace(":", "_")
        print(f"          regressor_{safe}.json  (+ .meta.json)")

    # ---- Persist sweep CSV -------------------------------------------------

    sweep_path = args.sweep_csv
    if sweep_path is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        sweep_path = f"runs/phase_c_sweep_{ts}.csv"
    Path(sweep_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_sweep_rows).to_csv(sweep_path, index=False)
    print(f"[write] sweep results → {sweep_path}")

    # ---- Per-date best summary --------------------------------------------

    print("\n=== Per-date best config (selected by val_rmse, county-level) ===")
    print(
        f"  {'date':>5}  {'depth':>5}  {'lr':>5}  {'mcw':>3}  "
        f"{'best_iter':>9}  {'train_rmse':>10}  {'val_rmse':>8}  "
        f"{'n_train':>7}  {'n_val':>5}"
    )
    for date in VALID_FORECAST_DATES:
        reg = bundle.regressors[date]
        p = reg.params
        print(
            f"  {date:>5}  "
            f"{p['max_depth']:>5}  "
            f"{p['learning_rate']:>5.2f}  "
            f"{p['min_child_weight']:>3}  "
            f"{reg.best_iteration:>9}  "
            f"{reg.train_metrics['train_rmse']:>10.2f}  "
            f"{reg.train_metrics['val_rmse']:>8.2f}  "
            f"{reg.n_train:>7,}  "
            f"{reg.n_val:>5,}"
        )

    print(
        "\nNote: val_rmse above is in-sample at the bundle level (val used for "
        "early stopping AND config selection). Generalization estimate is the "
        "2024 holdout, evaluated by scripts/backtest_phase_c.py + Phase G."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
