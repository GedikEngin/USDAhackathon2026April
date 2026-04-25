"""
scripts/train_regressor_d1.py — Phase D.1 XGBoost training driver (Row B).

Forks scripts/train_regressor.py to produce the engineered+Prithvi regressor
for the D.1.e gate test ("Row B"). Inherits Phase C's per-date booster
architecture verbatim — same xgb engine, same sweep grid, same SHAP path,
same save format — but with three Phase D.1-specific changes:

  1. Train pool narrowed to 2013-2022 (HLS only exists 2013+).
     Phase C's train pool is 2005-2022; we restrict to make the D.1.e
     ablation apples-to-apples (Row A engineered-only ALSO uses 2013-2022,
     so any Row B vs Row A delta is attributable to Prithvi, not pool size).

  2. Features extended with the Prithvi embedding columns.
     - 1024 prithvi_NNNN columns from data/v2/prithvi/embeddings_v1.parquet
     - 4 QC scalars (chip_count, chip_age_days_max, cloud_pct_max,
       corn_pixel_frac_min) per decisions log: become regressor features
       so the model learns to discount queries with low chip availability

  3. Output directory is models/forecast_d1/ (not models/forecast/).
     Phase C's bundle stays put; the gate-test runner loads both.

The narrow train pool also means the GEOID min-history filter from Phase B/C
must be RECOMPUTED, not reused. A GEOID with >= 10 qualifying years on the
2005-2022 pool may have <10 on the 2013-2022 pool. The driver applies
forecast.data.apply_min_history_filter() with n_min_history=8 (one less
than Phase C's 10) to keep most counties in the pool.

Embedding completeness:
  - Queries with no chips (chip_count==0) end up with 1024 NaN prithvi_NNNN
    cols. XGBoost handles NaN natively via missing-direction split, so we
    DO NOT drop those rows. The chip_count=0 rows still contribute to
    training via their engineered features.
  - The 4 QC scalar columns are also NaN for chip_count==0 rows. Same
    handling — let XGBoost route them.

Usage
-----
    python scripts/train_regressor_d1.py                          # default sweep
    python scripts/train_regressor_d1.py --no-sweep                # quick fit
    python scripts/train_regressor_d1.py --max-depths 6,8 \
        --learning-rates 0.05                                      # custom grid
    python scripts/train_regressor_d1.py --row A                   # Row A: engineered-only
    python scripts/train_regressor_d1.py --no-qc                   # skip QC features

The --row flag controls which ablation row this run produces:
    --row B   Row B: engineered + Prithvi + QC  (default; the headline run)
    --row A   Row A: engineered only on 2013-2022  (for D.1.e ablation)

Outputs
-------
    models/forecast_d1/regressor_{08-01,09-01,10-01,EOS}.json    booster files
    models/forecast_d1/regressor_*.json.meta.json                metadata sidecars
    models/forecast_d1/run_manifest.json                         row label, train pool, etc.
    runs/d1_sweep_<timestamp>.csv                                per-(date, config) sweep row

Re-runs are safe (overwrite-mode by default). Use --out-dir to write to a
versioned location (e.g. models/forecast_d1/v2/) without overwriting.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Make `forecast` importable when run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forecast.data import (
    DEFAULT_MASTER_PATH,
    load_master,
    apply_min_history_filter,
)
from forecast.features import VALID_FORECAST_DATES
from forecast.regressor import (
    DEFAULT_EARLY_STOPPING_ROUNDS,
    DEFAULT_NUM_BOOST_ROUND,
    DEFAULT_PARAMS,
    FEATURE_COLS,
    Regressor,
    RegressorBundle,
    fit as fit_regressor,
)
# Pull in Prithvi metadata; this import doesn't load torch (the constants are
# module-scope and cheap)
from forecast.prithvi import (
    EMBEDDING_COL_NAMES,
    EMBEDDING_DIM,
    MODEL_VERSION as PRITHVI_MODEL_VERSION,
)


# =============================================================================
# Phase D.1 specific defaults
# =============================================================================

# HLS coverage: 2013+ per decisions log D.1.kickoff
D1_TRAIN_YEARS: Tuple[int, ...] = tuple(range(2013, 2023))   # 2013-2022 inclusive
D1_VAL_YEARS:   Tuple[int, ...] = (2023,)
D1_HOLDOUT_YEARS: Tuple[int, ...] = (2024,)

# Sweep grid (same as Phase C — D.1.e gate test uses identical hyperparameter
# grid for fairness across rows)
DEFAULT_MAX_DEPTHS = (4, 6, 8)
DEFAULT_LEARNING_RATES = (0.05, 0.1)
DEFAULT_MIN_CHILD_WEIGHTS = (1, 5)

# QC scalar columns produced by extract_embeddings.py
QC_FEATURE_COLS: List[str] = [
    "chip_count",
    "chip_age_days_max",
    "cloud_pct_max",
    "corn_pixel_frac_min",
]

# Where embeddings live
DEFAULT_EMBEDDINGS_PATH = "data/v2/prithvi/embeddings_v1.parquet"
DEFAULT_OUT_DIR         = "models/forecast_d1"


# =============================================================================
# Embedding join
# =============================================================================

def join_embeddings(
    master_df: pd.DataFrame,
    embeddings_path: Path,
    *,
    log: bool = True,
) -> pd.DataFrame:
    """Left-join embeddings_v1.parquet onto master_df.

    Join keys: (GEOID, year, forecast_date).

    Rows with no matching embedding (e.g. years pre-2013, queries that had
    chip_count==0) get NaN prithvi_NNNN columns. XGBoost handles NaN natively.

    Pre-conditions checked:
      - GEOID in master is 5-char string (or coercible)
      - embeddings parquet exists and has expected columns
      - exactly one row per (GEOID, year, forecast_date) in embeddings
        (after filtering to the current model_version)

    Returns the augmented DataFrame.
    """
    if not embeddings_path.exists():
        raise SystemExit(
            f"Embeddings parquet not found: {embeddings_path}\n"
            f"Run scripts/extract_embeddings.py first."
        )

    if log:
        print(f"[join] loading embeddings from {embeddings_path}")
    emb = pd.read_parquet(embeddings_path)

    # Filter to the current model_version. Re-runs of extract_embeddings.py
    # under the same MODEL_VERSION will overwrite duplicates via its
    # crashsafe-write + dedupe path; this is just a defensive filter for the
    # case where multiple model_versions coexist.
    if "model_version" in emb.columns:
        before = len(emb)
        emb = emb[emb["model_version"] == PRITHVI_MODEL_VERSION].copy()
        if log:
            print(f"  filtered to model_version={PRITHVI_MODEL_VERSION}: "
                  f"{before} -> {len(emb)} rows")

    # Validate required columns
    required = ["GEOID", "year", "forecast_date"] + QC_FEATURE_COLS + EMBEDDING_COL_NAMES
    missing = [c for c in required if c not in emb.columns]
    if missing:
        raise KeyError(
            f"embeddings parquet missing required columns: {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''} (total {len(missing)} missing)"
        )

    # Validate uniqueness on join keys
    dup_mask = emb.duplicated(subset=["GEOID", "year", "forecast_date"], keep=False)
    if dup_mask.any():
        raise ValueError(
            f"embeddings parquet has {int(dup_mask.sum())} duplicate "
            f"(GEOID, year, forecast_date) rows after model_version filter; "
            f"this would multiply rows on join. Re-run extract_embeddings.py "
            f"--rebuild to fix."
        )

    # Coerce types for the join. Master uses str-padded GEOID; ensure
    # both sides do the same.
    emb["GEOID"] = emb["GEOID"].astype(str).str.zfill(5)
    master_df = master_df.copy()
    master_df["GEOID"] = master_df["GEOID"].astype(str).str.zfill(5)
    emb["year"] = emb["year"].astype(int)
    master_df["year"] = master_df["year"].astype(int)
    emb["forecast_date"] = emb["forecast_date"].astype(str)
    master_df["forecast_date"] = master_df["forecast_date"].astype(str)

    # Drop the model_version column from the embeddings before join (we already
    # filtered on it; carrying it through clutters the output and would
    # collide if master were ever joined with multiple versions).
    join_cols = ["GEOID", "year", "forecast_date"] + QC_FEATURE_COLS + EMBEDDING_COL_NAMES
    emb_for_join = emb[join_cols]

    n_before = len(master_df)
    out = master_df.merge(emb_for_join, on=["GEOID", "year", "forecast_date"], how="left")
    n_after = len(out)

    if n_after != n_before:
        raise RuntimeError(
            f"left-join changed row count from {n_before} to {n_after}; "
            f"embeddings parquet has duplicates on join keys (this should "
            f"have been caught above)."
        )

    n_matched = int(out[EMBEDDING_COL_NAMES[0]].notna().sum())
    if log:
        print(f"  master rows: {n_before:,}  matched-with-embedding: "
              f"{n_matched:,} ({100*n_matched/n_before:.1f}%)")
    return out


# =============================================================================
# FEATURE_COLS extension
# =============================================================================

def extend_feature_cols(
    *,
    include_prithvi: bool,
    include_qc: bool,
    log: bool = True,
) -> None:
    """Append D.1's new columns to FEATURE_COLS in place.

    Phase C's `regressor.FEATURE_COLS` is a module-scope dict mapping
    forecast_date -> list of feature names. The Phase C `_build_dmatrix`
    reads this dict at fit/predict time, so appending columns here is
    sufficient to get them into the trained booster's feature matrix.

    Order matters: appended columns become the rightmost feature_names in
    the saved booster. This means the saved booster's feature_names list is
    [Phase-C cols..., (4 QC cols), (1024 prithvi cols)]. The order is locked
    once a model is saved and must be reproduced on load — Regressor.load
    reads feature_cols from the meta.json sidecar so this is safe.

    Caveat: this mutates a module-level dict. Calling this twice in the
    same process would double-append. We guard against that.
    """
    extra: List[str] = []
    if include_qc:
        extra += QC_FEATURE_COLS
    if include_prithvi:
        extra += EMBEDDING_COL_NAMES

    if not extra:
        if log:
            print("[features] no D.1 extensions (Row A: engineered-only)")
        return

    for date in VALID_FORECAST_DATES:
        cols = FEATURE_COLS[date]
        # Idempotency guard: skip extras already present
        already = set(cols)
        new = [c for c in extra if c not in already]
        cols.extend(new)

    if log:
        n_qc = len(QC_FEATURE_COLS) if include_qc else 0
        n_pr = EMBEDDING_DIM if include_prithvi else 0
        first_date = VALID_FORECAST_DATES[0]
        print(f"[features] extended FEATURE_COLS: +{n_qc} QC, +{n_pr} prithvi")
        print(f"  total per-date count: {first_date}={len(FEATURE_COLS[first_date])}, "
              f"EOS={len(FEATURE_COLS['EOS'])}")


# =============================================================================
# Train pool slicer for D.1
# =============================================================================

def d1_train_pool(
    df: pd.DataFrame,
    *,
    n_min_history: int = 8,
) -> Tuple[pd.DataFrame, int]:
    """Slice df to the D.1 training pool: 2013-2022, post-min-history.

    Differs from forecast.data.train_pool() in two ways:
      1. Year filter is 2013-2022, not 2005-2022
      2. n_min_history default is 8, not 10 (the narrower year window has
         fewer qualifying years per GEOID by construction)
      3. Does NOT enforce embedding-completeness (we want chip_count==0
         rows IN the train pool — XGBoost handles via missing-direction split)
      4. Still requires yield_target non-null (training labels)

    Returns (train_df, n_geoids_kept).
    """
    filtered, mh = apply_min_history_filter(df, n=n_min_history)
    train_mask = filtered["year"].isin(D1_TRAIN_YEARS)
    train_df = filtered.loc[train_mask].copy()
    target_complete = train_df["yield_target"].notna()
    train_df = train_df.loc[target_complete].reset_index(drop=True)
    return train_df, mh.n_kept


def d1_val_pool(df: pd.DataFrame) -> pd.DataFrame:
    """2023 rows; no min-history; yield_target non-null required for early-stop."""
    val = df[df["year"].isin(D1_VAL_YEARS)].copy().reset_index(drop=True)
    val = val[val["yield_target"].notna()].reset_index(drop=True)
    return val


# =============================================================================
# Sweep (mirrors scripts/train_regressor.py::sweep_one_date verbatim)
# =============================================================================

def _build_grid(
    max_depths: Tuple[int, ...],
    learning_rates: Tuple[float, ...],
    min_child_weights: Tuple[int, ...],
) -> List[Dict[str, object]]:
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
    sweep_rows: List[Dict[str, object]] = []
    best_reg: Optional[Regressor] = None
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
        val_rmse   = reg.train_metrics["val_rmse"]
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
            tag = " ←best" if val_rmse < best_val_rmse else ""
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


# =============================================================================
# Driver
# =============================================================================

def _parse_csv_arg(s: str, kind: type = str) -> List:
    return [kind(x.strip()) for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase D.1 XGBoost training driver (Row B = engineered+Prithvi)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--master", default=DEFAULT_MASTER_PATH,
                    help=f"path to training_master.parquet (default: {DEFAULT_MASTER_PATH})")
    ap.add_argument("--embeddings", default=DEFAULT_EMBEDDINGS_PATH,
                    help=f"path to embeddings_v1.parquet (default: {DEFAULT_EMBEDDINGS_PATH})")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help=f"output bundle directory (default: {DEFAULT_OUT_DIR})")
    ap.add_argument("--sweep-csv", default=None,
                    help="output CSV for sweep results (default: runs/d1_sweep_<ts>.csv)")
    ap.add_argument("--n-min-history", type=int, default=8,
                    help="minimum qualifying training years per GEOID "
                         "(default 8 for D.1's 10-year pool; Phase C used 10 for 18-year pool)")

    # Ablation row selector
    ap.add_argument("--row", choices=("A", "B"), default="B",
                    help="ablation row: B (default) = engineered+Prithvi, "
                         "A = engineered only (no Prithvi cols, no QC cols)")
    ap.add_argument("--no-qc", action="store_true",
                    help="(Row B only) drop the 4 QC scalar features")

    # Sweep grid
    ap.add_argument("--max-depths", default=",".join(str(d) for d in DEFAULT_MAX_DEPTHS))
    ap.add_argument("--learning-rates", default=",".join(str(lr) for lr in DEFAULT_LEARNING_RATES))
    ap.add_argument("--min-child-weights", default=",".join(str(m) for m in DEFAULT_MIN_CHILD_WEIGHTS))
    ap.add_argument("--num-boost-round", type=int, default=DEFAULT_NUM_BOOST_ROUND)
    ap.add_argument("--early-stopping-rounds", type=int, default=DEFAULT_EARLY_STOPPING_ROUNDS)
    ap.add_argument("--no-sweep", action="store_true",
                    help="skip the sweep — fit one model per date with DEFAULT_PARAMS")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    verbose = not args.quiet

    include_prithvi = (args.row == "B")
    include_qc      = (args.row == "B") and not args.no_qc

    if verbose:
        print(f"[setup] D.1 ablation row: {args.row}  "
              f"(prithvi={'YES' if include_prithvi else 'no'}, "
              f"qc={'YES' if include_qc else 'no'})")
        print(f"[setup] train pool: years {D1_TRAIN_YEARS[0]}-{D1_TRAIN_YEARS[-1]}; "
              f"val: {D1_VAL_YEARS}")

    # --- Load master ---------------------------------------------------------
    print(f"[load] {args.master}")
    master_df = load_master(args.master)
    print(f"[load] {len(master_df):,} rows × {len(master_df.columns)} cols")

    # --- Optional join with embeddings (Row B only) -------------------------
    if include_prithvi or include_qc:
        master_df = join_embeddings(master_df, Path(args.embeddings))

    # --- Extend FEATURE_COLS in place ---------------------------------------
    extend_feature_cols(
        include_prithvi=include_prithvi,
        include_qc=include_qc,
        log=verbose,
    )

    # --- Slice train/val pools -----------------------------------------------
    train_df, n_kept = d1_train_pool(master_df, n_min_history=args.n_min_history)
    val_df = d1_val_pool(master_df)
    print(f"[setup] train pool: {len(train_df):,} rows, {n_kept} GEOIDs kept "
          f"(>={args.n_min_history} qualifying years)")
    print(f"[setup] val pool:   {len(val_df):,} rows (year={sorted(val_df['year'].unique())})")

    if include_prithvi:
        # Diagnostic: how many train rows have a real embedding vs NaN?
        first_emb = EMBEDDING_COL_NAMES[0]
        n_train_with_emb = int(train_df[first_emb].notna().sum())
        n_val_with_emb   = int(val_df[first_emb].notna().sum())
        print(f"[setup] train rows with non-NaN prithvi_0000: "
              f"{n_train_with_emb:,} / {len(train_df):,} "
              f"({100*n_train_with_emb/max(len(train_df),1):.1f}%)")
        print(f"[setup] val rows   with non-NaN prithvi_0000: "
              f"{n_val_with_emb:,} / {len(val_df):,} "
              f"({100*n_val_with_emb/max(len(val_df),1):.1f}%)")

    # --- Build grid ----------------------------------------------------------
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
        print(f"[setup] sweep grid: {len(max_depths)} max_depths × "
              f"{len(learning_rates)} lrs × "
              f"{len(min_child_weights)} mcws "
              f"= {len(grid)} configs/date × {len(VALID_FORECAST_DATES)} dates "
              f"= {len(grid) * len(VALID_FORECAST_DATES)} total fits")

    # --- Per-date sweep ------------------------------------------------------
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

    # --- Persist bundle ------------------------------------------------------
    out_dir = Path(args.out_dir)
    bundle.save(out_dir)
    print(f"\n[write] bundle → {out_dir}/")
    for date in VALID_FORECAST_DATES:
        safe = date.replace(":", "_")
        print(f"          regressor_{safe}.json  (+ .meta.json)")

    # --- Run manifest --------------------------------------------------------
    manifest = {
        "phase": "D.1",
        "row": args.row,
        "include_prithvi": include_prithvi,
        "include_qc": include_qc,
        "train_years": list(D1_TRAIN_YEARS),
        "val_years": list(D1_VAL_YEARS),
        "holdout_years": list(D1_HOLDOUT_YEARS),
        "n_min_history": args.n_min_history,
        "prithvi_model_version": PRITHVI_MODEL_VERSION if include_prithvi else None,
        "embeddings_path": str(args.embeddings) if include_prithvi else None,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_geoids_kept": n_kept,
        "feature_cols_per_date": {d: len(FEATURE_COLS[d]) for d in VALID_FORECAST_DATES},
        "best_val_rmse_per_date": {
            d: float(bundle.regressors[d].train_metrics["val_rmse"])
            for d in VALID_FORECAST_DATES
        },
        "sweep_grid_size": len(grid),
        "wall_time_sec": round(total_elapsed, 1),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[write] run manifest → {manifest_path}")

    # --- Sweep CSV -----------------------------------------------------------
    sweep_path = args.sweep_csv
    if sweep_path is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        sweep_path = f"runs/d1_sweep_row{args.row}_{ts}.csv"
    Path(sweep_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_sweep_rows).to_csv(sweep_path, index=False)
    print(f"[write] sweep results → {sweep_path}")

    # --- Per-date best summary ----------------------------------------------
    print("\n=== Per-date best config (selected by val_rmse, county-level) ===")
    print(f"  {'date':>5}  {'depth':>5}  {'lr':>5}  {'mcw':>3}  "
          f"{'best_iter':>9}  {'train_rmse':>10}  {'val_rmse':>8}  "
          f"{'n_train':>7}  {'n_val':>5}")
    for date in VALID_FORECAST_DATES:
        reg = bundle.regressors[date]
        p = reg.params
        print(f"  {date:>5}  {p['max_depth']:>5}  "
              f"{p['learning_rate']:>5.2f}  {p['min_child_weight']:>3}  "
              f"{reg.best_iteration:>9}  "
              f"{reg.train_metrics['train_rmse']:>10.2f}  "
              f"{reg.train_metrics['val_rmse']:>8.2f}  "
              f"{reg.n_train:>7,}  {reg.n_val:>5,}")

    print("\nNote: val_rmse above is in-sample at the bundle level (val used for "
          "early stopping AND config selection). The D.1.e gate test "
          "(scripts/ablation_d1.py — TODO) will compare Row B vs Row A and "
          "report the headline gate metric.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
