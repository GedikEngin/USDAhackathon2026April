"""
scripts/ablation_d1.py — Phase D.1.e ablation gate test.

Compares 2-4 pre-trained RegressorBundles on a common val set. The headline
metric is the **Row B vs Row A** RMSE delta at EOS on val 2023, which is the
D.1.e gate criterion (decisions log: "Row B engineered+Prithvi ≥5% RMSE
improvement vs Row A engineered-only on 2023 EOS").

Architecture
------------
This script does NOT train models. It loads pre-trained bundles from disk
and evaluates them. The user trains the rows separately:

    Row A:   python scripts/train_regressor_d1.py --row A --out-dir models/forecast_d1_rowA
    Row B:   python scripts/train_regressor_d1.py --row B --out-dir models/forecast_d1_rowB
    Row C:   reuse models/forecast/  (Phase C's existing bundle, trained 2005-2022)
    Row D:   custom — train on 2013-2022 with leaky MODIS NDVI added back
             (this row is for diagnostic context only; not strictly required
              for the gate decision)

Each row is independent, so re-running just one of them (e.g. tweaking
hyperparameters on Row B and re-fitting) doesn't require re-training the others.

The 4-row ablation
------------------
  Row A   2013-2022 train  | engineered only            | D.1's lower bound
  Row B   2013-2022 train  | engineered + Prithvi + QC  | THE GATE TEST RUN
  Row C   2005-2022 train  | engineered only (Phase C)  | "did narrowing pool hurt?"
  Row D   2013-2022 train  | engineered + leaky NDVI    | "what we'd have gotten
                                                          if we kept MODIS NDVI"

Decisions log gate criterion (entry 2-D.1.kickoff):
  - PRIMARY: Row B EOS val_rmse ≥ 5% lower than Row A EOS val_rmse.
  - INFORM: how does Row C compare? If C is much better than B, the train-pool
    narrowing cost more than Prithvi added.
  - INFORM: how does Row D compare? Tells us whether removing leaky MODIS NDVI
    in Phase C was the right call.

Outputs
-------
  runs/d1_ablation_<timestamp>.json        full result, machine-readable
  runs/d1_ablation_<timestamp>_summary.csv per-(row, forecast_date) RMSE table
  runs/d1_ablation_<timestamp>_per_state.csv per-(row, state) breakdown
  stdout: human-readable report ending in PASS / FAIL banner

Usage
-----
  # Minimum: just A vs B (the gate)
  python scripts/ablation_d1.py \\
      --row-a models/forecast_d1_rowA \\
      --row-b models/forecast_d1_rowB

  # Full 4-row table
  python scripts/ablation_d1.py \\
      --row-a models/forecast_d1_rowA \\
      --row-b models/forecast_d1_rowB \\
      --row-c models/forecast \\
      --row-d models/forecast_d1_rowD

  # Different val year (default 2023)
  python scripts/ablation_d1.py --row-a ... --row-b ... --val-year 2024

The val year defaults to 2023 (per decisions log split). Setting --val-year 2024
runs the holdout test (used once at end of Phase G).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Make `forecast` importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forecast.data import DEFAULT_MASTER_PATH, load_master
from forecast.features import VALID_FORECAST_DATES
from forecast.regressor import (
    FEATURE_COLS,
    RegressorBundle,
    _add_derived_columns,
)
from forecast.prithvi import (
    EMBEDDING_COL_NAMES,
    MODEL_VERSION as PRITHVI_MODEL_VERSION,
)


# =============================================================================
# Constants
# =============================================================================

GATE_RMSE_LIFT_THRESHOLD = 0.05    # Row B EOS RMSE must be 5%+ lower than Row A
DEFAULT_EMBEDDINGS_PATH  = "data/v2/prithvi/embeddings_v1.parquet"
DEFAULT_MASTER_PATH_LOCAL = DEFAULT_MASTER_PATH


# =============================================================================
# Result dataclasses
# =============================================================================

@dataclass
class RowResult:
    """One ablation row's evaluation results across all forecast_dates."""
    row_label: str
    bundle_dir: str
    n_features_per_date: Dict[str, int]
    val_year: int
    n_val_total: int
    rmse_per_date: Dict[str, float]
    rmse_per_state_date: Dict[str, Dict[str, float]]   # state -> date -> RMSE
    bias_per_date: Dict[str, float]
    n_per_date: Dict[str, int]


@dataclass
class GateResult:
    """The headline D.1.e gate decision."""
    row_a_label: str
    row_b_label: str
    val_year: int
    forecast_date: str
    rmse_a: float
    rmse_b: float
    lift: float           # (A - B) / A; positive = B better than A
    threshold: float
    passed: bool
    notes: str


# =============================================================================
# Bundle loading + feature-space alignment
# =============================================================================

def load_bundle_with_features(bundle_dir: Path) -> Tuple[RegressorBundle, Dict[str, int]]:
    """Load a RegressorBundle and report its feature_cols length per date.

    The booster's feature_names list (saved with each Regressor) is the
    authoritative source of "what features does this row need?" — different
    rows have different feature sets (Row A has only engineered, Row B has
    engineered+1024+QC, Row D has engineered+5 NDVI).

    Returns (bundle, n_features_per_date_dict).
    """
    bundle = RegressorBundle.load(bundle_dir)
    n_per_date = {}
    for date, reg in bundle.regressors.items():
        n_per_date[date] = len(reg.feature_cols)
    return bundle, n_per_date


def prepare_val_df_for_bundle(
    master_df: pd.DataFrame,
    embeddings_df: Optional[pd.DataFrame],
    bundle: RegressorBundle,
    val_year: int,
) -> pd.DataFrame:
    """Prepare val rows with the columns this specific bundle needs.

    Different rows need different columns:
      - Row A: only the Phase C base features (master alone is sufficient)
      - Row B: master + 1024 prithvi cols + 4 QC cols (need embeddings join)
      - Row C: same as Row A (Phase C bundle uses Phase C feature list)
      - Row D: master + leaky MODIS NDVI cols (those are already in master)

    We detect what's needed by inspecting the bundle's feature_cols and
    joining embeddings only if any prithvi_* col is in the feature list.

    The bundle.predict() path will fail loudly if a required column is missing
    from the input df, so we just need to ensure all required columns are
    present (NaN-filled is fine; XGBoost handles NaN).
    """
    val_df = master_df[master_df["year"] == val_year].copy().reset_index(drop=True)

    # Inspect the bundle's required columns (use first regressor as representative —
    # all dates share the same prithvi/QC additions; only grain weather differs)
    first_reg = next(iter(bundle.regressors.values()))
    required = set(first_reg.feature_cols)

    # Column-presence repair via FEATURE_COLS module-state mutation.
    # The Phase C _build_dmatrix reads FEATURE_COLS at predict time, so when we
    # call bundle.predict(val_df), the function looks up the booster's bound
    # feature_cols (from meta.json sidecar via Regressor.feature_cols) and
    # asserts those columns exist on the input df. We need to ensure the df
    # has them.

    needs_prithvi = any(c.startswith("prithvi_") for c in required)
    needs_qc = any(c in required for c in (
        "chip_count", "chip_age_days_max", "cloud_pct_max", "corn_pixel_frac_min"))

    if needs_prithvi or needs_qc:
        if embeddings_df is None:
            raise RuntimeError(
                f"Bundle {bundle.regressors[VALID_FORECAST_DATES[0]].forecast_date} "
                f"requires Prithvi/QC features but --embeddings was not provided."
            )
        # Standardize key types
        emb = embeddings_df.copy()
        emb["GEOID"] = emb["GEOID"].astype(str).str.zfill(5)
        emb["year"] = emb["year"].astype(int)
        emb["forecast_date"] = emb["forecast_date"].astype(str)
        val_df["GEOID"] = val_df["GEOID"].astype(str).str.zfill(5)
        val_df["year"] = val_df["year"].astype(int)
        val_df["forecast_date"] = val_df["forecast_date"].astype(str)

        # Pick the columns we'll join
        join_cols = ["GEOID", "year", "forecast_date"]
        feature_extras = []
        if needs_qc:
            feature_extras += [c for c in (
                "chip_count", "chip_age_days_max", "cloud_pct_max", "corn_pixel_frac_min")
                if c in emb.columns]
        if needs_prithvi:
            feature_extras += [c for c in EMBEDDING_COL_NAMES if c in emb.columns]
        emb_for_join = emb[join_cols + feature_extras]

        n_before = len(val_df)
        val_df = val_df.merge(emb_for_join, on=join_cols, how="left")
        if len(val_df) != n_before:
            raise RuntimeError(
                f"Embedding join changed val row count {n_before} -> {len(val_df)}; "
                f"embeddings parquet has duplicate keys."
            )

    # Materialize the derived columns Phase C's _build_dmatrix expects:
    #   - state_is_<S> one-hots from state_alpha
    #   - is_irrigated_reported from yield_bu_acre_irr
    # We call this here (eagerly) rather than letting _build_dmatrix do it
    # at predict-time, because the column-presence check below needs to
    # see them present.
    val_df = _add_derived_columns(val_df)

    # Sanity-check that all required columns exist now
    missing = [c for c in required if c not in val_df.columns]
    if missing:
        raise KeyError(
            f"Bundle requires {len(missing)} columns not present in val_df after join: "
            f"first 5 = {missing[:5]}"
        )

    # The Phase C _build_dmatrix derives state_is_<S> + is_irrigated_reported
    # at DMatrix-construction time — but the column-presence check above
    # runs BEFORE bundle.predict() is called, so we materialize them eagerly
    # via the _add_derived_columns call above. FEATURE_COLS module-state
    # must list this row's columns, though, because _build_dmatrix indexes
    # into FEATURE_COLS[forecast_date]. Ensure the current FEATURE_COLS
    # matches the bundle's expectation.
    _ensure_feature_cols_alignment(bundle)

    return val_df


def _ensure_feature_cols_alignment(bundle: RegressorBundle) -> None:
    """Make `forecast.regressor.FEATURE_COLS` match this bundle's per-date
    feature_cols. The Phase C _build_dmatrix reads FEATURE_COLS at runtime;
    if the global is misaligned with the loaded bundle, prediction fails or
    silently uses wrong columns.

    This mutates the global. Within ablation_d1.py we serially align before
    each bundle's predict() call, so the global ends up matching the
    last-evaluated bundle. Don't run other bundle predicts after this script
    in the same Python process.
    """
    for date in VALID_FORECAST_DATES:
        if date not in bundle.regressors:
            raise KeyError(
                f"Bundle missing booster for forecast_date={date!r}; "
                f"expected all 4 dates."
            )
        FEATURE_COLS[date] = list(bundle.regressors[date].feature_cols)


# =============================================================================
# Per-row evaluation
# =============================================================================

def evaluate_row(
    row_label: str,
    bundle_dir: Path,
    master_df: pd.DataFrame,
    embeddings_df: Optional[pd.DataFrame],
    val_year: int,
    *,
    log: bool = True,
) -> RowResult:
    """Load a bundle, prepare val_df, predict, compute per-date and
    per-state-date RMSE."""
    if log:
        print(f"\n--- evaluating {row_label}: {bundle_dir} ---")

    bundle, n_features_per_date = load_bundle_with_features(bundle_dir)
    val_df = prepare_val_df_for_bundle(master_df, embeddings_df, bundle, val_year)

    # Drop rows with missing yield_target — can't score
    val_df = val_df[val_df["yield_target"].notna()].reset_index(drop=True)
    if len(val_df) == 0:
        raise ValueError(f"No val rows with yield_target for {row_label} "
                         f"at val_year={val_year}")

    # Predict by date and accumulate
    rmse_per_date: Dict[str, float] = {}
    bias_per_date: Dict[str, float] = {}
    n_per_date: Dict[str, int] = {}
    rmse_per_state_date: Dict[str, Dict[str, float]] = {}

    for date in VALID_FORECAST_DATES:
        sub = val_df[val_df["forecast_date"] == date].reset_index(drop=True)
        if len(sub) == 0:
            rmse_per_date[date] = float("nan")
            bias_per_date[date] = float("nan")
            n_per_date[date] = 0
            continue

        # Phase C's _build_dmatrix drops rows where any feature column is NaN.
        # For Row B (with 1024 prithvi + 4 QC cols added), val rows with
        # chip_count==0 have NaN prithvi+QC and get dropped at predict time.
        # We need to apply the same drop mask to `truth` so preds and truth
        # stay aligned. Compute it explicitly, mirroring _build_dmatrix.
        feat_cols = bundle.regressors[date].feature_cols
        feat_block = sub[feat_cols].to_numpy(dtype=np.float64)
        row_complete = ~np.isnan(feat_block).any(axis=1)
        n_kept = int(row_complete.sum())
        n_dropped = int((~row_complete).sum())

        if n_kept == 0:
            rmse_per_date[date] = float("nan")
            bias_per_date[date] = float("nan")
            n_per_date[date] = 0
            if log:
                print(f"  {date}: n=0 kept (all {len(sub)} rows had NaN features); "
                      f"skipping")
            continue

        sub_kept = sub.loc[row_complete].reset_index(drop=True)

        # Predict on the kept rows. (bundle.regressors[date].predict goes
        # through _build_dmatrix again, which will produce the same n_kept
        # rows on a sub_kept input — every row in sub_kept is NaN-free
        # by construction.)
        preds = bundle.regressors[date].predict(sub_kept)
        truth = sub_kept["yield_target"].to_numpy(dtype=np.float64)

        if len(preds) != len(truth):
            # Defensive — shouldn't happen with the row_complete pre-filter,
            # but if it does, fail loudly with a helpful message
            raise RuntimeError(
                f"prediction shape {preds.shape} != truth shape {truth.shape} "
                f"for row_label={row_label!r} forecast_date={date!r}; "
                f"_build_dmatrix may be applying additional row drops."
            )

        errors = preds - truth   # signed; positive = model over-predicts
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        bias = float(np.mean(errors))
        rmse_per_date[date] = rmse
        bias_per_date[date] = bias
        n_per_date[date] = n_kept

        # Per-state breakdown — operate on sub_kept
        for state, state_sub in sub_kept.groupby("state_alpha"):
            state_idx = state_sub.index.to_numpy()
            state_truth = state_sub["yield_target"].to_numpy(dtype=np.float64)
            state_preds = preds[state_idx]
            state_errors = state_preds - state_truth
            if len(state_errors) == 0:
                continue
            state_rmse = float(np.sqrt(np.mean(state_errors ** 2)))
            rmse_per_state_date.setdefault(state, {})[date] = state_rmse

        if log:
            drop_note = f" (dropped {n_dropped} NaN-feat rows)" if n_dropped > 0 else ""
            print(f"  {date}: n={n_kept:>4}  rmse={rmse:>6.2f}  "
                  f"bias={bias:>+6.2f}{drop_note}")

    return RowResult(
        row_label=row_label,
        bundle_dir=str(bundle_dir),
        n_features_per_date=n_features_per_date,
        val_year=val_year,
        n_val_total=int(sum(n_per_date.values())),
        rmse_per_date=rmse_per_date,
        rmse_per_state_date=rmse_per_state_date,
        bias_per_date=bias_per_date,
        n_per_date=n_per_date,
    )


# =============================================================================
# Gate decision
# =============================================================================

def evaluate_gate(
    row_a: RowResult,
    row_b: RowResult,
    *,
    forecast_date: str = "EOS",
    threshold: float = GATE_RMSE_LIFT_THRESHOLD,
) -> GateResult:
    """Compute the D.1.e gate metric: (rmse_A - rmse_B) / rmse_A on val_year EOS.

    Pass iff lift >= threshold.
    """
    rmse_a = row_a.rmse_per_date.get(forecast_date, float("nan"))
    rmse_b = row_b.rmse_per_date.get(forecast_date, float("nan"))

    if not np.isfinite(rmse_a) or not np.isfinite(rmse_b):
        return GateResult(
            row_a_label=row_a.row_label, row_b_label=row_b.row_label,
            val_year=row_a.val_year, forecast_date=forecast_date,
            rmse_a=rmse_a, rmse_b=rmse_b, lift=float("nan"),
            threshold=threshold, passed=False,
            notes=f"non-finite RMSE; cannot evaluate gate",
        )
    if rmse_a <= 0:
        return GateResult(
            row_a_label=row_a.row_label, row_b_label=row_b.row_label,
            val_year=row_a.val_year, forecast_date=forecast_date,
            rmse_a=rmse_a, rmse_b=rmse_b, lift=float("nan"),
            threshold=threshold, passed=False,
            notes=f"row A rmse <= 0; cannot compute lift",
        )

    lift = (rmse_a - rmse_b) / rmse_a
    passed = lift >= threshold
    return GateResult(
        row_a_label=row_a.row_label, row_b_label=row_b.row_label,
        val_year=row_a.val_year, forecast_date=forecast_date,
        rmse_a=rmse_a, rmse_b=rmse_b, lift=lift,
        threshold=threshold, passed=passed,
        notes="",
    )


# =============================================================================
# Reporting
# =============================================================================

def print_summary_table(rows: List[RowResult]) -> None:
    """Per-(row, forecast_date) RMSE table to stdout."""
    print()
    print("=" * 75)
    print(f"  Per-row RMSE on val_year={rows[0].val_year}")
    print("=" * 75)
    header = f"  {'row':>6}  " + "  ".join(f"{d:>8}" for d in VALID_FORECAST_DATES) + "   features/date"
    print(header)
    print(f"  {'-'*6}  " + "  ".join("-"*8 for _ in VALID_FORECAST_DATES) + "   " + "-"*16)
    for r in rows:
        rmse_str = "  ".join(
            f"{r.rmse_per_date.get(d, float('nan')):>8.2f}"
            for d in VALID_FORECAST_DATES
        )
        feat = ", ".join(f"{d}={r.n_features_per_date.get(d,0)}" for d in VALID_FORECAST_DATES)
        print(f"  {r.row_label:>6}  {rmse_str}   {feat}")


def print_per_state_table(rows: List[RowResult]) -> None:
    """Per-(row, state) average RMSE across forecast_dates."""
    print()
    print("=" * 75)
    print(f"  Per-state RMSE (averaged across forecast_dates)")
    print("=" * 75)
    states = sorted({s for r in rows for s in r.rmse_per_state_date})
    header = f"  {'row':>6}  " + "  ".join(f"{s:>8}" for s in states)
    print(header)
    print(f"  {'-'*6}  " + "  ".join("-"*8 for _ in states))
    for r in rows:
        cells = []
        for s in states:
            per_date = r.rmse_per_state_date.get(s, {})
            if per_date:
                avg = float(np.mean(list(per_date.values())))
                cells.append(f"{avg:>8.2f}")
            else:
                cells.append(f"{'-':>8}")
        print(f"  {r.row_label:>6}  " + "  ".join(cells))


def print_gate_banner(gate: GateResult) -> None:
    """Big visible PASS/FAIL banner."""
    print()
    print("=" * 75)
    print(f"  GATE TEST: {gate.row_b_label} vs {gate.row_a_label} "
          f"on val_year={gate.val_year} at {gate.forecast_date}")
    print("=" * 75)
    print(f"  RMSE row A ({gate.row_a_label}): {gate.rmse_a:.2f}")
    print(f"  RMSE row B ({gate.row_b_label}): {gate.rmse_b:.2f}")
    print(f"  Lift: ({gate.rmse_a:.2f} - {gate.rmse_b:.2f}) / {gate.rmse_a:.2f} = "
          f"{100*gate.lift:+.2f}%   (threshold: +{100*gate.threshold:.0f}%)")
    if gate.notes:
        print(f"  Notes: {gate.notes}")
    print()
    if gate.passed:
        print("  ┌─────────────────┐")
        print("  │  GATE: PASSED ✓  │")
        print("  └─────────────────┘")
    else:
        print("  ┌─────────────────┐")
        print("  │  GATE: FAILED ✗  │")
        print("  └─────────────────┘")
    print()


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--master", default=DEFAULT_MASTER_PATH_LOCAL,
                    help=f"path to training_master.parquet (default: {DEFAULT_MASTER_PATH_LOCAL})")
    ap.add_argument("--embeddings", default=DEFAULT_EMBEDDINGS_PATH,
                    help=f"path to embeddings_v1.parquet (default: {DEFAULT_EMBEDDINGS_PATH})")
    ap.add_argument("--val-year", type=int, default=2023,
                    help="val year for the gate (default: 2023; pass 2024 for the holdout test)")
    ap.add_argument("--gate-date", default="EOS", choices=list(VALID_FORECAST_DATES),
                    help="forecast_date for the gate metric (default: EOS)")
    ap.add_argument("--threshold", type=float, default=GATE_RMSE_LIFT_THRESHOLD,
                    help=f"gate lift threshold (default: {GATE_RMSE_LIFT_THRESHOLD} = 5%%)")

    ap.add_argument("--row-a", required=True, help="bundle dir for Row A (engineered-only, 2013-2022)")
    ap.add_argument("--row-b", required=True, help="bundle dir for Row B (engineered+Prithvi+QC, 2013-2022)")
    ap.add_argument("--row-c", default=None, help="(optional) bundle dir for Row C (Phase C as-is, 2005-2022)")
    ap.add_argument("--row-d", default=None, help="(optional) bundle dir for Row D (engineered+leaky NDVI, 2013-2022)")

    ap.add_argument("--out-prefix", default=None,
                    help="output file prefix (default: runs/d1_ablation_<timestamp>)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    log = not args.quiet

    # --- Load master + embeddings ------------------------------------------
    if log:
        print(f"[load] master: {args.master}")
    master_df = load_master(args.master)
    if log:
        print(f"  {len(master_df):,} rows × {len(master_df.columns)} cols")

    embeddings_df: Optional[pd.DataFrame] = None
    if Path(args.embeddings).exists():
        if log:
            print(f"[load] embeddings: {args.embeddings}")
        embeddings_df = pd.read_parquet(args.embeddings)
        # Filter to current model_version
        if "model_version" in embeddings_df.columns:
            before = len(embeddings_df)
            embeddings_df = embeddings_df[
                embeddings_df["model_version"] == PRITHVI_MODEL_VERSION].copy()
            if log:
                print(f"  filtered to model_version={PRITHVI_MODEL_VERSION}: "
                      f"{before} -> {len(embeddings_df)} rows")
    elif log:
        print(f"[load] embeddings not found at {args.embeddings}; "
              f"only Rows A/C/D can be evaluated (no Prithvi).")

    # --- Evaluate each row --------------------------------------------------
    rows: List[RowResult] = []

    rows.append(evaluate_row(
        "Row A", Path(args.row_a),
        master_df, embeddings_df, args.val_year, log=log,
    ))
    rows.append(evaluate_row(
        "Row B", Path(args.row_b),
        master_df, embeddings_df, args.val_year, log=log,
    ))
    if args.row_c:
        rows.append(evaluate_row(
            "Row C", Path(args.row_c),
            master_df, embeddings_df, args.val_year, log=log,
        ))
    if args.row_d:
        rows.append(evaluate_row(
            "Row D", Path(args.row_d),
            master_df, embeddings_df, args.val_year, log=log,
        ))

    # --- Reports ------------------------------------------------------------
    print_summary_table(rows)
    print_per_state_table(rows)

    # --- Gate ---------------------------------------------------------------
    gate = evaluate_gate(rows[0], rows[1],
                         forecast_date=args.gate_date,
                         threshold=args.threshold)
    print_gate_banner(gate)

    # --- Persist ------------------------------------------------------------
    out_prefix = args.out_prefix
    if out_prefix is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_prefix = f"runs/d1_ablation_{ts}"
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)

    full = {
        "val_year": args.val_year,
        "gate_date": args.gate_date,
        "gate_threshold": args.threshold,
        "rows": [asdict(r) for r in rows],
        "gate": asdict(gate),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    json_path = Path(f"{out_prefix}.json")
    json_path.write_text(json.dumps(full, indent=2, default=str))
    if log:
        print(f"[write] full result -> {json_path}")

    summary_rows = []
    for r in rows:
        for date in VALID_FORECAST_DATES:
            summary_rows.append({
                "row": r.row_label,
                "bundle_dir": r.bundle_dir,
                "val_year": r.val_year,
                "forecast_date": date,
                "n_features": r.n_features_per_date.get(date, 0),
                "n_val": r.n_per_date.get(date, 0),
                "rmse": r.rmse_per_date.get(date, float("nan")),
                "bias": r.bias_per_date.get(date, float("nan")),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = Path(f"{out_prefix}_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    if log:
        print(f"[write] summary CSV -> {summary_csv}")

    state_rows = []
    for r in rows:
        for state, per_date in r.rmse_per_state_date.items():
            for date, rmse in per_date.items():
                state_rows.append({
                    "row": r.row_label,
                    "state_alpha": state,
                    "forecast_date": date,
                    "rmse": rmse,
                })
    state_df = pd.DataFrame(state_rows)
    state_csv = Path(f"{out_prefix}_per_state.csv")
    state_df.to_csv(state_csv, index=False)
    if log:
        print(f"[write] per-state CSV -> {state_csv}")

    return 0 if gate.passed else 1


if __name__ == "__main__":
    sys.exit(main())
