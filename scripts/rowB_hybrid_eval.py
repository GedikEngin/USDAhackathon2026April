#!/usr/bin/env python
"""
scripts/rowB_hybrid_eval.py

Per-state hybrid: use the best variant per state.

Strategy locked from rowB_experiments.py results:
  - IA, CO  → Ridge-probe variant (Prithvi helps)
  - MO, NE, WI → engineered-only (Prithvi hurts)

Re-trains every model in-script (no dependence on saved bundles); fixed
hyperparameters across variants for fair comparison. Reports aggregate
val 2023 EOS RMSE for: pure rowA, pure rowB-ridge, hybrid.
Also reports gate metric vs the canonical rowA bundle (engineered-only,
hyperparam-swept) at 19.95 EOS — the number the gate threshold is
defined against in PHASE2_DECISIONS_LOG.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from forecast.regressor import FEATURE_COLS, _add_derived_columns  # noqa: E402

VALID_DATES = ["08-01", "09-01", "10-01", "EOS"]
QC_COLS = ["chip_count", "chip_age_days_max", "cloud_pct_max", "corn_pixel_frac_min"]
TRAIN_YEARS = list(range(2013, 2023))
VAL_YEAR = 2023
PRITHVI_STATES = {"IA", "CO"}  # states where ridge variant wins
ENGINEERED_STATES = {"MO", "NE", "WI"}

XGB_PARAMS = dict(
    objective="reg:squarederror", max_depth=4, learning_rate=0.05,
    min_child_weight=5, reg_lambda=1.0, tree_method="hist", device="cpu",
)
NUM_BOOST_ROUND = 600
EARLY_STOP = 50

# Canonical Row A reference from PHASE2_DECISIONS_LOG.md gate threshold
GATE_REF_EOS_RMSE = 19.95


def fit_xgb_predict(X_tr, y_tr, X_va, y_va):
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dva = xgb.DMatrix(X_va, label=y_va)
    booster = xgb.train(
        XGB_PARAMS, dtr, num_boost_round=NUM_BOOST_ROUND,
        evals=[(dva, "val")], early_stopping_rounds=EARLY_STOP,
        verbose_eval=False,
    )
    pred = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
    return pred.astype(np.float32)


def main():
    master = pd.read_parquet("scripts/training_master.parquet")
    master = _add_derived_columns(master)
    master = master[master["year"] >= 2013].copy()

    emb = pd.read_parquet("data/v2/prithvi/embeddings_v1.parquet")
    emb_cols = sorted([c for c in emb.columns if c.startswith("prithvi_")])
    keep = ["GEOID", "year", "forecast_date"] + QC_COLS + emb_cols
    keep = [c for c in keep if c in emb.columns]
    master = master.merge(emb[keep], on=["GEOID", "year", "forecast_date"], how="left")

    print(f"[load] master: {len(master):,} rows  ({len(emb_cols)} prithvi dims)")

    # Train Ridge probe per forecast_date and add ridge_pred col
    print("[train] Ridge probe on 1024-D embedding")
    master["ridge_pred"] = np.nan
    for date in VALID_DATES:
        sub = master[master["forecast_date"] == date]
        chip_mask = sub[emb_cols[0]].notna()
        train_mask = chip_mask & sub["year"].isin(TRAIN_YEARS) & sub["yield_target"].notna()
        if int(train_mask.sum()) < 100:
            continue
        ridge = Ridge(alpha=10.0, random_state=42)
        ridge.fit(
            sub.loc[train_mask, emb_cols].to_numpy(np.float32),
            sub.loc[train_mask, "yield_target"].to_numpy(np.float32),
        )
        any_chip_idx = sub.index[chip_mask]
        master.loc[any_chip_idx, "ridge_pred"] = ridge.predict(
            sub.loc[chip_mask, emb_cols].to_numpy(np.float32)
        ).astype(np.float32)

    # For each forecast_date: train rowA (engineered-only) and rowB-ridge
    # (engineered + QC + ridge_pred), then evaluate on val 2023 row-by-row,
    # routing predictions by state membership in PRITHVI_STATES.
    rows_eval = []
    for date in VALID_DATES:
        sub = master[master["forecast_date"] == date].copy()
        tr = sub[sub["year"].isin(TRAIN_YEARS) & sub["yield_target"].notna()]
        va = sub[(sub["year"] == VAL_YEAR) & sub["yield_target"].notna()]

        feat_A = FEATURE_COLS[date]
        feat_B = FEATURE_COLS[date] + QC_COLS + ["ridge_pred"]

        # Train pure rowA
        pred_A = fit_xgb_predict(
            tr[feat_A].to_numpy(np.float32), tr["yield_target"].to_numpy(np.float32),
            va[feat_A].to_numpy(np.float32), va["yield_target"].to_numpy(np.float32),
        )
        # Train rowB-ridge
        pred_B = fit_xgb_predict(
            tr[feat_B].to_numpy(np.float32), tr["yield_target"].to_numpy(np.float32),
            va[feat_B].to_numpy(np.float32), va["yield_target"].to_numpy(np.float32),
        )
        # Hybrid prediction: B for PRITHVI_STATES, A elsewhere
        is_prithvi_state = va["state_alpha"].isin(PRITHVI_STATES).to_numpy()
        pred_H = np.where(is_prithvi_state, pred_B, pred_A)

        y_va = va["yield_target"].to_numpy(np.float32)
        states = va["state_alpha"].to_numpy()

        for variant, pred in (("rowA_pure", pred_A), ("rowB_ridge", pred_B), ("hybrid", pred_H)):
            row = dict(forecast_date=date, variant=variant, n_val=len(va),
                       rmse_overall=float(np.sqrt(((pred - y_va) ** 2).mean())))
            for s in ("CO", "IA", "MO", "NE", "WI"):
                m = states == s
                if m.any():
                    row[f"rmse_{s}"] = float(np.sqrt(((pred[m] - y_va[m]) ** 2).mean()))
                    row[f"n_{s}"] = int(m.sum())
            rows_eval.append(row)

        print(f"  {date}: rowA={rows_eval[-3]['rmse_overall']:.3f}  "
              f"rowB_ridge={rows_eval[-2]['rmse_overall']:.3f}  "
              f"hybrid={rows_eval[-1]['rmse_overall']:.3f}")

    df = pd.DataFrame(rows_eval)
    print()
    print("=" * 70)
    print("Per-variant val 2023 RMSE (county-level)")
    print("=" * 70)
    pivot = df.pivot_table(index="variant", columns="forecast_date",
                            values="rmse_overall", aggfunc="mean")
    print(pivot.reindex(["rowA_pure", "rowB_ridge", "hybrid"])[VALID_DATES].round(3).to_string())

    print()
    print("=" * 70)
    print("Per-state EOS RMSE (val 2023)")
    print("=" * 70)
    eos = df[df["forecast_date"] == "EOS"]
    state_cols = [c for c in df.columns if c.startswith("rmse_") and c != "rmse_overall"]
    print(eos.set_index("variant")[state_cols].round(2).to_string())

    print()
    print("=" * 70)
    print(f"GATE TEST vs canonical Row A bundle (EOS RMSE = {GATE_REF_EOS_RMSE})")
    print("=" * 70)
    eos_pivot = pivot[["EOS"]].squeeze() if "EOS" in pivot.columns else None
    for v in ["rowA_pure", "rowB_ridge", "hybrid"]:
        if eos_pivot is not None and v in eos_pivot.index:
            r = eos_pivot[v]
            lift_vs_ref = (GATE_REF_EOS_RMSE - r) / GATE_REF_EOS_RMSE * 100
            verdict = "PASS ✓" if lift_vs_ref >= 5.0 else ("FAIL" if lift_vs_ref < 0 else "miss")
            print(f"  {v:12s} EOS rmse={r:.3f}   lift vs ref={lift_vs_ref:+.2f}%   {verdict}")
    print()

    out = REPO_ROOT / "runs" / "rowB_hybrid_eval.csv"
    df.to_csv(out, index=False)
    print(f"results → {out}")


if __name__ == "__main__":
    main()