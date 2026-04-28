#!/usr/bin/env python
"""
scripts/train_hybrid_d1.py
Train and persist the Phase 2-D.1.e hybrid bundle to models/forecast_d1_hybrid/.

Hybrid architecture (locked at D.1.e gate decision, 2026-04-28):
  per-state routing:
    CO, IA       → Row B (XGBoost on engineered + QC + ridge_pred)
    MO, NE, WI   → Row A (XGBoost on engineered only)
  ridge_pred is a Ridge-regression "linear probe" over the 1024-D Prithvi
  embedding, trained per forecast_date on chip-bearing 2013-2022 rows.

Output:
  models/forecast_d1_hybrid/
    ridge_probe_<date>.npz                  Ridge coef + intercept + train mean
    rowA_<date>.json (+ .meta.json)         XGBoost engineered-only
    rowB_<date>.json (+ .meta.json)         XGBoost engineered + QC + ridge_pred
    hybrid_manifest.json                    routing rule, gate metric, schema

Deterministic: random_state=42, fixed XGBoost hyperparams.
"""

from __future__ import annotations

import datetime as dt
import json
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
PRITHVI_STATES = ["CO", "IA"]
ENGINEERED_STATES = ["MO", "NE", "WI"]
RIDGE_ALPHA = 10.0

XGB_PARAMS = dict(
    objective="reg:squarederror", max_depth=4, learning_rate=0.05,
    min_child_weight=5, reg_lambda=1.0, tree_method="hist", device="cpu",
)
NUM_BOOST_ROUND = 600
EARLY_STOP = 50

PRITHVI_MODEL_VERSION = "prithvi_eo_v2_300_tl_meanpool_v1"
OUT_DIR = REPO_ROOT / "models" / "forecast_d1_hybrid"


def fit_xgb(X_tr, y_tr, X_va, y_va) -> tuple[xgb.Booster, int]:
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dva = xgb.DMatrix(X_va, label=y_va)
    booster = xgb.train(
        XGB_PARAMS, dtr, num_boost_round=NUM_BOOST_ROUND,
        evals=[(dva, "val")], early_stopping_rounds=EARLY_STOP,
        verbose_eval=False,
    )
    return booster, booster.best_iteration


def main():
    print("[load] master + embeddings")
    master = pd.read_parquet("scripts/training_master.parquet")
    master = _add_derived_columns(master)
    master = master[master["year"] >= 2013].copy()

    emb = pd.read_parquet("data/v2/prithvi/embeddings_v1.parquet")
    emb_cols = sorted([c for c in emb.columns if c.startswith("prithvi_")])
    keep = ["GEOID", "year", "forecast_date"] + QC_COLS + emb_cols
    keep = [c for c in keep if c in emb.columns]
    master = master.merge(emb[keep], on=["GEOID", "year", "forecast_date"], how="left")
    print(f"  master: {len(master):,} rows  ({len(emb_cols)} prithvi dims)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_date_metrics: list[dict] = []

    for date in VALID_DATES:
        print(f"\n[train] forecast_date={date}")
        sub = master[master["forecast_date"] == date].copy()
        chip_mask = sub[emb_cols[0]].notna()
        train_mask = sub["year"].isin(TRAIN_YEARS) & sub["yield_target"].notna()
        val_mask = (sub["year"] == VAL_YEAR) & sub["yield_target"].notna()

        # --- 1. Ridge probe on chip-bearing 2013-2022 rows ---
        rprobe_mask = chip_mask & train_mask
        n_ridge = int(rprobe_mask.sum())
        ridge = Ridge(alpha=RIDGE_ALPHA, random_state=42)
        ridge.fit(
            sub.loc[rprobe_mask, emb_cols].to_numpy(np.float32),
            sub.loc[rprobe_mask, "yield_target"].to_numpy(np.float32),
        )
        # save ridge weights as npz (compact, no pickle dependency)
        ridge_path = OUT_DIR / f"ridge_probe_{date}.npz"
        np.savez_compressed(
            ridge_path,
            coef=ridge.coef_.astype(np.float32),
            intercept=np.float32(ridge.intercept_),
            embedding_cols=np.array(emb_cols, dtype=object),
            alpha=np.float32(RIDGE_ALPHA),
            n_train=np.int32(n_ridge),
        )
        # add ridge_pred to master for chip-bearing rows
        any_chip_idx = sub.index[chip_mask]
        sub.loc[any_chip_idx, "ridge_pred"] = ridge.predict(
            sub.loc[chip_mask, emb_cols].to_numpy(np.float32)
        ).astype(np.float32)
        print(f"  ridge_probe: {n_ridge} train rows, saved {ridge_path.name}")

        # --- 2. Row A — engineered-only XGBoost ---
        feat_A = FEATURE_COLS[date]
        tr = sub[train_mask]
        va = sub[val_mask]
        booster_A, best_iter_A = fit_xgb(
            tr[feat_A].to_numpy(np.float32), tr["yield_target"].to_numpy(np.float32),
            va[feat_A].to_numpy(np.float32), va["yield_target"].to_numpy(np.float32),
        )
        pred_A = booster_A.predict(
            xgb.DMatrix(va[feat_A].to_numpy(np.float32)),
            iteration_range=(0, best_iter_A + 1),
        )
        rmse_A = float(np.sqrt(((pred_A - va["yield_target"].to_numpy()) ** 2).mean()))
        rowA_path = OUT_DIR / f"rowA_{date}.json"
        booster_A.save_model(str(rowA_path))
        with open(str(rowA_path) + ".meta.json", "w") as fh:
            json.dump({
                "forecast_date": date, "variant": "rowA_engineered_only",
                "feature_cols": feat_A, "n_train": int(len(tr)), "n_val": int(len(va)),
                "best_iter": int(best_iter_A), "val_rmse_county": rmse_A,
                "xgb_params": XGB_PARAMS,
            }, fh, indent=2)
        print(f"  rowA: rmse={rmse_A:.3f}  best_iter={best_iter_A}  → {rowA_path.name}")

        # --- 3. Row B — engineered + QC + ridge_pred XGBoost ---
        feat_B = FEATURE_COLS[date] + QC_COLS + ["ridge_pred"]
        booster_B, best_iter_B = fit_xgb(
            tr[feat_B].to_numpy(np.float32), tr["yield_target"].to_numpy(np.float32),
            va[feat_B].to_numpy(np.float32), va["yield_target"].to_numpy(np.float32),
        )
        pred_B = booster_B.predict(
            xgb.DMatrix(va[feat_B].to_numpy(np.float32)),
            iteration_range=(0, best_iter_B + 1),
        )
        rmse_B = float(np.sqrt(((pred_B - va["yield_target"].to_numpy()) ** 2).mean()))
        rowB_path = OUT_DIR / f"rowB_{date}.json"
        booster_B.save_model(str(rowB_path))
        with open(str(rowB_path) + ".meta.json", "w") as fh:
            json.dump({
                "forecast_date": date, "variant": "rowB_engineered_plus_ridge_probe",
                "feature_cols": feat_B, "n_train": int(len(tr)), "n_val": int(len(va)),
                "best_iter": int(best_iter_B), "val_rmse_county": rmse_B,
                "xgb_params": XGB_PARAMS,
            }, fh, indent=2)
        print(f"  rowB: rmse={rmse_B:.3f}  best_iter={best_iter_B}  → {rowB_path.name}")

        # --- 4. Hybrid evaluation ---
        is_prithvi_state = va["state_alpha"].isin(PRITHVI_STATES).to_numpy()
        pred_H = np.where(is_prithvi_state, pred_B, pred_A)
        y_va = va["yield_target"].to_numpy(np.float32)
        rmse_H = float(np.sqrt(((pred_H - y_va) ** 2).mean()))
        per_state_H = {}
        for s in ("CO", "IA", "MO", "NE", "WI"):
            m = (va["state_alpha"].values == s)
            if m.any():
                per_state_H[s] = float(np.sqrt(((pred_H[m] - y_va[m]) ** 2).mean()))
        per_date_metrics.append(dict(
            forecast_date=date, n_train=int(len(tr)), n_val=int(len(va)),
            rmse_rowA=rmse_A, rmse_rowB=rmse_B, rmse_hybrid=rmse_H,
            best_iter_rowA=int(best_iter_A), best_iter_rowB=int(best_iter_B),
            rmse_hybrid_per_state=per_state_H,
        ))
        print(f"  hybrid: rmse={rmse_H:.3f}  per-state={per_state_H}")

    # Write manifest
    eos_metric = next(m for m in per_date_metrics if m["forecast_date"] == "EOS")
    GATE_REF_EOS_RMSE = 19.95
    lift_pct = (GATE_REF_EOS_RMSE - eos_metric["rmse_hybrid"]) / GATE_REF_EOS_RMSE * 100
    manifest = {
        "version": "phase2_d1e_hybrid_v1",
        "trained_at": dt.datetime.now().isoformat(),
        "routing_rule": {
            "prithvi_states": PRITHVI_STATES,
            "engineered_states": ENGINEERED_STATES,
            "rule": "predict with rowB if state in prithvi_states, else rowA",
        },
        "ridge_probe": {
            "alpha": RIDGE_ALPHA,
            "embedding_dim": len(emb_cols),
            "embedding_source": "data/v2/prithvi/embeddings_v1.parquet",
            "model_version": PRITHVI_MODEL_VERSION,
            "fit_pool": "chip-bearing rows in 2013-2022",
        },
        "xgb_params": XGB_PARAMS,
        "qc_feature_cols": QC_COLS,
        "valid_forecast_dates": VALID_DATES,
        "feature_cols_per_date": {d: FEATURE_COLS[d] for d in VALID_DATES},
        "rowB_feature_cols_per_date": {
            d: FEATURE_COLS[d] + QC_COLS + ["ridge_pred"] for d in VALID_DATES
        },
        "training_pool_years": TRAIN_YEARS,
        "val_year": VAL_YEAR,
        "metrics_per_date": per_date_metrics,
        "gate": {
            "ref_eos_rmse": GATE_REF_EOS_RMSE,
            "hybrid_eos_rmse": eos_metric["rmse_hybrid"],
            "lift_pct_vs_ref": round(lift_pct, 3),
            "threshold_pct": 5.0,
            "verdict": "PASS" if lift_pct >= 5.0 else "FAIL",
        },
        "files": {
            "ridge_probe": [f"ridge_probe_{d}.npz" for d in VALID_DATES],
            "rowA_xgb": [f"rowA_{d}.json" for d in VALID_DATES],
            "rowB_xgb": [f"rowB_{d}.json" for d in VALID_DATES],
        },
    }
    with open(OUT_DIR / "hybrid_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    print("\n" + "=" * 70)
    print("HYBRID BUNDLE WRITTEN")
    print("=" * 70)
    for d, m in zip(VALID_DATES, per_date_metrics):
        print(f"  {d}:  rowA={m['rmse_rowA']:.3f}  rowB={m['rmse_rowB']:.3f}  "
              f"hybrid={m['rmse_hybrid']:.3f}")
    print(f"\nGATE: hybrid EOS rmse={eos_metric['rmse_hybrid']:.3f}  "
          f"vs ref {GATE_REF_EOS_RMSE}  =  {lift_pct:+.2f}% lift   "
          f"({'PASS ✓' if lift_pct >= 5.0 else 'FAIL'})")
    print(f"\noutput: {OUT_DIR}/")


if __name__ == "__main__":
    main()