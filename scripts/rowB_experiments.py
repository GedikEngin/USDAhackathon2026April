#!/usr/bin/env python
"""
scripts/rowB_experiments.py

Run multiple Row B variants side-by-side and report which one wins the gate.

Variants:
  raw1024   1024-D Prithvi → XGBoost (= the existing Row B; baseline reference)
  pca32     PCA-32 (95.4% var) → XGBoost
  pca64     PCA-64 (97.4% var) → XGBoost
  ridge     Ridge probe on 1024-D → 1 scalar feature → XGBoost
            (this variant uses ALL 1024 dims through linear weights, the
             foundation-model best-practice approach)

Each variant uses the same XGBoost hyperparameters (depth=4, lr=0.05,
mcw=5, num_boost_round=600 with early stop on val 2023). For honest
comparison we pin the architecture across variants.

Engineered features come from Phase C's `forecast.regressor.FEATURE_COLS`
plus 4 QC columns (chip_count, chip_age_days_max, cloud_pct_max,
corn_pixel_frac_min) where applicable.

Output:
  runs/rowB_experiments_<ts>.csv
  stdout: per-variant per-forecast_date table + per-state breakdown
"""

from __future__ import annotations

import argparse
import datetime as dt
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

XGB_PARAMS = dict(
    objective="reg:squarederror",
    max_depth=4,
    learning_rate=0.05,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    device="cpu",
)
NUM_BOOST_ROUND = 600
EARLY_STOP = 50


def load_master(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = _add_derived_columns(df)
    return df


def load_emb(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(path)
    cols = sorted([c for c in df.columns if c.startswith("prithvi_")])
    return df, cols


def join_and_split(master: pd.DataFrame, emb: pd.DataFrame, emb_cols: list[str]
                   ) -> pd.DataFrame:
    keep = ["GEOID", "year", "forecast_date"] + QC_COLS + emb_cols
    keep = [c for c in keep if c in emb.columns]
    return master.merge(emb[keep], on=["GEOID", "year", "forecast_date"], how="left")


def fit_xgb(X_tr, y_tr, X_va, y_va):
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dva = xgb.DMatrix(X_va, label=y_va)
    booster = xgb.train(
        XGB_PARAMS, dtr,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dva, "val")],
        early_stopping_rounds=EARLY_STOP,
        verbose_eval=False,
    )
    pred = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
    return float(np.sqrt(((pred - y_va) ** 2).mean())), pred, booster.best_iteration


def evaluate_variant(name: str, master: pd.DataFrame, feature_cols_per_date: dict
                     ) -> list[dict]:
    """Train one xgb per forecast_date and return rows of {variant, date, rmse, n_train, n_val, ...}."""
    rows = []
    for date in VALID_DATES:
        feat = feature_cols_per_date[date]
        sub = master[master["forecast_date"] == date].copy()
        tr = sub[sub["year"].isin(TRAIN_YEARS)]
        va = sub[sub["year"] == VAL_YEAR]
        # require non-NaN target on both sides
        tr = tr[tr["yield_target"].notna()]
        va = va[va["yield_target"].notna()]
        # drop rows where ANY required feature is NaN-ALL (allow XGBoost to handle individual NaN)
        # XGBoost handles NaN; we keep all rows with target.
        X_tr = tr[feat].to_numpy(dtype=np.float32)
        y_tr = tr["yield_target"].to_numpy(dtype=np.float32)
        X_va = va[feat].to_numpy(dtype=np.float32)
        y_va = va["yield_target"].to_numpy(dtype=np.float32)
        rmse, pred, best_iter = fit_xgb(X_tr, y_tr, X_va, y_va)
        per_state = {}
        for s in va["state_alpha"].unique():
            mask = va["state_alpha"].values == s
            if mask.sum() > 0:
                per_state[s] = float(np.sqrt(((pred[mask] - y_va[mask]) ** 2).mean()))
        rows.append(dict(
            variant=name, forecast_date=date,
            n_train=len(tr), n_val=len(va),
            rmse_overall=rmse, best_iter=best_iter,
            n_features=len(feat),
            **{f"rmse_{s}": v for s, v in per_state.items()},
        ))
        print(f"  {name:8s} {date:5s} rmse={rmse:6.3f}  n_train={len(tr):4d} "
              f"n_val={len(va):3d}  feat={len(feat):4d}  best_iter={best_iter}")
    return rows


def add_ridge_pred_columns(master: pd.DataFrame, emb_cols_1024: list[str]
                           ) -> pd.DataFrame:
    """Train a Ridge probe per (forecast_date) on 2013-2022 chip-bearing rows
    over 1024-D embedding → produce a `ridge_pred` column on `master`.

    chip-less rows get NaN ridge_pred, which XGBoost handles natively.
    """
    print("[ridge] training Ridge probe on 1024-D embedding (per forecast_date)")
    out = master.copy()
    out["ridge_pred"] = np.nan
    for date in VALID_DATES:
        sub = out[out["forecast_date"] == date].copy()
        chip_mask = sub[emb_cols_1024[0]].notna()
        train_mask = chip_mask & sub["year"].isin(TRAIN_YEARS) & sub["yield_target"].notna()
        if int(train_mask.sum()) < 100:
            print(f"  [skip] {date}: only {int(train_mask.sum())} chip-bearing train rows")
            continue
        X_tr = sub.loc[train_mask, emb_cols_1024].to_numpy(dtype=np.float32)
        y_tr = sub.loc[train_mask, "yield_target"].to_numpy(dtype=np.float32)
        # Strong L2 — 1024 dims with ~3000 rows needs heavy regularization
        ridge = Ridge(alpha=10.0, random_state=42)
        ridge.fit(X_tr, y_tr)
        # predict on every chip-bearing row of this date
        any_chip_mask = chip_mask
        X_all = sub.loc[any_chip_mask, emb_cols_1024].to_numpy(dtype=np.float32)
        pred = ridge.predict(X_all).astype(np.float32)
        # write back via index
        idx = sub.loc[any_chip_mask].index
        out.loc[idx, "ridge_pred"] = pred
        # diagnostic: ridge RMSE on val
        val_mask = sub["year"].eq(VAL_YEAR) & chip_mask & sub["yield_target"].notna()
        if val_mask.sum() > 0:
            X_va = sub.loc[val_mask, emb_cols_1024].to_numpy(dtype=np.float32)
            y_va = sub.loc[val_mask, "yield_target"].to_numpy(dtype=np.float32)
            r = ridge.predict(X_va)
            ridge_rmse = float(np.sqrt(((r - y_va) ** 2).mean()))
            print(f"  [ridge] {date}: ridge-only val rmse = {ridge_rmse:.3f}  "
                  f"on {int(val_mask.sum())} chip-bearing val rows")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master", default="scripts/training_master.parquet")
    ap.add_argument("--emb-raw", default="data/v2/prithvi/embeddings_v1.parquet")
    ap.add_argument("--emb-pca32", default="data/v2/prithvi/embeddings_v1_pca32.parquet")
    ap.add_argument("--emb-pca64", default="data/v2/prithvi/embeddings_v1_pca64.parquet")
    args = ap.parse_args()

    master = load_master(Path(args.master))
    print(f"[load] master: {len(master):,} rows")

    # Filter to 2013+ (D.1 pool), drop any rows where target is NaN later inline.
    master = master[master["year"] >= 2013].copy()

    emb_raw, emb_cols_1024 = load_emb(Path(args.emb_raw))
    print(f"[load] raw 1024-D embeddings: {len(emb_raw):,} rows × {len(emb_cols_1024)} dims")

    emb_p32, emb_cols_32 = load_emb(Path(args.emb_pca32))
    print(f"[load] PCA-32 embeddings: {len(emb_p32):,} rows × {len(emb_cols_32)} dims")

    emb_p64, emb_cols_64 = load_emb(Path(args.emb_pca64))
    print(f"[load] PCA-64 embeddings: {len(emb_p64):,} rows × {len(emb_cols_64)} dims")

    all_results = []

    # === Variant: raw 1024 ===
    print("\n=== variant: raw1024 ===")
    m_raw = join_and_split(master, emb_raw, emb_cols_1024)
    feat_raw = {d: FEATURE_COLS[d] + QC_COLS + emb_cols_1024 for d in VALID_DATES}
    all_results += evaluate_variant("raw1024", m_raw, feat_raw)

    # === Variant: PCA-32 ===
    print("\n=== variant: pca32 ===")
    m_p32 = join_and_split(master, emb_p32, emb_cols_32)
    feat_p32 = {d: FEATURE_COLS[d] + QC_COLS + emb_cols_32 for d in VALID_DATES}
    all_results += evaluate_variant("pca32", m_p32, feat_p32)

    # === Variant: PCA-64 ===
    print("\n=== variant: pca64 ===")
    m_p64 = join_and_split(master, emb_p64, emb_cols_64)
    feat_p64 = {d: FEATURE_COLS[d] + QC_COLS + emb_cols_64 for d in VALID_DATES}
    all_results += evaluate_variant("pca64", m_p64, feat_p64)

    # === Variant: Ridge probe + XGBoost ===
    print("\n=== variant: ridge ===")
    m_ridge = add_ridge_pred_columns(m_raw, emb_cols_1024)
    feat_ridge = {d: FEATURE_COLS[d] + QC_COLS + ["ridge_pred"] for d in VALID_DATES}
    all_results += evaluate_variant("ridge", m_ridge, feat_ridge)

    # === Row A reference (engineered only, no Prithvi, no QC) ===
    print("\n=== reference: rowA (engineered-only) ===")
    feat_A = {d: FEATURE_COLS[d] for d in VALID_DATES}
    all_results += evaluate_variant("rowA_ref", master, feat_A)

    # === Summary ===
    df_res = pd.DataFrame(all_results)
    pivot = df_res.pivot_table(index="variant", columns="forecast_date",
                                values="rmse_overall", aggfunc="mean")
    pivot = pivot.reindex(["rowA_ref", "raw1024", "pca32", "pca64", "ridge"])
    pivot = pivot[VALID_DATES]
    print("\n" + "=" * 70)
    print("Per-variant val 2023 RMSE (county-level, all rows, NaN handled by XGBoost)")
    print("=" * 70)
    print(pivot.round(3).to_string())

    # Gate test: best Row B variant vs rowA_ref at EOS
    eos = df_res[df_res["forecast_date"] == "EOS"].set_index("variant")["rmse_overall"]
    rowa_eos = eos["rowA_ref"]
    print()
    print("=" * 70)
    print(f"GATE TEST (val 2023 EOS, threshold +5% lift vs Row A reference {rowa_eos:.3f})")
    print("=" * 70)
    for v in ["raw1024", "pca32", "pca64", "ridge"]:
        if v in eos.index:
            lift = (rowa_eos - eos[v]) / rowa_eos * 100
            verdict = "PASS ✓" if lift >= 5.0 else ("FAIL" if lift < 0 else "miss")
            print(f"  {v:8s} EOS rmse={eos[v]:.3f}   lift={lift:+.2f}%   {verdict}")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = REPO_ROOT / "runs" / f"rowB_experiments_{ts}.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nfull results → {out_csv}")


if __name__ == "__main__":
    main()