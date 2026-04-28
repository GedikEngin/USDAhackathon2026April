#!/usr/bin/env python
"""
scripts/pca_embeddings.py
Fit PCA on the 1024-D Prithvi embedding (2013-2022 chip-bearing rows only,
no val leak), apply to all rows, write a reduced-dim parquet that the
existing train_regressor_d1.py can consume directly.

The output schema matches embeddings_v1.parquet's: keys (GEOID, year,
forecast_date), QC features (chip_count, chip_age_days_max, cloud_pct_max,
corn_pixel_frac_min), model_version, and prithvi_0000..prithvi_NNNN columns.
The trainer auto-detects "prithvi_*" columns regardless of count.

Usage:
    python scripts/pca_embeddings.py --k 32
    python scripts/pca_embeddings.py --k 64 --out data/v2/prithvi/embeddings_v1_pca64.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parent.parent
EMB_IN = REPO_ROOT / "data" / "v2" / "prithvi" / "embeddings_v1.parquet"
TRAIN_YEAR_MIN = 2013
TRAIN_YEAR_MAX = 2022  # exclude 2023 (val) and 2024 (holdout) from PCA fit


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=32, help="number of principal components")
    ap.add_argument("--in", dest="inp", default=str(EMB_IN),
                    help="input embeddings parquet")
    ap.add_argument("--out", default=None,
                    help="output parquet (default: embeddings_v1_pca{k}.parquet)")
    args = ap.parse_args()

    in_path = Path(args.inp)
    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "data" / "v2" / "prithvi" / f"embeddings_v1_pca{args.k}.parquet"
    )

    print(f"[load] {in_path}")
    df = pd.read_parquet(in_path)
    print(f"  {len(df):,} rows × {len(df.columns):,} cols")

    emb_cols = sorted([c for c in df.columns if c.startswith("prithvi_")])
    print(f"  found {len(emb_cols)} embedding columns")
    assert len(emb_cols) > 0, "no prithvi_* columns found"

    # chip-bearing mask = embedding is non-NaN (== chip_count > 0)
    has_chips = df[emb_cols[0]].notna()
    print(f"  chip-bearing rows: {int(has_chips.sum()):,} / {len(df):,}")

    # Fit PCA on TRAIN years only (no val/holdout leak)
    fit_mask = has_chips & df["year"].between(TRAIN_YEAR_MIN, TRAIN_YEAR_MAX)
    n_fit = int(fit_mask.sum())
    print(f"  PCA fit pool ({TRAIN_YEAR_MIN}-{TRAIN_YEAR_MAX}, chip-bearing): {n_fit:,} rows")
    if n_fit < args.k * 2:
        sys.exit(f"too few fit rows ({n_fit}) for k={args.k}")

    X_fit = df.loc[fit_mask, emb_cols].to_numpy(dtype=np.float32)
    print(f"[fit] PCA k={args.k}")
    pca = PCA(n_components=args.k, svd_solver="auto", random_state=42)
    pca.fit(X_fit)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    print(f"  cumulative variance explained at k={args.k}: {cum_var[-1]*100:.1f}%")
    print(f"  first 8 component variances: " +
          " ".join(f"{v*100:.1f}%" for v in pca.explained_variance_ratio_[:8]))

    # Transform all chip-bearing rows; non-chip rows stay NaN
    print("[transform] applying to all chip-bearing rows")
    X_all = df.loc[has_chips, emb_cols].to_numpy(dtype=np.float32)
    Z_all = pca.transform(X_all)

    # Build output dataframe
    pc_cols = [f"prithvi_{i:04d}" for i in range(args.k)]
    out = df.drop(columns=emb_cols).copy()
    for col in pc_cols:
        out[col] = np.nan
    out.loc[has_chips, pc_cols] = Z_all.astype(np.float32)

    # Stamp model_version so downstream traceability is preserved
    if "model_version" in out.columns:
        out["model_version"] = out["model_version"].astype(str) + f"+pca{args.k}"

    # Reorder cols: keys + QC + model_version + extracted_at + pc_cols
    key_cols = ["GEOID", "year", "forecast_date"]
    qc_cols = [c for c in ["chip_count", "chip_age_days_max", "cloud_pct_max",
                            "corn_pixel_frac_min"] if c in out.columns]
    extra_cols = [c for c in out.columns
                  if c not in key_cols + qc_cols + pc_cols
                  and not c.startswith("prithvi_")]
    out = out[key_cols + qc_cols + extra_cols + pc_cols]

    print(f"[write] {out_path}  ({len(out):,} rows × {len(out.columns):,} cols)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    out.to_parquet(tmp, index=False, compression="snappy")
    tmp.replace(out_path)
    size_mb = out_path.stat().st_size / 1e6
    print(f"  done ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()