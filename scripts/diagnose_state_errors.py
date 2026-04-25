"""
scripts/diagnose_state_errors.py — decompose state-level errors into bias + variance.

For the Phase B same_geoid K=5 results, print per-state:
  - signed mean error across (year, forecast_date) tuples = bias
  - RMSE = total error magnitude
  - bias-vs-variance decomposition: RMSE^2 = bias^2 + variance

If a state's bias dominates RMSE, it's a fixable systematic offset (e.g. WI
always over-predicts, MO always under-predicts). If variance dominates, the
model's central tendency is right and the residuals are just noise — the cone
covers them but no point-estimate fix can reduce RMSE.

Reads the most recent backtest_baseline_*.csv from runs/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    runs = sorted(Path("runs").glob("backtest_baseline_*.csv"))
    if not runs:
        print("No backtest CSV found in runs/. Run scripts.backtest_baseline first.")
        return
    path = runs[-1]
    print(f"Reading {path}\n")
    df = pd.read_csv(path)

    # Filter to same_geoid pool, K=5
    sub = df[(df["pool"] == "same_geoid") & (df["k"] == 5)].copy()
    if sub.empty:
        print("No same_geoid K=5 rows. Re-run with --k-sweep including 5.")
        return

    # Per-state bias + variance decomposition
    print("=== Per-state error decomposition (same_geoid, K=5) ===")
    print(f"  {'state':<6} {'n':>3}  {'bias':>8}  {'std':>8}  {'rmse':>8}  "
          f"{'bias^2/rmse^2':>14}  {'best_dir':>10}")
    rows = []
    for state, grp in sub.groupby("state_alpha"):
        errs = grp["point_error"].dropna().to_numpy()
        if len(errs) == 0:
            continue
        bias = float(errs.mean())
        std = float(errs.std(ddof=0))
        rmse = float(np.sqrt(np.mean(errs ** 2)))
        bias_share = bias ** 2 / max(rmse ** 2, 1e-9)
        best_dir = "over" if bias > 0 else "under"
        print(f"  {state:<6} {len(errs):>3}  {bias:>+8.2f}  {std:>8.2f}  "
              f"{rmse:>8.2f}  {bias_share:>14.1%}  {best_dir:>10}")
        rows.append((state, bias, std, rmse, bias_share))

    # Print per-(state, date) signed errors so we can see if the bias is steady
    print("\n=== Signed point_error per (state, year, date) — same_geoid, K=5 ===")
    pivot = (
        sub.assign(label=lambda d: d["holdout_year"].astype(str) + "_" + d["forecast_date"])
        .pivot_table(
            index="state_alpha",
            columns="label",
            values="point_error",
            aggfunc="first",
        )
        .round(1)
    )
    print(pivot.to_string())

    # Overall: is the same_geoid K=5 cone covering on bias-dominated states?
    print("\n=== Coverage on bias-dominated vs variance-dominated states ===")
    for state, bias, std, rmse, bias_share in rows:
        cov = float(sub[sub["state_alpha"] == state]["in_cone_80"].mean())
        kind = "BIAS-DOMINATED" if bias_share > 0.5 else "variance-dominated"
        print(f"  {state}: bias={bias:+.1f}  rmse={rmse:.1f}  coverage={cov:.0%}  ({kind})")


if __name__ == "__main__":
    main()
