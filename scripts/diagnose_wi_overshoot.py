"""
scripts/diagnose_wi_overshoot.py — figure out why WI predictions are ~16 bu high.

Quick diagnostic. Picks one queryable WI county at the EOS forecast date for
holdout year 2024, prints:
  - the 5 analog years and counties chosen
  - their raw yields and detrended yields
  - the WI trend line value at each year
  - the trend line value at 2024 (where retrending happens)
  - the actual WI 2024 truth
  - what the prediction would be under three counterfactual detrend strategies
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forecast.analog import AnalogIndex
from forecast.data import load_master, train_pool
from forecast.detrend import fit as fit_trend
from forecast.features import fit_standardizer


def main() -> None:
    master_df = load_master("scripts/training_master.parquet")
    train_df, _ = train_pool(master_df, n_min_history=10)
    standardizer = fit_standardizer(train_df)
    trend = fit_trend(train_df)
    index = AnalogIndex.fit(train_df, standardizer, trend)

    # State-level: what does the WI trend predict for each year vs reality?
    print("=== WI state trend vs. observed acres-weighted state yield ===")
    print(f"  WI slope = {trend.slopes['WI']:+.3f} bu/acre/yr")
    print(f"  WI intercept = {trend.intercepts['WI']:+.1f}")
    print()
    print(f"  {'year':>4}  {'trend_pred':>11}  {'actual_state':>13}  {'residual':>10}")
    for year in range(2005, 2025):
        sub = master_df[
            (master_df["state_alpha"] == "WI") & (master_df["year"] == year)
        ].drop_duplicates(subset=["GEOID"])
        yields = sub["yield_target"].to_numpy(dtype=np.float64)
        acres = sub["acres_planted_all"].to_numpy(dtype=np.float64)
        valid = ~np.isnan(yields) & ~np.isnan(acres) & (acres > 0)
        actual = (
            float((yields[valid] * acres[valid]).sum() / acres[valid].sum())
            if valid.any()
            else float("nan")
        )
        pred = float(trend.predict("WI", year))
        marker = "  <- TRAIN" if year <= 2022 else ("  <- val" if year == 2023 else "  <- HOLDOUT")
        print(f"  {year:>4}  {pred:>11.1f}  {actual:>13.1f}  {actual - pred:>+10.1f}{marker}")

    # Drill into one queryable WI county at 2024 EOS
    target_geoid = None
    candidates = master_df[
        (master_df["state_alpha"] == "WI")
        & (master_df["year"] == 2024)
        & (master_df["forecast_date"] == "EOS")
    ]
    # Pick the largest-acres WI county for visibility
    if len(candidates):
        target_geoid = (
            candidates.sort_values("acres_planted_all", ascending=False)
            .iloc[0]["GEOID"]
        )
    if target_geoid is None:
        print("\nNo WI 2024 EOS rows found — aborting county drill-down.")
        return

    print(f"\n=== Drilling into WI GEOID={target_geoid} at 2024 EOS ===")
    qrow = master_df[
        (master_df["GEOID"] == target_geoid)
        & (master_df["year"] == 2024)
        & (master_df["forecast_date"] == "EOS")
    ].iloc[0]
    print(f"  county actual yield: {qrow['yield_target']:.1f}")
    print(f"  county acres planted: {qrow['acres_planted_all']:,.0f}")

    analogs = index.find(
        geoid=target_geoid,
        year=2024,
        forecast_date="EOS",
        query_features=qrow,
        k=5,
        pool="same_geoid",
    )
    print(f"\n  {'analog_year':>11}  {'raw_yield':>9}  {'trend@year':>10}  {'detrended':>9}  {'distance':>8}")
    for a in analogs:
        trend_at_year = float(trend.predict(a.state_alpha, a.year))
        print(
            f"  {a.year:>11}  {a.observed_yield:>9.1f}  {trend_at_year:>10.1f}  "
            f"{a.detrended_yield:>+9.1f}  {a.distance:>8.3f}"
        )

    median_detrended = float(np.median([a.detrended_yield for a in analogs]))
    trend_at_2024 = float(trend.predict("WI", 2024))
    pred_current = trend_at_2024 + median_detrended

    print(f"\n  median detrended: {median_detrended:+.1f}")
    print(f"  WI trend at 2024: {trend_at_2024:.1f}")
    print(f"  prediction (= trend@2024 + median_detrended): {pred_current:.1f}")
    print(f"  county actual:    {qrow['yield_target']:.1f}")
    print(f"  county error:     {pred_current - qrow['yield_target']:+.1f}")

    # Counterfactuals
    print("\n=== Counterfactual detrend strategies for this query ===")

    # CF1: no retrend — just predict median raw analog yield
    median_raw = float(np.median([a.observed_yield for a in analogs]))
    print(f"  CF1 (no detrend, median raw analog yield):  {median_raw:.1f}  "
          f"(error {median_raw - qrow['yield_target']:+.1f})")

    # CF2: trend fit on last 5 train years only (2018-2022)
    recent_train = train_df[train_df["year"].between(2018, 2022)]
    recent_trend = fit_trend(recent_train)
    if "WI" in recent_trend.slopes:
        recent_trend_at_2024 = float(recent_trend.predict("WI", 2024))
        # Re-detrend the analogs using the recent trend, take median, retrend
        re_detrended = [
            a.observed_yield - float(recent_trend.predict(a.state_alpha, a.year))
            for a in analogs
        ]
        med_re = float(np.median(re_detrended))
        pred_cf2 = recent_trend_at_2024 + med_re
        print(f"  CF2 (trend fit on 2018-2022 only):          {pred_cf2:.1f}  "
              f"(error {pred_cf2 - qrow['yield_target']:+.1f})  "
              f"[WI slope = {recent_trend.slopes['WI']:+.3f}]")

    # CF3: clip the retrend horizon — use trend@(min(query_year, analog_year+5))
    clipped = []
    for a in analogs:
        clip_year = min(2024, a.year + 5)
        retrended = float(trend.predict("WI", clip_year)) + a.detrended_yield
        clipped.append(retrended)
    pred_cf3 = float(np.median(clipped))
    print(f"  CF3 (clip retrend to analog_year+5):        {pred_cf3:.1f}  "
          f"(error {pred_cf3 - qrow['yield_target']:+.1f})")

    # CF4: don't detrend at all, AND apply state-level mean shift (2018-2022 vs 2024)
    # i.e. take median raw, shift by the mean residual of the recent train years.
    recent_state_actual = master_df[
        (master_df["state_alpha"] == "WI") & (master_df["year"].between(2018, 2022))
    ].drop_duplicates(subset=["GEOID", "year"])["yield_target"].mean()
    print(f"  reference: 2018-2022 train WI mean county yield = {recent_state_actual:.1f}")
    print(f"  reference: 2023 + 2024 truth WI state yield avg ~ "
          f"{(175.4 + 172.0) / 2:.1f}")


if __name__ == "__main__":
    main()
