"""
forecast.baseline — naive 5-year county-mean baseline.

Required by the Phase B gate: "RMSE on point estimate better than the naive
'5-year county mean' baseline." This module defines exactly what that means
so the comparison is unambiguous.

Definition
----------
For a query (geoid_q, year_q, forecast_date_q):
    baseline_yield = mean of yield_target over the 5 most recent
                     non-holdout years strictly before year_q for which
                     yield_target is non-null in that GEOID.

Notes:
    - "Non-holdout" means we never look at year 2024 (holdout) when computing
      a baseline for any query, regardless of query year. When backtesting
      2023, we *do* allow 2018-2022 in the lookback. When backtesting 2024,
      we allow 2019-2023 in the lookback (2023 is the val year, fair game
      for the baseline since the baseline is competing against analog
      retrieval which itself sees 2023 as a non-holdout candidate when
      forecasting 2024).
    - "5 most recent" — if a county has fewer than 5 prior non-null years,
      we use what we have, with a minimum of 3. Counties with <3 prior years
      get NaN baseline (caller decides how to handle).
    - Same value across all 4 forecast_dates within a year. The baseline does
      not respond to in-season weather; that's the point of comparison.
    - State aggregation: planted-acres-weighted mean of county baselines,
      same convention as cone aggregation. Caveat-free here because we're
      averaging point estimates, not percentiles.

Public surface:
    county_baseline(master_df, geoid, year, forecast_date,
                    holdout_years=(2024,), min_years=3, lookback=5) -> float
    state_baseline(master_df, state_alpha, year, forecast_date, ...) -> float
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def county_baseline(
    master_df: pd.DataFrame,
    geoid: str,
    year: int,
    forecast_date: str = "EOS",
    holdout_years: Iterable[int] = (2024,),
    min_years: int = 3,
    lookback: int = 5,
) -> float:
    """Mean yield_target over the last `lookback` non-holdout years strictly
    before `year` for the given GEOID.

    Parameters
    ----------
    master_df : DataFrame
        The master table (or any superset with GEOID, year, forecast_date,
        yield_target).
    geoid : str
        5-digit zero-padded FIPS.
    year : int
        Query year. Years strictly less than this contribute.
    forecast_date : str
        Used only to dedupe to one row per (GEOID, year). The baseline does
        not vary across forecast_dates; the parameter exists for API symmetry
        with the analog retrieval flow.
    holdout_years : iterable of int
        Years to exclude from the lookback regardless of query year. Default
        excludes 2024 only.
    min_years : int
        Minimum prior years required to compute a baseline. Below this, returns NaN.
    lookback : int
        Maximum number of prior years to average. We take the 5 most recent
        eligible years.

    Returns
    -------
    float
        Mean yield in bu/acre, or NaN if fewer than `min_years` eligible.
    """
    holdout_set = set(int(y) for y in holdout_years)

    # Dedupe to one row per (GEOID, year). yield_target is the same across all
    # 4 forecast_dates in a year, so we just take the EOS row (or first if EOS missing).
    sub = master_df[
        (master_df["GEOID"] == geoid)
        & (master_df["year"] < year)
        & (~master_df["year"].isin(holdout_set))
        & (master_df["yield_target"].notna())
    ]
    if len(sub) == 0:
        return float("nan")

    per_year = (
        sub.drop_duplicates(subset=["year"])
        .sort_values("year", ascending=False)
        .head(lookback)
    )
    if len(per_year) < min_years:
        return float("nan")

    return float(per_year["yield_target"].mean())


def state_baseline(
    master_df: pd.DataFrame,
    state_alpha: str,
    year: int,
    forecast_date: str = "EOS",
    holdout_years: Iterable[int] = (2024,),
    min_years: int = 3,
    lookback: int = 5,
) -> Tuple[float, int]:
    """Planted-acres-weighted state baseline.

    Returns
    -------
    (baseline_yield, n_counties)
        baseline_yield in bu/acre. n_counties is how many counties contributed
        a non-NaN county baseline; if 0, baseline_yield is NaN.
    """
    sub = master_df[
        (master_df["state_alpha"] == state_alpha) & (master_df["year"] == year)
    ]
    if len(sub) == 0:
        return float("nan"), 0

    # Take one row per GEOID for acres (same value across forecast_dates).
    per_geoid = sub.drop_duplicates(subset=["GEOID"])

    geoids = per_geoid["GEOID"].tolist()
    acres = per_geoid["acres_planted_all"].to_numpy(dtype=np.float64)

    baselines = np.array(
        [
            county_baseline(
                master_df,
                g,
                year,
                forecast_date=forecast_date,
                holdout_years=holdout_years,
                min_years=min_years,
                lookback=lookback,
            )
            for g in geoids
        ],
        dtype=np.float64,
    )

    valid = ~np.isnan(baselines) & ~np.isnan(acres) & (acres > 0)
    if not valid.any():
        return float("nan"), 0

    w = acres[valid]
    b = baselines[valid]
    weighted = float((b * w).sum() / w.sum())
    return weighted, int(valid.sum())
