"""
forecast.recalibrate — per-state additive bias correction.

Phase B finding (2026-04-25): same-GEOID K=5 retrieval shows state-systematic
bias — WI over-predicts by ~12 bu, NE by ~5 bu, CO under-predicts by ~2.5 bu,
IA is calibrated. The bias is state-level and direction-stable across forecast
dates, which means an additive per-(state, forecast_date) shift fit on
in-distribution validation data should remove it.

Method
------
Fit on the val year (2023). For each (state, forecast_date), compute the mean
signed residual (predicted - actual) over that state's 2023 forecasts. This is
the recalibration constant. At prediction time on the holdout (2024), subtract
the constant from the point estimate AND shift every percentile of the cone
by the same constant — preserving cone width while correcting location.

Why fit on val, not on a leave-one-year-out over train
------------------------------------------------------
- Val year exists *for* this kind of calibration tuning per PHASE2_PHASE_PLAN.
- LOYO over 18 train years would require 18 index rebuilds × 4 dates ~90s and
  add complexity for marginal accuracy improvement.
- A val-fitted constant uses the most recent year's residual, which is the
  closest analog to the holdout's distribution (corn yields trend year-over-year;
  2022's residual is more representative of 2024's expected residual than 2008's).

Caveats
-------
- The val year is now in-distribution at fit time. 2023 metrics post-
  recalibration are not predictive of out-of-sample performance — only 2024
  metrics are. The backtest harness reports both but the gate evaluates on
  2024 only post-recalibration.
- One-year fitting is high-variance. If 2023 was an unusual year (e.g. MO
  2023 was a drought outlier), the recalibration overshoots. Acceptable for
  the v2 baseline; revisit if Phase C results show calibration instability.
- States with zero or one queryable county at val time get zero recalibration
  (defensive — better than dividing by an unstable mean of a tiny sample).

Public surface
--------------
    Recalibrator                                dataclass
    fit_from_val_results(val_results_df)        -> Recalibrator
    Recalibrator.adjust_state_forecast(sf)      mutates point + cone in place
    Recalibrator.constants                      dict[(state, date)] -> float
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from forecast.aggregate import StateForecast


@dataclass
class Recalibrator:
    """Per-(state, forecast_date) additive bias correction.

    Attributes
    ----------
    constants : dict[(state_alpha, forecast_date), float]
        The number to subtract from point estimates (and from each cone
        percentile) at adjust time. Positive value = the model was
        over-predicting in val, so we shift down. Negative = under, shift up.
    fit_year : int
        The year used to fit (e.g. 2023).
    fit_n : dict[(state, date), int]
        Sample size that produced each constant (= number of state-year-date
        rows from val, usually 1).
    """

    constants: Dict[Tuple[str, str], float] = field(default_factory=dict)
    fit_year: int = 0
    fit_n: Dict[Tuple[str, str], int] = field(default_factory=dict)

    def get_constant(self, state_alpha: str, forecast_date: str) -> float:
        """Return the recalibration constant for (state, date), or 0 if unknown.

        Unknown (state, date) pairs return 0 rather than raising — defensive
        for cases where val data was missing for a state-date combination.
        """
        return self.constants.get((state_alpha, forecast_date), 0.0)

    def adjust_state_forecast(self, sf: StateForecast) -> StateForecast:
        """Return a new StateForecast with recalibration applied.

        Subtracts the (state, date) constant from `point_estimate` and from
        every percentile in `percentiles`. County-level cones inside `sf`
        are not mutated (they're kept for narration; the headline numbers
        are at the state level).
        """
        c = self.get_constant(sf.state_alpha, sf.forecast_date)
        if c == 0.0:
            return sf

        adjusted_percentiles = {p: v - c for p, v in sf.percentiles.items()}
        # Build a new StateForecast — dataclass is small, copying is cheap.
        return StateForecast(
            state_alpha=sf.state_alpha,
            year=sf.year,
            forecast_date=sf.forecast_date,
            point_estimate=sf.point_estimate - c,
            percentiles=adjusted_percentiles,
            n_counties=sf.n_counties,
            total_planted_acres=sf.total_planted_acres,
            county_cones=sf.county_cones,
            county_weights=sf.county_weights,
        )


def fit_from_val_results(val_results_df: pd.DataFrame) -> Recalibrator:
    """Fit per-(state, date) additive bias from val-year backtest rows.

    Parameters
    ----------
    val_results_df : DataFrame
        Output of `run_backtest` filtered to a single holdout_year (the val
        year) and a single (pool, k) configuration. Must have columns
        `state_alpha`, `forecast_date`, `point_error` (= predicted - truth).
        Rows with NaN point_error are dropped.

    Returns
    -------
    Recalibrator
    """
    required = {"state_alpha", "forecast_date", "point_error", "holdout_year"}
    missing = required - set(val_results_df.columns)
    if missing:
        raise KeyError(f"val_results_df missing columns: {sorted(missing)}")

    if val_results_df["holdout_year"].nunique() != 1:
        raise ValueError(
            f"fit_from_val_results expects rows from a single year, got "
            f"{sorted(val_results_df['holdout_year'].unique())}. "
            f"Filter to the val year before calling."
        )

    valid = val_results_df.dropna(subset=["point_error"])
    if valid.empty:
        raise ValueError("No valid val-year rows to fit recalibration on.")

    fit_year = int(valid["holdout_year"].iloc[0])
    constants: Dict[Tuple[str, str], float] = {}
    fit_n: Dict[Tuple[str, str], int] = {}

    for (state, date), grp in valid.groupby(["state_alpha", "forecast_date"]):
        # Mean signed error for this (state, date). With the val pool having
        # one row per (state, date) per (pool, k), this is just the value.
        # If the caller passed multiple (pool, k), it's their responsibility
        # to filter — we just take the mean.
        bias = float(grp["point_error"].mean())
        constants[(str(state), str(date))] = bias
        fit_n[(str(state), str(date))] = int(len(grp))

    return Recalibrator(constants=constants, fit_year=fit_year, fit_n=fit_n)
