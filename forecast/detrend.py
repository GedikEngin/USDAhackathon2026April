"""
forecast.detrend — per-state linear yield trend, fit on the training pool.

Why this exists
---------------
Corn yields trend upward over time (genetics, agronomy, management). Across the
2005-2024 modeling window, the U.S. corn-belt trend is roughly +1.5 bu/acre/year.
If we don't detrend, an analog from 2008 systematically under-predicts a 2023
query — the K-NN finds weather-similar years, but the yield distribution it
returns is anchored to whatever decade those analogs come from.

Convention
----------
Per-state OLS on `yield_target ~ year` over the training pool (2005-2022,
post min-history filter). We fit per-state, not per-county, because:
  - per-state has 18 years × ~80 counties = ~1,400 obs per state, very stable
  - per-county trends are noisy with only 18 points; year-to-year weather noise
    overwhelms the genetic-gain signal
  - state-level matches the brief's reporting unit (we ultimately publish state
    forecasts) and the level at which agronomic improvements are reported

Sign convention: a "detrended yield" is the residual after subtracting the
per-state linear prediction at that year. So a 2008 IA county that yielded
175 bu/acre against a 165 bu/acre state-trend fit gets detrended_yield = +10.
A 2023 query gets retrend(median(detrended_analogs)) added back at the
state-trend prediction for 2023.

Public surface
--------------
    fit(train_df)                  -> StateTrend
    StateTrend.detrend(df)         -> df with detrended_yield_target column
    StateTrend.retrend(state, year, value)  -> value + per-state trend at year
    StateTrend.save(path) / load(path)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class StateTrend:
    """Per-state linear trend `yield_target = slope * year + intercept`.

    Attributes
    ----------
    slopes      : dict[state_alpha, float]    bu/acre per year
    intercepts  : dict[state_alpha, float]    bu/acre at year=0
    fit_years   : tuple[int, int]             (min, max) train-pool years used
    fit_n       : dict[state_alpha, int]      training row count per state
    """

    slopes: Dict[str, float] = field(default_factory=dict)
    intercepts: Dict[str, float] = field(default_factory=dict)
    fit_years: tuple = (0, 0)
    fit_n: Dict[str, int] = field(default_factory=dict)

    def predict(self, state_alpha: str, year: int | np.ndarray) -> float | np.ndarray:
        """Trend prediction for `state_alpha` at `year`.

        Accepts scalar or array `year`; returns matching shape.
        """
        if state_alpha not in self.slopes:
            raise KeyError(
                f"No trend fit for state {state_alpha!r}. "
                f"Fit states: {sorted(self.slopes.keys())}"
            )
        return self.slopes[state_alpha] * np.asarray(year) + self.intercepts[state_alpha]

    def detrend(self, df: pd.DataFrame, *, target_col: str = "yield_target") -> pd.DataFrame:
        """Return a copy of `df` with a `detrended_<target_col>` column added.

        Rows with state_alpha not in the fit are passed through with NaN
        detrended value (and a one-time warning is the caller's responsibility).
        Rows with NaN target keep NaN detrended target (e.g. unharvested holdout).
        """
        if "state_alpha" not in df.columns or "year" not in df.columns:
            raise KeyError("df must have state_alpha and year columns")
        if target_col not in df.columns:
            raise KeyError(f"df missing target column {target_col!r}")

        out = df.copy()
        # Vectorize trend prediction by mapping per-state slope/intercept.
        # Rows whose state isn't in the fit get NaN trend (preserves yield_target NaN behavior).
        slope_arr = out["state_alpha"].map(self.slopes).to_numpy(dtype=np.float64)
        intercept_arr = out["state_alpha"].map(self.intercepts).to_numpy(dtype=np.float64)
        year_arr = out["year"].to_numpy(dtype=np.float64)
        trend = slope_arr * year_arr + intercept_arr  # NaN where state unmapped

        out[f"detrended_{target_col}"] = out[target_col].to_numpy(dtype=np.float64) - trend
        return out

    def retrend(
        self,
        state_alpha: str,
        year: int,
        detrended_value: float | np.ndarray,
    ) -> float | np.ndarray:
        """Inverse of detrend at the (state, year) level. Add back the trend."""
        return np.asarray(detrended_value) + self.predict(state_alpha, year)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "slopes": self.slopes,
            "intercepts": self.intercepts,
            "fit_years": list(self.fit_years),
            "fit_n": self.fit_n,
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "StateTrend":
        payload = json.loads(Path(path).read_text())
        return cls(
            slopes=payload["slopes"],
            intercepts=payload["intercepts"],
            fit_years=tuple(payload["fit_years"]),
            fit_n=payload.get("fit_n", {}),
        )


def fit(train_df: pd.DataFrame, *, target_col: str = "yield_target") -> StateTrend:
    """Fit per-state linear trend on the training pool.

    The caller passes the training subset of the master table. We collapse to
    one row per (GEOID, year) before fitting (the master table has 4 forecast_dates
    per year; yield_target is identical across them). This prevents the regression
    from being implicitly weighted 4x.

    Raises
    ------
    ValueError if any state has fewer than 5 distinct years of non-null target,
    which is far below what we expect (18 train years × ~80 counties per state)
    and indicates an upstream problem.
    """
    required = ["GEOID", "year", "state_alpha", target_col]
    missing = [c for c in required if c not in train_df.columns]
    if missing:
        raise KeyError(f"train_df missing required columns: {missing}")

    one_per_geoid_year = (
        train_df[required]
        .drop_duplicates(subset=["GEOID", "year"])
        .dropna(subset=[target_col])
    )
    if len(one_per_geoid_year) == 0:
        raise ValueError("No non-null target rows in train_df after deduplication.")

    fit_min = int(one_per_geoid_year["year"].min())
    fit_max = int(one_per_geoid_year["year"].max())

    slopes: Dict[str, float] = {}
    intercepts: Dict[str, float] = {}
    fit_n: Dict[str, int] = {}

    for state in sorted(one_per_geoid_year["state_alpha"].unique()):
        sub = one_per_geoid_year[one_per_geoid_year["state_alpha"] == state]
        n_distinct_years = sub["year"].nunique()
        if n_distinct_years < 5:
            raise ValueError(
                f"State {state!r} has only {n_distinct_years} distinct training "
                f"year(s); refusing to fit a trend on that. Check the train pool."
            )
        # Plain OLS via numpy. polyfit deg=1 returns [slope, intercept].
        slope, intercept = np.polyfit(
            sub["year"].to_numpy(dtype=np.float64),
            sub[target_col].to_numpy(dtype=np.float64),
            deg=1,
        )
        slopes[state] = float(slope)
        intercepts[state] = float(intercept)
        fit_n[state] = int(len(sub))

    return StateTrend(
        slopes=slopes,
        intercepts=intercepts,
        fit_years=(fit_min, fit_max),
        fit_n=fit_n,
    )
