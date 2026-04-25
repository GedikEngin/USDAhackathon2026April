"""
forecast.detrend — per-county linear yield trend, fit on the training pool.

Why per-county
--------------
Phase B diagnostic (2026-04-25) revealed that a per-state trend fit by OLS
under-fits in WI: 14 of 18 train years had positive residuals, meaning the
linear line systematically sat below the actual yield curve. When K-NN
analog retrieval picks weather-similar years, those analogs inherit the
same bias (their detrended residuals skew positive), and retrending forward
through the same too-flat trend bakes the bias into every prediction. WI
ended up systematically over-predicted by ~16 bu/acre.

A per-county trend doesn't share the cross-county confounding that drives
this bias. Each county gets its own slope and intercept fit on its own
18-year history (post min-history filter we have ≥10 years per county).
When a county's analog pool returns same-GEOID matches, the detrended
residuals are measured against that county's own trend line, and the
median residual centers near zero by construction.

Trade-offs
----------
- 18 OLS points per county is noisier than 1,400 points per state. Some
  county slopes will be implausibly large/small. We don't clamp slopes —
  the cone width absorbs the noise via the analog distribution.
- Counties below the min-history threshold can still be queried (we
  forecast for them) but their per-county trend isn't reliable. As a
  defensive fallback we use the state median of per-county fits within
  that state — this represents "typical county trend" rather than a slope
  dragged by the largest counties.

API (signatures changed: keying on GEOID instead of state_alpha)
----------------------------------------------------------------
    fit(train_df)                       -> CountyTrend
    CountyTrend.predict(geoid, year)    scalar/array, fallback to state if unknown
    CountyTrend.detrend(df)             adds detrended_yield_target column
    CountyTrend.retrend(geoid, year, v) inverse
    CountyTrend.save / .load
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class CountyTrend:
    """Per-county linear trend `yield_target = slope * year + intercept`,
    with per-state median fallback for counties without a per-county fit.

    Attributes
    ----------
    county_slopes / county_intercepts : per-GEOID slope and intercept
    state_fallback_slopes / state_fallback_intercepts :
        per-state median of the per-county fits, used when a queried GEOID
        has no per-county fit.
    geoid_to_state : lookup so retrend can find the right fallback if the
        per-county fit is missing.
    fit_years : (min, max) training-window years used for the fit.
    fit_n : per-GEOID training row count.
    """

    county_slopes: Dict[str, float] = field(default_factory=dict)
    county_intercepts: Dict[str, float] = field(default_factory=dict)
    state_fallback_slopes: Dict[str, float] = field(default_factory=dict)
    state_fallback_intercepts: Dict[str, float] = field(default_factory=dict)
    geoid_to_state: Dict[str, str] = field(default_factory=dict)
    fit_years: tuple = (0, 0)
    fit_n: Dict[str, int] = field(default_factory=dict)

    # ---- prediction ---------------------------------------------------------

    def _slope_intercept_for(self, geoid: str) -> tuple[float, float]:
        """Return (slope, intercept) for a GEOID, falling back to state median
        if the GEOID has no per-county fit."""
        if geoid in self.county_slopes:
            return self.county_slopes[geoid], self.county_intercepts[geoid]
        state = self.geoid_to_state.get(geoid)
        if state is None or state not in self.state_fallback_slopes:
            raise KeyError(
                f"GEOID {geoid!r} has no per-county trend and no state fallback. "
                f"Fit GEOIDs: {len(self.county_slopes)}, "
                f"fit states: {sorted(self.state_fallback_slopes.keys())}"
            )
        return (
            self.state_fallback_slopes[state],
            self.state_fallback_intercepts[state],
        )

    def predict(self, geoid: str, year: int | np.ndarray) -> float | np.ndarray:
        """Trend prediction for `geoid` at `year`. Accepts scalar or array year."""
        slope, intercept = self._slope_intercept_for(geoid)
        return slope * np.asarray(year) + intercept

    def detrend(self, df: pd.DataFrame, *, target_col: str = "yield_target") -> pd.DataFrame:
        """Add `detrended_<target_col>` column = target − per-county trend at each row's year.

        Rows whose GEOID has no per-county fit AND no state fallback get NaN
        detrended target — caller is responsible for filtering or handling.
        """
        if "GEOID" not in df.columns or "year" not in df.columns:
            raise KeyError("df must have GEOID and year columns")
        if target_col not in df.columns:
            raise KeyError(f"df missing target column {target_col!r}")

        out = df.copy()
        geoids = out["GEOID"].to_numpy()
        slopes = np.empty(len(out), dtype=np.float64)
        intercepts = np.empty(len(out), dtype=np.float64)
        for i, g in enumerate(geoids):
            try:
                s, b = self._slope_intercept_for(str(g))
            except KeyError:
                s, b = np.nan, np.nan
            slopes[i] = s
            intercepts[i] = b

        years = out["year"].to_numpy(dtype=np.float64)
        trend = slopes * years + intercepts
        out[f"detrended_{target_col}"] = (
            out[target_col].to_numpy(dtype=np.float64) - trend
        )
        return out

    def retrend(
        self, geoid: str, year: int, detrended_value: float | np.ndarray
    ) -> float | np.ndarray:
        """Inverse of detrend at the (geoid, year) level. Add back the trend."""
        return np.asarray(detrended_value) + self.predict(geoid, year)

    # ---- serialization ------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "county_slopes": self.county_slopes,
            "county_intercepts": self.county_intercepts,
            "state_fallback_slopes": self.state_fallback_slopes,
            "state_fallback_intercepts": self.state_fallback_intercepts,
            "geoid_to_state": self.geoid_to_state,
            "fit_years": list(self.fit_years),
            "fit_n": self.fit_n,
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CountyTrend":
        payload = json.loads(Path(path).read_text())
        return cls(
            county_slopes=payload["county_slopes"],
            county_intercepts=payload["county_intercepts"],
            state_fallback_slopes=payload["state_fallback_slopes"],
            state_fallback_intercepts=payload["state_fallback_intercepts"],
            geoid_to_state=payload["geoid_to_state"],
            fit_years=tuple(payload["fit_years"]),
            fit_n=payload.get("fit_n", {}),
        )


# Backwards-compat alias — old code/scripts still importing StateTrend
# don't have to change. The class itself is the same; only the API keying
# moved from state_alpha to GEOID.
StateTrend = CountyTrend


# -----------------------------------------------------------------------------
# Fitting
# -----------------------------------------------------------------------------


def fit(
    train_df: pd.DataFrame,
    *,
    target_col: str = "yield_target",
    min_county_years: int = 5,
) -> CountyTrend:
    """Fit per-county OLS on the training pool, with state-median fallback.

    Parameters
    ----------
    train_df : DataFrame
        Training-pool slice. Must have GEOID, year, state_alpha, target_col.
        Caller is responsible for prior filtering (min-history, complete
        embedding, non-null target).
    target_col : str
        Column to fit. Default 'yield_target'.
    min_county_years : int
        Minimum distinct years required for a per-county fit. Below this,
        the county is left out of the per-county map and will hit the state
        fallback at predict time. Default 5 (generous; the upstream
        min-history filter is 10).

    Returns
    -------
    CountyTrend
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

    county_slopes: Dict[str, float] = {}
    county_intercepts: Dict[str, float] = {}
    fit_n: Dict[str, int] = {}
    geoid_to_state: Dict[str, str] = {}

    for (geoid, state), sub in one_per_geoid_year.groupby(["GEOID", "state_alpha"]):
        geoid_to_state[str(geoid)] = str(state)
        n_distinct = sub["year"].nunique()
        if n_distinct < min_county_years:
            continue
        slope, intercept = np.polyfit(
            sub["year"].to_numpy(dtype=np.float64),
            sub[target_col].to_numpy(dtype=np.float64),
            deg=1,
        )
        county_slopes[str(geoid)] = float(slope)
        county_intercepts[str(geoid)] = float(intercept)
        fit_n[str(geoid)] = int(len(sub))

    if len(county_slopes) == 0:
        raise ValueError(
            f"No counties had ≥{min_county_years} training years; cannot fit any "
            f"per-county trend. Check the train pool."
        )

    # State fallback = median of per-county slopes/intercepts within the state.
    # Median not mean: robust to one or two implausible per-county fits, and
    # represents a "typical county trend in this state" more naturally.
    state_fallback_slopes: Dict[str, float] = {}
    state_fallback_intercepts: Dict[str, float] = {}
    by_state = pd.DataFrame(
        {
            "geoid": list(county_slopes.keys()),
            "slope": list(county_slopes.values()),
            "intercept": list(county_intercepts.values()),
        }
    )
    by_state["state_alpha"] = by_state["geoid"].map(geoid_to_state)
    for state, grp in by_state.groupby("state_alpha"):
        state_fallback_slopes[str(state)] = float(grp["slope"].median())
        state_fallback_intercepts[str(state)] = float(grp["intercept"].median())

    return CountyTrend(
        county_slopes=county_slopes,
        county_intercepts=county_intercepts,
        state_fallback_slopes=state_fallback_slopes,
        state_fallback_intercepts=state_fallback_intercepts,
        geoid_to_state=geoid_to_state,
        fit_years=(fit_min, fit_max),
        fit_n=fit_n,
    )
