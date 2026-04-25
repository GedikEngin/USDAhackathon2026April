"""
backend/forecast_models.py — Pydantic schemas for the /forecast/* endpoints.

The JSON contract was locked before any code was written; these schemas are
the source of truth for what the backend serves and what the frontend
expects.

Three response shapes:

  StatesIndex            -> GET /forecast/states
  StateForecastResponse  -> GET /forecast/{state}
                            (with or without ?date=)
  NarrateResponse        -> POST /forecast/narrate

Plus the substructures (DateForecast, ConeBand, AnalogRecord, AnalogAnchor,
DriverRecord, HistoryStats).

Conventions:
  - All bu/acre values are floats. None means "not available" (e.g. 2025
    has no truth, NDVI-missing counties produce no cone).
  - forecast_date is the canonical "08-01"/"09-01"/"10-01"/"EOS" string.
    The frontend formats display ("Aug 1, 2025") locally.
  - All percentile keys are int (10, 50, 90), not str ("p10").
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------- shared sub-shapes ----------------------------------------------


class ConeBand(BaseModel):
    """Cone of uncertainty — analog-derived percentile band, state-level."""
    p10: float
    p50: float
    p90: float
    width_80: float                                 # p90 - p10, precomputed for chart axes


class AnalogRecord(BaseModel):
    """One analog (geoid, year) returned for the anchor county."""
    geoid: str
    county_name: str
    state_alpha: str
    year: int
    distance: float                                 # standardized-embedding L2
    observed_yield_bu_acre: float
    detrended_yield_bu_acre: float                  # observed - county/state trend at analog year


class AnalogAnchor(BaseModel):
    """Documents WHICH county the analog list came from. The 5 analogs are
    one county's analogs, not a state-aggregated set; this surfaces the
    choice so the frontend and the agent can both narrate it honestly."""
    geoid: str
    county_name: str
    state_alpha: str
    acres_planted: float
    rationale: Literal["largest-acres county in state"] = "largest-acres county in state"


class DriverRecord(BaseModel):
    """One feature's contribution to the state-level prediction.

    SHAP values from per-county XGBoost predictions, then acres-weighted
    across counties. Top-K returned."""
    feature: str
    shap_bu_acre: float                             # signed; + raises prediction, - lowers
    feature_value_state_mean: float                 # acres-weighted mean of the feature
    direction: Literal["+", "-", "0"]


class HistoryStats(BaseModel):
    """5-year and 10-year acres-weighted state means, computed at lifespan
    time and reused for every request. Excludes holdout years per Phase B
    convention (state_baseline default holdout_years=(2024,))."""
    mean_5yr_bu_acre: Optional[float]
    mean_10yr_bu_acre: Optional[float]


class DateForecast(BaseModel):
    """One (state, year, forecast_date) bundle. The unit of work the
    Phase F narration agent will consume."""
    forecast_date: str                              # "08-01"|"09-01"|"10-01"|"EOS"
    point_estimate_bu_acre: Optional[float]
    cone: Optional[ConeBand]
    n_counties_regressor: int                       # how many counties contributed to the point
    n_counties_cone: int                            # how many to the cone (may be 0 if NDVI missing)
    cone_status: Literal[
        "ok",
        "unavailable_pending_ndvi",                 # the 2025 case
        "unavailable_no_analogs",                   # would only happen with bad data
    ]
    analog_years: List[AnalogRecord]                # may be empty if cone_status != ok
    analog_anchor: Optional[AnalogAnchor]           # None if cone_status != ok
    top_drivers: List[DriverRecord]                 # top biophysical drivers (weather/drought/soil/etc.)
    structural_drivers: List[DriverRecord] = Field(default_factory=list)
    # ^^ "year" (long-run trend), "state" (regional baseline), and
    # "acres_planted_all" (size weight). These are real model contributions
    # but not biophysical/actionable, so we surface them separately so the
    # narrator can describe them as "model context" without crowding out
    # the actual season-driven signals.


# ---------- /forecast/states -----------------------------------------------


class StateInfo(BaseModel):
    alpha: str                                      # "IA"
    name: str                                       # "Iowa"
    n_counties: int
    available_years: List[int]                      # years queryable in the master parquet


class StatesIndex(BaseModel):
    states: List[StateInfo]
    forecast_dates: List[str]                       # ["08-01","09-01","10-01","EOS"]
    default_year: int
    default_date: str
    model_version: str


# ---------- /forecast/{state} ----------------------------------------------


class StateForecastResponse(BaseModel):
    """Full response for /forecast/{state}. When ?date= is set, only one
    DateForecast is populated and forecast_date is set at the top level.
    When omitted, by_date holds all four (08-01, 09-01, 10-01, EOS)."""
    state: str
    state_name: str                                 # "Iowa" (display)
    year: int
    model_version: str

    # When date= is set: forecast_date is populated and forecast holds the single result.
    # When date= is omitted: forecast_date is None and by_date has all 4.
    forecast_date: Optional[str] = None
    forecast: Optional[DateForecast] = None
    by_date: Optional[Dict[str, DateForecast]] = None

    # Truth (acres-weighted state yield from NASS) for years that have it.
    # 2023, 2024 -> float; 2025 -> None.
    truth_state_yield_bu_acre: Optional[float]
    history: HistoryStats


# ---------- /forecast/narrate (Phase F stub in E.1) -----------------------


class NarrateRequest(BaseModel):
    """Frontend echoes back a forecast object to narrate. Stateless: the
    backend re-fetches what it needs by (state, year, forecast_date) rather
    than trusting the echoed payload."""
    state: str
    year: int
    forecast_date: str                              # "08-01"|"09-01"|"10-01"|"EOS"


class NarrateResponse(BaseModel):
    """E.1 stub. Phase F replaces the body; frontend layout doesn't change."""
    narrative: str
    stub: bool
    tool_calls: List[dict] = Field(default_factory=list)
    model_version: str
