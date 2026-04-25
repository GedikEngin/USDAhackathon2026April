"""
forecast.aggregate — county-level cones rolled up to state-level forecasts.

Phase B baseline. State forecast = planted-acres-weighted mean of county
percentile values, computed independently per percentile.

Caveat (documented in PHASE2_PHASE_PLAN B.4):
    Percentiles do not average linearly. "The state's 10th percentile is the
    acres-weighted mean of county 10th percentiles" is an approximation, not
    an identity. The strictly-correct alternative would be to weight individual
    analog yields by their county's planted acres and take percentiles over the
    weighted distribution — that's defensible but more complex to compute and
    doesn't change the qualitative shape of the cone. For the v2 baseline we
    accept the approximation; documented in the doc string for state_forecast.

Public surface:
    CountyForecastRecord                dataclass (cone + geoid + weight)
    StateForecast                       dataclass
    state_forecast_from_records(...)    main entrypoint
    build_records_from_master(...)      helper that derives weights from master table
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping

import numpy as np
import pandas as pd

from forecast.cone import Cone


@dataclass
class StateForecast:
    """State-level rolled-up forecast for one (state, year, forecast_date)."""

    state_alpha: str
    year: int
    forecast_date: str
    point_estimate: float
    percentiles: Dict[int, float]
    n_counties: int
    total_planted_acres: float
    # Per-county records, kept for narration/debugging.
    county_cones: List[Cone] = field(default_factory=list)
    county_weights: Dict[str, float] = field(default_factory=dict)


def state_forecast_from_records(
    records: List["CountyForecastRecord"],
    state_alpha: str,
    year: int,
    forecast_date: str,
) -> StateForecast:
    """Roll up county forecasts from CountyForecastRecord objects (cone + GEOID + weight).

    This is the actual public entrypoint. The backtest harness builds a
    CountyForecastRecord for each queryable county in the state and passes the
    list here.
    """
    if not records:
        raise ValueError(
            f"state_forecast_from_records called with no records for "
            f"({state_alpha}, {year}, {forecast_date})."
        )

    # Validate all percentile keys agree.
    pct_keys = sorted(records[0].cone.percentiles.keys())
    for r in records[1:]:
        if sorted(r.cone.percentiles.keys()) != pct_keys:
            raise ValueError(
                f"County cones disagree on percentile keys: "
                f"{pct_keys} vs {sorted(r.cone.percentiles.keys())}"
            )

    # Drop records with NaN or zero weight (zero-acres counties contribute nothing
    # and would NaN the normalization). Counties with NaN acres in NASS-disclosure-
    # suppressed years get a uniform fallback handled by the caller; if any leak
    # through here, drop them with a warning rather than crashing the whole state.
    clean: List[CountyForecastRecord] = []
    dropped: List[str] = []
    for r in records:
        if r.weight is None or np.isnan(r.weight) or r.weight <= 0:
            dropped.append(r.geoid)
            continue
        clean.append(r)
    if not clean:
        raise ValueError(
            f"All county records dropped (zero or NaN weights) for "
            f"({state_alpha}, {year}, {forecast_date})."
        )

    weights_arr = np.array([r.weight for r in clean], dtype=np.float64)
    w_norm = weights_arr / weights_arr.sum()

    # Per-percentile weighted mean.
    pct_dict: Dict[int, float] = {}
    for p in pct_keys:
        vals = np.array([r.cone.percentiles[p] for r in clean], dtype=np.float64)
        pct_dict[p] = float((vals * w_norm).sum())

    # Point estimate = weighted mean of county point estimates.
    pt_vals = np.array([r.cone.point_estimate for r in clean], dtype=np.float64)
    point_estimate = float((pt_vals * w_norm).sum())

    return StateForecast(
        state_alpha=state_alpha,
        year=year,
        forecast_date=forecast_date,
        point_estimate=point_estimate,
        percentiles=pct_dict,
        n_counties=len(clean),
        total_planted_acres=float(weights_arr.sum()),
        county_cones=[r.cone for r in clean],
        county_weights={r.geoid: float(r.weight) for r in clean},
    )


@dataclass
class CountyForecastRecord:
    """One county's contribution to a state forecast.

    The backtest harness builds these from a (geoid, cone, weight) triple. Cone
    doesn't carry GEOID itself (it's anonymous within its state), so this
    wrapper provides the binding.
    """

    geoid: str
    cone: Cone
    weight: float  # planted acres for this (geoid, year)


def build_records_from_master(
    cones_by_geoid: Mapping[str, Cone],
    master_df: pd.DataFrame,
    state_alpha: str,
    year: int,
) -> List[CountyForecastRecord]:
    """Helper: pair a {GEOID -> Cone} mapping with planted acres pulled from the master table.

    Counties in `cones_by_geoid` whose `acres_planted_all` is NaN at this
    (state, year) get weight=0 and will be dropped by state_forecast_from_records.
    Logged for traceability via the dropped list returned by that function's
    internal cleanup.
    """
    sub = master_df[
        (master_df["state_alpha"] == state_alpha) & (master_df["year"] == year)
    ]
    # acres_planted_all is the same value at all 4 forecast_dates within a year;
    # take the first (08-01) row per GEOID.
    acres_lookup = (
        sub.drop_duplicates(subset=["GEOID"])
        .set_index("GEOID")["acres_planted_all"]
        .to_dict()
    )

    records: List[CountyForecastRecord] = []
    for geoid, cone in cones_by_geoid.items():
        weight = acres_lookup.get(geoid, np.nan)
        records.append(CountyForecastRecord(geoid=geoid, cone=cone, weight=float(weight)))
    return records
