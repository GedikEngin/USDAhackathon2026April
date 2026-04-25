"""
forecast.cone — percentile cone construction from K analog records.

Phase B baseline. Given K Analog records (each carrying a detrended yield),
take percentiles in *detrended* space and retrend at the query's (state, year)
to produce a cone in raw bu/acre.

Why detrended-then-retrended:
    A 2008 analog and a 2023 query come from very different points on the
    genetic-gain trend. If we take percentiles directly over raw analog yields,
    the cone is anchored to whatever decade the analogs come from. By taking
    percentiles in detrended space (each analog measured against its own
    state-year trend) and then retrending at the query, we isolate the
    "weather/management deviation" the analog is supposed to communicate.

Public surface:
    Cone                                dataclass (p10/p50/p90 + median in raw space)
    build_cone(analogs, trend, query_state, query_year,
               percentiles=(10,50,90))  -> Cone
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import numpy as np

from forecast.analog import Analog
from forecast.detrend import CountyTrend


@dataclass
class Cone:
    """Cone-of-uncertainty for one query.

    All values are in raw bu/acre (post-retrend).

    Attributes
    ----------
    percentiles : dict[int, float]
        e.g. {10: 142.3, 50: 178.1, 90: 201.4}. Keys are integer percentiles.
    point_estimate : float
        Median analog yield in raw bu/acre. Phase B baseline point estimate;
        Phase C replaces with the trained regressor's prediction while keeping
        this cone untouched.
    n_analogs : int
        How many analogs the cone is built on (usually K, can be < K for the
        same_geoid sanity-baseline pool when a county has thin history).
    state_alpha : str
        Query state — useful for downstream aggregation/logging.
    year : int
        Query year — same.
    forecast_date : str
        Query forecast_date.
    detrended_percentiles : dict[int, float]
        Internal — percentiles in detrended space, before the retrend was
        applied. Kept for diagnostics (e.g. "is the cone in detrended space
        symmetric around 0?").
    """

    percentiles: Dict[int, float]
    point_estimate: float
    n_analogs: int
    state_alpha: str
    year: int
    forecast_date: str
    detrended_percentiles: Dict[int, float] = field(default_factory=dict)


def build_cone(
    analogs: List[Analog],
    trend: CountyTrend,
    query_geoid: str,
    query_state: str,
    query_year: int,
    query_forecast_date: str,
    percentiles: Iterable[int] = (10, 50, 90),
) -> Cone:
    """Build a percentile cone from a list of analogs.

    Parameters
    ----------
    analogs : list[Analog]
        Output of AnalogIndex.find(...). Empty list raises ValueError —
        caller must handle counties with no analogs upstream.
    trend : CountyTrend
        Used to retrend the detrended percentiles back to raw bu/acre at
        (query_geoid, query_year).
    query_geoid : str
        5-digit FIPS of the query county. Used for retrending — the cone
        is anchored to this county's own trend line, not a state aggregate.
    query_state : str
        State of the query, recorded on the returned Cone for the
        aggregation layer. Not used in math.
    query_year, query_forecast_date :
        Query identifiers, attached to the returned Cone for traceability.
    percentiles : iterable of int
        Percentiles to compute. Default (10, 50, 90).

    Returns
    -------
    Cone
    """
    if not analogs:
        raise ValueError(
            f"build_cone called with no analogs for query "
            f"({query_geoid}, {query_year}, {query_forecast_date}). "
            f"Caller must filter or short-circuit upstream."
        )

    pcts: Tuple[int, ...] = tuple(int(p) for p in percentiles)
    if any(p < 0 or p > 100 for p in pcts):
        raise ValueError(f"percentiles must be in [0, 100], got {pcts}")

    detrended_vals = np.array([a.detrended_yield for a in analogs], dtype=np.float64)
    if np.isnan(detrended_vals).any():
        raise ValueError(
            "Analog list contains NaN detrended_yield. This indicates a county "
            "missing from the CountyTrend fit, or an upstream merge bug."
        )

    # Percentiles in detrended space.
    detrended_pcts = np.percentile(detrended_vals, pcts)
    detrended_dict = {int(p): float(v) for p, v in zip(pcts, detrended_pcts)}

    # Retrend each percentile back to raw bu/acre at the query (geoid, year).
    retrended_dict = {
        p: float(trend.retrend(query_geoid, query_year, v))
        for p, v in detrended_dict.items()
    }

    # Median analog yield as the Phase B point estimate. Take the median in
    # detrended space and retrend; equivalent to retrend(median(detrended)).
    median_detrended = float(np.median(detrended_vals))
    point_estimate = float(trend.retrend(query_geoid, query_year, median_detrended))

    return Cone(
        percentiles=retrended_dict,
        point_estimate=point_estimate,
        n_analogs=len(analogs),
        state_alpha=query_state,
        year=query_year,
        forecast_date=query_forecast_date,
        detrended_percentiles=detrended_dict,
    )
