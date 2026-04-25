"""
backend/forecast_routes.py — FastAPI router for the v2 yield-forecast endpoints.

Three endpoints:
  GET  /forecast/states                  index of states + available years
  GET  /forecast/{state}                 full state forecast (4 dates or 1)
  POST /forecast/narrate                 Phase F stub (returns placeholder)

All inference math reuses the validated forecast/ stack — same code paths
the smoke test (scripts/smoke_forecast_2025.py) exercises end-to-end.
Route handlers are thin HTTP wrappers around already-validated functions.

Lifespan contract (set in backend/main.py):
  app.state.master_df            full 2005-2025 master parquet, defensive-padded
  app.state.train_df             train pool 2005-2022, post min-history filter
  app.state.standardizer         fit on train_df
  app.state.trend                CountyTrend fit on train_df
  app.state.analog_index         AnalogIndex fit on train_df + std + trend
  app.state.bundle               RegressorBundle from FORECAST_BUNDLE_DIR
  app.state.model_version        from {FORECAST_BUNDLE_DIR}/VERSION.txt
  app.state.history_lookup       precomputed 5/10-year means by (state, year_query)
  app.state.county_name_lookup   {GEOID -> county_name} (for analog narration)

Hot-swappable model: change FORECAST_BUNDLE_DIR env var, restart, done.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request

# Make `forecast` importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.forecast_models import (
    AnalogAnchor,
    AnalogRecord,
    ConeBand,
    DateForecast,
    DriverRecord,
    HistoryStats,
    NarrateRequest,
    NarrateResponse,
    StateForecastResponse,
    StateInfo,
    StatesIndex,
)
from forecast.aggregate import (
    StateForecast as StateForecastDC,                    # dataclass; alias to avoid Pydantic clash
    build_records_from_master,
    state_forecast_from_records,
)
from forecast.cone import Cone, build_cone
from forecast.explain import attribution_table
from forecast.features import EMBEDDING_COLS, VALID_FORECAST_DATES


log = logging.getLogger("forecast_routes")
router = APIRouter(prefix="/forecast", tags=["forecast"])


def _require_forecast_loaded(app_state) -> None:
    """Raise 503 if forecast artifacts didn't load at lifespan time. Each
    route calls this before touching app.state.master_df / .bundle / etc."""
    if not getattr(app_state, "forecast_loaded", False):
        raise HTTPException(
            status_code=503,
            detail=(
                "Forecast endpoints unavailable: forecast artifacts didn't load "
                "at startup. Check that scripts/training_master_v2.parquet and "
                "models/forecast/ exist (override via FORECAST_MASTER_PATH and "
                "FORECAST_BUNDLE_DIR env vars). See backend logs for details."
            ),
        )


# ---------- constants ------------------------------------------------------

STATE_NAMES = {
    "CO": "Colorado",
    "IA": "Iowa",
    "MO": "Missouri",
    "NE": "Nebraska",
    "WI": "Wisconsin",
}
ALL_STATES = ("CO", "IA", "MO", "NE", "WI")

# Cone retrieval config (locked in PHASE2_DECISIONS_LOG Phase B).
ANALOG_K = 5
ANALOG_POOL = "same_geoid"
PERCENTILES = (10, 50, 90)


# ---------- compute primitives — used by both /states and /{state} --------


def _state_truth(master_df: pd.DataFrame, state: str, year: int) -> Optional[float]:
    """Acres-weighted state yield from NASS truth.

    Mirrors backtest_phase_c.state_truth_from_master / backtest_baseline.
    Returns None when the master has no truth (2025 forecast year, or thin
    coverage). Truth is the same value at all 4 forecast_dates within a
    year (NASS reports annually), so we dedupe to the EOS row for safety.
    """
    sub = master_df[
        (master_df["state_alpha"] == state)
        & (master_df["year"] == year)
        & (master_df["forecast_date"] == "EOS")
    ]
    if len(sub) == 0:
        return None
    yields = sub["yield_target"].to_numpy(dtype=np.float64)
    acres = sub["acres_planted_all"].to_numpy(dtype=np.float64)
    valid = ~np.isnan(yields) & ~np.isnan(acres) & (acres > 0)
    if not valid.any():
        return None
    return float((yields[valid] * acres[valid]).sum() / acres[valid].sum())


def _county_has_complete_embedding(row: pd.Series, forecast_date: str) -> bool:
    """True iff every embedding column for this forecast_date is non-null
    on this row. Used to decide whether a county can contribute to the cone."""
    cols = EMBEDDING_COLS[forecast_date]
    return not any(pd.isna(row.get(c)) for c in cols)


def _predict_state_point(
    bundle, query_df: pd.DataFrame, state: str, year: int, forecast_date: str
) -> Tuple[Optional[float], int]:
    """Acres-weighted state-level point estimate. Mirrors the smoke test
    exactly. Returns (point, n_counties). XGBoost handles NaN feature cells
    natively; counties only drop out if acres_planted_all is NaN/zero."""
    sub = query_df[
        (query_df["state_alpha"] == state)
        & (query_df["year"] == year)
        & (query_df["forecast_date"] == forecast_date)
    ]
    if len(sub) == 0:
        return None, 0
    preds = bundle.regressors[forecast_date].predict(sub)
    # XGBoost may return shape (0,) when every row is structurally bad. Guard.
    if len(preds) != len(sub):
        return None, 0
    acres = sub["acres_planted_all"].to_numpy(dtype=np.float64)
    valid = ~np.isnan(preds) & ~np.isnan(acres) & (acres > 0)
    if not valid.any():
        return None, 0
    point = float((preds[valid] * acres[valid]).sum() / acres[valid].sum())
    return point, int(valid.sum())


def _build_state_cone(
    analog_index, trend, query_rows: pd.DataFrame,
    master_df: pd.DataFrame, state: str, year: int, forecast_date: str,
) -> Tuple[Optional[StateForecastDC], int]:
    """Per-county cones via analog retrieval, then state aggregation.
    Returns (StateForecast | None, n_counties_contributing).

    Returns (None, 0) when no county has a complete embedding (NDVI-missing
    2025 case; cone_status='unavailable_pending_ndvi' surfaces upstream)."""
    cones_by_geoid: Dict[str, Cone] = {}
    for _, row in query_rows.iterrows():
        if not _county_has_complete_embedding(row, forecast_date):
            continue
        try:
            analogs = analog_index.find(
                geoid=str(row["GEOID"]),
                year=int(row["year"]),
                forecast_date=str(row["forecast_date"]),
                query_features=row,
                k=ANALOG_K,
                pool=ANALOG_POOL,
            )
        except Exception as e:
            log.warning(
                "analog_index.find failed for (%s, %s, %s): %s",
                row["GEOID"], row["year"], forecast_date, e,
            )
            continue
        if not analogs:
            continue
        cone = build_cone(
            analogs=analogs, trend=trend,
            query_geoid=str(row["GEOID"]),
            query_state=str(row["state_alpha"]),
            query_year=int(row["year"]),
            query_forecast_date=str(row["forecast_date"]),
            percentiles=PERCENTILES,
        )
        cones_by_geoid[str(row["GEOID"])] = cone

    if not cones_by_geoid:
        return None, 0

    records = build_records_from_master(cones_by_geoid, master_df, state, year)
    try:
        sf = state_forecast_from_records(records, state, year, forecast_date)
    except ValueError:
        # All county weights ended up NaN/zero. Treat as "no cone".
        return None, 0
    return sf, sf.n_counties


def _largest_acres_anchor_county(
    query_rows: pd.DataFrame, master_df: pd.DataFrame, year: int, forecast_date: str
) -> Optional[pd.Series]:
    """Pick the county with the largest acres_planted_all that ALSO has a
    complete embedding for forecast_date. The cone's analog list comes from
    THIS county (not the whole state), so the anchor must be queryable.

    Returns the row from query_rows or None if no county qualifies."""
    candidates = query_rows[
        (query_rows["forecast_date"] == forecast_date)
        & query_rows["acres_planted_all"].notna()
    ].copy()
    if len(candidates) == 0:
        return None
    # Filter to embedding-complete.
    keep = candidates.apply(
        lambda r: _county_has_complete_embedding(r, forecast_date), axis=1
    )
    candidates = candidates[keep]
    if len(candidates) == 0:
        return None
    return candidates.sort_values("acres_planted_all", ascending=False).iloc[0]


def _analogs_for_anchor(
    analog_index, trend, anchor_row: pd.Series, master_df: pd.DataFrame,
    county_name_lookup: Dict[str, str],
) -> List[AnalogRecord]:
    """K=5 same-GEOID analogs for the anchor county, formatted as AnalogRecords.

    The trend's retrend at analog (geoid, year) gives the analog's expected
    yield given its own trend; observed - retrend = detrended_yield. We
    return both so the agent can describe relative-to-trend deviations."""
    try:
        analogs = analog_index.find(
            geoid=str(anchor_row["GEOID"]),
            year=int(anchor_row["year"]),
            forecast_date=str(anchor_row["forecast_date"]),
            query_features=anchor_row,
            k=ANALOG_K,
            pool=ANALOG_POOL,
        )
    except Exception as e:
        log.warning("anchor analog find failed: %s", e)
        return []
    out: List[AnalogRecord] = []
    for a in analogs:
        # county_name comes from a master_df lookup; analog records carry GEOID.
        cn = county_name_lookup.get(a.geoid, "?")
        out.append(AnalogRecord(
            geoid=a.geoid,
            county_name=cn,
            state_alpha=a.state_alpha,
            year=a.year,
            distance=a.distance,
            observed_yield_bu_acre=a.observed_yield,
            detrended_yield_bu_acre=a.detrended_yield,
        ))
    return out


_STRUCTURAL_FEATURES = {"year", "acres_planted_all", "state"}
# 'state' here is the merged label produced below from the state_is_* one-hots.


def _state_top_drivers(
    bundle, query_rows: pd.DataFrame, forecast_date: str, k: int = 3,
) -> Tuple[List[DriverRecord], List[DriverRecord]]:
    """Acres-weighted state-level SHAP via attribution_table + groupby.

    Returns (biophysical_top_k, structural_drivers).

    Biophysical = weather, drought, soil, NDVI, management ratios, etc. —
    the actionable agronomic signals. Top-K by |shap_bu_acre|.

    Structural = the model's structural priors (year=long-run trend,
    state=regional baseline, acres_planted_all=size weight). Always
    surfaced separately so the narrator can describe them as "model
    context" without crowding out the season-driven signals.

    For each county in query_rows at this forecast_date, compute SHAP values
    via Booster.predict(pred_contribs=True). Acres-weight each row's SHAP
    by acres_planted_all. Group by feature, take signed weighted mean."""
    sub = query_rows[query_rows["forecast_date"] == forecast_date]
    if len(sub) == 0:
        return [], []

    regressor = bundle.regressors[forecast_date]
    table = attribution_table(regressor, sub, include_index_cols=True)
    # table has columns: GEOID, year, forecast_date, [state_alpha], feature,
    # feature_value, shap_value, prediction, base_value

    # Join acres back via GEOID + year (within the forecast_date already-filtered slice).
    acres_lookup = (
        sub.set_index("GEOID")["acres_planted_all"].to_dict()
    )
    table["acres"] = table["GEOID"].map(acres_lookup)
    valid = table["acres"].notna() & (table["acres"] > 0)
    table = table[valid].copy()
    if len(table) == 0:
        return [], []

    # Acres-weighted mean of shap_value and feature_value per feature.
    table["w_shap"] = table["shap_value"] * table["acres"]
    table["w_fval"] = table["feature_value"] * table["acres"]
    grouped = table.groupby("feature").agg(
        w_shap_sum=("w_shap", "sum"),
        w_fval_sum=("w_fval", "sum"),
        acres_sum=("acres", "sum"),
    ).reset_index()
    grouped["shap_bu_acre"] = grouped["w_shap_sum"] / grouped["acres_sum"]
    grouped["feature_value_state_mean"] = grouped["w_fval_sum"] / grouped["acres_sum"]

    # Merge state_is_* one-hots into a single 'state' driver. Only one is on
    # for any given row; acres-weighted SHAP collapses to the contribution
    # of being in this state.
    state_mask = grouped["feature"].str.startswith("state_is_")
    if state_mask.any():
        merged_shap = float(grouped.loc[state_mask, "shap_bu_acre"].sum())
        grouped = grouped.loc[~state_mask].copy()
        grouped = pd.concat([
            grouped,
            pd.DataFrame([{
                "feature": "state",
                "shap_bu_acre": merged_shap,
                "feature_value_state_mean": 1.0,
            }]),
        ], ignore_index=True)

    def _to_record(row) -> DriverRecord:
        s = float(row["shap_bu_acre"])
        return DriverRecord(
            feature=str(row["feature"]),
            shap_bu_acre=s,
            feature_value_state_mean=float(row["feature_value_state_mean"]),
            direction="+" if s > 0 else ("-" if s < 0 else "0"),
        )

    # Split.
    bio_mask = ~grouped["feature"].isin(_STRUCTURAL_FEATURES)
    bio_df = grouped[bio_mask].copy()
    struct_df = grouped[~bio_mask].copy()

    # Top-K biophysical by absolute SHAP.
    bio_df["abs_shap"] = bio_df["shap_bu_acre"].abs()
    bio_top = bio_df.sort_values("abs_shap", ascending=False).head(k)
    biophysical = [_to_record(r) for _, r in bio_top.iterrows()]

    # All structural drivers (sorted by absolute SHAP for stable ordering).
    struct_df["abs_shap"] = struct_df["shap_bu_acre"].abs()
    struct_sorted = struct_df.sort_values("abs_shap", ascending=False)
    structural = [_to_record(r) for _, r in struct_sorted.iterrows()]

    return biophysical, structural


def _build_date_forecast(
    *, app_state, master_df: pd.DataFrame, state: str, year: int, forecast_date: str,
) -> DateForecast:
    """Compute the full DateForecast for one (state, year, forecast_date).
    All inference math is here; everything else in the routes is plumbing."""
    # Slice query rows. Master holds query rows for ALL years (2005-2025);
    # for forecast year 2025 these are the rows the regressor scores.
    query_rows = master_df[
        (master_df["state_alpha"] == state) & (master_df["year"] == year)
    ].copy()

    if len(query_rows) == 0:
        # Year not in master — surface as no-op DateForecast.
        return DateForecast(
            forecast_date=forecast_date,
            point_estimate_bu_acre=None,
            cone=None,
            n_counties_regressor=0,
            n_counties_cone=0,
            cone_status="unavailable_no_analogs",
            analog_years=[],
            analog_anchor=None,
            top_drivers=[],
        )

    # Restrict to the forecast_date slice for downstream regressor + cone.
    fd_rows = query_rows[query_rows["forecast_date"] == forecast_date]

    # ---- regressor point ----
    point, n_reg = _predict_state_point(
        app_state.bundle, master_df, state, year, forecast_date
    )

    # ---- cone ----
    sf, n_cone = _build_state_cone(
        app_state.analog_index, app_state.trend, fd_rows,
        master_df, state, year, forecast_date,
    )
    if sf is not None:
        cone = ConeBand(
            p10=sf.percentiles[10],
            p50=sf.percentiles[50],
            p90=sf.percentiles[90],
            width_80=sf.percentiles[90] - sf.percentiles[10],
        )
        cone_status = "ok"
    else:
        cone = None
        # Distinguish "NDVI missing" (the 2025 case — embedding incomplete on
        # every county) from "no analogs" (would imply data corruption).
        # Heuristic: if at least one county would be queryable but the analog
        # search returned nothing, that's no_analogs. If every county has an
        # incomplete embedding, that's pending_ndvi.
        any_complete = fd_rows.apply(
            lambda r: _county_has_complete_embedding(r, forecast_date), axis=1
        ).any() if len(fd_rows) > 0 else False
        cone_status = (
            "unavailable_no_analogs" if any_complete else "unavailable_pending_ndvi"
        )

    # ---- analogs (only if cone is available) ----
    analogs: List[AnalogRecord] = []
    anchor: Optional[AnalogAnchor] = None
    if cone is not None:
        anchor_row = _largest_acres_anchor_county(
            fd_rows, master_df, year, forecast_date
        )
        if anchor_row is not None:
            anchor = AnalogAnchor(
                geoid=str(anchor_row["GEOID"]),
                county_name=str(anchor_row.get("county_name", "?")),
                state_alpha=str(anchor_row["state_alpha"]),
                acres_planted=float(anchor_row["acres_planted_all"]),
            )
            analogs = _analogs_for_anchor(
                app_state.analog_index, app_state.trend,
                anchor_row, master_df, app_state.county_name_lookup,
            )

    # ---- drivers (always populated when point is available) ----
    drivers: List[DriverRecord] = []
    structural: List[DriverRecord] = []
    if point is not None and n_reg > 0:
        drivers, structural = _state_top_drivers(
            app_state.bundle, fd_rows, forecast_date, k=3
        )

    return DateForecast(
        forecast_date=forecast_date,
        point_estimate_bu_acre=point,
        cone=cone,
        n_counties_regressor=n_reg,
        n_counties_cone=n_cone,
        cone_status=cone_status,
        analog_years=analogs,
        analog_anchor=anchor,
        top_drivers=drivers,
        structural_drivers=structural,
    )


# ---------- helpers for /forecast/states -----------------------------------

def _available_years_for_state(master_df: pd.DataFrame, state: str) -> List[int]:
    """Years for which this state has at least one queryable county at
    forecast_date='EOS'. EOS is the most-permissive date (all features
    populated by definition for years with full data); a year that has
    queryable EOS rows is queryable at all 4 dates."""
    sub = master_df[
        (master_df["state_alpha"] == state) & (master_df["forecast_date"] == "EOS")
    ]
    if len(sub) == 0:
        return []
    # A year is available if at least one row has acres_planted_all.
    sub = sub[sub["acres_planted_all"].notna()]
    return sorted(int(y) for y in sub["year"].unique())


# ---------- routes ---------------------------------------------------------


@router.get("/states", response_model=StatesIndex)
def list_states(request: Request) -> StatesIndex:
    """Index of states + per-state available years. Cheap; the lookups all
    hit cached app.state."""
    _require_forecast_loaded(request.app.state)
    app_state = request.app.state
    master = app_state.master_df

    states_out: List[StateInfo] = []
    for alpha in ALL_STATES:
        years = _available_years_for_state(master, alpha)
        n_counties = int(
            master[(master["state_alpha"] == alpha) & (master["forecast_date"] == "EOS")]
            ["GEOID"].nunique()
        )
        states_out.append(StateInfo(
            alpha=alpha,
            name=STATE_NAMES[alpha],
            n_counties=n_counties,
            available_years=years,
        ))

    # Default year = max queryable year across all states, with bias toward
    # the forecast year if it's in master.
    all_years = sorted({y for s in states_out for y in s.available_years})
    if not all_years:
        raise HTTPException(500, "Master parquet has no queryable years.")
    default_year = all_years[-1]                    # newest

    return StatesIndex(
        states=states_out,
        forecast_dates=list(VALID_FORECAST_DATES),
        default_year=default_year,
        default_date="EOS",
        model_version=app_state.model_version,
    )


@router.get("/{state}", response_model=StateForecastResponse)
def get_state_forecast(
    state: str,
    request: Request,
    year: int = Query(..., description="Forecast year (e.g. 2025)"),
    date: Optional[str] = Query(
        None,
        description="Optional forecast_date: '08-01'|'09-01'|'10-01'|'EOS'. "
                    "Omitted = return all 4 in by_date.",
    ),
) -> StateForecastResponse:
    """Full state forecast for one year. Either single date or all 4."""
    _require_forecast_loaded(request.app.state)
    app_state = request.app.state
    master = app_state.master_df

    # Validate state.
    if state not in ALL_STATES:
        raise HTTPException(400, f"Unknown state: {state!r}. Expected one of {ALL_STATES}.")

    # Validate year.
    available = _available_years_for_state(master, state)
    if year not in available:
        raise HTTPException(
            404,
            f"Year {year} not available for state {state}. "
            f"Available: {available}",
        )

    # Validate date if set.
    if date is not None and date not in VALID_FORECAST_DATES:
        raise HTTPException(
            400,
            f"Unknown forecast_date: {date!r}. Expected one of {VALID_FORECAST_DATES}.",
        )

    # Truth lookup (None for forecast year, real for val/holdout/historical).
    truth = _state_truth(master, state, year)

    # History stats from the precomputed lookup.
    h5, h10 = app_state.history_lookup.get((state, year), (None, None))
    history = HistoryStats(mean_5yr_bu_acre=h5, mean_10yr_bu_acre=h10)

    # Compute one or four DateForecasts.
    if date is not None:
        df = _build_date_forecast(
            app_state=app_state, master_df=master,
            state=state, year=year, forecast_date=date,
        )
        return StateForecastResponse(
            state=state,
            state_name=STATE_NAMES[state],
            year=year,
            model_version=app_state.model_version,
            forecast_date=date,
            forecast=df,
            by_date=None,
            truth_state_yield_bu_acre=truth,
            history=history,
        )

    by_date: Dict[str, DateForecast] = {}
    for fd in VALID_FORECAST_DATES:
        by_date[fd] = _build_date_forecast(
            app_state=app_state, master_df=master,
            state=state, year=year, forecast_date=fd,
        )
    return StateForecastResponse(
        state=state,
        state_name=STATE_NAMES[state],
        year=year,
        model_version=app_state.model_version,
        forecast_date=None,
        forecast=None,
        by_date=by_date,
        truth_state_yield_bu_acre=truth,
        history=history,
    )


@router.post("/narrate", response_model=NarrateResponse)
def narrate(req: NarrateRequest, request: Request) -> NarrateResponse:
    """Narrate a single (state, year, forecast_date) forecast via Claude
    Haiku 4.5. Stateless — we re-fetch the structured forecast on each
    call rather than trusting the frontend to echo it back, since the
    structured form is cheap to recompute and trusting echoed payloads is
    a footgun.
    """
    _require_forecast_loaded(request.app.state)
    app_state = request.app.state

    # Validate request shape against the same constraints the GET route uses.
    if req.state not in ALL_STATES:
        raise HTTPException(400, f"Unknown state: {req.state!r}.")
    if req.forecast_date not in VALID_FORECAST_DATES:
        raise HTTPException(400, f"Unknown forecast_date: {req.forecast_date!r}.")

    master = app_state.master_df
    available = _available_years_for_state(master, req.state)
    if req.year not in available:
        raise HTTPException(
            404,
            f"Year {req.year} not available for state {req.state}. "
            f"Available: {available}",
        )

    # The narrator needs the agent client. v1 lifespan loads it from
    # ANTHROPIC_API_KEY; if that wasn't set, the agent is None and we
    # gracefully surface "narrator unavailable" rather than 500.
    agent = getattr(app_state, "agent", None)
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Narrator unavailable: ANTHROPIC_API_KEY is not set on this "
                "server. The structured forecast at GET /forecast/{state} "
                "is unaffected."
            ),
        )

    # Re-build the DateForecast for this exact (state, year, forecast_date).
    forecast = _build_date_forecast(
        app_state=app_state, master_df=master,
        state=req.state, year=req.year, forecast_date=req.forecast_date,
    )

    # Build the StateForecastResponse skeleton needed by the narrator
    # (truth + history + state name; we don't need by_date for one-shot).
    truth = _state_truth(master, req.state, req.year)
    h5, h10 = app_state.history_lookup.get((req.state, req.year), (None, None))
    state_response = StateForecastResponse(
        state=req.state,
        state_name=STATE_NAMES[req.state],
        year=req.year,
        model_version=app_state.model_version,
        forecast_date=req.forecast_date,
        forecast=forecast,
        by_date=None,
        truth_state_yield_bu_acre=truth,
        history=HistoryStats(mean_5yr_bu_acre=h5, mean_10yr_bu_acre=h10),
    )

    # Local import keeps the routes file independent of the narrator at
    # import time (so a missing anthropic package doesn't break /forecast/states).
    from backend.forecast_narrator import narrate_forecast

    return narrate_forecast(
        client=agent.client,
        state_response=state_response,
        forecast=forecast,
    )


# ---------- lifespan helpers (called from backend/main.py) -----------------


def build_history_lookup(master_df: pd.DataFrame) -> Dict[Tuple[str, int], Tuple[Optional[float], Optional[float]]]:
    """Pre-compute (5yr_mean, 10yr_mean) for every (state, year) we may serve.

    Mirrors forecast.baseline.state_baseline conventions: acres-weighted
    state mean, lookback excludes the query year and any holdout years.

    Returns {(state, year_query): (mean_5yr_or_None, mean_10yr_or_None)}.
    """
    from forecast.baseline import state_baseline
    out: Dict[Tuple[str, int], Tuple[Optional[float], Optional[float]]] = {}
    for state in ALL_STATES:
        years = sorted(int(y) for y in master_df["year"].unique())
        for y in years:
            try:
                m5, _ = state_baseline(
                    master_df, state, y, forecast_date="EOS",
                    holdout_years=(),                # empty: no special exclusion at API time
                    min_years=3, lookback=5,
                )
            except Exception:
                m5 = float("nan")
            try:
                m10, _ = state_baseline(
                    master_df, state, y, forecast_date="EOS",
                    holdout_years=(),
                    min_years=3, lookback=10,
                )
            except Exception:
                m10 = float("nan")
            out[(state, y)] = (
                None if np.isnan(m5) else float(m5),
                None if np.isnan(m10) else float(m10),
            )
    return out


def build_county_name_lookup(master_df: pd.DataFrame) -> Dict[str, str]:
    """{GEOID -> county_name} for analog narration."""
    return (
        master_df.drop_duplicates(subset=["GEOID"])
        .set_index("GEOID")["county_name"]
        .astype(str)
        .to_dict()
    )
