"""
backend/forecast_narrator.py — Claude Haiku 4.5 narration for the
v2 yield-forecast endpoints.

Phase F.1: one-shot prompting, no tool calls. The structured forecast is
small (~1 KB of JSON) so we just hand it to Claude and let the model write
the narrative. A future F.2 could add tools for deeper drill-downs (e.g.
"show me the worst-county breakdown") but F.1 is intentionally minimal —
single API call, predictable wall time, predictable token budget.

The route handler stays a thin wrapper around `narrate_forecast(...)`.

The system prompt is tuned to:
  - Audience: agronomists, extension agents, ag-finance analysts.
  - Length: 200-400 words, markdown.
  - Honesty: do not invent numbers; only use what's in the forecast JSON.
    Mention `cone_status: unavailable_pending_ndvi` explicitly when applicable.
  - Tone: confident where the data supports it, calibrated where the cone
    is wide. No hedging beyond what the cone width justifies.
  - Structure: trajectory -> drivers -> analogs -> caveats. Name analog
    years explicitly. Call out the top-3 drivers in plain language.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from backend.forecast_models import (
    DateForecast,
    NarrateResponse,
    StateForecastResponse,
)


log = logging.getLogger("forecast_narrator")


# ---------- system prompt -------------------------------------------------

SYSTEM_PROMPT = """\
You are an agronomy analyst writing concise narratives for a county-level corn-yield
forecast tool. Your audience is agronomists, extension agents, and ag-finance analysts
who want a clear read on what the model is saying and why.

You will receive a JSON object describing one (state, year, forecast_date) bundle. It
contains:
  - state, year, forecast_date          identifiers
  - point_estimate_bu_acre              acres-weighted state yield prediction (THE FORECAST)
  - cone                                {p10, p50, p90, width_80} or null if unavailable
  - cone_status                         "ok" | "unavailable_pending_ndvi" | "unavailable_no_analogs"
  - n_counties_regressor                counties contributing to the point estimate
  - n_counties_cone                     counties contributing to the cone
  - top_drivers                         top-3 BIOPHYSICAL features (weather/drought/soil/NDVI/management)
                                        ranked by acres-weighted SHAP. Each has `feature`,
                                        `shap_bu_acre` (signed), `feature_value_state_mean`,
                                        `direction` ("+", "-", "0").
  - structural_drivers                  the model's structural priors (long-run trend, regional
                                        baseline, planted-acres weighting). Same shape as top_drivers.
                                        These are real model contributions but not biophysical drivers.
  - analog_years                        up to 5 historical analog seasons for the
                                        anchor county (largest-acres in the state)
  - analog_anchor                       which county the analogs came from
  - history                             {mean_5yr_bu_acre, mean_10yr_bu_acre} for the state
  - truth_state_yield_bu_acre           NASS-reported state yield for that year, or null
                                        if it's a current forecast year

# Critical: the headline number
THE FORECAST IS `point_estimate_bu_acre`. The headline always reports the forecast.
`truth_state_yield_bu_acre` is the NASS-reported actual yield, used only for
retrospective comparison. NEVER write a headline like "Iowa's 2024 yield came in
at <truth>" — even if truth is available, the headline reports the forecast and
the comparison-to-truth goes in Caveats or Trajectory.

# What `feature_value_state_mean` means
For each driver, `feature_value_state_mean` is the ACRES-WEIGHTED COUNTY-MEAN of
that feature across the state's counties. It is NOT a state total.

For example: `acres_planted_all` `feature_value_state_mean` of 146,800 means
"the average county in this state, weighted by its own acreage, planted ~146,800
acres" — not "the state planted 146,800 acres total." Iowa's actual state total
is ~13 million acres.

For weather features the county-mean is intuitive (state-average GDD, state-average
VPD). For acreage and structural features, do NOT report the raw value as if it
were a state aggregate. Either skip the number or describe it in plain language
("a typical Iowa corn county").

# Feature glossary (do not confuse these)
  - gdd_cum_f50_c86: cumulative growing degree days, base 50°F cap 86°F (units: °F-days).
                     A typical Corn Belt season is ~2,400–2,800 °F-days by EOS.
  - edd_hours_gt86f: degree-hours above 86°F via single-sine integration (units: °F-hours).
                     Heat stress on grain fill; values >800 °F-hrs by EOS are notable.
  - edd_hours_gt90f: degree-hours above 90°F (units: °F-hours). Severe heat. Distinct
                     from GDD. NEVER call this "growing degree days." Call it
                     "extreme heat" or "degree-hours above 90°F."
  - vpd_kpa_veg/silk/grain: average vapor-pressure deficit during the named phase (kPa).
                     1.0 = mild, 1.5 = moderate, 2.0+ = stressful.
  - prcp_cum_mm: cumulative precipitation May 1 → cutoff (mm). 400 mm by EOS is around
                 average for a non-irrigated Corn Belt site.
  - dry_spell_max_days: longest run of <2 mm/day precip in the season window.
  - srad_total_*: cumulative solar radiation by phase (MJ/m²).
  - irrigated_share, harvest_ratio: management priors (proportions / ratios).
  - nccpi3corn, aws0_*, soc0_*: static soil productivity / water / carbon indices.

# Structural priors (in `structural_drivers`, not `top_drivers`)
  - year: a long-run yield trend the model has learned from 2005–2022 training data.
          Refer to as "long-run trend" or "genetic and management gains over time,"
          NEVER as "the year 2025" or "the calendar year 2024."
  - state: a regional/state indicator. Refer to as "regional baseline" or
          "state-level offset," not "state."
  - acres_planted_all: a size-weighting prior. Refer to as "scale of planted acreage"
          or "size of corn footprint," NEVER as a numeric acreage value.

Write a tight 200-400-word narrative in markdown. Use this structure:

  ## Headline
  One sentence. State, year, forecast date in plain English; the FORECAST point
  estimate (NOT the truth); how it compares to the 5-year mean (above / below /
  in line with).

  ## Trajectory
  Two or three sentences. Mention how confident the cone is at this stage
  (cone width tightens over the season; an EOS cone is the tightest we get).
  If `cone_status == "unavailable_pending_ndvi"`, say so plainly: the cone is
  unavailable because in-season NDVI for this year hasn't been pulled yet, but
  the point estimate (driven by weather, drought, soil, and management priors)
  is real. If `truth_state_yield_bu_acre` is non-null, mention how the forecast
  compares to truth here.

  ## Drivers
  Walk through the top 3 BIOPHYSICAL drivers (from `top_drivers`) in plain
  language. For each: name the feature using the glossary above (do NOT confuse
  units — °F-hours is not °F-days), say whether it pushed the prediction up or
  down (with the signed bu/ac), and translate the feature value into something
  an agronomist would recognize. Do NOT just restate the JSON; interpret it.

  ## Model context
  Briefly note (1–2 sentences) the structural priors from `structural_drivers`.
  These are not weather/biology — they're how the model thinks about long-run
  trend, regional baseline, and farm scale. Surface their summed contribution
  in bu/ac so the reader can see how much of the prediction is "structural"
  vs. "season-driven." Do NOT report raw feature values for structural drivers.

  ## Analogs
  If `analog_years` is non-empty, name the years explicitly and the anchor county.
  Note whether they cluster in dry, wet, hot, or cool seasons if you can tell from
  context.

  If `analog_years` is empty: state the operational reason from `cone_status`. For
  `unavailable_pending_ndvi`, say something like: "Analogs are unavailable because
  the analog-retrieval embedding requires in-season NDVI, which has not yet been
  pulled for this forecast year." Do NOT speculate that the season is unusual or
  unprecedented — the absence of analogs here is a data-pipeline state, not a
  signal about the season.

  ## Caveats
  One or two sentences. Mention if NASS truth is missing (forecast year), if cone is
  pending, if the regressor's per-state bias matters here. Don't invent caveats not
  in the data.

Hard rules:
  - The headline number is `point_estimate_bu_acre`, NOT `truth_state_yield_bu_acre`.
  - Do not invent numbers. Use the values in the JSON, rounded sensibly.
  - Do not write "I" or "we" — neutral, third-person analyst voice.
  - Do not hedge more than the cone width supports. If the cone is tight, the
    forecast is confident. If it's wide or absent, say so plainly.
  - Do not add markdown beyond what's specified above. No tables. No code fences.
  - Stop at the end of "## Caveats". Do not summarize at the end.
"""


# ---------- payload formatter ---------------------------------------------

def _strip_for_prompt(forecast: DateForecast) -> Dict[str, Any]:
    """Return a JSON-serializable dict with only the fields the narrator
    needs. Drops Pydantic noise and round-trips floats to a sane precision
    so the model doesn't fixate on spurious decimals."""
    def r(v: Optional[float], n: int = 1) -> Optional[float]:
        if v is None:
            return None
        return round(float(v), n)

    cone = None
    if forecast.cone is not None:
        cone = {
            "p10": r(forecast.cone.p10),
            "p50": r(forecast.cone.p50),
            "p90": r(forecast.cone.p90),
            "width_80": r(forecast.cone.width_80),
        }

    drivers = [
        {
            "feature": d.feature,
            "shap_bu_acre": r(d.shap_bu_acre),
            "feature_value_state_mean": r(d.feature_value_state_mean, 3),
            "direction": d.direction,
        }
        for d in forecast.top_drivers
    ]

    structural = [
        {
            "feature": d.feature,
            "shap_bu_acre": r(d.shap_bu_acre),
            "feature_value_state_mean": r(d.feature_value_state_mean, 3),
            "direction": d.direction,
        }
        for d in forecast.structural_drivers
    ]

    analogs = [
        {
            "year": a.year,
            "geoid": a.geoid,
            "county_name": a.county_name,
            "state_alpha": a.state_alpha,
            "distance": r(a.distance, 3),
            "observed_yield_bu_acre": r(a.observed_yield_bu_acre),
            "detrended_yield_bu_acre": r(a.detrended_yield_bu_acre),
        }
        for a in forecast.analog_years
    ]

    anchor = None
    if forecast.analog_anchor is not None:
        anchor = {
            "geoid": forecast.analog_anchor.geoid,
            "county_name": forecast.analog_anchor.county_name,
            "state_alpha": forecast.analog_anchor.state_alpha,
            "acres_planted": r(forecast.analog_anchor.acres_planted, 0),
            "rationale": forecast.analog_anchor.rationale,
        }

    return {
        "forecast_date": forecast.forecast_date,
        "point_estimate_bu_acre": r(forecast.point_estimate_bu_acre),
        "cone": cone,
        "cone_status": forecast.cone_status,
        "n_counties_regressor": forecast.n_counties_regressor,
        "n_counties_cone": forecast.n_counties_cone,
        "top_drivers": drivers,
        "structural_drivers": structural,
        "analog_years": analogs,
        "analog_anchor": anchor,
    }


def _build_user_message(
    state_response: StateForecastResponse,
    forecast: DateForecast,
) -> str:
    """User-side payload: the structured forecast plus a short framing line.
    Claude responds in markdown directly; no transformation needed on output."""
    payload = {
        "state": state_response.state,
        "state_name": state_response.state_name,
        "year": state_response.year,
        "model_version": state_response.model_version,
        "truth_state_yield_bu_acre": (
            None if state_response.truth_state_yield_bu_acre is None
            else round(float(state_response.truth_state_yield_bu_acre), 1)
        ),
        "history": {
            "mean_5yr_bu_acre": (
                None if state_response.history.mean_5yr_bu_acre is None
                else round(float(state_response.history.mean_5yr_bu_acre), 1)
            ),
            "mean_10yr_bu_acre": (
                None if state_response.history.mean_10yr_bu_acre is None
                else round(float(state_response.history.mean_10yr_bu_acre), 1)
            ),
        },
        "forecast": _strip_for_prompt(forecast),
    }
    return (
        f"Write the narrative for the following forecast.\n\n"
        f"```json\n{json.dumps(payload, indent=2)}\n```"
    )


# ---------- main entry point ----------------------------------------------

# Token budget: 400-word target * ~1.4 tokens/word * 2x safety = ~1100. Round up.
MAX_OUTPUT_TOKENS = 1200
MODEL = "claude-haiku-4-5"


def narrate_forecast(
    *,
    client: Any,                    # anthropic.Anthropic
    state_response: StateForecastResponse,
    forecast: DateForecast,
    model: str = MODEL,
    max_tokens: int = MAX_OUTPUT_TOKENS,
) -> NarrateResponse:
    """Single Claude API call. Returns a NarrateResponse; the route handler
    serializes it directly back to the client."""
    user_msg = _build_user_message(state_response, forecast)

    log.info(
        "narrate_forecast: state=%s year=%s date=%s model=%s",
        state_response.state, state_response.year, forecast.forecast_date, model,
    )

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as exc:
        log.exception("narrate_forecast: API call failed")
        # Graceful surface — return a NarrateResponse with the error so the
        # frontend's error-rendering still has a structured response.
        return NarrateResponse(
            narrative=(
                f"## Narrative unavailable\n\n"
                f"Claude API call failed: `{type(exc).__name__}: {exc}`. "
                f"The structured forecast above is unaffected."
            ),
            stub=False,
            tool_calls=[],
            model_version=state_response.model_version,
        )

    # Extract text from the response. The SDK returns a list of content
    # blocks; for a tool-free one-shot it's always one TextBlock.
    text_chunks = []
    for block in resp.content:
        # block.type == "text" carries .text. Other types (tool_use) shouldn't
        # appear since we don't pass tools, but guard defensively.
        bt = getattr(block, "type", None)
        if bt == "text":
            text_chunks.append(getattr(block, "text", ""))
    narrative = "\n".join(t for t in text_chunks if t).strip()
    if not narrative:
        narrative = (
            "## Narrative unavailable\n\nClaude returned an empty response. "
            "Try again, or fall back to the structured forecast above."
        )

    log.info(
        "narrate_forecast: ok, %d chars, stop_reason=%s, usage in/out=%s/%s",
        len(narrative),
        getattr(resp, "stop_reason", "?"),
        getattr(getattr(resp, "usage", None), "input_tokens", "?"),
        getattr(getattr(resp, "usage", None), "output_tokens", "?"),
    )

    return NarrateResponse(
        narrative=narrative,
        stub=False,
        tool_calls=[],
        model_version=state_response.model_version,
    )
