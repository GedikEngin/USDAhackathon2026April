"""
Tool definitions for the ReasoningAgent.

Each tool has three parts bundled together:
  1. A JSON schema (what we send to Claude in the `tools` parameter)
  2. A Python implementation (what actually runs)
  3. Unit conventions documented inline

All tools operate on AgentState (see agent.base), which holds the
pre-classified image data. The agent does NOT classify images itself —
that happens once at startup in agent_repl.py and the result is injected
into AgentState.

Unit conventions (locked Phase 7, 2026-04-24):
  - All "how much of class X" numbers are 0-100 percentages, in every
    tool input and output. `simulate_intervention`'s `fraction` parameter
    is 0-100 here; the tool layer divides by 100 before calling the
    underlying `scripts.emissions.simulate_intervention()` (which uses
    0.0-1.0 internally). One scale in Haiku's context; no unit bugs.
  - Areas in hectares (ha).
  - Emissions in tCO2e (annual: tCO2e/yr; embodied: tCO2e one-time).
  - Negative annual = net sink, positive = net source.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Match the backend's import pattern for scripts.emissions.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.emissions import (  # noqa: E402
    LAND_USE_EMISSIONS,
    SOURCES,
    EmissionsResult,
    simulate_intervention as _simulate_intervention_raw,
)

# Valid from/to classes for interventions. Excludes no_data (0) and
# background (1) because those aren't emissions-relevant.
_EMISSIONS_CLASS_NAMES = sorted(LAND_USE_EMISSIONS.keys())


# ---------------------------------------------------------------------------
# Shared state the tools read from
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """Pre-classified image data shared across all tool calls in one session."""
    # Percentages keyed by class name, 0-100. Includes no_data/background.
    percentages: Dict[str, float]
    # Raw emissions aggregation from compute_emissions(). Classes excluded
    # from emissions (no_data, background) are in `.excluded_breakdown`,
    # the rest in `.per_class`.
    emissions: EmissionsResult
    # Total ground area, in hectares, for the whole image.
    total_area_ha: float
    # Optional context the agent may surface (image path, inference time, etc.)
    image_label: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dominant_class(pcts: Dict[str, float]) -> str:
    if not pcts:
        return ""
    return max(pcts.items(), key=lambda kv: kv[1])[0]


def _net_sink_or_source(annual: float) -> str:
    if annual < -0.5:
        return "sink"
    if annual > 0.5:
        return "source"
    return "neutral"


def _model_caveats_for_breakdown(pcts: Dict[str, float]) -> List[str]:
    """
    Surface honest model-quality caveats based on class composition.
    Per Phase 5 findings: forest and barren are the weakest classes,
    and rural forest-heavy tiles sometimes collapse to background.
    """
    caveats: List[str] = []
    forest_pct = pcts.get("forest", 0.0)
    background_pct = pcts.get("background", 0.0)
    barren_pct = pcts.get("barren", 0.0)
    water_pct = pcts.get("water", 0.0)

    if forest_pct > 20.0:
        caveats.append(
            "Forest class has val IoU of 0.39; on mixed-vegetation rural "
            "scenes the model sometimes misclassifies forest as background. "
            "Treat forest percentage as approximate."
        )
    if background_pct > 40.0:
        caveats.append(
            "Background class (heterogeneous 'other impervious/misc' "
            "category) accounts for a large share of the image. Background "
            "is excluded from emissions aggregation but may contain "
            "unidentified sources or sinks."
        )
    if barren_pct > 10.0:
        caveats.append(
            "Barren class has val IoU of 0.35 — the weakest class. "
            "Barren pixels may in fact be sparsely-vegetated forest or "
            "fallow agriculture."
        )
    if water_pct > 5.0:
        caveats.append(
            "Water is treated as neutral open freshwater (IPCC AR6 baseline). "
            "Wetlands, which LoveDA does not distinguish from open water, "
            "would instead be a strong sink (-8 to -12 tCO2e/ha/yr)."
        )
    return caveats


def _assumptions_for_classes_present(classes_present: List[str]) -> List[str]:
    """Per-class modeling-assumption caveats, surfaced only when relevant."""
    out: List[str] = []
    if "building" in classes_present:
        out.append(
            "Building annual emissions (65 tCO2e/ha/yr) are operational "
            "energy only and assume a single-story footprint. Multi-story "
            "buildings scale roughly linearly with floor-area ratio."
        )
    if "water" in classes_present:
        out.append(
            "Water factor assumes open freshwater. Wetlands would be a "
            "strong sink; flag if imagery may contain wetlands."
        )
    if "forest" in classes_present:
        out.append(
            "Forest annual sequestration (-8 tCO2e/ha/yr) reflects "
            "steady-state mature forest. Newly-planted forest takes "
            "~20 years to reach this rate; first-year plantings are "
            "near-neutral or slightly positive."
        )
    if "agriculture" in classes_present:
        out.append(
            "Agriculture annual (+2.5 tCO2e/ha/yr) is a global average "
            "across crop types; rice, livestock-intensive, and "
            "heavily-fertilized systems emit substantially more."
        )
    return out


def _collect_sources_cited(classes_present: List[str]) -> Dict[str, str]:
    """Return SRC-n -> full citation, only for sources used by classes present."""
    keys: set = set()
    for name in classes_present:
        if name in LAND_USE_EMISSIONS:
            factor = LAND_USE_EMISSIONS[name]
            keys.add(factor.annual_source)
            keys.add(factor.embodied_source)
    return {k: SOURCES[k] for k in sorted(keys) if k in SOURCES}


# ---------------------------------------------------------------------------
# Tool 1: get_land_breakdown
# ---------------------------------------------------------------------------

GET_LAND_BREAKDOWN_SCHEMA: Dict[str, Any] = {
    "name": "get_land_breakdown",
    "description": (
        "Return the land-cover composition of the classified image as "
        "percentages (0-100) by class, plus the dominant class, total "
        "area in hectares, and any model-quality caveats that apply to "
        "this particular image. Call this first to understand what you "
        "are looking at. Takes no arguments."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


def tool_get_land_breakdown(state: AgentState) -> Dict[str, Any]:
    pcts = state.percentages
    return {
        "percentages": {k: round(v, 2) for k, v in pcts.items() if v > 0.0},
        "dominant_class": _dominant_class(pcts),
        "total_area_ha": round(state.total_area_ha, 3),
        "assessed_area_ha": round(state.emissions.assessed_area_ha, 3),
        "excluded_fraction": round(state.emissions.excluded_fraction, 4),
        "excluded_breakdown": {
            k: round(v, 4) for k, v in state.emissions.excluded_breakdown.items()
        },
        "model_caveats": _model_caveats_for_breakdown(pcts),
    }


# ---------------------------------------------------------------------------
# Tool 2: get_emissions_estimate
# ---------------------------------------------------------------------------

GET_EMISSIONS_ESTIMATE_SCHEMA: Dict[str, Any] = {
    "name": "get_emissions_estimate",
    "description": (
        "Return the full emissions estimate for the classified image: "
        "per-class annual flux (tCO2e/yr) and embodied stock (tCO2e), "
        "totals, whether the parcel is a net sink or source, the "
        "[SRC-n] citation tags used and their full references, and "
        "per-class modeling assumptions. Units: tCO2e, hectares. "
        "Negative annual = sink, positive = source. No arguments."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


def tool_get_emissions_estimate(state: AgentState) -> Dict[str, Any]:
    em = state.emissions
    classes_present = list(em.per_class.keys())

    per_class_out: Dict[str, Dict[str, Any]] = {}
    for name, row in em.per_class.items():
        factor = LAND_USE_EMISSIONS[name]
        per_class_out[name] = {
            "area_ha": round(row["area_ha"], 3),
            "annual_tco2e": round(row["annual_tco2e"], 2),
            "embodied_tco2e": round(row["embodied_tco2e"], 2),
            "annual_src": factor.annual_source,
            "embodied_src": factor.embodied_source,
        }

    return {
        "total_annual_tco2e_per_yr": round(em.total_annual_tco2e_per_yr, 2),
        "total_embodied_tco2e": round(em.total_embodied_tco2e, 2),
        "net_sink_or_source": _net_sink_or_source(em.total_annual_tco2e_per_yr),
        "per_class": per_class_out,
        "sources_cited": _collect_sources_cited(classes_present),
        "assumptions": _assumptions_for_classes_present(classes_present),
    }


# ---------------------------------------------------------------------------
# Tool 3: simulate_intervention
# ---------------------------------------------------------------------------

SIMULATE_INTERVENTION_SCHEMA: Dict[str, Any] = {
    "name": "simulate_intervention",
    "description": (
        "Simulate converting some percentage of one land class to another "
        "and return the change in annual emissions and embodied stock. "
        "Use this to evaluate 'what if' mitigation scenarios. "
        "Valid from/to classes: building, road, water, barren, forest, "
        "agriculture. `fraction_pct` is the percent (0-100) of the "
        "`from_class` area to convert — e.g. fraction_pct=50 converts "
        "half of whatever is currently `from_class`. The from_class must "
        "be present in the image (use get_land_breakdown to check)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "from_class": {
                "type": "string",
                "enum": _EMISSIONS_CLASS_NAMES,
                "description": "Land class to convert FROM (must be present in the image).",
            },
            "to_class": {
                "type": "string",
                "enum": _EMISSIONS_CLASS_NAMES,
                "description": "Land class to convert TO.",
            },
            "fraction_pct": {
                "type": "number",
                "minimum": 0,
                "maximum": 100,
                "description": (
                    "Percent (0-100) of the from_class area to convert. "
                    "100 = convert all of it. 50 = half."
                ),
            },
        },
        "required": ["from_class", "to_class", "fraction_pct"],
    },
}


def tool_simulate_intervention(
    state: AgentState,
    from_class: str,
    to_class: str,
    fraction_pct: float,
) -> Dict[str, Any]:
    # Convert 0-100 -> 0.0-1.0 for the underlying pure function.
    fraction = float(fraction_pct) / 100.0
    result = _simulate_intervention_raw(
        current_percentages=state.percentages,
        total_area_ha=state.total_area_ha,
        from_class=from_class,
        to_class=to_class,
        fraction=fraction,
    )

    delta_annual = result.delta_annual_tco2e_per_yr
    if delta_annual < -0.05:
        effect = "reduction"
    elif delta_annual > 0.05:
        effect = "increase"
    else:
        effect = "no change"

    return {
        "converted_area_ha": round(result.converted_area_ha, 3),
        "delta_annual_tco2e_per_yr": round(delta_annual, 2),
        "delta_embodied_tco2e": round(result.delta_embodied_tco2e, 2),
        "annual_effect": effect,
        "narrative": result.narrative,
        "before_totals": {
            "annual_tco2e_per_yr": round(result.before.total_annual_tco2e_per_yr, 2),
            "embodied_tco2e": round(result.before.total_embodied_tco2e, 2),
        },
        "after_totals": {
            "annual_tco2e_per_yr": round(result.after.total_annual_tco2e_per_yr, 2),
            "embodied_tco2e": round(result.after.total_embodied_tco2e, 2),
        },
    }


# ---------------------------------------------------------------------------
# Tool 4: recommend_mitigation
# ---------------------------------------------------------------------------

RECOMMEND_MITIGATION_SCHEMA: Dict[str, Any] = {
    "name": "recommend_mitigation",
    "description": (
        "Return a ranked menu of candidate land-use interventions for this "
        "image, sorted by a chosen priority. Each candidate is a "
        "(from_class, to_class) pair evaluated at 100% conversion of the "
        "available from_class area. Only from_classes actually present in "
        "the image are considered. You should weigh the tradeoffs in the "
        "menu and write the actual recommendation yourself — this tool "
        "does not choose for you. "
        "priority='annual' ranks by most-negative annual flux change (best "
        "for long-run carbon). "
        "priority='embodied' ranks by most-negative embodied change (best "
        "for reducing built stock, e.g. un-building). "
        "priority='balanced' ranks by annual*20 + embodied (rough 20-year "
        "horizon blend)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "priority": {
                "type": "string",
                "enum": ["annual", "embodied", "balanced"],
                "description": "Which axis to rank by. Default 'annual'.",
            },
        },
        "required": ["priority"],
    },
}


def _available_from_classes(state: AgentState) -> List[str]:
    """Emissions-relevant classes actually present in the image (pct > 0)."""
    return [
        name
        for name in _EMISSIONS_CLASS_NAMES
        if state.percentages.get(name, 0.0) > 0.0
    ]


def tool_recommend_mitigation(
    state: AgentState,
    priority: str,
) -> Dict[str, Any]:
    if priority not in ("annual", "embodied", "balanced"):
        raise ValueError(f"priority must be annual, embodied, or balanced; got {priority!r}")

    available_from = _available_from_classes(state)
    candidates: List[Dict[str, Any]] = []

    for from_cls in available_from:
        from_pct = state.percentages.get(from_cls, 0.0)
        available_area_ha = (from_pct / 100.0) * state.total_area_ha
        if available_area_ha <= 0.0:
            continue

        from_factor = LAND_USE_EMISSIONS[from_cls]
        for to_cls in _EMISSIONS_CLASS_NAMES:
            if to_cls == from_cls:
                continue
            to_factor = LAND_USE_EMISSIONS[to_cls]

            annual_delta = available_area_ha * (
                to_factor.annual_tco2e_per_ha_per_yr
                - from_factor.annual_tco2e_per_ha_per_yr
            )
            embodied_delta = available_area_ha * (
                to_factor.embodied_tco2e_per_ha
                - from_factor.embodied_tco2e_per_ha
            )

            candidates.append({
                "from_class": from_cls,
                "to_class": to_cls,
                "available_area_ha": round(available_area_ha, 3),
                "annual_delta_if_full": round(annual_delta, 2),
                "embodied_delta_if_full": round(embodied_delta, 2),
            })

    # Rank.
    if priority == "annual":
        candidates.sort(key=lambda c: c["annual_delta_if_full"])
    elif priority == "embodied":
        candidates.sort(key=lambda c: c["embodied_delta_if_full"])
    else:  # balanced
        candidates.sort(
            key=lambda c: c["annual_delta_if_full"] * 20.0 + c["embodied_delta_if_full"]
        )

    # Surface-friendly notes.
    notes: List[str] = []
    missing_from_emissions_classes = [
        c for c in _EMISSIONS_CLASS_NAMES if c not in available_from
    ]
    if missing_from_emissions_classes:
        notes.append(
            "Not present in image (no interventions available from these): "
            + ", ".join(missing_from_emissions_classes)
        )
    notes.append(
        "Deltas shown assume 100% conversion of the available from_class "
        "area. Use simulate_intervention with a fraction_pct for partial "
        "scenarios."
    )
    if priority == "balanced":
        notes.append(
            "'balanced' ranking weighs 20 years of annual flux against the "
            "one-time embodied cost/benefit. Longer horizons favor "
            "sequestration-heavy interventions; shorter favor low-embodied ones."
        )
    notes.append(
        "Forest conversion assumes steady-state sequestration rate, not "
        "first-year plantings. Real-world abatement is back-loaded over ~20 years."
    )

    return {
        "priority": priority,
        "top_interventions": candidates[:5],
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Tool registry — the one place that ties schemas to implementations
# ---------------------------------------------------------------------------

# Each implementation is a (state, **tool_input) -> dict callable.
ToolImpl = Callable[..., Dict[str, Any]]

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    GET_LAND_BREAKDOWN_SCHEMA,
    GET_EMISSIONS_ESTIMATE_SCHEMA,
    SIMULATE_INTERVENTION_SCHEMA,
    RECOMMEND_MITIGATION_SCHEMA,
]

TOOL_IMPLS: Dict[str, ToolImpl] = {
    "get_land_breakdown": tool_get_land_breakdown,
    "get_emissions_estimate": tool_get_emissions_estimate,
    "simulate_intervention": tool_simulate_intervention,
    "recommend_mitigation": tool_recommend_mitigation,
}


def dispatch_tool(
    name: str,
    tool_input: Dict[str, Any],
    state: AgentState,
) -> Dict[str, Any]:
    """
    Look up a tool by name and execute it with the given input.

    Raises ValueError if the tool is unknown. Tool implementations may
    raise their own ValueErrors on invalid inputs (e.g. from_class not
    present in image); those propagate up and are formatted into
    tool_result content with is_error=True by the agent loop.
    """
    impl = TOOL_IMPLS.get(name)
    if impl is None:
        raise ValueError(f"Unknown tool: {name!r}. Available: {list(TOOL_IMPLS)}")
    return impl(state, **tool_input)
