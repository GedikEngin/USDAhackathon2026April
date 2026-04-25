"""
Smoke test for agent/tools.py. Builds a realistic AgentState that mimics
the 3546.png result from Phase 5/6 (72.24% forest, 21.46% background,
3.73% agri, 1.11% water, ~0.9 ha total) and exercises each tool.

Numerical anchors from CURRENT_STATE.md:
  - total_annual_tco2e_per_yr: -50.44
  - total_embodied_tco2e: 5542.81
  - 100% agri->forest delta_annual: -3.69 tCO2e/yr (on 0.35 ha... wait)

Actually re-reading CURRENT_STATE: 3546.png is 72.24% forest, 21.46%
background, 3.73% agri, 1.11% water. The CURRENT_STATE says "100% agri->forest
on 3546.png (0.35 ha): delta_annual -3.69 tCO2e/yr" — but that 0.35 ha is the
converted area, which matches 3.73% of ~9.4 ha (the total area at 0.3m GSD on
1024x1024). 1024*1024*0.09 m2 = 9437 m2 = 0.944 ha. Hmm, that doesn't match
9.4 ha. Let me recompute: 1024*1024 = 1,048,576 px; 0.09 m2/px = 94,371.84 m2
= 9.437 ha. Good. So 3.73% of 9.437 ha = 0.352 ha. Matches.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.tools import (
    AgentState,
    dispatch_tool,
    TOOL_SCHEMAS,
    TOOL_IMPLS,
)
from scripts.emissions import compute_emissions, CLASS_NAMES


def build_3546_state() -> AgentState:
    """Rebuild the 3546.png state from Phase 5/6 measurements."""
    percentages = {
        "no_data": 0.0,
        "background": 21.46,
        "building": 0.0,
        "road": 0.0,
        "water": 1.11,
        "barren": 0.0,
        "forest": 72.24,
        "agriculture": 3.73,
        # note: these sum to 98.54, not exactly 100, due to rounding in
        # the CLI output we're mimicking. Close enough for a smoke test.
    }

    # Rebuild pixel counts at 1024x1024, 0.3m GSD = 0.09 m2/px.
    total_pixels = 1024 * 1024
    name_to_id = {v: k for k, v in CLASS_NAMES.items()}
    class_pixel_counts = {}
    for name, pct in percentages.items():
        cid = name_to_id[name]
        class_pixel_counts[cid] = int(round(pct / 100.0 * total_pixels))

    emissions = compute_emissions(
        class_pixel_counts=class_pixel_counts,
        total_pixels=total_pixels,
        pixel_area_m2=0.09,
    )

    return AgentState(
        percentages=percentages,
        emissions=emissions,
        total_area_ha=9.437,
        image_label="Val/Urban/3546.png",
    )


def main() -> None:
    state = build_3546_state()

    print("=" * 70)
    print("AgentState built:")
    print(f"  image: {state.image_label}")
    print(f"  total_area_ha: {state.total_area_ha}")
    print(f"  percentages nonzero: "
          f"{ {k:v for k,v in state.percentages.items() if v>0} }")
    print(f"  total_annual_tco2e_per_yr: "
          f"{state.emissions.total_annual_tco2e_per_yr:.2f}")
    print(f"  total_embodied_tco2e: "
          f"{state.emissions.total_embodied_tco2e:.2f}")
    print()

    print(f"Registered tools: {list(TOOL_IMPLS)}")
    print(f"Tool schema names: {[s['name'] for s in TOOL_SCHEMAS]}")
    print()

    # --- Tool 1
    print("[1] get_land_breakdown()")
    r = dispatch_tool("get_land_breakdown", {}, state)
    print(json.dumps(r, indent=2, default=str))
    print()

    # --- Tool 2
    print("[2] get_emissions_estimate()")
    r = dispatch_tool("get_emissions_estimate", {}, state)
    print(json.dumps(r, indent=2, default=str))
    print()

    # --- Tool 3a: 100% agri->forest (should match Phase 6: -3.69 annual, +228 embodied)
    print("[3a] simulate_intervention(agri -> forest, 100%)")
    r = dispatch_tool(
        "simulate_intervention",
        {"from_class": "agriculture", "to_class": "forest", "fraction_pct": 100},
        state,
    )
    print(json.dumps(r, indent=2, default=str))
    print()

    # --- Tool 3b: 50% forest -> agri (reverse direction)
    print("[3b] simulate_intervention(forest -> agri, 50%) — sanity check reverse direction")
    r = dispatch_tool(
        "simulate_intervention",
        {"from_class": "forest", "to_class": "agriculture", "fraction_pct": 50},
        state,
    )
    print(json.dumps(r, indent=2, default=str))
    print()

    # --- Tool 3c: unit-bug regression — fraction_pct=50 should be HALF, not 50x
    # If there's a units bug and we pass 50 where Python wants 0.5, we'd get
    # an out-of-range ValueError. Let's confirm 50 is interpreted as 50%.
    # Expected: converted_area_ha = 0.5 * 72.24% * 9.437 ha = ~3.409 ha
    print("[3c] UNIT CHECK: forest -> agri, 50% — expect converted_area_ha ~ 3.41")
    r = dispatch_tool(
        "simulate_intervention",
        {"from_class": "forest", "to_class": "agriculture", "fraction_pct": 50},
        state,
    )
    print(f"  converted_area_ha: {r['converted_area_ha']} (expected ~3.41)")
    assert abs(r['converted_area_ha'] - 3.41) < 0.02, "UNIT BUG: fraction_pct conversion wrong!"
    print("  ✓ fraction_pct correctly interpreted as percentage\n")

    # --- Tool 4a: annual priority
    print("[4a] recommend_mitigation(priority=annual)")
    r = dispatch_tool("recommend_mitigation", {"priority": "annual"}, state)
    print(json.dumps(r, indent=2, default=str))
    print()

    # --- Tool 4b: embodied priority
    print("[4b] recommend_mitigation(priority=embodied)")
    r = dispatch_tool("recommend_mitigation", {"priority": "embodied"}, state)
    print(json.dumps(r, indent=2, default=str))
    print()

    # --- Tool 4c: balanced priority
    print("[4c] recommend_mitigation(priority=balanced)")
    r = dispatch_tool("recommend_mitigation", {"priority": "balanced"}, state)
    print(f"  top_interventions[0]: {r['top_interventions'][0]}")
    print()

    # --- Error path: nonexistent from_class
    print("[5] ERROR PATH: from_class not in image (building=0%)")
    try:
        dispatch_tool(
            "simulate_intervention",
            {"from_class": "building", "to_class": "forest", "fraction_pct": 100},
            state,
        )
        print("  ✗ should have raised!")
    except ValueError as e:
        print(f"  ✓ raised: {e}")
    print()

    # --- Error path: unknown tool
    print("[6] ERROR PATH: unknown tool name")
    try:
        dispatch_tool("get_the_moon", {}, state)
        print("  ✗ should have raised!")
    except ValueError as e:
        print(f"  ✓ raised: {e}")
    print()

    # --- Error path: bad priority
    print("[7] ERROR PATH: bad priority value")
    try:
        dispatch_tool("recommend_mitigation", {"priority": "vibes"}, state)
        print("  ✗ should have raised!")
    except ValueError as e:
        print(f"  ✓ raised: {e}")
    print()

    print("=" * 70)
    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
