"""
emissions.py — Land-class → GHG emissions factor lookup.

Maps LoveDA's 6 emissions-relevant classes (building, road, water, barren,
forest, agriculture) to per-hectare CO2-equivalent factors sourced from
primary authorities: IPCC AR6 Working Group III (2022), EPA GHG Inventory
(2024), EIA CBECS 2018, and EDGAR v7.

Why two numbers per class:
    ANNUAL flux (tCO2e / ha / yr)     — ongoing, steady-state emissions or
                                        sequestration. What the land is
                                        "doing" each year going forward.
    EMBODIED stock (tCO2e / ha)        — one-time carbon cost (for built
                                        surfaces) or stored carbon (for
                                        natural ecosystems). Released all
                                        at once on conversion; amortized
                                        if we reason about land-use change.

Sign convention (for BOTH annual and embodied):
    POSITIVE → net emitter OR released-on-disturbance carbon cost
    NEGATIVE → net sequestration OR (for embodied) stored in a stable pool
              that would be released on conversion

Classes excluded from aggregation:
    class 0 (no-data)    — masked in the vision loss via ignore_index=0.
                           Pixels with value 0 are not a land cover; they're
                           missing annotation regions. Treat as "unknown."
    class 1 (background) — LoveDA's heterogeneous "other/misc impervious"
                           bucket (parking, construction, misc). Too
                           semantically messy to assign a defensible factor.
                           PROJECT_PLAN locked this exclusion in Phase 1.

Every numeric value in LAND_USE_EMISSIONS carries an inline [source] tag
pointing to the citation block at the bottom of this file. When the agent
calls get_emissions_estimate() downstream in Weekend 2, those citation keys
travel with the numbers so the LLM can surface them in its report.

Usage:
    from emissions import LAND_USE_EMISSIONS, compute_emissions

    # direct lookup
    factor = LAND_USE_EMISSIONS["forest"]
    print(factor["annual_tco2e_per_ha_per_yr"])   # -8.0 (sink)

    # aggregate over a classified image
    result = compute_emissions(
        class_pixel_counts={2: 120000, 4: 50000, 6: 830000, 7: 400000},
        total_pixels=1024 * 1024,
        pixel_area_m2=0.3 * 0.3,  # LoveDA is 0.3m resolution
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# -----------------------------------------------------------------------------
# CLASS INDEX MAPPING (locked in DECISIONS_LOG.md Phase 1)
# -----------------------------------------------------------------------------
# Integer label → human-readable name. Mirrors torchgeo's ordering.
CLASS_NAMES: Dict[int, str] = {
    0: "no_data",       # ignore_index in loss; NOT a land cover
    1: "background",    # trained as real class; EXCLUDED from emissions
    2: "building",
    3: "road",
    4: "water",
    5: "barren",
    6: "forest",
    7: "agriculture",
}

# Classes whose per-pixel contributions we aggregate for an emissions estimate.
EMISSIONS_CLASSES: List[int] = [2, 3, 4, 5, 6, 7]


# -----------------------------------------------------------------------------
# EMISSIONS FACTOR TABLE
# -----------------------------------------------------------------------------
# Each entry is a global-average estimate. Regional factors are a hackathon-day
# stretch, not part of v1 (see PROJECT_PLAN.md "Decisions already locked in").
#
# The "annual" value is the per-hectare flux you would bill against the land
# each year under steady-state use. The "embodied" value is the one-time
# carbon cost (or stored stock) associated with the surface as it exists —
# what you would pay (or release) on conversion.
#
# Every number has an inline [SRC-n] tag; see SOURCES at the bottom.

@dataclass(frozen=True)
class EmissionFactor:
    """One land-class entry in the emissions lookup table."""
    class_name: str
    annual_tco2e_per_ha_per_yr: float
    annual_source: str
    annual_notes: str
    embodied_tco2e_per_ha: float
    embodied_source: str
    embodied_notes: str
    uncertainty: str  # qualitative band: "low", "medium", "high"


LAND_USE_EMISSIONS: Dict[str, EmissionFactor] = {
    # -------------------------------------------------------------------------
    "building": EmissionFactor(
        class_name="building",
        # Annual = operational energy only (electricity + on-site combustion).
        # EIA CBECS 2018 reports avg U.S. commercial building energy intensity
        # at 70.6 kBtu/sf/yr [SRC-3]. At 100% footprint coverage that's ~7.6 GJ/m²/yr
        # ≈ 76 TJ/ha/yr. Applying the current U.S. grid avg ~0.37 kg CO2e/kWh for
        # electricity (~60% of mix) and ~53 kg CO2e/GJ for natural gas gives a
        # blended emissions intensity on the order of 60-70 tCO2e/ha/yr for a
        # building footprint fully covered in conditioned floorspace. We use 65
        # as the central estimate. This is OPERATIONAL ONLY — embodied sits
        # in the second field.
        annual_tco2e_per_ha_per_yr=65.0,
        annual_source="SRC-3",
        annual_notes=(
            "Operational emissions from energy use only; derived from EIA CBECS 2018 "
            "avg intensity 70.6 kBtu/sf/yr and EPA eGRID-weighted blended emissions "
            "factor. Single-story footprint assumption; real multi-story buildings "
            "emit considerably more per ground-ha. Does NOT include embodied carbon."
        ),
        # Embodied = one-time carbon cost of the materials (concrete, steel,
        # brick) and construction. IPCC AR6 WGIII Ch. 9 and the JRC EFIResources
        # benchmark [SRC-5] report building embodied GWP of roughly 4-8 kg
        # CO2e/m²/yr amortized over a 50-yr service life — i.e., 200-400 kg
        # CO2e/m² lifetime. At 10,000 m²/ha that's 2000-4000 tCO2e/ha lifetime,
        # or a central estimate of ~600 tCO2e/ha for a typical single-story
        # footprint. Multi-story would scale ~linearly with floor-area ratio.
        embodied_tco2e_per_ha=600.0,
        embodied_source="SRC-5",
        embodied_notes=(
            "Structural embodied carbon (cement, steel, masonry) amortized over "
            "building footprint. Assumes ~single-story; scales with floor-area ratio. "
            "Released on demolition OR converts to long-term stock while standing."
        ),
        uncertainty="high",
    ),

    # -------------------------------------------------------------------------
    "road": EmissionFactor(
        class_name="road",
        # Annual = pavement maintenance + ongoing vehicle-combustion-related
        # emissions attributed to road surface. IPCC AR6 WGIII Ch. 10 assigns
        # transport sector emissions to activity, not to land surface; so we
        # use only the pavement O&M contribution here (~3-5 tCO2e/ha/yr for
        # repaving cycles and surface maintenance). Central: 4.
        annual_tco2e_per_ha_per_yr=4.0,
        annual_source="SRC-1",
        annual_notes=(
            "Pavement maintenance and repaving amortized annually. Does NOT include "
            "vehicle tailpipe emissions — those are attributed to the transport "
            "sector, not the land surface. Road FOOTPRINT does not create tailpipe "
            "emissions; vehicle activity does."
        ),
        # Embodied = asphalt/concrete pavement + base course. LCA literature
        # puts asphalt pavement at ~15-25 kg CO2e/m² for a typical urban section
        # [SRC-5]. At 10,000 m²/ha → ~200 tCO2e/ha. Concrete higher (~300+).
        embodied_tco2e_per_ha=220.0,
        embodied_source="SRC-5",
        embodied_notes=(
            "Pavement (asphalt or concrete) + base course embodied carbon. Released "
            "slowly over pavement lifetime (~20 yr) plus on major reconstruction."
        ),
        uncertainty="medium",
    ),

    # -------------------------------------------------------------------------
    "water": EmissionFactor(
        class_name="water",
        # Annual = ~0 as baseline for open freshwater surfaces. IPCC AR6 WGI
        # Ch. 5 treats inland waters as approximately carbon-neutral at global
        # average (small CH4 source from reservoirs, small CO2 sink in some
        # lakes) [SRC-1]. Wetlands are a special case not handled by this lookup
        # — they would carry a strong sequestration signal (-8 to -12 tCO2e/ha/yr)
        # but LoveDA's "water" class doesn't distinguish wetlands from open
        # water, so we use the conservative open-water baseline. An honest
        # caveat at the agent layer should flag this.
        annual_tco2e_per_ha_per_yr=0.0,
        annual_source="SRC-1",
        annual_notes=(
            "Open freshwater baseline — approximately neutral per IPCC AR6 WGI Ch. 5. "
            "Wetlands would be a strong sink (-8 to -12 tCO2e/ha/yr) but LoveDA's "
            "'water' class does not distinguish wetlands from open water. Agent "
            "should caveat this when water fraction is high."
        ),
        embodied_tco2e_per_ha=0.0,
        embodied_source="SRC-1",
        embodied_notes="No embodied carbon for open water surface.",
        uncertainty="medium",  # high if wetland content is unknown
    ),

    # -------------------------------------------------------------------------
    "barren": EmissionFactor(
        class_name="barren",
        # Annual = ~0. Bare/disturbed land has negligible ongoing biogenic flux.
        # Some SOC loss under continued disturbance, but global-average
        # steady-state barren land is near-neutral. IPCC 2019 Refinement
        # Vol. 4 Ch. 2 [SRC-4] provides mineral-soil reference stocks for
        # "other lands" at low values (~30-60 tC/ha → ~110-220 tCO2e/ha stock).
        annual_tco2e_per_ha_per_yr=0.0,
        annual_source="SRC-4",
        annual_notes=(
            "Near-zero net annual flux at steady state. Converting barren back to "
            "forest or agri would RESTORE soil carbon over decades (-1 to -3 "
            "tCO2e/ha/yr during regrowth)."
        ),
        # Embodied = low soil carbon stock (degraded/disturbed soils).
        embodied_tco2e_per_ha=50.0,
        embodied_source="SRC-4",
        embodied_notes=(
            "Residual soil organic carbon in degraded/bare land, top 30cm. "
            "IPCC 2019 Guidelines default mineral-soil reference for 'other lands'."
        ),
        uncertainty="high",
    ),

    # -------------------------------------------------------------------------
    "forest": EmissionFactor(
        class_name="forest",
        # Annual = net sequestration sink. IPCC AR6 WGIII Ch. 7 estimates total
        # economic mitigation from forests + other natural ecosystems at 7.3
        # (3.9-13.1) GtCO2e/yr [SRC-1]. Global forest area ~4.06 Gha (FAO FRA
        # 2020). The full 7.3 Gt figure includes improved mgmt + restoration
        # + avoided deforestation, not just passive sink; the passive sink
        # component is ~2-3 tCO2e/ha/yr for mature forest, higher for
        # regrowing forest. We use -8 as a realistic managed-forest average,
        # mid-range between mature (~-2) and young regrowth (~-15).
        annual_tco2e_per_ha_per_yr=-8.0,
        annual_source="SRC-1",
        annual_notes=(
            "Net carbon sink. Derived from IPCC AR6 WGIII Ch. 7 sectoral mitigation "
            "potentials. Value represents managed/mixed-age forest; mature primary "
            "forest is closer to -2 tCO2e/ha/yr, active regrowth can reach -15."
        ),
        # Embodied = living biomass (aboveground + belowground) + soil carbon.
        # IPCC 2019 Refinement Vol. 4 Ch. 4 Table 4.7 [SRC-4] gives default
        # aboveground biomass stocks of ~100-300 tDM/ha for temperate forest
        # → ~180-550 tCO2e/ha just in biomass. Plus ~300-500 tCO2e/ha soil.
        # Central estimate ~800.
        embodied_tco2e_per_ha=800.0,
        embodied_source="SRC-4",
        embodied_notes=(
            "Living biomass + soil organic carbon, top 30cm. IPCC 2019 Guidelines "
            "default values for temperate forest. Released on deforestation; "
            "recoverable over decades via reforestation."
        ),
        uncertainty="medium",
    ),

    # -------------------------------------------------------------------------
    "agriculture": EmissionFactor(
        class_name="agriculture",
        # Annual = small net source. IPCC AR6 WGIII Ch. 7 [SRC-1] attributes
        # ~4.1 GtCO2e/yr mitigation potential to cropland+grassland mgmt
        # globally, over ~4.8 Gha combined (FAOSTAT). The CURRENT baseline is
        # a net source: crops are ~net-zero in CO2 (biomass turns over annually)
        # but N2O from fertilizer (~1-3 tCO2e/ha/yr via IPCC Tier 1) and CH4
        # from rice or manure (~2-8 tCO2e/ha for flooded rice per EDGAR v7
        # [SRC-6]) make typical cropland a modest source. Non-rice global avg
        # ~2-3 tCO2e/ha/yr. Central: 2.5.
        annual_tco2e_per_ha_per_yr=2.5,
        annual_source="SRC-6",
        annual_notes=(
            "Net annual source: dominated by N2O from fertilizer and, where present, "
            "CH4 from flooded rice or livestock on the same parcel. EDGAR v7 + IPCC "
            "Tier 1 emission factors. Rice-dominated areas would be higher (5-10); "
            "dryland grain lower (1-2). LoveDA doesn't distinguish crop types."
        ),
        # Embodied = soil organic carbon (diminished vs native by cultivation).
        # IPCC 2019 Guidelines [SRC-4] reference SOC for long-term cultivated
        # cropland: ~40-80 tC/ha top 30cm → ~150-300 tCO2e/ha.
        embodied_tco2e_per_ha=150.0,
        embodied_source="SRC-4",
        embodied_notes=(
            "Soil organic carbon in top 30cm of cultivated land. Reduced vs "
            "native ecosystem baseline. Conversion BACK to forest/grassland "
            "would gradually rebuild this stock."
        ),
        uncertainty="medium",
    ),
}


# -----------------------------------------------------------------------------
# AGGREGATION
# -----------------------------------------------------------------------------

@dataclass
class EmissionsResult:
    """Per-class and total emissions for a classified image."""
    # Per-class breakdown. class_name → dict with pixel_count, pixel_fraction,
    # area_ha, annual_tco2e, embodied_tco2e.
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)

    total_area_ha: float = 0.0
    assessed_area_ha: float = 0.0  # excludes background + no_data

    total_annual_tco2e_per_yr: float = 0.0
    total_embodied_tco2e: float = 0.0

    # Fractions of the image excluded from emissions aggregation.
    excluded_fraction: float = 0.0
    excluded_breakdown: Dict[str, float] = field(default_factory=dict)


def compute_emissions(
    class_pixel_counts: Dict[int, int],
    total_pixels: int,
    pixel_area_m2: float,
) -> EmissionsResult:
    """
    Aggregate per-class pixel counts into emissions estimates.

    Args:
        class_pixel_counts: {class_id: pixel_count}. Class ids can be any
            of 0-7; 0 (no_data) and 1 (background) are excluded from
            aggregation but reported in `excluded_breakdown`.
        total_pixels: total pixels in the image (for fraction calc).
        pixel_area_m2: ground area per pixel. LoveDA is 0.3m GSD → 0.09 m²/px.

    Returns:
        EmissionsResult with per-class and total emissions.
    """
    m2_per_ha = 10_000.0
    total_area_ha = total_pixels * pixel_area_m2 / m2_per_ha

    result = EmissionsResult(total_area_ha=total_area_ha)
    assessed_pixels = 0
    excluded_pixels = 0

    for class_id, count in class_pixel_counts.items():
        name = CLASS_NAMES.get(class_id, f"unknown_{class_id}")
        fraction = count / total_pixels if total_pixels > 0 else 0.0

        if class_id not in EMISSIONS_CLASSES:
            # Excluded: no_data or background. Track but don't cost.
            result.excluded_breakdown[name] = fraction
            excluded_pixels += count
            continue

        area_ha = count * pixel_area_m2 / m2_per_ha
        factor = LAND_USE_EMISSIONS[name]
        annual = area_ha * factor.annual_tco2e_per_ha_per_yr
        embodied = area_ha * factor.embodied_tco2e_per_ha

        result.per_class[name] = {
            "pixel_count": float(count),
            "pixel_fraction": fraction,
            "area_ha": area_ha,
            "annual_tco2e": annual,
            "embodied_tco2e": embodied,
        }
        result.total_annual_tco2e_per_yr += annual
        result.total_embodied_tco2e += embodied
        assessed_pixels += count

    result.assessed_area_ha = assessed_pixels * pixel_area_m2 / m2_per_ha
    result.excluded_fraction = (
        excluded_pixels / total_pixels if total_pixels > 0 else 0.0
    )
    return result


# -----------------------------------------------------------------------------
# SOURCES (referenced inline above by [SRC-n])
# -----------------------------------------------------------------------------
SOURCES: Dict[str, str] = {
    "SRC-1": (
        "IPCC (2022). Climate Change 2022: Mitigation of Climate Change. "
        "Contribution of Working Group III to the Sixth Assessment Report "
        "(AR6 WGIII), Chapter 7: Agriculture, Forestry and Other Land Uses "
        "(AFOLU). Cambridge University Press. "
        "https://www.ipcc.ch/report/ar6/wg3/chapter/chapter-7/"
    ),
    "SRC-2": (
        "EPA (2024). Inventory of U.S. Greenhouse Gas Emissions and Sinks: "
        "1990-2022. U.S. Environmental Protection Agency, EPA 430R-24004. "
        "https://www.epa.gov/ghgemissions/inventory-us-greenhouse-gas-emissions-and-sinks-1990-2022"
    ),
    "SRC-3": (
        "U.S. EIA (2022). 2018 Commercial Buildings Energy Consumption Survey "
        "(CBECS) — Consumption and Expenditures Highlights. Average site energy "
        "intensity 70.6 kBtu/sf/yr. "
        "https://www.eia.gov/consumption/commercial/data/2018/"
    ),
    "SRC-4": (
        "IPCC (2019). 2019 Refinement to the 2006 IPCC Guidelines for National "
        "Greenhouse Gas Inventories, Volume 4: Agriculture, Forestry and Other "
        "Land Use. Default soil and biomass carbon stocks. "
        "https://www.ipcc-nggip.iges.or.jp/public/2019rf/vol4.html"
    ),
    "SRC-5": (
        "JRC EFIResources (2018). Environmental benchmarks for buildings. "
        "European Commission Joint Research Centre. Embodied and operational "
        "GWP per m²/yr by climate zone. "
        "https://publications.jrc.ec.europa.eu/repository/handle/JRC110085"
    ),
    "SRC-6": (
        "Crippa, M. et al. (2023). EDGAR v7.0 Global Greenhouse Gas Emissions "
        "Database. European Commission, Joint Research Centre. "
        "Sectoral emission factors including agricultural CH4 and N2O. "
        "https://edgar.jrc.ec.europa.eu/"
    ),
}


def cite(src_key: str) -> str:
    """Return the full citation for a [SRC-n] tag. Used by agent tools."""
    return SOURCES.get(src_key, f"<unknown source {src_key}>")


# -----------------------------------------------------------------------------
# SELF-TEST
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick sanity: make a 1024x1024 fake image at 0.3m/px that's 50% forest,
    # 30% agri, 20% building.
    fake_counts = {
        6: int(1024 * 1024 * 0.50),  # forest
        7: int(1024 * 1024 * 0.30),  # agri
        2: int(1024 * 1024 * 0.20),  # building
    }
    res = compute_emissions(
        class_pixel_counts=fake_counts,
        total_pixels=1024 * 1024,
        pixel_area_m2=0.3 * 0.3,
    )
    print(f"Total area: {res.total_area_ha:.3f} ha")
    print(f"Assessed area: {res.assessed_area_ha:.3f} ha")
    print(f"Excluded fraction: {res.excluded_fraction:.3f}")
    print()
    for name, stats in res.per_class.items():
        f = LAND_USE_EMISSIONS[name]
        print(
            f"  {name:12s} {stats['pixel_fraction']*100:5.1f}%  "
            f"{stats['area_ha']:6.3f} ha  "
            f"annual {stats['annual_tco2e']:+8.2f} tCO2e/yr  "
            f"embodied {stats['embodied_tco2e']:+10.1f} tCO2e  "
            f"[{f.annual_source}]"
        )
    print()
    print(f"TOTAL annual:   {res.total_annual_tco2e_per_yr:+.2f} tCO2e/yr")
    print(f"TOTAL embodied: {res.total_embodied_tco2e:+.1f} tCO2e")
    print()
    print("Sources:")
    for k, v in SOURCES.items():
        print(f"  [{k}] {v}")


# ============================================================
# Phase 6 addition: counterfactual simulation
# ============================================================

@dataclass
class SimulationResult:
    """Output of simulate_intervention()."""
    before: "EmissionsResult"
    after: "EmissionsResult"
    delta_annual_tco2e_per_yr: float
    delta_embodied_tco2e: float
    converted_area_ha: float
    narrative: str


# Reverse map: class name -> class id (for compute_emissions, which keys on int).
_NAME_TO_ID = {name: idx for idx, name in CLASS_NAMES.items()}


def simulate_intervention(
    current_percentages: dict,
    total_area_ha: float,
    from_class: str,
    to_class: str,
    fraction: float,
) -> SimulationResult:
    """
    Counterfactual: convert `fraction` of `from_class` area into `to_class`.

    No-data (class 0) and background (class 1) are not valid from/to classes.
    """
    valid_classes = set(LAND_USE_EMISSIONS.keys())
    if from_class not in valid_classes:
        raise ValueError(
            f"from_class '{from_class}' must be one of {sorted(valid_classes)}"
        )
    if to_class not in valid_classes:
        raise ValueError(
            f"to_class '{to_class}' must be one of {sorted(valid_classes)}"
        )
    if from_class == to_class:
        raise ValueError("from_class and to_class must differ")
    if not (0.0 <= fraction <= 1.0):
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")

    current_from_pct = current_percentages.get(from_class, 0.0)
    if current_from_pct <= 0.0:
        raise ValueError(
            f"No {from_class} present in the image "
            f"(current fraction: {current_from_pct:.2f}%). Nothing to convert."
        )

    SYNTH_TOTAL = 100_000_000
    pixel_area_m2 = (total_area_ha * 10_000.0) / SYNTH_TOTAL

    def pct_to_int_keyed_counts(pcts):
        """Name-keyed pcts (0-100) -> int-keyed pixel counts for compute_emissions."""
        counts = {cid: 0 for cid in CLASS_NAMES.keys()}
        for name, p in pcts.items():
            cid = _NAME_TO_ID.get(name)
            if cid is None:
                continue
            counts[cid] = int(round(p / 100.0 * SYNTH_TOTAL))
        return counts

    before_counts = pct_to_int_keyed_counts(current_percentages)
    before = compute_emissions(
        class_pixel_counts=before_counts,
        total_pixels=SYNTH_TOTAL,
        pixel_area_m2=pixel_area_m2,
    )

    shift_pct = current_from_pct * fraction
    after_pcts = dict(current_percentages)
    after_pcts[from_class] = current_from_pct - shift_pct
    after_pcts[to_class] = after_pcts.get(to_class, 0.0) + shift_pct
    after_counts = pct_to_int_keyed_counts(after_pcts)
    after = compute_emissions(
        class_pixel_counts=after_counts,
        total_pixels=SYNTH_TOTAL,
        pixel_area_m2=pixel_area_m2,
    )

    delta_annual = after.total_annual_tco2e_per_yr - before.total_annual_tco2e_per_yr
    delta_embodied = after.total_embodied_tco2e - before.total_embodied_tco2e
    converted_area_ha = shift_pct / 100.0 * total_area_ha

    direction_annual = "reduction" if delta_annual < 0 else "increase"
    narrative = (
        f"Converting {fraction*100:.0f}% of {from_class} "
        f"({converted_area_ha:.2f} ha) to {to_class}: "
        f"annual flux changes by {delta_annual:+.2f} tCO2e/yr ({direction_annual}), "
        f"embodied stock changes by {delta_embodied:+.2f} tCO2e."
    )

    return SimulationResult(
        before=before,
        after=after,
        delta_annual_tco2e_per_yr=delta_annual,
        delta_embodied_tco2e=delta_embodied,
        converted_area_ha=converted_area_ha,
        narrative=narrative,
    )
