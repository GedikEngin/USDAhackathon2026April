"""
Pydantic schemas for FastAPI request/response bodies.

Aligned to the actual scripts/emissions.py shapes:
  EmissionsResult.per_class = Dict[str, Dict[str, float]]
      row keys: pixel_count, pixel_fraction, area_ha, annual_tco2e, embodied_tco2e
  excluded_breakdown values are fractions 0.0-1.0 (not 0-100).
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


# ---------- shared sub-shapes ----------

class PerClassEmissions(BaseModel):
    """Single class's emissions row — matches emissions.py exactly."""
    pixel_count: float           # emissions.py stores as float
    pixel_fraction: float        # 0.0-1.0
    area_ha: float
    annual_tco2e: float          # positive = emitter, negative = sink
    embodied_tco2e: float


class EmissionsReport(BaseModel):
    """Full EmissionsResult serialized for transport."""
    per_class: dict[str, PerClassEmissions]
    excluded_breakdown: dict[str, float]  # class_name -> fraction 0.0-1.0
    total_area_ha: float
    assessed_area_ha: float
    excluded_fraction: float
    total_annual_tco2e_per_yr: float
    total_embodied_tco2e: float
    sources_cited: dict[str, str]         # SRC-n -> full citation


# ---------- /classify ----------

class ClassifyResponse(BaseModel):
    percentages: dict[str, float]         # 0-100 (convenience, from pixel counts)
    emissions: EmissionsReport
    mask_png_base64: str
    inference_ms: int
    input_shape: tuple[int, int]
    warnings: list[str] = Field(default_factory=list)


# ---------- /emissions ----------

class EmissionFactorOut(BaseModel):
    annual_tco2e_per_ha_per_yr: float
    embodied_tco2e_per_ha: float
    annual_source: str
    embodied_source: str
    annual_notes: str
    embodied_notes: str
    uncertainty: str


class EmissionsTableResponse(BaseModel):
    factors: dict[str, EmissionFactorOut]
    sources: dict[str, str]
    sign_convention: str


# ---------- /simulate ----------

class SimulateRequest(BaseModel):
    current_percentages: dict[str, float]
    total_area_ha: float = Field(..., gt=0)
    from_class: str
    to_class: str
    fraction: float = Field(..., ge=0.0, le=1.0)

    @field_validator("current_percentages")
    @classmethod
    def _pct_sum_sane(cls, v: dict[str, float]) -> dict[str, float]:
        total = sum(v.values())
        if not (95.0 <= total <= 105.0):
            raise ValueError(
                f"current_percentages must sum to ~100, got {total:.2f}"
            )
        return v


class SimulateResponse(BaseModel):
    before: EmissionsReport
    after: EmissionsReport
    delta_annual_tco2e_per_yr: float
    delta_embodied_tco2e: float
    converted_area_ha: float
    narrative: str


# ---------- /agent/report (Phase 8) ----------

class AgentReportRequest(BaseModel):
    """
    Input to /agent/report.

    The frontend calls /classify first, keeps the result in its own state,
    and then posts that result back here alongside the user's query. This
    keeps the endpoint stateless (no server-side session cache) and avoids
    re-running inference for follow-up questions about the same image.
    """
    percentages: dict[str, float]          # 0-100, from ClassifyResponse.percentages
    emissions: EmissionsReport             # echoed back from ClassifyResponse.emissions
    total_area_ha: float = Field(..., gt=0)
    image_label: str = ""                  # optional label the agent may surface
    query: str = Field(..., min_length=1)  # the user's natural-language question
    max_turns: Optional[int] = Field(
        default=None, ge=1, le=10,
        description="Override the agent's hard turn cap. Defaults to the agent's own default (5).",
    )

    @field_validator("percentages")
    @classmethod
    def _pct_sum_sane(cls, v: dict[str, float]) -> dict[str, float]:
        total = sum(v.values())
        if not (95.0 <= total <= 105.0):
            raise ValueError(
                f"percentages must sum to ~100, got {total:.2f}"
            )
        return v


class ToolCallOut(BaseModel):
    """One tool invocation by the agent, surfaced for UI rendering."""
    turn: int
    name: str
    input: dict[str, Any]
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class AgentReportResponse(BaseModel):
    """Mirrors agent.base.AgentReport."""
    final_text: str
    tool_calls: list[ToolCallOut] = Field(default_factory=list)
    turns_used: int
    stop_reason: str
    usage: dict[str, Any] = Field(default_factory=dict)


# ---------- errors ----------

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
