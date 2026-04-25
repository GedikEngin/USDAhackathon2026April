"""
FastAPI backend for the Land Use & GHG Analysis System
+ v2 Yield Forecast endpoints.

Run (from repo root):
  uvicorn backend.main:app --app-dir . --host 0.0.0.0 --port 8000
Swagger:
  http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env so ANTHROPIC_API_KEY is available at startup when the user
# runs uvicorn from the repo root without pre-exporting it.
load_dotenv(PROJECT_ROOT / ".env")

from backend.inference import InferenceEngine  # noqa: E402
from backend.models import (  # noqa: E402
    AgentReportRequest,
    AgentReportResponse,
    ClassifyResponse,
    EmissionFactorOut,
    EmissionsReport,
    EmissionsTableResponse,
    PerClassEmissions,
    SimulateRequest,
    SimulateResponse,
    ToolCallOut,
)
from scripts.emissions import (  # noqa: E402
    LAND_USE_EMISSIONS,
    SOURCES,
    EmissionsResult,
    cite,
    simulate_intervention,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("backend")

DEFAULT_CHECKPOINT = PROJECT_ROOT / "model" / "segformer-b1-run1" / "best.pt"

# v2 forecast defaults. Both overridable via env vars to enable hot-swap (D.1
# bundle, alternative master parquet) without touching code.
DEFAULT_FORECAST_BUNDLE_DIR = PROJECT_ROOT / "models" / "forecast"
DEFAULT_FORECAST_MASTER     = PROJECT_ROOT / "scripts" / "training_master_v2.parquet"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- vision model (v1) ----
    ckpt_path = Path(os.environ.get("LANDUSE_CHECKPOINT", DEFAULT_CHECKPOINT))
    log.info("Starting up: loading model from %s", ckpt_path)
    app.state.engine = InferenceEngine(checkpoint_path=ckpt_path)
    log.info("Model loaded")

    # ---- reasoning agent (v1; optional at startup) ----
    app.state.agent = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from agent.claude import ClaudeAgent  # local: anthropic optional
            app.state.agent = ClaudeAgent()
            log.info(
                "Reasoning agent ready (model=%s, max_turns=%d)",
                app.state.agent.model, app.state.agent.max_turns,
            )
        except Exception as exc:
            log.warning("Agent unavailable: %s", exc)
    else:
        log.warning(
            "ANTHROPIC_API_KEY not set — /agent/report will return 503. "
            "Other endpoints are unaffected."
        )

    # ---- v2 forecast artifacts ----
    # Loaded once. The route handlers are pure read-only over these objects,
    # so no locking is needed under FastAPI's single-process default.
    app.state.forecast_loaded = False
    forecast_bundle_dir = Path(
        os.environ.get("FORECAST_BUNDLE_DIR", str(DEFAULT_FORECAST_BUNDLE_DIR))
    )
    forecast_master_path = Path(
        os.environ.get("FORECAST_MASTER_PATH", str(DEFAULT_FORECAST_MASTER))
    )
    try:
        log.info("Loading forecast bundle from %s", forecast_bundle_dir)
        log.info("Loading forecast master parquet %s", forecast_master_path)

        # Local imports — keep boot light if forecast deps are absent.
        from forecast.analog import AnalogIndex
        from forecast.data import load_master, train_pool
        from forecast.detrend import fit as fit_trend
        from forecast.features import fit_standardizer
        from forecast.regressor import RegressorBundle
        from backend.forecast_routes import (
            build_county_name_lookup, build_history_lookup,
        )

        master_df = load_master(str(forecast_master_path))
        train_df, mh_result = train_pool(master_df, n_min_history=10)
        standardizer = fit_standardizer(train_df)
        trend = fit_trend(train_df)
        analog_index = AnalogIndex.fit(train_df, standardizer, trend)
        bundle = RegressorBundle.load(str(forecast_bundle_dir))

        # Read VERSION.txt next to the bundle, default if absent.
        version_path = forecast_bundle_dir / "VERSION.txt"
        if version_path.exists():
            model_version = version_path.read_text().strip()
        else:
            model_version = "unversioned"
            log.warning("No VERSION.txt at %s; model_version='unversioned'.", version_path)

        history_lookup = build_history_lookup(master_df)
        county_name_lookup = build_county_name_lookup(master_df)

        # Stash on app.state for the routes.
        app.state.master_df = master_df
        app.state.train_df = train_df
        app.state.standardizer = standardizer
        app.state.trend = trend
        app.state.analog_index = analog_index
        app.state.bundle = bundle
        app.state.model_version = model_version
        app.state.history_lookup = history_lookup
        app.state.county_name_lookup = county_name_lookup
        app.state.forecast_loaded = True

        log.info(
            "Forecast ready: bundle=%s, master=%s rows=%d, train=%d rows, %d GEOIDs kept",
            model_version, forecast_master_path.name, len(master_df), len(train_df),
            mh_result.n_kept,
        )
    except Exception as exc:
        log.exception("Forecast artifacts unavailable: %s", exc)
        log.warning(
            "/forecast/* will return 503 until lifespan succeeds. Other endpoints unaffected."
        )

    yield
    log.info("Shutting down")


app = FastAPI(
    title="Land Use & GHG Analysis + Yield Forecast",
    description=(
        "v1: satellite tile -> land-class segmentation -> emissions estimate -> "
        "LLM-generated sustainability report.\n"
        "v2: county-level corn yield forecast for 5 states (CO, IA, MO, NE, WI) "
        "at 4 forecast dates, with analog-year cones of uncertainty and Claude-narrated "
        "interpretation."
    ),
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# v2 forecast router. Endpoints: /forecast/states, /forecast/{state}, /forecast/narrate.
# Returns 503 from the route layer if the forecast lifespan didn't load.
from backend.forecast_routes import router as forecast_router  # noqa: E402
app.include_router(forecast_router)

# Serve the single-page frontend at /ui if it exists.
_frontend_dir = PROJECT_ROOT / "frontend"
if _frontend_dir.is_dir():
    app.mount(
        "/ui",
        StaticFiles(directory=str(_frontend_dir), html=True),
        name="ui",
    )


def _emissions_result_to_report(result: Any) -> EmissionsReport:
    """
    Convert an EmissionsResult dataclass into the pydantic EmissionsReport.

    Source citations are looked up from LAND_USE_EMISSIONS, since emissions.py
    does not duplicate source strings onto each per_class row.
    """
    d = asdict(result)
    per_class = {
        name: PerClassEmissions(**row) for name, row in d["per_class"].items()
    }

    src_keys: set[str] = set()
    for class_name in d["per_class"].keys():
        factor = LAND_USE_EMISSIONS.get(class_name)
        if factor is not None:
            src_keys.add(factor.annual_source)
            src_keys.add(factor.embodied_source)
    sources_cited = {k: cite(k) for k in sorted(src_keys)}

    return EmissionsReport(
        per_class=per_class,
        excluded_breakdown=d.get("excluded_breakdown", {}),
        total_area_ha=d["total_area_ha"],
        assessed_area_ha=d["assessed_area_ha"],
        excluded_fraction=d["excluded_fraction"],
        total_annual_tco2e_per_yr=d["total_annual_tco2e_per_yr"],
        total_embodied_tco2e=d["total_embodied_tco2e"],
        sources_cited=sources_cited,
    )


def _rehydrate_emissions_result(report: EmissionsReport) -> EmissionsResult:
    """Reverse of _emissions_result_to_report."""
    per_class: dict[str, dict[str, float]] = {}
    for name, row in report.per_class.items():
        per_class[name] = {
            "pixel_count": row.pixel_count,
            "pixel_fraction": row.pixel_fraction,
            "area_ha": row.area_ha,
            "annual_tco2e": row.annual_tco2e,
            "embodied_tco2e": row.embodied_tco2e,
        }
    return EmissionsResult(
        per_class=per_class,
        total_area_ha=report.total_area_ha,
        assessed_area_ha=report.assessed_area_ha,
        total_annual_tco2e_per_yr=report.total_annual_tco2e_per_yr,
        total_embodied_tco2e=report.total_embodied_tco2e,
        excluded_fraction=report.excluded_fraction,
        excluded_breakdown=dict(report.excluded_breakdown),
    )


@app.get("/", tags=["meta"])
def root():
    if (PROJECT_ROOT / "frontend" / "index.html").exists():
        return RedirectResponse(url="/ui/")
    return {
        "service": "land-use-ghg + yield-forecast",
        "endpoints": [
            "/classify", "/emissions", "/simulate", "/agent/report",
            "/forecast/states", "/forecast/{state}", "/forecast/narrate",
            "/docs",
        ],
    }


@app.get("/health", tags=["meta"])
def health():
    engine = getattr(app.state, "engine", None)
    agent = getattr(app.state, "agent", None)
    forecast_loaded = bool(getattr(app.state, "forecast_loaded", False))
    return {
        "status": "ok" if engine is not None else "loading",
        "device": str(engine.device) if engine else None,
        "agent_available": agent is not None,
        "forecast_available": forecast_loaded,
        "model_version": getattr(app.state, "model_version", None),
    }


@app.post("/classify", response_model=ClassifyResponse, tags=["inference"])
async def classify(
    file: UploadFile = File(..., description="Satellite image (PNG or JPG)"),
    tta: bool = Form(True, description="Enable test-time augmentation"),
    pixel_size_m: float = Form(0.3, description="Ground sample distance in meters per pixel"),
):
    engine: InferenceEngine = app.state.engine
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty upload")

    try:
        result = engine.classify(image_bytes, tta=tta, pixel_size_m=pixel_size_m)
    except Exception as exc:
        log.exception("classify() failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    return ClassifyResponse(
        percentages=result.percentages,
        emissions=_emissions_result_to_report(result.emissions),
        mask_png_base64=result.mask_png_base64,
        inference_ms=result.inference_ms,
        input_shape=result.input_shape,
        warnings=result.warnings,
    )


@app.get("/emissions", response_model=EmissionsTableResponse, tags=["grounding"])
def emissions_table():
    factors = {}
    for class_name, factor in LAND_USE_EMISSIONS.items():
        factors[class_name] = EmissionFactorOut(
            annual_tco2e_per_ha_per_yr=factor.annual_tco2e_per_ha_per_yr,
            embodied_tco2e_per_ha=factor.embodied_tco2e_per_ha,
            annual_source=factor.annual_source,
            embodied_source=factor.embodied_source,
            annual_notes=factor.annual_notes,
            embodied_notes=factor.embodied_notes,
            uncertainty=factor.uncertainty,
        )
    return EmissionsTableResponse(
        factors=factors,
        sources=dict(SOURCES),
        sign_convention=(
            "Annual flux: positive = net emitter, negative = net sink. "
            "Embodied: positive = stored stock (forest biomass) or released-on-disturbance cost."
        ),
    )


@app.post("/simulate", response_model=SimulateResponse, tags=["counterfactual"])
def simulate(req: SimulateRequest):
    try:
        sim = simulate_intervention(
            current_percentages=req.current_percentages,
            total_area_ha=req.total_area_ha,
            from_class=req.from_class,
            to_class=req.to_class,
            fraction=req.fraction,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log.exception("simulate_intervention failed")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {exc}")

    return SimulateResponse(
        before=_emissions_result_to_report(sim.before),
        after=_emissions_result_to_report(sim.after),
        delta_annual_tco2e_per_yr=sim.delta_annual_tco2e_per_yr,
        delta_embodied_tco2e=sim.delta_embodied_tco2e,
        converted_area_ha=sim.converted_area_ha,
        narrative=sim.narrative,
    )


@app.post("/agent/report", response_model=AgentReportResponse, tags=["agent"])
def agent_report(req: AgentReportRequest):
    """
    Run the reasoning agent against a pre-classified image.

    Stateless. Frontend echoes the /classify result back; we re-inflate the
    EmissionsResult dataclass and run the agent loop.
    """
    agent = getattr(app.state, "agent", None)
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Reasoning agent is not available on this server. "
                "Set ANTHROPIC_API_KEY and restart to enable /agent/report."
            ),
        )

    from agent.tools import AgentState  # noqa: E402

    state = AgentState(
        percentages=dict(req.percentages),
        emissions=_rehydrate_emissions_result(req.emissions),
        total_area_ha=req.total_area_ha,
        image_label=req.image_label,
    )

    if req.max_turns is not None and req.max_turns != agent.max_turns:
        from agent.claude import ClaudeAgent  # noqa: E402
        run_agent = ClaudeAgent(
            client=agent.client,
            model=agent.model,
            max_turns=req.max_turns,
            max_tokens=agent.max_tokens,
        )
    else:
        run_agent = agent

    try:
        report = run_agent.run(state, req.query)
    except Exception as exc:
        log.exception("agent.run failed")
        raise HTTPException(status_code=500, detail=f"Agent failed: {exc}")

    tool_calls = [
        ToolCallOut(
            turn=tc.turn,
            name=tc.name,
            input=dict(tc.input),
            result=dict(tc.result) if tc.result is not None else None,
            error=tc.error,
        )
        for tc in report.tool_calls
    ]

    return AgentReportResponse(
        final_text=report.final_text,
        tool_calls=tool_calls,
        turns_used=report.turns_used,
        stop_reason=report.stop_reason,
        usage=dict(report.usage),
    )
