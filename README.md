# USDAhackathon2026April

Two geospatial-AI projects living in one repo. Both share a FastAPI shell, a
Claude Haiku reasoning agent, and a single-page frontend; the modeling
pipelines and data layers are otherwise independent.

| | **v1 — Terra** | **v2 — Corn Yield Forecasting** |
|---|---|---|
| **Status** | ✅ Shipped (pre-hackathon build complete, 2026-04-24) | 🚧 In progress — Phases A, B, C done; D.1 (Prithvi) underway |
| **Input** | a satellite image (PNG/JPG, ~1024² @ 0.3m GSD) | a `(state, year, forecast_date)` tuple |
| **Output** | per-class land breakdown + grounded GHG estimate + cited sustainability report | county-rolled-up state yield (bu/acre) + cone of uncertainty + analog-driven narrative |
| **Models** | SegFormer-B1 (fine-tuned on LoveDA) + Claude Haiku 4.5 | XGBoost per-forecast-date regressors + Prithvi-EO-2.0-300M (frozen, planned) + Claude Haiku 4.5 |
| **Brief** | Hackathon theme: Land Use & Sustainability | CSU Geospatial AI Crop Yield Forecasting brief (Iowa, Colorado, Wisconsin, Missouri, Nebraska; 2005–2024) |
| **Backend hooked up?** | yes — 5 FastAPI endpoints + `/ui` | not yet — Phases E/F/G |

The two projects are kept deliberately separate. v1's source of truth is
`PROJECT_PLAN.md` / `PHASE_PLAN.md` / `DECISIONS_LOG.md` / `CURRENT_STATE.md`
/ `README.md`. v2 mirrors that with a `PHASE2_*.md` prefix. **When a v2 doc
and `PHASE2_DECISIONS_LOG.md` disagree, the decisions log wins.**

---

## Why both projects share design DNA

A theme runs through both:

1. **Measurement is its own thing.** The vision model (v1) and the regressor +
   analog retriever (v2) produce numbers, with documented training procedures
   and reportable metrics.
2. **Grounding is its own thing.** Emissions factors (v1) cite IPCC AR6 / EPA /
   EDGAR with `[SRC-n]` tags inline in the code. NASS yield, gSSURGO soils,
   gridMET weather, USDM drought (v2) each pass through a `*_pull.py` /
   `*_features.py` pair with full schema and provenance.
3. **The LLM presents, it does not compute.** Claude Haiku 4.5 has tool access
   to read measurements and cite them — never to invent a number that wasn't
   measured. Tools surface model caveats; the system prompt instructs the
   model to own uncertainty rather than paper over it.

This separation is the point. The agent is a presenter over measured data,
not an oracle.

---

## Repo layout

```
USDAhackathon2026April/
  agent/                      v1 reasoning agent (Phase 7)
    base.py                     ReasoningAgent Protocol + dataclasses
    tools.py                    AgentState + 4 tool schemas + dispatcher
    claude.py                   ClaudeAgent — Haiku 4.5 tool loop
  backend/                    v1 FastAPI app (Phase 6 + 8)
    main.py                     5 endpoints, lifespan, /ui mount
    inference.py                InferenceEngine (SegFormer-B1 singleton)
    models.py                   Pydantic request/response schemas
  frontend/                   v1 single-page UI (Phase 8)
    index.html                  instrument-panel layout
    app.js                      fetch wiring, markdown renderer
    demos/                      place 3546.png and 2523.png here
  forecast/                   v2 modeling package (Phase B / C)
    data.py                     master-table loader + train/val/holdout splits
    features.py                 EMBEDDING_COLS + standardizer + VALID_FORECAST_DATES
    analog.py                   K-NN analog retrieval over the embedding
    cone.py                    percentile band over analog yields
    aggregate.py                county → state planted-acres-weighted rollup
    baseline.py                 5-yr county-mean naive (gate reference)
    detrend.py                  per-county trend + state-median fallback
    recalibrate.py              optional per-(state, date) bias correction
    regressor.py                XGBoost RegressorBundle (one model per forecast date)
    explain.py                  SHAP wrappers
  scripts/                    v1 + v2 utility scripts and pipelines
    # v1 — vision + emissions
    train.py                    SegFormer-B1 fine-tune
    dataset.py                  LoveDADataset + albumentations
    infer.py                    one-shot inference CLI
    emissions.py                LAND_USE_EMISSIONS table + simulate_intervention
    smoke_tools.py              tool-level smoke harness
    smoke_agent.py              agent-loop smoke harness
    # v2 — corn yield pipelines
    nass_pull.py / nass_features.py            NASS QuickStats yields, 2005–2024
    ndvi_county_extraction.js                  GEE script (corn-masked MODIS)
    gssurgo_county_features.py                 gSSURGO Valu1 county aggregation
    gridmet_pull.py / weather_features.py      gridMET daily → per-cutoff features
    drought_features.py                        USDM weekly → per-cutoff features
    hls_features.py                            provisional state-level HLS VIs
    merge_all.py                               outer-join all 6 sources
    train_regressor.py                         Phase C training driver
    backtest_baseline.py                       Phase B gate harness
    backtest_phase_c.py                        Phase C gate harness
    diagnose_state_errors.py                   per-state residual breakdown
    diagnose_wi_overshoot.py                   targeted WI bias inspection
  data/                       v1 LoveDA tiles + masks (not committed)
  phase2/                     v2 raw + intermediate data
    cdl/                        binary corn masks (Phase D.1.a output)
    data/{drought,gSSURGO,hls,ndvi,tiger}/    raw inputs by source
  model/                      v1 SegFormer checkpoint
    segformer-b1-run1/
      best.pt                   epoch 10, val mIoU 0.5147
  models/                     v2 trained models
    forecast/regressor_*.json   XGBoost bundle (per forecast date)
  inference_outputs/          v1 per-image artifacts (masks, overlays)
  runs/                       v2 backtest CSVs (Phase B + C)
  docs/                       all plan + decision docs (v1 + PHASE2_*)
```

---

## v1 — Terra (Land-Use & GHG Analysis)

Satellite image → pixel-level land-cover segmentation → emissions estimate
grounded in a cited factor table → LLM-generated sustainability report with
inline citations.

### Architecture

```
                            ┌───────────────────────┐
                            │      FRONTEND /ui     │
                            │ single-page HTML + JS │
                            └──────────┬────────────┘
                                       │
                         HTTP (same origin, no CORS)
                                       │
          ┌────────────────────────────┴───────────────────────────┐
          │                  FASTAPI BACKEND                       │
          │   GET  /health        agent & device status            │
          │   POST /classify      image → mask + percentages       │
          │   POST /simulate      counterfactual deltas            │
          │   GET  /emissions     full cited factor table          │
          │   POST /agent/report  Claude-generated report          │
          └──┬───────────────────┬──────────────────────────────┬──┘
             │                   │                              │
   ┌─────────▼─────────┐ ┌───────▼───────────┐ ┌────────────────▼──────────┐
   │ InferenceEngine   │ │  emissions.py     │ │ ClaudeAgent (Haiku 4.5)   │
   │ SegFormer-B1, CUDA│ │  LAND_USE_        │ │ 4 tools, AgentState       │
   │ ~300ms / 1024² tile│ │  EMISSIONS table │ │ max_turns = 5             │
   └───────────────────┘ └───────────────────┘ └───────────────────────────┘
        measurement              grounding                 reasoning
```

### Quickstart (v1)

**Requirements:** Linux + NVIDIA GPU + CUDA (CPU works but ~5s/image instead
of ~300ms), conda, an Anthropic API key.

```bash
# 1. Install
conda create -n landuse python=3.11 -y
conda activate landuse
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. Drop your key in .env
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# 3. Stage the demo tiles
mkdir -p frontend/demos
cp data/loveda/Val/Urban/images_png/3546.png frontend/demos/
cp data/loveda/Val/Rural/images_png/2523.png frontend/demos/

# 4. Run (serves API + UI on port 8000)
uvicorn backend.main:app --app-dir . --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. Three green status pills (`Backend ok · Device
cuda · Agent ready`) means everything is wired. Click **Demo · 3546 (forest)**
for a clean run; **Demo · 2523** is a pathological case the agent flags as a
model-quality caveat.

If you don't have a checkpoint, train one (~50 min on RTX 4070 Ti SUPER):

```bash
bash scripts/download_loveda.sh
python scripts/train.py \
  --data-root data/loveda \
  --output-dir model/segformer-b1-run1 \
  --epochs 15 --batch-size 2 --grad-accum 4 --lr 6e-5
```

### Vision model performance (val mIoU 0.5147, 7-class)

| class | IoU | quality |
|---|---|---|
| water | 0.667 | strong |
| agriculture | 0.596 | strong |
| building | 0.570 | strong |
| road | 0.544 | usable |
| background | 0.497 | usable |
| forest | 0.385 | weak — surfaces as caveat |
| barren | 0.346 | weak — surfaces as caveat |

Forest and barren are weak by design (LoveDA class imbalance, B1 over B2 for
iteration speed on 16GB VRAM). The agent's tools are built to surface these
caveats, not bury them.

### Emissions grounding

Six primary-authority sources, tagged inline in `scripts/emissions.py`:

| Tag | Source |
|---|---|
| SRC-1 | IPCC (2022). AR6 WGIII Chapter 7: AFOLU |
| SRC-2 | EPA (2024). U.S. GHG Inventory 1990–2022 |
| SRC-3 | U.S. EIA (2022). 2018 CBECS |
| SRC-4 | IPCC (2019). 2019 Refinement, Vol 4: AFOLU |
| SRC-5 | JRC EFIResources (2018). Building benchmarks |
| SRC-6 | Crippa et al. (2023). EDGAR v7.0 |

| class | annual tCO2e/ha/yr | embodied tCO2e/ha |
|---|---:|---:|
| building | +65.0 | +600 |
| road | +4.0 | +220 |
| water | 0.0 | 0 |
| barren | 0.0 | +50 |
| forest | −8.0 | −800 |
| agriculture | +2.5 | +150 |

Sign convention: positive = net emitter / released-on-disturbance; negative =
net sequestration / stored stock.

### v1 API

| Endpoint | Behavior |
|---|---|
| `GET /health` | `{status, device, agent_available}` |
| `POST /classify` | image → mask PNG (base64) + percentages + per-class emissions |
| `GET /emissions` | the full cited factor table |
| `POST /simulate` | `{from_class, to_class, fraction}` → annual + embodied deltas |
| `POST /agent/report` | classified state + query → cited sustainability report |
| `GET /ui/` | Terra single-page app |

Swagger at `/docs`. `/agent/report` is intentionally stateless: the frontend
sends back `{percentages, emissions, total_area_ha}` so follow-up queries
don't re-run inference.

### v1 costs

Pre-build budget: $10 of Anthropic credit. Total spent: **~$0.25**. Per
end-to-end demo run: ~$0.015–0.025 on Haiku 4.5. Vision is free (local CUDA).

---

## v2 — Corn Yield Forecasting

A `(state, year, forecast_date)` tuple → a county-level point estimate, an
analog-derived cone of uncertainty, and a Claude-narrated forecast. Outputs
roll up to the state via planted-acres-weighted aggregation.

**Forecast dates** (locked by the brief): Aug 1, Sep 1, Oct 1, end-of-season.
**States:** IA, CO, WI, MO, NE. **Years:** 2005–2024 (train 2005–2022, val
2023, holdout 2024).

### Pipeline

```
   ┌────────────────────────────────────────────────────────────────┐
   │            scripts/training_master.parquet                     │
   │            25,872 rows × 48 columns                            │
   │            grain: (GEOID, year, forecast_date)                 │
   └────────────────────────────────────────────────────────────────┘
       ▲           ▲          ▲          ▲           ▲          ▲
       │           │          │          │           │          │
   NASS yield   MODIS NDVI  gSSURGO   gridMET     USDM       HLS VIs
   (truth +     (corn-     (Valu1   (GDD/EDD/   drought    (provisional;
   engineered    masked    soil     VPD/precip/   weekly     redone in D.1)
   features)    via CDL)   static)  srad)         → per-cutoff
                                    per-cutoff
                                                                          │
                                                                          ▼
       ┌─────────────────────────┬────────────────────────────┐
       │  Phase B — analog cone  │  Phase C — XGBoost regressor│
       │  K-NN over standardized │  one model per forecast date│
       │  embedding; same_geoid  │  county-level prediction    │
       │  pool; K=5; (10, 50, 90)│  early-stopped on val=2023  │
       └────────────┬────────────┴───────────────┬─────────────┘
                    │                            │
                    └─────────────┬──────────────┘
                                  ▼
                  forecast.aggregate (planted-acres-weighted)
                                  │
                                  ▼
                       state-level point + cone
                                  │
                                  ▼
                    Claude Haiku narrative (Phase F, planned)
```

### Status

| Phase | Goal | Status |
|---|---|---|
| **A** | 6 data pipelines → master table (25,872 × 48) | ✅ closed; full data dictionary in `docs/PHASE2_DATA_DICTIONARY.md` |
| **B** | Analog-year retrieval baseline + cone-of-uncertainty MVP | ✅ gate passed: cone in [70%, 90%] coverage band; analog-median beats 5-yr-mean naive |
| **C** | XGBoost per-forecast-date regressors | ✅ gate passed: **+46.7% RMSE improvement** vs Phase B analog-median at EOS (threshold: +15%). NDVI columns stripped after SHAP showed `ndvi_peak` leaking end-of-season info into Aug forecasts |
| **D.1** | Prithvi as frozen feature extractor over HLS chips | 🚧 D.1.a done (60 CDL corn masks, 5 states × 2013–2024). D.1.b (HLS pull + chip extraction) is up next |
| **D.2** | End-to-end Prithvi fine-tune | stretch — contingent on D.1 ablation lift |
| **E** | Backend endpoints (`/forecast/*`) | not started |
| **F** | Agent tools for forecast narration | not started |
| **G** | Validation, ablations, presentation | not started |

### Data sources

| Dataset | Source | Granularity | Status |
|---|---|---|---|
| NASS corn yield + acres | USDA NASS QuickStats | (GEOID, year) | 🟢 6,837 × 16 raw, 6,834 × 10 engineered |
| MODIS NDVI (corn-masked via CDL) | Earth Engine `MOD13Q1` | (GEOID, year) | 🟢 21 per-year CSVs, 2004–2024 |
| gSSURGO Valu1 (NCCPI, AWS, SOC, root-zone, droughty) | USDA NRCS via SSURGO Portal | (GEOID,) static | 🟢 443 × 13 |
| gridMET daily weather → derived per-cutoff | gridMET | (GEOID, year, forecast_date) | 🟢 35,440 × 14 |
| US Drought Monitor weekly | NDMC / USDM | (GEOID, year, forecast_date) | 🟢 27,336 × 9, zero nulls |
| HLS L30/S30 | NASA LP DAAC | provisional state-level VIs | 🟡 redone county-level in Phase D.1 |
| CDL annual corn masks | NASS CropScape | 30 m raster, per state-year | 🟢 60 masks (5 states × 2013–2024) |
| Prithvi-EO-2.0-300M-TL | NASA / IBM HuggingFace | 600 M params | 🔴 to download in D.1.d |
| NAIP aerial imagery | USDA FSA | sub-meter | ⚫ excluded — see `PHASE2_DECISIONS_LOG.md` for rationale |

### Master table schema (excerpt)

`scripts/training_master.parquet` — 25,872 rows × 48 columns, full-coverage
rate 98.1% (excluding HLS, which is intentionally redone in D.1). Each row is
one `(GEOID, year, forecast_date)`. Full column-by-column reference in
`docs/PHASE2_DATA_DICTIONARY.md`.

Key columns:

- **Target.** `yield_target` — combined-practice corn-grain bu/acre from NASS.
- **Static soils.** `nccpi3corn`, `aws0_100`, `soc0_30`, `rootznaws`,
  `droughty`, `pwsl1pomu` (from gSSURGO Valu1).
- **Per-cutoff weather.** `gdd_cum_f50_c86`, `edd_hours_gt86f`,
  `edd_hours_gt90f`, `vpd_kpa_{veg,silk,grain}`, `prcp_cum_mm`,
  `dry_spell_max_days`, `srad_total_{veg,silk,grain}` (from gridMET).
- **Per-cutoff drought.** USDM-derived severity-weighted features.
- **Engineered NASS.** `irrigated_share`, `harvest_ratio`,
  `acres_harvested_noirr_derived`.

### Reproducing Phase B + C

```bash
# Pull and engineer all six sources (Phase A — slow, mostly already on disk)
python scripts/nass_pull.py
python scripts/nass_features.py
python scripts/gridmet_pull.py
python scripts/weather_features.py
python scripts/drought_features.py
python scripts/gssurgo_county_features.py
# (NDVI pulls run server-side via Earth Engine — see scripts/ndvi_county_extraction.js)

# Outer-join all six into the master parquet
python scripts/merge_all.py

# Phase B — analog-retrieval baseline + cone calibration
python -m scripts.backtest_baseline

# Phase C — train XGBoost regressors (per forecast date) and score gate
python -m scripts.train_regressor
python -m scripts.backtest_phase_c
```

Phase C gate verdict at EOS, val=2023:

```
EOS rmse_regressor:     ~9–10 bu/acre
EOS rmse_analog_median: ~17–18 bu/acre
EOS lift vs analog:     +46.7%   (threshold ≥ 15%)
PHASE C GATE: PASS
```

### Important v2 caveats

- **NDVI is currently unused by the regressor.** SHAP attribution at the end of
  Phase C showed `ndvi_peak` dominating predictions across every forecast
  date — it was leaking late-season info into August forecasts because the
  five MODIS NDVI columns were derived without per-cutoff masking. Phase D.1
  re-introduces remote-sensing signal via Prithvi over HLS chips, with strict
  per-cutoff temporal slicing. Until D.1 lands, the regressor has zero
  remote-sensing features.
- **HLS coverage starts in 2013.** Phase D.1 narrows the train pool to
  2013–2022 by design; the 2013-cutoff ablation gate test (4-row table) is
  what decides whether Prithvi features are kept.
- **NAIP is intentionally excluded.** Wrong sensor for Prithvi (HLS-pretrained),
  wrong cadence for in-season forecasting (2–3-yr per state, growing-season
  acquisition), wrong job for corn masking (CDL is purpose-built). Reasoning
  documented in `PHASE2_DECISIONS_LOG.md` so reviewers don't ask.

---

## Working with the project docs

For each project, the docs are layered:

| File | Purpose |
|---|---|
| `PROJECT_PLAN.md` / `PHASE2_PROJECT_PLAN.md` | the vision; what we're building and why |
| `PHASE_PLAN.md` / `PHASE2_PHASE_PLAN.md` | phase breakdown with go/no-go gates |
| `DECISIONS_LOG.md` / `PHASE2_DECISIONS_LOG.md` | chronological decision record (the source of truth) |
| `CURRENT_STATE.md` / `PHASE2_CURRENT_STATE.md` | what's in the repo right now (overwritten each phase end) |
| `PHASE2_DATA_INVENTORY.md` | v2-only: data status tracker |
| `PHASE2_DATA_DICTIONARY.md` | v2-only: column-by-column master-table reference |

When picking up a new chat session: read the project plan, phase plan,
decisions log, and current state — in that order — before doing anything else.

---

## Known gotchas

- **`torch.load` needs `weights_only=False`** on v1 checkpoints (they carry
  pathlib Path objects in saved args). The loader already handles this.
- **`python-multipart` is a separate install** from `fastapi`. In the deps
  list but easy to miss when cherry-picking.
- **Uvicorn must launch with `--app-dir .`** (or `PYTHONPATH=.`) from the repo
  root so the `backend` package imports correctly.
- **Prompt caching is inactive on Haiku 4.5** for v1: system prompt + tool
  schemas total ~1,500 tokens, below the 4,096-token caching floor. The
  `cache_control` markers are accepted silently. Budget impact is trivial.
- **Vision model is weakest on forest and barren.** Surfaced in every v1 agent
  report through `model_caveats`. Don't hide it.
- **MODIS NDVI is pre-scaled** (× 0.0001) server-side in the GEE script. Do
  NOT re-scale in Python.
- **NASS rows < 2005 are dropped at load time** in `forecast.data.load_master`
  (`min_year=2005`). The NDVI pull incidentally produced a 2004 block, which
  without the filter inherits 1,464 structural-NaN rows from missing weather
  features. Coverage rate jumped 92.8% → 98.1% after the fix.
- **`num_workers=0`** during v1 SegFormer training — there's an intermittent
  DataLoader worker segfault on Ubuntu. Data loading isn't the bottleneck.

---

## Acknowledgements

**v1**
- LoveDA (Wang et al. 2021) — semantic segmentation benchmark, Zenodo 5706578.
- SegFormer (Xie et al. 2021) — NVIDIA's efficient transformer architecture.
- Emissions factors from IPCC, U.S. EPA, U.S. EIA, JRC, EDGAR — cited inline
  in `scripts/emissions.py`.

**v2**
- USDA NASS QuickStats — corn yield ground truth.
- USDA NRCS gSSURGO — county-aggregated soil features (Valu1 table).
- Earth Engine + MODIS MOD13Q1 — NDVI features.
- gridMET (Climatology Lab, U. Idaho) — daily gridded weather.
- US Drought Monitor (NDMC / USDA / NOAA) — weekly drought severity.
- NASA LP DAAC HLS L30/S30 — Harmonized Landsat-Sentinel imagery.
- USDA NASS CDL — Cropland Data Layer annual corn masks.
- Prithvi-EO-2.0-300M-TL (NASA / IBM, HuggingFace) — geospatial foundation
  model (Phase D.1).
- CSU Geospatial AI Crop Yield Forecasting brief — defines the v2 problem.

**Both projects**
- Anthropic Claude Haiku 4.5 — reasoning agent, public Messages API with tool
  use.

---

## License

Add a license file appropriate to your distribution intent. The data sources
above each carry their own terms (NASS QuickStats, gSSURGO, gridMET, USDM, and
HLS are all open / public-domain in the U.S.; LoveDA is CC BY-NC-SA 4.0 — note
the non-commercial clause if redistributing v1 weights).
