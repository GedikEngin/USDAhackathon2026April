# Terra — Land-Use & Greenhouse-Gas Analysis

Satellite image → pixel-level land-cover segmentation → emissions estimate grounded
in a cited factor table → LLM-generated sustainability report with inline citations.

A portfolio project built over two weekends as pre-hackathon infrastructure.
The vision model and the emissions table are both fully measurable, auditable
artifacts; the LLM is a *consumer* of those measurements, not a source of
numbers. Every figure in a generated report traces to a source in the code.

![Terra UI — 3546 demo tile, full loop rendered.](docs/preview.jpg)

*Place the screenshot of the rendered demo at `docs/screenshot-3546.png` so
this image link resolves.*

---

## Architecture

```
                                ┌───────────────────────┐
                                │      FRONTEND /ui     │
                                │ single-page HTML + JS │
                                │ upload → preview →    │
                                │ classify → analyze    │
                                └──────────┬────────────┘
                                           │
                             HTTP (same origin, no CORS)
                                           │
              ┌────────────────────────────┴───────────────────────────┐
              │                      FASTAPI BACKEND                   │
              │   GET  /health      agent & device status pills        │
              │   POST /classify    multipart image upload →           │
              │                       colored mask PNG + percentages   │
              │                       + per-class emissions rows       │
              │   POST /simulate    pure counterfactual calculator     │
              │   GET  /emissions   the full cited factor table        │
              │   POST /agent/report  ← runs ClaudeAgent on preclass   │
              │                                                        │
              │   /ui/*    static frontend (served at startup)         │
              └──┬───────────────────┬──────────────────────────────┬──┘
                 │                   │                              │
     ┌───────────▼─────────┐ ┌───────▼───────────┐ ┌────────────────▼──────────┐
     │  InferenceEngine    │ │  emissions.py     │ │  ClaudeAgent (agent/)     │
     │  SegFormer-B1, CUDA │ │  LAND_USE_        │ │  Haiku 4.5, max_turns=5   │
     │  loaded once at     │ │  EMISSIONS table  │ │  4 tools, AgentState      │
     │  startup, ~300ms    │ │  + compute_       │ │  injected from /classify  │
     │  per 1024² tile     │ │  emissions() +    │ │                           │
     │                     │ │  simulate_        │ │  tool → emissions.py      │
     │  best.pt, epoch 10, │ │  intervention()   │ │  tool → AgentState        │
     │  val mIoU 0.5147    │ │                   │ │                           │
     └─────────────────────┘ └───────────────────┘ └───────────────────────────┘
          measurement              grounding                 reasoning
        (what's on the          (what each class            (narrative over
          ground)                costs or stores)            the measurements)
```

Three pillars, kept deliberately separate:

1. **Measurement** — a fine-tuned SegFormer-B1 classifies every pixel into one of
   eight land-cover classes. Runs once per image (~300ms on an RTX 4070 Ti SUPER).
   Produces pixel counts, not opinions.
2. **Grounding** — `scripts/emissions.py` maps each emissions-relevant class to a
   per-hectare annual flux and per-hectare embodied stock, each carrying an
   inline `[SRC-n]` citation to IPCC AR6 WGIII, EPA GHG Inventory, EIA CBECS,
   JRC EFIResources, IPCC 2019 Refinement, or EDGAR v7. Every number in the
   table has a derivation comment next to it.
3. **Reasoning** — Claude Haiku 4.5 has four tools that expose (a) the
   composition, (b) the emissions estimate with sources, (c) a counterfactual
   simulator, (d) a mechanical mitigation-candidate ranker. The model's job is
   to weigh tradeoffs, surface model-quality caveats honestly, and compose a
   readable report. It does not compute numbers.

This separation is the point of the project. The LLM is a presenter over
measured data, not an oracle.

---

## Quickstart

### Requirements

- **Linux** box with NVIDIA GPU + CUDA (tested on Ubuntu 24.04, RTX 4070 Ti SUPER).
  CPU-only will work but classify takes ~5s per image instead of 300ms.
- **Conda** (Miniconda/Anaconda). A `venv` would also work but conda handles the
  CUDA PyTorch install more reliably.
- **Anthropic API key** for the agent. Other endpoints work without one.
- **Trained checkpoint** at `model/segformer-b1-run1/best.pt`. If you're cloning
  from scratch and don't have one, see *Training from scratch* below.

### Two-command demo

```bash
# 1. Install dependencies (one-time)
conda create -n landuse python=3.11 -y
conda activate landuse
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt         # or see "Python deps" below

# 2. Put your key in .env (one-time)
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# 3. Run the backend (serves API + frontend on port 8000)
uvicorn backend.main:app --app-dir . --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000` in a browser. It redirects to `/ui/` and
the Terra interface loads. Click **Demo · 3546 (forest)** to run the full
pipeline on a pre-shipped LoveDA tile.

If you see three green pills at the top right — `Backend ok · Device cuda ·
Agent ready` — everything is wired. If `Agent` is amber ("Agent offline"),
your API key wasn't picked up; check `.env`.

### Demo images

Copy two LoveDA validation tiles into `frontend/demos/` so the demo buttons
have something to fetch:

```bash
mkdir -p frontend/demos
cp data/loveda/Val/Urban/images_png/3546.png frontend/demos/
cp data/loveda/Val/Rural/images_png/2523.png frontend/demos/
```

- **3546** — forest-dominant (72% forest). The clean "model succeeds" case.
  Produces a net-sink sustainability report.
- **2523** — pathological input. The vision model classifies mixed rural
  vegetation as background, surfacing zero forest. The agent correctly flags
  this as a model-quality caveat and labels the report a lower bound.

Arbitrary PNG/JPG uploads also work; LoveDA-style 1024×1024 tiles at ~0.3m
GSD will classify best.

### Python deps

If you don't have a `requirements.txt` yet:

```
anthropic
python-dotenv
fastapi
uvicorn[standard]
python-multipart
pydantic
transformers
accelerate
albumentations
opencv-python-headless
numpy
Pillow
# For training only:
datasets
matplotlib
seaborn
tqdm
```

---

## Repo layout

```
USDAhackathon2026April/
  agent/                              Reasoning agent (Phase 7)
    base.py                             ReasoningAgent Protocol + dataclasses
    tools.py                            AgentState + 4 tool schemas + dispatcher
    claude.py                           ClaudeAgent — Haiku 4.5 tool loop
  backend/                            FastAPI app (Phase 6 + Phase 8)
    main.py                             endpoints, lifespan, /ui mount
    inference.py                        InferenceEngine (SegFormer-B1 singleton)
    models.py                           Pydantic request/response schemas
  frontend/                           Single-page UI (Phase 8)
    index.html                          instrument-panel layout, CSS
    app.js                              fetch wiring, markdown renderer
    demos/                              place 3546.png and 2523.png here
  scripts/
    train.py                            SegFormer-B1 fine-tune pipeline
    dataset.py                          LoveDADataset + albumentations
    infer.py                            CLI inference (reference impl)
    emissions.py                        the cited factor table + simulator
    agent_repl.py                       terminal REPL for the agent
    dataset_stats.py                    Phase 1 class-balance analysis
  model/segformer-b1-run1/
    best.pt                             epoch 10 checkpoint, val mIoU 0.5147
    train_log.csv
  data/loveda/                        not in git — download via Zenodo
    Train/, Val/                        Urban + Rural splits
  smoke_tools.py                      offline test: 4 tools + dispatch
  smoke_agent.py                      offline test: agent loop w/ fake client
  .env                                (gitignored) holds ANTHROPIC_API_KEY
  PROJECT_PLAN.md                     original vision document
  PHASE_PLAN.md                       locked phase breakdown + checkpoints
  DECISIONS_LOG.md                    append-only record of choices made
  CURRENT_STATE.md                    end-of-phase snapshot of what exists
  HACKATHON_TODO.md                   tiered task list for the hackathon team
  README.md                           this file
```

---

## What the trained model can and can't do

The SegFormer-B1 fine-tune hit **val mIoU 0.5147** at epoch 10, over the
Saturday Weekend-1 gate of 0.38. Per-class IoU at best checkpoint:

| class       | IoU    | quality                   |
|-------------|--------|---------------------------|
| water       | 0.667  | strong                    |
| agriculture | 0.596  | strong                    |
| building    | 0.570  | strong                    |
| road        | 0.544  | usable                    |
| background  | 0.497  | usable                    |
| forest      | 0.385  | weak — surfaces as caveat |
| barren      | 0.346  | weak — surfaces as caveat |

The weak forest IoU is a known limitation. On mixed-vegetation rural scenes,
the model will sometimes classify forest as background with high confidence
(see `data/loveda/Val/Rural/images_png/2523.png` for a canonical example).
The agent's tools are built to **surface these caveats, not bury them**:
`get_land_breakdown` flags conditional concerns, `get_emissions_estimate`
labels assumptions, and the system prompt explicitly instructs the model to
own uncertainty honestly rather than paper over it.

---

## Emissions grounding

`scripts/emissions.py` is the heart of the grounding layer. Its `SOURCES`
dict contains six primary-authority references:

| Tag     | Source                                                                 |
|---------|------------------------------------------------------------------------|
| SRC-1   | IPCC (2022). AR6 WGIII Chapter 7: AFOLU                                |
| SRC-2   | EPA (2024). Inventory of U.S. GHG Emissions and Sinks: 1990–2022       |
| SRC-3   | U.S. EIA (2022). 2018 CBECS — energy intensity 70.6 kBtu/sf/yr         |
| SRC-4   | IPCC (2019). 2019 Refinement, Vol 4: AFOLU — default biomass/soil      |
| SRC-5   | JRC EFIResources (2018). Environmental benchmarks for buildings        |
| SRC-6   | Crippa et al. (2023). EDGAR v7.0 Global GHG Emissions Database         |

Every factor in `LAND_USE_EMISSIONS` has both an `annual_source` and an
`embodied_source` tag, plus inline prose notes in the same file deriving the
specific number from the cited reference. The tool layer surfaces these tags
with every response, so the LLM can weave citations into natural-language
output without fabricating them — the `[SRC-N]` chips you see in the rendered
report trace directly to the emissions.py table and through to the citation
block below it.

Assumptions worth knowing (all surfaced by the agent in practice):

- **Building operational emissions assume single-story footprint.** Multi-story
  buildings scale roughly linearly with floor-area ratio.
- **Agriculture is a global average.** Rice-dominated cropland would be higher;
  dryland grain lower. LoveDA doesn't distinguish crop type.
- **Water is treated as open freshwater (neutral).** Wetlands are a strong sink
  that the current factor undercounts.

---

## API reference

Swagger UI is auto-generated at `http://localhost:8000/docs`.

| Endpoint                   | What it does                                                   |
|----------------------------|----------------------------------------------------------------|
| `GET /health`              | `{status, device, agent_available}`                            |
| `POST /classify`           | multipart image → mask PNG (base64) + percentages + emissions  |
| `GET /emissions`           | the full cited factor table                                    |
| `POST /simulate`           | counterfactual: `{from_class, to_class, fraction}` → deltas    |
| `POST /agent/report`       | pre-classified state + query → cited sustainability report     |
| `GET /`                    | redirects to `/ui/` if frontend present, else meta JSON        |
| `GET /ui/`                 | Terra single-page app                                          |

`/agent/report` is intentionally stateless: the frontend sends back the
`{percentages, emissions, total_area_ha}` it got from a prior `/classify`
call, so follow-up queries on the same image don't re-run inference.

---

## Training from scratch

If you don't have a checkpoint:

```bash
# 1. Download LoveDA (~6 GB, goes to data/loveda/)
bash scripts/download_loveda.sh

# 2. Fine-tune SegFormer-B1 (one run: ~50 min on RTX 4070 Ti SUPER)
python scripts/train.py \
  --data-root data/loveda \
  --output-dir model/segformer-b1-run1 \
  --epochs 15 --batch-size 2 --grad-accum 4 --lr 6e-5

# 3. Verify
python scripts/infer.py --image data/loveda/Val/Urban/images_png/3546.png
```

Training specifics (decisions documented in `DECISIONS_LOG.md`):

- **Loss:** weighted cross-entropy, median-frequency-balanced weights, `ignore_index=0`.
- **Resolution:** 1024×1024 full tiles; no scale/crop augmentation.
- **Augmentation:** horizontal/vertical flips, rotate90, mild color jitter.
- **Optimizer:** AdamW + cosine schedule, batch 2 × grad_accum 4 (effective 8).
- **Class convention:** class 0 is no-data (ignored in loss), class 1 is background
  (trained but excluded from emissions).

---

## Known gotchas

- **`torch.load` needs `weights_only=False`** on our checkpoints (they carry
  pathlib Path objects in saved args). The loader already passes this.
- **`python-multipart` is a separate install** from `fastapi`. Already in the
  deps list but easy to miss if you're cherry-picking.
- **`num_workers=0`** during training — there's an intermittent DataLoader
  worker segfault on Ubuntu that cost a restart once in Phase 3. Data loading
  isn't the bottleneck.
- **Uvicorn must be launched with `--app-dir .`** (or `PYTHONPATH=.`) from the
  repo root, so the `backend` package is importable.
- **Prompt caching is inactive** on Haiku 4.5 because our system prompt + tool
  schemas total ~1,500 tokens, below the model's 4,096-token caching floor.
  The `cache_control` markers are accepted silently. Budget impact is
  trivial — see `DECISIONS_LOG.md` for the full trace.
- **The vision model is weakest on forest and barren.** This is surfaced in
  every agent report through `model_caveats`. Do not hide it.

---

## Costs

The pre-build budget was $10 of Anthropic API credit.

- Phase 7 (agent development + live testing): ~$0.05
- Phase 8 (frontend wiring + demo runs + race-condition debugging): ~$0.20
- **Per-demo-run cost:** ~$0.015–0.025 on Haiku 4.5

Classification is free — the SegFormer-B1 checkpoint runs locally on CUDA.

---

## Design notes

- The agent's abstraction (`ReasoningAgent` Protocol in `agent/base.py`) is
  cleanly swappable. `ClaudeAgent` is the only implementation shipped, but a
  fallback (OpenRouter, Gemini, a local model) is a one-file addition. This
  is an intentional hackathon-day escape hatch.
- `simulate_intervention` lives as a pure function in `emissions.py`, not
  inside the FastAPI handler. The agent calls it directly (in-process), and
  the `/simulate` HTTP endpoint is a thin wrapper around the same function.
  No logic duplication; no round-trips when the agent uses it.
- The frontend is same-origin with the backend (FastAPI serves the static
  files at `/ui`). This sidesteps CORS entirely and lets "clone, set key,
  run one command, see demo" actually hold.
- Status pills on the UI probe `/health` every 30 seconds. A backend restart
  eventually shows green again without a page reload.

---

## What's next

See `HACKATHON_TODO.md` for the tiered task list. Headline items:

- **P0** — replace the upload flow with a Leaflet map picker that fetches
  live satellite tiles (NAIP / Mapbox / Esri, **not** Sentinel-2 at 10m/pixel).
- **P1** — add more agent tools: bbox region selection, time-series queries,
  what-if sliders tied to `/simulate`.
- **P2** — React migration, streaming SSE for the agent response, better
  progress feedback.
- **P3** — region-specific emissions factors keyed on location.

---

## Acknowledgements

- **LoveDA** (Wang et al. 2021) — land-cover semantic segmentation benchmark at
  0.3m GSD, Zenodo record 5706578.
- **SegFormer** (Xie et al. 2021) — NVIDIA's efficient transformer segmentation
  architecture; we fine-tune the B1 variant.
- **Anthropic Claude Haiku 4.5** — the reasoning engine, accessed via the
  public Messages API with tool use.
- Emissions factors derive from public publications by **IPCC**, **U.S. EPA**,
  **U.S. EIA**, **JRC**, and the **EDGAR** consortium. All cited inline in
  `scripts/emissions.py`.
