# Current State

> Overwritten at end of every phase. Tells the next chat what actually exists in the repo right now. Do NOT paste code here — reference file paths and describe behavior.

---

## Last updated: 2026-04-24, end of Phase 8 (Weekend 2, Sunday afternoon) — **PRE-BUILD COMPLETE**

## Repo layout

```
USDAhackathon2026April/
  agent/                                 (Phase 7)
    __init__.py
    base.py                              (ReasoningAgent Protocol + AgentReport / ToolCallRecord dataclasses)
    tools.py                             (AgentState + 4 tool schemas/impls + dispatch_tool)
    claude.py                            (ClaudeAgent — Haiku 4.5 via Anthropic API, tool loop, max_turns=5)
  backend/                               (Phase 6 + Phase 8 additions)
    __init__.py
    main.py                              (FastAPI app, lifespan, 5 endpoints, /ui mount, CORS, dotenv)
    inference.py                         (InferenceEngine singleton)
    models.py                            (Pydantic schemas incl. AgentReportRequest/Response, ToolCallOut)
  frontend/                              ← Phase 8, NEW (single-page UI, no build step)
    index.html                           (~620 lines; instrument-panel CSS, two Google Fonts)
    app.js                               (~540 lines; fetch wiring, race-safe state, markdown renderer)
    demos/                               (user populates: 3546.png, 2523.png)
  data/
    loveda/
      Train/{Urban,Rural}/{images_png,masks_png}/
      Val/{Urban,Rural}/{images_png,masks_png}/
      samples/                           (8 overlay PNGs for visual sanity)
      stats.json
      Train.zip, Val.zip                 (deletable, ~6 GB)
  docs/                                  (plan docs + optional: screenshot-3546.png, demo_tiles.md)
    PROJECT_PLAN.md
    PHASE_PLAN.md
    DECISIONS_LOG.md
    CURRENT_STATE.md
    dataset_stats.md
    screenshot-3546.png                  ← Phase 8: referenced from README (user saves here)
  inference_outputs/                     (Phase 5 artifacts retained for reference)
    2522_mask_colored.png, 2522_mask_raw.png, 2522_overlay.png
    2523_confidence.png, 2523_mask_colored.png, 2523_mask_raw.png, 2523_overlay.png
    3546_mask_colored.png, 3546_mask_raw.png, 3546_overlay.png
  model/
    segformer-b1-run1/
      best.pt                            (epoch 10 weights, val_mIoU=0.5147)
      epoch_15.pt                        (last epoch, for reference)
      train_log.csv                      (15 epoch rows, deduplicated)
      run_config.json                    (CLI args from the run)
    segformer-b1-run1.log                (full stdout log of training)
  scripts/
    download_loveda.sh
    dataset_stats.py
    dataset.py                           (Phase 2)
    test_loader.py                       (Phase 2)
    train.py                             (Phase 3)
    infer.py                             (Phase 5)
    emissions.py                         (Phase 5 + Phase 6 additions; 555 lines)
    agent_repl.py                        (Phase 7; pre-classify + one-shot or interactive REPL)
    smoke_agent_endpoint.sh              ← Phase 8: curl-based end-to-end smoke test
  smoke_tools.py                         (Phase 7, repo-root smoke test for the 4 tools, offline)
  smoke_agent.py                         (Phase 7, repo-root smoke test for the agent loop, offline)
  .env                                   (gitignored; holds ANTHROPIC_API_KEY)
  README.md                              ← Phase 8, NEW (architecture diagram, quickstart, API ref)
  HACKATHON_TODO.md                      ← Phase 8, NEW (tiered P0–P3 task list, day-of triage)
```

## What works

- Conda env `landuse` (Python 3.11), CUDA confirmed on RTX 4070 Ti SUPER
- LoveDA Train + Val on disk, MD5-verified
- `scripts/dataset.py`: `LoveDADataset` + `build_transforms(split)`
- `scripts/train.py`: full SegFormer-B1 fine-tune pipeline
- Trained checkpoint: `model/segformer-b1-run1/best.pt` — epoch 10, val_mIoU 0.5147
- `scripts/infer.py`: full CLI inference pipeline from Phase 5 (still works, used as reference for Phase 6's InferenceEngine)
- `scripts/emissions.py`: lookup table + `compute_emissions` + `cite` + `simulate_intervention`
  - Keys on int class IDs (`0-7`), returns `EmissionsResult` dataclass
  - `EmissionsResult.per_class` row shape: `pixel_count / pixel_fraction / area_ha / annual_tco2e / embodied_tco2e` (5 float fields)
  - `simulate_intervention(current_percentages, total_area_ha, from_class, to_class, fraction) → SimulationResult`
  - Pure function; called by `/simulate` endpoint, the Phase 7 agent, and `/agent/report`
- **`backend/` FastAPI app (Phase 6 + 8):**
  - Run: `uvicorn backend.main:app --app-dir . --host 0.0.0.0 --port 8000`
  - Swagger UI at `/docs`, JSON schema at `/openapi.json`
  - SegFormer model loaded once in `lifespan`; checkpoint path overridable via `LANDUSE_CHECKPOINT` env var
  - `ClaudeAgent` also instantiated in `lifespan` **if `ANTHROPIC_API_KEY` is set**; `app.state.agent = None` if missing. `.env` auto-loaded via `python-dotenv` at import time
  - CORS permissive (`allow_origins=["*"]`)
  - **Frontend served at `/ui`** via `StaticFiles` mount; `GET /` redirects to `/ui/` when `frontend/index.html` exists
  - **Endpoints:**
    - `GET /` — redirect to `/ui/` or meta JSON
    - `GET /health` — `{status, device, agent_available}`
    - `POST /classify` — multipart image → mask PNG (base64) + percentages + emissions
    - `GET /emissions` — full cited factor table
    - `POST /simulate` — counterfactual deltas (pure math, sub-10ms)
    - `POST /agent/report` — takes `{percentages, emissions, total_area_ha, image_label, query, max_turns?}` → runs `ClaudeAgent.run()` → returns `{final_text, tool_calls, turns_used, stop_reason, usage}`. Stateless (frontend echoes classify result). Sync handler (runs in threadpool; does not block event loop during ~15s agent calls). 503 if agent unavailable.
- **`agent/` package (Phase 7):**
  - `agent/base.py`: `ReasoningAgent` Protocol; `AgentReport` dataclass holds `final_text`, `tool_calls: list[ToolCallRecord]`, `turns_used`, `stop_reason`, `usage` dict.
  - `agent/tools.py`: `AgentState` dataclass (`percentages`, `emissions: EmissionsResult`, `total_area_ha`, `image_label`). Four tools, each with a JSON schema and a Python impl, wired through `dispatch_tool(name, input, state)`:
    1. `get_land_breakdown()` — percentages 0–100, dominant_class, areas, excluded fractions, and conditional `model_caveats` surfaced when forest >20%, background >40%, barren >10%, or water >5%.
    2. `get_emissions_estimate()` — per_class rows (area_ha, annual_tco2e, embodied_tco2e, annual_src, embodied_src), totals, net_sink_or_source verdict, `sources_cited` dict (only sources used by classes actually present), `assumptions` list filtered to classes present.
    3. `simulate_intervention(from_class, to_class, fraction_pct)` — **fraction_pct is 0–100**, converted to 0.0–1.0 internally before calling `emissions.simulate_intervention()`. Returns converted_area_ha, delta_annual, delta_embodied, annual_effect label, narrative, before/after totals.
    4. `recommend_mitigation(priority)` — deterministic ranking over all `(from_class in image, to_class ≠ from_class)` pairs at 100% conversion. `priority ∈ {annual, embodied, balanced}`. Returns top 5 candidates with `available_area_ha`, `annual_delta_if_full`, `embodied_delta_if_full`, plus `notes` (missing classes, horizon assumption, forest-planting caveat). LLM does the judgment; tool just ranks.
  - `agent/claude.py`: `ClaudeAgent` uses `anthropic` SDK. Model `claude-haiku-4-5`. `max_turns=5`. Tool loop: POST messages with system+tools → if `stop_reason == "tool_use"` execute every `tool_use` block and append a `user` message with `tool_result` blocks → loop. On the final turn (max_turns reached) tools are **not** passed, forcing a text synthesis. `cache_control: {"type": "ephemeral"}` is set on the system prompt and the last tool schema. Per-tool errors surface as `is_error: True` tool_results so the model can recover.
  - System prompt: ~770 tokens, sustainability-analyst persona blend (data-first + practical tradeoffs + honest about model uncertainty). Explicit instruction to weigh `recommend_mitigation` menu tradeoffs rather than blindly pick #1.
- **`scripts/agent_repl.py` (Phase 7):**
  - Pre-classifies image once at startup using `backend.inference.InferenceEngine` directly (no HTTP). Builds `AgentState`, hands it to `ClaudeAgent`.
  - `--dry-run`: classify + dump state, skip API call. No key required.
  - `--query "..."`: one-shot. No query → run default sustainability-report prompt, then drop into interactive REPL for follow-ups.
  - `--verbose`: print per-tool-call detail (name, input, truncated result).
  - `--max-turns N` (default 5), `--model ...` overrides available.
  - Loads `ANTHROPIC_API_KEY` from `.env` via `python-dotenv`. `.env` is gitignored.
- **`frontend/` (Phase 8, NEW):**
  - `frontend/index.html` — single-page layout with CSS variables mirroring `backend/inference.py`'s PALETTE byte-for-byte. Two fonts via Google Fonts CDN (Fraunces serif, JetBrains Mono). Four panels: Input (drop zone + demo buttons), Imagery/Segmentation (source + mask side-by-side), Composition (class bar + legend + KPI tiles + Generate button), Sustainability Report (progress stepper → rendered markdown → tool trace + citations footer). Top-bar "instrument chrome" with three status pills (Backend, Device, Agent).
  - `frontend/app.js` — vanilla JS, no build step, no framework. `BASE_URL = window.location.origin` (same-origin; override via `?api=...`). Features:
    - `probeHealth()` polls `/health` on load and every 30s
    - `loadFile()` handles both drop-zone input and demo-button fetches; bumps a monotonic `state.gen` token on every load
    - `runClassify(gen)` fires `POST /classify`; rechecks `gen` at every await-resume to abort stale responses
    - `runAgent()` builds the `/agent/report` body from `state.classify`, fires POST, runs a 5-step fake progress stepper (3s per step) during the ~15s wall-clock
    - `renderComposition()` draws the horizontal stacked class bar using PALETTE colors, populates the legend and four KPI tiles
    - `markdownToHtml()` is a tuned ~60-line renderer handling `## h2`, `### h3`, `**bold**`, `*em*`, lists, `` `code` ``, `[SRC-N]` and `[SRC-1, SRC-2]` citation chips, and `---` horizontal rules (leading rule dropped, mid-doc rule rendered). Dry-run-tested against real Haiku `final_text`.
    - `renderReport()` paints the rendered markdown, groups tool calls by turn, surfaces usage stats (tokens in/out, cached pct, wallclock), and lists full source citations at the bottom
    - Errors render a dismissable banner; classify/agent failures reset the UI cleanly
  - `frontend/demos/` — user populates with `3546.png` and `2523.png` from LoveDA Val (Urban/Rural respectively). Demo buttons fetch from `/ui/demos/<id>.png`.
- **`scripts/smoke_agent_endpoint.sh` (Phase 8):** curl-based end-to-end smoke test. Hits `/health`, classifies a demo image, extracts fields from the response via `jq`, posts to `/agent/report`, prints the final text + tool call summary + usage stats. Requires running backend and `jq`. `IMAGE=...` env-var overridable.

## What's half-done

- Windows backup machine still not smoke-tested (still never needed, pre-build is done)
- Train.zip + Val.zip still on disk; deletable to reclaim ~6 GB
- MISMATCH warning from `transformers` fires on every model load (cosmetic only)
- Demo-image shortlist beyond 3546 and 2523 still pending (planned in Phase 5 log; P0.2 in HACKATHON_TODO for the hackathon team)
- `docs/screenshot-3546.png` referenced from README but user needs to save the browser screenshot there

## What's next

**Pre-build is complete.** All Phase 8 deliverables shipped:
- ✅ `POST /agent/report` endpoint wired to live ClaudeAgent
- ✅ Single-page frontend mounted at `/ui` (same-origin, no CORS)
- ✅ Full upload → classify → analyze → report loop in browser
- ✅ Race-safe against rapid demo-button clicks (monotonic `gen` token)
- ✅ README.md with ASCII architecture diagram, quickstart, API reference, known gotchas
- ✅ HACKATHON_TODO.md with strict P0–P3 tiering, time estimates, definitions of done, and day-of triage

**Hackathon handoff:** the team inherits a working portfolio-defensible system. Read `HACKATHON_TODO.md` first. Day-of priority:
1. **P0.1 Map picker + live tile fetching** (4–6h) — NAIP via USGS/Esri preferred; do NOT use Sentinel-2 at 10m/pixel
2. **P0.2 Demo-image shortlist** (1–2h)
3. **P0.3 Error-state polish** (1h)
4. Pick one P1 item based on team energy (P1.4 sliders is easiest; P1.3 chat mode is most impressive; P1.1 bbox is cleanest)

## Known issues / gotchas

- Activate env each SSH/AnyDesk session: `conda activate landuse`
- `torch.load` needs `weights_only=False` when loading our checkpoints. Handled everywhere.
- **Checkpoint state dict lives under `ckpt["model"]`, NOT `ckpt["model_state_dict"]`.** Any future refactor should adopt the more standard convention.
- PyTorch version is 2.11.0+cu126. Working fine.
- `num_workers=0` used after a mid-epoch-5 DataLoader worker segfault in Phase 3. Not relevant for inference.
- **Forest↔background confusion** (not forest↔agri). Phase 5 diagnostic: on rural mixed-vegetation tiles, 100% of GT-forest pixels can collapse to background with high confidence. This is what 2523.png demonstrates and what the agent's `model_caveats` layer is designed to surface.
- Barren (0.35) and forest (0.39) are the weakest classes. Accepted for v1.
- **Phase 6 gotchas (still relevant):**
  - Run uvicorn with `--app-dir .` (or `PYTHONPATH=.`)
  - `python-multipart` is a separate `pip install` from `fastapi`
  - `compute_emissions` expects int-keyed pixel counts, not name-keyed
  - `EmissionsResult.per_class` row shape: 5 floats, no per-row source strings. Sources derive at response-build time.
  - `excluded_breakdown` values are 0.0–1.0 fractions, not percentages.
- **Phase 7 gotchas (still relevant):**
  - **Prompt caching does NOT fire on Haiku 4.5.** Minimum cacheable prefix is **4,096 tokens** for Haiku 4.5 (raised from the older 1,024 threshold). Our system prompt + tool schemas total ~1,500 tokens, below the threshold. The `cache_control` markers are accepted silently but no cache entry is created. Every `input_tokens` count is uncached. Budget impact is trivial (~$0.001 extra per turn) so we shipped as-is.
  - **REPL runs each user query as a fresh `ClaudeAgent.run()` session.** No cross-query memory. The agent re-runs `get_land_breakdown` + `get_emissions_estimate` on every follow-up. Functionally correct; wastes ~$0.005 per follow-up. Phase 8 frontend inherits the same behavior (single-shot per Generate click); chat mode is P1.3 for hackathon.
  - **`/quit` REPL command sometimes isn't recognized** — gets passed through as a literal query. Quick fix is `query.strip().lower().startswith("/quit")`. Not addressed in Phase 8 (frontend is click-driven; REPL is dev-only).
  - **Model calls occasionally attach `[SRC-1,SRC-4]` bundles** to claims where only one is strictly applicable, especially in short follow-up responses. Not fabrication — just imprecise. Low priority.
  - **Path convention matters:** LoveDA image paths are `data/loveda/Val/{Urban,Rural}/images_png/NNNN.png`. 3546 is in Urban, 2523 is in Rural. `find data/loveda -name 'NNNN.png'` is the lookup pattern.
  - **The wacky `recommend_mitigation` top candidates** (e.g. priority=embodied ranks "destroy all 6.8 ha of forest to reduce embodied stock" as #1) are **working as designed**: the tool returns a mechanical menu, the agent's system prompt instructs it to rule out absurd conversions. Verified live in Phase 7 and Phase 8 demos.
  - **Empty-input tool schemas use `properties: {}, required: []`.** Tested on live calls; Haiku correctly sends `input: {}` and the dispatch handles it.
- **Phase 8 gotchas (NEW):**
  - **Async race in the frontend:** rapid demo-button clicks fire multiple classify calls; response ordering is not guaranteed. Fixed via monotonic `gen` token in `state` — every `loadFile()` bumps it, every `await` recheck drops stale results. Same guard on `runAgent()` so mid-report image swaps abort cleanly. If extending the frontend, any new async state writer needs the same guard.
  - **Backgrounding uvicorn with `&` + immediate relaunch causes port-already-in-use** (the old process doesn't die instantly). Use `pkill -f 'uvicorn backend.main:app'` before relaunching, or foreground it. Sleep 10–12s after relaunch before hitting `/health` — startup loads the model and agent, takes ~10s.
  - **Haiku sometimes opens reports with a leading `---`** markdown horizontal rule before the first heading. Frontend markdown renderer now detects `/^(-{3,}|\*{3,}|={3,})\s*$/` and drops leading rules; mid-doc rules render as styled `<hr />`.
  - **`ClassifyResult` doesn't surface raw `class_pixel_counts` or `total_pixels`** — they're consumed internally for `EmissionsResult`. The `/agent/report` endpoint instead rehydrates `EmissionsResult` from the pydantic `EmissionsReport` echoed back by the frontend. 15-line reconstruction; no recompute.
  - **FastAPI serves static files from disk on each request.** HTML/CSS/JS edits show up on Ctrl+Shift+R without a uvicorn restart. Only Python changes need the restart.
  - **Favicon 404 on every page load** — harmless, just noise in the uvicorn log. Fix would be a single-line route or a file in `frontend/favicon.ico`.

## Key numbers

### Model (measured Phase 3)
- **Best val_mIoU: 0.5147** (epoch 10)
- **Best val_mIoU (no background): 0.5178** (epoch 10)
- Per-class IoU at best.pt: background 0.4966 / building 0.5696 / road 0.5437 / water 0.6672 / barren 0.3455 / forest 0.3850 / agriculture 0.5955
- Total training time: ~50 minutes for 15 epochs

### Inference
- Single-image (1024×1024, TTA on, 4070 Ti SUPER): **~0.2–0.4 s**
- Via HTTP `/classify`: **313 ms end-to-end** (Phase 6)
- Via `agent_repl.py` preclassify (direct InferenceEngine import): **282–302 ms** (Phase 7, three live runs)
- Via frontend `/classify` (browser → FastAPI → InferenceEngine): **309 ms** (Phase 8, live browser test)
- Preprocessing parity with training: max abs diff **4.8e-7**
- Numerical parity across Phase 5 / Phase 6 / Phase 7 / Phase 8: `total_annual_tco2e_per_yr` = **−50.44** and `total_embodied_tco2e` = **5542.81** on 3546.png, identical to the decimal across all four pipelines.

### Simulation (measured Phase 6; re-verified Phase 7 agent tool; re-verified Phase 8 live report)
- 100% agri→forest on 3546.png (0.352 ha): **delta_annual −3.69 tCO2e/yr, delta_embodied +228.53 tCO2e**
- 50% building→forest (0.025 ha): delta_annual −1.79 tCO2e/yr, delta_embodied +4.91 tCO2e
- Error path verified: `from_class` not in image → ValueError surfaces as `is_error: True` tool_result; agent recovers.

### Agent sessions
- **Phase 7 live (2523 + 3546 default reports + 1 follow-up each):** 19,127 in / 1,669 out, 0 cached, ~$0.025 per image across 2 sessions.
- **Phase 8 live (3546 via `/agent/report` through frontend):** 13,888 in / 1,006 out, 0 cached, **~14s wallclock**, 4 turns, `stop_reason: end_turn`, 5 tool calls (2 in turn 1, 1 each in turns 2 and 3 including two parallel `simulate_intervention` in turn 3 for candidate comparison). Cost ~$0.017.
- **Offline smoke tests:** all 7 tool-level scenarios pass; all 4 agent-loop scenarios pass; Phase 8 adds `smoke_agent_endpoint.sh` for HTTP-level verification.

### Emissions factor snapshot (from `emissions.py`, global averages)

| class | annual tCO2e/ha/yr | embodied tCO2e/ha | source |
|---|---:|---:|---|
| building | +65.0 | +600 | SRC-3 (EIA CBECS) / SRC-5 (JRC) |
| road | +4.0 | +220 | SRC-1 (IPCC AR6 WGIII Ch. 7) / SRC-5 (JRC) |
| water | 0.0 | 0 | SRC-4 (IPCC 2019) |
| barren | 0.0 | +50 | SRC-4 / SRC-4 |
| forest | −8.0 | −800 | SRC-1 / SRC-4 |
| agriculture | +2.5 | +150 | SRC-6 (EDGAR) / SRC-4 |

Sign convention: positive = net emitter / released-on-disturbance; negative = net sequestration / stored stock.

### Budget
- **Total API spend pre-build: ~$0.25** (Phase 7 ~$0.05 + Phase 8 ~$0.20)
- **Pre-build budget remaining: ~$9.75 of $10** (2.5% spent)
- Sunday buffer untouched across Weekend 2

## Hackathon readiness checklist

- [x] Backend + agent exercised end-to-end via live demo run on 3546.png
- [x] Frontend race conditions fixed
- [x] README covers quickstart, architecture, API, costs, gotchas
- [x] HACKATHON_TODO.md tiered P0→P3 with time estimates and day-of triage
- [x] `.env` pattern documented; graceful degradation if API key missing
- [x] `/health` surfaces agent availability for UI feedback
- [x] Swagger at `/docs` reflects all 5 endpoints (incl. `/agent/report`)
- [ ] **User tasks before hackathon:**
  - Save the working browser screenshot as `docs/screenshot-3546.png` so README image link resolves
  - Copy `frontend/demos/3546.png` and `frontend/demos/2523.png` (may already be done locally; verify on any new clone)
  - Generate `requirements.txt` via `pip freeze > requirements.txt` inside `landuse` env for clean reinstalls
  - Commit README.md and HACKATHON_TODO.md to git
