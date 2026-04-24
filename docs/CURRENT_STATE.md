# Current State

> Overwritten at end of every phase. Tells the next chat what actually exists in the repo right now. Do NOT paste code here — reference file paths and describe behavior.

---

## Last updated: 2026-04-24, end of Phase 7 (Weekend 2, Saturday afternoon)

## Repo layout

```
USDAhackathon2026April/
  agent/                                 ← Phase 7, NEW
    __init__.py
    base.py                              (ReasoningAgent Protocol + AgentReport / ToolCallRecord dataclasses)
    tools.py                             (AgentState + 4 tool schemas/impls + dispatch_tool)
    claude.py                            (ClaudeAgent — Haiku 4.5 via Anthropic API, tool loop, max_turns=5)
  backend/
    __init__.py
    main.py                              (FastAPI app, lifespan, 3 endpoints, CORS)
    inference.py                         (InferenceEngine singleton)
    models.py                            (Pydantic schemas)
  data/
    loveda/
      Train/{Urban,Rural}/{images_png,masks_png}/
      Val/{Urban,Rural}/{images_png,masks_png}/
      samples/                           (8 overlay PNGs for visual sanity)
      stats.json
      Train.zip, Val.zip                 (deletable, ~6 GB)
  docs/
    PROJECT_PLAN.md
    PHASE_PLAN.md
    DECISIONS_LOG.md
    CURRENT_STATE.md
    dataset_stats.md
  frontend/                              (empty; Phase 8 will populate)
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
    agent_repl.py                        ← Phase 7, NEW (pre-classify + one-shot or interactive REPL)
  smoke_tools.py                         ← Phase 7, repo-root smoke test for the 4 tools (offline, no API)
  smoke_agent.py                         ← Phase 7, repo-root smoke test for the agent loop (fake client, no API)
  .env                                   ← Phase 7 (gitignored; holds ANTHROPIC_API_KEY)
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
  - Pure function; called by both `/simulate` endpoint and the Phase 7 agent
- `backend/` FastAPI app (Phase 6):
  - Run: `uvicorn backend.main:app --app-dir . --host 0.0.0.0 --port 8000`
  - Swagger UI at `/docs`, JSON schema at `/openapi.json`
  - Model loaded once in the `lifespan` startup handler; checkpoint path overridable via `LANDUSE_CHECKPOINT` env var
  - CORS permissive (`allow_origins=["*"]`)
  - Endpoints: `GET /`, `GET /health`, `POST /classify`, `GET /emissions`, `POST /simulate`
- **`agent/` package (Phase 7, NEW):**
  - `agent/base.py`: `ReasoningAgent` Protocol; `AgentReport` dataclass holds `final_text`, `tool_calls: list[ToolCallRecord]`, `turns_used`, `stop_reason`, `usage` dict.
  - `agent/tools.py`: `AgentState` dataclass (`percentages`, `emissions: EmissionsResult`, `total_area_ha`, `image_label`). Four tools, each with a JSON schema and a Python impl, wired through `dispatch_tool(name, input, state)`:
    1. `get_land_breakdown()` — percentages 0–100, dominant_class, areas, excluded fractions, and conditional `model_caveats` surfaced when forest >20%, background >40%, barren >10%, or water >5%.
    2. `get_emissions_estimate()` — per_class rows (area_ha, annual_tco2e, embodied_tco2e, annual_src, embodied_src), totals, net_sink_or_source verdict, `sources_cited` dict (only sources used by classes actually present), `assumptions` list filtered to classes present.
    3. `simulate_intervention(from_class, to_class, fraction_pct)` — **fraction_pct is 0–100**, converted to 0.0–1.0 internally before calling `emissions.simulate_intervention()`. Returns converted_area_ha, delta_annual, delta_embodied, annual_effect label, narrative, before/after totals.
    4. `recommend_mitigation(priority)` — deterministic ranking over all `(from_class in image, to_class ≠ from_class)` pairs at 100% conversion. `priority ∈ {annual, embodied, balanced}`. Returns top 5 candidates with `available_area_ha`, `annual_delta_if_full`, `embodied_delta_if_full`, plus `notes` (missing classes, horizon assumption, forest-planting caveat). LLM does the judgment; tool just ranks.
  - `agent/claude.py`: `ClaudeAgent` uses `anthropic` SDK. Model `claude-haiku-4-5`. `max_turns=5`. Tool loop: POST messages with system+tools → if `stop_reason == "tool_use"` execute every `tool_use` block and append a `user` message with `tool_result` blocks → loop. On the final turn (max_turns reached) tools are **not** passed, forcing a text synthesis. `cache_control: {"type": "ephemeral"}` is set on the system prompt and the last tool schema. Per-tool errors surface as `is_error: True` tool_results so the model can recover.
  - System prompt: ~770 tokens, sustainability-analyst persona blend (data-first + practical tradeoffs + honest about model uncertainty). Explicit instruction to weigh `recommend_mitigation` menu tradeoffs rather than blindly pick #1.
- **`scripts/agent_repl.py` (Phase 7, NEW):**
  - Pre-classifies image once at startup using `backend.inference.InferenceEngine` directly (no HTTP). Builds `AgentState`, hands it to `ClaudeAgent`.
  - `--dry-run`: classify + dump state, skip API call. No key required.
  - `--query "..."`: one-shot. No query → run default sustainability-report prompt, then drop into interactive REPL for follow-ups.
  - `--verbose`: print per-tool-call detail (name, input, truncated result).
  - `--max-turns N` (default 5), `--model ...` overrides available.
  - Loads `ANTHROPIC_API_KEY` from `.env` via `python-dotenv`. `.env` is gitignored.
- **`smoke_tools.py` (Phase 7, NEW, at repo root):** builds an `AgentState` mimicking 3546.png, exercises all 4 tools + dispatch, verifies the 0–100 → 0.0–1.0 unit conversion, and hits 3 error paths (nonexistent from_class, unknown tool, bad priority). Passes offline, no API key needed.
- **`smoke_agent.py` (Phase 7, NEW, at repo root):** scripted fake Anthropic client verifies 4 scenarios: happy path (multi-tool + end_turn), tool error recovery (is_error=True + agent retries), max_turns cap (final turn omits `tools` kwarg), edge case (model still emits tool_use on forced-final turn). Also verifies cache_control placement on system and last-tool. Passes offline.

## What's half-done

- Windows backup machine still not smoke-tested (still never needed)
- Train.zip + Val.zip still on disk; deletable to reclaim ~6 GB
- MISMATCH warning from `transformers` fires on every model load (cosmetic only)

## What's next

**Phase 8: frontend + demo polish.** The backend + agent both exist, are both exercised, and produce numerically-parity results against Phase 5/6. Next steps:

1. Build a minimal frontend (probably Vite + React, served on a separate localhost port) that:
   - Accepts an image upload, calls `POST /classify`, shows the colored mask overlay and percentages.
   - Calls `POST /simulate` on slider-driven scenarios ("what if 50% of agri → forest?").
   - Has a separate agent-report pane that calls a new backend endpoint (TBD) which wraps `ClaudeAgent.run()`. Or keeps the agent as a CLI-only feature for the demo and shows pre-recorded report text.
2. Select demo images: 3546.png (clean "model succeeds" case), 2523.png (optional "model honestly surfaces uncertainty" case), and ~5 more candidates. Phase 8 prep task.
3. Fix the `/quit` parsing bug in `scripts/agent_repl.py` (see Known issues).
4. Consider adding a `POST /agent/report` endpoint in `backend/` for the frontend to use, wrapping `ClaudeAgent.run()` with a fresh `AgentState` built from the uploaded image's classify result. Would share the preclassify → state pipeline already in `agent_repl.py`.

## Known issues / gotchas

- Activate env each SSH/AnyDesk session: `conda activate landuse`
- `torch.load` needs `weights_only=False` when loading our checkpoints. Handled everywhere.
- **Checkpoint state dict lives under `ckpt["model"]`, NOT `ckpt["model_state_dict"]`.** Any future refactor should adopt the more standard convention.
- PyTorch version is 2.11.0+cu126. Working fine.
- `num_workers=0` used after a mid-epoch-5 DataLoader worker segfault in Phase 3. Not relevant for inference.
- **Forest↔background confusion** (not forest↔agri). Phase 5 diagnostic: on rural mixed-vegetation tiles, 100% of GT-forest pixels can collapse to background with high confidence. This is what 2523.png demonstrates and what the agent's `model_caveats` layer is designed to surface.
- Barren (0.35) and forest (0.39) are the weakest classes. Accepted for v1.
- Phase 6 gotchas (still relevant):
  - Run uvicorn with `--app-dir .` (or `PYTHONPATH=.`)
  - `python-multipart` is a separate `pip install` from `fastapi`
  - `compute_emissions` expects int-keyed pixel counts, not name-keyed
  - `EmissionsResult.per_class` row shape: 5 floats, no per-row source strings. Sources derive at response-build time.
  - `excluded_breakdown` values are 0.0–1.0 fractions, not percentages.
- **Phase 7 gotchas (NEW):**
  - **Prompt caching does NOT fire on Haiku 4.5.** Minimum cacheable prefix is **4,096 tokens** for Haiku 4.5 (raised from the older 1,024 threshold). Our system prompt + tool schemas total ~1,500 tokens, below the threshold. The `cache_control` markers are accepted silently but no cache entry is created. Every `input_tokens` count is uncached. **Budget impact is trivial** (~$0.001 extra per turn, ~$0.15 across the full demo budget) so we shipped as-is. To activate caching later: pad the system prompt to ≥4,096 tokens with few-shot examples or extended instructions. Verified empirically: `cache_read_input_tokens=0` and `cache_creation_input_tokens=0` on every live call.
  - **REPL runs each user query as a fresh `ClaudeAgent.run()` session.** No cross-query memory. The agent re-runs `get_land_breakdown` + `get_emissions_estimate` on every follow-up even though `AgentState` is identical. Functionally correct; wastes ~$0.005 per follow-up. Fix would be to keep a `messages` list alive across queries inside the REPL loop.
  - **`/quit` REPL command sometimes isn't recognized** — gets passed through as a literal query to the agent. Cause is probably trailing whitespace on the input (blank-line submit captures `/quit\n` and `.strip()` should handle it, but didn't on one test). Benign — wastes one API call. Quick Phase 8 fix: `query.strip().lower().startswith("/quit")`.
  - **Model calls occasionally attach `[SRC-1,SRC-4]` bundles** to claims where only one is strictly applicable, especially in short follow-up responses. Not fabrication — both tags genuinely exist in the tool outputs, just imprecisely applied. Prompt-level nudge possible in Phase 8 ("cite only the exact source backing the numeric claim") but not important.
  - **Path convention matters:** LoveDA image paths are `data/loveda/Val/{Urban,Rural}/images_png/NNNN.png`. 3546 is in Urban, 2523 is in Rural. Easy to forget; `find data/loveda -name 'N.png'` is the lookup pattern.
  - **The wacky `recommend_mitigation` top candidates** (e.g. priority=embodied ranks "destroy all 6.8 ha of forest to reduce embodied stock" as #1) are **working as designed**: the tool returns a mechanical menu, and the agent's system prompt explicitly instructs it to rule out absurd conversions. Verified on 2523.png that the agent does in fact skip the absurd top pick in favor of a realistic one. Don't "fix" this.
  - **Empty-input tool schemas use `properties: {}, required: []`.** Tested on live calls; Haiku correctly sends `input: {}` and the dispatch handles it.

## Key numbers

### Model (measured Phase 3)
- **Best val_mIoU: 0.5147** (epoch 10)
- **Best val_mIoU (no background): 0.5178** (epoch 10)
- Per-class IoU at best.pt: background 0.4966 / building 0.5696 / road 0.5437 / water 0.6672 / barren 0.3455 / forest 0.3850 / agriculture 0.5955
- Total training time: ~50 minutes for 15 epochs

### Inference
- Single-image (1024×1024, TTA on, 4070 Ti SUPER): **~0.2–0.4 s**
- Via HTTP `/classify`: **313 ms end-to-end** (Phase 6 measurement)
- Via `agent_repl.py` preclassify (direct InferenceEngine import): **282–302 ms** (Phase 7 measurement, three live runs)
- Preprocessing parity with training: max abs diff **4.8e-7**
- Numerical parity across Phase 5 / Phase 6 / Phase 7: `total_annual_tco2e_per_yr` = **−50.44** and `total_embodied_tco2e` = **5542.81** on 3546.png, identical to the decimal across all three pipelines.

### Simulation (measured Phase 6; re-verified via agent tool in Phase 7)
- 100% agri→forest on 3546.png (0.352 ha): **delta_annual −3.69 tCO2e/yr, delta_embodied +228.53 tCO2e**
- 50% building→forest (0.025 ha): delta_annual −1.79 tCO2e/yr, delta_embodied +4.91 tCO2e
- Error path verified: `from_class` not in image → ValueError surfaces as `is_error: True` tool_result; agent recovers.

### Agent sessions (measured Phase 7, live)
- **3546.png, default sustainability-report prompt + 1 follow-up:** 19,127 input / 1,669 output tokens across 2 sessions. 0 cached. ~$0.025 per run. 3 turns on the report, 3 turns on the follow-up.
- **2523.png, same prompt structure:** 15,357 input / 1,312 output tokens, 0 cached. ~$0.02. Agent correctly headlined the 90.6% background caveat as a "critical limitation" and labeled the report as "a lower bound on the parcel's true emissions profile" — the honest-uncertainty behavior the pipeline was designed for.
- **Offline smoke tests:** all 7 tool-level scenarios (3 dispatch + 3 priority modes + 1 unit check + error paths) pass; all 4 agent-loop scenarios (happy path, error recovery, max_turns cap, forced-final-turn edge case) pass.

### Emissions factor snapshot (from `emissions.py`, global averages)

| class | annual tCO2e/ha/yr | embodied tCO2e/ha | source |
|---|---:|---:|---|
| building | +65.0 | +600 | SRC-3 (EIA CBECS) / SRC-5 (JRC) |
| road | +4.0 | +220 | SRC-1 (IPCC AR6 WGIII Ch. 7) / SRC-5 |
| water | 0.0 | 0 | SRC-1 |
| barren | 0.0 | +50 | SRC-4 (IPCC 2019 Guidelines) |
| forest | **−8.0** | +800 | SRC-1 |
| agriculture | +2.5 | +150 | SRC-6 (EDGAR v7) / SRC-4 |

### Budget
- API cost spent so far: **~$0.05** (Phase 7 live tests: one 3-turn cheap probe + one 5-turn + two 3-turn reports on 2523 + 3546)
- Remaining budget: **~$9.95** of $10
- Time spent Weekend 2 through Phase 7: ~3.5h total, inside estimate
- Sunday buffer: **still 0 hours used**, carried forward to Phase 8

## Saturday gate verdict (from PHASE_PLAN.md)

Both gates passed cleanly at Phase 3:

1. ✅ **val_mIoU ≥ 0.38** — achieved 0.5147 at epoch 10
2. ✅ **Forest OR agri IoU ≥ 0.55** — agri hit 0.596 at epoch 10

### Locked class weights used in Phase 3 (MFB, mean-normalized)

| Class | Weight |
|---|---|
| background | 0.255 |
| building | 0.824 |
| road | 1.730 |
| water | 1.426 |
| barren | 1.748 |
| forest | 0.567 |
| agriculture | 0.451 |

### Normalization constants (locked)

ImageNet: mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225).

### Dependency notes (accumulated)
- PyTorch 2.11.0+cu126
- transformers requires PyTorch ≥2.6 (CVE-2025-32434)
- `python-multipart` required for FastAPI multipart uploads (Phase 6)
- uvicorn invocation needs `--app-dir .` when running `backend.main:app` from repo root
- `python-dotenv` required for `.env` loading in `agent_repl.py` (Phase 7)
- `anthropic` SDK required for `ClaudeAgent` (Phase 7)