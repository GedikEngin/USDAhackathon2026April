# Phase Plan — Land Use & GHG Analysis System

> This file is the canonical phase breakdown and locked-in decisions. `PROJECT_PLAN.md` is the original vision doc. `DECISIONS_LOG.md` tracks what actually happened. `CURRENT_STATE.md` reflects the repo right now.

---

## How to start each phase chat

Open a new chat in this Claude Project. First message should be literally:

> Starting Phase N. Read project knowledge, confirm you're oriented, then let's go.

Claude will read `PROJECT_PLAN.md`, this file, `DECISIONS_LOG.md`, and `CURRENT_STATE.md`, confirm context, and begin the phase.

## How to end each phase chat

Before closing, send:

> Update DECISIONS_LOG.md and CURRENT_STATE.md with what we just did. Give me the full file contents to paste in.

Paste the output into project knowledge. Close the chat. Takes ~2 minutes and saves ~15 next session.

## When to break these rules

If a phase goes sideways mid-session (e.g. training debug rabbit hole), stay in that chat until it's resolved. Split on phase boundaries, not task boundaries within a phase.

---

## Locked decisions (do not re-litigate)

- **Vision model:** SegFormer-B1, fine-tuned on LoveDA Urban + Rural combined
- **Training resolution:** 1024×1024 full tiles, batch 2, gradient accumulation 4 (effective batch 8). Fallback to 768 if first epoch >45 min.
- **Class handling:** 7 classes (background, building, road, water, barren, forest, agriculture). Background uses `ignore_index=0` in loss; excluded from emissions aggregation.
- **Region semantics:** Whole image aggregated. No bbox, no connected components. Agent tools operate on aggregate percentages.
- **Emissions factors:** Global averages, IPCC AR6 / EPA / EDGAR cited inline in code.
- **LLM:** Claude Haiku 4.5 via Anthropic API, prompt caching on system prompt + tool defs, `max_turns=5` hard cap on agent loop.
- **Agent abstraction:** `ReasoningAgent` protocol; `ClaudeAgent` is the only implementation for pre-build. Fallback is a hackathon-day task.
- **Backend:** FastAPI. **Frontend:** single-page HTML + vanilla JS.
- **Budget:** $10 cap for pre-build. Agent dev is the only meaningful cost center.
- **Demo:** 2-3 pre-selected demo images primary; arbitrary upload secondary.

## Saturday Weekend 1 go/no-go checkpoint

Both gates must pass to continue with the fine-tune on Sunday:

1. **Val mIoU ≥ 0.38 after ~10 epochs**
2. **Forest IoU ≥ 0.55 OR agriculture IoU ≥ 0.55** (at least one)

If either gate fails: pivot to HuggingFace pretrained LoveDA checkpoint Sunday morning. No "one more epoch." Reframe portfolio narrative around agent + grounding layer.

---

## Phases

### Weekend 1 — Vision (Ubuntu via SSH)

**Phase 0 — Environment & remote dev (1.5h, Fri eve)**
Conda env on Ubuntu box, CUDA/PyTorch sanity check, SSH + tmux workflow, 15-min smoke test on Windows backup (just `torch.cuda.is_available()`). Deliverable: you can edit locally, run on Ubuntu, see GPU util.

**Phase 1 — LoveDA download + dataset analysis (2h, Sat AM)**
Download Urban + Rural train/val. Write `dataset_stats.py` — per-class pixel frequency train vs val, 8 sample overlays, class imbalance ratios. Deliverable: `dataset_stats.md`. Decide loss weighting together before Phase 3.

**Phase 2 — Data loading + augmentation (2h, Sat AM/midday)**
PyTorch `Dataset`, albumentations pipeline (flips + rotate90 + color jitter, no scale/crop), stratified Urban/Rural train/val split, `ignore_index=0`. Deliverable: DataLoader produces correct-shape batches for 20 iterations without errors.

**Phase 3 — SegFormer-B1 fine-tune, first pass (4h, Sat midday/PM)**
Pretrained B1 (ADE20K init), 7-class head, CE loss with class weights + `ignore_index=0`, AdamW + cosine, batch 2 grad-accum 4, ~15 epochs, per-class IoU logged every epoch. Script has `--crop-size` CLI arg for 768 fallback. Deliverable: checkpoint + val log with checkpoint-gate numbers clearly visible.

**🔴 GO/NO-GO CHECKPOINT — Saturday evening**
Apply the two gates above. Decide path A or B.

**Phase 4 — Sunday path (3h)**
- **Path A (converged):** Iterate — class weight tuning, more epochs, TTA. Stop at mIoU 0.45 or 1pm, whichever first.
- **Path B (pivoted):** Load HF pretrained LoveDA checkpoint, adapt class mapping, validate, done by lunch.

**Phase 5 — Inference pipeline + emissions lookup (2.5h, Sun PM)**
- `infer.py`: image → seg mask + per-class % dict. Handle non-1024 inputs via resize + documented warning.
- `emissions.py`: lookup dict with inline citations (IPCC AR6 WG3 Ch. 7, EPA GHG Inventory, EDGAR v7). Every number has source + table ref + units in a comment.

Deliverable: `python infer.py --image demo.png` prints breakdown + CO2e.

### Weekend 2 — Agent + integration (MacBook)

**Phase 6 — FastAPI backend (2h, Sat AM)**
Endpoints: `POST /classify`, `GET /emissions`, `POST /simulate`. Pydantic models, CORS, model loaded once at startup. Deliverable: all three endpoints work via curl; Swagger at `/docs`.

**Phase 7 — ReasoningAgent + Claude Haiku + tools (4h, Sat PM)**
`ReasoningAgent` protocol, `ClaudeAgent` implementation, prompt caching on system + tools, `max_turns=5`. Four tools:
1. `get_land_breakdown()`
2. `get_emissions_estimate()`
3. `simulate_intervention(from_class, to_class, fraction)`
4. `recommend_mitigation(priority)`

Agree tool schemas before implementing. **If Phase 7 hits 5h, cut to 2 tools (breakdown + simulate), document the others as hackathon TODOs.** Deliverable: `agent_repl.py` produces a sustainability report for a pre-classified image.

**Phase 8 — Frontend + docs (3h, Sun)**
Single `index.html` + `app.js` (~200 lines JS): upload, preview, analyze, results panel with streaming report. `README.md` with architecture diagram. `HACKATHON_TODO.md` with tiered tasks:
- P0: map picker + live tile integration (note Sentinel-2 resolution warning)
- P1: more agent tools, bbox region selection
- P2: frontend polish, React migration
- P3: region-specific emissions factors

Deliverable: full loop in browser; teammate can clone, set API key, run two commands, see demo.

**Buffer: ~2h Sunday evening** for the thing that always goes wrong.

---

## Risk register

- **Phase 7 agent loop** is the riskiest estimate. Cut to 2 tools if it blows past 5h.
- **Phase 3 training speed** — if first epoch >45 min at 1024, drop to 768 via CLI flag.
- **$10 budget** — main risk is runaway agent loop. `max_turns=5` from day one.
- **Portfolio framing** — "LLM grounded in measured physical data via segmentation + cited emissions table," not "I fine-tuned SegFormer." Shape phase priorities accordingly.
