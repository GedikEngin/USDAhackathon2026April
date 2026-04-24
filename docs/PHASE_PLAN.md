# Phase Plan — Land Use & GHG Analysis System
 
> This file is the canonical phase breakdown and locked-in decisions. `PROJECT_PLAN.md` is the original vision doc. `DECISIONS_LOG.md` tracks what actually happened. `CURRENT_STATE.md` reflects the repo right now.
>
> **When this file and `DECISIONS_LOG.md` disagree, the decisions log wins.** The log is append-only and records actual events; this document is updated periodically to match reality.
 
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
- **Training resolution:** 1024×1024 full tiles, batch 2, gradient accumulation 4 (effective batch 8). Mixed precision (fp16). Fallback to 768 not needed — actual epoch time was ~3-4 min.
- **Class handling:** 8 logit slots. Class 0 (no-data) uses `ignore_index=0` in loss. Class 1 (background) is trained as a real class but excluded from emissions aggregation at inference.
- **Region semantics:** Whole image aggregated. No bbox, no connected components. Agent tools operate on aggregate percentages.
- **Emissions factors:** Global averages, IPCC AR6 / EPA / EDGAR cited inline in code.
- **LLM:** Claude Haiku 4.5 via Anthropic API, prompt caching on system prompt + tool defs, `max_turns=5` hard cap on agent loop.
- **Agent abstraction:** `ReasoningAgent` protocol; `ClaudeAgent` is the only implementation for pre-build. Fallback is a hackathon-day task.
- **Backend:** FastAPI. **Frontend:** single-page HTML + vanilla JS.
- **Budget:** $10 cap for pre-build. Agent dev is the only meaningful cost center.
- **Demo:** 2-3 pre-selected demo images primary; arbitrary upload secondary.
## Saturday Weekend 1 go/no-go checkpoint — ✅ PASSED (2026-04-23)
 
Both gates required passing; both did, from `best.pt` (epoch 10):
 
1. ✅ **Val mIoU ≥ 0.38** — achieved **0.5147** (35% over threshold)
2. ✅ **Forest IoU ≥ 0.55 OR agri IoU ≥ 0.55** — agri hit **0.596** at epoch 10 (forest peaked 0.406 across all epochs; agri carried the OR clause)
Full per-class IoU at best.pt: background 0.497, building 0.570, road 0.544, water 0.667, barren 0.346, forest 0.385, agri 0.596.
 
Decision: Phase 4 iteration skipped. See Phase 3 entry in `DECISIONS_LOG.md` for reasoning.
 
---
 
## Phases
 
### Weekend 1 — Vision (Ubuntu via SSH)
 
**Phase 0 — Environment & remote dev (1.5h, Fri eve) — ✅ DONE**
Conda env on Ubuntu box, CUDA/PyTorch sanity check, SSH + tmux workflow, 15-min smoke test on Windows backup (just `torch.cuda.is_available()`). Deliverable: you can edit locally, run on Ubuntu, see GPU util.
 
*Actual:* Conda env `landuse` live on Ubuntu. Windows backup smoke test deferred (never needed).
 
**Phase 1 — LoveDA download + dataset analysis (2h, Sat AM) — ✅ DONE**
Download Urban + Rural train/val. Write `dataset_stats.py` — per-class pixel frequency train vs val, 8 sample overlays, class imbalance ratios. Deliverable: `dataset_stats.md`. Decide loss weighting together before Phase 3.
 
*Actual:* Locked MFB weights and `ignore_index=0` convention. Imbalance milder than feared (6.9×); no focal loss needed.
 
**Phase 2 — Data loading + augmentation (2h, Sat AM/midday) — ✅ DONE**
PyTorch `Dataset`, albumentations pipeline (flips + rotate90 + color jitter, no scale/crop), Urban+Rural combined, `ignore_index=0`. Deliverable: DataLoader produces correct-shape batches for 20 iterations without errors.
 
*Actual:* `scripts/dataset.py` + `scripts/test_loader.py`. Throughput 0.13s/batch — not the bottleneck.
 
**Phase 3 — SegFormer-B1 fine-tune, first pass (4h, Sat midday/PM) — ✅ DONE**
Pretrained B1 (ADE20K init), 7-class head, CE loss with class weights + `ignore_index=0`, AdamW + cosine, batch 2 grad-accum 4, ~15 epochs, per-class IoU logged every epoch. Script has `--crop-size` CLI arg for 768 fallback. Deliverable: checkpoint + val log with checkpoint-gate numbers clearly visible.
 
*Actual:* `scripts/train.py` + `model/segformer-b1-run1/best.pt` (epoch 10, val_mIoU 0.5147). Training took ~50 min total, not the forecast 2¼ hours. Both Saturday gates passed decisively.
 
**🔴 GO/NO-GO CHECKPOINT — Saturday evening — ✅ PASSED**
 
---
 
**Phase 4 — SKIPPED**
 
Original plan had two paths:
- ~~**Path A (converged):** Iterate — class weight tuning, more epochs, TTA. Stop at mIoU 0.45 or 1pm, whichever first.~~
- ~~**Path B (pivoted):** Load HF pretrained LoveDA checkpoint, adapt class mapping, validate, done by lunch.~~
We hit val_mIoU 0.5147 — already past Path A's stop target of 0.45. Continuing to iterate has low expected value (val loss already diverging at epoch 15, class weight tuning is high-risk, TTA belongs in inference). Full reasoning in `DECISIONS_LOG.md` Phase 3 entry.
 
**Sunday hours reallocated to Phase 5.**
 
---
 
**Phase 5 — Inference pipeline + emissions lookup (2.5h, Sun AM/PM) — NEXT**
- `scripts/infer.py`: load `best.pt` → run on an image → produce (a) colored seg mask PNG and (b) per-class pixel % dict. Handle non-1024 inputs via resize + documented warning. Add `--tta` flag (hflip+vflip averaging) for +0.01-0.03 mIoU freebie; on by default.
- `scripts/emissions.py`: lookup dict with inline citations (IPCC AR6 WG3 Ch. 7, EPA GHG Inventory, EDGAR v7). Every number has source + table ref + units in a comment. Background class excluded from aggregation.
Must use `torch.load(..., weights_only=False)` to load `best.pt` (carries `pathlib.PosixPath` in stored args).
 
Deliverable: `python scripts/infer.py --image demo.png` prints breakdown + CO2e + saves colored mask PNG.
 
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
 
**Buffer: ~2h Sunday evening** for the thing that always goes wrong. *Weekend 1 used ~0 buffer (Phase 3 finished ahead of schedule) — extra cushion carried into Weekend 2.*
 
---
 
## Risk register
 
- ~~**Phase 3 training speed** — if first epoch >45 min at 1024, drop to 768 via CLI flag.~~ *Resolved — epochs ran in 3-4 min.*
- **Phase 7 agent loop** is now the riskiest remaining estimate. Cut to 2 tools if it blows past 5h.
- **$10 budget** — main risk is runaway agent loop. `max_turns=5` from day one.
- **Portfolio framing** — "LLM grounded in measured physical data via segmentation + cited emissions table," not "I fine-tuned SegFormer." Shape phase priorities accordingly. *This framing now has even more weight given we skipped Phase 4 iteration.*
- **Weak classes in the model:** barren (IoU 0.35) and forest (0.39) will visibly confuse on demo images. Consider picking demo images that don't hinge on these classes, or frame weakness honestly in the portfolio narrative.
## Environment gotchas (accumulated through Phase 3)
 
- Activate env each SSH/AnyDesk session: `conda activate landuse`
- PyTorch 2.11.0+cu126 (upgraded from 2.5 to satisfy transformers' CVE-2025-32434 check)
- `torch.load` on our checkpoints requires `weights_only=False` (Path objects inside args)
- DataLoader has intermittent worker segfaults; use `num_workers=0` for training. Data loading is not the bottleneck anyway.