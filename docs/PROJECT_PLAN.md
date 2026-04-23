# Project: AI-Powered Land Use & Greenhouse Gas Analysis System

I'm building an AI system for an upcoming hackathon on the theme "Land Use & Sustainability." I need you to be my technical mentor and walk me through building this project step by step, starting from zero.

## What the project does

The system takes a satellite image as input and produces:

**Primary outputs:**
- Per-class land breakdown (e.g., "42% forest, 28% cropland, 18% urban, 12% water")
- Estimated greenhouse gas production/sequestration per region
- A natural-language sustainability report

**Secondary outputs (agent-driven):**
- Counterfactual analysis: "if I convert this cropland to solar panels, the emissions impact would be X"
- Mitigation recommendations grounded in real data

**Stretch goals for the hackathon itself (NOT for the pre-build):**
- Map picker (Leaflet + live satellite tile fetching) to replace the upload flow
- Live coordinate/bounding-box selection

## Architecture

Three pillars:

1. **Vision model** — a fine-tuned **SegFormer-B1** on LoveDA (combined Urban + Rural splits), producing pixel-level semantic segmentation across 7 classes: background, building, road, water, barren, forest, agriculture. Note: "background" is "other/unlabeled" and does NOT get an emissions factor — it's excluded from the breakdown or shown separately.
2. **Emissions grounding layer** — a documented lookup table sourced from IPCC AR6 / EPA / EDGAR, mapping land classes to CO2e factors (tons per hectare per year). **Global averages only** for v1; region-specific factors are a hackathon-day stretch.
3. **Reasoning agent** — Claude Haiku 4.5 via Anthropic API, with structured tool-calling for region queries, intervention simulation, and mitigation recommendations. Prompt caching on system prompt + tool definitions.

The three pieces connect via a FastAPI backend with clean endpoints. A minimal HTML/JS frontend demonstrates the end-to-end loop.

## Why this shape — "core + extensions"

The pre-hackathon build produces a working backend and minimal proof-of-concept frontend. During the hackathon, the team extends it — adding the map picker, polishing the frontend, adding more tools to the agent, handling edge cases. The pre-built part is infrastructure, not the whole project.

## Decisions already locked in

These are settled; don't re-litigate them unless you spot a critical problem:

- **Vision model:** SegFormer-B1 (not B2) — chosen for iteration speed on 16GB VRAM over peak accuracy. LoveDA is 1024×1024 at 0.3m/pixel.
- **LoveDA split strategy:** Train on combined Urban + Rural for robustness. Demo images may come from either domain.
- **Emissions factors:** Global averages, sourced from IPCC AR6 / EPA / EDGAR, each cited in code.
- **LLM:** Claude Haiku 4.5 via Anthropic API. API key already set up. Prompt caching expected for system prompt + tool defs.
- **Budget:** $10 hard spending cap for pre-hackathon dev only. Hackathon itself will have separate credits (team budget or event-provided). Cache aggressively for dev, but don't over-optimize the agent loop for token cost at the expense of reliability.
- **Agent abstraction:** `ReasoningAgent` interface with a Claude implementation as primary. Designed so a fallback (OpenRouter, Gemini, local) is a one-line swap. Do NOT build the fallback now — just make the abstraction clean.
- **Backend:** Python FastAPI.
- **Frontend:** minimal single-page HTML + vanilla JS (save React/polish for the hackathon team).
- **Demo format:** 2-3 pre-selected demo images as primary flow, arbitrary upload as secondary. Optimize the pipeline for the scripted demo, don't hardcode anything that breaks arbitrary input.
- **Image input for pre-build:** upload only, no live tile fetching.
- **Resolution constraint:** LoveDA is 0.3m/pixel aerial imagery. Any live tile fetching added during the hackathon MUST use comparable-resolution imagery (NAIP, Mapbox satellite, Esri World Imagery ~0.3m). Sentinel-2 at 10m/pixel will silently tank model performance — document this as an explicit constraint in `HACKATHON_TODO.md` so nobody wastes time on Sentinel Hub.

## Pre-committed fallback (critical)

**Explicit go/no-go checkpoint: end of Saturday, Weekend 1.** If the SegFormer fine-tune isn't converging well by this point, we pivot to a HuggingFace pretrained LoveDA checkpoint and reframe the project around the agent + grounding layer as the primary differentiator. This is committed now — don't let me rationalize pushing through on Sunday if the model is struggling. Include this checkpoint explicitly in the phase breakdown.

## Hardware setup

- **Primary coding machine:** MacBook Pro M5 Pro, 48GB unified memory. All application code, agent development, frontend, FastAPI work. Light inference only.
- **Primary training machine:** Ubuntu desktop with RTX 4070 Ti Super (16GB VRAM, CUDA) — accessed via SSH. This is where the SegFormer fine-tune runs.
- **Backup training machine:** Windows laptop (Intel Core Ultra 7 255HX, 32GB RAM, RTX 5070 Ti mobile GPU). **Native Windows with CUDA, NOT WSL** — I don't want WSL debugging as a failure mode. Pure emergency backup only.

I'm comfortable with SSH, remote dev workflows, Python, and PyTorch. I've done QLoRA fine-tuning on Qwen3-8B before, so I know the shape of a training pipeline. **This is my first computer vision project**, so explain CV-specific concepts (segmentation losses, mIoU, class imbalance handling, augmentation strategies, etc.) as they come up. Don't assume I know them.

## Time constraint

Two weekends, ~20-30 focused hours. Be aggressive about scope cuts. If something risks eating the timeline, call it out and recommend deferring it to hackathon day.

## How I want you to work with me

1. **Confirm you've read and understood this plan.** Flag anything you disagree with before we start.
2. **Propose a concrete phase breakdown** with hour estimates per phase and explicit go/no-go checkpoints (at minimum: end-of-Saturday Weekend 1 model convergence check).
3. **Work phase by phase.** Don't start the next phase until I confirm the current one works.
4. **Explain CV concepts as they come up.** Assume I'm technically capable but new to vision.
5. **Call out risks proactively.** If something is likely to break, eat time, or bite at the hackathon, tell me upfront.
6. **Portfolio-defensibility matters.** I want to discuss this project in technical interviews. Prefer choices that make for a stronger story over marginally easier ones.
7. **Check in before big decisions.** Dataset preprocessing strategy, loss function, augmentation pipeline, agent tool signatures — ask before committing.
8. **Keep me honest about scope.** If I try to add something that would bust the timeline, push back.

## Suggested phase structure (refine this)

**Weekend 1 — Vision model (primary training machine: Ubuntu via SSH)**
- Phase 0: Environment setup, remote dev workflow
- Phase 1: LoveDA download, class balance analysis, dataset exploration
- Phase 2: Data loading pipeline with augmentation
- Phase 3: SegFormer-B1 fine-tune — first pass
- **GO/NO-GO CHECKPOINT (end of Saturday): model converging?** If no → pivot to pretrained checkpoint
- Phase 4: Iterate on training (loss weighting, augmentation) OR integrate pretrained checkpoint
- Phase 5: Inference pipeline — image in → segmentation mask + percentages out
- Phase 6: Export weights, document results

**Weekend 2 — Agent + integration (primary coding machine: MacBook)**
- Phase 7: Emissions lookup table with IPCC/EPA/EDGAR citations
- Phase 8: FastAPI backend with `/classify`, `/emissions`, `/simulate` endpoints
- Phase 9: `ReasoningAgent` abstraction + Claude Haiku implementation with prompt caching
- Phase 10: 4-5 well-defined tools (get_emissions_for_region, simulate_intervention, recommend_mitigation, etc.)
- Phase 11: Minimal HTML/JS frontend proving the full loop end-to-end
- Phase 12: README, architecture doc, `HACKATHON_TODO.md` with tiered tasks for the team

## First step

Read this plan, confirm understanding, flag any remaining concerns, then propose the detailed phase breakdown with hour estimates and explicit checkpoints.