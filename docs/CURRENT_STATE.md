# Current State

> Overwritten at end of every phase. Tells the next chat what actually exists in the repo right now. Do NOT paste code here — reference file paths and describe behavior.

---

## Last updated: 2026-04-23, end of Phase 0

## Repo layout

```
USDAhackathon2026April/
  data/
  model/
  scripts/
  frontend/
  .gitignore
  (plan docs)
```

## What works

- Conda env `landuse` (Python 3.11) active on Ubuntu box
- `torch.cuda.is_available()` returns `True`, device confirmed as RTX 4070 Ti SUPER
- All packages installed: transformers, datasets, accelerate, albumentations, opencv-python-headless, matplotlib, seaborn, tqdm, fastapi, uvicorn, python-multipart, anthropic, pydantic

## What's half-done

- tmux confirmed installed but detach shortcut not yet verified

## What's next

Phase 1: LoveDA download + dataset analysis.
- Download Urban + Rural train/val splits
- Write `dataset_stats.py` — per-class pixel frequency, 8 sample overlays, class imbalance ratios
- Deliverable: `dataset_stats.md` to inform loss weighting decision before Phase 3

## Known issues / gotchas

- Activate env each SSH session with `conda activate landuse`
- nvcc reports 12.0, driver reports CUDA 13.0 — normal, no action needed
- Windows backup machine not yet smoke-tested

## Key numbers (fill in as they're measured)

- Val mIoU: —
- Per-class IoU (bg, building, road, water, barren, forest, agri): —
- Training epoch time: —
- API cost spent so far: $0
