# Decisions Log

> Append-only record of decisions made during each phase. Update at end of every phase chat. Never edit previous entries — if you reverse a decision, add a new entry noting the reversal.

## Format

Each entry:

```
### Phase N — [short title] — [date]

**Decided:**
- Thing, with brief rationale

**Rejected alternatives:**
- Thing we considered but didn't pick, with why

**Surprises / learnings:**
- Anything that didn't match expectations

**Open questions carried forward:**
- Things to decide in a later phase
```

---

## Entries

### Phase 0 — Environment & Remote Dev — 2026-04-23

**Decided:**
- Conda env named `landuse`, Python 3.11, on Ubuntu box at `~/Desktop/dev/USDAhackathon2026April`
- PyTorch installed via `pip` with `--index-url https://download.pytorch.org/whl/cu121`
- All project dependencies installed into `landuse` env (transformers, datasets, accelerate, albumentations, fastapi, anthropic, etc.)

**Rejected alternatives:**
- Plain `venv` — stuck with conda since it handles CUDA-linked packages more reliably

**Surprises / learnings:**
- Fresh Miniconda install required accepting Anaconda ToS explicitly via `conda tos accept` before env creation would proceed
- Driver reports CUDA 13.0, but nvcc toolkit is 12.0 — normal mismatch, cu121 PyTorch works fine
- Torch was accidentally installed into `base` env (Python 3.13) during smoke test before `landuse` env existed — not a problem, just noise
- tmux detach shortcut (Ctrl+B then D) not yet confirmed working — not blocking

**Open questions carried forward:**
- tmux detach to be verified when training actually starts in Phase 3
- Windows backup machine not smoke-tested (no access to machine during this session)
