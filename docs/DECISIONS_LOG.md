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

### Phase 1 — LoveDA download + dataset analysis — 2026-04-23

**Decided:**
- Download LoveDA from Zenodo (record 5706578) directly via wget, not HuggingFace `datasets`. Canonical source, MD5-verifiable, raw PNGs on disk in known layout.
- Train on Train+Val only. Official Test split has withheld labels (LoveDA benchmark server), so val doubles as our model-selection set. No separate held-out test for us.
- **Class convention (clarifies earlier phase plan):** class 0 is "no-data" (ignored in loss AND excluded from stats), class 1 is "background" (trained as a real class, excluded from emissions aggregation at inference). Earlier plan said "background uses ignore_index=0" which conflated two separate things in LoveDA's labeling.
- **Loss function (locked for Phase 3):** weighted cross-entropy with median frequency balancing weights, mean-normalized. `ignore_index=0`. No focal loss.
- Exact MFB weights for classes 1–7: [0.255, 0.824, 1.730, 1.426, 1.748, 0.567, 0.451].
- Sampling strategy: uniform random across combined Urban+Rural. No weighted sampler — 6.9× imbalance is mild enough that MFB weighting alone handles it.

**Rejected alternatives:**
- HuggingFace `datasets` for download — `chloechia/loveda` is image-classification only (no segmentation masks). No clean HF mirror with masks found. Zenodo was the right call.
- Focal loss — imbalance is 6.9× (mild). Adds a γ hyperparameter to tune with no clear benefit at this scale. Reserved as a Phase 4 iteration lever if rare-class IoU refuses to move.
- Inverse frequency weighting — overweights rare classes, less stable than MFB for segmentation.
- Weighted/stratified batch sampler — overkill given mild imbalance. Urban+Rural combining already handles the domain-shift aspect.

**Surprises / learnings:**
- Imbalance is much milder than expected for a segmentation dataset. 6.9× ratio vs Cityscapes (~400×) or ADE20K (~2000×). LoveDA's 7-class aggregation is remarkably balanced; the "challenge" is more about domain shift and the amorphous background class than rare-class learning.
- **Val distribution differs meaningfully from train.** Val has ~2× more water pixels, ~1.4× more agri, and ~half the forest pixels compared to train. Implication: the Saturday forest-IoU gate is noisier than it looks (fewer forest pixels in val = less stable IoU). Agri is the safer bet for the gate.
- **Urban vs rural domain shift is dramatic for agri specifically:** 1.86% of urban train pixels vs 35.41% of rural. A 20× shift. Combined-split training isn't just "more data" — it's the only way the model sees agri often enough to learn it.
- Background at 35.8% of pixels is the largest class. LoveDA's "background" means miscellaneous labeled stuff (parking lots, construction, misc impervious surfaces) — heterogeneous by design, hard to learn. We weight it down hard (0.255) and it's excluded from emissions anyway, so low background IoU is tolerable for our use case.
- Visual sanity check on Train/Urban/1417: class palette aligned correctly with aerial features. No labeling bugs detected. User noted barren/forest can be visually confusing even to a human on 0.3m imagery — expect the model to confuse them too.

**Open questions carried forward:**
- tmux detach shortcut still not verified (verify when training starts in Phase 3)
- Windows backup machine still not smoke-tested
- If forest IoU is weak at the Saturday gate, do we pivot on forest-vs-agri OR clause, or iterate on class weights in Phase 4? Decision deferred to gate-evening.
- Background IoU likely to be weak due to class heterogeneity. If mIoU including background is embarrassingly low for portfolio purposes, consider reporting mIoU with/without background — cosmetic decision, not a training decision.

### Phase 2 — Data loading + augmentation — 2026-04-23

**Decided:**
- Two files in `scripts/`: `dataset.py` (the `LoveDADataset` class + `build_transforms()`) and `test_loader.py` (the 20-iter smoke test that is the phase deliverable).
- Combined Urban + Rural via a single sample list built by globbing both directories at construction time. No weighted sampler, no stratified batch sampler. Uniform random shuffle. This supersedes the "stratified Urban/Rural" wording in the original phase plan — Phase 1 already decided MFB + uniform sampling is sufficient.
- Mask dtype returned as `torch.long`, values in `{0..7}` as-is (standard PyTorch CE convention). Loss will use `ignore_index=0`.
- Resize to 1024×1024 in the Dataset itself (not assert-only). Rationale: user cannot guarantee every downstream image will be 1024 (demo images, arbitrary uploads during hackathon). Better to unify here than have a separate inference pipeline later.
- **Bilinear interpolation for images, nearest-neighbor for masks.** Albumentations `Resize` handles this via `mask_interpolation=0`. Bilinear on masks would produce fractional class indices that are meaningless — a pixel is either road or it isn't.
- Train aug order: resize → hflip → vflip → rotate90 → color jitter → normalize → ToTensor. Geometric before photometric is convention. All geometric ops p=0.5.
- Color jitter params: brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05. Mild. Aerial color carries real class signal (green channel distinguishes forest / agri / barren), so we don't want aggressive photometric aug destroying it.
- Val pipeline: resize → normalize → ToTensor. No augmentation.
- **Normalization: ImageNet stats** (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). Locked because SegFormer backbone is ImageNet-pretrained. NOT recomputing from LoveDA — that would mismatch what the backbone expects and silently hurt accuracy.
- TTA (test-time augmentation) deferred to Phase 5 inference pipeline, not used during val loop. Rationale: training-time val TTA slows epochs and measures "TTA mIoU" rather than "model mIoU," which is less useful for model selection.
- DataLoader defaults for smoke test: `batch_size=2`, `num_workers=2`, `pin_memory=True`, `drop_last=True` on train, `drop_last=False` on val, `shuffle=True` on train only.

**Rejected alternatives:**
- Shifting masks so `ignore_index=-100` (PyTorch default). Considered, rejected: unnecessary remap, no benefit, `CrossEntropyLoss(ignore_index=0)` works directly on the raw labels.
- Assert-only at 1024×1024 with no resize in the Dataset. Rejected because arbitrary inputs at inference time will require resizing anyway, so having the Dataset handle it keeps the train/inference pipelines aligned.
- Adding `RandomCrop` or scale augmentation. Rejected per PHASE_PLAN.md ("no scale/crop"). LoveDA tiles are already 1024×1024 and we train at full tile size; crop aug would fragment spatial context.
- Stronger color jitter (0.4/0.4/0.4/0.1). Rejected: aerial imagery has narrower natural color variation than ground-level photos, and class separability partially depends on color (forest green vs agri yellow-green vs barren brown). Mild is right.
- Val TTA during training. Rejected: slows epochs, confounds model-selection signal.

**Surprises / learnings:**
- Throughput was better than expected: 0.13s/batch at batch_size=2, num_workers=2. Training will be GPU-bound, not I/O-bound. Means we don't need to preemptively crank num_workers or enable persistent_workers for Phase 3.
- Both splits surfaced all 8 classes across just 20 random batches, including class 0 (no-data). Confirms `ignore_index=0` is doing real work, not overkill.
- Measured normalized image range `[-2.118, 2.640]` matches ImageNet normalization bounds exactly, confirming the normalization is wired correctly.
- 2522 train samples at batch 2 w/ grad-accum 4 = 1261 optimizer steps/epoch. At ~1.5-2s/step in Phase 3 (GPU-bound estimate), that's 30-40 min/epoch — right at the 45-min fallback threshold. Flagged for Phase 3 to watch carefully.

**Open questions carried forward:**
- All items from Phase 1 still open (tmux, Windows backup, Saturday gate pivot strategy, mIoU reporting cosmetics).
- First epoch in Phase 3 must be timed carefully — if it drifts past 45 min, drop to `--crop-size 768` per the phase plan's fallback rule.
