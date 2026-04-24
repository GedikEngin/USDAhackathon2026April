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
### Phase 3 — SegFormer-B1 fine-tune, first pass — 2026-04-23
 
**Decided:**
- `scripts/train.py` implements the full training loop. ~500 lines, heavy docstrings.
- Pretrained checkpoint: `nvidia/segformer-b1-finetuned-ade-512-512` (not mit-b1). Decode head reinitialized for 8 classes via `ignore_mismatched_sizes=True`; backbone + rest of decode head transferred cleanly.
- `num_labels=8` passed to `from_pretrained` (not 7). Class 0 (no-data) occupies index 0, ignored in loss via `ignore_index=0`. Keeps logit indices aligned with label values — simpler than remapping.
- Optimizer: AdamW, LR 6e-5, weight_decay 0.01. Same LR for backbone and decode head — differentiated LR deferred as a Phase 4 lever we ended up not needing.
- Schedule: linear warmup 500 steps → cosine decay to 0 over 15 epochs. Warmup completed ~epoch 1.6.
- Mixed precision enabled by default via `torch.amp.autocast(dtype=torch.float16)` + `GradScaler`. Stable throughout training, no loss scale blow-ups.
- IoU computed from a confusion matrix accumulated over the full val set, not averaged per-batch. Per-batch averaging is numerically incorrect (classes absent from a batch poison the average).
- Two mIoU variants logged each epoch: `val_mIoU` (classes 1-7) and `val_mIoU_no_bg` (classes 2-7). Closes the Phase 1 open question about reporting cosmetics — we log both and decide at portfolio time.
- CSV log written to `model/segformer-b1-run1/train_log.csv` with per-class IoU columns. Grep-friendly. No W&B/TensorBoard needed.
- `best.pt` tracked by val_mIoU; epoch checkpoints retained to last 3 via `prune_old_checkpoints`. After Phase 3 closed, additionally pruned to just `best.pt` + `epoch_15.pt` to save disk.
**Rejected alternatives:**
- `nvidia/mit-b1` (ImageNet backbone only). Rejected because the ADE20K checkpoint also gives us useful decode-head-except-classifier weights for free, and the mismatch machinery in transformers handles the 150→8 swap cleanly.
- W&B / TensorBoard. Overkill for a single 15-epoch run; stdout + CSV is sufficient for the Saturday gate check.
- Per-batch mIoU averaging. Mathematically wrong; confusion matrix is the right approach.
- Differentiated LR (backbone vs decode head). Kept as a Phase 4 lever; never needed.
**Surprises / learnings:**
- **Training was dramatically faster than planned.** Estimated 2¼ hours; actual 50 minutes. Epochs averaged 3-4 minutes, not the 30-40 min I'd forecast in Phase 2. Consequence: grad_accum 4 was unnecessary. At batch 2, only 7 GB of 16 GB VRAM was used — could have used batch 4 or even 8 without grad accumulation. Noting for any future training runs.
- **Gate passed in ~12 minutes of training.** val_mIoU cleared 0.38 at epoch 2. Agri cleared 0.55 at epoch 3 (0.599). The "~10 epochs to the gate" assumption was very conservative.
- **Overfitting started at epoch 10-11.** Train loss kept dropping (0.48 at ep10 → 0.41 at ep15) but val loss rose (0.85 → 1.03) and val_mIoU plateaued around 0.51. `best.pt` correctly captures the epoch 10 peak.
- **Forest never cleared 0.55; agri carried the OR clause.** Phase 1 flagged agri as the safer bet for exactly this reason. Forest peaked at 0.406 (epoch 11). Barren was weakest (peak 0.38).
- **Three unexpected errors hit during training, all environment-level.**
  - First: `transformers` refused to load the `.bin` checkpoint due to CVE-2025-32434 requiring PyTorch ≥2.6. Fixed by upgrading torch 2.5 → 2.11.0+cu126.
  - Second: `torch.load` on our own checkpoint failed during `--resume` because torch 2.6+ defaults `weights_only=True`, which can't unpickle the `pathlib.PosixPath` stored inside `args`. Fixed by passing `weights_only=False` explicitly.
  - Third: DataLoader worker segfault mid-epoch-5. Root cause unknown (/dev/shm was fine, no kernel OOM, no trace in dmesg). Resumed with `num_workers=0` and trained cleanly to completion. Candidate causes: torch 2.11 × albumentations × multiprocessing interaction; or a corrupted worker state at that exact moment. No time was spent diagnosing further — `num_workers=0` is a fine permanent workaround since data loading wasn't the bottleneck.
- **Checkpoint format has a small footgun.** `save_checkpoint` stores `vars(args)`, which contains `Path` objects. Combined with torch 2.6+ `weights_only=True` default, this breaks resume-from-disk without `weights_only=False`. Phase 5's inference script must do the same override. Cleanup for a future refactor: serialize `args` as a plain dict of strings.
**Open questions carried forward:**
- All previous open questions now resolved or moot:
  - tmux detach — confirmed working during Phase 3.
  - Saturday gate pivot — not triggered; both gates passed.
  - mIoU reporting cosmetics — resolved by logging both variants.
  - Windows backup — never needed; leaving untested.
- **New for Phase 5:** TTA (horizontal + vertical flip averaging) is a cheap +0.01 to +0.03 mIoU freebie for inference. Should be a `--tta` flag in `infer.py`, on by default for demo images.
- **Phase 4 SKIPPED.** Per the decision below, we are going directly from Phase 3 to Phase 5.
**Decision: skip Phase 4 entirely.**
 
PHASE_PLAN.md's Phase 4 Path A target was val_mIoU 0.45. We hit 0.5147 at epoch 10. Continuing to iterate would yield low expected value:
 
- More epochs won't help — val loss is already diverging from train loss at epoch 15.
- Class weight tuning could push forest/barren up a few points but risks hurting stronger classes.
- TTA at inference belongs in Phase 5, not Phase 4.
The project's portfolio story is "LLM agent grounded in measured physical data via segmentation + cited emissions table." A marginally better SegFormer IoU is not the story. Sunday hours go to Phase 5 (inference + emissions) instead. This decision supersedes PHASE_PLAN.md's Phase 4 section.

### Phase 5 — Inference pipeline + emissions lookup — 2026-04-23

**Decided:**
- Two files in `scripts/`: `infer.py` (full inference pipeline, CLI-driven) and `emissions.py` (pure-lookup module, no torch dependency).
- **Dual emissions reporting per class.** Every emissions-relevant class (building, road, water, barren, forest, agriculture) carries BOTH an `annual_tco2e_per_ha_per_yr` (ongoing flux) AND an `embodied_tco2e_per_ha` (one-time stock / cost of the surface). User explicitly requested two numbers for annual ongoing vs stored stock. Agent will consume both in Phase 7.
- **Every number has an inline `[SRC-n]` tag** pointing to a citation block at the bottom of `emissions.py`. Six sources: IPCC AR6 WGIII Ch. 7, EPA GHG Inventory 2024, EIA CBECS 2018, IPCC 2019 Guidelines Vol. 4, JRC EFIResources 2018, EDGAR v7. `cite(src_key)` helper exposes full citation string for agent tools to surface in reports.
- **Palette is intuitive, not official.** Confirmed no official LoveDA palette exists anywhere in upstream sources — neither the Junjue-Wang repo nor torchgeo defines one. Both use matplotlib's default colormap for example plots. We chose: water=dodger blue, forest=forest green, building=crimson, road=dark grey, barren=tan, agri=gold, background=light grey, no-data=black. Defined as `PALETTE` dict in `infer.py`, easy to swap.
- **TTA on by default with `--no-tta` escape hatch.** 3 forward passes (original + hflip + vflip), logits averaged before argmax. Adds ~2x wall-clock vs no-TTA, still sub-second per image at 1024x1024.
- **Non-1024 inputs are resized with a warning.** Bilinear on input, nearest-neighbor on the output mask if we need to resize back. Flagged explicitly because LoveDA is 0.3m/px and substantially different GSD will silently hurt accuracy.
- Classes 0 (no-data) and 1 (background) are EXCLUDED from emissions aggregation but tracked and reported in a separate `excluded_breakdown` dict on the result. This matches the Phase 1 decision ("background trained as real class, excluded from emissions aggregation at inference").
- `compute_emissions()` returns an `EmissionsResult` dataclass with `per_class`, `total_area_ha`, `assessed_area_ha`, `excluded_fraction`, `total_annual_tco2e_per_yr`, `total_embodied_tco2e`. That's the exact shape Phase 7's `get_emissions_estimate` tool will surface to Claude.
- `--json-out` CLI flag writes the full report dict as JSON. Weekend 2 agent will consume this shape.
- `infer.py` loads checkpoint with `weights_only=False` per the Phase 3 gotcha, and reads state dict from `ckpt["model"]` (NOT `ckpt["model_state_dict"]` — see Surprises).

**Rejected alternatives:**
- Single annual-only number for emissions (option (c) from the phase-opening discussion). User chose option (b) — annual + embodied both — for richer agent reasoning surface.
- Single embodied-only. Would've erased the "net sink vs net source" framing that makes forest sequestration legible.
- Matplotlib colormap for the mask. Rejected because inline SVG-style legend-able palette maps better to Phase 8's frontend, and a deterministic palette makes unit-testable output.
- Using the `transformers` default post-processor. Rejected because we want full control over the upsample path (SegFormer outputs 1/4 resolution logits; we bilinear-upsample explicitly to input size).
- TTA off by default. Rejected because demo-path quality matters more than the ~2x inference latency, and Phase 6 FastAPI can expose a param if latency becomes an issue.

**Surprises / learnings:**
- **Checkpoint key footgun (round 2).** Phase 3 flagged that `weights_only=False` is needed. What we didn't catch then: `train.py`'s `save_checkpoint` stores the state dict under `ckpt["model"]`, not `ckpt["model_state_dict"]` (which is the HuggingFace / pytorch-lightning convention I assumed). The first `infer.py` draft looked for `"model_state_dict"`, fell through silently to the `else` branch, and loaded the *bare checkpoint dict* (including `optimizer`, `scheduler`, etc. as "unexpected keys") as a state dict. The result: the model ran with the ADE20K-pretrained backbone + a randomly-initialized 8-class head, and "worked" well enough to produce vaguely-plausible output. Warning signs were there — 207 missing keys on load, `unexpected keys: ['model', 'optimizer', 'scheduler']` — but easy to miss in a noisy first run. **Fix: one-line change to `ckpt["model"]` in `infer.py`.** The broader lesson: any future refactor of `save_checkpoint` should serialize args as plain strings AND adopt the `"model_state_dict"` convention; until then, infer.py stays pinned to the current format.
- **Preprocessing parity verified.** Built a side-by-side tensor comparison between `dataset.py`'s val pipeline (PIL → albumentations `A.Resize` + `A.Normalize` + `ToTensorV2`) and `infer.py`'s pipeline (PIL → `.resize` + manual `/255` + mean/std + `permute`). Max absolute diff: `4.8e-7`. Pure float32 roundoff. This removes an entire class of "is inference preprocessing broken?" worries going forward.
- **Forest's confusion partner is background, NOT agriculture.** Phase 1 predicted forest↔agri confusion on the assumption that rural tiles with ambiguous vegetation would drift between the two green classes. Reality: on diagnostic runs, ~45% of GT-forest pixels that got misclassified went to background (class 1), not agri. The model has learned an extremely strong prior "ambiguous green vegetation → background" — confidence gap of 0.90 on those predictions. This is consistent with the Phase 1 observation that background is heterogeneous and captures a lot of rural "stuff that's vegetation but not clearly forest." Agri was fine; forest is the collapse vector.
- **Rural tile 2523 is a pathological input** and a useful demonstration of the limits of the model. GT is 63% background, 28% forest, 8% water. Model predicts 91% background, 9% water, zero forest. Every single GT-forest pixel (100%) got classified as background with mean probability 0.928. Not a pipeline bug — the model genuinely considers this type of mixed-vegetation rural scene to be background-class territory. Consistent with Phase 3 forest IoU of 0.39 and the training distribution (val has half the forest pixels of train). **Not fixable in Phase 5.** Demo-image selection in Phase 8 should avoid images of this pattern, OR include them deliberately so the agent can surface the uncertainty honestly ("91% of pixels classified as heterogeneous-background; consider multi-spectral imagery for stronger vegetation discrimination").
- **Inference is fast.** ~0.2–0.4 s per 1024×1024 tile on the 4070 Ti SUPER with TTA on. Phase 6 FastAPI can serve this in a single thread without breaking a sweat; no need for batching or async model calls until we have real concurrent load.
- **Hit the MISMATCH warning on every load.** It's benign — it's `from_pretrained` telling us the ADE20K 150-class head was dropped before our 8-class `best.pt` weights were applied. Cosmetic only, but noisy. A `warnings.filterwarnings` or `transformers.logging.set_verbosity_error()` call would suppress it cleanly. Left as-is for now: the noise is useful during integration debugging.

**Open questions carried forward:**
- For Phase 8 demo-image selection: pre-run inference on 10–20 candidate val tiles and rank by (a) visual appeal, (b) model quality (pixel accuracy vs GT), (c) class diversity. 3546.png (86.7% forest) is a good "model succeeds" tile. 2523.png is a good "model's limits" tile *if* we want to showcase the agent surfacing uncertainty. Avoid tiles with >60% forest GT where the rural/mixed-vegetation style collapses to background.
- **Wetlands caveat on water class.** LoveDA doesn't distinguish wetlands from open water. Our emission factor treats water as ~neutral (baseline open freshwater per IPCC AR6 WGI Ch. 5), but wetlands would be a strong sink (-8 to -12 tCO2e/ha/yr). Agent should flag this when water fraction is high. Noted in `emissions.py` inline; not a blocker.
- **Building operational number assumes single-story footprint.** 65 tCO2e/ha/yr is derived from EIA CBECS 2018 avg energy intensity × eGRID-weighted emissions factor, at 100% footprint coverage. Multi-story scales roughly linearly with floor-area ratio. Noted inline in `emissions.py`; agent should caveat if asked about commercial/urban-core imagery.
- **For Phase 7 agent tool design:** the `EmissionsResult` dataclass plus the `SOURCES` dict give the agent everything it needs to compose grounded reports with citations. Tool schemas should surface `[SRC-n]` keys explicitly in responses so the LLM can weave citations into natural-language output without making them up.

**Weekend 1 closed.** Phase 5 ran well under its 2.5h budget; total Weekend 1 time was roughly half the planned envelope. Zero hours used from the Sunday buffer. Entering Weekend 2 with extra cushion carried forward to Phase 7 (the riskiest remaining estimate).

### Phase 6 — FastAPI backend — 2026-04-24

**Decided:**
- Three files under new `backend/` package: `main.py` (FastAPI app, lifespan, endpoints, CORS), `inference.py` (InferenceEngine singleton, loads best.pt once at startup, exposes `classify(image_bytes)`), `models.py` (Pydantic request/response schemas). Plus empty `__init__.py`.
- Endpoints as planned in PHASE_PLAN.md: `POST /classify`, `GET /emissions`, `POST /simulate`. Added `GET /` and `GET /health` as cheap meta endpoints for smoke-testing and frontend probes.
- **Multipart upload for `/classify`** (FastAPI `UploadFile`). Base64-in-JSON rejected: multipart plays nice with the eventual frontend `<input type="file">` and is standard for file posts.
- **Base64-encoded colored mask PNG in the JSON response body**, not a separate `GET /mask/{id}` endpoint. Rejected the separate-endpoint version because it would require server-side state. Inline base64 is stateless and ~500 KB over localhost is a non-issue. If it becomes one in Phase 8, we can switch then.
- **`simulate_intervention()` lives in `scripts/emissions.py` as a pure function**, `/simulate` is a thin wrapper. This was the core Phase 6 architecture decision: the Phase 7 agent will call the pure function directly without an HTTP round-trip, and the backend gets free reuse. `SimulationResult` dataclass returned with `before`, `after`, `delta_annual_tco2e_per_yr`, `delta_embodied_tco2e`, `converted_area_ha`, `narrative`.
- **Model loaded once at startup** via FastAPI `lifespan` async context manager. Checkpoint path overridable via `LANDUSE_CHECKPOINT` env var; defaults to `model/segformer-b1-run1/best.pt`. Single-worker uvicorn is sufficient — inference is sub-400 ms and the demo will have queue depth 0.
- **CORS permissive (`allow_origins=["*"]`)** for dev. Frontend will be served separately in Phase 8 and may live on any localhost port. Not a production concern for this project.
- **Run command: `uvicorn backend.main:app --app-dir . --host 0.0.0.0 --port 8000`.** The `--app-dir .` flag is required because without it uvicorn imports `backend.main` before our `sys.path.insert` runs, and can't find the `backend` package.
- **Source citations derived at response-build time** from `LAND_USE_EMISSIONS`, not duplicated onto each `per_class` row. Original assumption was wrong — `EmissionsResult.per_class` rows have only `pixel_count/pixel_fraction/area_ha/annual_tco2e/embodied_tco2e`, no source strings per-row. Backend now collects SRC keys by walking classes actually present in the result and looking up `LAND_USE_EMISSIONS[name].annual_source / embodied_source`.
- **Validation on `SimulateRequest`:** `total_area_ha > 0`, `fraction ∈ [0, 1]`, `current_percentages` must sum to ~100 (tolerance 95–105 for float drift), `from_class` and `to_class` must be valid emission-classes (no no-data/background), `from_class` must be non-zero in the image. All enforced via Pydantic + the `simulate_intervention` function, returning HTTP 400 with a useful message on violation.

**Rejected alternatives:**
- Running on the MacBook with MPS (original Phase 6 suggestion). User chose to stay on Ubuntu — keeps the model where the checkpoint lives, avoids a copy, avoids MPS edge cases. Demo portability (if any) is a Phase 8 concern.
- Base64-in-JSON upload. Multipart is standard, simpler on the frontend side.
- Auth, rate limiting, request logging middleware. Unnecessary for local dev + demo.
- Streaming responses. Not needed — full response fits in <1 MB and inference is sub-second.
- Saving uploads to disk. Process in-memory, return result, forget.
- Implementing `/simulate` logic inline in the FastAPI handler. Pure function in `emissions.py` is cleaner and enables Phase 7 agent tool reuse without HTTP.
- Auto-injecting `total_area_ha` into `/simulate` from a prior `/classify` call. Considered making the endpoint stateful; rejected. Client passes `total_area_ha` explicitly, which keeps the endpoint pure. Phase 8 frontend will just remember the value from its own `/classify` response.

**Surprises / learnings:**
- **`compute_emissions` keys on int class IDs, not names.** Phase 5's decisions log didn't flag this explicitly — Phase 5's `infer.py` apparently handled the int conversion internally, or I didn't read it closely enough when planning Phase 6. First `/classify` call came back with `per_class: {}` and an `excluded_breakdown` full of `unknown_forest`, `unknown_building`, etc. Root cause: `compute_emissions` does `CLASS_NAMES.get(class_id, f"unknown_{class_id}")`, so string keys got turned into `unknown_`-prefixed names and routed to excluded. **Fix:** backend passes `{0: count_0, 1: count_1, ..., 7: count_7}` (int keys from the module-level `CLASS_NAMES` dict in `emissions.py`). Same bug hit `simulate_intervention` in its first draft; fixed with a module-level `_NAME_TO_ID` reverse map and a `pct_to_int_keyed_counts` helper.
- **`per_class` row field names are different and simpler than I assumed.** Real shape: `pixel_count/pixel_fraction/area_ha/annual_tco2e/embodied_tco2e` (5 fields, all floats). I'd originally speced `pixel_count/pixel_pct/area_ha/annual_tco2e_per_yr/embodied_tco2e/annual_source/embodied_source/annual_factor_per_ha/embodied_factor_per_ha` in Pydantic. Had to introspect `emissions.py` to get the truth, then rewrite `PerClassEmissions` and `_emissions_result_to_report` to match. Sources moved to a derive-at-response-time pattern.
- **`excluded_breakdown` values are 0.0–1.0 fractions, not 0–100 percentages.** Consistent with `pixel_fraction` in per_class rows. Frontend display in Phase 8 will need to multiply by 100.
- **Numerical match with Phase 5 is exact.** `/classify` on `Val/Urban/3546.png` returned `total_annual_tco2e_per_yr: -50.44` and `total_embodied_tco2e: 5542.81` — same to the decimal as Phase 5's CLI run logged in `CURRENT_STATE.md`. Confirms the backend is reusing Phase 5 math correctly, no silent conversion drift. Inference latency 313 ms end-to-end via HTTP (including JSON + base64 encode), same order as Phase 5's CLI measurement.
- **Simulation math is arithmetically exact against the factor table.** Hand-verified: 100% agri→forest on 3546.png (0.352 ha) delta_annual = `0.352 × (−8 − 2.5)` = −3.70; got −3.69 (float rounding). 50% building→forest (0.025 ha) delta_annual = `0.025 × (−8 − 65)` = −1.79; got −1.79. Embodied deltas also exact.
- **The `--app-dir .` footgun.** First `uvicorn` invocation died with `ModuleNotFoundError: No module named 'backend'` because uvicorn's importlib step runs before our in-module `sys.path.insert`. Two workarounds: `--app-dir .` (cleanest) or `PYTHONPATH=. uvicorn ...`. The `sys.path.insert` at the top of `backend/main.py` is still useful for importing `scripts/emissions` within the app, but it doesn't help uvicorn find the `backend` package itself.
- **`python-multipart` is a separate install from `fastapi`.** Not obvious from the FastAPI error. Would have bitten us without the pre-install.
- **Benign `MISMATCH` noise on startup**, same as Phase 5's `infer.py`. Comes from `from_pretrained` dropping the ADE20K 150-class head before applying our 8-class weights. Harmless; ignored.

**Open questions carried forward:**
- For Phase 7 agent tool design: the backend's `EmissionsReport` (pydantic, with `sources_cited` dict populated) is a good template for the shape agent tools should surface. Tool outputs should surface `[SRC-n]` keys plus the full citation string so the LLM can weave citations into natural-language output without fabricating them.
- `response_model=ClassifyResponse` on `/classify` makes FastAPI re-validate and re-serialize the response, which for a 500 KB base64 string is technically wasteful. If Swagger or latency ever complains, drop `response_model` and return a plain dict. Not urgent — still sub-400ms end-to-end.
- `simulate_intervention` silently drops unknown class names in `current_percentages` (via `_NAME_TO_ID.get(...) or continue`). Trade-off: more resilient to frontend drift, but could mask real bugs. Revisit if Phase 8 ever passes stale class names.
- Validation on `SimulateRequest.current_percentages` accepts sums in [95, 105] to tolerate float drift. If the agent in Phase 7 ever synthesizes inputs from non-classify sources (e.g. a hand-specified scenario), this tolerance might need to widen or tighten.

**Phase 6 closed well under budget.** ~2h including a ~30 min debug detour on the int-key + field-name mismatches. Weekend 2 cushion carried into Phase 7 (the riskiest remaining estimate: agent loop with `max_turns=5` cap + tool schema design).

### Phase 7 — ReasoningAgent + Claude Haiku 4.5 + tools — 2026-04-24

**Decided:**
- **Four files under new `agent/` package**, not a single `scripts/agent.py`: `__init__.py`, `base.py` (ReasoningAgent Protocol + AgentReport + ToolCallRecord), `tools.py` (AgentState + 4 tool schemas/impls + dispatch_tool), `claude.py` (ClaudeAgent). Plus `scripts/agent_repl.py` as the CLI entry. A protocol when we only ship one impl is justified because PHASE_PLAN locked "fallback is a hackathon-day task" — a package beats a single file for that extensibility, and the cost is ~80 lines.
- **Four tools, all via pre-classified AgentState injection (option A, not tool-driven classify):** `get_land_breakdown`, `get_emissions_estimate`, `simulate_intervention`, `recommend_mitigation`. The PHASE_PLAN wording ("sustainability report for a pre-classified image") made option A the natural read; option B would burn a turn on a deterministic classify step and bloat context with the mask.
- **Unit convention: all "how much of X" values are 0–100 percentages in every tool input and output.** `simulate_intervention`'s `fraction_pct` parameter is 0–100 at the tool boundary; the tool layer divides by 100 before calling `emissions.simulate_intervention()` (which uses 0.0–1.0 internally). Rationale: Haiku reasons more reliably when scales are consistent across the conversation — mixing 0–1 and 0–100 is exactly the kind of thing that produces `fraction=50` meaning 50× the intended amount.
- **`recommend_mitigation` is a deterministic menu, not an LLM call inside the tool.** Tool enumerates all `(from_class in image, to_class ≠ from_class)` pairs, computes deltas at 100% conversion, sorts by the requested priority axis, returns top 5 plus notes. The system prompt explicitly instructs the agent to weigh tradeoffs and rule out absurd candidates. Keeps Python doing arithmetic/ranking, LLM doing judgment/narrative. Verified working: on 2523.png priority=annual the tool ranks "convert water to forest" as the top option, and the agent correctly caveats that "reforestation of open water is ecologically unusual and may not be practical."
- **Tool outputs surface `[SRC-n]` tags AND full citation strings together.** `get_emissions_estimate` returns per-class rows each carrying `annual_src` / `embodied_src` (the SRC-n key), plus a separate `sources_cited` dict mapping each SRC-n that actually applies to the full citation string. Deduplicated to only classes present in the image. This gives the model both the tag to inline into prose and the reference text to render first-mention, without fabrication. Same pattern was carried forward from Phase 6's `/classify` response shape.
- **Conditional `model_caveats` in `get_land_breakdown`**, surfaced based on composition: forest >20% triggers the forest-IoU caveat (Phase 5 collapse-to-background finding), background >40% triggers the excluded-fraction caveat (pathological 2523-type tiles), barren >10% triggers the weakest-class caveat, water >5% triggers the wetlands-vs-open-freshwater caveat. These caveats are the honest-uncertainty mechanism — the tool hands the model pre-formulated concerns that the prompt instructs it not to bury.
- **`agent/claude.py` default model: `claude-haiku-4-5`. Default `max_turns=5`. Default `max_tokens=2048`.** Haiku was locked in PHASE_PLAN for budget; 5 turns is the phase-plan cap; 2048 output tokens is well above typical report length (~1000 tokens observed).
- **Prompt caching: `cache_control: {"type": "ephemeral"}` set on `system[0]` and on the last entry of `tools[]`.** Standard two-breakpoint placement. See Surprises for what happened.
- **Max-turns behavior: on the final turn, call without `tools` kwarg to force synthesis.** Otherwise a model that keeps emitting tool_use at the cap would hand back a fragmented response. Tested in `smoke_agent.py`: final-turn call omits tools, model is forced to produce text.
- **Tool errors surface as `is_error: True` tool_result blocks.** The agent loop wraps every `dispatch_tool` call in try/except; a raised `ValueError` (e.g., `simulate_intervention(from_class="building")` when no building is in the image) returns the error string as the tool_result content with `is_error=True`, so the model can try a different strategy rather than the session erroring out. Verified on the `building→forest` error-recovery scenario (offline) and implicitly on live calls via the schema's `enum` constraints preventing most bad calls from reaching the dispatch.
- **System prompt persona: sustainability-first blend** (data-first analyst + practical-tradeoffs consultant + honest-about-uncertainty scientist). Explicit rules: start with breakdown, chain independent tool calls in one turn, never call same tool twice with same input, never blindly trust `recommend_mitigation` ranking, every number carries its SRC-n tag, full citation on first mention, surface `model_caveats` / `assumptions` honestly. Output format guidance: headline → composition → current footprint → intervention + rationale → caveats; concise prose.
- **REPL pre-classifies once at startup** using `backend.inference.InferenceEngine` imported directly (no HTTP round-trip). Builds `AgentState`, hands it to `ClaudeAgent`. Each user query becomes a fresh `agent.run(state, query)` call. This is deliberate but see Known Issues.
- **`.env` file + `python-dotenv`** for `ANTHROPIC_API_KEY` loading. Added `.env` to `.gitignore`. `scripts/agent_repl.py` calls `load_dotenv(PROJECT_ROOT / ".env")` at import time, before any agent imports. Verified key loads and isn't tracked by git.
- **Two offline smoke tests at repo root:** `smoke_tools.py` exercises every tool + dispatch + unit conversion + 3 error paths against a realistic AgentState rebuilt from 3546.png measurements; `smoke_agent.py` exercises the loop using a scripted fake Anthropic client covering happy path, tool-error recovery, max_turns cap, and the edge case of a model still emitting tool_use on the forced-final turn. Both pass without API credentials. These gave us full confidence before spending a cent on live calls.

**Rejected alternatives:**
- **Option B (tool-driven classify)** — having the agent call a `classify_image(path)` tool on turn 1. Would burn a turn on something deterministic, add mask-sized blobs to context, and muddle the "agent reasons over measurements" story. Option A fits the phase-plan wording exactly.
- **`fraction` as 0.0–1.0 in tool 3**, matching the underlying Python signature. Rejected in favor of 0–100 because internal consistency across all tools is more important than zero-conversion-layer purity; LLMs are demonstrably more error-prone with mixed scales.
- **Tool 4 returning a single "best" intervention** rather than a ranked menu. Rejected because the ranking axis is mechanical and a single recommendation hides the tradeoff surface the LLM needs to reason over. Menu + LLM judgment > deterministic pick.
- **Tool 4 giving the LLM the raw factor table** and letting it do its own ranking. Rejected because arithmetic over six factors × six priorities is exactly the thing LLMs slip on (seen in other projects: subtle sign errors, dropped terms), and we already have a working pure function. Keep Python math, let LLM narrate.
- **Dropping Tool 4 entirely** (the phase-plan pre-authorized cut if we ran long). Not needed — Phase 7 landed inside its time budget, and the tool added real value on the live runs.
- **A single-file `scripts/agent.py`** instead of the `agent/` package. Cheaper upfront but worse for the "hackathon teammate adds fallback" scenario the phase plan anticipates.
- **Fallback agent implementation (non-Anthropic or rules-based)** for pre-build. Locked as a hackathon-day task per PHASE_PLAN. Protocol exists so this slots in cleanly when the time comes.
- **Model: Claude Sonnet 4.5** — would have solved the caching-minimum problem (Sonnet threshold is 1,024 vs Haiku 4.5's 4,096) but costs ~5× per token and was explicitly locked to Haiku in PHASE_PLAN for budget reasons. Not worth breaking the lock for a $0.15 budget optimization.
- **Padding the system prompt to >4,096 tokens** with few-shot examples to activate caching. Real work (improving agent behavior) but not justified at our scale — budget impact of no-caching is trivial, and we'd rather spend Phase 7 cushion on live testing than on prompt engineering for a feature that pays back $0.15 total.
- **Having `agent_repl.py` call the backend via HTTP** instead of importing `InferenceEngine` directly. Rejected because the agent is a Python process and there's no reason to make it an HTTP client of a separate FastAPI process on the same box. Direct import is simpler, faster, and avoids a uvicorn-running-in-background dependency. The Phase 6 `/classify` endpoint is for the frontend's benefit, not the agent's.
- **A persistent cross-query session in the REPL** (keep the `messages` list alive across user queries so tool results from query 1 remain available in query 2). Considered; rejected for Phase 7. Pro: saves ~$0.005 per follow-up. Con: blurs turn-budget accounting, risks context growing unbounded, adds one more thing to reason about. Filed as Phase 8 polish.

**Surprises / learnings:**
- **Prompt caching silently did not fire** on any live call. Both `cache_read_input_tokens` and `cache_creation_input_tokens` were 0 on every turn across 4 live sessions. Investigation (per Anthropic docs): **Haiku 4.5's minimum cacheable prefix is 4,096 tokens**, raised from the 1,024-token floor of older Haiku models. Our system prompt (~770 tokens) + 4 tool schemas (~750 tokens) total ~1,500 tokens, well below the floor. The `cache_control` markers are accepted by the API but no checkpoint is actually created. Budget impact: trivial (~$0.15 across the full demo envelope). Decision: **ship as-is, document it, move on.** Padding the prompt is real work for a $0.15 payoff. If a Phase 8 iteration wants to improve agent behavior anyway (more few-shot examples, richer tool-usage guidance), caching activation is a free side effect. Empirical proof: `in=5161 out=135 cached_read=0 cached_new=0` on the first live call.
- **Every smoke test passed first try after a test-harness fix.** The one failure was in the test: `FakeMessages.create` stored kwargs by reference, and the agent mutates its `messages` list across turns, so post-hoc inspection saw a mutated log. Fixed with `copy.deepcopy(kwargs)` at append time. Every other scenario (tool dispatch, unit conversion, is_error path, max_turns cap, cache_control placement) passed on the first run. The early Phase 7 investment in rigorous smoke tests paid back immediately — the first live call produced correct output.
- **The agent honestly surfaces uncertainty exactly as designed.** On 2523.png (the pathological tile — model predicts 90.6% background, 9.4% water, 0% forest; GT is 28% forest), the agent's report headlined the composition caveat as a "Critical limitation" in its own section, stated "the true annual flux could be substantially higher if background contains buildings or roads," and explicitly labeled the report as "a lower bound on the parcel's true emissions profile" that should not be acted on until ground-truthing is done. This is the portfolio story working. The 3546.png report by contrast is confident ("this parcel is a strong net carbon sink"), cites cleanly, and recommends the forest-expansion intervention with a calculated 62-year payback — the "model is confident, here's what a real analyst output looks like" case. Having both reports side-by-side **is** the narrative.
- **The agent computed break-even horizons in prose.** On 3546 it derived 228.53 ÷ 3.69 ≈ 62 years for agri→forest payback. On 2523 it derived 709.7 ÷ 7.1 ≈ 100 years for water→forest. Both correct. This is the kind of "LLM does what the tool can't" work we wanted — the tools return per-axis deltas, the LLM reasons over their ratio to produce a decision-relevant number.
- **The agent correctly rejected absurd `recommend_mitigation` top picks.** On 2523 with priority=annual, the top candidate is "convert 0.887 ha of water to forest." The agent recommended it but explicitly noted "reforestation of open water is ecologically unusual and may not be practical" and tied the recommendation to the wetlands caveat. Did not blindly parrot the #1 menu item. The "weigh tradeoffs, rule out absurd conversions" prompt instruction is landing.
- **The REPL `/quit` command didn't trigger on one test run** — the agent received "/quit" as a query and ran the full 3-tool pipeline in response. Code path looks correct (`if query in ("/quit", "/exit"): return` after `.strip()`); suspicion is an invisible trailing character on that terminal's Enter-handling. Benign — one wasted API call (~$0.005). Filed as Phase 8 polish; easy fix is `query.strip().lower().startswith("/quit")`.
- **Each REPL query is a fresh `agent.run()` session.** Observed in the live runs: the follow-up query "What's the single best intervention available here?" re-ran `get_land_breakdown` + `get_emissions_estimate` + `recommend_mitigation` all three tools, even though turn 1 of the main report had already fetched that data. The agent literally doesn't know the prior tool calls happened — the REPL feeds it only the current query, system prompt, and tools. This is functionally correct (tools are idempotent, results are identical) but wastes ~$0.005 per follow-up on redundant re-fetches. Worth fixing in Phase 8 if the frontend uses a chatty pattern; not worth fixing now.
- **Occasional imprecise citation bundling:** in shorter follow-up responses the model sometimes tags `[SRC-1,SRC-4]` on a claim where only one is strictly applicable (e.g., bundling SRC-1 with embodied forest stock when the embodied source is SRC-4 alone). Both sources do appear in the `sources_cited` dict, so not a fabrication — just imprecise attribution under brevity pressure. Could be tightened with a prompt-level "cite only the exact source backing the numeric claim" nudge. Low priority.
- **Every live run produced reports with section structure matching the prompt request** (headline → composition → current footprint → interventions → caveats) with no explicit format scaffolding in the context. System prompt guidance alone was sufficient. This was a risk going in — Haiku has been less reliable than Sonnet for structural instructions in past work.
- **No max_tokens hits.** All four live sessions finished well under the 2048 output-token ceiling (largest was 1098 on the 3546 default prompt). `max_tokens=2048` is comfortable; no need to tune up.
- **Inference via direct `InferenceEngine` import from `scripts/agent_repl.py` works identically to the FastAPI path.** 282–302 ms per classify across three live runs, same order as Phase 6's HTTP measurement (313 ms). Numerical output bit-for-bit identical. Confirms the Phase 6 architectural decision to keep `InferenceEngine` importable rather than hide it behind the HTTP boundary.
- **Total Phase 7 API spend: ~$0.05** across cheap probe + 5-turn 2523 run + 3-turn 2523 follow-up + 5-turn 3546 run + 3-turn 3546 follow-up. Budget posture is excellent: **~$9.95 remaining** of the $10 pre-build budget entering Phase 8.

**Open questions carried forward:**
- **For Phase 8: should the frontend call the agent via a new `POST /agent/report` endpoint in `backend/`?** Would require the backend to instantiate `ClaudeAgent` at startup (cheap — no model load) and offer an endpoint that takes an uploaded image, pre-classifies, builds AgentState, calls `agent.run(state, query)`, returns the structured AgentReport. Alternative: keep the agent CLI-only for the demo and show a pre-recorded report text in the frontend. Decision depends on whether we want to show live agent reasoning in the demo (expensive if the crowd hits it; impressive) or keep it behind glass (safer, loses some wow). Punt to Phase 8 opening.
- **Prompt padding to activate caching** becomes worthwhile if Phase 8 produces a pattern where the same system prompt hits hundreds of times (e.g., a demo playground with many users). At that scale, the ~$0.001/turn caching savings × turns would pay for the padding work. Not needed for a single-image-at-a-time demo.
- **Cross-query session memory in the REPL.** Low priority (trivial cost), but clean to implement — keep `messages` alive across queries inside `repl_loop`, pass it into a new `agent.continue_run(messages, state, query)` method that reuses the existing conversation. If Phase 8 adopts a chat-style frontend interface, this becomes more relevant.
- **Citation-precision prompt nudge.** A one-line addition to the system prompt ("cite only the exact SRC-n backing each numeric claim; do not bundle multiple SRCs per claim") would likely fix the imprecise-attribution observation. Not attempted in Phase 7 to avoid scope creep; test in Phase 8 if the demo output quality demands it.
- **Demo-image shortlist for Phase 8.** 3546.png is the canonical "model succeeds" tile. 2523.png is the canonical "model honestly surfaces uncertainty" tile. Both belong in the demo. Need ~3–5 more candidates covering: a clean urban (building-heavy) tile to exercise the building caveat, a mixed-use tile to show more balanced reports, maybe a water-dominant tile to exercise the wetlands assumption. Phase 5's decisions log flagged "pre-run inference on 10–20 candidate val tiles and rank" — that prep work is still pending.

**Phase 7 closed inside budget.** Estimate was 5h (riskiest remaining); actual was ~3.5h including two scripted offline smoke tests before any live API call. Pre-build budget posture entering Phase 8: ~$9.95 of $10 remaining (0.5% spent), Sunday buffer still untouched, all Weekend 2 phases so far on or under estimate.