# Current State

> Overwritten at end of every phase. Tells the next chat what actually exists in the repo right now. Do NOT paste code here — reference file paths and describe behavior.

---

## Last updated: 2026-04-23, end of Phase 2

## Repo layout

```
USDAhackathon2026April/
  data/
    loveda/
      Train/
        Urban/{images_png, masks_png}/   (1156 each)
        Rural/{images_png, masks_png}/   (1366 each)
      Val/
        Urban/{images_png, masks_png}/   (677 each)
        Rural/{images_png, masks_png}/   (992 each)
      samples/                            (8 overlay PNGs for visual sanity)
      stats.json                          (machine-readable class stats)
      Train.zip, Val.zip                  (can delete to reclaim ~6 GB)
  model/
  scripts/
    download_loveda.sh
    dataset_stats.py
    dataset.py                            (Phase 2)
    test_loader.py                        (Phase 2)
  frontend/
  dataset_stats.md
  .gitignore
  (plan docs)
```

## What works

- Conda env `landuse` (Python 3.11) active on Ubuntu box, CUDA confirmed on RTX 4070 Ti SUPER
- LoveDA Train + Val on disk, MD5-verified, counts match LoveDA paper
- `scripts/dataset_stats.py` produces stats.json + dataset_stats.md + 8 sample overlays
- `scripts/dataset.py`: `LoveDADataset` class + `build_transforms(split)` helper
  - Globs Urban + Rural into a single sample list, asserts every image has a matching mask at construction
  - Returns `(image: float32 (3,1024,1024) ImageNet-normalized, mask: long (1024,1024) values 0..7)`
  - Train aug pipeline: Resize(1024, bilinear img / nearest mask) → HFlip → VFlip → RandomRotate90 → ColorJitter(0.2/0.2/0.2/0.05) → Normalize(ImageNet) → ToTensorV2
  - Val aug pipeline: Resize(1024) → Normalize → ToTensorV2
- `scripts/test_loader.py`: Phase 2 smoke test. Pulls 20 batches from each split, asserts shapes/dtypes/class-value validity/normalization applied. PASSES.
- Measured throughput: ~0.13s/batch at batch_size=2, num_workers=2 on RTX 4070 Ti SUPER. Data loading will not be the bottleneck in Phase 3.

## What's half-done

- tmux detach shortcut still not verified (not blocking until Phase 3)
- Train.zip + Val.zip still on disk; can be deleted to reclaim ~6 GB

## What's next

Phase 3: SegFormer-B1 fine-tune, first pass.
- Load `nvidia/segformer-b1-finetuned-ade-512-512` or equivalent, swap the classification head to 7 classes (ignoring class 0 in the loss)
- CE loss with MFB class weights (already locked in Phase 1 decisions), `ignore_index=0`
- AdamW + cosine schedule, batch 2, grad-accum 4 (effective batch 8), ~15 epochs
- `--crop-size` CLI flag for 768 fallback if first epoch >45 min
- Per-class IoU logged every epoch, not just mIoU
- Deliverable: checkpoint + val log with Saturday gate numbers clearly visible

## Known issues / gotchas

- Activate env each SSH/AnyDesk session with `conda activate landuse`
- nvcc reports 12.0, driver reports CUDA 13.0 — normal, no action needed
- Windows backup machine not yet smoke-tested
- Barren vs forest: noted visual confusion even to human eye on 0.3m imagery. Model will likely confuse them. Accepted for v1.
- Val distribution differs from train: ~2× more water, ~1.4× more agri, ~half the forest. Per-epoch val IoU will be noisy; agri is a safer Saturday gate than forest.
- **Epoch time to watch in Phase 3:** at batch 2 w/ grad-accum 4, training will be 1261 optimizer steps per epoch on 2522 train samples. First epoch >45 min triggers fallback to 768.

## Key numbers (fill in as they're measured)

- Val mIoU: —
- Per-class IoU (bg, building, road, water, barren, forest, agri): —
- Training epoch time: —
- Data loader throughput: 0.13s/batch, B=2, num_workers=2 (Phase 2 measured)
- API cost spent so far: $0

### Locked class weights for Phase 3 (MFB, mean-normalized)

Indices 1–7 (class 0 ignored):

| Class | Weight |
|---|---|
| background | 0.255 |
| building | 0.824 |
| road | 1.730 |
| water | 1.426 |
| barren | 1.748 |
| forest | 0.567 |
| agriculture | 0.451 |

### Class distribution (Train, combined)

background 35.8%, agri 20.2%, forest 16.1%, building 11.1%, water 6.4%, road 5.3%, barren 5.2%. Overall imbalance ratio: 6.9× (mild).

### Normalization constants (locked)

ImageNet: mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225). Used because SegFormer backbone is ImageNet-pretrained. Do NOT re-derive from LoveDA.
