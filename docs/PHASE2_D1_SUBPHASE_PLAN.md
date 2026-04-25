# Phase 2-D.1 — Prithvi Frozen-Feature-Extractor Sub-phase Plan

> Working doc for Phase D.1. Once approved, conventions land in `PHASE2_DECISIONS_LOG.md` (decisions row) and `PHASE2_PHASE_PLAN.md` (sub-phases). When this file and `PHASE2_DECISIONS_LOG.md` disagree, the decisions log wins.

**Locked from the kickoff conversation:**

- **Variant:** `terratorch_prithvi_eo_v2_300_tl` — Prithvi-EO-2.0, 300M params, temporal+location embeddings.
- **Granularity:** county-level — one embedding per `(GEOID, year, forecast_date)`.
- **Sequence:** T=3 chips per query, drawn from vegetative (DOY 152–195), silking (196–227), grain-fill (228–273) windows, each clipped to ≤ forecast_date. At 08-01 grain-fill hasn't started → T=2 padded with the silking chip duplicated to keep tensor shape constant.
- **Pooling:** mean across spatial patches and across T → one fixed-length vector per query.
- **Train pool:** 2013–2022 (drop pre-HLS years from D.1; preserves Phase C-as-is bundle as a separate ablation row trained on 2005–2022).
- **Val:** 2023. **Holdout:** 2024.
- **CDL:** annual masks 2013–2024 from CropScape, plus the existing 2025 mask.
- **Compute:** WSL2 single-machine, RTX 5070 Ti Laptop 12 GB, conda env `forecast-d1`, Python 3.11, torch 2.10 / cu130, terratorch 1.2.6.
- **Storage:** WSL2 native ext4 at `~/dev/USDAhackathon2026April/data/v2/`. 881 GB free. Granule cache rolling-deleted; chips and embeddings persisted.
- **Phase C preservation:** `models/forecast/regressor_*.json` is **read-only from D.1's perspective**. D.1 retrain writes to `models/forecast_d1/`. Both bundles required for the G.2 ablation table.

**Deliverables (final):**

1. `data/v2/cdl/cdl_corn_mask_<state>_<year>.tif` × 65 (5 states × 13 years) — binary corn masks.
2. `data/v2/hls/chips/<GEOID>/<year>/<phase>_<scene_date>.tif` — county-clipped, corn-masked HLS chips.
3. `data/v2/hls/chip_index.parquet` — resumable inventory of which (GEOID, year, scene_date) chips exist + QC metadata.
4. `data/v2/prithvi/embeddings_v1.parquet` — the canonical Prithvi embeddings table.
5. `models/forecast_d1/regressor_*.json` — retrained per-date XGBoost bundle.
6. `runs/d1_ablation_<timestamp>.csv` — the ablation table for the gate decision.

**Gate (locked from `PHASE2_PHASE_PLAN.md` D.1):** Prithvi-augmented model ≥ 5% RMSE better than engineered-only on 2023 val. Engineered-only baseline retrained on 2013–2022 to match D.1's row pool — apples-to-apples.

---

## D.1.a — CDL prep (annual corn masks, 5 states × 12 years)

**Goal:** 65 annual binary corn-mask geotiffs on disk, schema-uniform with the existing 2025 masks.

**Why annual, not 2025-only:** corn-soybean rotation in the Midwest means a 2025 corn mask applied to 2018 HLS pulls 30–50% of its "corn" pixels from soybean fields that year. That's exactly the feature pollution Phase 2-C.1 stripped MODIS NDVI for. Annual masks are an extra ~30 minutes of automated download + ~5 GB; the accuracy gain is worth it.

**Scope:** 5 states (CO, IA, MO, NE, WI) × 12 years (2013–2024). Plus the existing 2025 mask. = 61 files total to materialize.

**Steps:**

1. **Rename existing 2025 masks** to year-suffixed naming so the script's loop is uniform:
   ```
   phase2/cdl/cdl_corn_mask_iowa.tif → data/v2/cdl/cdl_corn_mask_iowa_2025.tif
   ```
   Five files. Move + rename in one pass.
2. **Write `scripts/download_cdl.py`:**
   - Input: state list, year list, output dir.
   - Hits CropScape `GetCDLFile` web service: `https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile?year=YYYY&fips=SS`.
   - Response is XML containing a temporary geotiff URL on the GMU server. Parse, GET, write to disk.
   - For 2024 specifically: request the 30m-resampled version (CDL went 30m → 10m starting 2024 and we need pixel alignment with HLS, gSSURGO, and historical CDL).
   - **Resumable**: skip files already on disk. Idempotent re-run is safe.
   - **Polite rate limiting**: 2 s between requests, 4 retries with exponential backoff (mirror `nass_pull.py` convention).
3. **Write `scripts/cdl_to_corn_mask.py`:**
   - Input: raw CDL geotiffs (categorical raster, 256 classes).
   - Output: binary corn-mask geotiffs (1 = corn, 0 = anything else). Reproject to EPSG:5070 if not already (gSSURGO, HLS, and CDL all use Albers Equal Area; consistency at 30 m enables zero-resample masking later).
   - Corn class = pixel value 1 in CDL.
   - Naming: `cdl_corn_mask_<state>_<year>.tif`.

**Sub-phase deliverable:** 65 binary geotiffs in `data/v2/cdl/`, all 30m EPSG:5070, all binary {0,1} uint8.

**QC:** scripted check that prints (a) total pixels, (b) corn pixels, (c) corn fraction per state-year. Drift in corn fraction year-over-year is normal; a state-year reading 0% corn is a bug.

**Estimated time:** 30–60 min download + 10 min mask conversion. One sitting.

---

## D.1.b — HLS download orchestration

**Goal:** for every (GEOID, year, scene_date) we eventually need a chip from, the corresponding HLS granule has been pulled, processed, and the granule deleted.

**Scope:** 5 states × 10 years (2013–2022 train + 2023 val + 2024 holdout = 12 years actually, see below) × growing-season scenes.

**Strategy: pull-process-delete loop, never accumulate granules.** Hardware limit isn't compute, it's transient disk pressure during download. We process each granule immediately into county chips, then delete the source granule before moving on.

**Steps:**

1. **Earthdata auth check.** `python -c "import earthaccess; earthaccess.login(strategy='netrc')"` should succeed silently. If `~/.netrc` doesn't have the credentials, prompt-then-cache.
2. **Write `scripts/download_hls.py`:**
   - Iterate (state, year) outer loop.
   - For each, search HLS L30 (Landsat) and S30 (Sentinel-2) collections via `earthaccess.search_data` for the growing season (May 1 → Oct 31), bounded by the state polygon.
   - For 2013–2014: L30 only (S30 doesn't exist before mid-2015).
   - For each granule returned: download to `data/v2/hls/raw/<state>/<year>/<granule_id>/`, then immediately invoke chip extraction (D.1.c) to write the relevant county chips, then `rm -rf` the raw granule directory.
   - **Resumable**: granules already represented in `chip_index.parquet` are skipped without re-downloading.
   - **Cloud filter**: skip granules where Fmask reports >70% cloud cover at the granule level (rough first-pass; per-county cloud check happens during chip extraction). 70% is the default LP DAAC cloud threshold for many vegetation studies; tunable.
3. **Track progress** in `data/v2/hls/granule_log.parquet` (granule_id, status, downloaded_at, chips_extracted, granule_deleted).

**Risk:** earthaccess + LP DAAC throughput is the binding constraint. Bandwidth-limited at residential speeds. Estimate: ~50–100 GB total raw granule volume across the full pull (peak transient ~5–10 GB at any moment). At 25 MB/s residential, that's a long evening per state-year. Plan for this to run in the background overnight, multiple sessions if needed. The script must be safely interruptible (Ctrl+C) and resumable (re-run picks up where it left off).

**Sub-phase deliverable:** every chip we'll need is in `data/v2/hls/chips/`, no granules left over in `data/v2/hls/raw/`, `granule_log.parquet` shows all (state, year) batches at status="complete".

---

## D.1.c — Chip extraction and corn-masking

**Goal:** convert each downloaded HLS granule into per-county chips clipped to the corn mask.

This sub-phase runs **inline inside D.1.b**, not as a separate script invocation — but it deserves its own design section because it's where the most subtle correctness choices live.

**Per granule:**

1. Open the 6 HLS bands we feed Prithvi: Blue, Green, Red, Narrow NIR, SWIR1, SWIR2 (the order Prithvi-EO-2.0 was pretrained on, per the model card).
2. Reproject to EPSG:5070 if needed (HLS L30/S30 are usually UTM tiles; CDL and county polygons are 5070).
3. For each county polygon that intersects this granule's footprint:
   - Window-read the 6 bands clipped to county bbox.
   - Apply that county-year's CDL corn mask: pixels where `cdl == 0` get NaN (or 0 with a separate mask channel — Prithvi's data loader handles either).
   - Apply per-pixel cloud mask from HLS Fmask layer (drop cloudy pixels).
   - Compute QC stats: corn-pixel fraction, cloud fraction, mean reflectance per band (sanity check).
   - **If corn-pixel fraction < 5%**, skip this county for this granule (insufficient signal — Prithvi will encode mostly noise).
   - Else, save chip as 6-band geotiff to `data/v2/hls/chips/<GEOID>/<year>/<phase>_<scene_date>.tif`.
4. Append a row to `chip_index.parquet`: `(GEOID, year, scene_date, scene_doy, phase_window, chip_path, cloud_pct, corn_pixel_frac, sensor)`.

**Phase window assignment:**
- DOY 152–195 → "veg"
- DOY 196–227 → "silk"
- DOY 228–273 → "grain"
- Outside windows → "other" (kept in index but not used for embedding sequences; might be useful for ablation)

**Chip dimensions:** Prithvi-EO-2.0 was pretrained on 224×224 chips at 30m. County polygons vary widely in size — Iowa counties are roughly 36 km × 36 km (1200×1200 pixels at 30m), Colorado mountain counties can be much larger. **Decision:** for each (county, scene), extract a single 224×224 chip centered on the county centroid (or nudged to the corn-richest 224×224 sub-window if the centroid happens to be non-corn rural/urban). Document this choice in the data dictionary — it's a real simplification (large counties' embedding doesn't reflect their full extent), but it matches Prithvi's native input contract and avoids resampling artifacts.

**Sub-phase deliverable:** every (GEOID, year, scene_date) we'll embed has a 224×224×6 geotiff on disk; `chip_index.parquet` is the master inventory.

---

## D.1.d — Prithvi inference and embeddings parquet

**Goal:** for every (GEOID, year, forecast_date) row in the master training table, a Prithvi embedding vector is in `embeddings_v1.parquet`.

**Steps:**

1. **Write `forecast/prithvi.py`:**
   - `load_backbone()` — uses `BACKBONE_REGISTRY.build("terratorch_prithvi_eo_v2_300_tl", pretrained=True)`. Cached in `~/.cache/terratorch/`.
   - `pick_chips_for_query(geoid, year, forecast_date, chip_index_df) -> list[Path]` — returns 3 chip paths, one per phase window, each the most recent cloud-free chip in that window with `scene_date < forecast_date`. At 08-01 grain-fill is empty → returns 2 chips, with the silking chip duplicated to make T=3.
   - `extract_embedding(chip_paths, lat, lon, doy_of_most_recent) -> np.ndarray` — loads chips, stacks to (1, 6, T, H, W), normalizes per Prithvi's published mean/std, runs forward pass, mean-pools across spatial and temporal tokens, returns 1D embedding. T+L variant requires `lat`, `lon`, `temporal_coords` as auxiliary inputs — these come from county centroid + chip dates.
   - `extract_embedding_batch(queries, chip_index_df, batch_size=4)` — batched version for the full pass.
2. **Write `scripts/run_prithvi.py`:**
   - For every `(GEOID, year, forecast_date)` row in `training_master.parquet` where `year ≥ 2013`:
     - Pick chips via D.1.c's index.
     - Run inference.
     - Append row to `embeddings_v1.parquet` with the schema below.
   - **Resumable**: skip queries already in the parquet (keyed on the three-tuple).
   - **Batch size**: start at 4 (T=3 sequences × 6 bands × 224² × 4 batch = manageable on 12 GB at fp16). If OOM, drop to 2. If easy, raise to 8.
   - **fp16**: yes, with autocast. Frozen feature extractor doesn't need fp32 precision.

**Embeddings parquet schema** (locked):
```
data/v2/prithvi/embeddings_v1.parquet
  GEOID                str    5-char zero-padded
  year                 int    2013-2024
  forecast_date        str    {"08-01","09-01","10-01","EOS"}
  state_alpha          str    2-char USPS (denormalized)
  chip_age_days        int    days between most-recent-chip date and forecast_date
  chip_count           int    1, 2, or 3 (how many distinct phase chips were available)
  cloud_pct_max        float  worst cloud fraction across the chips used
  corn_pixel_frac_min  float  smallest corn-mask fraction across the chips used
  prithvi_emb_000      float
  ...                  
  prithvi_emb_<D-1>    float  D = embedding dim (TBD on first model load; expect ~1024)
  model_version        str    "prithvi-eo-v2-300-tl@<weights_sha>"
  extracted_at         str    ISO timestamp
```

The 4 QC columns (`chip_age_days`, `chip_count`, `cloud_pct_max`, `corn_pixel_frac_min`) become explicit features in the regressor — model learns to discount predictions when chips are stale, sparse, cloudy, or corn-poor. They're not just diagnostics.

**Sub-phase deliverable:** `embeddings_v1.parquet` covers every `(GEOID, year, forecast_date)` triple with `year ≥ 2013` that's in the master table.

**Estimated time:** ~12 years × 4 forecast dates × ~388 GEOIDs × T=3 = ~56K forward passes. At batch 4 / fp16 on a 5070 Ti Laptop, expect ~2–4 hours wall-clock. Single sitting; can run in parallel with the late stages of D.1.b for batches that are ready.

---

## D.1.e — Regressor retrain and ablation

**Goal:** retrained per-date XGBoost bundle on (engineered features + Prithvi embeddings), plus an ablation table comparing it against the engineered-only baseline.

**Steps:**

1. **Write `scripts/train_regressor_d1.py`** — clone of `scripts/train_regressor.py`, with:
   - Master table loaded with year filter `year ≥ 2013`.
   - Left-join `embeddings_v1.parquet` on `(GEOID, year, forecast_date)`.
   - Feature list extended: existing engineered features + 4 QC columns + D Prithvi embedding columns.
   - Same hyperparameter sweep as Phase C (`max_depth ∈ {4,6,8}`, `learning_rate ∈ {0.05, 0.1}`, `min_child_weight ∈ {1, 5}`).
   - Save to `models/forecast_d1/regressor_<date>.json`.
2. **Write `scripts/ablation_d1.py`** — produces the gate-decision table:
   - **Row A (reference, 2013–2022 pool):** retrain Phase C on 2013–2022 only, no Prithvi. This is the apples-to-apples baseline.
   - **Row B (D.1, 2013–2022 pool):** engineered + Prithvi.
   - **Row C (Phase C-as-is, 2005–2022 pool):** existing `models/forecast/` bundle, untouched. Reference for "more data, no Prithvi."
   - **Row D (reference only, leaky):** engineered + leaky MODIS NDVI on 2013–2022 pool. Documents what the gate would look like with the previously-stripped feature included. Not a candidate for production.
   - Columns: RMSE Aug 1 / Sep 1 / Oct 1 / EOS, on val 2023, state-aggregated.
3. **Decide gate:** Row B vs Row A end-of-season RMSE. Need ≥ 5% improvement to pass.

**Sub-phase deliverable:** `runs/d1_ablation_<timestamp>.csv` + a written paragraph in the decisions log interpreting the result.

---

## Risks and mitigations

- **HLS Earthdata auth fails or rate-limits.** Mitigation: earthaccess uses ~/.netrc; pre-test before the long pull. Polite rate limiting in the loop.
- **Sentinel-2 (S30) coverage gaps in 2015 transitional year.** Mitigation: fall back to L30 if S30 returns empty for a (state, week) cell.
- **Per-county chip-window choice (centroid-centered 224×224)** is a real simplification — large counties' embedding only reflects a single sub-tile. Documented; revisit if D.1 misses the gate.
- **VRAM ceiling on 12 GB.** Mitigation: fp16 default; drop batch size if OOM; 100M-TL is a fallback if 300M won't fit (we confirmed both register cleanly).
- **CropScape API flakiness.** Mitigation: 4 retries with exponential backoff; resumable script.
- **2024 CDL is 10m natively.** Mitigation: request the 30m-resampled version explicitly via the API.
- **Pre-2013 training rows are dropped from D.1.** Trade-off: we lose ~44% of train data in exchange for an apples-to-apples ablation. Mitigated by keeping the Phase C-as-is bundle as a separate ablation row.
- **Embedding dim D is currently TBD.** Will be set on first model load and locked in the parquet schema before the long inference run starts.
- **Prithvi version weights drift.** Mitigation: `model_version` baked into parquet rows; `embeddings_v2.parquet` if we ever re-extract, never overwrite.

---

## Sub-phase ordering (sequential — each sub-phase must complete before the next starts)

```
D.1.a  CDL annual download + binary mask conversion        [~1 hr, automated]
D.1.b  HLS download orchestration                          [overnight, multi-session]
D.1.c  Chip extraction + corn-masking                      [inline with D.1.b]
D.1.d  Prithvi inference + embeddings parquet               [~3 hr, automated]
D.1.e  Regressor retrain + ablation table                  [~30 min, automated]
GATE   ≥5% RMSE improvement vs Row A baseline               [decision]
```

If any earlier sub-phase produces something unexpected (e.g. CDL API returns 0 corn for a state-year, HLS coverage has bigger gaps than estimated), we pause the pipeline and reconvene before barreling forward.

---

## Definition of done for Phase D.1

- [ ] All 65 CDL annual masks materialized in `data/v2/cdl/`.
- [ ] All HLS chips for 2013–2024 in `data/v2/hls/chips/`.
- [ ] `chip_index.parquet` has full coverage per the QC spec.
- [ ] `embeddings_v1.parquet` covers every `(GEOID, year, forecast_date)` row with `year ≥ 2013`.
- [ ] `models/forecast_d1/regressor_*.json` × 4 trained.
- [ ] Ablation CSV produced.
- [ ] Decisions log entry written interpreting the gate result and naming the production model.
- [ ] `PHASE2_DATA_INVENTORY.md` updated to reflect HLS now county-level, CDL annual.
- [ ] `PHASE2_DATA_DICTIONARY.md` updated with the 4 QC columns and the Prithvi embedding columns.
