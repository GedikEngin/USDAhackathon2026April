# Phase 2 — Current State

> Overwritten at end of every v2 phase. Tells the next chat what actually exists in the v2 portion of the repo right now. Reference file paths and describe behavior; do not paste code here.

**Project:** USDA Hackathon 2026 April — Corn Yield Forecasting
**Repo:** `~/dev/USDAhackathon2026April`
**Last updated:** 2026-04-25, end of Phase 2-D.1.a (CDL annual masks landed).

---

## v2 status: Phases A, B, C done. D.1.a done. D.1.b next.

**Pipeline state at a glance:**

- **Phase A** — six data pipelines feeding `scripts/training_master.parquet` (25,872 × 48). Closed with full data dictionary. ✅
- **Phase B** — analog-year retrieval baseline + cone-of-uncertainty MVP. Cone calibrated within target band on 2023+2024 backtest; analog-median point estimate beats the 5-year county-mean naive baseline. Gate passed. ✅
- **Phase C** — XGBoost per-forecast-date regressors trained, EOS gate passed at +46.7% RMSE improvement vs Phase B analog-median (threshold was +15%). Bundle in `models/forecast/regressor_*.json`. **All 5 MODIS NDVI columns stripped from the regressor's feature set** (Phase 2-C.1) after SHAP showed `ndvi_peak` dominating predictions across every forecast date — leaking end-of-season info into August forecasts. The Phase C regressor currently has zero remote-sensing features. ✅
- **Phase D.1** — Prithvi frozen-feature-extractor sub-phase. Replaces the stripped MODIS NDVI with HLS chips encoded by Prithvi-EO-2.0-300M-TL.
  - **D.1.a CDL annual masks**: 60 binary corn masks at `phase2/cdl/cdl_corn_mask_<state>_<year>.tif` for 5 states × 2013–2024. ✅
  - D.1.b HLS download + county chip extraction. **NEXT.**
  - D.1.c (inline with D.1.b) chip extraction with corn-richest 224×224 sub-window per county.
  - D.1.d Prithvi inference + `embeddings_v1.parquet`.
  - D.1.e Regressor retrain on 2013–2022 + 4-row ablation table → gate decision.

No backend endpoints, no agent tools, no frontend forecast view yet — those are Phases E, F, G.

## Project goal (locked)

Build an ML pipeline that predicts corn-for-grain yield (bu/acre) at the **county** level across **5 US states** at four fixed in-season forecast dates (Aug 1, Sep 1, Oct 1, end-of-season), aggregates to state, and produces a calibrated cone of uncertainty using **analog-year retrieval**. A Claude Haiku agent narrates the forecast.

- **Target states:** Colorado (CO), Iowa (IA), Missouri (MO), Nebraska (NE), Wisconsin (WI)
- **Time range:** **2005–2024** (Phase B/C train 2005–2022, val 2023, holdout 2024). **Phase D.1 train pool narrows to 2013–2022** because HLS only exists 2013+; this is intentional for the ablation gate test.
- **Spatial unit:** US county, joined via 5-digit GEOID
- **Target variable:** `yield_bu_acre_all` (combined-practice corn grain yield, bu/acre), exposed in the master table as `yield_target`

See `PHASE2_PROJECT_PLAN.md` for the full vision, `PHASE2_PHASE_PLAN.md` for the phase breakdown with go/no-go gates, and `PHASE2_DECISIONS_LOG.md` for the chronological decision record.

## What exists in the repo today

### v1 (untouched, fully shipped)

The full v1 land-use & GHG analysis system is in place. v2 reuses v1's FastAPI shell, frontend chrome, and Claude agent infrastructure but does not modify them.

### v2 — Phase A artifacts

- `scripts/nass_pull.py` + `scripts/nass_features.py` — NASS yield 2005–2024.
- `scripts/nass_corn_5states_2005_2024.csv` (raw, 6,837 × 16); `scripts/nass_corn_5states_features.csv` (engineered, 6,834 × 10).
- `scripts/ndvi_county_extraction.js` — version-controlled GEE script.
- `phase2/data/ndvi/corn_ndvi_5states_<year>.csv × 21` — MODIS NDVI 2004–2024.
- `scripts/gssurgo_county_features.py`; `scripts/gssurgo_county_features.csv` (443 × 13, static across years).
- `scripts/gridmet_pull.py`; `data/v2/weather/raw/gridmet_county_daily_<year>.parquet × 20`; `scripts/gridmet_county_daily_2005_2024.parquet`; `scripts/weather_features.py`; `scripts/weather_county_features.csv` (35,440 × 14).
- `phase2/data/drought/drought_USDM-...csv` (raw, state-level); `scripts/drought_features.py`; `scripts/drought_county_features.csv` (27,336 × 9, 0 nulls).
- `scripts/merge_all.py`; `scripts/training_master.parquet` (**25,872 × 48**, full coverage 98.1%).
- `docs/PHASE2_DATA_DICTIONARY.md` — column-by-column reference.

### v2 — Phase B artifacts (forecast/ package)

- `forecast/data.py` — `load_master`, train/val/holdout splits, min-history filter (10 train years per county to be eligible as analog).
- `forecast/features.py` — `EMBEDDING_COLS` (the standardized retrieval embedding); `VALID_FORECAST_DATES = ("08-01", "09-01", "10-01", "EOS")`.
- `forecast/analog.py` — K-NN over the embedding; returns analog (GEOID, year) pairs with distances and observed yields.
- `forecast/cone.py` — percentile band over analog yields.
- `forecast/aggregate.py` — county→state planted-acres-weighted rollup.
- `forecast/baseline.py` — naive 5-year county mean (gate reference).
- `forecast/detrend.py`, `forecast/recalibrate.py` — utility transforms used in the cone calibration loop.
- `scripts/backtest_baseline.py` — Phase B gate harness; results at `runs/backtest_baseline_*.csv`.

### v2 — Phase C artifacts

- `forecast/regressor.py` — `RegressorBundle` (4 per-date XGBoost boosters); `FEATURE_COLS` per forecast_date; `_NDVI = []` (intentionally empty, see decisions log entry 2-C.1).
- `forecast/explain.py` — SHAP-based `top_drivers(geoid, year, date, k=3)`.
- `scripts/train_regressor.py` — driver for the hyperparameter sweep.
- `scripts/backtest_phase_c.py` — Phase C gate harness; results at `runs/backtest_phase_c_*.csv`.
- `models/forecast/regressor_<date>.json + .meta.json` — the trained bundle. **Read-only from D.1's perspective.** D.1 retrain writes to `models/forecast_d1/`.

### v2 — Phase D.1.a artifacts (this session)

- `scripts/download_cdl.py` — pulls per-state CDL geotiffs from CropScape `GetCDLFile` API. Resumable, polite rate-limiting (mirrors `nass_pull.py`).
- `scripts/cdl_to_corn_mask.py` — converts categorical CDL to binary corn masks at EPSG:5070 / 30 m / uint8 / LZW. Reprojection-aware (handles 2024's native 10 m by nearest-neighbor downsample).
- `phase2/cdl/raw/cdl_<state>_<year>.tif × 60` — raw CDL (17.6 GB). Retained through D.1; deleted in cleanup pass at end of phase.
- `phase2/cdl/cdl_corn_mask_<state>_<year>.tif × 60` — binary corn masks (uint8, EPSG:5070, 30 m). Inputs to D.1.b/c chip masking.
- QC table inline in the run log shows corn fractions: IA 27.5–29.9%, NE 14.0–16.2%, WI 6.7–7.5%, MO 3.9–5.0%, CO 1.4–2.1%. Year-over-year drift ±2 pp confirms masks track real rotation.

### v2 — superseded / deleted in this session

- `phase2/data/hls/hls_vi_features*.csv` — old state-level HLS slice CSVs from the prior pull (5 files). Superseded by D.1.b's per-county chip extraction; deleted.
- `scripts/hls_county_features.csv` — output of the old `hls_features.py` against the state-level slices. Deleted.
- `scripts/hls_pull.py`, `scripts/hls_features.py` — kept on disk as **read-only references** for D.1.b; the GDAL/earthaccess auth pattern, Fmask bit decoding, and L30/S30 band-name asymmetry handling are reused. The state-level VI computation logic is dropped. To be deleted in cleanup pass at end of D.1.

### v2 — environment

- New conda env: `forecast-d1` (Python 3.11). Phase A/B/C envs untouched.
- `torch 2.10.0+cu130`, `torchvision 0.25.0+cu130`, `torchaudio 2.10.0+cu130` — installed from PyTorch's CUDA 13.0 wheel index. Stable, not nightly. Blackwell sm_120 kernels confirmed working.
- `terratorch 1.2.6`, `rasterio 1.4.4`, `earthaccess 0.17.0`, `xgboost 3.2.0`. Standard geospatial deps (geopandas, shapely, pyproj, fiona) on top.
- Filesystem: WSL2 native `/dev/sdd`, 881 GB free. All D.1 outputs land under `~/dev/USDAhackathon2026April/data/v2/...` or `~/dev/USDAhackathon2026April/phase2/...`.
- GPU: RTX 5070 Ti Laptop, 12 GB VRAM, compute capability sm_120. Driver 591.44 / CUDA 13.1. `nvidia-smi` clean. `torch.cuda.is_available() == True`, matmul on GPU verified.

## Phase D.1 plan (what's coming)

See `PHASE2_DECISIONS_LOG.md` entries 2-D.1.kickoff and 2-D.1.a for the full decision rationale. Quick reference:

- **Prithvi variant:** `terratorch_prithvi_eo_v2_300_tl` (300M, temporal+location).
- **Chip granularity:** county-level. Corn-richest 224×224 sub-window per (county, granule), masked with the year-matched CDL.
- **Sequence shape:** T=3 chips (vegetative + silking + grain-fill phases). T=2 padded with silking-dup at 08-01 (grain-fill empty).
- **Pooling:** mean across spatial patches and across T → 1 vector per `(GEOID, year, forecast_date)`.
- **Train pool for D.1 retrain:** 2013–2022. Phase C-as-is bundle (2005–2022, no Prithvi) preserved as ablation row.
- **HLS phase windows:** calendar-aligned. `aug1` = Jul 17 – Aug 15; `sep1` = Aug 17 – Sep 15; `oct1` = Sep 17 – Oct 15; `final` = Oct 17 – Nov 15.
- **Pull architecture:** pull-once per `(state, year)` over full growing season `<year>-05-01 to <year>-11-15`; label chips at index time; pick chips at embed time.
- **Filters:** granule-level cloud filter 70% (CMR `eo:cloud_cover` ≥ 70% drops the granule pre-download); chip-level corn-fraction filter 5%; granule cap 100 per (state, year, phase).
- **Output:** `data/v2/hls/chips/<GEOID>/<year>/<phase>_<scene_date>.tif`; index in `data/v2/hls/chip_index.parquet`. Embeddings in `data/v2/prithvi/embeddings_v1.parquet`.
- **Gate:** Row B (engineered + Prithvi, 2013–2022) ≥ 5% RMSE improvement vs Row A (engineered-only, 2013–2022) on 2023 val EOS.

## Master table — at-a-glance (unchanged from end of Phase A)

```
scripts/training_master.parquet
  shape:        25,872 × 48
  size:         2.46 MB
  grain:        (GEOID, year, forecast_date)
  years:        2005–2024 (20 years)
  states:       CO, IA, MO, NE, WI (5)
  GEOIDs:       388 distinct (subset of 443 TIGER 2018 counties)
  forecast_d:   08-01, 09-01, 10-01, EOS
  target:       yield_target (bu/acre, combined-practice)
  full coverage: 25,380 / 25,872 (98.1%) excl. HLS
                 10,575 / 25,872 (40.9%) incl. HLS (HLS only exists 2013+)
```

48 columns: 5 keys + 1 target + 5 NASS-aux + 5 NDVI + 11 gSSURGO + 11 weather + 6 drought + 4 HLS-state-level (the 4 stale HLS columns will be replaced in D.1's master-table rebuild via left-join on `embeddings_v1.parquet`).

## Hackathon readiness checklist

- [x] Phase A definition-of-done met
- [x] Phase B analog cone shipped + Phase B gate passed
- [x] Phase C XGBoost regressor shipped + Phase C gate passed (+46.7% lift, threshold +15%)
- [ ] **Phase D.1 — IN PROGRESS**
  - [x] D.1.a CDL annual masks (60 files, 5 states × 12 years)
  - [ ] D.1.b HLS download orchestration + chip extraction
  - [ ] D.1.c chip extraction (inline with D.1.b)
  - [ ] D.1.d Prithvi inference + embeddings parquet
  - [ ] D.1.e Regressor retrain + 4-row ablation table → gate decision
- [ ] Phase D.2 (conditional on D.1 gate) Prithvi fine-tune
- [ ] Phase E `/forecast/{state}` endpoint + frontend forecast view
- [ ] Phase F forecast narration agent with 4 tools
- [ ] Phase G holdout evaluation, ablation table, presentation deck

## Immediate next steps (Phase D.1.b)

1. **Earthdata auth smoke test.** `python -c "import earthaccess; auth = earthaccess.login(strategy='netrc'); print(auth.authenticated)"`. User confirmed `~/.netrc` is set up. Verify it works in the new env.
2. **Read `scripts/hls_pull.py` and `scripts/hls_features.py` as reference.** Pull forward the GDAL config, Fmask helpers, band-map asymmetry handling, fsspec-opener pattern, year-by-year checkpointing.
3. **Write `scripts/download_hls.py`** — pull-once-per-(state, year), granule-level cloud filter 70%, chip extraction inline, granule deletion after chips written.
4. **Write `scripts/extract_chips.py`** (or fold into `download_hls.py`) — for each (granule, county) intersection: window-read 6 bands clipped to county polygon, slide 224×224 footprint, pick corn-richest position, write chip if ≥ 5% corn pixels, append row to chip index.
5. **Write `forecast/hls_common.py`** — shared band map (L30 vs S30), Fmask bit decoder, calendar-phase labeler, scaling factor (0.0001 SR scale).
6. **Write the chip index schema spec** as a doc artifact before code, so the schema is reviewable separately.
7. **Run end-to-end on one (state, year) cell** as a smoke test before the long pull (e.g. IA 2018, expected ~80–120 granules → process in 30–60 min).
8. **Long pull** across all 5 states × 12 years over multiple sessions.

## Budget note

v1 used ~$0.25 of the $10 API cap. v2 token usage will be similar in shape (Haiku narration, ~$0.02 per forecast). Compute cost for HLS download + Prithvi inference is local and outside the API budget.