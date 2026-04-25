# Data Inventory — v2 Corn Yield Forecasting

> Living tracker. Update whenever data status changes — pulled, parsed, cleaned, joined, validated, or discovered missing. This is the single source of truth for "do we have what we need to start the next phase."

**Last updated:** 2026-04-25, end of Phase 2-D.1.a (CDL annual masks landed).

## Status legend

- 🟢 **Have & ready** — on disk, parsed, schema-validated, joinable
- 🟡 **Have, partial / needs cleaning** — on disk but not yet in canonical schema, or has known gaps
- 🔵 **In flight** — actively pulling/processing
- 🔴 **Need** — not yet acquired
- ⚫ **Deprioritized** — known, decided not worth the integration cost

**Coverage target:** 5 states (IA, CO, WI, MO, NE), county level, **2005–2024** per the brief. Phase D.1 narrows the model train pool to **2013–2024** because HLS only exists 2013+.

---

## Master training table — Phase A.6 ✅

**THIS IS THE CANONICAL DOWNSTREAM INPUT** for Phase B (retrieval), Phase C (XGBoost), Phase D (Prithvi), and Phase F (agent tools).

| Property | Value |
|---|---|
| File | `scripts/training_master.parquet` |
| Format | Parquet, snappy compression, 2.46 MB |
| Shape | 25,872 rows × 48 columns |
| Grain | one row per `(GEOID, year, forecast_date)` |
| Years | 2005–2024 |
| GEOIDs | 388 distinct (subset of 443 TIGER counties) |
| Forecast dates | `08-01`, `09-01`, `10-01`, `EOS` |
| Target | `yield_target` (combined-practice corn-grain yield, bu/acre) |
| Full feature coverage | 25,380 / 25,872 (98.1%) excl. HLS; 10,575 / 25,872 (40.9%) incl. HLS |

**Reference doc:** `docs/PHASE2_DATA_DICTIONARY.md`. **Source script:** `scripts/merge_all.py`.

The 4 HLS columns currently in the master table (`hls_ndvi_mean`, `hls_ndvi_std`, `hls_evi_mean`, `hls_evi_std`) are state-level broadcast and 59% null. **They will be replaced in D.1's master-table rebuild** via left-join on the new `embeddings_v1.parquet` (D Prithvi embedding columns + 4 QC columns). The `merge_all.py` HLS allowlist code stays in place but starts pulling from the new source.

---

## Required by the brief

| Dataset | Source | Use | Coverage needed | Status | Notes |
|---|---|---|---|---|---|
| Prithvi foundation model | NASA / IBM via HuggingFace | Geospatial feature extraction | Model weights | 🔴 Need (D.1.d) | terratorch backbone string `terratorch_prithvi_eo_v2_300_tl`. ~1.2 GB weights, downloaded to `~/.cache/terratorch/` on first model load. |
| HLS (Harmonized Landsat-Sentinel) | NASA LP DAAC | Multispectral imagery, Prithvi input | 5 states, 2013–2024, growing season May–Nov | 🔴 **Need (D.1.b — NEXT)** | Pull-once per (state, year) via earthaccess + CMR. Granule-level cloud filter 70%. Top 100 cleanest granules per phase window. Process-and-delete loop. Outputs: `data/v2/hls/chips/<GEOID>/<year>/<phase>_<scene_date>.tif` + `data/v2/hls/chip_index.parquet`. |
| NAIP aerial imagery | USDA FSA | High-res visual context | 5 states, recent years | ⚫ Excluded | Decided 2026-04-25 (see decisions log). |
| USDA NASS yield (CORN, GRAIN — bu/acre) | NASS QuickStats API | Ground truth | 5 states, county level, 2005–2024 | 🟢 Have | `scripts/nass_corn_5states_2005_2024.csv` (6,837 × 16); `nass_corn_5states_features.csv` (6,834 × 10). |
| Cropland Data Layer (CDL) | NASS CropScape | Annual corn-pixel mask for HLS | 5 states, 2013–2024 | 🟢 **Have (D.1.a)** | `phase2/cdl/cdl_corn_mask_<state>_<year>.tif × 60` — uint8 binary masks at EPSG:5070 / 30 m / LZW. Built by `scripts/cdl_to_corn_mask.py` from `phase2/cdl/raw/cdl_<state>_<year>.tif × 60` (raw categorical, 17.6 GB). Year-over-year drift confirms masks track real rotation (IA 27.5–29.9%, NE 14.0–16.2%, WI 6.7–7.5%, MO 3.9–5.0%, CO 1.4–2.1%). |
| Weather / climate (daily, gridded) | gridMET | GDD, EDD, VPD, precip accumulation, srad | 5 states, daily, 2005–2024 | 🟢 Have | `scripts/gridmet_county_daily_2005_2024.parquet`; derived features `scripts/weather_county_features.csv` (35,440 × 14). |

## Strongly recommended additions

| Dataset | Source | Use | Coverage needed | Status | Notes |
|---|---|---|---|---|---|
| US Drought Monitor | NDMC / USDM | Drought severity feature | 5 states, weekly, 2005–2024 | 🟢 Have | `scripts/drought_county_features.csv` (27,336 × 9, 0 nulls). State-level broadcast. |

## Already on hand

| Dataset | Source | Use in v2 | Status | Notes |
|---|---|---|---|---|
| NASS combined-practice corn yield 2005–2024 | NASS QuickStats API | Ground truth | 🟢 | Full brief range. 100% on combined-practice columns. |
| NASS engineered features | `scripts/nass_features.py` | Engineered yield features | 🟢 | Merged into master table. |
| MODIS NDVI mapping, 5 states | GEE / MODIS MOD13Q1 | **Phase B retrieval embedding only** — stripped from Phase C regressor at 2-C.1 (SHAP showed leakage) | 🟢 | 21 per-year CSVs in `phase2/data/ndvi/`. Pre-scaled (× 0.0001 server-side). Whole-season summary (same value at all 4 forecast dates within a year). |
| GEE NDVI script (version-controlled) | Local | Reproducibility | 🟢 | `scripts/ndvi_county_extraction.js`. |
| gSSURGO Valu1 county features, 5 states | `scripts/gssurgo_county_features.py` | Soil features | 🟢 | `scripts/gssurgo_county_features.csv` (443 × 13). |
| gridMET daily weather, 5 states, 2005–2024 | `scripts/gridmet_pull.py` | Climate source | 🟢 | Raw daily parquets per year + combined parquet. |
| gridMET-derived per-cutoff features | `scripts/weather_features.py` | Climate features | 🟢 | Merged into master table. |
| US Drought Monitor weekly raw | NDMC | Drought source | 🟢 | State-level. |
| USDM-derived per-cutoff features | `scripts/drought_features.py` | Drought features | 🟢 | Merged. |
| TIGER 2018 county polygons | US Census | Spatial reduction layer | 🟢 | `phase2/data/tiger/tl_2018_us_county/` + `phase2/data/tiger/tl_2018_us_county_5states_5070.gpkg` (5-state subset, EPSG:5070). |
| **CDL annual binary corn masks** | **CropScape via `download_cdl.py`** | **HLS chip masking (D.1.b/c)** | 🟢 **(D.1.a)** | **`phase2/cdl/cdl_corn_mask_<state>_<year>.tif × 60` for 5 states × 2013–2024. Plus the 2025 mask rebuilt via the same pipeline. EPSG:5070 / 30 m / uint8 / LZW. Pixel-aligned with HLS, gSSURGO.** |
| **CDL raw categorical geotiffs** | **CropScape** | **Source for corn masks; retained for mask-rule iteration** | 🟢 **(D.1.a)** | **`phase2/cdl/raw/cdl_<state>_<year>.tif × 60` (~17.6 GB). Will be deleted in cleanup pass at end of D.1.** |
| NASS state yields | NASS | State-level validation | 🟢 | |

## Superseded / deleted in this session

- **State-level HLS slice CSVs** at `phase2/data/hls/hls_vi_features*.csv` (5 files). Deleted. Superseded by D.1.b's per-county chip extraction pipeline.
- **`scripts/hls_county_features.csv`**. Deleted. Output of the old `hls_features.py` against the now-deleted slices.
- **`scripts/hls_pull.py`, `scripts/hls_features.py`**: kept as **read-only references** for D.1.b's `download_hls.py`. The GDAL/earthaccess auth pattern, Fmask bit decoding, and L30/S30 band-name asymmetry handling are reused. To be deleted at end of Phase D.1 cleanup.

## Not yet acquired

| Dataset | Source | Use | Coverage needed | Status | Notes |
|---|---|---|---|---|---|
| HLS imagery (raw multispectral) → county chips | NASA LP DAAC + `download_hls.py` | Prithvi input | 5 states, 2013–2024, growing season | 🔴 **Need (D.1.b — NEXT)** | Pull architecture: pull-once-per-(state, year), label chips at index time, pick chips at embed time. ~24,000 granules expected; ~150 GB peak transient disk; ~50 GB persisted as 224×224×6 county chips. Multi-evening pull. |
| Prithvi model weights | HuggingFace via terratorch | Geospatial feature extraction | 1 model | 🔴 Need (D.1.d) | `BACKBONE_REGISTRY.build("terratorch_prithvi_eo_v2_300_tl", pretrained=True)` triggers the ~1.2 GB download to `~/.cache/terratorch/`. |

## Deprioritized (with rationale)

- Hourly temperature — daily Tmin/Tmax + GDD is sufficient.
- Sunrise/sunset hours — modern hybrids photoperiod-insensitive; gridMET srad covers actually-useful signal.
- CO₂ concentration — spatially uniform across CONUS.
- Tornado activity — highly localized, near-zero signal at county-year aggregation.
- PRISM (vs. gridMET) — picked gridMET; PRISM remains a fallback.
- NAIP aerial imagery — see decisions log entry 2026-04-25.

## CDL coverage by state and year (NDVI/HLS corn-masking dependency)

The MOD13Q1 (MODIS NDVI) source is available 2000–present. The CDL corn-mask coverage was uneven 2005–2007 (relevant for Phase A.2 NDVI), but is **complete for Phase D.1's 2013–2024 window** as confirmed in D.1.a's QC table. The 2005–2007 caveat (~1.9% NDVI null rate from CO 2005–2007) is documented and present only in the MODIS NDVI columns, not in the new D.1 HLS chip pipeline.

| State | FIPS | Counties (TIGER 2018) | NDVI 2005–2007 (Phase A) | NDVI 2008+ (Phase A) | CDL corn-mask 2013–2024 (D.1.a) |
|---|---|---|---|---|---|
| Iowa | 19 | 99 | 99 ✅ | 99 ✅ | ✅ all 12 years |
| Nebraska | 31 | 93 | 93 ✅ | 93 ✅ | ✅ all 12 years |
| Wisconsin | 55 | 72 | 72 ✅ | 72 ✅ | ✅ all 12 years |
| Missouri | 29 | 115 | ~28–110 ⚠️ | 110–113 ✅ | ✅ all 12 years |
| Colorado | 08 | 64 | ~3–40 ❌ | 37–38 ✅ | ✅ all 12 years |

**Phase B retrieval is per-county** and **Phase D.1 chip extraction is per-county**, so coverage variation across states does not contaminate analog matching or chip embedding.

## Phase D.1 outputs (planned)

| File | Sub-phase | Status |
|---|---|---|
| `phase2/cdl/raw/cdl_<state>_<year>.tif × 60` | D.1.a | ✅ Built |
| `phase2/cdl/cdl_corn_mask_<state>_<year>.tif × 60` | D.1.a | ✅ Built |
| `data/v2/hls/chips/<GEOID>/<year>/<phase>_<scene_date>.tif` | D.1.b/c | 🔴 Need |
| `data/v2/hls/chip_index.parquet` | D.1.b/c | 🔴 Need |
| `data/v2/prithvi/embeddings_v1.parquet` | D.1.d | 🔴 Need |
| `models/forecast_d1/regressor_<date>.json × 4` | D.1.e | 🔴 Need |
| `runs/d1_ablation_<timestamp>.csv` | D.1.e | 🔴 Need |

## Data schema (locked, all sources)

- **Spatial key:** `GEOID` as 5-character zero-padded string ("19153" for Polk County, IA).
- **Temporal keys:** `year` as int; `forecast_date` as one of `{"08-01", "09-01", "10-01", "EOS"}`.
- **State key:** `state_alpha` as 2-char USPS code (IA, CO, WI, MO, NE).
- **Yield units:** `bu/acre`.
- **Area units:** acres.
- **All tabular outputs:** parquet (snappy) or CSV. Rasters: GeoTIFF / COG.
- **As-of rule:** features for forecast date `D` in year `Y` use only data with timestamps strictly before `D`. Enforced in feature-construction layer (`weather_features.py`, `drought_features.py`, D.1.d's chip-picker via `scene_date < forecast_date`).

## Storage plan

- Tabular: `scripts/*.csv`, `scripts/*.parquet` for derived/feature outputs; `phase2/data/<source>/` for raw multi-file pulls; `data/v2/<source>/raw/` for raw caches.
- HLS county chips (D.1.b): `data/v2/hls/chips/<GEOID>/<year>/<phase>_<scene_date>.tif`. Granule cache `data/v2/hls/raw/` rolling-deleted.
- CDL: `phase2/cdl/raw/` (raw categorical, ~17.6 GB; deleted at end of D.1) + `phase2/cdl/` (binary masks, ~3 GB; retained).
- gridMET raw: `data/v2/weather/raw/` ✅; derived: `scripts/weather_county_features.csv` ✅
- USDM raw: `phase2/data/drought/` ✅; derived: `scripts/drought_county_features.csv` ✅
- Prithvi embeddings (D.1.d): `data/v2/prithvi/embeddings_v1.parquet`.
- Master table: `scripts/training_master.parquet` ✅. Rebuilt at end of D.1 to incorporate Prithvi embeddings via left-join.

## Open data questions

- **Wisconsin silage vs grain.** CDL doesn't disambiguate corn-for-silage from corn-for-grain at the pixel level (both are class 1). NASS yield target is grain-only. Phase D.1 chip embeddings will encode silage signal mixed with grain signal in WI. Documented; revisit only if WI per-state RMSE underperforms in the D.1.e ablation.
- **MODIS NDVI as-of fidelity is weak.** Same value at all 4 forecast dates within a year. Stripped from Phase C regressor (2-C.1); retained in Phase B retrieval embedding (less leakage-sensitive there). Will not appear in Phase D.1's HLS pipeline at all — replaced by phase-windowed chips with strict as-of fidelity.
- **NASS-aux fields are post-hoc reported.** Same value across all 4 forecast dates. Treat as structural priors. Switch to lagged variants if D.1 ablations expose dependence.
- **Within-state spatial heterogeneity is lost for USDM** (state-level broadcast). Acceptable for v2 baseline.
- **Pre-2013 train years** absent from D.1's pool by construction (HLS doesn't exist). Phase C-as-is bundle (2005–2022 train) is the fallback if D.1 fails the gate.
- **2013–2014 Landsat-only era cadence is lower** than 2015+. May see per-county chip count drop below 1 in some 2013–2014 (state, phase) cells. If so, those `(GEOID, year, forecast_date)` rows get NaN embeddings — XGBoost handles via missing-direction split.

## Decisions resolved since last inventory update

- ✅ **D.1 sub-phase plan locked** — Prithvi-EO-2.0-300M-TL, county granularity, T=3 sequence, corn-richest sub-window, calendar phase windows, pull-once architecture, 70% granule cloud filter, 5% chip corn-fraction filter, 100 granule cap, 2013–2022 train pool, 4-row ablation. See decisions log entry 2-D.1.kickoff.
- ✅ **CDL annual masks built** — 60 binary corn masks for 5 states × 2013–2024 in `phase2/cdl/`. See decisions log entry 2-D.1.a.

## Phase A definition-of-done — MET ✅

(Unchanged from prior inventory.)

## Phase D.1 sub-phase tracking

- [x] D.1.a — CDL annual masks pulled and built
- [ ] D.1.b — **HLS download orchestration (NEXT)**
- [ ] D.1.c — Chip extraction (inline with D.1.b)
- [ ] D.1.d — Prithvi inference + embeddings parquet
- [ ] D.1.e — Regressor retrain + 4-row ablation table → gate decision

## Budget note

v1 used ~$0.25 of the $10 API cap. v2 token usage will be similar in shape. Compute cost for HLS download + Prithvi inference is local and outside the API budget.