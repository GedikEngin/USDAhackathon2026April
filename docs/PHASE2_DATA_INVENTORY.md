# Data Inventory — v2 Corn Yield Forecasting

> Living tracker. Update whenever data status changes — pulled, parsed, cleaned, joined, validated, or discovered missing. This is the single source of truth for "do we have what we need to start the next phase."

**Last updated:** 2026-04-25, end of Phase C — **PHASES A, B, C COMPLETE.** No new data acquired in Phase B or C; both phases consumed `training_master.parquet`. Open items below updated to reflect what Phase B and C resolved or deferred.

## Status legend

- 🟢 **Have & ready** — on disk, parsed, schema-validated, joinable
- 🟡 **Have, partial / needs cleaning** — on disk but not yet in canonical schema, or has known gaps
- 🔵 **In flight** — actively pulling/processing
- 🔴 **Need** — not yet acquired
- ⚫ **Deprioritized** — known, decided not worth the integration cost

**Coverage target:** 5 states (IA, CO, WI, MO, NE), county level, **2005–2024** per the brief.

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

**Reference doc:** `docs/PHASE2_DATA_DICTIONARY.md` — column-by-column with units, ranges, NaN patterns, formulas, phase windows, and a worked example. **This is the file Phase B/C/F engineers (and LLM agents) read.**

**Source script:** `scripts/merge_all.py`. Outer-joins six sources (NASS / NDVI / gSSURGO / weather / drought / HLS), respects all locked schema decisions, has per-layer row-count invariants and end-of-script asserts.

---

## Required by the brief

| Dataset | Source | Use | Coverage needed | Status | Notes |
|---|---|---|---|---|---|
| Prithvi foundation model | NASA / IBM via HuggingFace | Geospatial feature extraction | Model weights | 🔴 Need | Download in Phase D.1. ~600M params. |
| HLS (Harmonized Landsat-Sentinel) | NASA LP DAAC | Multispectral imagery, Prithvi input | 5 states, 2005–2024, growing season | 🟡 Partial | Provisional state-level VI features (4 cols) merged in `training_master.parquet`. **Will be redone in Phase D.1** with consistent slice schema and county-level granularity for Prithvi feature extraction. Current slices: `phase2/data/hls/hls_vi_features_*.csv` (5 files, ~260 deduped rows). 2005–2013 is Landsat-only, low cadence. |
| NAIP aerial imagery | USDA FSA | High-res visual context | 5 states, recent years | ⚫ Excluded | Decided 2026-04-25 (see decisions log). NAIP doesn't fit any v2 phase: wrong sensor for Prithvi (HLS-pretrained), wrong cadence for in-season forecasting, wrong job for corn masking (CDL is purpose-built). Documented as "considered, excluded" in the final write-up. |
| USDA NASS yield (CORN, GRAIN — bu/acre) | NASS QuickStats API | Ground truth | 5 states, county level, 2005–2024 | 🟢 Have | `scripts/nass_corn_5states_2005_2024.csv` (6,837 × 16). 100% combined-practice coverage. Engineered features in `nass_corn_5states_features.csv` (6,834 × 10). Merged into `training_master.parquet`. |
| Cropland Data Layer (CDL) | NASS CropScape | Corn-pixel mask | 5 states, 2005–2024 | 🟡 Have via Earth Engine | Currently used inside the GEE NDVI script (`cropland == 1` mask). Standalone local download still needed if the HLS pipeline is built (Phase D.1) for offline masking. CDL coverage is uneven before 2008 — see "CDL coverage by state and year" below. |
| Weather / climate (daily, gridded) | gridMET | GDD, EDD, VPD, precip accumulation, srad | 5 states, daily, 2005–2024 | 🟢 Have | `scripts/gridmet_county_daily_2005_2024.parquet` + 20 per-year parquets. Derived per-cutoff features in `scripts/weather_county_features.csv` (35,440 × 14). Merged into `training_master.parquet`. |

## Strongly recommended additions

| Dataset | Source | Use | Coverage needed | Status | Notes |
|---|---|---|---|---|---|
| US Drought Monitor | NDMC / USDM | Drought severity feature for retrieval embedding | 5 states, weekly, 2005–2024 | 🟢 Have | `phase2/data/drought/drought_USDM-...csv` raw (6,865 weekly state-level rows). Derived per-cutoff features in `scripts/drought_county_features.csv` (27,336 × 9, 0 nulls). Merged into `training_master.parquet`. |

## Already on hand

| Dataset | Source | Use in v2 | Status | Notes |
|---|---|---|---|---|
| NASS combined-practice corn yield 2005–2024 | NASS QuickStats API | Ground truth | 🟢 | `scripts/nass_corn_5states_2005_2024.csv`. Full brief range. 100% on combined-practice columns; ~7% irrigated (CO + NE only). |
| NASS engineered features | `scripts/nass_features.py` | Engineered yield features | 🟢 | `scripts/nass_corn_5states_features.csv` (6,834 × 10). Merged into master table. |
| MODIS NDVI mapping, 5 states | GEE / MODIS MOD13Q1 | Vegetation index features | 🟢 | 21 per-year CSVs in `phase2/data/ndvi/`. Merged into master table. **Pre-scaled (× 0.0001 server-side); do NOT re-scale.** Coverage caveat: CO 2005–2007 has 1.9% null due to CDL pre-rollout. **NDVI ≠ HLS** — NDVI is a derived index from MODIS; raw HLS for Prithvi is still partial. |
| GEE NDVI script (version-controlled) | Local | Reproducibility | 🟢 | `scripts/ndvi_county_extraction.js`. Header documents bug fixes, output schema, how to extend or re-run. Per-year export task pattern. |
| gSSURGO Valu1 county features, 5 states | `scripts/gssurgo_county_features.py` | Soil features | 🟢 | `scripts/gssurgo_county_features.csv` (443 × 13). Static across years. Merged into master table. |
| gridMET daily weather, 5 states, 2005–2024 | `scripts/gridmet_pull.py` | GDD, EDD, VPD, precip, srad source | 🟢 | Raw daily parquets per year + combined parquet. Cached netcdfs in `data/v2/weather/raw/_gridmet_nc_cache/`. |
| gridMET-derived per-cutoff features | `scripts/weather_features.py` | Climate features | 🟢 | `scripts/weather_county_features.csv` (35,440 × 14). Merged into master table. |
| US Drought Monitor weekly raw | NDMC | Drought source | 🟢 | `phase2/data/drought/drought_USDM-Colorado,Iowa,Missouri,Nebraska,Wisconsin.csv`. **State-level, not county-level.** Cumulative percent area (D0 ≥ D1 ≥ D2 ≥ D3 ≥ D4 by construction). |
| USDM-derived per-cutoff features | `scripts/drought_features.py` | Drought features | 🟢 | `scripts/drought_county_features.csv` (27,336 × 9, 0 nulls). Merged into master table. State readings broadcast to all GEOIDs in state; as-of join uses `valid_end < forecast_date` (strict). |
| TIGER 2018 county polygons | US Census | Spatial reduction layer | 🟢 | `phase2/data/tiger/tl_2018_us_county/` (full shapefile) + `phase2/data/tiger/tl_2018_us_county_5states_5070.gpkg` (5-state subset, EPSG:5070 for gSSURGO zonal stats). |
| HLS-derived state-level VI features (provisional) | (HLS pull pipeline, separate) | Vegetation indices for the master table; redone in Phase D.1 | 🟡 | 5 slice CSVs in `phase2/data/hls/hls_vi_features_*.csv`. **Inconsistent slice schemas** (some have `is_forecast` and `n_granules`, others don't). `merge_all.py` allowlists only `ndvi_mean, ndvi_std, evi_mean, evi_std`; meta columns dropped. 59.1% null in master table (pre-2013 structural). |
| NASS state yields | NASS | State-level validation | 🟢 | |
| Principal Crops Area Planted/Harvested 2023–2025 | NASS | Acres feature | 🟡 | Useful; need to extend back to 2005 if used. |
| Corn Area Planted/Harvested + Yield + Production 2023–2025 | NASS | Acres feature | 🟡 | Same — extend if used. |
| Corn for Silage Area Harvested 2023–2025 | NASS | Disambiguate grain vs silage | 🟢 | Niche but useful. |
| Corn Plant Population per Acre 2021–2025 | NASS | Density feature | 🟢 | Limited history; recent-years feature only. |
| Water applied to corn/grain, 5 states, county, 2018–2023 | USDA Irrigation Survey | Irrigation feature | 🟡 | Sparse temporally (every ~5 years); use as quasi-static county feature. |

## Not yet acquired

| Dataset | Source | Use | Coverage needed | Status | Notes |
|---|---|---|---|---|---|
| HLS imagery (raw multispectral, county-level) | NASA LP DAAC | Prithvi input | 5 states, 2005–2024, growing season | 🔴 Need (Phase D.1) | Largest dataset by far. TB-scale. Settle storage strategy before pulling. The currently-merged `hls_vi_features_*.csv` are state-level VI summaries from a separate pipeline; not the raw cubes Prithvi needs. |
| CDL standalone (local) | NASS CropScape | Offline corn masking for HLS pipeline | 5 states, 2005–2024 | 🔴 Need (Phase D.1) | Currently used only inside GEE; pull a local copy when HLS pipeline lands. |
| Prithvi model weights | HuggingFace | Geospatial feature extraction | 1 model | 🔴 Need (Phase D.1) | ~600M params. |

## Deprioritized (with rationale)

- Hourly temperature — daily Tmin/Tmax + GDD is sufficient for county-annual yield.
- Sunrise/sunset hours — deterministic from latitude+date; modern corn hybrids photoperiod-insensitive. gridMET solar radiation covers actually-useful signal.
- CO₂ concentration — spatially uniform across CONUS; absorbed by `year` as a feature.
- Tornado activity — highly localized damage; near-zero signal at county-year aggregation.
- PRISM (vs. gridMET) — picked gridMET for daily weather. PRISM remains a viable fallback if gridMET ever shows quality issues.
- NAIP aerial imagery — see decisions log entry 2026-04-25.

## CDL coverage by state and year (NDVI corn-masking dependency)

The MOD13Q1 source is available 2000–present, but corn-masking depends on USDA CDL coverage, which started rolling out state-by-state in the early 2000s. Empirical coverage from the actual exports (counties with non-null `ndvi_peak`):

| State | FIPS | Counties (TIGER 2018) | 2004–2005 | 2006–2007 | 2008+ |
|---|---|---|---|---|---|
| Iowa | 19 | 99 | 99 ✅ | 99 ✅ | 99 ✅ |
| Nebraska | 31 | 93 | 93 ✅ | 93 ✅ | 93 ✅ |
| Wisconsin | 55 | 72 | 72 ✅ | 72 ✅ | 72 ✅ |
| Missouri | 29 | 115 | ~28 ⚠️ | ~110 ✅ | 110–113 ✅ |
| Colorado | 08 | 64 | ~3–4 ❌ | ~40 ⚠️ | 37–38 ✅ |

**Implications confirmed in master table:**
- Brief specifies 2005–2024 — fully usable for IA/NE/WI all years; MO from 2006; CO from 2008.
- **Strong block: 2008–2024** (~17 years × ~410 counties).
- 2005–2007 is usable but heterogeneous; CO essentially no NDVI 2005–2007 (visible as the 1.9% NDVI null rate in `training_master.parquet`).
- The ~30 counties always missing post-2008 are corn-absent counties (St. Louis City, Denver, CO mountain counties, urban WI). They drop out of training because they have no NASS yield either — explains why master table has 388 GEOIDs vs. 443 TIGER counties.
- **Phase B retrieval is per-county**, so coverage variation across states does not contaminate analog matching.

## Data schema (locked, all sources)

- **Spatial key:** `GEOID` as 5-character zero-padded string ("19153" for Polk County, IA).
- **Temporal keys:** `year` as int; `forecast_date` as one of `{"08-01", "09-01", "10-01", "EOS"}`.
- **State key:** `state_alpha` as 2-char USPS code (IA, CO, WI, MO, NE).
- **Yield units:** `bu/acre`.
- **Area units:** acres.
- **All tabular outputs:** parquet (snappy) or CSV. Rasters: GeoTIFF / COG.
- **As-of rule:** features for forecast date `D` in year `Y` use only data with timestamps strictly before `D`. Enforced in feature-construction layer (`weather_features.py`, `drought_features.py`); MODIS NDVI and NASS-aux are documented exceptions (see `PHASE2_DATA_DICTIONARY.md`).

## Storage plan

- Tabular: `scripts/*.csv` and `scripts/*.parquet` for derived/feature outputs; `phase2/data/<source>/` for raw multi-file pulls; `data/v2/weather/raw/` for the gridMET parquet cache.
- HLS imagery (when pulled in D.1): `data/v2/hls/{state}/{year}/`.
- CDL standalone (when pulled): `data/v2/cdl/{state}/{year}.tif`.
- Master table: `scripts/training_master.parquet` ✅
- gSSURGO aggregated: `scripts/gssurgo_county_features.csv` ✅
- gridMET raw: `data/v2/weather/raw/` ✅; derived: `scripts/weather_county_features.csv` ✅
- USDM raw: `phase2/data/drought/` ✅; derived: `scripts/drought_county_features.csv` ✅

## Open data questions

- **HLS slice cleanup deferred to Phase D.1** — current 5 slice CSVs have inconsistent schemas; will be redone with proper county-level pulls when Prithvi work starts.
- **MODIS NDVI as-of fidelity is weak.** Same value at all 4 forecast dates within a year. **RESOLVED for Phase C (2026-04-25):** SHAP analysis on the first trained Phase C bundle confirmed the regressor was treating MODIS NDVI as a hard answer rather than a soft prior — `ndvi_peak` dominated SHAP at every forecast date including 08-01. All 5 MODIS NDVI columns dropped from `forecast/regressor.py::_NDVI` (now `[]`); see `PHASE2_DECISIONS_LOG.md` Phase 2-C.1 entry. **Phase B retrieval embedding still uses MODIS NDVI** (2 columns: `ndvi_peak`, `ndvi_gs_mean`) — retrieval matching is less sensitive to as-of leakage than point prediction, and Phase B's cone is calibrated to 80% coverage with NDVI in the embedding. Replacement for the regressor will be Phase D.1's HLS-derived running NDVI clipped to forecast_date.
- **NASS-aux fields (`acres_*`, `irrigated_share`, `harvest_ratio`) are post-hoc reported.** Same value across all 4 forecast dates. Treat as structural priors. **Phase C SHAP analysis (2026-04-25):** `acres_planted_all` is the #2 driver by mean |SHAP| (8.6 bu/acre, after `year` at 13.4); `harvest_ratio` is #3 (4.3); `irrigated_share` is #6 (0.5 mean signed). The dominance is at every forecast_date including 08-01, which is suspicious per the original flag — but in practice these values are reasonably stable year-over-year for a given county and the model is using them as county-typing priors rather than in-season measurements. Acceptable for Phase C; revisit with prior-year-lagged variants if Phase D.1 ablations show the dependence is harmful.
- **Within-state spatial heterogeneity is lost for HLS and USDM** (both broadcast from state level). Acceptable for v2 baseline.
- **County coverage filter:** **RESOLVED:** `n_min_history=10` qualifying training years per GEOID for analog candidacy (in `forecast/data.py::train_pool`). Used by Phase B retrieval and Phase C training. 345 of 388 GEOIDs kept; 43 dropped. Counties below threshold can still be QUERIED (forecast for them) but never appear AS analogs. Revisit only if Phase D.1 finds a counterexample.
- **Drought feature enrichment.** **RESOLVED (default ship was sufficient through Phase C):** Phase C SHAP shows `d2_pct` in top-10 mean signed SHAP (−0.85 — drought reduces predicted yield, as expected); `d0_pct` similar (−0.87). Minimal D0–D4 set proved adequate. Candidates if Phase D.1+ wants richer signal stay on the shelf: DSCI, season-cum drought weeks, silking-peak DSCI, trailing-4-week mean of d2plus.

## Decisions resolved since last inventory update

- ✅ **`merge_all.py` ships with min-year filter, HLS feature allowlist, NDVI schema-drift guard, per-layer row-count asserts.** Phase A.6.
- ✅ **`PHASE2_DATA_DICTIONARY.md` written as a comprehensive reference doc** (quick-facts table, source-grouped sections with units/ranges/formulas/phase windows, NaN-patterns triage table, as-of fidelity reference, worked example). Phase A.7.
- ✅ **NAIP excluded from all v2 phases.** See decisions log 2026-04-25.
- ✅ **Phase B analog-cone baseline shipped** with `same_geoid` pool, K=5, percentiles (10, 50, 90), per-county trend with state-median fallback, per-(state, forecast_date) recalibration fit on val 2023. Gate passed post-recal on 2024 holdout.
- ✅ **Phase C XGBoost regressor shipped.** Bundle in `models/forecast/regressor_*.json`. Gate passed on val 2023 EOS at +46.7% lift (threshold 15%). `forecast/regressor.py` + `forecast/explain.py` modules; `scripts/train_regressor.py` + `scripts/backtest_phase_c.py` drivers.
- ✅ **MODIS NDVI dropped from regressor feature set after Phase 2-C.1 SHAP-driven leakage discovery.** Phase B retrieval embedding unaffected. Replacement is Phase D.1.
- ✅ **Recalibration is dropped for Phase C.** Val-2023-fitted constants would help val and hurt 2024 holdout (sign flips between years on CO/MO/WI).

## Phase A definition-of-done — MET

- [x] Every dataset listed above is on disk, parsed, and joinable on `(GEOID, year)` or `(GEOID, year, forecast_date)`.
- [x] `training_master.parquet` builds end-to-end without errors.
- [x] `merge_all.py` asserts pass (key uniqueness, GEOID padding, valid forecast_date, per-layer row-count invariants).
- [x] Every NaN cell in the master table has a documented cause (structural / sparse-coverage / disclosure suppression).
- [x] `PHASE2_DATA_DICTIONARY.md` written and exhaustive.

## Budget note

v1 used ~$0.25 of the $10 API cap. v2 token usage will be similar in shape (Haiku narration, ~$0.02 per forecast). Compute cost for HLS download + Prithvi inference is local (Ubuntu box) and outside the API budget.