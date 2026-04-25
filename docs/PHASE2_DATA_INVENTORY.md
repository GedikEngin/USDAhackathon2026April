# Data Inventory — v2 Corn Yield Forecasting

> Living tracker. Update whenever data status changes — pulled, parsed, cleaned, joined, validated, or discovered missing. This is the single source of truth for "do we have what we need to start the next phase."

**Last updated:** 2026-04-25, mid-Phase A (data acquisition in progress)

## Status legend

- 🟢 **Have & ready** — on disk, parsed, schema-validated, joinable
- 🟡 **Have, partial / needs cleaning** — on disk but not yet in canonical schema, or has known gaps
- 🔵 **In flight** — actively pulling/processing
- 🔴 **Need** — not yet acquired
- ⚫ **Deprioritized** — known, decided not worth the integration cost

**Coverage target:** 5 states (IA, CO, WI, MO, NE), county level, **2005–2024** per the brief.

---

## Required by the brief

| Dataset | Source | Use | Coverage needed | Status | Notes |
|---|---|---|---|---|---|
| Prithvi foundation model | NASA / IBM via HuggingFace | Geospatial feature extraction | Model weights | 🔴 Need | Download in Phase D.1. ~600M params. |
| HLS (Harmonized Landsat-Sentinel) | NASA LP DAAC | Multispectral imagery, Prithvi input | 5 states, 2005–2024, growing season | 🔴 Need | Largest dataset by far. TB-scale. 2005–2013 is **Landsat-only** (Sentinel-2 didn't launch until 2015), lower revisit cadence in early years. Settle storage strategy before pulling. Phase D.1. |
| NAIP aerial imagery | USDA FSA | High-res visual context | 5 states, recent years | 🔴 Need | Tertiary — only if a phase explicitly calls for sub-meter. |
| USDA NASS yield (CORN, GRAIN — bu/acre) | NASS QuickStats API | Ground truth | 5 states, county level, 2005–2024 | 🟡 Have, partial | Have 2015–2024 (2,998 county-year rows, 374 unique counties, 100% coverage on combined-practice). **Need to extend back to 2005** — re-run `scripts/nass_pull.py`. The rate-limit fixes are committed but not yet exercised on the 2005–2014 range. |
| Cropland Data Layer (CDL) | NASS CropScape | Corn-pixel mask | 5 states, 2005–2024 | 🟡 Have via Earth Engine | Currently used inside the GEE NDVI script (`cropland == 1` mask). Standalone local download may be needed if the HLS pipeline is built (Phase D.1) for offline masking. |
| Weather / climate (daily, gridded) | PRISM (or gridMET) | GDD, EDD, VPD, precip accumulation | 5 states, daily, 2005–2024 | 🔴 Need | Existing monthly mean temp/precip is too coarse for GDD. Phase A.4. |

## Strongly recommended additions

| Dataset | Source | Use | Coverage needed | Status | Notes |
|---|---|---|---|---|---|
| US Drought Monitor — Cumulative Percent Area | NDMC / USDM data export | Drought severity feature for retrieval embedding | 5 states, weekly, 2005–2024 | 🟢 Have | Format chosen: **Cumulative Percent Area** (each D-level column reports % of county at that level *or worse*; population-weighted variants rejected — corn fields don't care about population). Categorical splits derivable as `cat_D{n} = cum_D{n} − cum_D{n+1}`. DSCI derivable as `D0+D1+D2+D3+D4` cumulative sum (0–500). Drought stress is a top-3 yield predictor; DSCI is a candidate for the retrieval embedding itself. Phase A.5 will write `drought_features.py`. |

## Already on hand

| Dataset | Source | Use in v2 | Status | Notes |
|---|---|---|---|---|
| NASS combined-practice corn yield 2015–2024 | NASS QuickStats API | Ground truth | 🟡 | `scripts/nass_corn_5states_2015_2024.csv`, 2,998 rows, 12 cols. 100% coverage on `yield_bu_acre_all`, `production_bu_all`, `acres_harvested_all`, `acres_planted_all`. ~7% coverage on irrigated columns (CO + NE only). 0% on non-irrigated (rate-limited out; derivable from combined − irrigated). **Extend to 2005 in Phase A.1.** |
| NASS state yields | NASS | State-level validation | 🟢 | |
| Principal Crops Area Planted/Harvested 2023–2025 | NASS | Acres feature | 🟡 | Useful; need to extend back to 2005. |
| Corn Area Planted/Harvested + Yield + Production 2023–2025 | NASS | Acres feature | 🟡 | Same — extend. |
| Corn for Silage Area Harvested 2023–2025 | NASS | Disambiguate grain vs silage | 🟢 | Niche but useful. |
| Corn Plant Population per Acre 2021–2025 | NASS | Density feature | 🟢 | Limited history; recent-years feature only. |
| Water applied to corn/grain, 5 states, county, 2018–2023 | USDA Irrigation Survey | Irrigation feature | 🟡 | Sparse temporally (every ~5 years); use as quasi-static county feature. |
| Corn/grain yield 2015–2024 county-level | NASS | Ground truth | 🟢 | Subset of the bigger NASS pull above. |
| GEOID-keyed NASS yield+irrigation+harvest_ratio table | User-prepared | Master join key | 🟢 | The `GEOID,year,state_alpha,county_name,yield_target,...` table is the join skeleton. |
| gSSURGO state .gdb files (NE, CO, IA, WI, MO, 10m) | USDA NRCS | Soil features | 🟡 | Files downloaded; **`scripts/gssurgo_extract.py` not yet written**. Plan to use Valu1 table for county-level aggregation. Phase A.3. |
| MODIS NDVI mapping, 5 states (Google Earth Engine) | GEE / MODIS MOD13Q1 | Vegetation index features | 🔵 In flight | Export task `corn_ndvi_5states_2015_2024` submitted; lands in Drive `EarthEngineExports/`. Schema: `GEOID, NAME, STATEFP, year, ndvi_peak, ndvi_gs_mean, ndvi_gs_integral, ndvi_silking_mean, ndvi_veg_mean`. **Re-run for 2005–2014 to backfill.** **NDVI ≠ HLS** — NDVI is a derived index; raw HLS is still 🔴 Need for Phase D.1 (Prithvi). |
| Mean monthly temperature heatmap + CSV | User-provided | Coarse smoothed temp feature | 🟡 | Keep as a coarse feature. Does NOT replace daily for GDD. |
| Mean monthly precipitation heatmap + CSV | User-provided | Coarse smoothed precip feature | 🟡 | Same — coarse feature, does not replace daily. |
| USDA NASS API access | NASS | Source for NASS pulls | 🟢 | Key in `.env` as `NASS_API_KEY`. `python-dotenv` loaded. |

## Deprioritized (with rationale)

| Dataset | Reason |
|---|---|
| Tornado activity per state 2000–2022 | Tornado damage is highly localized; aggregated to county-year it's near-zero signal for state-level yield. ⚫ NOAA Storm Events is a possible replacement if a broader severe-weather signal is wanted. |
| Hourly temperature | Overkill for county-annual yield; daily Tmin/Tmax + GDD/EDD captures the heat signal. ⚫ |
| Sunrise/sunset hours | Day length is deterministic from latitude+date; modern corn hybrids are largely photoperiod-insensitive. Solar radiation from gridMET/NSRDB covers what's needed. ⚫ |
| CO₂ concentration | Spatially uniform across CONUS; absorbed by `year` as a feature (which also captures technology + genetics trends). ⚫ |
| Pixel-level gSSURGO in modeling | Soil features are static across years, so pixel-level only explains between-county variance, not between-year. Aggregate to county-level via Valu1 + zonal stats. ⚫ |

## Things we have NOT yet decided

These are deferred until a relevant phase forces the call. Listed here so they aren't forgotten.

- **PRISM vs. gridMET** — both work for daily weather. Decide when `prism_pull.py` is written.
- **Final list of features that go into the retrieval embedding** vs. covariates only. Decided in Phase B.1.
- **K** for K-nearest analog retrieval. Default 10; tune in Phase B.5.
- **Cone percentiles to publish** in the API. Default {10, 50, 90}; could expand to {5, 25, 50, 75, 95}. Decided in Phase B / E.
- **State-aggregation method** for percentile rollup. Default planted-acres-weighted mean of each percentile; document caveat. Could revisit in Phase G.
- **Prithvi exact band order and normalization.** Discovered during D.1 implementation, from Prithvi's HuggingFace card.
- **Whether to attempt Phase D.2 (Prithvi fine-tune).** Conditional on D.1 ablation.
- **Drought source: USDM Cumulative Percent Area locked.** SPI/SPEI from PRISM still derivable as a redundant signal if needed.
- **County coverage filter:** minimum N years of complete data per county. Decide after the full 2005–2024 NASS extension lands.

## Schema conventions (locked)

- **Spatial key:** `GEOID` as 5-character zero-padded string (e.g. "19153" for Polk County, IA).
- **Temporal keys:** `year` as 4-digit int, `forecast_date` as one of `{"08-01", "09-01", "10-01", "EOS"}`.
- **State key:** `state_alpha` as 2-char USPS code (IA, CO, WI, MO, NE).
- **Yield units:** `bu/acre` always. No silent conversions.
- **Area units:** acres always. No silent conversions to hectares.
- **All tabular outputs:** parquet (snappy) or CSV. Rasters: GeoTIFF / COG.
- **As-of rule for forecast features:** when constructing features for forecast date `D` in year `Y`, use only data with timestamps strictly before `D`. Enforced in `forecast/features.py`.

## Storage plan (provisional)

- Tabular data: `scripts/*.csv` and `scripts/*.parquet` for now (matches existing pattern); migrate to `data/v2/tabular/` if it gets crowded.
- HLS imagery (when pulled): `data/v2/hls/{state}/{year}/` — too big for project knowledge; lives on the Ubuntu box only.
- CDL standalone (if downloaded): `data/v2/cdl/{state}/{year}.tif`.
- Cached features: `scripts/training_master.parquet`.
- gSSURGO aggregated: `scripts/gssurgo_county_features.csv`.
- PRISM raw: `data/v2/weather/raw/` (large; possibly netcdf), derived: `scripts/prism_county_features.csv`.

## NASS pipeline state (recap from `PHASE2_CURRENT_STATE.md`)

**Pull script:** `scripts/nass_pull.py` — written, debugged, rate-limit fixes committed (2.0s delay, exponential backoff up to 240s, 4 retries on 403).

**Output (current):** `scripts/nass_corn_5states_2015_2024.csv` — 2,998 rows × 12 cols.

| Column | Coverage | Notes |
|---|---|---|
| `yield_bu_acre_all` | 100% | Primary target |
| `production_bu_all` | 100% | |
| `acres_harvested_all` | 100% | |
| `acres_planted_all` | 100% | |
| `yield_bu_acre_irr` | 6.9% | CO + NE only |
| `production_bu_irr` | 6.9% | CO + NE only |
| `acres_harvested_irr` | 6.9% | CO + NE only |
| `acres_planted_irr` | 0% | rate-limited out |
| `yield_bu_acre_noirr` | 0% | rate-limited out |
| `production_bu_noirr` | 0% | rate-limited out |
| `acres_harvested_noirr` | 0% | rate-limited out |
| `acres_planted_noirr` | 0% | rate-limited out |

**Per-state row counts (2015–2024):** CO 159, IA 906, MO 637, NE 710, WI 586.

**NASS data-quality fixes already applied in `nass_pull.py`:**
1. Pin `prodn_practice_desc` to avoid the multi-practice fan-out bug (without it, NASS returns 3 rows per county-year and merging 4 such tables produces 81 rows per county-year).
2. Filter rows where `county_name` starts with "OTHER" or `county_ansi` is empty (NASS rolls suppressed-data counties into a fake "OTHER (COMBINED) COUNTIES" GEOID ending in 000).
3. Strip commas before `pd.to_numeric` (large numbers come as `"14,200,000"` strings).

## Earth Engine NDVI state

**Script:** in GEE web editor. **TODO:** save to `scripts/ndvi_county_extraction.js` for version control.

**Export task:** `corn_ndvi_5states_2015_2024` → Google Drive `EarthEngineExports/corn_ndvi_5states_2015_2024.csv`.

**Backfill action:** re-run for 2005–2014. Suggested task name: `corn_ndvi_5states_2005_2014`. Combine into `corn_ndvi_5states_2005_2024.csv` after both land.

**Bug fixes already in script:**
- Use `ee.ImageCollection('USDA/NASS/CDL').filter(...)` instead of `'USDA/NASS/CDL/' + year` (server-side `ee.Number` cannot be JS-concatenated into asset paths).
- Wrap demo image with `ee.Image(...).set('system:description', 'demo')` to reset accumulated metadata.

**Caveats:**
- CDL 2024 may not exist yet (publishes Jan/Feb of following year). If the 2024 export fails, set `endYear=2023`.
- CDL 2005 is the earliest available; corn-pixel masking works back to 2005.
- MODIS NDVI scale factor 0.0001 (raw 0–10000); apply when consuming the CSV locally.

## gSSURGO state

**Source files:** state `.gdb` files for NE, CO, IA, WI, MO at 10m resolution. CONUS-level skipped (only 30m, >40GB).

**Extraction script:** `scripts/gssurgo_extract.py` — **not yet written**. Pseudocode plan in earlier conversation. Phase A.3.

**Plan recap:**
1. Read `.gdb` via geopandas/rasterio.
2. Use Valu1 table (USDA's pre-aggregated MUKEY-level table) — covers `nccpi3corn`, `aws0_100`, `soc0_30`, `rootznemc`, `droughty`, more.
3. Build per-property GeoTIFFs via MUKEY → property remap.
4. Zonal stats over TIGER/Line 2018 county polygons.
5. Reproject before zonal stats (gSSURGO is EPSG:5070 Albers).
6. Output: `scripts/gssurgo_county_features.csv` keyed on `GEOID` only (static across years).
