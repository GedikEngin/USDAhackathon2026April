# Data Inventory — v2 Corn Yield Forecasting

> Living tracker. Update whenever data status changes — pulled, parsed, cleaned, joined, validated, or discovered missing. This is the single source of truth for "do we have what we need to start the next phase."

**Last updated:** 2026-04-25, end of Phase A.1–A.2

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
| USDA NASS yield (CORN, GRAIN — bu/acre) | NASS QuickStats API | Ground truth | 5 states, county level, 2005–2024 | 🟢 Have | `scripts/nass_corn_5states_2005_2024.csv` covers the full brief-required range. 100% coverage on combined-practice columns. ~7% coverage on irrigated (CO + NE only). Non-irrigated columns sparse — derivable from `acres_harvested_all − acres_harvested_irr`. |
| Cropland Data Layer (CDL) | NASS CropScape | Corn-pixel mask | 5 states, 2005–2024 | 🟡 Have via Earth Engine | Currently used inside the GEE NDVI script (`cropland == 1` mask). Standalone local download may be needed if the HLS pipeline is built (Phase D.1) for offline masking. CDL coverage is uneven before 2008 — see "CDL coverage by state and year" below. |
| Weather / climate (daily, gridded) | PRISM (or gridMET) | GDD, EDD, VPD, precip accumulation | 5 states, daily, 2005–2024 | 🔴 Need | Existing monthly mean temp/precip is too coarse for GDD. Phase A.4. |

## Strongly recommended additions

| Dataset | Source | Use | Coverage needed | Status | Notes |
|---|---|---|---|---|---|
| US Drought Monitor | NDMC / USDM | Drought severity feature for retrieval embedding | 5 states, weekly, 2005–2024 | 🟡 Have, raw | `phase2/data/drought/drought_USDM-Colorado,Iowa,Missouri,Nebraska,Wisconsin.csv` on disk. Weekly D0–D4 percentages per county. **Needs feature engineering** (`scripts/drought_features.py` not yet written) to per-(GEOID, year, forecast_date) as-of values. Phase A.5. |

## Already on hand

| Dataset | Source | Use in v2 | Status | Notes |
|---|---|---|---|---|
| NASS combined-practice corn yield 2005–2024 | NASS QuickStats API | Ground truth | 🟢 | `scripts/nass_corn_5states_2005_2024.csv`. Full brief range. 100% coverage on `yield_bu_acre_all`, `production_bu_all`, `acres_harvested_all`, `acres_planted_all`. ~7% coverage on irrigated columns (CO + NE only). Non-irrigated derivable. |
| NASS engineered features | Local `scripts/nass_features.py` | Engineered yield features | 🟢 | `scripts/nass_corn_5states_features.csv`. Includes `yield_target`, `irrigated_share`, `harvest_ratio`, derived non-irrigated columns. Ready to merge. |
| MODIS NDVI mapping, 5 states | GEE / MODIS MOD13Q1 | Vegetation index features | 🟢 Have | 21 per-year CSVs in `phase2/data/ndvi/corn_ndvi_5states_<year>.csv` (2004–2024). 443 rows/file (TIGER 2018 county count for the 5 states). Schema: `GEOID, NAME, STATEFP, year, ndvi_peak, ndvi_gs_mean, ndvi_gs_integral, ndvi_silking_mean, ndvi_veg_mean`. **NDVI is pre-scaled (× 0.0001 applied server-side); CSV values are floats in [-0.2, 1.0] — do NOT re-scale.** Counties with no CDL corn pixels emit null NDVI columns (drop or null-impute downstream). **NDVI ≠ HLS** — NDVI is a derived index; raw HLS is still 🔴 Need for Phase D.1 (Prithvi). |
| GEE NDVI script (version-controlled) | Local | Reproducibility | 🟢 | `scripts/ndvi_county_extraction.js`. Header documents bug fixes, output schema, how to extend or re-run. Per-year export task pattern. |
| NASS state yields | NASS | State-level validation | 🟢 | |
| Principal Crops Area Planted/Harvested 2023–2025 | NASS | Acres feature | 🟡 | Useful; need to extend back to 2005 if used. |
| Corn Area Planted/Harvested + Yield + Production 2023–2025 | NASS | Acres feature | 🟡 | Same — extend if used. |
| Corn for Silage Area Harvested 2023–2025 | NASS | Disambiguate grain vs silage | 🟢 | Niche but useful. |
| Corn Plant Population per Acre 2021–2025 | NASS | Density feature | 🟢 | Limited history; recent-years feature only. |
| Water applied to corn/grain, 5 states, county, 2018–2023 | USDA Irrigation Survey | Irrigation feature | 🟡 | Sparse temporally (every ~5 years); use as quasi-static county feature. |
| GEOID-keyed NASS yield+irrigation+harvest_ratio table | User-prepared | Master join key | 🟢 | The `GEOID,year,state_alpha,county_name,yield_target,...` table is the join skeleton. |
| gSSURGO state .gdb files (NE, CO, IA, WI, MO, 10m) | USDA NRCS | Soil features | 🟡 | Files downloaded; **`scripts/gssurgo_extract.py` not yet written**. Plan to use Valu1 table for county-level aggregation. Phase A.3. |
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

## CDL coverage by state and year (drives NDVI coverage)

Empirical from the actual NDVI exports (counties with non-null `ndvi_peak`):

| State | FIPS | Counties (TIGER 2018) | 2004–2005 | 2006–2007 | 2008+ |
|---|---|---|---|---|---|
| Iowa | 19 | 99 | 99 ✅ | 99 ✅ | 99 ✅ |
| Nebraska | 31 | 93 | 93 ✅ | 93 ✅ | 93 ✅ |
| Wisconsin | 55 | 72 | 72 ✅ | 72 ✅ | 72 ✅ |
| Missouri | 29 | 115 | ~28 ⚠️ | ~110 ✅ | 110–113 ✅ |
| Colorado | 08 | 64 | ~3–4 ❌ | ~40 ⚠️ | 37–38 ✅ |

**Implications:**
- Strong block: **2008–2024** (~17 years × ~410 corn-bearing counties).
- 2005–2007 usable but heterogeneous. CO essentially absent 2005–2007.
- Brief is fully satisfied for IA/NE/WI all years; for MO from 2006; for CO from 2008.
- The ~30 counties always missing post-2008 are real corn-absent counties (St. Louis City 29510, Denver, CO mountain counties, urban WI).
- **Phase B retrieval is per-county**, so coverage variation across states does not contaminate analog matching — each county draws analogs from its own history.

## Things we have NOT yet decided

These are deferred until a relevant phase forces the call. Listed here so they aren't forgotten.

- **PRISM vs. gridMET** — both work for daily weather. Decide when `prism_pull.py` is written.
- **Final list of features that go into the retrieval embedding** vs. covariates only. Decided in Phase B.1.
- **K** for K-nearest analog retrieval. Default 10; tune in Phase B.5.
- **Cone percentiles to publish** in the API. Default {10, 50, 90}; could expand to {5, 25, 50, 75, 95}. Decided in Phase B / E.
- **State-aggregation method** for percentile rollup. Default planted-acres-weighted mean of each percentile; document caveat. Could revisit in Phase G.
- **Prithvi exact band order and normalization.** Discovered during D.1 implementation, from Prithvi's HuggingFace card.
- **Whether to attempt Phase D.2 (Prithvi fine-tune).** Conditional on D.1 ablation.
- **Drought feature aggregation** — D-level severities to include (sum of D2+? all D0–D4?), and resampling rule from weekly to forecast-date as-of (last reading? trailing-N-week mean?). Decide in Phase A.5.
- **County coverage filter:** minimum N years of complete data per county. Decide after PRISM and gSSURGO land.

## Schema conventions (locked)

- **Spatial key:** `GEOID` as 5-character zero-padded string (e.g. "19153" for Polk County, IA).
- **Temporal keys:** `year` as 4-digit int, `forecast_date` as one of `{"08-01", "09-01", "10-01", "EOS"}`.
- **State key:** `state_alpha` as 2-char USPS code (IA, CO, WI, MO, NE).
- **Yield units:** `bu/acre` always. No silent conversions.
- **Area units:** acres always. No silent conversions to hectares.
- **All tabular outputs:** parquet (snappy) or CSV. Rasters: GeoTIFF / COG.
- **As-of rule for forecast features:** when constructing features for forecast date `D` in year `Y`, use only data with timestamps strictly before `D`. Enforced in feature-construction layer.

## Storage plan (provisional)

- Tabular data: `scripts/*.csv` and `scripts/*.parquet` for now; `phase2/data/<source>/` for raw multi-file pulls (already used for ndvi/ and drought/).
- HLS imagery (when pulled): `data/v2/hls/{state}/{year}/` — too big for project knowledge; lives on the Ubuntu box only.
- CDL standalone (if downloaded): `data/v2/cdl/{state}/{year}.tif`.
- Cached features: `scripts/training_master.parquet`.
- gSSURGO aggregated: `scripts/gssurgo_county_features.csv`.
- PRISM raw: `data/v2/weather/raw/` (large; possibly netcdf), derived: `scripts/prism_county_features.csv`.

## NASS pipeline state

**Pull script:** `scripts/nass_pull.py` — written, debugged, rate-limit fixes committed (2.0s delay, exponential backoff up to 240s, 4 retries on 403). Run for full 2005–2024 range.

**Output:** `scripts/nass_corn_5states_2005_2024.csv`.

**Feature engineering:** `scripts/nass_features.py` — written. Output: `scripts/nass_corn_5states_features.csv`. Includes `yield_target`, `irrigated_share`, `harvest_ratio`, derived non-irrigated columns.

**Coverage by column (qualitative, post-extension):**

| Column | Coverage | Notes |
|---|---|---|
| `yield_bu_acre_all` | ~100% | Primary target |
| `production_bu_all` | ~100% | |
| `acres_harvested_all` | ~100% | |
| `acres_planted_all` | ~100% | |
| `*_irr` columns | sparse | CO + NE only, ~7% in original 2015–2024 cut |
| `*_noirr` columns | sparse | derivable from combined − irrigated |

Re-confirm exact post-extension percentages with a quick QC pass when convenient.

**NASS data-quality fixes already applied in `nass_pull.py`:**
1. Pin `prodn_practice_desc` to avoid the multi-practice fan-out bug (without it, NASS returns 3 rows per county-year and merging 4 such tables produces 81 rows per county-year).
2. Filter rows where `county_name` starts with "OTHER" or `county_ansi` is empty (NASS rolls suppressed-data counties into a fake "OTHER (COMBINED) COUNTIES" GEOID ending in 000).
3. Strip commas before `pd.to_numeric` (large numbers come as `"14,200,000"` strings).

## Earth Engine NDVI state

**Script:** `scripts/ndvi_county_extraction.js` — version-controlled. Per-year `Export.table.toDrive` task pattern (one task per year in the START_YEAR–END_YEAR range).

**Outputs:** 21 per-year CSVs in `phase2/data/ndvi/corn_ndvi_5states_<year>.csv` covering 2004–2024.

**Schema:** `GEOID, NAME, STATEFP, year, ndvi_peak, ndvi_gs_mean, ndvi_gs_integral, ndvi_silking_mean, ndvi_veg_mean`. NDVI columns are pre-scaled floats — do NOT apply `× 0.0001` again downstream.

**Bug fixes already in script:**
- Use `ee.ImageCollection('USDA/NASS/CDL').filter(...)` instead of `'USDA/NASS/CDL/' + year` (server-side `ee.Number` cannot be JS-concatenated into asset paths).
- Wrap demo image with `ee.Image(...).set('system:description', 'demo')` to reset accumulated metadata.

**Caveats:**
- CDL <year> may not exist yet for the latest year (publishes Jan/Feb of following year). If a year's export fails, drop `END_YEAR` by 1.
- CDL coverage is uneven before 2008 — see "CDL coverage by state and year" above.
- 443 rows per year-file = TIGER 2018 county count for the 5 states. Counties with zero CDL corn pixels emit null NDVI columns.

## gSSURGO state

**Source files:** state `.gdb` files for NE, CO, IA, WI, MO at 10m resolution. CONUS-level skipped (only 30m, >40GB).

**Extraction script:** `scripts/gssurgo_extract.py` — **not yet written**. Phase A.3.

**Plan recap:**
1. Read `.gdb` via geopandas/rasterio.
2. Use Valu1 table (USDA's pre-aggregated MUKEY-level table) — covers `nccpi3corn`, `aws0_100`, `soc0_30`, `rootznemc`, `droughty`, more.
3. Build per-property GeoTIFFs via MUKEY → property remap.
4. Zonal stats over TIGER/Line 2018 county polygons.
5. Reproject before zonal stats (gSSURGO is EPSG:5070 Albers).
6. Output: `scripts/gssurgo_county_features.csv` keyed on `GEOID` only (static across years).

## Drought monitor state

**Raw file:** `phase2/data/drought/drought_USDM-Colorado,Iowa,Missouri,Nebraska,Wisconsin.csv`. Weekly D0–D4 percentages per county.

**Feature script:** `scripts/drought_features.py` — **not yet written**. Phase A.5.

**Plan recap:**
1. Parse weekly CSV into per-(GEOID, week) tidy form.
2. For each forecast date in {Aug 1, Sep 1, Oct 1, EOS}, take the most recent USDM reading strictly before that date (as-of rule).
3. Output severity features per (GEOID, year, forecast_date). Decisions deferred: which D-level severities to expose, whether to also include trailing-N-week means.
4. Output: `scripts/drought_county_features.csv` keyed on `(GEOID, year, forecast_date)`.

## PRISM state

**Pull script:** `scripts/prism_pull.py` — **not yet written**. Phase A.4. Slow download — start early.

**Variables:** daily `tmax, tmin, prcp, srad, vp` for 5 states, 2005–2024.

**Feature script:** `scripts/prism_features.py` — **not yet written**. Phase A.4. Derives:
- GDD base-50°F cap-86°F, accumulated by growth phase and to forecast date
- EDD/KDD hours above 86°F or 90°F
- VPD
- Cumulative precip + dry-spell length
- Solar radiation totals
- All respecting the as-of rule per forecast date

Output: `scripts/prism_county_features.csv` keyed on `(GEOID, year, forecast_date)`.