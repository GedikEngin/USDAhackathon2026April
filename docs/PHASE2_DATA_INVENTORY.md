# Data Inventory — v2 Corn Yield Forecasting

> Living tracker. Update whenever data status changes — pulled, parsed, cleaned, joined, validated, or discovered missing. This is the single source of truth for "do we have what we need to start the next phase."

**Last updated:** 2026-04-25, end of Phase A.3 + A.4 + A.5 (gSSURGO + gridMET + USDM all derived to feature CSVs)

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
| USDA NASS yield (CORN, GRAIN — bu/acre) | NASS QuickStats API | Ground truth | 5 states, county level, 2005–2024 | 🟢 Have | `scripts/nass_corn_5states_2005_2024.csv` (6,837 rows × 16 cols) covers the full brief-required range. 100% coverage on combined-practice columns. ~7% coverage on irrigated (CO + NE only). Non-irrigated columns sparse — derivable from `acres_harvested_all − acres_harvested_irr`. |
| Cropland Data Layer (CDL) | NASS CropScape | Corn-pixel mask | 5 states, 2005–2024 | 🟡 Have via Earth Engine | Currently used inside the GEE NDVI script (`cropland == 1` mask). Standalone local download may be needed if the HLS pipeline is built (Phase D.1) for offline masking. CDL coverage is uneven before 2008 — see "CDL coverage by state and year" below. |
| Weather / climate (daily, gridded) | gridMET | GDD, EDD, VPD, precip accumulation, srad | 5 states, daily, 2005–2024 | 🟢 Have | `scripts/gridmet_county_daily_2005_2024.parquet` + 20 per-year parquets in `data/v2/weather/raw/`. Derived per-cutoff features in `scripts/weather_county_features.csv` (35,440 rows × 14 cols). Phase A.4 done. |

## Strongly recommended additions

| Dataset | Source | Use | Coverage needed | Status | Notes |
|---|---|---|---|---|---|
| US Drought Monitor | NDMC / USDM | Drought severity feature for retrieval embedding | 5 states, weekly, 2005–2024 | 🟢 Have | `phase2/data/drought/drought_USDM-Colorado,Iowa,Missouri,Nebraska,Wisconsin.csv` raw (6,865 weekly state-level rows, 2000-01-10..2026-04-27). Derived per-cutoff features in `scripts/drought_county_features.csv` (27,336 rows × 9 cols, 0 nulls). Phase A.5 done. |

## Already on hand

| Dataset | Source | Use in v2 | Status | Notes |
|---|---|---|---|---|
| NASS combined-practice corn yield 2005–2024 | NASS QuickStats API | Ground truth | 🟢 | `scripts/nass_corn_5states_2005_2024.csv`. Full brief range. 100% coverage on `yield_bu_acre_all`, `production_bu_all`, `acres_harvested_all`, `acres_planted_all`. ~7% coverage on irrigated columns (CO + NE only). Non-irrigated derivable. |
| NASS engineered features | Local `scripts/nass_features.py` | Engineered yield features | 🟢 | `scripts/nass_corn_5states_features.csv` (6,834 rows × 10 cols). Includes `yield_target`, `irrigated_share`, `harvest_ratio`, derived non-irrigated columns. Ready to merge. |
| MODIS NDVI mapping, 5 states | GEE / MODIS MOD13Q1 | Vegetation index features | 🟢 | 21 per-year CSVs in `phase2/data/ndvi/corn_ndvi_5states_<year>.csv` (2004–2024). 443 rows/file (TIGER 2018 county count for the 5 states). Schema: `GEOID, NAME, STATEFP, year, ndvi_peak, ndvi_gs_mean, ndvi_gs_integral, ndvi_silking_mean, ndvi_veg_mean`. **NDVI is pre-scaled (× 0.0001 applied server-side); CSV values are floats in [-0.2, 1.0] — do NOT re-scale.** Counties with no CDL corn pixels emit null NDVI columns (drop or null-impute downstream). **NDVI ≠ HLS** — NDVI is a derived index; raw HLS is still 🔴 Need for Phase D.1 (Prithvi). |
| GEE NDVI script (version-controlled) | Local | Reproducibility | 🟢 | `scripts/ndvi_county_extraction.js`. Header documents bug fixes, output schema, how to extend or re-run. Per-year export task pattern. |
| gSSURGO Valu1 county features, 5 states | `scripts/gssurgo_county_features.py` | Soil features | 🟢 | `scripts/gssurgo_county_features.csv` (443 rows × 13 cols). Keys on `GEOID` only (static across years). Columns: `nccpi3corn, nccpi3all, aws0_100, aws0_150, soc0_30, soc0_100, rootznemc, rootznaws, droughty, pctearthmc, pwsl1pomu`. Ready to merge. |
| gridMET daily weather, 5 states, 2005–2024 | `scripts/gridmet_pull.py` | GDD, EDD, VPD, precip, srad | 🟢 | Raw daily parquets per year in `data/v2/weather/raw/gridmet_county_daily_<year>.parquet` × 20, plus combined `scripts/gridmet_county_daily_2005_2024.parquet`. Cached netcdfs in `data/v2/weather/raw/_gridmet_nc_cache/`. |
| gridMET-derived per-cutoff features | `scripts/weather_features.py` | Climate features | 🟢 | `scripts/weather_county_features.csv` (35,440 rows × 14 cols). Keys: `(GEOID, year, forecast_date)`. Features: `gdd_cum_f50_c86, edd_hours_gt86f, edd_hours_gt90f, vpd_kpa_veg/silk/grain, prcp_cum_mm, dry_spell_max_days, srad_total_veg/silk/grain`. Ready to merge. |
| US Drought Monitor weekly raw | NDMC | Drought source | 🟢 | `phase2/data/drought/drought_USDM-Colorado,Iowa,Missouri,Nebraska,Wisconsin.csv`. **State-level, not county-level** (header is `MapDate,StateAbbreviation,StatisticFormatID,ValidStart,ValidEnd,D0,D1,D2,D3,D4,None,Missing`). Cumulative percent area: D0 ≥ D1 ≥ D2 ≥ D3 ≥ D4 by construction. |
| USDM-derived per-cutoff features | `scripts/drought_features.py` | Drought features | 🟢 | `scripts/drought_county_features.csv` (27,336 rows × 9 cols, 0 nulls). Keys: `(GEOID, year, forecast_date)`. Features: `d0_pct, d1_pct, d2_pct, d3_pct, d4_pct, d2plus`. State readings broadcast to all GEOIDs in the state. As-of join uses `valid_end < forecast_date` (strict). Ready to merge. |
| TIGER 2018 county polygons | US Census | Spatial reduction layer | 🟢 | `phase2/data/tiger/tl_2018_us_county/` (full shapefile) + `phase2/data/tiger/tl_2018_us_county_5states_5070.gpkg` (5-state subset, reprojected to EPSG:5070 for gSSURGO zonal stats). |
| NASS state yields | NASS | State-level validation | 🟢 | |
| Principal Crops Area Planted/Harvested 2023–2025 | NASS | Acres feature | 🟡 | Useful; need to extend back to 2005 if used. |
| Corn Area Planted/Harvested + Yield + Production 2023–2025 | NASS | Acres feature | 🟡 | Same — extend if used. |
| Corn for Silage Area Harvested 2023–2025 | NASS | Disambiguate grain vs silage | 🟢 | Niche but useful. |
| Corn Plant Population per Acre 2021–2025 | NASS | Density feature | 🟢 | Limited history; recent-years feature only. |
| Water applied to corn/grain, 5 states, county, 2018–2023 | USDA Irrigation Survey | Irrigation feature | 🟡 | Sparse temporally (every ~5 years); use as quasi-static county feature. |
| GEOID-keyed NASS yield+irrigation+harvest_ratio table | User-prepared | Master join key | 🟢 | The `GEOID,year,state_alpha,county_name,yield_target,...` table is the join skeleton. Also serves as the GEOID directory for `drought_features.py` broadcast. |
| gSSURGO state .gdb files (NE, CO, IA, WI, MO, 10m) | USDA NRCS | Soil source | 🟢 | `phase2/data/gSSURGO/gSSURGO_<state>/gSSURGO_<state>.gdb` × 5. Used by `gssurgo_county_features.py`. Kept on disk for reproducibility. |
| Mean monthly temperature heatmap + CSV | User-provided | Coarse smoothed temp feature | 🟡 | Superseded by gridMET for modeling. Could be retained as a coarse-feature optional layer. |
| Mean monthly precipitation heatmap + CSV | User-provided | Coarse smoothed precip feature | 🟡 | Same — superseded by gridMET. |
| USDA NASS API access | NASS | Source for NASS pulls | 🟢 | Key in `.env` as `NASS_API_KEY`. `python-dotenv` loaded. |

## Deprioritized (with rationale)

| Dataset | Reason |
|---|---|
| Tornado activity per state 2000–2022 | Tornado damage is highly localized; aggregated to county-year it's near-zero signal for state-level yield. ⚫ NOAA Storm Events is a possible replacement if a broader severe-weather signal is wanted. |
| Hourly temperature | Overkill for county-annual yield; daily Tmin/Tmax + GDD/EDD captures the heat signal. ⚫ |
| Sunrise/sunset hours | Day length is deterministic from latitude+date; modern corn hybrids are largely photoperiod-insensitive. gridMET solar radiation covers the actually-useful signal. ⚫ |
| CO₂ concentration | Spatially uniform across CONUS; absorbed by `year` as a feature (which also captures technology + genetics trends). ⚫ |
| Pixel-level gSSURGO in modeling | Soil features are static across years, so pixel-level only explains between-county variance, not between-year. Aggregate to county-level via Valu1 + zonal stats. ⚫ |
| PRISM (vs. gridMET) | gridMET picked for daily weather. PRISM was the original placeholder; both are equivalent at county-aggregation scale and gridMET was easier to acquire programmatically. ⚫ |

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

- **Final list of features that go into the retrieval embedding** vs. covariates only. Decided in Phase B.1.
- **K** for K-nearest analog retrieval. Default 10; tune in Phase B.5.
- **Cone percentiles to publish** in the API. Default {10, 50, 90}; could expand to {5, 25, 50, 75, 95}. Decided in Phase B / E.
- **State-aggregation method** for percentile rollup. Default planted-acres-weighted mean of each percentile; document caveat. Could revisit in Phase G.
- **Prithvi exact band order and normalization.** Discovered during D.1 implementation, from Prithvi's HuggingFace card.
- **Whether to attempt Phase D.2 (Prithvi fine-tune).** Conditional on D.1 ablation.
- **County coverage filter:** minimum N years of complete data per county. Decide after master table lands and a quick `notna().sum()` pass per county.
- **Whether to enrich drought features.** Current set is intentionally minimal (D0–D4 + d2plus). DSCI / season-cum drought weeks / silking-peak DSCI are easy adds if Phase B/C show the model wants them. Default: ship minimal, add only if signal warrants.

## Decisions resolved since last inventory update

- ✅ **PRISM vs. gridMET** — picked gridMET. (Phase A.4.)
- ✅ **Drought feature aggregation** — picked individual D0–D4 + d2plus, last-reading as-of with strict `<` (no trailing-N-week mean for now). (Phase A.5.)
- ✅ **As-of comparison: ValidEnd vs. ValidStart vs. MapDate** — picked `ValidEnd`. The USDM map for week W is generally not actionable until ValidEnd, so any forecast date strictly after a week's ValidEnd can use that week's reading. (Phase A.5.)

## Schema conventions (locked)

- **Spatial key:** `GEOID` as 5-character zero-padded string (e.g. "19153" for Polk County, IA).
- **Temporal keys:** `year` as 4-digit int, `forecast_date` as one of `{"08-01", "09-01", "10-01", "EOS"}`.
- **State key:** `state_alpha` as 2-char USPS code (IA, CO, WI, MO, NE).
- **Yield units:** `bu/acre` always. No silent conversions.
- **Area units:** acres always. No silent conversions to hectares.
- **All tabular outputs:** parquet (snappy) or CSV. Rasters: GeoTIFF / COG.
- **As-of rule for forecast features:** when constructing features for forecast date `D` in year `Y`, use only data with timestamps strictly before `D`. Enforced in feature-construction layer.

## Storage plan (provisional)

- Tabular data: `scripts/*.csv` and `scripts/*.parquet` for derived/feature outputs; `phase2/data/<source>/` for raw multi-file pulls (already used for `ndvi/`, `drought/`, `gSSURGO/`, `tiger/`); `data/v2/weather/raw/` for the gridMET parquet cache.
- HLS imagery (when pulled): `data/v2/hls/{state}/{year}/` — too big for project knowledge; lives on the Ubuntu box only.
- CDL standalone (if downloaded): `data/v2/cdl/{state}/{year}.tif`.
- Cached features: `scripts/training_master.parquet` (Phase A.6 deliverable).
- gSSURGO aggregated: `scripts/gssurgo_county_features.csv` ✅
- gridMET raw: `data/v2/weather/raw/` ✅; derived: `scripts/weather_county_features.csv` ✅
- USDM raw: `phase2/data/drought/` ✅; derived: `scripts/drought_county_features.csv` ✅

## NASS pipeline state

**Pull script:** `scripts/nass_pull.py` — written, debugged, rate-limit fixes committed (2.0s delay, exponential backoff up to 240s, 4 retries on 403). Run for full 2005–2024 range.

**Output:** `scripts/nass_corn_5states_2005_2024.csv` (6,837 rows × 16 cols).

**Feature engineering:** `scripts/nass_features.py` — written. Output: `scripts/nass_corn_5states_features.csv` (6,834 rows × 10 cols). Includes `yield_target`, `irrigated_share`, `harvest_ratio`, derived non-irrigated columns.

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
1. Pin `prodn_practice_desc` to avoid the multi-practice fan-out bug.
2. Filter rows where `county_name` starts with "OTHER" or `county_ansi` is empty.
3. Strip commas before `pd.to_numeric`.

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

## gSSURGO state — Phase A.3 ✅

**Source files:** state `.gdb` files for NE, CO, IA, WI, MO at 10m resolution in `phase2/data/gSSURGO/gSSURGO_<state>/`. CONUS-level skipped (only 30m, >40GB).

**Extraction script:** `scripts/gssurgo_county_features.py` — written, run.

**Method:**
1. Read each `.gdb` via geopandas/rasterio.
2. Pull the Valu1 table (USDA's pre-aggregated MUKEY-level table).
3. Build per-property GeoTIFFs by remapping MUKEY → property values.
4. Reproject to match TIGER 2018 county polygons (gSSURGO is EPSG:5070 Albers).
5. Run zonal statistics over county polygons.

**Output:** `scripts/gssurgo_county_features.csv` keyed on `GEOID` (443 rows × 13 cols, static across years).

**Columns:** `GEOID, state_alpha, nccpi3corn, nccpi3all, aws0_100, aws0_150, soc0_30, soc0_100, rootznemc, rootznaws, droughty, pctearthmc, pwsl1pomu`.

## gridMET / weather state — Phase A.4 ✅

**Pull script:** `scripts/gridmet_pull.py` — written, run for 2005–2024.

**Why gridMET (not PRISM):** picked gridMET when the script was written. Both are 4km daily surfaces over CONUS; both work for the heat/precip/srad signals we care about at county-aggregation scale. gridMET was easier to acquire programmatically.

**Variables:** daily `tmax, tmin, prcp, srad, vp` for the 5 states.

**Outputs:**
- `data/v2/weather/raw/gridmet_county_daily_<year>.parquet` × 20 (one per year, 2005–2024).
- `data/v2/weather/raw/_gridmet_nc_cache/` — cached source netcdfs (idempotent re-runs).
- `scripts/gridmet_county_daily_2005_2024.parquet` — combined daily, all years (input to `weather_features.py`).

**Feature script:** `scripts/weather_features.py` — written, run.

**Derives per `(GEOID, year, forecast_date)`:**
- `gdd_cum_f50_c86` — cumulative GDD May 1 → cutoff. Fahrenheit base 50 / cap 86, both endpoints capped (McMaster & Wilhelm / NDAWN convention; not the raw `tavg − 50` variant).
- `edd_hours_gt86f / edd_hours_gt90f` — degree-hours above 86°F / 90°F via single-sine hourly interpolation (Allen 1976 / Baskerville-Emin 1969). Captures sub-daily heat exposure.
- `vpd_kpa_veg / vpd_kpa_silk / vpd_kpa_grain` — VPD averaged over each phase window (DOY 152–195 / 196–227 / 228–273), clipped to cutoff.
- `prcp_cum_mm` — cumulative precipitation May 1 → cutoff.
- `dry_spell_max_days` — longest run of `<2 mm/day`.
- `srad_total_veg / srad_total_silk / srad_total_grain` — solar radiation totals per phase, MJ/m².

**As-of rule:** the function `build_features_for_cutoff(df, year, cutoff_date)` slices the daily df at the very top (`date <= cutoff_date`). Nothing downstream of that slice can see post-cutoff data. Phase windows are clipped to `cutoff_doy`.

**Output:** `scripts/weather_county_features.csv` (35,440 rows × 14 cols). Keys: `GEOID, year, forecast_date`.

## Drought monitor state — Phase A.5 ✅

**Raw file:** `phase2/data/drought/drought_USDM-Colorado,Iowa,Missouri,Nebraska,Wisconsin.csv`. Weekly readings for 5 states, 2000-01-10 through 2026-04-27 (6,865 rows).

**Important:** the source CSV is **state-level**, not county-level. Header is `MapDate,StateAbbreviation,StatisticFormatID,ValidStart,ValidEnd,D0,D1,D2,D3,D4,None,Missing` — there's no FIPS column. State readings are broadcast to every GEOID in the state at feature-derivation time. The earlier planning notes that said "weekly D0–D4 percentages per county" were aspirational.

**USDM percentages are cumulative:** D0 ≥ D1 ≥ D2 ≥ D3 ≥ D4 by construction. Each column reports % of state at that level *or worse*. So "D2+" is identical to D2 — `d2plus` exists in the output as a stable, descriptively-named alias for `d2_pct`, so downstream code doesn't have to remember the cumulative convention.

**Feature script:** `scripts/drought_features.py` — written, run.

**Method:**
1. Read the USDM CSV; rename `StateAbbreviation` → `state_alpha`; coerce `ValidEnd` → `valid_end` (datetime.date).
2. Read `nass_corn_5states_features.csv` as the GEOID directory (provides every `(GEOID, state_alpha, year)` triple in the modeling universe).
3. Pre-slice USDM by state for speed.
4. For each `(state, year, forecast_date)`, find the most recent reading whose `valid_end < forecast_date` (strict). 5 states × 20 years × 4 dates = 400 state-level rows derived.
5. Build the full `(GEOID, year, forecast_date)` skeleton (cartesian product of GEOID directory and forecast dates), left-join state-level features. Guarantees every row has all three keys populated; missing readings become NaN feature columns.

**Output:** `scripts/drought_county_features.csv` (27,336 rows × 9 cols). Keys: `GEOID, year, forecast_date`. Features: `d0_pct, d1_pct, d2_pct, d3_pct, d4_pct, d2plus`.

**Validation:**
- 0 nulls on every column (USDM goes back to 2000; even the NDVI-driven 2004 backfill has full coverage).
- 0 monotonicity violations (D0 ≥ D1 ≥ D2 ≥ D3 ≥ D4) across all 420 state-level cells.
- Spot-check: CO 2020-08-01 picks the 2020-07-27 reading (the closest week ending strictly before 2020-08-01). The 2020-08-03 reading is correctly excluded — confirms strict-`<` semantics.
