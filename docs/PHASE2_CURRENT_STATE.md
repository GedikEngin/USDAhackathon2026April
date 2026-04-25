# Phase 2 — Current State

> Overwritten at end of every v2 phase. Tells the next chat what actually exists in the v2 portion of the repo right now. Reference file paths and describe behavior; do not paste code here.

**Project:** USDA Hackathon 2026 April — Corn Yield Forecasting
**Repo:** `~/dev/USDAhackathon2026April`
**Last updated:** 2026-04-25, end of Phase A.3 + A.4 + A.5 (gSSURGO + gridMET + USDM all landed)

---

## v2 status: Phase A.1 through A.5 complete; A.6 (merge_all) is the only remaining Phase A item

Five data pipelines have output sitting on disk and joinable on the canonical keys:
- NASS yield 2005–2024 (full coverage on combined-practice columns) — `(GEOID, year)`
- MODIS NDVI 2004–2024 (corn-masked via CDL, county-aggregated; 21 per-year CSVs) — `(GEOID, year)`
- gSSURGO Valu1 features for all 5 states, county-aggregated — `(GEOID,)` (static across years)
- gridMET daily weather 2005–2024, derived per-cutoff features (GDD, EDD, VPD, precip, srad) — `(GEOID, year, forecast_date)`
- US Drought Monitor weekly readings, derived per-cutoff features (D0–D4, d2plus) — `(GEOID, year, forecast_date)`

`nass_features.py`, `gssurgo_county_features.py`, `weather_features.py`, and `drought_features.py` are all written and run. `gridmet_pull.py` is written and ran for the full 2005–2024 range. `ndvi_county_extraction.js` is version-controlled. No modeling, no backend endpoints, no agent tools yet. The v1 land-use system is shipped and untouched alongside.

## Project goal (locked)

Build an ML pipeline that predicts corn-for-grain yield (bu/acre) at the **county** level across **5 US states** at four fixed in-season forecast dates (Aug 1, Sep 1, Oct 1, end-of-season), aggregates to state, and produces a calibrated cone of uncertainty using **analog-year retrieval**. A Claude Haiku agent narrates the forecast.

- **Target states:** Colorado (CO), Iowa (IA), Missouri (MO), Nebraska (NE), Wisconsin (WI)
- **Time range:** **2005–2024** per the brief (20 years; 2024 holdout, 2023 validation, 2005–2022 train). NDVI data extends back to 2004 incidentally — kept but not required.
- **Spatial unit:** US county, joined via 5-digit GEOID (state FIPS + county FIPS, both zero-padded)
- **Target variable:** `yield_bu_acre_all` (combined-practice corn grain yield, bu/acre)

See `PHASE2_PROJECT_PLAN.md` for the full vision and `PHASE2_PHASE_PLAN.md` for the phase breakdown with go/no-go gates.

## What exists in the repo today

### v1 (untouched, fully shipped)

The full v1 land-use & GHG analysis system is in place. See the v1 `CURRENT_STATE.md` for canonical detail. v2 reuses v1's FastAPI shell, frontend chrome, and Claude agent infrastructure but does not modify them.

### v2 — written and run

- **`scripts/nass_pull.py`** — written, run for 2005–2024. Hits NASS QuickStats API for the 5 states, county-level corn yield/production/area at 3 practice levels (combined / irrigated / non-irrigated). 100% coverage on combined-practice columns. ~7% coverage on irrigated (CO + NE only). Non-irrigated columns sparse — derivable downstream from `acres_harvested_all − acres_harvested_irr`. Rate-limit fixes (2.0s delay, exponential backoff up to 240s, 4 retries) committed.

- **`scripts/nass_corn_5states_2005_2024.csv`** — output of the full 2005–2024 pull. The single canonical NASS CSV. 6,837 rows × 16 cols.

- **`scripts/nass_features.py`** — written. Local feature engineering on the NASS CSV. No API calls. Produces `scripts/nass_corn_5states_features.csv` with `yield_target`, `irrigated_share`, `harvest_ratio`, derived non-irrigated columns.

- **`scripts/nass_corn_5states_features.csv`** — output of `nass_features.py`. 6,834 rows × 10 cols. Ready to merge.

- **`scripts/ndvi_county_extraction.js`** — version-controlled GEE script. Pulls MODIS NDVI 2004–2024 for the 5 states, masked to corn pixels via USDA CDL, reduced to county-year. Header documents bug fixes (CDL `ImageCollection.filter` pattern, demo-image metadata reset), output schema, and how to extend or re-run. Script enqueues one Export task per year.

- **`phase2/data/ndvi/corn_ndvi_5states_<year>.csv`** (21 files, 2004–2024) — output of the GEE script. 443 rows per file (TIGER/2018 county polygons for the 5 states). Schema: `GEOID, NAME, STATEFP, year, ndvi_peak, ndvi_gs_mean, ndvi_gs_integral, ndvi_silking_mean, ndvi_veg_mean`. **NDVI values are pre-scaled (× 0.0001 applied server-side); CSV values are floats in roughly [-0.2, 1.0] — do NOT re-apply the scale factor downstream.** Counties with zero CDL corn pixels in a given year emit null NDVI columns; downstream merge must tolerate this.

- **`scripts/gssurgo_county_features.py`** — written, run for all 5 states. Reads each state `.gdb` from `phase2/data/gSSURGO/`, pulls the Valu1 table, builds per-property GeoTIFFs by remapping MUKEY → property values, runs zonal statistics over TIGER 2018 county polygons (reprojected from 5070 Albers to match counties).

- **`scripts/gssurgo_county_features.csv`** — output of gSSURGO extraction. 443 rows × 13 cols. Keyed on `GEOID` only (static across years; broadcast at merge time). Columns: `GEOID, state_alpha, nccpi3corn, nccpi3all, aws0_100, aws0_150, soc0_30, soc0_100, rootznemc, rootznaws, droughty, pctearthmc, pwsl1pomu`. Ready to merge.

- **`scripts/gridmet_pull.py`** — written, run for 2005–2024. Daily weather pull from gridMET (chosen over PRISM — see decisions log). Variables: `tmax, tmin, prcp, srad, vp` daily at the gridMET 4km grid, county-aggregated to TIGER 2018 polygons. Produces one parquet per year in `data/v2/weather/raw/gridmet_county_daily_<year>.parquet` plus a combined `scripts/gridmet_county_daily_2005_2024.parquet`.

- **`data/v2/weather/raw/gridmet_county_daily_<year>.parquet`** (20 files, 2005–2024) — raw daily county-aggregated weather output of `gridmet_pull.py`. Cached netcdf intermediates live in `_gridmet_nc_cache/`.

- **`scripts/gridmet_county_daily_2005_2024.parquet`** — combined daily weather, all years. Input to `weather_features.py`.

- **`scripts/weather_features.py`** — written, run. Derives per-`(GEOID, year, forecast_date)` features from the daily parquet. As-of slice happens once at the top of `build_features_for_cutoff`; nothing downstream of that slice can see post-cutoff data. Phase windows (vegetative DOY 152–195, silking DOY 196–227, grain DOY 228–273) are clipped to the cutoff. GDD uses Fahrenheit base 50 / cap 86 (corn standard, McMaster & Wilhelm). EDD uses single-sine hourly interpolation (Allen 1976 / Baskerville-Emin) for degree-hours above 86°F and 90°F.

- **`scripts/weather_county_features.csv`** — output of `weather_features.py`. 35,440 rows × 14 cols (= 443 GEOIDs × 20 years × 4 forecast dates, minus a handful of missing combos). Columns: `GEOID, year, forecast_date, gdd_cum_f50_c86, edd_hours_gt86f, edd_hours_gt90f, vpd_kpa_veg, vpd_kpa_silk, vpd_kpa_grain, prcp_cum_mm, dry_spell_max_days, srad_total_veg, srad_total_silk, srad_total_grain`. Ready to merge.

- **`phase2/data/drought/drought_USDM-Colorado,Iowa,Missouri,Nebraska,Wisconsin.csv`** — raw US Drought Monitor weekly D0–D4 percentages, 5 states, 2000-01-10 through 2026-04-27. State-level (StateAbbreviation column; no county FIPS in source — see `drought_features.py` notes). 6,865 weekly rows.

- **`scripts/drought_features.py`** — written, run. Reads the USDM CSV, broadcasts state-level readings to every GEOID in the state via `nass_corn_5states_features.csv` as the GEOID directory, applies the as-of rule (last reading with `valid_end < forecast_date` — strictly before, never on or after). Exposes individual `d0_pct, d1_pct, d2_pct, d3_pct, d4_pct` plus `d2plus` (alias for `d2_pct`, exposed under a stable name because USDM's percentages are cumulative so "D2 or worse" is identical to D2 itself).

- **`scripts/drought_county_features.csv`** — output of `drought_features.py`. 27,336 rows × 9 cols (= 6,834 (GEOID, year) × 4 forecast dates). 0 nulls on every column, 0 monotonicity violations on the state-level intermediate (D0 ≥ D1 ≥ D2 ≥ D3 ≥ D4 across all 420 (state, year, forecast_date) cells). Ready to merge.

### v2 — to write

- `scripts/merge_all.py` — outer-join all per-`(GEOID, year)` tables and per-`(GEOID, year, forecast_date)` tables. Output: single `training_master.parquet`. **Phase A.6 — only remaining Phase A item.**
- `PHASE2_DATA_DICTIONARY.md` — column-by-column doc for `training_master.parquet`. **Phase A.7.**
- HLS acquisition script (no name yet) — required for Phase D.1 (Prithvi). Not yet started; lower priority than the engineered baseline. See "Open architectural questions" below.

## Data status (summary; live tracker is `PHASE2_DATA_INVENTORY.md`)

### 🟢 Have & ready
- NASS combined-practice corn yield/production/area, 5 states, **2005–2024**, county level (`scripts/nass_corn_5states_2005_2024.csv`)
- NASS engineered features (`scripts/nass_corn_5states_features.csv`)
- MODIS NDVI features per (GEOID, year), **2004–2024** (`phase2/data/ndvi/corn_ndvi_5states_<year>.csv` × 21)
- gSSURGO Valu1 county features, all 5 states (`scripts/gssurgo_county_features.csv`)
- gridMET daily weather raw, 2005–2024 (20 per-year parquets + combined parquet)
- gridMET-derived per-cutoff weather features (`scripts/weather_county_features.csv`)
- US Drought Monitor weekly raw CSV (5 states, 2000–2026)
- USDM-derived per-cutoff drought features (`scripts/drought_county_features.csv`)
- Mean monthly temperature heatmap + CSV (coarse smoothed feature only — superseded by gridMET for modeling)
- Mean monthly precipitation heatmap + CSV (coarse smoothed feature only — superseded by gridMET for modeling)
- gSSURGO state .gdb files downloaded for all 5 states, 10m resolution (kept on disk for reproducibility)
- USDA NASS API access (key in `.env`)
- GEE script version-controlled (`scripts/ndvi_county_extraction.js`)

### 🔵 In flight
*(none currently — all queued pulls have landed)*

### 🔴 Need
- **`scripts/merge_all.py`** — final outer-join of all five feature tables into `training_master.parquet`. Phase A.6.
- **CDL standalone download** — currently used only inside the Earth Engine script for masking. Want a local copy if HLS pipeline is built (Phase D.1).
- **HLS imagery** — required for Phase D.1 (Prithvi as feature extractor). 2005–2013 is Landsat-only; Sentinel-2 component starts 2015. Not yet started.
- **NAIP imagery** — tertiary; only if a phase explicitly calls for sub-meter visual context.
- **Prithvi model weights** — download in Phase D.1.

### ⚫ Deprioritized (with rationale)
- Hourly temperature — overkill for county-annual yield; daily Tmin/Tmax + GDD is sufficient.
- Sunrise/sunset hours — deterministic from latitude+date and modern corn hybrids are largely photoperiod-insensitive. gridMET solar radiation covers the actually-useful signal.
- CO₂ concentration — spatially uniform across CONUS; absorbed by `year` as a feature (which also captures technology/genetics trends).
- Tornado activity — highly localized damage; near-zero signal at county-year aggregation. Could be replaced by NOAA Storm Events for broader severe-weather signal, but optional.
- PRISM (vs. gridMET) — gridMET picked for daily weather. PRISM remains a viable alternative if gridMET ever shows quality issues, but no current need.

## NDVI coverage by state and year (CDL-driven)

The MOD13Q1 source is available 2000–present, but corn-masking depends on USDA CDL coverage, which started rolling out state-by-state in the early 2000s. Empirical coverage from the actual exports (counties with non-null `ndvi_peak`):

| State | FIPS | Counties (TIGER 2018) | 2004–2005 | 2006–2007 | 2008+ |
|---|---|---|---|---|---|
| Iowa | 19 | 99 | 99 ✅ | 99 ✅ | 99 ✅ |
| Nebraska | 31 | 93 | 93 ✅ | 93 ✅ | 93 ✅ |
| Wisconsin | 55 | 72 | 72 ✅ | 72 ✅ | 72 ✅ |
| Missouri | 29 | 115 | ~28 ⚠️ | ~110 ✅ | 110–113 ✅ |
| Colorado | 08 | 64 | ~3–4 ❌ | ~40 ⚠️ | 37–38 ✅ |

**Implications:**
- Brief specifies 2005–2024 — fully usable for IA/NE/WI all years; MO from 2006; CO from 2008.
- **Strong block: 2008–2024** (~17 years × ~410 counties ≈ ~7,000 county-years).
- 2005–2007 is usable but heterogeneous; CO essentially no NDVI 2005–2007.
- The ~30 counties always missing post-2008 are real corn-absent counties (St. Louis City, Denver, CO mountain counties, urban WI). These will drop out of training because they have no NASS yield either.
- **Phase B retrieval is per-county**, so coverage variation across states does not contaminate analog matching — each county draws analogs from its own past only.

## Recommended full feature set (target schema for `training_master.parquet`)

This is the design target. Italicized rows are the columns currently on disk in the per-source CSVs and ready for `merge_all.py`.

### Soil (from gSSURGO Valu1 table, county-aggregated via zonal stats) — ✅ all on disk
- *`nccpi3corn`* — National Commodity Crop Productivity Index for corn
- *`nccpi3all`* — NCCPI averaged across all crops
- *`aws0_100`* — available water storage 0–100 cm, mm
- *`aws0_150`* — available water storage 0–150 cm, mm
- *`soc0_30`* — soil organic carbon 0–30 cm
- *`soc0_100`* — soil organic carbon 0–100 cm
- *`rootznemc`* — root zone effective moisture capacity
- *`rootznaws`* — root zone available water storage
- *`droughty`* — drought-vulnerable soil flag
- *`pctearthmc`* — percent Earth surface (excludes water, urban)
- *`pwsl1pomu`* — potential wetland soils

### Climate (gridMET daily, 4km, county-aggregated) — ✅ all on disk
- *`gdd_cum_f50_c86`* — cumulative GDD from May 1, base 50°F cap 86°F
- *`edd_hours_gt86f`* — degree-hours above 86°F (sinusoidal interpolation, season-cum)
- *`edd_hours_gt90f`* — degree-hours above 90°F
- *`vpd_kpa_veg / vpd_kpa_silk / vpd_kpa_grain`* — VPD averaged over each phase window
- *`prcp_cum_mm`* — cumulative precipitation May 1 → cutoff
- *`dry_spell_max_days`* — longest run of <2mm/day
- *`srad_total_veg / srad_total_silk / srad_total_grain`* — solar radiation totals per phase

### Drought (USDM weekly, state-level, broadcast to counties) — ✅ all on disk
- *`d0_pct, d1_pct, d2_pct, d3_pct, d4_pct`* — cumulative percent area at each severity
- *`d2plus`* — alias for `d2_pct` (severe-or-worse), stable name for retrieval embedding

### Remote sensing (MODIS via Earth Engine; HLS as a Phase D.1 layer) — ✅ all on disk for MODIS
- *`ndvi_peak / ndvi_gs_mean / ndvi_gs_integral / ndvi_silking_mean / ndvi_veg_mean`*
- Optional later: EVI, SIF, LAI
- Phase D.1: Prithvi embedding from raw HLS chips (separate from MODIS NDVI)

### Management (from NASS) — ✅ core columns on disk
- *`yield_target, irrigated_share, harvest_ratio`* (engineered in `nass_features.py`)
- *`acres_harvested_all, acres_planted_all, yield_bu_acre_irr`*
- Optional later: planting date / harvest date (NASS Crop Progress), hybrid relative maturity, seeding rate (have for 2021–2025), tillage, N/P/K (USDA ARMS), water applied (have 2018–2023)

### Topography (from USGS DEM) — ⚫ deprioritized for v2 baseline
Soil features already capture much of what topography would contribute (drainage, water holding). Add later if Phase B/C show a hole.

## Pipeline structure (best-practice separation)

For every external data source: **separate the pull from the feature engineering.** API calls and downloads are slow and rate-limited; features iterate fast. Splitting them prevents re-hitting external services on every feature change.

- `scripts/nass_pull.py` → raw `scripts/nass_corn_5states_2005_2024.csv` ✅
- `scripts/nass_features.py` → derived `scripts/nass_corn_5states_features.csv` ✅
- `scripts/ndvi_county_extraction.js` (GEE) → raw `phase2/data/ndvi/corn_ndvi_5states_<year>.csv` × 21 ✅
- `scripts/gssurgo_county_features.py` → `scripts/gssurgo_county_features.csv` ✅
- `scripts/gridmet_pull.py` → raw `data/v2/weather/raw/gridmet_county_daily_<year>.parquet` × 20 + combined parquet ✅
- `scripts/weather_features.py` → `scripts/weather_county_features.csv` ✅
- `scripts/drought_features.py` → `scripts/drought_county_features.csv` ✅
- `scripts/merge_all.py` → `scripts/training_master.parquet` ⬜

## Decisions made (with rationale)

These are now logged in `PHASE2_DECISIONS_LOG.md` as the canonical record. Restated here for orientation:

1. **County-level aggregation, not pixel-level** — matches NASS reporting unit.
2. **Server-side aggregation in Earth Engine** — NDVI CSVs already at county-year.
3. **Separate raw-pull script from feature-engineering script** — see "Pipeline structure".
4. **Use the Valu1 table for soil aggregation** — USDA's official pre-aggregation, covers 90% of common ML use cases including NCCPI.
5. **Skip hourly weather, day length, and CO₂.**
6. **Use combined-practice yield as the target.**
7. **TIGER/Line 2018 county boundaries** for spatial reduction.
8. **Time range 2005–2024** per the brief. Train 2005–2022, val 2023, holdout 2024.
9. **MODIS NDVI is the primary remote-sensing feature for the engineered baseline.** Raw HLS deferred to Phase D.1.
10. **gridMET (not PRISM) is the daily weather source** — decided when `gridmet_pull.py` was written. gridMET is a standardized 4km daily surface for 1979+, easier to acquire programmatically than PRISM, and for the heat/precip-stress signals we care about there's no quality difference at the county-aggregation scale.
11. **NDVI per-year file structure:** GEE script enqueues one Export task per year. Cleaner failure mode.
12. **NDVI is pre-scaled in the CSV:** the GEE script applies `× 0.0001` server-side; downstream code must NOT re-apply.
13. **GDD uses Fahrenheit base 50 / cap 86 with both endpoints capped** (McMaster & Wilhelm / NDAWN convention, not the raw `tavg − 50` variant).
14. **EDD/KDD uses single-sine hourly interpolation** (Allen 1976 / Baskerville-Emin 1969) for degree-hours above 86°F and 90°F. Captures sub-daily heat exposure that simple `max(0, tmax − 86)` misses.
15. **USDM is published at state level only.** Despite earlier planning notes saying "per county," the actual source CSV has only `StateAbbreviation`. State readings are broadcast to every GEOID in that state.
16. **USDM as-of join uses `valid_end < forecast_date` (strict).** Prevents same-week leakage even when the USDM map's validity span brackets the forecast date.
17. **Drought feature set is minimal: D0–D4 + d2plus.** No DSCI, no season-cum drought weeks, no silking-peak DSCI in this iteration. Can be added later if Phase B shows the model wants more drought signal than the cumulative percentages provide.

## Important data-quality fixes already discovered (NASS)

1. **Multi-practice fan-out bug:** without specifying `prodn_practice_desc`, NASS returns 3 rows per county-year (combined + irrigated + non-irrigated). Merging 4 such tables produced 81 rows per county-year (722,252 total instead of ~3,000). **Fix: always pin `prodn_practice_desc`.**
2. **"OTHER (COMBINED) COUNTIES" placeholder rows:** NASS rolls up suppressed-data counties into a fake "county" with GEOID ending in 000. **Fix: filter out rows where `county_name` starts with "OTHER" or `county_ansi` is empty.**
3. **Comma-separated values:** Large numbers come in as strings with commas (e.g., `"14,200,000"`). **Fix: strip commas before `pd.to_numeric`.**

## Earth Engine technical notes

- Used MODIS MOD13Q1 (250m, 16-day composite) as the NDVI source.
- Masked to corn-only pixels using USDA CDL `cropland == 1`.
- All aggregation done server-side in Earth Engine; output CSVs are at county-year level.
- **Bug fixed:** original script tried to concatenate `'USDA/NASS/CDL/' + year` where `year` was a server-side `ee.Number`, producing invalid asset paths. **Fix:** use `ee.ImageCollection('USDA/NASS/CDL').filter(...)` instead.
- **Bug fixed:** map description error fixed by wrapping demo image with `ee.Image(...).set('system:description', 'demo')` to reset accumulated metadata.
- MODIS NDVI scale factor 0.0001 (raw 0–10000) — applied server-side in the script; CSV values are already floats.
- 443 rows per year-file = TIGER 2018 county count for the 5 states. Counties with no CDL corn pixels emit null NDVI columns — downstream tolerates.

## Key technical notes for future sessions

### Join key convention
All datasets join on `GEOID` (5-digit string: state FIPS zero-padded to 2 + county FIPS zero-padded to 3) and `year` (integer). For per-forecast-date tables, the additional key is `forecast_date` ∈ `{"08-01", "09-01", "10-01", "EOS"}`.

### Data quirks to remember
- NASS suppresses data for low-corn counties → expect Colorado mountain counties to drop out, especially in extension years.
- NASS rate limits aggressively at sub-1s request intervals (Azure App Gateway).
- gSSURGO uses Albers Equal Area projection (EPSG:5070); reproject before zonal stats.
- MODIS NDVI scale factor 0.0001 — already applied in the existing CSVs.
- CDL value 1 = corn (used for masking in Earth Engine).
- Earth Engine `ee.Number` cannot be JS-concatenated into asset paths; use `ImageCollection.filter()` instead.
- CDL coverage is uneven before 2008 — see "NDVI coverage by state and year" table above.
- TIGER 2018 includes independent cities (e.g., St. Louis 29510) — they appear in row counts but always emit null NDVI.
- USDM source CSV is **state-level**, not county-level. Source has only `StateAbbreviation`. `drought_features.py` broadcasts to GEOIDs.
- USDM percentages are cumulative: D0 ≥ D1 ≥ D2 ≥ D3 ≥ D4. "D2+" is identical to D2 itself.
- Windows download artifacts: files brought across via the chat carry `:Zone.Identifier` siblings. Clean with `find . -name '*:Zone.Identifier' -delete`.

### Inherited from v1 (still apply for shared infrastructure)
- `conda activate landuse2` per session.
- `--app-dir .` required for uvicorn from repo root.
- `python-multipart` is a separate `pip install` from `fastapi`.
- `pkill -f uvicorn` before relaunch (background `&` doesn't kill cleanly).
- `.env` is gitignored.
- Frontend-only changes don't require a uvicorn restart; Python changes do.

### User coding-style preferences (carried from prior sessions)
- Minimal abstractions, explicit logic, flat structure.
- Inline comments.
- Avoid heavy frameworks unless necessary.
- Languages: C++, Python, Java, JavaScript familiar.

## Repo layout (current)

```
USDAhackathon2026April/
├── docs/
│   ├── PHASE2_PROJECT_PLAN.md                  # v2 vision + locked decisions
│   ├── PHASE2_PHASE_PLAN.md                    # v2 phase breakdown A–G
│   ├── PHASE2_CURRENT_STATE.md                 # ← THIS FILE
│   ├── PHASE2_DATA_INVENTORY.md                # v2 live data tracker
│   ├── PHASE2_DECISIONS_LOG.md                 # v2 append-only decisions
│   └── (v1 docs: CURRENT_STATE.md, etc.)
├── data/
│   └── v2/
│       ├── tiger/                              # county polygons used by zonal stats
│       └── weather/
│           └── raw/
│               ├── _gridmet_nc_cache/          # cached netcdfs
│               └── gridmet_county_daily_<year>.parquet × 20    ✅
├── phase2/
│   └── data/
│       ├── drought/
│       │   └── drought_USDM-Colorado,Iowa,Missouri,Nebraska,Wisconsin.csv  ✅
│       ├── gSSURGO/
│       │   └── gSSURGO_<state>/gSSURGO_<state>.gdb × 5         ✅
│       ├── ndvi/
│       │   └── corn_ndvi_5states_<year>.csv × 21               ✅
│       └── tiger/
│           ├── tl_2018_us_county/                               ✅
│           └── tl_2018_us_county_5states_5070.gpkg              ✅
├── scripts/                                    # ← Data pipeline lives here
│   ├── (v1) dataset.py, train.py, infer.py, emissions.py, etc.
│   ├── nass_pull.py                            ✅ written, run for 2005–2024
│   ├── nass_features.py                        ✅ written
│   ├── nass_corn_5states_2005_2024.csv         ✅ canonical NASS output
│   ├── nass_corn_5states_features.csv          ✅ engineered features
│   ├── ndvi_county_extraction.js               ✅ version-controlled (per-year exports)
│   ├── gssurgo_county_features.py              ✅ written, run for all 5 states (Phase A.3)
│   ├── gssurgo_county_features.csv             ✅ output (443 rows × 13 cols)
│   ├── gridmet_pull.py                         ✅ written, run for 2005–2024 (Phase A.4)
│   ├── gridmet_county_daily_2005_2024.parquet  ✅ combined daily weather
│   ├── weather_features.py                     ✅ written, run (Phase A.4)
│   ├── weather_county_features.csv             ✅ output (35,440 rows × 14 cols)
│   ├── drought_features.py                     ✅ written, run (Phase A.5)
│   ├── drought_county_features.csv             ✅ output (27,336 rows × 9 cols)
│   └── merge_all.py                            ⬜ final join → training_master.parquet (Phase A.6)
├── agent/                                       # v1 agent package (untouched)
├── backend/                                     # v1 FastAPI app (untouched)
├── frontend/                                    # v1 frontend (untouched)
├── inference_outputs/                           # v1 segmentation outputs
├── model/                                       # v1 SegFormer checkpoint
├── smoke_agent.py                               # v1 smoke test
└── smoke_tools.py                               # v1 smoke test
```

## What v2 will reuse from v1 (no modification)

- `backend/main.py` — FastAPI app, lifespan pattern, env loading. v2 adds new endpoints alongside.
- `backend/models.py` — Pydantic conventions; v2 adds new models.
- `frontend/index.html` + `frontend/app.js` — chrome, race-safe state pattern, status pills, markdown renderer. v2 adds a new view.
- `agent/base.py`, `agent/claude.py`, `agent/tools.py` — Claude agent infrastructure. v2 adds new tools alongside.
- `.env` and `python-dotenv` loading pattern.

## What v2 does NOT use from v1

- `backend/inference.py` — SegFormer-specific.
- `scripts/dataset.py`, `scripts/train.py`, `scripts/infer.py`, `scripts/emissions.py`, `scripts/agent_repl.py` — v1-specific.
- v1 agent tools (`get_land_breakdown`, etc.) — left in place; v2 tools live alongside.
- `model/segformer-b1-run1/` checkpoint — different problem.

## Known issues / open architectural questions

- **HLS pull strategy not yet designed.** TB-scale download for 5 states × 20 years × growing-season scenes. Need to settle storage location, idempotent download orchestration, and whether to pull every available scene or only the ~12 scenes leading up to each forecast date. **2005–2013 is Landsat-only HLS** (Sentinel-2 component begins 2015); cadence is lower in early years. Decision deferred until Phase B baseline ships and we know whether Phase D.1 is worth the data-engineering cost.
- **NASS non-irrigated columns** — sparse in the pull. Derivable from `acres_harvested_all − acres_harvested_irr` for fields where irrigated coverage exists; for fields where irrigated coverage is also sparse, this is a known gap. Re-running `nass_pull.py` with the rate-limit fixes for non-irrigated specifically is optional, low-priority.
- **NDVI 2005–2007 is patchy for CO and partially for MO.** Planned mitigation is per-county analog retrieval (a CO county draws analogs from its own history only, so coverage variation across states does not contaminate matching). If Phase B reveals problems, fall-back is to filter to ≥N years of complete data per county.
- **Drought feature parsimony.** Current set is intentionally minimal (D0–D4 + d2plus). DSCI / season-cumulative drought weeks / silking peak DSCI are easy adds if Phase B / C show the model wants richer drought signal. Current decision: ship the minimal set, see if it carries weight, add if not.

## Hackathon readiness checklist

- [x] Planning docs (`PHASE2_PROJECT_PLAN.md`, `PHASE2_PHASE_PLAN.md`, `PHASE2_DATA_INVENTORY.md`, `PHASE2_DECISIONS_LOG.md`) drafted
- [x] NASS API key + script, full pull done for 2005–2024 (full coverage on combined-practice columns)
- [x] `scripts/nass_features.py` written and `nass_corn_5states_features.csv` produced
- [x] GEE NDVI export complete for 2004–2024 (21 per-year CSVs in `phase2/data/ndvi/`)
- [x] `ndvi_county_extraction.js` saved into `scripts/` for version control
- [x] US Drought Monitor raw CSV pulled for 5 states
- [x] `scripts/gssurgo_county_features.py` written; per-county soil features CSV produced (Phase A.3)
- [x] `scripts/gridmet_pull.py` + `scripts/weather_features.py` written; per-county weather features produced 2005–2024 (Phase A.4)
- [x] `scripts/drought_features.py` written; per-(GEOID, year, forecast_date) drought features produced (Phase A.5)
- [ ] `scripts/merge_all.py` written; `training_master.parquet` built (Phase A.6)
- [ ] `PHASE2_DATA_DICTIONARY.md` written (Phase A.7)
- [ ] Phase A definition-of-done met (master table loads, no surprise nulls, spot-check passes)
- [ ] Phase B analog-year retrieval baseline shipped + Phase B gate passed
- [ ] Phase C XGBoost point-estimate model shipped + Phase C gate passed
- [ ] Phase D.1 Prithvi frozen-feature integration + Phase D gate decision
- [ ] Phase D.2 (conditional) Prithvi fine-tune
- [ ] Phase E `/forecast/{state}` endpoint + frontend forecast view
- [ ] Phase F forecast narration agent with 4 tools
- [ ] Phase G holdout evaluation, ablation table, presentation deck

## Immediate next steps (recommended order)

1. **Write `scripts/merge_all.py`** — outer-join NASS features (per-(GEOID, year)), NDVI (per-(GEOID, year), concatenated from 21 per-year CSVs), gSSURGO (per-GEOID, broadcast across years), gridMET-derived weather features (per-(GEOID, year, forecast_date)), drought features (per-(GEOID, year, forecast_date)). Output: `scripts/training_master.parquet`. **This is the only Phase A item left.**
2. **Write `PHASE2_DATA_DICTIONARY.md`** documenting every column in `training_master.parquet`. **Phase A.7.**
3. **Verify Phase A definition-of-done**: master table loads, no surprise nulls in feature columns, 5-row spot check against original sources passes.
4. **Begin Phase B** — feature standardization + nearest-neighbor analog retrieval. Backtest cone calibration on 2023 + 2024.
5. **Begin Phase C** — XGBoost baseline once the master table is stable.
6. **Decide on HLS pull** at the Phase B → Phase D boundary, after the engineered baseline tells us whether the additional data-engineering cost for Prithvi is justified.

## Optional / future work

- Re-run `nass_pull.py` to fill non-irrigated columns specifically. Low priority (derivable from existing data).
- NOAA Storm Events as a broader severe-weather signal (replaces deprioritized tornado data).
- Sentinel-2 NDVI as a higher-resolution complement to MODIS (10m vs. 250m).
- USDA ARMS for fertilizer/management features.
- Crop progress reports for planting-date features.
- Process-based crop model integration (APSIM / DSSAT) for counterfactual reasoning — the "design optimal" stretch goal.
- Add DSCI / season-cum drought weeks / silking-peak DSCI to drought features if Phase B/C show the model wants more drought signal.

## Budget note

v1 used ~$0.25 of the $10 API cap. v2 token usage will be similar in shape (Haiku narration, ~$0.02 per forecast). Compute cost for HLS download + Prithvi inference is local (Ubuntu box) and outside the API budget.
