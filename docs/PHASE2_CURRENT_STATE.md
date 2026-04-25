# Phase 2 — Current State

> Overwritten at end of every v2 phase. Tells the next chat what actually exists in the v2 portion of the repo right now. Reference file paths and describe behavior; do not paste code here.

**Project:** USDA Hackathon 2026 April — Corn Yield Forecasting
**Repo:** `~/dev/USDAhackathon2026April`
**Last updated:** 2026-04-25, end of Phase A.1–A.2 (NASS + NDVI both landed)

---

## v2 status: Phase A.1 + A.2 complete; A.3 (gSSURGO) and A.4 (PRISM) up next

Three data pipelines have output sitting on disk:
- NASS yield 2005–2024 (full coverage on combined-practice columns)
- MODIS NDVI 2004–2024 (corn-masked via CDL, county-aggregated; one CSV per year, 21 files)
- US Drought Monitor weekly per-county D0–D4 percentages (raw CSV in `phase2/data/drought/`)

`nass_features.py` is written and `nass_corn_5states_features.csv` is produced. `ndvi_county_extraction.js` is now version-controlled in the repo. No modeling, no backend endpoints, no agent tools yet. The v1 land-use system is shipped and untouched alongside.

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

- **`scripts/nass_corn_5states_2005_2024.csv`** — output of the full 2005–2024 pull. The single canonical NASS CSV.

- **`scripts/nass_features.py`** — written. Local feature engineering on the NASS CSV. No API calls. Produces `scripts/nass_corn_5states_features.csv` with `yield_target`, `irrigated_share`, `harvest_ratio`, derived non-irrigated columns.

- **`scripts/nass_corn_5states_features.csv`** — output of `nass_features.py`. Ready to merge.

- **`scripts/ndvi_county_extraction.js`** — version-controlled GEE script. Pulls MODIS NDVI 2004–2024 for the 5 states, masked to corn pixels via USDA CDL, reduced to county-year. Header documents bug fixes (CDL `ImageCollection.filter` pattern, demo-image metadata reset), output schema, and how to extend or re-run. Script enqueues one Export task per year.

- **`phase2/data/ndvi/corn_ndvi_5states_<year>.csv`** (21 files, 2004–2024) — output of the GEE script. 443 rows per file (TIGER/2018 county polygons for the 5 states). Schema: `GEOID, NAME, STATEFP, year, ndvi_peak, ndvi_gs_mean, ndvi_gs_integral, ndvi_silking_mean, ndvi_veg_mean`. **NDVI values are pre-scaled (× 0.0001 applied server-side); CSV values are floats in roughly [-0.2, 1.0] — do NOT re-apply the scale factor downstream.** Counties with zero CDL corn pixels in a given year emit null NDVI columns; downstream merge must tolerate this.

- **`phase2/data/drought/drought_USDM-Colorado,Iowa,Missouri,Nebraska,Wisconsin.csv`** — raw US Drought Monitor weekly D0–D4 percentages per county, 5 states. Pulled. Not yet feature-engineered into per-(GEOID, year, forecast_date) form.

### v2 — to write

- `scripts/gssurgo_extract.py` — read .gdb files, pull Valu1 table, build per-property GeoTIFFs from MUKEY → property mapping, run zonal statistics over TIGER 2018 county polygons. Output: per-county soil features CSV keyed on GEOID. **Phase A.3.**
- `scripts/prism_pull.py` — daily climate data (PRISM 4km or gridMET) for 5 states, **2005–2024**. Variables: tmax, tmin, prcp, srad, vp. **Phase A.4.**
- `scripts/prism_features.py` — derive GDD (base 50°F, cap 86°F), EDD/KDD heat-stress hours, VPD, cumulative precip, monthly summaries. Aggregate to county-day, then to per-(GEOID, year, forecast_date) feature rows respecting the as-of rule (no leakage of post-forecast-date data). **Phase A.4.**
- `scripts/drought_features.py` — derive per-(GEOID, year, forecast_date) drought severity features from the existing USDM CSV. Resample weekly observations to forecast-date as-of values. **Phase A.5.**
- `scripts/merge_all.py` — outer-join all per-(GEOID, year) tables and per-(GEOID, year, forecast_date) tables. Output: single `training_master.parquet`. **Phase A.6.**
- HLS acquisition script (no name yet) — required for Phase D.1 (Prithvi). Not yet started; lower priority than the engineered features. See "Open architectural questions" below.

## Data status (summary; live tracker is `PHASE2_DATA_INVENTORY.md`)

### 🟢 Have & ready
- NASS combined-practice corn yield/production/area, 5 states, **2005–2024**, county level (`scripts/nass_corn_5states_2005_2024.csv`)
- NASS engineered features (`scripts/nass_corn_5states_features.csv`)
- MODIS NDVI features per (GEOID, year), **2004–2024** (`phase2/data/ndvi/corn_ndvi_5states_<year>.csv` × 21)
- US Drought Monitor weekly raw CSV (5 states, multi-year — needs feature engineering)
- Mean monthly temperature heatmap + CSV (coarse smoothed feature only)
- Mean monthly precipitation heatmap + CSV (coarse smoothed feature only)
- gSSURGO state .gdb files downloaded for all 5 states, 10m resolution
- USDA NASS API access (key in `.env`)
- GEE script version-controlled (`scripts/ndvi_county_extraction.js`)

### 🔵 In flight
*(none currently — all queued pulls have landed)*

### 🔴 Need
- **PRISM (or gridMET) daily weather** — for GDD, EDD, VPD, solar radiation. 2005–2024. Slow download; start early. Phase A.4.
- **gSSURGO extraction** — files downloaded but extraction script not yet written. Phase A.3.
- **Drought feature engineering** — raw CSV on disk; needs aggregation to per-(GEOID, year, forecast_date) feature rows. Phase A.5.
- **CDL standalone download** — currently used only inside the Earth Engine script for masking. Want a local copy if HLS pipeline is built (Phase D.1).
- **HLS imagery** — required for Phase D.1 (Prithvi as feature extractor). 2005–2013 is Landsat-only; Sentinel-2 component starts 2015. Not yet started.
- **NAIP imagery** — tertiary; only if a phase explicitly calls for sub-meter visual context.
- **Prithvi model weights** — download in Phase D.1.

### ⚫ Deprioritized (with rationale)
- Hourly temperature — overkill for county-annual yield; daily Tmin/Tmax + GDD is sufficient.
- Sunrise/sunset hours — deterministic from latitude+date and modern corn hybrids are largely photoperiod-insensitive. Replaced with solar radiation from gridMET/NSRDB if needed.
- CO₂ concentration — spatially uniform across CONUS; absorbed by `year` as a feature (which also captures technology/genetics trends).
- Tornado activity — highly localized damage; near-zero signal at county-year aggregation. Could be replaced by NOAA Storm Events for broader severe-weather signal, but optional.

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

This is the design target — not all populated yet. Phase A is about getting them.

### Soil (from gSSURGO Valu1 table, county-aggregated via zonal stats)
- `nccpi3corn` — National Commodity Crop Productivity Index for corn (gold-standard single soil feature)
- `aws0_100` — available water storage in root zone, mm
- `soc0_30` — soil organic carbon, surface 0–30 cm, g C/m²
- `rootznemc` — root zone effective moisture capacity
- `droughty` — drought-vulnerable soil flag
- Optional: depth-weighted texture, pH, OM, CEC

### Climate (from PRISM or gridMET, daily, 4km, county-aggregated)
- GDD base-50°F cap-86°F, accumulated by growth phase and to forecast date
- EDD/KDD hours above 86°F or 90°F (heat stress, especially during silking)
- VPD (vapor pressure deficit)
- Tmin / Tmax monthly means
- Precipitation totals AND distribution (e.g., cumulative + dry-spell length)
- Drought indices (SPI / SPEI) — optional if not pulling US Drought Monitor separately
- Solar radiation (shortwave, MJ/m²/day)

### Remote sensing (from Google Earth Engine, MODIS-derived; HLS as a Phase D.1 layer)
- `ndvi_peak` — max NDVI during growing season
- `ndvi_gs_mean` — mean NDVI during growing season
- `ndvi_gs_integral` — sum NDVI during growing season
- `ndvi_silking_mean` — mean NDVI during silking window (DOY 196–227)
- `ndvi_veg_mean` — mean NDVI during vegetative phase (DOY 152–195)
- Optional later: EVI, SIF, LAI
- Phase D.1: Prithvi embedding from raw HLS chips (separate from MODIS NDVI)

### Management (from NASS; ARMS optional)
- Planting date / harvest date (from NASS Crop Progress)
- Hybrid relative maturity
- Seeding rate / plant population (already on hand for 2021–2025)
- Tillage practice
- N/P/K fertilizer rates (from USDA ARMS, optional)
- Crop rotation history (derivable from CDL year-over-year)
- Irrigation type and water applied (already on hand 2018–2023)
- `irrigated_share`, `harvest_ratio` (derived in `nass_features.py`, already done)

### Topography (from USGS DEM)
- Elevation, slope, aspect
- Topographic Wetness Index (TWI)

### Economic (optional, only for "design optimal" use case)
- Commodity prices, input costs, crop insurance data

## Pipeline structure (best-practice separation)

For every external data source: **separate the pull from the feature engineering.** API calls and downloads are slow and rate-limited; features iterate fast. Splitting them prevents re-hitting external services on every feature change.

- `scripts/nass_pull.py` → raw `scripts/nass_corn_5states_2005_2024.csv` ✅
- `scripts/nass_features.py` → derived `scripts/nass_corn_5states_features.csv` ✅
- `scripts/ndvi_county_extraction.js` (GEE) → raw `phase2/data/ndvi/corn_ndvi_5states_<year>.csv` × 21 ✅
- `scripts/gssurgo_extract.py` → `scripts/gssurgo_county_features.csv` ⬜
- `scripts/prism_pull.py` → raw daily netcdf or parquet ⬜
- `scripts/prism_features.py` → `scripts/prism_county_features.csv` ⬜
- `scripts/drought_features.py` → `scripts/drought_county_features.csv` ⬜
- `scripts/merge_all.py` → `scripts/training_master.parquet` ⬜

## Decisions made (with rationale)

These are now logged in `PHASE2_DECISIONS_LOG.md` as the canonical record. Restated here for orientation:

1. **County-level aggregation, not pixel-level** — matches NASS reporting unit and avoids massive computation for marginal gain.
2. **Server-side aggregation in Earth Engine** — output CSV is already summarized to county-year, eliminating local zonal stats on imagery.
3. **Separate raw-pull script from feature-engineering script** — see "Pipeline structure" above.
4. **Use the Valu1 table for soil aggregation** instead of manually weighting components/horizons — Valu1 is USDA's official pre-aggregation and covers 90% of common ML use cases including the corn-specific NCCPI index.
5. **Skip hourly weather, day length, and CO₂** — see Deprioritized section.
6. **Use combined-practice yield as the target** — `yield_bu_acre_all` has full coverage; irrigation effect is captured via `irrigated_share` where data exists.
7. **TIGER/Line 2018 county boundaries** for spatial reduction in Earth Engine — stable across the 2005–2024 time range.
8. **Time range 2005–2024 per the brief.** Train 2005–2022, val 2023, holdout 2024.
9. **MODIS NDVI is the primary remote-sensing feature for the engineered baseline.** Raw HLS is deferred to Phase D.1 (Prithvi). MODIS NDVI and HLS are *complementary*, not substitutes.
10. **PRISM (or gridMET) is the daily weather source** — both work; PRISM is US-only and more focused. Decide finally when `prism_pull.py` is written.
11. **NDVI per-year file structure (Phase A.2):** GEE script enqueues one Export task per year rather than a single multi-year export. Output is 21 CSVs; concatenated by `merge_all.py` later. Cleaner failure mode (one bad year doesn't kill the run) and matches what's already on disk.
12. **NDVI is pre-scaled in the CSV (Phase A.2):** the GEE script applies `× 0.0001` server-side; downstream code must NOT re-apply.

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

### Inherited from v1 (still apply for shared infrastructure)
- `conda activate landuse` per session.
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
├── PHASE2_PROJECT_PLAN.md                     # v2 vision + locked decisions
├── PHASE2_PHASE_PLAN.md                       # v2 phase breakdown A–G
├── PHASE2_CURRENT_STATE.md                    # ← THIS FILE
├── PHASE2_DATA_INVENTORY.md                   # v2 live data tracker
├── PHASE2_DECISIONS_LOG.md                    # v2 append-only decisions
├── fileStructure.txt
├── requirements.txt
├── requirements-windows.txt
├── agent/                                     # v1 agent package (untouched)
├── backend/                                   # v1 FastAPI app (untouched)
├── docs/                                      # v1 doc artifacts
├── frontend/                                  # v1 frontend (untouched)
├── inference_outputs/                         # v1 segmentation outputs
├── model/                                     # v1 SegFormer checkpoint
├── phase2/
│   └── data/
│       ├── drought/
│       │   └── drought_USDM-Colorado,Iowa,Missouri,Nebraska,Wisconsin.csv  ✅ raw
│       └── ndvi/
│           ├── corn_ndvi_5states_2004.csv     ✅ (21 files, 2004–2024)
│           └── ... corn_ndvi_5states_2024.csv
├── scripts/                                   # ← Data pipeline lives here
│   ├── (v1) dataset.py, train.py, infer.py, emissions.py, etc.
│   ├── nass_pull.py                           ✅ written, run for 2005–2024
│   ├── nass_features.py                       ✅ written
│   ├── nass_corn_5states_2005_2024.csv        ✅ canonical NASS output
│   ├── nass_corn_5states_features.csv         ✅ engineered features
│   ├── ndvi_county_extraction.js              ✅ version-controlled (per-year exports)
│   ├── gssurgo_extract.py                     ⬜ to write (Phase A.3)
│   ├── prism_pull.py                          ⬜ to write (Phase A.4)
│   ├── prism_features.py                      ⬜ to write (Phase A.4)
│   ├── drought_features.py                    ⬜ to write (Phase A.5)
│   └── merge_all.py                           ⬜ final join → training_master.parquet (Phase A.6)
├── smoke_agent.py                             # v1 smoke test
├── smoke_tools.py                             # v1 smoke test
└── preview.jpg
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
- **Drought feature engineering details.** USDM raw CSV is on disk but the per-(GEOID, year, forecast_date) feature table not yet built. Decisions to make: which D-level severities to include (sum of D2+? individual D0–D4?), how to as-of-resample weekly readings to forecast dates (last reading before forecast date? trailing-N-week mean?). Phase A.5.
- **gSSURGO MUKEY → property rasterization** — straightforward but slow. Consider caching the per-property GeoTIFFs to avoid recomputing on every feature pass.
- **NASS non-irrigated columns** — sparse in the pull. Derivable from `acres_harvested_all − acres_harvested_irr` for fields where irrigated coverage exists; for fields where irrigated coverage is also sparse, this is a known gap. Re-running `nass_pull.py` with the rate-limit fixes for non-irrigated specifically is optional, low-priority.

## Hackathon readiness checklist

- [x] Planning docs (`PHASE2_PROJECT_PLAN.md`, `PHASE2_PHASE_PLAN.md`, `PHASE2_DATA_INVENTORY.md`, `PHASE2_DECISIONS_LOG.md`) drafted
- [x] NASS API key + script, full pull done for 2005–2024 (full coverage on combined-practice columns)
- [x] `scripts/nass_features.py` written and `nass_corn_5states_features.csv` produced
- [x] GEE NDVI export complete for 2004–2024 (21 per-year CSVs in `phase2/data/ndvi/`)
- [x] `ndvi_county_extraction.js` saved into `scripts/` for version control
- [x] US Drought Monitor raw CSV pulled for 5 states
- [ ] `scripts/gssurgo_extract.py` written; per-county soil features CSV produced (Phase A.3)
- [ ] `scripts/prism_pull.py` + `scripts/prism_features.py` written; per-county weather features produced 2005–2024 (Phase A.4)
- [ ] `scripts/drought_features.py` written; per-(GEOID, year, forecast_date) drought features produced (Phase A.5)
- [ ] `scripts/merge_all.py` written; `training_master.parquet` built (Phase A.6)
- [ ] Phase A definition-of-done met (master table loads, no surprise nulls, spot-check passes)
- [ ] Phase B analog-year retrieval baseline shipped + Phase B gate passed
- [ ] Phase C XGBoost point-estimate model shipped + Phase C gate passed
- [ ] Phase D.1 Prithvi frozen-feature integration + Phase D gate decision
- [ ] Phase D.2 (conditional) Prithvi fine-tune
- [ ] Phase E `/forecast/{state}` endpoint + frontend forecast view
- [ ] Phase F forecast narration agent with 4 tools
- [ ] Phase G holdout evaluation, ablation table, presentation deck

## Immediate next steps (recommended order)

1. **Write `scripts/prism_pull.py`** to start the slow daily-weather download running in the background. PRISM (or gridMET) for 5 states, 2005–2024, daily Tmax/Tmin/Prcp/Srad/VP. The pull is the long pole; kick it off first so it can churn while other code is being written.
2. **Write `scripts/gssurgo_extract.py`** while PRISM downloads. Files are already on disk. Read .gdb, pull Valu1 table, build per-property GeoTIFFs, run zonal stats over TIGER 2018 county polygons. Output: `scripts/gssurgo_county_features.csv` keyed on GEOID. Soil features are static across years.
3. **Write `scripts/prism_features.py`** once raw PRISM data is on disk. Derive GDD (base 50°F, cap 86°F), EDD/KDD, VPD, cumulative precip, monthly summaries. **Critical:** respect the as-of rule for each forecast date — when constructing features for forecast date `D` in year `Y`, use only data with timestamps strictly before `D`.
4. **Write `scripts/drought_features.py`** — small, self-contained. Read USDM raw CSV; resample weekly readings to per-(GEOID, year, forecast_date) as-of values. Decide D-level severity aggregation.
5. **Write `scripts/merge_all.py`** — outer-join NASS features (per-(GEOID, year)), NDVI (per-(GEOID, year), concatenated from 21 per-year CSVs), gSSURGO (per-GEOID, broadcast across years), PRISM features (per-(GEOID, year, forecast_date)), drought features (per-(GEOID, year, forecast_date)). Output: `scripts/training_master.parquet`.
6. **Write `PHASE2_DATA_DICTIONARY.md`** documenting every column in `training_master.parquet`.
7. **Begin Phase B** — feature standardization + nearest-neighbor analog retrieval. Backtest cone calibration on 2023 + 2024.
8. **Begin Phase C** — XGBoost baseline once the master table is stable.
9. **Decide on HLS pull** at the Phase B → Phase D boundary, after the engineered baseline tells us whether the additional data-engineering cost for Prithvi is justified.

## Optional / future work

- Re-run `nass_pull.py` to fill non-irrigated columns specifically. Low priority (derivable from existing data).
- NOAA Storm Events as a broader severe-weather signal (replaces deprioritized tornado data).
- Sentinel-2 NDVI as a higher-resolution complement to MODIS (10m vs. 250m).
- USDA ARMS for fertilizer/management features.
- Crop progress reports for planting-date features.
- Process-based crop model integration (APSIM / DSSAT) for counterfactual reasoning — the "design optimal" stretch goal.

## Budget note

v1 used ~$0.25 of the $10 API cap. v2 token usage will be similar in shape (Haiku narration, ~$0.02 per forecast). Compute cost for HLS download + Prithvi inference is local (Ubuntu box) and outside the API budget.