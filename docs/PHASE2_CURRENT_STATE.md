# Phase 2 — Current State

> Overwritten at end of every v2 phase. Tells the next chat what actually exists in the v2 portion of the repo right now. Reference file paths and describe behavior; do not paste code here.

**Project:** USDA Hackathon 2026 April — Corn Yield Forecasting
**Repo:** `~/dev/USDAhackathon2026April`
**Last updated:** 2026-04-25, mid-Phase A (data acquisition in progress)

---

## v2 status: mid-Phase A

Planning docs done. Data pipeline scripts partially written; some pulls executed, some still queued. No modeling, no backend endpoints, no agent tools yet. The v1 land-use system is shipped and untouched alongside.

## Project goal (locked)

Build an ML pipeline that predicts corn-for-grain yield (bu/acre) at the **county** level across **5 US states** at four fixed in-season forecast dates (Aug 1, Sep 1, Oct 1, end-of-season), aggregates to state, and produces a calibrated cone of uncertainty using **analog-year retrieval**. A Claude Haiku agent narrates the forecast.

- **Target states:** Colorado (CO), Iowa (IA), Missouri (MO), Nebraska (NE), Wisconsin (WI)
- **Time range:** **2005–2024** per the brief's explicit instruction (20 years; 2024 holdout, 2023 validation, 2005–2022 train)
- **Spatial unit:** US county, joined via 5-digit GEOID (state FIPS + county FIPS, both zero-padded)
- **Target variable:** `yield_bu_acre_all` (combined-practice corn grain yield, bu/acre)

See `PHASE2_PROJECT_PLAN.md` for the full vision and `PHASE2_PHASE_PLAN.md` for the phase breakdown with go/no-go gates.

## What exists in the repo today

### v1 (untouched, fully shipped)

The full v1 land-use & GHG analysis system is in place. See the v1 `CURRENT_STATE.md` for canonical detail. v2 reuses v1's FastAPI shell, frontend chrome, and Claude agent infrastructure but does not modify them.

### v2 — written and run

- **`scripts/nass_pull.py`** — written, partial run complete. Hits NASS QuickStats API for the 5 states, **2015–2024**, county-level corn yield/production/area at 3 practice levels (combined / irrigated / non-irrigated). 100% coverage on combined-practice columns (4/4 metrics × 2,998 county-year rows). ~7% coverage on irrigated (CO + NE only). Non-irrigated columns are 0% — rate-limited out by NASS's Azure gateway. Rate-limit fixes (2.0s delay, exponential backoff up to 240s, 4 retries) committed to the script. **Action: re-run for 2005–2014 to extend to the brief-required 2005–2024 range.**

- **`scripts/nass_corn_5states_2015_2024.csv`** — partial output of the above run. 2,998 rows, 374 unique counties, 12 columns of which 4 are fully populated. After the 2005–2014 backfill, will be merged into `scripts/nass_corn_5states_2005_2024.csv`.

### v2 — written, in flight

- **Earth Engine NDVI script** — written, debugged, export task submitted to Google Drive as `corn_ndvi_5states_2015_2024`. **Action: save into `scripts/ndvi_county_extraction.js` for version control.** **Action: re-run for 2005–2014 to backfill.** Output drops as a CSV in Drive folder `EarthEngineExports`. Uses MODIS MOD13Q1 (250m, 16-day) masked to corn pixels via USDA CDL. All aggregation done server-side in Earth Engine; output is already at county-year granularity.

  **Output schema (one row per county-year):**
  | Column | Meaning |
  |---|---|
  | `GEOID` | 5-digit county FIPS |
  | `NAME` | County name |
  | `STATEFP` | State FIPS |
  | `year` | Year |
  | `ndvi_peak` | Max NDVI during growing season (DOY 121–273) |
  | `ndvi_gs_mean` | Mean NDVI during growing season |
  | `ndvi_gs_integral` | Sum of NDVI values during growing season |
  | `ndvi_silking_mean` | Mean NDVI during silking (DOY 196–227) |
  | `ndvi_veg_mean` | Mean NDVI during vegetative phase (DOY 152–195) |

  **Caveat:** CDL 2024 may not exist yet (publishes Jan/Feb of the following year). If 2024 fails, set `endYear=2023` in the GEE script and rerun. CDL 2005 is the earliest available; corn-pixel masking works back to 2005.

### v2 — to write

- `scripts/nass_features.py` — local feature engineering on the NASS pull. No API calls. Produces `nass_corn_5states_features.csv` with `yield_target`, `irrigated_share`, `harvest_ratio`, derived non-irrigated columns.
- `scripts/ndvi_county_extraction.js` — version-controlled copy of the GEE script (currently only in the GEE web editor).
- `scripts/gssurgo_extract.py` — read .gdb files, pull Valu1 table, build per-property GeoTIFFs from MUKEY → property mapping, run zonal statistics over TIGER county polygons. Output: per-county soil features CSV keyed on GEOID.
- `scripts/prism_pull.py` — daily climate data (PRISM 4km or gridMET) for 5 states, **2005–2024**. Variables: tmax, tmin, prcp, srad, vp.
- `scripts/prism_features.py` — derive GDD (base 50°F, cap 86°F), EDD/KDD heat-stress hours, VPD, cumulative precip, monthly summaries. Aggregate to county-day, then to per-(GEOID, year, forecast_date) feature rows respecting the as-of rule (no leakage of post-forecast-date data).
- `scripts/drought_features.py` — pure local processing on the USDM Cumulative Percent Area CSV. Derives most-recent-reading D0–D4, DSCI, season-cumulative drought weeks, peak DSCI during silking. Per-(GEOID, year, forecast_date). Honors the as-of join rule (most recent Thursday strictly before forecast date).
- `scripts/merge_all.py` — outer-join all per-(GEOID, year) tables and per-(GEOID, year, forecast_date) tables. Output: single `training_master.parquet`.
- HLS acquisition script (no name yet) — required for Phase D.1 (Prithvi). Not yet started; lower priority than the engineered features. See "Open architectural questions" below.

## Data status (summary; live tracker is `PHASE2_DATA_INVENTORY.md`)

### 🟢 Have & ready
- NASS combined-practice corn yield/production/area, 5 states, 2015–2024, county level (`scripts/nass_corn_5states_2015_2024.csv`) — extending to 2005 as next action
- US Drought Monitor — **Cumulative Percent Area** format, weekly, county-level (5 states, 2005–2024)
- Mean monthly temperature heatmap + CSV (coarse smoothed feature only)
- Mean monthly precipitation heatmap + CSV (coarse smoothed feature only)
- gSSURGO state .gdb files downloaded for all 5 states, 10m resolution
- Tornado activity 2000–2022 (deprioritized — low signal at county-year aggregation)
- USDA NASS API access (key in `.env`)

### 🔵 In flight
- MODIS NDVI features via Earth Engine (export running for 2015–2024; lands in Google Drive)
- NASS API key + python-dotenv setup (working; partial pull done)

### 🔴 Need
- **NASS extension to 2005–2014** — re-run `scripts/nass_pull.py` with the rate-limit fixes already committed.
- **GEE NDVI extension to 2005–2014** — re-run the script with `startYear=2005, endYear=2014`. Server-side, cheap.
- **PRISM (or gridMET) daily weather** — for GDD, EDD, VPD, solar radiation. 2005–2024.
- **gSSURGO extraction** — files downloaded but extraction script not yet written.
- **CDL standalone download** — currently used only inside the Earth Engine script for masking. May want a local copy if HLS pipeline is built (Phase D.1).
- **HLS imagery** — required for Phase D.1 (Prithvi as feature extractor). 2005–2013 is Landsat-only; Sentinel-2 component starts 2015. Not yet started.
- **NAIP imagery** — tertiary; only if a phase explicitly calls for sub-meter visual context.
- **Prithvi model weights** — download in Phase D.1.

### ⚫ Deprioritized (with rationale)
- Hourly temperature — overkill for county-annual yield; daily Tmin/Tmax + GDD is sufficient.
- Sunrise/sunset hours — deterministic from latitude+date and modern corn hybrids are largely photoperiod-insensitive. Replaced with solar radiation from gridMET/NSRDB if needed.
- CO₂ concentration — spatially uniform across CONUS; absorbed by `year` as a feature (which also captures technology/genetics trends).
- Tornado activity — highly localized damage; near-zero signal at county-year aggregation. Could be replaced by NOAA Storm Events for broader severe-weather signal, but optional.

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
- `irrigated_share`, `harvest_ratio` (derived in `nass_features.py`)

### Topography (from USGS DEM)
- Elevation, slope, aspect
- Topographic Wetness Index (TWI)

### Economic (optional, only for "design optimal" use case)
- Commodity prices, input costs, crop insurance data

## Pipeline structure (best-practice separation)

For every external data source: **separate the pull from the feature engineering.** API calls and downloads are slow and rate-limited; features iterate fast. Splitting them prevents re-hitting external services on every feature change.

- `scripts/nass_pull.py` → raw `scripts/nass_corn_5states_2005_2024.csv`
- `scripts/nass_features.py` → derived `scripts/nass_corn_5states_features.csv`
- `scripts/ndvi_county_extraction.js` (GEE) → raw `corn_ndvi_5states_2005_2024.csv` (Drive)
- `scripts/gssurgo_extract.py` → `scripts/gssurgo_county_features.csv`
- `scripts/prism_pull.py` → raw daily netcdf or parquet
- `scripts/prism_features.py` → `scripts/prism_county_features.csv`
- `scripts/merge_all.py` → `scripts/training_master.parquet`

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

## Important data-quality fixes already discovered (NASS)

1. **Multi-practice fan-out bug:** without specifying `prodn_practice_desc`, NASS returns 3 rows per county-year (combined + irrigated + non-irrigated). Merging 4 such tables produced 81 rows per county-year (722,252 total instead of ~3,000). **Fix: always pin `prodn_practice_desc`.**
2. **"OTHER (COMBINED) COUNTIES" placeholder rows:** NASS rolls up suppressed-data counties into a fake "county" with GEOID ending in 000. **Fix: filter out rows where `county_name` starts with "OTHER" or `county_ansi` is empty.**
3. **Comma-separated values:** Large numbers come in as strings with commas (e.g., `"14,200,000"`). **Fix: strip commas before `pd.to_numeric`.**

## Earth Engine technical notes

- Used MODIS MOD13Q1 (250m, 16-day composite) as the NDVI source.
- Masked to corn-only pixels using USDA CDL `cropland == 1`.
- All aggregation done server-side in Earth Engine; the output CSV is already at county-year level.
- **Bug fixed:** original script tried to concatenate `'USDA/NASS/CDL/' + year` where `year` was a server-side `ee.Number`, producing invalid asset paths. **Fix:** use `ee.ImageCollection('USDA/NASS/CDL').filter(...)` instead.
- **Bug fixed:** map description error fixed by wrapping demo image with `ee.Image(...).set('system:description', 'demo')` to reset accumulated metadata.
- MODIS NDVI scale factor is 0.0001 (raw values 0–10000) — apply before computing features locally.

## Key technical notes for future sessions

### Join key convention
All datasets join on `GEOID` (5-digit string: state FIPS zero-padded to 2 + county FIPS zero-padded to 3) and `year` (integer). For per-forecast-date tables, the additional key is `forecast_date` ∈ `{"08-01", "09-01", "10-01", "EOS"}`.

### Data quirks to remember
- NASS suppresses data for low-corn counties → expect Colorado mountain counties to drop out, especially in extension years (2005–2014).
- NASS rate limits aggressively at sub-1s request intervals (Azure App Gateway).
- gSSURGO uses Albers Equal Area projection (EPSG:5070); reproject before zonal stats.
- MODIS NDVI scale factor 0.0001 (raw 0–10000).
- CDL value 1 = corn (used for masking in Earth Engine).
- Earth Engine `ee.Number` cannot be JS-concatenated into asset paths; use `ImageCollection.filter()` instead.

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

## File structure (current and planned)

```
~/dev/USDAhackathon2026April/
├── .env                                       # API keys (gitignored)
├── .gitignore
├── HACKATHON_TODO.md                          # v1 hackathon roadmap (still in force)
├── README.md                                  # v1 README
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
├── phase2/                                    # ← v2 working dir (legacy notes)
├── scripts/                                   # ← Data pipeline lives here
│   ├── (v1) dataset.py, train.py, infer.py, emissions.py, etc.
│   ├── nass_pull.py                           # ✅ written, 2015-2024 partial run done; needs 2005-2014 backfill run
│   ├── nass_features.py                       # ⬜ to write
│   ├── nass_corn_5states_2015_2024.csv        # ✅ partial output (will become …2005_2024.csv after backfill)
│   ├── ndvi_county_extraction.js              # ⬜ save GEE script for VC
│   ├── gssurgo_extract.py                     # ⬜ to write
│   ├── prism_pull.py                          # ⬜ to write
│   ├── prism_features.py                      # ⬜ to write
│   └── merge_all.py                           # ⬜ final join → training_master.parquet
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
- **Drought feature source: USDM Cumulative Percent Area locked.** Per-county weekly D0–D4 percentages, format = each level reports % of county at that level *or worse*. SPI/SPEI from PRISM still derivable as a redundant signal if needed; not blocking.
- **gSSURGO MUKEY → property rasterization** — straightforward but slow. Consider caching the per-property GeoTIFFs to avoid recomputing on every feature pass.
- **NASS non-irrigated columns** — 0% coverage in the current pull. Derivable from `acres_harvested_all − acres_harvested_irr` for fields where irrigated coverage exists; for fields where irrigated coverage is also sparse, this is a known gap. Re-running `nass_pull.py` with the rate-limit fixes for non-irrigated specifically is optional, low-priority.
- **NASS county coverage in extension years (2005–2014).** Some Colorado mountain counties may drop out of the early years entirely; need to QC after the 2005–2014 backfill lands.

## Hackathon readiness checklist

- [x] Planning docs (`PHASE2_PROJECT_PLAN.md`, `PHASE2_PHASE_PLAN.md`, `PHASE2_DATA_INVENTORY.md`, `PHASE2_DECISIONS_LOG.md`) drafted
- [x] NASS API key + script, partial pull done for 2015–2024 (~3,000 county-year rows, full coverage on combined-practice columns)
- [x] Earth Engine NDVI export submitted for 2015–2024 (await Drive landing)
- [ ] NASS pull extended to 2005–2014 (re-run `nass_pull.py`)
- [ ] GEE NDVI export extended to 2005–2014 (re-run script)
- [ ] `scripts/nass_features.py` written and `nass_corn_5states_features.csv` produced
- [ ] `ndvi_county_extraction.js` saved into `scripts/` for version control
- [ ] `scripts/gssurgo_extract.py` written; per-county soil features CSV produced
- [ ] `scripts/prism_pull.py` + `scripts/prism_features.py` written; per-county weather features produced (2005–2024)
- [ ] `scripts/merge_all.py` written; `training_master.parquet` built
- [ ] Phase A definition-of-done met (master table loads, no surprise nulls, spot-check passes)
- [ ] Phase B analog-year retrieval baseline shipped + Phase B gate passed
- [ ] Phase C XGBoost point-estimate model shipped + Phase C gate passed
- [ ] Phase D.1 Prithvi frozen-feature integration + Phase D gate decision
- [ ] Phase D.2 (conditional) Prithvi fine-tune
- [ ] Phase E `/forecast/{state}` endpoint + frontend forecast view
- [ ] Phase F forecast narration agent with 4 tools
- [ ] Phase G holdout evaluation, ablation table, presentation deck

## Immediate next steps (recommended order)

1. **Wait for the 2015–2024 Earth Engine NDVI export to finish.** Verify CSV lands in Google Drive `EarthEngineExports/`. Save the GEE script into `scripts/ndvi_county_extraction.js`. If 2024 CDL fails, set `endYear=2023` and rerun.
2. **Re-run `scripts/nass_pull.py` for 2005–2014** to extend the NASS yield history. Rate-limit fixes are already committed; expect this to take longer than the 2015–2024 run because of the 2.0s delay + occasional backoffs. Merge into `scripts/nass_corn_5states_2005_2024.csv`.
3. **Re-run the GEE NDVI script for 2005–2014.** Server-side, cheap. Combine with the 2015–2024 export into `corn_ndvi_5states_2005_2024.csv`.
4. **Write `scripts/nass_features.py`** — local processing on the combined NASS CSV. Produces `nass_corn_5states_features.csv` with `yield_target`, `irrigated_share`, `harvest_ratio`, derived non-irrigated columns. No API calls.
5. **Write `scripts/gssurgo_extract.py`** — read .gdb files, pull Valu1, build per-property GeoTIFFs, run zonal stats over TIGER 2018 county polygons. Output per-county soil features CSV keyed on GEOID.
6. **Write `scripts/prism_pull.py` + `scripts/prism_features.py`** — daily PRISM (or gridMET); derive GDD/EDD/VPD/cumulative-precip respecting the as-of rule for each forecast date. 2005–2024.
7. **Write `scripts/merge_all.py`** — outer-join all features → `training_master.parquet`.
8. **Begin Phase B** — feature standardization + nearest-neighbor analog retrieval. Backtest cone calibration on 2023 + 2024.
9. **Begin Phase C** — XGBoost baseline once the master table is stable.
10. **Decide on HLS pull** at the Phase B → Phase D boundary, after the engineered baseline tells us whether the additional data-engineering cost for Prithvi is justified.

## Optional / future work

- Re-run `nass_pull.py` to fill non-irrigated columns specifically. Low priority (derivable from existing data).
- NOAA Storm Events as a broader severe-weather signal (replaces deprioritized tornado data).
- Sentinel-2 NDVI as a higher-resolution complement to MODIS (10m vs. 250m).
- USDA ARMS for fertilizer/management features.
- Crop progress reports for planting-date features.
- Process-based crop model integration (APSIM / DSSAT) for counterfactual reasoning — the "design optimal" stretch goal.

## Budget note

v1 used ~$0.25 of the $10 API cap. v2 token usage will be similar in shape (Haiku narration, ~$0.02 per forecast). Compute cost for HLS download + Prithvi inference is local (Ubuntu box) and outside the API budget.
