# Phase 2 — Current State

**Project:** USDA Hackathon 2026 April — Corn Yield Prediction ML Model
**Repo:** `~/dev/USDAhackathon2026April`
**Last updated:** April 25, 2026

---

## Project Goal

Build a machine learning model to predict and design optimal corn yields at
the county level across 5 US states using satellite, soil, climate, and
agricultural management data.

**Target states:** Colorado (CO), Iowa (IA), Missouri (MO), Nebraska (NE),
Wisconsin (WI)

**Time range:** 2015–2024 (10 years)

**Spatial unit:** US county (joined via 5-digit GEOID = state FIPS + county FIPS)

**Target variable:** Corn grain yield in bushels per acre (`yield_bu_acre_all`)

---

## Data Sources — Status

### ✅ Already gathered (pre-existing)
- Satellite images (raw)
- Mean temperature per month heatmap (CSV)
- Mean precipitation per month heatmap (CSV)
- Tornado activity per state, 2000–2022 *(low value for yield model — can be deprioritized)*
- Crop (corn) planted data
- Principal Crops Area Planted and Harvested — 2023–2025
- Corn Area Planted/Harvested/Yield/Production — 2023–2025
- Corn for Silage data — 2023–2025
- Corn for Grain Plant Population per Acre — 2021–2025
- NASS corn yields, county yields, state yields (manual pulls)
- Water applied to corn for 5 states/counties, 2018–2023
- Corn/grain yield for 5 states/counties, 2015–2024
- USDA NASS API access

### 🟡 In progress
- **NDVI from Google Earth Engine** — script written and submitted as export
  task. See "Earth Engine NDVI" section below.
- **NASS API pull** — partial success due to rate limiting. See "NASS Data"
  section below.

### ⬜ Identified but not yet gathered
- **gSSURGO soil data** — user has downloaded state .gdb files; extraction
  script not yet written. See "Soil Data — gSSURGO" section.
- **PRISM or gridMET climate data** — for GDD, EDD, VPD, solar radiation.
- **CDL (Cropland Data Layer)** — used inside Earth Engine script for corn
  masking; not separately downloaded.

### ❌ Discussed and intentionally skipped
- **Hourly temperature** — overkill for county-annual yield. Use daily
  Tmin/Tmax from PRISM and derive GDD/EDD instead.
- **Sunrise/sunset hours** — replaced with solar radiation (gridMET or NSRDB).
  Day length is deterministic from latitude+date, and modern corn hybrids are
  largely photoperiod-insensitive.
- **CO₂ concentration** — spatially uniform across US; use `year` as a
  feature instead to absorb CO₂ + technology + genetics trends.

---

## Recommended Full Feature Set (for reference)

### Soil (from gSSURGO Valu1 table)
- `nccpi3corn` — National Commodity Crop Productivity Index for corn (gold standard single soil feature)
- `aws0_100` — Available water storage in root zone (mm)
- `soc0_30` — Soil organic carbon, surface (g C/m²)
- `droughty` — Drought-vulnerable soil flag
- Plus depth-weighted texture, pH, OM, CEC if needed

### Climate (from PRISM or gridMET, daily 4km)
- GDD (base 50°F, cap 86°F) per growth phase
- EDD/KDD (hours above 86°F or 90°F) — heat stress, esp. silking
- VPD (vapor pressure deficit)
- Tmin/Tmax monthly means
- Precipitation totals AND distribution
- Drought indices (SPI/SPEI)
- Solar radiation (shortwave, MJ/m²/day)

### Remote Sensing (from Google Earth Engine)
- NDVI peak during growing season
- NDVI integrated over silking window
- NDVI mean per growth phase
- Optional: EVI, SIF, LAI

### Management (from NASS + USDA ARMS)
- Planting date / harvest date
- Hybrid relative maturity
- Seeding rate / plant population
- Tillage practice
- N/P/K fertilizer rates
- Crop rotation history (derivable from CDL year-over-year)
- Irrigation type and water applied

### Topography (from USGS DEM)
- Elevation, slope, aspect
- Topographic Wetness Index (TWI)

### Economic (optional — for "design optimal" use case)
- Commodity prices, input costs, crop insurance data

---

## Earth Engine NDVI

**Status:** Script written, debugged, and export task submitted to Google Drive.

**Script location (in editor):** Save as `ndvi_county_extraction` in your
Earth Engine repository.

**Export task name:** `corn_ndvi_5states_2015_2024`

**Output:** `nass_corn_5states_2015_2024.csv` will land in Google Drive folder
`EarthEngineExports`. **NOTE:** filename is technically the NDVI export — name
above is correct: `corn_ndvi_5states_2015_2024.csv`.

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

**Key technical decisions:**
- Used MODIS MOD13Q1 (250m, 16-day composite) as the NDVI source
- Masked to corn-only pixels using USDA CDL `cropland == 1`
- All aggregation done server-side in Earth Engine — output CSV is already
  county-year level, no further reduction needed
- Bug fixed: original script tried to concatenate `'USDA/NASS/CDL/' + year`
  where `year` was a server-side `ee.Number`, producing invalid asset paths.
  Fixed by using `ee.ImageCollection('USDA/NASS/CDL').filter(...)` instead.
- Map description error fixed by wrapping demo image with `ee.Image(...)
  .set('system:description', 'demo')` to reset accumulated metadata.

**Known caveat:** CDL 2024 may not exist yet (publishes Jan/Feb of following
year). If 2024 fails, change `endYear` to 2023 and rerun.

**Final script location:** Earlier in this conversation. Should be saved into
the project as `scripts/ndvi_county_extraction.js` for version control.

---

## NASS Data

### Setup
- API key obtained from `quickstats.nass.usda.gov/api`
- Stored in `.env` file as `NASS_API_KEY=...`
- Loaded in Python via `python-dotenv`
- `.env` and `*.csv` added to `.gitignore`

### Pipeline structure (best-practice separation)
- `scripts/nass_pull.py` — hits NASS API, produces raw CSV
- `scripts/nass_features.py` — local processing, produces ML-ready CSV
- Run `nass_pull.py` only when source data updates; iterate freely with
  `nass_features.py`

### Pull results (partial, due to rate limiting)
The script pulled **3 practice levels × 4 metrics = 12 columns**:
- ALL PRODUCTION PRACTICES (combined)
- IRRIGATED
- NON-IRRIGATED

**Coverage achieved:**
| Column | Rows | Coverage |
|---|---|---|
| `yield_bu_acre_all` | 2,998 | 100% |
| `production_bu_all` | 2,998 | 100% |
| `acres_harvested_all` | 2,998 | 100% |
| `acres_planted_all` | 2,998 | 100% |
| `yield_bu_acre_irr` | 207 | 6.9% (CO+NE only) |
| `production_bu_irr` | 207 | 6.9% (CO+NE only) |
| `acres_harvested_irr` | 207 | 6.9% (CO+NE only) |
| `acres_planted_irr` | 0 | 0% (rate-limited out) |
| `yield_bu_acre_noirr` | 0 | 0% (rate-limited out) |
| `production_bu_noirr` | 0 | 0% (rate-limited out) |
| `acres_harvested_noirr` | 0 | 0% (rate-limited out) |
| `acres_planted_noirr` | 0 | 0% (rate-limited out) |

**Per-state row counts:**
- CO: 159
- IA: 906
- MO: 637
- NE: 710
- WI: 586
- **Total: 2,998 county-year rows, 374 unique counties, 2015–2024**

### Rate-limiting issue (RESOLVED in code, NOT YET RE-RUN)
NASS is hosted behind Microsoft Azure Application Gateway, which rate-limits
aggressive request patterns. The script was making requests every 0.5s and
got 403 Forbidden after ~7 query batches. The blocked queries were all
NON-IRRIGATED + irrigated planted acres.

**Fix applied to script:**
- Increased delay from 0.5s → 2.0s between requests
- Added exponential backoff on 403 (30s → 60s → 120s → 240s)
- Up to 4 retries per request
- See latest version of `nass_pull.py` from this conversation

### Important data-quality fixes discovered
1. **Multi-practice fan-out bug:** without specifying `prodn_practice_desc`,
   NASS returns 3 rows per county-year (combined + irrigated + non-irrigated).
   Merging 4 such tables produced 81 rows per county-year (722,252 total
   rows instead of ~3,000). **Fix:** always pin `prodn_practice_desc`.
2. **"OTHER (COMBINED) COUNTIES" placeholder rows:** NASS rolls up
   suppressed-data counties into a fake "county" with GEOID ending in 000.
   **Fix:** filter out rows where `county_name` starts with "OTHER" or
   `county_ansi` is empty.
3. **Comma-separated values:** Large numbers come in as strings with commas
   (e.g., `"14,200,000"`). **Fix:** strip commas before `pd.to_numeric`.

### IMPORTANT: NASS data was already enough for v1
Despite the rate-limit incident, the pulled CSV has 100% coverage on
combined-practice yield/production/harvested/planted across all 5 states +
~7% coverage on irrigated for CO and NE. Non-irrigated can be derived as
`acres_harvested_all - acres_harvested_irr`. **No need to re-run unless
specifically pursuing more complete irrigated/non-irrigated splits.**

### Output file
`scripts/nass_corn_5states_2015_2024.csv`

### Feature engineering plan (`nass_features.py`)
```python
# Core ML features derived from NASS:
df["irrigated_share"] = (df["acres_harvested_irr"] / df["acres_harvested_all"]).fillna(0).clip(0,1)
df["yield_target"] = df["yield_bu_acre_all"]
df["harvest_ratio"] = (df["acres_harvested_all"] / df["acres_planted_all"]).clip(0,1)
df["acres_harvested_noirr_derived"] = df["acres_harvested_all"] - df["acres_harvested_irr"].fillna(0)
```

---

## Soil Data — gSSURGO

**Status:** State .gdb files downloaded; extraction script NOT YET written.

**Source:** `nrcs.usda.gov/resources/data-and-reports/gridded-soil-survey-geographic-gssurgo-database`

**Format downloaded:** State-level ESRI File Geodatabase (`.gdb`) at 10m
resolution. (Skipped CONUS download — only 30m resolution and >40GB.)

**Recommended extraction approach:**
1. Read `.gdb` with Python (GDAL/OGR via geopandas, rasterio)
2. Use the **Valu1 table** that ships inside each .gdb — already aggregated
   to MUKEY level (no need to manually weight components/horizons)
3. Key Valu1 columns for corn: `nccpi3corn`, `aws0_100`, `soc0_30`,
   `rootznemc`, `droughty`
4. Build per-property GeoTIFF rasters by remapping MUKEY → property value
5. Run zonal statistics over county polygons (TIGER/Line) to get one
   number per county per property
6. Output: CSV with `GEOID` + soil feature columns

**Tools:** `rasterio`, `geopandas`, `rasterstats` Python libraries.

**Pseudocode pipeline available** in earlier chat — write
`scripts/gssurgo_extract.py` when ready.

---

## File Structure (Current and Planned)

```
~/dev/USDAhackathon2026April/
├── .env                              # API keys (gitignored)
├── .gitignore
├── HACKATHON_TODO.md
├── PHASE2_CURRENT_STATE.md           # ← THIS FILE
├── fileStructure.txt
├── requirements.txt
├── requirements-windows.txt
├── agent/
├── backend/
├── docs/
├── frontend/
├── inference_outputs/
├── model/
├── phase2/
├── scripts/                          # ← Data pipeline lives here
│   ├── nass_pull.py                  # ✅ written, partial run done
│   ├── nass_features.py              # ⬜ to write
│   ├── nass_corn_5states_2015_2024.csv  # ✅ partial output
│   ├── ndvi_county_extraction.js     # ⬜ save GEE script here for VC
│   ├── gssurgo_extract.py            # ⬜ to write
│   ├── prism_pull.py                 # ⬜ to write
│   ├── prism_features.py             # ⬜ to write
│   └── merge_all.py                  # ⬜ final join on (GEOID, year)
├── smoke_agent.py
├── smoke_tools.py
└── preview.jpg
```

---

## Decisions Made (with rationale)

1. **County-level aggregation, not pixel-level** — matches NASS yield
   reporting unit; avoids massive computation for marginal gain.
2. **Server-side aggregation in Earth Engine** — output CSV is already
   summarized to county-year, eliminating need for local zonal stats on
   imagery.
3. **Separate raw-pull script from feature-engineering script** — API calls
   are slow/rate-limited; features iterate fast. Splitting them prevents
   re-hitting API on every feature change.
4. **Use Valu1 table for soil instead of manually aggregating components and
   horizons** — Valu1 is USDA's official pre-aggregation; covers 90% of
   common ML use cases including the corn-specific NCCPI index.
5. **Skip hourly weather data, day length, CO₂** — see "Identified but
   intentionally skipped" section above.
6. **Use combined-practice yield as target** — `yield_bu_acre_all` is the
   most reliable column with full coverage; irrigation effect captured via
   `irrigated_share` feature where data exists.
7. **TIGER/Line 2018 county boundaries** — used in Earth Engine script for
   spatial reduction. Stable across 2015–2024 time range.

---

## Immediate Next Steps

In recommended order:

1. **Wait for Earth Engine NDVI export to finish** (30–90 min); verify CSV
   lands in Google Drive. If 2024 CDL fails, set `endYear = 2023` and rerun.
2. **Write `nass_features.py`** — quick local processing of existing NASS
   CSV. Produces `nass_corn_5states_features.csv` with `yield_target`,
   `irrigated_share`, `harvest_ratio`, etc.
3. **Write `gssurgo_extract.py`** — read .gdb files, pull Valu1 + run
   zonal stats over TIGER counties → per-county soil features CSV.
4. **Write `prism_pull.py` + `prism_features.py`** — daily climate data,
   derive GDD/EDD/VPD per growth phase per county.
5. **Write `merge_all.py`** — outer-join all features on `(GEOID, year)` →
   single training table.
6. **Begin baseline modeling** — XGBoost or LightGBM on the merged table,
   target = `yield_target`. Cross-validate by year (train on 2015–2022,
   test on 2023–2024) for honest temporal generalization.

## Optional Future Work

- Re-run `nass_pull.py` with rate-limit fixes after >30 min cooldown to fill
  in non-irrigated columns (low priority — derivable from existing data).
- Add severe weather / hail data from NOAA Storm Events (replaces tornado
  data with broader, more relevant signal).
- Add Sentinel-2 NDVI as higher-resolution complement to MODIS (10m vs 250m).
- Add fertilizer/management data from USDA ARMS.
- Add planting date estimates from CDL phenology or Crop Progress reports.
- Process-based crop model integration (APSIM/DSSAT) for counterfactual
  reasoning (the "design optimal" part of the project goal).

---

## Key Technical Notes for Future Sessions

### Join key convention
All datasets join on `GEOID` (5-digit string: state FIPS zero-padded to 2 +
county FIPS zero-padded to 3) and `year` (integer).

### Key data quirks to remember
- NASS suppresses data for low-corn counties → expect Colorado mountain
  counties to drop out
- NASS rate limits aggressively at sub-1s request intervals (Azure gateway)
- gSSURGO uses Albers Equal Area projection (EPSG:5070); reproject before
  zonal stats
- MODIS NDVI scale factor is 0.0001 (raw values are 0–10000)
- CDL value 1 = corn (used for masking in Earth Engine)
- Earth Engine `ee.Number` cannot be JS-concatenated into asset paths;
  use `ImageCollection.filter()` instead

### User's coding style preferences (from memory)
- Minimal abstractions, explicit logic, flat structure
- Inline comments
- Avoid heavy frameworks unless necessary
- C++, Python, Java, JavaScript familiar