# Phase Plan v2 — Corn Yield Forecasting

> The canonical phase breakdown for v2. `PHASE2_PROJECT_PLAN.md` is the vision doc. `PHASE2_DECISIONS_LOG.md` tracks what actually happened. `PHASE2_CURRENT_STATE.md` reflects the repo right now. `PHASE2_DATA_INVENTORY.md` tracks data status.
>
> **When this file and `PHASE2_DECISIONS_LOG.md` disagree, the decisions log wins.**

---

## How to start each phase chat

Open a new chat in this Claude Project. First message:

> Starting Phase 2-X. Read project knowledge, confirm you're oriented, then let's go.

Claude will read `PHASE2_PROJECT_PLAN.md`, this file, `PHASE2_DECISIONS_LOG.md`, `PHASE2_CURRENT_STATE.md`, and `PHASE2_DATA_INVENTORY.md`, confirm context, and begin the phase.

## How to end each phase chat

> Update PHASE2_DECISIONS_LOG.md, PHASE2_CURRENT_STATE.md, and (if data status changed) PHASE2_DATA_INVENTORY.md with what we just did. Give me the full file contents to paste in.

---

## Phase index

- **Phase A — Data acquisition & cleaning** (currently in progress)
- **Phase B — Feature engineering & analog-year retrieval baseline** (cone-of-uncertainty MVP)
- **Phase C — Point-estimate model** (XGBoost / LightGBM baseline)
- **Phase D — Prithvi integration** (frozen feature extractor → optional fine-tune)
- **Phase E — Backend endpoints + frontend forecast view**
- **Phase F — Agent tools for forecast narration**
- **Phase G — Validation, ablations, presentation prep**

Explicit **GO/NO-GO checkpoints** at the end of B, C, and D.

---

## Phase A — Data acquisition & cleaning

**Goal:** every dataset listed in `PHASE2_DATA_INVENTORY.md` is on disk, parsed into a consistent schema, and joinable on `(GEOID, year)` or `(GEOID, year, forecast_date)`.

The Phase A pipeline follows the script-level structure already established in the repo: each external data source has a separate `*_pull.py` (raw acquisition, slow, rate-limited) and `*_features.py` (local processing, fast iteration). Final join in `merge_all.py`.

### A.1 — NASS yield + auxiliary, extended to 2005

- `scripts/nass_pull.py` — already written. Rate-limit fixes (2.0s delay, exponential backoff up to 240s, 4 retries) committed. Currently has 2,998 rows × 12 cols for 2015–2024 with full coverage on combined-practice columns.
- **Action:** re-run for 2005–2024 to extend the history backward. The brief specifies 2005-2024 explicitly. ~2× more training years means ~2× analog-year candidates, which makes Phase B materially stronger.
- **Action:** write `scripts/nass_features.py` — pure local processing on the raw CSV. Produces `scripts/nass_corn_5states_features.csv` with derived columns:
  - `yield_target = yield_bu_acre_all`
  - `irrigated_share = (acres_harvested_irr / acres_harvested_all).fillna(0).clip(0,1)`
  - `harvest_ratio = (acres_harvested_all / acres_planted_all).clip(0,1)`
  - `acres_harvested_noirr_derived = acres_harvested_all - acres_harvested_irr.fillna(0)`
- **Schema:** keys `GEOID, state_alpha, year`; outputs above; coverage notes documented inline.
- **Deliverable:** parquet/CSV with no missing years for major corn counties; QC report on county coverage by year.

### A.2 — MODIS NDVI via Earth Engine (already in flight)

- Earth Engine script written, debugged, export task `corn_ndvi_5states_2015_2024` submitted to Google Drive.
- **Action:** when current export lands, **re-run for 2005–2014** to backfill (server-side; cheap). New task name suggested: `corn_ndvi_5states_2005_2014`.
- **Action:** save the GEE script into `scripts/ndvi_county_extraction.js` for version control. Currently lives only in the GEE web editor.
- **Output schema (already locked):** `GEOID, NAME, STATEFP, year, ndvi_peak, ndvi_gs_mean, ndvi_gs_integral, ndvi_silking_mean, ndvi_veg_mean`.
- **CDL caveat:** CDL 2024 may not exist yet (publishes Jan/Feb of following year). If 2024 fails, set `endYear=2023`. CDL 2005 is the earliest available; corn-pixel masking works back to 2005.
- **Deliverable:** combined `corn_ndvi_5states_2005_2024.csv` keyed on `(GEOID, year)`.

### A.3 — gSSURGO soil features

- State `.gdb` files already downloaded (10m resolution) for all 5 states. Skipped the CONUS download (only 30m, >40GB).
- **Action:** write `scripts/gssurgo_extract.py`:
  1. Read `.gdb` with GDAL/OGR via geopandas + rasterio.
  2. Use the **Valu1 table** that ships inside each `.gdb` — already aggregated to MUKEY level (no need to manually weight components/horizons). Valu1 covers 90% of common ML use cases including the corn-specific NCCPI index.
  3. Key Valu1 columns for corn: `nccpi3corn`, `aws0_100`, `soc0_30`, `rootznemc`, `droughty`. Add depth-weighted texture, pH, OM, CEC if needed.
  4. Build per-property GeoTIFF rasters by remapping MUKEY → property value.
  5. Run zonal statistics over TIGER/Line 2018 county polygons to get one number per county per property.
- **Reproject before zonal stats:** gSSURGO is in EPSG:5070 (Albers Equal Area).
- **Deliverable:** `scripts/gssurgo_county_features.csv` with `GEOID` + ~6–10 soil columns. Static across years (soil doesn't change with year), so this table joins on `GEOID` alone.

### A.4 — Daily weather (PRISM or gridMET)

- **Action:** write `scripts/prism_pull.py` for 5 states, **2005–2024**, daily.
  - Variables: `tmax, tmin, prcp, srad, vp` daily.
- **Action:** write `scripts/prism_features.py` to derive:
  - **Growing Degree Days (GDD)** with base 50°F, cap 86°F (corn standard).
  - **Cumulative GDD** from May 1 through each forecast date.
  - **EDD/KDD** hours above 86°F or 90°F (heat stress, especially during silking).
  - **VPD** (vapor pressure deficit).
  - **Cumulative precip** May 1 → forecast date, plus dry-spell length.
  - Solar radiation totals.
- **Critical:** all aggregations respect the **as-of rule** — when constructing features for forecast date `D` in year `Y`, use only data with timestamps strictly before `D`. Enforced at the feature-construction layer.
- **Deliverable:** `scripts/prism_county_features.csv` keyed on `(GEOID, year, forecast_date)`.

### A.5 — US Drought Monitor features

- ✅ **Data acquired** in Cumulative Percent Area format (5 states, weekly, 2005–2024, county level). Each D-level column reports % of county at that level *or worse*.
- **Action:** write `scripts/drought_features.py`. No external pull; pure local processing on the USDM CSV.
- Derived features per `(GEOID, year, forecast_date)`:
  - **Most-recent reading**: D0/D1/D2/D3/D4 cumulative percentages as-of the last USDM Thursday strictly before the forecast date.
  - **DSCI** (Drought Severity Coverage Index): `D0 + D1 + D2 + D3 + D4` cumulative sum, range 0–500. Strong candidate for the retrieval embedding.
  - **Season-cumulative drought weeks**: count of weeks since May 1 where D2 ≥ 50% (sustained-stress signal).
  - **Peak DSCI during silking** (DOY 196–227): max DSCI in that 4-week window. Silking is when corn is most water-sensitive.
- **As-of join rule:** USDM week-ending dates are Thursdays. Use the most recent Thursday strictly *before* the forecast date (not ≤) to avoid same-week leakage.
- **Deliverable:** `scripts/drought_county_features.csv` keyed on `(GEOID, year, forecast_date)`.

### A.6 — Master training table

- **Action:** write `scripts/merge_all.py`. Outer-joins:
  - `nass_corn_5states_features.csv` on `(GEOID, year)`
  - `corn_ndvi_5states_2005_2024.csv` on `(GEOID, year)`
  - `gssurgo_county_features.csv` on `(GEOID,)` — broadcasts soil features across all years
  - `prism_county_features.csv` on `(GEOID, year, forecast_date)`
  - `drought_county_features.csv` on `(GEOID, year, forecast_date)`
- **Schema:**
  - keys: `GEOID, state_alpha, year, forecast_date`
  - target: `yield_target` (state-aggregated; null at forecast time, populated for training rows)
  - features: ~30–50 columns spanning weather, drought, soil, NDVI, NASS auxiliary
- **Deliverable:** `scripts/training_master.parquet`.

### A.7 — Data dictionary

- **Action:** produce `PHASE2_DATA_DICTIONARY.md` documenting every column in `training_master.parquet` — units, source, date range, computation method, any caveats.

**Phase A definition of done:** `training_master.parquet` loads, has no surprise nulls in feature columns, and survives a 5-row spot check against original sources.

---

## Phase B — Feature engineering & analog-year retrieval (cone-of-uncertainty MVP)

**Goal:** ship a working cone-of-uncertainty forecast **without any ML model**. Pure feature engineering + nearest-neighbor retrieval.

### B.1 — Standardize feature vectors

- For each `(GEOID, year, forecast_date)` row, build a fixed-length feature vector. Standardization is per-feature z-score over the training set (excluding holdout).
- Decide which subset of features goes into the **retrieval embedding** vs. which are kept as covariates only. Conservative starting set for retrieval: `[cum_GDD, cum_precip, drought_severity, ndvi_gs_mean_to_date, planted_acres_ratio, soil_aws0_100]`.
- **Deliverable:** `forecast/features.py` with `build_feature_vector(geoid, year, date)` and `embedding_for_retrieval(features)`.

### B.2 — Nearest-neighbor retrieval

- For a query `(geoid_q, year_q, date_q)`, find the K nearest historical `(geoid_h, year_h, date_q)` neighbors by L2 distance over the standardized retrieval embedding. K = 10 baseline.
- **Constraint:** when forecasting year Y, exclude all year-Y rows from the candidate pool (no temporal leakage).
- **Deliverable:** `forecast/analog.py::find_analogs(geoid, year, date, k=10) -> list[Analog]` where `Analog = {geoid, year, distance, observed_yield_bu_acre}`.

### B.3 — Cone construction

- Given K analogs, the cone at percentile p is the p-th percentile of the K observed yields.
- Default cone: 10th, 50th, 90th.
- "Point estimate" at this stage = median of analog yields. (Phase C replaces with regression.)
- **Deliverable:** `forecast/cone.py::build_cone(analogs, percentiles=[10,50,90])`.

### B.4 — State aggregation

- Roll up county-level point estimates and cones to state level via planted-acres-weighted mean of percentile values. (Strictly speaking, percentiles don't average linearly; for the baseline this is acceptable — document the caveat.)
- **Deliverable:** `forecast/aggregate.py::state_forecast(state, year, date)`.

### B.5 — Backtesting harness

- Hold out 2024 (and 2023 as a second holdout). Run the full forecast pipeline for every (state, holdout_year, date). Score against actual NASS state yield.
- Metrics: RMSE on point estimate, coverage% of the 80% cone (target ~80%), mean cone width.
- **Deliverable:** `scripts/backtest_baseline.py` + a results table.

### **🔴 GO/NO-GO checkpoint — end of Phase B**

Both gates:

1. Backtest 80% cone coverage on 2023+2024 between **70% and 90%** (cone is honestly calibrated, not absurdly wide or narrow).
2. RMSE on point estimate **better than the naive "5-year county mean" baseline**.

If either fails: revisit feature selection, K, or which features go in the retrieval embedding before proceeding to Phase C. Do NOT add ML complexity on top of a broken baseline.

---

## Phase C — Point-estimate model (XGBoost / LightGBM)

**Goal:** replace the analog-median point estimate with a trained gradient-boosted regressor. Keep the analog cone as the uncertainty quantification.

### C.1 — Train/val/holdout split

- Train: 2005–2022, all counties.
- Val: 2023 (hyperparameter selection, early stopping).
- Holdout: 2024 (touched once at the end of Phase G).
- **Deliverable:** documented split + `forecast/data.py::load_split(name)`.

### C.2 — Model training

- Train one model per forecast date (Aug 1, Sep 1, Oct 1, end-of-season). Same features as Phase B plus any covariates held back from the retrieval embedding.
- Modest hyperparameter sweep: `max_depth ∈ {4,6,8}`, `n_estimators` early-stopped, `learning_rate ∈ {0.05, 0.1}`.
- XGBoost or LightGBM — equivalent for this; pick one and stick with it.
- **Deliverable:** 4 trained models, `forecast/models/{model}_{date}.json` + a metrics CSV.

### C.3 — Replace point estimate, keep cone

- Forecast pipeline now: trained regressor gives the point; analog retrieval gives the cone. The two are *deliberately* not the same model — cone is interpretable, point is accurate.
- Sanity check: point estimate should usually fall inside the 10–90 cone. Log when it doesn't.

### C.4 — Driver attribution

- Compute SHAP values per prediction. Aggregate top 3 drivers per county-year-date for use by the agent.
- **Deliverable:** `forecast/explain.py::top_drivers(geoid, year, date, k=3)`.

### **🔴 GO/NO-GO checkpoint — end of Phase C**

Trained-regressor RMSE on 2023 val **at least 15% better than Phase B analog-median baseline** at end-of-season prediction. (Earlier dates can be weaker; we just need to confirm the model adds value.)

If gate fails: investigate feature leakage, target encoding, or whether some feature class is dominating the trees. Do NOT proceed to Prithvi if the simple model isn't beating the baseline.

---

## Phase D — Prithvi integration

**Goal:** add Prithvi-derived features to the model. The brief names the model; we use it.

### D.1 — Prithvi as frozen feature extractor (default path)

- Pull Prithvi from HuggingFace.
- For each `(GEOID, year, forecast_date)`, take the most recent cloud-free HLS chip(s) covering corn pixels in that county. Pass through Prithvi encoder, mean-pool to a single embedding vector per county-date.
- Concat Prithvi embeddings with the engineered feature vector. Retrain the gradient-boosted regressor.
- **Deliverable:** `forecast/prithvi.py::extract_embedding(hls_chip)` + retrained models + ablation table comparing engineered-only vs. engineered+Prithvi.

**Phase D.1 is conditional on the HLS pull being feasible.** HLS is TB-scale across 5 states × 20 years × growing-season scenes. Settle storage and idempotent download orchestration before pulling. The 2005–2013 era is **Landsat-only HLS** (no Sentinel-2 component until 2015), so revisit cadence is lower in early years; document the caveat.

### D.2 — Prithvi end-to-end fine-tune (stretch)

- Only if D.1 is shipped and there's signal Prithvi adds enough lift to justify the cost.
- Add a regression head, fine-tune on masked corn-pixel HLS chips against yield.
- **Deliverable:** fine-tuned checkpoint + ablation row in the metrics table.
- **Risk flag:** substantial effort. Decision to do D.2 is a separate go/no-go after D.1 ships.

### **🔴 GO/NO-GO checkpoint — end of Phase D.1**

Does Prithvi-augmented model beat engineered-only by **≥ 5% RMSE on 2023 val**?

- Yes → ship Prithvi as part of the production pipeline. Decide whether to attempt D.2.
- No → ship engineered-only as production and **report the ablation as a finding** ("Prithvi did not add lift over hand-engineered features at the county-year-date aggregation level"). This is a defensible result, not a failure.

---

## Phase E — Backend endpoints & frontend forecast view

### E.1 — New FastAPI endpoints

Add to `backend/main.py` (or a new `backend/forecast_routes.py` if it gets crowded):

- `GET /forecast/states` — list the 5 states + their 2025 forecast availability per date.
- `GET /forecast/{state}?year={year}&date={Aug1|Sep1|Oct1|EOS}` — returns:
  ```json
  {
    "state": "IA",
    "year": 2025,
    "forecast_date": "2025-08-01",
    "point_estimate_bu_acre": 192.4,
    "cone": {"p10": 174.1, "p50": 191.0, "p90": 208.7},
    "analog_years": [
      {"GEOID": "...", "year": 2014, "distance": 0.21, "observed_yield": 188.3},
      ...
    ],
    "top_drivers": [{"feature": "cum_GDD_to_date", "shap": 6.2, "direction": "+"}, ...],
    "model_version": "v2.0-xgb-prithvi"
  }
  ```
- `POST /forecast/narrate` — takes a forecast object, calls the agent, returns the narrative.

### E.2 — Frontend forecast view

New view alongside (not replacing) the v1 land-use UI. Reuses the chrome.

Components:
- State picker (5 buttons)
- Year picker (defaults to 2025)
- Forecast-date picker (4 buttons: Aug 1, Sep 1, Oct 1, EOS)
- Main panel: line chart of point estimate + shaded cone across the 4 dates for the selected state-year
- Side panel: list of analog years with distances and observed yields
- Drivers panel: top 3 drivers with SHAP values
- Narrative panel: rendered Claude output

### E.3 — Same-origin, same race-safety patterns as v1

Reuse the monotonic `gen` token pattern. Reuse the markdown renderer. Reuse the status pills.

---

## Phase F — Agent tools for forecast narration

**Goal:** Claude Haiku 4.5 produces a forecast narrative that names analog years, identifies drivers, and explains the trajectory.

### F.1 — Tool design

Four tools (mirrors v1 pattern):

1. `get_forecast_summary(state, year, date)` — returns the structured forecast (point, cone, model_version, basic context).
2. `get_analog_years(state, year, date, k=5)` — returns top K analog years with observed yields, distances, brief weather summaries.
3. `get_drivers(state, year, date, k=3)` — returns top K SHAP-ranked drivers with values and directions.
4. `compare_to_history(state, year, date)` — returns 5-year and 10-year mean state yield, plus where the current forecast sits (z-score, percentile).

### F.2 — System prompt

New persona — agricultural forecast analyst. Same structural conventions as v1 (cite tool outputs, surface uncertainty honestly, name analog years explicitly).

### F.3 — Wire to `/forecast/narrate`

Reuse v1's stateless agent endpoint pattern. Frontend echoes the forecast object back; backend builds a `ForecastState` and runs the agent loop.

---

## Phase G — Validation, ablations, presentation prep

### G.1 — Final holdout evaluation

Run the production pipeline on 2024 holdout. Report state-level RMSE, cone coverage, mean cone width per forecast date.

### G.2 — Ablation table

The defensibility story. Rows:

- Naive 5-year county mean
- Phase B analog-median
- Phase C XGBoost (engineered only)
- Phase D.1 XGBoost + Prithvi frozen embeddings
- (optional) Phase D.2 Prithvi fine-tuned

Columns: RMSE Aug 1 / Sep 1 / Oct 1 / EOS, cone coverage at 80%, mean cone width.

### G.3 — Presentation deck

5–7 minutes per the brief. Skeleton:

1. Problem (the brief, in 30s)
2. Pipeline architecture diagram (3 pillars)
3. Cone methodology — analog years, with one worked example
4. Validation: ablation table + 1 plot per holdout year
5. Live demo (or recorded fallback) hitting `/forecast/IA?date=Sep1`
6. Limitations and what we'd build next

### G.4 — README + handoff

Update top-level README to reference v2 alongside v1. New `PHASE2_README.md` if v2 deserves its own.

---

## Risk register

- **HLS data volume** for Phase D.1. Could be terabytes. Plan storage. May force scope cuts.
- **Prithvi integration complexity.** Foundation models have weird input expectations (band order, normalization, patch sizes). Plan a half-day buffer just for "make Prithvi accept our HLS chips correctly."
- **Cone calibration.** Analog-year retrieval is interpretable but not guaranteed to give well-calibrated coverage. Phase B gate exists precisely to catch this.
- **CDL spatial alignment.** CDL 30m, HLS 30m, but reprojections can introduce subpixel shifts that contaminate corn-only feature aggregation. Test alignment explicitly if HLS is pulled.
- **Temporal leakage.** When training, if features include data not available at the forecast date (e.g., October NDVI in the August forecast), the model looks great in val and dies in production. As-of rule enforced in `forecast/features.py`.
- **County coverage gaps.** Not every county grows enough corn to have a NASS yield reported every year. Filter to counties with ≥N years of complete data (decide N in Phase A.1 once full 2005–2024 pull is done).
- **NASS rate limiting.** Azure App Gateway throttles aggressive request patterns. Fixes already in `nass_pull.py`; the historical extension to 2005 will exercise them.
- **HLS 2005–2013 sparsity.** Landsat-only era; lower revisit cadence than 2015+. Phase D.1 should account for this when picking "most recent cloud-free chip" near forecast dates in early years.
- **gSSURGO size.** Aggregate to county-level features early in A.3; do not carry pixel-level soil data into modeling.

## Conventions (carried over from v1)

- All file paths in docs use repo-relative paths
- Decisions log is append-only
- Phase docs (CURRENT_STATE) overwritten at end of phase
- `PHASE2_DATA_INVENTORY.md` updated continuously, not just at phase boundaries
- Don't re-litigate locked decisions; if reversing, log it as a new decision
