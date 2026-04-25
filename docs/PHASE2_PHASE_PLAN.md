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

- **Phase A — Data acquisition & cleaning** (A.1–A.5 ✅, A.6 + A.7 remaining)
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

### A.1 — NASS yield + auxiliary, extended to 2005 ✅

- `scripts/nass_pull.py` — written. Rate-limit fixes (2.0s delay, exponential backoff up to 240s, 4 retries) committed. Run for full 2005–2024 range.
- `scripts/nass_features.py` — written. Pure local processing on the raw CSV. Produces `scripts/nass_corn_5states_features.csv` with derived columns:
  - `yield_target = yield_bu_acre_all`
  - `irrigated_share = (acres_harvested_irr / acres_harvested_all).fillna(0).clip(0,1)`
  - `harvest_ratio = (acres_harvested_all / acres_planted_all).clip(0,1)`
  - `acres_harvested_noirr_derived = acres_harvested_all - acres_harvested_irr.fillna(0)`
- **Schema:** keys `GEOID, state_alpha, year`; outputs above; coverage notes documented inline.
- **Outputs on disk:** `scripts/nass_corn_5states_2005_2024.csv` (6,837 × 16); `scripts/nass_corn_5states_features.csv` (6,834 × 10).

### A.2 — MODIS NDVI via Earth Engine ✅

- Earth Engine script written, debugged, ran for 2004–2024 (one task per year).
- `scripts/ndvi_county_extraction.js` — version-controlled in the repo.
- **Output schema (locked):** `GEOID, NAME, STATEFP, year, ndvi_peak, ndvi_gs_mean, ndvi_gs_integral, ndvi_silking_mean, ndvi_veg_mean`.
- **NDVI is pre-scaled** (× 0.0001 applied server-side); CSV values are floats in `[-0.2, 1.0]`. Downstream MUST NOT re-apply the scale factor.
- **CDL caveat:** uneven coverage 2005–2007 (CO essentially absent until 2008). Documented in `PHASE2_CURRENT_STATE.md`.
- **Outputs on disk:** 21 per-year CSVs in `phase2/data/ndvi/corn_ndvi_5states_<year>.csv`. To be concatenated by `merge_all.py`.

### A.3 — gSSURGO soil features ✅

- State `.gdb` files downloaded (10m resolution) for all 5 states in `phase2/data/gSSURGO/`. Skipped CONUS download (only 30m, >40GB).
- `scripts/gssurgo_county_features.py` — written, run for all 5 states.
- **Method:**
  1. Read each `.gdb` via geopandas/rasterio.
  2. Pull the **Valu1 table** that ships inside each `.gdb` (USDA's pre-aggregated MUKEY-level table).
  3. Build per-property GeoTIFFs by remapping MUKEY → property values.
  4. Reproject TIGER 2018 county polygons to EPSG:5070 (gSSURGO's native Albers).
  5. Run zonal statistics over county polygons.
- **Final column set:** `nccpi3corn, nccpi3all, aws0_100, aws0_150, soc0_30, soc0_100, rootznemc, rootznaws, droughty, pctearthmc, pwsl1pomu` (11 properties).
- **Output:** `scripts/gssurgo_county_features.csv` keyed on `GEOID` + `state_alpha` (443 rows × 13 cols, static across years; broadcast at merge time).

### A.4 — Daily weather (gridMET) ✅

- Picked **gridMET** over PRISM. Both work at county-aggregation scale; gridMET was easier to acquire programmatically. PRISM stays as a fallback if gridMET ever shows quality issues.
- `scripts/gridmet_pull.py` — written, run for 5 states, **2005–2024**, daily.
  - Variables: `tmax, tmin, prcp, srad, vp` daily.
  - Outputs: `data/v2/weather/raw/gridmet_county_daily_<year>.parquet` × 20, plus combined `scripts/gridmet_county_daily_2005_2024.parquet`. Cached netcdfs in `data/v2/weather/raw/_gridmet_nc_cache/` for idempotent re-runs.
- `scripts/weather_features.py` — written, run. Derives:
  - **GDD F50/C86** — Fahrenheit base 50 / cap 86, both endpoints capped before averaging (McMaster & Wilhelm / NDAWN convention; not the raw `tavg − 50` variant). Cumulative from May 1 → cutoff.
  - **EDD/KDD hours above 86°F / 90°F** via single-sine hourly interpolation (Allen 1976 / Baskerville-Emin 1969). Closed-form integral of the daily sine arc over the threshold.
  - **VPD** averaged over each phase window (vegetative DOY 152–195, silking DOY 196–227, grain fill DOY 228–273), clipped to cutoff.
  - **Cumulative precipitation** May 1 → cutoff plus longest dry-spell run (`<2 mm/day`).
  - **Solar radiation totals** per phase window, MJ/m².
- **As-of safety:** `build_features_for_cutoff(df, year, cutoff_date)` slices the daily df at the very top with `date <= cutoff_date`. Single point of leakage control. Phase windows clipped to `cutoff_doy`.
- **Output:** `scripts/weather_county_features.csv` (35,440 rows × 14 cols). Keys: `(GEOID, year, forecast_date)`.

### A.5 — US Drought Monitor features ✅

- **Source caveat discovered during this phase:** the USDM CSV is **state-level**, not county-level (header is `MapDate,StateAbbreviation,StatisticFormatID,ValidStart,ValidEnd,D0,D1,D2,D3,D4,None,Missing` — no county FIPS). Earlier planning notes saying "per-county" were aspirational. State readings are broadcast to GEOIDs at feature-derivation time.
- **USDM percentages are cumulative:** D0 ≥ D1 ≥ D2 ≥ D3 ≥ D4 by construction. Each column reports % of state at that level *or worse*.
- `scripts/drought_features.py` — written, run. No external pull; pure local processing.
- **Method:**
  1. Read the USDM CSV; rename `StateAbbreviation → state_alpha`; coerce `ValidEnd → valid_end`.
  2. Read `nass_corn_5states_features.csv` as the GEOID directory (provides every `(GEOID, state_alpha, year)` triple).
  3. For each `(state, year, forecast_date)` in `{08-01, 09-01, 10-01, EOS}`, find the most recent USDM reading whose `valid_end < forecast_date` (strict — strictly before, never on or after).
  4. Build full `(GEOID, year, forecast_date)` cartesian skeleton, left-join state features.
- **Final feature set (intentionally minimal):** `d0_pct, d1_pct, d2_pct, d3_pct, d4_pct, d2plus`. `d2plus` is a stable alias for `d2_pct` (severe-or-worse, exposed under a descriptive name because the cumulative convention is non-obvious).
- **DSCI / season-cum drought weeks / silking-peak DSCI deferred.** Easy adds if Phase B/C show the model wants more drought signal. See decisions log entry 2-A.5.
- **As-of comparison column:** `ValidEnd`. Strict `<` prevents same-week leakage when the USDM map's validity span brackets the forecast date.
- **Output:** `scripts/drought_county_features.csv` (27,336 rows × 9 cols, 0 nulls, 0 monotonicity violations). Keys: `(GEOID, year, forecast_date)`.

### A.6 — Master training table

- **Action:** write `scripts/merge_all.py`. Outer-joins:
  - `nass_corn_5states_features.csv` on `(GEOID, year)`
  - 21 NDVI per-year CSVs from `phase2/data/ndvi/corn_ndvi_5states_<year>.csv`, concatenated, on `(GEOID, year)`
  - `gssurgo_county_features.csv` on `GEOID` — broadcasts soil features across all years
  - `weather_county_features.csv` on `(GEOID, year, forecast_date)`
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
- Decide which subset of features goes into the **retrieval embedding** vs. which are kept as covariates only. Conservative starting set for retrieval: `[gdd_cum_f50_c86, prcp_cum_mm, d2plus, ndvi_gs_mean_to_date, irrigated_share, aws0_100]`.
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
- **Temporal leakage.** When training, if features include data not available at the forecast date (e.g., October NDVI in the August forecast), the model looks great in val and dies in production. As-of rule enforced in `weather_features.py` and `drought_features.py`.
- **County coverage gaps.** Not every county grows enough corn to have a NASS yield reported every year. Filter to counties with ≥N years of complete data (decide N after `merge_all.py` lands and a `notna().sum()` pass per county).
- **NASS rate limiting.** Azure App Gateway throttles aggressive request patterns. Fixes already in `nass_pull.py`.
- **HLS 2005–2013 sparsity.** Landsat-only era; lower revisit cadence than 2015+. Phase D.1 should account for this when picking "most recent cloud-free chip" near forecast dates in early years.
- **gSSURGO size.** Aggregate to county-level features early in A.3; do not carry pixel-level soil data into modeling. ✅ Done — `gssurgo_county_features.csv` is 443 rows.
- **Drought features are state-level.** USDM doesn't have county granularity in the source CSV — drought signal is constant within a state for a given week. May limit per-county analog matching value; if so, drop USDM from the retrieval embedding (keep as covariate) in Phase B.

## Conventions (carried over from v1)

- All file paths in docs use repo-relative paths
- Decisions log is append-only
- Phase docs (CURRENT_STATE) overwritten at end of phase
- `PHASE2_DATA_INVENTORY.md` updated continuously, not just at phase boundaries
- Don't re-litigate locked decisions; if reversing, log it as a new decision
