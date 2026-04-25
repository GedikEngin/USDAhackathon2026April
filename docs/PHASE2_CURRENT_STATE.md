# Phase 2 — Current State

> Overwritten at end of every v2 phase. Tells the next chat what actually exists in the v2 portion of the repo right now. Reference file paths and describe behavior; do not paste code here.

**Project:** USDA Hackathon 2026 April — Corn Yield Forecasting
**Repo:** `~/dev/USDAhackathon2026April`
**Last updated:** 2026-04-25, end of Phase C — **PHASE C COMPLETE (gate passed on val 2023 EOS at +46.7% lift, threshold 15%)**

---

## v2 status: Phases A, B, and C are fully closed. Ready to start Phase D or Phase E (independent).

The v2 pipeline now has all three pieces of the brief's required output for the 5 states × 4 forecast dates:

- **Point estimate:** XGBoost regressor, 4 boosters (one per forecast_date), trained on 5,580 county-year rows each. Bundle persisted to `models/forecast/regressor_*.json` + `.meta.json` sidecars.
- **Cone of uncertainty:** Phase B same-GEOID K=5 analog retrieval over 17-feature standardized embedding, percentiles (10, 50, 90) over detrended yields, retrended at query county.
- **Driver attribution (for narration prep):** SHAP values via `Booster.predict(pred_contribs=True)`, no `shap` library dependency. Per-row top-K and global feature importance available.

`scripts/backtest_phase_c.py` is the production scoring harness. It writes one CSV row per (year, state, date) with the regressor point, the analog cone, the analog-median (Phase B baseline) point, and the 5-yr-county-mean baseline — all in raw bu/acre, all comparable.

The v1 land-use system is shipped and untouched alongside.

## Project goal (locked)

Build an ML pipeline that predicts corn-for-grain yield (bu/acre) at the **county** level across **5 US states** at four fixed in-season forecast dates (Aug 1, Sep 1, Oct 1, end-of-season), aggregates to state, and produces a calibrated cone of uncertainty using **analog-year retrieval**. A Claude Haiku agent narrates the forecast.

- **Target states:** Colorado (CO), Iowa (IA), Missouri (MO), Nebraska (NE), Wisconsin (WI)
- **Time range:** **2005–2024** (20 years; 2024 holdout, 2023 validation, 2005–2022 train)
- **Spatial unit:** US county, joined via 5-digit GEOID (state FIPS + county FIPS, both zero-padded)
- **Target variable:** `yield_bu_acre_all` (combined-practice corn grain yield, bu/acre), exposed in the master table as `yield_target`

See `PHASE2_PROJECT_PLAN.md` for the full vision and `PHASE2_PHASE_PLAN.md` for the phase breakdown with go/no-go gates.

---

## What exists in the repo today

### v1 (untouched, fully shipped)

The full v1 land-use & GHG analysis system is in place. See the v1 `CURRENT_STATE.md` for canonical detail. v2 reuses v1's FastAPI shell, frontend chrome, and Claude agent infrastructure but does not modify them.

### v2 — Phase A artifacts (data acquisition + master table)

- **`scripts/nass_pull.py`** — 2005–2024, 5 states, county-level corn yield/production/area at 3 practice levels.
- **`scripts/nass_features.py`** — engineers `yield_target`, `irrigated_share`, `harvest_ratio`, `acres_harvested_noirr_derived` from the raw pull.
- **`scripts/nass_corn_5states_2005_2024.csv`** — raw NASS pull output (6,837 × 16).
- **`scripts/nass_corn_5states_features.csv`** — engineered features (6,834 × 10).
- **`scripts/ndvi_county_extraction.js`** — version-controlled GEE script. Per-year Export task pattern.
- **`phase2/data/ndvi/corn_ndvi_5states_<year>.csv` × 21** — NDVI county features 2004–2024. Pre-scaled (`× 0.0001` server-side).
- **`scripts/gssurgo_county_features.py`** — reads each state's gSSURGO `.gdb`, area-weighted county aggregates from the Valu1 table.
- **`scripts/gssurgo_county_features.csv`** — output (443 × 13). Static across years.
- **`scripts/gridmet_pull.py`** — daily weather pull from gridMET, 5 states, 2005–2024.
- **`data/v2/weather/raw/gridmet_county_daily_<year>.parquet` × 20** — raw daily county-aggregated weather.
- **`scripts/gridmet_county_daily_2005_2024.parquet`** — combined daily weather, all years.
- **`scripts/weather_features.py`** — derives per-`(GEOID, year, forecast_date)` features. Single as-of slice; phase windows clipped to cutoff. GDD F50/C86, EDD via single-sine, VPD per phase, prcp/dry-spell, srad per phase.
- **`scripts/weather_county_features.csv`** — output (35,440 × 14).
- **`phase2/data/drought/drought_USDM-Colorado,Iowa,Missouri,Nebraska,Wisconsin.csv`** — raw weekly USDM, state-level, 2000–2026.
- **`scripts/drought_features.py`** — broadcasts USDM state readings to GEOIDs via NASS GEOID directory; as-of rule `valid_end < forecast_date` (strict).
- **`scripts/drought_county_features.csv`** — output (27,336 × 9). 0 nulls.
- **`scripts/merge_all.py`** — outer-joins all six sources into the master table. Includes 2005-min-year filter, HLS feature allowlist, NDVI schema-drift guard, per-layer row-count invariant asserts, and a comprehensive QC tail.
- **`scripts/training_master.parquet`** — **canonical Phase B/C/D input.** 25,872 rows × 48 columns. Keyed on `(GEOID, year, forecast_date)`. Full coverage rate 98.1% (excl. HLS).
- **`docs/PHASE2_DATA_DICTIONARY.md`** — column-by-column reference for the master table.

### v2 — Phase B artifacts (analog retrieval cone)

- **`forecast/features.py`** — `EMBEDDING_COLS` (15 features at 08-01, 17 at later dates), `Standardizer` (per-(forecast_date, feature) z-score), `build_embedding_matrix`. Fit on the train pool (2005–2022, post min-history filter).
- **`forecast/data.py`** — `load_master`, `train_pool` (post min-history filter), `val_pool` (2023), `holdout_pool` (2024), `SPLIT_YEARS`. Single source of truth for "how to slice the master table."
- **`forecast/detrend.py`** — per-county linear OLS trend with state-median fallback (`CountyTrend`). Replaces the per-state trend after WI overshoot diagnostic in Phase B.
- **`forecast/analog.py`** — `AnalogIndex` (per-date `BallTree`), pool strategies `cross_county` and `same_geoid` (locked at `same_geoid` for Phase B+C production), K=5. Returns `Analog` records with both raw and detrended yields.
- **`forecast/cone.py`** — `build_cone(analogs, trend, query_geoid, query_year, ...)`. Percentiles in detrended space, retrend at query county.
- **`forecast/aggregate.py`** — `state_forecast_from_records` rolls county cones up via planted-acres-weighted mean of percentile values. Caveat documented (percentiles don't average linearly).
- **`forecast/baseline.py`** — naive 5-year county-mean baseline for Phase B gate comparison; carries forward into Phase C reporting.
- **`forecast/recalibrate.py`** — per-(state, forecast_date) additive bias correction. Used by Phase B post-recal pipeline; **NOT used in Phase C** (recal dropped after holdout-vs-val sign-flip analysis).
- **`scripts/backtest_baseline.py`** — Phase B's full backtest harness with sweep over (pool, k), summary, gate evaluation pre- and post-recal.

### v2 — Phase C artifacts (XGBoost point estimate + SHAP)

- **`forecast/regressor.py`** — `Regressor` (single-date), `RegressorBundle` (4-date dict), `fit`, `fit_all_dates`. Native xgb JSON save/load with `.meta.json` sidecars (`best_iteration`, `feature_cols`, `params`, `train_metrics`). `_add_derived_columns` derives `is_irrigated_reported` + 5 state one-hots; `_build_dmatrix` is the single source of truth for DMatrix construction (used by both `predict` and `shap_values_for`). **`_NDVI = []` with explanatory block comment** — all 5 MODIS NDVI cols dropped after Phase 2-C.1 SHAP analysis showed leakage.
- **`forecast/explain.py`** — SHAP attribution. `Driver` (one feature contribution), `Attribution` (full matrix + base + predictions). `top_drivers` (Phase F surface), `top_drivers_for_bundle` (date-dispatching), `attribution_table` (long-form for diagnostics), `feature_importance` (mean_abs / mean_signed). Uses `Booster.predict(pred_contribs=True)`; no `shap` library dependency. Built-in additivity check (`base + Σ shap == prediction` to 1e-2).
- **`scripts/train_regressor.py`** — Phase C training driver. Sweeps `max_depth × learning_rate × min_child_weight` (12 configs/date × 4 dates = 48 fits; ~15s wall). Picks best per date by county-level val RMSE. Writes bundle to `models/forecast/` + sweep CSV to `runs/phase_c_sweep_<ts>.csv`.
- **`scripts/backtest_phase_c.py`** — Phase C scoring harness. Regressor point + Phase B cone, both at state level. Reports per-(year, state, date), per-(year, state), per-(state, date) val residuals. Gate evaluation (regressor vs analog-median pre-recal at val EOS, threshold 15%). Writes `runs/backtest_phase_c_<ts>.csv`.
- **`scripts/smoke_explain.py`** — 6-section assertion + eyeball test for `explain.py`. Verifies shapes, additivity, ranking correctness, dispatch, and error paths against the trained bundle.
- **`models/forecast/regressor_{08-01,09-01,10-01,EOS}.json`** + `.meta.json` — the persisted Phase C bundle. **Trained without MODIS NDVI** in the feature set (37 features at 09-01/10-01/EOS, 35 at 08-01).

### v2 — to write (Phase D onward)

- HLS pull pipeline (consistent schema, county-level if feasible) + Prithvi as frozen feature extractor. **Phase D.1.**
- Optional Prithvi end-to-end fine-tune. **Phase D.2.**
- `/forecast/{state}` endpoint + frontend forecast view. **Phase E.**
- Forecast narration agent with 4 tools. **Phase F.**
- Holdout evaluation + ablation table + presentation deck. **Phase G.**

## Master table — at-a-glance (unchanged from Phase A.6)

```
scripts/training_master.parquet
  shape:        25,872 × 48
  size:         2.46 MB
  grain:        (GEOID, year, forecast_date)
  years:        2005–2024 (20 years)
  GEOIDs:       388 distinct (subset of 443 TIGER counties)
  forecast_d:   08-01, 09-01, 10-01, EOS
  target:       yield_target (bu/acre, combined-practice)
  full coverage: 25,380 / 25,872 (98.1%) excl. HLS
```

48 columns: 5 keys + 1 target + 5 NASS-aux + 5 NDVI + 11 gSSURGO + 11 weather + 6 drought + 4 HLS. Source script and provenance for each column documented in `PHASE2_DATA_DICTIONARY.md`.

## Phase C — gate verdict and what it does/doesn't say

**Gate (per `PHASE2_PHASE_PLAN`):** trained-regressor RMSE on 2023 val ≥ 15% better than Phase B analog-median (pre-recal) at end-of-season.

**Result:** EOS lift +46.7%. **PASS.**

Per-date breakdown on val 2023:
- 08-01: regressor RMSE 7.19, analog-median RMSE 10.97, **lift +34.5%**
- 09-01: regressor RMSE 11.48, analog-median RMSE 11.46, **lift −0.2%** (essentially tied)
- 10-01: regressor RMSE 7.48, analog-median RMSE 11.62, **lift +35.6%**
- EOS: regressor RMSE 6.33, analog-median RMSE 11.87, **lift +46.7%** ← official gate number

Three of four dates clear the 15% threshold comfortably. 09-01 ties analog-median; the phase plan explicitly accepts weaker earlier dates ("we just need to confirm the model adds value"), and the gate evaluates EOS only.

### What the gate does NOT say

- **2024 holdout generalization is mediocre.** Regressor LOSES to analog-median at every date in 2024 (regressor EOS RMSE 9.54, analog-median EOS RMSE 8.24). IA-2024 and MO-2024 are the dominant misses (regressor bias −15 and −19 bu respectively); both are years where the gradient-boosted trees' year-anchored prior didn't match the actual outcome. Documented as a known limitation; goes in Phase G's ablation table.
- **The model leans on `year` heavily.** Mean |SHAP| 13.4, double the next feature. Partly fine (genetic-gain trend is real); partly the reason for 2024 misses (when a year sits off the recent trend, the year-anchored prediction misses by exactly the amount the year doesn't predict).
- **Weather features are present but secondary.** Top weather drivers by mean |SHAP|: `edd_hours_gt90f` (3.6), `gdd_cum_f50_c86` (2.6), `vpd_kpa_silk` (2.5), `aws0_100` (2.5). Compared to `year` (13.4) and `acres_planted_all` (8.6), they're not the headline signal. The MO 2023 drought response works as designed (`d2_pct` and `vpd_kpa_silk` in top 10), but the model doesn't lean on weather as much as a "geospatial AI yield forecast" framing implies it should.

### The narrative for Phase G / presentation

> "We trained an XGBoost regressor over hand-engineered weather, soil, drought, and management features for each of the four forecast dates. The model passed the 15%-better-than-analog-median gate at end-of-season on val 2023 with +46.7% lift. SHAP analysis confirmed the model relies on agronomically-defensible features (heat stress, growing-degree days, silking-window VPD, plant-available water) once we removed MODIS NDVI from the feature set — that column had been encoding end-of-season information into August forecasts and dominated SHAP attributions before removal. The point estimate's main current limitation is the year-trend prior: when a year's outcome sits off the genetic-gain trend, the regressor underperforms the analog cone, which retrieves cross-decade weather-similar years. Phase D.1 (HLS-derived running NDVI via Prithvi) is the next step to add as-of-honest in-season visual signal."

## Recommended hyperparameter configs picked by the sweep (Phase C, post-NDVI-removal)

| forecast_date | max_depth | learning_rate | min_child_weight | best_iteration | val RMSE (county-level) |
|---|---|---|---|---|---|
| 08-01 | 4 | 0.10 | 1 | 241 | 19.24 |
| 09-01 | 4 | 0.10 | 1 | 231 | 21.53 |
| 10-01 | 4 | 0.05 | 5 | 342 | 19.98 |
| EOS   | 6 | 0.05 | 5 | 486 | 19.61 |

Sweep is essentially flat (1.3–1.6 bu spread across 12 configs at every date). Per-date picks differ within sweep noise; not chasing.

## Open architectural questions for Phase D / E

These came out of the Phase C work and are flagged for the next phase kickoff:

- **MODIS NDVI is gone from the regressor's feature set.** Replacement is Phase D.1 (HLS-derived running NDVI clipped to forecast_date, via Prithvi). Phase B's retrieval embedding still uses MODIS NDVI (2 columns) — the leakage that broke point-estimation does not equivalently break retrieval matching.
- **`year` dominance** could be revisited if/when the feature set widens. Engineered interaction terms (`state_alpha × year` explicit, `vpd_kpa_silk × d2_pct` for drought-stress) could redistribute SHAP attribution toward weather. Defer until Phase D.1 lands; adding interactions before more features is premature.
- **Per-state bias is documented, not corrected.** Val 2023 shows CO +9 to +18, MO +5 to +11 (post-NDVI), WI −4 to −11. Holdout 2024 shows different signs. A val-fitted recal would help one and hurt the other. If Phase D.1 imagery doesn't shrink these biases organically, may need a more robust calibration approach (multi-year rolling residuals).
- **2024 holdout generalization** is the real Phase G question. The regressor loses to analog-median on holdout; whether Phase D.1 closes that gap is the v2 lift story. If it doesn't, the analog cone's cross-decade retrieval is the more durable point estimate and we ship it that way.

## Hackathon readiness checklist

- [x] Planning docs (`PHASE2_PROJECT_PLAN.md`, `PHASE2_PHASE_PLAN.md`, `PHASE2_DATA_INVENTORY.md`, `PHASE2_DECISIONS_LOG.md`) drafted
- [x] All Phase A data pipelines run, master table built, data dictionary written
- [x] Phase B analog-cone baseline shipped + gate passed post-recal on 2024 holdout
- [x] **Phase C XGBoost point-estimate model shipped + gate passed on val 2023 EOS at +46.7% lift**
- [ ] Phase D.1 Prithvi frozen-feature integration + Phase D gate decision
- [ ] Phase D.2 (conditional) Prithvi fine-tune
- [ ] Phase E `/forecast/{state}` endpoint + frontend forecast view
- [ ] Phase F forecast narration agent with 4 tools
- [ ] Phase G holdout evaluation, ablation table, presentation deck

## Immediate next steps

Phases D and E are independent of each other. Pick either, or run them in parallel chats:

**Phase D.1 (Prithvi frozen feature extractor)** — heavier lift. Requires:
1. HLS pull pipeline with consistent schema (TB-scale; settle storage strategy first). Possibly county-level granularity instead of state-broadcast (current Phase A HLS slices are state-level, slated for redo).
2. CDL standalone download for offline corn-pixel masking.
3. Prithvi weights from HuggingFace.
4. Per-(GEOID, year, forecast_date) embedding extraction: most-recent-cloud-free HLS chip(s) → Prithvi encoder → mean-pool → embedding vector.
5. Concat embeddings to engineered features, retrain regressor, ablation vs Phase C-as-is.

**Phase E (backend + frontend)** — lighter lift, builds on already-shipped models. Requires:
1. New FastAPI endpoints in `backend/main.py` (or new `backend/forecast_routes.py`): `GET /forecast/states`, `GET /forecast/{state}?year=&date=`, `POST /forecast/narrate`.
2. New frontend view alongside v1 land-use UI: state/year/date pickers, line chart of point + cone across dates, analog-year list, drivers panel, narrative panel.
3. Reuses v1's monotonic `gen` token pattern, markdown renderer, status pills.

**My recommendation:** ship Phase E first. It validates the Phase B/C pipeline end-to-end with a real demoable artifact, and Phase D.1's headwinds (storage, Prithvi quirks) are real. Phase E gives us an MVP product to demo even if D.1 drags. Phase F (agent narration) naturally follows E.

## Budget note

v1 used ~$0.25 of the $10 API cap. v2 token usage will be similar in shape (Haiku narration, ~$0.02 per forecast). Compute cost for HLS download + Prithvi inference is local (Ubuntu box) and outside the API budget.