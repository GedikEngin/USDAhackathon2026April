# Current State — v2 Corn Yield Forecasting

> **End of Phase B (analog-year retrieval baseline + cone calibration).** Phase B gate PASSED post-recalibration on 2024 holdout. Ready to start Phase C (XGBoost point-estimate model).
>
> Last updated: 2026-04-25, end of Phase 2-B.

## TL;DR

- Phase A: ✅ done (master table built, 25,872 × 48, all NaNs documented)
- **Phase B: ✅ done** (analog-year retrieval baseline shipped + cone calibration; gate passed)
- Phase C: ⏳ next (XGBoost point-estimate, replaces analog-median; cone stays)
- Phase D.1: ⏳ pending (HLS pull + Prithvi frozen embeddings)
- Phase D.2 / E / F / G: ⏳ pending

## What's on disk

### Master training table (canonical Phase B/C/D input)

```
scripts/training_master.parquet
  shape:        25,872 × 48
  size:         2.46 MB
  grain:        (GEOID, year, forecast_date)
  years:        2005–2024
  states:       CO, IA, MO, NE, WI
  GEOIDs:       388 distinct (subset of 443 TIGER 2018 counties)
  forecast_d:   08-01, 09-01, 10-01, EOS
  target:       yield_target (bu/acre, combined-practice)
```

48 columns: 5 keys + 1 target + 5 NASS-aux + 5 NDVI-MODIS + 11 gSSURGO + 11 weather + 6 drought + 4 HLS. Source script and provenance for every column documented in `PHASE2_DATA_DICTIONARY.md`.

### Phase B forecast pipeline (shipped)

```
forecast/
  features.py     — embedding cols + per-(forecast_date) Standardizer
  detrend.py      — CountyTrend (per-GEOID linear, state-median fallback)
  data.py         — load_master, train/val/holdout pools, min-history filter
  analog.py       — AnalogIndex with per-date BallTrees, cross_county / same_geoid pools
  cone.py         — percentile cone in raw bu/acre
  aggregate.py    — county→state acres-weighted rollup
  baseline.py     — naive 5-yr county-mean baseline
  recalibrate.py  — per-(state, date) additive bias correction, fit on val
scripts/
  backtest_baseline.py    — end-to-end harness with K-sweep, recalibration, gate eval
  diagnose_wi_overshoot.py — exploratory diagnostic (preserved; pre-CountyTrend API)
  diagnose_state_errors.py — bias-vs-variance decomposition reading runs/*.csv
runs/
  backtest_baseline_<timestamp>.csv        — pre-recal per-row results
  backtest_baseline_<timestamp>_recal.csv  — post-recal per-row results (with *_recal cols)
```

## Phase B gate result

Both gates **PASS** post-recalibration on 2024 holdout, primary pool `same_geoid`, K=5:

| Gate | Target | Result | Pass? |
|---|---|---|---|
| Coverage of 80% cone on 2024 | [70%, 90%] | 80% across all forecast dates | ✅ |
| Point RMSE vs 5-yr-mean baseline | strictly less | 11.3 / 12.9 / 12.7 / 14.3 vs baseline 14.5 | ✅ |

Per-state recalibrated RMSE on 2024 (K=5):

| State | RMSE point (recal) | RMSE baseline | Coverage | Notes |
|---|---|---|---|---|
| CO | 1.2 | 0.65 | 100% | Tied with baseline; small-state high-stability |
| IA | 5.1 | 15.7 | 100% | **3× lift** — main driver of headline beat |
| MO | 26.0 | 27.1 | 0% | Recal overcorrected after MO 2023 drought outlier; flagged for Phase C |
| NE | 5.7 | 8.1 | 100% | Real lift |
| WI | 9.4 | 2.5 | 100% | Recal helped (was 12.6) but baseline still beats us locally |

## Phase B winning configuration (locked)

- **Pool:** `same_geoid` (within-county history)
- **K:** 5
- **Embedding:** 17 cols at 09-01/10-01/EOS, 15 at 08-01 (drop *_grain at 08-01)
- **Standardization:** per-(forecast_date) z-score on train pool only
- **Min-history filter:** N=10 qualifying years per candidate county (345 of 388 GEOIDs survive)
- **Detrend:** per-county OLS, state-median fallback for sub-threshold counties
- **Cone:** percentiles (10, 50, 90) in detrended space, retrended at (query_geoid, query_year)
- **State aggregation:** planted-acres-weighted, current-year `acres_planted_all`
- **Recalibration:** fit on val (2023) per (state, forecast_date), apply to holdout (2024)

## Open questions / known issues for Phase C

- **MO 2023 was a real-world drought outlier.** Recalibrating on it gave MO 2024 a -26 bu shift that wrecked MO coverage. Phase C XGBoost should learn the d2plus / vpd_kpa_silk / dry_spell signature of MO 2023 and absorb it as a *feature* response, not a constant offset. If Phase C still struggles on MO, the feature embedding is genuinely missing what drove MO 2023.
- **WI plateau.** Recent-era WI yields are flatter than the 18-year linear trend implies. Phase C should learn this from year/feature interactions; SHAP analysis can confirm.
- **One-year recalibration is fragile.** This is a v2-specific issue inherent to the analog-retrieval methodology. Phase C's regressor learns from features, removing single-point-of-failure structure.
- **Cross-county retrieval is parked, not retired.** With Prithvi embeddings (Phase D.1) the L2 distance may become discriminating enough that cross-county pool meaningfully helps. Flag as a Phase D.1 ablation.

## v2 — to write (Phase C onward)

- **Phase C:** XGBoost (or LightGBM) point-estimate model on engineered features. Replaces analog-median point estimate; **cone stays** (interpretable analog cone + accurate trained point). One model per forecast_date. Driver attribution via SHAP. **Gate: ≥15% RMSE improvement over Phase B analog-median at end-of-season.**
- **Phase D.1:** HLS pull pipeline (consistent schema, county-level if feasible) + Prithvi as frozen feature extractor. Concat Prithvi embeddings with engineered features, retrain Phase C model, ablate.
- **Phase D.2:** Optional Prithvi fine-tune (conditional on D.1 lift).
- **Phase E:** `/forecast/{state}` endpoint + frontend forecast view.
- **Phase F:** Forecast narration agent with 4 tools (analog years, top drivers, historical comparisons, current state).
- **Phase G:** Holdout evaluation, ablation table, presentation deck.

## Hackathon readiness checklist

- [x] Planning docs (`PHASE2_PROJECT_PLAN`, `PHASE2_PHASE_PLAN`, `PHASE2_DATA_INVENTORY`, `PHASE2_DECISIONS_LOG`) drafted
- [x] Data acquisition (NASS, NDVI, USDM, gSSURGO, gridMET) complete for 2005–2024
- [x] Feature engineering (`*_features.py`) shipped
- [x] `merge_all.py` + `training_master.parquet` built (Phase A.6)
- [x] `PHASE2_DATA_DICTIONARY.md` written (Phase A.7)
- [x] **Phase A definition-of-done met**
- [x] **Phase B analog-year retrieval baseline shipped + Phase B gate passed (post-recalibration on 2024)**
- [ ] Phase C XGBoost point-estimate model shipped + Phase C gate passed
- [ ] Phase D.1 Prithvi frozen-feature integration + Phase D gate decision
- [ ] Phase D.2 (conditional) Prithvi fine-tune
- [ ] Phase E `/forecast/{state}` endpoint + frontend forecast view
- [ ] Phase F forecast narration agent with 4 tools
- [ ] Phase G holdout evaluation, ablation table, presentation deck

## Immediate next steps (Phase C)

1. **Train/val/holdout split locked already** (2005–2022 / 2023 / 2024), implemented in `forecast/data.py::SPLIT_YEARS` and `train_pool()` / `val_pool()` / `holdout_pool()` slicers.
2. **Train one XGBoost (or LightGBM) regressor per forecast_date.** Same feature columns as the Phase B retrieval embedding, plus the held-back covariates (other NDVI cols, all NASS-aux, full drought tier, full gSSURGO). XGBoost handles native NaN, so the *_grain features at 08-01 are kept (default-direction split).
3. **Hyperparameter sweep:** `max_depth ∈ {4, 6, 8}`, `learning_rate ∈ {0.05, 0.1}`, early-stopped on val 2023.
4. **Replace point estimate, keep cone.** The trained regressor's prediction becomes the point; the analog-retrieval cone (P10/P90) stays. They are deliberately not the same model — cone is interpretable, point is accurate.
5. **Driver attribution:** SHAP top-3 features per (geoid, year, forecast_date). Stored for the Phase F narration agent.
6. **Phase C gate:** RMSE on 2023 val ≥15% better than Phase B analog-median at end-of-season.

## Budget note

v1 used ~$0.25 of the $10 API cap. v2 token usage will be similar in shape (Haiku narration, ~$0.02 per forecast). Phase C compute (XGBoost training) is local and fast — minutes, not hours. HLS download + Prithvi inference (Phase D.1) is local and outside the API budget.
