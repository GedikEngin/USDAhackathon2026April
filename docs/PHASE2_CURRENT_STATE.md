# Phase 2 — Current State

> Overwritten at end of every v2 phase. Tells the next chat what actually exists in the v2 portion of the repo right now. Reference file paths and describe behavior; do not paste code here.

**Project:** USDA Hackathon 2026 April — Corn Yield Forecasting
**Repo:** `~/dev/USDAhackathon2026April`
**Last updated:** 2026-04-25, end of Phase A.6 + A.7 — **PHASE A COMPLETE**

---

## v2 status: Phase A is fully closed. Ready to start Phase B.

Six data pipelines feed one master table:

- NASS yield 2005–2024 — `(GEOID, year)`
- MODIS NDVI 2004–2024 (corn-masked via CDL, county-aggregated; 21 per-year CSVs) — `(GEOID, year)`
- gSSURGO Valu1 features for all 5 states, county-aggregated — `(GEOID,)` static
- gridMET daily weather 2005–2024, derived per-cutoff features — `(GEOID, year, forecast_date)`
- US Drought Monitor weekly readings, derived per-cutoff features — `(GEOID, year, forecast_date)`
- HLS-derived state-level VIs (provisional; redone in Phase D.1) — `(state_alpha, year, forecast_date)`

`scripts/merge_all.py` outer-joins all six into `scripts/training_master.parquet` (25,872 rows × 48 columns, 2.46 MB). Every column is documented in `docs/PHASE2_DATA_DICTIONARY.md`. No surprise nulls; full-coverage rate is 98.1%. All `merge_all.py` asserts pass.

No modeling, no backend endpoints, no agent tools yet. The v1 land-use system is shipped and untouched alongside.

## Project goal (locked)

Build an ML pipeline that predicts corn-for-grain yield (bu/acre) at the **county** level across **5 US states** at four fixed in-season forecast dates (Aug 1, Sep 1, Oct 1, end-of-season), aggregates to state, and produces a calibrated cone of uncertainty using **analog-year retrieval**. A Claude Haiku agent narrates the forecast.

- **Target states:** Colorado (CO), Iowa (IA), Missouri (MO), Nebraska (NE), Wisconsin (WI)
- **Time range:** **2005–2024** (20 years; 2024 holdout, 2023 validation, 2005–2022 train)
- **Spatial unit:** US county, joined via 5-digit GEOID (state FIPS + county FIPS, both zero-padded)
- **Target variable:** `yield_bu_acre_all` (combined-practice corn grain yield, bu/acre), exposed in the master table as `yield_target`

See `PHASE2_PROJECT_PLAN.md` for the full vision and `PHASE2_PHASE_PLAN.md` for the phase breakdown with go/no-go gates.

## What exists in the repo today

### v1 (untouched, fully shipped)

The full v1 land-use & GHG analysis system is in place. See the v1 `CURRENT_STATE.md` for canonical detail. v2 reuses v1's FastAPI shell, frontend chrome, and Claude agent infrastructure but does not modify them.

### v2 — written and run

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
- **`scripts/merge_all.py`** — outer-joins all six sources into the master table. Includes 2005-min-year filter, HLS feature allowlist, NDVI schema-drift guard, per-layer row-count invariant asserts, and a comprehensive QC tail (year/state/forecast_date coverage breakdowns + full-coverage rate).
- **`scripts/training_master.parquet`** — **the canonical Phase B/C/D input.** 25,872 rows × 48 columns. Keyed on `(GEOID, year, forecast_date)`. Full coverage rate 98.1%. All asserts pass.
- **`docs/PHASE2_DATA_DICTIONARY.md`** — column-by-column reference for the master table. Quick-facts table, source-grouped sections with units/ranges/formulas/phase windows, NaN-patterns triage table, as-of fidelity reference table, worked example, versioning note. Heavy on precision, light on prose.

### v2 — to write (Phase B onward)

- Feature standardization + nearest-neighbor analog retrieval baseline → cone-of-uncertainty MVP. **Phase B.**
- XGBoost (or LightGBM) point-estimate model on engineered features. **Phase C.**
- HLS pull pipeline (consistent schema, county-level if feasible) + Prithvi as frozen feature extractor. **Phase D.1.**
- Optional Prithvi fine-tune. **Phase D.2.**
- `/forecast/{state}` endpoint + frontend forecast view. **Phase E.**
- Forecast narration agent with 4 tools. **Phase F.**
- Holdout evaluation + ablation table + presentation deck. **Phase G.**

## Master table — at-a-glance

```
scripts/training_master.parquet
  shape:        25,872 × 48
  size:         2.46 MB
  grain:        (GEOID, year, forecast_date)
  years:        2005–2024 (20 years)
  states:       CO, IA, MO, NE, WI (5)
  GEOIDs:       388 distinct (subset of 443 TIGER 2018 counties)
  forecast_d:   08-01, 09-01, 10-01, EOS
  target:       yield_target (bu/acre, combined-practice)
  full coverage: 25,380 / 25,872 (98.1%)  — rows with non-null target + at least one
                                            non-null per source group (excl. HLS)
                10,575 / 25,872 (40.9%)   — same, including HLS (HLS only exists 2013+)
```

48 columns: 5 keys + 1 target + 5 NASS-aux + 5 NDVI + 11 gSSURGO + 11 weather + 6 drought + 4 HLS. Source script and provenance for each column documented in `PHASE2_DATA_DICTIONARY.md`.

## Known sparseness (all documented in the dictionary)

- **`yield_bu_acre_irr`** — 83.7% null. Only CO and NE report irrigated practice; structural.
- **`vpd_kpa_grain`, `srad_total_grain`** — 25.0% null, every `08-01` row. Structural: grain fill (DOY 228+) hasn't started by Aug 1. Correct as-of behavior. **Do NOT median-impute.**
- **`ndvi_*`** — 1.9% null. CO 2005–2007 — USDA CDL didn't roll out CO until 2008; corn masking failed.
- **HLS `ndvi_*`, `evi_*`** — 59.1% null. Structural: HLS doesn't exist pre-2013. To be redone in Phase D.1.
- **`harvest_ratio`, `acres_harvested_all`** — 8 nulls (~0%). NASS disclosure suppression on 2 (GEOID, year) tuples.

Everything else (cumulative weather, all gSSURGO, all drought, NDVI 2008+) has 0 nulls.

## Recommended full feature set — DELIVERED

The "design target" feature set in the prior phase notes is now realized in `training_master.parquet`. Every italicized row from the prior `PHASE2_CURRENT_STATE.md` design table has been built:

- **Soil** (gSSURGO, 11 cols) — all on disk and merged.
- **Climate** (gridMET, 11 cols) — all on disk and merged.
- **Drought** (USDM, 6 cols) — all on disk and merged.
- **Remote sensing — MODIS** (NDVI, 5 cols) — all on disk and merged.
- **Remote sensing — HLS** (4 cols, provisional) — merged at state level; redone in Phase D.1.
- **Management — NASS** (5 cols) — all on disk and merged.

## Open architectural questions for Phase B

These came out of the Phase A.6 review and are flagged for the Phase B kickoff:

- **MODIS NDVI is weak as-of** (same whole-season-summary value at all four forecast dates). Locked decision: ship as-is, replace with HLS running-NDVI in Phase D.1. Watch for early-cutoff overfitting in Phase B/C.
- **NASS-aux fields are post-hoc reported.** Same value across all 4 forecast_dates within a year. Treat as structural priors. If Phase C ablations show these dominate at 08-01, switch to lagged variants.
- **HLS and USDM are state-level broadcast** — within-state spatial heterogeneity is lost for these two sources. Acceptable for v2 baseline; revisit if Phase B retrieval quality is dominated by intra-state drought differences.

## Hackathon readiness checklist

- [x] Planning docs (`PHASE2_PROJECT_PLAN.md`, `PHASE2_PHASE_PLAN.md`, `PHASE2_DATA_INVENTORY.md`, `PHASE2_DECISIONS_LOG.md`) drafted
- [x] NASS API key + script, full pull done for 2005–2024
- [x] `scripts/nass_features.py` written and run
- [x] GEE NDVI export complete for 2004–2024
- [x] `scripts/ndvi_county_extraction.js` saved into `scripts/` for version control
- [x] US Drought Monitor raw CSV pulled for 5 states
- [x] `scripts/gssurgo_county_features.py` written and run (Phase A.3)
- [x] `scripts/gridmet_pull.py` + `scripts/weather_features.py` written and run (Phase A.4)
- [x] `scripts/drought_features.py` written and run (Phase A.5)
- [x] `scripts/merge_all.py` written, audited, fixed; `training_master.parquet` built (Phase A.6)
- [x] `PHASE2_DATA_DICTIONARY.md` written (Phase A.7)
- [x] **Phase A definition-of-done met** — master table loads, no surprise nulls, asserts pass, every NaN documented
- [ ] Phase B analog-year retrieval baseline shipped + Phase B gate passed
- [ ] Phase C XGBoost point-estimate model shipped + Phase C gate passed
- [ ] Phase D.1 Prithvi frozen-feature integration + Phase D gate decision
- [ ] Phase D.2 (conditional) Prithvi fine-tune
- [ ] Phase E `/forecast/{state}` endpoint + frontend forecast view
- [ ] Phase F forecast narration agent with 4 tools
- [ ] Phase G holdout evaluation, ablation table, presentation deck

## Immediate next steps (Phase B)

1. **Decide on the retrieval feature set.** Standardize a subset of master-table columns into a per-`(GEOID, year, forecast_date)` embedding. Open question: which features go in (likely all weather + drought + NDVI + soil, NOT yield_target or NASS-aux), and how to handle the structural-NaN columns at 08-01.
2. **Per-county analog retrieval.** For a target `(GEOID, year_target, forecast_date)`, find the K nearest analogs from the same GEOID's history (excluding `year_target` and any held-out years if doing temporal CV).
3. **Cone construction.** Take the observed `yield_target` of the K analogs; the cone is a percentile band around them (e.g. P10–P90).
4. **Calibrate on 2023.** Compute cone coverage at the 80% nominal level; iterate K and percentile choice.
5. **Phase B gate:** is the analog cone defensibly better than a naive 5-year county-mean baseline? If yes, ship; if no, revisit retrieval-embedding design before Phase C.

## Budget note

v1 used ~$0.25 of the $10 API cap. v2 token usage will be similar in shape (Haiku narration, ~$0.02 per forecast). Compute cost for HLS download + Prithvi inference is local (Ubuntu box) and outside the API budget.
