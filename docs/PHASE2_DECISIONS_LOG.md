# Decisions Log — v2

> Append-only record of decisions made during each v2 phase. Update at end of every phase chat. Never edit previous entries — if you reverse a decision, add a new entry noting the reversal.
>
> v1 decisions are in `DECISIONS_LOG.md` and remain in force for v1 code. v2 decisions live here.

## Format

```
### Phase 2-X — [short title] — [date]

**Decided:**
- Thing, with brief rationale

**Rejected alternatives:**
- Thing we considered but didn't pick, with why

**Surprises / learnings:**
- Anything that didn't match expectations

**Open questions carried forward:**
- Things to decide in a later phase
```

---

## Entries

### Phase 2-pre — Project planning & data scoping — 2026-04-25

**Decided:**
- v2 builds **on top of** v1's FastAPI + frontend + Claude agent infrastructure, not in place of it. v1 docs and code stay untouched.
- All v2 planning docs prefixed `PHASE2_`. v1 docs (no prefix) are untouched.
- **Spatial unit: county-level training, state-level reporting** via planted-acres-weighted aggregation. ~80× more training rows than state-only.
- **Forecast dates locked at Aug 1 / Sep 1 / Oct 1 / End-of-Season** per brief.
- **Five states locked at IA / CO / WI / MO / NE** per brief.
- **Foundation model is Prithvi**, integration mode default = frozen feature extractor (Phase D.1). End-to-end fine-tune (D.2) is a stretch goal contingent on D.1 ablation showing lift.
- **Cone of uncertainty methodology = analog-year retrieval** (K-nearest neighbors over a feature embedding, percentile band over observed yields of analog (county, year, date) tuples). NOT quantile regression, NOT Bayesian credible intervals. Brief explicitly defines it this way.
- **Point estimate model = XGBoost (or LightGBM) on engineered features** as the v2 baseline; Prithvi features layered in as ablation.
- **Initial training data depth: 2005–2024.** 2024 reserved as holdout, 2023 reserved as validation. *(Note: this was reverted to 2015–2024 mid-day in the merge of the existing data state, then reverted back to 2005–2024 — see Phase 2-pre-revision below.)*
- **Feature/label "as-of" rule:** when constructing features for forecast date D in year Y, only data with timestamps strictly before D is allowed. Enforced in `forecast/features.py` (to be written).
- **Repo organization:** v2 code under new `forecast/` package + `scripts/forecast_*.py` for one-offs + `frontend/forecast/` for new UI pieces. v1 packages (`backend/`, `agent/`, `frontend/index.html`+`app.js`) gain new files but don't have existing files modified.
- **Phase plan defines 7 phases (A–G)** with explicit go/no-go gates at end of B, C, D.

**Rejected alternatives:**
- **State-level training only.** ~5 rows per year × 20 years = 100 rows total. Too small for meaningful ML.
- **Quantile regression / NGBoost / Bayesian deep learning for the cone.** Could give calibrated cones, but cones aren't interpretable and don't match the brief's literal definition. Analog retrieval is the brief's specified methodology.
- **Skip Prithvi, use only XGBoost.** Defensible but undersells the brief, which names Prithvi explicitly. At minimum Phase D.1 (frozen Prithvi as feature extractor) is in scope.
- **Prithvi end-to-end fine-tune as default.** High effort, uncertain lift, blocks shipping a complete pipeline. D.2 is conditional on D.1 results.
- **Use the v1 SegFormer model for corn-pixel masking.** CDL is purpose-built for this (USDA's own corn mask) and free.
- **Use NDVI alone instead of HLS for Prithvi.** NDVI is a derived index; Prithvi was pretrained on raw HLS. NDVI stays as a feature, HLS is the imagery layer for D.1.
- **Use the user's existing monthly mean temp/precip for GDD computation.** Monthly means lose the daily heat-stress signal that yield models need. Daily PRISM/gridMET is required; the monthly data stays as a coarse smoothed feature.
- **Treat tornado activity as a feature.** Tornado damage is highly localized; aggregated to county-year has near-zero signal. Deprioritized.
- **Carry pixel-level gSSURGO into the model.** Soil features are static across years, so pixel-level only explains between-county variance. Aggregate to county-level features early.

**Surprises / learnings:**
- The user's data inventory was very strong on the tabular/ground-truth side (NASS, water-applied, gSSURGO) and missing the satellite-imagery-as-feature side (no HLS, no CDL standalone, no Prithvi). The brief's centerpiece (foundation-model features over multispectral imagery) is exactly the part that needs the most acquisition work in Phase A / D.1.
- **NDVI vs. HLS** is a common point of confusion. They aren't substitutes — NDVI is derived from HLS bands, and Prithvi expects the raw bands. The user's GEE NDVI download is useful as a smoothed feature but does not eliminate the need for HLS in Phase D.1.
- **CDL** is needed for masking; for now it's used inside the Earth Engine NDVI script (`cropland == 1`). A standalone local download will be needed if HLS pulling for Phase D.1 happens.

**Open questions carried forward:**
- HLS storage strategy — TB-scale. Settle in Phase D.1 opening.
- HLS coverage for 2005–2013 (Landsat-only era) is sparser than 2015+. Document caveat in Phase D.1.
- Final feature set for the *retrieval embedding* (vs. covariates) — Phase B.1.
- K for K-NN analog retrieval — default 10, tune in Phase B.5.
- Cone percentiles to publish in the API — default {10, 50, 90}.
- Whether to ship Phase D.2 (Prithvi fine-tune) — conditional on D.1 ablation.
- PRISM vs. gridMET for daily weather — decide when `prism_pull.py` is written.
- Drought source — US Drought Monitor vs. SPI/SPEI from PRISM — Phase A.4/A.5.
- County coverage filter (minimum N years per county) — decide after full 2005–2024 NASS extension.

---

### Phase 2-pre-revision — Date range and existing-pipeline reconciliation — 2026-04-25

**Context:** the user shared the actual current state of their data pipeline mid-planning. Two reconciliation decisions resulted, plus a date-range correction.

**Decided:**
- **Time range stays at 2005–2024** per the brief's explicit "Data spans from 2005-2024" instruction. The merged-doc draft briefly used 2015–2024 to match the existing NASS pull, but the brief specifies 2005, and using only half the available history weakens analog-year retrieval (10 candidate years per query becomes ~17–18). User has agreed to extend the NASS pull and the GEE NDVI export to cover 2005–2014.
- **Adopted the user's pipeline architecture** as the canonical Phase A structure. Specifically: separate `*_pull.py` and `*_features.py` per source, then `merge_all.py` for the final join. This pattern is already followed by `nass_pull.py` and is now baked into the Phase A sub-phase deliverables.
- **MODIS NDVI (via Earth Engine, server-side, corn-masked via CDL) is the primary remote-sensing feature for Phases B and C.** Raw HLS is deferred to Phase D.1 (Prithvi). The two are explicitly complementary, not substitutes — MODIS gives smoothed seasonal indices at county-year granularity; HLS gives raw multispectral chips for Prithvi to encode.
- **PRISM (or gridMET) is the daily weather source.** Both work; PRISM is US-focused. Final pick deferred until `prism_pull.py` is written.
- **Train/val/holdout split scaled to the longer history:** train = 2005–2022 (18 years), val = 2023, holdout = 2024.
- **gSSURGO Valu1 table** is the canonical source for soil features, aggregated to county via zonal stats over TIGER/Line 2018 polygons. Reproject before zonal stats (gSSURGO is EPSG:5070 Albers).
- **Combined-practice yield (`yield_bu_acre_all`) is the target.** 100% coverage in the current NASS pull. Irrigation effect captured via `irrigated_share` derived feature where data exists.
- **Preserved the user's data-quality fixes already applied to `nass_pull.py`:** pin `prodn_practice_desc` (avoids 81-row fan-out), filter "OTHER (COMBINED) COUNTIES" placeholder rows, strip commas before `pd.to_numeric`. These are now documented in `PHASE2_DATA_INVENTORY.md` and `PHASE2_CURRENT_STATE.md`.
- **Preserved the user's deprioritization decisions:** hourly weather, day length / sunrise-sunset, CO₂ as a per-county feature. Documented with rationale.
- **Preserved the user's GEE bug fixes:** use `ee.ImageCollection('USDA/NASS/CDL').filter(...)` instead of string-concatenating `ee.Number` into asset paths; wrap demo image to reset accumulated `system:description` metadata.

**Rejected alternatives:**
- **Cutting time range to 2015–2024** — would reduce analog-year candidate pool by ~50% and contradict the brief's explicit "2005-2024" instruction. Rejected.
- **Holding 2023+2024 as a 2-year holdout** — clean validation discipline (separate val and holdout) is more portfolio-defensible. Rejected in favor of train 2005–2022 / val 2023 / holdout 2024.
- **Re-running NASS pull immediately to fill non-irrigated columns** — non-irrigated is derivable as `acres_harvested_all − acres_harvested_irr` for the 4 fields that have irrigated data. Re-run is optional, low priority. The historical-extension re-run (to add 2005–2014) is higher priority.

**Surprises / learnings:**
- The user's `nass_pull.py` already encodes three subtle NASS-API gotchas (multi-practice fan-out, OTHER-COUNTIES placeholders, comma-separated numeric strings). These would have cost ~half a day to rediscover; preserving them in the docs is worth more than re-deriving them.
- The user's GEE script is more complete than the planning docs assumed — it already does corn-masking via CDL and county-level reduction server-side. The output CSV is already at `(GEOID, year)` granularity, which simplifies `merge_all.py` substantially (no local zonal stats needed for NDVI).
- The Earth Engine `ee.Number` JS-concat bug is a real footgun and worth keeping documented for any future GEE work in this repo.

**Open questions carried forward:**
- Can the existing GEE export be re-run for 2005–2014 in one task, or does it need to be split? (CDL availability per year may force splitting.)
- After the NASS extension to 2005–2014, what's the actual county coverage by year? Some Colorado mountain counties may drop out for additional years; need to QC.

---

### Phase 2-A.1+A.2 — NASS extension and NDVI extraction landed — 2026-04-25

**Context:** Returned to the project after the queued pulls finished. Discovered the pipeline state was further along than the docs reflected: NASS pull extended to 2005–2024 successfully, NDVI export ran for 2004–2024 (one year wider than planned on the early side), `nass_features.py` written and run, US Drought Monitor raw CSV pulled. GEE script saved into `scripts/ndvi_county_extraction.js` for version control during this session.

**Decided:**
- **PRISM** — see above. Not categorically rejected; just unpicked because gridMET was easier and equivalent.
- **Compute GDD from monthly mean temp** (using the existing user-provided monthly CSVs). Already-rejected in Phase 2-pre; restating because the temptation is real when daily weather is slow to pull. Monthly means lose the day-to-day heat-stress signal.
- **Use raw `max(0, tmax_F − 86)` for EDD.** Simpler but throws away most of the day-night asymmetry. The single-sine integral is ~50 lines of vectorized code and gives a much better signal at the cost of a half-day of careful implementation.
- **Compute daily VPD from `(tmax, tmin, vp)` server-side at pull time.** Was tempted; ended up keeping VPD as a per-day pre-computed column in the parquet so feature work stays a pure aggregation pass. This is mostly a style call.
- **Carry hourly temperature as a feature.** Already deprioritized — 24× the storage for marginal gain at county-year aggregation, and the single-sine EDD already approximates the heat-exposure integral.

**Surprises / learnings:**
- **`vp` (vapor pressure)** in gridMET is in kPa, not Pa or hPa. Worth a unit-check at the head of the feature script.
- **gridMET shortwave radiation (`srad`)** is in W/m² daily mean; we convert to MJ/m² by `srad × 86400 / 1e6` per day before summing. Easy to forget.
- **Phase-clip-to-cutoff matters more than expected.** On 08-01 the silking window (196–227 = Jul 15–Aug 15) is only ~2 weeks in; aggregating the full silking range without clipping would silently use data from days 214–227 that haven't happened yet. The `phase_slice(df, lo, hi, cutoff_doy)` helper centralizes this.

**Open questions carried forward:**
- Phase B/C may want **monthly summaries** (e.g., June precip, July tmax) in addition to phase-window aggregates. Easy to add by extending `build_features_for_cutoff` with a month-by-month loop. Defer until modeling shows whether the phase windows alone are sufficient.

---

### Phase 2-A.5 — USDM drought features derived — 2026-04-25

**Context:** Raw USDM CSV already on disk for months. Phase A.5 was about deciding the feature shape and deriving it.

**Decided:**
- **USDM is state-level, not county-level.** The raw CSV header is `MapDate,StateAbbreviation,StatisticFormatID,ValidStart,ValidEnd,D0,D1,D2,D3,D4,None,Missing` — there's no county FIPS. Earlier planning notes said "weekly D0–D4 percentages per county" which was aspirational. Discovery during Phase A.5 was that the actual data is state-level, so feature derivation broadcasts each state's reading to every GEOID in that state at output time.
- **GEOID directory is `nass_corn_5states_features.csv`.** Provides the modeling universe — every (GEOID, state_alpha, year) triple we care about. This makes the broadcast deterministic and ties USDM coverage to the same county set as the rest of the pipeline.
- **As-of rule: `valid_end < forecast_date` (strictly before, never on or after).** USDM week-validity windows can bracket the forecast date (a map released Tuesday with `valid_start=Mon, valid_end=Sun` covers a week that may include the forecast day itself). Strict `<` on `valid_end` guarantees the selected reading was fully published and visible before the forecast date.
- **Feature set is intentionally minimal: `d0_pct, d1_pct, d2_pct, d3_pct, d4_pct, d2plus`.** Six columns. No DSCI, no season-cum drought weeks, no silking-peak DSCI. Rationale: USDM percentages are already cumulative (D0 ≥ D1 ≥ D2 ≥ D3 ≥ D4 by construction), so the individual columns carry the full signal. DSCI is `D0+D1+D2+D3+D4` and adds no information beyond what the five raw columns provide. Season-cum and silking-peak are nice-to-haves that can be added later if Phase B/C show the model wants richer drought signal.
- **`d2plus` is exposed under a stable alias even though it equals `d2_pct`.** USDM's cumulative convention is non-obvious and easy to forget. Naming the "severe-or-worse" feature explicitly avoids future code having to encode the convention. Cost is one extra column; benefit is descriptive clarity.
- **Output skeleton is full cartesian `(GEOID, year, forecast_date)` × left-join state features.** Guarantees every output row has all three keys populated even if no USDM reading precedes the forecast date for some (year, forecast_date) combo. NaN feature columns in that case; `merge_all.py` decides downstream how to handle.
- **Code style matches `weather_features.py`.** Same `argparse` defaults pattern, same QC tail (row count, columns, head sample, coverage by forecast_date, null counts), plus a monotonicity check on the state-level intermediate as a free integrity test.

**Rejected alternatives:**
- **DSCI as a separate column.** It's `D0+D1+D2+D3+D4` and adds no information. Rejected. (Could be re-added if the user wants a single 0–500 drought-severity scalar for the retrieval embedding, but the individual columns are strictly more expressive.)
- **Season-cumulative drought weeks (count of weeks since May 1 where D2 ≥ 50%).** Useful in principle for sustained-stress signal. Rejected for v1 of drought features because (a) the as-of rule is harder to enforce cleanly for cumulative-counts than for as-of readings; (b) cumulative GDD / cumulative precip already carry sustained-condition signal; (c) easy to add later if Phase B/C wants it.
- **Peak DSCI during silking (DOY 196–227).** Same logic — defer until needed. Silking-window precision conflicts slightly with the as-of rule for the 08-01 forecast date (silking window is only 18 days in by 08-01).
- **Trailing-N-week mean of D-levels.** Rejected — last reading strictly before is cleaner, and the model can learn smoothing if it wants by combining the as-of reading with cumulative GDD/precip.
- **Use `MapDate` or `ValidStart` as the as-of comparison date.** `ValidEnd` is the published validity boundary; using `MapDate` (Tuesday-released) or `ValidStart` (Monday-of-week) would either leak future info or undercount available data. `ValidEnd` is correct.
- **Broadcast via state-level FIPS lookup table instead of `nass_corn_5states_features.csv`.** Same result; using the NASS features file makes the GEOID universe explicit and shared with the rest of the pipeline.

**Surprises / learnings:**
- **The earlier planning docs claiming "per-county D0–D4" were just wrong about what's actually in the file.** Always inspect the header before writing the feature script. The `PHASE2_DATA_INVENTORY.md` and `PHASE2_PHASE_PLAN.md` were both updated to reflect reality.
- **`d2plus == d2_pct` because USDM cumulates.** Easy to mistake for redundancy; the descriptive alias is worth the duplicated column.
- **0 nulls in the output despite NDVI's 2004 backfill including a year before USDM coverage.** Real USDM goes back to 2000-01-10 in the source CSV, so even the NDVI-driven 2004 GEOID rows are fully covered. This was a relief.
- **0 monotonicity violations across all 420 state-level rows.** Real-world data sometimes has rounding noise that produces sub-1e-9 D0 < D1 cases; tightening the check to `1e-9` tolerance would catch nothing. The source data is clean.
- **`DEFAULT_IN` originally pointed to `scripts/drought_USDM-...csv` from a habitual path-pattern slip.** The actual file lives at `phase2/data/drought/...`. One round-trip with the user to fix. Worth checking actual on-disk paths before committing default arguments.
- **Windows `:Zone.Identifier` files** appear when files come across via the chat. Cleaned with `find . -name '*:Zone.Identifier' -delete`. Worth a `.gitignore` line.

**Open questions carried forward:**
- Whether to enrich the drought feature set after Phase B/C. Default: ship minimal, add only if signal warrants. Candidates: DSCI, season-cum drought weeks, silking-peak DSCI, trailing-4-week mean of d2plus.
- Whether the state-level granularity hurts retrieval quality. USDM is the only state-level feature in the pipeline; if Phase B shows the analog matching is dominated by drought differences within a state, may need to drop USDM from the retrieval embedding (keep as covariate).

---

### Dataset exclusion — NAIP aerial imagery — 2026-04-25

**Context:** The brief lists NAIP (USDA FSA aerial imagery, 0.6–1m resolution, RGB+NIR) alongside HLS, NASS, CDL, and weather as a candidate dataset. Question came up in conversation: which phase actually uses it? Answer: none. Decision is to keep it out and document the rationale here so it doesn't keep coming up.

**Decided:**
- **NAIP is excluded from all phases of v2.** No phase fits its strengths; every NAIP-adjacent need is better served by a dataset already in the pipeline.
- **Brief alignment.** The brief lists candidate datasets, not required ones. Documenting "considered and excluded" with rationale satisfies the brief's selectivity expectation. A team that pulls every named dataset without thinking about fit is a worse signal than one that justifies its choices.
- **Add a one-line "datasets considered but not used" note to the final write-up / presentation.** Pre-empts the question from judges/reviewers.

**Rejected alternatives:**
- **Use NAIP as Prithvi input.** Prithvi was pretrained on HLS bands at 30m. Feeding it NAIP would mean different band setup, different spectral characteristics, different ground sample distance — most of the pretraining benefit is lost. HLS is the right Prithvi input; NAIP is not.
- **Use NAIP as a substitute for HLS in 2005–2012 (the pre-HLS years).** NAIP coverage in those years is sparse and asynchronous across states. Even if it were dense, the 2–3-year flyover cadence per state is wrong for within-season forecasting at four time points. Doesn't solve the gap.
- **Use NAIP for corn-pixel masking instead of CDL.** CDL is purpose-built for this (USDA's own corn mask), free, annual, and aligned to the same grid as the rest of the imagery layer. NAIP would require a custom corn-segmentation pipeline before it could be used as a mask. Strictly worse.
- **Use NAIP for field-boundary extraction.** Possible in principle, but the model targets county-year yield. Per-field boundaries don't change the aggregation. Wrong granularity for the target.
- **Pull NAIP "just in case" something downstream wants it.** Multi-TB download for hypothetical future use. Storage and pull cost both nontrivial. Hard no.

**Surprises / learnings:**
- **NAIP's strength (sub-meter visual context) is exactly the dimension our target (county-year scalar) collapses.** Aggregating sub-meter imagery to county-year throws away ~99.99% of what makes NAIP special. Wrong tool for this problem. A different problem (per-field yield, equipment detection, field-boundary extraction) would be a much better fit.
- **NAIP's flight cadence (every 2–3 years per state, growing-season acquisition with variable date) is fundamentally incompatible with a 4-time-point within-season forecast schedule.** Even if the spatial mismatch were solved, the temporal mismatch alone would rule it out.
- **NAIP would have been a much more natural fit for the v1 SegFormer land-use project.** Visual segmentation at sub-meter res is exactly what NAIP is for. Worth noting if v3 or a follow-on project goes back to land-cover work.

**Open questions carried forward:**
- *(none — this decision is closed unless the project scope changes to per-field or equipment-level prediction)*

---

### Phase 2-A.6 + A.7 — `merge_all.py` audit & ship + data dictionary — 2026-04-25
 
**Context:** Entered the chat with `merge_all.py` already written and run end-to-end, producing `scripts/training_master.parquet` (27,336 × 50 at first pass). Goal of the session: code review the merge, fix anything wrong before declaring A.6 done, then write the data dictionary (A.7) so Phase A is closed cleanly.
 
**Decided:**
- **NASS rows with year < 2005 are dropped at load time inside `load_nass()`** (new `min_year=2005` parameter, default-active filter). Rationale: the NASS pull/features pipeline incidentally produced a 366-row 2004 block (NDVI extraction had run for 2004 on a script-level off-by-one and NASS upstream queries followed), but weather features only start in 2005 (gridMET extraction window). Without the filter, the skeleton inherits 2004 keys and produces 1,464 structural-NaN rows for every weather/drought-derived column — a silent data-quality bug masquerading as ~5% normal sparseness in the QC tail. Dropping at the source-load layer is the cleanest fix; the full-coverage rate jumped 92.8% → 98.1% after the fix landed.
- **HLS pull pipeline writes `is_forecast` and `n_granules` audit/meta columns alongside features.** New `HLS_FEATURE_COLS = ["ndvi_mean", "ndvi_std", "evi_mean", "evi_std"]` allowlist drops everything else inside `load_hls()`. Rationale: `is_forecast` is object-dtype (a semantic flag), and `n_granules` is a granule count, not a vegetation index. Either one flowing into the master table as a feature column would poison Phase C's XGBoost training (object-dtype crash; count column with no monotonic relationship to yield). Allowlisting also makes future audit-column additions in the HLS pipeline automatically safe — they get dropped by default with a printed log line.
- **Per-merge-layer row-count invariant asserts** (`assert len(df) == len(skeleton)`) added after each of the six layered merges in `main()`. Rationale: the most likely future-bug shape for this script is a duplicate-key fan-out from a source-side data-quality issue. The existing end-of-script assert (`(GEOID, year, forecast_date)` uniqueness) catches it eventually, but the per-layer asserts pinpoint *which* source caused it. Cheap (six lines), high-leverage.
- **NDVI loader schema-drift guard** added (`ValueError` if any of the five expected NDVI columns is absent from input CSVs after concat). Rationale: the per-file column filter inside `load_ndvi()` silently keeps only what's in `NDVI_FEATURE_COLS`; if the GEE export ever renames a column (it has happened mid-development), the merge would silently zero out NDVI features without anyone noticing. Loud failure beats silent corruption.
- **HLS slice-schema asymmetry left as documented Phase D.1 work, not fixed in A.6.** The five HLS slice CSVs have inconsistent columns (some have `is_forecast` and `n_granules`, others don't, evidenced by the 46.7% vs 61.3% null asymmetry in the original run). The Phase D.1 HLS pull will be redone with a consistent schema for the Prithvi pipeline anyway, so cleaning the existing slice files now is wasted effort. Noted in `PHASE2_DATA_DICTIONARY.md` and the data inventory.
- **Phase A definition-of-done is met.** Master table loads, all 48 columns documented, asserts pass, and every NaN cell now has a known-and-documented cause: 25.0% structural at `*_grain × 08-01` (correct as-of behavior), 1.9% sparse-coverage at NDVI for CO 2005–2007 (CDL pre-rollout), 59.1% structural at HLS pre-2013, 0.03% NASS disclosure suppression on 2 (GEOID, year) tuples. No surprise nulls anywhere.
- **`PHASE2_DATA_DICTIONARY.md` style: comprehensive reference doc**, structured for fast lookup. Format: quick-facts table → column-overview-by-source table → section per source (5 keys, 1 target, 5 NASS-aux, 5 NDVI-MODIS, 11 gSSURGO, 11 weather, 6 drought, 4 HLS) with units, ranges, null counts, formulas, phase-window definitions, and as-of caveats. Plus a triage table for NaN patterns (structural / sparse-coverage / should-never-be-NaN) with recommended downstream handling, an as-of fidelity reference table, a worked example unpacking one row fully, and a versioning note. Heavy on precision, light on prose. Designed to be the single doc Phase B/C/F engineers (or LLM agents) need to reason about feature semantics without reading 400 lines of `weather_features.py`.

**Rejected alternatives:**
- **Filter year < 2005 in train.py instead of in load_nass().** Considered. Rejected because the bug is structural to the master table (every consumer would have to remember to filter); centralising at load time is correct. Train.py still owns the train/val/holdout split (2005–2022 / 2023 / 2024).
- **Backfill weather and drought to 2004.** Out of scope. NDVI-driven 2004 rows aren't useful for training (no in-season weather signal) and aren't useful as feature history for analog retrieval (analogs match on weather/drought trajectories, which require those features at the candidate year too).
- **Median-impute the *_grain features at 08-01 instead of treating as structural NaN.** Considered. Rejected — at the 08-01 cutoff, grain fill literally hasn't started, so any imputed value is fictional. XGBoost handles native missing values (assigns to a default-direction split); analog retrieval should treat these as drop-this-feature-for-08-01-models, not impute. Documented this prescription in the dictionary's NaN-patterns triage table.
- **Drop `d2plus` from the dictionary because it's exactly equal to `d2_pct`.** Considered for parsimony. Rejected — the alias is in the merged table on purpose (cumulative-USDM convention is non-obvious; the alias exists so retrieval-embedding code doesn't have to remember it). Documenting both columns honestly is the right move.
- **Skip the data dictionary and let the PHASE2_DATA_INVENTORY notes serve.** Rejected — inventory tracks data status (have/in-flight/need), not column semantics. The dictionary is the artifact downstream code reads. Conflating them obscures both.
- **Cheatsheet-style dictionary instead of comprehensive reference.** Rejected per session input — Phase F's narration agent and Phase B's analog-retrieval embedding both benefit from precision (units, formulas, phase windows). A cheatsheet would force every consumer back into the source scripts.

**Surprises / learnings:**
- **The 2004 NASS leakage was invisible from the QC tail at first glance.** It showed up as ~5% nulls per weather column, which looked like normal sparseness from missing-county-year combos. The structural pattern (1,464 = 366 × 4 across every weather column) was only obvious once flagged. **Lesson: structural NaN counts that match exactly across columns are a tell — single-cause failures, not noise.**
- **`vpd_kpa_grain` and `srad_total_grain` showing 6,468 nulls (25.0%) was initially alarming until the math worked out: 6,468 = 1,617 (GEOID, year) tuples × 4 forecast_dates of 08-01-only NaN. That's correct as-of behavior, not a bug.** Documenting this in the NaN-patterns triage table is necessary because Phase B/C code that imputes the median would silently pretend grain fill was happening on Aug 1 and degrade model quality. **Lesson: as-of structural NaN looks like a bug to a downstream pipeline that doesn't know the convention.**
- **Full-coverage rate is the cleanest single signal of data health for this kind of merge.** Watched it go 92.8% → 98.1% after the 2004 fix. One number, no interpretation needed.
- **HLS state-level broadcast loses within-state heterogeneity.** A western Iowa drought-affected county and an eastern Iowa unaffected county get the same `ndvi_mean` value at a given (year, forecast_date) because HLS is broadcast from state level. USDM has the same property. Both are documented as caveats in the dictionary; both are fixable in Phase D.1 (HLS) and possibly Phase B (drought via richer source) if the model wants the spatial detail.
- **The script's `geoid_to_state` lookup at line 320 of the original `merge_all.py` is dead code.** It was built for an HLS broadcast strategy that ended up being unnecessary because the merged DataFrame already carries `state_alpha` from NASS. Harmless; flagged for cleanup later.

**Open questions carried forward:**
- **MODIS NDVI as-of fidelity is weak** (whole-season summary at every forecast date). The locked decision is to ship as-is; the dictionary makes the trade-off explicit. Phase D.1 will replace with HLS-derived running NDVI clipped to forecast_date. If Phase B/C show the model is overfitting to end-of-season NDVI signal at early forecast dates, may need to revisit before D.1.
- **NASS-aux fields (`acres_*`, `irrigated_share`, `harvest_ratio`) are post-hoc reported.** Same value at all 4 forecast dates within a year. Treated as structural priors, not in-season measurements. If Phase C ablations show these features dominate at 08-01 (which would be suspicious), need to switch to lagged variants.
- **HLS slice schema cleanup is deferred to Phase D.1.** Not blocking Phase B/C since HLS is an ablation feature, not a baseline feature.
- **`dry_spell_max_days` is `int64` (no NaN). If a future pipeline run produces NaN, pandas will widen to float64 silently** or crash on parquet write depending on the path. Dtype-pin it if it ever matters.

**Phase 2-A.6 + A.7 closed cleanly.** Phase A is fully done; ready to start Phase B (analog-year retrieval baseline + cone calibration).

---

### Phase 2-C — Point-estimate model (XGBoost regressor) — 2026-04-25

**Context:** Phase B shipped post-recalibration with `same_geoid` pool, K=5, percentiles (10, 50, 90), per-county linear trend with state-median fallback, and per-(state, forecast_date) additive recalibration fit on val 2023. The Phase B post-recal gate passed on the 2024 holdout (coverage 80%, RMSE point < baseline at every forecast_date for K=5). Phase B left two open issues: (1) MO 2023 was a real-world drought outlier whose val-fitted recalibration broke; the regressor should learn `d2plus`/`vpd_kpa_silk`/`dry_spell` instead of baking a state constant. (2) WI plateau — recent-era yields flatter than the 18-year linear trend; SHAP should confirm the model learns this. Phase C gate per `PHASE2_PHASE_PLAN`: trained-regressor RMSE on 2023 val ≥ 15% better than Phase B analog-median (pre-recal) at end-of-season.

**Decided:**

- **Train target: raw `yield_target` + `year` as a feature.** Hackathon spec is silent on training-target choice; default applied. The WI plateau argument carries — a per-county linear detrend bakes in the assumption recent yields are still on the 2005–2022 line, exactly the failure mode Phase B documented. Letting the trees learn nonlinear time effects via `year` directly addresses it. Side note: this also keeps Phase C's point and Phase B's cone in two different conventions on purpose — the cone is interpretable (detrended-then-retrended), the point is accurate (raw with year-as-feature).
- **One model per forecast_date.** Four boosters, each trained only on its own date's slice (~5,580 train rows after min-history filter). Mirrors Phase B's per-date BallTree indices. At 08-01, the structurally-NaN grain features (`vpd_kpa_grain`, `srad_total_grain`) are dropped from the feature list rather than imputed.
- **State and year both in as features.** Spec is silent; default applied. State enters as 5 one-hots (`state_is_CO/IA/MO/NE/WI`). Year enters as a numeric. Together they let the trees learn state-specific time trends (the WI plateau can become a `state_is_WI × year ≥ 2018` interaction). Risk acknowledged: the model may shortcut through these structural features instead of weather signal — SHAP analysis is the verification step.
- **Feature set: full superset of master-table numerics.** Wider than Phase B's retrieval embedding (15/17 features) by design — XGBoost can absorb covariates the embedding holds back. Initial feature set was 35 cols at 08-01 / 37 at later dates: 9 weather (including held-back `edd_hours_gt90f`) + 2 grain (later dates only) + 5 drought (full D0–D4) + 5 MODIS NDVI + 11 gSSURGO soil + 4 management (`irrigated_share`, `is_irrigated_reported`, `harvest_ratio`, `acres_planted_all`) + 1 time + 5 state one-hots. Dropped vs the master table: `acres_harvested_all` (collinear with planted × harvest_ratio, post-hoc reported, target-leakage feel), `d2plus` (exact alias of `d2_pct` — including both adds zero info and breaks SHAP attribution), `yield_bu_acre_irr` (84% structural null, replaced by `is_irrigated_reported` indicator), HLS columns (59% null pre-2013, slated for redo in D.1).
- **Sample weights: none.** Every county-year is a real observation. Acres-weighting collapses CO into noise. State aggregation handles the weighting at evaluation time.
- **Hyperparameter sweep:** `max_depth ∈ {4,6,8}` × `learning_rate ∈ {0.05, 0.1}` × `min_child_weight ∈ {1,5}` = 12 configs/date × 4 dates = 48 fits. Each fit early-stopped on val RMSE (county-level, the early-stopping metric). Selection: per-date best by county-level val RMSE. Sweep wall time ~15s; deterministic with `seed=42 + tree_method=hist`.
- **Recalibration: dropped.** Initial Phase 2-C kickoff deferred this decision until residuals were seen. Looking at val 2023 residuals (post-NDVI-removal) — CO +9 to +18, MO +11, WI −10 — the pattern looks recal-shaped. But holdout 2024 residuals show totally different signs (CO ~0, MO −19, WI ~0). A val-fitted additive constant would help val and hurt holdout. Same failure mode that hit Phase B's MO 2023 recal, but more starkly in C. Dropping recal is the honest move; per-state bias goes in the Phase G writeup as a known limitation.
- **Gate comparator: Phase B analog-median PRE-recalibration on val 2023.** Pre-recal is the honest baseline for "did the regressor learn more than retrieval + simple bias correction." Comparing to post-recal would be a circular bar (the recal was tuned to val itself).
- **NDVI columns dropped from regressor's feature list (mid-Phase-C reversal).** This is the single biggest decision of Phase C and merits a full sub-entry — see **Phase 2-C.1** below. Brief version: SHAP analysis on the first trained bundle showed `ndvi_peak` dominating predictions at every forecast date including 08-01, confirming the data dictionary's documented as-of leakage warning had become a real model behavior. All 5 MODIS NDVI columns were removed from `_NDVI` in `forecast/regressor.py` (set to `[]` with a long block comment) and the bundle retrained.
- **Phase C gate PASSED on val 2023, post-NDVI-removal.** EOS lift +46.7% (threshold 15%); 08-01 lift +34.5%; 10-01 lift +35.6%; 09-01 lift −0.2% (regressor and analog-median tied). The phase plan explicitly says "earlier dates can be weaker; we just need to confirm the model adds value" — the EOS-only gate evaluation passes comfortably.
- **Module layout:** `forecast/regressor.py` (per-date `Regressor` + `RegressorBundle`, native xgb JSON save/load), `forecast/explain.py` (SHAP via `Booster.predict(pred_contribs=True)`, no `shap` library dependency), `scripts/train_regressor.py` (sweep driver), `scripts/backtest_phase_c.py` (regressor point + Phase B cone, gate eval), `scripts/smoke_explain.py` (assertion + eyeball test for explain). Bundle persists to `models/forecast/regressor_*.json` + `.meta.json` sidecars.

**Rejected alternatives:**
- **Train on detrended residual + retrend at predict.** Considered as the natural mirror of Phase B's cone math. Rejected because the WI plateau is exactly a non-linear time effect that a linear detrend bakes in wrong; passing `year` as a feature lets the trees learn it directly without the detrend's prior commitment.
- **Use the analog retrieval embedding as the regressor feature set.** Considered for strict apples-to-apples comparison. Rejected — the embedding is intentionally narrow for L2-distance interpretability; the regressor can absorb more without distance metric concerns.
- **Hand-engineered interactions** (e.g. `vpd_kpa_silk × d2plus`, `gdd_cum_f50_c86 / aws0_100`). Considered. Rejected for the gate-passing build — XGBoost should find these through tree splits if the signal is there. SHAP confirmed weather features are reaching reasonable importance after NDVI removal; engineered interactions could be a Phase G refinement if the ablation table shows headroom.
- **Sample weighting by acres.** Rejected. Treats large-acre counties as more important for loss; collapses CO mountain counties and small WI counties into noise. State aggregation does the right weighting at evaluation time.
- **Rolling-origin cross-validation** (train through Y, val on Y+1 for Y ∈ {2020, 2021, 2022}, average). Considered for more stable hyperparam selection. Rejected per phase plan single-fold spec; 3× compute for marginal sweep stability.
- **State-rolled-up RMSE as the early-stopping metric.** Rejected — only 5 obs per (state, date), too few to drive sweep selection. County-level (308 obs/date) is the right gradient signal; state RMSE shows up at the gate eval, not the sweep.
- **Refit the picked config after the sweep.** Rejected — `tree_method=hist` with fixed seed is deterministic, so refit produces the identical booster. Saves ~25% of wall time.
- **Recalibrate the regressor's val errors per-(state, date)** (Phase B's pattern, applied to Phase C). Rejected after seeing val 2023 vs holdout 2024 residual sign flips. Same failure mode as B's MO 2023 recal.
- **Drop only the top 3 NDVI features (`ndvi_peak`, `ndvi_silking_mean`, `ndvi_gs_mean`).** Considered as a half-measure when the leakage was identified. Rejected — the as-of leakage applies equally to every MODIS NDVI column (all are whole-season summaries per the data dictionary). All 5 dropped; cleaner story.
- **Defer the NDVI fix to Phase D.1** (where HLS-derived running NDVI replaces the leaky MODIS columns anyway). Rejected. (a) The gate verdict +51.6% would be a number we'd cite in the deck despite knowing chunks of it came from a feature that shouldn't be available at 08-01 — dishonest. (b) Phase D.1 is weeks of work (TB-scale HLS pull, Prithvi integration); a known-broken Phase C sitting on shelf for that long blocks Phase E (frontend) and Phase F (agent), both of which consume Phase C outputs. The agent narrating "the top driver of this Aug 1 forecast is end-of-September NDVI" is incoherent. (c) The fix is small: drop 5 column names, retrain (15s), re-backtest (26s).

**Surprises / learnings:**
- **The val sweep is essentially flat (1.3–1.6 bu RMSE spread across 12 configs at every date).** Tabular yield prediction with 5,580 training rows is a well-regularized problem; the loss surface isn't sharply peaked. Implication: don't sweep harder, change the features.
- **Per-date best-config picks were inconsistent** (depth=4 at one date, 6 at another, learning_rate split between 0.05 and 0.1). Single-year val with 5 state-level obs/date is too noisy to drive consistent selection. A more conservative move would have been one config across all dates picked by averaged val RMSE; not done because the per-date picks all landed within sweep noise of each other anyway.
- **Initial training (with NDVI in the feature set) gate-passed at +51.6% lift at EOS,** which masked the underlying problem. Without SHAP we'd have shipped the leaky model. **The data dictionary explicitly warned about MODIS NDVI's whole-season-summary as-of weakness** ("This violates strict as-of for the 08-01 and 09-01 forecast dates"). Phase A documented the risk; Phase C had to actually feel it to act on it. **Lesson: documented data caveats need to be paired with model-behavior checks (SHAP on the first trained bundle), not just feature-engineering choices.**
- **`ndvi_peak` was the #1 driver at 08-01, 09-01, 10-01, AND EOS** (mean |SHAP| 17.4 vs 6.2 for the next feature). The model didn't treat it as a soft prior — it treated it as the answer. SHAP is the only way this would have surfaced; Phase B's analog cone uses NDVI in retrieval but the percentile construction over K=5 detrended yields makes any one feature less load-bearing.
- **After NDVI removal, `year` became the dominant feature** (mean |SHAP| 13.4, double the next). Partly fine: corn yield trends ~2 bu/acre/year from genetic gain. Partly worrying: the trees are essentially "predict the trend, perturb by weather." When 2024 sits off the recent trend (IA exceptional, MO normal-after-drought), the year-anchored prediction misses by exactly the amount the year doesn't predict.
- **Holdout 2024 numbers tell a different story than val 2023.** Regressor wins on val by a large margin (+46.7% EOS lift) but LOSES to analog-median on holdout at every date. This is the textbook sign of single-fold val + hyperparam selection on the same year + small state-level sample (5 obs/state-date). The honest read: val numbers reflect "did the model train sanely"; 2024 holdout reflects "does it generalize." It doesn't, on the two strongest states.
- **IA-2024 (regressor bias −15 bu) and MO-2024 (regressor bias −19 bu) are the holdout failures.** Both are years where the model's year-anchored prior didn't match the actual outcome. IA 2024 was an exceptionally good year that sits at the high end of the 2005–2022 distribution; MO 2024 was a normal year following 2023's drought, and the model likely learned to hedge MO downward. The analog-median doesn't have this prior — it retrieves similar-weather years from any decade. **Lesson: gradient-boosted trees are conservative at the tails of the training distribution by design; this is a structural limitation, not a sweep tuning problem.**
- **09-01 is consistently the worst date** (val RMSE highest across the sweep, lift collapses to −0.2% at the gate) — surprising because it's the first date where grain-phase weather features are populated, so we'd expect more signal not less. Plausible: 2023's grain-phase profile may not be representative; or the trees haven't yet found a stable use of the new features with only one val year. Not chasing this — fixing 09-01 specifically would be tuning to noise.
- **MO 2023 drought story works as predicted post-NDVI-removal.** Pre-recal analog-median over-predicts MO 2023 by +23 bu (the WI/MO drought outlier issue from Phase B). Regressor over-predicts by +8 to +11 bu. The regressor is reading the drought signature (`d2_pct`, `vpd_kpa_silk`, `dry_spell_max_days` all in top 10 mean |SHAP|) instead of baking MO as a state constant — exactly what we wanted from kickoff issue (1).
- **Gate verdict is real but qualified.** +46.7% EOS lift on val is the official Phase C gate number and is real. But: leans heavily on `year` and `acres_planted_all` (structural priors) more than on weather signal; loses to analog-median on 2024 holdout; 09-01 essentially ties analog-median. Phase G's ablation table is the natural place to surface all of this; the gate passes per the locked rule.

**Open questions carried forward:**
- **Phase D.1 imagery is the natural next lever** for in-season visual signal. The regressor has no remote-sensing features at all post-NDVI-removal (HLS columns excluded due to pre-2013 sparsity + state-broadcast); D.1's Prithvi-over-HLS chips clipped to forecast_date is the as-of-honest replacement.
- **2024 holdout generalization gap** is documented as a known limitation. Phase G's ablation table and per-state error breakdown will be the official record. If Phase D.1 + Prithvi narrows the IA-2024 / MO-2024 misses, the v2 pipeline is genuinely better; if not, the analog cone's cross-decade retrieval is the more durable point estimate and we ship it that way.
- **`year` dominance** could be revisited if/when the feature set widens. Engineered interaction terms (`state_alpha × year` explicit, `vpd_kpa_silk × d2_pct` for drought-stress, etc.) could redistribute SHAP attribution toward weather. Defer until Phase D.1 results are in — adding interactions before more features is premature.
- **Per-state bias is documented, not corrected.** Val 2023 shows CO +9 to +18, MO +5 to +11 (post-NDVI), WI −4 to −11. Holdout 2024 shows different signs. A val-fitted recal would help one and hurt the other. If Phase D.1 imagery doesn't shrink these biases organically, may need to revisit — possibly with a more robust calibration approach (e.g. fit on multiple years' rolling residuals).
- **09-01 anomaly** — fix candidates if a future iteration wants to chase it: month-resolution weather aggregates instead of phase-window; per-date feature lists that drop noisy features at 09-01 specifically; multi-year val to stabilize selection.

**Phase 2-C closed.** Gate passed on val 2023 EOS at +46.7% lift. Bundle persisted to `models/forecast/`. Phase E (backend + frontend) and Phase D (Prithvi) can both proceed; they're independent. Phase G holdout evaluation reuses `scripts/backtest_phase_c.py` against the 2024 pool with the as-built bundle.

---

### Phase 2-C.1 — NDVI leakage discovery and feature-set fix — 2026-04-25

**Context:** Sub-entry inside Phase 2-C, broken out because the decision had its own deliberation arc and reverses an implicit Phase 2-pre commitment ("MODIS NDVI is the primary remote-sensing feature for Phases B and C"). The reversal is *only* for Phase C (point estimate); MODIS NDVI stays in Phase B's retrieval embedding.

**Trigger:** First Phase C training run gate-passed at +51.6% EOS lift on val 2023, but `scripts/smoke_explain.py` (run as a sanity check on `forecast/explain.py`) showed:
- `ndvi_peak` mean |SHAP| = 17.4 — 3× the next feature
- The 3 NDVI columns combined exceeded every other feature put together
- `ndvi_peak` was the #1 driver at 08-01, 09-01, 10-01, AND EOS

The data dictionary (`PHASE2_DATA_DICTIONARY.md`) had explicitly flagged this as a risk: "NDVI columns are WHOLE-SEASON SUMMARIES — same value at all 4 forecast_dates within a year, integrated over DOY 121-273 (May 1 → Sep 30). They are *not* clipped to the forecast_date — at `08-01` you still see end-of-September NDVI. **This violates strict as-of for the 08-01 and 09-01 forecast dates.**" The Phase 2-pre locked decision had been to ship anyway as a "trend feature" (soft prior); SHAP showed the regressor was treating it as a hard answer, not a soft prior.

**Decided:**
- **All 5 MODIS NDVI columns (`ndvi_peak`, `ndvi_gs_mean`, `ndvi_gs_integral`, `ndvi_silking_mean`, `ndvi_veg_mean`) removed from `_NDVI` in `forecast/regressor.py`.** `_NDVI = []` with a long block comment explaining the leakage and pointing at the Phase D.1 replacement (HLS-derived running NDVI clipped to forecast_date, embedded via Prithvi).
- **The reversal is point-estimate-only.** Phase B's retrieval embedding keeps its 2 MODIS NDVI columns (`ndvi_peak`, `ndvi_gs_mean` per `forecast/features.py`). Rationale: retrieval matching is less sensitive to as-of leakage than point prediction. An analog with similar end-of-season NDVI is *also* an analog with similar weather/management (NDVI is partly downstream of those); the cone takes percentiles over K=5 detrended yields, so no single feature dominates the answer. The leakage that breaks point-estimation does not equivalently break retrieval matching. Documented in the long block comment in `regressor.py`.
- **Bundle retrained immediately.** Sweep wall time 14.7s; backtest 25.8s; smoke test seconds. Total ~5 minutes of compute to reach the post-fix gate verdict.
- **Post-fix gate result is the official Phase C number.** EOS lift +46.7% (threshold 15%); pre-fix +51.6% number is mentioned in this entry but not used as the headline. The +46.7% reflects the legitimate signal the regressor learned without leakage.

**Rejected alternatives:**
- **Drop only the top-3 NDVI features by SHAP.** Half-measure. The as-of leakage applies to every MODIS NDVI column (all whole-season summaries); cherry-picking three feels like model-tuning rather than principled fix.
- **Defer to Phase D.1.** D.1 will replace MODIS NDVI with HLS-derived running NDVI anyway, so "the fix is coming." Rejected because (a) the +51.6% gate verdict would be a number cited in the presentation while we knew chunks of it came from a feature that shouldn't be available at 08-01 — dishonest; (b) D.1 takes weeks, leaving Phases E and F (which consume the regressor) blocked or built against a known-broken model. The agent narrating "the top driver of this August 1 forecast is end-of-season NDVI" would be incoherent and visible to judges.
- **Keep NDVI but apply a per-forecast-date clip** (manual zero-out of NDVI at 08-01 and 09-01, keep at 10-01 and EOS). Rejected — the value is still identical across dates within a year (it's the same summary), so the clip is just "set to zero at early dates," which is information the model could reverse-engineer through interactions with `year`. Also conceptually muddy: the column's semantics are the same at every date by construction. Drop or replace, don't fudge.

**Surprises / learnings:**
- **A documented data caveat became a model behavior** the moment the model had a chance to exploit it. The data dictionary's warning had been written; the Phase 2-pre decision to ship anyway as a "trend feature" had been logged. Neither prevented the leakage from showing up. **Lesson: written caveats are necessary but not sufficient. Pair them with empirical checks (SHAP on the first trained bundle) before declaring a model done.**
- **Phase B's analog cone showed a different (legitimate) reliance on NDVI.** B uses MODIS NDVI in the retrieval embedding to find weather-similar analogs; the cone's percentile construction over K=5 detrended yields prevents any one feature from being load-bearing. C's regressor had no such constraint — gradient-boosted trees, given a feature that's strongly predictive in-distribution, will lean on it as hard as the loss function rewards. **Lesson: the same data feature can be safe in one model and unsafe in another; the model architecture's response to feature dominance is part of the leakage analysis.**
- **The `forecast/explain.py` SHAP module justified its own existence within an hour of being built.** Originally scoped as Phase F's narration support tool. Used in this iteration as a leakage diagnostic and feature-set audit. Same primitives serve both jobs; building it before the gate-pass declaration was the right call.
- **The sweep got slightly *less* flat post-NDVI-removal** (val RMSE spread 19.2–23.9 across configs vs 15.3–18.1 before). Removing a dominant feature exposed more sweep sensitivity, as expected. Configs converged on simpler trees (`max_depth=4` picked at 3 of 4 dates vs depth=4 at 2 of 4 before) — the model genuinely has less signal to work with and benefits from less capacity.
- **The post-fix +46.7% lift on val is real signal**, not artifact. With NDVI removed and `year` dominant, the model is "predict-the-trend + weather-perturb" — which still beats acres-weighted-mean-of-K=5-county-analog-medians by a big margin because the analog cone has no genetic-gain mechanism (it pulls 2008 analogs as readily as 2022 analogs).

**Open questions carried forward:**
- *(none — decision is closed for Phase C; Phase D.1 will revisit remote-sensing features with the right as-of fidelity)*

---