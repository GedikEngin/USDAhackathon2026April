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
- **NDVI script structure: per-year Export tasks, not a single combined task.** Mirrors what produced the 21 per-year CSVs already on disk. Cleaner failure mode (one bad year doesn't kill the whole run) and the per-year files merge trivially in `merge_all.py`. Locked in the version-controlled `scripts/ndvi_county_extraction.js`.
- **NDVI is pre-scaled in the CSVs** — the GEE script applies `× 0.0001` server-side, so the NDVI columns are already floats in roughly `[-0.2, 1.0]`. Downstream code (specifically `merge_all.py` and any feature engineering) MUST NOT re-apply the scale factor. Documented in both the script header and the inventory doc.
- **Keep all 21 NDVI years (2004–2024), do not filter early years.** The brief asks for 2005–2024; 2004 is "free" from the modified script and stays. CDL-coverage variation across years is real but Phase B retrieval is per-county, so a CO county with thin pre-2008 history just gets fewer analog candidates than an IA county — that's a feature, not a bug.
- **CDL coverage map locked into the docs.** Empirical coverage by state and year (counted from the actual exports): IA/NE/WI full from 2004; MO full from 2006; CO full from 2008. Strong block 2008–2024. The ~30 counties always missing post-2008 are real corn-absent counties (St. Louis City, Denver, mountain counties).
- **TIGER 2018 county polygons stay** as the spatial reduction layer. 443 counties = 5-state count. Independent cities (St. Louis 29510) come along for the ride and emit nulls — handled downstream.
- **Drought feature engineering deferred to Phase A.5** as a small standalone script (`scripts/drought_features.py`). Raw weekly CSV is already on disk, so this is local-only work; no more pulls.

**Rejected alternatives:**
- **Reconstruct the GEE script from memory or have the user paste it in** — Reconstruction-from-spec was chosen instead. The version-controlled script is functionally equivalent to what produced the existing CSVs (same schema, same bug fixes, same per-year structure) but is not byte-identical to the editor version. Risk is bounded because all 21 years already exported successfully; the saved script is for re-runs and 6th-state extension.
- **Filter out 2004–2007 to avoid heterogeneous coverage** — Rejected. Per-county retrieval makes coverage-variation a non-issue, and dropping years narrows the analog pool unnecessarily.
- **Update `merge_all.py` plan to apply NDVI scale factor at merge time** — Rejected. The CSVs are already scaled; re-applying would silently corrupt the data. Safer to lock "do not re-scale" as a documented invariant.

**Surprises / learnings:**
- **Docs were stale.** `PHASE2_CURRENT_STATE.md` claimed NASS extension and NDVI export were "to do" or "in flight" — both were complete and on disk. `nass_features.py` and the engineered features CSV existed but weren't acknowledged. Doc-drift is the failure mode in this kind of workflow; the fix is updating the current-state doc at the end of each working session, before the next chat starts.
- **CDL coverage cliff is at 2008, not 2005.** The plan-side language ("CDL 2005 is the earliest available; corn-pixel masking works back to 2005") was technically true but misleading: CDL existed for *some* states in 2005, not all five. Iowa/Nebraska/Wisconsin had CDL all the way back; Missouri came online 2006; Colorado 2008. This is now in the inventory doc.
- **NDVI per-year file count (443) is non-obvious.** It's the TIGER 2018 county count for the 5 states, including independent cities and corn-absent mountain counties. The "useful" count is closer to 410 in good years. Don't be alarmed by the 443 in `wc -l`.

**Open questions carried forward:**
- Drought-feature aggregation: which D-level severities to expose (sum of D2+? all five percentages?), and the resampling rule from weekly to forecast-date as-of (last reading? trailing-N-week mean?). Decide in Phase A.5.
- Once PRISM lands, the SPI/SPEI vs. USDM question can be settled empirically — both will be available. Default plan: keep both, let the model decide.
- Confirm exact non-irrigated coverage in the post-extension NASS CSV. Original 2015–2024 cut had 0% on those columns; need a one-line `df.notna().mean()` to confirm whether the rate-limit fixes filled them in for 2005–2014.

---

### Phase 2-A.3 — gSSURGO soil features extracted — 2026-04-25

**Context:** State `.gdb` files for all 5 states already on disk in `phase2/data/gSSURGO/gSSURGO_<state>/`. Phase A.3 was about turning those rasters into a per-county tabular file.

**Decided:**
- **Use the Valu1 table from each state `.gdb`** as the source of soil properties, not the raw component/horizon tables. Valu1 is USDA's own pre-aggregation to MUKEY level and covers 90% of common ML use cases including the corn-specific NCCPI index. Saves a substantial amount of weighted-mean computation that would just reproduce what USDA has already done.
- **Final column set:** `nccpi3corn, nccpi3all, aws0_100, aws0_150, soc0_30, soc0_100, rootznemc, rootznaws, droughty, pctearthmc, pwsl1pomu`. Eleven properties. NCCPI corn is the gold-standard single feature; the rest cover water-holding capacity at two depths, organic carbon at two depths, root-zone properties, the drought-susceptible-soil flag, and the wetland-soil indicator. `pctearthmc` is included because it's a useful denominator for sanity-checking Earth-vs-water vs-urban-pixel ratios when interpreting other features.
- **Reproject TIGER 2018 county polygons into EPSG:5070 (Albers Equal Area)** to match gSSURGO's native CRS, then run zonal stats. Reverse-direction (reprojecting gSSURGO into county CRS) would resample 10m soil rasters and burn time + introduce sampling error.
- **Output keys on `GEOID` only.** Soil is static across years; broadcast across `year` happens at merge time in `merge_all.py`. This avoids 20× duplication on disk and makes the table semantically obvious.
- **TIGER 2018 5-state subset cached as `phase2/data/tiger/tl_2018_us_county_5states_5070.gpkg`** to skip re-filtering + re-projecting on every run.

**Rejected alternatives:**
- **Use raw component / horizon tables and compute weighted means manually.** Higher fidelity in principle but burns substantial time, and Valu1 already encodes the depth-weighting convention USDA recommends. Not worth the complexity at the county-aggregation scale we're operating at.
- **Use the CONUS gSSURGO instead of state-by-state.** CONUS is only available at 30m, not 10m, and >40GB. State 10m is already on disk and gives 9× the spatial resolution for free.
- **Store gSSURGO features per `(GEOID, year)` for symmetry with other tables.** Rejected — soil doesn't change year-over-year and storing 20 copies of the same row is silly. Broadcast at merge time is the cleaner pattern.

**Surprises / learnings:**
- **443 county polygons, but only ~410 corn-bearing.** The remaining ~30 (independent cities, urban counties, CO mountain counties) get gSSURGO values fine — soil doesn't disappear just because you can't grow corn there. They emit non-null gSSURGO rows but null NDVI/NASS rows; the merge handles this naturally.
- **Valu1 is well-documented but the column names are USDA-internal.** `nccpi3corn` (National Commodity Crop Productivity Index, corn variant), `aws0_100` (available water storage 0–100 cm), `pwsl1pomu` (potential wetland soils, level 1, percent of mapunit). Worth one-line glosses in the data dictionary.

**Open questions carried forward:**
- Whether to add depth-weighted texture / pH / OM / CEC. The brief's "soil quality" intent is well-served by NCCPI alone; the other Valu1 columns add flesh. Adding more requires going back to component/horizon tables. Defer until Phase B/C show the model wants more soil signal.

---

### Phase 2-A.4 — Daily weather pulled and feature-engineered — 2026-04-25

**Context:** The slow long-pole pull. Originally scoped as PRISM, decided as gridMET when the script was written.

**Decided:**
- **gridMET, not PRISM, as the daily weather source.** Both are 4km daily surfaces over CONUS; both publish `tmax, tmin, prcp, srad, vp` (gridMET also has wind and humidity if needed later). At the county-aggregation scale we're operating at, there's no quality difference. gridMET was easier to acquire programmatically and the netcdf cache pattern (`data/v2/weather/raw/_gridmet_nc_cache/`) makes idempotent re-runs trivial. PRISM is a fine fallback if gridMET ever shows quality issues — moves to ⚫ deprioritized for now.
- **Daily county aggregation done in the pull script**, not at feature time. `gridmet_pull.py` produces `gridmet_county_daily_<year>.parquet` with one row per (GEOID, date). Then `weather_features.py` runs over the combined parquet with no further spatial reduction. Clean separation.
- **Combined parquet (`scripts/gridmet_county_daily_2005_2024.parquet`)** as the input to feature engineering, alongside the 20 per-year parquets that are kept for traceability and partial re-runs.
- **GDD: Fahrenheit base 50 / cap 86, with both endpoints capped before averaging.** This is the McMaster & Wilhelm / NDAWN / Iowa State Extension convention, not the raw `(tavg − 50)` variant. Capping `tmin` as well as `tmax` matters in the early/late season when nights are cold (raw average would credit growth that didn't happen).
- **EDD/KDD: degree-hours above 86°F and 90°F via single-sine hourly interpolation** (Allen 1976 / Baskerville-Emin 1969). Closed-form integral over the daily sine arc gives a smooth, vectorizable formula that handles the three cases: entire day below threshold (0), entire day above threshold (24×excess), and threshold-crossing (sine-arc integral). Captures sub-daily heat exposure that simple `max(0, tmax − 86)` misses.
- **Phase windows (DOY ranges):** vegetative 152–195, silking 196–227, grain fill 228–273. Locked. These are the standard corn phenology windows for the Corn Belt and align with how agronomists talk about heat/water stress.
- **VPD computed daily, then averaged within each phase window.** Mean (not max) because corn responds to integrated stress, not peak.
- **Solar radiation summed (not averaged) within each phase window.** Cumulative MJ/m² is the agronomically meaningful quantity for biomass accumulation.
- **Cumulative precipitation from May 1 → cutoff** + **longest dry-spell run (`<2 mm/day`)** as separate features. Captures both volume and distribution of water stress.
- **As-of safety lives in one place.** `build_features_for_cutoff(df, year, cutoff_date)` slices the daily df at the very top with a single `date <= cutoff_date` mask. Nothing downstream of that slice can see post-cutoff data. Phase windows are clipped to `cutoff_doy` so e.g. on 08-01 the silking aggregate only covers DOY ≤ 213, not the full silking window. This is the single point of leakage control and the function header documents it.

**Rejected alternatives:**
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

### Phase 2-B — Analog-year retrieval baseline shipped, gate passed post-recalibration — 2026-04-25

**Decided:**
- **Primary analog pool = `same_geoid` (within-county history), not cross-county.** The PHASE_PLAN B.2 spec said cross-county; the first backtest run revealed cross-county was actively broken on CO (RMSE 14-15 bu vs same_geoid 10 bu) because flat-L2 distance over the engineered embedding pulls weather-similar but soil/management-dissimilar neighbors. Same_geoid retrieval is interpretable ("this county's K most weather-similar past years"), aligns with the brief's wording, and produces calibrated cones. Cross-county retained as a `pool="cross_county"` flag in `forecast/analog.py` for diagnostic comparison and for future Phase D.1 work where Prithvi embeddings may make the cross-county distance more meaningful.
- **K = 5 chosen as winning configuration.** K-sweep across {5, 10, 15} on same_geoid showed K=5 had best RMSE at every forecast_date for every state, lowest mean cone width (~25 bu) while still passing 80% coverage, and is the smallest K that meaningfully averages out individual-analog noise. Larger K produces wider cones and worse RMSE because it pulls in less-similar analogs.
- **Per-county linear trend, not per-state.** Original `StateTrend` used per-state OLS; diagnostic on WI showed the per-state trend systematically under-fit (14 of 18 train years had positive residuals), which biased every WI same-GEOID prediction upward by ~12 bu. Refactored to `CountyTrend`: per-GEOID OLS with state-median fallback for counties below the min-history threshold (none in practice with min_history=10). API keying changed from `state_alpha` to `GEOID`. Per-county refactor fixed CO (RMSE 4.8 → 2.8) but did NOT fix WI — proving per-state vs per-county wasn't the WI problem.
- **Retrieval-embedding feature set (17 cols at 09-01/10-01/EOS, 15 at 08-01):** `gdd_cum_f50_c86`, `edd_hours_gt86f`, `vpd_kpa_veg`, `vpd_kpa_silk`, `prcp_cum_mm`, `dry_spell_max_days`, `srad_total_veg`, `srad_total_silk`, then `vpd_kpa_grain` + `srad_total_grain` (only at 09-01/10-01/EOS — structural NaN at 08-01), `d0_pct`, `d2plus`, `ndvi_gs_mean`, `ndvi_peak`, `nccpi3corn`, `aws0_100`, `rootznaws`, `irrigated_share`. Per-forecast-date column lists: at 08-01 the *_grain features are dropped from the embedding entirely (rather than imputed) because grain-fill physically hasn't started.
- **Per-forecast-date standardization.** Z-score parameters fit independently per (forecast_date, feature) on the train pool, because cumulative weather variables have wildly different distributions at Aug 1 vs EOS (cumulative GDD ~1,600 vs ~3,700). Single-pool standardization would compress the 08-01 rows into the low end of each feature axis and destroy embedding discrimination at early forecast dates.
- **Min-history filter N=10.** A county is a *candidate* in the analog pool if it has ≥10 qualifying training years (all 4 forecast_dates complete + non-null target). 345 of 388 GEOIDs survive. Counties with <10 qualifying years can still be *queried* (we forecast for them); they just can't pollute another county's analog pool with noisy partial history.
- **Per-(state, forecast_date) additive recalibration, fit on val (2023), applied to holdout (2024).** Required for the Phase B gate. State-systematic bias diagnosed in the un-recalibrated run: WI +12 bu over-predict, NE +5, CO -2.5, MO ±20+ (split year), IA calibrated. Mean signed `point_error` per (state, date) on val rows = the recalibration constant; subtract from holdout point estimate AND from every cone percentile (preserves cone width, shifts location). Implementation in `forecast/recalibrate.py`. Gate evaluation now scores **2024-only post-recalibration** because the val year is in-distribution at fit time.
- **Per-county yields detrended → analogs matched on weather/soil/management → percentiles taken in detrended space → retrended at the query (geoid, year).** This is the analog matching/retrieval flow. Detrending isolates the weather-deviation signal from genetic-gain signal. The cone is "deviation from this county's trend line" before retrending; retrending puts it back in raw bu/acre at the query point.
- **Cone percentiles = (10, 50, 90).** Reports as P10 / P50 / P90. Point estimate = retrended median of analog detrended yields. Cone width target ~25-30 bu/acre at K=5.
- **State aggregation = planted-acres-weighted mean of county percentile values.** Documented caveat: percentiles don't average linearly. Acceptable for v2 baseline; the strictly-correct alternative (pool all analog yields across the state weighted by their counties' acres, then take percentiles) is a Phase B v2 if the gate fails. We pass the gate, so we ship as-is.
- **Naive baseline = trailing 5-year county mean of `yield_target`, planted-acres-weighted to state.** Min 3 years required, holdout years (2024) excluded from lookback regardless of query year. Implementation in `forecast/baseline.py`.
- **Phase B GATE PASSED post-recalibration on 2024 holdout for primary pool same_geoid:** coverage 80% across all (K, forecast_date) configurations, RMSE point < RMSE baseline at every forecast_date for K=5. Pre-recalibration: gate FAILED on coverage being too high (80-100%, conservative-side failure). Post-recalibration on 2024: coverage exactly 80%, RMSE 11.3-14.3 vs baseline 14.5.

**Files shipped:**
- `forecast/__init__.py` (implicit; package directory exists via the modules below)
- `forecast/features.py` — embedding column lists per forecast_date + Standardizer
- `forecast/detrend.py` — CountyTrend with per-county OLS + state-median fallback. StateTrend alias preserved.
- `forecast/data.py` — load_master() + train/val/holdout pool slicers + min-history filter
- `forecast/analog.py` — AnalogIndex with per-date BallTrees, supports `pool='cross_county'` and `pool='same_geoid'`
- `forecast/cone.py` — build_cone() over detrended analog yields, retrends at (query_geoid, query_year)
- `forecast/aggregate.py` — CountyForecastRecord + state_forecast_from_records (planted-acres weighted)
- `forecast/baseline.py` — county_baseline + state_baseline naive 5-yr mean
- `forecast/recalibrate.py` — Recalibrator fit_from_val_results + adjust_state_forecast
- `scripts/backtest_baseline.py` — end-to-end harness with K-sweep, pool-sweep, recalibration, gate evaluation
- `scripts/diagnose_wi_overshoot.py` — exploratory diagnostic (preserved as-is for git history; pre-CountyTrend API)
- `scripts/diagnose_state_errors.py` — bias-vs-variance decomposition reading the latest backtest CSV

**Rejected alternatives:**
- **Cross-county retrieval as primary.** Rejected after first backtest run showed CO RMSE 14+ bu and systematic under-prediction across all states except CO. Diagnosis: flat L2 distance treats every embedding dimension equally, so 13 weather/drought/NDVI dimensions drown out the 4 soil/management dimensions, and the K-NN finds weather-similar neighbors from other counties whose soil/management profiles are wrong for the query. Retained as a `pool="cross_county"` flag for future use.
- **K=10 baseline (per PHASE_PLAN B.2).** Tried; same_geoid K=10 had RMSE 9.8-10.2 with 90% coverage. K=5 was tighter on every state with 80% coverage. K=10 is preserved as a sweep option; K=5 is the shipped default.
- **Larger K to chase cross-county lift.** Tried K=20 on cross-county. Made things worse — averaged over more dissimilar analogs.
- **No-detrend variant.** Considered after diagnosing WI bias. Diagnostic showed median raw analog yield was 183 vs truth 195 for the WI county we drilled into; raw under-predicts by 11. Detrend-and-retrend over-predicts by 7. Neither is right; the issue is bias in the analog pool, not in the detrend itself. Decision: keep detrending, fix bias via recalibration.
- **Trend fit on last 5 train years only (CF2 in `diagnose_wi_overshoot.py`).** Tested. Produced a *steeper* WI slope (+3.26 vs +2.52) which made the over-prediction worse. Recent trend fit is not a fix.
- **Clip retrend horizon to analog_year + 5 (CF3 in `diagnose_wi_overshoot.py`).** Promising for one WI county (error +1.1 vs +7.2 default) but is a hack — the principle is "trends are real but only valid over short horizons," which is empirically defensible but adds a tunable parameter. Rejected in favor of the more principled per-state recalibration that does the same job downstream.
- **Per-county recalibration constants.** Considered. Rejected — 18 train years per county is too few for stable per-county constants. State-level (5 states × 4 dates = 20 constants) is the right grain.
- **Leave-one-year-out recalibration over training pool instead of fitting on val.** Considered. Would require 18 BallTree rebuilds × 4 dates ~90s of additional compute. Rejected because: (a) val year exists *for* this kind of calibration, (b) the val year is closer in distribution to the holdout than 2008-era LOYO residuals would be (yields trend over time), (c) plan explicitly defines val as 2023 for hyperparameter/calibration tuning.
- **Symmetric percentiles (P15/P50/P85).** Briefly considered to chase the [70, 90] coverage band before recalibration was implemented. Rejected because narrowing the cone is dishonest — the cone was over-covering, not miscalibrated. Recalibration fixed the gate the principled way (correcting the location bias) without artificially narrowing.

**Surprises / learnings:**
- **The detrend/retrend math near-cancels for same-GEOID retrieval.** Per-county trend refactor changed CO numbers (CO RMSE 4.8 → 2.8) but left WI numbers identical to 4 sig figs. Because in same-GEOID retrieval the analog years are detrended against the *same county's trend line* and retrended at *the same county's trend line at query year*, the trend transformation contributes only the slope × (query_year - mean_analog_year) term. For analog years tightly clustered near the query year, this is near-zero. The fix had to be at the recalibration layer, not the trend layer.
- **The K-NN doesn't pull "average" analog years; it pulls weather-similar years which can be biased toward recent good years.** For a "normal" 2024 WI query, the 5 most weather-similar past years included 2022, 2021, 2015, 2008, 2006. Four of five were positive-residual years against the trend. Median detrended residual = +25.5. That's not a bug in the K-NN — it's the K-NN faithfully matching weather similarity, but weather-similar ≠ residual-balanced.
- **MO 2023 was a real-world drought outlier that broke the val-fitted recalibration on MO 2024.** MO 2023 truth = 150 bu/acre (state-aggregated), forecast = 170+, error = +22. Recalibration constant for MO became -22. MO 2024 truth = 186 (a normal year), forecast pre-recal ~175, post-recal ~152. We now under-predict MO 2024 by 26 bu. **MO is now in 0% coverage.** Headline gate still passes because IA's huge baseline-beat (15.7 → 5.1 RMSE) carries the state aggregate, and 4 of 5 states are in cone at every date. Phase C should learn the MO 2023 drought signal from weather/USDM features and not bake it in as a state constant.
- **The cone-of-uncertainty methodology has a structural blindspot at one-year-fitted recalibration.** If val has an outlier state-year, the recalibration overshoots. This is a v2-specific issue — Phase C's XGBoost will learn signal-from-features rather than constants-by-state, removing the single-point-of-failure structure. Documenting here so we don't relitigate when Phase C results show different state-error patterns.
- **Bias-vs-variance decomposition is the cleanest diagnostic for whether a fix is plausible.** `scripts/diagnose_state_errors.py` showed CO and NE were ~75-86% bias-dominated (fixable by additive shift); MO was ~38% bias / ~62% variance (split year, additive shift can't fix). That decomposition is what told us recalibration would work for some states and overcorrect for others. Worth preserving as a permanent diagnostic.
- **The pre-recalibration gate FAILED only on the conservative side (coverage too high, 80-100%, vs target 70-90%).** This is qualitatively different from a too-narrow cone (which would lie about uncertainty). The post-recalibration coverage drops to exactly 80% across all configurations because shifting the cone center brings 1 of 5 states out of cone (MO 2024). The gate's coverage band is now satisfied at the extreme, not by being well-centered — flagging in case Phase C's improved point estimates re-shift coverage.

**Open questions carried forward:**
- **MO 2023 drought**: Phase C XGBoost should learn the d2plus / dry_spell_max_days / vpd_kpa_silk signature of that year. If it does, MO 2024 recalibration becomes near-zero (no false bias) and MO 2024 RMSE drops from ~26 to single-digit. If it doesn't, the weather features genuinely don't capture what drove MO 2023 — possibly an issue with NDVI/HLS not being available at county-level, or a non-weather factor (e.g. nitrogen availability, planting delay).
- **WI plateau**: Phase C should learn that recent-era WI yields are flatter than the 18-year trend implies. SHAP analysis on Phase C trained model can confirm whether the year feature (or a year-proxy) carries a flat-then-flat structure.
- **Cross-county pool revival in Phase D.1.** With Prithvi-derived embeddings, the L2 distance over (engineered + Prithvi) features may be discriminating enough to make cross-county retrieval meaningful. Test by swapping the pool flag in `analog.py` after D.1.
- **Recalibration on a swap-validation set (fit on 2024, apply to 2023).** Would test whether 2024-as-holdout passing is a one-shot fluke or a real result. Not run; flagged for Phase G validation pass.
- **Cone aggregation caveat (percentiles don't average linearly).** Currently we acres-weight per-percentile. Strictly correct: pool weighted analog yields, take percentiles. Same Phase G validation work.
- **State aggregation does NOT use NASS-published state yields as truth.** We use planted-acres-weighted mean of county yields per `state_truth_from_master()` in the backtest harness. NASS's own state aggregation uses the same construction modulo small-county disclosure suppression. Documented; revisit if the state-level Phase G eval shows >2 bu/acre divergence from NASS published numbers.

**Phase 2-B closed cleanly. Gate passed (post-recalibration on 2024 holdout). Ready to start Phase C (XGBoost point-estimate model).**
