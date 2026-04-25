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
