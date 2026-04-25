# PHASE2 Data Dictionary — `training_master.parquet`

> Reference doc for every column in `scripts/training_master.parquet`, the canonical input to Phase B (analog retrieval), Phase C (XGBoost point-estimate model), Phase D (Prithvi integration), and Phase F (forecast narration agent).
>
> Generated at end of Phase A.7. The output of `scripts/merge_all.py`. **When this dictionary disagrees with the source `*_features.py` script, the source script wins** — keep this doc in sync when feature definitions change.

---

## Quick facts

| Property | Value |
|---|---|
| File | `scripts/training_master.parquet` |
| Format | Parquet (snappy compression) |
| Size on disk | ~2.46 MB |
| Shape | 25,872 rows × 48 columns |
| Grain (one row per…) | `(GEOID, year, forecast_date)` |
| Years | 2005–2024 (20 years; 2005–2022 train / 2023 val / 2024 holdout) |
| States | CO, IA, MO, NE, WI |
| GEOIDs (5-digit FIPS) | 388 distinct (subset of 443 TIGER 2018 counties — 55 are corn-absent and have no NASS yield) |
| Forecast dates | `08-01`, `09-01`, `10-01`, `EOS` (= Nov 30) |
| Target | `yield_target` (combined-practice corn-grain yield, bu/acre) |

**Read it back:**
```python
import pandas as pd
df = pd.read_parquet("scripts/training_master.parquet")
df["GEOID"] = df["GEOID"].astype(str).str.zfill(5)  # parquet preserves str, but defensive
```

---

## Column overview by source

| Source | Cols | Grain at source | How it broadcasts | Script |
|---|---|---|---|---|
| **Keys** | 5 | n/a | n/a | `merge_all.py` (skeleton) |
| **Target + NASS** | 6 | `(GEOID, year)` | broadcast across 4 forecast_dates | `nass_features.py` |
| **NDVI (MODIS)** | 5 | `(GEOID, year)` | broadcast across 4 forecast_dates | `ndvi_county_extraction.js` (GEE) |
| **Soil (gSSURGO)** | 11 | `(GEOID,)` static | broadcast across years AND forecast_dates | `gssurgo_county_features.py` |
| **Weather (gridMET)** | 11 | `(GEOID, year, forecast_date)` | direct merge | `weather_features.py` |
| **Drought (USDM)** | 6 | `(state, year, forecast_date)` | broadcast to all GEOIDs in state | `drought_features.py` |
| **HLS** | 4 | `(state, year, forecast_date)` | broadcast to all GEOIDs in state | (HLS pull pipeline; redone in Phase D.1) |
| **TOTAL** | **48** |  |  |  |

---

## Keys (5 columns)

The composite primary key is `(GEOID, year, forecast_date)`. Asserted unique by `merge_all.py`.

### `GEOID` — string, length 5
5-digit county FIPS (state FIPS + county FIPS, both zero-padded). Example: `"19153"` = Polk County, IA. **Always 5 characters; always strings, not ints.** Asserted by `merge_all.py`.

### `state_alpha` — string, length 2
USPS state code: one of `{"CO", "IA", "MO", "NE", "WI"}`. Sourced from NASS (authoritative). Drives drought and HLS broadcasts.

### `county_name` — string
Uppercased county name from NASS (e.g. `"POLK"`, `"ADAMS"`). Useful for human-readable output (agent narration, log messages). **Do not use as a join key** — county names collide across states (e.g. `ADAMS` exists in CO, IA, NE, WI). Always join on `GEOID`.

### `year` — int64
Calendar year, 2005–2024 inclusive. The crop year and the calendar year are the same for corn in this region.

### `forecast_date` — string
One of `{"08-01", "09-01", "10-01", "EOS"}`. The "as-of" date for in-season features. Data with timestamp `<` forecast_date contributes to the row's features (strict before; same-day data is excluded). `EOS` = November 30 = end of growing season; effectively a post-harvest snapshot.

**Sort order:** the script writes rows ordered chronologically (`08-01 < 09-01 < 10-01 < EOS`) within each `(GEOID, year)` group. If you re-sort, use a `pd.Categorical` with `categories=["08-01","09-01","10-01","EOS"]` to preserve this; lexicographic sorting happens to also work but is fragile.

---

## Target (1 column)

### `yield_target` — float64, units: **bu/acre**
Combined-practice (irrigated + non-irrigated) corn-for-grain yield. From NASS QuickStats 2005–2024 county-level annual surveys. **0 nulls in the full table** (rows with no NASS yield were filtered out in `nass_features.py` via `.dropna(subset=["yield_target"])`).

**Range observed:** roughly 50–250 bu/acre, with state means ~140–195 in IA, lower and more variable in CO and WI. Drought years (2012, 2024 in some areas) produce large negative excursions.

**For training:** this is the ground-truth label. Held-out by year (2023 val, 2024 holdout). For forecast inference at a current year, this column is unknown and the row exists only as a feature carrier — but the master table itself does not contain rows without `yield_target` (filtered upstream).

---

## NASS features (5 cols beyond target)

All sourced from NASS QuickStats. Per-`(GEOID, year)` and broadcast across all 4 forecast_dates within a year. **Same value at 08-01 / 09-01 / 10-01 / EOS** — these are annual statistics, not in-season measurements. Source: `scripts/nass_features.py`.

> **As-of caveat:** NASS yield/production are reported *post-harvest*. Strictly speaking, including these annual values at the 08-01 row violates the as-of rule for the current target year. **`yield_target` is the label, not a feature, so this is not leakage** — it's only "leakage" if used as a feature. The other NASS columns (`acres_*`, `irrigated_share`, `harvest_ratio`) are partly post-hoc (final harvested acres are not knowable on Aug 1) and should be treated as **structural/management priors** in modeling — i.e. lagged or replaced with prior-year proxies if Phase C wants stricter as-of fidelity. Current decision: ship as-is, document here, revisit if Phase B/C show issues.

### `irrigated_share` — float64, range [0, 1]
`acres_harvested_irr / acres_harvested_all`, with NaN-as-rainfed convention (`.fillna(0)`) and clip to [0, 1]. **0 nulls.** Only CO and NE report `acres_harvested_irr` separately; for IA / MO / WI the input is NaN and `irrigated_share` becomes 0. State means: CO ≈ 0.7, NE ≈ 0.6, others ~0.

### `harvest_ratio` — float64, range [0, 1]
`acres_harvested_all / acres_planted_all`, clipped to [0, 1]. Proxy for in-season abandonment (drought, hail, late freeze). Typical values 0.95–1.00 in normal years; drops in drought years. **8 nulls** (4 (GEOID, year) tuples × 2-of-4 forecast_dates? actually × 4 = 8 means 2 (GEOID, year) tuples have NaN — likely tiny counties where NASS suppressed `acres_planted_all` for disclosure reasons).

### `acres_harvested_all` — float64, units: acres
Combined-practice harvested corn-grain acreage. **8 nulls** (same rows as `harvest_ratio`). Range: hundreds (small CO/WI counties) to ~250,000 (large IA counties).

### `acres_planted_all` — float64, units: acres
Combined-practice planted corn-grain acreage. **0 nulls.** Always ≥ `acres_harvested_all`.

### `yield_bu_acre_irr` — float64, units: bu/acre
Irrigated-only yield. Reported only for CO and NE counties with material irrigation. **21,660 nulls (83.7%)** — this is structural sparseness, not a bug. Use as a feature only with explicit imputation strategy or as an indicator with `is_irrigated_reported = yield_bu_acre_irr.notna()`.

---

## NDVI features (5 cols)

MODIS MOD13Q1 (16-day, 250 m) NDVI, masked to corn pixels via USDA CDL, county-aggregated by mean reducer. Per-`(GEOID, year)` and broadcast across all 4 forecast_dates. Source: `scripts/ndvi_county_extraction.js` (Earth Engine).

> **CRITICAL: NDVI values are pre-scaled in the CSVs.** The GEE script applies `× 0.0001` server-side, so columns are floats in roughly `[-0.2, 1.0]`. **Do not re-apply the scale factor downstream.**

> **As-of caveat:** NDVI columns are **whole-season summaries** (e.g. `ndvi_gs_integral` integrates over DOY 121–273 = May 1 → Sep 30 of the target year). They are *not* clipped to the forecast_date — at `08-01` you still see end-of-September NDVI. **This violates strict as-of for the 08-01 and 09-01 forecast dates.** The locked decision (see `PHASE2_DECISIONS_LOG.md`) is to ship MODIS NDVI as a *trend feature* — its value at the current year is treated as observable when forecasting, with the rationale that MODIS NDVI is updated near-real-time and end-of-season summary at forecast time is a reasonable approximation of "the NDVI that will have been observed by harvest." For strict as-of, replace with HLS-derived running-NDVI in Phase D.1.

> **Coverage:** 492 nulls per NDVI column (1.9%). These are CO 2005–2007 counties where USDA CDL hadn't fully rolled out yet, so corn-masking failed. Treat as missing-not-at-random; documented in `PHASE2_CURRENT_STATE.md`.

### `ndvi_peak` — float64
Maximum NDVI observed during the growing season (DOY 121–273). Year-level proxy for canopy vigor at biomass peak.

### `ndvi_gs_mean` — float64
Mean NDVI across the growing season.

### `ndvi_gs_integral` — float64
Sum of NDVI values across the growing season. Proxy for cumulative canopy vigor; correlates with biomass.

### `ndvi_silking_mean` — float64
Mean NDVI during silking (DOY 196–227 ≈ Jul 15–Aug 15). Phenologically the most yield-relevant window for corn (kernel set is decided here).

### `ndvi_veg_mean` — float64
Mean NDVI during vegetative phase (DOY 152–195 ≈ Jun 1–Jul 14). Pre-silking canopy development.

---

## Soil features (11 cols, gSSURGO)

USDA gSSURGO Valu1 table, area-weighted county aggregates. **Static across years** — soil doesn't change at year-decade timescales. Source: `scripts/gssurgo_county_features.py`. **0 nulls on every column.** All values are area-weighted means computed via per-county windowed reads of the state MUKEY raster (CRS: EPSG:5070 Albers Equal Area).

### `nccpi3corn` — float64, range [0, 1]
National Commodity Crop Productivity Index for corn. Higher = better corn productivity inherent to the soil. Strong yield prior.

### `nccpi3all` — float64, range [0, 1]
NCCPI averaged across all crops. Use alongside `nccpi3corn` if the model wants a corn-specific signal vs. general productivity.

### `aws0_100` — float64, units: mm (or cm-equivalent)
Available water storage 0–100 cm depth.

### `aws0_150` — float64, units: mm
Available water storage 0–150 cm depth. Deeper-rooted crops draw on more of this.

### `soc0_30` — float64, units: g/m² or t/ha (per gSSURGO Valu1 convention; see source)
Soil organic carbon 0–30 cm.

### `soc0_100` — float64
Soil organic carbon 0–100 cm.

### `rootznemc` — float64
Root zone effective moisture capacity. Combines depth and AWS over the rooting zone.

### `rootznaws` — float64
Root zone available water storage.

### `droughty` — float64, [0, 100]
Percent of county area flagged as drought-vulnerable per gSSURGO classification. Higher = soil more prone to dry-out.

### `pctearthmc` — float64, [0, 100]
Percent of county classified as "earth" surface (excludes water bodies, urban, bedrock). Useful for normalizing acreage features and for understanding what fraction of the county can grow anything.

### `pwsl1pomu` — float64, [0, 100]
Potential wetland soils, percent of county. Negatively correlated with arable area in some regions.

---

## Weather features (11 cols, gridMET)

gridMET daily 4 km weather, county-aggregated, with phase-window and as-of-clipped derivations. Per-`(GEOID, year, forecast_date)`; **direct merge, not broadcast.** Source: `scripts/weather_features.py`. Underlying daily data: `scripts/gridmet_county_daily_2005_2024.parquet` (from `gridmet_pull.py`).

**As-of safety:** the function `build_features_for_cutoff(df, year, cutoff_date)` slices the daily DataFrame at the very top with `date <= cutoff_date`. This is the **single point of leakage control** — every feature below is derived only from that pre-sliced data. Phase windows are additionally clipped to `cutoff_doy` (e.g. on 08-01 the silking window only includes DOYs 196–213).

**Phase windows (DOY, inclusive):**
- Vegetative: 152–195 (Jun 1 – Jul 14)
- Silking: 196–227 (Jul 15 – Aug 15)
- Grain fill: 228–273 (Aug 16 – Sep 30)
- May 1 = DOY 121 (start of cumulative-from-planting features)

**Cumulative-from-May-1 features** (`gdd_cum_*`, `edd_*`, `prcp_cum_mm`, `dry_spell_max_days`) are season-cumulative through the cutoff. These don't structurally NaN at any forecast date because by 08-01 they have at least 92 days of accumulation.

**Phase-aggregate features** (`vpd_kpa_*`, `srad_total_*`) are means/sums over the phase window clipped to cutoff. **Structural NaN pattern:** if the cutoff falls before the phase starts, the slice is empty and the feature is NaN. See "NaN patterns" section below.

### `gdd_cum_f50_c86` — float64, units: °F-days
Cumulative growing degree days, base 50°F / cap 86°F (industry standard for corn; McMaster & Wilhelm, NDAWN). **Both `tmin` and `tmax` are clamped** to [50, 86]°F before averaging — not just `tavg − 50`. Cumulative from DOY 121 → cutoff. **0 nulls.** Range: ~1500 (08-01, cooler counties) to ~4000 (EOS, hotter counties).

### `edd_hours_gt86f` — float64, units: degree-hours
Heat-stress: degree-hours above 86°F, summed over DOY 121 → cutoff. Computed via single-sine hourly interpolation (Allen 1976 / Baskerville-Emin 1969 closed-form integral). **0 nulls.**

### `edd_hours_gt90f` — float64, units: degree-hours
Severe heat-stress: degree-hours above 90°F, same method as above. **0 nulls.** Will be ≤ `edd_hours_gt86f` by construction.

### `vpd_kpa_veg` — float64, units: kPa
Mean daily vapor pressure deficit during vegetative phase (DOY 152–195), clipped to cutoff. **0 nulls** at all 4 forecast dates (vegetative window ends Jul 14, well before 08-01).

### `vpd_kpa_silk` — float64, units: kPa
Mean daily VPD during silking (DOY 196–227), clipped to cutoff. **0 nulls** at all 4 forecast dates: at 08-01 cutoff the slice is DOYs 196–213 (Jul 15–Aug 1), still ~17 days of silking data. **Critical yield-determining feature** — silking VPD strongly predicts pollination success.

### `vpd_kpa_grain` — float64, units: kPa
Mean daily VPD during grain fill (DOY 228–273). **6,468 nulls (25.0%) — STRUCTURAL.** At the 08-01 cutoff, grain fill (which starts DOY 228 = Aug 16) hasn't begun. Every 08-01 row has NaN here by construction. See "NaN patterns" section.

### `prcp_cum_mm` — float64, units: mm
Cumulative precipitation, May 1 → cutoff. **0 nulls.**

### `dry_spell_max_days` — int64, units: days
Longest run of consecutive days with `< 2 mm/day` precipitation, May 1 → cutoff. **0 nulls.** Note: `int64` dtype (no NaN); if a future year produces NaN, pandas will widen to float64 — pin this if it matters.

### `srad_total_veg` — float64, units: MJ/m²
Total solar radiation during vegetative phase, clipped to cutoff. **0 nulls.**

### `srad_total_silk` — float64, units: MJ/m²
Total solar radiation during silking, clipped to cutoff. **0 nulls.**

### `srad_total_grain` — float64, units: MJ/m²
Total solar radiation during grain fill. **6,468 nulls (25.0%) — STRUCTURAL**, same pattern as `vpd_kpa_grain`.

---

## Drought features (6 cols, USDM)

US Drought Monitor weekly state-level percentages, broadcast to GEOIDs within state. Per-`(GEOID, year, forecast_date)`. Source: `scripts/drought_features.py`. **0 nulls on every column** (USDM coverage extends back to 2000; full coverage for the 2005–2024 modeling window).

**Cumulative convention:** USDM percentages are cumulative — `D0 ≥ D1 ≥ D2 ≥ D3 ≥ D4` by construction. Each column reports % of state at that severity *or worse*. The asserts in `drought_features.py` confirmed 0 monotonicity violations across all 420 (state × year × forecast_date) state-level rows.

**As-of rule:** for each `(state, year, forecast_date)`, the most recent reading whose `valid_end < forecast_date` (strict) is selected. USDM week-validity windows can bracket the forecast date (a Tuesday-released map covers Mon–Sun); the strict-`<` semantics guarantee the reading was fully published before the forecast.

**Spatial broadcast:** state readings flow to all GEOIDs in that state. **Within-state spatial heterogeneity in drought is not captured.** A western Iowa drought that doesn't reach eastern Iowa shows up as the same `d2_pct` value in both. Acceptable for v2 baseline; richer spatial drought could come from PRISM SPI/SPEI or satellite drought indices in a later phase.

### `d0_pct` — float64, range [0, 100]
Percent of state in drought category D0 ("Abnormally Dry") or worse. **0 nulls.**

### `d1_pct` — float64, range [0, 100]
Percent of state in D1 ("Moderate Drought") or worse. ≤ `d0_pct`.

### `d2_pct` — float64, range [0, 100]
Percent of state in D2 ("Severe Drought") or worse. ≤ `d1_pct`.

### `d3_pct` — float64, range [0, 100]
Percent of state in D3 ("Extreme Drought") or worse. ≤ `d2_pct`.

### `d4_pct` — float64, range [0, 100]
Percent of state in D4 ("Exceptional Drought") or worse. ≤ `d3_pct`.

### `d2plus` — float64, range [0, 100]
**Stable alias for `d2_pct`.** Identical value (because USDM percentages are cumulative — "D2 or worse" is exactly D2). Exposed under a descriptive name for retrieval-embedding code that wants to reference "severe drought share" without remembering the cumulative convention. Drop one or the other if you want a smaller feature set; they carry the same signal.

---

## HLS features (4 cols)

Harmonized Landsat–Sentinel surface-reflectance vegetation indices, state-aggregated, derived per forecast date. Per-`(state, year, forecast_date)`, **broadcast to all GEOIDs in the state.** **High null rate (59.1%) — pre-2013 has no rows by design** (HLS Sentinel-2 component starts 2015; Landsat-only era is too sparse). The HLS pull will be redone in Phase D.1; treat current values as provisional.

> **Note on losing within-state heterogeneity:** like USDM, HLS is broadcast from state level, so all counties in a state share the same `ndvi_mean` value at a given (year, forecast_date). The MODIS-derived NDVI columns (`ndvi_peak`, `ndvi_gs_mean`, etc.) preserve county-level detail; HLS does not. Don't double-count.

> **Note on column name overlap with MODIS:** `ndvi_mean` (HLS) vs. `ndvi_gs_mean` (MODIS) are different things from different sensors at different scales. HLS columns have no `gs_` prefix; if a feature column starts with `ndvi_` and the suffix is `mean` or `std`, it's HLS. If the suffix includes `gs`, `peak`, `silking`, or `veg`, it's MODIS.

### `ndvi_mean` — float64
Mean NDVI from HLS surface reflectance, state-aggregated, as-of forecast_date. 15,297 nulls (59.1%, pre-2013).

### `ndvi_std` — float64
Standard deviation of NDVI across pixels in the state, as-of forecast_date. 15,297 nulls.

### `evi_mean` — float64
Mean Enhanced Vegetation Index from HLS. 15,297 nulls.

### `evi_std` — float64
Standard deviation of EVI. 15,297 nulls.

> **Dropped at load time:** `is_forecast` (object-dtype semantic flag from the pull pipeline) and `n_granules` (count column). These are **not** in `training_master.parquet`. See `merge_all.py` `HLS_FEATURE_COLS` allowlist if you need to re-include or rename.

---

## NaN patterns — what's intentional vs. what isn't

This section is **important for downstream modeling.** A naive median-imputation pipeline will mishandle several of these patterns and silently degrade model quality. The recommended downstream handling assumes Phase B (analog retrieval) and Phase C (XGBoost).

### Intentional structural NaN (treat as "not applicable", not "missing")

| Column | Null count | Pattern | Cause | Recommended handling |
|---|---|---|---|---|
| `vpd_kpa_grain` | 6,468 (25.0%) | Every `08-01` row | Grain fill (DOY 228+) hasn't started by Aug 1 | **Drop the column for `forecast_date == "08-01"` models, or use forecast_date × feature interactions in XGBoost (handles missing natively).** Do NOT median-impute. |
| `srad_total_grain` | 6,468 (25.0%) | Every `08-01` row | Same | Same |
| `yield_bu_acre_irr` | 21,660 (83.7%) | Most rows outside CO/NE | NASS only reports practice-split for irrigated states | Derive an `is_irrigated_reported` indicator and use that. Do NOT impute the value. |

### Sparse-coverage NaN (real signal of "data not available for this county/year")

| Column | Null count | Pattern | Cause | Recommended handling |
|---|---|---|---|---|
| `ndvi_*` (5 cols) | 492 (1.9%) each | CO 2005–2007 | USDA CDL didn't cover CO until 2008 | XGBoost handles natively. For analog retrieval, use a feature-presence-aware distance. |
| `harvest_ratio`, `acres_harvested_all` | 8 (~0%) | 2 (GEOID, year) tuples × 4 fd | NASS disclosure suppression for tiny counties | Drop those rows from training; they're 0.03% of data. |
| HLS `ndvi_*` / `evi_*` (4 cols) | 15,297 (59.1%) each | All pre-2013 rows | HLS doesn't exist pre-2013 | Use HLS only as ablation feature; don't include for years where it's structurally missing. Phase D.1 will redo this. |

### "Should never be NaN" — assert before training

| Column | Expected null count |
|---|---|
| `GEOID`, `state_alpha`, `county_name`, `year`, `forecast_date` | 0 (asserted by `merge_all.py`) |
| `yield_target` | 0 (filtered upstream in `nass_features.py`) |
| `irrigated_share`, `acres_planted_all` | 0 |
| All gSSURGO columns | 0 |
| Cumulative-from-May-1 weather (`gdd_cum_*`, `edd_*`, `prcp_cum_mm`, `dry_spell_max_days`) | 0 |
| Veg-window and silk-window weather (`vpd_kpa_veg`, `vpd_kpa_silk`, `srad_total_veg`, `srad_total_silk`) | 0 |
| All drought columns | 0 |

If any of these are non-zero in a future re-run, something upstream broke.

---

## As-of fidelity by source — reference table

How strictly each source respects "data with timestamp `<` forecast_date only":

| Source | As-of fidelity | Why |
|---|---|---|
| Weather (gridMET-derived) | **STRICT** | `build_features_for_cutoff` slices once at the top of each call; phase windows clipped to `cutoff_doy`. |
| Drought (USDM) | **STRICT** | Selects most recent USDM reading with `valid_end < forecast_date`. Bracket-spanning weeks are excluded. |
| Soil (gSSURGO) | **N/A** (static) | Soil doesn't change year to year. |
| NDVI (MODIS) | **WEAK — whole-season summary** | Same NDVI value at 08-01 / 09-01 / 10-01 / EOS within a year. Treated as observable trend feature; replace with running-NDVI in Phase D.1. |
| NASS aux (acres, irrigated_share, harvest_ratio) | **WEAK — annual reported post-harvest** | Same value across all 4 forecast_dates. Treat as structural priors, not as in-season measurements. Lagged variants (prior year's irrigated_share) would be strict. |
| HLS | **AS-OF by design** (per pipeline spec; redone in Phase D.1) | Each forecast_date has its own snapshot. Provisional until Phase D.1 redo. |

---

## Worked example — one row, fully unpacked

```
GEOID:           "19153"
state_alpha:     "IA"
county_name:     "POLK"
year:            2020
forecast_date:   "08-01"
yield_target:    194.5            # bu/acre, combined-practice corn-grain (2020 reported value)
irrigated_share: 0.0              # IA: rainfed
harvest_ratio:   0.99             # 99% of planted was harvested
acres_planted_all: 200000         # ~200k acres planted to corn
ndvi_peak:       0.84             # MODIS, whole-season max (DOY 121-273)
ndvi_silking_mean: 0.82           # MODIS, silking-window mean
nccpi3corn:      0.78             # gSSURGO, high productivity soil
aws0_100:        225              # mm, available water 0-100cm
gdd_cum_f50_c86: 1620             # °F-days, May 1 -> Jul 31
edd_hours_gt86f: 110              # mild heat stress so far
vpd_kpa_silk:    1.2              # mean VPD during silking-up-to-Aug-1 window
vpd_kpa_grain:   NaN              # STRUCTURAL — grain fill hasn't started by 08-01
prcp_cum_mm:     280              # cumulative rain May 1 -> Jul 31
dry_spell_max_days: 8             # longest dry run
srad_total_grain: NaN             # STRUCTURAL — same as vpd_kpa_grain
d0_pct:          12.5             # 12.5% of IA in D0+ on the most recent USDM reading before 2020-08-01
d2_pct:          0.0              # no severe drought
d2plus:          0.0              # alias of d2_pct
ndvi_mean:       0.65             # HLS state-level, as-of 08-01
```

---

## Versioning

This dictionary describes the **Phase A.7 build** of `training_master.parquet`. If `merge_all.py` is changed (new columns, new sources, schema fixes), this file must be updated in the same commit.

Next planned changes:
- **Phase D.1:** HLS pull will be redone with consistent schema and per-county (not per-state) granularity. The 4 HLS columns above will be replaced; some will be renamed.
- **Phase B/C:** if the model wants richer drought signal, `dsci`, `season_drought_weeks`, `silking_peak_d2plus` may be added (per the deferred items in `PHASE2_DECISIONS_LOG.md`).

When those land, append to this dictionary; do not delete prior entries.
