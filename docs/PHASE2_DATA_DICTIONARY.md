# PHASE2_DATA_DICTIONARY.md

> Reference for every column in `phase2/data/training_master.parquet` (the output of `scripts/merge_all.py`). Source-of-truth for what each feature means, where it came from, and how it was constructed. Update whenever a feature is added, removed, or has its semantics changed.
>
> **Approximate shape:** 44 columns × ~27k–37k rows (5 states × 388 corn-bearing counties
> × 21 years × 4 forecast dates, minus rows where NASS suppressed the yield report).
> See [Coverage summary](#coverage-summary) for the precise breakdown.

**Last updated:** 2026-04-25

---

## Conventions

- **Keys** are the columns used to join across sources: `GEOID`, `year`, `forecast_date`. Every row in `training_master.parquet` is unique on `(GEOID, year, forecast_date)`.
- **Granularity** indicates the natural resolution of the source. Static features (gSSURGO) broadcast across years; state-level features (USDM, HLS) broadcast across counties within a state.
- **As-of safety:** all `(GEOID, year, forecast_date)`-keyed features were constructed using only data with timestamps strictly before `forecast_date` in the corresponding year. This is the non-negotiable temporal-leakage rule. Static and per-year features are not affected.
- **Forecast-date strings** are the project canonical: `"08-01"`, `"09-01"`, `"10-01"`, `"EOS"`. (HLS pull's internal labels `aug1`/`sep1`/`oct1`/`final` are normalized to the canonical form by `scripts/hls_features.py`.)
- **GEOID** is always 5-character zero-padded county FIPS code, stored as string.

---

## Keys

| Column | Type | Description |
|---|---|---|
| `GEOID` | str (5 chars) | County FIPS code, zero-padded. Joinable to TIGER 2018 county polygons. |
| `state_alpha` | str (2 chars) | USPS state code, one of `IA, CO, WI, MO, NE`. Source: NASS. |
| `county_name` | str | County name from NASS QuickStats. Sometimes spelled differently than TIGER (e.g. `"DE KALB"` vs `"DeKalb"`); not used for joining, only for display. |
| `year` | int | Calendar year. Range 2005–2025. 2025 rows have NaN target and are the forecast queries. |
| `forecast_date` | str | One of `"08-01"`, `"09-01"`, `"10-01"`, `"EOS"`. EOS is operationalized as 11-30. Each `(GEOID, year)` pair has 4 rows, one per forecast date. |

---

## Target

| Column | Type | Source | Description |
|---|---|---|---|
| `yield_target` | float | NASS QuickStats | Combined-practice corn yield, **bushels per acre**. The single target the model predicts. NaN for 2025 rows (the forecast queries) and for any (GEOID, year) where NASS suppressed the row. |

---

## NASS-derived features (per `(GEOID, year)`, broadcast across forecast_date)

Source: `scripts/nass_corn_5states_features.csv`, derived from `scripts/nass_corn_5states_2005_2024.csv` (raw NASS QuickStats pull).

| Column | Type | Description |
|---|---|---|
| `irrigated_share` | float | Irrigated acres harvested / combined-practice acres harvested. Coverage is sparse — only Colorado and Nebraska report irrigated NASS rows. NaN where unreported. Range [0, 1]. |
| `harvest_ratio` | float | Acres harvested / acres planted. Indicator of crop failure rate. Range [0, 1] in normal years; rare values <0.5 indicate severe loss. |
| `acres_harvested_all` | float | Combined-practice acres harvested for corn-grain. Used to area-weight if we ever roll up to state. |
| `acres_planted_all` | float | Combined-practice acres planted for corn-grain. |
| `yield_bu_acre_irr` | float | Yield from irrigated rows only. Sparse — same coverage as `irrigated_share`. NaN otherwise. |

**Excluded from feature set:** `yield_bu_acre_all` (= the target), `production_bu_*` (redundant with yield × acres), `yield_bu_acre_noirr` (sparse and computable from combined − irrigated).

---

## MODIS NDVI features (per `(GEOID, year)`, broadcast across forecast_date)

Source: 21 yearly CSVs `phase2/data/ndvi/corn_ndvi_5states_<year>.csv` (2004–2024), exported server-side via Earth Engine using `scripts/ndvi_county_extraction.js`. CDL-masked to corn pixels (`cropland == 1`). MODIS MOD13Q1 product (250m, 16-day composites).

> **NDVI is pre-scaled (× 0.0001 applied server-side).** Values are floats in `[-0.2, 1.0]`. Do not multiply by 0.0001 downstream.

| Column | Type | Description |
|---|---|---|
| `ndvi_peak` | float | Maximum NDVI observed in the growing season (DOY 121–273) for corn pixels in the county. |
| `ndvi_gs_mean` | float | Mean NDVI over the full growing season (DOY 121–273). |
| `ndvi_gs_integral` | float | Integrated NDVI (sum × 16-day spacing). Proxy for cumulative biomass. |
| `ndvi_silking_mean` | float | Mean NDVI over the silking window (DOY 196–227). Most yield-correlated subseason. |
| `ndvi_veg_mean` | float | Mean NDVI over the vegetative window (DOY 152–195). |

**Coverage caveats:**
- Counties with no CDL corn pixels in a year emit NaN NDVI (rare in IA/NE, more common in mountain CO).
- 2004 backfill: NDVI was extended one year earlier than NASS to give a one-year warmup for trend features. 2004 rows are *not* included in the master table (range is 2005–2025).
- **As-of caveat:** these are *seasonal* NDVI summaries computed over the full growing season — they include composites collected after the forecast date. For Phases B/C they are safe to use for the EOS forecast date; for Aug 1 / Sep 1 / Oct 1 they technically violate the as-of rule. Phase B may construct cutoff-respecting trailing means from a per-date NDVI table if needed. For now, treat NDVI as smoothed seasonal context, not a strict as-of feature.
- Pre-2008 CO and MO data is patchy (~6% null overall) due to CDL coverage rolling out state-by-state.

---

## HLS NDVI/EVI features (per `(state, year, forecast_date)`, broadcast across GEOIDs)

Source: `phase2/data/hls/hls_vi_features.csv` (output of `scripts/hls_pull.py`), reshaped by `scripts/hls_features.py` to `scripts/hls_county_features.csv`.

State-level state-of-vegetation summary derived from NASA HLS (Harmonized Landsat-Sentinel) v2.0 imagery. State bounding boxes; up to 5 cloud-masked granules per (state, year, forecast_date) window; NDVI/EVI computed per granule and median-aggregated.

> **HLS coverage starts 2013.** All HLS columns are NaN for 2005–2012 rows by design — Landsat-only era revisit cadence is too sparse for reliable in-season features. 2013–2014 are Landsat-only (lower revisit cadence) — sparser than 2015+.

| Column | Type | Description |
|---|---|---|
| `hls_ndvi_mean` | float | Median across granules of within-granule mean NDVI (state bbox, cloud-masked via Fmask). |
| `hls_ndvi_std` | float | Median across granules of within-granule NDVI std. Indicator of within-state heterogeneity. |
| `hls_evi_mean` | float | Median across granules of within-granule mean EVI. |
| `hls_evi_std` | float | Median across granules of within-granule EVI std. |
| `hls_n_granules` | Int64 (nullable) | Number of granules that contributed. Higher = more confident; ≤2 = sparse coverage warning. |

**Caveats:**
- State-level only — every county in a state gets the same HLS values for a given `(year, forecast_date)`. Within-state heterogeneity is not captured. (Phase D.1 Prithvi feature extraction will address this at the chip/county level.)
- As-of safe by construction: each forecast date's window is the ~30 days preceding the forecast date.
- Pre-2013 NaN: fundamental data gap, not a feature failure.
- When `hls_vi_features.csv` is absent (current state as of 2026-04-25), HLS columns are not emitted by `merge_all.py`. Downstream code must handle their absence gracefully.

---

## gSSURGO soil features (per `GEOID`, broadcast across `(year, forecast_date)`)

Source: `scripts/gssurgo_county_features.csv`, derived from gSSURGO state .gdb files (10m resolution) by `scripts/gssurgo_county_features.py`. Zonal stats over TIGER 2018 county polygons (reprojected to EPSG:5070 Albers before zonal stats — gSSURGO native CRS is Albers). Source table: gSSURGO Valu1 (depth-weighted summary table USDA recommends for cross-state aggregation).

> Static across years. Soil properties do not change at the project's time scale.

| Column | Type | Description |
|---|---|---|
| `nccpi3corn` | float | National Commodity Crop Productivity Index, corn variant. Range [0, 1], higher = better corn productivity potential. The single most relevant gSSURGO column for corn yield. |
| `nccpi3all` | float | NCCPI all-crops variant. Slightly broader productivity index. Range [0, 1]. |
| `aws0_100` | float | Available water storage 0–100 cm, mm. How much plant-available water the top 1m of soil can hold. |
| `aws0_150` | float | Available water storage 0–150 cm, mm. Deeper version. |
| `soc0_30` | float | Soil organic carbon 0–30 cm, g/m². Topsoil organic content. |
| `soc0_100` | float | Soil organic carbon 0–100 cm, g/m². Deeper version. |
| `rootznemc` | float | Rooting zone depth — effective max, cm. How deep roots can reach. |
| `rootznaws` | float | Rooting zone available water storage, mm. Effectively `aws` clipped to root depth. |
| `droughty` | float | Fraction of map unit area classified as droughty soils. Range [0, 1]. Higher = more drought-prone. |
| `pctearthmc` | float | Percent earthy mapunit components. Indirect proxy for fraction of county that is non-rock/non-water cropland-suitable. |
| `pwsl1pomu` | float | Potential wetland soils, level 1, percent of map unit. Higher = more wet/poorly-drained soils. |

**Notes:**
- 443 GEOIDs covered across the 5 states; not every county grows corn but soil features still exist for all.
- `Valu1` is USDA-published and well-documented; column-name glosses above paraphrase the official descriptions.

---

## Weather features (per `(GEOID, year, forecast_date)`)

Source: `scripts/weather_county_features.csv`, derived from `scripts/gridmet_county_daily_2005_2024.parquet` by `scripts/weather_features.py`. Daily county-aggregated gridMET (4km, CONUS, 2005–2024). All features respect the as-of rule by slicing the daily dataframe at `cutoff_date ≤ forecast_date − 1 day` before any aggregation.

**Phenology windows used (DOY ranges, inclusive):**
- Vegetative: 152–195 (Jun 1 – Jul 14)
- Silking: 196–227 (Jul 15 – Aug 15)
- Grain fill: 228–273 (Aug 16 – Sep 30)

Phase windows are clipped to the cutoff DOY in the year — e.g., on `08-01` the silking aggregate only covers DOY ≤ 213.

| Column | Type | Description |
|---|---|---|
| `gdd_cum_f50_c86` | float | Cumulative growing degree days, Fahrenheit base 50 / cap 86. Both `tmin` and `tmax` are clamped to `[50, 86] °F` before averaging (McMaster & Wilhelm / NDAWN convention). Sum from May 1 to cutoff. The standard corn GDD calculation. |
| `edd_hours_gt86f` | float | Excessive degree-hours above 86°F over May 1 → cutoff, computed by single-sine hourly interpolation between daily tmin and tmax (Allen 1976 / Baskerville-Emin 1969). Captures sub-daily heat stress that simple `max(0, tmax − 86)` misses. |
| `edd_hours_gt90f` | float | Same calculation, threshold 90°F. Severe heat stress indicator. |
| `vpd_kpa_veg` | float | Mean daily VPD (kPa) over the vegetative phase (DOY 152–195, clipped to cutoff). NaN if cutoff is before the phase starts. |
| `vpd_kpa_silk` | float | Mean daily VPD (kPa) over silking (DOY 196–227, clipped to cutoff). The most yield-critical VPD window. |
| `vpd_kpa_grain` | float | Mean daily VPD (kPa) over grain fill (DOY 228–273, clipped to cutoff). NaN for early-season forecast dates that don't reach this window. |
| `prcp_cum_mm` | float | Cumulative precipitation, May 1 → cutoff, mm. |
| `dry_spell_max_days` | int | Longest run of consecutive days with `<2 mm/day` in May 1 → cutoff. Captures drought distribution, not just total volume. |
| `srad_total_veg` | float | Sum daily shortwave radiation (MJ/m²) over vegetative phase, clipped to cutoff. Cumulative biomass-driving energy. |
| `srad_total_silk` | float | Sum daily shortwave radiation (MJ/m²) over silking, clipped to cutoff. |
| `srad_total_grain` | float | Sum daily shortwave radiation (MJ/m²) over grain fill, clipped to cutoff. NaN at `08-01`. |

**Coverage caveat:** gridMET pull is 2005–2024. Any 2004 skeleton rows (if retained) have NaN weather columns.

---

## Drought features (per `(state, year, forecast_date)`, broadcast across GEOIDs)

Source: `scripts/drought_county_features.csv`, derived from `phase2/data/drought/drought_USDM-Colorado,Iowa,Missouri,Nebraska,Wisconsin.csv` by `scripts/drought_features.py`. US Drought Monitor weekly data, **state-level** (not county-level — the raw CSV has no county FIPS). State values broadcast to every GEOID in that state.

**As-of rule:** the selected USDM reading for `(state, year, forecast_date)` is the most recent reading where `valid_end < forecast_date` (strictly before, never on or after). USDM week-validity windows can bracket a forecast date; strict `<` guarantees the reading was fully published before the cutoff.

> USDM percentages are cumulative by construction: `D0 ≥ D1 ≥ D2 ≥ D3 ≥ D4`.

| Column | Type | Description |
|---|---|---|
| `d0_pct` | float | Percentage of state in D0 (Abnormally Dry) or worse. Range [0, 100]. |
| `d1_pct` | float | Percentage of state in D1 (Moderate Drought) or worse. |
| `d2_pct` | float | Percentage of state in D2 (Severe Drought) or worse. |
| `d3_pct` | float | Percentage of state in D3 (Extreme Drought) or worse. |
| `d4_pct` | float | Percentage of state in D4 (Exceptional Drought). |
| `d2plus` | float | Convenience alias for `d2_pct` ("severe-or-worse"). Same value, descriptive name. Worth the duplicated column for readability of model code. |

**Excluded from feature set:** DSCI (= sum of all five columns, redundant); season-cum drought weeks; trailing-N-week means. Defer until Phase B/C show the model wants more drought signal.

---

## Coverage summary

Approximate row count in `training_master.parquet`:

```
443 GEOIDs × 21 years (2005–2025) × 4 forecast_dates = ~37,212 rows
```

Actual row count will be slightly less because some `(GEOID, year)` combinations are suppressed in NASS and dropped.

Feature group coverage (rough, post-merge):

| Feature group | Coverage |
|---|---|
| `yield_target` | ~95% (sparse for some CO mountain counties; NaN for 2025 forecast queries) |
| NASS features (combined-practice) | ~95–100% |
| NASS features (irrigated) | ~7% (CO + NE only) |
| NDVI | ~95–100% for major corn states; sparser for low-corn counties; ~6% null in CO/MO pre-2008 |
| gSSURGO | 100% (all 443 counties) |
| Weather | 100% (gridMET covers all CONUS counties) |
| Drought | 100% (state-broadcast) |
| HLS | 0% for 2005–2012; ~60–95% for 2013–2017; increasing 2018+ |

---

## Train / val / holdout split

Locked in `PHASE2_DECISIONS_LOG.md`. `merge_all.py` does **not** filter rows — all years are present in the master table; splits are enforced in Phase B/C training scripts.

| Split | Years |
|---|---|
| Train | 2005–2022 (18 years) |
| Val | 2023 |
| Holdout | 2024 |
| Forecast queries | 2025 (NaN target) |

---

## Adding a new feature

When adding a new column to `training_master.parquet`:

1. Update the relevant `*_features.py` to emit the new column.
2. Update `merge_all.py` if the join keys change.
3. **Add an entry here** with: column name, type, source script, granularity, and a one-sentence description of what it represents and how it was derived.
4. Note any caveats (sparsity, as-of-rule status, coverage gaps).
5. Log the addition in `PHASE2_DECISIONS_LOG.md` if the rationale is non-obvious.

---

## See also

- `docs/PHASE2_DATA_INVENTORY.md` — what's on disk and what's missing
- `docs/PHASE2_DECISIONS_LOG.md` — *why* each feature is what it is
- `docs/PHASE2_PHASE_PLAN.md` — Phase A.6 spec for `merge_all.py`
- `docs/PHASE2_CURRENT_STATE.md` — current pipeline state