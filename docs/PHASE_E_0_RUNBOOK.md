# Phase E.0 runbook — extending the master parquet to 2025

The hackathon brief deliverable #2 is "yield forecast outputs for each of
the chosen states at all four time points **for the 2025 season**." E.0
extends the existing 2005-2024 master parquet to include 2025 query rows
so the existing Phase B/C inference machinery can produce 2025 forecasts.

## What changed since the first version of this runbook

A NASS API probe (April 25, 2026) confirmed NASS publishes **no
county-level 2025 corn data** for any of our 5 states. The state-level
data NASS does publish (weekly CONDITION/PROGRESS ratings, prices) is not
in our feature set. Therefore the original `nass_pull_2025.py` is
abandoned. We synthesize 2025 NASS-aux from 2024 verbatim.

This is defensible: the regressor's NASS-aux features
(`acres_planted_all`, `irrigated_share`, `harvest_ratio`) are structural
management priors — "this is how this county typically operates." Last
year's value is a fine prior. The regressor's predictive signal for 2025
comes from the IN-SEASON inputs (weather, drought, soil), all of which
ARE real for 2025. A `nass_aux_provenance` column on every row makes the
provenance auditable.

## What this runbook ships

| File | Status |
|---|---|
| `scripts/nass_features_v2.py` | NEW — synthesizes 2025 NASS-aux from 2024, marks provenance |
| `scripts/smoke_forecast_2025.py` | NEW — validates the entire forecast pipeline against 2025 data |
| `scripts/nass_pull.py` | UNCHANGED — Phase A.1 baseline, untouched |
| `scripts/nass_features.py` | UNCHANGED — feeds the v2 synthesizer with its 2005-2024 input |
| `scripts/merge_all.py` | UNCHANGED — already accepts `--nass` CLI flag |
| `scripts/gridmet_pull.py` | minor patch (one-line filename update) |
| `scripts/weather_features.py` | minor patch (one-line filename update) |
| `scripts/drought_features.py` | UNCHANGED — already covers 2025 from the on-disk USDM CSV |
| `forecast/*` modules | UNCHANGED |
| ~~`scripts/nass_pull_2025.py`~~ | **REMOVED** (abandoned approach; NASS has no 2025 county data) |

## Run order

From repo root.

### Step 1 — Synthesize 2025 NASS-aux from 2024

```
python scripts/nass_features_v2.py
```

Wall: <1s.

Writes `scripts/nass_corn_5states_features_v2.csv`. Reads the canonical
v1 file (`scripts/nass_corn_5states_features.csv`) and appends one
forecast row per county for 2025, copying 2024's NASS-aux verbatim with
`nass_aux_provenance="prior_year"` and `yield_target=NaN`.

Verify:
```
python -c "
import pandas as pd
df = pd.read_csv('scripts/nass_corn_5states_features_v2.csv')
print('total:', len(df))
print(df.year.value_counts().sort_index().tail(5).to_string())
print('2025 yield_target nulls:', df.loc[df.year==2025, 'yield_target'].isna().sum())
print('2025 provenance:', df.loc[df.year==2025, 'nass_aux_provenance'].value_counts().to_dict())
"
```

You want: 2025 yield_target 100% null, all 2025 provenance = `prior_year`,
and 2025 row count matches 2024's row count.

### Step 2 — Patch and pull gridMET 2025

Two files have hard-coded `2005_2024` paths. One-line patches:

```
sed -i 's|gridmet_county_daily_2005_2024.parquet|gridmet_county_daily_2005_2025.parquet|' scripts/gridmet_pull.py
sed -i 's|gridmet_county_daily_2005_2024.parquet|gridmet_county_daily_2005_2025.parquet|' scripts/weather_features.py
```

Verify:
```
grep "2005_2025" scripts/gridmet_pull.py scripts/weather_features.py
```

(Should match in both files.)

Then pull 2025 + combine:
```
python scripts/gridmet_pull.py --years 2025
python scripts/gridmet_pull.py --combine
```

Wall: ~3-8 min depending on network. Pulls ~5 NetCDFs (one per gridMET
variable) for 2025, runs zonal stats over 388 counties, writes a per-year
parquet plus the combined 2005-2025 parquet.

Verify:
```
ls -la data/v2/weather/raw/gridmet_county_daily_2025.parquet
ls -la scripts/gridmet_county_daily_2005_2025.parquet
```

Both should exist.

### Step 3 — Re-derive weather features over 2005-2025

```
python scripts/weather_features.py
```

Wall: ~30s. Overwrites `scripts/weather_county_features.csv` to include
2025 (the script auto-iterates over whatever years are in the gridMET
parquet).

Verify:
```
python -c "
import pandas as pd
w = pd.read_csv('scripts/weather_county_features.csv')
print('rows:', len(w), '  years:', sorted(w.year.unique())[-5:])
print('2025 by forecast_date:', w[w.year==2025].groupby('forecast_date').size().to_dict())
"
```

You want: ~1,552 rows for 2025 (388 GEOIDs × 4 forecast_dates), or close
to it depending on gridMET's data-availability cutoff.

### Step 4 — Re-derive drought features

```
python scripts/drought_features.py
```

Wall: ~5s. The on-disk USDM CSV already covers 2025 through October. Just
re-run; the script auto-picks up.

Verify:
```
python -c "
import pandas as pd
d = pd.read_csv('scripts/drought_county_features.csv')
print('rows:', len(d), '  years:', sorted(d.year.unique())[-5:])
print('2025 by forecast_date:', d[d.year==2025].groupby('forecast_date').size().to_dict())
"
```

### Step 5 — Re-merge to v2 master parquet

```
python scripts/merge_all.py \
    --nass scripts/nass_corn_5states_features_v2.csv \
    --out  scripts/training_master_v2.parquet
```

Wall: ~10s.

Verify:
```
python -c "
import pandas as pd
m = pd.read_parquet('scripts/training_master_v2.parquet')
print('shape:', m.shape, '  years:', sorted(m.year.unique())[-5:])
sub = m[m.year==2025]
print('2025 rows:', len(sub))
print('2025 by (state, forecast_date):')
print(sub.groupby(['state_alpha','forecast_date']).size().unstack(fill_value=0))
print()
print('2025 NaN counts in regressor-critical cols:')
for c in ['acres_planted_all','irrigated_share','harvest_ratio',
          'gdd_cum_f50_c86','vpd_kpa_silk','prcp_cum_mm','d2plus','nccpi3corn']:
    if c in sub.columns:
        print(f'  {c:25s} {sub[c].isna().sum():>5}/{len(sub)}')
"
```

What you want for 2025:
- ~1,552 rows total (388 GEOIDs × 4 forecast_dates)
- All 5 states populated
- `acres_planted_all`, `irrigated_share`, `harvest_ratio` — 0 nulls (carried from 2024)
- All weather and drought columns — 0 nulls (or close to it; structural NaN at 08-01 for grain-window features is expected)
- All gSSURGO columns — 0 nulls (static)
- NDVI columns — many nulls expected (no 2025 GEE NDVI pulled), and that's fine

### Step 6 — Smoke test

```
python scripts/smoke_forecast_2025.py
```

Wall: ~5-10s.

Twenty rows of output (5 states × 4 forecast_dates), one per (state, date).
Per-row outcomes:

- **`ok`** — point inside cone. Won't see this for 2025 because NDVI 2025
  is missing → the cone has no contributing counties.
- **`<-- WARN`** — point estimate fine, cone empty. **Expected for 2025.**
  The state forecasts are still real; the cone story for 2025 specifically
  lands later (when GEE NDVI 2025 is pulled).
- **`<-- FAIL`** — regressor returned NaN for every county. Indicates a
  data-layer problem; investigate.

Send me the output of step 6. If it's all `<-- WARN` or `ok`, E.1 (the
backend route) is unblocked.

## What can fail and how to recover

- **Step 1 fails with "historical features file not found"** — run
  `python scripts/nass_features.py` first to produce the v1 baseline.

- **Step 5 fails on a `merge_all.py` assertion** about state_alpha or
  forecast_date — check that the v2 NASS file has the same column names
  and dtypes as v1.

- **Step 6 prints `<-- FAIL`** on every (state, date) — most likely the
  master parquet doesn't have 2025 rows. Re-check step 1 verifier and
  step 5 verifier. The pipeline has 5 places where 2025 rows can get
  silently dropped, each verifier above narrows it down.

- **gridMET 2025 has only partial coverage** (e.g. data only through
  early April since today is April 25, 2026 in our setup) — that's fine
  for the 08-01 forecast date, which only needs data through Aug 1 of
  the previous year... wait no, 08-01 of 2025 means data through Aug 1,
  2025, which gridMET has. All four 2025 forecast dates are in the past,
  so gridMET should have full coverage for them. Check `weather_features.py`
  output to confirm.

## After this passes, I'll write E.1

E.1 = `backend/forecast_routes.py` + the lifespan additions in
`backend/main.py`. The smoke test exercises every function the route
will call, in the same order, so once it's green the route becomes a
thin HTTP wrapper around already-validated math.
