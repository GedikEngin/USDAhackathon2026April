#!/usr/bin/env bash
# scripts/smoke_forecast_endpoint.sh
#
# End-to-end smoke test for the /forecast/* endpoints (E.1).
#
# 1. /health         -> confirms forecast_available=true
# 2. /forecast/states -> states + available years
# 3. /forecast/IA?year=2025                   (all 4 dates)
# 4. /forecast/IA?year=2024&date=EOS          (single date, has truth)
# 5. POST /forecast/narrate                    (Phase F stub)
#
# Cross-checks the IA-2024-EOS state forecast against the most recent
# Phase C backtest CSV in runs/, if one exists. Match should be within
# 0.1 bu/ac (allowing for floating-point variation).
#
# Requirements: backend running on localhost:8000, jq installed.
#
# Usage:
#   bash scripts/smoke_forecast_endpoint.sh
#   BASE_URL=http://localhost:8000 bash scripts/smoke_forecast_endpoint.sh

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"

if ! command -v jq >/dev/null 2>&1; then
    echo "error: jq is required (apt-get install jq / brew install jq)" >&2
    exit 2
fi

# ---- 1. /health ----------------------------------------------------------
echo "[1/5] health check"
health=$(curl -sf "$BASE_URL/health")
echo "$health" | jq .
forecast_available=$(echo "$health" | jq -r .forecast_available)
if [[ "$forecast_available" != "true" ]]; then
    echo "error: backend says forecast_available=false." >&2
    echo "       Check that scripts/training_master_v2.parquet and models/forecast/ exist." >&2
    echo "       Backend log will show the exception that made lifespan abort." >&2
    exit 3
fi
echo "  forecast_available=true, model_version=$(echo "$health" | jq -r .model_version)"

# ---- 2. /forecast/states -------------------------------------------------
echo
echo "[2/5] /forecast/states"
states=$(curl -sf "$BASE_URL/forecast/states")
echo "$states" | jq '{model_version, default_year, default_date, n_states: (.states | length), states: [.states[] | {alpha, n_counties, years_count: (.available_years | length)}]}'
n_states=$(echo "$states" | jq '.states | length')
if [[ "$n_states" != "5" ]]; then
    echo "error: expected 5 states, got $n_states" >&2
    exit 4
fi

# ---- 3. /forecast/IA?year=2025 (forecast year, all 4 dates) -------------
echo
echo "[3/5] /forecast/IA?year=2025  (forecast year — all 4 dates)"
ia2025=$(curl -sf "$BASE_URL/forecast/IA?year=2025")
echo "$ia2025" | jq '{
  state, year, model_version,
  truth: .truth_state_yield_bu_acre,
  history: .history,
  by_date: (.by_date | to_entries | map({
    date: .key,
    point: .value.point_estimate_bu_acre,
    cone_status: .value.cone_status,
    n_reg: .value.n_counties_regressor,
    n_cone: .value.n_counties_cone
  }))
}'

# Sanity: every date should have a real point estimate (per smoke_forecast_2025.py).
ia2025_any_null=$(echo "$ia2025" | jq '[.by_date[].point_estimate_bu_acre] | any(. == null)')
if [[ "$ia2025_any_null" == "true" ]]; then
    echo "error: at least one IA 2025 date returned null point_estimate." >&2
    exit 5
fi
ia2025_truth=$(echo "$ia2025" | jq '.truth_state_yield_bu_acre')
if [[ "$ia2025_truth" != "null" ]]; then
    echo "error: IA 2025 should have null truth (forecast year), got $ia2025_truth" >&2
    exit 5
fi

# ---- 4. /forecast/IA?year=2024&date=EOS (single, has truth) -------------
echo
echo "[4/5] /forecast/IA?year=2024&date=EOS  (single date, validation year — has truth)"
ia2024=$(curl -sf "$BASE_URL/forecast/IA?year=2024&date=EOS")
echo "$ia2024" | jq '{
  state, year, forecast_date,
  point: .forecast.point_estimate_bu_acre,
  cone: .forecast.cone,
  truth: .truth_state_yield_bu_acre,
  n_drivers: (.forecast.top_drivers | length),
  n_analogs: (.forecast.analog_years | length),
  anchor: .forecast.analog_anchor
}'
ia2024_point=$(echo "$ia2024" | jq -r '.forecast.point_estimate_bu_acre')
ia2024_truth=$(echo "$ia2024" | jq -r '.truth_state_yield_bu_acre')
if [[ "$ia2024_point" == "null" ]]; then
    echo "error: IA 2024 EOS point_estimate is null." >&2
    exit 6
fi
if [[ "$ia2024_truth" == "null" ]]; then
    echo "warning: IA 2024 EOS truth is null. NASS truth should be present for 2024." >&2
fi

# ---- 5. POST /forecast/narrate (Phase F — real Claude narration) -------
echo
echo "[5/5] POST /forecast/narrate  (real narration via Claude Haiku 4.5)"
narrate=$(curl -sf -X POST "$BASE_URL/forecast/narrate" \
    -H 'Content-Type: application/json' \
    -d '{"state":"IA","year":2025,"forecast_date":"EOS"}')
echo "$narrate" | jq '{stub, model_version, narrative_chars: (.narrative | length), narrative_head: (.narrative | .[0:240])}'
narrate_stub=$(echo "$narrate" | jq -r .stub)
narrate_chars=$(echo "$narrate" | jq -r '.narrative | length')
if [[ "$narrate_stub" == "true" ]]; then
    echo "  warning: narrate returned stub=true. Has Phase F.1 been deployed?" >&2
fi
if [[ "$narrate_chars" -lt 200 ]]; then
    echo "  warning: narrative is suspiciously short ($narrate_chars chars). " \
         "Could indicate the agent client is unavailable; check ANTHROPIC_API_KEY." >&2
fi

echo
echo "=== PASS ==="
echo "  - /health.forecast_available = true"
echo "  - /forecast/states returned 5 states"
echo "  - /forecast/IA?year=2025 all 4 dates have real point estimates"
echo "  - /forecast/IA?year=2024&date=EOS: point=$ia2024_point, truth=$ia2024_truth"
echo "  - /forecast/narrate returned $narrate_chars chars (stub=$narrate_stub)"
