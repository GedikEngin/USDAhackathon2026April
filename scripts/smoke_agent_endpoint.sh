#!/usr/bin/env bash
# scripts/smoke_agent_endpoint.sh
#
# End-to-end smoke test for POST /agent/report.
#
# 1. Uploads demo image to /classify
# 2. Extracts the fields /agent/report needs
# 3. Posts a sustainability-report query to /agent/report
# 4. Prints a human-readable summary
#
# Requirements: the backend must be running on localhost:8000 with
# ANTHROPIC_API_KEY set, and jq must be installed.
#
# Usage:
#   bash scripts/smoke_agent_endpoint.sh
#   IMAGE=data/loveda/Val/Urban/images_png/2523.png bash scripts/smoke_agent_endpoint.sh

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
IMAGE="${IMAGE:-data/loveda/Val/Urban/images_png/3546.png}"
QUERY="${QUERY:-Produce a short sustainability report for this parcel. Cover composition, current footprint, one realistic intervention, and any model-quality caveats.}"

if ! command -v jq >/dev/null 2>&1; then
    echo "error: jq is required (brew install jq / apt-get install jq)" >&2
    exit 2
fi

if [[ ! -f "$IMAGE" ]]; then
    echo "error: image not found: $IMAGE" >&2
    exit 2
fi

echo "[1/3] health check"
health=$(curl -sf "$BASE_URL/health")
echo "$health" | jq .
agent_available=$(echo "$health" | jq -r .agent_available)
if [[ "$agent_available" != "true" ]]; then
    echo "error: backend says agent_available=false. Set ANTHROPIC_API_KEY and restart." >&2
    exit 3
fi

echo
echo "[2/3] POST /classify  (image=$IMAGE)"
classify_response=$(curl -sf -X POST "$BASE_URL/classify" \
    -F "file=@$IMAGE" -F "tta=true" -F "pixel_size_m=0.3")
echo "$classify_response" | jq '{
  percentages,
  total_area_ha: .emissions.total_area_ha,
  total_annual: .emissions.total_annual_tco2e_per_yr,
  inference_ms
}'

# Build the agent-report request body by piping classify_response through jq.
# We keep percentages, emissions, total_area_ha and add query + image_label.
image_label=$(basename "$IMAGE")
agent_request=$(echo "$classify_response" | jq \
    --arg query "$QUERY" \
    --arg label "$image_label" \
    '{
      percentages,
      emissions,
      total_area_ha: .emissions.total_area_ha,
      image_label: $label,
      query: $query
    }')

echo
echo "[3/3] POST /agent/report  (query: \"$QUERY\")"
t0=$(date +%s)
agent_response=$(echo "$agent_request" | curl -sf -X POST "$BASE_URL/agent/report" \
    -H "Content-Type: application/json" \
    --data @-)
t1=$(date +%s)
echo "  wallclock: $((t1 - t0))s"
echo

echo "=== final_text ==="
echo "$agent_response" | jq -r .final_text
echo
echo "=== tool calls ==="
echo "$agent_response" | jq '.tool_calls | map({turn, name, error})'
echo
echo "=== usage ==="
echo "$agent_response" | jq '{turns_used, stop_reason, usage}'
