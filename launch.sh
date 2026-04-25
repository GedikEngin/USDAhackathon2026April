#!/usr/bin/env bash
# launch.sh — start FastAPI (Land Use + Forecast) and Next.js (Aurum) together.
#   uvicorn  → http://localhost:8000/ui/
#   next dev → http://localhost:3000/   (embedded in the "Alternate Model" tab)
# Ctrl-C tears down both.

set -u
cd "$(dirname "$0")"

# Load nvm so `node`/`npm` are on PATH even in non-interactive shells.
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
if [ -s "$NVM_DIR/nvm.sh" ]; then
  # shellcheck disable=SC1091
  . "$NVM_DIR/nvm.sh"
  nvm use 20 >/dev/null 2>&1 || true
fi

PIDS=()
cleanup() {
  echo
  echo "[launch] shutting down…"
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM

echo "[launch] starting FastAPI (uvicorn) on :8000"
uvicorn backend.main:app --app-dir . --host 0.0.0.0 --port 8000 &
PIDS+=($!)

echo "[launch] starting Next.js (Aurum) on :3000"
( cd ian/corn-yield-app && npm run dev ) &
PIDS+=($!)

echo "[launch] both up. Open http://localhost:8000/ui/ — Ctrl-C to stop."
wait
