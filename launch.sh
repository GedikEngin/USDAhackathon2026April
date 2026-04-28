#!/usr/bin/env bash
# launch.sh — start FastAPI backend + Next.js Aurum app together.
#
#   FastAPI (uvicorn)  → http://localhost:8000/ui/         (v1 Land Use + /forecast/* shell)
#   Next.js (Aurum)    → http://localhost:3000/            (corn-yield app)
#                      → http://localhost:3000/predictions  (Sagemaker pipeline forecasts)
#                      → http://localhost:3000/updated-forecast (Phase 2-D.1.e hybrid model)
#
# Ctrl-C tears down both.

set -u
cd "$(dirname "$0")"

# ── pick the right Python env for the FastAPI backend ────────────────────────
# `landuse2` is the working env on this machine (`landuse` has a broken scipy
# install). Try them in order; fall back to whatever uvicorn is already on PATH.
PYTHON_BIN=""
for env in landuse2 landuse; do
  candidate="$HOME/miniconda3/envs/$env/bin/python"
  if [ -x "$candidate" ]; then
    if "$candidate" -c "import uvicorn, fastapi, backend.main" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      ACTIVE_ENV="$env"
      break
    fi
  fi
done
if [ -z "$PYTHON_BIN" ]; then
  if command -v uvicorn >/dev/null 2>&1; then
    PYTHON_BIN="$(which python)"
    ACTIVE_ENV="(system)"
    echo "[launch] WARNING: no working conda env found, falling back to system python at $PYTHON_BIN"
  else
    echo "[launch] ERROR: no conda env (landuse2/landuse) and no system uvicorn — cannot start backend." >&2
    echo "[launch]        Install uvicorn or fix one of the conda envs, then re-run." >&2
    exit 1
  fi
fi

# Make `node`/`npm` available in non-interactive shells via nvm.
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
if [ -s "$NVM_DIR/nvm.sh" ]; then
  # shellcheck disable=SC1091
  . "$NVM_DIR/nvm.sh"
  nvm use 20 >/dev/null 2>&1 || true
fi
if ! command -v npm >/dev/null 2>&1; then
  echo "[launch] WARNING: npm not on PATH — Next.js will fail to start." >&2
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

# ── 1. FastAPI on :8000 ──────────────────────────────────────────────────────
echo "[launch] starting FastAPI (uvicorn) on :8000  [env=$ACTIVE_ENV]"
"$PYTHON_BIN" -m uvicorn backend.main:app --app-dir . --host 0.0.0.0 --port 8000 \
  > runs/launch_uvicorn.log 2>&1 &
PIDS+=($!)

# ── 2. Next.js Aurum on :3000 ────────────────────────────────────────────────
echo "[launch] starting Next.js (Aurum) on :3000"
mkdir -p runs
( cd ian/corn-yield-app && npm run dev > ../../runs/launch_next.log 2>&1 ) &
PIDS+=($!)

# ── 3. quick health probe so failures surface immediately ────────────────────
echo "[launch] waiting up to 30s for both ports to bind…"
ok_back=0; ok_front=0
for i in $(seq 1 30); do
  sleep 1
  [ "$ok_back" -eq 0 ] && curl -fsS -o /dev/null http://localhost:8000/health 2>/dev/null && \
    { ok_back=1; echo "[launch] :8000 backend ready ($(($i))s)"; }
  [ "$ok_front" -eq 0 ] && curl -fsS -o /dev/null http://localhost:3000 2>/dev/null && \
    { ok_front=1; echo "[launch] :3000 frontend ready ($(($i))s)"; }
  [ "$ok_back" -eq 1 ] && [ "$ok_front" -eq 1 ] && break
done
[ "$ok_back" -eq 0 ]  && echo "[launch] WARNING: :8000 not responding — check runs/launch_uvicorn.log"
[ "$ok_front" -eq 0 ] && echo "[launch] WARNING: :3000 not responding — check runs/launch_next.log"

cat <<EOF

[launch] both servers running. URLs:
  Land Use & GHG (v1)        →  http://localhost:8000/ui/
  Corn Yield home (Aurum)    →  http://localhost:3000/
  Yield Forecasts (Sagemaker)→  http://localhost:3000/predictions
  Updated Model (Phase D.1.e)→  http://localhost:3000/updated-forecast
  FastAPI Swagger            →  http://localhost:8000/docs

  Logs:  tail -f runs/launch_uvicorn.log runs/launch_next.log
  Stop:  Ctrl-C
EOF

wait