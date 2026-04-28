#!/usr/bin/env python
"""
scripts/nass_2025_state_forecasts.py

Pulls 2025 USDA NASS state-level corn yield records for IA / NE / WI / MO / CO
and writes a JSON the corn-yield-app can overlay on the cone-of-uncertainty
chart.

What NASS exposes:
  - Source: SURVEY (current-year), CENSUS (final/longer-term)
  - For 2025 corn (today = 2026-04-28), the in-season Aug/Sep/Oct/Nov forecast
    vintages and the January final estimate are all queryable via QuickStats.
  - The vintage is encoded in `reference_period_desc` (e.g. "YEAR - AUG FORECAST",
    "YEAR - SEP FORECAST", "YEAR - OCT FORECAST", "YEAR - NOV FORECAST",
    "YEAR") and `begin_code` (08, 09, 10, 11, or no value for the final).

This script queries with no `reference_period_desc` filter, accepts every
record returned, then maps each row to one of our 4 forecast stages
(Aug 1 / Sep 1 / Oct 1 / Final). The mapping prefers explicit forecast
labels; falls back to the latest non-forecast record as the "Final".

Output:
  data/v2/nass_2025_state_forecasts.json   (also copied into corn-yield-app/app/lib/)

Schema:
  {
    "Iowa":     [ {"stage": "Aug 1", "usda": 196}, {"stage": "Sep 1", "usda": 198}, ... ],
    "Nebraska": [ ... ],
    ...
    "_meta": { "pulled_at": "...", "raw_records": 23 }
  }
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")
API_KEY = os.environ.get("NASS_API_KEY")
if not API_KEY:
    print("ERROR: NASS_API_KEY not in .env", file=sys.stderr)
    sys.exit(1)

BASE_URL = "https://quickstats.nass.usda.gov/api/api_GET/"
STATES = {"CO": "Colorado", "IA": "Iowa", "MO": "Missouri", "NE": "Nebraska", "WI": "Wisconsin"}
YEAR = 2025
REQUEST_DELAY_SEC = 2.0
MAX_RETRIES = 4
BACKOFF_BASE = 30

OUT_REPO = REPO_ROOT / "data" / "v2" / "nass_2025_state_forecasts.json"
OUT_APP = REPO_ROOT / "ian" / "corn-yield-app" / "app" / "lib" / "nass_2025_state_forecasts.json"

# Mapping from NASS reference_period_desc → (cone chart stage, priority).
# Lower priority number = preferred when multiple records map to the same stage.
# Specifically: for "Final", prefer YEAR (the January final estimate) over
# NOV FORECAST (the last in-season figure that gets revised in January).
STAGE_FROM_REFERENCE: dict[str, tuple[str, int]] = {
    "year - aug forecast": ("Aug 1", 0),
    "year - sep forecast": ("Sep 1", 0),
    "year - oct forecast": ("Oct 1", 0),
    "year": ("Final", 0),                # January final — preferred
    "year - jan estimate": ("Final", 1),
    "marketing year": ("Final", 2),
    "year - nov forecast": ("Final", 3), # last in-season — fallback only
}


def request_with_retry(params: dict, ctx: str) -> requests.Response | None:
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(BASE_URL, params=params, timeout=60)
        except requests.exceptions.RequestException as e:
            print(f"    {ctx}: network error: {e}")
            return None
        if r.status_code in (200, 400):
            return r
        if r.status_code == 403:
            wait = BACKOFF_BASE * (2 ** attempt)
            print(f"    {ctx}: rate-limited (403), waiting {wait}s before retry {attempt + 1}/{MAX_RETRIES}")
            time.sleep(wait)
            continue
        print(f"    {ctx}: HTTP {r.status_code}: {r.text[:150]}")
        return r
    print(f"    {ctx}: gave up after {MAX_RETRIES} retries")
    return None


def pull_state(state_alpha: str) -> list[dict]:
    """Return all records NASS has for 2025 corn yield in this state."""
    params = {
        "key": API_KEY,
        "agg_level_desc": "STATE",
        "source_desc": "SURVEY",
        "commodity_desc": "CORN",
        "statisticcat_desc": "YIELD",
        "unit_desc": "BU / ACRE",
        "util_practice_desc": "GRAIN",
        "prodn_practice_desc": "ALL PRODUCTION PRACTICES",
        "year": YEAR,
        "state_alpha": state_alpha,
        "format": "JSON",
    }
    r = request_with_retry(params, state_alpha)
    if r is None or r.status_code == 400:
        return []
    try:
        payload = r.json()
    except ValueError:
        print(f"    {state_alpha}: non-JSON response")
        return []
    return payload.get("data", []) or []


def normalize_stage(record: dict) -> tuple[str | None, int]:
    """Map a NASS record to (stage_label, priority). Lower priority wins."""
    rpd = (record.get("reference_period_desc") or "").strip().lower()
    if rpd in STAGE_FROM_REFERENCE:
        return STAGE_FROM_REFERENCE[rpd]
    # Fallback: detect a month token
    m = re.search(r"\b(aug|sep|oct|nov|jan)\b", rpd)
    if m:
        token = m.group(1)
        stage = {"aug": "Aug 1", "sep": "Sep 1", "oct": "Oct 1",
                 "nov": "Final", "jan": "Final"}[token]
        return stage, 9  # unknown vintage → low priority
    return None, 99


def parse_value(v) -> float | None:
    if v is None:
        return None
    s = str(v).replace(",", "").strip()
    if not s or s.upper() in ("(D)", "(NA)", "(Z)", "(X)"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def main():
    print(f"[pull] NASS state-level 2025 corn yield, 5 states")
    raw_records = []
    # by_state[name][stage] = (value, priority)
    by_state: dict[str, dict[str, tuple[float, int]]] = {state: {} for state in STATES.values()}
    raw_dump: dict[str, list[dict]] = {}

    for alpha, name in STATES.items():
        print(f"  {alpha} ({name}):")
        records = pull_state(alpha)
        print(f"    {len(records)} records returned")
        raw_dump[alpha] = []
        for rec in records:
            stage, prio = normalize_stage(rec)
            value = parse_value(rec.get("Value"))
            rpd = rec.get("reference_period_desc")
            print(f"      reference_period_desc={rpd!r:40s} stage={stage} prio={prio} value={value}")
            raw_dump[alpha].append({
                "reference_period_desc": rpd,
                "begin_code": rec.get("begin_code"),
                "end_code": rec.get("end_code"),
                "stage": stage,
                "priority": prio,
                "value": value,
                "load_time": rec.get("load_time"),
            })
            if stage and value is not None:
                prev = by_state[name].get(stage)
                if prev is None or prio < prev[1]:
                    by_state[name][stage] = (value, prio)
            raw_records.append({"state": alpha, **rec})
        time.sleep(REQUEST_DELAY_SEC)

    # Build the per-state arrays in chart order
    stage_order = ["Aug 1", "Sep 1", "Oct 1", "Final"]
    out: dict = {"_meta": {
        "pulled_at": datetime.now(timezone.utc).isoformat(),
        "raw_records": len(raw_records),
        "year": YEAR,
        "stages_in_order": stage_order,
    }}
    for name in STATES.values():
        s_records = []
        for stage in stage_order:
            entry = by_state[name].get(stage)
            if entry is not None:
                s_records.append({"stage": stage, "usda": round(entry[0], 1)})
        out[name] = s_records

    # Persist
    OUT_REPO.parent.mkdir(parents=True, exist_ok=True)
    OUT_APP.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPO, "w") as fh:
        json.dump(out, fh, indent=2)
    with open(OUT_APP, "w") as fh:
        json.dump(out, fh, indent=2)

    # Also dump the raw records side-by-side for debugging the mapping
    raw_path = OUT_REPO.with_suffix(".raw.json")
    with open(raw_path, "w") as fh:
        json.dump(raw_dump, fh, indent=2)

    print()
    print(f"[write] {OUT_REPO}")
    print(f"[write] {OUT_APP}  (consumed by /predictions and /updated-forecast)")
    print(f"[write] {raw_path}  (raw records dump for debugging)")
    print()
    print("Summary:")
    for name in STATES.values():
        stages = ", ".join(f"{r['stage']}={r['usda']}" for r in out[name])
        print(f"  {name:10s}  {stages or '(no records mapped)'}")


if __name__ == "__main__":
    main()