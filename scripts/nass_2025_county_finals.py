#!/usr/bin/env python
"""
scripts/nass_2025_county_finals.py

Path 3: pull county-level USDA NASS 2025 final corn yields for 5 states
(IA / NE / WI / MO / CO) and write a JSON keyed by FIPS for the corn-yield-app
to render side-by-side with the model's 2025 predictions.

NASS county final estimates for 2025 are typically available Feb–Apr 2026.
A handful of small counties may be suppressed via the (D) disclosure flag —
those just appear as missing in the output.

Output:
  data/v2/nass_2025_county_finals.json
  ian/corn-yield-app/app/lib/nass_2025_county_finals.json   (consumed by the UI)

Schema:
  {
    "_meta": { "pulled_at": "...", "year": 2025, "n_counties": 263, "n_missing": 4 },
    "19015": {"fips": "19015", "state": "IA", "yield": 218.4},  # Boone, IA
    ...
  }
"""

from __future__ import annotations

import json
import os
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
STATES = ["CO", "IA", "MO", "NE", "WI"]
YEAR = 2025
REQUEST_DELAY_SEC = 2.0
MAX_RETRIES = 4
BACKOFF_BASE = 30

OUT_REPO = REPO_ROOT / "data" / "v2" / "nass_2025_county_finals.json"
OUT_APP = REPO_ROOT / "ian" / "corn-yield-app" / "app" / "lib" / "nass_2025_county_finals.json"


def request_with_retry(params: dict, ctx: str) -> requests.Response | None:
    for attempt in range(MAX_RETRIES):
        try:
            # County-level queries can return a lot of rows; allow 180s.
            r = requests.get(BASE_URL, params=params, timeout=180)
        except requests.exceptions.RequestException as e:
            wait = 5 * (attempt + 1)
            print(f"    {ctx}: network error ({type(e).__name__}); retry {attempt + 1}/{MAX_RETRIES} in {wait}s")
            time.sleep(wait)
            continue
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


def parse_yield(v) -> float | None:
    if v is None:
        return None
    s = str(v).replace(",", "").strip()
    if not s or s.upper() in ("(D)", "(NA)", "(Z)", "(X)"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def pull_state_counties(state_alpha: str) -> list[dict]:
    """Pull all 2025 county-level corn yield records for one state.
    No reference_period_desc filter — NASS labels vary at county level.
    We accept whatever YEAR/MARKETING-YEAR record they publish; in practice
    NASS doesn't issue county-level monthly forecasts, so the year-level
    record IS the final."""
    params = {
        "key": API_KEY,
        "agg_level_desc": "COUNTY",
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
        return []
    return payload.get("data", []) or []


def main():
    print(f"[pull] NASS county-level 2025 corn yield finals — 5 states")
    out: dict = {"_meta": {
        "pulled_at": datetime.now(timezone.utc).isoformat(),
        "year": YEAR,
        "stages_in_order": ["Final"],
    }}
    n_records = 0
    n_missing = 0
    for alpha in STATES:
        print(f"  {alpha}:")
        records = pull_state_counties(alpha)
        n_kept = 0
        for rec in records:
            county_ansi = (rec.get("county_ansi") or "").strip()
            state_ansi = (rec.get("state_ansi") or "").strip()
            county_name = (rec.get("county_name") or "").strip()
            if not county_ansi or county_name.upper().startswith("OTHER"):
                continue  # skip "OTHER (COMBINED) COUNTIES" placeholder rows
            fips = state_ansi.zfill(2) + county_ansi.zfill(3)
            yield_val = parse_yield(rec.get("Value"))
            if yield_val is None:
                n_missing += 1
                continue
            out[fips] = {
                "fips": fips,
                "state": alpha,
                "county": f"{county_name.title()}, {alpha}",
                "yield": round(yield_val, 1),
            }
            n_kept += 1
            n_records += 1
        print(f"    {len(records)} records → {n_kept} counties kept")
        time.sleep(REQUEST_DELAY_SEC)

    out["_meta"]["n_counties"] = n_records
    out["_meta"]["n_missing"] = n_missing
    out["_meta"]["status"] = "available" if n_records > 0 else "pending_usda_publication"
    if n_records == 0:
        out["_meta"]["note"] = (
            "NASS county-level final yields for 2025 are not yet in QuickStats. "
            "USDA typically publishes them May–June following the crop year. "
            "Re-run this script then to populate the UI's USDA-vs-model county comparison."
        )

    OUT_REPO.parent.mkdir(parents=True, exist_ok=True)
    OUT_APP.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPO, "w") as fh:
        json.dump(out, fh, indent=2)
    with open(OUT_APP, "w") as fh:
        json.dump(out, fh, indent=2)

    print()
    print(f"[write] {OUT_REPO}")
    print(f"[write] {OUT_APP}")
    print()
    print(f"Summary: {n_records} county finals, {n_missing} suppressed/missing")
    # Sample by state
    by_state: dict[str, list[float]] = {}
    for k, v in out.items():
        if k.startswith("_"):
            continue
        by_state.setdefault(v["state"], []).append(v["yield"])
    for st, vals in sorted(by_state.items()):
        avg = sum(vals) / len(vals) if vals else 0
        print(f"  {st}: {len(vals)} counties, mean={avg:.1f}, "
              f"range=[{min(vals):.0f}, {max(vals):.0f}]")


if __name__ == "__main__":
    main()