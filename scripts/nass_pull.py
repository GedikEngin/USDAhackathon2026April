"""
Pull NASS county-level corn data for 5 states, 2015-2024.
Handles rate limiting (HTTP 403 from Azure gateway) with backoff.
"""

import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("NASS_API_KEY")
if not API_KEY:
    raise RuntimeError("NASS_API_KEY not found in .env")

BASE_URL = "https://quickstats.nass.usda.gov/api/api_GET/"
STATES = ["CO", "IA", "MO", "NE", "WI"]
YEARS = list(range(2004, 2025))

# Slower default + retry on rate limit
REQUEST_DELAY_SEC = 2.0      # was 0.5 — bump to 2s between requests
MAX_RETRIES = 4
BACKOFF_BASE = 30            # seconds; doubles each retry

PRACTICES = [
    ("ALL PRODUCTION PRACTICES", "all"),
    ("IRRIGATED", "irr"),
    ("NON-IRRIGATED", "noirr"),
]

BASE_ITEMS = [
    {"label": "yield_bu_acre",    "params": {"commodity_desc":"CORN","statisticcat_desc":"YIELD",         "unit_desc":"BU / ACRE","util_practice_desc":"GRAIN"}},
    {"label": "production_bu",    "params": {"commodity_desc":"CORN","statisticcat_desc":"PRODUCTION",    "unit_desc":"BU",       "util_practice_desc":"GRAIN"}},
    {"label": "acres_harvested",  "params": {"commodity_desc":"CORN","statisticcat_desc":"AREA HARVESTED","unit_desc":"ACRES",    "util_practice_desc":"GRAIN"}},
    {"label": "acres_planted",    "params": {"commodity_desc":"CORN","statisticcat_desc":"AREA PLANTED",  "unit_desc":"ACRES"}},
]


def request_with_retry(params, state):
    """GET a NASS query, with backoff on 403 rate limits."""
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(BASE_URL, params=params, timeout=60)
        except requests.exceptions.RequestException as e:
            print(f"    {state}: network error: {e}")
            return None

        # 200 OK
        if r.status_code == 200:
            return r

        # 400 = "no data for this combination" — not an error
        if r.status_code == 400:
            return r

        # 403 = rate limit. Back off.
        if r.status_code == 403:
            wait = BACKOFF_BASE * (2 ** attempt)
            print(f"    {state}: rate limited (403). Waiting {wait}s before retry {attempt+1}/{MAX_RETRIES}...")
            time.sleep(wait)
            continue

        # Anything else
        print(f"    {state}: HTTP {r.status_code}: {r.text[:150]}")
        return r

    print(f"    {state}: gave up after {MAX_RETRIES} retries")
    return None


def fetch(params):
    """Run one query across all states."""
    base_params = {
        "key": API_KEY,
        "agg_level_desc": "COUNTY",
        "source_desc": "SURVEY",
        "year__GE": min(YEARS),
        "year__LE": max(YEARS),
        "format": "JSON",
        **params,
    }
    frames = []
    for st in STATES:
        q = {**base_params, "state_alpha": st}
        r = request_with_retry(q, st)
        if r is None:
            continue

        if r.status_code == 400:
            print(f"    {st}: no data for this combination")
            time.sleep(REQUEST_DELAY_SEC)
            continue

        try:
            payload = r.json()
        except ValueError:
            print(f"    {st}: non-JSON response")
            time.sleep(REQUEST_DELAY_SEC)
            continue

        if "data" in payload:
            frames.append(pd.DataFrame(payload["data"]))
            print(f"    {st}: {len(payload['data'])} rows")
        elif "error" in payload:
            print(f"    {st}: {payload['error']}")

        time.sleep(REQUEST_DELAY_SEC)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def clean(df, value_col):
    cols = ["GEOID", "year", "state_alpha", "county_name", value_col]
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = df.copy()
    df = df[df["county_ansi"].astype(str).str.strip() != ""]
    df = df[~df["county_name"].astype(str).str.upper().str.startswith("OTHER")]
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["GEOID"] = (df["state_ansi"].astype(str).str.zfill(2)
                   + df["county_ansi"].astype(str).str.zfill(3))
    df["Value"] = df["Value"].astype(str).str.replace(",", "", regex=False)
    df[value_col] = pd.to_numeric(df["Value"], errors="coerce")
    df["year"] = df["year"].astype(int)
    out = df[cols].drop_duplicates(subset=["GEOID", "year"])
    return out


# --- Main ----------------------------------------------------
print("Pulling NASS data with rate-limit-aware delays...")
print(f"Delay between requests: {REQUEST_DELAY_SEC}s\n")

cleaned_tables = {}
for practice_long, practice_short in PRACTICES:
    print(f"=== Practice: {practice_long} ===")
    for item in BASE_ITEMS:
        col_name = f"{item['label']}_{practice_short}"
        print(f"  {col_name}:")
        params = {**item["params"], "prodn_practice_desc": practice_long}
        df = fetch(params)
        print(f"    Total: {len(df):,} rows")
        cleaned_tables[col_name] = clean(df, col_name)
    print()

# --- Merge ---------------------------------------------------
print("Merging...")
spine = "yield_bu_acre_all"
merged = cleaned_tables[spine].copy()
for col, df in cleaned_tables.items():
    if col == spine:
        continue
    if df.empty:
        merged[col] = pd.NA
    else:
        merged = merged.merge(df[["GEOID", "year", col]], on=["GEOID", "year"], how="outer")

yield_cols = [c for c in merged.columns if c.startswith("yield_bu_acre")]
merged = merged.dropna(subset=yield_cols, how="all")

id_cols = ["GEOID", "year", "state_alpha", "county_name"]
metric_cols = sorted(c for c in merged.columns if c not in id_cols)
merged = merged[id_cols + metric_cols]

print(f"\nTotal rows: {len(merged):,}")
print(f"Unique counties: {merged['GEOID'].nunique()}")
print(f"\nColumn coverage:")
for col in metric_cols:
    nn = merged[col].notna().sum()
    print(f"  {col:30s} {nn:>5,} ({100*nn/len(merged):5.1f}%)")

merged.to_csv("nass_corn_5states_2005_2024.csv", index=False)
print(f"\nSaved to nass_corn_5states_2005_2024.csv")