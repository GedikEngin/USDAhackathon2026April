"""
Download per-state Cropland Data Layer (CDL) geotiffs from the USDA NASS
CropScape web service for use as annual corn masks in Phase 2-D.1.

CropScape exposes a `GetCDLFile` SOAP/REST endpoint hosted at GMU that takes
(year, state_fips) and returns an XML document containing a URL to a
temporarily-hosted geotiff on the GMU server. This script:

  1. Hits GetCDLFile for each (state, year) in the request set.
  2. Parses the response XML to extract the geotiff URL.
  3. Downloads the geotiff to disk.
  4. Skips already-downloaded files (resumable).
  5. Backs off + retries on transient errors.

The 2024 CDL is published natively at 10 m (a resolution change from prior
30 m). Per the CropScape announcement page, USDA also publishes a 30 m
nearest-neighbor-resampled version for consistency with historical CDL. The
GetCDLFile API does NOT, as of this writing, expose a resolution toggle — it
returns whatever the server has for that (year, fips). We download whatever
comes back; the corn-mask conversion script (cdl_to_corn_mask.py) handles
resampling to 30 m EPSG:5070 if the source is finer.

Phase 2-D.1 sub-phase: D.1.a (CDL prep).

Usage:
    python scripts/download_cdl.py
    python scripts/download_cdl.py --years 2013-2024 --states IA,NE
    python scripts/download_cdl.py --out phase2/cdl/raw

Outputs:
    phase2/cdl/raw/cdl_<state_alpha>_<year>.tif

Notes
-----
- The CropScape endpoint is occasionally flaky (slow XML, brief 5xx).
  Pattern follows nass_pull.py: 2s base delay between requests, exponential
  backoff up to 240s, 4 retries. Intentionally polite.
- No authentication needed — CropScape is open-access.
- Output filename is keyed on state alpha (IA, CO, ...), not FIPS, to match
  the rest of the pipeline's naming convention. State alpha → FIPS mapping
  is local to this module.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import requests


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# State alpha → FIPS. The 5 states in scope for v2 (locked).
STATE_FIPS: Dict[str, str] = {
    "CO": "08",
    "IA": "19",
    "MO": "29",
    "NE": "31",
    "WI": "55",
}

# CropScape endpoints. The /axis2/services/CDLService/ path is the live one
# as of 2026-04. If GMU ever migrates this, update here.
GETCDL_FILE_URL = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile"

# Request etiquette knobs — match nass_pull.py.
DEFAULT_DELAY_SEC = 2.0
DEFAULT_MAX_RETRIES = 4
DEFAULT_BACKOFF_CAP_SEC = 240.0
DEFAULT_REQUEST_TIMEOUT_SEC = 120  # XML can be slow to generate on the server

DEFAULT_YEAR_RANGE = (2013, 2024)
DEFAULT_OUT_DIR = "phase2/cdl/raw"


# -----------------------------------------------------------------------------
# Argparse
# -----------------------------------------------------------------------------


def _parse_year_range(s: str) -> List[int]:
    """Accept "2013-2024" or "2013,2014,2015" or a single year."""
    s = s.strip()
    if "-" in s:
        lo, hi = s.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    if "," in s:
        return [int(x) for x in s.split(",") if x.strip()]
    return [int(s)]


def _parse_states(s: str) -> List[str]:
    states = [x.strip().upper() for x in s.split(",") if x.strip()]
    bad = [x for x in states if x not in STATE_FIPS]
    if bad:
        raise argparse.ArgumentTypeError(
            f"Unknown state alpha codes: {bad}. Valid: {sorted(STATE_FIPS)}"
        )
    return states


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download per-state CDL geotiffs from CropScape (D.1.a)."
    )
    p.add_argument(
        "--out",
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    p.add_argument(
        "--years",
        type=_parse_year_range,
        default=list(range(DEFAULT_YEAR_RANGE[0], DEFAULT_YEAR_RANGE[1] + 1)),
        help=(
            f"Years to download. Format: '2013-2024' (range), '2013,2014,2015' "
            f"(list), or a single year. Default: {DEFAULT_YEAR_RANGE[0]}-"
            f"{DEFAULT_YEAR_RANGE[1]}"
        ),
    )
    p.add_argument(
        "--states",
        type=_parse_states,
        default=list(STATE_FIPS),
        help=(
            f"State alpha codes, comma-separated. "
            f"Default: all 5 = {','.join(sorted(STATE_FIPS))}"
        ),
    )
    p.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Max retries per request (default: {DEFAULT_MAX_RETRIES})",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_SEC,
        help=f"Base delay between requests in seconds (default: {DEFAULT_DELAY_SEC})",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist on disk.",
    )
    return p


# -----------------------------------------------------------------------------
# CropScape API calls
# -----------------------------------------------------------------------------


def _request_cdl_url(
    year: int,
    fips: str,
    *,
    max_retries: int,
    base_delay: float,
    timeout: int = DEFAULT_REQUEST_TIMEOUT_SEC,
) -> str:
    """Hit GetCDLFile and return the temporary geotiff URL.

    Raises RuntimeError if all retries are exhausted.
    """
    params = {"year": str(year), "fips": fips}
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.get(GETCDL_FILE_URL, params=params, timeout=timeout)
            r.raise_for_status()
            # Response is XML like:
            #   <?xml version='1.0' encoding='UTF-8'?>
            #   <ns:returnURL xmlns:ns='...'>https://nassgeodata.gmu.edu/.../CDL_2018_19.tif</ns:returnURL>
            # Some responses use slightly different element names; we walk
            # the tree and pull the first .tif URL we find.
            root = ET.fromstring(r.text)
            url = _extract_tif_url(root, raw_text=r.text)
            if url is None:
                raise RuntimeError(
                    f"GetCDLFile returned XML with no .tif URL for "
                    f"year={year} fips={fips}: {r.text[:300]}"
                )
            return url
        except (requests.RequestException, ET.ParseError, RuntimeError) as e:
            last_err = e
            if attempt >= max_retries:
                break
            sleep_s = min(base_delay * (2**attempt), DEFAULT_BACKOFF_CAP_SEC)
            print(
                f"  retry {attempt + 1}/{max_retries} "
                f"(year={year} fips={fips}) after {sleep_s:.0f}s: {e}",
                file=sys.stderr,
            )
            time.sleep(sleep_s)
    raise RuntimeError(
        f"GetCDLFile failed after {max_retries} retries "
        f"(year={year} fips={fips}): {last_err}"
    )


def _extract_tif_url(root: ET.Element, *, raw_text: str) -> str | None:
    """Find the .tif URL in the GetCDLFile XML response.

    The XML namespace varies; instead of a strict path lookup, walk the tree
    and pick the first text node ending with .tif (case-insensitive). Falls
    back to a substring scan of the raw text if XML parsing yielded nothing.
    """
    for elem in root.iter():
        text = (elem.text or "").strip()
        if text.lower().endswith(".tif") and text.lower().startswith("http"):
            return text
    # Last-ditch: substring scan. CropScape has historically returned XML
    # with malformed namespaces; the raw URL is still in the body.
    lower = raw_text.lower()
    idx = lower.find("https://")
    while idx != -1:
        end = lower.find("</", idx)
        if end == -1:
            end = len(lower)
        candidate = raw_text[idx:end].strip()
        if candidate.lower().endswith(".tif"):
            return candidate
        idx = lower.find("https://", idx + 1)
    return None


def _download_geotiff(url: str, out_path: Path, *, timeout: int = 600) -> int:
    """Stream-download a geotiff to out_path. Returns bytes written.

    Writes to a temp suffix and renames on success, so an interrupted
    download never leaves a half-file at the final path.
    """
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    written = 0
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with tmp_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                if chunk:
                    f.write(chunk)
                    written += len(chunk)
    tmp_path.replace(out_path)
    return written


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------


def main() -> int:
    args = _build_argparser().parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets: List[Tuple[str, str, int]] = [
        (state_alpha, STATE_FIPS[state_alpha], year)
        for state_alpha in args.states
        for year in args.years
    ]
    total = len(targets)

    print(f"CDL download — {total} (state, year) targets")
    print(f"  states: {','.join(args.states)}")
    print(f"  years:  {min(args.years)}-{max(args.years)}")
    print(f"  out:    {out_dir.resolve()}")
    print(f"  delay:  {args.delay}s base, exponential backoff up to "
          f"{DEFAULT_BACKOFF_CAP_SEC}s, {args.retries} retries")
    print()

    n_skipped = 0
    n_downloaded = 0
    n_failed = 0
    bytes_total = 0
    failures: List[str] = []

    for i, (state_alpha, fips, year) in enumerate(targets, start=1):
        out_path = out_dir / f"cdl_{state_alpha}_{year}.tif"
        prefix = f"[{i:>3}/{total}] {state_alpha} {year}"

        if out_path.exists() and not args.force:
            n_skipped += 1
            print(f"{prefix}  skip (exists, {out_path.stat().st_size / 1e6:.1f} MB)")
            continue

        try:
            tif_url = _request_cdl_url(
                year=year,
                fips=fips,
                max_retries=args.retries,
                base_delay=args.delay,
            )
            print(f"{prefix}  GET {tif_url}")
            n_bytes = _download_geotiff(tif_url, out_path)
            n_downloaded += 1
            bytes_total += n_bytes
            print(f"{prefix}  -> {out_path.name} ({n_bytes / 1e6:.1f} MB)")
        except Exception as e:
            n_failed += 1
            failures.append(f"{state_alpha} {year}: {e}")
            print(f"{prefix}  FAIL: {e}", file=sys.stderr)
            # Don't bail on first failure — keep going, surface failures at end.

        # Polite delay between requests, even on skip.
        if i < total:
            time.sleep(args.delay)

    print()
    print("Summary:")
    print(f"  downloaded: {n_downloaded}  ({bytes_total / 1e9:.2f} GB)")
    print(f"  skipped:    {n_skipped}")
    print(f"  failed:     {n_failed}")
    if failures:
        print()
        print("Failures (re-run to retry):")
        for f in failures:
            print(f"  {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
