#!/usr/bin/env python
"""
scripts/download_hls.py
HLS download orchestration for Phase 2-D.1.b.

Per (state, year) cell:
  1. CMR query HLSL30 + HLSS30 over the growing season (May 1 - Nov 15)
     with state bbox and cloud_cover <= 70%.
  2. Bucket granules by calendar phase (aug1/sep1/oct1/final), keep the
     top 100 cleanest per phase.
  3. Walk granules. For each:
       a. download via earthaccess.download() to data/v2/hls/raw/<granule_id>/
       b. call extract_chips_for_granule() (from scripts.extract_chips)
          --> returns a list of ChipIndexRow (positive + negative)
       c. flush rows to the per-cell shard parquet (every N granules)
       d. delete the granule directory
  4. Resumability: granule_id presence in the shard parquet means done.
     Re-running the same cell skips already-processed granules.

Sharding for parallel terminals
-------------------------------
Each (state, year) cell writes to its own shard:
  data/v2/hls/chip_index_shards/chip_index_<state>_<year>.parquet
A separate one-shot merge script (D.1.b's last sub-step) concatenates all
shards into the final data/v2/hls/chip_index.parquet.

Run multiple terminals in parallel with --shard K/N. Cells are assigned by
a deterministic hash so two terminals with --shard 1/4 and --shard 2/4 see
disjoint cell sets without coordination. Recommended max: 4 concurrent
terminals (Earthdata practical rate limit).

Usage examples
--------------
  # smoke test: one cell, capped granules
  python scripts/download_hls.py --cells IA-2018 --max-granules 5

  # all 12 years for one state, single terminal
  python scripts/download_hls.py --state IA

  # full grid across 4 terminals (run each in its own terminal):
  python scripts/download_hls.py --all --shard 1/4
  python scripts/download_hls.py --all --shard 2/4
  python scripts/download_hls.py --all --shard 3/4
  python scripts/download_hls.py --all --shard 4/4

  # dry run: print resolved CMR granule lists, no downloads
  python scripts/download_hls.py --cells IA-2018 --dry-run

Notes
-----
- Earthdata auth: assumes ~/.netrc is configured.
  earthaccess.login(strategy='netrc') is called once at startup.
- All paths resolved relative to the current working directory; intended to
  be run from the repo root.
- Granule cache directory (data/v2/hls/raw/) is rolling-deleted: each
  granule is removed after its chips are written. Steady-state disk
  pressure < 5 GB.
- Logs at runs/download_hls_<shard>_<timestamp>.log (live tail with
  `tail -f`). Stdout still gets pretty progress.
"""

from __future__ import annotations

# Stable hash across worker processes (so --shard partitioning is
# deterministic). Must be set before any string is hashed in a child proc;
# we set it explicitly here and document the contract.
import os
os.environ.setdefault("PYTHONHASHSEED", "0")

import argparse
import dataclasses
import datetime as dt
import hashlib
import logging
import shutil
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Make `forecast/` importable when run as a script from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402

from forecast.hls_common import (  # noqa: E402
    ChipIndexRow,
    CHIP_INDEX_SHARD_DIR_DEFAULT,
    CHIP_ROOT_DEFAULT,
    GRANULE_CACHE_DIR_DEFAULT,
    PHASE_TO_FORECAST_DATE,
    PHASE_WINDOWS,
    SENTINEL_GEOID_NO_INTERSECT,
    STATE_BBOX,
    cmr_short_names_both,
    configure_gdal_for_cloud_native_hls,
    label_calendar_phase,
    parse_granule_id,
    temporal_window_for_year,
)

# -----------------------------------------------------------------------------
# Year range and grid definition (locked in decisions log: 2013-2024 inclusive)
# -----------------------------------------------------------------------------
YEARS_DEFAULT: list[int] = list(range(2013, 2025))   # 2013..2024
STATES_DEFAULT: list[str] = ["IA", "NE", "WI", "MO", "CO"]

# Per-phase granule cap. Top 100 cleanest cloud cover per (state, year, phase)
# captures essentially all granules in 2013-2014 Landsat-only era and the
# cleanest 100 in 2015+ combined-sensor era. (Decisions log 2-D.1.kickoff.)
GRANULE_CAP_PER_PHASE_DEFAULT = 100
CMR_CLOUD_COVER_MAX_DEFAULT   = 70   # CMR-side prefilter

# Flush index parquet every N granules processed. With ~24,000 granules total
# at avg ~30s/granule, N=20 means a flush every ~10 minutes -- bounded loss
# on crash; trivial write overhead.
FLUSH_EVERY_N_GRANULES_DEFAULT = 20

EXTRACTOR_VERSION = "d1b-v1"


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def setup_logging(log_path: Optional[Path]) -> logging.Logger:
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="a", encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt,
                        handlers=handlers, force=True)
    return logging.getLogger("download_hls")


# -----------------------------------------------------------------------------
# Cell sharding
# -----------------------------------------------------------------------------
def cell_in_shard(state: str, year: int, shard_k: int, shard_n: int) -> bool:
    """Deterministic hash assignment of (state, year) to one of N shards.

    Uses md5 (stable across Python interpreters/processes regardless of
    PYTHONHASHSEED) so multiple terminals agree on the assignment without
    needing to coordinate via env vars.
    """
    if shard_n <= 0 or not (1 <= shard_k <= shard_n):
        raise ValueError(f"invalid shard {shard_k}/{shard_n}")
    key = f"{state}-{year}".encode()
    h = int.from_bytes(hashlib.md5(key).digest()[:8], "big")
    return (h % shard_n) == (shard_k - 1)


def parse_cells_arg(cells_arg: list[str]) -> list[tuple[str, int]]:
    """Parse '--cells IA-2018 IA-2019 ...' into [(state, year), ...]."""
    out: list[tuple[str, int]] = []
    for tok in cells_arg:
        try:
            state, year = tok.split("-")
            year = int(year)
        except (ValueError, AttributeError) as e:
            raise SystemExit(f"--cells token {tok!r} must be like 'IA-2018': {e}")
        if state not in STATES_DEFAULT:
            raise SystemExit(f"--cells: unknown state {state!r}, expected one of {STATES_DEFAULT}")
        if year not in YEARS_DEFAULT:
            raise SystemExit(f"--cells: year {year} outside D.1 range {YEARS_DEFAULT[0]}..{YEARS_DEFAULT[-1]}")
        out.append((state, year))
    return out


def resolve_grid(args: argparse.Namespace) -> list[tuple[str, int]]:
    """Apply --cells / --state / --year / --all and --shard K/N to produce
    the final list of (state, year) cells this process should run."""
    if args.cells:
        cells = parse_cells_arg(args.cells)
    elif args.all:
        cells = [(s, y) for s in STATES_DEFAULT for y in YEARS_DEFAULT]
    else:
        states = [args.state] if args.state else STATES_DEFAULT
        years = [args.year] if args.year else YEARS_DEFAULT
        cells = [(s, y) for s in states for y in years]

    if args.shard:
        try:
            k_str, n_str = args.shard.split("/")
            shard_k, shard_n = int(k_str), int(n_str)
        except ValueError:
            raise SystemExit(f"--shard {args.shard!r} must be like '1/4'")
        cells = [(s, y) for (s, y) in cells if cell_in_shard(s, y, shard_k, shard_n)]
        if not cells:
            raise SystemExit(f"shard {args.shard} matched no cells in the requested grid")
    return cells


# -----------------------------------------------------------------------------
# Shard parquet I/O
# -----------------------------------------------------------------------------
def shard_path_for(state: str, year: int, shard_dir: Path) -> Path:
    return shard_dir / f"chip_index_{state}_{year}.parquet"


def load_shard(path: Path) -> pd.DataFrame:
    """Read existing shard parquet or return an empty DataFrame with the
    canonical schema. The canonical schema is defined by the ChipIndexRow
    dataclass; we let pandas infer dtypes from the rows (they're already
    typed via the dataclass default factories)."""
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=[f.name for f in dataclasses.fields(ChipIndexRow)])


def save_shard(df: pd.DataFrame, path: Path) -> None:
    """Crash-safe write: tmp + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False, compression="snappy")
    os.replace(tmp, path)


def rows_to_dataframe(rows: list[ChipIndexRow]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=[f.name for f in dataclasses.fields(ChipIndexRow)])
    return pd.DataFrame([dataclasses.asdict(r) for r in rows])


def already_processed_granule_ids(shard_df: pd.DataFrame) -> set[str]:
    """Set of granule_ids present in the shard. Includes negative rows
    (skip_reason set) and the sentinel-GEOID 'no county intersect' rows.

    Resumability invariant: if a granule_id appears in the shard, it has
    been fully processed and must not be downloaded again."""
    if shard_df.empty:
        return set()
    return set(shard_df["granule_id"].dropna().unique())


# -----------------------------------------------------------------------------
# CMR query + per-phase bucketing
# -----------------------------------------------------------------------------
def cmr_query_cell(
    state: str,
    year: int,
    cloud_cover_max: int,
    log: logging.Logger,
) -> list:
    """One CMR round-trip per (state, year). Queries HLSL30 + HLSS30 over
    the full growing season with the state bbox and cloud filter.

    Returns the list of earthaccess Granule objects (mixed L30 and S30).
    """
    import earthaccess

    bbox = STATE_BBOX[state]
    temporal = temporal_window_for_year(year)

    log.info(
        "CMR query: state=%s year=%d bbox=%s temporal=%s..%s cloud_cover<=%d",
        state, year, bbox, temporal[0], temporal[1], cloud_cover_max,
    )

    results = earthaccess.search_data(
        short_name=cmr_short_names_both(),
        bounding_box=bbox,
        temporal=temporal,
        cloud_cover=(0, cloud_cover_max),
    )
    log.info("CMR returned %d granules for %s-%d", len(results), state, year)
    return results


def granule_meta_from_result(g) -> dict:
    """Pull the bits we need out of an earthaccess Granule.

    earthaccess exposes the underlying CMR/STAC dict via __getitem__ (it
    inherits from `cmr.UmmGranule`). We need:
      - granule_id  (the umm.GranuleUR or the data-link basename)
      - cloud_cover (eo:cloud_cover, 0..100)

    We try the umm-G key first, then fall back to the STAC properties.
    """
    granule_id = None
    cloud_pct = None

    # Try umm-G shape: g['umm']['GranuleUR']
    try:
        umm = g["umm"]
        granule_id = umm.get("GranuleUR")
        # Cloud cover in umm-G is in AdditionalAttributes
        for aa in umm.get("AdditionalAttributes", []) or []:
            if aa.get("Name") == "CLOUD_COVERAGE":
                vals = aa.get("Values") or []
                if vals:
                    try:
                        cloud_pct = float(vals[0])
                    except (TypeError, ValueError):
                        pass
                break
    except (KeyError, TypeError):
        pass

    # STAC fallback
    if granule_id is None:
        try:
            granule_id = g.get("id") or g.get("name")
        except (AttributeError, TypeError):
            pass
    if cloud_pct is None:
        try:
            props = g.get("properties", {}) if hasattr(g, "get") else {}
            cv = props.get("eo:cloud_cover") if isinstance(props, dict) else None
            if cv is not None:
                cloud_pct = float(cv)
        except Exception:
            pass

    # Last-resort: derive granule_id from a data link basename
    if granule_id is None:
        try:
            urls = g.data_links() or []
            if urls:
                # url like .../HLS.L30.T15TVH.2018196T172224.v2.0.B04.tif
                base = os.path.basename(urls[0])
                # strip trailing .Bxx.tif / .Fmask.tif
                parts = base.split(".")
                # Find the last 'v2.0' segment, granule_id ends there
                for i, p in enumerate(parts):
                    if p == "v2" and i + 1 < len(parts) and parts[i + 1].startswith("0"):
                        granule_id = ".".join(parts[: i + 2])
                        break
        except Exception:
            pass

    return {"granule_id": granule_id, "cloud_pct": cloud_pct, "raw": g}


def bucket_by_phase(
    metas: list[dict],
    log: logging.Logger,
) -> dict[str, list[dict]]:
    """Group granule metas by calendar phase (aug1/sep1/oct1/final),
    dropping any granule outside all phase windows (e.g. early-May scenes
    that the growing-season window picks up but no phase wants).

    Each meta gets enriched with parsed sensor/scene_date/mgrs_tile in place.
    """
    buckets: dict[str, list[dict]] = defaultdict(list)
    n_unparseable = 0
    n_outside_phases = 0

    for m in metas:
        gid = m["granule_id"]
        if not gid:
            n_unparseable += 1
            continue
        try:
            parsed = parse_granule_id(gid)
        except ValueError:
            n_unparseable += 1
            continue
        m["sensor"] = parsed.sensor
        m["scene_date"] = parsed.scene_date
        m["mgrs_tile"] = parsed.mgrs_tile
        phase = label_calendar_phase(parsed.scene_date)
        if phase is None:
            n_outside_phases += 1
            continue
        m["phase"] = phase
        buckets[phase].append(m)

    if n_unparseable:
        log.warning("dropped %d granules with unparseable ids", n_unparseable)
    if n_outside_phases:
        log.info("dropped %d granules outside phase windows (early-season etc)", n_outside_phases)
    for p in PHASE_WINDOWS:
        log.info("  phase %-5s : %d granules pre-cap", p, len(buckets.get(p, [])))
    return buckets


def cap_per_phase(
    buckets: dict[str, list[dict]],
    cap: int,
    log: logging.Logger,
) -> list[dict]:
    """Per phase, keep the top `cap` granules sorted by ascending cloud_pct
    (NaN-cloud values sorted to end). Returns a flat de-duplicated list."""
    seen: set[str] = set()
    out: list[dict] = []
    for phase, metas in buckets.items():
        # NaN-cloud values sort to end so we prefer known-clean granules
        metas_sorted = sorted(
            metas,
            key=lambda m: (m["cloud_pct"] is None, m["cloud_pct"] or 0.0),
        )
        kept = metas_sorted[:cap]
        if len(metas) > cap:
            log.info("  phase %s: capped %d -> %d", phase, len(metas), cap)
        for m in kept:
            if m["granule_id"] in seen:
                continue
            seen.add(m["granule_id"])
            out.append(m)
    return out


# -----------------------------------------------------------------------------
# extract_chips stub
# -----------------------------------------------------------------------------
# Real implementation lands in scripts/extract_chips.py next turn. Until then
# we stub it: the stub emits one sentinel "negative" row per granule so the
# resumability mechanism sees the granule as processed and we can
# functionally exercise the full orchestrator (CMR query, phase bucketing,
# per-cell sharding, dry-run output) end-to-end.

def extract_chips_for_granule_stub(
    *,
    granule_meta: dict,
    state_alpha: str,
    year: int,
    cell_root: Path,  # unused in stub; will be granule-local download dir later
) -> list[ChipIndexRow]:
    """STUB. Emits one sentinel row so the granule registers as processed.
    Replace with the real implementation from scripts/extract_chips.py."""
    parsed = parse_granule_id(granule_meta["granule_id"])
    return [
        ChipIndexRow(
            GEOID=SENTINEL_GEOID_NO_INTERSECT,
            state_alpha=state_alpha,
            year=year,
            phase=granule_meta["phase"],
            scene_date=parsed.scene_date,
            granule_id=granule_meta["granule_id"],
            sensor=parsed.sensor,
            mgrs_tile=parsed.mgrs_tile,
            cmr_cloud_pct=granule_meta.get("cloud_pct"),
            chip_path=None,
            skip_reason="county_not_in_granule",  # stub-only sentinel reason
            extractor_version=EXTRACTOR_VERSION + "-stub",
        )
    ]


# Real extractor wiring point: when scripts/extract_chips.py exists we'll
# replace the call site below to pass in the downloaded granule directory,
# the counties_in_state_gdf, and the cdl_mask_path. For now we use the stub.
try:
    from scripts.extract_chips import extract_chips_for_granule  # type: ignore  # noqa: F401
    _EXTRACTOR_AVAILABLE = True
except ImportError:
    _EXTRACTOR_AVAILABLE = False


# -----------------------------------------------------------------------------
# Per-cell driver
# -----------------------------------------------------------------------------
def process_cell(
    *,
    state: str,
    year: int,
    args: argparse.Namespace,
    log: logging.Logger,
) -> None:
    cell_label = f"{state}-{year}"
    log.info("=" * 70)
    log.info("CELL %s", cell_label)
    log.info("=" * 70)

    shard_dir = Path(args.shard_dir)
    shard_path = shard_path_for(state, year, shard_dir)
    shard_df = load_shard(shard_path)
    done_ids = already_processed_granule_ids(shard_df)
    log.info("loaded shard %s: %d existing rows, %d granules already processed",
             shard_path, len(shard_df), len(done_ids))

    # 1. CMR query
    try:
        results = cmr_query_cell(state, year, args.cloud_cover_max, log)
    except Exception as e:
        log.error("CMR query failed for %s: %s", cell_label, e)
        log.error("traceback:\n%s", traceback.format_exc())
        return

    if not results:
        log.warning("no CMR results for %s", cell_label)
        return

    # 2. Pull metadata
    metas = [granule_meta_from_result(g) for g in results]

    # 3. Phase-bucket and cap
    buckets = bucket_by_phase(metas, log)
    capped = cap_per_phase(buckets, args.granule_cap, log)
    log.info("after per-phase cap: %d granules to consider", len(capped))

    # 4. Skip already-processed
    todo = [m for m in capped if m["granule_id"] not in done_ids]
    log.info("after resume-skip: %d granules to download (%d already done)",
             len(todo), len(capped) - len(todo))

    if args.max_granules is not None:
        before = len(todo)
        todo = todo[: args.max_granules]
        log.info("--max-granules %d: capped %d -> %d", args.max_granules, before, len(todo))

    if args.dry_run:
        log.info("[DRY RUN] would download %d granules:", len(todo))
        for m in todo[:20]:
            log.info("    %-7s %s  cloud=%s  phase=%s",
                     m["sensor"], m["granule_id"],
                     f"{m['cloud_pct']:.1f}" if m["cloud_pct"] is not None else "?",
                     m["phase"])
        if len(todo) > 20:
            log.info("    ... and %d more", len(todo) - 20)
        return

    if not todo:
        log.info("nothing to do for %s", cell_label)
        return

    if not _EXTRACTOR_AVAILABLE:
        log.warning(
            "scripts.extract_chips.extract_chips_for_granule is NOT importable; "
            "using STUB extractor (emits sentinel rows only, no real chips). "
            "Wire up extract_chips.py to produce real chips."
        )

    # 5. Walk granules: download -> extract -> append rows -> delete
    import earthaccess

    granule_cache = Path(args.granule_cache)
    granule_cache.mkdir(parents=True, exist_ok=True)

    new_rows: list[ChipIndexRow] = []
    n_processed = 0
    t_cell_start = time.time()

    for i, m in enumerate(todo, start=1):
        gid = m["granule_id"]
        granule_local_dir = granule_cache / gid
        granule_local_dir.mkdir(parents=True, exist_ok=True)
        t_g_start = time.time()
        try:
            paths = earthaccess.download(
                [m["raw"]], local_path=str(granule_local_dir)
            )
            log.info("[%d/%d] downloaded %s (%d files, %.1fs)",
                     i, len(todo), gid, len(paths or []), time.time() - t_g_start)
        except Exception as e:
            log.error("[%d/%d] download failed for %s: %s", i, len(todo), gid, e)
            log.error("traceback:\n%s", traceback.format_exc())
            # don't add to done set; let next run retry
            shutil.rmtree(granule_local_dir, ignore_errors=True)
            continue

        try:
            if _EXTRACTOR_AVAILABLE:
                rows = extract_chips_for_granule(
                    granule_meta=m,
                    state_alpha=state,
                    year=year,
                    granule_local_dir=granule_local_dir,
                    chip_root=Path(args.chip_root),
                )
            else:
                rows = extract_chips_for_granule_stub(
                    granule_meta=m, state_alpha=state, year=year,
                    cell_root=granule_local_dir,
                )
        except Exception as e:
            log.error("[%d/%d] extract_chips failed for %s: %s", i, len(todo), gid, e)
            log.error("traceback:\n%s", traceback.format_exc())
            # We still want to mark the granule as processed (don't infinite-
            # retry on a deterministic extraction bug). Emit a single
            # all-failure sentinel row.
            parsed = parse_granule_id(gid)
            rows = [ChipIndexRow(
                GEOID=SENTINEL_GEOID_NO_INTERSECT,
                state_alpha=state, year=year,
                phase=m["phase"], scene_date=parsed.scene_date,
                granule_id=gid, sensor=parsed.sensor, mgrs_tile=parsed.mgrs_tile,
                cmr_cloud_pct=m.get("cloud_pct"),
                chip_path=None,
                skip_reason="county_not_in_granule",
                extractor_version=EXTRACTOR_VERSION + "-error",
            )]

        new_rows.extend(rows)
        n_chips_pos = sum(1 for r in rows if r.chip_path is not None)
        log.info("[%d/%d] %s -> %d rows (%d chips written)",
                 i, len(todo), gid, len(rows), n_chips_pos)

        # delete the downloaded granule directory now that chips are saved
        shutil.rmtree(granule_local_dir, ignore_errors=True)

        n_processed += 1
        if n_processed % args.flush_every == 0:
            shard_df = pd.concat(
                [shard_df, rows_to_dataframe(new_rows)], ignore_index=True
            )
            save_shard(shard_df, shard_path)
            log.info("checkpoint: flushed %d new rows -> %s (%d total)",
                     len(new_rows), shard_path, len(shard_df))
            new_rows = []

    # final flush
    if new_rows:
        shard_df = pd.concat(
            [shard_df, rows_to_dataframe(new_rows)], ignore_index=True
        )
        save_shard(shard_df, shard_path)
        log.info("final flush: %d new rows -> %s (%d total)",
                 len(new_rows), shard_path, len(shard_df))

    log.info("CELL %s done in %.1fs (%d granules processed)",
             cell_label, time.time() - t_cell_start, n_processed)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    grid_grp = ap.add_argument_group("grid selection (use one)")
    grid_grp.add_argument("--cells", nargs="+",
                          help="explicit cells, e.g. IA-2018 IA-2019 NE-2020")
    grid_grp.add_argument("--state", choices=STATES_DEFAULT,
                          help="all years for this state")
    grid_grp.add_argument("--year", type=int, choices=YEARS_DEFAULT,
                          help="all states for this year")
    grid_grp.add_argument("--all", action="store_true",
                          help="full grid (5 states x 12 years)")

    ap.add_argument("--shard", type=str, default=None,
                    help="shard K/N for parallel terminals, e.g. --shard 1/4")

    ap.add_argument("--max-granules", type=int, default=None,
                    help="hard cap on per-cell downloads; smoke testing")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve CMR queries and bucket granules; do NOT download")

    ap.add_argument("--cloud-cover-max", type=int, default=CMR_CLOUD_COVER_MAX_DEFAULT,
                    help=f"CMR-side cloud cover max %% (default: {CMR_CLOUD_COVER_MAX_DEFAULT})")
    ap.add_argument("--granule-cap", type=int, default=GRANULE_CAP_PER_PHASE_DEFAULT,
                    help=f"per-phase top-N cleanest cap (default: {GRANULE_CAP_PER_PHASE_DEFAULT})")
    ap.add_argument("--flush-every", type=int, default=FLUSH_EVERY_N_GRANULES_DEFAULT,
                    help=f"flush shard parquet every N granules (default: {FLUSH_EVERY_N_GRANULES_DEFAULT})")

    ap.add_argument("--shard-dir", default=CHIP_INDEX_SHARD_DIR_DEFAULT,
                    help=f"per-cell shard parquet dir (default: {CHIP_INDEX_SHARD_DIR_DEFAULT})")
    ap.add_argument("--chip-root", default=CHIP_ROOT_DEFAULT,
                    help=f"chip output root (default: {CHIP_ROOT_DEFAULT})")
    ap.add_argument("--granule-cache", default=GRANULE_CACHE_DIR_DEFAULT,
                    help=f"granule download cache (default: {GRANULE_CACHE_DIR_DEFAULT})")
    ap.add_argument("--log-dir", default="runs",
                    help="directory for log files (default: runs/)")

    args = ap.parse_args()

    # selection sanity
    if not (args.cells or args.state or args.year or args.all):
        ap.error("must specify one of: --cells, --state, --year, --all")
    if args.shard and not args.all:
        ap.error("--shard requires --all (sharding the full grid only)")

    # logging setup
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    shard_tag = args.shard.replace("/", "of") if args.shard else "single"
    log_path = Path(args.log_dir) / f"download_hls_{shard_tag}_{ts}.log"
    log = setup_logging(log_path)
    log.info("download_hls.py starting; args=%s", vars(args))
    log.info("log file: %s", log_path)
    log.info("repo root: %s", _REPO_ROOT)

    # earthaccess auth + GDAL config
    if not args.dry_run:
        configure_gdal_for_cloud_native_hls()
        log.info("GDAL configured for cloud-native HLS")

    import earthaccess
    auth = earthaccess.login(strategy="netrc")
    if not auth or not auth.authenticated:
        raise SystemExit(
            "Earthdata authentication failed. Verify ~/.netrc has machine "
            "urs.earthdata.nasa.gov credentials."
        )
    log.info("Earthdata authenticated as %s", getattr(auth, "username", "?"))

    cells = resolve_grid(args)
    log.info("resolved grid: %d cells", len(cells))
    for s, y in cells:
        log.info("    %s-%d", s, y)

    if not args.dry_run:
        Path(args.chip_root).mkdir(parents=True, exist_ok=True)
        Path(args.shard_dir).mkdir(parents=True, exist_ok=True)
        Path(args.granule_cache).mkdir(parents=True, exist_ok=True)

    t_run_start = time.time()
    for s, y in cells:
        try:
            process_cell(state=s, year=y, args=args, log=log)
        except KeyboardInterrupt:
            log.warning("interrupted by user; partial shard saved")
            raise
        except Exception as e:
            log.error("cell %s-%d crashed (continuing to next): %s", s, y, e)
            log.error("traceback:\n%s", traceback.format_exc())

    log.info("=" * 70)
    log.info("ALL CELLS DONE in %.1f minutes", (time.time() - t_run_start) / 60.0)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
