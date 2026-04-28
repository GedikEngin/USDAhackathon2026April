#!/usr/bin/env python
"""
scripts/merge_chip_index.py
Concatenate per-cell shards into the canonical chip index.

Inputs:
    data/v2/hls/chip_index_shards/chip_index_<state>_<year>.parquet   (one per cell)

Output:
    data/v2/hls/chip_index.parquet   (the canonical index; consumed by D.1.d's chip-picker)

Behavior:
    - Idempotent: always rewrites the canonical file from current shards.
    - Safe to run on partial shards (i.e. while download_hls.py is still running).
      The output reflects whatever's been flushed to disk so far.
    - Validates uniqueness of (granule_id, GEOID) -- duplicates are a bug, fail loudly.
    - Reports per-state / per-year / per-phase coverage stats.
    - Optionally verifies chip GeoTIFFs exist on disk (--verify-chips).

Exit codes:
    0  success (merged + validated)
    1  uniqueness violation or other validation failure
    2  no shards on disk

Usage:
    python scripts/merge_chip_index.py
    python scripts/merge_chip_index.py --verify-chips    # check chip files exist
    python scripts/merge_chip_index.py --out alt.parquet # alternate output path
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from forecast.hls_common import (  # noqa: E402
    CHIP_INDEX_PATH_DEFAULT,
    CHIP_INDEX_SHARD_DIR_DEFAULT,
    PHASE_TO_FORECAST_DATE,
    SENTINEL_GEOID_NO_INTERSECT,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def section(title: str) -> None:
    print()
    print("=" * 72)
    print(f" {title}")
    print("=" * 72)


def load_all_shards(shard_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    """Read every chip_index_<state>_<year>.parquet under shard_dir and concat."""
    shard_paths = sorted(shard_dir.glob("chip_index_*.parquet"))
    if not shard_paths:
        return pd.DataFrame(), []

    frames = []
    for p in shard_paths:
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            print(f"  ! failed to read {p}: {e}; skipping")
            continue
        # Tag the source for downstream debugging
        df["_shard_source"] = p.name
        frames.append(df)
    if not frames:
        return pd.DataFrame(), []
    out = pd.concat(frames, ignore_index=True)
    return out, shard_paths


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of validation-error strings. Empty list = clean."""
    errors: list[str] = []

    # 1. (granule_id, GEOID) uniqueness.
    # The orchestrator emits at most one row per (granule, county) pair.
    # Sentinel rows (GEOID == "00000", "county_not_in_granule") are exempt:
    # cross-state-border MGRS tiles (e.g. T15TYE on the IA/MO line) get
    # included in BOTH states' CMR queries and both shards write an
    # identical sentinel — that's expected, not a bug.
    dup_mask = df.duplicated(subset=["granule_id", "GEOID"], keep=False)
    sentinel_dup = dup_mask & (df["GEOID"] == SENTINEL_GEOID_NO_INTERSECT)
    real_dup = dup_mask & ~sentinel_dup
    if real_dup.any():
        n_dups = int(real_dup.sum())
        sample = df.loc[real_dup, ["granule_id", "GEOID", "_shard_source"]].head(10)
        errors.append(
            f"duplicate (granule_id, GEOID) rows: {n_dups}\n"
            f"  sample:\n{sample.to_string(index=False)}"
        )

    # 2. Positive rows must have all chip_* fields populated.
    pos = df["chip_path"].notna()
    pos_with_missing = (
        pos
        & (df["corn_pixel_count"].isna()
           | df["corn_pixel_frac"].isna()
           | df["valid_pixel_frac"].isna())
    )
    if pos_with_missing.any():
        n = int(pos_with_missing.sum())
        errors.append(f"positive rows missing QC fields: {n}")

    # 3. Negative rows must have skip_reason set.
    neg = df["chip_path"].isna()
    neg_no_reason = neg & df["skip_reason"].isna()
    if neg_no_reason.any():
        n = int(neg_no_reason.sum())
        errors.append(f"negative rows with no skip_reason: {n}")

    # 4. GEOID must be 5-char string (allow sentinel '00000')
    bad_geoid = df["GEOID"].astype(str).str.len() != 5
    if bad_geoid.any():
        n = int(bad_geoid.sum())
        sample = df.loc[bad_geoid, "GEOID"].head(5).tolist()
        errors.append(f"non-5-char GEOIDs: {n}; sample: {sample}")

    return errors


# -----------------------------------------------------------------------------
# Coverage stats
# -----------------------------------------------------------------------------
def report_coverage(df: pd.DataFrame) -> None:
    section("OVERALL")
    n_total = len(df)
    n_pos   = int(df["chip_path"].notna().sum())
    n_neg   = int(df["chip_path"].isna().sum())
    n_sentinel = int((df["GEOID"] == SENTINEL_GEOID_NO_INTERSECT).sum())
    n_real_neg = n_neg - n_sentinel
    n_granules = int(df["granule_id"].nunique())

    print(f"  rows total:                   {n_total:>10,}")
    print(f"  positive rows (chip written): {n_pos:>10,}  ({100*n_pos/max(n_total,1):.1f}%)")
    print(f"  negative rows (real county):  {n_real_neg:>10,}  ({100*n_real_neg/max(n_total,1):.1f}%)")
    print(f"  sentinel rows (no intersect): {n_sentinel:>10,}  ({100*n_sentinel/max(n_total,1):.1f}%)")
    print(f"  unique granules processed:    {n_granules:>10,}")
    print(f"  unique GEOIDs (incl sentinel):{df['GEOID'].nunique():>10,}")

    section("BY STATE × YEAR (chips written)")
    pos_df = df[df["chip_path"].notna()].copy()
    if len(pos_df) == 0:
        print("  no positive rows yet")
        return
    pivot = pos_df.pivot_table(
        index="state_alpha", columns="year", values="chip_path",
        aggfunc="count", fill_value=0,
    )
    print(pivot.to_string())

    section("BY STATE × PHASE (chips written)")
    pivot2 = pos_df.pivot_table(
        index="state_alpha", columns="phase", values="chip_path",
        aggfunc="count", fill_value=0,
    )
    # Order phases sensibly
    phase_order = ["aug1", "sep1", "oct1", "final"]
    pivot2 = pivot2.reindex(columns=[c for c in phase_order if c in pivot2.columns])
    print(pivot2.to_string())

    section("SKIP-REASON BREAKDOWN")
    print(df["skip_reason"].value_counts(dropna=False).to_string())

    section("CHIP FRACTION (corn / valid pixel) STATS — positive rows")
    if len(pos_df) > 0:
        for c in ("corn_pixel_frac", "valid_pixel_frac"):
            print(f"  {c}:")
            s = pos_df[c]
            print(f"    min={s.min():.3f}  q25={s.quantile(0.25):.3f}  "
                  f"median={s.median():.3f}  q75={s.quantile(0.75):.3f}  max={s.max():.3f}")

    # Coverage of the (GEOID, year, forecast_date) grid we ultimately need
    # at D.1.d. Each row's phase maps to a forecast_date; the chip-picker
    # uses this mapping to assign chips to forecast queries.
    section("COVERAGE OF (GEOID × YEAR × FORECAST_DATE) — at least one chip")
    pos_df["forecast_date"] = pos_df["phase"].map(PHASE_TO_FORECAST_DATE)
    # Note: aug1 -> 08-01, sep1 -> 09-01, oct1 -> 10-01, final -> EOS
    # but the actual chip-picker logic has more nuance; this is just a
    # rough coverage report.
    coverage = pos_df.groupby(["GEOID", "year", "forecast_date"]).size().reset_index(name="n_chips")
    n_unique_cells = len(coverage)
    print(f"  unique (GEOID, year, forecast_date) cells with >=1 chip: {n_unique_cells:>8,}")
    print(f"  median chips per cell: {coverage['n_chips'].median():.1f}")
    print(f"  min/max chips per cell: {coverage['n_chips'].min()} / {coverage['n_chips'].max()}")


def maybe_verify_chips(df: pd.DataFrame) -> None:
    """Walk every positive row's chip_path and check the file exists."""
    section("CHIP-FILE EXISTENCE CHECK")
    pos = df[df["chip_path"].notna()]
    if len(pos) == 0:
        print("  no positive rows to check")
        return
    paths = pos["chip_path"].astype(str).tolist()
    n_total = len(paths)
    n_missing = 0
    missing_sample = []
    for p in paths:
        if not os.path.exists(p):
            n_missing += 1
            if len(missing_sample) < 5:
                missing_sample.append(p)
    if n_missing == 0:
        print(f"  all {n_total:,} chip files present on disk ✓")
    else:
        print(f"  MISSING: {n_missing:,} of {n_total:,} chip files ({100*n_missing/n_total:.2f}%)")
        for p in missing_sample:
            print(f"    {p}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--shard-dir", default=CHIP_INDEX_SHARD_DIR_DEFAULT,
                    help=f"per-cell shard dir (default: {CHIP_INDEX_SHARD_DIR_DEFAULT})")
    ap.add_argument("--out", default=CHIP_INDEX_PATH_DEFAULT,
                    help=f"output parquet path (default: {CHIP_INDEX_PATH_DEFAULT})")
    ap.add_argument("--verify-chips", action="store_true",
                    help="verify each positive row's chip file exists on disk")
    ap.add_argument("--no-write", action="store_true",
                    help="don't write the merged parquet; only print stats")
    args = ap.parse_args()

    shard_dir = Path(args.shard_dir)
    out_path = Path(args.out)

    if not shard_dir.exists():
        print(f"shard dir does not exist: {shard_dir}", file=sys.stderr)
        sys.exit(2)

    print(f"reading shards from {shard_dir}/")
    df, shard_paths = load_all_shards(shard_dir)
    if df.empty:
        print("no shards found (empty dir or all reads failed)", file=sys.stderr)
        sys.exit(2)

    print(f"  loaded {len(shard_paths)} shard files")
    print(f"  concat: {len(df):,} rows total")

    # Drop the _shard_source bookkeeping column before validation/output;
    # it's only useful for debugging.
    df_clean = df.drop(columns=["_shard_source"])

    # Dedupe sentinel rows (GEOID == "00000") that appear in multiple state
    # shards because a cross-state-border MGRS tile (e.g. T15TYE on the IA/MO
    # line) gets returned by both states' CMR queries and both shards write an
    # identical "county_not_in_granule" sentinel. Real chip rows are NOT
    # touched — those have unique (granule_id, GEOID).
    sentinel_mask = df_clean["GEOID"] == SENTINEL_GEOID_NO_INTERSECT
    n_sentinel_before = int(sentinel_mask.sum())
    sent_df = (
        df_clean[sentinel_mask]
        .drop_duplicates(subset=["granule_id", "GEOID"], keep="first")
    )
    df_clean = pd.concat([df_clean[~sentinel_mask], sent_df], ignore_index=True)
    n_sentinel_after = int((df_clean["GEOID"] == SENTINEL_GEOID_NO_INTERSECT).sum())
    n_dropped = n_sentinel_before - n_sentinel_after
    if n_dropped > 0:
        print(f"  deduped {n_dropped} cross-state sentinel rows")

    # Validate
    section("VALIDATION")
    errors = validate(df)  # use df with _shard_source for debugging
    if errors:
        print("  ! VALIDATION ERRORS:")
        for e in errors:
            print(f"    - {e}")
        print()
        print("  refusing to write canonical index until clean.")
        sys.exit(1)
    print("  all validation checks passed ✓")

    # Coverage stats
    report_coverage(df_clean)

    # Optional file-existence verification
    if args.verify_chips:
        maybe_verify_chips(df_clean)

    # Write merged parquet
    if args.no_write:
        section("OUTPUT (skipped due to --no-write)")
        return
    section("WRITING CANONICAL INDEX")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df_clean.to_parquet(tmp, index=False, compression="snappy")
    os.replace(tmp, out_path)
    size_mb = out_path.stat().st_size / 1e6
    print(f"  wrote {out_path}  ({size_mb:.2f} MB, {len(df_clean):,} rows)")


if __name__ == "__main__":
    main()
