#!/usr/bin/env python
"""
scripts/extract_embeddings.py
Phase 2-D.1.d orchestrator: produce data/v2/prithvi/embeddings_v1.parquet
from chip_index.parquet + the trained Prithvi-EO-2.0-300M-TL backbone.

Pipeline (per (GEOID, year, forecast_date) query):
  1. Generate the query list from scripts/training_master.parquet, filtered
     to 2013-2024 (HLS only exists 2013+, per decisions log D.1.kickoff).
  2. For each query, call forecast.chip_picker.pick_chips() to select T=3
     chips obeying the as-of rule. Empty queries (no chips) are emitted
     with NaN embeddings.
  3. Batch real queries to the PrithviEmbedder; queries with chips load
     their 3 GeoTIFFs, get the model output, and the embedding is
     mean-pooled to a 1024-d vector.
  4. Write output parquet keyed on (GEOID, year, forecast_date) with:
        - 3 keys
        - 4 QC features (chip_count, chip_age_days_max, cloud_pct_max,
          corn_pixel_frac_min)  -- decisions log: become regressor features
        - model_version          -- baked in for ablation traceability
        - 1024 embedding columns prithvi_0000..prithvi_1023

Resumability:
  - Output parquet is keyed on (GEOID, year, forecast_date, model_version)
    where model_version is constant per run.
  - On startup, if the output exists, we read it and skip queries that
    are already present AND have a non-NaN embedding (or chip_count == 0,
    indicating they were correctly emitted as NaN).
  - This means re-running after the HLS pull finishes will only re-embed
    queries whose chip_count just went from 0 -> >0 (i.e. cells that
    finished pulling chips since the last run).
  - --rebuild flag: ignore the existing parquet, embed everything.

Usage:
  # Full run (resumable). Run after the HLS pull is substantially complete.
  python scripts/extract_embeddings.py

  # Test on a single year for a single state
  python scripts/extract_embeddings.py --states IA --years 2018

  # Force full rebuild (no resume)
  python scripts/extract_embeddings.py --rebuild

  # Dry-run: report query counts and chip availability, but skip inference
  python scripts/extract_embeddings.py --dry-run

  # Custom batch size (default 8 per probe; reduce if OOM)
  python scripts/extract_embeddings.py --batch-size 4
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# We import lazily where torch/terratorch are needed, to keep --dry-run fast
from forecast.chip_picker import (  # noqa: E402
    ChipQuery,
    VALID_FORECAST_DATES,
    pick_chips,
)
from forecast.hls_common import (  # noqa: E402
    CHIP_INDEX_PATH_DEFAULT,
)


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_MASTER_PATH    = "scripts/training_master.parquet"
DEFAULT_CHIP_INDEX     = CHIP_INDEX_PATH_DEFAULT  # 'data/v2/hls/chip_index.parquet'
DEFAULT_OUTPUT_PATH    = "data/v2/prithvi/embeddings_v1.parquet"
DEFAULT_LOG_DIR        = "runs"

# HLS coverage: 2013+ per decisions log D.1.kickoff. (HLS v2.0 archive
# starts 2013, no chips will exist for earlier years.)
HLS_FIRST_YEAR = 2013
HLS_LAST_YEAR  = 2024  # inclusive; covers train (2013-2022) + val (2023) + holdout (2024)


def setup_logging(log_path: Optional[Path]) -> logging.Logger:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("extract_embeddings")


# =============================================================================
# Query generation
# =============================================================================

def generate_queries(
    master_df: pd.DataFrame,
    states: Optional[list[str]] = None,
    years: Optional[list[int]] = None,
    forecast_dates: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Build the query DataFrame from training_master.parquet.

    Columns: GEOID (str, 5-char), state_alpha (str, 2-char), year (int),
             forecast_date (str in {'08-01','09-01','10-01','EOS'}).

    Filters:
      - year in [HLS_FIRST_YEAR, HLS_LAST_YEAR]
      - state in `states` if given
      - year in `years` if given
      - forecast_date in `forecast_dates` if given
    """
    df = master_df.copy()
    df["GEOID"] = df["GEOID"].astype(str).str.zfill(5)
    if "state_alpha" not in df.columns:
        raise KeyError("master table missing state_alpha column")
    df["state_alpha"] = df["state_alpha"].astype(str)
    df["year"] = df["year"].astype(int)
    df["forecast_date"] = df["forecast_date"].astype(str)

    df = df[(df["year"] >= HLS_FIRST_YEAR) & (df["year"] <= HLS_LAST_YEAR)]

    if states:
        df = df[df["state_alpha"].isin(states)]
    if years:
        df = df[df["year"].isin(years)]
    if forecast_dates:
        df = df[df["forecast_date"].isin(forecast_dates)]

    # Dedupe on the query keys -- master table has 1 row per query already,
    # but defensive
    df = df[["GEOID", "state_alpha", "year", "forecast_date"]].drop_duplicates()
    df = df.sort_values(["state_alpha", "year", "GEOID", "forecast_date"]).reset_index(drop=True)
    return df


# =============================================================================
# Output schema
# =============================================================================

def build_output_row(result, model_version: str, embedding_dim: int,
                     embedding_col_names: list[str]) -> dict:
    """Convert one EmbeddingResult to a flat dict suitable for parquet."""
    row = {
        "GEOID":         result.GEOID,
        "year":          int(result.year),
        "forecast_date": result.forecast_date,
        "model_version": model_version,
        "chip_count":         result.chip_count,
        "chip_age_days_max":  result.chip_age_days_max,
        "cloud_pct_max":      result.cloud_pct_max,
        "corn_pixel_frac_min": result.corn_pixel_frac_min,
    }
    if result.embedding is None:
        # NaN embedding for queries with no chips. XGBoost handles via
        # missing-direction split (per decisions log).
        for col in embedding_col_names:
            row[col] = np.nan
    else:
        emb = result.embedding
        for i, col in enumerate(embedding_col_names):
            row[col] = float(emb[i])
    return row


def empty_output_df(embedding_col_names: list[str]) -> pd.DataFrame:
    """Return an empty DataFrame with the canonical schema columns set."""
    cols = (
        ["GEOID", "year", "forecast_date", "model_version",
         "chip_count", "chip_age_days_max", "cloud_pct_max", "corn_pixel_frac_min"]
        + embedding_col_names
    )
    return pd.DataFrame({c: [] for c in cols})


def crashsafe_write(df: pd.DataFrame, path: Path) -> None:
    """Atomic write: tmp + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False, compression="snappy")
    os.replace(tmp, path)


# =============================================================================
# Resume logic
# =============================================================================

def load_existing_output(path: Path, model_version: str,
                         log: logging.Logger) -> pd.DataFrame:
    """Read the existing embeddings parquet, filter to current model_version.
    Returns empty DataFrame if file doesn't exist or schema mismatch."""
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        log.warning("could not read existing %s: %s; starting fresh", path, e)
        return pd.DataFrame()
    if "model_version" not in df.columns:
        log.warning("existing %s has no model_version column; starting fresh", path)
        return pd.DataFrame()
    df = df[df["model_version"] == model_version].copy()
    log.info("loaded %d existing rows (model_version=%s) from %s",
             len(df), model_version, path)
    return df


def already_done_keys(existing_df: pd.DataFrame) -> set:
    """Set of (GEOID, year, forecast_date) tuples already embedded.

    A row is 'done' if it's in the parquet AND either:
      - has a non-NaN embedding, OR
      - has chip_count == 0 (correctly emitted as NaN; nothing to redo
        unless the chip_index gains new chips, in which case we'd rerun
        the picker and chip_count would change)

    For now we treat any row in the parquet as done -- the simple
    interpretation. To force re-embedding of NaN rows (e.g. after pulls
    add chips), use --rebuild or pass --redo-empty.
    """
    if existing_df.empty:
        return set()
    keys = set(zip(
        existing_df["GEOID"].astype(str),
        existing_df["year"].astype(int),
        existing_df["forecast_date"].astype(str),
    ))
    return keys


# =============================================================================
# Main pipeline
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--master", default=DEFAULT_MASTER_PATH,
                    help=f"path to training_master.parquet (default: {DEFAULT_MASTER_PATH})")
    ap.add_argument("--chip-index", default=DEFAULT_CHIP_INDEX,
                    help=f"path to chip_index.parquet (default: {DEFAULT_CHIP_INDEX})")
    ap.add_argument("--out", default=DEFAULT_OUTPUT_PATH,
                    help=f"output embeddings parquet (default: {DEFAULT_OUTPUT_PATH})")
    ap.add_argument("--log-dir", default=DEFAULT_LOG_DIR)

    ap.add_argument("--states", nargs="+", default=None,
                    help="restrict to these states (default: all 5)")
    ap.add_argument("--years", type=int, nargs="+", default=None,
                    help="restrict to these years (default: 2013-2024)")
    ap.add_argument("--forecast-dates", nargs="+", default=None,
                    choices=list(VALID_FORECAST_DATES),
                    help="restrict to these forecast dates (default: all 4)")

    ap.add_argument("--batch-size", type=int, default=None,
                    help="Prithvi inference batch size (default from prithvi.py)")
    ap.add_argument("--device", default=None, choices=("cuda", "cpu"),
                    help="device override (default: auto)")
    ap.add_argument("--rebuild", action="store_true",
                    help="ignore existing output; embed everything")
    ap.add_argument("--redo-empty", action="store_true",
                    help="re-embed queries currently emitted as NaN (after chip pull "
                         "adds new chips, those queries' chip_count may have changed)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report query/chip counts; skip inference and output write")
    ap.add_argument("--flush-every", type=int, default=500,
                    help="flush output parquet every N processed queries (default: 500)")
    ap.add_argument("--max-queries", type=int, default=None,
                    help="cap total queries processed (smoke testing)")
    args = ap.parse_args()

    # Logging
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.log_dir) / f"extract_embeddings_{ts}.log"
    log = setup_logging(log_path)
    log.info("extract_embeddings.py starting; args=%s", vars(args))
    log.info("log file: %s", log_path)

    # Load master + chip_index
    master_path = Path(args.master)
    if not master_path.exists():
        raise SystemExit(f"master table not found: {master_path}")
    log.info("loading master table from %s", master_path)
    master_df = pd.read_parquet(master_path)
    log.info("  master: %d rows, %d cols", len(master_df), len(master_df.columns))

    chip_idx_path = Path(args.chip_index)
    if not chip_idx_path.exists():
        raise SystemExit(
            f"chip index not found: {chip_idx_path}\n"
            f"Run scripts/merge_chip_index.py first."
        )
    log.info("loading chip index from %s", chip_idx_path)
    chip_idx = pd.read_parquet(chip_idx_path)
    log.info("  chip_index: %d rows, %d positive, %d negative",
             len(chip_idx),
             int(chip_idx["chip_path"].notna().sum()),
             int(chip_idx["chip_path"].isna().sum()))

    # Generate queries
    queries_df = generate_queries(
        master_df,
        states=args.states,
        years=args.years,
        forecast_dates=args.forecast_dates,
    )
    log.info("generated %d queries", len(queries_df))

    if args.max_queries and args.max_queries < len(queries_df):
        queries_df = queries_df.head(args.max_queries)
        log.info("--max-queries %d: capped to %d", args.max_queries, len(queries_df))

    # Lazy import prithvi (heavy: torch + terratorch + cuda init)
    from forecast.prithvi import (
        EMBEDDING_COL_NAMES,
        EMBEDDING_DIM,
        MODEL_VERSION,
        PrithviEmbedder,
    )

    # Resume: load existing output, build done-set
    out_path = Path(args.out)
    if args.rebuild:
        log.info("--rebuild: ignoring existing output (will overwrite)")
        existing_df = pd.DataFrame()
    else:
        existing_df = load_existing_output(out_path, MODEL_VERSION, log)

    if args.redo_empty and not existing_df.empty:
        # Drop existing rows where chip_count == 0 so they re-process
        n_before = len(existing_df)
        existing_df = existing_df[existing_df["chip_count"] != 0].copy()
        log.info("--redo-empty: dropped %d zero-chip rows from existing output",
                 n_before - len(existing_df))

    done_keys = already_done_keys(existing_df)

    # Filter queries to those not already done
    if done_keys:
        before = len(queries_df)
        keys_in_queries = list(zip(
            queries_df["GEOID"].astype(str),
            queries_df["year"].astype(int),
            queries_df["forecast_date"].astype(str),
        ))
        mask = [k not in done_keys for k in keys_in_queries]
        queries_df = queries_df[mask].reset_index(drop=True)
        log.info("resume: skipped %d already-done queries; %d remaining",
                 before - len(queries_df), len(queries_df))

    # ======= Pre-flight: chip availability summary =======
    log.info("running chip-picker preflight on all queries...")
    t0 = time.monotonic()
    chip_queries: list[ChipQuery] = []
    for _, row in queries_df.iterrows():
        cq = pick_chips(
            chip_idx,
            row["GEOID"], int(row["year"]), row["forecast_date"],
        )
        chip_queries.append(cq)
    log.info("preflight: %d ChipQuery objects in %.1fs",
             len(chip_queries), time.monotonic() - t0)

    # Coverage report
    real_cnt = sum(1 for q in chip_queries if q.chip_count > 0)
    nan_cnt  = sum(1 for q in chip_queries if q.chip_count == 0)
    if chip_queries:
        cnt_dist = pd.Series([q.chip_count for q in chip_queries]).value_counts().sort_index()
        log.info("chip_count distribution:")
        for k, v in cnt_dist.items():
            log.info("    chip_count=%d : %d queries", k, v)
    log.info("  real-chip queries: %d  (will go to Prithvi)", real_cnt)
    log.info("  empty queries:     %d  (will be NaN-emitted)", nan_cnt)

    if args.dry_run:
        log.info("[DRY RUN] skipping inference and output write")
        return

    if not chip_queries:
        log.info("nothing to embed; exiting")
        return

    # ======= Build embedder =======
    embedder_kwargs = {}
    if args.batch_size is not None:
        embedder_kwargs["batch_size"] = args.batch_size
    if args.device is not None:
        embedder_kwargs["device"] = args.device

    log.info("instantiating PrithviEmbedder %s", embedder_kwargs)
    embedder = PrithviEmbedder(**embedder_kwargs)

    # ======= Run embedding extraction in chunks with periodic flush =======
    log.info("running embedding extraction; flush every %d queries", args.flush_every)
    out_rows_buffer: list[dict] = []
    n_done = 0
    t_start = time.monotonic()

    # Process in chunks of flush_every so we can periodically write to disk
    # without ever loading all chip tensors at once
    chunk_size = max(args.flush_every, embedder.batch_size * 4)
    for chunk_start in range(0, len(chip_queries), chunk_size):
        chunk = chip_queries[chunk_start: chunk_start + chunk_size]
        log.info("processing chunk %d-%d (%d queries)",
                 chunk_start, chunk_start + len(chunk) - 1, len(chunk))

        results = embedder.embed_query_sequences(chunk, chip_root="")

        for r in results:
            out_rows_buffer.append(build_output_row(
                r, MODEL_VERSION, EMBEDDING_DIM, EMBEDDING_COL_NAMES,
            ))
        n_done += len(results)
        elapsed = time.monotonic() - t_start
        rate = n_done / max(elapsed, 1e-9)
        eta_min = (len(chip_queries) - n_done) / max(rate, 1e-9) / 60
        log.info("  done %d/%d  %.1f q/s  ETA %.1f min",
                 n_done, len(chip_queries), rate, eta_min)

        # Flush
        new_df = pd.DataFrame(out_rows_buffer)
        merged = pd.concat([existing_df, new_df], ignore_index=True) if not existing_df.empty else new_df
        # Dedupe keeping last (in case --rebuild + a partial flush)
        merged = merged.drop_duplicates(
            subset=["GEOID", "year", "forecast_date", "model_version"],
            keep="last",
        )
        crashsafe_write(merged, out_path)
        size_mb = out_path.stat().st_size / 1e6
        log.info("  flushed -> %s (%.1f MB, %d rows)",
                 out_path, size_mb, len(merged))

    log.info("=" * 70)
    log.info("DONE: %d queries embedded in %.1f minutes",
             n_done, (time.monotonic() - t_start) / 60)
    log.info("output: %s", out_path)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
