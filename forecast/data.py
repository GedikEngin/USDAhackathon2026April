"""
forecast.data — master table loading, train/val/holdout slicing, min-history filter.

Single source of truth for "how do I get a clean DataFrame for downstream code".
Centralises the conventions documented in PHASE2_DATA_DICTIONARY and
PHASE2_DECISIONS_LOG so analog.py / backtest_baseline.py / Phase C training
all see the same view of the data.

Public surface:
    load_master(path=None)                  -> pd.DataFrame  (defensive: GEOID padded)
    apply_min_history_filter(df, n=10)      -> (df, kept_geoids, dropped_geoids)
    SPLIT_YEARS                             -> dict with train/val/holdout year tuples
    train_pool(df, n_min_history=10)        -> filtered training rows
    val_pool(df)                            -> 2023 rows
    holdout_pool(df)                        -> 2024 rows

Conventions (locked):
    - Train years:     2005-2022
    - Val year:        2023
    - Holdout year:    2024
    - Min-history filter: a county is a *candidate* in the analog pool if it has
      >= n complete-data train years across all 4 forecast_dates. Counties with
      <n years can still be queried (we forecast for them) but never appear as
      analogs.
    - "Complete data" = all 17 retrieval embedding columns non-null at the
      relevant forecast_date (so the 08-01 contract is the 15-col version).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from forecast.features import EMBEDDING_COLS, VALID_FORECAST_DATES


# -----------------------------------------------------------------------------
# Split definitions (locked per PHASE2_PHASE_PLAN C.1)
# -----------------------------------------------------------------------------

SPLIT_YEARS = {
    "train": tuple(range(2005, 2023)),  # 2005-2022 inclusive
    "val": (2023,),
    "holdout": (2024,),
}

DEFAULT_MASTER_PATH = "scripts/training_master.parquet"


# -----------------------------------------------------------------------------
# Loader
# -----------------------------------------------------------------------------


def load_master(path: str | Path | None = None) -> pd.DataFrame:
    """Load `training_master.parquet` and apply defensive normalization.

    - GEOID coerced to 5-char zero-padded string (parquet preserves str, but the
      data dictionary explicitly recommends a defensive pad in case a downstream
      consumer round-trips through int).
    - forecast_date validated against VALID_FORECAST_DATES.
    - state_alpha validated against the 5-state set.

    Does NOT filter by year or by min-history; those are separate ops.
    """
    if path is None:
        path = DEFAULT_MASTER_PATH
    df = pd.read_parquet(path)

    # Required columns sanity check
    required = {"GEOID", "year", "forecast_date", "state_alpha", "yield_target"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Master table missing required columns: {sorted(missing)}")

    # GEOID defensive pad
    df["GEOID"] = df["GEOID"].astype(str).str.zfill(5)

    # forecast_date validation
    bad_dates = set(df["forecast_date"].unique()) - set(VALID_FORECAST_DATES)
    if bad_dates:
        raise ValueError(
            f"Master table has unexpected forecast_date values: {sorted(bad_dates)}. "
            f"Expected subset of {VALID_FORECAST_DATES}."
        )

    # state_alpha validation
    expected_states = {"CO", "IA", "MO", "NE", "WI"}
    bad_states = set(df["state_alpha"].unique()) - expected_states
    if bad_states:
        raise ValueError(
            f"Master table has unexpected state_alpha values: {sorted(bad_states)}. "
            f"Expected subset of {sorted(expected_states)}."
        )

    return df


# -----------------------------------------------------------------------------
# Min-history filter
# -----------------------------------------------------------------------------


@dataclass
class MinHistoryFilterResult:
    """Diagnostic record from apply_min_history_filter."""

    n_min: int
    kept_geoids: List[str]
    dropped_geoids: List[str]
    geoid_year_counts: pd.Series  # per-GEOID count of qualifying training years

    @property
    def n_kept(self) -> int:
        return len(self.kept_geoids)

    @property
    def n_dropped(self) -> int:
        return len(self.dropped_geoids)


def _row_has_complete_embedding(df: pd.DataFrame) -> pd.Series:
    """Per-row boolean: True iff this row's embedding columns are all non-null
    given its forecast_date.

    This is the per-date contract used elsewhere — at 08-01 the 15 non-grain cols
    must be non-null; at 09-01/10-01/EOS the full 17 must be non-null.
    """
    # Build a mask per forecast_date and union them.
    out = pd.Series(False, index=df.index)
    for date in VALID_FORECAST_DATES:
        cols = EMBEDDING_COLS[date]
        date_mask = df["forecast_date"] == date
        if not date_mask.any():
            continue
        # Sub-frame for this date
        sub = df.loc[date_mask, cols]
        complete = sub.notna().all(axis=1)
        out.loc[date_mask] = complete
    return out


def apply_min_history_filter(
    df: pd.DataFrame,
    n: int = 10,
    *,
    train_years: Tuple[int, ...] = SPLIT_YEARS["train"],
) -> Tuple[pd.DataFrame, MinHistoryFilterResult]:
    """Drop counties with fewer than `n` qualifying training years from the
    analog candidate pool.

    A "qualifying" training year for a county = a year in `train_years` where
    *all 4 forecast_dates* have complete embedding data (per the per-date
    column contract) AND yield_target is non-null.

    Returns
    -------
    filtered_df : DataFrame
        All rows of `df` whose GEOID survives the filter. NOTE: this includes
        non-train years (val 2023, holdout 2024) for those GEOIDs — the filter
        decides who's a *candidate*, not who's queryable. (Counties dropped
        from candidacy can still be queried; the caller decides.)
    result : MinHistoryFilterResult
        Diagnostic with the per-GEOID year count and the kept/dropped lists.
    """
    if "GEOID" not in df.columns or "year" not in df.columns:
        raise KeyError("df must have GEOID and year columns")

    train_mask = df["year"].isin(train_years)
    train_df = df.loc[train_mask].copy()

    # Per-row completeness of the embedding (per the per-date contract).
    embedding_complete = _row_has_complete_embedding(train_df)
    target_complete = train_df["yield_target"].notna()

    # A (GEOID, year) is "qualifying" iff *all 4 forecast_dates* are complete
    # in both embedding and target.
    train_df["_qualifies"] = embedding_complete & target_complete

    per_gy = (
        train_df.groupby(["GEOID", "year"])["_qualifies"]
        .all()  # all 4 forecast_dates must qualify
        .reset_index()
    )
    qualifying_years_per_geoid = (
        per_gy[per_gy["_qualifies"]]
        .groupby("GEOID")
        .size()
        .rename("n_qualifying_train_years")
    )

    # GEOIDs absent from the count series have 0 qualifying years.
    all_geoids = pd.Index(df["GEOID"].unique(), name="GEOID")
    counts = qualifying_years_per_geoid.reindex(all_geoids, fill_value=0)

    kept = counts[counts >= n].index.tolist()
    dropped = counts[counts < n].index.tolist()

    filtered_df = df[df["GEOID"].isin(kept)].copy()
    result = MinHistoryFilterResult(
        n_min=n,
        kept_geoids=sorted(kept),
        dropped_geoids=sorted(dropped),
        geoid_year_counts=counts.sort_values(ascending=False),
    )
    return filtered_df, result


# -----------------------------------------------------------------------------
# Pool helpers (semantic sugar over year filters)
# -----------------------------------------------------------------------------


def train_pool(
    df: pd.DataFrame, *, n_min_history: int = 10
) -> Tuple[pd.DataFrame, MinHistoryFilterResult]:
    """Slice `df` to the training pool: 2005-2022, post-min-history, complete embedding.

    Returned rows are guaranteed to have:
      - year in 2005-2022
      - GEOID with >= n_min_history qualifying train years
      - the row's own embedding complete (per per-date contract)
      - yield_target non-null

    These are the rows safe to use as analog candidates and as fit data for
    the standardizer / state trend.
    """
    # First filter to candidate GEOIDs (using the full master table so the
    # candidacy decision sees all years).
    filtered, result = apply_min_history_filter(df, n=n_min_history)

    # Now slice to train years and require row-level completeness.
    train_mask = filtered["year"].isin(SPLIT_YEARS["train"])
    train_df = filtered.loc[train_mask].copy()

    embedding_complete = _row_has_complete_embedding(train_df)
    target_complete = train_df["yield_target"].notna()
    train_df = train_df.loc[embedding_complete & target_complete].reset_index(drop=True)

    return train_df, result


def val_pool(df: pd.DataFrame) -> pd.DataFrame:
    """2023 rows. No min-history filter — we forecast every queryable county.
    Caller is responsible for any per-row embedding-completeness check."""
    return df[df["year"].isin(SPLIT_YEARS["val"])].copy().reset_index(drop=True)


def holdout_pool(df: pd.DataFrame) -> pd.DataFrame:
    """2024 rows. Same caveats as val_pool. Touched once at end of Phase G."""
    return df[df["year"].isin(SPLIT_YEARS["holdout"])].copy().reset_index(drop=True)
