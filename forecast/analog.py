"""
forecast.analog — nearest-neighbor analog retrieval over the standardized embedding.

Phase B baseline. Builds one K-NN index per forecast_date over the train pool's
standardized embedding (cross-county candidate pool by default; same-GEOID
sanity baseline available via pool='same_geoid'). At query time, the query's
own year is excluded post-search to enforce the no-temporal-leakage rule.

Yields are returned **detrended** — the cone-construction layer (cone.py) takes
percentiles over detrended yields and the aggregation layer retrends at
(query_state, query_year). Keeping the detrending out of this module means the
analog retrieval is purely about weather/soil/management similarity, which is
exactly what the brief calls for.

Public surface:
    Analog                                  dataclass
    AnalogIndex                             holds per-date BallTrees + row tables
    AnalogIndex.fit(train_df, std, trend)   builds indices
    AnalogIndex.find(geoid, year, date,
                     k=10, pool='cross_county') -> list[Analog]
    AnalogIndex.find_for_query_row(row, ...)    convenience for batch backtest

Performance notes:
    BallTree query is O(log n) per neighbor. For ~7,000 train rows per date,
    each query takes <1 ms. The full backtest (~12k queries) is seconds, not
    minutes — index build dominates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from forecast.detrend import StateTrend
from forecast.features import (
    EMBEDDING_COLS,
    Standardizer,
    VALID_FORECAST_DATES,
    build_embedding_matrix,
)


PoolStrategy = Literal["cross_county", "same_geoid"]


# -----------------------------------------------------------------------------
# Analog record
# -----------------------------------------------------------------------------


@dataclass
class Analog:
    """One analog match returned from the K-NN search.

    Attributes
    ----------
    geoid : str
        5-digit FIPS of the analog county.
    year : int
        Calendar year of the analog (always != query year).
    state_alpha : str
        State of the analog (used by cone.py for retrending — analogs are
        retrended *back through their own state's trend* before percentile
        construction; the cone is then in detrended space).
    distance : float
        Euclidean distance in standardized embedding space. Smaller = closer.
    observed_yield : float
        Raw bu/acre yield of the analog. NASS-reported, post-harvest.
    detrended_yield : float
        observed_yield minus the analog's own (state, year) trend prediction.
    forecast_date : str
        Always equals the query's forecast_date (the index is per-date).
    """

    geoid: str
    year: int
    state_alpha: str
    distance: float
    observed_yield: float
    detrended_yield: float
    forecast_date: str


# -----------------------------------------------------------------------------
# Index
# -----------------------------------------------------------------------------


@dataclass
class _DateIndex:
    """Per-forecast-date index data. Internal."""

    tree: BallTree
    matrix: np.ndarray              # (n_train_rows, n_features)
    rows: pd.DataFrame              # [GEOID, year, state_alpha, yield_target,
                                    #  detrended_yield_target] aligned to matrix rows
    max_year_population: int        # max number of rows sharing a single year (for K' headroom)


class AnalogIndex:
    """Per-forecast-date K-NN indices over the train pool's standardized embedding.

    Use AnalogIndex.fit(...) to construct from a train_df that has already been
    sliced through forecast.data.train_pool(...).
    """

    def __init__(self) -> None:
        self._indices: Dict[str, _DateIndex] = {}
        self._standardizer: Optional[Standardizer] = None
        self._trend: Optional[StateTrend] = None

    @classmethod
    def fit(
        cls,
        train_df: pd.DataFrame,
        standardizer: Standardizer,
        trend: StateTrend,
    ) -> "AnalogIndex":
        """Build per-date BallTree indices over the train pool.

        `train_df` must already be the post-min-history, post-completeness
        train slice (use forecast.data.train_pool). It must have a
        `state_alpha` column (used downstream by Analog records).
        """
        required = ["GEOID", "year", "state_alpha", "forecast_date", "yield_target"]
        missing = [c for c in required if c not in train_df.columns]
        if missing:
            raise KeyError(f"train_df missing required columns: {missing}")

        # Detrend once up-front so analog yields are immediately available.
        train_df = trend.detrend(train_df)
        if train_df["detrended_yield_target"].isna().any():
            n_null = int(train_df["detrended_yield_target"].isna().sum())
            raise ValueError(
                f"Train pool has {n_null} null detrended yields after detrending. "
                f"Likely a state in the data is missing from the StateTrend fit."
            )

        # Build the standardized matrices per date (this also validates no nulls
        # in embedding columns).
        per_date = build_embedding_matrix(train_df, standardizer)

        idx = cls()
        idx._standardizer = standardizer
        idx._trend = trend

        for date, (matrix, row_index) in per_date.items():
            # row_index from build_embedding_matrix has [GEOID, year, forecast_date,
            # yield_target]. We need state_alpha and detrended_yield_target too —
            # join those back from train_df.
            sub = train_df[train_df["forecast_date"] == date].reset_index(drop=True)
            # Sanity: row_index built from sub directly, so order matches.
            assert len(sub) == len(row_index), (
                f"Row count mismatch building index for date={date!r}"
            )
            rows = pd.DataFrame(
                {
                    "GEOID": sub["GEOID"].values,
                    "year": sub["year"].values.astype(int),
                    "state_alpha": sub["state_alpha"].values,
                    "yield_target": sub["yield_target"].values.astype(float),
                    "detrended_yield_target": sub["detrended_yield_target"]
                    .values.astype(float),
                }
            )

            tree = BallTree(matrix, metric="euclidean")
            max_year_pop = int(rows.groupby("year").size().max())

            idx._indices[date] = _DateIndex(
                tree=tree,
                matrix=matrix,
                rows=rows,
                max_year_population=max_year_pop,
            )

        return idx

    # -- query API ------------------------------------------------------------

    def find(
        self,
        geoid: str,
        year: int,
        forecast_date: str,
        query_features: pd.Series | pd.DataFrame,
        k: int = 10,
        pool: PoolStrategy = "cross_county",
    ) -> List[Analog]:
        """K-nearest analogs for one query.

        Parameters
        ----------
        geoid : str
            5-digit FIPS of the query county. Used for the same_geoid pool
            constraint and (via state lookup) for retrending downstream.
        year : int
            Query year. Excluded from the analog pool (no temporal leakage).
        forecast_date : str
            Selects which per-date index to query.
        query_features : Series or 1-row DataFrame
            The query row from the master table. Must contain all embedding
            columns for `forecast_date`. Will be standardized internally using
            the same Standardizer used at fit time.
        k : int
            Number of analogs to return. Default 10.
        pool : 'cross_county' | 'same_geoid'
            'cross_county': any analog (geoid_h, year_h) with year_h != year.
            'same_geoid': require geoid_h == geoid (within-county history).
        """
        if forecast_date not in self._indices:
            raise KeyError(
                f"No index for forecast_date={forecast_date!r}. "
                f"Fit dates: {sorted(self._indices.keys())}"
            )
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        date_idx = self._indices[forecast_date]
        cols = EMBEDDING_COLS[forecast_date]

        # Coerce query to a single-row DataFrame for the standardizer.
        if isinstance(query_features, pd.Series):
            query_df = query_features.to_frame().T
        else:
            if len(query_features) != 1:
                raise ValueError(
                    f"query_features must be 1 row, got {len(query_features)}"
                )
            query_df = query_features.copy()

        missing = [c for c in cols if c not in query_df.columns]
        if missing:
            raise KeyError(f"query_features missing embedding columns: {missing}")
        # Ensure correct types — Series.to_frame() can yield object dtype.
        for c in cols:
            query_df[c] = pd.to_numeric(query_df[c], errors="raise")

        assert self._standardizer is not None, "AnalogIndex not fit"
        query_vec = self._standardizer.transform(query_df, forecast_date)
        if query_vec.shape[0] != 1:
            raise ValueError(
                f"Standardized query has shape {query_vec.shape}, expected (1, n_features)."
            )

        # K' = headroom for post-filter exclusion. Cross-county worst case: all
        # neighbors share the query year, so we need K + max_year_population.
        # Same-GEOID worst case: 1 row per year for the query GEOID.
        n_train = date_idx.matrix.shape[0]
        if pool == "cross_county":
            k_query = min(n_train, k + date_idx.max_year_population)
        elif pool == "same_geoid":
            # Restrict to same-GEOID rows by post-filtering; but if those are
            # very few, we still need the broader search to have enough hits.
            k_query = min(n_train, k + n_train)  # = n_train; just take everything
        else:
            raise ValueError(f"Unknown pool strategy: {pool!r}")

        distances, indices = date_idx.tree.query(query_vec, k=k_query)
        distances = distances[0]
        indices = indices[0]

        rows = date_idx.rows
        analogs: List[Analog] = []
        for d, i in zip(distances, indices):
            cand = rows.iloc[int(i)]
            cand_geoid = str(cand["GEOID"])
            cand_year = int(cand["year"])

            # Year exclusion (always enforced).
            if cand_year == year:
                continue

            # Pool constraint.
            if pool == "same_geoid" and cand_geoid != geoid:
                continue

            analogs.append(
                Analog(
                    geoid=cand_geoid,
                    year=cand_year,
                    state_alpha=str(cand["state_alpha"]),
                    distance=float(d),
                    observed_yield=float(cand["yield_target"]),
                    detrended_yield=float(cand["detrended_yield_target"]),
                    forecast_date=forecast_date,
                )
            )
            if len(analogs) >= k:
                break

        if len(analogs) < k:
            # We exhausted the search radius without getting K. For cross_county
            # this should be impossible given the headroom calculation; for
            # same_geoid it's normal when a county has < K-many non-query years.
            if pool == "cross_county":
                raise RuntimeError(
                    f"Cross-county search exhausted before K={k} analogs found "
                    f"(got {len(analogs)}). This indicates the K' headroom "
                    f"calculation is wrong — please report. "
                    f"query=({geoid}, {year}, {forecast_date}), n_train={n_train}, "
                    f"k_query={k_query}."
                )
            # Same-GEOID: silently return fewer than K. Caller handles.

        return analogs

    def find_for_query_row(
        self,
        query_row: pd.Series,
        k: int = 10,
        pool: PoolStrategy = "cross_county",
    ) -> List[Analog]:
        """Convenience: pull geoid/year/forecast_date out of `query_row` and
        delegate to find()."""
        return self.find(
            geoid=str(query_row["GEOID"]),
            year=int(query_row["year"]),
            forecast_date=str(query_row["forecast_date"]),
            query_features=query_row,
            k=k,
            pool=pool,
        )

    # -- introspection --------------------------------------------------------

    def n_candidates(self, forecast_date: str) -> int:
        """Number of train-pool rows in the index for the given date."""
        return int(self._indices[forecast_date].matrix.shape[0])

    def fit_dates(self) -> List[str]:
        return sorted(self._indices.keys())
