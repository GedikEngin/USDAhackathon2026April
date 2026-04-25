"""
forecast.features — retrieval-embedding column lists and per-date standardization.

Phase B baseline. The retrieval embedding is intentionally narrow (17 features at
09-01/10-01/EOS, 15 at 08-01) and per-forecast-date because cumulative weather
distributions are wildly different at Aug 1 vs. EOS. See PHASE2_DECISIONS_LOG
'Phase B kickoff' entry for the why on each column.

Public surface:
    EMBEDDING_COLS                       dict[str, list[str]]   per-forecast_date column lists
    fit_standardizer(df)                 -> Standardizer
    Standardizer.transform(df)           -> np.ndarray (n_rows, n_features) per date
    Standardizer.save(path) / load(path)
    build_embedding_matrix(df, std)      -> dict[str, (matrix, row_index_df)]

Notes:
- Standardization scope is the **training pool only** (2005-2022, post min-history filter).
  The caller (analog.py / backtest_baseline.py) is responsible for slicing before fit.
- We do NOT impute. *_grain features are excluded from the 08-01 column list entirely
  rather than imputed, per the per-date embedding decision.
- All other embedding columns have 0 nulls per the data dictionary; if a future
  feature run produces a null in a non-grain column, fit_standardizer raises.
- Embedding dtype is float32 (sufficient precision; halves memory of the index).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Column groups (documented for traceability; the union is what gets standardized)
# -----------------------------------------------------------------------------

# Weather features available at every forecast_date (cumulative + pre-grain phase).
# 0 nulls in master table per data dictionary.
_WEATHER_ALWAYS = [
    "gdd_cum_f50_c86",       # cumulative GDD, base 50 / cap 86 F
    "edd_hours_gt86f",       # heat-stress degree-hours
    "vpd_kpa_veg",           # mean VPD during vegetative window (DOY 152-195)
    "vpd_kpa_silk",          # mean VPD during silking (DOY 196-227) — top yield driver
    "prcp_cum_mm",           # cumulative precip May 1 -> cutoff
    "dry_spell_max_days",    # longest dry run May 1 -> cutoff
    "srad_total_veg",        # solar radiation, vegetative
    "srad_total_silk",       # solar radiation, silking
]

# Weather features only available once grain fill has started (DOY 228+).
# Structurally NaN at 08-01; included at 09-01 / 10-01 / EOS.
_WEATHER_GRAIN = [
    "vpd_kpa_grain",
    "srad_total_grain",
]

# USDM drought, state-broadcast. d0_pct = drought-of-any-kind breadth,
# d2plus = severe-or-worse severity. Other D-tiers (d1/d3/d4) excluded as
# collinear-monotone with d2plus.
_DROUGHT = [
    "d0_pct",
    "d2plus",
]

# MODIS NDVI, county-level, whole-season summary (same value at all 4 dates within a year).
# Weak as-of fidelity is documented; ships as-is for v2 baseline, replaced by HLS
# running-NDVI in Phase D.1.
_NDVI = [
    "ndvi_gs_mean",
    "ndvi_peak",
]

# gSSURGO soil features, static per-GEOID (broadcast across all years and forecast_dates).
_SOIL = [
    "nccpi3corn",   # corn productivity index (0-1) — primary
    "aws0_100",     # plant-available water 0-100 cm (mm)
    "rootznaws",    # root-zone available water (mm) — best single soil-water summary
]

# Management — only irrigated_share enters the embedding. Other NASS-aux columns
# (acres_*, harvest_ratio) are post-hoc reported (same value across forecast_dates)
# and treated as covariates / structural priors, not in-season retrieval signal.
_MANAGEMENT = [
    "irrigated_share",
]


def _embedding_cols_for_date(forecast_date: str) -> List[str]:
    """Column list for the given forecast_date. Order is stable and is the
    canonical column ordering for the embedding matrix."""
    base = _WEATHER_ALWAYS + _DROUGHT + _NDVI + _SOIL + _MANAGEMENT
    if forecast_date == "08-01":
        return base
    # At 09-01, 10-01, EOS: grain-phase features are populated (cutoff has passed
    # the grain-fill window start of DOY 228 on Aug 16).
    return _WEATHER_ALWAYS + _WEATHER_GRAIN + _DROUGHT + _NDVI + _SOIL + _MANAGEMENT


VALID_FORECAST_DATES = ("08-01", "09-01", "10-01", "EOS")

EMBEDDING_COLS: Dict[str, List[str]] = {
    d: _embedding_cols_for_date(d) for d in VALID_FORECAST_DATES
}


# -----------------------------------------------------------------------------
# Standardizer
# -----------------------------------------------------------------------------


@dataclass
class Standardizer:
    """Per-(forecast_date, feature) mean and std, fit on the training pool.

    Attributes
    ----------
    means : dict[forecast_date, dict[feature, float]]
    stds  : dict[forecast_date, dict[feature, float]]
    n_train_rows : dict[forecast_date, int]   for sanity/logging
    """

    means: Dict[str, Dict[str, float]] = field(default_factory=dict)
    stds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    n_train_rows: Dict[str, int] = field(default_factory=dict)

    def transform(self, df: pd.DataFrame, forecast_date: str) -> np.ndarray:
        """Z-score the embedding columns for `forecast_date` rows in `df`.

        The caller must pass rows already filtered to a single forecast_date.
        Returns a (n_rows, n_features) float32 array in the canonical column
        order from EMBEDDING_COLS[forecast_date].
        """
        if forecast_date not in self.means:
            raise KeyError(
                f"Standardizer not fit for forecast_date={forecast_date!r}. "
                f"Fit dates: {sorted(self.means.keys())}"
            )
        cols = EMBEDDING_COLS[forecast_date]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing embedding columns in df: {missing}")

        out = np.empty((len(df), len(cols)), dtype=np.float32)
        for j, col in enumerate(cols):
            mu = self.means[forecast_date][col]
            sd = self.stds[forecast_date][col]
            vals = df[col].to_numpy(dtype=np.float64)
            if np.isnan(vals).any():
                n_null = int(np.isnan(vals).sum())
                raise ValueError(
                    f"Null values in {col!r} at forecast_date={forecast_date!r} "
                    f"({n_null} rows). Embedding columns must be fully populated "
                    f"after grain-column exclusion at 08-01."
                )
            out[:, j] = ((vals - mu) / sd).astype(np.float32)
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "means": self.means,
            "stds": self.stds,
            "n_train_rows": self.n_train_rows,
            "embedding_cols": {d: EMBEDDING_COLS[d] for d in self.means},
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Standardizer":
        payload = json.loads(Path(path).read_text())
        return cls(
            means=payload["means"],
            stds=payload["stds"],
            n_train_rows=payload.get("n_train_rows", {}),
        )


def fit_standardizer(train_df: pd.DataFrame) -> Standardizer:
    """Fit per-(forecast_date, feature) mean and std on the training pool.

    Parameters
    ----------
    train_df : DataFrame
        Rows from the master table with `forecast_date` ∈ VALID_FORECAST_DATES.
        Caller must already have applied the train-year filter (e.g. year ≤ 2022)
        and the per-county min-history filter.

    Returns
    -------
    Standardizer
    """
    if "forecast_date" not in train_df.columns:
        raise KeyError("train_df missing forecast_date column")

    means: Dict[str, Dict[str, float]] = {}
    stds: Dict[str, Dict[str, float]] = {}
    n_rows: Dict[str, int] = {}

    for date in VALID_FORECAST_DATES:
        sub = train_df[train_df["forecast_date"] == date]
        if len(sub) == 0:
            raise ValueError(
                f"No training rows at forecast_date={date!r}. "
                f"Check the train-pool filter."
            )
        cols = EMBEDDING_COLS[date]

        date_means: Dict[str, float] = {}
        date_stds: Dict[str, float] = {}
        for col in cols:
            if col not in sub.columns:
                raise KeyError(f"Train df missing embedding column {col!r}")
            vals = sub[col].to_numpy(dtype=np.float64)
            if np.isnan(vals).any():
                n_null = int(np.isnan(vals).sum())
                raise ValueError(
                    f"Train pool has {n_null} null(s) in {col!r} at "
                    f"forecast_date={date!r}. The embedding contract requires "
                    f"complete data for all listed columns; null counts must "
                    f"be addressed in the min-history filter or in column selection."
                )
            mu = float(vals.mean())
            sd = float(vals.std(ddof=0))
            if sd == 0.0:
                raise ValueError(
                    f"Zero std for {col!r} at forecast_date={date!r}. "
                    f"Constant feature — drop from EMBEDDING_COLS."
                )
            date_means[col] = mu
            date_stds[col] = sd

        means[date] = date_means
        stds[date] = date_stds
        n_rows[date] = int(len(sub))

    return Standardizer(means=means, stds=stds, n_train_rows=n_rows)


# -----------------------------------------------------------------------------
# Embedding matrix builder
# -----------------------------------------------------------------------------


def build_embedding_matrix(
    df: pd.DataFrame, standardizer: Standardizer
) -> Dict[str, Tuple[np.ndarray, pd.DataFrame]]:
    """Standardize all rows in `df`, sliced by forecast_date.

    Returns one (matrix, row_index) pair per forecast_date present in df:
        matrix     : float32 array, shape (n_rows_at_date, n_features_at_date)
        row_index  : DataFrame with columns [GEOID, year, forecast_date, yield_target]
                     in the same row order as `matrix`. yield_target may be NaN
                     for query rows (e.g. holdout 2024 at 08-01 before harvest).

    The caller (analog.py) uses this to build per-date K-NN indices and to look
    up analog yields by row position.
    """
    required_index_cols = ["GEOID", "year", "forecast_date", "yield_target"]
    missing = [c for c in required_index_cols if c not in df.columns]
    if missing:
        raise KeyError(f"df missing required index columns: {missing}")

    out: Dict[str, Tuple[np.ndarray, pd.DataFrame]] = {}
    for date in df["forecast_date"].unique():
        if date not in VALID_FORECAST_DATES:
            raise ValueError(
                f"Unexpected forecast_date in df: {date!r}. "
                f"Must be one of {VALID_FORECAST_DATES}."
            )
        sub = df[df["forecast_date"] == date].copy()
        sub = sub.reset_index(drop=True)
        matrix = standardizer.transform(sub, date)
        row_index = sub[required_index_cols].copy()
        out[date] = (matrix, row_index)
    return out
