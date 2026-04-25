"""
forecast.regressor — per-forecast-date XGBoost point-estimate model.

Phase C baseline. Trains one gradient-boosted regressor per forecast_date
(08-01, 09-01, 10-01, EOS) on the full Phase A master table, predicting raw
yield_target in bu/acre. The Phase B analog cone is kept untouched; only the
point estimate swaps from analog-median to this regressor's prediction.

Design decisions (locked in Phase 2-C kickoff, see PHASE2_DECISIONS_LOG):
  - Train target: raw yield_target. `year` is included as a feature so the
    trees can learn nonlinear time effects (WI plateau).
  - Feature set: full superset (~30 numeric) + state one-hot + year.
    Wider than the analog retrieval embedding by design — the embedding is
    intentionally narrow for L2-distance interpretability; the regressor
    can absorb covariates the embedding holds back.
  - One model per forecast_date. At 08-01, the structurally-NaN grain
    features (vpd_kpa_grain, srad_total_grain) are dropped from the feature
    list rather than imputed.
  - irrigated_share + an `is_irrigated_reported` indicator replace
    `yield_bu_acre_irr` (84% structural null).
  - acres_harvested_all is dropped: collinear with planted * harvest_ratio
    and post-hoc reported (would feel like target leakage at 08-01).
  - Sample weights: none. Every county-year is a real observation; state
    aggregation does the acres weighting at evaluation time.

Public surface:
    FEATURE_COLS                          dict[forecast_date, list[str]]
    DEFAULT_PARAMS                        baseline xgb params (pre-sweep)
    Regressor                             dataclass — one trained booster + metadata
    RegressorBundle                       four Regressors keyed by forecast_date
    fit(train_df, val_df, params=None)    -> Regressor (single date)
    fit_all_dates(train_df, val_df, ...)  -> RegressorBundle
    Regressor.predict(df)                 -> np.ndarray
    RegressorBundle.predict(df)           -> np.ndarray (auto per-date)
    Regressor.save / load
    RegressorBundle.save / load

Notes
-----
- The module does NOT do hyperparameter sweeps internally. The driver script
  (scripts/train_regressor.py) is responsible for the {max_depth, lr,
  min_child_weight} sweep. fit() takes a single param dict and returns a
  single Regressor; the driver constructs the sweep and selects.
- Training uses XGBoost's native xgb.train (not the sklearn wrapper) for
  early-stopping symmetry with the JSON save/load path.
- DMatrix construction is encapsulated in _build_dmatrix; if we ever switch
  to native categorical handling for state_alpha (xgboost >= 1.5) we change
  it in one place.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from forecast.features import VALID_FORECAST_DATES


# -----------------------------------------------------------------------------
# Feature column groups (canonical ordering — do not reorder casually; the
# saved booster's feature_names are derived from this and reordering breaks
# old models).
# -----------------------------------------------------------------------------

# Weather features available at every forecast_date.
_WEATHER_ALWAYS = [
    "gdd_cum_f50_c86",
    "edd_hours_gt86f",
    "edd_hours_gt90f",        # held back from embedding; included here
    "vpd_kpa_veg",
    "vpd_kpa_silk",
    "prcp_cum_mm",
    "dry_spell_max_days",
    "srad_total_veg",
    "srad_total_silk",
]

# Grain-phase weather (structurally NaN at 08-01; included at 09-01+).
_WEATHER_GRAIN = [
    "vpd_kpa_grain",
    "srad_total_grain",
]

# USDM drought, full set.
_DROUGHT = [
    "d0_pct",
    "d1_pct",
    "d2_pct",
    "d3_pct",
    "d4_pct",
    # NOTE: d2plus is omitted — it equals d2_pct exactly (alias from
    # drought_features.py); including both adds zero information and creates
    # spurious redundancy in SHAP attributions.
]

# MODIS NDVI — full set (whole-season summary, broadcast across forecast_dates).
_NDVI = [
    "ndvi_peak",
    "ndvi_gs_mean",
    "ndvi_gs_integral",
    "ndvi_silking_mean",
    "ndvi_veg_mean",
]

# gSSURGO soil — full set (static per-GEOID).
_SOIL = [
    "nccpi3corn",
    "nccpi3all",
    "aws0_100",
    "aws0_150",
    "soc0_30",
    "soc0_100",
    "rootznemc",
    "rootznaws",
    "droughty",
    "pctearthmc",
    "pwsl1pomu",
]

# Management — irrigated_share + a presence indicator. acres_harvested_all is
# dropped (collinear with planted * harvest_ratio and post-hoc reported —
# would feel like target leakage at 08-01).
_MANAGEMENT = [
    "irrigated_share",
    "is_irrigated_reported",   # 1 iff yield_bu_acre_irr was non-null pre-impute
    "harvest_ratio",
    "acres_planted_all",
]

# Time. State enters via one-hots built at DMatrix-construction time.
_TIME = ["year"]

# State one-hots — column NAMES are stable; the values are 0/1 floats.
# Order matches alphabetical state_alpha so a saved model's feature_names
# round-trip is deterministic.
STATE_ALPHAS = ("CO", "IA", "MO", "NE", "WI")
_STATE_ONEHOT = [f"state_is_{s}" for s in STATE_ALPHAS]


def _feature_cols_for_date(forecast_date: str) -> List[str]:
    """Canonical feature column list for the given forecast_date.

    Order is the saved booster's feature_names — once a model is trained
    and saved, this order is locked. Adding a feature later means appending,
    not inserting.
    """
    base_weather = _WEATHER_ALWAYS.copy()
    if forecast_date != "08-01":
        base_weather += _WEATHER_GRAIN
    return (
        base_weather
        + _DROUGHT
        + _NDVI
        + _SOIL
        + _MANAGEMENT
        + _TIME
        + _STATE_ONEHOT
    )


FEATURE_COLS: Dict[str, List[str]] = {
    d: _feature_cols_for_date(d) for d in VALID_FORECAST_DATES
}


# -----------------------------------------------------------------------------
# DMatrix construction — single source of truth
# -----------------------------------------------------------------------------


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the columns the regressor expects but that don't exist in the
    master table as-is.

    Adds:
      - is_irrigated_reported : 1.0 if yield_bu_acre_irr.notna() else 0.0
      - state_is_<S> for S in STATE_ALPHAS : one-hot of state_alpha

    Non-destructive: returns a new df. If the columns already exist, leaves
    them alone (idempotent — caller can pre-add them if desired).
    """
    out = df.copy()

    if "is_irrigated_reported" not in out.columns:
        if "yield_bu_acre_irr" not in out.columns:
            raise KeyError(
                "Cannot derive is_irrigated_reported: yield_bu_acre_irr column "
                "missing. Pass the raw master table or add the indicator column "
                "upstream."
            )
        out["is_irrigated_reported"] = out["yield_bu_acre_irr"].notna().astype(np.float32)

    for s in STATE_ALPHAS:
        col = f"state_is_{s}"
        if col not in out.columns:
            out[col] = (out["state_alpha"] == s).astype(np.float32)

    return out


def _build_dmatrix(
    df: pd.DataFrame,
    forecast_date: str,
    *,
    include_label: bool,
) -> Tuple[xgb.DMatrix, np.ndarray]:
    """Construct a DMatrix for one forecast_date's slice.

    Parameters
    ----------
    df : DataFrame
        Rows already filtered to a single forecast_date. Must include all
        feature columns for that date plus state_alpha (for one-hot derivation)
        and yield_target (if include_label).
    forecast_date : str
        One of VALID_FORECAST_DATES.
    include_label : bool
        If True, attach yield_target as the DMatrix label and assert no nulls.

    Returns
    -------
    (DMatrix, row_kept_mask)
        row_kept_mask is a boolean array indexed against the input df telling
        the caller which rows survived feature-completeness checks. Rows
        with any NaN in the feature columns are dropped (XGBoost handles
        NaN natively at predict time, but for training we want clean data
        and the master table is supposed to be NaN-free for these columns
        per the data dictionary).
    """
    if forecast_date not in VALID_FORECAST_DATES:
        raise ValueError(
            f"Unexpected forecast_date={forecast_date!r}; "
            f"must be one of {VALID_FORECAST_DATES}"
        )

    enriched = _add_derived_columns(df)
    cols = FEATURE_COLS[forecast_date]
    missing = [c for c in cols if c not in enriched.columns]
    if missing:
        raise KeyError(
            f"DataFrame missing required feature columns for {forecast_date!r}: "
            f"{missing}"
        )

    feat_block = enriched[cols].to_numpy(dtype=np.float32)
    # Dictionary contract: all listed feature columns are NaN-free in the
    # master table (per PHASE2_DATA_DICTIONARY 'should never be NaN' table,
    # plus the per-date grain-column exclusion). If a NaN slips in, drop the
    # row and warn — fail closed at training time.
    row_complete = ~np.isnan(feat_block).any(axis=1)
    n_dropped = int((~row_complete).sum())
    if n_dropped > 0:
        # Don't print in module code — caller logs. Just drop.
        feat_block = feat_block[row_complete]
        enriched = enriched.iloc[row_complete].reset_index(drop=True)

    if include_label:
        if "yield_target" not in enriched.columns:
            raise KeyError("yield_target missing — cannot attach label")
        label = enriched["yield_target"].to_numpy(dtype=np.float32)
        if np.isnan(label).any():
            n_null = int(np.isnan(label).sum())
            raise ValueError(
                f"yield_target has {n_null} null(s) at forecast_date={forecast_date!r}. "
                f"Filter upstream — training labels must be complete."
            )
        dm = xgb.DMatrix(feat_block, label=label, feature_names=cols)
    else:
        dm = xgb.DMatrix(feat_block, feature_names=cols)

    return dm, row_complete


# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------


# Defaults the driver overrides during a sweep. The values here represent a
# reasonable mid-point of the {max_depth, lr, min_child_weight} grid we plan to
# sweep over and are NOT necessarily the gate-passing config.
DEFAULT_PARAMS: Dict[str, object] = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",      # fast, deterministic with seed
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_child_weight": 5,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "seed": 42,
    "verbosity": 0,
}

DEFAULT_NUM_BOOST_ROUND = 2000
DEFAULT_EARLY_STOPPING_ROUNDS = 50


# -----------------------------------------------------------------------------
# Regressor — one trained booster + metadata
# -----------------------------------------------------------------------------


@dataclass
class Regressor:
    """One trained per-forecast-date XGBoost booster.

    Attributes
    ----------
    forecast_date : str
        Which forecast_date this booster was trained on.
    booster : xgb.Booster
        The trained model. None until fit() returns.
    feature_cols : list[str]
        Locked at fit time — the order matches booster.feature_names. Used
        by predict() to enforce column ordering and presence.
    params : dict
        The xgb params used at fit (post-merge with DEFAULT_PARAMS).
    best_iteration : int
        From early stopping. predict() uses this iteration count.
    train_metrics : dict[str, float]
        Final-iteration train and val RMSE.
    n_train : int
        Training row count after NaN-row drop.
    n_val : int
        Val row count after NaN-row drop.
    """

    forecast_date: str
    booster: Optional[xgb.Booster] = None
    feature_cols: List[str] = field(default_factory=list)
    params: Dict[str, object] = field(default_factory=dict)
    best_iteration: int = 0
    train_metrics: Dict[str, float] = field(default_factory=dict)
    n_train: int = 0
    n_val: int = 0

    # ---- prediction ---------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict yield_target for rows in `df` at this booster's forecast_date.

        The caller is responsible for filtering `df` to a single forecast_date
        — predict() does NOT auto-filter. Use RegressorBundle.predict() if
        you have mixed-date rows.

        Returns
        -------
        np.ndarray
            Shape (n_rows,). Predictions in raw bu/acre.
        """
        if self.booster is None:
            raise RuntimeError("Regressor not fit (booster is None).")
        if "forecast_date" in df.columns and not (df["forecast_date"] == self.forecast_date).all():
            raise ValueError(
                f"predict() called with mixed forecast_dates; this Regressor "
                f"is bound to {self.forecast_date!r}. Use RegressorBundle for "
                f"mixed-date rows."
            )
        dm, _ = _build_dmatrix(df, self.forecast_date, include_label=False)
        return self.booster.predict(
            dm, iteration_range=(0, self.best_iteration + 1)
        )

    # ---- serialization ------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write booster (XGBoost JSON) + metadata sidecar (.meta.json).

        Two files: <path> for the booster, <path>.meta.json for the metadata.
        Both are required to load.
        """
        if self.booster is None:
            raise RuntimeError("Regressor not fit; cannot save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path))
        meta = {
            "forecast_date": self.forecast_date,
            "feature_cols": self.feature_cols,
            "params": self.params,
            "best_iteration": self.best_iteration,
            "train_metrics": self.train_metrics,
            "n_train": self.n_train,
            "n_val": self.n_val,
        }
        path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Regressor":
        path = Path(path)
        booster = xgb.Booster()
        booster.load_model(str(path))
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Metadata sidecar {meta_path} missing. "
                f"Save and load are paired; cannot reconstruct Regressor "
                f"from booster alone."
            )
        meta = json.loads(meta_path.read_text())
        return cls(
            forecast_date=meta["forecast_date"],
            booster=booster,
            feature_cols=meta["feature_cols"],
            params=meta["params"],
            best_iteration=meta["best_iteration"],
            train_metrics=meta.get("train_metrics", {}),
            n_train=meta.get("n_train", 0),
            n_val=meta.get("n_val", 0),
        )


# -----------------------------------------------------------------------------
# Fit (single date)
# -----------------------------------------------------------------------------


def fit(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    forecast_date: str,
    *,
    params: Optional[Dict[str, object]] = None,
    num_boost_round: int = DEFAULT_NUM_BOOST_ROUND,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    verbose_eval: bool | int = False,
) -> Regressor:
    """Train one XGBoost booster on the rows for `forecast_date`.

    Parameters
    ----------
    train_df : DataFrame
        Training rows (caller has applied year filter, e.g. 2005-2022).
        Must include rows at `forecast_date`; rows at other dates are
        ignored. yield_target must be non-null in the kept rows.
    val_df : DataFrame
        Validation rows for early stopping (e.g. 2023). Same shape contract
        as train_df. yield_target must be non-null.
    forecast_date : str
        One of VALID_FORECAST_DATES. Selects the per-date feature list.
    params : dict, optional
        XGBoost parameters. Merged on top of DEFAULT_PARAMS.
    num_boost_round, early_stopping_rounds : int
        Standard xgb.train arguments.
    verbose_eval : bool | int
        Passed through to xgb.train.

    Returns
    -------
    Regressor
        With booster fit, best_iteration captured from early stopping, and
        train/val final-iteration RMSE in train_metrics.
    """
    if forecast_date not in VALID_FORECAST_DATES:
        raise ValueError(
            f"Unexpected forecast_date={forecast_date!r}; "
            f"must be one of {VALID_FORECAST_DATES}"
        )

    # Slice to the forecast_date's rows.
    train_sub = train_df[train_df["forecast_date"] == forecast_date]
    val_sub = val_df[val_df["forecast_date"] == forecast_date]
    if len(train_sub) == 0:
        raise ValueError(
            f"No training rows at forecast_date={forecast_date!r}. "
            f"Check the train-pool filter."
        )
    if len(val_sub) == 0:
        raise ValueError(
            f"No validation rows at forecast_date={forecast_date!r}. "
            f"Cannot early-stop without val data."
        )

    dtrain, train_kept = _build_dmatrix(train_sub, forecast_date, include_label=True)
    dval, val_kept = _build_dmatrix(val_sub, forecast_date, include_label=True)

    merged_params = {**DEFAULT_PARAMS, **(params or {})}

    evals_result: Dict[str, Dict[str, list]] = {}
    booster = xgb.train(
        merged_params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=verbose_eval,
    )

    # XGBoost stores best_iteration as an attribute on the booster after
    # early stopping. Fall back to the last iteration if no early stop fired.
    best_iter = getattr(booster, "best_iteration", num_boost_round - 1)

    final_train_rmse = float(evals_result["train"]["rmse"][best_iter])
    final_val_rmse = float(evals_result["val"]["rmse"][best_iter])

    return Regressor(
        forecast_date=forecast_date,
        booster=booster,
        feature_cols=FEATURE_COLS[forecast_date],
        params=merged_params,
        best_iteration=int(best_iter),
        train_metrics={
            "train_rmse": final_train_rmse,
            "val_rmse": final_val_rmse,
        },
        n_train=int(train_kept.sum()),
        n_val=int(val_kept.sum()),
    )


# -----------------------------------------------------------------------------
# RegressorBundle — four Regressors keyed by forecast_date
# -----------------------------------------------------------------------------


@dataclass
class RegressorBundle:
    """Four Regressors, one per forecast_date.

    The natural unit Phase C trains and ships. predict() dispatches to the
    right per-date booster automatically.
    """

    regressors: Dict[str, Regressor] = field(default_factory=dict)

    # ---- prediction ---------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict yield_target for `df`, dispatching by forecast_date.

        Returns predictions in the same row order as the input df.
        Raises if any row's forecast_date has no fit booster, or if any
        forecast_date in the bundle is missing.
        """
        if "forecast_date" not in df.columns:
            raise KeyError("df missing forecast_date column")

        out = np.full(len(df), np.nan, dtype=np.float64)
        for date in df["forecast_date"].unique():
            if date not in self.regressors:
                raise KeyError(
                    f"No regressor fit for forecast_date={date!r}. "
                    f"Bundle has: {sorted(self.regressors.keys())}"
                )
            mask = (df["forecast_date"] == date).to_numpy()
            sub = df.loc[mask].reset_index(drop=True)
            preds = self.regressors[date].predict(sub)
            out[mask] = preds
        return out

    # ---- serialization ------------------------------------------------------

    def save(self, dir_path: str | Path) -> None:
        """Save each Regressor to <dir_path>/regressor_<forecast_date>.json
        (+ matching .meta.json sidecar).

        Filenames replace the `:` in forecast_date with `_` so paths are
        Windows-friendly. Forecast dates already use `-` and a literal
        `EOS`, so this is safe with no information loss.
        """
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        for date, reg in self.regressors.items():
            safe = date.replace(":", "_")
            reg.save(dir_path / f"regressor_{safe}.json")

    @classmethod
    def load(cls, dir_path: str | Path) -> "RegressorBundle":
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Bundle directory {dir_path} does not exist.")
        regressors: Dict[str, Regressor] = {}
        for date in VALID_FORECAST_DATES:
            safe = date.replace(":", "_")
            path = dir_path / f"regressor_{safe}.json"
            if not path.exists():
                raise FileNotFoundError(
                    f"Bundle directory {dir_path} missing per-date model "
                    f"{path.name}. All 4 forecast_dates must be present."
                )
            regressors[date] = Regressor.load(path)
        return cls(regressors=regressors)


# -----------------------------------------------------------------------------
# Fit all dates — convenience for the driver script
# -----------------------------------------------------------------------------


def fit_all_dates(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    params: Optional[Dict[str, object]] = None,
    num_boost_round: int = DEFAULT_NUM_BOOST_ROUND,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    verbose_eval: bool | int = False,
) -> RegressorBundle:
    """Train one Regressor per forecast_date and bundle them.

    Same `params` are used for every date — caller (the driver script) is
    responsible for picking different params per date if a sweep selects them.
    """
    bundle = RegressorBundle()
    for date in VALID_FORECAST_DATES:
        bundle.regressors[date] = fit(
            train_df,
            val_df,
            forecast_date=date,
            params=params,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
        )
    return bundle
