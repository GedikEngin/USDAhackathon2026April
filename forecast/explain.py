"""
forecast.explain — SHAP-based driver attribution for the Phase C regressor.

Two consumers in mind:
  - Phase F narration agent: needs `top_drivers(geoid, year, date, k=3)` —
    returns the K features that pushed this specific prediction farthest from
    the model's average, signed (+/-) and ranked by absolute magnitude.
  - Phase C/G diagnostics (us): need per-row attribution tables and
    state-aggregated views to debug things like "why does the regressor
    underpredict IA-2024 by 15 bu?".

Computation
-----------
Uses XGBoost's native SHAP path (`Booster.predict(dm, pred_contribs=True)`).
No `shap` library dependency — that package is only needed for plotting,
which we don't do here. SHAP values from XGBoost satisfy the additivity
identity:

    prediction = base_value + Σ_i shap_value_i

where `base_value` is the model's mean prediction over the training set
(roughly the average corn yield, ~150 bu/acre). A SHAP value is the additive
contribution of one feature to one prediction: positive = pushes prediction
up vs base, negative = pushes it down.

State one-hots and `year` get attribution like any other feature; the agent
will naturally describe them as "this state's typical pattern contributed
+8 bu vs the cross-state baseline."

Public surface
--------------
    Driver                                  dataclass — one feature attribution
    Attribution                             dataclass — full per-row attribution + base
    shap_values_for(regressor, df)          -> Attribution  (raw matrix + base)
    top_drivers(regressor, query_row, k=3)  -> list[Driver]
    top_drivers_for_bundle(bundle, query_row, k=3) -> list[Driver]
    attribution_table(regressor, df)        -> pd.DataFrame  (long form)
    feature_importance(regressor, df,
                       method='mean_abs')   -> pd.Series

Notes
-----
- The Phase C `Regressor` stores `best_iteration` from early stopping. SHAP
  computation respects that via `iteration_range=(0, best_iteration+1)`,
  matching `Regressor.predict()` exactly. SHAP values therefore decompose
  the same prediction the model serves.
- All functions accept either a single-row DataFrame/Series or a multi-row
  DataFrame. For per-row work (Phase F), use `top_drivers`. For bulk work,
  use `attribution_table` and slice it.
- Feature names in the returned objects match the booster's feature_names
  (which match `FEATURE_COLS[forecast_date]` from regressor.py). One-hot
  state columns surface as `state_is_IA` etc.; downstream code can choose
  to merge those into a single `state` driver if desired.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np
import pandas as pd

from forecast.regressor import (
    FEATURE_COLS,
    Regressor,
    RegressorBundle,
    _add_derived_columns,
    _build_dmatrix,
)


# -----------------------------------------------------------------------------
# Per-feature record (Phase F's unit of consumption)
# -----------------------------------------------------------------------------


@dataclass
class Driver:
    """One feature's contribution to one prediction.

    Attributes
    ----------
    feature : str
        Column name from FEATURE_COLS — e.g. 'vpd_kpa_silk', 'state_is_IA',
        'year'.
    shap_value : float
        Signed contribution in bu/acre. prediction = base_value + Σ shap_value
        across all features.
    feature_value : float
        The raw feature value at this row (e.g. 1.42 for vpd_kpa_silk in kPa,
        2020 for year, 1.0 for a state one-hot that's "on"). Useful for
        narration ("at vpd_kpa_silk = 1.42 the model added +6 bu").
    direction : str
        '+' if shap_value > 0, '-' if < 0, '0' if exactly zero (rare; a
        feature with no influence at this row).
    """

    feature: str
    shap_value: float
    feature_value: float
    direction: str

    @classmethod
    def make(cls, feature: str, shap_value: float, feature_value: float) -> "Driver":
        if shap_value > 0:
            d = "+"
        elif shap_value < 0:
            d = "-"
        else:
            d = "0"
        return cls(
            feature=feature,
            shap_value=float(shap_value),
            feature_value=float(feature_value),
            direction=d,
        )


# -----------------------------------------------------------------------------
# Full attribution (raw matrix; base value)
# -----------------------------------------------------------------------------


@dataclass
class Attribution:
    """Result of running SHAP over a batch of rows for one Regressor.

    Attributes
    ----------
    shap_matrix : np.ndarray
        Shape (n_rows, n_features). Entry [i, j] is the contribution of
        feature j to the prediction of row i, in bu/acre.
    base_value : float
        The model's mean prediction over training data; same for all rows.
    feature_cols : list[str]
        Columns of `shap_matrix`, in the booster's feature order. Matches
        FEATURE_COLS[forecast_date].
    feature_values : np.ndarray
        Shape (n_rows, n_features). The raw input values that produced
        each SHAP entry. Useful for narration without needing to re-pull
        from the source df.
    forecast_date : str
        Which forecast_date this attribution is bound to.
    predictions : np.ndarray
        Shape (n_rows,). The model's predicted yields. Equals
        base_value + shap_matrix.sum(axis=1) by SHAP additivity.
    """

    shap_matrix: np.ndarray
    base_value: float
    feature_cols: List[str]
    feature_values: np.ndarray
    forecast_date: str
    predictions: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self) -> None:
        # Cheap sanity check: SHAP additivity within numerical tolerance.
        # XGBoost computes contribs in float32, so 1e-3 tolerance is generous.
        if len(self.predictions) > 0:
            implied = self.base_value + self.shap_matrix.sum(axis=1)
            max_dev = float(np.max(np.abs(implied - self.predictions)))
            if max_dev > 1e-2:
                raise ValueError(
                    f"SHAP additivity violated: max deviation between "
                    f"(base + Σ shap) and prediction = {max_dev:.4f}. "
                    f"Likely a feature_names mismatch between DMatrix and booster."
                )


# -----------------------------------------------------------------------------
# Core: compute SHAP values for one Regressor on a slice of rows
# -----------------------------------------------------------------------------


def shap_values_for(regressor: Regressor, df: pd.DataFrame) -> Attribution:
    """Compute SHAP values for every row in `df` against `regressor`.

    Parameters
    ----------
    regressor : Regressor
        Trained, with `best_iteration` set. Caller is responsible for
        passing rows whose forecast_date matches `regressor.forecast_date`.
    df : DataFrame
        Rows to attribute. Must contain all feature columns for the
        regressor's forecast_date (or columns from which they can be
        derived — `_add_derived_columns` is called internally). Mixed
        forecast_dates are rejected; use the bundle helper for that.

    Returns
    -------
    Attribution
    """
    if regressor.booster is None:
        raise RuntimeError("Regressor not fit — booster is None.")
    forecast_date = regressor.forecast_date
    if "forecast_date" in df.columns and not (df["forecast_date"] == forecast_date).all():
        raise ValueError(
            f"shap_values_for() called with rows at multiple forecast_dates; "
            f"this regressor is bound to {forecast_date!r}. Use bundle helpers "
            f"for mixed-date input."
        )

    # Build the DMatrix the same way predict() does — single source of truth.
    dm, row_kept = _build_dmatrix(df, forecast_date, include_label=False)

    # Native XGBoost SHAP. The returned matrix has shape (n_rows, n_features+1);
    # the last column is the bias term (= base_value), constant across rows.
    contribs = regressor.booster.predict(
        dm,
        pred_contribs=True,
        iteration_range=(0, regressor.best_iteration + 1),
    )
    if contribs.ndim != 2 or contribs.shape[1] != len(regressor.feature_cols) + 1:
        raise RuntimeError(
            f"Unexpected SHAP output shape {contribs.shape}; expected "
            f"({contribs.shape[0]}, {len(regressor.feature_cols) + 1})."
        )

    base_value = float(contribs[0, -1])
    if not np.allclose(contribs[:, -1], base_value, atol=1e-6):
        # XGBoost returns a constant bias column; if it varies per row,
        # something is wrong with the booster or the iteration range.
        raise RuntimeError(
            f"SHAP bias column is not constant across rows "
            f"(min={float(contribs[:, -1].min())}, max={float(contribs[:, -1].max())})."
        )
    shap_matrix = contribs[:, :-1].astype(np.float64)

    # Recover the same feature-value matrix _build_dmatrix used. We need this
    # for Driver.feature_value and for the attribution table.
    enriched = _add_derived_columns(df.iloc[row_kept].reset_index(drop=True))
    feature_values = enriched[regressor.feature_cols].to_numpy(dtype=np.float64)

    # Predictions for the additivity check.
    preds = regressor.booster.predict(
        dm, iteration_range=(0, regressor.best_iteration + 1)
    ).astype(np.float64)

    return Attribution(
        shap_matrix=shap_matrix,
        base_value=base_value,
        feature_cols=list(regressor.feature_cols),
        feature_values=feature_values,
        forecast_date=forecast_date,
        predictions=preds,
    )


# -----------------------------------------------------------------------------
# Top-K drivers — Phase F's primary surface
# -----------------------------------------------------------------------------


def top_drivers(
    regressor: Regressor,
    query_row: pd.Series | pd.DataFrame,
    k: int = 3,
) -> List[Driver]:
    """Top-K signed drivers for one query row, ranked by |shap_value| desc.

    Parameters
    ----------
    regressor : Regressor
    query_row : Series or 1-row DataFrame
        One row's worth of features at this regressor's forecast_date.
    k : int
        Number of drivers to return. If the model has fewer non-zero
        attributions than k, returns all of them.

    Returns
    -------
    list[Driver]
        Length min(k, n_nonzero), ranked by |shap_value| descending.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if isinstance(query_row, pd.Series):
        df = query_row.to_frame().T
    else:
        if len(query_row) != 1:
            raise ValueError(
                f"top_drivers expects 1 row, got {len(query_row)}. "
                f"Use attribution_table() for batch input."
            )
        df = query_row

    attr = shap_values_for(regressor, df)
    shap_row = attr.shap_matrix[0]                     # (n_features,)
    feat_row = attr.feature_values[0]                  # (n_features,)

    # Rank by |shap_value|; drop exact zeros (true non-contributors).
    abs_vals = np.abs(shap_row)
    order = np.argsort(-abs_vals)                      # descending
    drivers: List[Driver] = []
    for j in order:
        if abs_vals[j] == 0.0:
            break
        drivers.append(
            Driver.make(
                feature=attr.feature_cols[j],
                shap_value=shap_row[j],
                feature_value=feat_row[j],
            )
        )
        if len(drivers) >= k:
            break
    return drivers


def top_drivers_for_bundle(
    bundle: RegressorBundle,
    query_row: pd.Series | pd.DataFrame,
    k: int = 3,
) -> List[Driver]:
    """Bundle convenience: read forecast_date off the row, dispatch to the
    right per-date regressor.
    """
    if isinstance(query_row, pd.Series):
        forecast_date = str(query_row["forecast_date"])
    else:
        if len(query_row) != 1:
            raise ValueError("top_drivers_for_bundle expects exactly 1 row")
        forecast_date = str(query_row["forecast_date"].iloc[0])
    if forecast_date not in bundle.regressors:
        raise KeyError(
            f"Bundle has no regressor for forecast_date={forecast_date!r}. "
            f"Has: {sorted(bundle.regressors.keys())}"
        )
    return top_drivers(bundle.regressors[forecast_date], query_row, k=k)


# -----------------------------------------------------------------------------
# Attribution table — diagnostic surface (long-form DataFrame)
# -----------------------------------------------------------------------------


def attribution_table(
    regressor: Regressor,
    df: pd.DataFrame,
    *,
    include_index_cols: bool = True,
) -> pd.DataFrame:
    """Long-form attribution: one row per (input_row, feature).

    Columns:
        GEOID, year, forecast_date  (if include_index_cols and present in df)
        feature                     str
        feature_value               float
        shap_value                  float
        prediction                  float (same value across a given input row)
        base_value                  float (constant)

    Useful for:
        - Per-state SHAP aggregation: groupby('state_alpha').shap_value.mean()
        - Finding the rows where a specific feature dominates
        - Plotting in a notebook
    """
    attr = shap_values_for(regressor, df)
    n_rows, n_feat = attr.shap_matrix.shape

    # Build the long-form table efficiently — repeat the index columns
    # n_feat times, and tile the feature names n_rows times.
    feat_names = np.array(attr.feature_cols)
    feature_col = np.tile(feat_names, n_rows)
    shap_col = attr.shap_matrix.reshape(-1)
    fval_col = attr.feature_values.reshape(-1)
    pred_col = np.repeat(attr.predictions, n_feat)

    out = pd.DataFrame(
        {
            "feature": feature_col,
            "feature_value": fval_col,
            "shap_value": shap_col,
            "prediction": pred_col,
            "base_value": attr.base_value,
        }
    )

    if include_index_cols:
        # Recover the same row order _build_dmatrix used (after NaN-row drop).
        # Use the helper's row_kept by re-deriving it deterministically.
        enriched = _add_derived_columns(df)
        cols = FEATURE_COLS[regressor.forecast_date]
        feat_block = enriched[cols].to_numpy(dtype=np.float32)
        row_kept = ~np.isnan(feat_block).any(axis=1)
        index_df = df.iloc[row_kept].reset_index(drop=True)

        for ic in ("GEOID", "year", "forecast_date", "state_alpha"):
            if ic in index_df.columns:
                out.insert(0, ic, np.repeat(index_df[ic].to_numpy(), n_feat))

    return out


# -----------------------------------------------------------------------------
# Global feature importance — model-level (averaged over a batch)
# -----------------------------------------------------------------------------


def feature_importance(
    regressor: Regressor,
    df: pd.DataFrame,
    *,
    method: Literal["mean_abs", "mean_signed"] = "mean_abs",
) -> pd.Series:
    """Global feature importance, computed as a function of SHAP over `df`.

    Parameters
    ----------
    method :
        'mean_abs'   → mean |shap| per feature (typical "importance")
        'mean_signed'→ mean signed shap per feature (shows direction bias —
                       positive means the feature usually pushes predictions
                       up vs base)

    Returns
    -------
    pd.Series
        Index = feature column names, sorted descending by value (or by
        |value| if method='mean_signed'). Ties broken by feature name.
    """
    attr = shap_values_for(regressor, df)
    if method == "mean_abs":
        vals = np.mean(np.abs(attr.shap_matrix), axis=0)
        s = pd.Series(vals, index=attr.feature_cols, name="mean_abs_shap")
        return s.sort_values(ascending=False)
    elif method == "mean_signed":
        vals = np.mean(attr.shap_matrix, axis=0)
        s = pd.Series(vals, index=attr.feature_cols, name="mean_signed_shap")
        # Sort by absolute magnitude so the strongest directional features
        # appear at the top regardless of sign.
        return s.reindex(s.abs().sort_values(ascending=False).index)
    else:
        raise ValueError(f"Unknown method={method!r}; expected 'mean_abs' or 'mean_signed'.")
