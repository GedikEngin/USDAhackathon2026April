"""
scripts/smoke_explain.py — smoke test for forecast.explain.

Loads the trained RegressorBundle from models/forecast/, runs each public
function in forecast.explain against val (2023) data, asserts shape and
additivity invariants, and prints qualitative output for eyeballing.

Run from repo root:
    python -m scripts.smoke_explain

Exit code:
    0 if every assertion passes
    1 if any assertion fails (asserts raise; non-zero exit naturally)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forecast.data import DEFAULT_MASTER_PATH, load_master, val_pool
from forecast.explain import (
    Attribution,
    Driver,
    attribution_table,
    feature_importance,
    shap_values_for,
    top_drivers,
    top_drivers_for_bundle,
)
from forecast.features import VALID_FORECAST_DATES
from forecast.regressor import FEATURE_COLS, RegressorBundle


BUNDLE_DIR = "models/forecast"


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# -----------------------------------------------------------------------------
# Setup: load bundle + data
# -----------------------------------------------------------------------------


def main() -> int:
    _section("Setup")
    print(f"Loading master ← {DEFAULT_MASTER_PATH}")
    master_df = load_master(DEFAULT_MASTER_PATH)
    val_df = val_pool(master_df)
    print(f"  master: {len(master_df):,} rows")
    print(f"  val:    {len(val_df):,} rows (2023)")

    print(f"Loading bundle ← {BUNDLE_DIR}")
    bundle = RegressorBundle.load(BUNDLE_DIR)
    print(f"  bundle dates: {sorted(bundle.regressors.keys())}")

    eos_reg = bundle.regressors["EOS"]
    eos_val = val_df[val_df["forecast_date"] == "EOS"].reset_index(drop=True)
    print(f"  EOS val rows: {len(eos_val)}")

    # ------------------------------------------------------------------------
    # Test 1: shap_values_for — runs, shape, additivity
    # ------------------------------------------------------------------------
    _section("Test 1: shap_values_for(eos_reg, eos_val)")

    attr: Attribution = shap_values_for(eos_reg, eos_val)
    n_rows = len(eos_val)
    n_feat = len(FEATURE_COLS["EOS"])

    # Shapes
    assert attr.shap_matrix.shape == (n_rows, n_feat), (
        f"shap_matrix shape {attr.shap_matrix.shape} != ({n_rows}, {n_feat})"
    )
    assert attr.feature_values.shape == (n_rows, n_feat), (
        f"feature_values shape {attr.feature_values.shape} != ({n_rows}, {n_feat})"
    )
    assert len(attr.predictions) == n_rows, "predictions length mismatch"
    assert len(attr.feature_cols) == n_feat, "feature_cols length mismatch"
    assert attr.feature_cols == list(FEATURE_COLS["EOS"]), (
        "feature_cols order does not match FEATURE_COLS['EOS']"
    )
    assert attr.forecast_date == "EOS"
    print(f"  shape OK: ({n_rows}, {n_feat})")

    # Additivity (already checked in __post_init__ but re-assert here for the test record)
    implied = attr.base_value + attr.shap_matrix.sum(axis=1)
    max_dev = float(np.max(np.abs(implied - attr.predictions)))
    assert max_dev < 1e-2, f"additivity max_dev {max_dev:.4f} >= 1e-2"
    print(f"  additivity OK: max |base + Σshap - pred| = {max_dev:.6f}")

    print(f"  base_value = {attr.base_value:.2f} bu/acre")
    print(f"  prediction range: {attr.predictions.min():.1f} to {attr.predictions.max():.1f}")
    print(f"  prediction mean:  {attr.predictions.mean():.1f}")

    # ------------------------------------------------------------------------
    # Test 2: top_drivers — ranking, sign agreement with raw matrix
    # ------------------------------------------------------------------------
    _section("Test 2: top_drivers ranking + sign correctness")

    # Pick the first IA-2023 EOS row; print + check.
    ia_rows = eos_val[eos_val["state_alpha"] == "IA"].reset_index(drop=True)
    assert len(ia_rows) > 0, "no IA val rows at EOS — check data"
    sample = ia_rows.iloc[0]

    drivers = top_drivers(eos_reg, sample, k=5)
    assert isinstance(drivers, list)
    assert len(drivers) == 5
    assert all(isinstance(d, Driver) for d in drivers)

    # Ranked by |shap| descending
    abs_vals = [abs(d.shap_value) for d in drivers]
    assert abs_vals == sorted(abs_vals, reverse=True), (
        f"top_drivers not ranked by |shap| desc: {abs_vals}"
    )
    # Direction matches sign
    for d in drivers:
        if d.shap_value > 0:
            assert d.direction == "+"
        elif d.shap_value < 0:
            assert d.direction == "-"
        else:
            assert d.direction == "0"
    print("  ranking + sign OK")

    # Cross-check against the raw matrix for this row.
    row_idx = 0  # first IA row in eos_val? not necessarily — find it
    ia_indices = eos_val.index[eos_val["state_alpha"] == "IA"].tolist()
    row_idx = ia_indices[0]
    raw_shap = attr.shap_matrix[row_idx]
    raw_top_idx = np.argsort(-np.abs(raw_shap))[:5]
    raw_top_features = [attr.feature_cols[i] for i in raw_top_idx]
    drivers_features = [d.feature for d in drivers]
    assert drivers_features == raw_top_features, (
        f"top_drivers features {drivers_features} != raw matrix top "
        f"{raw_top_features}"
    )
    print("  top_drivers matches raw matrix for sample row")

    # Print the drivers for eyeball check.
    print(f"\n  Top 5 drivers for IA county {sample['GEOID']} ({sample['county_name']}) at EOS 2023:")
    print(f"    truth      = {sample['yield_target']:.1f} bu/acre")
    print(f"    base_value = {attr.base_value:.1f} bu/acre")
    print(f"    prediction = {attr.predictions[row_idx]:.1f} bu/acre")
    print(f"    {'feature':<25} {'shap':>8} {'fvalue':>10}  dir")
    for d in drivers:
        print(f"    {d.feature:<25} {d.shap_value:>+7.2f}  {d.feature_value:>10.3f}   {d.direction}")

    # ------------------------------------------------------------------------
    # Test 3: top_drivers_for_bundle dispatches by forecast_date
    # ------------------------------------------------------------------------
    _section("Test 3: top_drivers_for_bundle dispatch")

    # Pull one row at each date for the same county; verify dispatch returns
    # something for each.
    polk_rows = val_df[val_df["GEOID"] == sample["GEOID"]].sort_values("forecast_date")
    if len(polk_rows) == 4:
        for _, r in polk_rows.iterrows():
            ds = top_drivers_for_bundle(bundle, r, k=3)
            assert len(ds) == 3
            print(f"  {r['forecast_date']:>5}: top driver = {ds[0].feature:<25} "
                  f"({ds[0].shap_value:+.2f})")
    else:
        print(f"  (skipping — sample county has {len(polk_rows)} rows, expected 4)")

    # Wrong-date row passed to a single-date regressor should raise.
    aug_row = val_df[
        (val_df["GEOID"] == sample["GEOID"]) & (val_df["forecast_date"] == "08-01")
    ]
    if len(aug_row) == 1:
        try:
            top_drivers(eos_reg, aug_row.iloc[0], k=3)
        except ValueError as e:
            print(f"  rejection of wrong forecast_date OK: {str(e)[:60]}...")
        else:
            raise AssertionError("top_drivers did not reject wrong forecast_date")

    # ------------------------------------------------------------------------
    # Test 4: attribution_table — shape + index columns
    # ------------------------------------------------------------------------
    _section("Test 4: attribution_table shape + index cols")

    table = attribution_table(eos_reg, eos_val)
    expected_rows = n_rows * n_feat
    assert len(table) == expected_rows, (
        f"table has {len(table)} rows, expected {expected_rows}"
    )

    expected_cols = {
        "GEOID", "year", "forecast_date", "state_alpha",
        "feature", "feature_value", "shap_value", "prediction", "base_value",
    }
    missing = expected_cols - set(table.columns)
    assert not missing, f"attribution_table missing cols: {missing}"
    print(f"  shape OK: {len(table):,} rows × {len(table.columns)} cols")

    # Round-trip: groupby (GEOID, year) sum of shap_value should equal
    # prediction - base_value for that input row.
    per_row = table.groupby(["GEOID", "year"]).agg(
        sum_shap=("shap_value", "sum"),
        prediction=("prediction", "first"),
        base=("base_value", "first"),
    )
    per_row["implied"] = per_row["base"] + per_row["sum_shap"]
    max_dev = float(np.max(np.abs(per_row["implied"] - per_row["prediction"])))
    assert max_dev < 1e-2, f"long-form additivity max_dev {max_dev:.4f}"
    print(f"  long-form additivity OK: max dev = {max_dev:.6f}")

    # ------------------------------------------------------------------------
    # Test 5: feature_importance — both methods, sorted, indexed correctly
    # ------------------------------------------------------------------------
    _section("Test 5: feature_importance (mean_abs, mean_signed)")

    imp_abs = feature_importance(eos_reg, eos_val, method="mean_abs")
    assert isinstance(imp_abs, pd.Series)
    assert len(imp_abs) == n_feat
    assert (imp_abs >= 0).all(), "mean_abs has negative values"
    # Sorted descending
    assert imp_abs.is_monotonic_decreasing, "mean_abs not sorted descending"
    print("  mean_abs OK")

    imp_signed = feature_importance(eos_reg, eos_val, method="mean_signed")
    assert isinstance(imp_signed, pd.Series)
    assert len(imp_signed) == n_feat
    # Sorted by |value| descending
    assert imp_signed.abs().is_monotonic_decreasing, "mean_signed not sorted by |value| desc"
    print("  mean_signed OK")

    # Eyeball top 10 of each
    print("\n  Top 10 features by mean |SHAP| (EOS, val 2023):")
    print(f"    {'feature':<28} {'mean|shap|':>11}")
    for feat, val in imp_abs.head(10).items():
        print(f"    {feat:<28} {val:>10.3f}")

    print("\n  Top 10 features by |mean signed SHAP| (shows directional bias):")
    print(f"    {'feature':<28} {'mean_shap':>11}")
    for feat, val in imp_signed.head(10).items():
        print(f"    {feat:<28} {val:>+10.3f}")

    # ------------------------------------------------------------------------
    # Test 6: known-bad path — bundle missing date
    # ------------------------------------------------------------------------
    _section("Test 6: error paths")

    # Constructed sub-bundle missing EOS.
    partial = RegressorBundle()
    partial.regressors["08-01"] = bundle.regressors["08-01"]
    eos_sample_row = eos_val.iloc[0]
    try:
        top_drivers_for_bundle(partial, eos_sample_row, k=3)
    except KeyError as e:
        print(f"  KeyError on missing date OK: {str(e)[:80]}...")
    else:
        raise AssertionError("top_drivers_for_bundle did not raise on missing date")

    # k=0 should fail.
    try:
        top_drivers(eos_reg, eos_sample_row, k=0)
    except ValueError as e:
        print(f"  ValueError on k=0 OK: {e}")
    else:
        raise AssertionError("top_drivers did not reject k=0")

    _section("All tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
