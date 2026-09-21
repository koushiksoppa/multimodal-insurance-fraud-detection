"""Structured feature engineering and preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.features.tabular import (
    ORDINAL_ORDERS,
    OrdinalBandEncoder,
    build_preprocessor,
    engineer_features,
    feature_names,
    narrative_spec,
    tabular_spec,
)


def test_ordinal_orders_have_no_duplicate_levels():
    for name, order in ORDINAL_ORDERS.items():
        assert len(order) == len(set(order)), f"{name} lists a level twice"


def test_band_encoder_respects_the_declared_order():
    frame = pd.DataFrame(
        {"vehicle_price_band": ["more than 69000", "less than 20000", "30000 to 39000"]}
    )
    encoded = OrdinalBandEncoder(["vehicle_price_band"]).fit_transform(frame)
    assert encoded[1, 0] < encoded[2, 0] < encoded[0, 0]


def test_band_encoder_maps_unknown_levels_to_nan():
    """An unrecognised band must not silently become position zero."""
    frame = pd.DataFrame({"vehicle_price_band": ["not a real band"]})
    encoded = OrdinalBandEncoder(["vehicle_price_band"]).fit_transform(frame)
    assert np.isnan(encoded[0, 0])


def test_engineering_is_pure(claims: pd.DataFrame):
    before = claims.copy()
    engineer_features(claims)
    pd.testing.assert_frame_equal(claims, before)


def test_age_sentinel_becomes_explicit_missingness(claims: pd.DataFrame):
    frame = claims.copy()
    frame.loc[frame.index[:5], "driver_age"] = 0

    engineered = engineer_features(frame)
    assert engineered["driver_age_missing"].sum() == 5
    assert engineered.loc[engineered.index[:5], "driver_age"].isna().all()
    # A zero age must never survive into the feature matrix as a real value.
    assert not (engineered["driver_age"] == 0).any()


def test_policy_type_mismatch_flag_detects_the_inconsistency():
    frame = pd.DataFrame(
        {
            "policy_type": ["Sedan - Liability", "Sport - Collision"],
            "vehicle_category": ["Sport", "Sport"],
        }
    )
    engineered = engineer_features(frame)
    assert engineered["policy_type_vehicle_mismatch"].tolist() == [1, 0]


def test_preprocessor_produces_a_finite_dense_matrix(claims: pd.DataFrame):
    engineered = engineer_features(claims)
    spec = tabular_spec()
    pre = build_preprocessor(spec)
    matrix = pre.fit_transform(engineered[spec.columns])

    # The preprocessor emits named columns so that feature names survive into
    # SHAP and the importance reports.
    assert isinstance(matrix, pd.DataFrame)
    assert matrix.shape[0] == len(claims)
    assert np.isfinite(matrix.to_numpy()).all()


def test_feature_names_match_the_matrix_width(claims: pd.DataFrame):
    engineered = engineer_features(claims)
    spec = tabular_spec()
    pre = build_preprocessor(spec)
    matrix = pre.fit_transform(engineered[spec.columns])
    assert len(feature_names(pre)) == matrix.shape[1]


def test_preprocessor_fitted_on_train_transforms_unseen_rows(claims: pd.DataFrame):
    """The fit/transform boundary is what keeps preprocessing out of the leak path."""
    engineered = engineer_features(claims)
    spec = tabular_spec()
    train, test = engineered.iloc[:600], engineered.iloc[600:]

    pre = build_preprocessor(spec)
    pre.fit(train[spec.columns])
    transformed = pre.transform(test[spec.columns])

    assert transformed.shape[1] == len(feature_names(pre))
    assert np.isfinite(transformed.to_numpy()).all()


def test_unseen_category_at_transform_time_does_not_raise(claims: pd.DataFrame):
    """A live claim may carry a vehicle make the training data never saw."""
    engineered = engineer_features(claims)
    spec = tabular_spec()
    pre = build_preprocessor(spec).fit(engineered[spec.columns])

    novel = engineered.iloc[[0]].copy()
    novel["make"] = "A Make That Does Not Exist"
    assert np.isfinite(pre.transform(novel[spec.columns]).to_numpy()).all()


def test_excluding_sex_removes_it_from_the_feature_set():
    with_sex = tabular_spec(include_sex=True)
    without_sex = tabular_spec(include_sex=False)
    assert "driver_sex" in with_sex.columns
    assert "driver_sex" not in without_sex.columns


def test_year_is_excluded_by_default():
    """Under the temporal split the calendar year is constant in the test period."""
    assert "policy_year" not in tabular_spec().columns
    assert "policy_year" in tabular_spec(include_year=True).columns


def test_narrative_spec_shares_no_column_with_the_tabular_spec():
    assert not set(tabular_spec().columns) & set(narrative_spec().columns)


def test_narrative_preprocessor_runs(claims: pd.DataFrame):
    spec = narrative_spec()
    pre = build_preprocessor(spec)
    matrix = pre.fit_transform(claims[spec.columns])
    assert matrix.shape[0] == len(claims)
    assert np.isfinite(matrix.to_numpy()).all()


def test_preprocessor_composes_inside_a_pipeline(claims: pd.DataFrame):
    from sklearn.linear_model import LogisticRegression

    engineered = engineer_features(claims)
    spec = tabular_spec()
    pipeline = Pipeline(
        [
            ("pre", build_preprocessor(spec)),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    pipeline.fit(engineered[spec.columns], engineered["fraud_label"])
    probabilities = pipeline.predict_proba(engineered[spec.columns])[:, 1]
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


@pytest.mark.parametrize("scale", [True, False])
def test_scaling_toggle_changes_the_numeric_columns(claims: pd.DataFrame, scale: bool):
    engineered = engineer_features(claims)
    spec = tabular_spec()
    matrix = build_preprocessor(spec, scale_numeric=scale).fit_transform(
        engineered[spec.columns]
    )
    age_column = matrix["driver_age"].to_numpy()
    if scale:
        assert abs(age_column.mean()) < 1e-6
    else:
        assert age_column.mean() > 10
