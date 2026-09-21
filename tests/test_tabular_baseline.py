"""Structured baseline models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.tabular import tabular_spec
from src.models.tabular.baseline import (
    DEFAULT_PARAMS,
    TabularBaseline,
    TabularModelConfig,
    build_estimator,
)

MODELS = ["logistic_regression", "random_forest", "gradient_boosting"]


@pytest.mark.parametrize("name", MODELS)
def test_each_model_fits_and_produces_valid_probabilities(claims: pd.DataFrame, name: str):
    model = TabularBaseline(tabular_spec(), TabularModelConfig(name=name)).fit(
        claims, claims["fraud_label"]
    )
    probabilities = model.predict_proba(claims)
    assert probabilities.shape == (len(claims),)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


@pytest.mark.parametrize("name", MODELS)
def test_each_model_ranks_better_than_chance_in_sample(claims: pd.DataFrame, name: str):
    """A sanity check, not a performance claim.

    In-sample ROC-AUC above chance only confirms the pipeline is wired up. Any
    real claim about performance comes from the held-out test split.
    """
    from sklearn.metrics import roc_auc_score

    model = TabularBaseline(tabular_spec(), TabularModelConfig(name=name)).fit(
        claims, claims["fraud_label"]
    )
    assert roc_auc_score(claims["fraud_label"], model.predict_proba(claims)) > 0.6


def test_predicting_before_fitting_is_refused(claims: pd.DataFrame):
    model = TabularBaseline(tabular_spec())
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict_proba(claims)


def test_missing_input_columns_are_reported_clearly(claims: pd.DataFrame):
    model = TabularBaseline(tabular_spec())
    with pytest.raises(KeyError, match="missing required columns"):
        model.fit(claims.drop(columns=["make"]), claims["fraud_label"])


def test_fitting_is_deterministic_for_a_fixed_seed(claims: pd.DataFrame):
    def run():
        return TabularBaseline(
            tabular_spec(), TabularModelConfig(name="gradient_boosting", seed=7)
        ).fit(claims, claims["fraud_label"]).predict_proba(claims)

    np.testing.assert_allclose(run(), run())


def test_feature_names_match_the_transformed_width(claims: pd.DataFrame):
    model = TabularBaseline(tabular_spec()).fit(claims, claims["fraud_label"])
    assert model.transform(claims).shape[1] == len(model.feature_names)


def test_only_the_logistic_model_scales_its_inputs(claims: pd.DataFrame):
    """Trees are scale-invariant, and unscaled inputs keep SHAP readable."""
    linear = TabularBaseline(
        tabular_spec(), TabularModelConfig(name="logistic_regression")
    ).fit(claims, claims["fraud_label"])
    trees = TabularBaseline(
        tabular_spec(), TabularModelConfig(name="gradient_boosting")
    ).fit(claims, claims["fraud_label"])

    assert abs(linear.transform(claims)["driver_age"].mean()) < 1e-6
    assert trees.transform(claims)["driver_age"].mean() > 10


def test_class_weighting_is_passed_through_to_the_estimator():
    weighted = build_estimator(
        TabularModelConfig(name="random_forest", class_weight=True)
    )
    plain = build_estimator(
        TabularModelConfig(name="random_forest", class_weight=False)
    )
    assert weighted.class_weight == "balanced"
    assert plain.class_weight is None


def test_class_weighting_raises_predicted_probabilities(claims: pd.DataFrame):
    """Reweighting changes the implied prior, which inflates scores.

    This is the mechanism behind the calibration cost reported in the results:
    a reweighted model ranks about as well but its probabilities no longer mean
    what they say.
    """
    weighted = TabularBaseline(
        tabular_spec(), TabularModelConfig(name="gradient_boosting", class_weight=True)
    ).fit(claims, claims["fraud_label"])
    plain = TabularBaseline(
        tabular_spec(), TabularModelConfig(name="gradient_boosting", class_weight=False)
    ).fit(claims, claims["fraud_label"])

    assert weighted.predict_proba(claims).mean() > plain.predict_proba(claims).mean()


def test_unweighted_probabilities_sit_near_the_base_rate(claims: pd.DataFrame):
    plain = TabularBaseline(
        tabular_spec(), TabularModelConfig(name="gradient_boosting", class_weight=False)
    ).fit(claims, claims["fraud_label"])
    assert plain.predict_proba(claims).mean() == pytest.approx(
        claims["fraud_label"].mean(), abs=0.05
    )


def test_custom_parameters_override_the_defaults():
    config = TabularModelConfig(name="random_forest", params={"n_estimators": 11})
    assert build_estimator(config).n_estimators == 11
    assert DEFAULT_PARAMS["random_forest"]["n_estimators"] != 11


def test_an_unknown_model_name_is_rejected():
    config = TabularModelConfig()
    config.name = "not_a_model"  # type: ignore[assignment]
    with pytest.raises(KeyError):
        build_estimator(config)
