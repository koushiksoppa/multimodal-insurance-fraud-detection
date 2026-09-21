"""Structured baselines.

Three model families, chosen to span a real trade-off rather than to pad a
list:

``logistic_regression``
    Linear, fast, and its coefficients are directly readable. The reference
    point: if a gradient-boosted ensemble cannot beat a well-specified logistic
    regression on this data, the extra complexity is not earning its place.

``random_forest``
    Bagged trees. Captures interactions without tuning, resistant to the
    irrelevant features this table contains, and cheap on four CPU cores.

``gradient_boosting``
    LightGBM. Usually the strongest family on tabular data of this size, and
    the one a production fraud system would most likely use.

On class imbalance. At a 6 % base rate, the choice is between class weighting,
threshold tuning, and resampling, and they are not equivalent:

* **Threshold tuning is always applied**, on validation data only. It changes
  the operating point without touching what the model learned, and it is the
  intervention that most directly reflects how a triage queue is run.
* **Class weighting is offered and evaluated**, not assumed. Reweighting changes
  the implied prior, which systematically inflates predicted probabilities --
  fine for ranking, actively harmful for calibration. Since this project
  evaluates calibration, both variants are measured and the cost is reported.
* **SMOTE and friends are deliberately not used.** Synthesising minority points
  by interpolating between neighbours assumes the space between two fraudulent
  claims is itself fraudulent. Here the features are overwhelmingly categorical
  one-hot indicators, where that interpolation produces fractional memberships
  corresponding to no real claim at all. It would also have to happen strictly
  inside the training fold to avoid leaking, at which point it mostly
  reproduces what class weighting does more cheaply and more honestly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.tabular import FeatureSpec, build_preprocessor, engineer_features
from src.utils.logging import get_logger

logger = get_logger(__name__)

ModelName = Literal["logistic_regression", "random_forest", "gradient_boosting"]


@dataclass
class TabularModelConfig:
    name: ModelName = "gradient_boosting"
    class_weight: bool = True
    seed: int = 20260921
    params: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "class_weight": self.class_weight,
            "seed": self.seed,
            "params": self.params,
        }


#: Defaults tuned for a small, wide, heavily imbalanced table on four CPU
#: cores. They are conservative on depth and strong on regularisation, because
#: with 550 positives in the training split an unconstrained ensemble memorises
#: rather than generalises.
DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "max_iter": 2000,
        "C": 1.0,
        "solver": "lbfgs",
    },
    "random_forest": {
        "n_estimators": 400,
        "max_depth": 12,
        "min_samples_leaf": 20,
        "max_features": "sqrt",
        "n_jobs": -1,
    },
    "gradient_boosting": {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "n_jobs": 4,
        "verbose": -1,
    },
}


def build_estimator(config: TabularModelConfig):
    """Instantiate the bare classifier, without preprocessing."""
    params = {**DEFAULT_PARAMS[config.name], **config.params}
    weight = "balanced" if config.class_weight else None

    if config.name == "logistic_regression":
        return LogisticRegression(random_state=config.seed, class_weight=weight, **params)

    if config.name == "random_forest":
        return RandomForestClassifier(
            random_state=config.seed, class_weight=weight, **params
        )

    if config.name == "gradient_boosting":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(random_state=config.seed, class_weight=weight, **params)

    raise ValueError(f"unknown model {config.name!r}")  # pragma: no cover


def build_pipeline(spec: FeatureSpec, config: TabularModelConfig) -> Pipeline:
    """Preprocessing plus classifier, as one fittable object.

    Bundling them matters: it is what guarantees the imputer, scaler and encoder
    are fitted on the training split alone. Fitting a scaler on the full frame
    before splitting is the most common quiet leak in tabular ML, and this
    structure makes it impossible rather than merely discouraged.
    """
    # Trees do not benefit from centred, unit-variance inputs, and leaving the
    # raw scale intact makes their SHAP values directly readable.
    scale = config.name == "logistic_regression"
    return Pipeline(
        [
            ("preprocess", build_preprocessor(spec, scale_numeric=scale)),
            ("classifier", build_estimator(config)),
        ]
    )


class TabularBaseline:
    """Fit-and-score wrapper around the pipeline.

    Holds the feature spec alongside the fitted pipeline so that callers never
    have to remember which columns the model expects, and applies the shared
    feature engineering on the way in so that it cannot be forgotten at
    inference time.
    """

    def __init__(self, spec: FeatureSpec, config: TabularModelConfig | None = None) -> None:
        self.spec = spec
        self.config = config or TabularModelConfig()
        self.pipeline = build_pipeline(spec, self.config)
        self.is_fitted = False

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.spec.columns) - set(engineer_features(df.head(1)).columns)
        if missing:
            raise KeyError(f"input is missing required columns: {sorted(missing)}")
        return engineer_features(df)[self.spec.columns]

    def fit(self, train: pd.DataFrame, y: np.ndarray | pd.Series) -> TabularBaseline:
        self.pipeline.fit(self._prepare(train), np.asarray(y))
        self.is_fitted = True
        logger.info(
            "fitted %s on %d rows, %d features",
            self.config.name, len(train), len(self.feature_names),
        )
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("model is not fitted")
        return self.pipeline.predict_proba(self._prepare(df))[:, 1]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """The preprocessed feature matrix, for SHAP and for fusion."""
        return self.pipeline.named_steps["preprocess"].transform(self._prepare(df))

    @property
    def feature_names(self) -> list[str]:
        return [
            str(n)
            for n in self.pipeline.named_steps["preprocess"].get_feature_names_out()
        ]

    @property
    def classifier(self):
        return self.pipeline.named_steps["classifier"]
