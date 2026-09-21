"""Explaining the structured model.

Two complementary methods, because they answer different questions and
disagreeing with each other is informative:

**SHAP** attributes a single prediction to its input features, and averaging
the magnitudes gives a global ranking. For tree ensembles the TreeExplainer
computes exact Shapley values rather than sampling them, so it is both fast and
not itself a source of noise.

**Permutation importance** measures how much a metric degrades when one feature
is shuffled. It is model-agnostic, it is computed on held-out data, and --
unlike SHAP -- it asks about *predictive* value rather than about how much the
model happened to use the feature. A feature the model leans on heavily but
which does not actually help will rank high on SHAP and low on permutation.

A caution that belongs in the code rather than only in the write-up: neither
method establishes causation. SHAP says the model's output moves with this
feature; permutation says the model's accuracy depends on it. Neither says the
feature causes fraud. With correlated inputs -- and this dataset has several,
including `policy_type` which literally embeds `vehicle_category` in its own
string -- attribution is also split arbitrarily among the correlated group, so
a low score can mean "redundant" rather than "irrelevant".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AttributionResult:
    method: str
    importances: pd.DataFrame
    """Columns: feature, importance, and method-specific extras."""

    shap_values: np.ndarray | None = None
    base_value: float | None = None
    notes: list[str] = field(default_factory=list)

    def top(self, n: int = 20) -> pd.DataFrame:
        return self.importances.head(n).reset_index(drop=True)

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "importances": self.importances.to_dict(orient="records"),
            "base_value": self.base_value,
            "notes": self.notes,
        }


def shap_importance(
    model,
    features: pd.DataFrame,
    *,
    max_samples: int = 2000,
    seed: int = 20260921,
) -> AttributionResult:
    """Global SHAP importance for a fitted tree model.

    ``model`` is the bare estimator and ``features`` the already-preprocessed
    matrix it was fitted on -- not the raw claim frame -- so that attribution
    is reported against the features the model actually saw.
    """
    import shap

    if len(features) > max_samples:
        # SHAP over 4,000 rows of a 400-tree ensemble is slow on four cores and
        # the global ranking is stable well before that.
        features = features.sample(max_samples, random_state=seed)

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(features)

    # Recent SHAP returns (n, features, classes) for binary classifiers, older
    # versions a list of two arrays. Take the positive class either way.
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    elif values.ndim == 3:
        values = values[:, :, 1]

    base = explainer.expected_value
    if isinstance(base, list | np.ndarray):
        base = float(np.asarray(base).ravel()[-1])

    importances = (
        pd.DataFrame(
            {
                "feature": list(features.columns),
                "importance": np.abs(values).mean(axis=0),
                "mean_signed": values.mean(axis=0),
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    logger.info(
        "SHAP over %d rows; top features: %s",
        len(features), ", ".join(importances["feature"].head(5)),
    )
    return AttributionResult(
        method="shap",
        importances=importances,
        shap_values=values,
        base_value=float(base),
        notes=[
            "Mean absolute SHAP value per feature, over the sampled rows.",
            "Attribution is split arbitrarily among correlated features, so a "
            "low score can mean redundant rather than uninformative.",
        ],
    )


def permutation_importance(
    predict_proba,
    features: pd.DataFrame,
    y: np.ndarray,
    *,
    metric: str = "average_precision",
    n_repeats: int = 10,
    seed: int = 20260921,
) -> AttributionResult:
    """Drop in a metric when one feature is shuffled, on held-out data.

    ``predict_proba`` takes the feature frame and returns positive-class
    probabilities, so this works for any model, including the fusion network.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    scorer = {
        "average_precision": average_precision_score,
        "roc_auc": roc_auc_score,
    }[metric]

    generator = np.random.default_rng(seed)
    y = np.asarray(y)
    baseline = float(scorer(y, predict_proba(features)))

    rows = []
    for column in features.columns:
        drops = []
        for _ in range(n_repeats):
            shuffled = features.copy()
            shuffled[column] = generator.permutation(shuffled[column].to_numpy())
            drops.append(baseline - float(scorer(y, predict_proba(shuffled))))
        rows.append(
            {
                "feature": column,
                "importance": float(np.mean(drops)),
                "std": float(np.std(drops)),
            }
        )

    importances = (
        pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
    )
    logger.info(
        "permutation importance (%s, baseline %.4f); top: %s",
        metric, baseline, ", ".join(importances["feature"].head(5)),
    )
    return AttributionResult(
        method=f"permutation_{metric}",
        importances=importances,
        base_value=baseline,
        notes=[
            f"Mean drop in {metric} over {n_repeats} shuffles of each feature.",
            "Computed on held-out data, so this measures predictive value "
            "rather than how much the model happened to use the feature.",
            "Negative values are within noise: shuffling an irrelevant feature "
            "sometimes improves the metric by chance.",
        ],
    )


def explain_claim(
    model,
    features: pd.DataFrame,
    row_index: int,
    *,
    top_n: int = 8,
) -> pd.DataFrame:
    """Per-feature contributions for one claim.

    This is what the API returns as the explanation for a single prediction:
    the features that pushed this particular claim's score up or down, signed
    and ordered by magnitude.
    """
    import shap

    explainer = shap.TreeExplainer(model)
    row = features.iloc[[row_index]]
    values = explainer.shap_values(row)

    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    elif values.ndim == 3:
        values = values[:, :, 1]

    contributions = pd.DataFrame(
        {
            "feature": list(features.columns),
            "value": row.iloc[0].to_numpy(),
            "contribution": values[0],
        }
    )
    contributions["abs_contribution"] = contributions["contribution"].abs()
    return (
        contributions.sort_values("abs_contribution", ascending=False)
        .head(top_n)
        .drop(columns="abs_contribution")
        .reset_index(drop=True)
    )
