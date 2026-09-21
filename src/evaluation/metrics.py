"""Evaluation metrics for an imbalanced detection problem.

Accuracy is deliberately absent from the headline metrics. At the roughly six
percent fraud rate in this data, a model that predicts "not fraud" for every
claim scores about 94 percent accuracy while catching nothing, so accuracy
measures the base rate rather than the model. It is computed and stored anyway,
because leaving it out entirely would make the reports harder to compare with
other work, but it is never used to choose anything.

What each headline metric is for:

ROC-AUC
    Ranking quality, independent of threshold and of prevalence. Comparable
    across datasets with different fraud rates -- which matters here, because
    the temporal test split has a lower fraud rate than the training period.

PR-AUC (average precision)
    Ranking quality in the region that actually matters when positives are
    rare. Unlike ROC-AUC it is sensitive to prevalence, so a bare PR-AUC is not
    comparable across splits. Every PR-AUC reported here is accompanied by the
    prevalence and by the lift over it, which is the comparable quantity.

Precision / recall / F1
    Threshold-dependent, so a threshold has to be chosen, and it is chosen on
    validation data only.

Brier score and calibration error
    Whether the predicted probability means anything. A score that ranks well
    but is badly calibrated cannot be used to set a review threshold in terms
    of expected cost, so this is checked rather than assumed.

Confidence intervals come from a stratified bootstrap over the evaluation set.
With a few hundred positives in a test split, differences of a couple of points
in PR-AUC are not distinguishable from noise, and the intervals are there to
keep the ablation discussion honest about that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logging import get_logger

logger = get_logger(__name__)

ThresholdStrategy = Literal["f1", "fbeta", "precision_at_recall", "review_budget", "fixed"]


@dataclass
class ClassificationMetrics:
    """Everything measured for one model on one evaluation set."""

    n: int
    n_positive: int
    prevalence: float
    threshold: float

    roc_auc: float
    pr_auc: float
    pr_auc_lift: float
    """PR-AUC divided by prevalence. 1.0 means no better than random ranking."""

    precision: float
    recall: float
    f1: float
    f2: float
    accuracy: float
    balanced_accuracy: float

    brier: float
    ece: float

    tp: int
    fp: int
    tn: int
    fn: int

    intervals: dict[str, tuple[float, float]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        payload = {
            k: v for k, v in self.__dict__.items() if k != "intervals"
        }
        payload["intervals"] = {k: list(v) for k, v in self.intervals.items()}
        return payload

    def summary(self) -> str:
        return (
            f"PR-AUC {self.pr_auc:.4f} (lift {self.pr_auc_lift:.2f}x) | "
            f"ROC-AUC {self.roc_auc:.4f} | P {self.precision:.4f} "
            f"R {self.recall:.4f} F1 {self.f1:.4f} @ thr {self.threshold:.4f}"
        )


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 10
) -> float:
    """Binned |confidence - accuracy| gap, weighted by bin occupancy.

    Equal-width bins on the probability axis. With a low base rate most
    predictions land in the lowest bins, so the statistic is dominated by the
    region where the model says "probably not fraud"; the reliability diagram
    in :mod:`src.evaluation.plots` shows the rest.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize puts values equal to an edge into the upper bin; clip the top
    # so a prediction of exactly 1.0 lands in the last bin rather than past it.
    bins = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)

    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        weight = mask.mean()
        total += weight * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(total)


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
    n_bootstrap: int = 0,
    bootstrap_seed: int = 20260921,
    ci: float = 0.95,
) -> ClassificationMetrics:
    """Compute every metric for one set of scores.

    ``y_score`` must be a probability-like score in [0, 1]; calibration
    metrics are meaningless otherwise.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    if len(y_true) != len(y_score):
        raise ValueError(f"length mismatch: {len(y_true)} labels, {len(y_score)} scores")
    if len(y_true) == 0:
        raise ValueError("cannot evaluate an empty set")

    n_positive = int(y_true.sum())
    prevalence = n_positive / len(y_true)
    if n_positive == 0 or n_positive == len(y_true):
        raise ValueError(
            "evaluation set contains a single class; ranking metrics are undefined"
        )

    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    pr_auc = float(average_precision_score(y_true, y_score))

    metrics = ClassificationMetrics(
        n=len(y_true),
        n_positive=n_positive,
        prevalence=prevalence,
        threshold=float(threshold),
        roc_auc=float(roc_auc_score(y_true, y_score)),
        pr_auc=pr_auc,
        pr_auc_lift=pr_auc / prevalence if prevalence else float("nan"),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        f2=float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        brier=float(brier_score_loss(y_true, y_score)),
        ece=expected_calibration_error(y_true, y_score),
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
    )

    if n_bootstrap:
        metrics.intervals = bootstrap_intervals(
            y_true, y_score,
            threshold=threshold,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed,
            ci=ci,
        )
    return metrics


def bootstrap_intervals(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float,
    n_bootstrap: int = 1000,
    seed: int = 20260921,
    ci: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Stratified bootstrap confidence intervals.

    Resampling is stratified by class so that every replicate keeps the same
    number of positives as the original set. An unstratified bootstrap on data
    this imbalanced occasionally draws a replicate with no positives at all,
    for which the ranking metrics are undefined.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    generator = np.random.default_rng(seed)

    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)

    collected: dict[str, list[float]] = {
        "roc_auc": [], "pr_auc": [], "precision": [], "recall": [], "f1": [], "brier": []
    }

    for _ in range(n_bootstrap):
        idx = np.concatenate(
            [
                generator.choice(pos_idx, size=len(pos_idx), replace=True),
                generator.choice(neg_idx, size=len(neg_idx), replace=True),
            ]
        )
        bt, bs = y_true[idx], y_score[idx]
        bp = (bs >= threshold).astype(int)
        collected["roc_auc"].append(roc_auc_score(bt, bs))
        collected["pr_auc"].append(average_precision_score(bt, bs))
        collected["precision"].append(precision_score(bt, bp, zero_division=0))
        collected["recall"].append(recall_score(bt, bp, zero_division=0))
        collected["f1"].append(f1_score(bt, bp, zero_division=0))
        collected["brier"].append(brier_score_loss(bt, bs))

    alpha = (1.0 - ci) / 2.0
    return {
        name: (
            float(np.quantile(values, alpha)),
            float(np.quantile(values, 1.0 - alpha)),
        )
        for name, values in collected.items()
    }


# --------------------------------------------------------------------------
# threshold selection
# --------------------------------------------------------------------------


@dataclass
class ThresholdChoice:
    threshold: float
    strategy: str
    objective_value: float
    rationale: str
    curve: dict[str, list[float]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "strategy": self.strategy,
            "objective_value": self.objective_value,
            "rationale": self.rationale,
        }


def select_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    strategy: ThresholdStrategy = "f1",
    beta: float = 2.0,
    target_recall: float = 0.5,
    review_budget: float = 0.05,
    fixed: float = 0.5,
    n_candidates: int = 200,
) -> ThresholdChoice:
    """Choose an operating threshold.

    This must only ever be given validation data. Choosing a threshold on the
    test set is a subtle way of fitting to it, and it is the single easiest way
    to make a fraud model look better than it is.

    Strategies:

    ``f1``
        Maximise F1. A reasonable default with no cost information.
    ``fbeta``
        Maximise F-beta with beta > 1, which weights recall higher. Defensible
        for fraud triage, where a missed fraud usually costs more than an
        unnecessary review.
    ``precision_at_recall``
        Highest-precision threshold that still reaches ``target_recall``.
    ``review_budget``
        Flag the top ``review_budget`` fraction of claims. This is how a triage
        queue is actually operated: an investigations team has a fixed capacity,
        not a fixed probability cut-off.
    ``fixed``
        Use a supplied constant, for comparison.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    if strategy == "fixed":
        return ThresholdChoice(
            float(fixed), "fixed", float("nan"), f"threshold fixed at {fixed}"
        )

    if strategy == "review_budget":
        threshold = float(np.quantile(y_score, 1.0 - review_budget))
        flagged = float((y_score >= threshold).mean())
        caught = float(y_true[y_score >= threshold].sum() / max(1, y_true.sum()))
        return ThresholdChoice(
            threshold,
            "review_budget",
            caught,
            f"flags the top {review_budget:.1%} of claims ({flagged:.1%} of the "
            f"validation set), capturing {caught:.1%} of validation fraud",
        )

    candidates = np.unique(
        np.quantile(y_score, np.linspace(0.0, 1.0, n_candidates))
    )

    best_threshold, best_value = float(candidates[0]), -np.inf
    curve: dict[str, list[float]] = {"threshold": [], "precision": [], "recall": [], "score": []}

    for threshold in candidates:
        y_pred = (y_score >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        if strategy == "f1":
            value = f1_score(y_true, y_pred, zero_division=0)
        elif strategy == "fbeta":
            value = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        elif strategy == "precision_at_recall":
            # Disqualify thresholds that miss the recall floor, then maximise
            # precision among those that meet it.
            value = precision if recall >= target_recall else -1.0
        else:  # pragma: no cover - guarded by the Literal type
            raise ValueError(f"unknown threshold strategy {strategy!r}")

        curve["threshold"].append(float(threshold))
        curve["precision"].append(float(precision))
        curve["recall"].append(float(recall))
        curve["score"].append(float(value))

        if value > best_value:
            best_threshold, best_value = float(threshold), float(value)

    rationale = {
        "f1": "maximises F1 on the validation split",
        "fbeta": f"maximises F{beta:g} on the validation split, weighting recall above precision",
        "precision_at_recall": (
            f"highest-precision validation threshold reaching recall >= {target_recall:.0%}"
        ),
    }[strategy]

    if strategy == "precision_at_recall" and best_value < 0:
        logger.warning(
            "no threshold reaches recall %.0f%% on validation; falling back to F1",
            100 * target_recall,
        )
        return select_threshold(y_true, y_score, strategy="f1", n_candidates=n_candidates)

    choice = ThresholdChoice(best_threshold, strategy, best_value, rationale, curve)
    logger.info(
        "threshold %.4f selected on validation (%s, objective %.4f)",
        choice.threshold, strategy, best_value,
    )
    return choice


def reliability_curve(
    y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 10
) -> dict[str, list[float]]:
    """Data for a reliability diagram, using equal-count bins.

    Equal-count rather than equal-width bins, because with a six percent base
    rate equal-width bins leave the upper ones nearly empty and the resulting
    diagram is mostly noise.
    """
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob, dtype=float)

    order = np.argsort(y_prob)
    chunks = np.array_split(order, n_bins)

    return {
        "mean_predicted": [float(y_prob[c].mean()) for c in chunks if len(c)],
        "observed_frequency": [float(y_true[c].mean()) for c in chunks if len(c)],
        "count": [int(len(c)) for c in chunks if len(c)],
    }
