"""The shared evaluation harness.

Every model in this project -- tabular, text, image, and each fusion variant --
is scored through this one function. That is the point: an ablation table is
only meaningful if every cell was produced under identical conditions. Routing
all of them through a single code path removes the possibility that one arm was
scored with a differently-chosen threshold, a different bootstrap seed or a
different evaluation subset.

Two rules are enforced here rather than left to discipline:

1. **The threshold is selected on validation and reused verbatim on test.**
   The function does not accept a test-set threshold at all.
2. **Test data is scored once, at the end.** Nothing in this module inspects
   test performance before the threshold is fixed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    ClassificationMetrics,
    ThresholdChoice,
    ThresholdStrategy,
    compute_metrics,
    reliability_curve,
    select_threshold,
)
from src.evaluation.registry import ExperimentRecord, ExperimentRegistry, make_experiment_id
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_BOOTSTRAP = 1000


@dataclass
class EvaluationResult:
    """Everything one model produced on one dataset."""

    experiment_id: str
    model: str
    modalities: list[str]
    threshold: ThresholdChoice
    val: ClassificationMetrics
    test: ClassificationMetrics
    predictions: pd.DataFrame
    reliability: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.experiment_id}\n"
            f"  val : {self.val.summary()}\n"
            f"  test: {self.test.summary()}"
        )


def evaluate_model(
    *,
    model_name: str,
    modalities: list[str],
    y_val: np.ndarray,
    score_val: np.ndarray,
    y_test: np.ndarray,
    score_test: np.ndarray,
    claim_ids_test: pd.Series | np.ndarray | None = None,
    claim_ids_val: pd.Series | np.ndarray | None = None,
    threshold_strategy: ThresholdStrategy = "review_budget",
    threshold_beta: float = 2.0,
    review_budget: float = 0.10,
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
    track: str = "real",
    dataset_version: str = "real-v1",
    split_strategy: str = "temporal",
    seed: int = 20260921,
    hyperparameters: dict[str, Any] | None = None,
    notes: str = "",
    tags: list[str] | None = None,
    duration_seconds: float | None = None,
    registry: ExperimentRegistry | None = None,
    save: bool = True,
) -> EvaluationResult:
    """Score one model and, by default, record the run.

    ``threshold_strategy`` defaults to a fixed review capacity of 10 percent,
    chosen on validation.

    F-beta was the obvious first choice -- a missed fraudulent claim is paid out
    in full while a false positive only costs an investigator's time, so
    weighting recall seems right. In practice it is degenerate on this data: the
    signal is weak enough that F2 is maximised by flagging around seventy
    percent of all claims, which is not an operating point any insurer could
    staff. A capacity threshold avoids that failure mode, matches how a triage
    queue is actually run, and keeps precision and recall comparable across
    models instead of each arm being scored at its own convenient cut-off.

    The F1- and F2-optimal thresholds are still computed and reported, in the
    threshold analysis, where the degeneracy is visible rather than hidden
    inside a headline number.
    """
    y_val = np.asarray(y_val).astype(int)
    y_test = np.asarray(y_test).astype(int)
    score_val = np.asarray(score_val, dtype=float)
    score_test = np.asarray(score_test, dtype=float)

    # Threshold comes from validation. The test scores are not consulted.
    choice = select_threshold(
        y_val,
        score_val,
        strategy=threshold_strategy,
        beta=threshold_beta,
        review_budget=review_budget,
    )

    # Every alternative strategy is also evaluated on validation, so the
    # threshold analysis can show what each one would have picked and what it
    # would have cost. All of these are validation-only; none touches test.
    alternatives = _threshold_alternatives(
        y_val, score_val, beta=threshold_beta, review_budget=review_budget
    )

    val_metrics = compute_metrics(
        y_val, score_val, threshold=choice.threshold, n_bootstrap=0
    )
    test_metrics = compute_metrics(
        y_test, score_test, threshold=choice.threshold,
        n_bootstrap=n_bootstrap, bootstrap_seed=seed,
    )

    experiment_id = make_experiment_id(
        model=model_name,
        modalities=modalities,
        track=track,
        seed=seed,
        split_strategy=split_strategy,
    )

    predictions = pd.DataFrame(
        {
            "claim_id": (
                np.asarray(claim_ids_test)
                if claim_ids_test is not None
                else np.arange(len(y_test))
            ),
            "split": "test",
            "y_true": y_test,
            "score": score_test,
            "y_pred": (score_test >= choice.threshold).astype(int),
        }
    )
    if claim_ids_val is not None:
        val_frame = pd.DataFrame(
            {
                "claim_id": np.asarray(claim_ids_val),
                "split": "val",
                "y_true": y_val,
                "score": score_val,
                "y_pred": (score_val >= choice.threshold).astype(int),
            }
        )
        predictions = pd.concat([val_frame, predictions], ignore_index=True)

    result = EvaluationResult(
        experiment_id=experiment_id,
        model=model_name,
        modalities=modalities,
        threshold=choice,
        val=val_metrics,
        test=test_metrics,
        predictions=predictions,
        reliability={
            "val": reliability_curve(y_val, score_val),
            "test": reliability_curve(y_test, score_test),
        },
        extra={"threshold_alternatives": alternatives},
    )

    if save:
        registry = registry or ExperimentRegistry()
        record = ExperimentRecord(
            experiment_id=experiment_id,
            model=model_name,
            modalities=modalities,
            dataset_version=dataset_version,
            track=track,
            split_strategy=split_strategy,
            seed=seed,
            hyperparameters=hyperparameters or {},
            metrics={"val": val_metrics.as_dict(), "test": test_metrics.as_dict()},
            threshold=choice.as_dict(),
            notes=notes,
            tags=tags or [],
            duration_seconds=duration_seconds,
        )
        registry.save(record)
        registry.save_artifact(experiment_id, "predictions.parquet", predictions)
        registry.save_artifact(experiment_id, "reliability.json", result.reliability)
        registry.save_artifact(
            experiment_id, "threshold_alternatives.json", alternatives
        )
        if choice.curve:
            registry.save_artifact(experiment_id, "threshold_curve.json", choice.curve)

    logger.info("%s", result.summary())
    return result


def _threshold_alternatives(
    y_val: np.ndarray,
    score_val: np.ndarray,
    *,
    beta: float,
    review_budget: float,
) -> dict[str, dict[str, Any]]:
    """What each threshold strategy would choose, measured on validation.

    Recorded for the threshold analysis. Includes the fraction of claims each
    strategy would flag, which is what exposes a degenerate choice: a strategy
    that reaches high recall by flagging most of the book is not a usable
    operating point regardless of its F-score.
    """
    strategies: dict[str, dict[str, Any]] = {
        "f1": {"strategy": "f1"},
        "f2": {"strategy": "fbeta", "beta": beta},
        "recall_50": {"strategy": "precision_at_recall", "target_recall": 0.5},
        "budget_5pct": {"strategy": "review_budget", "review_budget": 0.05},
        "budget_10pct": {"strategy": "review_budget", "review_budget": review_budget},
        "budget_20pct": {"strategy": "review_budget", "review_budget": 0.20},
    }

    out: dict[str, dict[str, Any]] = {}
    for label, kwargs in strategies.items():
        try:
            picked = select_threshold(y_val, score_val, **kwargs)
        except ValueError:  # pragma: no cover - defensive
            continue
        flagged = float((score_val >= picked.threshold).mean())
        at_threshold = compute_metrics(y_val, score_val, threshold=picked.threshold)
        out[label] = {
            "threshold": picked.threshold,
            "flagged_fraction": flagged,
            "precision": at_threshold.precision,
            "recall": at_threshold.recall,
            "f1": at_threshold.f1,
            "rationale": picked.rationale,
        }
    return out


class Timer:
    """Wall-clock timing for a training run, recorded with the experiment."""

    def __init__(self) -> None:
        self.seconds: float | None = None
        self._start = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self._start
