"""Missing-modality evaluation.

Real claims arrive incomplete. A photograph may never be submitted, a narrative
may be a single uninformative line, a policy record may be mid-migration. A
system that only works when all three modalities are present is not a system
anyone can deploy, so the question here is not whether performance drops -- it
will -- but by how much, and whether the drop is graceful or catastrophic.

The evaluation takes one already-trained multimodal model and re-scores the
test split with modalities masked out, one scenario at a time. Crucially it
does *not* retrain per scenario: the point is how a single deployed model
behaves when its inputs are incomplete, not how well a model trained
specifically for a degraded input performs.

One methodological trap worth naming. If a model was trained with every text
and tabular block present, then masking text at test time is an
out-of-distribution request, and the resulting collapse measures the model's
surprise rather than the modality's value. That is why
:class:`~src.models.multimodal.trainer.TrainConfig` offers modality dropout,
and why this module reports a dropout-trained model alongside a plain one. The
comparison between them is the actual finding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics
from src.models.multimodal.dataset import ModalityBlock, MultimodalDataset
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Scenario:
    name: str
    available: list[str]
    """Modalities the claim still has."""

    description: str = ""


def enumerate_scenarios(modalities: list[str]) -> list[Scenario]:
    """Every non-empty subset, from complete down to a single modality."""
    scenarios: list[Scenario] = []
    for size in range(len(modalities), 0, -1):
        for subset in combinations(modalities, size):
            available = list(subset)
            missing = [m for m in modalities if m not in available]
            if not missing:
                name = "complete"
                description = "every modality present"
            else:
                name = "missing_" + "_".join(missing)
                description = f"{', '.join(missing)} unavailable"
            scenarios.append(
                Scenario(name=name, available=available, description=description)
            )
    return scenarios


@dataclass
class MissingModalityResult:
    rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    baseline_pr_auc: float = 0.0
    trained_with_dropout: bool = False

    def as_dict(self) -> dict:
        return {
            "baseline_pr_auc": self.baseline_pr_auc,
            "trained_with_dropout": self.trained_with_dropout,
            "scenarios": self.rows.to_dict(orient="records"),
        }

    def summary(self) -> str:
        if self.rows.empty:
            return "no scenarios evaluated"
        lines = []
        for _, row in self.rows.iterrows():
            lines.append(
                f"  {row['scenario']:<28} PR-AUC {row['pr_auc']:.4f} "
                f"({row['pr_auc_retained']:.1%} of complete)"
            )
        return "\n".join(lines)


def mask_dataset(dataset: MultimodalDataset, available: list[str]) -> MultimodalDataset:
    """Return a copy of ``dataset`` with the unavailable modalities masked out.

    The features are zeroed as well as masked, so a bug in the model's masking
    logic would show up as degraded performance rather than being silently
    covered by features that are still readable.
    """
    blocks: dict[str, ModalityBlock] = {}
    for name, block in dataset.blocks.items():
        if name in available:
            blocks[name] = block
        else:
            blocks[name] = ModalityBlock(
                name=name,
                features=np.zeros_like(block.features),
                present=np.zeros(len(block.features), dtype=bool),
                feature_names=block.feature_names,
            )
    return MultimodalDataset(blocks, dataset.labels, dataset.claim_ids)


def evaluate_missing_modalities(
    trainer,
    dataset: MultimodalDataset,
    *,
    threshold: float,
    scenarios: list[Scenario] | None = None,
) -> MissingModalityResult:
    """Score one trained model across every availability scenario."""
    modalities = list(dataset.blocks)
    scenarios = scenarios or enumerate_scenarios(modalities)

    rows = []
    baseline = None

    for scenario in scenarios:
        masked = mask_dataset(dataset, scenario.available)
        scores = trainer.predict_proba(masked)
        metrics = compute_metrics(masked.labels, scores, threshold=threshold)

        if baseline is None:
            baseline = metrics.pr_auc

        rows.append(
            {
                "scenario": scenario.name,
                "description": scenario.description,
                "available": "+".join(scenario.available),
                "n_modalities": len(scenario.available),
                "pr_auc": metrics.pr_auc,
                "pr_auc_retained": metrics.pr_auc / baseline if baseline else np.nan,
                "roc_auc": metrics.roc_auc,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "lift_at_10pct": metrics.capacity["lift_at_10pct"],
                "brier": metrics.brier,
                "mean_score": float(scores.mean()),
            }
        )
        logger.info(
            "scenario %-28s PR-AUC %.4f (%.1f%% of complete)",
            scenario.name, metrics.pr_auc,
            100 * metrics.pr_auc / baseline if baseline else float("nan"),
        )

    result = MissingModalityResult(
        rows=pd.DataFrame(rows),
        baseline_pr_auc=float(baseline or 0.0),
        trained_with_dropout=getattr(trainer.config, "modality_dropout", 0.0) > 0,
    )
    return result


def compare_dropout_training(
    plain: MissingModalityResult, with_dropout: MissingModalityResult
) -> pd.DataFrame:
    """Side-by-side degradation curves, with and without modality dropout.

    The interesting column is ``retained_gain``: how much more of its complete
    performance the dropout-trained model keeps when a modality disappears. A
    positive value means training the model to expect absence made it more
    robust to it.
    """
    merged = plain.rows.merge(
        with_dropout.rows,
        on=["scenario", "available", "n_modalities"],
        suffixes=("_plain", "_dropout"),
    )
    merged["pr_auc_gain"] = merged["pr_auc_dropout"] - merged["pr_auc_plain"]
    merged["retained_gain"] = (
        merged["pr_auc_retained_dropout"] - merged["pr_auc_retained_plain"]
    )
    return merged[
        [
            "scenario", "available", "n_modalities",
            "pr_auc_plain", "pr_auc_dropout", "pr_auc_gain",
            "pr_auc_retained_plain", "pr_auc_retained_dropout", "retained_gain",
        ]
    ]
