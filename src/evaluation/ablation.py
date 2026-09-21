"""The modality ablation study.

Seven modality combinations, one shared architecture, repeated across seeds:

    tabular | text | image
    tabular+text | tabular+image | text+image
    tabular+text+image

Every cell is the same network given a different subset of its inputs, so a
difference between cells is attributable to the modality that changed and not
to a change of model family. Fusion strategies are only varied for cells with
more than one modality: with a single modality, gating and attention have
nothing to weigh against anything and reduce to the concatenation case, so
running them would pad the table with duplicates.

Two evaluation views, because a single number would be misleading here:

**Full test split (headline).** All 4,083 test claims. Claims without a
photograph carry a zero image block and a presence bit of zero, which is the
deployment reality: 85 % of real claims have no photo. Every cell is scored on
exactly the same claims, so the comparison is fair.

**Image-covered subset.** The 585 test claims that do have a photograph. This
is where the image modality can contribute at all, so it is where an image
effect would show up if one existed. It is reported separately rather than
mixed in, because the subset has a different prevalence and only 32 positives,
so its estimates are wide and must not be read as if they had the precision of
the full-split numbers.

Multiple seeds are run because with 213 test positives, a single-seed
difference of a point or two of PR-AUC is noise. The table reports mean and
standard deviation across seeds, and the bootstrap intervals from the harness
give the within-seed uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.evaluation.harness import evaluate_model
from src.evaluation.metrics import compute_metrics
from src.features.assemble import AssembledFeatures
from src.models.multimodal.dataset import MultimodalDataset, build_blocks, subset_blocks
from src.models.multimodal.fusion import FusionConfig
from src.models.multimodal.trainer import FusionTrainer, TrainConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)

MODALITY_COMBINATIONS: list[list[str]] = [
    ["tabular"],
    ["text"],
    ["image"],
    ["tabular", "text"],
    ["tabular", "image"],
    ["text", "image"],
    ["tabular", "text", "image"],
]


@dataclass
class AblationCell:
    """One trained model: a modality combination, a strategy and a seed."""

    modalities: list[str]
    strategy: str
    seed: int
    experiment_id: str
    val_pr_auc: float
    test_metrics: dict
    subset_metrics: dict | None
    gates: dict[str, float] | None
    history: dict
    n_parameters: int

    @property
    def key(self) -> str:
        return "+".join(self.modalities)


@dataclass
class AblationResult:
    cells: list[AblationCell] = field(default_factory=list)

    def table(self, *, view: str = "test") -> pd.DataFrame:
        """Per-cell metrics, averaged over seeds."""
        rows = []
        for cell in self.cells:
            metrics = cell.test_metrics if view == "test" else cell.subset_metrics
            if metrics is None:
                continue
            rows.append(
                {
                    "modalities": cell.key,
                    "strategy": cell.strategy,
                    "seed": cell.seed,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                    "pr_auc_lift": metrics["pr_auc_lift"],
                    "lift_at_10pct": metrics["capacity"]["lift_at_10pct"],
                    "brier": metrics["brier"],
                    "ece": metrics["ece"],
                    "val_pr_auc": cell.val_pr_auc,
                }
            )
        return pd.DataFrame(rows)

    def summary(self, *, view: str = "test") -> pd.DataFrame:
        """Mean and standard deviation across seeds, one row per cell."""
        raw = self.table(view=view)
        if raw.empty:
            return raw

        numeric = ["precision", "recall", "f1", "roc_auc", "pr_auc", "pr_auc_lift",
                   "lift_at_10pct", "brier", "ece"]
        grouped = raw.groupby(["modalities", "strategy"], as_index=False).agg(
            n_seeds=("seed", "nunique"),
            **{f"{c}_mean": (c, "mean") for c in numeric},
            **{f"{c}_std": (c, "std") for c in numeric},
        )
        # Ordered by the canonical modality sequence, not alphabetically, so
        # the table reads as a progression from single modalities upwards.
        order = {"+".join(c): i for i, c in enumerate(MODALITY_COMBINATIONS)}
        grouped["_order"] = grouped["modalities"].map(order).fillna(99)
        return grouped.sort_values(["_order", "strategy"]).drop(columns="_order")


def run_ablation(
    features: AssembledFeatures,
    *,
    strategies: list[str] | None = None,
    seeds: list[int] | None = None,
    combinations: list[list[str]] | None = None,
    train_config: TrainConfig | None = None,
    projection_dim: int = 64,
    hidden_dims: tuple[int, ...] = (64,),
    dropout: float = 0.3,
    dataset_version: str = "real-v1",
    track: str = "real",
    split_strategy: str = "temporal",
    n_bootstrap: int = 1000,
    save: bool = True,
) -> AblationResult:
    """Train and evaluate every requested ablation cell."""
    strategies = strategies or ["concat"]
    seeds = seeds or [20260921]
    combinations = combinations or MODALITY_COMBINATIONS

    claims = features.claims
    all_blocks = build_blocks(
        claims,
        tabular=features.tabular,
        tabular_names=features.tabular_names,
        text=features.text,
        image=features.image,
    )

    index = {
        split: np.flatnonzero((claims["split"] == split).to_numpy())
        for split in ("train", "val", "test")
    }
    labels = claims["fraud_label"].to_numpy()
    claim_ids = claims["claim_id"].to_numpy()

    # Test claims that actually carry a photograph: the subset where the image
    # modality can contribute anything at all.
    has_image_test = claims.iloc[index["test"]]["has_image"].to_numpy().astype(bool)

    result = AblationResult()

    for modalities in combinations:
        # With one modality there is nothing to fuse, so the strategies
        # collapse to the same model. Running them all would duplicate rows.
        cell_strategies = strategies if len(modalities) > 1 else ["concat"]

        for strategy in cell_strategies:
            for seed in seeds:
                cell = _run_cell(
                    modalities=modalities,
                    strategy=strategy,
                    seed=seed,
                    all_blocks=all_blocks,
                    index=index,
                    labels=labels,
                    claim_ids=claim_ids,
                    has_image_test=has_image_test,
                    train_config=train_config,
                    projection_dim=projection_dim,
                    hidden_dims=hidden_dims,
                    dropout=dropout,
                    dataset_version=dataset_version,
                    track=track,
                    split_strategy=split_strategy,
                    n_bootstrap=n_bootstrap,
                    save=save,
                )
                result.cells.append(cell)

    return result


def _run_cell(
    *,
    modalities: list[str],
    strategy: str,
    seed: int,
    all_blocks,
    index,
    labels,
    claim_ids,
    has_image_test,
    train_config,
    projection_dim,
    hidden_dims,
    dropout,
    dataset_version,
    track,
    split_strategy,
    n_bootstrap,
    save,
) -> AblationCell:
    blocks = {m: all_blocks[m] for m in modalities}

    datasets = {
        split: MultimodalDataset(
            subset_blocks(blocks, index[split]),
            labels[index[split]],
            claim_ids[index[split]],
        )
        for split in ("train", "val", "test")
    }

    config = TrainConfig(seed=seed) if train_config is None else _with_seed(train_config, seed)
    fusion_config = FusionConfig(
        strategy=strategy,
        input_dims=datasets["train"].input_dims,
        projection_dim=projection_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )

    logger.info(
        "--- ablation cell: %s | %s | seed %d ---",
        "+".join(modalities), strategy, seed,
    )
    trainer = FusionTrainer(fusion_config, config).fit(datasets["train"], datasets["val"])

    score_val = trainer.predict_proba(datasets["val"])
    score_test = trainer.predict_proba(datasets["test"])
    y_val = labels[index["val"]]
    y_test = labels[index["test"]]

    model_name = f"fusion_{strategy}" if len(modalities) > 1 else f"mlp_{modalities[0]}"

    evaluation = evaluate_model(
        model_name=model_name,
        modalities=modalities,
        y_val=y_val,
        score_val=score_val,
        y_test=y_test,
        score_test=score_test,
        claim_ids_val=claim_ids[index["val"]],
        claim_ids_test=claim_ids[index["test"]],
        n_bootstrap=n_bootstrap,
        track=track,
        dataset_version=dataset_version,
        split_strategy=split_strategy,
        seed=seed,
        hyperparameters={
            **config.as_dict(),
            **fusion_config.as_dict(),
            "n_parameters": trainer.model.n_parameters(),
        },
        notes=f"ablation cell {'+'.join(modalities)} with {strategy} fusion",
        tags=["ablation", track, strategy],
        duration_seconds=trainer.history.seconds,
        save=save,
    )

    # Same model, re-scored on the claims that carry a photograph.
    subset_metrics = None
    if has_image_test.any() and y_test[has_image_test].sum() > 0:
        subset_metrics = compute_metrics(
            y_test[has_image_test],
            score_test[has_image_test],
            threshold=evaluation.threshold.threshold,
            n_bootstrap=0,
        ).as_dict()

    return AblationCell(
        modalities=modalities,
        strategy=strategy,
        seed=seed,
        experiment_id=evaluation.experiment_id,
        val_pr_auc=evaluation.val.pr_auc,
        test_metrics=evaluation.test.as_dict(),
        subset_metrics=subset_metrics,
        gates=trainer.modality_gates(datasets["test"]),
        history=trainer.history.as_dict(),
        n_parameters=trainer.model.n_parameters(),
    )


def _with_seed(config: TrainConfig, seed: int) -> TrainConfig:
    from dataclasses import replace

    return replace(config, seed=seed)
