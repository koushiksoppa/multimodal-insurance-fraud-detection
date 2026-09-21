"""Run the modality ablation study.

Usage::

    python scripts/run_ablation.py
    python scripts/run_ablation.py --strategies concat gated attention
    python scripts/run_ablation.py --seeds 1 2 3 4 5
    python scripts/run_ablation.py --quick

Requires ``scripts/extract_features.py`` to have been run once; the cached
embeddings are what make this affordable on CPU.

Results are written to ``evaluation/reports/`` as both a machine-readable JSON
and a markdown table, and every cell is also recorded in the experiment
registry.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data.build import DATASET_VERSION, load_built  # noqa: E402
from src.evaluation.ablation import MODALITY_COMBINATIONS, run_ablation  # noqa: E402
from src.features.assemble import assemble  # noqa: E402
from src.models.multimodal.trainer import TrainConfig  # noqa: E402
from src.utils.config import configure_threads, get_paths  # noqa: E402
from src.utils.io import write_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402

logger = get_logger("ablation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-version", default=DATASET_VERSION)
    parser.add_argument("--track", default="real")
    parser.add_argument("--split-strategy", default="temporal")
    parser.add_argument(
        "--strategies", nargs="+", default=["concat"],
        choices=["concat", "gated", "attention"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260921, 20260922, 20260923])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument(
        "--quick", action="store_true",
        help="one seed, few epochs; for checking the pipeline runs, not for results",
    )
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def to_markdown(summary: pd.DataFrame, *, title: str, caption: str) -> str:
    """Render one ablation view as a markdown table."""
    lines = [f"### {title}", "", caption, ""]
    lines.append(
        "| Modalities | Fusion | Precision | Recall | F1 | ROC-AUC | PR-AUC | "
        "PR lift | Lift@10% | Brier | ECE |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    def cell(row: pd.Series, name: str, places: int = 4) -> str:
        """Mean, with the across-seed standard deviation when there is one."""
        mean = row[f"{name}_mean"]
        std = row[f"{name}_std"]
        if pd.isna(std) or row["n_seeds"] < 2:
            return f"{mean:.{places}f}"
        return f"{mean:.{places}f} ± {std:.{places}f}"

    for _, row in summary.iterrows():
        lines.append(
            f"| {row['modalities']} | {row['strategy']} | {cell(row, 'precision')} | "
            f"{cell(row, 'recall')} | {cell(row, 'f1')} | {cell(row, 'roc_auc')} | "
            f"{cell(row, 'pr_auc')} | {cell(row, 'pr_auc_lift', 2)} | "
            f"{cell(row, 'lift_at_10pct', 2)} | {cell(row, 'brier')} | {cell(row, 'ece')} |"
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    configure_threads()
    set_seed(args.seeds[0])

    seeds = args.seeds[:1] if args.quick else args.seeds
    epochs = 8 if args.quick else args.epochs

    claims = load_built(args.dataset_version)
    logger.info("loaded %d claims", len(claims))

    features = assemble(claims)
    logger.info("feature dimensions: %s", features.dims)

    train_config = TrainConfig(
        epochs=epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        class_weight_power=args.class_weight_power,
        patience=args.patience,
    )

    result = run_ablation(
        features,
        strategies=args.strategies,
        seeds=seeds,
        combinations=MODALITY_COMBINATIONS,
        train_config=train_config,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
        dataset_version=args.dataset_version,
        track=args.track,
        split_strategy=args.split_strategy,
        n_bootstrap=0 if args.quick else args.bootstrap,
        save=not args.no_save,
    )

    paths = get_paths()
    full = result.summary(view="test")
    subset = result.summary(view="subset")

    print()
    print(full.to_string(index=False))

    n_test_with_image = int(claims[(claims["split"] == "test") & claims["has_image"]].shape[0])
    n_pos_with_image = int(
        claims[(claims["split"] == "test") & claims["has_image"]]["fraud_label"].sum()
    )

    markdown = "\n\n".join(
        [
            "# Modality ablation study",
            "",
            f"Dataset `{args.dataset_version}`, track `{args.track}`, "
            f"`{args.split_strategy}` split, {len(seeds)} seed(s): "
            f"{', '.join(str(s) for s in seeds)}.",
            "Values are mean ± standard deviation across seeds. Thresholds were "
            "selected on validation and reused on test.",
            to_markdown(
                full,
                title="Full test split",
                caption=(
                    "All test claims. Every cell is scored on the same claims, so "
                    "the comparison is like-for-like. Claims without a photograph "
                    "carry a zero image block and a presence bit of zero, which is "
                    "the deployment reality."
                ),
            ),
            to_markdown(
                subset,
                title="Image-covered test subset",
                caption=(
                    f"The {n_test_with_image} test claims that carry a photograph, "
                    f"of which {n_pos_with_image} are fraud. This is the only place "
                    f"the image modality can contribute. With so few positives the "
                    f"estimates are wide; read them as indicative, not decisive."
                ),
            ),
        ]
    )

    report_path = paths.reports / f"ablation_{args.track}_{args.split_strategy}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(markdown, encoding="utf-8")

    json_path = write_json(
        paths.reports / f"ablation_{args.track}_{args.split_strategy}.json",
        {
            "dataset_version": args.dataset_version,
            "track": args.track,
            "split_strategy": args.split_strategy,
            "seeds": seeds,
            "strategies": args.strategies,
            "n_test_with_image": n_test_with_image,
            "n_positive_with_image": n_pos_with_image,
            "cells": [
                {
                    "modalities": c.modalities,
                    "strategy": c.strategy,
                    "seed": c.seed,
                    "experiment_id": c.experiment_id,
                    "val_pr_auc": c.val_pr_auc,
                    "test": c.test_metrics,
                    "image_subset": c.subset_metrics,
                    "gates": c.gates,
                    "history": c.history,
                    "n_parameters": c.n_parameters,
                }
                for c in result.cells
            ],
        },
    )
    logger.info("wrote %s", report_path)
    logger.info("wrote %s", json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
