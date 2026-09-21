"""Train and evaluate the damage intelligence model.

Usage::

    python scripts/run_damage_model.py
    python scripts/run_damage_model.py --regimes frozen finetune_last_block
    python scripts/run_damage_model.py --epochs 40

The secondary vision task: classify a damage photograph by location (front,
rear) and condition (normal, breakage, crushed). Labels are real and
human-assigned; none is invented. Splits follow the claim splits, so no image
can be in one task's training data and another's test data.

Fine-tuning the final residual block is considerably slower on CPU than
training a head on frozen features -- minutes per epoch rather than seconds --
so run it once rather than sweeping it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data.build import DATASET_VERSION, load_built  # noqa: E402
from src.data.sources import image_root_path  # noqa: E402
from src.models.vision.damage import (  # noqa: E402
    DamageConfig,
    DamageModel,
    build_damage_datasets,
)
from src.utils.config import configure_threads, get_paths, get_settings  # noqa: E402
from src.utils.io import write_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402

logger = get_logger("damage_model")

REGIMES = ["frozen", "finetune_last_block"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-version", default=DATASET_VERSION)
    parser.add_argument("--regimes", nargs="+", default=["frozen"], choices=REGIMES)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=get_settings().seed)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_threads()
    set_seed(args.seed)

    claims = load_built(args.dataset_version)
    root = image_root_path()

    covered = claims[claims["has_image"]]
    logger.info(
        "damage dataset: %d images, class distribution %s",
        len(covered),
        covered["damage_class"].value_counts().to_dict(),
    )

    results = {}
    for regime in args.regimes:
        config = DamageConfig(
            regime=regime,
            backbone=args.backbone,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            augment=not args.no_augment,
            seed=args.seed,
        )
        datasets = build_damage_datasets(claims, root, config=config)

        logger.info("=== training damage model: %s ===", regime)
        model = DamageModel(config).fit(datasets["train"], datasets["val"])

        result = model.evaluate(datasets["test"])
        result.n_train = len(datasets["train"])
        result.seconds = model.history.get("seconds", 0.0)
        logger.info("%s", result.summary())

        results[regime] = {**result.as_dict(), "config": config.as_dict()}

    report_path = write_json(
        get_paths().reports / f"damage_model_{args.dataset_version}.json",
        {
            "dataset_version": args.dataset_version,
            "seed": args.seed,
            "n_images_total": int(len(covered)),
            "class_distribution": covered["damage_class"].value_counts().to_dict(),
            "results": results,
        },
    )
    logger.info("wrote %s", report_path)

    print()
    print(
        pd.DataFrame(
            [
                {
                    "regime": name,
                    "accuracy": round(r["accuracy"], 4),
                    "balanced_accuracy": round(r["balanced_accuracy"], 4),
                    "macro_f1": round(r["macro_f1"], 4),
                    "location_acc": round(r["location_accuracy"], 4),
                    "condition_acc": round(r["condition_accuracy"], 4),
                    "severity_mae": round(r["severity_mae"], 4),
                    "seconds": round(r["seconds"], 1),
                }
                for name, r in results.items()
            ]
        ).to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
