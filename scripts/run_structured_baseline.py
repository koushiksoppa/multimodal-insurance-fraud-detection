"""Train and evaluate the structured (tabular-only) baselines.

Usage::

    python scripts/run_structured_baseline.py
    python scripts/run_structured_baseline.py --models gradient_boosting
    python scripts/run_structured_baseline.py --compare-class-weight
    python scripts/run_structured_baseline.py --split-strategy stratified_random

This is Baseline 1 of the ablation study, and it is also the reference point
the multimodal models have to beat. Everything is fitted on train, tuned on
validation and scored once on test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data.build import DATASET_VERSION, load_built  # noqa: E402
from src.evaluation.harness import Timer, evaluate_model  # noqa: E402
from src.features.tabular import narrative_spec, tabular_spec  # noqa: E402
from src.models.tabular.baseline import TabularBaseline, TabularModelConfig  # noqa: E402
from src.utils.config import get_settings  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402

logger = get_logger("structured_baseline")

ALL_MODELS = ["logistic_regression", "random_forest", "gradient_boosting"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument("--seed", type=int, default=get_settings().seed)
    parser.add_argument("--split-strategy", default="temporal")
    parser.add_argument("--dataset-version", default=DATASET_VERSION)
    parser.add_argument(
        "--compare-class-weight",
        action="store_true",
        help="also fit each model without class weighting, to measure what "
             "reweighting costs in calibration and buys in ranking",
    )
    parser.add_argument(
        "--narrative-ceiling",
        action="store_true",
        help="additionally fit a model on the narrative fields as columns; the "
             "gap to the text model measures what is lost by expressing those "
             "facts as prose",
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def run_one(
    *,
    name: str,
    spec,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    args: argparse.Namespace,
    class_weight: bool,
    modalities: list[str],
    model_label: str | None = None,
    notes: str = "",
    tags: list[str] | None = None,
):
    config = TabularModelConfig(name=name, class_weight=class_weight, seed=args.seed)

    with Timer() as timer:
        model = TabularBaseline(spec, config).fit(train, train["fraud_label"])

    return evaluate_model(
        model_name=model_label or name,
        modalities=modalities,
        y_val=val["fraud_label"].to_numpy(),
        score_val=model.predict_proba(val),
        y_test=test["fraud_label"].to_numpy(),
        score_test=model.predict_proba(test),
        claim_ids_val=val["claim_id"],
        claim_ids_test=test["claim_id"],
        n_bootstrap=args.bootstrap,
        dataset_version=args.dataset_version,
        split_strategy=args.split_strategy,
        seed=args.seed,
        hyperparameters={**config.as_dict(), "n_features": len(model.feature_names)},
        notes=notes,
        tags=tags or [],
        duration_seconds=timer.seconds,
        save=not args.no_save,
    )


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    claims = load_built(args.dataset_version)
    train = claims[claims["split"] == "train"]
    val = claims[claims["split"] == "val"]
    test = claims[claims["split"] == "test"]
    logger.info(
        "train %d (%d fraud) | val %d (%d) | test %d (%d)",
        len(train), int(train["fraud_label"].sum()),
        len(val), int(val["fraud_label"].sum()),
        len(test), int(test["fraud_label"].sum()),
    )

    spec = tabular_spec(include_year=args.split_strategy != "temporal")
    results = []

    for name in args.models:
        results.append(
            run_one(
                name=name, spec=spec, train=train, val=val, test=test, args=args,
                class_weight=True, modalities=["tabular"],
                notes="structured baseline, class-weighted",
                tags=["baseline", "tabular"],
            )
        )

        if args.compare_class_weight:
            results.append(
                run_one(
                    name=name, spec=spec, train=train, val=val, test=test, args=args,
                    class_weight=False, modalities=["tabular"],
                    model_label=f"{name}_unweighted",
                    notes="structured baseline without class weighting; compare "
                          "calibration against the weighted variant",
                    tags=["baseline", "tabular", "calibration-study"],
                )
            )

    if args.narrative_ceiling:
        results.append(
            run_one(
                name="gradient_boosting", spec=narrative_spec(), train=train, val=val,
                test=test, args=args, class_weight=True, modalities=["text"],
                model_label="narrative_fields_ceiling",
                notes="narrative-channel fields supplied as columns rather than "
                      "prose; upper bound on what the text model could recover",
                tags=["diagnostic", "narrative-ceiling"],
            )
        )

    print()
    summary = pd.DataFrame(
        [
            {
                "model": r.model,
                "modalities": "+".join(r.modalities),
                "val_pr_auc": round(r.val.pr_auc, 4),
                "test_pr_auc": round(r.test.pr_auc, 4),
                "test_pr_lift": round(r.test.pr_auc_lift, 2),
                "test_roc_auc": round(r.test.roc_auc, 4),
                "test_precision": round(r.test.precision, 4),
                "test_recall": round(r.test.recall, 4),
                "test_f1": round(r.test.f1, 4),
                "test_brier": round(r.test.brier, 4),
                "test_ece": round(r.test.ece, 4),
            }
            for r in results
        ]
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
