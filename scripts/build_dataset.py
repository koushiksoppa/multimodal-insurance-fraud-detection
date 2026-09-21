"""Build the unified multimodal claim dataset from the raw sources.

Usage::

    python scripts/build_dataset.py
    python scripts/build_dataset.py --split-strategy stratified_random
    python scripts/build_dataset.py --no-verify-images

Writes ``data/processed/claims_<version>.parquet`` and a build report under
``evaluation/reports/``. The build is deterministic for a given seed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.build import BuildConfig, build_real_dataset  # noqa: E402
from src.data.splits import SplitConfig  # noqa: E402
from src.utils.config import get_settings  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger("build_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=get_settings().seed)
    parser.add_argument(
        "--split-strategy",
        choices=["temporal", "stratified_random"],
        default="temporal",
        help="temporal is the headline protocol; the random split is for comparison",
    )
    parser.add_argument(
        "--no-verify-images",
        action="store_true",
        help="skip decoding every image (faster, but corrupt files go unnoticed)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = BuildConfig(
        seed=args.seed,
        split=SplitConfig(strategy=args.split_strategy, seed=args.seed),
        verify_images=not args.no_verify_images,
    )

    claims, report = build_real_dataset(config)

    logger.info("built %d claims, %d columns", len(claims), claims.shape[1])
    logger.info(
        "fraud rate %.4f | image coverage %.4f",
        claims["fraud_label"].mean(),
        claims["has_image"].mean(),
    )
    n_warnings = report["quality"]["n_warnings"]
    if n_warnings:
        logger.warning("%d data quality warnings recorded in the build report", n_warnings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
