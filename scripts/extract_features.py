"""Encode text and images once and cache the embeddings.

Usage::

    python scripts/extract_features.py
    python scripts/extract_features.py --refresh
    python scripts/extract_features.py --text-model distilbert-base-uncased

This is the expensive step: roughly fifteen minutes on a four-core CPU for the
full corpus. Run it once. Every experiment afterwards reads the cache, which is
what makes the seven-cell ablation across multiple seeds practical without a
GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.build import DATASET_VERSION, load_built  # noqa: E402
from src.features.assemble import assemble  # noqa: E402
from src.models.nlp.encoder import TextEncoderConfig  # noqa: E402
from src.models.vision.encoder import VisionEncoderConfig  # noqa: E402
from src.utils.config import configure_threads, get_paths  # noqa: E402
from src.utils.io import write_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger("extract_features")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-version", default=DATASET_VERSION)
    parser.add_argument("--text-model", default=TextEncoderConfig().model_name)
    parser.add_argument("--text-max-length", type=int, default=160)
    parser.add_argument("--vision-backbone", default="resnet18")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--refresh", action="store_true", help="re-encode even if a cache exists"
    )
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument("--skip-image", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_threads()

    claims = load_built(args.dataset_version)
    features = assemble(
        claims,
        text_config=TextEncoderConfig(
            model_name=args.text_model,
            max_length=args.text_max_length,
            batch_size=args.batch_size,
        ),
        vision_config=VisionEncoderConfig(
            backbone=args.vision_backbone, batch_size=args.batch_size
        ),
        refresh=args.refresh,
        include_text=not args.skip_text,
        include_image=not args.skip_image,
    )

    logger.info("feature dimensions: %s", features.dims)
    report_path = write_json(
        get_paths().reports / f"feature_report_{args.dataset_version}.json",
        {"dataset_version": args.dataset_version, "dims": features.dims,
         "reports": features.reports},
    )
    logger.info("wrote %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
