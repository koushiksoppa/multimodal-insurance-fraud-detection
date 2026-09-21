"""Download the raw datasets this project depends on.

Usage::

    python scripts/download_data.py
    python scripts/download_data.py --force
    python scripts/download_data.py --only structured

The downloaded files land under ``data/raw/`` and are git-ignored. Running the
script twice is cheap: existing files are left alone unless ``--force`` is
given.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running the script directly, without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import sources  # noqa: E402
from src.utils.config import get_paths  # noqa: E402
from src.utils.io import file_sha256, write_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger("download_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=["structured", "images"],
        help="download a single modality instead of everything",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even if the files are already present",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = get_paths().ensure()

    acquired: dict[str, Path] = {}
    if args.only in (None, "structured"):
        acquired["structured"] = sources.download_structured(force=args.force)
    if args.only in (None, "images"):
        acquired["images"] = sources.download_images(force=args.force)

    manifest = {
        "sources": {
            key: {
                "name": src.name,
                "modality": src.modality,
                "origin": src.origin,
                "retrieved_from": src.retrieved_from,
                "licence": src.licence,
                "licence_notes": src.licence_notes,
                "expected_records": src.expected_records,
                "citation": src.citation,
            }
            for key, src in sources.SOURCES.items()
        },
        # Recorded as POSIX so the manifest reads the same on any platform.
        "local_paths": {
            k: v.relative_to(paths.root).as_posix() for k, v in acquired.items()
        },
    }

    if "structured" in acquired:
        manifest["structured_sha256"] = file_sha256(acquired["structured"])
    if "images" in acquired:
        manifest["image_count"] = sum(1 for _ in sources.iter_image_files())

    target = write_json(paths.raw / "manifest.json", manifest)
    logger.info("wrote acquisition manifest to %s", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
