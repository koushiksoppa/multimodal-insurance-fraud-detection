"""Dataset sources and acquisition.

Every dataset this project uses is declared here with its provenance, so the
origin of a number in the results tables can always be traced back to a
specific third-party artefact. Nothing is vendored into the repository: the
raw data is downloaded on demand and is git-ignored.

Two real datasets are used. There is, to the best of my knowledge, no public
dataset that pairs insurance claim narratives, damage photographs and a fraud
label for the same claim, so the tri-modal corpus has to be assembled. How that
is done -- and which parts are therefore synthetic -- is documented in
``docs/DATASETS.md`` and implemented in :mod:`src.data.build`.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.config import get_paths
from src.utils.io import file_sha256
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DatasetSource:
    """Provenance record for one third-party dataset."""

    key: str
    name: str
    modality: str
    origin: str
    """Where the data originally comes from, not merely where it is hosted."""
    retrieved_from: str
    licence: str
    licence_notes: str
    expected_records: int
    notes: str = ""
    citation: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


STRUCTURED_SOURCE = DatasetSource(
    key="vehicle_claims",
    name="Vehicle Insurance Claim Fraud Detection (carclaims / fraud_oracle)",
    modality="structured",
    origin=(
        "A vehicle insurance claims table distributed with the Angoss "
        "KnowledgeSeeker software and used in the academic insurance-fraud "
        "literature (for example Phua et al., 2004). It circulates publicly "
        "under the file names carclaims.txt, fraud_oracle.csv and "
        "claims_oracle.csv."
    ),
    retrieved_from=(
        "https://raw.githubusercontent.com/ezzaimsoufiane/"
        "Auto-Insurance-Claims-Fraud-Detection-with-ML/master/claims_oracle.csv"
    ),
    licence="Not formally stated by the original distributor.",
    licence_notes=(
        "The original distributor attached no explicit licence. The table is "
        "widely redistributed for research and teaching and contains no "
        "identifiable individuals. This repository does not redistribute it: "
        "the file is downloaded at build time and is git-ignored. Treat it as "
        "research-use-only rather than as freely licensed data."
    ),
    expected_records=15_420,
    notes=(
        "Real claim records with a real, naturally imbalanced fraud label "
        "(FraudFound_P). Includes sex and age-band attributes, which permits a "
        "genuine subgroup fairness analysis rather than a declared omission."
    ),
    citation=(
        "Phua, C., Alahakoon, D., & Lee, V. (2004). Minority report in fraud "
        "detection: classification of skewed data. ACM SIGKDD Explorations, "
        "6(1), 50-59."
    ),
    tags=("real", "tabular", "fraud-label"),
)

IMAGE_SOURCE = DatasetSource(
    key="car_damage",
    name="Comprehensive car damage classification dataset",
    modality="image",
    origin=(
        "Photographs of vehicle damage, manually organised into classes by "
        "damage location and damage type."
    ),
    retrieved_from="https://huggingface.co/datasets/DrBimmer/comprehensive-car-damage",
    licence="MIT",
    licence_notes=(
        "Declared MIT on the dataset card. Images are downloaded at build time "
        "and are not redistributed by this repository."
    ),
    expected_records=2_300,
    notes=(
        "Six classes formed by damage location (F = front, R = rear) crossed "
        "with condition (Normal, Breakage, Crushed). These are real, "
        "human-assigned labels and are used directly as the damage "
        "intelligence targets; no damage label in this project is invented."
    ),
    tags=("real", "image", "damage-label"),
)

#: Folder name -> (location, condition).
IMAGE_CLASS_MAP: dict[str, tuple[str, str]] = {
    "F_Normal": ("front", "normal"),
    "F_Breakage": ("front", "breakage"),
    "F_Crushed": ("front", "crushed"),
    "R_Normal": ("rear", "normal"),
    "R_Breakage": ("rear", "breakage"),
    "R_Crushed": ("rear", "crushed"),
}

#: Ordinal severity used by the damage-intelligence head. "Breakage" (a broken
#: component such as a lamp or bumper) is treated as less severe than "Crushed"
#: (structural deformation of a panel), which matches how the two classes
#: actually look in the source images.
SEVERITY_ORDER: dict[str, int] = {"normal": 0, "breakage": 1, "crushed": 2}

SOURCES: dict[str, DatasetSource] = {
    STRUCTURED_SOURCE.key: STRUCTURED_SOURCE,
    IMAGE_SOURCE.key: IMAGE_SOURCE,
}


# --------------------------------------------------------------------------
# acquisition
# --------------------------------------------------------------------------


def structured_csv_path() -> Path:
    return get_paths().raw_structured / "claims_oracle.csv"


def image_root_path() -> Path:
    return get_paths().raw_images / "comprehensive-car-damage"


def download_structured(*, force: bool = False, timeout: int = 120) -> Path:
    """Download the structured claims table. Returns the local path."""
    import requests

    target = structured_csv_path()
    if target.exists() and not force:
        logger.info("structured data already present at %s", target)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("downloading structured claims table ...")
    response = requests.get(STRUCTURED_SOURCE.retrieved_from, timeout=timeout)
    response.raise_for_status()

    # Write via a temporary file so an interrupted download cannot leave a
    # truncated CSV that would later look like a valid but short dataset.
    tmp = target.with_suffix(".csv.part")
    tmp.write_bytes(response.content)
    tmp.replace(target)

    logger.info(
        "saved %s (%.1f MB, sha256=%s)",
        target,
        target.stat().st_size / 1e6,
        file_sha256(target)[:16],
    )
    return target


def download_images(*, force: bool = False) -> Path:
    """Download the car damage image dataset from the Hugging Face Hub."""
    from huggingface_hub import snapshot_download

    target = image_root_path()
    if target.exists() and any(target.rglob("*.jpg")) and not force:
        logger.info("image data already present at %s", target)
        return target

    if force and target.exists():
        shutil.rmtree(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("downloading car damage images from the Hugging Face Hub ...")
    snapshot_download(
        repo_id="DrBimmer/comprehensive-car-damage",
        repo_type="dataset",
        local_dir=str(target),
        allow_patterns=["*.jpg", "*.jpeg", "*.png", "README.md"],
    )
    count = sum(1 for _ in iter_image_files(target))
    logger.info("saved %d images under %s", count, target)
    return target


def iter_image_files(root: Path | None = None):
    """Yield ``(path, folder_label)`` for every image in the damage dataset."""
    root = root or image_root_path()
    for folder in sorted(IMAGE_CLASS_MAP):
        folder_path = root / folder
        if not folder_path.is_dir():
            continue
        for path in sorted(folder_path.iterdir()):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                yield path, folder


def download_all(*, force: bool = False) -> dict[str, Path]:
    get_paths().ensure()
    return {
        "structured": download_structured(force=force),
        "images": download_images(force=force),
    }
