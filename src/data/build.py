"""Assemble the unified multimodal claim dataset.

This is the single place where the real structured table, the constructed
narratives and the real damage photographs are brought together into one
claim-level frame, split, audited and written to disk. Everything downstream --
feature extraction, every model, the API, the database -- reads the output of
this module and never reaches back to the raw sources.

The build is deterministic. Given the same seed and the same raw inputs it
produces byte-identical output, which is what makes the experiment results
reproducible rather than merely repeatable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.data import quality, schema
from src.data.linkage import ImageInventory, attach_images, build_image_inventory
from src.data.splits import SplitConfig, assign_splits, audit_splits, raise_on_leakage
from src.data.structured import load_unified
from src.data.text_generation import DEFAULT_CONFIG as TEXT_CONFIG
from src.data.text_generation import GenerationConfig, generate_narratives
from src.utils.config import get_paths
from src.utils.io import write_json, write_table
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Bumped whenever the construction changes in a way that alters the data.
#: Recorded on every row and in every experiment record, so a result can always
#: be traced to the dataset it was produced from.
DATASET_VERSION = "real-v1"


@dataclass
class BuildConfig:
    seed: int = 20260921
    split: SplitConfig = None  # type: ignore[assignment]
    text: GenerationConfig = TEXT_CONFIG
    verify_images: bool = True
    prevalence_preserving_linkage: bool = True

    def __post_init__(self) -> None:
        if self.split is None:
            self.split = SplitConfig(seed=self.seed)


def build_real_dataset(
    config: BuildConfig | None = None,
    *,
    write: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Build the real-label track and return the frame plus a build report."""
    config = config or BuildConfig()
    paths = get_paths().ensure()

    schema.assert_disjoint_channels()

    # --- structured -------------------------------------------------------
    claims, load_report = load_unified()
    logger.info("loaded %d claims from the structured source", len(claims))

    # --- narratives -------------------------------------------------------
    claims["claim_text"] = generate_narratives(
        claims, base_seed=config.seed, config=config.text
    )

    # --- images -----------------------------------------------------------
    inventory = build_image_inventory(verify=config.verify_images)
    claims, linkage_report = attach_images(
        claims,
        inventory,
        base_seed=config.seed,
        prevalence_preserving=config.prevalence_preserving_linkage,
    )

    # --- splits -----------------------------------------------------------
    claims["split"], split_report = assign_splits(claims, config.split)
    claims["track"] = "real"
    claims["dataset_version"] = DATASET_VERSION

    # --- audits -----------------------------------------------------------
    findings = audit_splits(claims, strategy=config.split.strategy)
    raise_on_leakage(findings)

    quality_report = quality.run_all(
        claims,
        image_root=inventory.root,
        all_image_ids=inventory.frame["image_id"].tolist(),
        verify_images=False,  # already verified while building the inventory
    )
    quality_report.raise_on_error()

    report = {
        "dataset_version": DATASET_VERSION,
        "track": "real",
        "seed": config.seed,
        "n_claims": len(claims),
        "n_columns": claims.shape[1],
        "structured_load": load_report.as_dict(),
        "linkage": linkage_report.as_dict(),
        "image_inventory": {
            "n_usable": len(inventory),
            "n_unreadable": inventory.n_unreadable,
            "unreadable_examples": inventory.unreadable_examples,
            "class_counts": inventory.frame["damage_class"].value_counts().to_dict(),
        },
        "split": split_report.as_dict(),
        "leakage_audit": [
            {"name": f.name, "passed": f.passed, "message": f.message, **f.details}
            for f in findings
        ],
        "quality": quality_report.as_dict(),
        "text_generation": asdict(config.text),
    }

    if write:
        table_path = write_table(claims, paths.processed / f"claims_{DATASET_VERSION}.parquet")
        report_path = write_json(
            paths.reports / f"build_report_{DATASET_VERSION}.json", report
        )
        logger.info("wrote %s", table_path)
        logger.info("wrote %s", report_path)
        report["paths"] = {"table": str(table_path), "report": str(report_path)}

    return claims, report


def load_built(version: str = DATASET_VERSION) -> pd.DataFrame:
    """Read a previously built dataset."""
    path = get_paths().processed / f"claims_{version}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/build_dataset.py"
        )
    return pd.read_parquet(path)


def image_root_for(df: pd.DataFrame) -> Path:
    """The directory that ``image_path`` values are relative to."""
    from src.data.sources import image_root_path

    del df  # signature kept for symmetry; the root is fixed by configuration
    return image_root_path()


__all__ = [
    "DATASET_VERSION",
    "BuildConfig",
    "ImageInventory",
    "build_real_dataset",
    "load_built",
    "image_root_for",
]
