"""Attach damage photographs to claims.

The photographs and the claim records come from different sources: nobody has
published a corpus where a fraud-labelled claim ships with its own damage
photo. Attachment therefore has to be constructed, and how it is constructed
determines what the image arm of the experiment actually measures.

The rule used on the real-label track is deliberately **label-independent and
prevalence-preserving**: a photograph is drawn uniformly at random and attached
to a claim drawn uniformly at random within each class, with the covered
fraction held equal across classes. Consequences, stated plainly:

* The photograph is not a photograph of that claim's vehicle. It carries real
  damage labels, but no information about whether *that* claim was fraudulent.
* The image arm on the real-label track is therefore a **negative control**.
  A correctly implemented fusion model should extract no fraud signal from it.
  If it appears to, something is wrong -- a leak, a bug, or an over-fitted
  evaluation -- and the ablation table is the place that would show up.
* Because the covered fraction is equal across classes, fraud prevalence in the
  image-covered subset matches prevalence in the full table, so precision-recall
  metrics remain comparable between the two.

Whether a fusion architecture can exploit image information *when the
information is really there* is a separate question, answered on the synthetic
track where the generating process puts it there on purpose.

Each photograph is attached to at most one claim. Nothing is reused, so an
image cannot appear on both sides of a split by construction -- the strongest
available guarantee against image leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.sources import IMAGE_CLASS_MAP, SEVERITY_ORDER, image_root_path, iter_image_files
from src.utils.logging import get_logger
from src.utils.seeding import derive_seed

logger = get_logger(__name__)


@dataclass
class ImageInventory:
    """Every usable photograph, with its real damage labels."""

    frame: pd.DataFrame
    root: Path
    n_unreadable: int = 0
    unreadable_examples: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frame)


def build_image_inventory(
    root: Path | None = None, *, verify: bool = True
) -> ImageInventory:
    """Catalogue the image corpus, dropping files that do not decode.

    Verification happens here, once, rather than during training: a corrupted
    JPEG discovered mid-epoch on a CPU-bound machine is an expensive way to
    learn about a bad file.
    """
    root = root or image_root_path()
    if not root.exists():
        raise FileNotFoundError(
            f"{root} not found. Run: python scripts/download_data.py --only images"
        )

    records: list[dict] = []
    unreadable: list[str] = []

    verifier = None
    if verify:
        from PIL import Image

        verifier = Image

    for path, folder in iter_image_files(root):
        rel = path.relative_to(root).as_posix()
        if verifier is not None:
            try:
                with verifier.open(path) as img:
                    img.verify()
            except Exception as exc:  # noqa: BLE001 - any decode failure disqualifies
                unreadable.append(f"{rel}: {type(exc).__name__}")
                continue

        location, condition = IMAGE_CLASS_MAP[folder]
        records.append(
            {
                "image_id": rel.replace("/", "__"),
                "image_path": rel,
                "damage_class": folder,
                "damage_location": location,
                "damage_condition": condition,
                "damage_severity": SEVERITY_ORDER[condition],
            }
        )

    frame = pd.DataFrame.from_records(records)
    if unreadable:
        logger.warning("%d image files failed to decode and were excluded", len(unreadable))
    logger.info(
        "image inventory: %d usable images across %d classes",
        len(frame),
        frame["damage_class"].nunique() if len(frame) else 0,
    )
    return ImageInventory(
        frame=frame,
        root=root,
        n_unreadable=len(unreadable),
        unreadable_examples=unreadable[:10],
    )


@dataclass
class LinkageReport:
    n_claims: int
    n_images_available: int
    n_attached: int
    coverage: float
    coverage_by_class: dict[str, float]
    prevalence_full: float
    prevalence_covered: float
    strategy: str

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def attach_images(
    claims: pd.DataFrame,
    inventory: ImageInventory,
    *,
    base_seed: int,
    label_column: str = "fraud_label",
    prevalence_preserving: bool = True,
) -> tuple[pd.DataFrame, LinkageReport]:
    """Attach photographs to a subset of claims.

    Parameters
    ----------
    prevalence_preserving:
        Attach the same *fraction* of claims within each label class, so the
        image-covered subset has the same fraud prevalence as the full table.
        Turning this off attaches uniformly at random across all claims, which
        leaves prevalence to sampling noise.
    """
    generator = np.random.default_rng(derive_seed(base_seed, "image-linkage"))

    n_images = len(inventory)
    n_claims = len(claims)
    if n_images == 0:
        raise ValueError("image inventory is empty")
    if n_images > n_claims:
        raise ValueError(
            f"{n_images} images but only {n_claims} claims; each image is attached "
            "to at most one claim, so the inventory cannot exceed the claim count"
        )

    labels = claims[label_column].to_numpy()

    if prevalence_preserving:
        chosen_positions = _stratified_positions(labels, n_images, generator)
        strategy = "stratified, prevalence-preserving, no image reuse"
    else:
        chosen_positions = generator.choice(n_claims, size=n_images, replace=False)
        strategy = "uniform at random, no image reuse"

    # Shuffle the image order so that attachment is not correlated with the
    # inventory's class ordering.
    image_order = generator.permutation(n_images)
    images = inventory.frame.iloc[image_order].reset_index(drop=True)

    out = claims.copy()
    for col, dtype in (
        ("image_id", object),
        ("image_path", object),
        ("damage_class", object),
        ("damage_location", object),
        ("damage_condition", object),
    ):
        out[col] = pd.Series([None] * n_claims, index=out.index, dtype=dtype)
    out["damage_severity"] = pd.Series(
        [np.nan] * n_claims, index=out.index, dtype="float64"
    )

    positions = np.sort(chosen_positions)
    target_index = out.index[positions]
    for col in ("image_id", "image_path", "damage_class", "damage_location",
                "damage_condition"):
        out.loc[target_index, col] = images[col].to_numpy()
    out.loc[target_index, "damage_severity"] = images["damage_severity"].to_numpy()

    out["has_image"] = out["image_id"].notna()

    covered = out[out["has_image"]]
    coverage_by_class = {
        str(int(cls)): float((out[label_column] == cls).pipe(
            lambda mask: out.loc[mask, "has_image"].mean()
        ))
        for cls in sorted(pd.unique(labels))
    }

    report = LinkageReport(
        n_claims=n_claims,
        n_images_available=n_images,
        n_attached=int(out["has_image"].sum()),
        coverage=float(out["has_image"].mean()),
        coverage_by_class=coverage_by_class,
        prevalence_full=float(out[label_column].mean()),
        prevalence_covered=float(covered[label_column].mean()) if len(covered) else 0.0,
        strategy=strategy,
    )

    logger.info(
        "attached %d images to %d claims (%.1f%% coverage); prevalence full %.4f, "
        "covered %.4f",
        report.n_attached, n_claims, 100 * report.coverage,
        report.prevalence_full, report.prevalence_covered,
    )
    return out, report


def _stratified_positions(
    labels: np.ndarray, n_to_attach: int, generator: np.random.Generator
) -> np.ndarray:
    """Choose row positions so each class is covered at the same rate.

    The per-class quota is rounded with a largest-remainder pass, so the quotas
    sum exactly to ``n_to_attach`` without biasing any class.
    """
    classes = np.unique(labels)
    counts = {c: int((labels == c).sum()) for c in classes}
    total = len(labels)

    exact = {c: n_to_attach * counts[c] / total for c in classes}
    quota = {c: int(np.floor(v)) for c, v in exact.items()}

    shortfall = n_to_attach - sum(quota.values())
    if shortfall:
        remainders = sorted(classes, key=lambda c: exact[c] - quota[c], reverse=True)
        for c in remainders[:shortfall]:
            quota[c] += 1

    positions: list[np.ndarray] = []
    for c in classes:
        class_positions = np.flatnonzero(labels == c)
        take = min(quota[c], len(class_positions))
        positions.append(generator.choice(class_positions, size=take, replace=False))

    return np.concatenate(positions)
