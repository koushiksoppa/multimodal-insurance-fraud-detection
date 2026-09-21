"""Assembling modality feature blocks into tensors.

The feature blocks are prepared once for the whole dataset and then sliced per
split, because the expensive parts -- transformer and CNN forward passes --
are cached upstream and must not be repeated seven times for seven ablation
cells.

The presence mask is the mechanism that makes missing modalities work. Rather
than dropping claims that lack a photograph, or training a separate model per
modality combination, every claim carries a bit per modality saying whether
that modality is real. A missing block is filled with zeros and its bit is set
to zero, and the network is told. This is what lets one trained model serve a
claim with a photo and a claim without one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.multimodal.fusion import MODALITIES
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModalityBlock:
    """One modality's features for every claim, with a presence mask."""

    name: str
    features: np.ndarray
    """Shape (n_claims, dim). Rows for absent claims are zero."""

    present: np.ndarray
    """Shape (n_claims,), boolean."""

    feature_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.features) != len(self.present):
            raise ValueError(
                f"{self.name}: {len(self.features)} feature rows but "
                f"{len(self.present)} mask entries"
            )
        # Absent rows must be exactly zero; otherwise a stale value could reach
        # the network behind a mask that says the modality is missing.
        if self.present.dtype != bool:
            self.present = self.present.astype(bool)
        self.features = np.asarray(self.features, dtype="float32")
        self.features[~self.present] = 0.0

    @property
    def dim(self) -> int:
        return int(self.features.shape[1])

    def subset(self, index: np.ndarray) -> ModalityBlock:
        return ModalityBlock(
            name=self.name,
            features=self.features[index],
            present=self.present[index],
            feature_names=self.feature_names,
        )

    def coverage(self) -> float:
        return float(self.present.mean())


class MultimodalDataset(Dataset):
    """Torch dataset over a set of modality blocks."""

    def __init__(
        self,
        blocks: dict[str, ModalityBlock],
        labels: np.ndarray,
        claim_ids: np.ndarray | pd.Series | None = None,
    ) -> None:
        if not blocks:
            raise ValueError("at least one modality block is required")
        lengths = {len(b.features) for b in blocks.values()} | {len(labels)}
        if len(lengths) != 1:
            raise ValueError(f"inconsistent lengths across blocks and labels: {lengths}")

        self.blocks = blocks
        self.labels = np.asarray(labels, dtype="float32")
        self.claim_ids = (
            np.asarray(claim_ids) if claim_ids is not None else np.arange(len(labels))
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        features = {
            name: torch.from_numpy(block.features[index])
            for name, block in self.blocks.items()
        }
        masks = {
            name: torch.tensor(bool(block.present[index]))
            for name, block in self.blocks.items()
        }
        return features, masks, torch.tensor(self.labels[index])

    @property
    def input_dims(self) -> dict[str, int]:
        return {name: block.dim for name, block in self.blocks.items()}

    def coverage(self) -> dict[str, float]:
        return {name: block.coverage() for name, block in self.blocks.items()}


def collate(batch):
    """Stack a batch, keeping the per-modality dict structure."""
    features_list, masks_list, labels = zip(*batch, strict=True)
    names = features_list[0].keys()
    features = {
        name: torch.stack([f[name] for f in features_list]) for name in names
    }
    masks = {name: torch.stack([m[name] for m in masks_list]) for name in names}
    return features, masks, torch.stack(labels)


def make_loader(
    dataset: MultimodalDataset,
    *,
    batch_size: int = 256,
    shuffle: bool = False,
    seed: int | None = None,
) -> DataLoader:
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        generator=generator,
        # Single-process loading: the blocks are already in memory as arrays,
        # so worker processes would add startup cost and memory pressure on an
        # 8 GB machine without speeding anything up.
        num_workers=0,
    )


def build_blocks(
    claims: pd.DataFrame,
    *,
    tabular: np.ndarray | None = None,
    tabular_names: list[str] | None = None,
    text: np.ndarray | None = None,
    image: np.ndarray | None = None,
    modalities: list[str] | None = None,
) -> dict[str, ModalityBlock]:
    """Assemble the requested modality blocks, aligned to ``claims``.

    Presence is derived from the data itself: text is present for every claim
    because a narrative is always generated, while an image is present only
    where one was attached, which is where the missing-modality behaviour comes
    from on real data rather than from an artificial ablation.
    """
    requested = modalities or list(MODALITIES)
    n = len(claims)
    blocks: dict[str, ModalityBlock] = {}

    if "tabular" in requested:
        if tabular is None:
            raise ValueError("tabular features requested but not supplied")
        blocks["tabular"] = ModalityBlock(
            name="tabular",
            features=np.asarray(tabular),
            present=np.ones(n, dtype=bool),
            feature_names=tabular_names or [],
        )

    if "text" in requested:
        if text is None:
            raise ValueError("text features requested but not supplied")
        has_text = (
            claims["claim_text"].fillna("").str.strip().ne("").to_numpy()
            if "claim_text" in claims.columns
            else np.ones(n, dtype=bool)
        )
        blocks["text"] = ModalityBlock(
            name="text", features=np.asarray(text), present=has_text
        )

    if "image" in requested:
        if image is None:
            raise ValueError("image features requested but not supplied")
        has_image = (
            claims["has_image"].to_numpy().astype(bool)
            if "has_image" in claims.columns
            else np.ones(n, dtype=bool)
        )
        blocks["image"] = ModalityBlock(
            name="image", features=np.asarray(image), present=has_image
        )

    logger.info(
        "modality blocks: %s",
        ", ".join(
            f"{k} dim={v.dim} coverage={v.coverage():.3f}" for k, v in blocks.items()
        ),
    )
    return blocks


def subset_blocks(
    blocks: dict[str, ModalityBlock], index: np.ndarray
) -> dict[str, ModalityBlock]:
    return {name: block.subset(index) for name, block in blocks.items()}
