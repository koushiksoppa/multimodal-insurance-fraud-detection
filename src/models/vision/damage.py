"""Damage intelligence: the secondary vision task.

This is the part of the project where the images carry labels that are real,
human-assigned and genuinely about what is in the photograph. Six classes,
formed by damage location (front, rear) crossed with condition (normal,
breakage, crushed). No label here is invented.

It is deliberately kept secondary to the fraud task. It is an auxiliary output
of the system -- "what does the photo show?" -- not evidence about whether a
claim is fraudulent, and on the real-label track it cannot be, because the
photographs were never attached to these claims in reality.

Two training regimes are implemented and compared:

``frozen``
    ImageNet features extracted once and cached, with a small MLP head trained
    on top. Seconds per epoch on CPU.

``finetune_last_block``
    The final residual block is unfrozen and trained along with the head, at a
    reduced learning rate. Minutes per epoch on CPU rather than seconds, so it
    is run once rather than swept.

The comparison is the point. With roughly 380 images per class, unfreezing more
of the network risks overfitting faster than it gains representational power,
and the honest way to find out is to measure both rather than to assert which
is better.

Splits follow the claim splits, not a separate image split. Using one split
regime for everything means an image can never be in the fraud model's training
data and the damage model's test data at once, which keeps the leakage
guarantee intact across the whole project rather than only within one task.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.sources import IMAGE_CLASS_MAP, SEVERITY_ORDER
from src.models.vision.encoder import VisionEncoderConfig, build_backbone, build_transform
from src.utils.config import configure_threads, resolve_device
from src.utils.logging import get_logger

logger = get_logger(__name__)

CLASS_NAMES = sorted(IMAGE_CLASS_MAP)
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
LOCATIONS = ["front", "rear"]
CONDITIONS = ["normal", "breakage", "crushed"]


@dataclass
class DamageConfig:
    regime: str = "frozen"
    """frozen or finetune_last_block."""

    backbone: str = "resnet18"
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    backbone_learning_rate: float = 1e-4
    """Lower than the head's: the pretrained filters need nudging, not retraining."""

    weight_decay: float = 1e-4
    dropout: float = 0.3
    hidden_dim: int = 128
    patience: int = 6
    augment: bool = True
    seed: int = 20260921
    device: str = "auto"

    def as_dict(self) -> dict:
        return {
            "regime": self.regime,
            "backbone": self.backbone,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "backbone_learning_rate": self.backbone_learning_rate,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "hidden_dim": self.hidden_dim,
            "augment": self.augment,
            "seed": self.seed,
        }


class DamageImageDataset(Dataset):
    """Images plus their real damage labels."""

    def __init__(
        self,
        image_paths: list[Path],
        class_labels: np.ndarray,
        transform,
    ) -> None:
        self.image_paths = image_paths
        self.class_labels = np.asarray(class_labels, dtype="int64")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        from PIL import Image

        with Image.open(self.image_paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(self.class_labels[index])


class DamageHead(nn.Module):
    """Classification head over backbone features."""

    def __init__(self, in_dim: int, n_classes: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DamageResult:
    regime: str
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    location_accuracy: float
    condition_accuracy: float
    condition_macro_f1: float
    severity_mae: float
    """Mean absolute error on the ordinal severity scale (0, 1, 2).

    Reported alongside accuracy because the classes are ordered: predicting
    'crushed' when the truth is 'normal' is a worse error than predicting
    'breakage', and plain accuracy cannot express that.
    """

    per_class: dict = field(default_factory=dict)
    confusion: list[list[int]] = field(default_factory=list)
    class_names: list[str] = field(default_factory=lambda: list(CLASS_NAMES))
    history: dict = field(default_factory=dict)
    n_train: int = 0
    n_test: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return self.__dict__.copy()

    def summary(self) -> str:
        return (
            f"{self.regime}: accuracy {self.accuracy:.4f} | balanced "
            f"{self.balanced_accuracy:.4f} | macro-F1 {self.macro_f1:.4f} | "
            f"location {self.location_accuracy:.4f} | condition "
            f"{self.condition_accuracy:.4f} | severity MAE {self.severity_mae:.4f}"
        )


class DamageModel:
    """Train and evaluate the damage classifier."""

    def __init__(self, config: DamageConfig | None = None) -> None:
        self.config = config or DamageConfig()
        self.device = resolve_device(self.config.device)
        self.backbone = None
        self.head: DamageHead | None = None
        self.history: dict = {"train_loss": [], "val_loss": [], "val_macro_f1": []}
        configure_threads()

    def _build(self) -> None:
        torch.manual_seed(self.config.seed)
        backbone, dim = build_backbone(self.config.backbone)

        finetuning = self.config.regime == "finetune_last_block"
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)

        if finetuning:
            # Only the last residual block. The early layers encode generic
            # edges and textures that 2,300 images cannot improve on, and
            # unfreezing them on this much data mostly produces overfitting.
            last_block = getattr(backbone, "layer4", None)
            if last_block is None:  # efficientnet and friends
                last_block = list(backbone.children())[-2]
            for parameter in last_block.parameters():
                parameter.requires_grad_(True)

        self.backbone = backbone.to(self.device)
        self.head = DamageHead(
            dim, len(CLASS_NAMES), self.config.hidden_dim, self.config.dropout
        ).to(self.device)

    def _loaders(self, splits: dict[str, DamageImageDataset]) -> dict[str, DataLoader]:
        return {
            name: DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=(name == "train"),
                num_workers=0,
            )
            for name, dataset in splits.items()
        }

    def fit(
        self, train: DamageImageDataset, val: DamageImageDataset
    ) -> DamageModel:
        self._build()
        started = time.perf_counter()

        finetuning = self.config.regime == "finetune_last_block"
        groups: list[dict] = [
            {"params": self.head.parameters(), "lr": self.config.learning_rate}
        ]
        if finetuning:
            trainable = [p for p in self.backbone.parameters() if p.requires_grad]
            groups.append(
                {"params": trainable, "lr": self.config.backbone_learning_rate}
            )

        optimiser = torch.optim.AdamW(groups, weight_decay=self.config.weight_decay)
        criterion = nn.CrossEntropyLoss()
        loaders = self._loaders({"train": train, "val": val})

        best_state = None
        best_score = -np.inf
        stale = 0

        for epoch in range(1, self.config.epochs + 1):
            self.head.train()
            self.backbone.train() if finetuning else self.backbone.eval()

            total, seen = 0.0, 0
            for images, labels in loaders["train"]:
                images, labels = images.to(self.device), labels.to(self.device)
                optimiser.zero_grad(set_to_none=True)

                if finetuning:
                    features = self.backbone(images)
                else:
                    with torch.no_grad():
                        features = self.backbone(images)

                loss = criterion(self.head(features), labels)
                loss.backward()
                optimiser.step()
                total += loss.item() * len(labels)
                seen += len(labels)

            val_pred, val_true, val_loss = self._predict(loaders["val"], criterion)
            macro_f1 = float(f1_score(val_true, val_pred, average="macro", zero_division=0))

            self.history["train_loss"].append(total / max(seen, 1))
            self.history["val_loss"].append(val_loss)
            self.history["val_macro_f1"].append(macro_f1)

            if macro_f1 > best_score:
                best_score = macro_f1
                best_state = (
                    copy.deepcopy(self.head.state_dict()),
                    copy.deepcopy(self.backbone.state_dict()) if finetuning else None,
                )
                stale = 0
            else:
                stale += 1

            logger.info(
                "epoch %2d | train loss %.4f | val loss %.4f | val macro-F1 %.4f",
                epoch, self.history["train_loss"][-1], val_loss, macro_f1,
            )

            if stale >= self.config.patience:
                logger.info("early stop at epoch %d (best macro-F1 %.4f)", epoch, best_score)
                break

        if best_state is not None:
            self.head.load_state_dict(best_state[0])
            if best_state[1] is not None:
                self.backbone.load_state_dict(best_state[1])

        self.history["seconds"] = time.perf_counter() - started
        logger.info("damage model trained in %.1fs", self.history["seconds"])
        return self

    def _predict(self, loader: DataLoader, criterion=None):
        self.head.eval()
        self.backbone.eval()
        predictions, truths = [], []
        total, seen = 0.0, 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.head(self.backbone(images))
                if criterion is not None:
                    total += criterion(logits, labels).item() * len(labels)
                    seen += len(labels)
                predictions.append(logits.argmax(dim=1).cpu().numpy())
                truths.append(labels.cpu().numpy())
        return (
            np.concatenate(predictions),
            np.concatenate(truths),
            total / max(seen, 1),
        )

    def predict(self, dataset: DamageImageDataset) -> np.ndarray:
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        predictions, _, _ = self._predict(loader)
        return predictions

    def predict_proba(self, dataset: DamageImageDataset) -> np.ndarray:
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        self.head.eval()
        self.backbone.eval()
        out = []
        with torch.no_grad():
            for images, _ in loader:
                logits = self.head(self.backbone(images.to(self.device)))
                out.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(out)

    def evaluate(self, dataset: DamageImageDataset) -> DamageResult:
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        predicted, truth, _ = self._predict(loader)
        return summarise_predictions(
            truth, predicted, regime=self.config.regime, history=self.history
        )


def summarise_predictions(
    truth: np.ndarray,
    predicted: np.ndarray,
    *,
    regime: str,
    history: dict | None = None,
) -> DamageResult:
    """Score six-class predictions, and the location and condition axes separately.

    The six classes are a product of two attributes, so overall accuracy hides
    which axis the model is actually getting wrong. Decomposing it says whether
    the model can tell front from rear, whether it can tell a scratch from a
    crushed panel, or neither.
    """
    location_of = {
        CLASS_TO_INDEX[name]: LOCATIONS.index(IMAGE_CLASS_MAP[name][0])
        for name in CLASS_NAMES
    }
    condition_of = {
        CLASS_TO_INDEX[name]: CONDITIONS.index(IMAGE_CLASS_MAP[name][1])
        for name in CLASS_NAMES
    }
    severity_of = {
        CLASS_TO_INDEX[name]: SEVERITY_ORDER[IMAGE_CLASS_MAP[name][1]]
        for name in CLASS_NAMES
    }

    true_location = np.array([location_of[t] for t in truth])
    pred_location = np.array([location_of[p] for p in predicted])
    true_condition = np.array([condition_of[t] for t in truth])
    pred_condition = np.array([condition_of[p] for p in predicted])
    true_severity = np.array([severity_of[t] for t in truth], dtype=float)
    pred_severity = np.array([severity_of[p] for p in predicted], dtype=float)

    report = classification_report(
        truth, predicted, labels=range(len(CLASS_NAMES)),
        target_names=CLASS_NAMES, output_dict=True, zero_division=0,
    )

    return DamageResult(
        regime=regime,
        accuracy=float(accuracy_score(truth, predicted)),
        balanced_accuracy=float(balanced_accuracy_score(truth, predicted)),
        macro_f1=float(f1_score(truth, predicted, average="macro", zero_division=0)),
        location_accuracy=float(accuracy_score(true_location, pred_location)),
        condition_accuracy=float(accuracy_score(true_condition, pred_condition)),
        condition_macro_f1=float(
            f1_score(true_condition, pred_condition, average="macro", zero_division=0)
        ),
        severity_mae=float(np.abs(true_severity - pred_severity).mean()),
        per_class={
            name: report[name] for name in CLASS_NAMES if name in report
        },
        confusion=confusion_matrix(
            truth, predicted, labels=range(len(CLASS_NAMES))
        ).tolist(),
        history=history or {},
        n_test=len(truth),
    )


def build_damage_datasets(
    claims: pd.DataFrame,
    image_root: Path,
    *,
    config: DamageConfig | None = None,
) -> dict[str, DamageImageDataset]:
    """Build train/val/test damage datasets from the claim splits.

    Only claims with an attached photograph contribute. Because splits
    partition claims and each photograph belongs to exactly one claim, the
    damage splits inherit the same disjointness guarantee.
    """
    config = config or DamageConfig()
    encoder_config = VisionEncoderConfig(backbone=config.backbone)

    datasets: dict[str, DamageImageDataset] = {}
    for split in ("train", "val", "test"):
        part = claims[(claims["split"] == split) & claims["has_image"]]
        paths = [image_root / rel for rel in part["image_path"]]
        labels = np.array([CLASS_TO_INDEX[c] for c in part["damage_class"]])
        datasets[split] = DamageImageDataset(
            paths,
            labels,
            build_transform(
                encoder_config, train=(split == "train" and config.augment)
            ),
        )
        logger.info("damage %s split: %d images", split, len(paths))

    return datasets
