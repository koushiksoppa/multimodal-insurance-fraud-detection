"""Training loop for the fusion models.

Written by hand rather than pulled from a framework: the loop is fifty lines,
it has to run on four CPU cores, and the three things that matter here --
imbalance handling, model selection, and reproducibility -- are all decisions I
want visible rather than buried in a trainer's configuration.

Imbalance is handled with a positive class weight in the loss rather than by
resampling, and the weight is a tunable rather than being pinned to the
inverse base rate. Full inverse weighting maximises recall but wrecks
calibration, which this project measures; the default is a square-root
tempering that recovers most of the ranking benefit at a fraction of the
calibration cost. The resulting trade-off is reported, not asserted.

Model selection is on validation PR-AUC, not validation loss. At a six percent
base rate the loss is dominated by the negatives, so a checkpoint with a better
loss is frequently a worse ranker, which is the only thing this model is for.

The test split is never touched by this module.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from src.models.multimodal.dataset import MultimodalDataset, make_loader
from src.models.multimodal.fusion import FusionConfig, FusionModel
from src.utils.config import configure_threads, resolve_device
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    #: Exponent applied to the inverse class ratio. 0 disables weighting,
    #: 1 is full inverse weighting, 0.5 is the tempered default.
    class_weight_power: float = 0.5

    #: Stop when validation PR-AUC has not improved for this many epochs.
    patience: int = 10

    #: Probability of masking out each present modality during training.
    #:
    #: Without this, a model whose text and tabular blocks are always present
    #: at training time has never seen them absent, so asking it to score a
    #: claim with a missing narrative is an out-of-distribution request and the
    #: missing-modality experiment measures nothing but that surprise. Dropping
    #: modalities at random during training makes the degradation meaningful.
    #: The image modality needs no help here -- it is genuinely absent for 85%
    #: of real claims -- but the mechanism applies uniformly.
    #:
    #: Off by default so the ablation reports an unmodified architecture, and
    #: turned on for the missing-modality study, where the two are compared.
    modality_dropout: float = 0.0

    grad_clip: float = 1.0
    seed: int = 20260921
    device: str = "auto"
    verbose: bool = True

    def as_dict(self) -> dict:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "class_weight_power": self.class_weight_power,
            "patience": self.patience,
            "modality_dropout": self.modality_dropout,
            "grad_clip": self.grad_clip,
            "seed": self.seed,
        }


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_pr_auc: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_pr_auc: float = 0.0
    seconds: float = 0.0
    stopped_early: bool = False

    def as_dict(self) -> dict:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_pr_auc": self.val_pr_auc,
            "best_epoch": self.best_epoch,
            "best_val_pr_auc": self.best_val_pr_auc,
            "seconds": self.seconds,
            "stopped_early": self.stopped_early,
            "n_epochs_run": len(self.train_loss),
        }


def positive_weight(labels: np.ndarray, power: float) -> float:
    """Loss weight for the positive class.

    ``power`` interpolates between no weighting (0) and full inverse-frequency
    weighting (1). At this dataset's roughly 16:1 ratio, full weighting pushes
    mean predicted probability far above the base rate; the square-root default
    keeps the ranking benefit while leaving the probabilities interpretable.
    """
    labels = np.asarray(labels)
    n_positive = int(labels.sum())
    n_negative = len(labels) - n_positive
    if n_positive == 0 or power == 0:
        return 1.0
    return float((n_negative / n_positive) ** power)


class FusionTrainer:
    """Fit a :class:`FusionModel` and score datasets with it."""

    def __init__(
        self,
        fusion_config: FusionConfig,
        train_config: TrainConfig | None = None,
    ) -> None:
        self.fusion_config = fusion_config
        self.config = train_config or TrainConfig()
        self.device = resolve_device(self.config.device)
        self.model: FusionModel | None = None
        self.history = TrainHistory()
        configure_threads()

    def fit(
        self, train: MultimodalDataset, val: MultimodalDataset
    ) -> FusionTrainer:
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        model = FusionModel(self.fusion_config).to(self.device)
        logger.info(
            "training %s fusion over %s: %d parameters",
            self.fusion_config.strategy,
            "+".join(self.fusion_config.modalities),
            model.n_parameters(),
        )

        pos_weight = positive_weight(train.labels, self.config.class_weight_power)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, device=self.device)
        )
        optimiser = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=self.config.epochs
        )

        train_loader = make_loader(
            train, batch_size=self.config.batch_size, shuffle=True, seed=self.config.seed
        )
        val_loader = make_loader(val, batch_size=self.config.batch_size)

        best_state = copy.deepcopy(model.state_dict())
        best_score = -np.inf
        epochs_without_improvement = 0
        started = time.perf_counter()

        for epoch in range(1, self.config.epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_seen = 0
            for features, masks, labels in train_loader:
                features = {k: v.to(self.device) for k, v in features.items()}
                masks = {k: v.to(self.device) for k, v in masks.items()}
                labels = labels.to(self.device)

                if self.config.modality_dropout > 0:
                    masks = self._drop_modalities(masks)

                optimiser.zero_grad(set_to_none=True)
                loss = criterion(model(features, masks), labels)
                loss.backward()
                if self.config.grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                optimiser.step()

                epoch_loss += loss.item() * len(labels)
                n_seen += len(labels)

            scheduler.step()

            val_scores, val_labels, val_loss = self._evaluate(model, val_loader, criterion)
            val_pr_auc = float(average_precision_score(val_labels, val_scores))

            self.history.train_loss.append(epoch_loss / max(n_seen, 1))
            self.history.val_loss.append(val_loss)
            self.history.val_pr_auc.append(val_pr_auc)

            if val_pr_auc > best_score:
                best_score = val_pr_auc
                best_state = copy.deepcopy(model.state_dict())
                self.history.best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if self.config.verbose and (epoch % 10 == 0 or epoch == 1):
                logger.info(
                    "epoch %3d | train loss %.4f | val loss %.4f | val PR-AUC %.4f",
                    epoch, self.history.train_loss[-1], val_loss, val_pr_auc,
                )

            if epochs_without_improvement >= self.config.patience:
                logger.info(
                    "early stop at epoch %d; best validation PR-AUC %.4f at epoch %d",
                    epoch, best_score, self.history.best_epoch,
                )
                self.history.stopped_early = True
                break

        # Restore the best checkpoint, not the last one. Without this, early
        # stopping would keep whichever weights happened to be current when
        # patience ran out.
        model.load_state_dict(best_state)
        self.model = model
        self.history.best_val_pr_auc = float(best_score)
        self.history.seconds = time.perf_counter() - started
        logger.info(
            "trained in %.1fs, %d epochs, best val PR-AUC %.4f (epoch %d)",
            self.history.seconds, len(self.history.train_loss),
            self.history.best_val_pr_auc, self.history.best_epoch,
        )
        return self

    def _drop_modalities(
        self, masks: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Randomly mask out present modalities during training.

        Applied only to modalities that are actually present -- dropping an
        already-absent one is a no-op. Claims that would lose every modality
        keep one at random, since a claim with no inputs at all carries no
        information and only adds noise to the gradient.
        """
        names = list(masks)
        if len(names) < 2:
            return masks

        stacked = torch.stack([masks[n] for n in names], dim=1)
        keep = torch.rand_like(stacked, dtype=torch.float) >= self.config.modality_dropout
        dropped = stacked & keep

        # Restore one modality for any claim left with nothing.
        empty = ~dropped.any(dim=1)
        if empty.any():
            available = stacked[empty].float()
            # Sample among the modalities that were genuinely present.
            choice = torch.multinomial(available + 1e-9, num_samples=1)
            restored = torch.zeros_like(available, dtype=torch.bool)
            restored.scatter_(1, choice, True)
            dropped[empty] = restored & stacked[empty]

        return {name: dropped[:, i] for i, name in enumerate(names)}

    def _evaluate(self, model: FusionModel, loader, criterion):
        model.eval()
        scores: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        total_loss = 0.0
        n_seen = 0
        with torch.no_grad():
            for features, masks, batch_labels in loader:
                features = {k: v.to(self.device) for k, v in features.items()}
                masks = {k: v.to(self.device) for k, v in masks.items()}
                batch_labels = batch_labels.to(self.device)

                logits = model(features, masks)
                total_loss += criterion(logits, batch_labels).item() * len(batch_labels)
                n_seen += len(batch_labels)
                scores.append(torch.sigmoid(logits).cpu().numpy())
                labels.append(batch_labels.cpu().numpy())
        return (
            np.concatenate(scores),
            np.concatenate(labels),
            total_loss / max(n_seen, 1),
        )

    def predict_proba(self, dataset: MultimodalDataset) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        loader = make_loader(dataset, batch_size=self.config.batch_size)
        self.model.eval()
        scores: list[np.ndarray] = []
        with torch.no_grad():
            for features, masks, _ in loader:
                features = {k: v.to(self.device) for k, v in features.items()}
                masks = {k: v.to(self.device) for k, v in masks.items()}
                scores.append(torch.sigmoid(self.model(features, masks)).cpu().numpy())
        return np.concatenate(scores)

    def modality_gates(self, dataset: MultimodalDataset) -> dict[str, float] | None:
        """Mean gate weight per modality, for the gated strategy.

        These are the model's own learned answer to "which modality mattered?",
        read off directly rather than inferred from an ablation.
        """
        if self.model is None or self.fusion_config.strategy != "gated":
            return None

        loader = make_loader(dataset, batch_size=self.config.batch_size)
        self.model.eval()
        collected: list[np.ndarray] = []
        with torch.no_grad():
            for features, masks, _ in loader:
                features = {k: v.to(self.device) for k, v in features.items()}
                masks = {k: v.to(self.device) for k, v in masks.items()}
                self.model(features, masks)
                gates = self.model.last_gates
                if gates is not None:
                    collected.append(gates.cpu().numpy())

        if not collected:
            return None
        means = np.vstack(collected).mean(axis=0)
        return dict(zip(self.fusion_config.modalities, means.tolist(), strict=True))
