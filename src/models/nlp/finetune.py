"""End-to-end fine-tuning of the text transformer.

Why this exists. The frozen-embedding pipeline treats a sentence encoder as a
fixed feature extractor: the narrative goes in, a 384-dimensional vector comes
out, and a small head learns to classify it. That works when the task is close
to what the encoder was trained for -- semantic similarity -- and much less well
when the task hinges on discrete facts that barely move a sentence embedding.
"No police report was filed" and "A police report was filed at the scene" are
near-identical under a similarity objective and opposite under a fraud
objective. Fine-tuning is what lets the representation be reshaped around the
second objective instead of the first.

So the comparison here is not decoration. It is the direct test of whether the
text modality's weakness is a property of the data or an artefact of using
frozen general-purpose embeddings, and the answer changes what the ablation
means.

Making it fit on a CPU. A full fine-tune of six transformer layers over 8,762
training narratives is roughly twenty minutes per epoch on four cores. Three
levers bring that down without making the exercise dishonest:

* **Freeze the embedding matrix and the lowest layers.** The word embeddings
  and the earliest layers encode lexical and syntactic regularities that 8,762
  formulaic narratives are not going to improve. Gradients stop at the first
  unfrozen layer, so the backward pass shortens accordingly.
* **Shorten the window to 128 tokens.** Measured truncation at 128 is 0.53 % of
  the corpus, against 0 % at 160. Attention is quadratic in sequence length, so
  the saving is real and the cost is five claims in a thousand losing their
  final clause.
* **Keep the epoch count low.** With 550 positives, a transformer overfits
  within a small number of epochs regardless; early stopping on validation
  PR-AUC decides when to stop rather than a fixed schedule.

Each of these is a compute trade-off with a measured cost, recorded in the run
so the result can be read with the constraint in view.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.utils.config import configure_threads, resolve_device
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FineTuneConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_length: int = 128
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 3e-5
    """Small, as fine-tuning conventionally requires: a pretrained network is
    being nudged, and a head-sized learning rate would destroy what it knows."""

    head_learning_rate: float = 1e-3
    """The classification head is randomly initialised and needs to move fast."""

    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    dropout: float = 0.2

    #: Number of lowest transformer layers to keep frozen, plus the embeddings.
    #: MiniLM-L6 has six layers, so 2 leaves four trainable.
    freeze_layers: int = 2
    freeze_embeddings: bool = True

    class_weight_power: float = 0.5
    patience: int = 2
    grad_clip: float = 1.0
    seed: int = 20260921
    device: str = "auto"

    #: Cap on training rows. None uses all of them. Present because a full pass
    #: is roughly a quarter-hour of CPU and a smaller subsample is useful while
    #: iterating.
    max_train_samples: int | None = None

    def as_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "head_learning_rate": self.head_learning_rate,
            "weight_decay": self.weight_decay,
            "freeze_layers": self.freeze_layers,
            "freeze_embeddings": self.freeze_embeddings,
            "class_weight_power": self.class_weight_power,
            "seed": self.seed,
            "max_train_samples": self.max_train_samples,
        }


class TextClassificationDataset(Dataset):
    """Pre-tokenised narratives with labels.

    Tokenising once up front rather than per batch: the corpus is small enough
    to hold tokenised in memory, and re-tokenising every epoch would add a
    measurable fraction of the CPU budget for no benefit.
    """

    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        self.labels = torch.tensor(np.asarray(labels), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            {key: value[index] for key, value in self.encodings.items()},
            self.labels[index],
        )


class TransformerClassifier(nn.Module):
    """Transformer encoder with a pooled binary classification head."""

    def __init__(self, model_name: str, dropout: float = 0.2):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, **kwargs) -> torch.Tensor:
        output = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        pooled = self._mean_pool(output.last_hidden_state, attention_mask)
        return self.classifier(self.dropout(pooled)).squeeze(-1)

    @staticmethod
    def _mean_pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor):
        mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
        return (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def embeddings(self, input_ids, attention_mask) -> torch.Tensor:
        """Pooled representation, for use as features by the fusion model."""
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self._mean_pool(output.last_hidden_state, attention_mask)


@dataclass
class FineTuneHistory:
    train_loss: list[float] = field(default_factory=list)
    val_pr_auc: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_pr_auc: float = 0.0
    seconds: float = 0.0
    n_trainable: int = 0
    n_total: int = 0

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class TextFineTuner:
    """Fine-tune a transformer on the claim narratives."""

    def __init__(self, config: FineTuneConfig | None = None) -> None:
        self.config = config or FineTuneConfig()
        self.device = resolve_device(self.config.device)
        self.model: TransformerClassifier | None = None
        self.tokenizer = None
        self.history = FineTuneHistory()
        configure_threads()

    def _build(self) -> None:
        from transformers import AutoTokenizer

        torch.manual_seed(self.config.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        model = TransformerClassifier(self.config.model_name, self.config.dropout)
        self._freeze(model)
        self.model = model.to(self.device)

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.history.n_total = total
        self.history.n_trainable = trainable
        logger.info(
            "fine-tuning %s: %d of %d parameters trainable (%.1f%%)",
            self.config.model_name, trainable, total, 100 * trainable / total,
        )

    def _freeze(self, model: TransformerClassifier) -> None:
        """Freeze the embeddings and the lowest transformer layers."""
        encoder = model.encoder

        if self.config.freeze_embeddings and hasattr(encoder, "embeddings"):
            for parameter in encoder.embeddings.parameters():
                parameter.requires_grad_(False)

        layers = None
        if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
            layers = encoder.encoder.layer          # BERT / MiniLM
        elif hasattr(encoder, "transformer") and hasattr(encoder.transformer, "layer"):
            layers = encoder.transformer.layer      # DistilBERT

        if layers is not None:
            for layer in layers[: self.config.freeze_layers]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(False)

    def fit(
        self,
        train_texts: list[str],
        train_labels: np.ndarray,
        val_texts: list[str],
        val_labels: np.ndarray,
    ) -> TextFineTuner:
        self._build()
        started = time.perf_counter()

        train_labels = np.asarray(train_labels)
        if self.config.max_train_samples and len(train_texts) > self.config.max_train_samples:
            generator = np.random.default_rng(self.config.seed)
            # Stratified subsample: an unstratified one would thin the 550
            # positives that carry all the signal.
            keep = _stratified_sample(
                train_labels, self.config.max_train_samples, generator
            )
            train_texts = [train_texts[i] for i in keep]
            train_labels = train_labels[keep]
            logger.info("subsampled training set to %d rows", len(train_texts))

        train_set = TextClassificationDataset(
            train_texts, train_labels, self.tokenizer, self.config.max_length
        )
        val_set = TextClassificationDataset(
            val_texts, np.asarray(val_labels), self.tokenizer, self.config.max_length
        )

        train_loader = DataLoader(
            train_set, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_set, batch_size=self.config.batch_size * 2)

        ratio = (len(train_labels) - train_labels.sum()) / max(train_labels.sum(), 1)
        pos_weight = float(ratio**self.config.class_weight_power)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, device=self.device)
        )

        # Separate learning rates: the pretrained encoder is nudged, the fresh
        # head is trained.
        encoder_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and n.startswith("encoder")
        ]
        head_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and not n.startswith("encoder")
        ]
        optimiser = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": self.config.learning_rate},
                {"params": head_params, "lr": self.config.head_learning_rate},
            ],
            weight_decay=self.config.weight_decay,
        )

        total_steps = max(1, len(train_loader) * self.config.epochs)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimiser,
            max_lr=[self.config.learning_rate, self.config.head_learning_rate],
            total_steps=total_steps,
            pct_start=self.config.warmup_ratio,
            anneal_strategy="linear",
        )

        best_state = None
        best_score = -np.inf
        stale = 0

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            running, seen = 0.0, 0
            epoch_started = time.perf_counter()

            for step, (encoding, labels) in enumerate(train_loader, start=1):
                encoding = {k: v.to(self.device) for k, v in encoding.items()}
                labels = labels.to(self.device)

                optimiser.zero_grad(set_to_none=True)
                loss = criterion(self.model(**encoding), labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                optimiser.step()
                scheduler.step()

                running += loss.item() * len(labels)
                seen += len(labels)

                if step % 100 == 0:
                    rate = seen / (time.perf_counter() - epoch_started)
                    logger.info(
                        "  epoch %d step %d/%d | loss %.4f | %.1f samples/s",
                        epoch, step, len(train_loader), running / seen, rate,
                    )

            scores, truths = self._score(val_loader)
            val_pr_auc = float(average_precision_score(truths, scores))

            self.history.train_loss.append(running / max(seen, 1))
            self.history.val_pr_auc.append(val_pr_auc)
            logger.info(
                "epoch %d | train loss %.4f | val PR-AUC %.4f | %.1fs",
                epoch, self.history.train_loss[-1], val_pr_auc,
                time.perf_counter() - epoch_started,
            )

            if val_pr_auc > best_score:
                best_score = val_pr_auc
                best_state = copy.deepcopy(self.model.state_dict())
                self.history.best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale >= self.config.patience:
                    logger.info("early stop at epoch %d", epoch)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.history.best_val_pr_auc = float(best_score)
        self.history.seconds = time.perf_counter() - started
        logger.info(
            "fine-tuned in %.1fs; best val PR-AUC %.4f at epoch %d",
            self.history.seconds, best_score, self.history.best_epoch,
        )
        return self

    def _score(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        scores, truths = [], []
        with torch.no_grad():
            for encoding, labels in loader:
                encoding = {k: v.to(self.device) for k, v in encoding.items()}
                logits = self.model(**encoding)
                scores.append(torch.sigmoid(logits).cpu().numpy())
                truths.append(labels.numpy())
        return np.concatenate(scores), np.concatenate(truths)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        dataset = TextClassificationDataset(
            texts, np.zeros(len(texts)), self.tokenizer, self.config.max_length
        )
        loader = DataLoader(dataset, batch_size=self.config.batch_size * 2)
        scores, _ = self._score(loader)
        return scores

    def encode(self, texts: list[str], *, batch_size: int | None = None) -> np.ndarray:
        """Pooled representations from the fine-tuned encoder.

        These replace the frozen embeddings in the fusion model, which is how
        fine-tuning feeds back into the multimodal result rather than only into
        the text-only arm.
        """
        if self.model is None:
            raise RuntimeError("model is not fitted")
        dataset = TextClassificationDataset(
            texts, np.zeros(len(texts)), self.tokenizer, self.config.max_length
        )
        loader = DataLoader(dataset, batch_size=batch_size or self.config.batch_size * 2)

        self.model.eval()
        out = []
        with torch.no_grad():
            for encoding, _ in loader:
                encoding = {k: v.to(self.device) for k, v in encoding.items()}
                pooled = self.model.embeddings(
                    encoding["input_ids"], encoding["attention_mask"]
                )
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                out.append(pooled.cpu().numpy().astype("float32"))
        return np.vstack(out)


def _stratified_sample(
    labels: np.ndarray, n: int, generator: np.random.Generator
) -> np.ndarray:
    """Sample ``n`` indices while preserving the class ratio."""
    indices = []
    for value in np.unique(labels):
        positions = np.flatnonzero(labels == value)
        take = max(1, int(round(n * len(positions) / len(labels))))
        indices.append(
            generator.choice(positions, size=min(take, len(positions)), replace=False)
        )
    return np.sort(np.concatenate(indices))
