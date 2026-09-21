"""Transformer text encoder with on-disk embedding cache.

Model choice. The workhorse is MiniLM-L6 (6 layers, 384-dimensional), not
BERT-base or RoBERTa. The reason is measured, not aesthetic: on the four-core
CPU this project targets, MiniLM encodes the 15,420-claim corpus in about
thirteen minutes, where BERT-base would take roughly an hour for embeddings
that a 79-word semi-structured narrative does not need. MiniLM is a distilled
BERT trained with a sentence-level objective, so its mean-pooled output is
already a sensible sentence representation rather than something that only
becomes useful after fine-tuning. DistilBERT is available through the same
interface and is used in the fine-tuning comparison.

Sequence length. Measured on the real corpus: median 95 WordPiece tokens, 95th
percentile 118, maximum 134. A 160-token window therefore truncates nothing.
The truncation rate is recorded at encode time rather than assumed, because
"how did you handle long claims?" deserves a number and not a shrug.

Pooling. Attention-masked mean pooling over the final hidden states. For a
sentence-transformers checkpoint this matches the objective the model was
trained under; taking the raw CLS token instead would use a representation that
this particular checkpoint never optimised.

Caching. Encoding is by far the most expensive step in the project and its
output never changes for fixed inputs, so embeddings are cached on disk keyed
by a hash of the model, the settings and the text itself. Every ablation cell
that needs text then costs seconds rather than minutes, which is what makes
running seven modality combinations across several seeds practical here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.utils.config import get_paths, resolve_device
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TextEncoderConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_length: int = 160
    batch_size: int = 32
    pooling: str = "mean"
    normalize: bool = True
    device: str = "auto"

    def cache_key(self, texts: list[str]) -> str:
        """Stable identifier for one encoding of one corpus."""
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    "model": self.model_name,
                    "max_length": self.max_length,
                    "pooling": self.pooling,
                    "normalize": self.normalize,
                },
                sort_keys=True,
            ).encode("utf-8")
        )
        for text in texts:
            digest.update(text.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()[:24]


@dataclass
class EncodingReport:
    n_texts: int
    embedding_dim: int
    max_length: int
    n_truncated: int
    truncation_rate: float
    median_tokens: float
    p95_tokens: float
    max_tokens: int
    seconds: float
    from_cache: bool

    def as_dict(self) -> dict:
        return asdict(self)


class TextEncoder:
    """Wraps a Hugging Face encoder and produces pooled sentence embeddings."""

    def __init__(self, config: TextEncoderConfig | None = None) -> None:
        self.config = config or TextEncoderConfig()
        self._tokenizer = None
        self._model = None
        self.device = resolve_device(self.config.device)

    # -- lazy loading ----------------------------------------------------

    def _load(self) -> None:
        """Load the checkpoint on first use.

        Deferred so that importing this module -- which the API and the test
        suite both do -- does not pull a hundred megabytes of weights into
        memory for code paths that never encode anything.
        """
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        logger.info("loading text encoder %s on %s", self.config.model_name, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self._model = AutoModel.from_pretrained(self.config.model_name)
        self._model.eval().to(self.device)
        # Gradients are disabled per call site with `torch.no_grad()`, never
        # globally. A process-wide `set_grad_enabled(False)` here would
        # silently stop any model trained later in the same process from
        # learning -- a failure that looks like a bad model rather than a bug.

    @property
    def tokenizer(self):
        self._load()
        return self._tokenizer

    @property
    def model(self):
        self._load()
        return self._model

    @property
    def embedding_dim(self) -> int:
        return int(self.model.config.hidden_size)

    # -- encoding --------------------------------------------------------

    def encode(
        self, texts: list[str], *, show_progress: bool = True
    ) -> tuple[np.ndarray, EncodingReport]:
        """Encode texts to pooled embeddings, measuring truncation as it goes."""
        import time

        import torch

        self._load()
        started = time.perf_counter()

        token_counts: list[int] = []
        vectors: list[np.ndarray] = []

        iterator = range(0, len(texts), self.config.batch_size)
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="encoding text", unit="batch")

        for start in iterator:
            batch = texts[start : start + self.config.batch_size]

            # Untruncated lengths, so the truncation rate can be reported
            # rather than silently absorbed by the tokenizer.
            token_counts.extend(
                len(ids) for ids in self._tokenizer(batch, truncation=False)["input_ids"]
            )

            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                output = self._model(**encoded)

            pooled = self._pool(output.last_hidden_state, encoded["attention_mask"])
            if self.config.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy().astype("float32"))

        embeddings = np.vstack(vectors)
        counts = np.asarray(token_counts)
        n_truncated = int((counts > self.config.max_length).sum())

        report = EncodingReport(
            n_texts=len(texts),
            embedding_dim=embeddings.shape[1],
            max_length=self.config.max_length,
            n_truncated=n_truncated,
            truncation_rate=n_truncated / len(texts) if texts else 0.0,
            median_tokens=float(np.median(counts)) if len(counts) else 0.0,
            p95_tokens=float(np.percentile(counts, 95)) if len(counts) else 0.0,
            max_tokens=int(counts.max()) if len(counts) else 0,
            seconds=time.perf_counter() - started,
            from_cache=False,
        )
        logger.info(
            "encoded %d texts to %d dims in %.1fs (%.1f texts/s); truncated %d (%.3f%%)",
            report.n_texts, report.embedding_dim, report.seconds,
            report.n_texts / max(report.seconds, 1e-9),
            report.n_truncated, 100 * report.truncation_rate,
        )
        return embeddings, report

    def _pool(self, hidden_state, attention_mask):
        """Pool token states into one vector per text."""

        if self.config.pooling == "cls":
            return hidden_state[:, 0]

        if self.config.pooling != "mean":
            raise ValueError(f"unknown pooling {self.config.pooling!r}")

        # Mask out padding before averaging. Without the mask, a short text in
        # a batch with a long one is diluted by its own padding, so the
        # embedding would depend on what else happened to be in the batch.
        mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
        summed = (hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode_cached(
        self,
        texts: list[str],
        *,
        cache_dir: Path | None = None,
        refresh: bool = False,
    ) -> tuple[np.ndarray, EncodingReport]:
        """Encode, reusing a cached result when the inputs are unchanged."""
        cache_dir = cache_dir or (get_paths().cache / "text")
        cache_dir.mkdir(parents=True, exist_ok=True)

        key = self.config.cache_key(texts)
        vectors_path = cache_dir / f"{key}.npy"
        meta_path = cache_dir / f"{key}.json"

        if vectors_path.exists() and meta_path.exists() and not refresh:
            embeddings = np.load(vectors_path)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            report = EncodingReport(**{**meta["report"], "from_cache": True})
            logger.info(
                "loaded %d cached text embeddings from %s", len(embeddings), vectors_path
            )
            return embeddings, report

        embeddings, report = self.encode(texts)
        np.save(vectors_path, embeddings)
        meta_path.write_text(
            json.dumps(
                {"config": asdict(self.config), "report": report.as_dict()}, indent=2
            ),
            encoding="utf-8",
        )
        return embeddings, report
