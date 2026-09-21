"""Explaining the text model.

Two methods, chosen to match the structure of the data rather than out of
habit.

**Sentence occlusion (primary).** Remove one sentence, re-encode the narrative,
re-score it, and record how far the fraud probability moved. Each sentence in
these narratives asserts exactly one fact -- whether a police report was filed,
how many prior claims there are -- so a sentence-level attribution maps
directly onto a claim attribute an adjuster would recognise. Token-level
attribution on the same text would spread the same signal across "no", "police"
and "report" and say less.

It is also model-agnostic: it treats the encoder and the classifier as one
black box, which means it works identically for the frozen-embedding pipeline
and for a fine-tuned model.

**Token attribution via gradient times input (secondary).** For a fine-tuned
model, the gradient of the output with respect to the input embeddings, dotted
with the embeddings themselves. Cheaper than integrated gradients and adequate
for the short sequences here. Available only when the classifier is
differentiable end-to-end from the tokens.

The honest caveat, as elsewhere: occlusion measures *this model's sensitivity*
to removing a sentence. A sentence with a large effect is one the model relies
on, which is not the same as a sentence that is evidence of fraud.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Split on sentence-ending punctuation followed by whitespace. The generated
#: narratives use plain declarative sentences with no abbreviations, so a
#: regex is sufficient and avoids a sentence-splitter dependency.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass
class SentenceAttribution:
    claim_id: str
    baseline_score: float
    sentences: list[str] = field(default_factory=list)
    effects: list[float] = field(default_factory=list)
    """Baseline score minus the score with that sentence removed.

    Positive means removing the sentence lowered the fraud score, so the
    sentence was pushing the score up.
    """

    def ranked(self, n: int | None = None) -> list[tuple[str, float]]:
        pairs = sorted(
            zip(self.sentences, self.effects, strict=True),
            key=lambda pair: abs(pair[1]),
            reverse=True,
        )
        return pairs[:n] if n else pairs

    def as_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "baseline_score": self.baseline_score,
            "attributions": [
                {"sentence": sentence, "effect": effect}
                for sentence, effect in self.ranked()
            ],
        }


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text.strip()) if s.strip()]


def occlude_sentences(
    text: str,
    score_fn: Callable[[list[str]], np.ndarray],
    *,
    claim_id: str = "",
) -> SentenceAttribution:
    """Attribute a claim's score to its individual sentences.

    ``score_fn`` takes a list of narrative variants and returns one
    positive-class probability each. Batching all variants into a single call
    matters: re-encoding one narrative at a time would make this far slower
    than it needs to be on CPU.
    """
    sentences = split_sentences(text)
    if len(sentences) < 2:
        scores = score_fn([text])
        return SentenceAttribution(
            claim_id=claim_id, baseline_score=float(scores[0]), sentences=sentences,
            effects=[0.0] * len(sentences),
        )

    variants = [text]
    variants.extend(
        " ".join(sentences[:i] + sentences[i + 1 :]) for i in range(len(sentences))
    )

    scores = np.asarray(score_fn(variants), dtype=float)
    baseline = float(scores[0])
    effects = (baseline - scores[1:]).tolist()

    return SentenceAttribution(
        claim_id=claim_id,
        baseline_score=baseline,
        sentences=sentences,
        effects=effects,
    )


def aggregate_sentence_effects(
    attributions: list[SentenceAttribution],
    *,
    normalise: Callable[[str], str] | None = None,
) -> dict[str, dict[str, float]]:
    """Pool per-claim sentence effects into a global picture.

    Individual narratives use several interchangeable phrasings of the same
    fact, so a raw tally would fragment one signal across four wordings.
    ``normalise`` maps a sentence to the fact it expresses; by default it
    collapses on the first few words, which is enough to group the variants
    the generator produces.
    """
    normalise = normalise or (lambda s: " ".join(s.lower().split()[:4]))

    buckets: dict[str, list[float]] = {}
    for attribution in attributions:
        for sentence, effect in zip(attribution.sentences, attribution.effects, strict=True):
            buckets.setdefault(normalise(sentence), []).append(effect)

    return {
        key: {
            "mean_effect": float(np.mean(values)),
            "mean_abs_effect": float(np.mean(np.abs(values))),
            "count": len(values),
        }
        for key, values in sorted(
            buckets.items(), key=lambda kv: -float(np.mean(np.abs(kv[1])))
        )
    }


def token_attribution(
    model,
    tokenizer,
    text: str,
    *,
    max_length: int = 160,
    device: str = "cpu",
) -> list[tuple[str, float]]:
    """Gradient-times-input attribution over WordPiece tokens.

    Requires a model that is differentiable from its input embeddings to a
    scalar output -- that is, a fine-tuned classifier rather than the frozen
    encoder plus a separate head fitted on cached vectors.
    """
    import torch

    model.eval().to(device)
    encoded = tokenizer(
        text, truncation=True, max_length=max_length, return_tensors="pt"
    ).to(device)

    embedding_layer = model.get_input_embeddings()
    embeddings = embedding_layer(encoded["input_ids"]).detach().requires_grad_(True)

    with torch.enable_grad():
        output = model(
            inputs_embeds=embeddings, attention_mask=encoded["attention_mask"]
        )
        logits = output.logits if hasattr(output, "logits") else output
        target = logits[0, -1] if logits.dim() > 1 and logits.shape[-1] > 1 else logits.squeeze()
        model.zero_grad(set_to_none=True)
        target.backward()

    # Gradient times input, summed over the embedding dimension: the
    # first-order effect of zeroing each token's embedding.
    attributions = (embeddings.grad * embeddings).sum(dim=-1).squeeze(0)
    attributions = attributions.detach().cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"].squeeze(0))
    scale = np.abs(attributions).max()
    if scale > 0:
        attributions = attributions / scale

    return [
        (token, float(value))
        for token, value in zip(tokens, attributions, strict=True)
        if token not in tokenizer.all_special_tokens
    ]
