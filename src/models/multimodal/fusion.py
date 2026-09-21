"""Fusion architectures.

One architecture family serves every cell of the ablation study, including the
single-modality cells. That is deliberate. If the tabular-only baseline were a
gradient-boosted ensemble and the multimodal model were a neural network, any
difference between them would confound *modality* with *model family*, and the
ablation would measure the wrong thing. Here every cell is the same network
given a different subset of its inputs, so a difference between cells is
attributable to the modality that was added.

(A LightGBM tabular model is still trained and reported separately, as a check
that the shared neural head is not simply weaker than a competent tabular
model. That comparison belongs outside the ablation, not inside it.)

Three fusion strategies, in increasing order of how much they assume:

``concat``
    Project each modality to a common width, concatenate, pass through an MLP.
    The standard baseline. Every modality contributes with a fixed weight
    learned once for the whole dataset.

``gated``
    Learn a per-claim gate for each modality from the joint context, and take a
    weighted sum rather than a concatenation. Lets the model down-weight a
    modality on a claim-by-claim basis -- for example, leaning on the narrative
    when the photograph is uninformative. The gates are recoverable after
    training, which makes this the most interpretable of the three: the average
    gate per modality is a direct, learned statement about how much each
    modality was worth.

``attention``
    Treat the projected modalities as a three-token sequence and run multi-head
    self-attention over it, then pool. The most expressive, and on three tokens
    also the most likely to overfit. Included so the comparison is empirical.

Complexity is not assumed to help. The ablation reports all three, and if
concatenation wins, that is the result.

Missing modalities are first-class. Every forward pass takes a presence mask.
A missing modality contributes a zero vector, its presence bit is fed to the
network explicitly, and under gated and attention fusion it is excluded from
the weighting rather than being allowed to dilute the average. This matters
here because 85 % of real claims have no photograph.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

MODALITIES = ("tabular", "text", "image")


@dataclass
class FusionConfig:
    """Architecture settings shared by every ablation cell."""

    strategy: str = "concat"
    """One of concat, gated, attention."""

    input_dims: dict[str, int] = field(default_factory=dict)
    """Input width per active modality, for example {'tabular': 46, 'text': 384}."""

    projection_dim: int = 64
    hidden_dims: tuple[int, ...] = (64,)
    dropout: float = 0.3
    attention_heads: int = 4

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "input_dims": dict(self.input_dims),
            "projection_dim": self.projection_dim,
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "attention_heads": self.attention_heads,
        }

    @property
    def modalities(self) -> list[str]:
        return [m for m in MODALITIES if m in self.input_dims]


class ModalityProjection(nn.Module):
    """Map one modality's features into the shared fusion space.

    LayerNorm before the non-linearity keeps the three modalities on comparable
    scales. Without it the 384-dimensional text embedding and the ~46-dimensional
    tabular block enter the fusion layer with very different magnitudes, and the
    network spends its early capacity correcting for that rather than learning.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _mlp_head(in_dim: int, hidden_dims: tuple[int, ...], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = in_dim
    for hidden in hidden_dims:
        layers += [
            nn.Linear(width, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        width = hidden
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


class FusionModel(nn.Module):
    """Shared network for every modality combination and fusion strategy.

    Inputs arrive as a dict of ``{modality: tensor}`` plus a matching dict of
    presence masks. A model configured with a single modality is the
    single-modality baseline for that arm; nothing else changes.
    """

    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        self.config = config
        self.modalities = config.modalities
        if not self.modalities:
            raise ValueError("at least one modality is required")
        if config.strategy not in {"concat", "gated", "attention"}:
            raise ValueError(f"unknown fusion strategy {config.strategy!r}")

        dim = config.projection_dim
        self.projections = nn.ModuleDict(
            {
                m: ModalityProjection(config.input_dims[m], dim, config.dropout)
                for m in self.modalities
            }
        )

        n = len(self.modalities)

        if config.strategy == "concat":
            # Presence bits are appended so the head can tell "this modality is
            # absent" apart from "this modality is present and near zero".
            head_in = dim * n + n
        elif config.strategy == "gated":
            self.gate = nn.Sequential(
                nn.Linear(dim * n + n, dim),
                nn.GELU(),
                nn.Linear(dim, n),
            )
            head_in = dim + n
        else:  # attention
            self.attention = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=config.attention_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.attention_norm = nn.LayerNorm(dim)
            head_in = dim + n

        self.head = _mlp_head(head_in, config.hidden_dims, config.dropout)
        self._last_gates: torch.Tensor | None = None

    # -- forward ---------------------------------------------------------

    def forward(
        self,
        features: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return one logit per claim."""
        projected = []
        presence = []
        for m in self.modalities:
            mask = masks[m].to(features[m].dtype).unsqueeze(-1)
            # Zero out absent modalities *after* projection, so a missing input
            # cannot leak through the projection's bias term.
            projected.append(self.projections[m](features[m]) * mask)
            presence.append(masks[m].to(features[m].dtype))

        presence_bits = torch.stack(presence, dim=1)

        if self.config.strategy == "concat":
            fused = torch.cat([*projected, presence_bits], dim=1)
            return self.head(fused).squeeze(-1)

        stacked = torch.stack(projected, dim=1)  # (batch, n_modalities, dim)

        if self.config.strategy == "gated":
            context = torch.cat(
                [stacked.flatten(start_dim=1), presence_bits], dim=1
            )
            logits = self.gate(context)
            # An absent modality must not receive gate weight; masking before
            # the softmax renormalises over the modalities actually present.
            logits = logits.masked_fill(presence_bits == 0, float("-inf"))
            weights = torch.softmax(logits, dim=1)
            # A claim with no modality at all would produce all -inf; guard it.
            weights = torch.nan_to_num(weights, nan=0.0)
            self._last_gates = weights.detach()
            fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        else:  # attention
            key_padding = presence_bits == 0
            attended, _ = self.attention(
                stacked, stacked, stacked, key_padding_mask=key_padding
            )
            attended = self.attention_norm(attended + stacked)
            denominator = presence_bits.sum(dim=1, keepdim=True).clamp(min=1.0)
            fused = (attended * presence_bits.unsqueeze(-1)).sum(dim=1) / denominator

        return self.head(torch.cat([fused, presence_bits], dim=1)).squeeze(-1)

    # -- introspection ---------------------------------------------------

    @property
    def last_gates(self) -> torch.Tensor | None:
        """Gate weights from the most recent forward pass, gated strategy only.

        Used by the explainability report: averaged over a split, these are the
        model's own learned statement about how much each modality contributed.
        """
        return self._last_gates

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
