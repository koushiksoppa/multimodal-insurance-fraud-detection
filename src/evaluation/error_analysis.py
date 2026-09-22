"""Error analysis.

A single aggregate number says a model is wrong 90 % of the time it fires. It
does not say *which* claims it is wrong about, and that is the part that
decides whether a fraud model is usable. This module breaks the errors apart
along the axes that matter here:

* **Which kind of error.** False positives cost an investigator's time; false
  negatives are claims paid out in full. They are not interchangeable and are
  reported separately.
* **Confidence.** A confident mistake is a different failure from a borderline
  one. A model whose errors cluster near its threshold is merely uncertain; one
  that is confidently wrong has learned something false.
* **Segment.** Error rates by policy type, fault, claim period and whether a
  photograph was attached. A model that is uniformly mediocre is a different
  problem from one that fails on an identifiable subgroup.
* **Modality disagreement.** Where the single-modality models disagree with
  each other, and whether the fusion model resolves those cases better or worse
  than either. This is the one analysis that only a multimodal project can run,
  and it is where the fusion architecture either earns its place or does not.

Representative examples are surfaced with their narratives, so the failures can
be read rather than only counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_SEGMENTS = [
    "base_policy",
    "fault",
    "vehicle_category",
    "accident_area",
    "police_report_filed",
    "past_number_of_claims",
    "has_image",
    "driver_sex",
]


@dataclass
class ErrorAnalysis:
    threshold: float
    counts: dict[str, int] = field(default_factory=dict)
    confidence: pd.DataFrame = field(default_factory=pd.DataFrame)
    segments: dict[str, pd.DataFrame] = field(default_factory=dict)
    examples: dict[str, pd.DataFrame] = field(default_factory=dict)
    disagreement: pd.DataFrame | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "counts": self.counts,
            "confidence": self.confidence.to_dict(orient="records"),
            "segments": {k: v.to_dict(orient="records") for k, v in self.segments.items()},
            "examples": {
                k: v.to_dict(orient="records") for k, v in self.examples.items()
            },
            "disagreement": (
                self.disagreement.to_dict(orient="records")
                if self.disagreement is not None
                else None
            ),
            "notes": self.notes,
        }


def _error_kind(row) -> str:
    if row["y_true"] == 1 and row["y_pred"] == 1:
        return "true_positive"
    if row["y_true"] == 1 and row["y_pred"] == 0:
        return "false_negative"
    if row["y_true"] == 0 and row["y_pred"] == 1:
        return "false_positive"
    return "true_negative"


def analyse(
    predictions: pd.DataFrame,
    claims: pd.DataFrame,
    *,
    threshold: float,
    segments: list[str] | None = None,
    n_examples: int = 5,
    min_segment_size: int = 30,
) -> ErrorAnalysis:
    """Break down one model's test-set errors.

    ``predictions`` needs ``claim_id``, ``y_true``, ``score`` and ``y_pred``.
    """
    segments = segments or DEFAULT_SEGMENTS

    merged = predictions.merge(claims, on="claim_id", how="left", suffixes=("", "_claim"))
    merged["error_kind"] = merged.apply(_error_kind, axis=1)
    # Distance from the operating point, so "confidently wrong" is measurable
    # rather than impressionistic.
    merged["margin"] = merged["score"] - threshold

    counts = merged["error_kind"].value_counts().to_dict()
    analysis = ErrorAnalysis(threshold=threshold, counts={k: int(v) for k, v in counts.items()})

    analysis.confidence = _confidence_breakdown(merged)
    analysis.segments = _segment_breakdown(merged, segments, min_segment_size)
    analysis.examples = _representative_examples(merged, n_examples)
    analysis.notes = _notes(merged, analysis)

    logger.info(
        "error analysis at threshold %.4f: %s", threshold,
        ", ".join(f"{k}={v}" for k, v in sorted(analysis.counts.items())),
    )
    return analysis


def _confidence_breakdown(merged: pd.DataFrame) -> pd.DataFrame:
    """How errors distribute across score bands."""
    bands = pd.cut(
        merged["score"],
        bins=[0.0, 0.05, 0.10, 0.20, 0.40, 1.0],
        include_lowest=True,
    )
    grouped = merged.groupby(bands, observed=True).agg(
        n=("y_true", "size"),
        n_fraud=("y_true", "sum"),
        mean_score=("score", "mean"),
    )
    grouped["observed_fraud_rate"] = grouped["n_fraud"] / grouped["n"]
    return grouped.reset_index().rename(columns={"score": "score_band"})


def _segment_breakdown(
    merged: pd.DataFrame, segments: list[str], min_size: int
) -> dict[str, pd.DataFrame]:
    """Error rates within each segment of each attribute.

    Segments smaller than ``min_size`` are dropped rather than reported: a
    false-negative rate computed over four claims is noise that looks like a
    finding.
    """
    out: dict[str, pd.DataFrame] = {}
    for column in segments:
        if column not in merged.columns:
            continue

        rows = []
        for value, part in merged.groupby(column, observed=True):
            if len(part) < min_size:
                continue
            positives = int(part["y_true"].sum())
            flagged = int(part["y_pred"].sum())
            true_positives = int(((part["y_true"] == 1) & (part["y_pred"] == 1)).sum())
            rows.append(
                {
                    "segment": str(value),
                    "n": len(part),
                    "n_fraud": positives,
                    "fraud_rate": positives / len(part),
                    "flag_rate": flagged / len(part),
                    "recall": true_positives / positives if positives else np.nan,
                    "precision": true_positives / flagged if flagged else np.nan,
                    "mean_score": float(part["score"].mean()),
                }
            )
        if rows:
            out[column] = pd.DataFrame(rows).sort_values("n", ascending=False)
    return out


def _representative_examples(merged: pd.DataFrame, n: int) -> dict[str, pd.DataFrame]:
    """The most instructive individual failures.

    Not a random sample: the *most confident* mistakes of each kind, because
    those are the ones that reveal what the model has actually learned. A
    borderline miss tells you the threshold is in an awkward place; a confident
    miss tells you something about the model.
    """
    columns = [
        c
        for c in [
            "claim_id", "y_true", "score", "margin", "base_policy", "fault",
            "vehicle_category", "past_number_of_claims", "has_image",
            "damage_condition", "claim_text",
        ]
        if c in merged.columns
    ]

    out: dict[str, pd.DataFrame] = {}

    false_negatives = merged[merged["error_kind"] == "false_negative"]
    if len(false_negatives):
        out["most_confident_false_negatives"] = (
            false_negatives.nsmallest(n, "score")[columns].reset_index(drop=True)
        )
        out["narrowest_false_negatives"] = (
            false_negatives.nlargest(n, "score")[columns].reset_index(drop=True)
        )

    false_positives = merged[merged["error_kind"] == "false_positive"]
    if len(false_positives):
        out["most_confident_false_positives"] = (
            false_positives.nlargest(n, "score")[columns].reset_index(drop=True)
        )

    true_positives = merged[merged["error_kind"] == "true_positive"]
    if len(true_positives):
        out["most_confident_true_positives"] = (
            true_positives.nlargest(n, "score")[columns].reset_index(drop=True)
        )

    return out


def _notes(merged: pd.DataFrame, analysis: ErrorAnalysis) -> list[str]:
    notes = []

    false_negatives = merged[merged["error_kind"] == "false_negative"]
    if len(false_negatives):
        very_low = int((false_negatives["score"] < merged["score"].quantile(0.25)).sum())
        notes.append(
            f"{very_low} of {len(false_negatives)} missed frauds scored in the "
            f"bottom quartile of all claims: the model is not merely uncertain "
            f"about them, it is confident they are legitimate."
        )

    if "has_image" in merged.columns:
        with_image = merged[merged["has_image"].fillna(False).astype(bool)]
        without = merged[~merged["has_image"].fillna(False).astype(bool)]
        if len(with_image) > 30 and len(without) > 30:
            notes.append(
                f"Mean score {with_image['score'].mean():.4f} on claims with a "
                f"photograph versus {without['score'].mean():.4f} without. A large "
                f"gap here would mean the model is reacting to the presence of an "
                f"image rather than to its content."
            )

    total_errors = analysis.counts.get("false_positive", 0) + analysis.counts.get(
        "false_negative", 0
    )
    notes.append(f"{total_errors} errors out of {len(merged)} test claims.")
    return notes


def modality_disagreement(
    predictions_by_modality: dict[str, pd.DataFrame],
    claims: pd.DataFrame,
    *,
    fusion_key: str = "tabular+text+image",
    quantile: float = 0.9,
) -> pd.DataFrame:
    """Where single-modality models disagree, and whether fusion helps.

    Each modality's scores are converted to within-split ranks first. Raw
    scores from differently-calibrated models are not comparable -- one may
    simply run hotter than another -- whereas ranks ask the question that
    matters: did the two models put this claim in different parts of the queue?

    A claim is "disagreed" when one modality ranks it in its top decile and
    another does not. The table then reports how the fusion model performs on
    exactly those claims, relative to the modalities it was built from.
    """
    frames = []
    for key, frame in predictions_by_modality.items():
        part = frame[frame["split"] == "test"] if "split" in frame.columns else frame
        renamed = part[["claim_id", "y_true", "score"]].copy()
        renamed[f"rank_{key}"] = renamed["score"].rank(pct=True)
        frames.append(renamed.drop(columns="score").rename(columns={"y_true": "y_true"}))

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["claim_id", "y_true"], how="inner")

    single = [
        k for k in predictions_by_modality
        if "+" not in k and f"rank_{k}" in merged.columns
    ]
    if len(single) < 2:
        raise ValueError("need at least two single-modality models to compare")

    rank_columns = [f"rank_{k}" for k in single]
    high = merged[rank_columns] >= quantile
    merged["n_flagging"] = high.sum(axis=1)
    merged["disagreement"] = (merged["n_flagging"] > 0) & (
        merged["n_flagging"] < len(single)
    )

    fusion_column = f"rank_{fusion_key}"
    rows = []
    for label, part in (
        ("all", merged),
        ("agreed_high", merged[merged["n_flagging"] == len(single)]),
        ("disagreed", merged[merged["disagreement"]]),
        ("agreed_low", merged[merged["n_flagging"] == 0]),
    ):
        if not len(part):
            continue
        row = {
            "group": label,
            "n": len(part),
            "n_fraud": int(part["y_true"].sum()),
            "fraud_rate": float(part["y_true"].mean()),
        }
        for key in [*single, fusion_key]:
            column = f"rank_{key}"
            if column in part.columns:
                row[f"mean_rank_{key}"] = float(part[column].mean())
        if fusion_column in part.columns and part["y_true"].nunique() > 1:
            from sklearn.metrics import roc_auc_score

            row["fusion_roc_auc"] = float(
                roc_auc_score(part["y_true"], part[fusion_column])
            )
        rows.append(row)

    del claims  # merged on claim_id only; claim attributes are not needed here
    return pd.DataFrame(rows)
