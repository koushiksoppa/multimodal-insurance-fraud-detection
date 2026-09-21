"""Train / validation / test splitting, and the leakage audit that guards it.

Two strategies are implemented, and the difference between them is itself a
result worth reporting.

``temporal``
    Train on the earliest claims, validate on the next period, test on the
    latest. This is the honest protocol for fraud detection: a deployed model
    only ever sees the past. The source data spans three calendar years and the
    recorded fraud rate declines across them, so a model evaluated on a random
    split is being scored on a period it has partly already seen. Temporal is
    the default.

``stratified_random``
    Stratified on the label, ignoring time. Reported alongside the temporal
    result so the optimism introduced by ignoring time can be quantified rather
    than assumed.

Leakage is prevented structurally rather than by hoping:

* Splits partition **claims**, and each photograph is attached to at most one
  claim, so no image can straddle a split boundary.
* Preprocessing is fitted on the training split only; :mod:`src.features`
  enforces this by fitting inside the pipeline rather than on the full frame.
* The test split is never used to select a threshold, a hyperparameter or a
  model. Thresholds come from validation.
* :func:`audit_splits` re-checks all of the above on the built dataset and is
  wired into the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from src.utils.logging import get_logger
from src.utils.seeding import derive_seed

logger = get_logger(__name__)

SplitName = Literal["train", "val", "test"]
Strategy = Literal["temporal", "stratified_random"]

MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]
MONTH_TO_NUMBER = {name: i + 1 for i, name in enumerate(MONTHS)}


@dataclass
class SplitConfig:
    strategy: Strategy = "temporal"

    #: Temporal strategy: claims strictly before this (year, month) go to train,
    #: claims before ``test_start`` go to validation, the rest go to test.
    val_start: tuple[int, int] = (1995, 7)
    test_start: tuple[int, int] = (1996, 1)

    #: Random strategy proportions. Ignored by the temporal strategy, whose
    #: proportions follow from the calendar.
    train_frac: float = 0.6
    val_frac: float = 0.2

    seed: int = 20260921


@dataclass
class SplitReport:
    strategy: str
    sizes: dict[str, int] = field(default_factory=dict)
    positives: dict[str, int] = field(default_factory=dict)
    positive_rates: dict[str, float] = field(default_factory=dict)
    image_coverage: dict[str, float] = field(default_factory=dict)
    boundaries: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "sizes": self.sizes,
            "positives": self.positives,
            "positive_rates": self.positive_rates,
            "image_coverage": self.image_coverage,
            "boundaries": self.boundaries,
            "notes": self.notes,
        }


def _time_key(df: pd.DataFrame) -> pd.Series:
    """Order claims in time as precisely as the source data allows.

    The source has no dates, only a calendar year plus the incident month and
    week-of-month, so the key is year * 10000 + month * 100 + week. Ties within
    a week are unordered, which is fine: the split boundaries fall on month
    boundaries.
    """
    month = df["incident_month"].map(MONTH_TO_NUMBER)
    week = pd.to_numeric(df["incident_week_of_month"], errors="coerce").fillna(1)
    return df["policy_year"].astype(int) * 10_000 + month.fillna(1).astype(int) * 100 + week.astype(int)


def assign_splits(
    df: pd.DataFrame,
    config: SplitConfig | None = None,
    *,
    label_column: str = "fraud_label",
) -> tuple[pd.Series, SplitReport]:
    """Return a split label per row, plus a report describing the result."""
    config = config or SplitConfig()

    if config.strategy == "temporal":
        splits, report = _assign_temporal(df, config)
    elif config.strategy == "stratified_random":
        splits, report = _assign_stratified_random(df, config, label_column)
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"unknown split strategy {config.strategy!r}")

    for name in ("train", "val", "test"):
        part = df[splits == name]
        positives = int(part[label_column].sum()) if len(part) else 0
        report.sizes[name] = len(part)
        report.positives[name] = positives
        report.positive_rates[name] = positives / len(part) if len(part) else 0.0
        if "has_image" in part.columns:
            report.image_coverage[name] = (
                float(part["has_image"].mean()) if len(part) else 0.0
            )

    logger.info(
        "split (%s): %s",
        report.strategy,
        ", ".join(
            f"{k} n={report.sizes[k]} pos={report.positives[k]} "
            f"({report.positive_rates[k]:.4f})"
            for k in ("train", "val", "test")
        ),
    )
    return splits, report


def _assign_temporal(
    df: pd.DataFrame, config: SplitConfig
) -> tuple[pd.Series, SplitReport]:
    key = _time_key(df)
    val_key = config.val_start[0] * 10_000 + config.val_start[1] * 100
    test_key = config.test_start[0] * 10_000 + config.test_start[1] * 100

    splits = pd.Series("train", index=df.index, dtype=object)
    splits[key >= val_key] = "val"
    splits[key >= test_key] = "test"

    report = SplitReport(
        strategy="temporal",
        boundaries={
            "train": f"< {config.val_start[0]}-{config.val_start[1]:02d}",
            "val": f"{config.val_start[0]}-{config.val_start[1]:02d} "
                   f"to < {config.test_start[0]}-{config.test_start[1]:02d}",
            "test": f">= {config.test_start[0]}-{config.test_start[1]:02d}",
        },
        notes=[
            "Splits are ordered in time; no claim in train occurs after a claim "
            "in test.",
            "policy_year is constant within the test split and nearly so within "
            "the others, so it is dropped from the feature set under this "
            "strategy to avoid a degenerate feature.",
        ],
    )
    return splits, report


def _assign_stratified_random(
    df: pd.DataFrame, config: SplitConfig, label_column: str
) -> tuple[pd.Series, SplitReport]:
    generator = np.random.default_rng(derive_seed(config.seed, "split:stratified"))
    splits = pd.Series("train", index=df.index, dtype=object)

    for label in sorted(pd.unique(df[label_column])):
        positions = np.flatnonzero((df[label_column] == label).to_numpy())
        generator.shuffle(positions)
        n = len(positions)
        n_train = int(round(n * config.train_frac))
        n_val = int(round(n * config.val_frac))
        splits.iloc[positions[n_train : n_train + n_val]] = "val"
        splits.iloc[positions[n_train + n_val :]] = "test"

    report = SplitReport(
        strategy="stratified_random",
        boundaries={"train": "random", "val": "random", "test": "random"},
        notes=[
            "Stratified on the label only. Claims from all three years appear "
            "in every split, so a model can benefit from period-specific "
            "patterns it would not have at deployment time. Reported for "
            "comparison with the temporal split, not as the headline protocol.",
        ],
    )
    return splits, report


# --------------------------------------------------------------------------
# leakage audit
# --------------------------------------------------------------------------


@dataclass
class LeakageFinding:
    name: str
    passed: bool
    message: str
    details: dict = field(default_factory=dict)


def audit_splits(
    df: pd.DataFrame,
    *,
    split_column: str = "split",
    strategy: str = "temporal",
) -> list[LeakageFinding]:
    """Check every leakage route this dataset could plausibly have.

    Returns a list of findings rather than raising, so the caller can log all
    of them at once. :func:`raise_on_leakage` turns failures into an error.
    """
    findings: list[LeakageFinding] = []

    # 1. A claim may belong to exactly one split.
    per_claim = df.groupby("claim_id")[split_column].nunique()
    n_claim_leaks = int((per_claim > 1).sum())
    findings.append(
        LeakageFinding(
            "claim_disjoint",
            n_claim_leaks == 0,
            f"{n_claim_leaks} claims appear in more than one split",
            {"n_leaked": n_claim_leaks},
        )
    )

    # 2. A photograph may belong to exactly one split. Guaranteed by the
    #    attachment rule, but checked because the guarantee is load-bearing.
    if "image_id" in df.columns:
        attached = df[df["image_id"].notna()]
        n_image_leaks = 0
        if len(attached):
            per_image = attached.groupby("image_id")[split_column].nunique()
            n_image_leaks = int((per_image > 1).sum())
        findings.append(
            LeakageFinding(
                "image_disjoint",
                n_image_leaks == 0,
                f"{n_image_leaks} images appear in more than one split",
                {"n_attached": len(attached), "n_leaked": n_image_leaks},
            )
        )

    # 3. Policy-level overlap. The source table has one row per policy number,
    #    so this should be vacuously true; it is checked because a future
    #    dataset with repeated policies would silently break the assumption.
    if "policy_number" in df.columns:
        per_policy = df.groupby("policy_number")[split_column].nunique()
        n_policy_leaks = int((per_policy > 1).sum())
        findings.append(
            LeakageFinding(
                "policy_disjoint",
                n_policy_leaks == 0,
                f"{n_policy_leaks} policy numbers appear in more than one split",
                {"n_leaked": n_policy_leaks},
            )
        )

    # 4. Near-duplicate records straddling a split would let a model recall a
    #    training row at test time.
    feature_cols = [
        c for c in df.columns
        if c not in {
            "claim_id", "policy_number", split_column, "fraud_label",
            "image_id", "image_path", "claim_text", "dataset_version", "track",
        }
    ]
    dup_mask = df.duplicated(subset=feature_cols, keep=False)
    n_cross = 0
    if dup_mask.any():
        groups = df[dup_mask].groupby(feature_cols, dropna=False)[split_column].nunique()
        n_cross = int((groups > 1).sum())
    findings.append(
        LeakageFinding(
            "duplicate_across_splits",
            n_cross == 0,
            f"{n_cross} identical feature vectors span more than one split",
            {"n_duplicate_rows": int(dup_mask.sum()), "n_cross_split_groups": n_cross},
        )
    )

    # 5. Under the temporal strategy, no training claim may postdate a test claim.
    if strategy == "temporal" and "incident_month" in df.columns:
        key = _time_key(df)
        train_max = key[df[split_column] == "train"].max()
        test_min = key[df[split_column] == "test"].min()
        ordered = bool(train_max < test_min)
        findings.append(
            LeakageFinding(
                "temporal_order",
                ordered,
                f"latest train key {train_max}, earliest test key {test_min}",
                {"train_max": int(train_max), "test_min": int(test_min)},
            )
        )

    for finding in findings:
        (logger.info if finding.passed else logger.error)(
            "leakage audit [%s] %s: %s",
            "pass" if finding.passed else "FAIL",
            finding.name,
            finding.message,
        )
    return findings


def raise_on_leakage(findings: list[LeakageFinding]) -> None:
    failed = [f for f in findings if not f.passed]
    if failed:
        bullet = "\n  - ".join(f"{f.name}: {f.message}" for f in failed)
        raise ValueError(f"leakage audit failed:\n  - {bullet}")
