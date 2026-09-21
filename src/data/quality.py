"""Reproducible data quality checks.

Each check is a small function that returns a :class:`CheckResult`. They are
registered in :data:`CHECKS` and run together by :func:`run_all`, which
produces a JSON-serialisable report. Checks report; they never repair. A check
that quietly fixed its own finding would make the report useless.

Severity has a precise meaning here:

``error``
    Violates an assumption the modelling code relies on. The pipeline should
    not proceed.
``warning``
    A real defect that the pipeline can proceed past, but which must be
    accounted for downstream (for example an undeclared missing-value
    sentinel).
``info``
    A property worth recording, such as the class balance.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

Severity = str  # "error" | "warning" | "info"


@dataclass
class CheckResult:
    name: str
    severity: Severity
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "severity": self.severity,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class QualityReport:
    results: list[CheckResult] = field(default_factory=list)

    @property
    def errors(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed and r.severity == "error"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed and r.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "n_checks": len(self.results),
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "results": [r.as_dict() for r in self.results],
        }

    def summary_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "check": r.name,
                    "severity": r.severity,
                    "status": "pass" if r.passed else "FAIL",
                    "message": r.message,
                }
                for r in self.results
            ]
        )

    def raise_on_error(self) -> None:
        if self.errors:
            bullet = "\n  - ".join(f"{r.name}: {r.message}" for r in self.errors)
            raise ValueError(f"data quality errors:\n  - {bullet}")


# --------------------------------------------------------------------------
# individual checks
# --------------------------------------------------------------------------


def check_claim_id_present(df: pd.DataFrame, **_: Any) -> CheckResult:
    if "claim_id" not in df.columns:
        return CheckResult("claim_id_present", "error", False, "claim_id column is absent")
    n_missing = int(df["claim_id"].isna().sum() + (df["claim_id"] == "").sum())
    return CheckResult(
        "claim_id_present",
        "error",
        n_missing == 0,
        f"{n_missing} rows have no claim_id",
        {"n_missing": n_missing},
    )


def check_claim_id_unique(df: pd.DataFrame, **_: Any) -> CheckResult:
    n_dupes = int(df["claim_id"].duplicated().sum())
    return CheckResult(
        "claim_id_unique",
        "error",
        n_dupes == 0,
        f"{n_dupes} duplicate claim_id values",
        {"n_duplicates": n_dupes},
    )


def check_duplicate_claims(df: pd.DataFrame, **_: Any) -> CheckResult:
    """Identical feature rows carrying different labels are a labelling defect."""
    feature_cols = [
        c for c in df.columns
        if c not in {"claim_id", "fraud_label", "split", "image_id", "image_path"}
    ]
    dupes = df.duplicated(subset=feature_cols, keep=False)
    n_dupe_rows = int(dupes.sum())

    inconsistent = 0
    if n_dupe_rows and "fraud_label" in df.columns:
        grouped = df[dupes].groupby(feature_cols, dropna=False)["fraud_label"].nunique()
        inconsistent = int((grouped > 1).sum())

    return CheckResult(
        "duplicate_claims",
        "warning",
        inconsistent == 0,
        f"{n_dupe_rows} rows share a feature vector with another row; "
        f"{inconsistent} such groups carry conflicting fraud labels",
        {"n_duplicate_rows": n_dupe_rows, "n_inconsistent_groups": inconsistent},
    )


def check_label_values(df: pd.DataFrame, **_: Any) -> CheckResult:
    if "fraud_label" not in df.columns:
        return CheckResult("label_values", "error", False, "fraud_label column is absent")
    observed = set(pd.unique(df["fraud_label"].dropna()))
    valid = observed <= {0, 1}
    n_missing = int(df["fraud_label"].isna().sum())
    return CheckResult(
        "label_values",
        "error",
        valid and n_missing == 0,
        f"fraud_label values {sorted(observed)}, {n_missing} missing",
        {"observed": sorted(int(v) for v in observed), "n_missing": n_missing},
    )


def check_class_balance(df: pd.DataFrame, **_: Any) -> CheckResult:
    counts = df["fraud_label"].value_counts().to_dict()
    positives = int(counts.get(1, 0))
    rate = positives / len(df) if len(df) else 0.0
    return CheckResult(
        "class_balance",
        "info",
        True,
        f"fraud rate {rate:.4f} ({positives} of {len(df)} claims)",
        {
            "n_total": len(df),
            "n_positive": positives,
            "n_negative": int(counts.get(0, 0)),
            "positive_rate": rate,
            "imbalance_ratio": (len(df) - positives) / positives if positives else None,
        },
    )


def check_numeric_validity(df: pd.DataFrame, **_: Any) -> CheckResult:
    """Look for infinities and for ages outside any plausible range."""
    issues: dict[str, Any] = {}
    numeric = df.select_dtypes(include="number")

    import numpy as np

    inf_counts = {
        col: int(np.isinf(numeric[col]).sum())
        for col in numeric.columns
        if np.isinf(numeric[col]).any()
    }
    if inf_counts:
        issues["infinite"] = inf_counts

    if "driver_age" in df.columns:
        implausible = int(((df["driver_age"] < 15) | (df["driver_age"] > 110)).sum())
        if implausible:
            issues["driver_age_out_of_range"] = implausible

    if "deductible" in df.columns:
        negative = int((df["deductible"] < 0).sum())
        if negative:
            issues["negative_deductible"] = negative

    return CheckResult(
        "numeric_validity",
        "warning",
        not issues,
        "no invalid numeric values" if not issues else f"issues found: {issues}",
        issues,
    )


def check_text_non_empty(df: pd.DataFrame, **_: Any) -> CheckResult:
    if "claim_text" not in df.columns:
        return CheckResult(
            "text_non_empty", "info", True, "claim_text not present at this stage"
        )
    text = df["claim_text"].fillna("")
    n_empty = int((text.str.strip() == "").sum())
    lengths = text.str.split().str.len()
    n_very_short = int((lengths < 5).sum())
    return CheckResult(
        "text_non_empty",
        "error",
        n_empty == 0,
        f"{n_empty} empty narratives, {n_very_short} shorter than 5 words",
        {
            "n_empty": n_empty,
            "n_very_short": n_very_short,
            "median_words": float(lengths.median()) if len(lengths) else 0.0,
            "p95_words": float(lengths.quantile(0.95)) if len(lengths) else 0.0,
            "max_words": int(lengths.max()) if len(lengths) else 0,
        },
    )


def check_images_resolve(
    df: pd.DataFrame, *, image_root: Path | None = None, **_: Any
) -> CheckResult:
    """Every referenced image file must exist on disk."""
    if "image_path" not in df.columns or image_root is None:
        return CheckResult(
            "images_resolve", "info", True, "no image references at this stage"
        )
    referenced = df.loc[df["image_path"].notna(), "image_path"]
    missing = [p for p in referenced if not (image_root / p).exists()]
    return CheckResult(
        "images_resolve",
        "error",
        not missing,
        f"{len(missing)} of {len(referenced)} referenced images are missing from disk",
        {"n_referenced": len(referenced), "n_missing": len(missing),
         "examples": missing[:5]},
    )


def check_image_reuse(df: pd.DataFrame, **_: Any) -> CheckResult:
    """An image attached to more than one claim could leak across splits."""
    if "image_id" not in df.columns:
        return CheckResult("image_reuse", "info", True, "no image references at this stage")
    attached = df.loc[df["image_id"].notna(), "image_id"]
    n_reused = int(attached.duplicated().sum())
    return CheckResult(
        "image_reuse",
        "error",
        n_reused == 0,
        f"{n_reused} images are attached to more than one claim",
        {"n_attached": len(attached), "n_reused": n_reused},
    )


def check_corrupted_images(
    df: pd.DataFrame, *, image_root: Path | None = None, verify_images: bool = False, **_: Any
) -> CheckResult:
    """Open every referenced image to confirm it decodes.

    Off by default because decoding thousands of JPEGs is slow; the dataset
    build turns it on once, and records the result.
    """
    if not verify_images or "image_path" not in df.columns or image_root is None:
        return CheckResult(
            "corrupted_images", "info", True, "image decoding not verified in this run"
        )
    from PIL import Image

    corrupted: list[str] = []
    referenced = df.loc[df["image_path"].notna(), "image_path"].tolist()
    for rel in referenced:
        try:
            with Image.open(image_root / rel) as img:
                img.verify()
        except Exception as exc:  # noqa: BLE001 - any decode failure counts
            corrupted.append(f"{rel}: {type(exc).__name__}")

    return CheckResult(
        "corrupted_images",
        "error",
        not corrupted,
        f"{len(corrupted)} of {len(referenced)} referenced images failed to decode",
        {"n_checked": len(referenced), "n_corrupted": len(corrupted),
         "examples": corrupted[:5]},
    )


def check_orphan_images(
    df: pd.DataFrame, *, all_image_ids: Iterable[str] | None = None, **_: Any
) -> CheckResult:
    """Images on disk that no claim references.

    Expected and benign in this project -- the image corpus is much smaller
    than the claim table, so the unattached remainder is reported for
    visibility rather than treated as a defect.
    """
    if all_image_ids is None or "image_id" not in df.columns:
        return CheckResult("orphan_images", "info", True, "image inventory not supplied")
    attached = set(df.loc[df["image_id"].notna(), "image_id"])
    orphans = sorted(set(all_image_ids) - attached)
    return CheckResult(
        "orphan_images",
        "info",
        True,
        f"{len(orphans)} images on disk are not attached to any claim",
        {"n_orphans": len(orphans), "n_attached": len(attached)},
    )


def check_split_disjoint(df: pd.DataFrame, **_: Any) -> CheckResult:
    """No claim, and no image, may appear in more than one split."""
    if "split" not in df.columns:
        return CheckResult("split_disjoint", "info", True, "not split yet")

    per_claim = df.groupby("claim_id")["split"].nunique()
    claim_leaks = int((per_claim > 1).sum())

    image_leaks = 0
    if "image_id" in df.columns:
        attached = df[df["image_id"].notna()]
        if len(attached):
            per_image = attached.groupby("image_id")["split"].nunique()
            image_leaks = int((per_image > 1).sum())

    return CheckResult(
        "split_disjoint",
        "error",
        claim_leaks == 0 and image_leaks == 0,
        f"{claim_leaks} claims and {image_leaks} images appear in more than one split",
        {"claim_leaks": claim_leaks, "image_leaks": image_leaks},
    )


def check_split_coverage(df: pd.DataFrame, **_: Any) -> CheckResult:
    """Every split must exist and contain both classes."""
    if "split" not in df.columns:
        return CheckResult("split_coverage", "info", True, "not split yet")

    details: dict[str, Any] = {}
    ok = True
    for name in ("train", "val", "test"):
        part = df[df["split"] == name]
        positives = int(part["fraud_label"].sum()) if len(part) else 0
        details[name] = {
            "n": len(part),
            "n_positive": positives,
            "positive_rate": positives / len(part) if len(part) else 0.0,
        }
        if len(part) == 0 or positives == 0:
            ok = False

    return CheckResult(
        "split_coverage",
        "error",
        ok,
        "; ".join(
            f"{k}: n={v['n']}, pos={v['n_positive']} ({v['positive_rate']:.4f})"
            for k, v in details.items()
        ),
        details,
    )


CHECKS: list[Callable[..., CheckResult]] = [
    check_claim_id_present,
    check_claim_id_unique,
    check_duplicate_claims,
    check_label_values,
    check_class_balance,
    check_numeric_validity,
    check_text_non_empty,
    check_images_resolve,
    check_image_reuse,
    check_corrupted_images,
    check_orphan_images,
    check_split_disjoint,
    check_split_coverage,
]


def run_all(df: pd.DataFrame, **context: Any) -> QualityReport:
    """Run every registered check and return the collected report."""
    report = QualityReport()
    for check in CHECKS:
        try:
            result = check(df, **context)
        except Exception as exc:  # noqa: BLE001 - a broken check is itself a finding
            result = CheckResult(
                check.__name__, "error", False, f"check raised {type(exc).__name__}: {exc}"
            )
        report.results.append(result)

        level = logger.info
        if not result.passed:
            level = logger.error if result.severity == "error" else logger.warning
        level("[%s] %s: %s", result.severity, result.name, result.message)

    return report
