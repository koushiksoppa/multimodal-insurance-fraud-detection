"""Load the real structured claims table into the unified schema.

This module does the mapping and the *derivations* only. It deliberately does
not impute, clip or drop anything: every anomaly it finds is recorded in a
:class:`LoadReport` so that the data quality stage can report on it and the
modelling stage can make an explicit decision. Silently repairing data here
would hide exactly the problems the quality report is supposed to surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import sources
from src.data.schema import SOURCE_RENAMES
from src.utils.logging import get_logger

logger = get_logger(__name__)

MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]
MONTH_TO_NUMBER = {name: i + 1 for i, name in enumerate(MONTHS)}

#: Values the source table uses to mean "missing" without saying so.
#: ``Age`` uses a numeric zero; the categorical date columns use the string
#: "0". Both are recorded rather than repaired at load time.
SENTINEL_AGE = 0
SENTINEL_CATEGORICAL = "0"


@dataclass
class LoadReport:
    """What the loader observed while mapping the source table."""

    n_rows: int = 0
    n_columns_source: int = 0
    sentinel_age_rows: int = 0
    sentinel_categorical: dict[str, int] = field(default_factory=dict)
    unmapped_source_columns: list[str] = field(default_factory=list)
    negative_report_delay_rows: int = 0
    missing_after_load: dict[str, int] = field(default_factory=dict)
    duplicate_policy_numbers: int = 0
    fully_duplicated_rows: int = 0

    def as_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_columns_source": self.n_columns_source,
            "sentinel_age_rows": self.sentinel_age_rows,
            "sentinel_categorical": self.sentinel_categorical,
            "unmapped_source_columns": self.unmapped_source_columns,
            "negative_report_delay_rows": self.negative_report_delay_rows,
            "missing_after_load": self.missing_after_load,
            "duplicate_policy_numbers": self.duplicate_policy_numbers,
            "fully_duplicated_rows": self.fully_duplicated_rows,
        }


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Read the source CSV exactly as distributed, with no transformation."""
    path = path or sources.structured_csv_path()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/download_data.py --only structured"
        )
    # keep_default_na=False: the source encodes absence with explicit strings
    # such as "none" and "no change", which are meaningful categories rather
    # than missing values. Letting pandas coerce them to NaN would destroy
    # information.
    return pd.read_csv(path, keep_default_na=False, na_values=[])


def to_unified(raw: pd.DataFrame) -> tuple[pd.DataFrame, LoadReport]:
    """Map the source table onto the unified schema.

    Returns the mapped frame and a report of every anomaly encountered.
    """
    report = LoadReport(n_rows=len(raw), n_columns_source=raw.shape[1])

    known = set(SOURCE_RENAMES) | {"FraudFound_P"}
    report.unmapped_source_columns = sorted(set(raw.columns) - known)
    if report.unmapped_source_columns:
        logger.warning(
            "source columns not present in the unified schema: %s",
            report.unmapped_source_columns,
        )

    df = raw.rename(columns=SOURCE_RENAMES).copy()
    df["fraud_label"] = raw["FraudFound_P"].astype("int64")
    df = df.drop(columns=["FraudFound_P"], errors="ignore")

    # Stable primary key. Source row order is the only ordering available, and
    # it is preserved by pandas, so this is reproducible across runs.
    df.insert(0, "claim_id", [f"CLM-{i:07d}" for i in range(1, len(df) + 1)])

    _record_sentinels(df, report)
    df = _derive_report_delay(df, report)
    _record_integrity(raw, df, report)

    report.missing_after_load = {
        col: int(n) for col, n in df.isna().sum().items() if n > 0
    }

    logger.info(
        "mapped %d claims, %d source columns -> %d unified columns",
        len(df), raw.shape[1], df.shape[1],
    )
    return df, report


def _record_sentinels(df: pd.DataFrame, report: LoadReport) -> None:
    """Count undeclared missing-value sentinels without altering them."""
    if "driver_age" in df.columns:
        report.sentinel_age_rows = int((df["driver_age"] == SENTINEL_AGE).sum())

    for col in df.columns:
        if df[col].dtype == object:
            n = int((df[col] == SENTINEL_CATEGORICAL).sum())
            if n:
                report.sentinel_categorical[col] = n

    if report.sentinel_age_rows:
        logger.warning(
            "driver_age is zero for %d rows; treated as an undeclared missing "
            "value, not as a real age",
            report.sentinel_age_rows,
        )


def _derive_report_delay(df: pd.DataFrame, report: LoadReport) -> pd.DataFrame:
    """Approximate the incident-to-claim delay in weeks.

    The source has no dates, only (month name, week-of-month) pairs for the
    incident and for the claim, plus a single calendar year per row. The delay
    is therefore an approximation, and rows where the claim month precedes the
    incident month are assumed to cross a year boundary. Rows that still come
    out negative are counted, not silently clipped.
    """
    incident_month = df["incident_month"].map(MONTH_TO_NUMBER)
    claim_month = df["claim_month"].map(MONTH_TO_NUMBER)

    incident_week = pd.to_numeric(df["incident_week_of_month"], errors="coerce")
    claim_week = pd.to_numeric(df["claim_week_of_month"], errors="coerce")

    # A claim filed in an earlier month than the incident crosses into the next
    # calendar year; add twelve months to the claim side in that case.
    wrapped = claim_month < incident_month
    adjusted_claim_month = claim_month + wrapped.astype("float") * 12

    delay = (adjusted_claim_month - incident_month) * 4.0 + (claim_week - incident_week)

    report.negative_report_delay_rows = int((delay < 0).sum())
    if report.negative_report_delay_rows:
        logger.warning(
            "%d rows have a negative approximate report delay; retained as NaN",
            report.negative_report_delay_rows,
        )
    df["report_delay_weeks"] = delay.where(delay >= 0, np.nan)
    return df


def _record_integrity(raw: pd.DataFrame, df: pd.DataFrame, report: LoadReport) -> None:
    """Check the assumptions the splitting strategy depends on."""
    if "policy_number" in df.columns:
        report.duplicate_policy_numbers = int(
            len(df) - df["policy_number"].nunique()
        )
        if report.duplicate_policy_numbers:
            logger.warning(
                "%d duplicate policy numbers; grouped splitting is required to "
                "avoid policy-level leakage",
                report.duplicate_policy_numbers,
            )
        else:
            logger.info(
                "policy_number is unique per row: there is no repeated-customer "
                "structure in this table, so customer-level leakage cannot occur"
            )

    report.fully_duplicated_rows = int(raw.duplicated().sum())
    if report.fully_duplicated_rows:
        logger.warning(
            "%d fully duplicated source rows", report.fully_duplicated_rows
        )


def load_unified(path: Path | None = None) -> tuple[pd.DataFrame, LoadReport]:
    """Convenience wrapper: read the source CSV and map it in one call."""
    return to_unified(load_raw(path))
