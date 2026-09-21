"""Shared fixtures.

The fixtures build a small in-memory claim table with the same column names,
category levels and quirks as the real source data, so the whole test suite
runs without the downloaded datasets. Tests that genuinely need the real files
are marked ``network`` and are skipped when the data is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.tabular import ORDINAL_ORDERS

MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]
DAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
]


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


def _make_claims(n: int, seed: int = 1234) -> pd.DataFrame:
    """A synthetic claim table shaped like the real one.

    The label is generated from a couple of the columns with substantial noise,
    so tests that train a model see something learnable but not separable.
    """
    generator = np.random.default_rng(seed)

    years = generator.choice([1994, 1995, 1996], size=n, p=[0.4, 0.34, 0.26])
    months = generator.choice(MONTHS, size=n)
    month_num = np.array([MONTHS.index(m) + 1 for m in months])

    # Claims are filed at or after the incident, within a few weeks.
    claim_month_num = np.clip(month_num + generator.integers(0, 2, size=n), 1, 12)

    frame = pd.DataFrame(
        {
            "claim_id": [f"CLM-{i:07d}" for i in range(1, n + 1)],
            "policy_number": np.arange(1, n + 1),
            "rep_number": generator.integers(1, 17, size=n),
            # tabular channel
            "make": generator.choice(["Honda", "Toyota", "Ford", "Mazda", "BMW"], size=n),
            "vehicle_category": generator.choice(["Sedan", "Sport", "Utility"], size=n),
            "vehicle_price_band": generator.choice(
                ORDINAL_ORDERS["vehicle_price_band"], size=n
            ),
            "vehicle_age_band": generator.choice(ORDINAL_ORDERS["vehicle_age_band"], size=n),
            "driver_age": generator.integers(18, 75, size=n),
            "driver_sex": generator.choice(["Male", "Female"], size=n),
            "marital_status": generator.choice(
                ["Single", "Married", "Divorced", "Widow"], size=n
            ),
            "policyholder_age_band": generator.choice(
                ORDINAL_ORDERS["policyholder_age_band"], size=n
            ),
            "policy_type": generator.choice(
                ["Sedan - Collision", "Sedan - Liability", "Sport - All Perils"], size=n
            ),
            "base_policy": generator.choice(["Liability", "Collision", "All Perils"], size=n),
            "deductible": generator.choice([300, 400, 500, 700], size=n),
            "driver_rating": generator.integers(1, 5, size=n),
            "number_of_cars": generator.choice(ORDINAL_ORDERS["number_of_cars"], size=n),
            "agent_type": generator.choice(["External", "Internal"], size=n),
            "policy_year": years,
            # narrative channel
            "accident_area": generator.choice(["Urban", "Rural"], size=n, p=[0.9, 0.1]),
            "fault": generator.choice(["Policy Holder", "Third Party"], size=n),
            "police_report_filed": generator.choice(["No", "Yes"], size=n, p=[0.98, 0.02]),
            "witness_present": generator.choice(["No", "Yes"], size=n, p=[0.98, 0.02]),
            "days_policy_accident": generator.choice(
                ORDINAL_ORDERS["days_policy_accident"], size=n
            ),
            "days_policy_claim": generator.choice(
                ORDINAL_ORDERS["days_policy_claim"], size=n
            ),
            "past_number_of_claims": generator.choice(
                ORDINAL_ORDERS["past_number_of_claims"], size=n
            ),
            "address_change_claim": generator.choice(
                ORDINAL_ORDERS["address_change_claim"], size=n
            ),
            "number_of_supplements": generator.choice(
                ORDINAL_ORDERS["number_of_supplements"], size=n
            ),
            "incident_month": months,
            "incident_week_of_month": generator.integers(1, 6, size=n),
            "incident_day_of_week": generator.choice(DAYS, size=n),
            "claim_month": [MONTHS[m - 1] for m in claim_month_num],
            "claim_week_of_month": generator.integers(1, 6, size=n),
            "claim_day_of_week": generator.choice(DAYS, size=n),
            "report_delay_weeks": (claim_month_num - month_num) * 4.0,
        }
    )

    # A noisy label driven by one tabular and one narrative field, so both
    # channels carry signal and neither determines the outcome.
    logit = (
        -2.9
        + 0.8 * (frame["base_policy"] == "All Perils")
        + 0.7 * (frame["fault"] == "Policy Holder")
        + 0.4 * (frame["address_change_claim"] == "under 6 months")
        + generator.normal(0, 1.0, size=n)
    )
    frame["fraud_label"] = (1 / (1 + np.exp(-logit)) > generator.random(n)).astype(int)
    return frame


@pytest.fixture
def claims() -> pd.DataFrame:
    """A 900-row synthetic claim table with both classes present."""
    return _make_claims(900)


@pytest.fixture
def small_claims() -> pd.DataFrame:
    return _make_claims(120, seed=7)


@pytest.fixture
def scored() -> tuple[np.ndarray, np.ndarray]:
    """Labels and probability-like scores for metric tests.

    Scores are informative but far from perfect, which is the regime the
    project's metrics actually operate in.
    """
    generator = np.random.default_rng(99)
    n = 2000
    y = (generator.random(n) < 0.08).astype(int)
    noise = generator.normal(0, 1.1, size=n)
    raw = 0.9 * y + noise
    prob = 1 / (1 + np.exp(-(raw - 0.5)))
    return y, prob


@pytest.fixture
def real_data_available() -> bool:
    from src.data.sources import image_root_path, structured_csv_path

    return structured_csv_path().exists() and image_root_path().exists()


@pytest.fixture
def tmp_registry(tmp_path: Path):
    from src.evaluation.registry import ExperimentRegistry

    return ExperimentRegistry(root=tmp_path / "runs")
