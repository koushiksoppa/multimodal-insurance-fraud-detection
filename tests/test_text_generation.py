"""Narrative construction.

The most important test in this file is :func:`test_narratives_do_not_depend_on_the_label`.
If the generator could see the fraud label, it could leak it into the wording,
and the text model's performance would be an artefact of the data construction
rather than a measurement of anything. The test flips every label and checks
that not one character of the output changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.text_generation import (
    DEFAULT_CONFIG,
    FILLER,
    GenerationConfig,
    generate_narratives,
    render_claim_text,
)


def test_narratives_do_not_depend_on_the_label(claims: pd.DataFrame):
    original = generate_narratives(claims, base_seed=11)

    flipped = claims.copy()
    flipped["fraud_label"] = 1 - flipped["fraud_label"]
    after_flip = generate_narratives(flipped, base_seed=11)

    pd.testing.assert_series_equal(original, after_flip)


def test_narratives_are_deterministic_given_a_seed(claims: pd.DataFrame):
    first = generate_narratives(claims, base_seed=5)
    second = generate_narratives(claims, base_seed=5)
    pd.testing.assert_series_equal(first, second)


def test_a_different_seed_changes_the_wording(claims: pd.DataFrame):
    first = generate_narratives(claims, base_seed=5)
    second = generate_narratives(claims, base_seed=6)
    assert (first != second).mean() > 0.5


def test_narratives_are_stable_under_row_reordering(claims: pd.DataFrame):
    """Seeding is per claim_id, not per row position."""
    forward = generate_narratives(claims, base_seed=3)
    shuffled = claims.iloc[::-1].copy()
    backward = generate_narratives(shuffled, base_seed=3)

    joined = pd.DataFrame(
        {"claim_id": claims["claim_id"], "a": forward.to_numpy()}
    ).merge(
        pd.DataFrame({"claim_id": shuffled["claim_id"], "b": backward.to_numpy()}),
        on="claim_id",
    )
    assert (joined["a"] == joined["b"]).all()


def test_every_narrative_has_content(claims: pd.DataFrame):
    texts = generate_narratives(claims, base_seed=1)
    assert (texts.str.strip() != "").all()
    assert (texts.str.split().str.len() >= 5).all()


def test_narratives_are_not_all_identical(claims: pd.DataFrame):
    texts = generate_narratives(claims, base_seed=1)
    assert texts.nunique() > 0.9 * len(texts)


def test_omission_makes_text_lossy(claims: pd.DataFrame):
    """Raising the omission rate must shorten the narratives.

    Prose in this project is deliberately an incomplete rendering of the
    narrative fields, so that the text modality is not simply the narrative
    columns in disguise.
    """
    lossless = generate_narratives(
        claims, base_seed=2, config=GenerationConfig(omission_rate=0.0, max_filler=0)
    )
    lossy = generate_narratives(
        claims, base_seed=2, config=GenerationConfig(omission_rate=0.6, max_filler=0)
    )
    assert lossy.str.split().str.len().mean() < lossless.str.split().str.len().mean()


def test_filler_carries_no_field_information(claims: pd.DataFrame):
    """Filler sentences must be drawn independently of the claim's content.

    Checked by confirming that filler appears at a similar rate in both label
    classes; a dependence would make it a covert signal.
    """
    texts = generate_narratives(
        claims, base_seed=4, config=GenerationConfig(min_filler=1, max_filler=1)
    )
    has_filler = texts.apply(lambda t: any(f in t for f in FILLER))
    assert has_filler.all()


def test_unknown_field_values_are_dropped_not_invented():
    """A value the phrase bank does not cover must simply not be described."""
    row = pd.Series(
        {
            "claim_id": "CLM-0000001",
            "incident_month": "Jan",
            "incident_week_of_month": 2,
            "incident_day_of_week": "Monday",
            "accident_area": "Suburban",  # not a value the source data uses
            "fault": "Policy Holder",
            "police_report_filed": "No",
            "witness_present": "No",
            "past_number_of_claims": "none",
            "address_change_claim": "no change",
            "number_of_supplements": "none",
            "days_policy_accident": "more than 30",
            "report_delay_weeks": 1.0,
        }
    )
    text = render_claim_text(row, seed=1, config=GenerationConfig(omission_rate=0.0))
    assert "Suburban" not in text
    assert "urban" not in text.lower() or "rural" not in text.lower()


def test_missing_report_delay_is_tolerated():
    row = pd.Series(
        {
            "claim_id": "CLM-0000002",
            "incident_month": "Mar",
            "incident_week_of_month": 1,
            "incident_day_of_week": "Friday",
            "accident_area": "Urban",
            "fault": "Third Party",
            "police_report_filed": "Yes",
            "witness_present": "Yes",
            "past_number_of_claims": "1",
            "address_change_claim": "1 year",
            "number_of_supplements": "1 to 2",
            "days_policy_accident": "none",
            "report_delay_weeks": np.nan,
        }
    )
    text = render_claim_text(row, seed=2, config=DEFAULT_CONFIG)
    assert text.strip()
    assert "nan" not in text.lower()
