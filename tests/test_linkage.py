"""Image-to-claim attachment.

The attachment rule underpins two claims made elsewhere in the project: that no
photograph can leak across a split, and that the image arm on the real-label
track is a negative control rather than a source of hidden signal. Both are
tested here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.linkage import ImageInventory, attach_images


@pytest.fixture
def inventory() -> ImageInventory:
    conditions = ["normal", "breakage", "crushed"]
    rows = []
    for i in range(300):
        condition = conditions[i % 3]
        location = "front" if i % 2 == 0 else "rear"
        rows.append(
            {
                "image_id": f"img_{i:04d}",
                "image_path": f"{location[0].upper()}_{condition}/{i}.jpg",
                "damage_class": f"{location[0].upper()}_{condition}",
                "damage_location": location,
                "damage_condition": condition,
                "damage_severity": conditions.index(condition),
            }
        )
    return ImageInventory(frame=pd.DataFrame(rows), root=None)  # type: ignore[arg-type]


def test_no_image_is_attached_to_more_than_one_claim(claims, inventory):
    linked, _ = attach_images(claims, inventory, base_seed=1)
    attached = linked.loc[linked["image_id"].notna(), "image_id"]
    assert not attached.duplicated().any()


def test_every_image_is_used_exactly_once(claims, inventory):
    linked, report = attach_images(claims, inventory, base_seed=1)
    assert report.n_attached == len(inventory)
    attached = set(linked.loc[linked["image_id"].notna(), "image_id"])
    assert attached == set(inventory.frame["image_id"])


def test_uncovered_claims_have_no_image_fields(claims, inventory):
    linked, _ = attach_images(claims, inventory, base_seed=1)
    uncovered = linked[~linked["has_image"]]
    assert uncovered["image_id"].isna().all()
    assert uncovered["image_path"].isna().all()
    assert uncovered["damage_condition"].isna().all()


def test_attachment_preserves_the_fraud_base_rate(claims, inventory):
    linked, report = attach_images(claims, inventory, base_seed=1)
    assert report.prevalence_covered == pytest.approx(report.prevalence_full, abs=0.02)


def test_coverage_is_equal_across_classes(claims, inventory):
    _, report = attach_images(claims, inventory, base_seed=1)
    rates = list(report.coverage_by_class.values())
    assert max(rates) - min(rates) < 0.02


def test_attachment_carries_no_information_about_the_label(claims, inventory):
    """The damage condition attached to a claim must be independent of fraud.

    This is the property that makes the image arm a negative control. It is
    checked with a chi-squared test of independence between the attached damage
    condition and the label.
    """
    from scipy.stats import chi2_contingency

    linked, _ = attach_images(claims, inventory, base_seed=1)
    covered = linked[linked["has_image"]]
    table = pd.crosstab(covered["damage_condition"], covered["fraud_label"])
    _, p_value, _, _ = chi2_contingency(table)
    assert p_value > 0.01, (
        "attached damage condition is associated with the fraud label; the image "
        f"arm would not be a clean negative control (p={p_value:.4f})"
    )


def test_attachment_is_deterministic(claims, inventory):
    first, _ = attach_images(claims, inventory, base_seed=9)
    second, _ = attach_images(claims, inventory, base_seed=9)
    pd.testing.assert_frame_equal(first, second)


def test_a_different_seed_changes_the_attachment(claims, inventory):
    first, _ = attach_images(claims, inventory, base_seed=9)
    second, _ = attach_images(claims, inventory, base_seed=10)
    assert not first["image_id"].equals(second["image_id"])


def test_more_images_than_claims_is_refused(small_claims, inventory):
    with pytest.raises(ValueError, match="at most one claim"):
        attach_images(small_claims, inventory, base_seed=1)


def test_empty_inventory_is_refused(claims):
    empty = ImageInventory(frame=pd.DataFrame(columns=["image_id"]), root=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty"):
        attach_images(claims, empty, base_seed=1)


def test_severity_is_ordinal_and_matches_the_condition(claims, inventory):
    linked, _ = attach_images(claims, inventory, base_seed=1)
    covered = linked[linked["has_image"]]
    mapping = (
        covered.groupby("damage_condition")["damage_severity"].nunique().to_dict()
    )
    assert all(v == 1 for v in mapping.values()), "a condition maps to two severities"
