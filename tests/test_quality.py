"""Data quality checks.

Each test injects the defect the check exists to find, because a check that has
never been shown to fail is not evidence of anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import quality


def _named(results, name):
    return next(r for r in results.results if r.name == name)


def test_a_clean_frame_passes_every_error_level_check(claims: pd.DataFrame):
    claims = claims.copy()
    claims["claim_text"] = "The policyholder reports a collision on a city road."
    report = quality.run_all(claims)
    assert report.ok, [r.message for r in report.errors]


def test_missing_claim_id_is_an_error(claims: pd.DataFrame):
    broken = claims.copy()
    broken.loc[broken.index[0], "claim_id"] = ""
    assert not _named(quality.run_all(broken), "claim_id_present").passed


def test_duplicate_claim_id_is_an_error(claims: pd.DataFrame):
    broken = claims.copy()
    broken.loc[broken.index[1], "claim_id"] = broken.loc[broken.index[0], "claim_id"]
    assert not _named(quality.run_all(broken), "claim_id_unique").passed


def test_a_label_outside_zero_one_is_an_error(claims: pd.DataFrame):
    broken = claims.copy()
    broken.loc[broken.index[0], "fraud_label"] = 7
    assert not _named(quality.run_all(broken), "label_values").passed


def test_class_balance_is_reported_as_information(claims: pd.DataFrame):
    result = _named(quality.run_all(claims), "class_balance")
    assert result.severity == "info"
    assert result.passed
    assert 0 < result.details["positive_rate"] < 1
    assert result.details["n_positive"] + result.details["n_negative"] == len(claims)


def test_conflicting_labels_on_identical_records_are_flagged(claims: pd.DataFrame):
    broken = claims.copy()
    donor = broken.iloc[0]
    target = broken.index[1]
    for col in broken.columns:
        if col not in {"claim_id", "fraud_label", "policy_number"}:
            broken.loc[target, col] = donor[col]
    broken.loc[target, "fraud_label"] = 1 - donor["fraud_label"]

    result = _named(quality.run_all(broken), "duplicate_claims")
    assert not result.passed
    assert result.details["n_inconsistent_groups"] >= 1


def test_infinite_numeric_values_are_flagged(claims: pd.DataFrame):
    broken = claims.copy()
    broken["report_delay_weeks"] = broken["report_delay_weeks"].astype(float)
    broken.loc[broken.index[0], "report_delay_weeks"] = np.inf
    result = _named(quality.run_all(broken), "numeric_validity")
    assert not result.passed
    assert "infinite" in result.details


def test_an_implausible_age_is_flagged(claims: pd.DataFrame):
    broken = claims.copy()
    broken.loc[broken.index[0], "driver_age"] = 400
    result = _named(quality.run_all(broken), "numeric_validity")
    assert not result.passed
    assert result.details["driver_age_out_of_range"] == 1


def test_empty_narrative_is_an_error(claims: pd.DataFrame):
    broken = claims.copy()
    broken["claim_text"] = "a normal sentence about the claim goes here"
    broken.loc[broken.index[0], "claim_text"] = "   "
    result = _named(quality.run_all(broken), "text_non_empty")
    assert not result.passed
    assert result.severity == "error"


def test_a_missing_image_file_is_an_error(claims: pd.DataFrame, tmp_path):
    framed = claims.copy()
    framed["image_id"] = None
    framed["image_path"] = None
    framed.loc[framed.index[0], "image_id"] = "img_0"
    framed.loc[framed.index[0], "image_path"] = "does/not/exist.jpg"

    result = _named(
        quality.run_all(framed, image_root=tmp_path), "images_resolve"
    )
    assert not result.passed
    assert result.details["n_missing"] == 1


def test_a_present_image_file_passes(claims: pd.DataFrame, tmp_path):
    from PIL import Image

    (tmp_path / "F_Normal").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "F_Normal" / "a.jpg")

    framed = claims.copy()
    framed["image_id"] = None
    framed["image_path"] = None
    framed.loc[framed.index[0], "image_id"] = "F_Normal__a.jpg"
    framed.loc[framed.index[0], "image_path"] = "F_Normal/a.jpg"

    results = quality.run_all(framed, image_root=tmp_path, verify_images=True)
    assert _named(results, "images_resolve").passed
    assert _named(results, "corrupted_images").passed


def test_a_corrupted_image_is_detected(claims: pd.DataFrame, tmp_path):
    (tmp_path / "F_Normal").mkdir()
    (tmp_path / "F_Normal" / "broken.jpg").write_bytes(b"this is not a jpeg")

    framed = claims.copy()
    framed["image_id"] = None
    framed["image_path"] = None
    framed.loc[framed.index[0], "image_id"] = "F_Normal__broken.jpg"
    framed.loc[framed.index[0], "image_path"] = "F_Normal/broken.jpg"

    result = _named(
        quality.run_all(framed, image_root=tmp_path, verify_images=True),
        "corrupted_images",
    )
    assert not result.passed
    assert result.details["n_corrupted"] == 1


def test_image_reuse_is_an_error(claims: pd.DataFrame):
    framed = claims.copy()
    framed["image_id"] = None
    framed.loc[framed.index[:2], "image_id"] = "shared.jpg"
    assert not _named(quality.run_all(framed), "image_reuse").passed


def test_orphan_images_are_reported_but_not_a_failure(claims: pd.DataFrame):
    framed = claims.copy()
    framed["image_id"] = None
    framed.loc[framed.index[0], "image_id"] = "used.jpg"

    result = _named(
        quality.run_all(framed, all_image_ids=["used.jpg", "spare_a.jpg", "spare_b.jpg"]),
        "orphan_images",
    )
    assert result.passed
    assert result.severity == "info"
    assert result.details["n_orphans"] == 2


def test_a_split_with_no_positives_is_an_error(claims: pd.DataFrame):
    framed = claims.copy()
    framed["split"] = "train"
    framed.loc[framed.index[:50], "split"] = "val"
    framed.loc[framed.index[:50], "fraud_label"] = 0
    framed.loc[framed.index[50:100], "split"] = "test"

    assert not _named(quality.run_all(framed), "split_coverage").passed


def test_a_check_that_raises_becomes_a_failing_result(claims: pd.DataFrame, monkeypatch):
    def exploding_check(df, **kwargs):
        raise RuntimeError("boom")

    exploding_check.__name__ = "exploding_check"
    monkeypatch.setattr(quality, "CHECKS", [*quality.CHECKS, exploding_check])

    result = _named(quality.run_all(claims), "exploding_check")
    assert not result.passed
    assert "RuntimeError" in result.message


def test_raise_on_error_reports_every_failure(claims: pd.DataFrame):
    broken = claims.copy()
    broken.loc[broken.index[0], "claim_id"] = ""
    broken.loc[broken.index[0], "fraud_label"] = 9

    with pytest.raises(ValueError) as excinfo:
        quality.run_all(broken).raise_on_error()
    assert "claim_id_present" in str(excinfo.value)
    assert "label_values" in str(excinfo.value)


def test_summary_table_has_one_row_per_check(claims: pd.DataFrame):
    report = quality.run_all(claims)
    assert len(report.summary_table()) == len(report.results) == len(quality.CHECKS)
