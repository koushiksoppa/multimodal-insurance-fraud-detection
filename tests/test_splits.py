"""Splitting and the leakage audit."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.splits import (
    SplitConfig,
    assign_splits,
    audit_splits,
    raise_on_leakage,
)


def _with_splits(claims: pd.DataFrame, strategy: str) -> pd.DataFrame:
    out = claims.copy()
    out["split"], _ = assign_splits(out, SplitConfig(strategy=strategy))
    return out


@pytest.mark.parametrize("strategy", ["temporal", "stratified_random"])
def test_every_row_lands_in_exactly_one_split(claims: pd.DataFrame, strategy: str):
    splits, report = assign_splits(claims, SplitConfig(strategy=strategy))
    assert set(splits.unique()) <= {"train", "val", "test"}
    assert sum(report.sizes.values()) == len(claims)


@pytest.mark.parametrize("strategy", ["temporal", "stratified_random"])
def test_all_three_splits_contain_both_classes(claims: pd.DataFrame, strategy: str):
    _, report = assign_splits(claims, SplitConfig(strategy=strategy))
    for name in ("train", "val", "test"):
        assert report.sizes[name] > 0, f"{name} is empty"
        assert report.positives[name] > 0, f"{name} has no positives"


def test_temporal_split_does_not_put_the_future_in_train(claims: pd.DataFrame):
    framed = _with_splits(claims, "temporal")
    findings = {f.name: f for f in audit_splits(framed, strategy="temporal")}
    assert findings["temporal_order"].passed


def test_stratified_split_preserves_the_base_rate(claims: pd.DataFrame):
    _, report = assign_splits(claims, SplitConfig(strategy="stratified_random"))
    overall = claims["fraud_label"].mean()
    for name in ("train", "val", "test"):
        assert report.positive_rates[name] == pytest.approx(overall, abs=0.02)


def test_stratified_split_is_deterministic(claims: pd.DataFrame):
    first, _ = assign_splits(claims, SplitConfig(strategy="stratified_random", seed=42))
    second, _ = assign_splits(claims, SplitConfig(strategy="stratified_random", seed=42))
    pd.testing.assert_series_equal(first, second)


@pytest.mark.parametrize("strategy", ["temporal", "stratified_random"])
def test_leakage_audit_passes_on_a_correct_split(claims: pd.DataFrame, strategy: str):
    framed = _with_splits(claims, strategy)
    raise_on_leakage(audit_splits(framed, strategy=strategy))


def test_leakage_audit_catches_a_claim_in_two_splits(claims: pd.DataFrame):
    framed = _with_splits(claims, "stratified_random")
    # Copy one training row into the test split under the same claim_id.
    leaked = framed[framed["split"] == "train"].iloc[[0]].copy()
    leaked["split"] = "test"
    contaminated = pd.concat([framed, leaked], ignore_index=True)

    findings = {f.name: f for f in audit_splits(contaminated, strategy="stratified_random")}
    assert not findings["claim_disjoint"].passed
    with pytest.raises(ValueError, match="leakage audit failed"):
        raise_on_leakage(list(findings.values()))


def test_leakage_audit_catches_a_reused_image(claims: pd.DataFrame):
    framed = _with_splits(claims, "stratified_random")
    framed["image_id"] = None
    train_idx = framed.index[framed["split"] == "train"][0]
    test_idx = framed.index[framed["split"] == "test"][0]
    framed.loc[[train_idx, test_idx], "image_id"] = "shared.jpg"

    findings = {f.name: f for f in audit_splits(framed, strategy="stratified_random")}
    assert not findings["image_disjoint"].passed


def test_leakage_audit_catches_a_duplicate_record_across_splits(claims: pd.DataFrame):
    framed = _with_splits(claims, "stratified_random")
    donor = framed[framed["split"] == "train"].iloc[0]
    target_idx = framed.index[framed["split"] == "test"][0]

    feature_cols = [
        c for c in framed.columns
        if c not in {"claim_id", "policy_number", "split", "fraud_label"}
    ]
    for col in feature_cols:
        framed.loc[target_idx, col] = donor[col]

    findings = {f.name: f for f in audit_splits(framed, strategy="stratified_random")}
    assert not findings["duplicate_across_splits"].passed


def test_temporal_boundaries_are_reported(claims: pd.DataFrame):
    _, report = assign_splits(claims, SplitConfig(strategy="temporal"))
    assert report.boundaries["test"].startswith(">=")
    assert any("time" in note for note in report.notes)
