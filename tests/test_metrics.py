"""Evaluation metrics and threshold selection."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    bootstrap_intervals,
    compute_metrics,
    expected_calibration_error,
    reliability_curve,
    select_threshold,
)


def test_perfect_scores_give_perfect_ranking_metrics():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.01, 0.02, 0.03, 0.97, 0.99])
    m = compute_metrics(y, scores, threshold=0.5)
    assert m.roc_auc == pytest.approx(1.0)
    assert m.pr_auc == pytest.approx(1.0)
    assert m.precision == pytest.approx(1.0)
    assert m.recall == pytest.approx(1.0)
    assert m.fp == 0 and m.fn == 0


def test_random_scores_give_pr_auc_near_prevalence():
    """PR-AUC of an uninformative ranker equals the base rate.

    This is why the reports carry PR-AUC lift alongside PR-AUC: the raw number
    is not interpretable without knowing the prevalence it is being compared
    against.
    """
    generator = np.random.default_rng(0)
    y = (generator.random(20000) < 0.1).astype(int)
    scores = generator.random(20000)
    m = compute_metrics(y, scores, threshold=0.5)
    assert m.pr_auc == pytest.approx(m.prevalence, abs=0.02)
    assert m.pr_auc_lift == pytest.approx(1.0, abs=0.2)
    assert m.roc_auc == pytest.approx(0.5, abs=0.02)


def test_confusion_counts_sum_to_the_sample_size(scored):
    y, prob = scored
    m = compute_metrics(y, prob, threshold=0.5)
    assert m.tp + m.fp + m.tn + m.fn == len(y)
    assert m.tp + m.fn == m.n_positive


def test_single_class_evaluation_is_rejected():
    with pytest.raises(ValueError, match="single class"):
        compute_metrics(np.zeros(10, dtype=int), np.linspace(0, 1, 10))


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="length mismatch"):
        compute_metrics(np.array([0, 1]), np.array([0.5]))


def test_a_perfectly_calibrated_predictor_has_near_zero_calibration_error():
    generator = np.random.default_rng(7)
    prob = generator.uniform(0.01, 0.99, size=60000)
    y = (generator.random(60000) < prob).astype(int)
    assert expected_calibration_error(y, prob, n_bins=10) < 0.01


def test_a_systematically_overconfident_predictor_is_detected():
    generator = np.random.default_rng(8)
    truth = generator.uniform(0.0, 0.4, size=40000)
    y = (generator.random(40000) < truth).astype(int)
    inflated = np.clip(truth + 0.3, 0, 1)
    assert expected_calibration_error(y, inflated, n_bins=10) > 0.2


def test_reliability_curve_bins_cover_every_sample(scored):
    y, prob = scored
    curve = reliability_curve(y, prob, n_bins=10)
    assert sum(curve["count"]) == len(y)
    assert len(curve["mean_predicted"]) == len(curve["observed_frequency"])


# --------------------------------------------------------------------------
# threshold selection
# --------------------------------------------------------------------------


def test_f1_threshold_beats_the_default_on_the_data_it_was_chosen_on(scored):
    y, prob = scored
    choice = select_threshold(y, prob, strategy="f1")
    tuned = compute_metrics(y, prob, threshold=choice.threshold)
    default = compute_metrics(y, prob, threshold=0.5)
    assert tuned.f1 >= default.f1


def test_fbeta_threshold_favours_recall_over_f1(scored):
    y, prob = scored
    f1_choice = select_threshold(y, prob, strategy="f1")
    f2_choice = select_threshold(y, prob, strategy="fbeta", beta=2.0)

    f1_recall = compute_metrics(y, prob, threshold=f1_choice.threshold).recall
    f2_recall = compute_metrics(y, prob, threshold=f2_choice.threshold).recall
    assert f2_recall >= f1_recall


def test_review_budget_threshold_flags_the_requested_fraction(scored):
    y, prob = scored
    choice = select_threshold(y, prob, strategy="review_budget", review_budget=0.05)
    flagged = (prob >= choice.threshold).mean()
    assert flagged == pytest.approx(0.05, abs=0.01)


def test_precision_at_recall_reaches_its_recall_floor(scored):
    y, prob = scored
    choice = select_threshold(y, prob, strategy="precision_at_recall", target_recall=0.4)
    assert compute_metrics(y, prob, threshold=choice.threshold).recall >= 0.4


def test_unreachable_recall_target_falls_back_rather_than_failing(scored):
    y, prob = scored
    choice = select_threshold(
        y, prob, strategy="precision_at_recall", target_recall=1.01
    )
    assert choice.strategy == "f1"


def test_fixed_strategy_returns_what_it_was_given(scored):
    y, prob = scored
    assert select_threshold(y, prob, strategy="fixed", fixed=0.31).threshold == 0.31


def test_threshold_selection_is_deterministic(scored):
    y, prob = scored
    assert (
        select_threshold(y, prob, strategy="f1").threshold
        == select_threshold(y, prob, strategy="f1").threshold
    )


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------


def test_bootstrap_intervals_bracket_the_point_estimate(scored):
    y, prob = scored
    point = compute_metrics(y, prob, threshold=0.5)
    intervals = bootstrap_intervals(y, prob, threshold=0.5, n_bootstrap=200, seed=1)

    for name, value in (
        ("roc_auc", point.roc_auc),
        ("pr_auc", point.pr_auc),
        ("brier", point.brier),
    ):
        low, high = intervals[name]
        assert low <= value <= high, f"{name}: {value} outside [{low}, {high}]"


def test_bootstrap_is_reproducible(scored):
    y, prob = scored
    a = bootstrap_intervals(y, prob, threshold=0.5, n_bootstrap=100, seed=3)
    b = bootstrap_intervals(y, prob, threshold=0.5, n_bootstrap=100, seed=3)
    assert a == b


def test_stratified_bootstrap_never_loses_the_positive_class():
    """An unstratified bootstrap on rare positives can draw an all-negative
    replicate, for which ranking metrics are undefined. The stratified
    resampler used here cannot."""
    generator = np.random.default_rng(12)
    y = np.zeros(400, dtype=int)
    y[:6] = 1
    prob = generator.random(400)
    intervals = bootstrap_intervals(y, prob, threshold=0.5, n_bootstrap=100, seed=2)
    assert np.isfinite(intervals["roc_auc"][0])


def test_metrics_round_trip_through_a_dict(scored):
    y, prob = scored
    m = compute_metrics(y, prob, threshold=0.4, n_bootstrap=20)
    payload = m.as_dict()
    assert payload["pr_auc"] == m.pr_auc
    assert "pr_auc" in payload["intervals"]
