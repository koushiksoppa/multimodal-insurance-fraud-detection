"""The shared evaluation harness and the experiment registry.

The harness is the single code path every model is scored through, so the
properties tested here -- that the threshold comes from validation and that the
test set is scored once with that same threshold -- hold for every cell of the
ablation table, not just for the models that happened to be checked.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.harness import Timer, evaluate_model
from src.evaluation.metrics import capacity_metrics
from src.evaluation.registry import ExperimentRecord, make_experiment_id


@pytest.fixture
def two_splits():
    generator = np.random.default_rng(21)

    def make(n: int, strength: float):
        y = (generator.random(n) < 0.07).astype(int)
        score = 1 / (1 + np.exp(-(strength * y + generator.normal(0, 1.0, n) - 0.4)))
        return y, score

    y_val, score_val = make(1500, 1.2)
    y_test, score_test = make(2000, 1.0)
    return y_val, score_val, y_test, score_test


def test_threshold_is_taken_from_validation_and_reused_on_test(two_splits, tmp_registry):
    y_val, score_val, y_test, score_test = two_splits
    result = evaluate_model(
        model_name="dummy", modalities=["tabular"],
        y_val=y_val, score_val=score_val, y_test=y_test, score_test=score_test,
        n_bootstrap=0, registry=tmp_registry,
    )
    assert result.val.threshold == result.test.threshold == result.threshold.threshold


def test_the_default_operating_point_flags_the_review_capacity(two_splits, tmp_registry):
    """Validated because the earlier F2 default was degenerate on this data."""
    y_val, score_val, y_test, score_test = two_splits
    result = evaluate_model(
        model_name="dummy", modalities=["tabular"],
        y_val=y_val, score_val=score_val, y_test=y_test, score_test=score_test,
        n_bootstrap=0, registry=tmp_registry,
    )
    flagged_on_val = (score_val >= result.threshold.threshold).mean()
    assert flagged_on_val == pytest.approx(0.10, abs=0.01)


def test_predictions_cover_both_evaluation_splits(two_splits, tmp_registry):
    y_val, score_val, y_test, score_test = two_splits
    result = evaluate_model(
        model_name="dummy", modalities=["tabular"],
        y_val=y_val, score_val=score_val, y_test=y_test, score_test=score_test,
        claim_ids_val=np.arange(len(y_val)), claim_ids_test=np.arange(len(y_test)),
        n_bootstrap=0, registry=tmp_registry,
    )
    counts = result.predictions["split"].value_counts()
    assert counts["val"] == len(y_val)
    assert counts["test"] == len(y_test)


def test_the_run_is_recorded_and_can_be_read_back(two_splits, tmp_registry):
    y_val, score_val, y_test, score_test = two_splits
    result = evaluate_model(
        model_name="dummy", modalities=["tabular", "text"],
        y_val=y_val, score_val=score_val, y_test=y_test, score_test=score_test,
        n_bootstrap=0, registry=tmp_registry, hyperparameters={"C": 1.0},
    )
    loaded = tmp_registry.load(result.experiment_id)
    assert loaded.modality_key == "tabular+text"
    assert loaded.hyperparameters["C"] == 1.0
    assert loaded.metrics["test"]["pr_auc"] == pytest.approx(result.test.pr_auc)


def test_rerunning_the_same_configuration_replaces_its_index_row(two_splits, tmp_registry):
    y_val, score_val, y_test, score_test = two_splits
    for _ in range(3):
        evaluate_model(
            model_name="dummy", modalities=["tabular"],
            y_val=y_val, score_val=score_val, y_test=y_test, score_test=score_test,
            n_bootstrap=0, registry=tmp_registry,
        )
    assert len(tmp_registry.index()) == 1


def test_threshold_alternatives_are_recorded(two_splits, tmp_registry):
    y_val, score_val, y_test, score_test = two_splits
    result = evaluate_model(
        model_name="dummy", modalities=["tabular"],
        y_val=y_val, score_val=score_val, y_test=y_test, score_test=score_test,
        n_bootstrap=0, registry=tmp_registry,
    )
    alternatives = result.extra["threshold_alternatives"]
    assert {"f1", "f2", "budget_5pct", "budget_10pct", "budget_20pct"} <= set(alternatives)
    for entry in alternatives.values():
        assert 0.0 <= entry["flagged_fraction"] <= 1.0


def test_experiment_ids_are_deterministic_and_encode_the_configuration():
    first = make_experiment_id(
        model="gbm", modalities=["text", "tabular"], track="real",
        seed=1, split_strategy="temporal",
    )
    second = make_experiment_id(
        model="gbm", modalities=["tabular", "text"], track="real",
        seed=1, split_strategy="temporal",
    )
    assert first == second, "modality order must not change the identifier"
    assert "tab-txt" in first
    assert "temporal" in first


def test_modality_key_is_canonically_ordered():
    record = ExperimentRecord(
        experiment_id="x", model="m", modalities=["image", "tabular"],
        dataset_version="v", track="real", split_strategy="temporal", seed=1,
    )
    assert record.modality_key == "tabular+image"


# --------------------------------------------------------------------------
# capacity metrics
# --------------------------------------------------------------------------


def test_capacity_lift_is_one_for_an_uninformative_ranker():
    generator = np.random.default_rng(5)
    y = (generator.random(20000) < 0.1).astype(int)
    out = capacity_metrics(y, generator.random(20000))
    assert out["lift_at_10pct"] == pytest.approx(1.0, abs=0.25)


def test_capacity_lift_is_maximal_for_a_perfect_ranker():
    y = np.zeros(1000, dtype=int)
    y[:50] = 1  # 5% prevalence
    score = y.astype(float)  # perfect ranking
    out = capacity_metrics(y, score)
    assert out["precision_at_5pct"] == pytest.approx(1.0)
    assert out["recall_at_5pct"] == pytest.approx(1.0)
    assert out["lift_at_5pct"] == pytest.approx(20.0)


def test_recall_at_capacity_increases_with_capacity(scored):
    y, prob = scored
    out = capacity_metrics(y, prob)
    recalls = [out[f"recall_at_{k}pct"] for k in (1, 5, 10, 20)]
    assert recalls == sorted(recalls)


def test_capacity_metrics_appear_in_the_computed_metrics(scored):
    from src.evaluation.metrics import compute_metrics

    y, prob = scored
    m = compute_metrics(y, prob, threshold=0.5)
    assert "lift_at_10pct" in m.capacity
    assert "lift_at_10pct" in m.as_dict()["capacity"]


def test_timer_measures_a_positive_duration():
    with Timer() as timer:
        sum(range(100000))
    assert timer.seconds is not None and timer.seconds > 0
