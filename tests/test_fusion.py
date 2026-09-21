"""Fusion architectures, modality blocks and the training loop.

The tests that matter most here are the missing-modality ones. The project's
claim that the system degrades gracefully rather than crashing is only worth
making if a masked-out modality provably contributes nothing to the output, so
that is checked directly rather than inferred from the code reading correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.multimodal.dataset import (
    ModalityBlock,
    MultimodalDataset,
    build_blocks,
    collate,
    make_loader,
    subset_blocks,
)
from src.models.multimodal.fusion import FusionConfig, FusionModel
from src.models.multimodal.trainer import FusionTrainer, TrainConfig, positive_weight

STRATEGIES = ["concat", "gated", "attention"]


def _blocks(n: int = 200, *, image_coverage: float = 0.3, seed: int = 0):
    generator = np.random.default_rng(seed)
    present_image = generator.random(n) < image_coverage
    return {
        "tabular": ModalityBlock(
            "tabular", generator.normal(size=(n, 12)), np.ones(n, dtype=bool)
        ),
        "text": ModalityBlock(
            "text", generator.normal(size=(n, 16)), np.ones(n, dtype=bool)
        ),
        "image": ModalityBlock(
            "image", generator.normal(size=(n, 8)), present_image
        ),
    }


def _dataset(n: int = 200, seed: int = 0) -> MultimodalDataset:
    generator = np.random.default_rng(seed + 100)
    labels = (generator.random(n) < 0.15).astype(int)
    # Guarantee both classes: a single-class split makes PR-AUC undefined.
    labels[0], labels[1] = 0, 1
    return MultimodalDataset(_blocks(n, seed=seed), labels)


# --------------------------------------------------------------------------
# modality blocks
# --------------------------------------------------------------------------


def test_absent_rows_are_zeroed_on_construction():
    features = np.ones((10, 4))
    present = np.array([True] * 5 + [False] * 5)
    block = ModalityBlock("image", features, present)
    assert (block.features[5:] == 0).all()
    assert (block.features[:5] == 1).all()


def test_mismatched_mask_length_is_rejected():
    with pytest.raises(ValueError, match="mask entries"):
        ModalityBlock("image", np.ones((10, 4)), np.ones(9, dtype=bool))


def test_subsetting_keeps_features_and_mask_aligned():
    block = _blocks(50)["image"]
    index = np.array([3, 7, 11])
    subset = block.subset(index)
    np.testing.assert_array_equal(subset.present, block.present[index])
    np.testing.assert_array_equal(subset.features, block.features[index])


def test_build_blocks_derives_image_presence_from_the_claim_table(claims: pd.DataFrame):
    frame = claims.copy()
    frame["has_image"] = [True, False] * (len(frame) // 2)
    blocks = build_blocks(
        frame,
        tabular=np.zeros((len(frame), 5)),
        text=np.zeros((len(frame), 6)),
        image=np.zeros((len(frame), 7)),
    )
    assert blocks["image"].coverage() == pytest.approx(0.5)
    assert blocks["tabular"].coverage() == 1.0


def test_requesting_a_modality_without_supplying_it_fails(claims: pd.DataFrame):
    with pytest.raises(ValueError, match="text features requested"):
        build_blocks(claims, tabular=np.zeros((len(claims), 3)), modalities=["tabular", "text"])


def test_inconsistent_block_lengths_are_rejected():
    blocks = _blocks(50)
    blocks["image"] = blocks["image"].subset(np.arange(10))
    with pytest.raises(ValueError, match="inconsistent lengths"):
        MultimodalDataset(blocks, np.zeros(50))


def test_collate_preserves_the_per_modality_structure():
    dataset = _dataset(16)
    features, masks, labels = collate([dataset[i] for i in range(8)])
    assert set(features) == {"tabular", "text", "image"}
    assert features["tabular"].shape == (8, 12)
    assert masks["image"].shape == (8,)
    assert labels.shape == (8,)


def test_loader_shuffling_is_reproducible():
    dataset = _dataset(64)

    def first_batch_labels(seed):
        loader = make_loader(dataset, batch_size=16, shuffle=True, seed=seed)
        return next(iter(loader))[2].numpy()

    np.testing.assert_array_equal(first_batch_labels(1), first_batch_labels(1))


# --------------------------------------------------------------------------
# fusion models
# --------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_each_strategy_produces_one_logit_per_claim(strategy: str):
    config = FusionConfig(
        strategy=strategy, input_dims={"tabular": 12, "text": 16, "image": 8}
    )
    model = FusionModel(config)
    features = {k: torch.randn(5, d) for k, d in config.input_dims.items()}
    masks = {k: torch.ones(5, dtype=torch.bool) for k in config.input_dims}
    assert model(features, masks).shape == (5,)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_a_masked_modality_cannot_influence_the_output(strategy: str):
    """The central missing-modality guarantee.

    Changing the image features of a claim whose image is masked out must not
    move its prediction by one bit. If it did, the model would be reading data
    it has been told is absent, and every missing-modality result in the
    project would be meaningless.
    """
    torch.manual_seed(0)
    config = FusionConfig(
        strategy=strategy, input_dims={"tabular": 12, "text": 16, "image": 8}
    )
    model = FusionModel(config).eval()

    features = {k: torch.randn(4, d) for k, d in config.input_dims.items()}
    masks = {
        "tabular": torch.ones(4, dtype=torch.bool),
        "text": torch.ones(4, dtype=torch.bool),
        "image": torch.zeros(4, dtype=torch.bool),
    }

    with torch.no_grad():
        before = model(features, masks)
        features["image"] = torch.randn(4, 8) * 100
        after = model(features, masks)

    torch.testing.assert_close(before, after)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_a_present_modality_does_influence_the_output(strategy: str):
    """The converse: the masking must not be disabling everything."""
    torch.manual_seed(0)
    config = FusionConfig(
        strategy=strategy, input_dims={"tabular": 12, "text": 16, "image": 8}
    )
    model = FusionModel(config).eval()

    features = {k: torch.randn(4, d) for k, d in config.input_dims.items()}
    masks = {k: torch.ones(4, dtype=torch.bool) for k in config.input_dims}

    with torch.no_grad():
        before = model(features, masks)
        features["image"] = torch.randn(4, 8) * 10
        after = model(features, masks)

    assert not torch.allclose(before, after)


def test_gated_fusion_gives_a_masked_modality_no_weight():
    torch.manual_seed(0)
    config = FusionConfig(
        strategy="gated", input_dims={"tabular": 12, "text": 16, "image": 8}
    )
    model = FusionModel(config).eval()

    features = {k: torch.randn(6, d) for k, d in config.input_dims.items()}
    masks = {
        "tabular": torch.ones(6, dtype=torch.bool),
        "text": torch.ones(6, dtype=torch.bool),
        "image": torch.zeros(6, dtype=torch.bool),
    }
    with torch.no_grad():
        model(features, masks)

    gates = model.last_gates
    image_position = config.modalities.index("image")
    assert torch.allclose(gates[:, image_position], torch.zeros(6))
    # The remaining modalities must still form a distribution.
    assert torch.allclose(gates.sum(dim=1), torch.ones(6), atol=1e-5)


def test_a_single_modality_model_is_valid():
    config = FusionConfig(strategy="concat", input_dims={"text": 16})
    model = FusionModel(config)
    out = model({"text": torch.randn(3, 16)}, {"text": torch.ones(3, dtype=torch.bool)})
    assert out.shape == (3,)


def test_no_modality_is_rejected():
    with pytest.raises(ValueError, match="at least one modality"):
        FusionModel(FusionConfig(input_dims={}))


def test_an_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown fusion strategy"):
        FusionModel(FusionConfig(strategy="telepathy", input_dims={"text": 4}))


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_gradients_reach_every_projection(strategy: str):
    config = FusionConfig(
        strategy=strategy, input_dims={"tabular": 12, "text": 16, "image": 8}
    )
    model = FusionModel(config)
    features = {k: torch.randn(8, d) for k, d in config.input_dims.items()}
    masks = {k: torch.ones(8, dtype=torch.bool) for k in config.input_dims}

    model(features, masks).sum().backward()
    for name in config.modalities:
        weight = model.projections[name].net[0].weight
        assert weight.grad is not None and weight.grad.abs().sum() > 0, name


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def test_positive_weight_interpolates_between_none_and_full():
    labels = np.array([0] * 90 + [1] * 10)  # 9:1
    assert positive_weight(labels, 0.0) == pytest.approx(1.0)
    assert positive_weight(labels, 1.0) == pytest.approx(9.0)
    assert positive_weight(labels, 0.5) == pytest.approx(3.0)


def test_positive_weight_handles_a_split_with_no_positives():
    assert positive_weight(np.zeros(50, dtype=int), 0.5) == 1.0


@pytest.mark.slow
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_training_runs_and_produces_valid_probabilities(strategy: str):
    train, val = _dataset(300, seed=1), _dataset(150, seed=2)
    config = FusionConfig(strategy=strategy, input_dims=train.input_dims)
    trainer = FusionTrainer(
        config, TrainConfig(epochs=4, patience=4, verbose=False)
    ).fit(train, val)

    probabilities = trainer.predict_proba(val)
    assert probabilities.shape == (len(val),)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


@pytest.mark.slow
def test_training_restores_the_best_checkpoint_not_the_last():
    train, val = _dataset(300, seed=3), _dataset(150, seed=4)
    config = FusionConfig(strategy="concat", input_dims=train.input_dims)
    trainer = FusionTrainer(
        config, TrainConfig(epochs=6, patience=6, verbose=False)
    ).fit(train, val)

    from sklearn.metrics import average_precision_score

    restored = average_precision_score(val.labels, trainer.predict_proba(val))
    assert restored == pytest.approx(trainer.history.best_val_pr_auc, abs=1e-6)


@pytest.mark.slow
def test_predicting_before_fitting_is_refused():
    dataset = _dataset(50)
    trainer = FusionTrainer(
        FusionConfig(strategy="concat", input_dims=dataset.input_dims)
    )
    with pytest.raises(RuntimeError, match="not fitted"):
        trainer.predict_proba(dataset)


@pytest.mark.slow
def test_gates_are_reported_only_for_the_gated_strategy():
    train, val = _dataset(200, seed=5), _dataset(100, seed=6)

    gated = FusionTrainer(
        FusionConfig(strategy="gated", input_dims=train.input_dims),
        TrainConfig(epochs=2, verbose=False),
    ).fit(train, val)
    concat = FusionTrainer(
        FusionConfig(strategy="concat", input_dims=train.input_dims),
        TrainConfig(epochs=2, verbose=False),
    ).fit(train, val)

    gates = gated.modality_gates(val)
    assert gates is not None
    assert set(gates) == {"tabular", "text", "image"}
    assert sum(gates.values()) == pytest.approx(1.0, abs=0.05)
    assert concat.modality_gates(val) is None


@pytest.mark.slow
def test_training_is_reproducible_for_a_fixed_seed():
    train, val = _dataset(200, seed=7), _dataset(100, seed=8)
    config = FusionConfig(strategy="concat", input_dims=train.input_dims)

    def run():
        return FusionTrainer(
            config, TrainConfig(epochs=3, seed=99, verbose=False)
        ).fit(train, val).predict_proba(val)

    np.testing.assert_allclose(run(), run(), rtol=1e-5)


def test_subset_blocks_matches_manual_indexing():
    blocks = _blocks(40)
    index = np.array([1, 5, 9])
    subset = subset_blocks(blocks, index)
    for name, block in subset.items():
        np.testing.assert_array_equal(block.features, blocks[name].features[index])
