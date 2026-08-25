import math

import pytest
import torch

from werewolf.models.twd_tom.baselines import (
    EMPIRICAL_PRIOR_SMOOTHING,
    EMPIRICAL_PRIOR_VERSION,
    evaluate_dense_empirical_priors,
    fit_dense_empirical_priors,
)
from werewolf.models.twd_tom.dense_dataset import DenseTWDToMDataset


def test_dense_empirical_priors_are_fit_on_unaugmented_training_labels(
    training_sample_factory,
):
    training = DenseTWDToMDataset([training_sample_factory()])

    priors = fit_dense_empirical_priors(training)
    report = evaluate_dense_empirical_priors(training, priors)

    assert priors["version"] == EMPIRICAL_PRIOR_VERSION
    assert priors["smoothing"] == EMPIRICAL_PRIOR_SMOOTHING
    assert priors["training_game_count"] == 1
    assert priors["training_boundary_count"] == 1
    assert priors["global"].shape == (7, 7)
    assert torch.allclose(
        priors["global"].sum(dim=-1),
        torch.ones(7, dtype=priors["global"].dtype),
    )
    assert report["train_global_prior"]["aggregate"][
        "mean_belief_cross_entropy"
    ] < math.log(6)
    assert set(report["train_global_prior"]["by_game"]) == {
        "synthetic_game_001"
    }


def test_dense_empirical_priors_reject_augmented_fit(training_sample_factory):
    augmented = DenseTWDToMDataset(
        [training_sample_factory()],
        enable_cyclic_rotation=True,
    )

    with pytest.raises(ValueError, match="unaugmented"):
        fit_dense_empirical_priors(augmented)


def test_dense_empirical_prior_contract_rejects_unknown_version(
    training_sample_factory,
):
    dataset = DenseTWDToMDataset([training_sample_factory()])
    priors = fit_dense_empirical_priors(dataset)
    priors["version"] = "changed"

    with pytest.raises(ValueError, match="version"):
        evaluate_dense_empirical_priors(dataset, priors)
