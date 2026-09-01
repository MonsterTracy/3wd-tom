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


def test_private_dataset_reports_private_admissible_uniform_baseline(
    training_sample_factory,
):
    dataset = DenseTWDToMDataset(
        [training_sample_factory()],
        include_private_features=True,
    )
    report = evaluate_dense_empirical_priors(
        dataset,
        fit_dense_empirical_priors(dataset),
    )

    assert "private_admissible_uniform" in report
    aggregate = report["private_admissible_uniform"]["aggregate"]
    assert "private_admissible_normalized_reducible_gap_improvement" in aggregate


def test_public_empirical_priors_accept_empty_target_with_private_knowledge(
    training_sample_factory,
):
    sample = training_sample_factory()
    sample["suspected_werewolves"] = {
        observer: [] for observer in sample["suspected_werewolves"]
    }
    sample["known_non_werewolves"]["player1"] = ["player1", "player4"]
    dataset = DenseTWDToMDataset([sample])

    report = evaluate_dense_empirical_priors(
        dataset,
        fit_dense_empirical_priors(dataset),
    )

    for name in ("train_global_prior", "train_phase_prior"):
        aggregate = report[name]["aggregate"]
        assert aggregate["valid_observer_count"] == 4
        assert aggregate[
            "uniform_non_self_baseline_mean_kl_divergence"
        ] == pytest.approx(0.0)
        assert aggregate["mean_belief_cross_entropy"] == pytest.approx(
            aggregate["mean_belief_target_entropy"]
        )
        assert "private_admissible_uniform_baseline_kl_sum" not in aggregate


def test_dense_baselines_keep_zero_supervision_game_as_explicit_unscored(
    training_sample_factory,
):
    scored = training_sample_factory(game_id="scored_game")
    unscored = training_sample_factory(game_id="unscored_game")
    unscored["suspected_werewolves"] = {
        observer: None for observer in unscored["suspected_werewolves"]
    }
    unscored["belief_status"] = {
        observer: "parse_error" for observer in unscored["belief_status"]
    }
    unscored["belief_errors"] = {
        observer: "synthetic failed report"
        for observer in unscored["belief_errors"]
    }
    dataset = DenseTWDToMDataset([scored, unscored])

    report = evaluate_dense_empirical_priors(
        dataset,
        fit_dense_empirical_priors(dataset),
    )["train_global_prior"]

    assert report["game_count"] == 2
    assert report["scored_game_count"] == 1
    assert report["unscored_game_count"] == 1
    assert report["unscored_game_ids"] == ["unscored_game"]
    assert report["by_game"]["unscored_game"]["status"] == (
        "unscored_no_supervised_observers"
    )
    assert report["by_game"]["unscored_game"]["valid_observer_count"] == 0
    assert report["aggregate"]["valid_observer_count"] == 4
