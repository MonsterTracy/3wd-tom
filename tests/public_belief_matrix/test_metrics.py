import math

import pytest
import torch

from werewolf.models.public_belief_matrix.losses import (
    masked_row_soft_target_cross_entropy,
)
from werewolf.models.public_belief_matrix.metrics import (
    masked_mean_row_cross_entropy,
    masked_mean_row_entropy,
    mean_observer_pairwise_tv,
    mean_prediction_diagonal_mass,
)


def _probabilities():
    return torch.full((1, 7, 7), 1.0 / 7.0)


def test_metric_cross_entropy_reuses_loss_contract():
    logits = torch.zeros((1, 7, 7))
    targets = _probabilities()
    mask = torch.ones((1, 7), dtype=torch.bool)

    assert masked_mean_row_cross_entropy(logits, targets, mask) == pytest.approx(
        masked_row_soft_target_cross_entropy(logits, targets, mask).item()
    )


def test_identical_observer_rows_have_zero_pairwise_tv():
    probabilities = _probabilities()
    mask = torch.tensor([[True, True, False, False, False, False, False]])
    assert mean_observer_pairwise_tv(probabilities, mask) == pytest.approx(0.0)


def test_disjoint_one_hot_observer_rows_have_unit_pairwise_tv():
    probabilities = torch.zeros((1, 7, 7))
    probabilities[0, 0, 0] = 1.0
    probabilities[0, 1, 1] = 1.0
    mask = torch.tensor([[True, True, False, False, False, False, False]])
    assert mean_observer_pairwise_tv(probabilities, mask) == pytest.approx(1.0)


def test_invalid_observer_rows_are_excluded_from_pairwise_tv():
    probabilities = _probabilities()
    probabilities[0, 2] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mask = torch.tensor([[True, True, False, False, False, False, False]])
    assert mean_observer_pairwise_tv(probabilities, mask) == pytest.approx(0.0)


def test_pairwise_tv_is_none_without_a_valid_pair():
    probabilities = _probabilities()
    mask = torch.tensor([[True, False, False, False, False, False, False]])
    assert mean_observer_pairwise_tv(probabilities, mask) is None


def test_prediction_diagonal_mass_matches_manual_mean():
    probabilities = _probabilities()
    probabilities[0, 0] = torch.tensor([0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    probabilities[0, 2] = torch.tensor([0.0, 0.0, 0.7, 0.1, 0.1, 0.1, 0.0])
    mask = torch.tensor([[True, False, True, False, False, False, False]])
    assert mean_prediction_diagonal_mass(probabilities, mask) == pytest.approx(
        0.55
    )


def test_row_entropy_matches_one_hot_and_uniform_manual_mean():
    probabilities = _probabilities()
    probabilities[0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mask = torch.tensor([[True, True, False, False, False, False, False]])
    assert masked_mean_row_entropy(probabilities, mask) == pytest.approx(
        math.log(7.0) / 2.0
    )
