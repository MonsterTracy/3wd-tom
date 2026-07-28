"""Tests for masked subjective-pair KL divergence."""

import inspect
import math

import pytest
import torch
import torch.nn.functional as F

from werewolf.models.twd_tom.losses import (
    masked_pair_kl_divergence,
)
from werewolf.models.twd_tom.schema import (
    NUM_PLAYERS,
    NUM_WOLF_PAIR_CLASSES,
)


def make_targets():
    targets = torch.zeros((2, NUM_PLAYERS, NUM_WOLF_PAIR_CLASSES))
    mask = torch.zeros((2, NUM_PLAYERS), dtype=torch.bool)
    targets[0, 0, 1] = 1.0
    targets[0, 2, 3:8] = 1.0 / 5.0
    targets[1, 5].fill_(1.0 / NUM_WOLF_PAIR_CLASSES)
    mask[0, 0] = True
    mask[0, 2] = True
    mask[1, 5] = True
    return targets, mask


def test_matches_manual_masked_pair_kl_mean():
    logits = torch.randn(
        (2, NUM_PLAYERS, NUM_WOLF_PAIR_CLASSES),
        generator=torch.Generator().manual_seed(7),
    )
    targets, mask = make_targets()
    loss = masked_pair_kl_divergence(logits, targets, mask)
    manual = F.kl_div(
        F.log_softmax(logits, dim=-1),
        targets,
        reduction="none",
    ).sum(dim=-1)[mask].mean()
    torch.testing.assert_close(loss, manual)


def test_none_and_sum_reductions_use_only_valid_subjects():
    logits = torch.zeros((2, NUM_PLAYERS, NUM_WOLF_PAIR_CLASSES))
    targets, mask = make_targets()
    losses = masked_pair_kl_divergence(
        logits, targets, mask, reduction="none"
    )
    summed = masked_pair_kl_divergence(
        logits, targets, mask, reduction="sum"
    )
    assert losses.shape == (2, NUM_PLAYERS)
    assert torch.all(losses[~mask] == 0.0)
    assert torch.all(losses[mask] >= 0.0)
    torch.testing.assert_close(summed, losses.sum())


def test_perfect_pair_prediction_has_near_zero_loss():
    targets = torch.zeros((1, NUM_PLAYERS, NUM_WOLF_PAIR_CLASSES))
    mask = torch.zeros((1, NUM_PLAYERS), dtype=torch.bool)
    targets[0, 3, 5] = 1.0
    mask[0, 3] = True
    good = torch.full_like(targets, -20.0)
    wrong = good.clone()
    good[0, 3, 5] = 20.0
    wrong[0, 3, 1] = 20.0
    good_loss = masked_pair_kl_divergence(good, targets, mask)
    wrong_loss = masked_pair_kl_divergence(wrong, targets, mask)
    assert good_loss.item() < 1e-6
    assert good_loss < wrong_loss


def test_uniform_prediction_one_hot_target_equals_log_twenty_one():
    logits = torch.zeros((1, NUM_PLAYERS, NUM_WOLF_PAIR_CLASSES))
    targets = torch.zeros_like(logits)
    mask = torch.zeros((1, NUM_PLAYERS), dtype=torch.bool)
    targets[0, 1, 4] = 1.0
    mask[0, 1] = True
    loss = masked_pair_kl_divergence(logits, targets, mask)
    torch.testing.assert_close(
        loss,
        torch.tensor(math.log(NUM_WOLF_PAIR_CLASSES)),
    )


def test_masked_subject_rows_receive_no_gradient():
    logits = torch.zeros(
        (1, NUM_PLAYERS, NUM_WOLF_PAIR_CLASSES),
        requires_grad=True,
    )
    targets = torch.zeros_like(logits)
    mask = torch.zeros((1, NUM_PLAYERS), dtype=torch.bool)
    targets[0, 2, 6] = 1.0
    mask[0, 2] = True
    masked_pair_kl_divergence(logits, targets, mask).backward()
    assert logits.grad is not None
    assert logits.grad[0, 2].abs().sum().item() > 0.0
    other_gradients = logits.grad.clone()
    other_gradients[0, 2] = 0.0
    assert other_gradients.count_nonzero().item() == 0


def test_all_masked_batch_returns_differentiable_zero():
    logits = torch.randn(
        (2, NUM_PLAYERS, NUM_WOLF_PAIR_CLASSES),
        requires_grad=True,
    )
    targets = torch.zeros_like(logits)
    mask = torch.zeros((2, NUM_PLAYERS), dtype=torch.bool)
    loss = masked_pair_kl_divergence(logits, targets, mask)
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.count_nonzero().item() == 0


def test_supervised_rows_must_sum_to_one():
    logits = torch.zeros((1, NUM_PLAYERS, NUM_WOLF_PAIR_CLASSES))
    targets = torch.zeros_like(logits)
    mask = torch.zeros((1, NUM_PLAYERS), dtype=torch.bool)
    targets[0, 0, 1] = 0.7
    mask[0, 0] = True
    with pytest.raises(ValueError, match="sum to one"):
        masked_pair_kl_divergence(logits, targets, mask)


def test_unsupervised_rows_must_remain_zero():
    logits = torch.zeros((1, NUM_PLAYERS, NUM_WOLF_PAIR_CLASSES))
    targets = torch.zeros_like(logits)
    mask = torch.zeros((1, NUM_PLAYERS), dtype=torch.bool)
    targets[0, 4, 2] = 1.0
    with pytest.raises(ValueError, match="must remain all zero"):
        masked_pair_kl_divergence(logits, targets, mask)


def test_negative_and_non_finite_values_are_rejected():
    logits = torch.zeros((1, NUM_PLAYERS, NUM_WOLF_PAIR_CLASSES))
    targets = torch.zeros_like(logits)
    mask = torch.zeros((1, NUM_PLAYERS), dtype=torch.bool)
    targets[0, 0, 0] = 1.1
    targets[0, 0, 1] = -0.1
    mask[0, 0] = True
    with pytest.raises(ValueError, match="negative"):
        masked_pair_kl_divergence(logits, targets, mask)
    targets[0, 0] = 0.0
    targets[0, 0, 0] = 1.0
    logits[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        masked_pair_kl_divergence(logits, targets, mask)


def test_invalid_shapes_and_mask_dtype_are_rejected():
    logits = torch.zeros((2, NUM_PLAYERS, NUM_WOLF_PAIR_CLASSES))
    targets = torch.zeros_like(logits)
    mask = torch.zeros((2, NUM_PLAYERS), dtype=torch.bool)
    with pytest.raises(ValueError, match=r"shape \[B, 7, 21\]"):
        masked_pair_kl_divergence(logits[:, 0], targets, mask)
    with pytest.raises(ValueError, match="same shape"):
        masked_pair_kl_divergence(logits, targets[..., :14], mask)
    with pytest.raises(ValueError, match="subject_mask"):
        masked_pair_kl_divergence(logits, targets, mask[:, :6])
    with pytest.raises(TypeError, match="torch.bool"):
        masked_pair_kl_divergence(logits, targets, mask.long())


def test_invalid_reduction_is_rejected():
    logits = torch.zeros((1, NUM_PLAYERS, NUM_WOLF_PAIR_CLASSES))
    targets = torch.zeros_like(logits)
    mask = torch.zeros((1, NUM_PLAYERS), dtype=torch.bool)
    with pytest.raises(ValueError, match="reduction"):
        masked_pair_kl_divergence(
            logits, targets, mask, reduction="batchmean"
        )


def test_pair_loss_api_has_no_truth_inputs_or_old_names():
    parameters = inspect.signature(masked_pair_kl_divergence).parameters
    assert tuple(parameters) == (
        "pair_logits",
        "pair_targets",
        "subject_mask",
        "reduction",
    )
    forbidden = {
        "roles",
        "true_roles",
        "wolf_labels",
        "truth",
        "actual_wolves",
        "alive_mask",
        "observer_id",
        "belief_logits",
        "belief_targets",
    }
    assert forbidden.isdisjoint(parameters)
