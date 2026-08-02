"""Tests for masked soft-target pair cross entropy."""

import math

import pytest
import torch

from werewolf.models.twd_tom.losses import masked_pair_cross_entropy


def tensors():
    logits = torch.zeros((1, 7, 21), requires_grad=True)
    targets = torch.zeros_like(logits)
    targets[0, 2, 4] = 1
    mask = torch.zeros((1, 7), dtype=torch.bool)
    mask[0, 2] = True
    return logits, targets, mask


def test_soft_target_cross_entropy_matches_manual_formula():
    logits, targets, mask = tensors()
    targets[0, 2].zero_()
    targets[0, 2, 4] = 0.25
    targets[0, 2, 8] = 0.75
    expected = -(targets[0, 2] * logits[0, 2].log_softmax(-1)).sum()
    torch.testing.assert_close(
        masked_pair_cross_entropy(logits, targets, mask), expected
    )


def test_uniform_logits_one_hot_target_equals_log_twenty_one():
    logits, targets, mask = tensors()
    assert masked_pair_cross_entropy(logits, targets, mask).item() == pytest.approx(
        math.log(21)
    )


def test_masked_rows_receive_no_gradient():
    logits, targets, mask = tensors()
    masked_pair_cross_entropy(logits, targets, mask).backward()
    assert logits.grad[0, 2].count_nonzero().item() > 0
    assert logits.grad[0, ~mask[0]].count_nonzero().item() == 0


def test_no_valid_observer_fails_instead_of_returning_zero():
    logits = torch.zeros((1, 7, 21))
    targets = torch.zeros_like(logits)
    mask = torch.zeros((1, 7), dtype=torch.bool)
    with pytest.raises(ValueError, match="at least one valid observer"):
        masked_pair_cross_entropy(logits, targets, mask)


def test_shapes_targets_and_mask_are_strict():
    logits, targets, mask = tensors()
    targets[0, 0, 0] = 1
    with pytest.raises(ValueError, match="unsupervised"):
        masked_pair_cross_entropy(logits, targets, mask)
    with pytest.raises(ValueError, match="shape"):
        masked_pair_cross_entropy(logits[:, 0], targets, mask)
    with pytest.raises(TypeError, match="torch.bool"):
        masked_pair_cross_entropy(logits, targets.zero_(), mask.long())
