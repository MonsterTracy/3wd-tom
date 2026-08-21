import pytest
import torch
import torch.nn.functional as F

from werewolf.models.public_belief_matrix.losses import (
    masked_row_soft_target_cross_entropy,
)


def _inputs():
    return (
        torch.zeros((1, 7, 7)),
        torch.zeros((1, 7, 7)),
        torch.zeros((1, 7), dtype=torch.bool),
    )


def test_one_hot_target_matches_manual_cross_entropy():
    logits, targets, mask = _inputs()
    logits[0, 2] = torch.tensor([2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
    targets[0, 2, 0] = 1.0
    mask[0, 2] = True

    loss = masked_row_soft_target_cross_entropy(logits, targets, mask)
    expected = -F.log_softmax(logits[0, 2], dim=-1)[0]

    torch.testing.assert_close(loss, expected)


def test_two_suspect_soft_target_matches_manual_cross_entropy():
    logits, targets, mask = _inputs()
    logits[0, 4] = torch.arange(7, dtype=torch.float32)
    targets[0, 4, [1, 5]] = 0.5
    mask[0, 4] = True

    loss = masked_row_soft_target_cross_entropy(logits, targets, mask)
    log_probabilities = F.log_softmax(logits[0, 4], dim=-1)
    expected = -0.5 * (log_probabilities[1] + log_probabilities[5])

    torch.testing.assert_close(loss, expected)


def test_masked_invalid_row_does_not_affect_loss():
    logits, targets, mask = _inputs()
    targets[0, 0, 0] = 1.0
    mask[0, 0] = True
    baseline = masked_row_soft_target_cross_entropy(logits, targets, mask)
    targets[0, 1] = torch.nan

    torch.testing.assert_close(
        masked_row_soft_target_cross_entropy(logits, targets, mask),
        baseline,
    )


def test_all_rows_masked_fails_closed():
    logits, targets, mask = _inputs()
    with pytest.raises(ValueError, match="at least one"):
        masked_row_soft_target_cross_entropy(logits, targets, mask)


def test_invalid_valid_target_row_sum_fails_closed():
    logits, targets, mask = _inputs()
    targets[0, 0, 0] = 0.5
    mask[0, 0] = True
    with pytest.raises(ValueError, match="sum to one"):
        masked_row_soft_target_cross_entropy(logits, targets, mask)


@pytest.mark.parametrize("invalid_value", [-0.1, float("nan")])
def test_negative_or_nan_valid_target_fails_closed(invalid_value):
    logits, targets, mask = _inputs()
    targets[0, 0, 0] = invalid_value
    targets[0, 0, 1] = 1.0
    mask[0, 0] = True
    with pytest.raises(ValueError):
        masked_row_soft_target_cross_entropy(logits, targets, mask)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("targets", torch.zeros((1, 7, 6))),
        ("observer_mask", torch.zeros((1, 6), dtype=torch.bool)),
        ("logits", torch.zeros((1, 7, 6))),
    ],
)
def test_shape_mismatch_fails_closed(field, replacement):
    logits, targets, mask = _inputs()
    values = {"logits": logits, "targets": targets, "observer_mask": mask}
    values[field] = replacement
    with pytest.raises(ValueError):
        masked_row_soft_target_cross_entropy(**values)
