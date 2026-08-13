import pytest
import torch
import torch.nn.functional as F

from werewolf.models.tom.losses import masked_soft_target_cross_entropy


def _empty_inputs(batch_size=1):
    return (
        torch.zeros((batch_size, 7, 7)),
        torch.zeros((batch_size, 7, 7)),
        torch.zeros((batch_size, 7), dtype=torch.bool),
    )


def test_one_valid_one_hot_row_matches_manual_cross_entropy_and_is_finite():
    logits, targets, mask = _empty_inputs()
    logits[0, 3] = torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0])
    targets[0, 3, 5] = 1.0
    mask[0, 3] = True

    loss = masked_soft_target_cross_entropy(logits, targets, mask)
    expected = -F.log_softmax(logits[0, 3], dim=-1)[5]

    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, expected)


def test_multiple_valid_rows_use_one_global_row_mean():
    logits, targets, mask = _empty_inputs(batch_size=2)
    logits[0, 0] = torch.arange(7, dtype=torch.float32)
    logits[1, 1] = torch.arange(7, dtype=torch.float32).flip(0)
    logits[1, 2] = torch.tensor([2.0, 0.0, -1.0, 3.0, 1.0, -2.0, 4.0])
    targets[0, 0] = 1.0 / 7.0
    targets[1, 1, 0] = 1.0
    targets[1, 2, [2, 4]] = 0.5
    mask[0, 0] = True
    mask[1, 1:3] = True

    loss = masked_soft_target_cross_entropy(logits, targets, mask)
    expected = -(
        targets[mask] * F.log_softmax(logits[mask], dim=-1)
    ).sum(dim=-1).mean()

    torch.testing.assert_close(loss, expected)


def test_masked_rows_and_zero_masked_targets_do_not_affect_loss():
    logits, targets, mask = _empty_inputs()
    targets[0, 0] = 1.0 / 7.0
    mask[0, 0] = True
    baseline = masked_soft_target_cross_entropy(logits, targets, mask)

    logits[0, 1:] = torch.nan
    targets[0, 1] = 0.0
    targets[0, 2:] = torch.nan
    actual = masked_soft_target_cross_entropy(logits, targets, mask)

    torch.testing.assert_close(actual, baseline)


def test_valid_uniform_and_one_hot_rows_are_accepted():
    logits, targets, mask = _empty_inputs()
    targets[0, 0] = 1.0 / 7.0
    targets[0, 4, 2] = 1.0
    mask[0, [0, 4]] = True

    loss = masked_soft_target_cross_entropy(logits, targets, mask)

    assert torch.isfinite(loss)


def test_all_rows_masked_raises_explicit_error():
    logits, targets, mask = _empty_inputs()
    with pytest.raises(ValueError, match="at least one valid row"):
        masked_soft_target_cross_entropy(logits, targets, mask)


@pytest.mark.parametrize(
    "row",
    [
        torch.tensor([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        torch.tensor([1.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
        torch.tensor([float("nan"), 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ],
)
def test_malformed_valid_target_row_raises_explicit_error(row):
    logits, targets, mask = _empty_inputs()
    targets[0, 0] = row
    mask[0, 0] = True
    with pytest.raises(ValueError):
        masked_soft_target_cross_entropy(logits, targets, mask)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("logits", torch.zeros((1, 7, 6))),
        ("targets", torch.zeros((1, 6, 7))),
        ("observer_mask", torch.zeros((1, 6), dtype=torch.bool)),
    ],
)
def test_exact_shapes_are_required(field, replacement):
    logits, targets, observer_mask = _empty_inputs()
    inputs = {
        "logits": logits,
        "targets": targets,
        "observer_mask": observer_mask,
    }
    inputs[field] = replacement
    with pytest.raises(ValueError):
        masked_soft_target_cross_entropy(**inputs)


def test_loss_backpropagates_to_logits():
    logits = torch.randn((1, 7, 7), requires_grad=True)
    targets = torch.zeros((1, 7, 7))
    mask = torch.zeros((1, 7), dtype=torch.bool)
    targets[0, 2, [1, 6]] = 0.5
    mask[0, 2] = True

    loss = masked_soft_target_cross_entropy(logits, targets, mask)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 2].abs().sum() > 0
