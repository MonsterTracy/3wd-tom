import math

import pytest
import torch

from werewolf.models.twd_tom.losses import (
    masked_belief_distribution_loss,
    masked_belief_probabilities,
)


def make_contract():
    logits = torch.zeros((1, 7, 7), requires_grad=True)
    targets = torch.zeros((1, 7, 7))
    alive = torch.tensor([[True, True, False, False, False, False, False]])
    diagonal = (~torch.eye(7, dtype=torch.bool)).unsqueeze(0)
    targets[0, 0, 1:] = 1 / 6
    targets[0, 1, [0, 2, 3, 4, 5, 6]] = 1 / 6
    return logits, targets, alive, diagonal


def test_masked_belief_loss_is_cross_entropy_over_alive_rows():
    logits, targets, alive, diagonal = make_contract()
    loss = masked_belief_distribution_loss(logits, targets, alive, diagonal)
    assert loss.item() == pytest.approx(math.log(6))
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[0, 2:]) == 0


def test_masked_probabilities_have_zero_diagonal_and_normalized_rows():
    logits, _, _, diagonal = make_contract()
    probabilities = masked_belief_probabilities(logits, diagonal)
    assert torch.equal(probabilities.diagonal(dim1=-2, dim2=-1), torch.zeros((1, 7)))
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones((1, 7)))


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_supported_reductions(reduction):
    logits, targets, alive, diagonal = make_contract()
    result = masked_belief_distribution_loss(
        logits, targets, alive, diagonal, reduction=reduction
    )
    assert result.shape == ((1, 7) if reduction == "none" else ())


def test_invalid_shape_mask_and_dead_targets_fail_closed():
    logits, targets, alive, diagonal = make_contract()
    with pytest.raises(ValueError, match=r"\[B, 7, 7\]"):
        masked_belief_distribution_loss(logits[..., :6], targets, alive, diagonal)
    broken_diagonal = diagonal.clone()
    broken_diagonal[0, 0, 0] = True
    with pytest.raises(ValueError, match="exclude exactly the diagonal"):
        masked_belief_distribution_loss(logits, targets, alive, broken_diagonal)
    targets[0, 2, 0] = 1.0
    with pytest.raises(ValueError, match="dead observer"):
        masked_belief_distribution_loss(logits, targets, alive, diagonal)


def test_no_alive_observer_is_rejected():
    logits, targets, alive, diagonal = make_contract()
    targets.zero_()
    alive.zero_()
    with pytest.raises(ValueError, match="at least one observer"):
        masked_belief_distribution_loss(logits, targets, alive, diagonal)


def test_supervision_mask_changes_only_selected_loss_rows():
    logits, targets, alive, diagonal = make_contract()
    original_targets = targets.clone()
    supervision = torch.tensor(
        [[False, True, False, False, False, False, False]]
    )
    with torch.no_grad():
        logits[0, 1, 2] = 1.0

    loss = masked_belief_distribution_loss(
        logits,
        targets,
        alive,
        diagonal,
        observer_supervision_mask=supervision,
    )
    loss.backward()

    assert loss.item() > math.log(6)
    assert torch.equal(targets, original_targets)
    assert torch.count_nonzero(logits.grad[0, 0]) == 0
    assert torch.count_nonzero(logits.grad[0, 1]) > 0


def test_unobserved_alive_row_may_have_zero_target_when_not_supervised():
    logits, targets, alive, diagonal = make_contract()
    targets[0, 0].zero_()
    supervision = torch.tensor(
        [[False, True, False, False, False, False, False]]
    )

    loss = masked_belief_distribution_loss(
        logits,
        targets,
        alive,
        diagonal,
        observer_supervision_mask=supervision,
    )

    assert loss.item() == pytest.approx(math.log(6))
