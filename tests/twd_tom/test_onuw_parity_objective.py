import math

import pytest
import torch

from werewolf.models.twd_tom.onuw_parity_objective import (
    onuw_parity_belief_objective,
)


def test_training_loss_is_observer_row_micro_and_reports_all_aggregations():
    logits = torch.zeros((2, 2, 7, 7))
    targets = torch.zeros_like(logits)
    query_mask = torch.tensor([[True, True], [True, False]])
    alive = torch.zeros((2, 2, 7), dtype=torch.bool)
    alive[0, 0, :2] = True
    alive[0, 1, :1] = True
    alive[1, 0, :3] = True
    valid = query_mask[:, :, None] & alive
    targets[..., 0] = valid.to(targets.dtype)
    values = onuw_parity_belief_objective(logits, targets, query_mask, alive)
    assert values["loss"].item() == pytest.approx(math.log(7))
    assert values["row_micro_ce"].item() == pytest.approx(math.log(7))
    assert values["row_micro_kl"].item() == pytest.approx(math.log(7))
    assert values["query_macro_ce"].item() == pytest.approx(math.log(7))
    assert values["game_macro_ce"].item() == pytest.approx(math.log(7))
    assert values["valid_row_count"].item() == 6


def test_kl_is_ce_minus_target_entropy():
    logits = torch.zeros((1, 1, 7, 7))
    targets = torch.zeros_like(logits)
    targets[0, 0, 0] = 1 / 7
    alive = torch.zeros((1, 1, 7), dtype=torch.bool)
    alive[0, 0, 0] = True
    query_mask = torch.ones((1, 1), dtype=torch.bool)
    values = onuw_parity_belief_objective(logits, targets, query_mask, alive)
    assert values["row_micro_ce"].item() == pytest.approx(math.log(7))
    assert values["row_micro_kl"].item() == pytest.approx(0.0, abs=1e-6)
