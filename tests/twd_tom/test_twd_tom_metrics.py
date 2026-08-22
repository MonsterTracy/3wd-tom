import math

import pytest
import torch

from werewolf.models.twd_tom.metrics import compute_belief_metrics


def make_contract():
    logits = torch.zeros((1, 7, 7))
    targets = torch.zeros((1, 7, 7))
    alive = torch.tensor([[True, False, False, False, False, False, False]])
    diagonal = (~torch.eye(7, dtype=torch.bool)).unsqueeze(0)
    targets[0, 0, 1:] = 1 / 6
    return logits, targets, alive, diagonal


def test_uniform_prediction_matches_uniform_target():
    logits, targets, alive, diagonal = make_contract()
    metrics = compute_belief_metrics(logits, targets, alive, diagonal)
    assert metrics == {
        "valid_observer_count": 1,
        "mean_belief_cross_entropy": pytest.approx(math.log(6)),
        "mean_belief_total_variation": pytest.approx(0.0, abs=1e-7),
        "mean_belief_absolute_error": pytest.approx(0.0, abs=1e-7),
        "mean_belief_top1_agreement": pytest.approx(1.0),
    }


def test_dead_observer_rows_do_not_affect_metrics():
    logits, targets, alive, diagonal = make_contract()
    baseline = compute_belief_metrics(logits, targets, alive, diagonal)
    logits[0, 1:] = 1000
    assert compute_belief_metrics(logits, targets, alive, diagonal) == baseline


def test_metrics_expose_no_pair_or_order_specific_names():
    metrics = compute_belief_metrics(*make_contract())
    assert all("pair" not in name for name in metrics)
    assert all("order" not in name for name in metrics)
