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
        "mean_belief_top1_support_hit": pytest.approx(1.0),
        "uniform_non_self_baseline_mean_cross_entropy": pytest.approx(math.log(6)),
        "uniform_non_self_baseline_mean_total_variation": pytest.approx(
            0.0,
            abs=1e-7,
        ),
        "uniform_non_self_baseline_mean_absolute_error": pytest.approx(
            0.0,
            abs=1e-7,
        ),
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


def test_top1_is_a_support_hit_for_soft_targets():
    logits, targets, alive, diagonal = make_contract()
    targets.zero_()
    targets[0, 0, 1] = 0.5
    targets[0, 0, 2] = 0.5
    logits[0, 0, 2] = 10.0

    metrics = compute_belief_metrics(logits, targets, alive, diagonal)
    assert metrics["mean_belief_top1_support_hit"] == pytest.approx(1.0)

    logits[0, 0, 2] = 0.0
    logits[0, 0, 3] = 10.0
    metrics = compute_belief_metrics(logits, targets, alive, diagonal)
    assert metrics["mean_belief_top1_support_hit"] == pytest.approx(0.0)


def test_uniform_non_self_baseline_is_reported_for_sparse_target():
    logits, targets, alive, diagonal = make_contract()
    targets.zero_()
    targets[0, 0, 1] = 1.0

    metrics = compute_belief_metrics(logits, targets, alive, diagonal)
    assert metrics["uniform_non_self_baseline_mean_cross_entropy"] == pytest.approx(
        math.log(6)
    )
    assert metrics[
        "uniform_non_self_baseline_mean_total_variation"
    ] == pytest.approx(5 / 6)
    assert metrics[
        "uniform_non_self_baseline_mean_absolute_error"
    ] == pytest.approx(5 / 18)
