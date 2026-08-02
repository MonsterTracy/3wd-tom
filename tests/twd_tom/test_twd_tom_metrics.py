"""Tests for subjective pair-distribution metrics."""

import inspect
import math

import pytest
import torch

from werewolf.models.twd_tom.metrics import (
    compute_subjective_pair_diagnostics,
    compute_subjective_pair_metrics,
)
from werewolf.models.twd_tom.schema import (
    NUM_PLAYERS,
    NUM_WOLF_PAIR_CLASSES,
)


EXPECTED_KEYS = {
    "valid_subject_count",
    "mean_pair_kl_divergence",
    "mean_pair_cross_entropy",
    "mean_pair_total_variation",
    "mean_marginal_mae",
    "mean_marginal_row_sum_error",
    "mean_predicted_diagonal_marginal",
    "mean_target_diagonal_marginal",
}
DIAGNOSTIC_KEYS = {
    "mean_target_pair_entropy",
    "mean_predicted_pair_entropy",
    "mean_target_pair_top1_top2_margin",
    "mean_predicted_pair_top1_top2_margin",
    "mean_target_marginal_spread",
    "mean_predicted_marginal_spread",
    "mean_target_observer_pairwise_tv",
    "mean_predicted_observer_pairwise_tv",
}


def make_metric_inputs():
    targets = torch.zeros((1, NUM_PLAYERS, NUM_WOLF_PAIR_CLASSES))
    subject_mask = torch.zeros((1, NUM_PLAYERS), dtype=torch.bool)

    targets[0, 0].fill_(1.0 / NUM_WOLF_PAIR_CLASSES)
    subject_mask[0, 0] = True

    targets[0, 1, :5] = 1.0 / 5.0
    subject_mask[0, 1] = True

    targets[0, 2, 7] = 1.0
    subject_mask[0, 2] = True
    return targets, subject_mask


def test_perfect_pair_prediction_has_near_zero_kl():
    targets, subject_mask = make_metric_inputs()
    logits = targets.clamp_min(torch.finfo(torch.float32).tiny).log()
    metrics = compute_subjective_pair_metrics(
        logits,
        targets,
        subject_mask,
    )
    assert set(metrics) == EXPECTED_KEYS
    assert metrics["valid_subject_count"] == 3
    assert metrics["mean_pair_kl_divergence"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["mean_pair_total_variation"] == pytest.approx(0.0, abs=1e-6)


def test_uniform_target_is_a_valid_supervised_row():
    targets = torch.zeros((1, NUM_PLAYERS, NUM_WOLF_PAIR_CLASSES))
    targets[0, 0].fill_(1.0 / NUM_WOLF_PAIR_CLASSES)
    subject_mask = torch.zeros((1, NUM_PLAYERS), dtype=torch.bool)
    subject_mask[0, 0] = True
    metrics = compute_subjective_pair_metrics(
        torch.zeros_like(targets),
        targets,
        subject_mask,
    )
    assert metrics["valid_subject_count"] == 1
    assert metrics["mean_pair_kl_divergence"] == pytest.approx(0.0, abs=1e-6)


def test_uniform_pair_prediction_one_hot_target_metrics():
    logits = torch.zeros((1, NUM_PLAYERS, NUM_WOLF_PAIR_CLASSES))
    targets = torch.zeros_like(logits)
    targets[0, 0, 4] = 1.0
    mask = torch.zeros((1, NUM_PLAYERS), dtype=torch.bool)
    mask[0, 0] = True
    metrics = compute_subjective_pair_metrics(logits, targets, mask)
    assert metrics["mean_pair_kl_divergence"] == pytest.approx(
        math.log(NUM_WOLF_PAIR_CLASSES)
    )
    assert metrics["mean_pair_cross_entropy"] == pytest.approx(
        math.log(NUM_WOLF_PAIR_CLASSES)
    )
    assert metrics["mean_pair_total_variation"] == pytest.approx(
        1.0 - 1.0 / NUM_WOLF_PAIR_CLASSES
    )


def test_masked_rows_do_not_affect_metrics():
    targets, subject_mask = make_metric_inputs()
    logits = targets.clamp_min(torch.finfo(torch.float32).tiny).log()
    logits[0, 6] = 1000.0
    metrics = compute_subjective_pair_metrics(
        logits,
        targets,
        subject_mask,
    )
    assert metrics["mean_pair_kl_divergence"] == pytest.approx(0.0, abs=1e-6)


def test_all_masked_metrics_are_safe_zeroes():
    logits = torch.zeros((1, NUM_PLAYERS, NUM_WOLF_PAIR_CLASSES))
    targets = torch.zeros_like(logits)
    mask = torch.zeros((1, NUM_PLAYERS), dtype=torch.bool)
    metrics = compute_subjective_pair_metrics(logits, targets, mask)
    assert metrics["valid_subject_count"] == 0
    assert all(value == 0.0 for key, value in metrics.items() if key.startswith("mean_"))


def test_pair_metrics_api_has_three_explicit_parameters():
    parameters = inspect.signature(compute_subjective_pair_metrics).parameters
    assert tuple(parameters) == (
        "pair_logits",
        "pair_targets",
        "subject_mask",
    )


@pytest.mark.parametrize("tom_order", [1, 2])
def test_both_orders_use_only_pair_and_marginal_metrics(tom_order):
    logits = torch.zeros((1, NUM_PLAYERS, NUM_WOLF_PAIR_CLASSES))
    targets = torch.zeros_like(logits)
    targets[0, 2, 5] = 1.0
    mask = torch.zeros((1, NUM_PLAYERS), dtype=torch.bool)
    mask[0, 2] = True
    metrics = compute_subjective_pair_metrics(logits, targets, mask)
    assert set(metrics) == EXPECTED_KEYS
    assert not any("suspicion" in key for key in metrics)


def test_collapse_diagnostics_are_finite_and_match_known_pair_rows():
    targets = torch.zeros((1, NUM_PLAYERS, NUM_WOLF_PAIR_CLASSES))
    targets[0, 0, 0] = 1
    targets[0, 1, 1] = 1
    mask = torch.zeros((1, NUM_PLAYERS), dtype=torch.bool)
    mask[0, :2] = True
    logits = targets.clamp_min(torch.finfo(torch.float32).tiny).log()
    diagnostics = compute_subjective_pair_diagnostics(logits, targets, mask)
    assert set(diagnostics) == DIAGNOSTIC_KEYS
    assert all(math.isfinite(value) for value in diagnostics.values())
    assert diagnostics["mean_target_pair_entropy"] == pytest.approx(0.0)
    assert diagnostics["mean_target_pair_top1_top2_margin"] == pytest.approx(1.0)
    assert diagnostics["mean_target_marginal_spread"] == pytest.approx(1.0)
    assert diagnostics["mean_target_observer_pairwise_tv"] == pytest.approx(1.0)
    assert diagnostics["mean_predicted_observer_pairwise_tv"] == pytest.approx(
        1.0
    )


def test_observer_pairwise_tv_skips_snapshots_with_fewer_than_two_rows():
    targets = torch.zeros((2, NUM_PLAYERS, NUM_WOLF_PAIR_CLASSES))
    targets[0, 0, 0] = 1
    targets[1, 0, 0] = 1
    targets[1, 1, 1] = 1
    mask = torch.zeros((2, NUM_PLAYERS), dtype=torch.bool)
    mask[0, 0] = True
    mask[1, :2] = True
    logits = targets.clamp_min(torch.finfo(torch.float32).tiny).log()
    diagnostics = compute_subjective_pair_diagnostics(logits, targets, mask)
    assert diagnostics["mean_target_observer_pairwise_tv"] == pytest.approx(1.0)
    only_one = compute_subjective_pair_diagnostics(
        logits[:1], targets[:1], mask[:1]
    )
    assert only_one["mean_target_observer_pairwise_tv"] == 0.0
    assert only_one["mean_predicted_observer_pairwise_tv"] == 0.0
