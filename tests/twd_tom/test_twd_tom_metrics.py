import math

import pytest
import torch

from werewolf.models.twd_tom.metrics import (
    UNSCORED_NO_SUPERVISION_STATUS,
    compute_belief_metrics,
)


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
    assert metrics["total_row_count"] == 1
    assert metrics["valid_observer_count"] == 1
    assert metrics["zero_uniform_baseline_gap_row_count"] == 1
    assert metrics["positive_uniform_baseline_gap_row_count"] == 0
    assert metrics["model_kl_sum"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["uniform_non_self_baseline_kl_sum"] == pytest.approx(0.0)
    assert metrics["mean_belief_cross_entropy"] == pytest.approx(math.log(6))
    assert metrics["mean_belief_target_entropy"] == pytest.approx(math.log(6))
    assert metrics["mean_belief_kl_divergence"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["mean_belief_total_variation"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["mean_belief_absolute_error"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["mean_belief_max_probability_error"] == pytest.approx(
        0.0, abs=1e-7
    )
    assert metrics["max_belief_probability_error"] == pytest.approx(
        0.0, abs=1e-7
    )
    assert metrics["mean_belief_max_set_support_hit"] == pytest.approx(1.0)
    assert metrics[
        "mean_belief_deterministic_top1_support_hit"
    ] == pytest.approx(1.0)


def test_dead_observer_rows_do_not_affect_metrics():
    logits, targets, alive, diagonal = make_contract()
    baseline = compute_belief_metrics(logits, targets, alive, diagonal)
    logits[0, 1:] = 1000
    assert compute_belief_metrics(logits, targets, alive, diagonal) == baseline


def test_metrics_report_unobserved_rows_inside_scope_without_scoring_them():
    logits = torch.zeros((1, 7, 7))
    targets = torch.zeros((1, 7, 7))
    targets[0, 0, 1] = 1.0
    alive = torch.tensor([[True, True, False, False, False, False, False]])
    diagonal = (~torch.eye(7, dtype=torch.bool)).unsqueeze(0)
    scope = alive.clone()
    observed = torch.tensor(
        [[True, False, False, False, False, False, False]]
    )
    supervision = scope & observed

    metrics = compute_belief_metrics(
        logits,
        targets,
        alive,
        diagonal,
        observer_supervision_mask=supervision,
        observer_scope_mask=scope,
        label_observed_mask=observed,
    )

    assert metrics["scope_observer_count"] == 2
    assert metrics["observed_label_row_count_in_scope"] == 1
    assert metrics["unobserved_label_row_count_in_scope"] == 1
    assert metrics["valid_observer_count"] == 1


def test_metrics_mark_an_entirely_unobserved_slice_as_unscored():
    logits, targets, alive, diagonal = make_contract()
    targets.zero_()
    scope = alive.clone()
    observed = torch.zeros_like(alive)

    metrics = compute_belief_metrics(
        logits,
        targets,
        alive,
        diagonal,
        observer_supervision_mask=scope & observed,
        observer_scope_mask=scope,
        label_observed_mask=observed,
    )

    assert metrics == {
        "status": UNSCORED_NO_SUPERVISION_STATUS,
        "total_row_count": 0,
        "valid_observer_count": 0,
        "scope_observer_count": 1,
        "observed_label_row_count_in_scope": 0,
        "unobserved_label_row_count_in_scope": 1,
    }


def test_metrics_expose_no_pair_or_order_specific_names():
    metrics = compute_belief_metrics(*make_contract())
    assert all("pair" not in name for name in metrics)
    assert all("order" not in name for name in metrics)


def test_max_set_and_deterministic_top1_have_frozen_tie_semantics():
    logits, targets, alive, diagonal = make_contract()
    targets.zero_()
    targets[0, 0, 1] = 0.5
    targets[0, 0, 2] = 0.5
    logits[0, 0, 2] = 10.0

    metrics = compute_belief_metrics(logits, targets, alive, diagonal)
    assert metrics["mean_belief_max_set_support_hit"] == pytest.approx(1.0)
    assert metrics[
        "mean_belief_deterministic_top1_support_hit"
    ] == pytest.approx(1.0)

    logits[0, 0, 2] = 0.0
    logits[0, 0, 3] = 10.0
    metrics = compute_belief_metrics(logits, targets, alive, diagonal)
    assert metrics["mean_belief_max_set_support_hit"] == pytest.approx(0.0)
    assert metrics[
        "mean_belief_deterministic_top1_support_hit"
    ] == pytest.approx(0.0)

    logits.zero_()
    metrics = compute_belief_metrics(logits, targets, alive, diagonal)
    assert metrics["mean_belief_max_set_support_hit"] == pytest.approx(1.0)
    assert metrics[
        "mean_belief_deterministic_top1_support_hit"
    ] == pytest.approx(1.0)

    targets.zero_()
    targets[0, 0, 3] = 1.0
    metrics = compute_belief_metrics(logits, targets, alive, diagonal)
    assert metrics["mean_belief_max_set_support_hit"] == pytest.approx(1.0)
    assert metrics[
        "mean_belief_deterministic_top1_support_hit"
    ] == pytest.approx(0.0)


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


def test_kl_is_cross_entropy_minus_target_entropy():
    logits, targets, alive, diagonal = make_contract()
    targets.zero_()
    targets[0, 0, 1] = 0.5
    targets[0, 0, 2] = 0.5
    logits[0, 0, 1] = math.log(0.5)
    logits[0, 0, 2] = math.log(0.5)
    logits[0, 0, 3:] = -100.0

    metrics = compute_belief_metrics(logits, targets, alive, diagonal)
    assert metrics["mean_belief_cross_entropy"] == pytest.approx(math.log(2))
    assert metrics["mean_belief_target_entropy"] == pytest.approx(math.log(2))
    assert metrics["mean_belief_kl_divergence"] == pytest.approx(0.0, abs=1e-7)


def test_sparse_target_has_zero_entropy():
    logits, targets, alive, diagonal = make_contract()
    targets.zero_()
    targets[0, 0, 1] = 1.0

    metrics = compute_belief_metrics(logits, targets, alive, diagonal)
    assert metrics["mean_belief_target_entropy"] == pytest.approx(0.0)
    assert metrics["mean_belief_kl_divergence"] == pytest.approx(math.log(6))


def test_private_admissible_uniform_baseline_excludes_known_non_werewolves():
    logits, targets, alive, diagonal = make_contract()
    targets.zero_()
    targets[0, 0, 1] = 1.0
    known_non_wolf = torch.zeros_like(diagonal)
    known_non_wolf[0, 0, [0, 2, 3, 4, 5, 6]] = True

    metrics = compute_belief_metrics(
        logits,
        targets,
        alive,
        diagonal,
        known_non_werewolf_mask=known_non_wolf,
    )

    assert metrics[
        "private_admissible_uniform_baseline_mean_cross_entropy"
    ] == pytest.approx(0.0)
    assert metrics[
        "private_admissible_uniform_baseline_mean_total_variation"
    ] == pytest.approx(0.0)
    assert metrics[
        "private_admissible_uniform_baseline_mean_absolute_error"
    ] == pytest.approx(0.0)
    assert metrics["mean_illegal_known_nonwolf_mass"] == pytest.approx(5 / 6)


def test_private_admissible_diagnostic_rejects_incompatible_empty_target():
    logits, targets, alive, diagonal = make_contract()
    known_non_wolf = torch.zeros_like(diagonal)
    known_non_wolf[0, 0, [0, 3]] = True

    with pytest.raises(
        ValueError,
        match="belief targets cannot support known non-Werewolves",
    ):
        compute_belief_metrics(
            logits,
            targets,
            alive,
            diagonal,
            known_non_werewolf_mask=known_non_wolf,
        )
