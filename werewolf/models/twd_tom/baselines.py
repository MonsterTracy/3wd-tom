"""Train-only empirical baselines for dense belief prediction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from werewolf.models.twd_tom.dense_dataset import DenseTWDToMDataset
from werewolf.models.twd_tom.metrics import compute_belief_metrics
from werewolf.models.twd_tom.schema import NUM_PLAYERS


EMPIRICAL_PRIOR_VERSION = "train_only_observer_phase_prior_v1"
EMPIRICAL_PRIOR_SMOOTHING = 1.0


class _MetricMean:
    def __init__(self) -> None:
        self.count = 0
        self.sums: dict[str, float] = {}

    def update(self, metrics: Mapping[str, int | float]) -> None:
        count = int(metrics["valid_observer_count"])
        self.count += count
        for name, value in metrics.items():
            if name != "valid_observer_count":
                self.sums[name] = self.sums.get(name, 0.0) + float(value) * count

    def finalize(self) -> dict[str, int | float]:
        if self.count <= 0:
            raise ValueError("baseline evaluation has no valid observers")
        result: dict[str, int | float] = {
            "valid_observer_count": self.count,
            **{name: value / self.count for name, value in self.sums.items()},
        }
        result["mean_loss"] = result["mean_belief_cross_entropy"]
        uniform_kl = float(
            result["uniform_non_self_baseline_mean_cross_entropy"]
        ) - float(result["mean_belief_target_entropy"])
        result["uniform_non_self_baseline_mean_kl_divergence"] = uniform_kl
        result["normalized_reducible_gap_improvement"] = (
            1.0 - float(result["mean_belief_kl_divergence"]) / uniform_kl
            if uniform_kl > 0.0
            else 0.0
        )
        return result


def _uniform_non_self(*, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    mask = ~torch.eye(NUM_PLAYERS, dtype=torch.bool)
    probabilities = mask.to(dtype=dtype)
    return probabilities / probabilities.sum(dim=-1, keepdim=True)


def _smoothed_prior(
    sums: torch.Tensor,
    counts: torch.Tensor,
    *,
    fallback: torch.Tensor,
) -> torch.Tensor:
    if sums.shape != (NUM_PLAYERS, NUM_PLAYERS):
        raise ValueError("prior sums must have shape [7, 7]")
    if counts.shape != (NUM_PLAYERS,):
        raise ValueError("prior counts must have shape [7]")
    smoothed = (
        sums + EMPIRICAL_PRIOR_SMOOTHING * fallback
    ) / (counts.unsqueeze(-1) + EMPIRICAL_PRIOR_SMOOTHING)
    return smoothed / smoothed.sum(dim=-1, keepdim=True)


def fit_dense_empirical_priors(
    dataset: DenseTWDToMDataset,
) -> dict[str, Any]:
    """Fit observer and observer+phase priors using training labels only."""

    if not isinstance(dataset, DenseTWDToMDataset):
        raise TypeError("empirical priors require DenseTWDToMDataset")
    if dataset.enable_cyclic_rotation:
        raise ValueError("fit empirical priors on an unaugmented training dataset")
    global_sums = torch.zeros(
        (NUM_PLAYERS, NUM_PLAYERS), dtype=torch.float64
    )
    global_counts = torch.zeros(NUM_PLAYERS, dtype=torch.float64)
    phase_sums: dict[str, torch.Tensor] = {}
    phase_counts: dict[str, torch.Tensor] = {}
    for item_index in range(len(dataset)):
        item = dataset[item_index]
        targets = item["belief_targets"].to(dtype=torch.float64)
        alive = item["observer_alive_mask"]
        global_sums += (targets * alive.unsqueeze(-1)).sum(dim=0)
        global_counts += alive.sum(dim=0)
        for boundary_index, phase in enumerate(item["metadata"]["phase"]):
            phase_sums.setdefault(
                phase,
                torch.zeros_like(global_sums),
            )
            phase_counts.setdefault(
                phase,
                torch.zeros_like(global_counts),
            )
            phase_sums[phase] += (
                targets[boundary_index] * alive[boundary_index].unsqueeze(-1)
            )
            phase_counts[phase] += alive[boundary_index]

    uniform = _uniform_non_self()
    global_prior = _smoothed_prior(
        global_sums,
        global_counts,
        fallback=uniform,
    )
    return {
        "version": EMPIRICAL_PRIOR_VERSION,
        "smoothing": EMPIRICAL_PRIOR_SMOOTHING,
        "training_game_count": len(dataset),
        "training_boundary_count": dataset.boundary_count,
        "global": global_prior,
        "by_phase": {
            phase: _smoothed_prior(
                phase_sums[phase],
                phase_counts[phase],
                fallback=global_prior,
            )
            for phase in sorted(phase_sums)
        },
    }


def _prior_logits(probabilities: torch.Tensor) -> torch.Tensor:
    return probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()


def _evaluate_prior_for_game(
    item: Mapping[str, Any],
    *,
    global_prior: torch.Tensor,
    phase_priors: Mapping[str, torch.Tensor] | None,
) -> dict[str, int | float]:
    phases = item["metadata"]["phase"]
    probabilities = torch.stack(
        [
            global_prior
            if phase_priors is None
            else phase_priors.get(phase, global_prior)
            for phase in phases
        ]
    ).to(dtype=item["belief_targets"].dtype)
    return compute_belief_metrics(
        _prior_logits(probabilities),
        item["belief_targets"],
        item["observer_alive_mask"],
        item["diagonal_target_mask"],
    )


def evaluate_dense_empirical_priors(
    dataset: DenseTWDToMDataset,
    priors: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate train-only global and phase priors on a disjoint dataset."""

    if not isinstance(dataset, DenseTWDToMDataset):
        raise TypeError("empirical prior evaluation requires DenseTWDToMDataset")
    if dataset.enable_cyclic_rotation:
        raise ValueError("evaluate empirical priors on an unaugmented dataset")
    if priors.get("version") != EMPIRICAL_PRIOR_VERSION:
        raise ValueError("empirical prior version mismatch")
    global_prior = priors.get("global")
    phase_priors = priors.get("by_phase")
    if not isinstance(global_prior, torch.Tensor) or not isinstance(
        phase_priors, Mapping
    ):
        raise TypeError("empirical priors are incomplete")

    reports: dict[str, Any] = {
        "version": EMPIRICAL_PRIOR_VERSION,
        "smoothing": float(priors["smoothing"]),
    }
    for name, selected_phase_priors in {
        "train_global_prior": None,
        "train_phase_prior": phase_priors,
    }.items():
        aggregate = _MetricMean()
        by_game: dict[str, dict[str, int | float]] = {}
        for item_index in range(len(dataset)):
            item = dataset[item_index]
            game_id = item["metadata"]["game_id"]
            metrics = _evaluate_prior_for_game(
                item,
                global_prior=global_prior,
                phase_priors=selected_phase_priors,
            )
            aggregate.update(metrics)
            game_report = dict(metrics)
            game_report["mean_loss"] = game_report[
                "mean_belief_cross_entropy"
            ]
            by_game[game_id] = game_report
        reports[name] = {
            "aggregate": aggregate.finalize(),
            "by_game": dict(sorted(by_game.items())),
        }
    return reports


__all__ = [
    "EMPIRICAL_PRIOR_SMOOTHING",
    "EMPIRICAL_PRIOR_VERSION",
    "evaluate_dense_empirical_priors",
    "fit_dense_empirical_priors",
]
