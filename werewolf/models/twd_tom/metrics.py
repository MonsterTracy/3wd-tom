"""Metrics for observer-conditioned player belief distributions."""

from __future__ import annotations

import torch

from werewolf.models.twd_tom.losses import (
    masked_belief_distribution_loss,
    masked_belief_probabilities,
)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    return float(selected.mean().item()) if selected.numel() else 0.0


@torch.no_grad()
def compute_belief_metrics(
    belief_logits: torch.Tensor,
    belief_targets: torch.Tensor,
    observer_alive_mask: torch.Tensor,
    diagonal_target_mask: torch.Tensor,
) -> dict[str, int | float]:
    """Measure predictions only for living observers and non-self targets."""

    per_observer_loss = masked_belief_distribution_loss(
        belief_logits,
        belief_targets,
        observer_alive_mask,
        diagonal_target_mask,
        reduction="none",
    )
    valid_observers = observer_alive_mask.to(
        device=belief_logits.device,
        dtype=torch.bool,
    )
    target_mask = diagonal_target_mask.to(
        device=belief_logits.device,
        dtype=torch.bool,
    )
    targets = belief_targets.to(
        device=belief_logits.device,
        dtype=belief_logits.dtype,
    )
    probabilities = masked_belief_probabilities(
        belief_logits,
        target_mask,
    )
    target_entropy = torch.where(
        targets > 0,
        -targets * targets.clamp_min(
            torch.finfo(belief_logits.dtype).tiny
        ).log(),
        torch.zeros_like(targets),
    ).sum(dim=-1)
    kl_divergence = per_observer_loss - target_entropy
    uniform_probabilities = target_mask.to(dtype=belief_logits.dtype)
    uniform_probabilities /= target_mask.sum(dim=-1, keepdim=True).clamp_min(1)

    total_variation = 0.5 * (probabilities - targets).abs().sum(dim=-1)
    absolute_error = torch.where(
        target_mask,
        (probabilities - targets).abs(),
        torch.zeros_like(probabilities),
    ).sum(dim=-1) / target_mask.sum(dim=-1).clamp_min(1)
    top_probability = probabilities.max(dim=-1, keepdim=True).values
    predicted_top_support = probabilities == top_probability
    target_support = targets > 0
    top1_support_hit = (
        predicted_top_support & target_support & target_mask
    ).any(dim=-1).to(dtype=belief_logits.dtype)

    uniform_cross_entropy = -(
        targets
        * uniform_probabilities.clamp_min(
            torch.finfo(belief_logits.dtype).tiny
        ).log()
    ).sum(dim=-1)
    uniform_total_variation = 0.5 * (
        uniform_probabilities - targets
    ).abs().sum(dim=-1)
    uniform_absolute_error = torch.where(
        target_mask,
        (uniform_probabilities - targets).abs(),
        torch.zeros_like(uniform_probabilities),
    ).sum(dim=-1) / target_mask.sum(dim=-1).clamp_min(1)

    return {
        "valid_observer_count": int(valid_observers.sum().item()),
        "mean_belief_cross_entropy": _masked_mean(
            per_observer_loss,
            valid_observers,
        ),
        "mean_belief_target_entropy": _masked_mean(
            target_entropy,
            valid_observers,
        ),
        "mean_belief_kl_divergence": _masked_mean(
            kl_divergence,
            valid_observers,
        ),
        "mean_belief_total_variation": _masked_mean(
            total_variation,
            valid_observers,
        ),
        "mean_belief_absolute_error": _masked_mean(
            absolute_error,
            valid_observers,
        ),
        "mean_belief_top1_support_hit": _masked_mean(
            top1_support_hit,
            valid_observers,
        ),
        "uniform_non_self_baseline_mean_cross_entropy": _masked_mean(
            uniform_cross_entropy,
            valid_observers,
        ),
        "uniform_non_self_baseline_mean_total_variation": _masked_mean(
            uniform_total_variation,
            valid_observers,
        ),
        "uniform_non_self_baseline_mean_absolute_error": _masked_mean(
            uniform_absolute_error,
            valid_observers,
        ),
    }


__all__ = ["compute_belief_metrics"]
