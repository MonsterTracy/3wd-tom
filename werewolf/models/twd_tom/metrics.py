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

    total_variation = 0.5 * (probabilities - targets).abs().sum(dim=-1)
    absolute_error = torch.where(
        target_mask,
        (probabilities - targets).abs(),
        torch.zeros_like(probabilities),
    ).sum(dim=-1) / target_mask.sum(dim=-1).clamp_min(1)
    top1_agreement = (
        probabilities.argmax(dim=-1) == targets.argmax(dim=-1)
    ).to(dtype=belief_logits.dtype)

    return {
        "valid_observer_count": int(valid_observers.sum().item()),
        "mean_belief_cross_entropy": _masked_mean(
            per_observer_loss,
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
        "mean_belief_top1_agreement": _masked_mean(
            top1_agreement,
            valid_observers,
        ),
    }


__all__ = ["compute_belief_metrics"]
