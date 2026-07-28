"""Minimal valid-row pair-distribution metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from werewolf.models.twd_tom.losses import masked_pair_kl_divergence
from werewolf.models.twd_tom.belief_labels import (
    pair_probabilities_to_belief_marginals,
)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    return float(selected.mean().item()) if selected.numel() else 0.0


@torch.no_grad()
def compute_subjective_pair_metrics(
    pair_logits: torch.Tensor,
    pair_targets: torch.Tensor,
    subject_mask: torch.Tensor,
) -> dict[str, int | float]:
    """Compute required metrics without using truth or support weighting."""

    per_subject_kl = masked_pair_kl_divergence(
        pair_logits,
        pair_targets,
        subject_mask,
        reduction="none",
    )
    valid_mask = subject_mask.to(device=pair_logits.device, dtype=torch.bool)

    targets = pair_targets.to(device=pair_logits.device, dtype=pair_logits.dtype)
    log_probabilities = F.log_softmax(pair_logits, dim=-1)
    probabilities = log_probabilities.exp()
    cross_entropy = -(targets * log_probabilities).sum(dim=-1)
    total_variation = 0.5 * (probabilities - targets).abs().sum(dim=-1)
    predicted_marginals = pair_probabilities_to_belief_marginals(probabilities)
    target_marginals = pair_probabilities_to_belief_marginals(targets)
    marginal_mae = (predicted_marginals - target_marginals).abs().mean(dim=-1)
    marginal_row_sum_error = (
        predicted_marginals.sum(dim=-1) - 2.0
    ).abs()
    predicted_diagonal = predicted_marginals.diagonal(dim1=-2, dim2=-1)
    target_diagonal = target_marginals.diagonal(dim1=-2, dim2=-1)

    return {
        "valid_subject_count": int(valid_mask.sum().item()),
        "mean_pair_kl_divergence": _masked_mean(per_subject_kl, valid_mask),
        "mean_pair_cross_entropy": _masked_mean(cross_entropy, valid_mask),
        "mean_pair_total_variation": _masked_mean(total_variation, valid_mask),
        "mean_marginal_mae": _masked_mean(marginal_mae, valid_mask),
        "mean_marginal_row_sum_error": _masked_mean(
            marginal_row_sum_error,
            valid_mask,
        ),
        "mean_predicted_diagonal_marginal": _masked_mean(
            predicted_diagonal,
            valid_mask,
        ),
        "mean_target_diagonal_marginal": _masked_mean(
            target_diagonal,
            valid_mask,
        ),
    }


__all__ = ["compute_subjective_pair_metrics"]
