"""Order-specific metrics for observer belief distributions."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from werewolf.models.twd_tom.losses import masked_distribution_kl_divergence
from werewolf.models.twd_tom.belief_labels import (
    pair_probabilities_to_belief_marginals,
)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    return float(selected.mean().item()) if selected.numel() else 0.0


def _distribution_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    safe_log = probabilities.clamp_min(
        torch.finfo(probabilities.dtype).tiny
    ).log()
    return -(probabilities * safe_log).sum(dim=-1)


def _top1_top2_margin(probabilities: torch.Tensor) -> torch.Tensor:
    top_two = probabilities.topk(2, dim=-1).values
    return top_two[..., 0] - top_two[..., 1]


def _mean_observer_pairwise_tv(
    probabilities: torch.Tensor,
    valid_mask: torch.Tensor,
) -> float:
    snapshot_means = []
    for snapshot, snapshot_mask in zip(probabilities, valid_mask):
        valid_rows = snapshot[snapshot_mask]
        if valid_rows.shape[0] < 2:
            continue
        pairwise_tv = 0.5 * (
            valid_rows.unsqueeze(1) - valid_rows.unsqueeze(0)
        ).abs().sum(dim=-1)
        upper_triangle = torch.triu_indices(
            valid_rows.shape[0],
            valid_rows.shape[0],
            offset=1,
            device=valid_rows.device,
        )
        snapshot_means.append(
            pairwise_tv[upper_triangle[0], upper_triangle[1]].mean()
        )
    if not snapshot_means:
        return 0.0
    return float(torch.stack(snapshot_means).mean().item())


@torch.no_grad()
def compute_subjective_pair_metrics(
    pair_logits: torch.Tensor,
    pair_targets: torch.Tensor,
    subject_mask: torch.Tensor,
) -> dict[str, int | float]:
    """Compute required metrics without using truth or support weighting."""

    per_subject_kl = masked_distribution_kl_divergence(
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


@torch.no_grad()
def compute_subjective_pair_diagnostics(
    pair_logits: torch.Tensor,
    pair_targets: torch.Tensor,
    subject_mask: torch.Tensor,
) -> dict[str, float]:
    """Measure pair sharpness, marginal spread, and observer diversity."""

    compute_subjective_pair_metrics(pair_logits, pair_targets, subject_mask)
    valid_mask = subject_mask.to(device=pair_logits.device, dtype=torch.bool)
    targets = pair_targets.to(device=pair_logits.device, dtype=pair_logits.dtype)
    probabilities = F.softmax(pair_logits, dim=-1)
    predicted_marginals = pair_probabilities_to_belief_marginals(probabilities)
    target_marginals = pair_probabilities_to_belief_marginals(targets)
    return {
        "mean_target_pair_entropy": _masked_mean(
            _distribution_entropy(targets), valid_mask
        ),
        "mean_predicted_pair_entropy": _masked_mean(
            _distribution_entropy(probabilities), valid_mask
        ),
        "mean_target_pair_top1_top2_margin": _masked_mean(
            _top1_top2_margin(targets), valid_mask
        ),
        "mean_predicted_pair_top1_top2_margin": _masked_mean(
            _top1_top2_margin(probabilities), valid_mask
        ),
        "mean_target_marginal_spread": _masked_mean(
            target_marginals.max(dim=-1).values
            - target_marginals.min(dim=-1).values,
            valid_mask,
        ),
        "mean_predicted_marginal_spread": _masked_mean(
            predicted_marginals.max(dim=-1).values
            - predicted_marginals.min(dim=-1).values,
            valid_mask,
        ),
        "mean_target_observer_pairwise_tv": _mean_observer_pairwise_tv(
            targets, valid_mask
        ),
        "mean_predicted_observer_pairwise_tv": _mean_observer_pairwise_tv(
            probabilities, valid_mask
        ),
    }


__all__ = [
    "compute_subjective_pair_diagnostics",
    "compute_subjective_pair_metrics",
]
