"""Deterministic evaluation metrics for Public Belief Matrix V1."""

from __future__ import annotations

import torch

from werewolf.models.public_belief_matrix.losses import (
    masked_row_soft_target_cross_entropy,
)
from werewolf.models.twd_tom.schema import NUM_PLAYERS


@torch.no_grad()
def masked_mean_row_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    observer_mask: torch.Tensor,
) -> float:
    """Return the loss contract as an evaluation scalar."""

    return float(
        masked_row_soft_target_cross_entropy(
            logits, targets, observer_mask
        ).item()
    )


@torch.no_grad()
def masked_mean_row_entropy(
    probabilities: torch.Tensor,
    observer_mask: torch.Tensor,
) -> float:
    """Average categorical entropy over valid observer rows."""

    valid = _validated_probability_rows(probabilities, observer_mask)
    safe_log = valid.clamp_min(torch.finfo(valid.dtype).tiny).log()
    return float((-(valid * safe_log).sum(dim=-1)).mean().item())


@torch.no_grad()
def mean_observer_pairwise_tv(
    probabilities: torch.Tensor,
    observer_mask: torch.Tensor,
) -> float | None:
    """Average TV across all within-snapshot valid observer pairs."""

    probabilities, observer_mask = _validate_probability_inputs(
        probabilities, observer_mask
    )
    selected = probabilities[observer_mask]
    if selected.numel() > 0:
        _validate_selected_probabilities(selected)
    pairwise_values = []
    for snapshot, snapshot_mask in zip(probabilities, observer_mask):
        valid_rows = snapshot[snapshot_mask]
        if valid_rows.shape[0] < 2:
            continue
        indices = torch.triu_indices(
            valid_rows.shape[0],
            valid_rows.shape[0],
            offset=1,
            device=valid_rows.device,
        )
        pairwise_values.append(
            0.5
            * (
                valid_rows[indices[0]] - valid_rows[indices[1]]
            ).abs().sum(dim=-1)
        )
    if not pairwise_values:
        return None
    return float(torch.cat(pairwise_values).mean().item())


@torch.no_grad()
def mean_prediction_diagonal_mass(
    probabilities: torch.Tensor,
    observer_mask: torch.Tensor,
) -> float:
    """Average prediction p[observer, observer] over valid rows."""

    probabilities, observer_mask = _validate_probability_inputs(
        probabilities, observer_mask
    )
    valid = probabilities[observer_mask]
    if valid.numel() == 0:
        raise ValueError("observer_mask must select at least one valid row")
    _validate_selected_probabilities(valid)
    diagonal = probabilities.diagonal(dim1=-2, dim2=-1)
    return float(diagonal[observer_mask].mean().item())


def _validated_probability_rows(
    probabilities: torch.Tensor,
    observer_mask: torch.Tensor,
) -> torch.Tensor:
    probabilities, observer_mask = _validate_probability_inputs(
        probabilities, observer_mask
    )
    valid = probabilities[observer_mask]
    if valid.numel() == 0:
        raise ValueError("observer_mask must select at least one valid row")
    _validate_selected_probabilities(valid)
    return valid


def _validate_probability_inputs(
    probabilities: torch.Tensor,
    observer_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(probabilities, torch.Tensor):
        raise TypeError("probabilities must be a tensor")
    if not isinstance(observer_mask, torch.Tensor):
        raise TypeError("observer_mask must be a tensor")
    if probabilities.ndim != 3 or tuple(probabilities.shape[1:]) != (
        NUM_PLAYERS,
        NUM_PLAYERS,
    ):
        raise ValueError("probabilities must have shape [B, 7, 7]")
    if observer_mask.shape != probabilities.shape[:2]:
        raise ValueError("observer_mask must have shape [B, 7]")
    if probabilities.shape[0] <= 0:
        raise ValueError("batch size must be positive")
    if not torch.is_floating_point(probabilities):
        raise TypeError("probabilities must use a floating-point dtype")
    if observer_mask.dtype != torch.bool:
        raise TypeError("observer_mask must use torch.bool")
    return probabilities, observer_mask.to(device=probabilities.device)


def _validate_selected_probabilities(probabilities: torch.Tensor) -> None:
    if not torch.isfinite(probabilities).all():
        raise ValueError("valid probability rows must be finite")
    if torch.any(probabilities < 0):
        raise ValueError("valid probability rows cannot be negative")
    if not torch.allclose(
        probabilities.sum(dim=-1),
        torch.ones(
            probabilities.shape[0],
            device=probabilities.device,
            dtype=probabilities.dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("every valid probability row must sum to one")


__all__ = [
    "masked_mean_row_cross_entropy",
    "masked_mean_row_entropy",
    "mean_observer_pairwise_tv",
    "mean_prediction_diagonal_mass",
]
