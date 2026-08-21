"""Masked row-wise loss for Public Belief Matrix V1."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from werewolf.models.twd_tom.schema import NUM_PLAYERS


def masked_row_soft_target_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    observer_mask: torch.Tensor,
) -> torch.Tensor:
    """Average soft-target cross entropy over valid observer rows only."""

    logits, targets, observer_mask = _validate_inputs(
        logits, targets, observer_mask
    )
    valid_logits = logits[observer_mask]
    valid_targets = targets[observer_mask]
    return -(
        valid_targets * F.log_softmax(valid_logits, dim=-1)
    ).sum(dim=-1).mean()


def _validate_inputs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    observer_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    for name, tensor in {
        "logits": logits,
        "targets": targets,
        "observer_mask": observer_mask,
    }.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
    if logits.ndim != 3 or tuple(logits.shape[1:]) != (
        NUM_PLAYERS,
        NUM_PLAYERS,
    ):
        raise ValueError("logits must have shape [B, 7, 7]")
    if targets.shape != logits.shape:
        raise ValueError("targets must have the same shape as logits")
    if observer_mask.shape != logits.shape[:2]:
        raise ValueError("observer_mask must have shape [B, 7]")
    if logits.shape[0] <= 0:
        raise ValueError("batch size must be positive")
    if not torch.is_floating_point(logits):
        raise TypeError("logits must use a floating-point dtype")
    if not torch.is_floating_point(targets):
        raise TypeError("targets must use a floating-point dtype")
    if observer_mask.dtype != torch.bool:
        raise TypeError("observer_mask must use torch.bool")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must contain only finite values")

    targets = targets.to(device=logits.device, dtype=logits.dtype)
    observer_mask = observer_mask.to(device=logits.device)
    if not observer_mask.any():
        raise ValueError("observer_mask must select at least one valid row")
    valid_targets = targets[observer_mask]
    if not torch.isfinite(valid_targets).all():
        raise ValueError("valid target rows must contain only finite values")
    if torch.any(valid_targets < 0):
        raise ValueError("valid target rows cannot contain negative values")
    if not torch.allclose(
        valid_targets.sum(dim=-1),
        torch.ones(
            valid_targets.shape[0],
            device=logits.device,
            dtype=logits.dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("every valid target row must sum to one")
    return logits, targets, observer_mask


__all__ = ["masked_row_soft_target_cross_entropy"]
