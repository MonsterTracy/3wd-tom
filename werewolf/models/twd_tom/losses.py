"""Masked distribution loss for observer-conditioned player beliefs."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from werewolf.models.twd_tom.schema import NUM_PLAYERS


VALID_REDUCTIONS = {"none", "sum", "mean"}


def masked_belief_distribution_loss(
    belief_logits: torch.Tensor,
    belief_targets: torch.Tensor,
    observer_alive_mask: torch.Tensor,
    diagonal_target_mask: torch.Tensor,
    *,
    observer_supervision_mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute soft-target cross entropy over alive observer rows."""

    (
        belief_logits,
        belief_targets,
        observer_alive_mask,
        observer_supervision_mask,
        diagonal_target_mask,
    ) = _validate_belief_loss_inputs(
        belief_logits=belief_logits,
        belief_targets=belief_targets,
        observer_alive_mask=observer_alive_mask,
        observer_supervision_mask=observer_supervision_mask,
        diagonal_target_mask=diagonal_target_mask,
        reduction=reduction,
    )
    valid_observer_count = observer_supervision_mask.sum()
    if valid_observer_count.item() == 0:
        raise ValueError(
            "observer supervision must select at least one observer among alive rows"
        )

    masked_logits = belief_logits.masked_fill(~diagonal_target_mask, -torch.inf)
    log_probabilities = F.log_softmax(masked_logits, dim=-1)
    per_target_loss = torch.where(
        diagonal_target_mask,
        -belief_targets * log_probabilities,
        torch.zeros_like(belief_targets),
    )
    per_observer_loss = per_target_loss.sum(dim=-1)
    masked_loss = per_observer_loss * observer_supervision_mask.to(
        dtype=per_observer_loss.dtype
    )
    if reduction == "none":
        return masked_loss
    total_loss = masked_loss.sum()
    if reduction == "sum":
        return total_loss
    return total_loss / valid_observer_count.to(dtype=total_loss.dtype)


def masked_belief_probabilities(
    belief_logits: torch.Tensor,
    diagonal_target_mask: torch.Tensor,
) -> torch.Tensor:
    """Return row-normalized player beliefs with an exact zero diagonal."""

    if not isinstance(belief_logits, torch.Tensor):
        raise TypeError("belief_logits must be a tensor")
    if not isinstance(diagonal_target_mask, torch.Tensor):
        raise TypeError("diagonal_target_mask must be a tensor")
    if belief_logits.ndim != 3 or tuple(belief_logits.shape[-2:]) != (
        NUM_PLAYERS,
        NUM_PLAYERS,
    ):
        raise ValueError("belief_logits must have shape [B, 7, 7]")
    if diagonal_target_mask.shape != belief_logits.shape:
        raise ValueError("diagonal_target_mask must match belief_logits shape")
    if diagonal_target_mask.dtype is not torch.bool:
        raise TypeError("diagonal_target_mask must use torch.bool")
    if not torch.is_floating_point(belief_logits):
        raise TypeError("belief_logits must use a floating-point dtype")
    if not torch.isfinite(belief_logits).all():
        raise ValueError("belief_logits must contain only finite values")
    mask = diagonal_target_mask.to(device=belief_logits.device)
    if torch.any(mask.sum(dim=-1) == 0):
        raise ValueError("every observer row requires at least one target")
    probabilities = F.softmax(belief_logits.masked_fill(~mask, -torch.inf), dim=-1)
    return probabilities.masked_fill(~mask, 0.0)


def _validate_belief_loss_inputs(
    *,
    belief_logits: torch.Tensor,
    belief_targets: torch.Tensor,
    observer_alive_mask: torch.Tensor,
    observer_supervision_mask: torch.Tensor | None,
    diagonal_target_mask: torch.Tensor,
    reduction: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if reduction not in VALID_REDUCTIONS:
        raise ValueError(
            f"reduction must be one of {sorted(VALID_REDUCTIONS)}, got {reduction!r}"
        )
    tensors = {
        "belief_logits": belief_logits,
        "belief_targets": belief_targets,
        "observer_alive_mask": observer_alive_mask,
        "diagonal_target_mask": diagonal_target_mask,
    }
    if observer_supervision_mask is not None:
        tensors["observer_supervision_mask"] = observer_supervision_mask
    for field_name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{field_name} must be a tensor")

    if belief_logits.ndim != 3 or tuple(belief_logits.shape[-2:]) != (
        NUM_PLAYERS,
        NUM_PLAYERS,
    ):
        raise ValueError("belief_logits must have shape [B, 7, 7]")
    if belief_logits.shape[0] <= 0:
        raise ValueError("batch size must be positive")
    if belief_targets.shape != belief_logits.shape:
        raise ValueError("belief_targets must match belief_logits shape")
    expected_observer_shape = (belief_logits.shape[0], NUM_PLAYERS)
    if observer_alive_mask.shape != expected_observer_shape:
        raise ValueError("observer_alive_mask must have shape [B, 7]")
    if (
        observer_supervision_mask is not None
        and observer_supervision_mask.shape != expected_observer_shape
    ):
        raise ValueError("observer_supervision_mask must have shape [B, 7]")
    if diagonal_target_mask.shape != belief_logits.shape:
        raise ValueError("diagonal_target_mask must have shape [B, 7, 7]")
    if not torch.is_floating_point(belief_logits):
        raise TypeError("belief_logits must use a floating-point dtype")
    if not torch.is_floating_point(belief_targets):
        raise TypeError("belief_targets must use a floating-point dtype")
    if observer_alive_mask.dtype is not torch.bool:
        raise TypeError("observer_alive_mask must use torch.bool")
    if (
        observer_supervision_mask is not None
        and observer_supervision_mask.dtype is not torch.bool
    ):
        raise TypeError("observer_supervision_mask must use torch.bool")
    if diagonal_target_mask.dtype is not torch.bool:
        raise TypeError("diagonal_target_mask must use torch.bool")
    if not torch.isfinite(belief_logits).all():
        raise ValueError("belief_logits must contain only finite values")
    if not torch.isfinite(belief_targets).all():
        raise ValueError("belief_targets must contain only finite values")
    if torch.any(belief_targets < 0.0):
        raise ValueError("belief_targets cannot contain negative values")

    targets = belief_targets.to(
        device=belief_logits.device,
        dtype=belief_logits.dtype,
    )
    alive_mask = observer_alive_mask.to(device=belief_logits.device)
    supervision_mask = (
        alive_mask
        if observer_supervision_mask is None
        else observer_supervision_mask.to(device=belief_logits.device)
    )
    if torch.any(supervision_mask & ~alive_mask):
        raise ValueError("observer_supervision_mask must be a subset of alive rows")
    target_mask = diagonal_target_mask.to(device=belief_logits.device)
    expected_target_mask = ~torch.eye(
        NUM_PLAYERS,
        dtype=torch.bool,
        device=belief_logits.device,
    ).unsqueeze(0).expand_as(target_mask)
    if not torch.equal(target_mask, expected_target_mask):
        raise ValueError("diagonal_target_mask must exclude exactly the diagonal")
    if torch.any(targets.masked_select(~target_mask) != 0.0):
        raise ValueError("belief target diagonal must remain zero")

    row_sums = targets.sum(dim=-1)
    if not torch.allclose(
        row_sums[supervision_mask],
        torch.ones_like(row_sums[supervision_mask]),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("every supervised observer target row must sum to one")
    unsupervised_alive_sums = row_sums[alive_mask & ~supervision_mask]
    if unsupervised_alive_sums.numel() and not torch.all(
        torch.isclose(
            unsupervised_alive_sums,
            torch.zeros_like(unsupervised_alive_sums),
            rtol=0.0,
            atol=1e-6,
        )
        | torch.isclose(
            unsupervised_alive_sums,
            torch.ones_like(unsupervised_alive_sums),
            rtol=1e-5,
            atol=1e-6,
        )
    ):
        raise ValueError(
            "unsupervised alive target rows must sum to zero or one"
        )
    if torch.any(targets[~alive_mask] != 0.0):
        raise ValueError("dead observer target rows must remain all zero")
    return belief_logits, targets, alive_mask, supervision_mask, target_mask


__all__ = [
    "VALID_REDUCTIONS",
    "masked_belief_distribution_loss",
    "masked_belief_probabilities",
]
