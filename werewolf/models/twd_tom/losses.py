"""Masked categorical losses for observer-specific distributions."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from werewolf.models.twd_tom.schema import (
    NUM_PLAYERS,
    NUM_WOLF_PAIR_CLASSES,
)


VALID_REDUCTIONS = {
    "none",
    "sum",
    "mean",
}


def masked_distribution_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    subject_mask: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute soft-target cross entropy over supervised observer rows."""

    logits, targets, subject_mask = _validate_distribution_loss_inputs(
        logits=logits,
        targets=targets,
        subject_mask=subject_mask,
        reduction=reduction,
    )
    valid_subject_count = subject_mask.sum()
    if valid_subject_count.item() == 0:
        raise ValueError("subject_mask must select at least one valid observer")

    per_subject_loss = -(
        targets * F.log_softmax(logits, dim=-1)
    ).sum(dim=-1)
    masked_loss = per_subject_loss * subject_mask.to(per_subject_loss.dtype)
    if reduction == "none":
        return masked_loss
    total_loss = masked_loss.sum()
    if reduction == "sum":
        return total_loss
    return total_loss / valid_subject_count.to(dtype=total_loss.dtype)


def masked_distribution_kl_divergence(
    logits: torch.Tensor,
    targets: torch.Tensor,
    subject_mask: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute target-to-prediction KL over valid observer rows."""

    (
        logits,
        targets,
        subject_mask,
    ) = _validate_distribution_loss_inputs(
        logits=logits,
        targets=targets,
        subject_mask=subject_mask,
        reduction=reduction,
    )

    log_probabilities = F.log_softmax(
        logits,
        dim=-1,
    )

    per_class_loss = F.kl_div(
        log_probabilities,
        targets,
        reduction="none",
    )

    per_subject_loss = per_class_loss.sum(
        dim=-1
    )

    numeric_mask = subject_mask.to(
        dtype=per_subject_loss.dtype
    )

    masked_loss = (
        per_subject_loss
        * numeric_mask
    )

    if reduction == "none":
        return masked_loss

    total_loss = masked_loss.sum()

    if reduction == "sum":
        return total_loss

    valid_subject_count = (
        subject_mask.sum()
    )

    if valid_subject_count.item() == 0:
        # Keep the returned zero connected to the computation graph so
        # backward() remains valid for an all-failed collection batch.
        return logits.sum() * 0.0

    return total_loss / valid_subject_count.to(
        dtype=total_loss.dtype
    )


def _validate_distribution_loss_inputs(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    subject_mask: torch.Tensor,
    reduction: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Validate subjective-belief loss inputs."""

    if reduction not in VALID_REDUCTIONS:
        raise ValueError(
            "reduction must be one of "
            f"{sorted(VALID_REDUCTIONS)}, "
            f"got {reduction!r}"
        )

    tensors = {
        "logits": logits,
        "targets": targets,
        "subject_mask": subject_mask,
    }

    for field_name, tensor in tensors.items():
        if not isinstance(
            tensor,
            torch.Tensor,
        ):
            raise TypeError(
                f"{field_name} must be a tensor"
            )

    if logits.ndim != 3:
        raise ValueError("logits must have shape [B, 7, C]")

    batch_size = logits.shape[0]
    class_count = logits.shape[2]

    expected_shape = (
        batch_size,
        NUM_PLAYERS,
        class_count,
    )

    if class_count not in (NUM_PLAYERS, NUM_WOLF_PAIR_CLASSES):
        raise ValueError(
            "logits final dimension must contain 7 suspicion classes "
            "or 21 pair classes"
        )

    if tuple(logits.shape) != expected_shape:
        raise ValueError(f"logits must have shape [B, {NUM_PLAYERS}, C]")

    if tuple(targets.shape) != expected_shape:
        raise ValueError(
            "targets must have the same shape as logits"
        )

    expected_mask_shape = (
        batch_size,
        NUM_PLAYERS,
    )

    if (
        tuple(subject_mask.shape)
        != expected_mask_shape
    ):
        raise ValueError(
            "subject_mask must have shape "
            f"[B, {NUM_PLAYERS}]"
        )

    if batch_size <= 0:
        raise ValueError(
            "batch size must be positive"
        )

    if not torch.is_floating_point(
        logits
    ):
        raise TypeError(
            "logits must use a "
            "floating-point dtype"
        )

    if not torch.is_floating_point(
        targets
    ):
        raise TypeError(
            "targets must use a "
            "floating-point dtype"
        )

    if subject_mask.dtype != torch.bool:
        raise TypeError(
            "subject_mask must use torch.bool"
        )

    if not torch.isfinite(
        logits
    ).all():
        raise ValueError(
            "logits must contain only "
            "finite values"
        )

    if not torch.isfinite(
        targets
    ).all():
        raise ValueError(
            "targets must contain only "
            "finite values"
        )

    if torch.any(
        targets < 0.0
    ):
        raise ValueError(
            "targets cannot contain "
            "negative values"
        )

    targets = targets.to(
        device=logits.device,
        dtype=logits.dtype,
    )

    subject_mask = subject_mask.to(
        device=logits.device,
    )

    row_sums = targets.sum(
        dim=-1
    )

    valid_row_sums = row_sums[
        subject_mask
    ]

    if valid_row_sums.numel() > 0:
        expected_sums = torch.ones_like(
            valid_row_sums
        )

        if not torch.allclose(
            valid_row_sums,
            expected_sums,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError(
                "every supervised target "
                "row must sum to one"
            )

    invalid_rows = targets[
        ~subject_mask
    ]

    if (
        invalid_rows.numel() > 0
        and torch.any(
            invalid_rows != 0.0
        )
    ):
        raise ValueError(
            "unsupervised target rows "
            "must remain all zero"
        )

    return (
        logits,
        targets,
        subject_mask,
    )


__all__ = [
    "VALID_REDUCTIONS",
    "masked_distribution_cross_entropy",
    "masked_distribution_kl_divergence",
]
