"""Losses for subjective two-Werewolf pair distributions.

The model predicts one distribution over 21 possible Werewolf pairs
for every belief subject:

    pair_logits:  [B, 7, 21]
    pair_targets: [B, 7, 21]
    subject_mask:   [B, 7]

Only rows selected by ``subject_mask`` contribute to the loss.

The targets are global joint pair distributions. They are
not true-role labels and are not independent binary probabilities.
"""

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


def masked_pair_cross_entropy(
    pair_logits: torch.Tensor,
    pair_targets: torch.Tensor,
    subject_mask: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute soft-target cross entropy over supervised observer rows."""

    pair_logits, pair_targets, subject_mask = _validate_pair_loss_inputs(
        pair_logits=pair_logits,
        pair_targets=pair_targets,
        subject_mask=subject_mask,
        reduction=reduction,
    )
    valid_subject_count = subject_mask.sum()
    if valid_subject_count.item() == 0:
        raise ValueError("subject_mask must select at least one valid observer")

    per_subject_loss = -(
        pair_targets * F.log_softmax(pair_logits, dim=-1)
    ).sum(dim=-1)
    masked_loss = per_subject_loss * subject_mask.to(per_subject_loss.dtype)
    if reduction == "none":
        return masked_loss
    total_loss = masked_loss.sum()
    if reduction == "sum":
        return total_loss
    return total_loss / valid_subject_count.to(dtype=total_loss.dtype)


def masked_pair_kl_divergence(
    pair_logits: torch.Tensor,
    pair_targets: torch.Tensor,
    subject_mask: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute KL divergence over valid subjective-belief rows.

    Args:
        pair_logits:
            Floating-point tensor with shape ``[B, 7, 21]``. The final
            dimension contains unnormalized scores over pair classes.

        pair_targets:
            Floating-point tensor with shape ``[B, 7, 21]``.

            Every supervised row must:

            - contain only non-negative values;
            - sum to one.

            Every unsupervised row must remain all zero.

        subject_mask:
            Boolean tensor with shape ``[B, 7]``.

            ``True`` means that the corresponding subject row contains
            a valid subjective target and contributes to the loss.

        reduction:
            ``"none"``:
                Return masked per-subject losses with shape ``[B, 7]``.

            ``"sum"``:
                Sum over all valid subject rows.

            ``"mean"``:
                Average over valid subject rows.

                If a batch contains no valid rows, return a
                differentiable scalar zero.

    Returns:
        A loss tensor determined by ``reduction``.

    Notes:
        KL divergence differs from soft-label cross entropy only by the
        entropy of the fixed target distribution. They therefore have
        the same gradients with respect to model logits.
    """

    (
        pair_logits,
        pair_targets,
        subject_mask,
    ) = _validate_pair_loss_inputs(
        pair_logits=pair_logits,
        pair_targets=pair_targets,
        subject_mask=subject_mask,
        reduction=reduction,
    )

    log_probabilities = F.log_softmax(
        pair_logits,
        dim=-1,
    )

    per_pair_loss = F.kl_div(
        log_probabilities,
        pair_targets,
        reduction="none",
    )

    per_subject_loss = per_pair_loss.sum(
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
        return pair_logits.sum() * 0.0

    return total_loss / valid_subject_count.to(
        dtype=total_loss.dtype
    )


def _validate_pair_loss_inputs(
    *,
    pair_logits: torch.Tensor,
    pair_targets: torch.Tensor,
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
        "pair_logits": pair_logits,
        "pair_targets": pair_targets,
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

    if pair_logits.ndim != 3:
        raise ValueError(
            "pair_logits must have shape [B, 7, 21]"
        )

    batch_size = pair_logits.shape[0]

    expected_pair_shape = (
        batch_size,
        NUM_PLAYERS,
        NUM_WOLF_PAIR_CLASSES,
    )

    if (
        tuple(pair_logits.shape)
        != expected_pair_shape
    ):
        raise ValueError(
            "pair_logits must have shape "
            f"[B, {NUM_PLAYERS}, {NUM_WOLF_PAIR_CLASSES}]"
        )

    if (
        tuple(pair_targets.shape)
        != expected_pair_shape
    ):
        raise ValueError(
            "pair_targets must have the same "
            "shape as pair_logits"
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
        pair_logits
    ):
        raise TypeError(
            "pair_logits must use a "
            "floating-point dtype"
        )

    if not torch.is_floating_point(
        pair_targets
    ):
        raise TypeError(
            "pair_targets must use a "
            "floating-point dtype"
        )

    if subject_mask.dtype != torch.bool:
        raise TypeError(
            "subject_mask must use torch.bool"
        )

    if not torch.isfinite(
        pair_logits
    ).all():
        raise ValueError(
            "pair_logits must contain only "
            "finite values"
        )

    if not torch.isfinite(
        pair_targets
    ).all():
        raise ValueError(
            "pair_targets must contain only "
            "finite values"
        )

    if torch.any(
        pair_targets < 0.0
    ):
        raise ValueError(
            "pair_targets cannot contain "
            "negative values"
        )

    pair_targets = pair_targets.to(
        device=pair_logits.device,
        dtype=pair_logits.dtype,
    )

    subject_mask = subject_mask.to(
        device=pair_logits.device,
    )

    row_sums = pair_targets.sum(
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
                "every supervised pair target "
                "row must sum to one"
            )

    invalid_rows = pair_targets[
        ~subject_mask
    ]

    if (
        invalid_rows.numel() > 0
        and torch.any(
            invalid_rows != 0.0
        )
    ):
        raise ValueError(
            "unsupervised pair target rows "
            "must remain all zero"
        )

    return (
        pair_logits,
        pair_targets,
        subject_mask,
    )


__all__ = [
    "VALID_REDUCTIONS",
    "masked_pair_cross_entropy",
    "masked_pair_kl_divergence",
]
