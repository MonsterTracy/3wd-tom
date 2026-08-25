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


def _masked_sum(values: torch.Tensor, mask: torch.Tensor) -> float:
    return float(values[mask].sum().item())


@torch.no_grad()
def compute_belief_metrics(
    belief_logits: torch.Tensor,
    belief_targets: torch.Tensor,
    observer_alive_mask: torch.Tensor,
    diagonal_target_mask: torch.Tensor,
    *,
    observer_supervision_mask: torch.Tensor | None = None,
    observer_scope_mask: torch.Tensor | None = None,
    label_observed_mask: torch.Tensor | None = None,
    known_non_werewolf_mask: torch.Tensor | None = None,
) -> dict[str, int | float]:
    """Measure predictions for the selected subset of valid living observers."""

    per_observer_loss = masked_belief_distribution_loss(
        belief_logits,
        belief_targets,
        observer_alive_mask,
        diagonal_target_mask,
        observer_supervision_mask=observer_supervision_mask,
        reduction="none",
    )
    alive_observers = observer_alive_mask.to(
        device=belief_logits.device,
        dtype=torch.bool,
    )
    if observer_supervision_mask is None:
        valid_observers = alive_observers
    else:
        if not isinstance(observer_supervision_mask, torch.Tensor):
            raise TypeError("observer_supervision_mask must be a tensor")
        if observer_supervision_mask.shape != alive_observers.shape:
            raise ValueError("observer_supervision_mask must match alive rows")
        if observer_supervision_mask.dtype is not torch.bool:
            raise TypeError("observer_supervision_mask must use torch.bool")
        valid_observers = observer_supervision_mask.to(device=belief_logits.device)
        if torch.any(valid_observers & ~alive_observers):
            raise ValueError("observer supervision must be a subset of alive rows")
    if not torch.any(valid_observers):
        raise ValueError("metrics require at least one supervised observer")
    if (observer_scope_mask is None) != (label_observed_mask is None):
        raise ValueError(
            "observer_scope_mask and label_observed_mask must be supplied together"
        )
    if observer_scope_mask is None:
        scope_observers = valid_observers
        observed_labels = valid_observers
    else:
        for field_name, mask in {
            "observer_scope_mask": observer_scope_mask,
            "label_observed_mask": label_observed_mask,
        }.items():
            if not isinstance(mask, torch.Tensor):
                raise TypeError(f"{field_name} must be a tensor")
            if mask.shape != alive_observers.shape:
                raise ValueError(f"{field_name} must match alive rows")
            if mask.dtype is not torch.bool:
                raise TypeError(f"{field_name} must use torch.bool")
        scope_observers = observer_scope_mask.to(device=belief_logits.device)
        observed_labels = label_observed_mask.to(device=belief_logits.device)
        if torch.any(scope_observers & ~alive_observers):
            raise ValueError("observer scope must be a subset of alive rows")
        if torch.any(observed_labels & ~alive_observers):
            raise ValueError("observed labels must be a subset of alive rows")
        if not torch.equal(valid_observers, scope_observers & observed_labels):
            raise ValueError(
                "observer supervision must equal scope & label_observed"
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
    max_probability_error = (probabilities - targets).abs().max(dim=-1).values
    absolute_error = torch.where(
        target_mask,
        (probabilities - targets).abs(),
        torch.zeros_like(probabilities),
    ).sum(dim=-1) / target_mask.sum(dim=-1).clamp_min(1)
    top_probability = probabilities.max(dim=-1, keepdim=True).values
    predicted_top_support = probabilities == top_probability
    target_support = targets > 0
    max_set_support_hit = (
        predicted_top_support & target_support & target_mask
    ).any(dim=-1).to(dtype=belief_logits.dtype)
    deterministic_top_index = probabilities.argmax(dim=-1, keepdim=True)
    deterministic_top1_support_hit = target_support.gather(
        dim=-1,
        index=deterministic_top_index,
    ).squeeze(-1).to(dtype=belief_logits.dtype)

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
    uniform_kl = (uniform_cross_entropy - target_entropy).clamp_min(0.0)
    zero_gap = uniform_kl <= 1e-8
    model_kl_sum = _masked_sum(kl_divergence, valid_observers)
    uniform_kl_sum = _masked_sum(uniform_kl, valid_observers)

    result = {
        "total_row_count": int(valid_observers.sum().item()),
        "valid_observer_count": int(valid_observers.sum().item()),
        "scope_observer_count": int(scope_observers.sum().item()),
        "observed_label_row_count_in_scope": int(
            (scope_observers & observed_labels).sum().item()
        ),
        "unobserved_label_row_count_in_scope": int(
            (scope_observers & ~observed_labels).sum().item()
        ),
        "positive_uniform_baseline_gap_row_count": int(
            (valid_observers & ~zero_gap).sum().item()
        ),
        "zero_uniform_baseline_gap_row_count": int(
            (valid_observers & zero_gap).sum().item()
        ),
        "model_kl_sum": model_kl_sum,
        "uniform_non_self_baseline_kl_sum": uniform_kl_sum,
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
        "mean_belief_max_probability_error": _masked_mean(
            max_probability_error,
            valid_observers,
        ),
        "max_belief_probability_error": float(
            max_probability_error[valid_observers].max().item()
        ),
        "mean_belief_max_set_support_hit": _masked_mean(
            max_set_support_hit,
            valid_observers,
        ),
        "mean_belief_deterministic_top1_support_hit": _masked_mean(
            deterministic_top1_support_hit,
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
        "uniform_non_self_baseline_mean_kl_divergence": (
            uniform_kl_sum / int(valid_observers.sum().item())
        ),
        "normalized_reducible_gap_improvement": (
            1.0 - model_kl_sum / uniform_kl_sum
            if uniform_kl_sum > 0.0
            else 0.0
        ),
    }
    if known_non_werewolf_mask is not None:
        if not isinstance(known_non_werewolf_mask, torch.Tensor):
            raise TypeError("known_non_werewolf_mask must be a tensor")
        if known_non_werewolf_mask.shape != target_mask.shape:
            raise ValueError(
                "known_non_werewolf_mask must match diagonal_target_mask"
            )
        if known_non_werewolf_mask.dtype is not torch.bool:
            raise TypeError("known_non_werewolf_mask must use torch.bool")
        known_non_wolves = known_non_werewolf_mask.to(device=belief_logits.device)
        admissible = target_mask & ~known_non_wolves
        if torch.any(valid_observers & (admissible.sum(dim=-1) == 0)):
            raise ValueError(
                "every living observer requires a private-admissible target"
            )
        if torch.any((targets > 0) & ~admissible & valid_observers.unsqueeze(-1)):
            raise ValueError(
                "belief targets cannot support known non-Werewolves"
            )
        private_uniform = admissible.to(dtype=belief_logits.dtype)
        private_uniform /= admissible.sum(dim=-1, keepdim=True).clamp_min(1)
        private_cross_entropy = -(
            targets
            * private_uniform.clamp_min(
                torch.finfo(belief_logits.dtype).tiny
            ).log()
        ).sum(dim=-1)
        private_total_variation = 0.5 * (
            private_uniform - targets
        ).abs().sum(dim=-1)
        private_absolute_error = torch.where(
            target_mask,
            (private_uniform - targets).abs(),
            torch.zeros_like(private_uniform),
        ).sum(dim=-1) / target_mask.sum(dim=-1).clamp_min(1)
        private_kl = (private_cross_entropy - target_entropy).clamp_min(0.0)
        private_zero_gap = private_kl <= 1e-8
        private_kl_sum = _masked_sum(private_kl, valid_observers)
        illegal_mass = (probabilities * known_non_wolves).sum(dim=-1)
        result.update({
            "positive_private_admissible_baseline_gap_row_count": int(
                (valid_observers & ~private_zero_gap).sum().item()
            ),
            "zero_private_admissible_baseline_gap_row_count": int(
                (valid_observers & private_zero_gap).sum().item()
            ),
            "private_admissible_uniform_baseline_kl_sum": private_kl_sum,
            "mean_illegal_known_nonwolf_mass": _masked_mean(
                illegal_mass, valid_observers
            ),
            "private_admissible_uniform_baseline_mean_cross_entropy": _masked_mean(
                private_cross_entropy, valid_observers
            ),
            "private_admissible_uniform_baseline_mean_total_variation": _masked_mean(
                private_total_variation, valid_observers
            ),
            "private_admissible_uniform_baseline_mean_absolute_error": _masked_mean(
                private_absolute_error, valid_observers
            ),
            "private_admissible_uniform_baseline_mean_kl_divergence": (
                private_kl_sum / int(valid_observers.sum().item())
            ),
            "private_admissible_normalized_reducible_gap_improvement": (
                1.0 - model_kl_sum / private_kl_sum
                if private_kl_sum > 0.0
                else 0.0
            ),
        })
    return result


__all__ = ["compute_belief_metrics"]
