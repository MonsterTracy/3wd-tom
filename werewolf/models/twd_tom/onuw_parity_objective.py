"""Unmasked-column belief objective with observer-row-micro training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _validate(
    belief_logits: torch.Tensor,
    belief_targets: torch.Tensor,
    query_valid_mask: torch.Tensor,
    observer_alive_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if belief_logits.ndim != 4 or belief_logits.shape[-2:] != (7, 7):
        raise ValueError("belief_logits must have shape [B, Q, 7, 7]")
    if belief_targets.shape != belief_logits.shape:
        raise ValueError("belief_targets must match belief_logits")
    if query_valid_mask.shape != belief_logits.shape[:2]:
        raise ValueError("query_valid_mask must have shape [B, Q]")
    if observer_alive_mask.shape != (*belief_logits.shape[:2], 7):
        raise ValueError("observer_alive_mask must have shape [B, Q, 7]")
    if query_valid_mask.dtype is not torch.bool:
        raise TypeError("query_valid_mask must use torch.bool")
    if observer_alive_mask.dtype is not torch.bool:
        raise TypeError("observer_alive_mask must use torch.bool")
    if not torch.is_floating_point(belief_logits):
        raise TypeError("belief_logits must be floating point")
    if not torch.is_floating_point(belief_targets):
        raise TypeError("belief_targets must be floating point")
    if not torch.isfinite(belief_logits).all() or not torch.isfinite(
        belief_targets
    ).all():
        raise ValueError("belief tensors must be finite")
    if torch.any(belief_targets < 0):
        raise ValueError("belief_targets cannot be negative")
    valid_rows = query_valid_mask[:, :, None] & observer_alive_mask
    if not valid_rows.any():
        raise ValueError("batch requires at least one valid observer row")
    row_sums = belief_targets.sum(dim=-1)
    if not torch.allclose(
        row_sums[valid_rows],
        torch.ones_like(row_sums[valid_rows]),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("every supervised target row must sum to one")
    if torch.any(belief_targets[~valid_rows] != 0):
        raise ValueError("unsupervised observer rows must remain zero")
    return valid_rows, belief_targets.to(
        device=belief_logits.device, dtype=belief_logits.dtype
    )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denominator = mask.sum()
    if denominator.item() == 0:
        raise ValueError("cannot average an empty mask")
    return (values * mask.to(values.dtype)).sum() / denominator.to(values.dtype)


def onuw_parity_belief_objective(
    belief_logits: torch.Tensor,
    belief_targets: torch.Tensor,
    query_valid_mask: torch.Tensor,
    observer_alive_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return the frozen loss plus row/query/game CE and KL aggregations."""

    valid_rows, targets = _validate(
        belief_logits,
        belief_targets,
        query_valid_mask,
        observer_alive_mask,
    )
    log_probabilities = F.log_softmax(belief_logits, dim=-1)
    row_ce = -(targets * log_probabilities).sum(dim=-1)
    row_entropy = torch.where(
        targets > 0,
        -(targets * torch.log(targets.clamp_min(torch.finfo(targets.dtype).tiny))),
        torch.zeros_like(targets),
    ).sum(dim=-1)
    row_kl = row_ce - row_entropy

    row_micro_ce = _masked_mean(row_ce, valid_rows)
    row_micro_kl = _masked_mean(row_kl, valid_rows)

    rows_per_query = valid_rows.sum(dim=-1)
    valid_queries = query_valid_mask & (rows_per_query > 0)
    query_ce = (row_ce * valid_rows).sum(dim=-1) / rows_per_query.clamp_min(1)
    query_kl = (row_kl * valid_rows).sum(dim=-1) / rows_per_query.clamp_min(1)
    query_macro_ce = _masked_mean(query_ce, valid_queries)
    query_macro_kl = _masked_mean(query_kl, valid_queries)

    rows_per_game = valid_rows.sum(dim=(1, 2))
    valid_games = rows_per_game > 0
    game_ce = (row_ce * valid_rows).sum(dim=(1, 2)) / rows_per_game.clamp_min(1)
    game_kl = (row_kl * valid_rows).sum(dim=(1, 2)) / rows_per_game.clamp_min(1)
    game_macro_ce = _masked_mean(game_ce, valid_games)
    game_macro_kl = _masked_mean(game_kl, valid_games)
    return {
        "loss": row_micro_ce,
        "row_micro_ce": row_micro_ce,
        "row_micro_kl": row_micro_kl,
        "query_macro_ce": query_macro_ce,
        "query_macro_kl": query_macro_kl,
        "game_macro_ce": game_macro_ce,
        "game_macro_kl": game_macro_kl,
        "valid_row_count": valid_rows.sum(),
        "valid_query_count": valid_queries.sum(),
        "valid_game_count": valid_games.sum(),
    }


__all__ = ["onuw_parity_belief_objective"]
