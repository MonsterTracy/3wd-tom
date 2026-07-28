"""Build global two-Werewolf pair targets and player marginals."""

from __future__ import annotations

from typing import Any

import torch

from werewolf.models.twd_tom.schema import (
    NUM_WOLF_PAIR_CLASSES,
    NUM_PLAYERS,
    PLAYER_NAMES,
    PLAYER_TO_ID,
    canonicalize_player_set,
    canonical_wolf_pairs,
    validate_player_suspicion,
)


def canonicalize_known_players(values: Any, *, field_name: str) -> list[str]:
    """Validate and canonicalize one environment-derived knowledge set."""

    return canonicalize_player_set(values, field_name=field_name)


def close_hard_knowledge(
    known_werewolves: Any,
    known_non_werewolves: Any,
) -> tuple[list[str], list[str]]:
    """Apply only the deterministic consequence of exactly two Werewolves."""

    known_wolves = canonicalize_known_players(
        known_werewolves,
        field_name="known_werewolves",
    )
    known_non_wolves = canonicalize_known_players(
        known_non_werewolves,
        field_name="known_non_werewolves",
    )
    if set(known_wolves) & set(known_non_wolves):
        raise ValueError("hard knowledge sets must be disjoint")
    hard_support = [
        pair
        for pair in canonical_wolf_pairs()
        if set(known_wolves).issubset(pair)
        and set(pair).isdisjoint(known_non_wolves)
    ]
    if not hard_support:
        raise ValueError("hard knowledge has no legal two-Werewolf pair")
    closed_wolves = [
        player
        for player in PLAYER_NAMES
        if all(player in pair for pair in hard_support)
    ]
    closed_non_wolves = [
        player
        for player in PLAYER_NAMES
        if all(player not in pair for pair in hard_support)
    ]
    return closed_wolves, closed_non_wolves


def suspicion_set_to_pair_target(
    suspected_werewolves: Any,
    known_werewolves: Any,
    known_non_werewolves: Any,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Project one soft suspicion set over the global 21 pairs."""

    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError("dtype must be a floating-point torch dtype")
    known_wolves, known_non_wolves = close_hard_knowledge(
        known_werewolves,
        known_non_werewolves,
    )
    suspected = validate_player_suspicion(
        suspected_werewolves,
        known_wolves,
        known_non_wolves,
    )
    known_wolf_set = set(known_wolves)
    known_non_wolf_set = set(known_non_wolves)
    soft_suspects = set(suspected) - known_wolf_set
    pairs = canonical_wolf_pairs()
    target = torch.zeros(
        NUM_WOLF_PAIR_CLASSES,
        dtype=dtype,
        device=device,
    )
    for pair_index, pair in enumerate(pairs):
        if (
            known_wolf_set.issubset(pair)
            and set(pair).isdisjoint(known_non_wolf_set)
        ):
            target[pair_index] = 2 ** len(set(pair) & soft_suspects)
    target /= target.sum()

    if not torch.isfinite(target).all() or torch.any(target < 0.0):
        raise RuntimeError("constructed pair target must be finite and non-negative")
    if not torch.isclose(
        target.sum(),
        torch.ones((), dtype=dtype, device=device),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise RuntimeError("constructed pair target must sum to one")
    return target


def pair_probabilities_to_belief_marginals(
    pair_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Convert global pair probabilities to seven player marginals."""

    if not isinstance(pair_probabilities, torch.Tensor):
        raise TypeError("pair_probabilities must be a tensor")
    if not torch.is_floating_point(pair_probabilities):
        raise TypeError("pair_probabilities must use a floating-point dtype")
    if pair_probabilities.ndim < 2 or tuple(
        pair_probabilities.shape[-2:]
    ) != (NUM_PLAYERS, NUM_WOLF_PAIR_CLASSES):
        raise ValueError("pair_probabilities must end with shape [7, 21]")
    if not torch.isfinite(pair_probabilities).all():
        raise ValueError("pair_probabilities must contain only finite values")
    if torch.any(pair_probabilities < 0.0):
        raise ValueError("pair_probabilities cannot contain negative values")

    row_sums = pair_probabilities.sum(dim=-1)
    valid_or_zero = torch.isclose(
        row_sums,
        torch.ones_like(row_sums),
        rtol=1e-5,
        atol=1e-6,
    ) | (row_sums == 0.0)
    if not valid_or_zero.all():
        raise ValueError("pair probability rows must sum to one or remain zero")

    incidence = torch.zeros(
        (NUM_WOLF_PAIR_CLASSES, NUM_PLAYERS),
        dtype=pair_probabilities.dtype,
        device=pair_probabilities.device,
    )
    for pair_index, pair in enumerate(canonical_wolf_pairs()):
        for player in pair:
            incidence[pair_index, PLAYER_TO_ID[player] - 1] = 1.0

    return torch.einsum("...oc,cp->...op", pair_probabilities, incidence)


__all__ = [
    "suspicion_set_to_pair_target",
    "canonicalize_known_players",
    "close_hard_knowledge",
    "pair_probabilities_to_belief_marginals",
]
