"""Deterministic belief-label conversions used by tom-v2 and frozen models."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import torch

from werewolf.models.twd_tom.schema import (
    NUM_PLAYERS,
    PLAYER_NAMES,
    PLAYER_TO_ID,
    canonicalize_player_set,
    normalize_player,
    validate_player_suspicion,
)


def suspicion_set_to_belief_vector(
    suspected_werewolves: Any,
    *,
    observer_id: Any,
    known_werewolves: Any,
    known_non_werewolves: Any,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert a legal suspicion set to one sparse seven-player row."""

    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError("dtype must be a floating-point torch dtype")
    observer = normalize_player(observer_id)
    closed_wolves, closed_non_wolves = close_hard_knowledge(
        known_werewolves,
        known_non_werewolves,
    )
    suspected = validate_player_suspicion(
        suspected_werewolves,
        closed_wolves,
        closed_non_wolves,
        observer_id=observer,
    )
    target = torch.zeros(NUM_PLAYERS, dtype=dtype, device=device)
    if suspected:
        for player in suspected:
            target[PLAYER_TO_ID[player] - 1] = 1.0
    else:
        forbidden = set(closed_non_wolves) | {observer}
        for player in PLAYER_NAMES:
            if player not in forbidden:
                target[PLAYER_TO_ID[player] - 1] = 1.0
    if target.sum().item() == 0.0:
        raise RuntimeError("hard knowledge leaves no admissible target player")
    target /= target.sum()

    if not torch.isfinite(target).all() or torch.any(target < 0.0):
        raise RuntimeError("constructed belief target must be finite and non-negative")
    if target[PLAYER_TO_ID[observer] - 1].item() != 0.0:
        raise RuntimeError("constructed belief target diagonal must be zero")
    if not torch.isclose(
        target.sum(),
        torch.ones((), dtype=dtype, device=device),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise RuntimeError("constructed belief target must sum to one")
    return target


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
        for pair in combinations(PLAYER_NAMES, 2)
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
__all__ = [
    "suspicion_set_to_belief_vector",
    "canonicalize_known_players",
    "close_hard_knowledge",
]
