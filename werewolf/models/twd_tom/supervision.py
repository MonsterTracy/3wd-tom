"""Supervision-only observer role metadata and row-selection masks."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from werewolf.models.twd_tom.schema import NUM_PLAYERS, PLAYER_NAMES, PLAYER_TO_ID
from werewolf.trajectory import canonical_digest


ROLE_SIDECAR_SCHEMA_VERSION = "classic7_tom_v2_role_sidecar_v1"
ROLE_NAMES = ("Werewolf", "Villager", "Seer", "Witch")
ROLE_COUNTS = {"Werewolf": 2, "Villager": 3, "Seer": 1, "Witch": 1}

ALL_ALIVE_SCOPE = "all_alive"
NON_WOLF_ALIVE_SCOPE = "non_wolf_alive"
VILLAGER_ALIVE_SCOPE = "villager_alive"
SPEAKER_ALIVE_SCOPE = "speaker_alive"
SUPERVISION_SCOPES = (
    ALL_ALIVE_SCOPE,
    NON_WOLF_ALIVE_SCOPE,
    VILLAGER_ALIVE_SCOPE,
    SPEAKER_ALIVE_SCOPE,
)


def _require_lower_hex_digest(value: Any, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def normalize_observer_roles(value: Any) -> dict[str, str]:
    """Validate one complete Classic-7 player-to-role mapping."""

    if not isinstance(value, Mapping) or set(value) != set(PLAYER_NAMES):
        raise ValueError("observer roles must map every canonical player exactly once")
    roles: dict[str, str] = {}
    for player in PLAYER_NAMES:
        role = value[player]
        if role not in ROLE_NAMES:
            raise ValueError(f"unsupported Classic-7 role for {player}: {role!r}")
        roles[player] = role
    counts = {role: sum(value == role for value in roles.values()) for role in ROLE_NAMES}
    if counts != ROLE_COUNTS:
        raise ValueError(f"observer role counts must equal {ROLE_COUNTS}, got {counts}")
    return roles


def rotate_observer_roles(value: Any, *, shift: int) -> dict[str, str]:
    """Apply the same cyclic seat rotation used by training augmentation."""

    roles = normalize_observer_roles(value)
    if isinstance(shift, bool) or not isinstance(shift, int) or not 0 <= shift < NUM_PLAYERS:
        raise ValueError(f"shift must be an integer in [0, {NUM_PLAYERS - 1}]")
    return {
        PLAYER_NAMES[(PLAYER_TO_ID[player] - 1 + shift) % NUM_PLAYERS]: role
        for player, role in roles.items()
    }


def build_observer_supervision_mask(
    *,
    alive_mask: torch.Tensor,
    observer_roles: Any | None,
    speaker_id: int | Sequence[int],
    scope: str,
) -> torch.Tensor:
    """Return ``alive_mask & scope_mask`` without modifying targets."""

    if not isinstance(alive_mask, torch.Tensor) or alive_mask.dtype is not torch.bool:
        raise TypeError("alive_mask must be a torch.bool tensor")
    if alive_mask.ndim not in {1, 2} or alive_mask.shape[-1] != NUM_PLAYERS:
        raise ValueError("alive_mask must have shape [7] or [Q, 7]")
    if scope not in SUPERVISION_SCOPES:
        raise ValueError(f"scope must be one of {SUPERVISION_SCOPES}")

    if alive_mask.ndim == 1:
        speakers = [speaker_id]
    else:
        if isinstance(speaker_id, (str, bytes)) or not isinstance(speaker_id, Sequence):
            raise TypeError("dense speaker_id must be a sequence")
        speakers = list(speaker_id)
        if len(speakers) != alive_mask.shape[0]:
            raise ValueError("dense speaker_id must match the boundary count")
    if any(
        isinstance(speaker, bool)
        or not isinstance(speaker, int)
        or not 1 <= speaker <= NUM_PLAYERS
        for speaker in speakers
    ):
        raise ValueError("speaker_id must contain integers in [1, 7]")

    if scope in {NON_WOLF_ALIVE_SCOPE, VILLAGER_ALIVE_SCOPE}:
        roles = normalize_observer_roles(observer_roles)
    else:
        roles = None

    if scope == ALL_ALIVE_SCOPE:
        scope_mask = torch.ones_like(alive_mask)
    elif scope == NON_WOLF_ALIVE_SCOPE:
        assert roles is not None
        row = torch.tensor(
            [roles[player] != "Werewolf" for player in PLAYER_NAMES],
            dtype=torch.bool,
            device=alive_mask.device,
        )
        scope_mask = row if alive_mask.ndim == 1 else row.expand_as(alive_mask)
    elif scope == VILLAGER_ALIVE_SCOPE:
        assert roles is not None
        row = torch.tensor(
            [roles[player] == "Villager" for player in PLAYER_NAMES],
            dtype=torch.bool,
            device=alive_mask.device,
        )
        scope_mask = row if alive_mask.ndim == 1 else row.expand_as(alive_mask)
    else:
        scope_mask = torch.zeros_like(alive_mask)
        if alive_mask.ndim == 1:
            scope_mask[speakers[0] - 1] = True
        else:
            boundary = torch.arange(alive_mask.shape[0], device=alive_mask.device)
            speaker_indices = torch.tensor(
                [speaker - 1 for speaker in speakers],
                dtype=torch.long,
                device=alive_mask.device,
            )
            scope_mask[boundary, speaker_indices] = True

    return alive_mask & scope_mask


def load_role_sidecar_report(path: str | Path) -> dict[str, Any]:
    """Load and validate one immutable, digest-bound role sidecar."""

    sidecar_path = Path(path)
    value = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("role sidecar must contain one JSON object")
    required = {
        "schema_version",
        "canonical_batch_summary_digest",
        "canonical_batch_summary_sha256",
        "split_manifest_digest",
        "games",
        "sidecar_digest",
    }
    if set(value) != required:
        raise ValueError("role sidecar field set mismatch")
    if value["schema_version"] != ROLE_SIDECAR_SCHEMA_VERSION:
        raise ValueError("role sidecar schema version mismatch")
    for field_name in (
        "canonical_batch_summary_digest",
        "canonical_batch_summary_sha256",
        "split_manifest_digest",
        "sidecar_digest",
    ):
        _require_lower_hex_digest(value[field_name], field_name=field_name)
    payload = dict(value)
    recorded_digest = payload.pop("sidecar_digest")
    if recorded_digest != canonical_digest(payload):
        raise ValueError("role sidecar digest mismatch")
    games = value["games"]
    if not isinstance(games, Mapping) or not games:
        raise ValueError("role sidecar games must be a non-empty mapping")
    normalized_games: dict[str, dict[str, Any]] = {}
    for game_id, game in games.items():
        if not isinstance(game_id, str) or not game_id.strip():
            raise ValueError("role sidecar game IDs must be non-empty text")
        if not isinstance(game, Mapping) or set(game) != {
            "game_summary_digest",
            "trajectory_digest",
            "trajectory_sha256",
            "observer_roles",
        }:
            raise ValueError(f"role sidecar game contract mismatch: {game_id}")
        for field_name in (
            "game_summary_digest",
            "trajectory_digest",
            "trajectory_sha256",
        ):
            _require_lower_hex_digest(
                game[field_name],
                field_name=f"{game_id}.{field_name}",
            )
        normalized_games[game_id] = {
            **dict(game),
            "observer_roles": normalize_observer_roles(game["observer_roles"]),
        }
    return {**value, "games": normalized_games}


def load_role_sidecar(path: str | Path) -> dict[str, dict[str, str]]:
    """Load a sidecar as ``game_id -> player -> role`` for supervision only."""

    report = load_role_sidecar_report(path)
    return {
        game_id: dict(game["observer_roles"])
        for game_id, game in report["games"].items()
    }


__all__ = [
    "ALL_ALIVE_SCOPE",
    "NON_WOLF_ALIVE_SCOPE",
    "ROLE_COUNTS",
    "ROLE_NAMES",
    "ROLE_SIDECAR_SCHEMA_VERSION",
    "SPEAKER_ALIVE_SCOPE",
    "SUPERVISION_SCOPES",
    "VILLAGER_ALIVE_SCOPE",
    "build_observer_supervision_mask",
    "load_role_sidecar",
    "load_role_sidecar_report",
    "normalize_observer_roles",
    "rotate_observer_roles",
]
