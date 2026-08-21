"""Deterministic hard knowledge from one player's legal observation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from werewolf.models.twd_tom.schema import PLAYER_NAMES, normalize_player


def _log_payload(log: Any) -> dict[str, Any]:
    fields = ("event", "source", "target", "content", "day", "time")
    if isinstance(log, Mapping):
        if set(log) != {*fields, "viewer"}:
            raise TypeError(
                "serialized observation game_log fields do not match contract"
            )
        return {field: deepcopy(log[field]) for field in fields}
    if any(not hasattr(log, field) for field in fields):
        raise TypeError(
            "observation game_log entries must be Log objects or mappings"
        )
    return {field: deepcopy(getattr(log, field)) for field in fields}


def legal_observer_state(
    observer_id: int | str,
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Select the public and private state legally visible to one observer."""

    observer = normalize_player(observer_id)
    if not isinstance(observation, Mapping):
        raise TypeError("legal observation must be a mapping")
    if normalize_player(observation.get("observer_id")) != observer:
        raise ValueError("legal observation belongs to another observer")
    identity = observation.get("identity")
    if not isinstance(identity, str) or not identity:
        raise ValueError("legal observation requires observer identity")
    game_log = observation.get("game_log")
    if isinstance(game_log, (str, bytes)) or not isinstance(game_log, Sequence):
        raise TypeError("legal observation requires a game_log sequence")
    public_state = observation.get("authoritative_public_state")
    if not isinstance(public_state, Mapping):
        raise TypeError("legal observation requires authoritative_public_state")
    return {
        "observer_id": observer,
        "self_role": identity,
        "current_phase": observation.get("phase"),
        "current_public_actor": observation.get("current_act_idx"),
        "game_log": [_log_payload(log) for log in game_log],
        "authoritative_public_state": deepcopy(dict(public_state)),
    }


def derive_observer_hard_knowledge(
    observer_id: int | str,
    observation: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Derive exact role constraints from one legal observation."""

    state = legal_observer_state(observer_id, observation)
    observer = state["observer_id"]
    self_role = state["self_role"]
    known_werewolves = set()
    known_non_werewolves = set()

    if self_role == "Werewolf":
        known_werewolves.add(observer)
    else:
        known_non_werewolves.add(observer)

    wolf_team = set()
    public_wolf_count = None
    for log in state["game_log"]:
        event = log["event"]
        content = log["content"]
        if event == "game_setting" and isinstance(content, Mapping):
            count = content.get("Werewolf")
            if type(count) is int and count >= 0:
                public_wolf_count = count
        elif event == "werewolf_team_info" and self_role == "Werewolf":
            if not isinstance(content, Mapping):
                raise ValueError("wolf-team information must be a mapping")
            team = content.get("wolf_team")
            if isinstance(team, (str, bytes)) or not isinstance(team, Sequence):
                raise ValueError("wolf-team information must contain a team list")
            wolf_team.update(normalize_player(player) for player in team)
        elif event == "skill_seer" and self_role == "Seer":
            if not isinstance(content, Mapping):
                raise ValueError("Seer check information must be a mapping")
            checked = content.get("cheked_identity")
            if checked in {"bad", "werewolf"}:
                known_werewolves.add(normalize_player(log["target"]))
            elif checked in {"good", "non-werewolf"}:
                known_non_werewolves.add(normalize_player(log["target"]))
            elif checked is not None:
                raise ValueError("unsupported Seer check result")
        elif event == "kill_decision" and self_role == "Witch":
            target = log["target"]
            if type(target) is int and 1 <= target <= 7:
                known_non_werewolves.add(normalize_player(target))

    known_werewolves.update(wolf_team)
    if wolf_team and len(wolf_team) == public_wolf_count:
        known_non_werewolves.update(set(PLAYER_NAMES) - wolf_team)

    conflict = known_werewolves & known_non_werewolves
    if conflict:
        raise ValueError("conflicting observer hard knowledge")
    unknown_players = set(PLAYER_NAMES) - known_werewolves - known_non_werewolves
    return {
        "known_werewolves": sorted(known_werewolves, key=PLAYER_NAMES.index),
        "known_non_werewolves": sorted(
            known_non_werewolves,
            key=PLAYER_NAMES.index,
        ),
        "unknown_players": sorted(unknown_players, key=PLAYER_NAMES.index),
    }


__all__ = ["derive_observer_hard_knowledge", "legal_observer_state"]
