"""Archived projection of the public ledger into formal ToM input."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from archive.legacy_tom.werewolf.models.tom.schema import (
    SpeechAction,
    normalize_episode_context,
    normalize_player,
)


_PHASES = {
    "day_speech": "discussion",
    "day_speech_pk": "pk_discussion",
    "day_vote": "vote",
    "day_vote_pk": "pk_vote",
    "night_skill_wolf": "night",
}
_PHASE_PATTERN = re.compile(
    r"(?P<round>[0-9]+)_(?P<time>day|night)_"
    r"(?P<phase>skill_wolf|speech|speech_pk|vote|vote_pk)"
)
_LEDGER_EVENT_TYPES = {
    "phase_change",
    "turn_start",
    "public_speech",
    "vote_result",
    "exile_result",
    "death_announcement",
}


def _sequence(value: Any, *, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field} must be a sequence")
    return value


def _ledger_events(value: Any) -> list[Mapping[str, Any]]:
    events = _sequence(value, field="public_events")
    normalized = []
    for expected_idx, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise TypeError("public ledger events must be mappings")
        if event.get("event_idx") != expected_idx:
            raise ValueError("public ledger event_idx must be continuous")
        if event.get("event_type") not in _LEDGER_EVENT_TYPES:
            raise ValueError(f"unsupported public ledger event: {event.get('event_type')!r}")
        normalized.append(event)
    return normalized


def _round_and_phase(value: Any) -> tuple[int, str]:
    if not isinstance(value, str):
        raise TypeError("public phase must be text")
    match = _PHASE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"unsupported public phase: {value!r}")
    ledger_phase = f"{match.group('time')}_{match.group('phase')}"
    phase = _PHASES[ledger_phase]
    round_number = int(match.group("round"))
    return (round_number + 1 if phase == "night" else round_number), phase


def build_model_input(
    *,
    episode_context: str,
    public_events: Any,
) -> dict[str, Any]:
    """Return static context plus the four permitted public evidence types."""

    context = normalize_episode_context(episode_context)
    evidence: list[dict[str, Any]] = []
    current_round: int | None = None
    current_phase: str | None = None

    for event in _ledger_events(public_events):
        event_type = event["event_type"]
        if event_type == "phase_change":
            current_round, current_phase = _round_and_phase(event.get("phase"))
            continue
        if event_type == "turn_start":
            continue

        if event_type == "death_announcement":
            if current_phase is None:
                current_round = 1
            elif current_phase != "night":
                raise ValueError("night result must follow a night phase")
            evidence.append(
                {
                    "type": "night_result",
                    "dead_players": [
                        normalize_player(player)
                        for player in _sequence(
                            event.get("dead_players"), field="dead_players"
                        )
                    ],
                    "round": current_round,
                    "phase": "night",
                }
            )
            continue

        if current_round is None or current_phase is None:
            raise ValueError("public evidence has no preceding round and phase")

        if event_type == "public_speech":
            if current_phase not in {"discussion", "pk_discussion"}:
                raise ValueError("public speech must occur in a discussion phase")
            for raw_action in _sequence(
                event.get("sp_actions"), field="sp_actions"
            ):
                if len(raw_action) != 3:
                    raise ValueError("speech action must contain three fields")
                action = SpeechAction.from_values(
                    subject=raw_action[0],
                    action=raw_action[1],
                    object_=raw_action[2],
                )
                evidence.append(
                    {
                        "type": "speech_action",
                        "subject": action.subject,
                        "predicate": action.action,
                        "object": action.object,
                        "round": current_round,
                        "phase": current_phase,
                    }
                )
        elif event_type == "vote_result":
            if current_phase not in {"vote", "pk_vote"}:
                raise ValueError("vote result must occur in a vote phase")
            for vote in _sequence(event.get("votes"), field="votes"):
                if not isinstance(vote, Mapping):
                    raise TypeError("public votes must be mappings")
                target = vote.get("target")
                evidence.append(
                    {
                        "type": "vote",
                        "voter": normalize_player(vote.get("voter")),
                        "target": None if target is None else normalize_player(target),
                        "round": current_round,
                        "phase": current_phase,
                    }
                )
        elif event_type == "exile_result":
            if current_phase not in {"vote", "pk_vote"}:
                raise ValueError("exile result must occur in a vote phase")
            exiled_players = [
                normalize_player(player)
                for player in _sequence(
                    event.get("exiled_players"), field="exiled_players"
                )
            ]
            if len(exiled_players) > 1:
                raise ValueError("formal ToM input supports one exile result")
            evidence.append(
                {
                    "type": "exile",
                    "player": (
                        exiled_players[0]
                        if exiled_players
                        else None
                    ),
                    "round": current_round,
                    "phase": current_phase,
                }
            )

    return {"episode_context": context, "events": evidence}


__all__ = ["build_model_input"]
