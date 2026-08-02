"""Canonical public-event history for classic-seven pre-speech ToM."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from werewolf.models.twd_tom.schema import (
    PLAYER_NAMES,
    PLAYER_TO_ID,
    parse_speech_action,
)


PUBLIC_EVENT_SCHEMA_VERSION = "classic7_public_event_sequence_v1"
PUBLIC_EVENT_TYPES = frozenset(
    {
        "phase_change",
        "turn_start",
        "public_speech",
        "vote_result",
        "exile_result",
        "death_announcement",
    }
)
PUBLIC_PHASES = frozenset(
    {"skill_wolf", "speech", "speech_pk", "vote", "vote_pk"}
)
STRUCTURED_TOKEN_TYPES = (
    "phase_change",
    "turn_start",
    "public_speech",
    "speech_action",
    "vote_result",
    "vote",
    "exile_result",
    "exiled_player",
    "death_announcement",
    "dead_player",
)
STRUCTURED_TOKEN_TO_ID = {
    token_type: index
    for index, token_type in enumerate(STRUCTURED_TOKEN_TYPES, start=1)
}
PHASE_TO_ID = {
    phase: index
    for index, phase in enumerate(
        (
            "day_speech",
            "day_speech_pk",
            "day_vote",
            "day_vote_pk",
            "night_skill_wolf",
        ),
        start=1,
    )
}

_EVENT_FIELDS = {
    "phase_change": frozenset({"event_idx", "event_type", "phase"}),
    "turn_start": frozenset({"event_idx", "event_type", "speaker"}),
    "public_speech": frozenset(
        {"event_idx", "event_type", "speaker", "raw_text", "sp_actions"}
    ),
    "vote_result": frozenset({"event_idx", "event_type", "votes"}),
    "exile_result": frozenset(
        {"event_idx", "event_type", "exiled_players"}
    ),
    "death_announcement": frozenset(
        {"event_idx", "event_type", "dead_players"}
    ),
}
_PHASE_PATTERN = re.compile(
    r"(?P<day>[0-9]+)_(?P<time>day|night)_"
    r"(?P<phase>skill_wolf|speech|speech_pk|vote|vote_pk)"
)


def _sequence(value: Any, *, field_name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence")
    return value


def _canonical_player(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or value not in PLAYER_NAMES:
        raise ValueError(
            f"{field_name} must use a canonical player1...player7 ID"
        )
    return value


def _canonical_players(value: Any, *, field_name: str) -> list[str]:
    players = [
        _canonical_player(item, field_name=field_name)
        for item in _sequence(value, field_name=field_name)
    ]
    if len(players) != len(set(players)):
        raise ValueError(f"{field_name} cannot contain duplicates")
    expected = sorted(players, key=PLAYER_TO_ID.__getitem__)
    if players != expected:
        raise ValueError(f"{field_name} must use canonical player order")
    return players


def parse_public_phase(value: Any) -> tuple[int, str]:
    """Return the public day index and phase category from a canonical ID."""

    if not isinstance(value, str):
        raise TypeError("phase must be text")
    match = _PHASE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError("phase must be an existing canonical public phase ID")
    phase = match.group("phase")
    if phase not in PUBLIC_PHASES:
        raise ValueError("unsupported public phase")
    time = match.group("time")
    day = int(match.group("day"))
    if time == "day" and day < 1:
        raise ValueError("public daytime phases start at day 1")
    if (time, phase) not in {
        ("night", "skill_wolf"),
        ("day", "speech"),
        ("day", "speech_pk"),
        ("day", "vote"),
        ("day", "vote_pk"),
    }:
        raise ValueError("phase is not a public environment transition")
    return day, f"{time}_{phase}"


def normalize_public_event(
    event: Any,
    *,
    expected_idx: int,
) -> dict[str, Any]:
    """Validate one public event without adding or repairing fields."""

    if not isinstance(event, Mapping):
        raise TypeError("public event must be a mapping")
    event_type = event.get("event_type")
    if event_type not in PUBLIC_EVENT_TYPES:
        raise ValueError(f"unsupported public event_type: {event_type!r}")
    if set(event) != _EVENT_FIELDS[event_type]:
        missing = sorted(_EVENT_FIELDS[event_type] - set(event))
        extra = sorted(set(event) - _EVENT_FIELDS[event_type])
        raise ValueError(
            f"{event_type} field set mismatch; missing={missing}, extra={extra}"
        )
    event_idx = event.get("event_idx")
    if (
        isinstance(event_idx, bool)
        or not isinstance(event_idx, int)
        or event_idx != expected_idx
    ):
        raise ValueError(
            f"event_idx must be continuous; expected {expected_idx}"
        )

    normalized: dict[str, Any] = {
        "event_idx": event_idx,
        "event_type": event_type,
    }
    if event_type == "phase_change":
        phase = event.get("phase")
        parse_public_phase(phase)
        normalized["phase"] = phase
    elif event_type == "turn_start":
        normalized["speaker"] = _canonical_player(
            event.get("speaker"), field_name="speaker"
        )
    elif event_type == "public_speech":
        normalized["speaker"] = _canonical_player(
            event.get("speaker"), field_name="speaker"
        )
        raw_text = event.get("raw_text")
        if not isinstance(raw_text, str):
            raise TypeError("public_speech.raw_text must be text")
        normalized["raw_text"] = raw_text
        normalized["sp_actions"] = [
            parse_speech_action(action).to_list()
            for action in _sequence(
                event.get("sp_actions"), field_name="public_speech.sp_actions"
            )
        ]
    elif event_type == "vote_result":
        votes = []
        previous_voter_id = 0
        for vote in _sequence(event.get("votes"), field_name="vote_result.votes"):
            if not isinstance(vote, Mapping):
                raise TypeError("each public vote must be a mapping")
            if set(vote) != {"voter", "target"}:
                raise ValueError("each public vote requires voter and target")
            voter = _canonical_player(vote.get("voter"), field_name="voter")
            voter_id = PLAYER_TO_ID[voter]
            if voter_id <= previous_voter_id:
                raise ValueError("votes must use unique canonical voter order")
            previous_voter_id = voter_id
            target = vote.get("target")
            if target is not None:
                target = _canonical_player(target, field_name="vote target")
            votes.append({"voter": voter, "target": target})
        normalized["votes"] = votes
    elif event_type == "exile_result":
        normalized["exiled_players"] = _canonical_players(
            event.get("exiled_players"), field_name="exiled_players"
        )
    else:
        normalized["dead_players"] = _canonical_players(
            event.get("dead_players"), field_name="dead_players"
        )
    return normalized


def normalize_public_events(events: Any) -> list[dict[str, Any]]:
    """Validate a complete append-only public-event prefix."""

    normalized = []
    for index, event in enumerate(
        _sequence(events, field_name="public_events")
    ):
        normalized.append(normalize_public_event(event, expected_idx=index))
    return normalized


def public_speech_actions(events: Any) -> list[list[str]]:
    """Flatten exact speech actions for prompt/audit metadata only."""

    return [
        list(action)
        for event in normalize_public_events(events)
        if event["event_type"] == "public_speech"
        for action in event["sp_actions"]
    ]


def observer_public_action_counts(events: Any) -> tuple[int, ...]:
    """Count prior public speeches, speech actions, and votes per player."""

    counts = [0] * len(PLAYER_NAMES)
    for event in normalize_public_events(events):
        if event["event_type"] == "public_speech":
            counts[PLAYER_TO_ID[event["speaker"]] - 1] += 1
            for subject, _action, _object in event["sp_actions"]:
                counts[PLAYER_TO_ID[subject] - 1] += 1
        elif event["event_type"] == "vote_result":
            for vote in event["votes"]:
                counts[PLAYER_TO_ID[vote["voter"]] - 1] += 1
    return tuple(counts)


def structured_event_tokens(events: Any) -> list[dict[str, Any]]:
    """Project public events into the exact raw-text-free model token content."""

    tokens: list[dict[str, Any]] = []

    def append(
        token_type: str,
        *,
        subject: str | None = None,
        action: str | None = None,
        object_: str | None = None,
        phase: str | None = None,
        day: int = 0,
    ) -> None:
        tokens.append(
            {
                "token_type": token_type,
                "subject": subject,
                "action": action,
                "object": object_,
                "phase": phase,
                "day": day,
            }
        )

    for event in normalize_public_events(events):
        event_type = event["event_type"]
        if event_type == "phase_change":
            day, phase = parse_public_phase(event["phase"])
            append(event_type, phase=phase, day=day)
        elif event_type == "turn_start":
            append(event_type, subject=event["speaker"])
        elif event_type == "public_speech":
            append(event_type, subject=event["speaker"])
            for subject, action, object_ in event["sp_actions"]:
                append(
                    "speech_action",
                    subject=subject,
                    action=action,
                    object_=object_,
                )
        elif event_type == "vote_result":
            append(event_type)
            for vote in event["votes"]:
                append(
                    "vote",
                    subject=vote["voter"],
                    object_=vote["target"],
                )
        elif event_type == "exile_result":
            append(event_type)
            for player in event["exiled_players"]:
                append("exiled_player", object_=player)
        else:
            append(event_type)
            for player in event["dead_players"]:
                append("dead_player", object_=player)
    return tokens


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def public_event_digest(events: Any) -> str:
    return hashlib.sha256(
        _canonical_json(normalize_public_events(events)).encode("utf-8")
    ).hexdigest()


def structured_input_digest(events: Any) -> str:
    return hashlib.sha256(
        _canonical_json(structured_event_tokens(events)).encode("utf-8")
    ).hexdigest()


def copy_public_events(events: Any) -> list[dict[str, Any]]:
    """Return a validated detached copy of one history prefix."""

    return deepcopy(normalize_public_events(events))


__all__ = [
    "PHASE_TO_ID",
    "PUBLIC_EVENT_SCHEMA_VERSION",
    "PUBLIC_EVENT_TYPES",
    "STRUCTURED_TOKEN_TO_ID",
    "copy_public_events",
    "normalize_public_event",
    "normalize_public_events",
    "observer_public_action_counts",
    "parse_public_phase",
    "public_event_digest",
    "public_speech_actions",
    "structured_event_tokens",
    "structured_input_digest",
]
