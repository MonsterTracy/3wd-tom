"""Archived schemas for the formal ToM input representation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


NUM_PLAYERS = 7
PAD_TOKEN = "<pad>"
NONE_TOKEN = "<none>"
PLAYER_NAMES = tuple(f"player{index}" for index in range(1, NUM_PLAYERS + 1))
PLAYER_TO_ID = {
    PAD_TOKEN: 0,
    NONE_TOKEN: 1,
    **{name: index for index, name in enumerate(PLAYER_NAMES, start=2)},
}

ACTION_NAMES = (
    "point_as_werewolf",
    "point_as_villager",
    "point_as_seer",
    "point_as_witch",
    "point_as_guard",
    "support",
    "oppose",
    "check_as_good",
    "check_as_werewolf",
    "save",
    "poison",
    "guard",
    "vote_intent",
)
ACTION_TO_ID = {
    PAD_TOKEN: 0,
    **{name: index for index, name in enumerate(ACTION_NAMES, start=1)},
}
NONE_ACTION_ID = len(ACTION_TO_ID)

EPISODE_CONTEXTS = ("seer_guard", "seer_witch")
CONFIG_TO_ID = {
    context: index
    for index, context in enumerate(EPISODE_CONTEXTS)
}

EVENT_NAMES = ("speech_action", "vote", "exile", "night_result")
EVENT_TO_ID = {
    PAD_TOKEN: 0,
    **{name: index for index, name in enumerate(EVENT_NAMES, start=1)},
}

PHASE_NAMES = ("discussion", "pk_discussion", "vote", "pk_vote", "night")
PHASE_TO_ID = {
    PAD_TOKEN: 0,
    **{name: index for index, name in enumerate(PHASE_NAMES, start=1)},
}


def normalize_player(value: Any) -> str:
    if isinstance(value, bool):
        raise TypeError("boolean values are not player references")
    if isinstance(value, int):
        player_id = value
    elif isinstance(value, str):
        compact = re.sub(r"[\s_-]+", "", value.strip().lower())
        match = re.fullmatch(r"(?:player)?([1-7])", compact)
        if match is None:
            raise ValueError(f"invalid player reference: {value!r}")
        player_id = int(match.group(1))
    else:
        raise TypeError("player reference must be an integer or string")
    if not 1 <= player_id <= NUM_PLAYERS:
        raise ValueError(f"player ID must be in [1, {NUM_PLAYERS}]")
    return f"player{player_id}"


def normalize_action(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("speech action must be text")
    action = value.strip().lower()
    if action not in ACTION_NAMES:
        raise ValueError(
            f"unsupported speech action {value!r}; allowed actions are {ACTION_NAMES}"
        )
    return action


def normalize_episode_context(value: Any) -> str:
    if not isinstance(value, str) or value not in EPISODE_CONTEXTS:
        raise ValueError(f"episode context must be one of {EPISODE_CONTEXTS}")
    return value


@dataclass(frozen=True)
class SpeechAction:
    subject: str
    action: str
    object: str

    @classmethod
    def from_values(
        cls,
        subject: Any,
        action: Any,
        object_: Any,
    ) -> "SpeechAction":
        return cls(
            subject=normalize_player(subject),
            action=normalize_action(action),
            object=normalize_player(object_),
        )

    def to_list(self) -> list[str]:
        return [self.subject, self.action, self.object]


__all__ = [
    "ACTION_NAMES",
    "ACTION_TO_ID",
    "CONFIG_TO_ID",
    "EPISODE_CONTEXTS",
    "EVENT_NAMES",
    "EVENT_TO_ID",
    "NONE_ACTION_ID",
    "NONE_TOKEN",
    "NUM_PLAYERS",
    "PAD_TOKEN",
    "PHASE_NAMES",
    "PHASE_TO_ID",
    "PLAYER_NAMES",
    "PLAYER_TO_ID",
    "SpeechAction",
    "normalize_action",
    "normalize_episode_context",
    "normalize_player",
]
