"""Minimal schemas for the formal ToM input representation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


NUM_PLAYERS = 7
PAD_TOKEN = "<pad>"
PLAYER_NAMES = tuple(f"player{index}" for index in range(1, NUM_PLAYERS + 1))

ACTION_NAMES = (
    "point_as_werewolf",
    "point_as_villager",
    "point_as_seer",
    "point_as_witch",
    "point_as_guard",
    "support",
    "oppose",
)
ACTION_TO_ID = {
    PAD_TOKEN: 0,
    **{name: index for index, name in enumerate(ACTION_NAMES, start=1)},
}

EPISODE_CONTEXTS = ("seer_guard", "seer_witch")


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
    "EPISODE_CONTEXTS",
    "NUM_PLAYERS",
    "PAD_TOKEN",
    "PLAYER_NAMES",
    "SpeechAction",
    "normalize_action",
    "normalize_episode_context",
    "normalize_player",
]
