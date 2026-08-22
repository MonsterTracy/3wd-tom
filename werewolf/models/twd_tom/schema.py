"""Classic-seven schemas for observer-specific Theory of Mind.

Raw ToM data uses the public speech-action structure:
   [subject, action, object]

Raw data contains semantic strings. Integer IDs are introduced only when
the dataset is converted into tensors.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


NUM_PLAYERS = 7

NUM_WEREWOLVES = 2

RAW_LABEL_FIELD = "suspected_werewolves"
RAW_LABEL_TYPE = "finite_symbolic_player_set"
NUMERIC_ANNOTATION_PRESENT = False
RAW_LABEL_SEMANTICS = (
    "observer_internal_player_suspicion_set_v1"
)
SUPERVISION_SCOPE = "all_valid_alive_observer_rows_v1"
LABEL_PROVENANCE = "alive_observer_readonly_pre_speech_report_v1"
LABEL_SOURCE = "playing_agent_readonly_self_report"
LABEL_CONTEXT_SCOPE = "playing_agent_legally_available_information_state"
MODEL_INPUT_SCOPE = "structured_public_events_only"
REPORT_CONTEXT_MODE = "readonly_clone_of_playing_agent_context"
REPORT_SIDE_EFFECT_FREE = True
GLOBAL_TRUTH_INJECTED = False
OTHER_PLAYERS_PRIVATE_INFORMATION_VISIBLE = False
PRIVATE_CONTEXT_SERIALIZED = False
REPORT_TIMING = "pre_public_speech"
TRUTH_BASED_OBSERVER_SELECTION = False
OBSERVER_SELECTION = "publicly_alive_players"
LABEL_PROMPT_VERSION = "classic7_pre_speech_player_suspicion_prompt_v3"
PAD_TOKEN = "<pad>"
NONE_TOKEN = "<none>"


# Player names follow ONUW's player1, player2, ... convention.
PLAYER_NAMES: tuple[str, ...] = tuple(
    f"player{player_id}"
    for player_id in range(1, NUM_PLAYERS + 1)
)
CANONICAL_PLAYER_ORDERING = PLAYER_NAMES
# Minimal ONUW-style action vocabulary adapted to the roles supported by
# the seven-player environment.
#
# PAD_TOKEN is not a raw action. It is reserved for tensor padding.
ACTION_NAMES: tuple[str, ...] = (
    "point_as_werewolf",
    "point_as_villager",
    "point_as_seer",
    "point_as_witch",
    "support",
    "oppose",
    "check_as_good",
    "check_as_werewolf",
    "save",
    "poison",
    "vote_intent",
    "abstain_intent",
    "no_commitment",
)

TARGETLESS_ACTION_NAMES = frozenset(
    {"abstain_intent", "no_commitment"}
)


# Roles accepted when normalizing a player's known identity.
#
# "unknown" means that the subject has not assigned a concrete identity
# to the object. It is not a model target class for wolf belief.
GUESS_ROLE_NAMES: tuple[str, ...] = (
    "werewolf",
    "villager",
    "seer",
    "witch",
    "guard",
    "unknown",
)


# ID 0 is always padding. Real players keep IDs 1...7, while targetless
# speech-action objects use one distinct non-padding sentinel.
PLAYER_TO_ID: dict[str, int] = {
    PAD_TOKEN: 0,
    **{
        player_name: index
        for index, player_name in enumerate(PLAYER_NAMES, start=1)
    },
    NONE_TOKEN: NUM_PLAYERS + 1,
}


def canonicalize_player_set(values: Any, *, field_name: str) -> list[str]:
    """Validate and order one duplicate-free set of canonical player IDs."""

    if not isinstance(values, list):
        raise TypeError(f"{field_name} must be a list")
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} entries must be strings")
        if value not in PLAYER_NAMES:
            raise ValueError(
                f"{field_name} entries must be canonical player1...player7 names"
            )
        if value in seen:
            raise ValueError(f"duplicate player in {field_name}: {value}")
        seen.add(value)
        normalized.append(value)
    return sorted(normalized, key=PLAYER_TO_ID.__getitem__)


ACTION_TO_ID: dict[str, int] = {
    PAD_TOKEN: 0,
    **{
        action_name: index
        for index, action_name in enumerate(ACTION_NAMES, start=1)
    },
}

ID_TO_PLAYER: dict[int, str] = {
    value: key for key, value in PLAYER_TO_ID.items()
}

ID_TO_ACTION: dict[int, str] = {
    value: key for key, value in ACTION_TO_ID.items()
}


def normalize_player(value: Any) -> str:
    """Convert a player reference into canonical form such as ``player3``.

    Accepted examples:
        3
        "3"
        "player3"
        "Player_3"
        "player 3"
    """

    if isinstance(value, bool):
        raise TypeError("boolean values are not valid player references")

    if isinstance(value, int):
        player_id = value
    elif isinstance(value, str):
        compact = re.sub(r"[\s_-]+", "", value.strip().lower())
        match = re.fullmatch(r"(?:player)?([1-7])", compact)

        if match is None:
            raise ValueError(f"invalid player reference: {value!r}")

        player_id = int(match.group(1))
    else:
        raise TypeError(
            "player reference must be an integer or string; "
            f"got {type(value).__name__}"
        )

    if not 1 <= player_id <= NUM_PLAYERS:
        raise ValueError(
            f"player ID must be in [1, {NUM_PLAYERS}], got {player_id}"
        )

    return f"player{player_id}"


def normalize_action(value: Any) -> str:
    """Validate and normalize a raw speech action name."""

    if not isinstance(value, str):
        raise TypeError(
            f"action must be a string, got {type(value).__name__}"
        )

    action = value.strip().lower()

    if action not in ACTION_NAMES:
        raise ValueError(
            f"unsupported speech action {value!r}; "
            f"allowed actions are {ACTION_NAMES}"
        )

    return action


def normalize_guess_role(value: Any) -> str:
    """Normalize a player's known identity from its own observation."""

    if value is None:
        return "unknown"

    if not isinstance(value, str):
        raise TypeError(
            f"guess role must be a string or None, "
            f"got {type(value).__name__}"
        )

    role = value.strip().lower()

    if role not in GUESS_ROLE_NAMES:
        raise ValueError(
            f"unsupported guess role {value!r}; "
            f"allowed roles are {GUESS_ROLE_NAMES}"
        )

    return role


@dataclass(frozen=True)
class SpeechAction:
    """One structured ONUW-style speech action."""

    subject: str
    action: str
    object: str | None

    def __post_init__(self) -> None:
        if self.subject not in PLAYER_NAMES:
            raise ValueError(
                f"subject must be canonical player name, got {self.subject!r}"
            )

        if self.action not in ACTION_NAMES:
            raise ValueError(
                f"unsupported action: {self.action!r}"
            )

        if self.action in TARGETLESS_ACTION_NAMES:
            if self.object is not None:
                raise ValueError(
                    f"{self.action} must use object=None"
                )
        elif self.object not in PLAYER_NAMES:
            raise ValueError(
                f"{self.action} requires a canonical player object"
            )

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
            object=(
                None
                if object_ is None
                else normalize_player(object_)
            ),
        )

    def to_list(self) -> list[str | None]:
        """Return the raw ONUW-compatible triplet."""

        return [self.subject, self.action, self.object]


def parse_speech_action(item: Sequence[Any]) -> SpeechAction:
    """Parse ``[subject, action, object]`` into a validated object."""

    if isinstance(item, (str, bytes)) or not isinstance(item, Sequence):
        raise TypeError(
            "speech action must be a three-element sequence"
        )

    if len(item) != 3:
        raise ValueError(
            f"speech action must contain exactly three elements, got {item!r}"
        )

    return SpeechAction.from_values(
        subject=item[0],
        action=item[1],
        object_=item[2],
    )


__all__ = [
    "NUM_PLAYERS",
    "NUM_WEREWOLVES",
    "RAW_LABEL_FIELD",
    "RAW_LABEL_TYPE",
    "NUMERIC_ANNOTATION_PRESENT",
    "RAW_LABEL_SEMANTICS",
    "SUPERVISION_SCOPE",
    "LABEL_PROVENANCE",
    "LABEL_SOURCE",
    "LABEL_CONTEXT_SCOPE",
    "MODEL_INPUT_SCOPE",
    "REPORT_CONTEXT_MODE",
    "REPORT_SIDE_EFFECT_FREE",
    "GLOBAL_TRUTH_INJECTED",
    "OTHER_PLAYERS_PRIVATE_INFORMATION_VISIBLE",
    "PRIVATE_CONTEXT_SERIALIZED",
    "REPORT_TIMING",
    "TRUTH_BASED_OBSERVER_SELECTION",
    "OBSERVER_SELECTION",
    "LABEL_PROMPT_VERSION",
    "PAD_TOKEN",
    "NONE_TOKEN",
    "PLAYER_NAMES",
    "CANONICAL_PLAYER_ORDERING",
    "ACTION_NAMES",
    "TARGETLESS_ACTION_NAMES",
    "GUESS_ROLE_NAMES",
    "PLAYER_TO_ID",
    "ACTION_TO_ID",
    "ID_TO_PLAYER",
    "ID_TO_ACTION",
    "SpeechAction",
    "normalize_player",
    "normalize_action",
    "normalize_guess_role",
    "canonicalize_player_set",
    "parse_speech_action",
]
