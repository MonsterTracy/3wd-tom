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
from itertools import combinations
from typing import Any


NUM_PLAYERS = 7

# This objective environment configuration also defines the fixed pair-label
# space.
NUM_WEREWOLVES = 2

PROJECTED_SCHEMA_VERSION = (
    "classic7_pre_speech_suspicion_pair_distribution_v2"
)
PROJECTION_VERSION = (
    "classic7_player_suspicion_pair_projection_base2_v1"
)
TARGET_ENCODING = PROJECTION_VERSION
RAW_LABEL_FIELD = "suspected_werewolves"
RAW_LABEL_TYPE = "finite_symbolic_player_set"
NUMERIC_ANNOTATION_PRESENT = False
RAW_LABEL_SEMANTICS = (
    "observer_internal_player_suspicion_set_v1"
)
TARGET_INTERPRETATION = (
    "deterministic_base2_projection_of_player_suspicion_and_hard_knowledge_v1"
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
LABEL_PROMPT_VERSION = "classic7_pre_speech_player_suspicion_prompt_v2"
MARGINAL_SEMANTICS = "two_wolf_membership_probability_v1"
PAIR_ORDERING = "global_lexicographic_two_player_combinations"
MODEL_OUTPUT = "observer_pair_logits"
OUTPUT_ACTIVATION = "softmax_over_pair_classes"
TARGET_DISTRIBUTION_IS_REPORTER_PROBABILITY = False
TARGET_DISTRIBUTION_IS_DETERMINISTIC_ENCODING = True

PAD_TOKEN = "<pad>"


# Player names follow ONUW's player1, player2, ... convention.
PLAYER_NAMES: tuple[str, ...] = tuple(
    f"player{player_id}"
    for player_id in range(1, NUM_PLAYERS + 1)
)
CANONICAL_PLAYER_ORDERING = PLAYER_NAMES
SECOND_ORDER_TARGET_ENCODING = "classic7_second_order_wolf_pair_distribution_v2"
SECOND_ORDER_OBSERVER_READOUT = "public_event_query_attention_v1"
SECOND_ORDER_OBSERVER_EVENT_CONDITIONING = (
    "cyclic_relative_player_relations_v1"
)
SECOND_ORDER_SUBJECT_SUPERVISION = (
    "post_completed_public_speech_pre_next_action_v1"
)


CANONICAL_WOLF_PAIRS: tuple[tuple[str, str], ...] = tuple(
    combinations(
        PLAYER_NAMES,
        NUM_WEREWOLVES,
    )
)

NUM_WOLF_PAIR_CLASSES = len(
    CANONICAL_WOLF_PAIRS
)


def canonical_wolf_pairs() -> tuple[tuple[str, str], ...]:
    """Return the sole global 21-class two-Werewolf ordering."""

    return CANONICAL_WOLF_PAIRS


# Minimal ONUW-style action vocabulary adapted to the roles supported by
# the seven-player environment.
#
# PAD_TOKEN is not a raw action. It is reserved for tensor padding.
ACTION_NAMES: tuple[str, ...] = (
    "point_as_werewolf",
    "point_as_villager",
    "point_as_seer",
    "point_as_witch",
    "point_as_guard",
    "support",
    "oppose",
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


# ID 0 is always padding. Real players and actions start from 1.
PLAYER_TO_ID: dict[str, int] = {
    PAD_TOKEN: 0,
    **{
        player_name: index
        for index, player_name in enumerate(PLAYER_NAMES, start=1)
    },
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


def validate_player_suspicion(
    suspected_werewolves: Any,
    known_werewolves: Any,
    known_non_werewolves: Any,
) -> list[str]:
    """Validate one player-level suspicion set against observer hard knowledge."""

    suspected = canonicalize_player_set(
        suspected_werewolves,
        field_name="suspected_werewolves",
    )
    known_wolves = canonicalize_player_set(
        known_werewolves,
        field_name="known_werewolves",
    )
    known_non_wolves = canonicalize_player_set(
        known_non_werewolves,
        field_name="known_non_werewolves",
    )
    known_wolf_set = set(known_wolves)
    known_non_wolf_set = set(known_non_wolves)
    if known_wolf_set & known_non_wolf_set:
        raise ValueError("hard knowledge sets must be disjoint")
    suspected_set = set(suspected)
    if not known_wolf_set.issubset(suspected_set):
        raise ValueError(
            "suspected_werewolves must contain all known_werewolves"
        )
    if suspected_set & known_non_wolf_set:
        raise ValueError(
            "suspected_werewolves cannot contain known_non_werewolves"
        )
    legal_candidates = [
        player
        for player in PLAYER_NAMES
        if player not in known_non_wolf_set
    ]
    if suspected == legal_candidates and legal_candidates != known_wolves:
        raise ValueError(
            "suspected_werewolves cannot equal all legal candidates unless "
            "hard knowledge already determines the full candidate set"
        )
    return suspected

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
    object: str

    def __post_init__(self) -> None:
        if self.subject not in PLAYER_NAMES:
            raise ValueError(
                f"subject must be canonical player name, got {self.subject!r}"
            )

        if self.object not in PLAYER_NAMES:
            raise ValueError(
                f"object must be canonical player name, got {self.object!r}"
            )

        if self.action not in ACTION_NAMES:
            raise ValueError(
                f"unsupported action: {self.action!r}"
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
            object=normalize_player(object_),
        )

    def to_list(self) -> list[str]:
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
    "NUM_WOLF_PAIR_CLASSES",
    "CANONICAL_WOLF_PAIRS",
    "PROJECTED_SCHEMA_VERSION",
    "PROJECTION_VERSION",
    "TARGET_ENCODING",
    "RAW_LABEL_FIELD",
    "RAW_LABEL_TYPE",
    "NUMERIC_ANNOTATION_PRESENT",
    "RAW_LABEL_SEMANTICS",
    "TARGET_INTERPRETATION",
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
    "MARGINAL_SEMANTICS",
    "PAIR_ORDERING",
    "MODEL_OUTPUT",
    "OUTPUT_ACTIVATION",
    "TARGET_DISTRIBUTION_IS_REPORTER_PROBABILITY",
    "TARGET_DISTRIBUTION_IS_DETERMINISTIC_ENCODING",
    "PAD_TOKEN",
    "PLAYER_NAMES",
    "CANONICAL_PLAYER_ORDERING",
    "SECOND_ORDER_TARGET_ENCODING",
    "SECOND_ORDER_OBSERVER_READOUT",
    "SECOND_ORDER_OBSERVER_EVENT_CONDITIONING",
    "SECOND_ORDER_SUBJECT_SUPERVISION",
    "ACTION_NAMES",
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
    "validate_player_suspicion",
    "canonical_wolf_pairs",
    "parse_speech_action",
]
