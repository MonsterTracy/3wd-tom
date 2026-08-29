"""Game-sequence dataset for Classic7 ONUW parity.

Canonical unit:
    one game -> one chronological public token sequence -> many strict PRE
    queries -> one full [7, 7] target per query.

Sparse prefixes are intentionally not the training representation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch
from torch.utils.data import Dataset

from werewolf.models.twd_tom.onuw_parity_protocol import (
    CLASSIC7_ONUW_REFERENCE,
    CLASSIC7_PUBLIC_EVENTS,
    CONTENT_PROFILES,
    EMOTION_NAMES,
    MODALITY_PROFILES,
    ONUW_ACTION_ONLY,
    ONUW_AGENT_DECLARED_MULTIMODAL,
)
from werewolf.models.twd_tom.schema import (
    ACTION_TO_ID,
    NONE_TOKEN,
    PLAYER_NAMES,
    PLAYER_TO_ID,
    normalize_action,
)
from werewolf.models.twd_tom.public_events import (
    normalize_public_events,
    structured_event_tokens,
)
from werewolf.models.twd_tom.speech_annotations import (
    normalize_speech_annotations,
)


PARITY_GAME_SCHEMA_VERSION = "classic7_onuw_game_sequence_v1"
TOKEN_TYPES = (
    "bos",
    "speech_action",
    "phase_change",
    "turn_start",
    "public_speech",
    "vote_result",
    "vote",
    "exile_result",
    "exiled_player",
    "death_announcement",
    "dead_player",
)
TOKEN_TYPE_TO_ID = {
    name: index for index, name in enumerate(TOKEN_TYPES, start=1)
}
EMOTION_TO_ID = {
    name: index for index, name in enumerate(EMOTION_NAMES, start=1)
}
PHASE_NAMES = (
    "day_speech",
    "day_speech_pk",
    "day_vote",
    "day_vote_pk",
    "night_skill_wolf",
)
PHASE_TO_ID = {
    name: index for index, name in enumerate(PHASE_NAMES, start=1)
}
TOKEN_FIELDS = frozenset(
    {
        "token_type",
        "subject",
        "action",
        "object",
        "face",
        "tone",
        "phase",
        "day",
    }
)
QUERY_FIELDS = frozenset(
    {
        "query_id",
        "step_idx",
        "speaker",
        "token_cutoff",
        "observer_ids",
        "belief_target",
    }
)
FEATURE_FIELDS = (
    "subject_ids",
    "action_ids",
    "object_ids",
    "token_type_ids",
    "face_ids",
    "tone_ids",
    "phase_ids",
    "day_values",
)


def bos_token() -> dict[str, Any]:
    """Return the explicit empty-history query anchor."""

    return {
        "token_type": "bos",
        "subject": None,
        "action": None,
        "object": None,
        "face": None,
        "tone": None,
        "phase": None,
        "day": 0,
    }


def materialize_public_tokens(
    *,
    public_events: Sequence[Mapping[str, Any]],
    speech_annotations: Sequence[Mapping[str, Any]],
    speech_emotions: Sequence[Mapping[str, Any]],
    content_profile: str,
    modality_profile: str,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Project a complete public prefix without retaining raw speech text."""

    if content_profile not in CONTENT_PROFILES:
        raise ValueError("unsupported content_profile")
    if modality_profile not in MODALITY_PROFILES:
        raise ValueError("unsupported modality_profile")
    events = normalize_public_events(public_events)
    annotations = normalize_speech_annotations(
        speech_annotations,
        public_events=events,
        require_complete=True,
    )
    if isinstance(speech_emotions, (str, bytes)) or not isinstance(
        speech_emotions, Sequence
    ):
        raise TypeError("speech_emotions must be a sequence")
    emotions_by_event = {}
    for raw in speech_emotions:
        if not isinstance(raw, Mapping) or set(raw) != {
            "event_idx", "speaker", "face", "tone", "source"
        }:
            raise ValueError("speech emotion sidecar fields do not match contract")
        if raw["source"] != "agent_declared":
            raise ValueError("full emotion source must be agent_declared")
        if raw["speaker"] not in PLAYER_NAMES:
            raise ValueError("speech emotion speaker must be canonical")
        if raw["face"] not in EMOTION_TO_ID or raw["tone"] not in EMOTION_TO_ID:
            raise ValueError("speech emotion must use the 8-class vocabulary")
        event_idx = raw["event_idx"]
        if isinstance(event_idx, bool) or not isinstance(event_idx, int):
            raise TypeError("speech emotion event_idx must be an integer")
        if event_idx in emotions_by_event:
            raise ValueError("duplicate speech emotion event_idx")
        emotions_by_event[event_idx] = dict(raw)

    public_speeches = {
        event["event_idx"]: event
        for event in events
        if event["event_type"] == "public_speech"
    }
    if modality_profile == ONUW_AGENT_DECLARED_MULTIMODAL:
        if set(emotions_by_event) != set(public_speeches):
            raise ValueError(
                "full multimodal requires one agent-declared emotion per speech"
            )
        for event_idx, event in public_speeches.items():
            if emotions_by_event[event_idx]["speaker"] != event["speaker"]:
                raise ValueError("speech emotion speaker differs from public speech")

    annotations_by_event = {
        annotation["event_idx"]: annotation for annotation in annotations
    }
    speech_action_counts = [
        len(annotations_by_event[event_idx]["actions"])
        for event_idx in public_speeches
    ]
    expanded_emotions = []
    for event_idx in public_speeches:
        emotion = emotions_by_event.get(event_idx)
        expanded_emotions.extend(
            [emotion] * len(annotations_by_event[event_idx]["actions"])
        )

    raw_tokens = structured_event_tokens(events, annotations)
    tokens = [bos_token()]
    emotion_index = 0
    for raw_token in raw_tokens:
        if content_profile == ONUW_ACTION_ONLY and raw_token["token_type"] != (
            "speech_action"
        ):
            continue
        emotion = None
        if raw_token["token_type"] == "speech_action":
            emotion = expanded_emotions[emotion_index]
            emotion_index += 1
        tokens.append(
            {
                **raw_token,
                "face": emotion["face"] if emotion is not None else None,
                "tone": emotion["tone"] if emotion is not None else None,
            }
        )
    if emotion_index != len(expanded_emotions):
        raise RuntimeError("speech action/emotion projection count mismatch")
    return tokens, speech_action_counts


def _validate_optional_player(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in PLAYER_NAMES:
        raise ValueError(f"{field_name} must be a canonical player or null")
    return value


def _validate_token(
    value: Any,
    *,
    content_profile: str,
    modality_profile: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != TOKEN_FIELDS:
        raise ValueError("each public token must use the exact parity token fields")
    token = dict(value)
    token_type = token["token_type"]
    if token_type not in TOKEN_TYPE_TO_ID:
        raise ValueError(f"unsupported token_type: {token_type!r}")
    if content_profile == ONUW_ACTION_ONLY and token_type not in {
        "bos",
        "speech_action",
    }:
        raise ValueError("onuw_action_only accepts only BOS and speech actions")

    token["subject"] = _validate_optional_player(
        token["subject"], field_name="subject"
    )
    token["object"] = _validate_optional_player(
        token["object"], field_name="object"
    )
    action = token["action"]
    if token_type == "speech_action":
        if token["subject"] is None:
            raise ValueError("speech_action requires a subject")
        token["action"] = normalize_action(action)
    elif action is not None:
        raise ValueError("non-speech tokens cannot carry a speech action")

    for field in ("face", "tone"):
        emotion = token[field]
        if token_type == "speech_action" and (
            modality_profile == ONUW_AGENT_DECLARED_MULTIMODAL
        ):
            if emotion not in EMOTION_TO_ID:
                raise ValueError(
                    f"full multimodal speech_action requires 8-class {field}"
                )
        elif emotion is not None and emotion not in EMOTION_TO_ID:
            raise ValueError(f"unsupported {field}: {emotion!r}")

    phase = token["phase"]
    if phase is not None and phase not in PHASE_TO_ID:
        raise ValueError(f"unsupported public phase: {phase!r}")
    day = token["day"]
    if isinstance(day, bool) or not isinstance(day, int) or day < 0:
        raise ValueError("token day must be a non-negative integer")
    if content_profile == ONUW_ACTION_ONLY and (phase is not None or day != 0):
        raise ValueError("onuw_action_only does not encode phase/day")
    return token


def _validate_target(
    value: Any,
    *,
    observer_ids: Sequence[str],
) -> list[list[float]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("belief_target must be a 7x7 sequence")
    if len(value) != 7:
        raise ValueError("belief_target must have seven observer rows")
    alive = set(observer_ids)
    matrix = []
    for row_index, raw_row in enumerate(value):
        if isinstance(raw_row, (str, bytes)) or not isinstance(raw_row, Sequence):
            raise TypeError("each belief target row must be a sequence")
        if len(raw_row) != 7:
            raise ValueError("each belief target row must have seven columns")
        row = []
        for raw in raw_row:
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TypeError("belief target values must be numeric")
            probability = float(raw)
            if probability < 0.0:
                raise ValueError("belief target values cannot be negative")
            row.append(probability)
        observer = PLAYER_NAMES[row_index]
        if observer in alive:
            if abs(sum(row) - 1.0) > 1e-6:
                raise ValueError("each alive observer target row must sum to one")
        elif any(row):
            raise ValueError("dead observer target rows must remain zero")
        matrix.append(row)
    return matrix


def validate_parity_game(value: Any) -> dict[str, Any]:
    """Validate one full game without truncating or repairing it."""

    if not isinstance(value, Mapping):
        raise TypeError("parity game must be a mapping")
    required = {
        "schema_version",
        "protocol_id",
        "game_id",
        "content_profile",
        "modality_profile",
        "tokens",
        "queries",
        "speech_action_counts",
    }
    if set(value) != required:
        missing = sorted(required - set(value))
        extra = sorted(set(value) - required)
        raise ValueError(f"parity game fields mismatch; missing={missing}, extra={extra}")
    if value["schema_version"] != PARITY_GAME_SCHEMA_VERSION:
        raise ValueError("unsupported parity game schema_version")
    if value["protocol_id"] != CLASSIC7_ONUW_REFERENCE:
        raise ValueError("parity dataset accepts only Classic7-ONUW-reference")
    game_id = value["game_id"]
    if not isinstance(game_id, str) or not game_id.strip():
        raise ValueError("game_id must be non-empty text")
    content_profile = value["content_profile"]
    modality_profile = value["modality_profile"]
    if content_profile not in CONTENT_PROFILES:
        raise ValueError("unsupported content_profile")
    if modality_profile not in MODALITY_PROFILES:
        raise ValueError("unsupported modality_profile")

    raw_tokens = value["tokens"]
    if isinstance(raw_tokens, (str, bytes)) or not isinstance(raw_tokens, Sequence):
        raise TypeError("tokens must be a sequence")
    tokens = [
        _validate_token(
            token,
            content_profile=content_profile,
            modality_profile=modality_profile,
        )
        for token in raw_tokens
    ]
    if not tokens or tokens[0] != bos_token():
        raise ValueError("tokens must start with the canonical BOS token")
    if any(token["token_type"] == "bos" for token in tokens[1:]):
        raise ValueError("BOS may appear only at index zero")

    raw_queries = value["queries"]
    if isinstance(raw_queries, (str, bytes)) or not isinstance(raw_queries, Sequence):
        raise TypeError("queries must be a sequence")
    if not raw_queries:
        raise ValueError("each parity game requires at least one PRE query")
    queries = []
    previous_step = -1
    previous_cutoff = -1
    query_ids = set()
    for raw_query in raw_queries:
        if not isinstance(raw_query, Mapping) or set(raw_query) != QUERY_FIELDS:
            raise ValueError("each PRE query must use the exact query fields")
        query = dict(raw_query)
        query_id = query["query_id"]
        if not isinstance(query_id, str) or not query_id or query_id in query_ids:
            raise ValueError("query_id must be unique non-empty text")
        query_ids.add(query_id)
        step_idx = query["step_idx"]
        cutoff = query["token_cutoff"]
        if isinstance(step_idx, bool) or not isinstance(step_idx, int):
            raise TypeError("step_idx must be an integer")
        if step_idx <= previous_step:
            raise ValueError("PRE queries must use increasing step_idx")
        if isinstance(cutoff, bool) or not isinstance(cutoff, int):
            raise TypeError("token_cutoff must be an integer")
        if cutoff < previous_cutoff or not 0 <= cutoff < len(tokens):
            raise ValueError("token_cutoff must be nondecreasing and in range")
        speaker = query["speaker"]
        if speaker not in PLAYER_NAMES:
            raise ValueError("query speaker must be canonical")
        observers = query["observer_ids"]
        if isinstance(observers, (str, bytes)) or not isinstance(observers, Sequence):
            raise TypeError("observer_ids must be a sequence")
        observers = list(observers)
        if not observers or len(observers) != len(set(observers)):
            raise ValueError("observer_ids must be non-empty and unique")
        if observers != [player for player in PLAYER_NAMES if player in observers]:
            raise ValueError("observer_ids must follow canonical order")
        query["observer_ids"] = observers
        query["belief_target"] = _validate_target(
            query["belief_target"], observer_ids=observers
        )
        queries.append(query)
        previous_step = step_idx
        previous_cutoff = cutoff

    action_counts = value["speech_action_counts"]
    if isinstance(action_counts, (str, bytes)) or not isinstance(action_counts, Sequence):
        raise TypeError("speech_action_counts must be a sequence")
    action_counts = list(action_counts)
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in action_counts
    ):
        raise ValueError("speech_action_counts must contain non-negative integers")
    return {
        "schema_version": PARITY_GAME_SCHEMA_VERSION,
        "protocol_id": CLASSIC7_ONUW_REFERENCE,
        "game_id": game_id,
        "content_profile": content_profile,
        "modality_profile": modality_profile,
        "tokens": tokens,
        "queries": queries,
        "speech_action_counts": action_counts,
    }


def _encode_token(token: Mapping[str, Any], *, modality_profile: str) -> tuple:
    use_emotion = modality_profile == ONUW_AGENT_DECLARED_MULTIMODAL
    object_id = (
        PLAYER_TO_ID[NONE_TOKEN]
        if token["token_type"] == "speech_action" and token["object"] is None
        else PLAYER_TO_ID.get(token["object"], 0)
    )
    return (
        PLAYER_TO_ID.get(token["subject"], 0),
        ACTION_TO_ID.get(token["action"], 0),
        object_id,
        TOKEN_TYPE_TO_ID[token["token_type"]],
        EMOTION_TO_ID.get(token["face"], 0) if use_emotion else 0,
        EMOTION_TO_ID.get(token["tone"], 0) if use_emotion else 0,
        PHASE_TO_ID.get(token["phase"], 0),
        float(token["day"]),
    )


class OnuwParityGameDataset(Dataset):
    """One item per game, with multiple PRE targets in each item."""

    def __init__(self, games: Sequence[Mapping[str, Any]]) -> None:
        if isinstance(games, (str, bytes)) or not isinstance(games, Sequence):
            raise TypeError("games must be a sequence")
        if not games:
            raise ValueError("games cannot be empty")
        self.games = [validate_parity_game(game) for game in games]
        if len({game["game_id"] for game in self.games}) != len(self.games):
            raise ValueError("game_id values must be unique")

    def __len__(self) -> int:
        return len(self.games)

    def __getitem__(self, index: int) -> dict[str, Any]:
        game = self.games[index]
        encoded = torch.tensor(
            [
                _encode_token(
                    token,
                    modality_profile=game["modality_profile"],
                )
                for token in game["tokens"]
            ],
            dtype=torch.float32,
        )
        features = {}
        for column, field in enumerate(FEATURE_FIELDS):
            features[field] = encoded[:, column].to(
                dtype=torch.float32 if field == "day_values" else torch.long
            )
        queries = game["queries"]
        targets = torch.tensor(
            [query["belief_target"] for query in queries], dtype=torch.float32
        )
        alive = torch.zeros((len(queries), 7), dtype=torch.bool)
        for query_index, query in enumerate(queries):
            alive[query_index] = torch.tensor(
                [player in query["observer_ids"] for player in PLAYER_NAMES],
                dtype=torch.bool,
            )
        return {
            **features,
            "token_attention_mask": torch.ones(len(game["tokens"]), dtype=torch.bool),
            "query_positions": torch.tensor(
                [query["token_cutoff"] for query in queries], dtype=torch.long
            ),
            "query_valid_mask": torch.ones(len(queries), dtype=torch.bool),
            "observer_alive_mask": alive,
            "belief_targets": targets,
            "metadata": {
                "game_id": game["game_id"],
                "content_profile": game["content_profile"],
                "modality_profile": game["modality_profile"],
                "query_ids": [query["query_id"] for query in queries],
                "step_indices": [query["step_idx"] for query in queries],
                "speech_action_counts": list(game["speech_action_counts"]),
            },
        }


def collate_onuw_parity_games(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pad token and query axes with three distinct masks."""

    if isinstance(batch, (str, bytes)) or not isinstance(batch, Sequence) or not batch:
        raise ValueError("batch must be a non-empty sequence")
    max_tokens = max(item["token_attention_mask"].shape[0] for item in batch)
    max_queries = max(item["query_valid_mask"].shape[0] for item in batch)
    result = {
        field: batch[0][field].new_zeros((len(batch), max_tokens))
        for field in FEATURE_FIELDS
    }
    result["token_attention_mask"] = torch.zeros(
        (len(batch), max_tokens), dtype=torch.bool
    )
    result["query_positions"] = torch.zeros(
        (len(batch), max_queries), dtype=torch.long
    )
    result["query_valid_mask"] = torch.zeros(
        (len(batch), max_queries), dtype=torch.bool
    )
    result["observer_alive_mask"] = torch.zeros(
        (len(batch), max_queries, 7), dtype=torch.bool
    )
    result["belief_targets"] = torch.zeros(
        (len(batch), max_queries, 7, 7), dtype=torch.float32
    )
    result["metadata"] = []
    for batch_index, item in enumerate(batch):
        token_count = item["token_attention_mask"].shape[0]
        query_count = item["query_valid_mask"].shape[0]
        for field in FEATURE_FIELDS:
            if item[field].shape != (token_count,):
                raise ValueError(f"feature shape mismatch: {field}")
            result[field][batch_index, :token_count] = item[field]
        result["token_attention_mask"][batch_index, :token_count] = item[
            "token_attention_mask"
        ]
        result["query_positions"][batch_index, :query_count] = item[
            "query_positions"
        ]
        result["query_valid_mask"][batch_index, :query_count] = item[
            "query_valid_mask"
        ]
        result["observer_alive_mask"][batch_index, :query_count] = item[
            "observer_alive_mask"
        ]
        result["belief_targets"][batch_index, :query_count] = item[
            "belief_targets"
        ]
        result["metadata"].append(deepcopy(item["metadata"]))
    return result


__all__ = [
    "PARITY_GAME_SCHEMA_VERSION",
    "TOKEN_TYPES",
    "TOKEN_TYPE_TO_ID",
    "EMOTION_TO_ID",
    "PHASE_TO_ID",
    "FEATURE_FIELDS",
    "bos_token",
    "materialize_public_tokens",
    "validate_parity_game",
    "OnuwParityGameDataset",
    "collate_onuw_parity_games",
]
