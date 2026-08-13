"""Tensorize formal public prefixes and subjective belief targets."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from werewolf.models.tom.public_history import build_model_input
from werewolf.models.tom.schema import (
    ACTION_TO_ID,
    CONFIG_TO_ID,
    EVENT_TO_ID,
    NONE_ACTION_ID,
    NONE_TOKEN,
    PAD_TOKEN,
    PHASE_TO_ID,
    PLAYER_NAMES,
    PLAYER_TO_ID,
    SpeechAction,
)
from werewolf.models.tom.targets import materialize_target


_SEQUENCE_FIELDS = (
    "event_type_ids",
    "subject_ids",
    "action_ids",
    "object_ids",
    "rounds",
    "phase_ids",
)
_BOOKKEEPING_EVENTS = {"turn_start", "phase_change"}


def _sequence(value: Any, *, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field} must be a sequence")
    return value


def _current_actions(value: Any) -> list[list[str]]:
    actions = _sequence(value, field="formal_speech_actions")
    if not actions:
        raise ValueError("formal_speech_actions must not be empty")
    normalized = []
    for action in actions:
        if isinstance(action, (str, bytes)) or not isinstance(action, Sequence):
            raise TypeError("formal speech actions must be triplets")
        if len(action) != 3:
            raise ValueError("formal speech actions must be triplets")
        canonical = SpeechAction.from_values(action[0], action[1], action[2]).to_list()
        if list(action) != canonical:
            raise ValueError("formal speech actions must use canonical values")
        normalized.append(canonical)
    return normalized


def _visible_events(raw_sample: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    public_events = list(
        _sequence(raw_sample.get("public_events"), field="public_events")
    )
    cutoff = raw_sample.get("public_history_cutoff")
    if not isinstance(cutoff, Mapping):
        raise TypeError("public_history_cutoff must be a mapping")
    cutoff_index = cutoff.get("event_idx")
    if isinstance(cutoff_index, bool) or not isinstance(cutoff_index, int):
        raise TypeError("public history cutoff event_idx must be an integer")
    if not 0 <= cutoff_index < len(public_events):
        raise ValueError("public history cutoff event_idx is out of range")
    cutoff_event = public_events[cutoff_index]
    if (
        not isinstance(cutoff_event, Mapping)
        or cutoff_event.get("event_type") != "public_speech"
    ):
        raise ValueError("public history cutoff must identify public_speech")
    for event in public_events[cutoff_index + 1 :]:
        if not isinstance(event, Mapping):
            raise TypeError("public ledger events must be mappings")
        if event.get("event_type") not in _BOOKKEEPING_EVENTS:
            raise ValueError("public evidence exists after the speech cutoff")
    return public_events[: cutoff_index + 1]


def _encode_event(event: Mapping[str, Any]) -> tuple[list[int], list[bool]]:
    event_type = event.get("type")
    if event_type not in EVENT_TO_ID or event_type == PAD_TOKEN:
        raise ValueError(f"unknown formal event type: {event_type!r}")
    phase = event.get("phase")
    if phase not in PHASE_TO_ID or phase == PAD_TOKEN:
        raise ValueError(f"unknown formal phase: {phase!r}")
    round_number = event.get("round")
    if (
        isinstance(round_number, bool)
        or not isinstance(round_number, int)
        or round_number <= 0
    ):
        raise ValueError("formal event round must be a positive integer")

    none_player = PLAYER_TO_ID[NONE_TOKEN]
    subject_id = none_player
    action_id = NONE_ACTION_ID
    object_id = none_player
    dead = [False] * len(PLAYER_NAMES)
    if event_type == "speech_action":
        subject_id = PLAYER_TO_ID[event["subject"]]
        action_id = ACTION_TO_ID[event["predicate"]]
        object_id = PLAYER_TO_ID[event["object"]]
    elif event_type == "vote":
        subject_id = PLAYER_TO_ID[event["voter"]]
        target = event["target"]
        object_id = none_player if target is None else PLAYER_TO_ID[target]
    elif event_type == "exile":
        player = event["player"]
        object_id = none_player if player is None else PLAYER_TO_ID[player]
    elif event_type == "night_result":
        dead_players = event["dead_players"]
        if len(dead_players) != len(set(dead_players)):
            raise ValueError("night_result dead_players must not contain duplicates")
        for player in dead_players:
            dead[PLAYER_NAMES.index(player)] = True

    categorical = [
        EVENT_TO_ID[event_type],
        subject_id,
        action_id,
        object_id,
        round_number,
        PHASE_TO_ID[phase],
    ]
    return categorical, dead


def encode_sample(raw_sample: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Build one complete causal sequence and its Phase 3 target."""

    if not isinstance(raw_sample, Mapping):
        raise TypeError("raw sample must be a mapping")
    current_actions = _current_actions(raw_sample.get("formal_speech_actions"))
    public_events = _visible_events(raw_sample)
    cutoff_actions = _current_actions(public_events[-1].get("sp_actions"))
    if cutoff_actions != current_actions:
        raise ValueError("current formal speech actions do not match the cutoff speech")

    model_input = build_model_input(
        episode_context=raw_sample.get("episode_context"),
        public_events=public_events,
    )
    projected_events = model_input["events"]
    suffix = projected_events[-len(current_actions) :]
    projected_actions = [
        [event.get("subject"), event.get("predicate"), event.get("object")]
        for event in suffix
        if event.get("type") == "speech_action"
    ]
    if projected_actions != current_actions or len(suffix) != len(current_actions):
        raise ValueError("projected history does not end with current speech actions")

    categorical = []
    dead_sets = []
    for event in projected_events:
        values, dead = _encode_event(event)
        categorical.append(values)
        dead_sets.append(dead)
    if not categorical:
        raise ValueError("formal public sequence must not be empty")

    values = torch.tensor(categorical, dtype=torch.long)
    target, observer_mask = materialize_target(
        alive_observers=raw_sample.get("alive_observers"),
        observer_reports=raw_sample.get("observer_reports"),
    )
    return {
        "config_id": torch.tensor(
            CONFIG_TO_ID[model_input["episode_context"]],
            dtype=torch.long,
        ),
        "event_type_ids": values[:, 0],
        "subject_ids": values[:, 1],
        "action_ids": values[:, 2],
        "object_ids": values[:, 3],
        "rounds": values[:, 4],
        "phase_ids": values[:, 5],
        "dead_players": torch.tensor(dead_sets, dtype=torch.bool),
        "attention_mask": torch.ones(len(categorical), dtype=torch.bool),
        "sequence_length": torch.tensor(len(categorical), dtype=torch.long),
        "target": torch.tensor(target, dtype=torch.float32),
        "observer_mask": torch.tensor(observer_mask, dtype=torch.bool),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"line {line_number}: blank JSONL row")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise TypeError(f"line {line_number}: record must be a mapping")
            records.append(record)
    return records


class TomDataset(Dataset):
    """Load formal raw samples without exposing label-side state as features."""

    def __init__(self, source: str | Path | Sequence[Mapping[str, Any]]) -> None:
        if isinstance(source, (str, Path)):
            records = _load_jsonl(Path(source))
        elif isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
            records = list(source)
        else:
            raise TypeError("source must be a JSONL path or sample sequence")
        if not records:
            raise ValueError("Dataset must not be empty")
        self._items = [encode_sample(record) for record in records]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self._items[index]


def collate_batch(items: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Right-pad a batch to its longest complete formal prefix."""

    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
        raise TypeError("items must be a sequence")
    if not items:
        raise ValueError("cannot collate an empty batch")
    batch = {
        field: pad_sequence(
            [item[field] for item in items],
            batch_first=True,
            padding_value=0,
        )
        for field in _SEQUENCE_FIELDS
    }
    batch["dead_players"] = pad_sequence(
        [item["dead_players"] for item in items],
        batch_first=True,
        padding_value=False,
    )
    batch["attention_mask"] = pad_sequence(
        [item["attention_mask"] for item in items],
        batch_first=True,
        padding_value=False,
    )
    for field in ("config_id", "sequence_length", "target", "observer_mask"):
        batch[field] = torch.stack([item[field] for item in items])
    return batch


__all__ = ["TomDataset", "collate_batch", "encode_sample"]
