"""Strict materialized Dataset contract for Public Belief Matrix V1."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from werewolf.models.public_belief_matrix.collection import (
    PUBLIC_BELIEF_MATRIX_MAX_SEQ_LEN,
    PUBLIC_BELIEF_MATRIX_SAMPLE_SCHEMA_VERSION,
    validate_public_belief_matrix_sample,
)
from werewolf.models.public_belief_matrix.targets import (
    suspicion_reports_to_matrix_target,
)
from werewolf.models.twd_tom.action_features import PublicEventFeatureBuilder


PUBLIC_BELIEF_MATRIX_MATERIALIZATION_VERSION = (
    "classic7_public_belief_matrix_training_target_v1"
)
MATERIALIZED_RECORD_FIELDS = frozenset(
    {
        "materialization_version",
        "source_schema_version",
        "seed",
        "sample",
        "matrix_target",
        "observer_row_mask",
    }
)
_SEED_PATTERN = re.compile(r"_seed_(\d+)$")


def seed_from_formal_game_id(game_id: Any) -> int:
    """Return the explicit terminal seed encoded by formal collection."""

    if not isinstance(game_id, str):
        raise TypeError("game_id must be text")
    match = _SEED_PATTERN.search(game_id)
    if match is None:
        raise ValueError(f"game_id does not contain a terminal seed: {game_id}")
    return int(match.group(1))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"line {line_number}: blank JSONL rows are not allowed")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise TypeError(f"line {line_number}: record must be a mapping")
            records.append(record)
    return records


def validate_materialized_record(record: Mapping[str, Any]) -> None:
    """Validate one materialized record against its symbolic source."""

    if not isinstance(record, Mapping):
        raise TypeError("materialized record must be a mapping")
    if set(record) != MATERIALIZED_RECORD_FIELDS:
        raise ValueError("materialized record fields do not match the contract")
    if record["materialization_version"] != PUBLIC_BELIEF_MATRIX_MATERIALIZATION_VERSION:
        raise ValueError("unsupported PBM materialization version")
    if record["source_schema_version"] != PUBLIC_BELIEF_MATRIX_SAMPLE_SCHEMA_VERSION:
        raise ValueError("unsupported PBM source schema")
    seed = record["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    sample = record["sample"]
    validate_public_belief_matrix_sample(sample)
    if sample["schema_version"] != record["source_schema_version"]:
        raise ValueError("source schema lineage mismatch")
    if seed_from_formal_game_id(sample.get("game_id")) != seed:
        raise ValueError("materialized seed does not match source game_id")
    snapshot_id = sample.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise ValueError("snapshot_id must be non-empty text")
    target = suspicion_reports_to_matrix_target(sample["observer_reports"])
    if record["matrix_target"] != [list(row) for row in target.matrix_target]:
        raise ValueError("materialized matrix_target does not match symbolic reports")
    if record["observer_row_mask"] != list(target.observer_row_mask):
        raise ValueError("materialized observer_row_mask does not match symbolic reports")
    if not any(target.observer_row_mask):
        raise ValueError("PBM snapshot must contain at least one valid observer row")


class PublicBeliefMatrixDataset(Dataset):
    """Load audited PBM materialized records without exposing metadata to the model."""

    def __init__(self, source: str | Path | Sequence[Mapping[str, Any]]) -> None:
        if isinstance(source, (str, Path)):
            records = _load_jsonl(Path(source))
        elif isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
            records = list(source)
        else:
            raise TypeError("source must be a JSONL path or a record sequence")
        if not records:
            raise ValueError("PBM dataset cannot be empty")
        self._items = [self._build_item(record) for record in records]

    @staticmethod
    def _build_item(record: Mapping[str, Any]) -> dict[str, Any]:
        validate_materialized_record(record)
        sample = record["sample"]
        payload = sample["structured_prefix"]
        fields = PublicEventFeatureBuilder.FEATURE_FIELDS
        if set(payload) != set(fields):
            raise ValueError("structured_prefix fields do not match the canonical contract")

        features: dict[str, torch.Tensor] = {}
        lengths = set()
        for field in fields:
            values = payload[field]
            if not isinstance(values, list):
                raise TypeError(f"structured_prefix.{field} must be a list")
            dtype = torch.float32 if field == "day_values" else torch.long
            tensor = torch.tensor(values, dtype=dtype)
            if tensor.ndim != 1:
                raise ValueError(f"structured_prefix.{field} must be rank one")
            lengths.add(tensor.shape[0])
            features[field] = tensor
        if len(lengths) != 1:
            raise ValueError("structured_prefix fields must have equal lengths")
        sequence_length = next(iter(lengths))
        if sequence_length <= 0 or sequence_length > PUBLIC_BELIEF_MATRIX_MAX_SEQ_LEN:
            raise ValueError("structured_prefix length must be between 1 and 256")

        target = suspicion_reports_to_matrix_target(sample["observer_reports"])
        features.update(
            {
                "matrix_target": torch.tensor(target.matrix_target, dtype=torch.float32),
                "observer_row_mask": torch.tensor(
                    target.observer_row_mask, dtype=torch.bool
                ),
                "metadata": {
                    "game_id": sample["game_id"],
                    "snapshot_id": sample["snapshot_id"],
                    "seed": record["seed"],
                },
            }
        )
        return features

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._items[index]


def collate_public_belief_matrix_batch(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pad variable-length prefixes and keep audit metadata outside model inputs."""

    if not items:
        raise ValueError("cannot collate an empty batch")
    batch = {
        field: pad_sequence(
            [item[field] for item in items],
            batch_first=True,
            padding_value=0,
        )
        for field in PublicEventFeatureBuilder.FEATURE_FIELDS
    }
    batch["matrix_target"] = torch.stack([item["matrix_target"] for item in items])
    batch["observer_row_mask"] = torch.stack(
        [item["observer_row_mask"] for item in items]
    )
    batch["metadata"] = [item["metadata"] for item in items]
    return batch


__all__ = [
    "MATERIALIZED_RECORD_FIELDS",
    "PUBLIC_BELIEF_MATRIX_MATERIALIZATION_VERSION",
    "PublicBeliefMatrixDataset",
    "collate_public_belief_matrix_batch",
    "seed_from_formal_game_id",
    "validate_materialized_record",
]
