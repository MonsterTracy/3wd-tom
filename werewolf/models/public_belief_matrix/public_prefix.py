"""Shared model-visible public-prefix contract for Public Belief Matrix V1."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from werewolf.models.twd_tom.action_features import PublicEventFeatureBuilder
from werewolf.models.twd_tom.public_events import PHASE_TO_ID, STRUCTURED_TOKEN_TO_ID
from werewolf.models.twd_tom.schema import ACTION_TO_ID, PLAYER_TO_ID


def _inverse(mapping: Mapping[str, int], name: str) -> dict[int, str]:
    inverse = {value: key for key, value in mapping.items()}
    if len(inverse) != len(mapping):
        raise RuntimeError(f"{name} IDs must be unique")
    return inverse


def render_public_belief_matrix_visible_prefix(
    prefix: Mapping[str, torch.Tensor],
) -> str:
    """Render only the canonical model-visible tensors, failing closed."""

    if not isinstance(prefix, Mapping):
        raise TypeError("prefix must be a mapping")
    fields = PublicEventFeatureBuilder.FEATURE_FIELDS
    if set(prefix) != set(fields):
        raise ValueError("prefix fields do not match the canonical feature contract")
    lengths = set()
    for field in fields:
        value = prefix[field]
        if not isinstance(value, torch.Tensor) or value.ndim != 1:
            raise TypeError(f"prefix {field} must be a rank-one tensor")
        lengths.add(value.shape[0])
    if len(lengths) != 1:
        raise ValueError("prefix tensors must have equal lengths")

    vocabularies = {
        "subject_ids": _inverse(PLAYER_TO_ID, "player"),
        "action_ids": _inverse(ACTION_TO_ID, "action"),
        "object_ids": _inverse(PLAYER_TO_ID, "player"),
        "event_type_ids": _inverse(STRUCTURED_TOKEN_TO_ID, "event type"),
        "phase_ids": _inverse(PHASE_TO_ID, "phase"),
    }
    records = []
    for index in range(next(iter(lengths))):
        mask = prefix["attention_mask"][index].item()
        if mask not in (0, 1):
            raise ValueError("attention_mask values must be zero or one")
        if not mask:
            continue
        record: dict[str, Any] = {"index": index}
        for field, output_name in (
            ("event_type_ids", "event_type"),
            ("subject_ids", "subject"),
            ("action_ids", "action"),
            ("object_ids", "object"),
            ("phase_ids", "phase"),
        ):
            identifier = prefix[field][index].item()
            if isinstance(identifier, bool) or int(identifier) != identifier:
                raise ValueError(f"{field} must contain integer IDs")
            identifier = int(identifier)
            if identifier == 0:
                record[output_name] = "none"
            else:
                try:
                    record[output_name] = vocabularies[field][identifier]
                except KeyError as exc:
                    raise ValueError(f"invalid {field} vocabulary ID: {identifier}") from exc
        day = float(prefix["day_values"][index].item())
        if not torch.isfinite(torch.tensor(day)):
            raise ValueError("day_values must be finite")
        record["day"] = int(day) if day.is_integer() else day
        records.append(record)
    return json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_public_belief_matrix_visible_prefix(
    public_events: Sequence[Any],
    *,
    max_seq_len: int = 256,
) -> dict[str, torch.Tensor]:
    """Build the canonical raw-text-free, deterministically truncated prefix."""

    return PublicEventFeatureBuilder(max_seq_len=max_seq_len).encode_events(
        public_events
    )


__all__ = [
    "build_public_belief_matrix_visible_prefix",
    "render_public_belief_matrix_visible_prefix",
]
