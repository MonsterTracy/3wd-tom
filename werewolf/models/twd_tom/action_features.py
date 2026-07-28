"""Encode canonical public events into structured causal-model features."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from werewolf.models.twd_tom.public_events import (
    PHASE_TO_ID,
    STRUCTURED_TOKEN_TO_ID,
    structured_event_tokens,
)
from werewolf.models.twd_tom.schema import ACTION_TO_ID, PLAYER_TO_ID


class PublicEventFeatureBuilder:
    """Convert public-event prefixes into one padded structured token stream."""

    FEATURE_FIELDS = (
        "subject_ids",
        "action_ids",
        "object_ids",
        "event_type_ids",
        "phase_ids",
        "day_values",
        "attention_mask",
    )

    def __init__(
        self,
        *,
        max_seq_len: int = 256,
        device: torch.device | str | None = None,
    ):
        if (
            isinstance(max_seq_len, bool)
            or not isinstance(max_seq_len, int)
            or max_seq_len <= 0
        ):
            raise ValueError("max_seq_len must be a positive integer")
        self.max_seq_len = max_seq_len
        self.device = device

    def encode_events(
        self,
        public_events: Sequence[Any],
    ) -> dict[str, torch.Tensor]:
        batch = self.encode_batch([public_events])
        return {key: value[0] for key, value in batch.items()}

    def encode_batch(
        self,
        event_sequences: Sequence[Sequence[Any]],
    ) -> dict[str, torch.Tensor]:
        if (
            isinstance(event_sequences, (str, bytes))
            or not isinstance(event_sequences, Sequence)
        ):
            raise TypeError("event_sequences must be a sequence")
        if not event_sequences:
            raise ValueError("event_sequences cannot be empty")

        encoded = [
            self._encode_sequence(sequence)
            for sequence in event_sequences
        ]
        sequence_length = max(1, max(map(len, encoded), default=0))
        shape = (len(encoded), sequence_length)
        features = {
            field: torch.zeros(
                shape,
                dtype=(
                    torch.float32
                    if field == "day_values"
                    else torch.long
                ),
                device=self.device,
            )
            for field in self.FEATURE_FIELDS
        }
        for batch_index, sequence in enumerate(encoded):
            if not sequence:
                continue
            length = len(sequence)
            values = torch.tensor(
                sequence,
                dtype=torch.float32,
                device=self.device,
            )
            for column, field in enumerate(self.FEATURE_FIELDS[:-1]):
                features[field][batch_index, :length] = values[:, column].to(
                    dtype=features[field].dtype
                )
            features["attention_mask"][batch_index, :length] = 1
        return features

    def _encode_sequence(
        self,
        public_events: Sequence[Any],
    ) -> list[tuple[int, int, int, int, int, float]]:
        encoded = []
        tokens = structured_event_tokens(public_events)
        groups: list[list[dict[str, Any]]] = []
        boundary_types = {
            "phase_change",
            "turn_start",
            "public_speech",
            "vote_result",
            "exile_result",
            "death_announcement",
        }
        for token in tokens:
            if token["token_type"] in boundary_types:
                groups.append([token])
            else:
                if not groups:
                    raise RuntimeError(
                        "structured detail token has no event boundary"
                    )
                groups[-1].append(token)
        retained: list[list[dict[str, Any]]] = []
        remaining = self.max_seq_len
        for group in reversed(groups):
            if len(group) <= remaining:
                retained.append(group)
                remaining -= len(group)
                continue
            if not retained and remaining > 0:
                retained.append(
                    group[:1] + group[-(remaining - 1) :]
                    if remaining > 1
                    else group[:1]
                )
            break
        tokens = [
            token
            for group in reversed(retained)
            for token in group
        ]
        for token in tokens:
            encoded.append(
                (
                    PLAYER_TO_ID.get(token["subject"], 0),
                    ACTION_TO_ID.get(token["action"], 0),
                    PLAYER_TO_ID.get(token["object"], 0),
                    STRUCTURED_TOKEN_TO_ID[token["token_type"]],
                    PHASE_TO_ID.get(token["phase"], 0),
                    float(token["day"]),
                )
            )
        return encoded


__all__ = ["PublicEventFeatureBuilder"]
