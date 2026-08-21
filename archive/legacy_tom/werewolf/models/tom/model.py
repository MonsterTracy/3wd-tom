"""Archived causal model over the formal public-history representation."""

from __future__ import annotations

import torch
from torch import nn

from archive.legacy_tom.werewolf.models.tom.schema import (
    ACTION_TO_ID,
    CONFIG_TO_ID,
    EVENT_TO_ID,
    NONE_ACTION_ID,
    NUM_PLAYERS,
    PHASE_TO_ID,
    PLAYER_TO_ID,
)


HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 768
NUM_LAYERS = 4
NUM_HEADS = 8
DROPOUT = 0.1
MAX_SEQUENCE_LENGTH = 256


def _vocabulary_size(mapping: dict[str, int], *extra_ids: int) -> int:
    """Return the cardinality of one canonical ID space."""

    return max(*mapping.values(), *extra_ids) + 1


class BeliefModel(nn.Module):
    """Predict a complete observer-by-candidate belief matrix."""

    def __init__(self) -> None:
        super().__init__()
        player_count = _vocabulary_size(PLAYER_TO_ID)
        self.event_type_embedding = nn.Embedding(
            _vocabulary_size(EVENT_TO_ID), HIDDEN_SIZE, padding_idx=0
        )
        self.subject_embedding = nn.Embedding(
            player_count, HIDDEN_SIZE, padding_idx=0
        )
        self.action_embedding = nn.Embedding(
            _vocabulary_size(ACTION_TO_ID, NONE_ACTION_ID),
            HIDDEN_SIZE,
            padding_idx=0,
        )
        self.object_embedding = nn.Embedding(
            player_count, HIDDEN_SIZE, padding_idx=0
        )
        self.phase_embedding = nn.Embedding(
            _vocabulary_size(PHASE_TO_ID), HIDDEN_SIZE, padding_idx=0
        )
        self.round_projection = nn.Linear(1, HIDDEN_SIZE, bias=False)
        self.dead_set_projection = nn.Linear(
            NUM_PLAYERS, HIDDEN_SIZE, bias=False
        )
        self.config_embedding = nn.Embedding(
            _vocabulary_size(CONFIG_TO_ID), HIDDEN_SIZE
        )
        self.position_embedding = nn.Embedding(
            MAX_SEQUENCE_LENGTH, HIDDEN_SIZE
        )

        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=HIDDEN_SIZE,
                nhead=NUM_HEADS,
                dim_feedforward=INTERMEDIATE_SIZE,
                dropout=DROPOUT,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(NUM_LAYERS)
        )
        self.final_layer_norm = nn.LayerNorm(HIDDEN_SIZE)
        self.output_projection = nn.Linear(
            HIDDEN_SIZE, NUM_PLAYERS * NUM_PLAYERS
        )

    def forward(
        self,
        *,
        event_type_ids: torch.Tensor,
        subject_ids: torch.Tensor,
        action_ids: torch.Tensor,
        object_ids: torch.Tensor,
        phase_ids: torch.Tensor,
        rounds: torch.Tensor,
        dead_players: torch.Tensor,
        config_id: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return belief logits with shape ``[B, observer, candidate]``."""

        attention_mask = self._validate_inputs(
            event_type_ids=event_type_ids,
            subject_ids=subject_ids,
            action_ids=action_ids,
            object_ids=object_ids,
            phase_ids=phase_ids,
            rounds=rounds,
            dead_players=dead_players,
            config_id=config_id,
            attention_mask=attention_mask,
        )
        batch_size, sequence_length = event_type_ids.shape
        positions = torch.arange(
            sequence_length, device=event_type_ids.device
        ).unsqueeze(0)
        hidden_states = (
            self.event_type_embedding(event_type_ids)
            + self.subject_embedding(subject_ids)
            + self.action_embedding(action_ids)
            + self.object_embedding(object_ids)
            + self.phase_embedding(phase_ids)
            + self.round_projection(
                rounds.to(dtype=self.round_projection.weight.dtype).unsqueeze(-1)
            )
            + self.dead_set_projection(
                dead_players.to(dtype=self.dead_set_projection.weight.dtype)
            )
            + self.config_embedding(config_id).unsqueeze(1)
            + self.position_embedding(positions)
        )

        causal_mask = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=event_type_ids.device,
        ).triu(diagonal=1)
        padding_mask = ~attention_mask
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                src_mask=causal_mask,
                src_key_padding_mask=padding_mask,
            )
        hidden_states = self.final_layer_norm(hidden_states)

        positions = positions.expand(batch_size, -1)
        last_valid_indices = torch.where(
            attention_mask,
            positions,
            torch.full_like(positions, -1),
        ).max(dim=1).values
        pooled = hidden_states[
            torch.arange(batch_size, device=event_type_ids.device),
            last_valid_indices,
        ]
        return self.output_projection(pooled).reshape(
            batch_size, NUM_PLAYERS, NUM_PLAYERS
        )

    @staticmethod
    def _validate_inputs(
        *,
        event_type_ids: torch.Tensor,
        subject_ids: torch.Tensor,
        action_ids: torch.Tensor,
        object_ids: torch.Tensor,
        phase_ids: torch.Tensor,
        rounds: torch.Tensor,
        dead_players: torch.Tensor,
        config_id: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        sequence_tensors = {
            "event_type_ids": event_type_ids,
            "subject_ids": subject_ids,
            "action_ids": action_ids,
            "object_ids": object_ids,
            "phase_ids": phase_ids,
            "rounds": rounds,
            "attention_mask": attention_mask,
        }
        for name, tensor in {
            **sequence_tensors,
            "dead_players": dead_players,
            "config_id": config_id,
        }.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
        for name, tensor in sequence_tensors.items():
            if tensor.ndim != 2:
                raise ValueError(f"{name} must have shape [B, L]")
        expected_shape = event_type_ids.shape
        if any(tensor.shape != expected_shape for tensor in sequence_tensors.values()):
            raise ValueError("all sequence tensors must share shape [B, L]")
        batch_size, sequence_length = expected_shape
        if batch_size <= 0 or sequence_length <= 0:
            raise ValueError("feature tensors must have positive B and L")
        if sequence_length > MAX_SEQUENCE_LENGTH:
            raise ValueError(
                f"sequence length {sequence_length} exceeds the fixed "
                f"maximum of {MAX_SEQUENCE_LENGTH}"
            )
        if dead_players.shape != (batch_size, sequence_length, NUM_PLAYERS):
            raise ValueError("dead_players must have shape [B, L, 7]")
        if config_id.shape != (batch_size,):
            raise ValueError("config_id must have shape [B]")
        if any(
            tensor.device != event_type_ids.device
            for tensor in (*sequence_tensors.values(), dead_players, config_id)
        ):
            raise ValueError("all model inputs must use the same device")

        id_tensors = {
            "event_type_ids": (event_type_ids, _vocabulary_size(EVENT_TO_ID)),
            "subject_ids": (subject_ids, _vocabulary_size(PLAYER_TO_ID)),
            "action_ids": (
                action_ids,
                _vocabulary_size(ACTION_TO_ID, NONE_ACTION_ID),
            ),
            "object_ids": (object_ids, _vocabulary_size(PLAYER_TO_ID)),
            "phase_ids": (phase_ids, _vocabulary_size(PHASE_TO_ID)),
            "config_id": (config_id, _vocabulary_size(CONFIG_TO_ID)),
        }
        for name, (tensor, cardinality) in id_tensors.items():
            if tensor.dtype == torch.bool or torch.is_floating_point(tensor):
                raise TypeError(f"{name} must use an integer dtype")
            if torch.any(tensor < 0) or torch.any(tensor >= cardinality):
                raise ValueError(f"{name} contains an out-of-range ID")
        if rounds.dtype == torch.bool or torch.is_floating_point(rounds):
            raise TypeError("rounds must use an integer dtype")
        if torch.any(rounds < 0):
            raise ValueError("rounds cannot be negative")
        if dead_players.dtype != torch.bool:
            raise TypeError("dead_players must use torch.bool")
        if attention_mask.dtype != torch.bool:
            raise TypeError("attention_mask must use torch.bool")
        if torch.any(attention_mask.sum(dim=1) == 0):
            raise ValueError("every sequence must contain at least one real token")
        return attention_mask


__all__ = [
    "BeliefModel",
    "DROPOUT",
    "HIDDEN_SIZE",
    "MAX_SEQUENCE_LENGTH",
    "NUM_HEADS",
    "NUM_LAYERS",
]
