"""Causal structured-prefix backbone for Public Belief Matrix V1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from werewolf.models.twd_tom.belief_backbone import (
    GPT2BlockStack,
    HIDDEN_SIZE,
    NUM_ATTENTION_HEADS,
    NUM_HIDDEN_LAYERS,
)
from werewolf.models.twd_tom.public_events import (
    PHASE_TO_ID,
    STRUCTURED_TOKEN_TO_ID,
)
from werewolf.models.twd_tom.schema import (
    ACTION_TO_ID,
    NUM_PLAYERS,
    PLAYER_TO_ID,
)


def _vocabulary_size(mapping: dict[str, int]) -> int:
    """Return the cardinality of a canonical zero-based ID space."""

    return max(mapping.values()) + 1


@dataclass(frozen=True)
class PublicBeliefMatrixBackboneConfig:
    """Fixed V1 architecture with a configurable maximum prefix length."""

    num_players: int = NUM_PLAYERS
    max_seq_len: int = 256

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_players, bool)
            or not isinstance(self.num_players, int)
            or self.num_players != NUM_PLAYERS
        ):
            raise ValueError(f"num_players must equal {NUM_PLAYERS}")
        if (
            isinstance(self.max_seq_len, bool)
            or not isinstance(self.max_seq_len, int)
            or self.max_seq_len <= 0
        ):
            raise ValueError("max_seq_len must be a positive integer")


class PublicBeliefMatrixBackbone(nn.Module):
    """Predict one complete observer-by-player suspicion matrix per prefix."""

    def __init__(
        self,
        config: PublicBeliefMatrixBackboneConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or PublicBeliefMatrixBackboneConfig()

        self.subject_embedding = nn.Embedding(
            _vocabulary_size(PLAYER_TO_ID), HIDDEN_SIZE, padding_idx=0
        )
        self.action_embedding = nn.Embedding(
            _vocabulary_size(ACTION_TO_ID), HIDDEN_SIZE, padding_idx=0
        )
        self.object_embedding = nn.Embedding(
            _vocabulary_size(PLAYER_TO_ID), HIDDEN_SIZE, padding_idx=0
        )
        self.event_type_embedding = nn.Embedding(
            _vocabulary_size(STRUCTURED_TOKEN_TO_ID),
            HIDDEN_SIZE,
            padding_idx=0,
        )
        self.phase_embedding = nn.Embedding(
            _vocabulary_size(PHASE_TO_ID), HIDDEN_SIZE, padding_idx=0
        )
        self.day_projection = nn.Linear(1, HIDDEN_SIZE, bias=False)
        self.transformer = GPT2BlockStack(max_seq_len=self.config.max_seq_len)
        self.matrix_projection = nn.Linear(
            HIDDEN_SIZE,
            self.config.num_players * self.config.num_players,
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for embedding in (
            self.subject_embedding,
            self.action_embedding,
            self.object_embedding,
            self.event_type_embedding,
            self.phase_embedding,
        ):
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                embedding.weight[0].zero_()
        nn.init.normal_(self.day_projection.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.matrix_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.matrix_projection.bias)

    def forward(
        self,
        subject_ids: torch.Tensor,
        action_ids: torch.Tensor,
        object_ids: torch.Tensor,
        event_type_ids: torch.Tensor,
        phase_ids: torch.Tensor,
        day_values: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Encode one batch of Stage 1 visible-prefix feature tensors."""

        attention_mask = self._validate_inputs(
            subject_ids=subject_ids,
            action_ids=action_ids,
            object_ids=object_ids,
            event_type_ids=event_type_ids,
            phase_ids=phase_ids,
            day_values=day_values,
            attention_mask=attention_mask,
        )
        hidden_states = (
            self.subject_embedding(subject_ids)
            + self.action_embedding(action_ids)
            + self.object_embedding(object_ids)
            + self.event_type_embedding(event_type_ids)
            + self.phase_embedding(phase_ids)
            + self.day_projection(day_values.unsqueeze(-1))
        )
        hidden_states = self.transformer(
            hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states * attention_mask.unsqueeze(-1).to(
            dtype=hidden_states.dtype
        )

        sequence_positions = torch.arange(
            subject_ids.shape[1], device=subject_ids.device
        ).unsqueeze(0)
        last_valid_indices = torch.where(
            attention_mask,
            sequence_positions,
            torch.full_like(sequence_positions, -1),
        ).max(dim=1).values
        batch_indices = torch.arange(
            subject_ids.shape[0], device=subject_ids.device
        )
        pooled_hidden_state = hidden_states[batch_indices, last_valid_indices]
        matrix_logits = self.matrix_projection(pooled_hidden_state).reshape(
            subject_ids.shape[0],
            self.config.num_players,
            self.config.num_players,
        )
        return {
            "hidden_states": hidden_states,
            "pooled_hidden_state": pooled_hidden_state,
            "matrix_logits": matrix_logits,
            "matrix_probabilities": torch.softmax(matrix_logits, dim=-1),
        }

    def _validate_inputs(
        self,
        *,
        subject_ids: torch.Tensor,
        action_ids: torch.Tensor,
        object_ids: torch.Tensor,
        event_type_ids: torch.Tensor,
        phase_ids: torch.Tensor,
        day_values: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        tensors = {
            "subject_ids": subject_ids,
            "action_ids": action_ids,
            "object_ids": object_ids,
            "event_type_ids": event_type_ids,
            "phase_ids": phase_ids,
            "day_values": day_values,
            "attention_mask": attention_mask,
        }
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
            if tensor.ndim != 2:
                raise ValueError(f"{name} must have shape [B, L]")
        expected_shape = subject_ids.shape
        if any(tensor.shape != expected_shape for tensor in tensors.values()):
            raise ValueError("all feature tensors must have the same [B, L] shape")
        batch_size, sequence_length = expected_shape
        if batch_size <= 0 or sequence_length <= 0:
            raise ValueError("feature tensors must have positive B and L")
        if sequence_length > self.config.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")
        if any(tensor.device != subject_ids.device for tensor in tensors.values()):
            raise ValueError("all feature tensors must use the same device")

        id_tensors = {
            "subject_ids": (subject_ids, _vocabulary_size(PLAYER_TO_ID)),
            "action_ids": (action_ids, _vocabulary_size(ACTION_TO_ID)),
            "object_ids": (object_ids, _vocabulary_size(PLAYER_TO_ID)),
            "event_type_ids": (
                event_type_ids,
                _vocabulary_size(STRUCTURED_TOKEN_TO_ID),
            ),
            "phase_ids": (phase_ids, _vocabulary_size(PHASE_TO_ID)),
        }
        for name, (tensor, vocabulary_size) in id_tensors.items():
            if tensor.dtype == torch.bool or torch.is_floating_point(tensor):
                raise TypeError(f"{name} must use an integer dtype")
            if torch.any(tensor < 0) or torch.any(tensor >= vocabulary_size):
                raise ValueError(f"{name} contains an out-of-range ID")
        if not torch.is_floating_point(day_values):
            raise TypeError("day_values must use a floating-point dtype")
        if not torch.isfinite(day_values).all():
            raise ValueError("day_values must contain only finite values")
        if attention_mask.dtype == torch.bool:
            bool_mask = attention_mask
        elif not torch.is_floating_point(attention_mask):
            if torch.any((attention_mask != 0) & (attention_mask != 1)):
                raise ValueError("attention_mask must contain only zero or one")
            bool_mask = attention_mask.bool()
        else:
            raise TypeError("attention_mask must use bool or an integer dtype")
        if torch.any(bool_mask.sum(dim=1) == 0):
            raise ValueError("every history prefix must contain a valid token")
        return bool_mask


__all__ = [
    "HIDDEN_SIZE",
    "NUM_ATTENTION_HEADS",
    "NUM_HIDDEN_LAYERS",
    "PublicBeliefMatrixBackbone",
    "PublicBeliefMatrixBackboneConfig",
]
