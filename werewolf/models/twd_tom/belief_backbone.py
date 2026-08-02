"""Qwen2 backbone for subjective Werewolf belief prediction.

The model consumes one structured public-event token sequence. It reuses the
existing player/action/object embeddings and adds only event-type, public
phase, and scalar day fields:

    subject_ids
    action_ids
    object_ids

Speech-action positions still carry ``[subject, action, object]``. Event
boundaries and public system facts may leave those fields at padding zero.

The causal backbone is a randomly initialized Hugging Face ``Qwen2Model`` that
receives the structured action embeddings through ``inputs_embeds``. No
pretrained weights or tokenizer are used.

The final valid hidden state is combined with one observer embedding per
player. First-order samples may additionally provide the current observer's
two seven-player hard-knowledge vectors. A single linear projection maps those
vectors into the hidden size. The shared output head then predicts one pair
distribution per observer with shape ``[batch_size, num_players, 21]``.

This module does not consume raw public text, true roles, truth-derived
labels, observer IDs, alive masks, or private event fields.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import Qwen2Config, Qwen2Model

from werewolf.models.twd_tom.schema import (
    ACTION_TO_ID,
    NUM_WOLF_PAIR_CLASSES,
    NUM_PLAYERS,
    PLAYER_TO_ID,
)
from werewolf.models.twd_tom.public_events import (
    PHASE_TO_ID,
    STRUCTURED_TOKEN_TO_ID,
)
from werewolf.models.twd_tom.belief_labels import (
    pair_probabilities_to_belief_marginals,
)

BACKBONE_NAME = "qwen2_model"
HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 768
NUM_HIDDEN_LAYERS = 4
NUM_ATTENTION_HEADS = 8
NUM_KEY_VALUE_HEADS = 4
ATTENTION_DROPOUT = 0.1
RMS_NORM_EPS = 1e-6


def _mapping_vocab_size(mapping: dict[str, int]) -> int:
    """Return the embedding vocabulary size for an ID mapping."""

    return max(mapping.values()) + 1


@dataclass
class ToMBeliefBackboneConfig:
    """Configuration for the subjective belief backbone."""

    num_players: int = NUM_PLAYERS
    pair_class_count: int = NUM_WOLF_PAIR_CLASSES
    max_seq_len: int = 256

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_players, bool)
            or not isinstance(self.num_players, int)
            or self.num_players != NUM_PLAYERS
        ):
            raise ValueError(f"num_players must equal {NUM_PLAYERS}")

        if (
            isinstance(self.pair_class_count, bool)
            or not isinstance(self.pair_class_count, int)
            or self.pair_class_count != NUM_WOLF_PAIR_CLASSES
        ):
            raise ValueError(
                "pair_class_count must equal "
                f"{NUM_WOLF_PAIR_CLASSES}"
            )

        if (
            isinstance(self.max_seq_len, bool)
            or not isinstance(self.max_seq_len, int)
            or self.max_seq_len <= 0
        ):
            raise ValueError("max_seq_len must be a positive integer")


class ToMBeliefBackbone(nn.Module):
    """Predict a complete subjective belief matrix from public actions."""

    def __init__(
        self,
        config: ToMBeliefBackboneConfig | None = None,
    ):
        super().__init__()

        self.config = (
            ToMBeliefBackboneConfig()
            if config is None
            else config
        )

        player_vocab_size = _mapping_vocab_size(PLAYER_TO_ID)
        action_vocab_size = _mapping_vocab_size(ACTION_TO_ID)
        event_type_vocab_size = _mapping_vocab_size(STRUCTURED_TOKEN_TO_ID)
        phase_vocab_size = _mapping_vocab_size(PHASE_TO_ID)

        self.subject_embedding = nn.Embedding(
            player_vocab_size,
            HIDDEN_SIZE,
            padding_idx=0,
        )
        self.action_embedding = nn.Embedding(
            action_vocab_size,
            HIDDEN_SIZE,
            padding_idx=0,
        )
        self.object_embedding = nn.Embedding(
            player_vocab_size,
            HIDDEN_SIZE,
            padding_idx=0,
        )
        self.event_type_embedding = nn.Embedding(
            event_type_vocab_size,
            HIDDEN_SIZE,
            padding_idx=0,
        )
        self.phase_embedding = nn.Embedding(
            phase_vocab_size,
            HIDDEN_SIZE,
            padding_idx=0,
        )
        self.day_projection = nn.Linear(
            1,
            HIDDEN_SIZE,
            bias=False,
        )

        # Used only when the public action history is empty.
        self.empty_history_embedding = nn.Parameter(
            torch.zeros(HIDDEN_SIZE)
        )

        self.transformer = Qwen2Model(
            Qwen2Config(
                vocab_size=1,
                hidden_size=HIDDEN_SIZE,
                intermediate_size=INTERMEDIATE_SIZE,
                num_hidden_layers=NUM_HIDDEN_LAYERS,
                num_attention_heads=NUM_ATTENTION_HEADS,
                num_key_value_heads=NUM_KEY_VALUE_HEADS,
                hidden_act="silu",
                max_position_embeddings=self.config.max_seq_len,
                attention_dropout=ATTENTION_DROPOUT,
                rms_norm_eps=RMS_NORM_EPS,
                use_cache=False,
                bos_token_id=0,
                eos_token_id=0,
                pad_token_id=0,
            )
        )

        self.observer_embedding = nn.Embedding(
            self.config.num_players,
            HIDDEN_SIZE,
        )
        self.private_knowledge_projection = nn.Linear(
            2 * self.config.num_players,
            HIDDEN_SIZE,
            bias=False,
        )

        self.output_projection = nn.Linear(
            HIDDEN_SIZE,
            self.config.pair_class_count,
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize project-owned embeddings and output layers."""

        for embedding in (
            self.subject_embedding,
            self.action_embedding,
            self.object_embedding,
            self.event_type_embedding,
            self.phase_embedding,
            self.observer_embedding,
        ):
            nn.init.normal_(
                embedding.weight,
                mean=0.0,
                std=0.02,
            )

        # Padding IDs must remain neutral.
        with torch.no_grad():
            self.subject_embedding.weight[0].zero_()
            self.action_embedding.weight[0].zero_()
            self.object_embedding.weight[0].zero_()
            self.event_type_embedding.weight[0].zero_()
            self.phase_embedding.weight[0].zero_()
        nn.init.normal_(self.day_projection.weight, mean=0.0, std=0.02)
        nn.init.normal_(
            self.private_knowledge_projection.weight,
            mean=0.0,
            std=0.02,
        )

        nn.init.normal_(
            self.empty_history_embedding,
            mean=0.0,
            std=0.02,
        )
        nn.init.normal_(
            self.output_projection.weight,
            mean=0.0,
            std=0.02,
        )
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        subject_ids: torch.Tensor,
        action_ids: torch.Tensor,
        object_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        event_type_ids: torch.Tensor | None = None,
        phase_ids: torch.Tensor | None = None,
        day_values: torch.Tensor | None = None,
        known_werewolves: torch.Tensor | None = None,
        known_non_werewolves: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode actions and predict pair distributions and marginals.

        Args:
            subject_ids: Integer tensor with shape ``[B, T]``.
            action_ids: Integer tensor with shape ``[B, T]``.
            object_ids: Integer tensor with shape ``[B, T]``.
            attention_mask: Optional tensor with shape ``[B, T]``. Real
                actions use one and right-padding positions use zero. When
                omitted, padding is inferred from all-zero action triplets.

        Returns:
            A dictionary with ``hidden_states`` (``[B, T, 256]``),
            ``pooled_hidden_state`` (``[B, 256]``), ``observer_pair_logits``
            (``[B, 7, 21]``), pair probabilities, and the derived
            ``belief_matrix`` with shape ``[B, 7, 7]``.
        """

        (
            subject_ids,
            action_ids,
            object_ids,
            event_type_ids,
            phase_ids,
            day_values,
            attention_mask,
        ) = self._validate_inputs(
            subject_ids=subject_ids,
            action_ids=action_ids,
            object_ids=object_ids,
            event_type_ids=event_type_ids,
            phase_ids=phase_ids,
            day_values=day_values,
            attention_mask=attention_mask,
        )

        batch_size = subject_ids.shape[0]

        base_embeddings = (
            self.subject_embedding(subject_ids)
            + self.action_embedding(action_ids)
            + self.object_embedding(object_ids)
            + self.event_type_embedding(event_type_ids)
            + self.phase_embedding(phase_ids)
            + self.day_projection(day_values.unsqueeze(-1))
        )

        hidden_states = base_embeddings

        # Attention cannot safely operate on a row whose every key is masked.
        # Represent an empty public history with one learned internal token.
        safe_attention_mask = attention_mask.clone()
        empty_rows = safe_attention_mask.sum(dim=1) == 0

        if empty_rows.any():
            safe_attention_mask[empty_rows, 0] = True
            hidden_states = hidden_states.clone()
            hidden_states[empty_rows, 0] = self.empty_history_embedding

        hidden_states = self.transformer(
            inputs_embeds=hidden_states,
            attention_mask=safe_attention_mask.long(),
            use_cache=False,
            return_dict=True,
        ).last_hidden_state

        # Zero right-padding positions in returned hidden states.
        hidden_states = (
            hidden_states
            * safe_attention_mask.to(
                dtype=hidden_states.dtype
            ).unsqueeze(-1)
        )

        last_valid_indices = (
            safe_attention_mask.long()
            .sum(dim=1)
            .sub(1)
            .clamp_min(0)
        )
        batch_indices = torch.arange(
            batch_size,
            device=subject_ids.device,
        )
        pooled_hidden_state = hidden_states[
            batch_indices,
            last_valid_indices,
        ]

        observer_ids = torch.arange(
            self.config.num_players,
            device=subject_ids.device,
        )
        observer_hidden_states = (
            pooled_hidden_state.unsqueeze(1)
            + self.observer_embedding(observer_ids).unsqueeze(0)
        )
        private_knowledge = self._validate_private_knowledge(
            known_werewolves=known_werewolves,
            known_non_werewolves=known_non_werewolves,
            batch_size=batch_size,
            device=subject_ids.device,
        )
        if private_knowledge is not None:
            observer_hidden_states = (
                observer_hidden_states
                + self.private_knowledge_projection(private_knowledge)
            )

        pair_logits = self.output_projection(observer_hidden_states)
        pair_probabilities = torch.softmax(
            pair_logits,
            dim=-1,
        )
        belief_matrix = pair_probabilities_to_belief_marginals(
            pair_probabilities
        )

        return {
            "hidden_states": hidden_states,
            "pooled_hidden_state": pooled_hidden_state,
            "observer_hidden_states": observer_hidden_states,
            "observer_pair_logits": pair_logits,
            "pair_logits": pair_logits,
            "pair_probabilities": pair_probabilities,
            "belief_matrix": belief_matrix,
        }

    def _validate_private_knowledge(
        self,
        *,
        known_werewolves: torch.Tensor | None,
        known_non_werewolves: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Validate and concatenate optional first-order private features."""

        if (known_werewolves is None) != (known_non_werewolves is None):
            raise ValueError(
                "known_werewolves and known_non_werewolves must be provided together"
            )
        if known_werewolves is None:
            return None

        expected_shape = (
            batch_size,
            self.config.num_players,
            self.config.num_players,
        )
        normalized = []
        for field_name, tensor in (
            ("known_werewolves", known_werewolves),
            ("known_non_werewolves", known_non_werewolves),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{field_name} must be a tensor")
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"{field_name} must have shape [B, 7, 7]"
                )
            tensor = tensor.to(device=device, dtype=torch.float32)
            if not torch.isfinite(tensor).all() or (
                (tensor != 0) & (tensor != 1)
            ).any():
                raise ValueError(f"{field_name} must contain only 0 or 1")
            normalized.append(tensor)

        if (normalized[0] * normalized[1]).any():
            raise ValueError(
                "known_werewolves and known_non_werewolves must be disjoint"
            )
        return torch.cat(normalized, dim=-1)

    def _validate_inputs(
        self,
        *,
        subject_ids: torch.Tensor,
        action_ids: torch.Tensor,
        object_ids: torch.Tensor,
        event_type_ids: torch.Tensor | None,
        phase_ids: torch.Tensor | None,
        day_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Validate shapes, IDs, and right-padding invariants."""

        tensors = {
            "subject_ids": subject_ids,
            "action_ids": action_ids,
            "object_ids": object_ids,
        }

        for field_name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{field_name} must be a tensor")
            if tensor.ndim != 2:
                raise ValueError(
                    f"{field_name} must have shape [B, T]"
                )

        expected_shape = subject_ids.shape

        if action_ids.shape != expected_shape:
            raise ValueError(
                "action_ids must have the same shape as subject_ids"
            )
        if object_ids.shape != expected_shape:
            raise ValueError(
                "object_ids must have the same shape as subject_ids"
            )
        if event_type_ids is None or phase_ids is None or day_values is None:
            raise ValueError(
                "event_type_ids, phase_ids, and day_values are required"
            )
        for field_name, tensor in {
            "event_type_ids": event_type_ids,
            "phase_ids": phase_ids,
            "day_values": day_values,
        }.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{field_name} must be a tensor")
            if tensor.shape != expected_shape:
                raise ValueError(f"{field_name} must have shape [B, T]")
        complete_speech_action = (
            subject_ids.ne(0)
            & action_ids.ne(0)
            & object_ids.ne(0)
        )

        batch_size, seq_len = expected_shape

        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        if seq_len <= 0:
            raise ValueError("sequence length must be positive")
        if seq_len > self.config.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")

        device = subject_ids.device
        subject_ids = subject_ids.to(dtype=torch.long)
        action_ids = action_ids.to(device=device, dtype=torch.long)
        object_ids = object_ids.to(device=device, dtype=torch.long)
        event_type_ids = event_type_ids.to(device=device, dtype=torch.long)
        phase_ids = phase_ids.to(device=device, dtype=torch.long)
        day_values = day_values.to(device=device, dtype=torch.float32)

        self._validate_id_range(
            subject_ids,
            field_name="subject_ids",
            vocab_size=self.subject_embedding.num_embeddings,
        )
        self._validate_id_range(
            action_ids,
            field_name="action_ids",
            vocab_size=self.action_embedding.num_embeddings,
        )
        self._validate_id_range(
            object_ids,
            field_name="object_ids",
            vocab_size=self.object_embedding.num_embeddings,
        )
        self._validate_id_range(
            event_type_ids,
            field_name="event_type_ids",
            vocab_size=self.event_type_embedding.num_embeddings,
        )
        self._validate_id_range(
            phase_ids,
            field_name="phase_ids",
            vocab_size=self.phase_embedding.num_embeddings,
        )
        if not torch.isfinite(day_values).all() or (day_values < 0).any():
            raise ValueError("day_values must be finite and non-negative")

        any_real_id = (
            subject_ids.ne(0)
            | action_ids.ne(0)
            | object_ids.ne(0)
        )
        speech_action_tokens = event_type_ids.eq(
            STRUCTURED_TOKEN_TO_ID["speech_action"]
        )
        if (speech_action_tokens & ~complete_speech_action).any():
            raise ValueError(
                "speech_action tokens require complete subject/action/object IDs"
            )
        if ((~speech_action_tokens) & action_ids.ne(0)).any():
            raise ValueError("only speech_action tokens may carry action IDs")
        structured_content = (
            any_real_id
            | event_type_ids.ne(0)
            | phase_ids.ne(0)
            | day_values.ne(0)
        )
        complete_real_token = event_type_ids.ne(0)

        if attention_mask is None:
            normalized_mask = complete_real_token
        else:
            if not isinstance(attention_mask, torch.Tensor):
                raise TypeError("attention_mask must be a tensor")
            if attention_mask.shape != expected_shape:
                raise ValueError(
                    "attention_mask must have shape [B, T]"
                )

            normalized_mask = attention_mask.to(
                device=device,
                dtype=torch.bool,
            )

            if (normalized_mask & ~complete_real_token).any():
                raise ValueError(
                    "attention_mask marks an incomplete or padding event "
                    "as valid"
                )
            if (~normalized_mask & structured_content).any():
                raise ValueError(
                    "non-zero event fields cannot appear in masked padding "
                    "positions"
                )

        if seq_len > 1:
            zero_to_one_transition = (
                normalized_mask[:, 1:].long()
                > normalized_mask[:, :-1].long()
            )
            if zero_to_one_transition.any():
                raise ValueError(
                    "attention_mask must use right padding"
                )

        return (
            subject_ids,
            action_ids,
            object_ids,
            event_type_ids,
            phase_ids,
            day_values,
            normalized_mask,
        )

    @staticmethod
    def _validate_id_range(
        tensor: torch.Tensor,
        *,
        field_name: str,
        vocab_size: int,
    ) -> None:
        if tensor.numel() == 0:
            return

        minimum = int(tensor.min().item())
        maximum = int(tensor.max().item())

        if minimum < 0 or maximum >= vocab_size:
            raise ValueError(
                f"{field_name} contains IDs outside [0, {vocab_size - 1}]"
            )


__all__ = [
    "BACKBONE_NAME",
    "HIDDEN_SIZE",
    "ToMBeliefBackboneConfig",
    "ToMBeliefBackbone",
]
