"""Causal backbone for observer-conditioned Werewolf belief prediction.

The model consumes one structured public-event token sequence. It reuses the
existing player/action/object embeddings and adds only event-type, public
phase, and scalar day fields:

    subject_ids
    action_ids
    object_ids

Speech-action positions still carry ``[subject, action, object]``. Event
boundaries and public system facts may leave those fields at padding zero.

The causal backbone is selected explicitly between a randomly initialized
Hugging Face ``Qwen2Model`` and a direct stack of Hugging Face ``GPT2Block``
modules. Both consume the same structured action embeddings. No pretrained
weights or tokenizer are used.

The readout adds cyclic observer-relative speaker, subject, and object/target
relation embeddings to the public hidden sequence, then uses one shared query
per observer. Each query directly predicts a distribution over the seven
canonical target seats.

This module does not consume raw public text, true roles, truth-derived
labels, alive masks, or private event fields.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from transformers import GPT2Config, Qwen2Config, Qwen2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.pytorch_utils import Conv1D

from werewolf.models.twd_tom.schema import (
    ACTION_TO_ID,
    NUM_PLAYERS,
    NONE_TOKEN,
    PLAYER_TO_ID,
)
from werewolf.models.twd_tom.public_events import (
    PHASE_TO_ID,
    STRUCTURED_TOKEN_TO_ID,
)

QWEN2_BACKBONE_NAME = "qwen2_model"
GPT2_BLOCK_BACKBONE_NAME = "gpt2_block"
SUPPORTED_BACKBONE_NAMES = (
    QWEN2_BACKBONE_NAME,
    GPT2_BLOCK_BACKBONE_NAME,
)
HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 768
NUM_HIDDEN_LAYERS = 4
NUM_ATTENTION_HEADS = 8
NUM_KEY_VALUE_HEADS = 4
ATTENTION_DROPOUT = 0.1
RMS_NORM_EPS = 1e-6
GPT2_LAYER_NORM_EPS = 1e-5
GPT2_DROPOUT = 0.1
NONE_RELATIVE_PLAYER_INDEX = NUM_PLAYERS
NONE_PLAYER_ID = PLAYER_TO_ID[NONE_TOKEN]


class GPT2BlockStack(nn.Module):
    """Direct GPT-2 block stack for structured event embeddings."""

    def __init__(self, *, max_seq_len: int):
        super().__init__()
        config = GPT2Config(
            vocab_size=1,
            n_positions=max_seq_len,
            n_embd=HIDDEN_SIZE,
            n_layer=NUM_HIDDEN_LAYERS,
            n_head=NUM_ATTENTION_HEADS,
            n_inner=INTERMEDIATE_SIZE,
            activation_function="gelu_new",
            resid_pdrop=GPT2_DROPOUT,
            embd_pdrop=GPT2_DROPOUT,
            attn_pdrop=GPT2_DROPOUT,
            layer_norm_epsilon=GPT2_LAYER_NORM_EPS,
            use_cache=False,
            bos_token_id=0,
            eos_token_id=0,
            pad_token_id=0,
        )
        config._attn_implementation = "eager"
        self.position_embedding = nn.Embedding(max_seq_len, HIDDEN_SIZE)
        self.embedding_dropout = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.ModuleList(
            GPT2Block(config, layer_idx=index)
            for index in range(NUM_HIDDEN_LAYERS)
        )
        self.final_layer_norm = nn.LayerNorm(
            HIDDEN_SIZE,
            eps=config.layer_norm_epsilon,
        )
        self._reset_parameters(config.initializer_range)

    def _reset_parameters(self, initializer_range: float) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, Conv1D)):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=initializer_range,
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=initializer_range,
                )
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        residual_std = initializer_range / math.sqrt(
            2 * NUM_HIDDEN_LAYERS
        )
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(
                    parameter,
                    mean=0.0,
                    std=residual_std,
                )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        position_ids = torch.arange(
            sequence_length,
            device=hidden_states.device,
        )
        hidden_states = self.embedding_dropout(
            hidden_states + self.position_embedding(position_ids).unsqueeze(0)
        )

        causal = torch.ones(
            (sequence_length, sequence_length),
            dtype=torch.bool,
            device=hidden_states.device,
        ).tril()
        allowed = (
            causal.view(1, 1, sequence_length, sequence_length)
            & attention_mask.view(batch_size, 1, 1, sequence_length)
        )
        attention_bias = torch.zeros(
            (batch_size, 1, sequence_length, sequence_length),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        attention_bias.masked_fill_(
            ~allowed,
            torch.finfo(hidden_states.dtype).min,
        )

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_bias,
                use_cache=False,
            )
        return self.final_layer_norm(hidden_states)


def relative_player_indices(player_ids: torch.Tensor) -> torch.Tensor:
    """Map absolute player IDs to each observer's cyclic relative index."""

    if not isinstance(player_ids, torch.Tensor):
        raise TypeError("player_ids must be a tensor")
    if player_ids.ndim != 2:
        raise ValueError("player_ids must have shape [B, L]")
    if player_ids.dtype == torch.bool or torch.is_floating_point(player_ids):
        raise TypeError("player_ids must use an integer dtype")
    if torch.any(player_ids < 0) or torch.any(player_ids > NONE_PLAYER_ID):
        raise ValueError("player_ids must contain canonical player/none IDs")
    observer_indices = torch.arange(
        NUM_PLAYERS,
        device=player_ids.device,
    ).view(1, NUM_PLAYERS, 1)
    relative = (player_ids.unsqueeze(1) - 1 - observer_indices) % NUM_PLAYERS
    return torch.where(
        (player_ids.unsqueeze(1) == 0)
        | (player_ids.unsqueeze(1) == NONE_PLAYER_ID),
        torch.full_like(relative, NONE_RELATIVE_PLAYER_INDEX),
        relative,
    )


def _mapping_vocab_size(mapping: dict[str, int]) -> int:
    """Return the embedding vocabulary size for an ID mapping."""

    return max(mapping.values()) + 1


@dataclass
class ToMBeliefBackboneConfig:
    """Configuration for the subjective belief backbone."""

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


class ToMBeliefBackbone(nn.Module):
    """Predict one seven-player belief distribution for every observer."""

    def __init__(
        self,
        config: ToMBeliefBackboneConfig | None = None,
        *,
        backbone_name: str = QWEN2_BACKBONE_NAME,
    ):
        super().__init__()
        if backbone_name not in SUPPORTED_BACKBONE_NAMES:
            raise ValueError(
                "backbone_name must be one of "
                f"{SUPPORTED_BACKBONE_NAMES}"
            )
        self.backbone_name = backbone_name

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

        if self.backbone_name == QWEN2_BACKBONE_NAME:
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
        else:
            self.transformer = GPT2BlockStack(
                max_seq_len=self.config.max_seq_len
            )

        self.observer_embedding = nn.Embedding(
            self.config.num_players,
            HIDDEN_SIZE,
        )
        self.speaker_relative_embedding = nn.Embedding(
            NUM_PLAYERS + 1,
            HIDDEN_SIZE,
            padding_idx=NONE_RELATIVE_PLAYER_INDEX,
        )
        self.target_relative_embedding = nn.Embedding(
            NUM_PLAYERS + 1,
            HIDDEN_SIZE,
            padding_idx=NONE_RELATIVE_PLAYER_INDEX,
        )
        self.object_relative_embedding = nn.Embedding(
            NUM_PLAYERS + 1,
            HIDDEN_SIZE,
            padding_idx=NONE_RELATIVE_PLAYER_INDEX,
        )
        self.relation_flag_projection = nn.Linear(
            3,
            HIDDEN_SIZE,
            bias=False,
        )
        self.observer_query_attention = nn.MultiheadAttention(
            embed_dim=HIDDEN_SIZE,
            num_heads=NUM_ATTENTION_HEADS,
            batch_first=True,
        )
        self.observer_query_layer_norm = nn.LayerNorm(HIDDEN_SIZE)

        self.output_projection = nn.Linear(
            HIDDEN_SIZE,
            self.config.num_players,
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

        for embedding in (
            self.speaker_relative_embedding,
            self.target_relative_embedding,
            self.object_relative_embedding,
        ):
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                embedding.weight[NONE_RELATIVE_PLAYER_INDEX].zero_()
        nn.init.normal_(
            self.relation_flag_projection.weight,
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
        boundary_indices: torch.Tensor | None = None,
        boundary_valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode public actions and predict observer belief logits.

        Args:
            subject_ids: Integer tensor with shape ``[B, T]``.
            action_ids: Integer tensor with shape ``[B, T]``.
            object_ids: Integer tensor with shape ``[B, T]``.
            attention_mask: Optional tensor with shape ``[B, T]``. Real
                actions use one and right-padding positions use zero. When
                omitted, padding is inferred from all-zero action triplets.

        Returns:
            A dictionary with ``hidden_states`` (``[B, T, 256]``),
            ``pooled_hidden_state`` (``[B, 256]``), and ``belief_logits``.
            The logits have shape ``[B, 7, 7]`` for ordinary inference and
            ``[B, Q, 7, 7]`` when dense PRE boundaries are supplied.
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
            raise ValueError(
                "observer query attention requires a non-empty public history"
            )

        if self.backbone_name == QWEN2_BACKBONE_NAME:
            hidden_states = self.transformer(
                inputs_embeds=hidden_states,
                attention_mask=safe_attention_mask.long(),
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
        else:
            hidden_states = self.transformer(
                hidden_states,
                attention_mask=safe_attention_mask,
            )

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
        observer_queries = (
            self.observer_embedding(observer_ids)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        speaker_token_mask = (
            (event_type_ids == STRUCTURED_TOKEN_TO_ID["turn_start"])
            | (event_type_ids == STRUCTURED_TOKEN_TO_ID["public_speech"])
        )
        target_token_mask = (
            (event_type_ids == STRUCTURED_TOKEN_TO_ID["speech_action"])
            | (event_type_ids == STRUCTURED_TOKEN_TO_ID["vote"])
        )
        no_player = torch.zeros_like(subject_ids)
        speaker_relative = relative_player_indices(
            torch.where(speaker_token_mask, subject_ids, no_player)
        )
        target_relative = relative_player_indices(
            torch.where(target_token_mask, subject_ids, no_player)
        )
        object_relative = relative_player_indices(object_ids)
        relation_flags = torch.stack(
            (
                speaker_relative == 0,
                target_relative == 0,
                object_relative == 0,
            ),
            dim=-1,
        ).to(dtype=hidden_states.dtype)
        relative_public_hidden_states = (
            hidden_states.unsqueeze(1)
            + self.speaker_relative_embedding(speaker_relative)
            + self.target_relative_embedding(target_relative)
            + self.object_relative_embedding(object_relative)
            + self.relation_flag_projection(relation_flags)
        )
        relative_public_hidden_states = (
            relative_public_hidden_states
            * safe_attention_mask[:, None, :, None].to(
                dtype=hidden_states.dtype
            )
        )
        sequence_length = hidden_states.shape[1]
        flattened_relative_hidden = relative_public_hidden_states.reshape(
            batch_size * self.config.num_players,
            sequence_length,
            HIDDEN_SIZE,
        )
        flattened_padding_mask = (
            (~safe_attention_mask)
            .unsqueeze(1)
            .expand(-1, self.config.num_players, -1)
            .reshape(batch_size * self.config.num_players, sequence_length)
        )

        if boundary_indices is None and boundary_valid_mask is None:
            flattened_queries = observer_queries.reshape(
                batch_size * self.config.num_players,
                1,
                HIDDEN_SIZE,
            )
            observer_context, _ = self.observer_query_attention(
                query=flattened_queries,
                key=flattened_relative_hidden,
                value=flattened_relative_hidden,
                key_padding_mask=flattened_padding_mask,
                need_weights=False,
            )
            observer_context = observer_context.reshape(
                batch_size,
                self.config.num_players,
                HIDDEN_SIZE,
            )
            observer_hidden_states = self.observer_query_layer_norm(
                observer_queries + observer_context
            )
        else:
            boundary_indices, boundary_valid_mask = self._validate_dense_boundaries(
                boundary_indices=boundary_indices,
                boundary_valid_mask=boundary_valid_mask,
                batch_size=batch_size,
                sequence_length=sequence_length,
                last_valid_indices=last_valid_indices,
                device=subject_ids.device,
            )
            boundary_count = boundary_indices.shape[1]
            dense_queries = observer_queries.unsqueeze(2).expand(
                -1, -1, boundary_count, -1
            )
            flattened_queries = dense_queries.reshape(
                batch_size * self.config.num_players,
                boundary_count,
                HIDDEN_SIZE,
            )
            positions = torch.arange(
                sequence_length,
                device=subject_ids.device,
            ).view(1, 1, 1, sequence_length)
            effective_boundaries = torch.where(
                boundary_valid_mask,
                boundary_indices,
                torch.zeros_like(boundary_indices),
            )
            prefix_mask = positions > effective_boundaries.view(
                batch_size, 1, boundary_count, 1
            )
            prefix_mask = prefix_mask.expand(
                -1, self.config.num_players, -1, -1
            )
            prefix_mask = (
                prefix_mask.unsqueeze(2)
                .expand(
                    -1,
                    -1,
                    self.observer_query_attention.num_heads,
                    -1,
                    -1,
                )
                .reshape(
                    batch_size
                    * self.config.num_players
                    * self.observer_query_attention.num_heads,
                    boundary_count,
                    sequence_length,
                )
            )
            observer_context, _ = self.observer_query_attention(
                query=flattened_queries,
                key=flattened_relative_hidden,
                value=flattened_relative_hidden,
                key_padding_mask=flattened_padding_mask,
                attn_mask=prefix_mask,
                need_weights=False,
            )
            observer_context = observer_context.reshape(
                batch_size,
                self.config.num_players,
                boundary_count,
                HIDDEN_SIZE,
            )
            dense_queries = dense_queries + observer_context
            dense_queries = self.observer_query_layer_norm(dense_queries)
            dense_queries = dense_queries * boundary_valid_mask[:, None, :, None].to(
                dtype=dense_queries.dtype
            )
            observer_hidden_states = dense_queries.transpose(1, 2)

        belief_logits = self.output_projection(observer_hidden_states)
        return {
            "hidden_states": hidden_states,
            "pooled_hidden_state": pooled_hidden_state,
            "observer_hidden_states": observer_hidden_states,
            "relative_public_hidden_states": relative_public_hidden_states,
            "belief_logits": belief_logits,
        }

    @staticmethod
    def _validate_dense_boundaries(
        *,
        boundary_indices: torch.Tensor | None,
        boundary_valid_mask: torch.Tensor | None,
        batch_size: int,
        sequence_length: int,
        last_valid_indices: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if boundary_indices is None or boundary_valid_mask is None:
            raise ValueError(
                "boundary_indices and boundary_valid_mask must be supplied together"
            )
        if not isinstance(boundary_indices, torch.Tensor):
            raise TypeError("boundary_indices must be a tensor")
        if not isinstance(boundary_valid_mask, torch.Tensor):
            raise TypeError("boundary_valid_mask must be a tensor")
        if boundary_indices.ndim != 2 or boundary_indices.shape[0] != batch_size:
            raise ValueError("boundary_indices must have shape [B, Q]")
        if boundary_indices.shape[1] <= 0:
            raise ValueError("dense supervision requires at least one boundary")
        if boundary_valid_mask.shape != boundary_indices.shape:
            raise ValueError("boundary_valid_mask must match boundary_indices")
        if boundary_indices.dtype != torch.long:
            raise TypeError("boundary_indices must use torch.long")
        if boundary_valid_mask.dtype is not torch.bool:
            raise TypeError("boundary_valid_mask must use torch.bool")
        indices = boundary_indices.to(device=device)
        valid = boundary_valid_mask.to(device=device)
        if torch.any(valid.sum(dim=1) == 0):
            raise ValueError("every dense game requires at least one valid boundary")
        if torch.any(indices[~valid] != 0):
            raise ValueError("padded boundary indices must equal zero")
        if torch.any(indices[valid] < 0) or torch.any(indices[valid] >= sequence_length):
            raise ValueError("valid boundary indices must address the event sequence")
        if torch.any(
            valid & (indices > last_valid_indices.unsqueeze(1))
        ):
            raise ValueError("dense boundaries cannot address right padding")
        if indices.shape[1] > 1:
            if torch.any((~valid[:, :-1]) & valid[:, 1:]):
                raise ValueError("boundary_valid_mask must use right padding")
            adjacent_valid = valid[:, :-1] & valid[:, 1:]
            if torch.any(adjacent_valid & (indices[:, 1:] <= indices[:, :-1])):
                raise ValueError("valid boundary indices must be strictly increasing")
        return indices, valid

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
    "GPT2_BLOCK_BACKBONE_NAME",
    "GPT2BlockStack",
    "HIDDEN_SIZE",
    "NONE_RELATIVE_PLAYER_INDEX",
    "NONE_PLAYER_ID",
    "QWEN2_BACKBONE_NAME",
    "SUPPORTED_BACKBONE_NAMES",
    "ToMBeliefBackboneConfig",
    "ToMBeliefBackbone",
    "relative_player_indices",
]
