"""Literal-shape GPT2Block reference for Classic7 ONUW parity."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from transformers import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.pytorch_utils import Conv1D

from werewolf.models.twd_tom.onuw_parity_dataset import (
    EMOTION_TO_ID,
    PHASE_TO_ID,
    TOKEN_TYPE_TO_ID,
)
from werewolf.models.twd_tom.onuw_parity_protocol import (
    CLASSIC7_PUBLIC_EVENTS,
    CONTENT_PROFILES,
    MODALITY_PROFILES,
    ONUW_AGENT_DECLARED_MULTIMODAL,
)
from werewolf.models.twd_tom.schema import ACTION_TO_ID, PLAYER_TO_ID


HIDDEN_SIZE = 512
NUM_LAYERS = 8
NUM_HEADS = 8
INTERMEDIATE_SIZE = 2048
NUM_PLAYERS = 7


@dataclass(frozen=True)
class OnuwParityModelConfig:
    """Capacity is explicit; it must be chosen after sequence-length audit."""

    max_positions: int
    content_profile: str
    modality_profile: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_positions, bool)
            or not isinstance(self.max_positions, int)
            or self.max_positions <= 0
        ):
            raise ValueError("max_positions must be a positive audited capacity")
        if self.content_profile not in CONTENT_PROFILES:
            raise ValueError("unsupported content_profile")
        if self.modality_profile not in MODALITY_PROFILES:
            raise ValueError("unsupported modality_profile")


def _vocab_size(mapping: dict[str, int]) -> int:
    return max(mapping.values()) + 1


class OnuwParityBeliefModel(nn.Module):
    """Public sequence -> hidden at PRE positions -> direct [7, 7] head."""

    def __init__(self, config: OnuwParityModelConfig) -> None:
        super().__init__()
        self.config = config
        self.subject_embedding = nn.Embedding(
            _vocab_size(PLAYER_TO_ID), HIDDEN_SIZE, padding_idx=0
        )
        self.action_embedding = nn.Embedding(
            _vocab_size(ACTION_TO_ID), HIDDEN_SIZE, padding_idx=0
        )
        self.object_embedding = nn.Embedding(
            _vocab_size(PLAYER_TO_ID), HIDDEN_SIZE, padding_idx=0
        )
        self.bos_embedding = nn.Parameter(torch.zeros(HIDDEN_SIZE))
        self.face_embedding = nn.Embedding(
            _vocab_size(EMOTION_TO_ID), HIDDEN_SIZE, padding_idx=0
        )
        self.tone_embedding = nn.Embedding(
            _vocab_size(EMOTION_TO_ID), HIDDEN_SIZE, padding_idx=0
        )
        self.token_type_embedding = nn.Embedding(
            _vocab_size(TOKEN_TYPE_TO_ID), HIDDEN_SIZE, padding_idx=0
        )
        self.phase_embedding = nn.Embedding(
            _vocab_size(PHASE_TO_ID), HIDDEN_SIZE, padding_idx=0
        )
        self.day_projection = nn.Linear(1, HIDDEN_SIZE, bias=False)
        self.position_embedding = nn.Embedding(config.max_positions, HIDDEN_SIZE)

        gpt_config = GPT2Config(
            vocab_size=1,
            n_positions=config.max_positions,
            n_embd=HIDDEN_SIZE,
            n_layer=NUM_LAYERS,
            n_head=NUM_HEADS,
            n_inner=INTERMEDIATE_SIZE,
            activation_function="gelu_new",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            use_cache=False,
            bos_token_id=0,
            eos_token_id=0,
            pad_token_id=0,
        )
        gpt_config._attn_implementation = "eager"
        self.embedding_dropout = nn.Dropout(gpt_config.embd_pdrop)
        self.blocks = nn.ModuleList(
            GPT2Block(gpt_config, layer_idx=index)
            for index in range(NUM_LAYERS)
        )
        self.final_layer_norm = nn.LayerNorm(
            HIDDEN_SIZE, eps=gpt_config.layer_norm_epsilon
        )
        self.matrix_head = nn.Linear(HIDDEN_SIZE, NUM_PLAYERS * NUM_PLAYERS)
        self._reset_parameters(gpt_config.initializer_range)

    def _reset_parameters(self, initializer_range: float) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, Conv1D)):
                nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.bos_embedding, mean=0.0, std=initializer_range)
        residual_std = initializer_range / math.sqrt(2 * NUM_LAYERS)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=residual_std)

    @staticmethod
    def _validate_masks(
        *,
        token_attention_mask: torch.Tensor,
        query_valid_mask: torch.Tensor,
        observer_alive_mask: torch.Tensor,
    ) -> None:
        masks = {
            "token_attention_mask": token_attention_mask,
            "query_valid_mask": query_valid_mask,
            "observer_alive_mask": observer_alive_mask,
        }
        for name, mask in masks.items():
            if not isinstance(mask, torch.Tensor) or mask.dtype is not torch.bool:
                raise TypeError(f"{name} must be a distinct torch.bool tensor")
        if token_attention_mask is query_valid_mask:
            raise ValueError("token_attention_mask and query_valid_mask cannot alias")
        if query_valid_mask is observer_alive_mask:
            raise ValueError("query_valid_mask and observer_alive_mask cannot alias")

    def forward(
        self,
        *,
        subject_ids: torch.Tensor,
        action_ids: torch.Tensor,
        object_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        face_ids: torch.Tensor,
        tone_ids: torch.Tensor,
        phase_ids: torch.Tensor,
        day_values: torch.Tensor,
        token_attention_mask: torch.Tensor,
        query_positions: torch.Tensor,
        query_valid_mask: torch.Tensor,
        observer_alive_mask: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_masks(
            token_attention_mask=token_attention_mask,
            query_valid_mask=query_valid_mask,
            observer_alive_mask=observer_alive_mask,
        )
        token_shape = subject_ids.shape
        if subject_ids.ndim != 2:
            raise ValueError("token features must have shape [B, L]")
        for name, value in {
            "action_ids": action_ids,
            "object_ids": object_ids,
            "token_type_ids": token_type_ids,
            "face_ids": face_ids,
            "tone_ids": tone_ids,
            "phase_ids": phase_ids,
            "day_values": day_values,
            "token_attention_mask": token_attention_mask,
        }.items():
            if value.shape != token_shape:
                raise ValueError(f"{name} must match token feature shape")
        batch_size, sequence_length = token_shape
        if sequence_length > self.config.max_positions:
            raise ValueError(
                "sequence exceeds audited max_positions; silent truncation is forbidden"
            )
        if query_positions.ndim != 2 or query_valid_mask.shape != query_positions.shape:
            raise ValueError("query tensors must have shape [B, Q]")
        if query_positions.shape[0] != batch_size:
            raise ValueError("query batch dimension must match tokens")
        expected_alive = (*query_positions.shape, NUM_PLAYERS)
        if observer_alive_mask.shape != expected_alive:
            raise ValueError("observer_alive_mask must have shape [B, Q, 7]")
        valid_positions = query_positions[query_valid_mask]
        if valid_positions.numel() and (
            torch.any(valid_positions < 0)
            or torch.any(valid_positions >= sequence_length)
        ):
            raise ValueError("valid query_positions must index the token sequence")

        hidden = (
            self.subject_embedding(subject_ids)
            + self.action_embedding(action_ids)
            + self.object_embedding(object_ids)
        )
        bos_id = TOKEN_TYPE_TO_ID["bos"]
        hidden = hidden + (token_type_ids == bos_id).unsqueeze(-1) * self.bos_embedding
        if self.config.modality_profile == ONUW_AGENT_DECLARED_MULTIMODAL:
            hidden = (
                hidden
                + self.face_embedding(face_ids)
                + self.tone_embedding(tone_ids)
            )
        if self.config.content_profile == CLASSIC7_PUBLIC_EVENTS:
            hidden = (
                hidden
                + self.token_type_embedding(token_type_ids)
                + self.phase_embedding(phase_ids)
                + self.day_projection(day_values.unsqueeze(-1))
            )
        positions = torch.arange(sequence_length, device=hidden.device)
        hidden = self.embedding_dropout(
            hidden + self.position_embedding(positions).unsqueeze(0)
        )

        causal = torch.ones(
            (sequence_length, sequence_length),
            dtype=torch.bool,
            device=hidden.device,
        ).tril()
        allowed = causal.view(1, 1, sequence_length, sequence_length) & (
            token_attention_mask.view(batch_size, 1, 1, sequence_length)
        )
        attention_bias = torch.zeros(
            (batch_size, 1, sequence_length, sequence_length),
            dtype=hidden.dtype,
            device=hidden.device,
        )
        attention_bias.masked_fill_(~allowed, torch.finfo(hidden.dtype).min)
        for block in self.blocks:
            hidden = block(hidden, attention_mask=attention_bias, use_cache=False)
        hidden = self.final_layer_norm(hidden)

        safe_positions = query_positions.clamp(min=0, max=sequence_length - 1)
        gather_index = safe_positions.unsqueeze(-1).expand(-1, -1, HIDDEN_SIZE)
        query_hidden = hidden.gather(1, gather_index)
        logits = self.matrix_head(query_hidden).view(
            batch_size, query_positions.shape[1], NUM_PLAYERS, NUM_PLAYERS
        )
        return logits.masked_fill(
            ~query_valid_mask[:, :, None, None], 0.0
        )


__all__ = [
    "HIDDEN_SIZE",
    "NUM_LAYERS",
    "NUM_HEADS",
    "INTERMEDIATE_SIZE",
    "OnuwParityModelConfig",
    "OnuwParityBeliefModel",
]
