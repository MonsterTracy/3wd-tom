"""Stateless strict-PRE inference for the functional ToM belief model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from werewolf.models.twd_tom.action_features import PublicEventFeatureBuilder
from werewolf.models.twd_tom.belief_backbone import ToMBeliefBackbone
from werewolf.models.twd_tom.checkpoint import (
    MODEL_OUTPUT,
    build_model_from_checkpoint,
    load_checkpoint,
)
from werewolf.models.twd_tom.losses import masked_belief_probabilities
from werewolf.models.twd_tom.public_events import (
    completed_pre_speech_public_events,
)
from werewolf.models.twd_tom.schema import NUM_PLAYERS


class PrefixBeliefPredictor:
    """Recompute one relative-suspicion matrix from a full PRE prefix."""

    def __init__(self, model: ToMBeliefBackbone):
        if not isinstance(model, ToMBeliefBackbone):
            raise TypeError("model must be a ToMBeliefBackbone")
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.feature_builder = PublicEventFeatureBuilder(
            max_seq_len=model.config.max_seq_len,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: torch.device | str,
    ) -> "PrefixBeliefPredictor":
        resolved_device = torch.device(device)
        checkpoint = load_checkpoint(checkpoint_path)
        model = build_model_from_checkpoint(
            checkpoint,
            device=resolved_device,
        )
        return cls(model)

    def encode_prefix(
        self,
        public_events: Any,
        speech_annotations: Any,
        *,
        speaker_id: Any,
    ) -> dict[str, torch.Tensor]:
        """Encode exactly the completed events preceding the current turn."""

        completed_events = completed_pre_speech_public_events(
            public_events,
            speaker_id=speaker_id,
        )
        return self.feature_builder.encode_events(
            completed_events,
            speech_annotations,
        )

    @torch.inference_mode()
    def predict(
        self,
        public_events: Any,
        speech_annotations: Any,
        *,
        speaker_id: Any,
    ) -> dict[str, torch.Tensor]:
        """Return detached ``[7, 7]`` logits and row-normalized matrix."""

        features = self.encode_prefix(
            public_events,
            speech_annotations,
            speaker_id=speaker_id,
        )
        batch = {
            field_name: tensor.unsqueeze(0).to(self.device)
            for field_name, tensor in features.items()
        }
        logits = self.model(**batch)[MODEL_OUTPUT]
        diagonal_target_mask = (~torch.eye(
            NUM_PLAYERS,
            dtype=torch.bool,
            device=self.device,
        )).unsqueeze(0)
        matrix = masked_belief_probabilities(
            logits,
            diagonal_target_mask,
        )
        return {
            "belief_logits": logits[0].detach(),
            "belief_matrix": matrix[0].detach(),
        }


__all__ = ["PrefixBeliefPredictor"]
