"""Checkpoint contract and restoration for the functional ToM predictor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from werewolf.models.twd_tom.belief_backbone import (
    SUPPORTED_BACKBONE_NAMES,
    ToMBeliefBackbone,
    ToMBeliefBackboneConfig,
)
from werewolf.models.twd_tom.dataset import (
    CYCLIC_ROTATION_VERSION,
    MODEL_INPUT_SCOPE,
    TARGET_CONVERSION,
    TARGET_SEMANTICS,
)
from werewolf.models.twd_tom.public_events import (
    PHASE_TO_ID,
    PUBLIC_EVENT_SCHEMA_VERSION,
    STRUCTURED_TOKEN_TO_ID,
)
from werewolf.models.twd_tom.samples import SAMPLE_SCHEMA_VERSION
from werewolf.models.twd_tom.schema import ACTION_NAMES, ACTION_TO_ID, NUM_PLAYERS


OBJECTIVE = "observer_conditioned_belief_distribution_v1"
MODEL_OUTPUT = "belief_logits"


def checkpoint_task_contract() -> dict[str, Any]:
    """Return the single frozen task contract stored in every checkpoint."""

    return {
        "objective": OBJECTIVE,
        "model_input_scope": MODEL_INPUT_SCOPE,
        "model_output": MODEL_OUTPUT,
        "output_shape": [NUM_PLAYERS, NUM_PLAYERS],
        "target_semantics": TARGET_SEMANTICS,
        "target_conversion": TARGET_CONVERSION,
        "train_player_augmentation": CYCLIC_ROTATION_VERSION,
    }


def result_model_config(model: ToMBeliefBackbone) -> dict[str, Any]:
    return asdict(model.config)


def load_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain a dictionary")
    return checkpoint


def build_model_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    device: torch.device,
) -> ToMBeliefBackbone:
    """Strictly restore a model whose schemas match the current project."""

    backbone_name = checkpoint.get("backbone")
    if backbone_name not in SUPPORTED_BACKBONE_NAMES:
        raise ValueError(
            "checkpoint backbone mismatch: expected one of "
            f"{SUPPORTED_BACKBONE_NAMES!r}, got {backbone_name!r}"
        )
    expected = {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        **checkpoint_task_contract(),
        "public_event_schema_version": PUBLIC_EVENT_SCHEMA_VERSION,
        "speech_action_count": len(ACTION_NAMES),
        "speech_action_to_id": dict(ACTION_TO_ID),
        "structured_token_to_id": dict(STRUCTURED_TOKEN_TO_ID),
        "public_phase_to_id": dict(PHASE_TO_ID),
    }
    for field_name, expected_value in expected.items():
        if checkpoint.get(field_name) != expected_value:
            raise ValueError(
                f"checkpoint {field_name} mismatch: expected "
                f"{expected_value!r}, got {checkpoint.get(field_name)!r}"
            )
    raw_model_config = checkpoint.get("model_config")
    if not isinstance(raw_model_config, Mapping):
        raise TypeError("checkpoint has no valid model_config")
    try:
        model_config = ToMBeliefBackboneConfig(**dict(raw_model_config))
    except TypeError as exc:
        raise ValueError("checkpoint model_config is incompatible") from exc
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise TypeError("checkpoint has no valid model_state_dict")
    model = ToMBeliefBackbone(model_config, backbone_name=backbone_name)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError("checkpoint state_dict is incompatible") from exc
    return model.to(device).eval()


__all__ = [
    "MODEL_OUTPUT",
    "OBJECTIVE",
    "build_model_from_checkpoint",
    "checkpoint_task_contract",
    "load_checkpoint",
    "result_model_config",
]
