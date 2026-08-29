"""Frozen P0 contracts for the Classic7 ONUW-parity line.

This module is deliberately independent from the semantic-clean protocol.
The reference target may use an observer's legal private view, while model
features remain public-only.
"""

from __future__ import annotations

from dataclasses import dataclass


NUM_PLAYERS = 7
ONUW_CODE_REFERENCE = "onuw_code_reference_v1"
CLASSIC7_ONUW_REFERENCE = "classic7_onuw_reference_v1"
CLASSIC7_MAIN = "classic7_main_v1"

ONUW_ACTION_ONLY = "onuw_action_only"
CLASSIC7_PUBLIC_EVENTS = "classic7_public_events"
CONTENT_PROFILES = (ONUW_ACTION_ONLY, CLASSIC7_PUBLIC_EVENTS)

ONUW_AGENT_DECLARED_MULTIMODAL = "onuw_agent_declared_multimodal"
ONUW_NO_FACE_NO_TONE = "onuw_no_face_no_tone"
MODALITY_PROFILES = (
    ONUW_AGENT_DECLARED_MULTIMODAL,
    ONUW_NO_FACE_NO_TONE,
)

EMOTION_NAMES = (
    "sad",
    "anger",
    "neutral",
    "happy",
    "surprise",
    "fear",
    "disgust",
    "other",
)
ROLE_GUESS_NAMES = (
    "werewolf",
    "villager",
    "seer",
    "witch",
    "unknown",
)

ONUW_PAPER_TRAIN_RECORD = {
    "epochs": 80,
    "batch_size": 32,
    "learning_rate": 5e-5,
    "early_stopping": "validation_loss",
}


@dataclass(frozen=True)
class Classic7OnuwReferenceContract:
    """Machine-readable invariants for the P0 reference protocol."""

    protocol_id: str = CLASSIC7_ONUW_REFERENCE
    timing: str = "strict_pre"
    canonical_unit: str = "game_sequence_with_multiple_pre_queries"
    label_collector: str = "onuw_style_role_guess"
    label_information: str = "observer_legal_private_view"
    model_input: str = "public_only"
    output_shape: tuple[int, int] = (NUM_PLAYERS, NUM_PLAYERS)
    supervised_rows: str = "alive_observers_only"
    target_columns: str = "all_canonical_players_including_dead"
    empty_support: str = "uniform_over_all_players"
    self_included: bool = True
    target_column_mask: str = "none"
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    readout: str = "direct_full_matrix"
    training_reduction: str = "observer_row_micro"


REFERENCE_CONTRACT = Classic7OnuwReferenceContract()


__all__ = [
    "NUM_PLAYERS",
    "ONUW_CODE_REFERENCE",
    "CLASSIC7_ONUW_REFERENCE",
    "CLASSIC7_MAIN",
    "ONUW_ACTION_ONLY",
    "CLASSIC7_PUBLIC_EVENTS",
    "CONTENT_PROFILES",
    "ONUW_AGENT_DECLARED_MULTIMODAL",
    "ONUW_NO_FACE_NO_TONE",
    "MODALITY_PROFILES",
    "EMOTION_NAMES",
    "ROLE_GUESS_NAMES",
    "ONUW_PAPER_TRAIN_RECORD",
    "Classic7OnuwReferenceContract",
    "REFERENCE_CONTRACT",
]
