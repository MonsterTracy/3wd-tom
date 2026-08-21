"""Archived formal Theory-of-Mind input and collection contracts."""

from archive.legacy_tom.werewolf.models.tom.collection import Collector
from archive.legacy_tom.werewolf.models.tom.dataset import (
    TomDataset,
    collate_batch,
    encode_sample,
)
from archive.legacy_tom.werewolf.models.tom.losses import (
    masked_soft_target_cross_entropy,
)
from archive.legacy_tom.werewolf.models.tom.model import BeliefModel
from archive.legacy_tom.werewolf.models.tom.public_history import build_model_input
from archive.legacy_tom.werewolf.models.tom.reporter import BeliefReporter
from archive.legacy_tom.werewolf.models.tom.schema import (
    ACTION_NAMES,
    ACTION_TO_ID,
    EPISODE_CONTEXTS,
    SpeechAction,
)
from archive.legacy_tom.werewolf.models.tom.targets import (
    materialize_target,
    suspicion_to_row,
)

__all__ = [
    "ACTION_NAMES",
    "ACTION_TO_ID",
    "BeliefReporter",
    "BeliefModel",
    "Collector",
    "EPISODE_CONTEXTS",
    "SpeechAction",
    "TomDataset",
    "build_model_input",
    "collate_batch",
    "encode_sample",
    "materialize_target",
    "masked_soft_target_cross_entropy",
    "suspicion_to_row",
]
