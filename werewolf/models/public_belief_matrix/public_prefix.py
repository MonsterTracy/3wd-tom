"""Shared model-visible public-prefix contract for Public Belief Matrix V1."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from werewolf.models.twd_tom.action_features import PublicEventFeatureBuilder


def build_public_belief_matrix_visible_prefix(
    public_events: Sequence[Any],
    *,
    max_seq_len: int = 256,
) -> dict[str, torch.Tensor]:
    """Build the canonical raw-text-free, deterministically truncated prefix."""

    return PublicEventFeatureBuilder(max_seq_len=max_seq_len).encode_events(
        public_events
    )


__all__ = ["build_public_belief_matrix_visible_prefix"]
