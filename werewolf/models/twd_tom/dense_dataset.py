"""Game-level dense PRE supervision for the tom-v2 belief model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from werewolf.models.twd_tom.action_features import PublicEventFeatureBuilder
from werewolf.models.twd_tom.dataset import (
    MODEL_INPUT_SCOPE,
    PRIVATE_MODEL_INPUT_SCOPE,
    TARGET_CONVERSION,
    TARGET_SEMANTICS,
    TWDToMDataset,
    cyclically_rotate_belief_sample,
    deterministic_cyclic_shift,
)
from werewolf.models.twd_tom.schema import NUM_PLAYERS
from werewolf.models.twd_tom.supervision import (
    ALL_ALIVE_SCOPE,
    SUPERVISION_SCOPES,
    normalize_observer_roles,
    rotate_observer_roles,
)


DENSE_SUPERVISION_VERSION = "game_level_dense_pre_boundary_v1"
_FEATURE_FIELDS = PublicEventFeatureBuilder.FEATURE_FIELDS


def _group_raw_samples(
    samples: Sequence[Mapping[str, Any]],
) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise TypeError("each dense dataset sample must be a mapping")
        game_id = sample.get("game_id")
        if not isinstance(game_id, str) or not game_id.strip():
            raise ValueError("each dense dataset sample requires a game_id")
        groups.setdefault(game_id, []).append(deepcopy(dict(sample)))

    ordered: list[list[dict[str, Any]]] = []
    for game_id in sorted(groups):
        group = groups[game_id]
        steps = [sample.get("step_idx") for sample in group]
        if any(isinstance(step, bool) or not isinstance(step, int) for step in steps):
            raise TypeError(f"dense game {game_id} step_idx values must be integers")
        if len(steps) != len(set(steps)):
            raise ValueError(f"dense game {game_id} has duplicate step_idx values")
        ordered.append(sorted(group, key=lambda sample: sample["step_idx"]))
    return ordered


class DenseTWDToMDataset(Dataset):
    """One item per game with supervision at every strict PRE boundary."""

    def __init__(
        self,
        samples: Sequence[Mapping[str, Any]],
        *,
        feature_builder: PublicEventFeatureBuilder | None = None,
        target_dtype: torch.dtype = torch.float32,
        enable_cyclic_rotation: bool = False,
        augmentation_seed: int = 0,
        include_private_features: bool = False,
        observer_roles_by_game: Mapping[str, Mapping[str, str]] | None = None,
        supervision_scope: str = ALL_ALIVE_SCOPE,
    ) -> None:
        if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
            raise TypeError("samples must be a sequence")
        if not samples:
            raise ValueError("dense dataset samples cannot be empty")
        if not isinstance(enable_cyclic_rotation, bool):
            raise TypeError("enable_cyclic_rotation must be bool")
        if not isinstance(include_private_features, bool):
            raise TypeError("include_private_features must be bool")
        if supervision_scope not in SUPERVISION_SCOPES:
            raise ValueError(f"supervision_scope must be one of {SUPERVISION_SCOPES}")
        if observer_roles_by_game is not None and not isinstance(
            observer_roles_by_game, Mapping
        ):
            raise TypeError("observer_roles_by_game must be a mapping or None")
        deterministic_cyclic_shift(
            seed=augmentation_seed,
            epoch=0,
            sample_index=0,
        )
        self._raw_games = _group_raw_samples(samples)
        self.feature_builder = feature_builder or PublicEventFeatureBuilder()
        self.target_dtype = target_dtype
        self.enable_cyclic_rotation = enable_cyclic_rotation
        self.augmentation_seed = augmentation_seed
        self.include_private_features = include_private_features
        self.observer_roles_by_game = {
            game_id: normalize_observer_roles(roles)
            for game_id, roles in (observer_roles_by_game or {}).items()
        }
        self.supervision_scope = supervision_scope
        self._epoch = 0
        self.model_input_scope = (
            PRIVATE_MODEL_INPUT_SCOPE
            if include_private_features
            else MODEL_INPUT_SCOPE
        )
        self.target_semantics = TARGET_SEMANTICS
        self.target_conversion = TARGET_CONVERSION
        self.supervision_version = DENSE_SUPERVISION_VERSION

        # Validate every raw game once at construction. Raw private values are
        # materialized only as boolean masks when the explicit mode is enabled.
        for game in self._raw_games:
            TWDToMDataset(
                game,
                feature_builder=self.feature_builder,
                target_dtype=self.target_dtype,
                include_private_features=self.include_private_features,
                observer_roles_by_game=(
                    self.observer_roles_by_game or None
                ),
                supervision_scope=self.supervision_scope,
            )

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        **kwargs: Any,
    ) -> "DenseTWDToMDataset":
        from werewolf.models.twd_tom.dataset import load_twd_tom_jsonl

        return cls(load_twd_tom_jsonl(path), **kwargs)

    @property
    def samples(self) -> list[dict[str, Any]]:
        """Expose raw game identity only for split-overlap validation."""

        return [
            {"game_id": game[0]["game_id"]}
            for game in self._raw_games
        ]

    def __len__(self) -> int:
        return len(self._raw_games)

    @property
    def boundary_count(self) -> int:
        """Return the number of strict PRE targets across all games."""

        return sum(len(game) for game in self._raw_games)

    def set_epoch(self, epoch: int) -> None:
        deterministic_cyclic_shift(seed=0, epoch=epoch, sample_index=0)
        self._epoch = epoch

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw_game = self._raw_games[index]
        shift = 0
        if self.enable_cyclic_rotation:
            shift = deterministic_cyclic_shift(
                seed=self.augmentation_seed,
                epoch=self._epoch,
                sample_index=index,
            )
            raw_game = [
                cyclically_rotate_belief_sample(sample, shift=shift)
                for sample in raw_game
            ]

        game_id = raw_game[0]["game_id"]
        roles = self.observer_roles_by_game.get(game_id)
        if roles is not None and shift:
            roles = rotate_observer_roles(roles, shift=shift)
        roles_by_game = {game_id: roles} if roles is not None else None

        snapshot_dataset = TWDToMDataset(
            raw_game,
            feature_builder=self.feature_builder,
            target_dtype=self.target_dtype,
            include_private_features=self.include_private_features,
            observer_roles_by_game=roles_by_game,
            supervision_scope=self.supervision_scope,
        )
        snapshots = [
            snapshot_dataset[snapshot_index]
            for snapshot_index in range(len(snapshot_dataset))
        ]
        full_features = {
            field: snapshots[-1][field]
            for field in _FEATURE_FIELDS
        }
        boundary_indices: list[int] = []
        previous_boundary = -1
        for snapshot in snapshots:
            boundary_length = int(snapshot["attention_mask"].sum().item())
            if boundary_length <= 0:
                raise ValueError("dense PRE boundary requires non-empty public history")
            boundary_index = boundary_length - 1
            if boundary_index <= previous_boundary:
                raise ValueError("dense PRE boundaries must be strictly increasing")
            for field in _FEATURE_FIELDS:
                prefix = snapshot[field]
                if not torch.equal(prefix, full_features[field][:boundary_length]):
                    raise ValueError(
                        "dense PRE features must be exact prefixes of the final game "
                        f"sequence: field={field}"
                    )
            boundary_indices.append(boundary_index)
            previous_boundary = boundary_index

        result = {
            **full_features,
            "boundary_indices": torch.tensor(boundary_indices, dtype=torch.long),
            "boundary_valid_mask": torch.ones(len(snapshots), dtype=torch.bool),
            "belief_targets": torch.stack(
                [snapshot["belief_targets"] for snapshot in snapshots]
            ),
            "observer_alive_mask": torch.stack(
                [snapshot["observer_alive_mask"] for snapshot in snapshots]
            ),
            "observer_supervision_mask": torch.stack(
                [snapshot["observer_supervision_mask"] for snapshot in snapshots]
            ),
            "diagonal_target_mask": torch.stack(
                [snapshot["diagonal_target_mask"] for snapshot in snapshots]
            ),
            "supervision_known_non_werewolf_mask": torch.stack(
                [
                    snapshot["supervision_known_non_werewolf_mask"]
                    for snapshot in snapshots
                ]
            ),
            "metadata": {
                "game_id": snapshots[0]["metadata"]["game_id"],
                "step_idx": [
                    snapshot["metadata"]["step_idx"] for snapshot in snapshots
                ],
                "phase": [
                    snapshot["metadata"]["phase"] for snapshot in snapshots
                ],
                "speaker_id": [
                    snapshot["metadata"]["speaker_id"] for snapshot in snapshots
                ],
                "observer_roles": snapshots[0]["metadata"]["observer_roles"],
                "raw_support_size": [
                    snapshot["metadata"]["raw_support_size"]
                    for snapshot in snapshots
                ],
                "raw_empty": [
                    snapshot["metadata"]["raw_empty"] for snapshot in snapshots
                ],
                "hard_knowledge_count": [
                    snapshot["metadata"]["hard_knowledge_count"]
                    for snapshot in snapshots
                ],
                "day": [snapshot["metadata"]["day"] for snapshot in snapshots],
                "public_action_count": [
                    snapshot["metadata"]["public_action_count"]
                    for snapshot in snapshots
                ],
                "speaker_vs_non_speaker": [
                    snapshot["metadata"]["speaker_vs_non_speaker"]
                    for snapshot in snapshots
                ],
                "alive_count": [
                    snapshot["metadata"]["alive_count"] for snapshot in snapshots
                ],
                "supervision_scope": self.supervision_scope,
                "supervision_version": DENSE_SUPERVISION_VERSION,
                "target_semantics": TARGET_SEMANTICS,
                "target_conversion": TARGET_CONVERSION,
            },
        }
        if self.include_private_features:
            result.update({
                field: torch.stack([snapshot[field] for snapshot in snapshots])
                for field in (
                    "known_werewolf_mask",
                    "known_non_werewolf_mask",
                )
            })
        return result


def collate_dense_twd_tom_games(
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Right-pad game timelines and their PRE boundary contracts."""

    if isinstance(batch, (str, bytes)) or not isinstance(batch, Sequence):
        raise TypeError("batch must be a sequence")
    if not batch:
        raise ValueError("batch cannot be empty")
    max_length = max(item["subject_ids"].shape[0] for item in batch)
    max_boundaries = max(item["boundary_indices"].shape[0] for item in batch)
    padded_features = {
        field: batch[0][field].new_zeros((len(batch), max_length))
        for field in _FEATURE_FIELDS
    }
    boundary_indices = torch.zeros((len(batch), max_boundaries), dtype=torch.long)
    boundary_valid_mask = torch.zeros(
        (len(batch), max_boundaries), dtype=torch.bool
    )
    belief_targets = batch[0]["belief_targets"].new_zeros(
        (len(batch), max_boundaries, NUM_PLAYERS, NUM_PLAYERS)
    )
    observer_alive_mask = torch.zeros(
        (len(batch), max_boundaries, NUM_PLAYERS), dtype=torch.bool
    )
    observer_supervision_mask = torch.zeros_like(observer_alive_mask)
    diagonal = ~torch.eye(NUM_PLAYERS, dtype=torch.bool)
    diagonal_target_mask = diagonal.view(1, 1, NUM_PLAYERS, NUM_PLAYERS).expand(
        len(batch), max_boundaries, -1, -1
    ).clone()
    supervision_known_non_werewolf_mask = torch.zeros_like(
        diagonal_target_mask
    )
    private_fields = ("known_werewolf_mask", "known_non_werewolf_mask")
    private_presence = [
        tuple(field in item for field in private_fields)
        for item in batch
    ]
    if any(any(presence) and not all(presence) for presence in private_presence):
        raise ValueError("private knowledge masks must be supplied together")
    if len(set(private_presence)) != 1:
        raise ValueError("batch cannot mix public and private-conditioned games")
    private_masks = (
        {
            field: torch.zeros(
                (len(batch), max_boundaries, NUM_PLAYERS, NUM_PLAYERS),
                dtype=torch.bool,
            )
            for field in private_fields
        }
        if all(private_presence[0])
        else {}
    )

    for batch_index, item in enumerate(batch):
        length = item["subject_ids"].shape[0]
        boundary_count = item["boundary_indices"].shape[0]
        for field in _FEATURE_FIELDS:
            if item[field].shape != (length,):
                raise ValueError(f"dense feature length mismatch for {field}")
            padded_features[field][batch_index, :length] = item[field]
        boundary_indices[batch_index, :boundary_count] = item["boundary_indices"]
        boundary_valid_mask[batch_index, :boundary_count] = item[
            "boundary_valid_mask"
        ]
        belief_targets[batch_index, :boundary_count] = item["belief_targets"]
        observer_alive_mask[batch_index, :boundary_count] = item[
            "observer_alive_mask"
        ]
        observer_supervision_mask[batch_index, :boundary_count] = item[
            "observer_supervision_mask"
        ]
        supervision_known_non_werewolf_mask[
            batch_index, :boundary_count
        ] = item["supervision_known_non_werewolf_mask"]
        if not torch.equal(
            item["diagonal_target_mask"],
            diagonal_target_mask[batch_index, :boundary_count],
        ):
            raise ValueError("dense diagonal target masks must exclude only self")
        for field in private_masks:
            private_masks[field][batch_index, :boundary_count] = item[field]

    return {
        **padded_features,
        **private_masks,
        "boundary_indices": boundary_indices,
        "boundary_valid_mask": boundary_valid_mask,
        "belief_targets": belief_targets,
        "observer_alive_mask": observer_alive_mask,
        "observer_supervision_mask": observer_supervision_mask,
        "diagonal_target_mask": diagonal_target_mask,
        "supervision_known_non_werewolf_mask": (
            supervision_known_non_werewolf_mask
        ),
        "metadata": [deepcopy(item["metadata"]) for item in batch],
    }


__all__ = [
    "DENSE_SUPERVISION_VERSION",
    "DenseTWDToMDataset",
    "collate_dense_twd_tom_games",
]
