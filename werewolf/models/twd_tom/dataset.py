"""Strict tom-v2 observer-conditioned belief Dataset."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from werewolf.models.twd_tom.action_features import PublicEventFeatureBuilder
from werewolf.models.twd_tom.belief_labels import (
    close_hard_knowledge,
    suspicion_set_to_belief_vector,
)
from werewolf.models.twd_tom.public_events import (
    PUBLIC_EVENT_SCHEMA_VERSION,
    normalize_public_events,
    parse_public_phase,
    public_event_digest,
    public_speech_actions,
    structured_input_digest,
)
from werewolf.models.twd_tom.samples import (
    REPORT_TRIGGERS,
    SAMPLE_FIELDS,
    SAMPLE_SCHEMA_VERSION,
)
from werewolf.models.twd_tom.schema import (
    canonicalize_player_set,
    LABEL_PROMPT_VERSION,
    LABEL_PROVENANCE,
    NUM_PLAYERS,
    PLAYER_NAMES,
    PLAYER_TO_ID,
    normalize_player,
)
from werewolf.speech.private_belief_perceiver import STATUS_OK


MODEL_INPUT_SCOPE = "structured_public_events_only"
TARGET_CONVERSION = "sparse_suspicion_uniform_support_player_vector_v1"
CYCLIC_ROTATION_VERSION = "cyclic_rotation_v1"

_SUBJECT_MAPPING_FIELDS = (
    "suspected_werewolves",
    "known_werewolves",
    "known_non_werewolves",
    "belief_status",
    "belief_errors",
    "agent_backend_ids",
)
_PLAYER_LIST_MAPPING_FIELDS = frozenset(
    {"suspected_werewolves", "known_werewolves", "known_non_werewolves"}
)


def deterministic_cyclic_shift(
    *,
    seed: int,
    epoch: int,
    sample_index: int,
) -> int:
    """Return one reproducible classic-seven seat rotation."""

    for field_name, value in {
        "seed": seed,
        "epoch": epoch,
        "sample_index": sample_index,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field_name} must be a non-negative integer")
    return (seed + epoch + sample_index) % NUM_PLAYERS


def _rotate_player_name(value: Any, *, shift: int) -> str:
    if not isinstance(value, str) or value not in PLAYER_NAMES:
        raise ValueError("rotated player IDs must be canonical player1...player7")
    return PLAYER_NAMES[(PLAYER_TO_ID[value] - 1 + shift) % NUM_PLAYERS]


def _rotate_player_number(value: Any, *, shift: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("rotated numeric player IDs must be integers")
    if not 1 <= value <= NUM_PLAYERS:
        raise ValueError("rotated numeric player IDs must be in [1, 7]")
    return ((value - 1 + shift) % NUM_PLAYERS) + 1


def _rotate_player_list(value: Any, *, shift: int) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("rotated player collections must be sequences")
    return sorted(
        (_rotate_player_name(player, shift=shift) for player in value),
        key=PLAYER_TO_ID.__getitem__,
    )


def cyclically_rotate_belief_sample(
    sample: Mapping[str, Any],
    *,
    shift: int,
) -> dict[str, Any]:
    """Rotate every player reference in one detached raw belief sample."""

    if not isinstance(sample, Mapping):
        raise TypeError("sample must be a mapping")
    if isinstance(shift, bool) or not isinstance(shift, int):
        raise TypeError("shift must be an integer")
    shift %= NUM_PLAYERS
    rotated = deepcopy(dict(sample))
    rotated["observer_ids"] = [
        _rotate_player_number(player_id, shift=shift)
        for player_id in rotated["observer_ids"]
    ]
    rotated["speaker_id"] = _rotate_player_number(
        rotated["speaker_id"],
        shift=shift,
    )

    for field_name in _SUBJECT_MAPPING_FIELDS:
        mapping = rotated[field_name]
        if not isinstance(mapping, Mapping):
            raise TypeError(f"{field_name} must be a mapping")
        remapped: dict[str, Any] = {}
        for subject, value in mapping.items():
            rotated_subject = _rotate_player_name(subject, shift=shift)
            remapped[rotated_subject] = (
                _rotate_player_list(value, shift=shift)
                if field_name in _PLAYER_LIST_MAPPING_FIELDS
                else deepcopy(value)
            )
        rotated[field_name] = remapped

    for event in rotated["public_events"]:
        event_type = event.get("event_type")
        if event_type in {"turn_start", "public_speech"}:
            event["speaker"] = _rotate_player_name(event["speaker"], shift=shift)
        if event_type == "public_speech":
            event["sp_actions"] = [
                [
                    _rotate_player_name(action[0], shift=shift),
                    action[1],
                    None
                    if action[2] is None
                    else _rotate_player_name(action[2], shift=shift),
                ]
                for action in event["sp_actions"]
            ]
        elif event_type == "vote_result":
            event["votes"] = sorted(
                (
                    {
                        "voter": _rotate_player_name(vote["voter"], shift=shift),
                        "target": None
                        if vote["target"] is None
                        else _rotate_player_name(vote["target"], shift=shift),
                    }
                    for vote in event["votes"]
                ),
                key=lambda vote: PLAYER_TO_ID[vote["voter"]],
            )
        elif event_type == "exile_result":
            event["exiled_players"] = _rotate_player_list(
                event["exiled_players"], shift=shift
            )
        elif event_type == "death_announcement":
            event["dead_players"] = _rotate_player_list(
                event["dead_players"], shift=shift
            )

    rotated["public_event_digest"] = public_event_digest(rotated["public_events"])
    rotated["structured_input_digest"] = structured_input_digest(
        rotated["public_events"]
    )
    return rotated


def _require_non_empty_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return value


def _normalize_observer_ids(value: Any) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("observer_ids must be a sequence")
    if not value:
        raise ValueError("observer_ids cannot be empty")
    observer_ids: list[int] = []
    for player_id in value:
        if isinstance(player_id, bool) or not isinstance(player_id, int):
            raise TypeError("observer IDs must be integers")
        if not 1 <= player_id <= NUM_PLAYERS:
            raise ValueError("observer IDs must be in [1, 7]")
        if player_id in observer_ids:
            raise ValueError(f"duplicate observer ID: {player_id}")
        observer_ids.append(player_id)
    return observer_ids


def _normalize_subject_mapping(
    value: Any,
    *,
    field_name: str,
    expected_subjects: set[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    normalized: dict[str, Any] = {}
    for raw_subject, item in value.items():
        subject = normalize_player(raw_subject)
        if raw_subject != subject:
            raise ValueError(f"{field_name} keys must use canonical player IDs")
        if subject in normalized:
            raise ValueError(f"duplicate subject in {field_name}: {subject}")
        normalized[subject] = item
    if set(normalized) != expected_subjects:
        missing = sorted(expected_subjects - set(normalized))
        extra = sorted(set(normalized) - expected_subjects)
        raise ValueError(
            f"{field_name} subject set mismatch; missing={missing}, extra={extra}"
        )
    return normalized


def _normalize_sample(sample: Any) -> dict[str, Any]:
    """Validate one raw playing-agent readonly self-report snapshot."""

    if not isinstance(sample, Mapping):
        raise TypeError("each dataset sample must be a mapping")
    if set(sample) != SAMPLE_FIELDS:
        missing = sorted(SAMPLE_FIELDS - set(sample))
        extra = sorted(set(sample) - SAMPLE_FIELDS)
        raise ValueError(f"sample field set mismatch; missing={missing}, extra={extra}")
    if sample.get("schema_version") != SAMPLE_SCHEMA_VERSION:
        raise ValueError("unsupported sample schema_version")
    if sample.get("label_prompt_version") != LABEL_PROMPT_VERSION:
        raise ValueError("unsupported label_prompt_version")
    if sample.get("label_provenance") != LABEL_PROVENANCE:
        raise ValueError("unsupported label_provenance")

    observer_ids = _normalize_observer_ids(sample.get("observer_ids"))
    expected_subjects = {normalize_player(player_id) for player_id in observer_ids}
    speaker_id = sample.get("speaker_id")
    if isinstance(speaker_id, bool) or not isinstance(speaker_id, int):
        raise TypeError("speaker_id must be an integer")
    if not 1 <= speaker_id <= NUM_PLAYERS:
        raise ValueError("speaker_id must be in [1, 7]")
    if speaker_id not in observer_ids:
        raise ValueError("speaker_id must identify an alive observer")

    mappings = {
        field_name: _normalize_subject_mapping(
            sample.get(field_name),
            field_name=field_name,
            expected_subjects=expected_subjects,
        )
        for field_name in _SUBJECT_MAPPING_FIELDS
    }
    targets: dict[str, torch.Tensor] = {}
    for subject in sorted(expected_subjects, key=PLAYER_TO_ID.__getitem__):
        status = mappings["belief_status"][subject]
        error = mappings["belief_errors"][subject]
        suspicion = mappings["suspected_werewolves"][subject]
        if status != STATUS_OK:
            raise ValueError(
                "training Dataset requires status=ok for every alive observer"
            )
        if error is not None:
            raise ValueError(f"{subject} status=ok requires null belief error")
        if not isinstance(suspicion, list):
            raise ValueError(f"{subject} status=ok requires a suspicion list")

        known_wolves = mappings["known_werewolves"][subject]
        known_non_wolves = mappings["known_non_werewolves"][subject]
        closed_wolves, closed_non_wolves = close_hard_knowledge(
            known_wolves,
            known_non_wolves,
        )
        if known_wolves != closed_wolves or known_non_wolves != closed_non_wolves:
            raise ValueError(f"{subject} hard knowledge must already be closed")
        normalized_suspicion = canonicalize_player_set(
            suspicion,
            field_name="suspected_werewolves",
        )
        if suspicion != normalized_suspicion:
            raise ValueError(
                f"{subject} suspected_werewolves must use canonical order"
            )
        _require_non_empty_text(
            mappings["agent_backend_ids"][subject],
            field_name=f"{subject} agent_backend_id",
        )
        targets[subject] = suspicion_set_to_belief_vector(
            normalized_suspicion,
            observer_id=subject,
            dtype=torch.float64,
        )

    public_events = normalize_public_events(sample.get("public_events"))
    if not public_events:
        raise ValueError("public_events cannot be empty")
    if (
        public_events[-1]["event_type"] != "turn_start"
        or public_events[-1]["speaker"] != normalize_player(speaker_id)
    ):
        raise ValueError("pre-speech public_events must end with matching turn_start")

    step_idx = sample.get("step_idx")
    if isinstance(step_idx, bool) or not isinstance(step_idx, int) or step_idx < 0:
        raise ValueError("step_idx must be a non-negative integer")
    if sample.get("label_cutoff_step_idx") != step_idx:
        raise ValueError("label_cutoff_step_idx must equal step_idx")
    phase = _require_non_empty_text(sample.get("phase"), field_name="phase")
    parse_public_phase(phase)
    latest_phase = next(
        (
            event["phase"]
            for event in reversed(public_events)
            if event["event_type"] == "phase_change"
        ),
        None,
    )
    if latest_phase != phase:
        raise ValueError("sample phase must match latest public phase_change")
    _require_non_empty_text(sample.get("game_id"), field_name="game_id")
    if sample.get("report_trigger") not in REPORT_TRIGGERS:
        raise ValueError("unsupported report_trigger")
    if sample.get("public_event_schema_version") != PUBLIC_EVENT_SCHEMA_VERSION:
        raise ValueError("unsupported public_event_schema_version")
    if sample.get("public_action_count") != len(public_speech_actions(public_events)):
        raise ValueError("public_action_count must equal the public speech-action count")
    if sample.get("public_event_digest") != public_event_digest(public_events):
        raise ValueError("public_event_digest does not match public_events")
    if sample.get("structured_input_digest") != structured_input_digest(public_events):
        raise ValueError("structured_input_digest does not match public_events")

    normalized = deepcopy(dict(sample))
    normalized["observer_ids"] = observer_ids
    normalized["public_events"] = public_events
    for field_name, value in mappings.items():
        normalized[field_name] = value
    normalized["_belief_targets"] = targets
    return normalized


def load_twd_tom_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load raw tom-v2 snapshot objects from JSONL without mutating them."""

    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"dataset file not found: {input_path}")
    samples: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(sample, dict):
                raise TypeError(f"JSONL line {line_number} must contain an object")
            samples.append(sample)
    return samples


class TWDToMDataset(Dataset):
    """The sole tom-v2 Dataset: public history to observer belief matrix."""

    def __init__(
        self,
        samples: Sequence[Mapping[str, Any]],
        *,
        feature_builder: PublicEventFeatureBuilder | None = None,
        target_dtype: torch.dtype = torch.float32,
        enable_cyclic_rotation: bool = False,
        augmentation_seed: int = 0,
    ) -> None:
        if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
            raise TypeError("samples must be a sequence")
        if not isinstance(target_dtype, torch.dtype) or not target_dtype.is_floating_point:
            raise TypeError("target_dtype must be a floating-point dtype")
        if not isinstance(enable_cyclic_rotation, bool):
            raise TypeError("enable_cyclic_rotation must be bool")
        deterministic_cyclic_shift(
            seed=augmentation_seed,
            epoch=0,
            sample_index=0,
        )
        self._raw_samples = [deepcopy(dict(sample)) for sample in samples]
        self.samples = [_normalize_sample(sample) for sample in self._raw_samples]
        last_prefix_by_game: dict[str, list[dict[str, Any]]] = {}
        for sample in self.samples:
            game_id = sample["game_id"]
            current = sample["public_events"]
            previous = last_prefix_by_game.get(game_id)
            if previous is not None and (
                len(current) <= len(previous) or current[: len(previous)] != previous
            ):
                raise ValueError(
                    "public_events snapshots must be strictly monotonic prefixes"
                )
            last_prefix_by_game[game_id] = current
        self.feature_builder = feature_builder or PublicEventFeatureBuilder()
        self.target_dtype = target_dtype
        self.enable_cyclic_rotation = enable_cyclic_rotation
        self.augmentation_seed = augmentation_seed
        self._epoch = 0
        self.model_input_scope = MODEL_INPUT_SCOPE
        self.target_conversion = TARGET_CONVERSION

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        feature_builder: PublicEventFeatureBuilder | None = None,
        target_dtype: torch.dtype = torch.float32,
        enable_cyclic_rotation: bool = False,
        augmentation_seed: int = 0,
    ) -> "TWDToMDataset":
        return cls(
            load_twd_tom_jsonl(path),
            feature_builder=feature_builder,
            target_dtype=target_dtype,
            enable_cyclic_rotation=enable_cyclic_rotation,
            augmentation_seed=augmentation_seed,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic training rotation for an epoch."""

        deterministic_cyclic_shift(seed=0, epoch=epoch, sample_index=0)
        self._epoch = epoch

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        if self.enable_cyclic_rotation:
            shift = deterministic_cyclic_shift(
                seed=self.augmentation_seed,
                epoch=self._epoch,
                sample_index=index,
            )
            sample = _normalize_sample(
                cyclically_rotate_belief_sample(
                    self._raw_samples[index],
                    shift=shift,
                )
            )
        features = self.feature_builder.encode_events(sample["public_events"])
        belief_targets = torch.zeros(
            (NUM_PLAYERS, NUM_PLAYERS),
            dtype=self.target_dtype,
        )
        observer_alive_mask = torch.zeros(NUM_PLAYERS, dtype=torch.bool)
        for observer_id in sample["observer_ids"]:
            subject = normalize_player(observer_id)
            row_index = observer_id - 1
            belief_targets[row_index] = sample["_belief_targets"][subject].to(
                dtype=self.target_dtype
            )
            observer_alive_mask[row_index] = True

        diagonal_target_mask = ~torch.eye(NUM_PLAYERS, dtype=torch.bool)
        metadata = {
            "schema_version": sample["schema_version"],
            "game_id": sample["game_id"],
            "step_idx": sample["step_idx"],
            "phase": sample["phase"],
            "speaker_id": sample["speaker_id"],
            "observer_ids": deepcopy(sample["observer_ids"]),
            "report_trigger": sample["report_trigger"],
            "label_provenance": sample["label_provenance"],
            "target_conversion": TARGET_CONVERSION,
        }
        return {
            **features,
            "belief_targets": belief_targets,
            "observer_alive_mask": observer_alive_mask,
            "diagonal_target_mask": diagonal_target_mask,
            "metadata": metadata,
        }


def collate_twd_tom_samples(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Right-pad public-event tensors and stack the fixed 7x7 targets."""

    if isinstance(batch, (str, bytes)) or not isinstance(batch, Sequence):
        raise TypeError("batch must be a sequence")
    if not batch:
        raise ValueError("batch cannot be empty")
    required_fields = {
        "belief_targets",
        "observer_alive_mask",
        "diagonal_target_mask",
        "metadata",
    }
    if any(not required_fields.issubset(item) for item in batch):
        raise ValueError("every sample must contain the tom-v2 belief contract")

    feature_fields = (
        "subject_ids",
        "action_ids",
        "object_ids",
        "event_type_ids",
        "phase_ids",
        "day_values",
        "attention_mask",
    )
    max_length = max(item["subject_ids"].shape[0] for item in batch)
    padded = {
        field_name: batch[0][field_name].new_zeros((len(batch), max_length))
        for field_name in feature_fields
    }
    for batch_index, item in enumerate(batch):
        length = item["subject_ids"].shape[0]
        for field_name in feature_fields:
            if item[field_name].shape != (length,):
                raise ValueError(f"feature length mismatch for {field_name}")
            padded[field_name][batch_index, :length] = item[field_name]

    return {
        **padded,
        "belief_targets": torch.stack([item["belief_targets"] for item in batch]),
        "observer_alive_mask": torch.stack(
            [item["observer_alive_mask"] for item in batch]
        ),
        "diagonal_target_mask": torch.stack(
            [item["diagonal_target_mask"] for item in batch]
        ),
        "metadata": {
            field_name: [deepcopy(item["metadata"][field_name]) for item in batch]
            for field_name in batch[0]["metadata"]
        },
    }


__all__ = [
    "CYCLIC_ROTATION_VERSION",
    "MODEL_INPUT_SCOPE",
    "TARGET_CONVERSION",
    "TWDToMDataset",
    "collate_twd_tom_samples",
    "cyclically_rotate_belief_sample",
    "deterministic_cyclic_shift",
    "load_twd_tom_jsonl",
]
