"""Strict dataset adapter for current first- and second-order ToM JSONL."""

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
    suspicion_set_to_pair_target,
)
from werewolf.models.twd_tom.public_events import (
    PUBLIC_EVENT_SCHEMA_VERSION,
    is_post_completed_public_speech_pre_next_action,
    normalize_public_events,
    parse_public_phase,
    public_event_digest,
    public_speech_actions,
    structured_input_digest,
)
from werewolf.models.twd_tom.samples import (
    REPORT_TRIGGERS,
    SAMPLE_SCHEMA_VERSION,
)
from werewolf.models.twd_tom.schema import (
    FORMAL_ANNOTATION_SCHEMA_VERSION,
    FORMAL_LABEL_PROVENANCE,
    LABEL_PROMPT_VERSION,
    NUM_PLAYERS,
    NUM_WOLF_PAIR_CLASSES,
    PLAYER_NAMES,
    PLAYER_TO_ID,
    normalize_player,
    validate_player_suspicion,
)
from werewolf.speech.private_belief_perceiver import (
    STATUS_OK,
    STATUS_PARSE_ERROR,
    STATUS_REPORTER_ERROR,
    STATUS_SEMANTIC_ERROR,
)


TOM_INPUT_SCOPES = {
    1: "public_events_plus_current_observer_private_knowledge",
    2: "public_events_only",
}
PRIVATE_FIELDS_USAGE = {
    1: "first_order_model_input_and_label",
    2: "label_construction_and_audit_only",
}
ANNOTATION_SCHEMA_VERSION = FORMAL_ANNOTATION_SCHEMA_VERSION
ANNOTATED_LABEL_PROVENANCE = FORMAL_LABEL_PROVENANCE
SOURCE_LABEL_PROVENANCE = "alive_observer_readonly_pre_speech_report_v1"
CYCLIC_ROTATION_VERSION = "cyclic_rotation_v1"

SUBJECT_MAPPING_FIELDS = (
    "suspected_werewolves",
    "known_werewolves",
    "known_non_werewolves",
    "belief_status",
    "belief_errors",
    "agent_backend_ids",
    "observer_annotation_confidence",
    "observer_label_provenance",
    "source_belief_errors",
    "source_belief_status",
)
_PLAYER_LIST_MAPPING_FIELDS = frozenset(
    {
        "suspected_werewolves",
        "known_werewolves",
        "known_non_werewolves",
    }
)

RAW_TRAINING_SAMPLE_FIELDS = frozenset(
    {
        "agent_backend_ids",
        "annotation_schema_version",
        "belief_errors",
        "belief_status",
        "current_action_used",
        "expert_labels_used_as_later_evidence",
        "future_information_used",
        "game_id",
        "known_non_werewolves",
        "known_werewolves",
        "label_cutoff_step_idx",
        "label_prompt_version",
        "label_provenance",
        "model_input_scope",
        "observer_annotation_confidence",
        "observer_ids",
        "observer_label_provenance",
        "phase",
        "private_fields_usage",
        "public_action_count",
        "public_event_digest",
        "public_event_schema_version",
        "public_events",
        "report_trigger",
        "schema_version",
        "source_belief_errors",
        "source_belief_status",
        "source_label_provenance",
        "source_schema_version",
        "speaker_id",
        "step_idx",
        "structured_input_digest",
        "suspected_werewolves",
        "tom_order",
    }
)

_VALID_STATUSES = {
    STATUS_OK,
    STATUS_PARSE_ERROR,
    STATUS_REPORTER_ERROR,
    STATUS_SEMANTIC_ERROR,
}


def _validate_tom_order(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (1, 2):
        raise ValueError("tom_order must be 1 or 2")
    return value


def _normalize_observer_ids(value: Any) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("observer_ids must be a sequence")
    if not value:
        raise ValueError("observer_ids cannot be empty")
    result: list[int] = []
    for player_id in value:
        if isinstance(player_id, bool) or not isinstance(player_id, int):
            raise TypeError("observer IDs must be integers")
        if not 1 <= player_id <= NUM_PLAYERS:
            raise ValueError("observer IDs must be in [1, 7]")
        if player_id in result:
            raise ValueError(f"duplicate observer ID: {player_id}")
        result.append(player_id)
    return result


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


def _require_non_empty_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return value


def deterministic_cyclic_shift(
    *,
    seed: int,
    epoch: int,
    sample_index: int,
) -> int:
    """Return the reproducible classic-seven rotation for one train item."""

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
    player_index = PLAYER_TO_ID[value] - 1
    return PLAYER_NAMES[(player_index + shift) % NUM_PLAYERS]


def _rotate_player_number(value: Any, *, shift: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("rotated numeric player IDs must be integers")
    if not 1 <= value <= NUM_PLAYERS:
        raise ValueError("rotated numeric player IDs must be in [1, 7]")
    return ((value - 1 + shift) % NUM_PLAYERS) + 1


def _rotate_player_list(value: Any, *, shift: int) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("rotated player collections must be sequences")
    rotated = [_rotate_player_name(player, shift=shift) for player in value]
    return sorted(rotated, key=PLAYER_TO_ID.__getitem__)


def cyclically_rotate_second_order_sample(
    sample: Mapping[str, Any],
    *,
    shift: int,
) -> dict[str, Any]:
    """Rotate every structured player ID in one detached raw sample."""

    if not isinstance(sample, Mapping):
        raise TypeError("sample must be a mapping")
    if isinstance(shift, bool) or not isinstance(shift, int):
        raise TypeError("shift must be an integer")
    shift %= NUM_PLAYERS
    rotated = deepcopy(dict(sample))
    if rotated.get("tom_order") != 2:
        raise ValueError("cyclic player rotation is restricted to tom_order=2")

    rotated["observer_ids"] = [
        _rotate_player_number(player_id, shift=shift)
        for player_id in rotated["observer_ids"]
    ]
    rotated["speaker_id"] = _rotate_player_number(
        rotated["speaker_id"],
        shift=shift,
    )

    for field_name in SUBJECT_MAPPING_FIELDS:
        mapping = rotated[field_name]
        if not isinstance(mapping, Mapping):
            raise TypeError(f"{field_name} must be a mapping")
        remapped: dict[str, Any] = {}
        for subject, value in mapping.items():
            rotated_subject = _rotate_player_name(subject, shift=shift)
            if rotated_subject in remapped:
                raise ValueError(f"duplicate rotated subject in {field_name}")
            remapped[rotated_subject] = (
                _rotate_player_list(value, shift=shift)
                if field_name in _PLAYER_LIST_MAPPING_FIELDS
                else deepcopy(value)
            )
        rotated[field_name] = remapped

    for event in rotated["public_events"]:
        event_type = event.get("event_type")
        if event_type in {"turn_start", "public_speech"}:
            event["speaker"] = _rotate_player_name(
                event["speaker"],
                shift=shift,
            )
        if event_type == "public_speech":
            event["sp_actions"] = [
                [
                    _rotate_player_name(action[0], shift=shift),
                    action[1],
                    _rotate_player_name(action[2], shift=shift),
                ]
                for action in event["sp_actions"]
            ]
        elif event_type == "vote_result":
            votes = [
                {
                    "voter": _rotate_player_name(
                        vote["voter"],
                        shift=shift,
                    ),
                    "target": (
                        None
                        if vote["target"] is None
                        else _rotate_player_name(
                            vote["target"],
                            shift=shift,
                        )
                    ),
                }
                for vote in event["votes"]
            ]
            event["votes"] = sorted(
                votes,
                key=lambda vote: PLAYER_TO_ID[vote["voter"]],
            )
        elif event_type == "exile_result":
            event["exiled_players"] = _rotate_player_list(
                event["exiled_players"],
                shift=shift,
            )
        elif event_type == "death_announcement":
            event["dead_players"] = _rotate_player_list(
                event["dead_players"],
                shift=shift,
            )

    rotated["public_event_digest"] = public_event_digest(
        rotated["public_events"]
    )
    rotated["structured_input_digest"] = structured_input_digest(
        rotated["public_events"]
    )
    return rotated


def _normalize_sample(sample: Any, *, tom_order: int) -> dict[str, Any]:
    """Validate one current raw training record without repairing it."""

    tom_order = _validate_tom_order(tom_order)
    if not isinstance(sample, Mapping):
        raise TypeError("each dataset sample must be a mapping")
    if set(sample) != RAW_TRAINING_SAMPLE_FIELDS:
        missing = sorted(RAW_TRAINING_SAMPLE_FIELDS - set(sample))
        extra = sorted(set(sample) - RAW_TRAINING_SAMPLE_FIELDS)
        raise ValueError(f"sample field set mismatch; missing={missing}, extra={extra}")
    if sample.get("schema_version") != SAMPLE_SCHEMA_VERSION:
        raise ValueError("unsupported sample schema_version")
    if sample.get("source_schema_version") != SAMPLE_SCHEMA_VERSION:
        raise ValueError("unsupported source_schema_version")
    if sample.get("tom_order") != tom_order:
        raise ValueError(
            f"sample tom_order must match requested tom_order={tom_order}"
        )
    if sample.get("model_input_scope") != TOM_INPUT_SCOPES[tom_order]:
        raise ValueError(
            f"model_input_scope is inconsistent with tom_order={tom_order}"
        )
    if sample.get("private_fields_usage") != PRIVATE_FIELDS_USAGE[tom_order]:
        raise ValueError(
            f"private_fields_usage is inconsistent with tom_order={tom_order}"
        )
    if sample.get("annotation_schema_version") != ANNOTATION_SCHEMA_VERSION:
        raise ValueError("unsupported annotation_schema_version")
    if sample.get("label_provenance") != ANNOTATED_LABEL_PROVENANCE:
        raise ValueError("unsupported label_provenance")
    if sample.get("source_label_provenance") != SOURCE_LABEL_PROVENANCE:
        raise ValueError("unsupported source_label_provenance")
    for field_name in (
        "current_action_used",
        "expert_labels_used_as_later_evidence",
        "future_information_used",
    ):
        if sample.get(field_name) is not False:
            raise ValueError(f"{field_name} must be false")

    observer_ids = _normalize_observer_ids(sample.get("observer_ids"))
    expected_subjects = {normalize_player(player_id) for player_id in observer_ids}
    speaker_id = sample.get("speaker_id")
    if isinstance(speaker_id, bool) or not isinstance(speaker_id, int):
        raise TypeError("speaker_id must be an integer")
    if not 1 <= speaker_id <= NUM_PLAYERS:
        raise ValueError("speaker_id must be in [1, 7]")
    if tom_order == 1 and observer_ids != [speaker_id]:
        raise ValueError(
            "first-order samples must supervise only the current speaker observer"
        )

    mappings = {
        field_name: _normalize_subject_mapping(
            sample.get(field_name),
            field_name=field_name,
            expected_subjects=expected_subjects,
        )
        for field_name in SUBJECT_MAPPING_FIELDS
    }

    targets: dict[str, torch.Tensor | None] = {}
    for subject in sorted(expected_subjects, key=PLAYER_TO_ID.__getitem__):
        known_wolves = mappings["known_werewolves"][subject]
        known_non_wolves = mappings["known_non_werewolves"][subject]
        closed_wolves, closed_non_wolves = close_hard_knowledge(
            known_wolves,
            known_non_wolves,
        )
        if known_wolves != closed_wolves or known_non_wolves != closed_non_wolves:
            raise ValueError(f"{subject} hard knowledge must already be closed")

        status = mappings["belief_status"][subject]
        error = mappings["belief_errors"][subject]
        suspicion = mappings["suspected_werewolves"][subject]
        if status not in _VALID_STATUSES:
            raise ValueError(f"{subject} has unsupported belief status")
        if status == STATUS_OK:
            if error is not None:
                raise ValueError(f"{subject} status=ok requires null belief error")
            if not isinstance(suspicion, list):
                raise ValueError(f"{subject} status=ok requires a suspicion list")
            normalized_suspicion = validate_player_suspicion(
                suspicion,
                closed_wolves,
                closed_non_wolves,
            )
            if suspicion != normalized_suspicion:
                raise ValueError(
                    f"{subject} suspected_werewolves must use canonical order"
                )
            targets[subject] = suspicion_set_to_pair_target(
                normalized_suspicion,
                closed_wolves,
                closed_non_wolves,
                dtype=torch.float64,
            )
        else:
            if suspicion is not None:
                raise ValueError(
                    f"{subject} non-ok status requires null suspected_werewolves"
                )
            if not isinstance(error, str) or not error:
                raise ValueError(f"{subject} non-ok status requires an error")
            targets[subject] = None

        _require_non_empty_text(
            mappings["agent_backend_ids"][subject],
            field_name=f"{subject} agent_backend_id",
        )
        _require_non_empty_text(
            mappings["observer_annotation_confidence"][subject],
            field_name=f"{subject} observer_annotation_confidence",
        )
        _require_non_empty_text(
            mappings["observer_label_provenance"][subject],
            field_name=f"{subject} observer_label_provenance",
        )
        source_status = mappings["source_belief_status"][subject]
        source_error = mappings["source_belief_errors"][subject]
        if source_status not in _VALID_STATUSES:
            raise ValueError(f"{subject} has unsupported source belief status")
        if source_error is not None and not isinstance(source_error, str):
            raise TypeError(f"{subject} source belief error must be text or null")

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
    if sample.get("label_prompt_version") != LABEL_PROMPT_VERSION:
        raise ValueError("unsupported label_prompt_version")
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
    normalized["_pair_targets"] = targets
    return normalized


def load_twd_tom_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load JSON objects from a ToM JSONL file without changing the file."""

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


def second_order_effective_subject_mask(
    subject_mask: torch.Tensor,
    reasoning_player_id: torch.Tensor,
) -> torch.Tensor:
    """Return valid observers other than the current reasoning player."""

    if not isinstance(subject_mask, torch.Tensor):
        raise TypeError(
            "subject_mask must be a torch.Tensor; "
            f"got {type(subject_mask).__name__}"
        )
    if not isinstance(reasoning_player_id, torch.Tensor):
        raise TypeError(
            "reasoning_player_id must be a torch.Tensor; "
            f"got {type(reasoning_player_id).__name__}"
        )
    if subject_mask.dtype is not torch.bool:
        raise TypeError(
            f"subject_mask dtype must be torch.bool; got {subject_mask.dtype}"
        )
    if reasoning_player_id.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            "reasoning_player_id dtype must be integral; "
            f"got {reasoning_player_id.dtype}"
        )
    if reasoning_player_id.shape != subject_mask.shape[:-1]:
        raise ValueError(
            "reasoning_player_id shape must match subject_mask batch dimensions; "
            f"reasoning_player_id={tuple(reasoning_player_id.shape)}, "
            f"subject_mask={tuple(subject_mask.shape)}"
        )
    if subject_mask.ndim == 0 or subject_mask.shape[-1] != NUM_PLAYERS:
        raise ValueError(
            "effective mask last dimension must be 7; "
            f"got shape {tuple(subject_mask.shape)}"
        )
    if torch.any((reasoning_player_id < 1) | (reasoning_player_id > NUM_PLAYERS)):
        raise ValueError("reasoning_player_id values must be in [1, 7]")
    canonical_observer_ids = torch.arange(
        1,
        NUM_PLAYERS + 1,
        device=subject_mask.device,
    )
    other_player_mask = canonical_observer_ids != reasoning_player_id.unsqueeze(
        -1
    ).to(device=subject_mask.device)
    return subject_mask & other_player_mask


class TWDToMDataset(Dataset):
    """Dataset for one explicit ToM order."""

    def __init__(
        self,
        samples: Sequence[Mapping[str, Any]],
        *,
        tom_order: int,
        feature_builder: PublicEventFeatureBuilder | None = None,
        target_dtype: torch.dtype = torch.float32,
        enable_cyclic_rotation: bool = False,
        augmentation_seed: int = 0,
    ):
        self.tom_order = _validate_tom_order(tom_order)
        if not isinstance(enable_cyclic_rotation, bool):
            raise TypeError("enable_cyclic_rotation must be boolean")
        if enable_cyclic_rotation and self.tom_order != 2:
            raise ValueError("cyclic player rotation is restricted to tom_order=2")
        if (
            isinstance(augmentation_seed, bool)
            or not isinstance(augmentation_seed, int)
            or augmentation_seed < 0
        ):
            raise ValueError("augmentation_seed must be a non-negative integer")
        if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
            raise TypeError("samples must be a sequence")
        self._raw_samples = [deepcopy(dict(sample)) for sample in samples]
        self.samples = [
            _normalize_sample(sample, tom_order=self.tom_order)
            for sample in self._raw_samples
        ]
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
        if not isinstance(target_dtype, torch.dtype) or not target_dtype.is_floating_point:
            raise TypeError("target_dtype must be a floating-point dtype")
        self.target_dtype = target_dtype
        self.enable_cyclic_rotation = enable_cyclic_rotation
        self.augmentation_seed = augmentation_seed
        self._epoch = 0

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        tom_order: int,
        feature_builder: PublicEventFeatureBuilder | None = None,
        target_dtype: torch.dtype = torch.float32,
        enable_cyclic_rotation: bool = False,
        augmentation_seed: int = 0,
    ) -> "TWDToMDataset":
        return cls(
            load_twd_tom_jsonl(path),
            tom_order=tom_order,
            feature_builder=feature_builder,
            target_dtype=target_dtype,
            enable_cyclic_rotation=enable_cyclic_rotation,
            augmentation_seed=augmentation_seed,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic train-only rotation epoch."""

        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self._epoch = epoch

    def second_order_supervised_indices(self) -> tuple[int, ...]:
        """Return speech-boundary samples with valid other-player targets."""

        if self.tom_order != 2:
            raise ValueError(
                "second_order_supervised_indices requires tom_order=2"
            )
        eligible = []
        for index in range(len(self)):
            item = self[index]
            if not item["post_completed_public_speech_pre_next_action"]:
                continue
            effective_mask = second_order_effective_subject_mask(
                item["subject_mask"],
                item["reasoning_player_id"],
            )
            if effective_mask.any().item():
                eligible.append(index)
        return tuple(eligible)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        if self.enable_cyclic_rotation:
            shift = deterministic_cyclic_shift(
                seed=self.augmentation_seed,
                epoch=self._epoch,
                sample_index=index,
            )
            rotated = cyclically_rotate_second_order_sample(
                self._raw_samples[index],
                shift=shift,
            )
            sample = _normalize_sample(rotated, tom_order=2)
        features = self.feature_builder.encode_events(sample["public_events"])
        targets = torch.zeros(
            (NUM_PLAYERS, NUM_WOLF_PAIR_CLASSES),
            dtype=self.target_dtype,
        )
        subject_mask = torch.zeros(NUM_PLAYERS, dtype=torch.bool)
        for subject, target in sample["_pair_targets"].items():
            if target is None:
                continue
            subject_index = PLAYER_TO_ID[subject] - 1
            targets[subject_index] = target.to(dtype=self.target_dtype)
            subject_mask[subject_index] = True

        metadata = {
            "schema_version": sample["schema_version"],
            "tom_order": self.tom_order,
            "model_input_scope": sample["model_input_scope"],
            "game_id": sample["game_id"],
            "step_idx": sample["step_idx"],
            "phase": sample["phase"],
            "speaker_id": sample["speaker_id"],
            "observer_ids": deepcopy(sample["observer_ids"]),
            "belief_status": deepcopy(sample["belief_status"]),
            "belief_errors": deepcopy(sample["belief_errors"]),
            "known_werewolves": deepcopy(sample["known_werewolves"]),
            "known_non_werewolves": deepcopy(sample["known_non_werewolves"]),
        }
        item: dict[str, Any] = {
            **features,
            "pair_targets": targets,
            "subject_mask": subject_mask,
            "metadata": metadata,
        }
        if self.tom_order == 2:
            item["reasoning_player_id"] = torch.tensor(
                sample["speaker_id"], dtype=torch.int64
            )
            boundary = is_post_completed_public_speech_pre_next_action(
                sample["public_events"],
                reasoning_player_id=sample["speaker_id"],
            )
            item["post_completed_public_speech_pre_next_action"] = boundary
            item["metadata"][
                "post_completed_public_speech_pre_next_action"
            ] = boundary
        else:
            known_wolves = torch.zeros((NUM_PLAYERS, NUM_PLAYERS), dtype=torch.float32)
            known_non_wolves = torch.zeros_like(known_wolves)
            subject = normalize_player(sample["observer_ids"][0])
            subject_index = PLAYER_TO_ID[subject] - 1
            for player in sample["known_werewolves"][subject]:
                known_wolves[subject_index, PLAYER_TO_ID[player] - 1] = 1.0
            for player in sample["known_non_werewolves"][subject]:
                known_non_wolves[subject_index, PLAYER_TO_ID[player] - 1] = 1.0
            item["known_werewolves"] = known_wolves
            item["known_non_werewolves"] = known_non_wolves
        return item


def collate_twd_tom_samples(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Right-pad event tensors and stack labels and optional private inputs."""

    if isinstance(batch, (str, bytes)) or not isinstance(batch, Sequence):
        raise TypeError("batch must be a sequence")
    if not batch:
        raise ValueError("batch cannot be empty")
    has_private = "known_werewolves" in batch[0]
    if any(("known_werewolves" in item) != has_private for item in batch):
        raise ValueError("a batch cannot mix first- and second-order samples")
    if any("pair_targets" not in item for item in batch):
        raise ValueError("every sample must contain pair_targets")

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

    result: dict[str, Any] = {
        **padded,
        "pair_targets": torch.stack([item["pair_targets"] for item in batch]),
        "subject_mask": torch.stack([item["subject_mask"] for item in batch]),
        "metadata": {
            field_name: [deepcopy(item["metadata"][field_name]) for item in batch]
            for field_name in batch[0]["metadata"]
        },
    }
    if has_private:
        result["known_werewolves"] = torch.stack(
            [item["known_werewolves"] for item in batch]
        )
        result["known_non_werewolves"] = torch.stack(
            [item["known_non_werewolves"] for item in batch]
        )
    else:
        if any(
            "reasoning_player_id" not in item
            or "post_completed_public_speech_pre_next_action" not in item
            for item in batch
        ):
            raise ValueError(
                "second-order samples require formal supervision fields"
            )
        result["reasoning_player_id"] = torch.stack(
            [item["reasoning_player_id"] for item in batch]
        )
        result["post_completed_public_speech_pre_next_action"] = torch.tensor(
            [item["post_completed_public_speech_pre_next_action"] for item in batch],
            dtype=torch.bool,
        )
    return result


__all__ = [
    "CYCLIC_ROTATION_VERSION",
    "RAW_TRAINING_SAMPLE_FIELDS",
    "SUBJECT_MAPPING_FIELDS",
    "TOM_INPUT_SCOPES",
    "TWDToMDataset",
    "collate_twd_tom_samples",
    "cyclically_rotate_second_order_sample",
    "deterministic_cyclic_shift",
    "load_twd_tom_jsonl",
    "second_order_effective_subject_mask",
]
