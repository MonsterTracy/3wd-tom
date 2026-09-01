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
from werewolf.models.twd_tom.annotation_v2 import (
    BELIEF_ANNOTATION_SOURCES,
    LEGACY_V1_BELIEF_SOURCE,
    SPEECH_ANNOTATION_SOURCES,
    V1_ANNOTATION_SOURCE,
    V1_EMPTY_UNIFORM_NONSELF_BELIEF_SOURCE,
    V2_ANNOTATION_SOURCE,
    apply_speech_v2_to_sample,
    belief_v2_targets_for_sample,
)
from werewolf.models.twd_tom.belief_labels import (
    close_hard_knowledge,
    legacy_v1_suspicion_set_to_belief_vector,
    suspicion_set_to_belief_vector,
)
from werewolf.models.twd_tom.public_events import (
    PUBLIC_EVENT_SCHEMA_VERSION,
    completed_pre_speech_public_events,
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
    LABEL_PROMPT_VERSION,
    LABEL_PROVENANCE,
    NUM_PLAYERS,
    PLAYER_NAMES,
    PLAYER_TO_ID,
    normalize_player,
    validate_player_suspicion,
)
from werewolf.models.twd_tom.supervision import (
    ALL_ALIVE_SCOPE,
    SUPERVISION_SCOPES,
    build_observer_supervision_mask,
    normalize_observer_roles,
    rotate_observer_roles,
)
from werewolf.models.twd_tom.speech_annotations import (
    SPEECH_ACTION_ONTOLOGY_VERSION,
    SPEECH_ANNOTATION_SCHEMA_VERSION,
    normalize_speech_annotations,
    speech_annotation_digest,
)
from werewolf.speech.private_belief_perceiver import (
    STATUS_OK,
    STATUS_PARSE_ERROR,
    STATUS_REPORTER_ERROR,
    STATUS_SEMANTIC_ERROR,
)


MODEL_INPUT_SCOPE = (
    "completed_structured_public_events_without_terminal_turn_start_v1"
)
PRIVATE_MODEL_INPUT_SCOPE = (
    "completed_structured_public_events_plus_observer_hard_knowledge_v1"
)
PRIVATE_FEATURE_FIELDS = (
    "known_werewolves",
    "known_non_werewolves",
)
TARGET_SEMANTICS = "relative_suspicion_matrix_empty_uniform_nonself_v2"
TARGET_CONVERSION = (
    "nonempty_sparse_suspicion_uniform_support_empty_uniform_nonself_v4"
)
LEGACY_V1_TARGET_SEMANTICS = "relative_suspicion_matrix_v1"
LEGACY_V1_TARGET_CONVERSION = (
    "hard_knowledge_consistent_sparse_suspicion_uniform_support_v2"
)
V2_TARGET_SEMANTICS = "relative_suspicion_distribution_v2_compat_v1"
V2_TARGET_CONVERSION = "annotation_v2_distribution_loss_mask_v1"
LABEL_OBSERVATION_SEMANTICS = "successful_report_including_empty_is_observed_v2"
LEGACY_V1_LABEL_OBSERVATION_SEMANTICS = (
    "empty_suspicion_imputed_uniform_legacy_v1"
)
V2_LABEL_OBSERVATION_SEMANTICS = "annotation_v2_distribution_loss_mask_v1"
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
    for annotation in rotated["speech_annotations"]:
        annotation["speaker"] = _rotate_player_name(
            annotation["speaker"], shift=shift
        )
        annotation["actions"] = [
                [
                    _rotate_player_name(action[0], shift=shift),
                    action[1],
                    None
                    if action[2] is None
                    else _rotate_player_name(action[2], shift=shift),
                ]
                for action in annotation["actions"]
            ]
    for event in rotated["public_events"]:
        event_type = event.get("event_type")
        if event_type == "vote_result":
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
    rotated["speech_annotation_digest"] = speech_annotation_digest(
        rotated["speech_annotations"]
    )
    rotated["structured_input_digest"] = structured_input_digest(
        rotated["public_events"],
        rotated["speech_annotations"],
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
    v1_empty_uniform_nonself_targets: dict[str, torch.Tensor] = {}
    legacy_v1_targets: dict[str, torch.Tensor] = {}
    label_observed: dict[str, bool] = {}
    for subject in sorted(expected_subjects, key=PLAYER_TO_ID.__getitem__):
        status = mappings["belief_status"][subject]
        error = mappings["belief_errors"][subject]
        suspicion = mappings["suspected_werewolves"][subject]
        known_wolves = mappings["known_werewolves"][subject]
        known_non_wolves = mappings["known_non_werewolves"][subject]
        closed_wolves, closed_non_wolves = close_hard_knowledge(
            known_wolves,
            known_non_wolves,
        )
        if known_wolves != closed_wolves or known_non_wolves != closed_non_wolves:
            raise ValueError(f"{subject} hard knowledge must already be closed")
        _require_non_empty_text(
            mappings["agent_backend_ids"][subject],
            field_name=f"{subject} agent_backend_id",
        )
        if status == STATUS_OK:
            if error is not None:
                raise ValueError(f"{subject} status=ok requires null belief error")
            if not isinstance(suspicion, list):
                raise ValueError(f"{subject} status=ok requires a suspicion list")
            normalized_suspicion = validate_player_suspicion(
                suspicion,
                closed_wolves,
                closed_non_wolves,
                observer_id=subject,
            )
            if suspicion != normalized_suspicion:
                raise ValueError(
                    f"{subject} suspected_werewolves must use canonical order"
                )
            label_observed[subject] = True
            v1_empty_uniform_nonself_targets[subject] = (
                suspicion_set_to_belief_vector(
                    normalized_suspicion,
                    observer_id=subject,
                    known_werewolves=closed_wolves,
                    known_non_werewolves=closed_non_wolves,
                    dtype=torch.float64,
                )
            )
            legacy_v1_targets[subject] = (
                v1_empty_uniform_nonself_targets[subject].clone()
                if normalized_suspicion
                else legacy_v1_suspicion_set_to_belief_vector(
                    normalized_suspicion,
                    observer_id=subject,
                    known_werewolves=closed_wolves,
                    known_non_werewolves=closed_non_wolves,
                    dtype=torch.float64,
                )
            )
        elif status in {
            STATUS_PARSE_ERROR,
            STATUS_REPORTER_ERROR,
            STATUS_SEMANTIC_ERROR,
        }:
            if suspicion is not None:
                raise ValueError(f"{subject} failed report requires null suspicion")
            if not isinstance(error, str) or not error:
                raise ValueError(f"{subject} failed report requires an error")
            label_observed[subject] = False
            v1_empty_uniform_nonself_targets[subject] = torch.zeros(
                NUM_PLAYERS,
                dtype=torch.float64,
            )
            legacy_v1_targets[subject] = torch.zeros(
                NUM_PLAYERS,
                dtype=torch.float64,
            )
        else:
            raise ValueError(f"{subject} has unsupported belief status")

    public_events = normalize_public_events(sample.get("public_events"))
    completed_pre_speech_public_events(
        public_events,
        speaker_id=speaker_id,
    )

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
    if sample.get("speech_annotation_schema_version") != (
        SPEECH_ANNOTATION_SCHEMA_VERSION
    ):
        raise ValueError("unsupported speech_annotation_schema_version")
    if sample.get("speech_action_ontology_version") != (
        SPEECH_ACTION_ONTOLOGY_VERSION
    ):
        raise ValueError("unsupported speech_action_ontology_version")
    speech_annotations = normalize_speech_annotations(
        sample.get("speech_annotations"),
        public_events=public_events,
        require_complete=True,
    )
    if sample.get("speech_annotation_digest") != speech_annotation_digest(
        speech_annotations
    ):
        raise ValueError("speech_annotation_digest does not match annotations")
    if sample.get("public_action_count") != len(
        public_speech_actions(public_events, speech_annotations)
    ):
        raise ValueError("public_action_count must equal the public speech-action count")
    if sample.get("public_event_digest") != public_event_digest(public_events):
        raise ValueError("public_event_digest does not match public_events")
    if sample.get("structured_input_digest") != structured_input_digest(
        public_events,
        speech_annotations,
    ):
        raise ValueError("structured_input_digest does not match public_events")

    normalized = deepcopy(dict(sample))
    normalized["observer_ids"] = observer_ids
    normalized["public_events"] = public_events
    normalized["speech_annotations"] = speech_annotations
    for field_name, value in mappings.items():
        normalized[field_name] = value
    normalized["_v1_empty_uniform_nonself_belief_targets"] = (
        v1_empty_uniform_nonself_targets
    )
    normalized["_legacy_v1_belief_targets"] = legacy_v1_targets
    normalized["_label_observed"] = label_observed
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
    """Map strict-PRE history to an observer belief matrix."""

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
        speech_annotation_source: str = V1_ANNOTATION_SOURCE,
        belief_annotation_source: str = V1_EMPTY_UNIFORM_NONSELF_BELIEF_SOURCE,
        speech_v2_annotations: Mapping[
            tuple[str, int], Mapping[str, Any]
        ] | None = None,
        belief_v2_annotations: Mapping[
            tuple[str, int, str], Mapping[str, Any]
        ] | None = None,
        fixed_cyclic_shift: int | None = None,
    ) -> None:
        if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
            raise TypeError("samples must be a sequence")
        if not isinstance(target_dtype, torch.dtype) or not target_dtype.is_floating_point:
            raise TypeError("target_dtype must be a floating-point dtype")
        if not isinstance(enable_cyclic_rotation, bool):
            raise TypeError("enable_cyclic_rotation must be bool")
        if fixed_cyclic_shift is not None and (
            isinstance(fixed_cyclic_shift, bool)
            or not isinstance(fixed_cyclic_shift, int)
            or not 0 <= fixed_cyclic_shift < NUM_PLAYERS
        ):
            raise ValueError("fixed_cyclic_shift must be None or an integer in [0, 6]")
        if enable_cyclic_rotation and fixed_cyclic_shift is not None:
            raise ValueError(
                "random epoch rotation and fixed_cyclic_shift are mutually exclusive"
            )
        if not isinstance(include_private_features, bool):
            raise TypeError("include_private_features must be bool")
        if supervision_scope not in SUPERVISION_SCOPES:
            raise ValueError(f"supervision_scope must be one of {SUPERVISION_SCOPES}")
        if speech_annotation_source not in SPEECH_ANNOTATION_SOURCES:
            raise ValueError(
                "speech_annotation_source must be one of "
                f"{SPEECH_ANNOTATION_SOURCES}"
            )
        if belief_annotation_source not in BELIEF_ANNOTATION_SOURCES:
            raise ValueError(
                "belief_annotation_source must be one of "
                f"{BELIEF_ANNOTATION_SOURCES}"
            )
        if (
            speech_annotation_source == V2_ANNOTATION_SOURCE
            and speech_v2_annotations is None
        ):
            raise ValueError("V2 speech source requires speech_v2_annotations")
        if (
            belief_annotation_source == V2_ANNOTATION_SOURCE
            and belief_v2_annotations is None
        ):
            raise ValueError("V2 belief source requires belief_v2_annotations")
        if observer_roles_by_game is not None and not isinstance(
            observer_roles_by_game, Mapping
        ):
            raise TypeError("observer_roles_by_game must be a mapping or None")
        role_metadata_supplied = observer_roles_by_game is not None
        deterministic_cyclic_shift(
            seed=augmentation_seed,
            epoch=0,
            sample_index=0,
        )
        self.observer_roles_by_game = {
            game_id: normalize_observer_roles(roles)
            for game_id, roles in (observer_roles_by_game or {}).items()
        }
        self._v1_raw_samples = [deepcopy(dict(sample)) for sample in samples]
        self._raw_samples = [
            (
                apply_speech_v2_to_sample(sample, speech_v2_annotations)
                if speech_annotation_source == V2_ANNOTATION_SOURCE
                else deepcopy(sample)
            )
            for sample in self._v1_raw_samples
        ]
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
        self.fixed_cyclic_shift = fixed_cyclic_shift
        self.augmentation_seed = augmentation_seed
        self.include_private_features = include_private_features
        sample_game_ids = {sample["game_id"] for sample in self.samples}
        missing_roles = sorted(sample_game_ids - set(self.observer_roles_by_game))
        if (
            role_metadata_supplied
            or supervision_scope in {"non_wolf_alive", "villager_alive"}
        ) and missing_roles:
            raise ValueError(
                "role sidecar metadata must cover every dataset game; "
                f"missing={missing_roles[:10]}"
            )
        self.supervision_scope = supervision_scope
        self.speech_annotation_source = speech_annotation_source
        self.belief_annotation_source = belief_annotation_source
        self._belief_v2_targets: list[torch.Tensor] | None = None
        self._belief_v2_observed_masks: list[torch.Tensor] | None = None
        if belief_v2_annotations is not None:
            v2_rows = [
                belief_v2_targets_for_sample(
                    sample,
                    belief_v2_annotations,
                    observer_roles=self.observer_roles_by_game.get(
                        sample["game_id"]
                    ),
                    dtype=target_dtype,
                )
                for sample in self._v1_raw_samples
            ]
            self._belief_v2_targets = [row[0] for row in v2_rows]
            self._belief_v2_observed_masks = [row[1] for row in v2_rows]
        self._epoch = 0
        self.model_input_scope = (
            PRIVATE_MODEL_INPUT_SCOPE
            if include_private_features
            else MODEL_INPUT_SCOPE
        )
        if belief_annotation_source == V2_ANNOTATION_SOURCE:
            self.target_semantics = V2_TARGET_SEMANTICS
            self.target_conversion = V2_TARGET_CONVERSION
            self.label_observation_semantics = (
                V2_LABEL_OBSERVATION_SEMANTICS
            )
        elif belief_annotation_source == LEGACY_V1_BELIEF_SOURCE:
            self.target_semantics = LEGACY_V1_TARGET_SEMANTICS
            self.target_conversion = LEGACY_V1_TARGET_CONVERSION
            self.label_observation_semantics = (
                LEGACY_V1_LABEL_OBSERVATION_SEMANTICS
            )
        else:
            self.target_semantics = TARGET_SEMANTICS
            self.target_conversion = TARGET_CONVERSION
            self.label_observation_semantics = LABEL_OBSERVATION_SEMANTICS

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        feature_builder: PublicEventFeatureBuilder | None = None,
        target_dtype: torch.dtype = torch.float32,
        enable_cyclic_rotation: bool = False,
        augmentation_seed: int = 0,
        include_private_features: bool = False,
        observer_roles_by_game: Mapping[str, Mapping[str, str]] | None = None,
        supervision_scope: str = ALL_ALIVE_SCOPE,
        speech_annotation_source: str = V1_ANNOTATION_SOURCE,
        belief_annotation_source: str = V1_EMPTY_UNIFORM_NONSELF_BELIEF_SOURCE,
        speech_v2_annotations: Mapping[
            tuple[str, int], Mapping[str, Any]
        ] | None = None,
        belief_v2_annotations: Mapping[
            tuple[str, int, str], Mapping[str, Any]
        ] | None = None,
        fixed_cyclic_shift: int | None = None,
    ) -> "TWDToMDataset":
        return cls(
            load_twd_tom_jsonl(path),
            feature_builder=feature_builder,
            target_dtype=target_dtype,
            enable_cyclic_rotation=enable_cyclic_rotation,
            augmentation_seed=augmentation_seed,
            include_private_features=include_private_features,
            observer_roles_by_game=observer_roles_by_game,
            supervision_scope=supervision_scope,
            speech_annotation_source=speech_annotation_source,
            belief_annotation_source=belief_annotation_source,
            speech_v2_annotations=speech_v2_annotations,
            belief_v2_annotations=belief_v2_annotations,
            fixed_cyclic_shift=fixed_cyclic_shift,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic training rotation for an epoch."""

        deterministic_cyclic_shift(seed=0, epoch=epoch, sample_index=0)
        self._epoch = epoch

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        shift = self.fixed_cyclic_shift or 0
        if self.enable_cyclic_rotation:
            shift = deterministic_cyclic_shift(
                seed=self.augmentation_seed,
                epoch=self._epoch,
                sample_index=index,
            )
        if shift:
            sample = _normalize_sample(
                cyclically_rotate_belief_sample(
                    self._raw_samples[index],
                    shift=shift,
                )
            )
        observer_roles = self.observer_roles_by_game.get(sample["game_id"])
        if observer_roles is not None and shift:
            observer_roles = rotate_observer_roles(observer_roles, shift=shift)
        model_public_events = completed_pre_speech_public_events(
            sample["public_events"],
            speaker_id=sample["speaker_id"],
        )
        features = self.feature_builder.encode_events(
            model_public_events,
            sample["speech_annotations"],
        )
        v1_empty_uniform_nonself_belief_targets = torch.zeros(
            (NUM_PLAYERS, NUM_PLAYERS),
            dtype=self.target_dtype,
        )
        legacy_v1_belief_targets = torch.zeros_like(
            v1_empty_uniform_nonself_belief_targets
        )
        observer_alive_mask = torch.zeros(NUM_PLAYERS, dtype=torch.bool)
        v1_label_observed_mask = torch.zeros(NUM_PLAYERS, dtype=torch.bool)
        known_werewolf_mask = torch.zeros(
            (NUM_PLAYERS, NUM_PLAYERS), dtype=torch.bool
        )
        known_non_werewolf_mask = torch.zeros_like(known_werewolf_mask)
        for observer_id in sample["observer_ids"]:
            subject = normalize_player(observer_id)
            row_index = observer_id - 1
            v1_empty_uniform_nonself_belief_targets[row_index] = sample[
                "_v1_empty_uniform_nonself_belief_targets"
            ][subject].to(dtype=self.target_dtype)
            legacy_v1_belief_targets[row_index] = sample[
                "_legacy_v1_belief_targets"
            ][subject].to(dtype=self.target_dtype)
            observer_alive_mask[row_index] = True
            v1_label_observed_mask[row_index] = sample["_label_observed"][subject]
            for player in sample["known_werewolves"][subject]:
                known_werewolf_mask[row_index, PLAYER_TO_ID[player] - 1] = True
            for player in sample["known_non_werewolves"][subject]:
                known_non_werewolf_mask[
                    row_index, PLAYER_TO_ID[player] - 1
                ] = True

        v2_belief_targets = None
        v2_label_observed_mask = None
        if self._belief_v2_targets is not None:
            v2_belief_targets = self._belief_v2_targets[index].clone()
            v2_label_observed_mask = self._belief_v2_observed_masks[index].clone()
            if shift:
                v2_belief_targets = torch.roll(
                    v2_belief_targets,
                    shifts=(shift, shift),
                    dims=(0, 1),
                )
                v2_label_observed_mask = torch.roll(
                    v2_label_observed_mask,
                    shifts=shift,
                    dims=0,
                )
        if self.belief_annotation_source == V2_ANNOTATION_SOURCE:
            if v2_belief_targets is None or v2_label_observed_mask is None:
                raise RuntimeError("V2 belief targets were not materialized")
            belief_targets = v2_belief_targets
            label_observed_mask = v2_label_observed_mask
        elif self.belief_annotation_source == LEGACY_V1_BELIEF_SOURCE:
            belief_targets = legacy_v1_belief_targets
            label_observed_mask = v1_label_observed_mask
        else:
            belief_targets = v1_empty_uniform_nonself_belief_targets
            label_observed_mask = v1_label_observed_mask

        observer_scope_mask = build_observer_supervision_mask(
            alive_mask=observer_alive_mask,
            observer_roles=observer_roles,
            speaker_id=sample["speaker_id"],
            scope=self.supervision_scope,
        )
        observer_supervision_mask = observer_scope_mask & label_observed_mask

        diagonal_target_mask = ~torch.eye(NUM_PLAYERS, dtype=torch.bool)
        metadata = {
            "schema_version": sample["schema_version"],
            "game_id": sample["game_id"],
            "step_idx": sample["step_idx"],
            "phase": sample["phase"],
            "speaker_id": sample["speaker_id"],
            "observer_ids": deepcopy(sample["observer_ids"]),
            "observer_roles": (
                [observer_roles[player] for player in PLAYER_NAMES]
                if observer_roles is not None
                else None
            ),
            "raw_support_size": [
                (
                    len(sample["suspected_werewolves"][normalize_player(player_id)])
                    if isinstance(
                        sample["suspected_werewolves"].get(normalize_player(player_id)),
                        list,
                    )
                    else None
                )
                for player_id in range(1, NUM_PLAYERS + 1)
            ],
            "raw_empty": [
                (
                    not bool(
                        sample["suspected_werewolves"][normalize_player(player_id)]
                    )
                    if isinstance(
                        sample["suspected_werewolves"].get(normalize_player(player_id)),
                        list,
                    )
                    else None
                )
                for player_id in range(1, NUM_PLAYERS + 1)
            ],
            "label_observed": [
                (
                    bool(label_observed_mask[player_id - 1])
                    if observer_alive_mask[player_id - 1]
                    else None
                )
                for player_id in range(1, NUM_PLAYERS + 1)
            ],
            "hard_knowledge_count": [
                (
                    len(
                        set(sample["known_werewolves"][player])
                        | set(sample["known_non_werewolves"][player])
                    )
                    if player in sample["known_werewolves"]
                    else None
                )
                for player in PLAYER_NAMES
            ],
            "day": parse_public_phase(sample["phase"])[0],
            "public_action_count": sample["public_action_count"],
            "speaker_vs_non_speaker": [
                player_id == sample["speaker_id"]
                for player_id in range(1, NUM_PLAYERS + 1)
            ],
            "alive_count": len(sample["observer_ids"]),
            "supervision_scope": self.supervision_scope,
            "report_trigger": sample["report_trigger"],
            "label_provenance": sample["label_provenance"],
            "target_semantics": self.target_semantics,
            "target_conversion": self.target_conversion,
            "label_observation_semantics": self.label_observation_semantics,
            "speech_annotation_source": self.speech_annotation_source,
            "belief_annotation_source": self.belief_annotation_source,
        }
        result = {
            **features,
            "belief_targets": belief_targets,
            "legacy_v1_belief_targets": legacy_v1_belief_targets,
            "legacy_v1_label_observed_mask": v1_label_observed_mask.clone(),
            "v1_empty_uniform_nonself_belief_targets": (
                v1_empty_uniform_nonself_belief_targets
            ),
            "v1_empty_uniform_nonself_label_observed_mask": (
                v1_label_observed_mask
            ),
            "observer_alive_mask": observer_alive_mask,
            "observer_scope_mask": observer_scope_mask,
            "label_observed_mask": label_observed_mask,
            "observer_supervision_mask": observer_supervision_mask,
            "diagonal_target_mask": diagonal_target_mask,
            "supervision_known_non_werewolf_mask": known_non_werewolf_mask,
            "metadata": metadata,
        }
        if v2_belief_targets is not None:
            result["v2_belief_targets"] = v2_belief_targets
            result["v2_label_observed_mask"] = v2_label_observed_mask
        if self.include_private_features:
            metadata["private_feature_fields"] = list(PRIVATE_FEATURE_FIELDS)
            result.update({
                "known_werewolf_mask": known_werewolf_mask,
                "known_non_werewolf_mask": known_non_werewolf_mask,
            })
        return result


def collate_twd_tom_samples(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Right-pad public-event tensors and stack the fixed 7x7 targets."""

    if isinstance(batch, (str, bytes)) or not isinstance(batch, Sequence):
        raise TypeError("batch must be a sequence")
    if not batch:
        raise ValueError("batch cannot be empty")
    required_fields = {
        "belief_targets",
        "legacy_v1_belief_targets",
        "legacy_v1_label_observed_mask",
        "v1_empty_uniform_nonself_belief_targets",
        "v1_empty_uniform_nonself_label_observed_mask",
        "observer_alive_mask",
        "observer_scope_mask",
        "label_observed_mask",
        "observer_supervision_mask",
        "diagonal_target_mask",
        "supervision_known_non_werewolf_mask",
        "metadata",
    }
    if any(not required_fields.issubset(item) for item in batch):
        raise ValueError("every sample must contain the tom-v2 belief contract")
    private_fields = ("known_werewolf_mask", "known_non_werewolf_mask")
    private_presence = [
        tuple(field in item for field in private_fields)
        for item in batch
    ]
    if any(any(presence) and not all(presence) for presence in private_presence):
        raise ValueError("private knowledge masks must be supplied together")
    if len(set(private_presence)) != 1:
        raise ValueError("batch cannot mix public and private-conditioned samples")

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

    result = {
        **padded,
        "belief_targets": torch.stack([item["belief_targets"] for item in batch]),
        "legacy_v1_belief_targets": torch.stack(
            [item["legacy_v1_belief_targets"] for item in batch]
        ),
        "legacy_v1_label_observed_mask": torch.stack(
            [item["legacy_v1_label_observed_mask"] for item in batch]
        ),
        "v1_empty_uniform_nonself_belief_targets": torch.stack(
            [item["v1_empty_uniform_nonself_belief_targets"] for item in batch]
        ),
        "v1_empty_uniform_nonself_label_observed_mask": torch.stack(
            [
                item["v1_empty_uniform_nonself_label_observed_mask"]
                for item in batch
            ]
        ),
        "observer_alive_mask": torch.stack(
            [item["observer_alive_mask"] for item in batch]
        ),
        "observer_scope_mask": torch.stack(
            [item["observer_scope_mask"] for item in batch]
        ),
        "label_observed_mask": torch.stack(
            [item["label_observed_mask"] for item in batch]
        ),
        "observer_supervision_mask": torch.stack(
            [item["observer_supervision_mask"] for item in batch]
        ),
        "diagonal_target_mask": torch.stack(
            [item["diagonal_target_mask"] for item in batch]
        ),
        "supervision_known_non_werewolf_mask": torch.stack(
            [item["supervision_known_non_werewolf_mask"] for item in batch]
        ),
        "metadata": {
            field_name: [deepcopy(item["metadata"][field_name]) for item in batch]
            for field_name in batch[0]["metadata"]
        },
    }
    if all(private_presence[0]):
        result.update({
            field: torch.stack([item[field] for item in batch])
            for field in private_fields
        })
    v2_fields = ("v2_belief_targets", "v2_label_observed_mask")
    v2_presence = [tuple(field in item for field in v2_fields) for item in batch]
    if any(any(presence) and not all(presence) for presence in v2_presence):
        raise ValueError("V2 belief targets and observation masks must be paired")
    if len(set(v2_presence)) != 1:
        raise ValueError("batch cannot mix samples with and without V2 targets")
    if all(v2_presence[0]):
        result.update({
            field: torch.stack([item[field] for item in batch])
            for field in v2_fields
        })
    return result


__all__ = [
    "CYCLIC_ROTATION_VERSION",
    "MODEL_INPUT_SCOPE",
    "LABEL_OBSERVATION_SEMANTICS",
    "LEGACY_V1_LABEL_OBSERVATION_SEMANTICS",
    "LEGACY_V1_TARGET_CONVERSION",
    "LEGACY_V1_TARGET_SEMANTICS",
    "PRIVATE_FEATURE_FIELDS",
    "PRIVATE_MODEL_INPUT_SCOPE",
    "TARGET_SEMANTICS",
    "TARGET_CONVERSION",
    "V2_TARGET_CONVERSION",
    "V2_LABEL_OBSERVATION_SEMANTICS",
    "V2_TARGET_SEMANTICS",
    "TWDToMDataset",
    "collate_twd_tom_samples",
    "cyclically_rotate_belief_sample",
    "deterministic_cyclic_shift",
    "load_twd_tom_jsonl",
]
