"""Canonical C1-to-D offline PRE-speech ToM materialization."""

from __future__ import annotations

import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from werewolf.models.twd_tom.public_events import (
    PUBLIC_EVENT_SCHEMA_VERSION,
    normalize_public_events,
    parse_public_phase,
    public_event_digest,
    public_speech_actions,
    structured_input_digest,
)
from werewolf.models.twd_tom.schema import PLAYER_NAMES
from werewolf.offline_annotation import (
    OFFLINE_ANNOTATION_SCHEMA_VERSION,
    PRIVATE_CONDITIONED_SUSPICION_TASK,
    PRIVATE_PROMPT_VERSION,
    PUBLIC_ONLY_SUSPICION_TASK,
    PUBLIC_PROMPT_VERSION,
    STATUS_OK,
    validate_offline_annotation_record,
    validate_offline_annotation_sources,
)
from werewolf.speech.private_belief_perceiver import (
    PlayingAgentBeliefReporter,
)
from werewolf.trajectory import PRE_PUBLIC_SPEECH, canonical_digest, canonical_json


D_SCHEMA_VERSION = "classic7_offline_pre_speech_tom_training_record_v1"
D_MATERIALIZATION_POLICY_VERSION = (
    "classic7_c1_ok_only_tom1_tom2_materialization_v1"
)

OFFLINE_PRIVATE_CONDITIONED_TOM1_TASK = (
    "offline_private_conditioned_tom1_v1"
)
OFFLINE_PUBLIC_ONLY_TOM2_TASK = "offline_public_only_tom2_v1"
D_MATERIALIZATION_TASKS = frozenset(
    {
        OFFLINE_PRIVATE_CONDITIONED_TOM1_TASK,
        OFFLINE_PUBLIC_ONLY_TOM2_TASK,
    }
)

TOM1_MODEL_INPUT_SCOPE = (
    "public_events_plus_current_observer_private_knowledge"
)
TOM1_PRIVATE_FIELDS_USAGE = "first_order_model_input_and_label"
TOM2_MODEL_INPUT_SCOPE = "public_events_only"
TOM2_PRIVATE_FIELDS_USAGE = "none"

TOM1_OBSERVER_PROVENANCE = (
    "offline_private_conditioned_annotation_ok_v1"
)
TOM2_OBSERVER_PROVENANCE = "offline_public_only_annotation_ok_v1"
OBSERVER_ANNOTATION_CONFIDENCE = "model_reported_source"

_GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_STATUS_ORDER = (
    "ok",
    "parse_error",
    "semantic_error",
    "reporter_error",
)

D_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "materialization_task",
        "materialization_policy_version",
        "materializer_code_commit",
        "game_id",
        "source_trajectory_commit",
        "trajectory_digest",
        "observer_view_artifact_digest",
        "boundary_id",
        "step_idx",
        "phase",
        "speaker_id",
        "report_trigger",
        "public_event_schema_version",
        "public_events",
        "public_event_digest",
        "structured_input_digest",
        "public_action_count",
        "label_cutoff_step_idx",
        "tom_order",
        "model_input_scope",
        "private_fields_usage",
        "observer_ids",
        "suspected_werewolves",
        "known_werewolves",
        "known_non_werewolves",
        "belief_status",
        "belief_errors",
        "source_annotation_schema_version",
        "source_annotation_task",
        "source_prompt_version",
        "source_annotation_run_ids",
        "source_annotation_code_commits",
        "source_annotation_record_digests",
        "reporter_backend_ids",
        "reporter_model_ids",
        "observer_label_provenance",
        "observer_annotation_confidence",
        "current_action_used",
        "expert_labels_used_as_later_evidence",
        "future_information_used",
        "record_digest",
    }
)

_SUBJECT_MAPPING_FIELDS = (
    "suspected_werewolves",
    "known_werewolves",
    "known_non_werewolves",
    "belief_status",
    "belief_errors",
    "source_annotation_run_ids",
    "source_annotation_code_commits",
    "source_annotation_record_digests",
    "reporter_backend_ids",
    "reporter_model_ids",
    "observer_label_provenance",
    "observer_annotation_confidence",
)

_TASK_CONTRACTS = {
    OFFLINE_PRIVATE_CONDITIONED_TOM1_TASK: {
        "source_task": PRIVATE_CONDITIONED_SUSPICION_TASK,
        "prompt_version": PRIVATE_PROMPT_VERSION,
        "tom_order": 1,
        "model_input_scope": TOM1_MODEL_INPUT_SCOPE,
        "private_fields_usage": TOM1_PRIVATE_FIELDS_USAGE,
        "observer_provenance": TOM1_OBSERVER_PROVENANCE,
    },
    OFFLINE_PUBLIC_ONLY_TOM2_TASK: {
        "source_task": PUBLIC_ONLY_SUSPICION_TASK,
        "prompt_version": PUBLIC_PROMPT_VERSION,
        "tom_order": 2,
        "model_input_scope": TOM2_MODEL_INPUT_SCOPE,
        "private_fields_usage": TOM2_PRIVATE_FIELDS_USAGE,
        "observer_provenance": TOM2_OBSERVER_PROVENANCE,
    },
}


def _required_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return value


def _validate_sha(value: Any, *, field_name: str, pattern) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be lowercase hexadecimal")
    return value


def _validate_player_list(value: Any, *, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    if any(player not in PLAYER_NAMES for player in value):
        raise ValueError(f"{field_name} contains a non-canonical player")
    if len(value) != len(set(value)):
        raise ValueError(f"{field_name} contains duplicate players")
    if value != sorted(value, key=PLAYER_NAMES.index):
        raise ValueError(f"{field_name} must use canonical player order")
    return value


def _validate_observer_ids(value: Any) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("observer_ids must be a sequence")
    observer_ids = list(value)
    if not observer_ids:
        raise ValueError("observer_ids cannot be empty")
    if observer_ids != sorted(set(observer_ids)):
        raise ValueError("observer_ids must be unique and ascending")
    for observer_id in observer_ids:
        if isinstance(observer_id, bool) or not isinstance(observer_id, int):
            raise TypeError("observer IDs must be integers")
        if not 1 <= observer_id <= 7:
            raise ValueError("observer IDs must be in [1, 7]")
    return observer_ids


def _subject_mappings(
    record: Mapping[str, Any],
    observer_ids: Sequence[int],
) -> dict[str, Mapping[str, Any]]:
    subjects = {f"player{observer_id}" for observer_id in observer_ids}
    mappings = {}
    for field_name in _SUBJECT_MAPPING_FIELDS:
        value = record.get(field_name)
        if not isinstance(value, Mapping) or set(value) != subjects:
            raise ValueError(
                f"{field_name} must exactly match observer_ids subjects"
            )
        mappings[field_name] = value
    return mappings


def validate_offline_tom_training_record(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless one value is an exact canonical D V1 record."""

    if not isinstance(record, Mapping):
        raise TypeError("offline ToM training record must be a mapping")
    if set(record) != D_RECORD_FIELDS:
        raise ValueError("offline ToM training record fields do not match V1")
    if record.get("schema_version") != D_SCHEMA_VERSION:
        raise ValueError("unsupported offline ToM training schema")
    if (
        record.get("materialization_policy_version")
        != D_MATERIALIZATION_POLICY_VERSION
    ):
        raise ValueError("unsupported materialization policy")
    task = record.get("materialization_task")
    if task not in _TASK_CONTRACTS:
        raise ValueError("unsupported materialization_task")
    contract = _TASK_CONTRACTS[task]
    if record.get("source_annotation_schema_version") != (
        OFFLINE_ANNOTATION_SCHEMA_VERSION
    ):
        raise ValueError("unsupported source annotation schema")
    for field_name, expected in (
        ("source_annotation_task", contract["source_task"]),
        ("source_prompt_version", contract["prompt_version"]),
        ("model_input_scope", contract["model_input_scope"]),
        ("private_fields_usage", contract["private_fields_usage"]),
    ):
        if record.get(field_name) != expected:
            raise ValueError(f"{field_name} does not match materialization task")
    if (
        type(record.get("tom_order")) is not int
        or record["tom_order"] != contract["tom_order"]
    ):
        raise ValueError("tom_order does not match materialization task")

    for field_name in ("game_id", "boundary_id"):
        _required_text(record.get(field_name), field_name=field_name)
    _validate_sha(
        record.get("materializer_code_commit"),
        field_name="materializer_code_commit",
        pattern=_GIT_SHA_PATTERN,
    )
    _validate_sha(
        record.get("source_trajectory_commit"),
        field_name="source_trajectory_commit",
        pattern=_GIT_SHA_PATTERN,
    )
    for field_name in (
        "trajectory_digest",
        "observer_view_artifact_digest",
        "public_event_digest",
        "structured_input_digest",
        "record_digest",
    ):
        _validate_sha(
            record.get(field_name),
            field_name=field_name,
            pattern=_SHA256_PATTERN,
        )

    step_idx = record.get("step_idx")
    speaker_id = record.get("speaker_id")
    public_action_count = record.get("public_action_count")
    if isinstance(step_idx, bool) or not isinstance(step_idx, int) or step_idx < 0:
        raise ValueError("step_idx must be a non-negative integer")
    if (
        isinstance(speaker_id, bool)
        or not isinstance(speaker_id, int)
        or not 1 <= speaker_id <= 7
    ):
        raise ValueError("speaker_id must be an integer in [1, 7]")
    if (
        isinstance(public_action_count, bool)
        or not isinstance(public_action_count, int)
        or public_action_count < 0
    ):
        raise ValueError("public_action_count must be a non-negative integer")
    if (
        type(record.get("label_cutoff_step_idx")) is not int
        or record["label_cutoff_step_idx"] != step_idx
    ):
        raise ValueError("label_cutoff_step_idx must equal step_idx")
    expected_boundary_id = (
        f"{record['game_id']}:step_{step_idx:06d}:{PRE_PUBLIC_SPEECH}"
    )
    if record["boundary_id"] != expected_boundary_id:
        raise ValueError("boundary_id is not canonical")

    phase = record.get("phase")
    trigger = record.get("report_trigger")
    expected_trigger = {
        "speech": "pre_public_speech",
        "speech_pk": "pre_public_speech_pk",
    }.get(phase)
    if trigger != expected_trigger:
        raise ValueError("phase/report_trigger identity is invalid")
    if record.get("public_event_schema_version") != PUBLIC_EVENT_SCHEMA_VERSION:
        raise ValueError("unsupported public-event schema")
    public_events = normalize_public_events(record.get("public_events"))
    if not public_events:
        raise ValueError("public_events cannot be empty")
    if public_events != record["public_events"]:
        raise ValueError("public_events must be canonical")
    if (
        public_events[-1].get("event_type") != "turn_start"
        or public_events[-1].get("speaker") != f"player{speaker_id}"
    ):
        raise ValueError("public prefix must end with matching turn_start")
    latest_phase = next(
        (
            event["phase"]
            for event in reversed(public_events)
            if event["event_type"] == "phase_change"
        ),
        None,
    )
    if (
        latest_phase is None
        or parse_public_phase(latest_phase)[1] != f"day_{phase}"
    ):
        raise ValueError("phase must match the public prefix")
    if public_event_digest(public_events) != record["public_event_digest"]:
        raise ValueError("public_event_digest does not match public_events")
    if structured_input_digest(public_events) != record["structured_input_digest"]:
        raise ValueError("structured_input_digest does not match public_events")
    if len(public_speech_actions(public_events)) != public_action_count:
        raise ValueError("public_action_count does not match public_events")

    observer_ids = _validate_observer_ids(record.get("observer_ids"))
    if task == OFFLINE_PRIVATE_CONDITIONED_TOM1_TASK:
        if observer_ids != [speaker_id]:
            raise ValueError("ToM1 must contain only the speaker observer")
    elif speaker_id in observer_ids:
        raise ValueError("ToM2 must exclude the speaker observer")
    mappings = _subject_mappings(record, observer_ids)
    for observer_id in observer_ids:
        subject = f"player{observer_id}"
        suspected = _validate_player_list(
            mappings["suspected_werewolves"][subject],
            field_name=f"{subject} suspected_werewolves",
        )
        known_werewolves = _validate_player_list(
            mappings["known_werewolves"][subject],
            field_name=f"{subject} known_werewolves",
        )
        known_non_werewolves = _validate_player_list(
            mappings["known_non_werewolves"][subject],
            field_name=f"{subject} known_non_werewolves",
        )
        if set(known_werewolves) & set(known_non_werewolves):
            raise ValueError(f"{subject} hard knowledge conflicts")
        PlayingAgentBeliefReporter.validate_semantics(
            observer_id=observer_id,
            suspected_werewolves=suspected,
            known_werewolves=known_werewolves,
            known_non_werewolves=known_non_werewolves,
        )
        if task == OFFLINE_PUBLIC_ONLY_TOM2_TASK and (
            known_werewolves or known_non_werewolves
        ):
            raise ValueError("ToM2 hard-knowledge mappings must be empty")
        if mappings["belief_status"][subject] != STATUS_OK:
            raise ValueError("all emitted belief_status values must be ok")
        if mappings["belief_errors"][subject] is not None:
            raise ValueError("all emitted belief_errors values must be null")
        for field_name in (
            "source_annotation_run_ids",
            "reporter_backend_ids",
            "reporter_model_ids",
        ):
            _required_text(
                mappings[field_name][subject],
                field_name=f"{subject} {field_name}",
            )
        _validate_sha(
            mappings["source_annotation_code_commits"][subject],
            field_name=f"{subject} source annotation commit",
            pattern=_GIT_SHA_PATTERN,
        )
        _validate_sha(
            mappings["source_annotation_record_digests"][subject],
            field_name=f"{subject} source annotation digest",
            pattern=_SHA256_PATTERN,
        )
        if mappings["observer_label_provenance"][subject] != (
            contract["observer_provenance"]
        ):
            raise ValueError("observer_label_provenance does not match task")
        if mappings["observer_annotation_confidence"][subject] != (
            OBSERVER_ANNOTATION_CONFIDENCE
        ):
            raise ValueError("observer_annotation_confidence does not match V1")

    for field_name in (
        "current_action_used",
        "expert_labels_used_as_later_evidence",
        "future_information_used",
    ):
        if record.get(field_name) is not False:
            raise ValueError(f"{field_name} must be false")
    payload = dict(record)
    payload.pop("record_digest")
    if canonical_digest(payload) != record["record_digest"]:
        raise ValueError("record_digest does not match record")
    return dict(record)


def _boundary_observer_ids(boundary: Mapping[str, Any]) -> set[int]:
    views = boundary.get("observer_views")
    if isinstance(views, (str, bytes)) or not isinstance(views, Sequence):
        raise TypeError("boundary observer_views must be a sequence")
    result = set()
    for view in views:
        if not isinstance(view, Mapping):
            raise TypeError("observer view must be a mapping")
        observer_id = view.get("observer_id")
        if (
            isinstance(observer_id, bool)
            or not isinstance(observer_id, int)
            or not 1 <= observer_id <= 7
        ):
            raise ValueError("observer view ID must be an integer in [1, 7]")
        if observer_id in result:
            raise ValueError("boundary observer IDs must be unique")
        result.add(observer_id)
    return result


def _validate_c1_lineage(
    record: Mapping[str, Any],
    *,
    trajectory: Mapping[str, Any],
    provenance: Mapping[str, Any],
    boundary: Mapping[str, Any],
    prefix: Sequence[Mapping[str, Any]],
) -> None:
    if record["game_id"] != trajectory["game_id"]:
        raise ValueError("C1 game_id does not match A")
    if record["source_trajectory_commit"] != trajectory["source_commit"]:
        raise ValueError("C1 source trajectory commit does not match A")
    if record["trajectory_digest"] != trajectory["trajectory_digest"]:
        raise ValueError("C1 trajectory_digest does not match A")
    if record["observer_view_artifact_digest"] != provenance["artifact_digest"]:
        raise ValueError("C1 observer-view artifact digest does not match C0")
    if record["boundary_id"] != boundary["boundary_id"]:
        raise ValueError("C1 boundary_id does not match C0")
    if record["step_idx"] != boundary["step_idx"]:
        raise ValueError("C1 step_idx does not match C0")
    if record["observer_id"] not in _boundary_observer_ids(boundary):
        raise ValueError("C1 observer_id is not present in C0 boundary")
    source = record["source"]
    expected_count = boundary["public_event_count_at_materialization"]
    expected_digest = public_event_digest(prefix)
    expected_structured_digest = structured_input_digest(prefix)
    expected_action_count = len(public_speech_actions(prefix))
    if source["public_event_count"] != expected_count:
        raise ValueError("C1 public_event_count cutoff does not match C0")
    if source["public_event_digest"] != expected_digest:
        raise ValueError("C1 public_event_digest does not match A/C0")
    if source["structured_input_digest"] != expected_structured_digest:
        raise ValueError("C1 structured_input_digest does not match A prefix")
    if source["public_action_count"] != expected_action_count:
        raise ValueError("C1 public_action_count does not match A prefix")


def _provenance_mappings(
    records: Sequence[Mapping[str, Any]],
    *,
    observer_provenance: str,
) -> dict[str, dict[str, Any]]:
    result = {
        "source_annotation_run_ids": {},
        "source_annotation_code_commits": {},
        "source_annotation_record_digests": {},
        "reporter_backend_ids": {},
        "reporter_model_ids": {},
        "observer_label_provenance": {},
        "observer_annotation_confidence": {},
    }
    for record in records:
        subject = f"player{record['observer_id']}"
        result["source_annotation_run_ids"][subject] = record[
            "annotation_run_id"
        ]
        result["source_annotation_code_commits"][subject] = record[
            "annotation_code_commit"
        ]
        result["source_annotation_record_digests"][subject] = record[
            "record_digest"
        ]
        result["reporter_backend_ids"][subject] = record[
            "reporter_backend_id"
        ]
        result["reporter_model_ids"][subject] = record["reporter_model_id"]
        result["observer_label_provenance"][subject] = observer_provenance
        result["observer_annotation_confidence"][subject] = (
            OBSERVER_ANNOTATION_CONFIDENCE
        )
    return result


def _make_record(
    *,
    task: str,
    materializer_code_commit: str,
    trajectory: Mapping[str, Any],
    provenance: Mapping[str, Any],
    boundary: Mapping[str, Any],
    prefix: Sequence[Mapping[str, Any]],
    source_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    contract = _TASK_CONTRACTS[task]
    observer_ids = [record["observer_id"] for record in source_records]
    subjects = [f"player{observer_id}" for observer_id in observer_ids]
    phase = boundary["speech_kind"]
    report_trigger = {
        "speech": "pre_public_speech",
        "speech_pk": "pre_public_speech_pk",
    }[phase]
    suspected = {
        subject: deepcopy(source["result"]["suspected_werewolves"])
        for subject, source in zip(subjects, source_records)
    }
    if task == OFFLINE_PRIVATE_CONDITIONED_TOM1_TASK:
        known_werewolves = {
            subjects[0]: deepcopy(
                source_records[0]["source"]["derived_hard_knowledge"][
                    "known_werewolves"
                ]
            )
        }
        known_non_werewolves = {
            subjects[0]: deepcopy(
                source_records[0]["source"]["derived_hard_knowledge"][
                    "known_non_werewolves"
                ]
            )
        }
    else:
        known_werewolves = {subject: [] for subject in subjects}
        known_non_werewolves = {subject: [] for subject in subjects}
    public_events = deepcopy(list(prefix))
    record = {
        "schema_version": D_SCHEMA_VERSION,
        "materialization_task": task,
        "materialization_policy_version": D_MATERIALIZATION_POLICY_VERSION,
        "materializer_code_commit": materializer_code_commit,
        "game_id": trajectory["game_id"],
        "source_trajectory_commit": trajectory["source_commit"],
        "trajectory_digest": trajectory["trajectory_digest"],
        "observer_view_artifact_digest": provenance["artifact_digest"],
        "boundary_id": boundary["boundary_id"],
        "step_idx": boundary["step_idx"],
        "phase": phase,
        "speaker_id": boundary["speaker_id"],
        "report_trigger": report_trigger,
        "public_event_schema_version": PUBLIC_EVENT_SCHEMA_VERSION,
        "public_events": public_events,
        "public_event_digest": public_event_digest(public_events),
        "structured_input_digest": structured_input_digest(public_events),
        "public_action_count": len(public_speech_actions(public_events)),
        "label_cutoff_step_idx": boundary["step_idx"],
        "tom_order": contract["tom_order"],
        "model_input_scope": contract["model_input_scope"],
        "private_fields_usage": contract["private_fields_usage"],
        "observer_ids": observer_ids,
        "suspected_werewolves": suspected,
        "known_werewolves": known_werewolves,
        "known_non_werewolves": known_non_werewolves,
        "belief_status": {subject: STATUS_OK for subject in subjects},
        "belief_errors": {subject: None for subject in subjects},
        "source_annotation_schema_version": OFFLINE_ANNOTATION_SCHEMA_VERSION,
        "source_annotation_task": contract["source_task"],
        "source_prompt_version": contract["prompt_version"],
        **_provenance_mappings(
            source_records,
            observer_provenance=contract["observer_provenance"],
        ),
        "current_action_used": False,
        "expert_labels_used_as_later_evidence": False,
        "future_information_used": False,
    }
    record["record_digest"] = canonical_digest(record)
    validate_offline_tom_training_record(record)
    return record


def materialize_offline_tom_records(
    trajectory: Mapping[str, Any],
    observer_view_provenance: Mapping[str, Any],
    annotation_records: Sequence[Mapping[str, Any]],
    *,
    materializer_code_commit: str,
) -> dict[str, Any]:
    """Materialize deterministic ToM1/ToM2 D rows from frozen A/C0/C1."""

    _validate_sha(
        materializer_code_commit,
        field_name="materializer_code_commit",
        pattern=_GIT_SHA_PATTERN,
    )
    if isinstance(annotation_records, (str, bytes)) or not isinstance(
        annotation_records,
        Sequence,
    ):
        raise TypeError("annotation_records must be a sequence")
    _, boundaries = validate_offline_annotation_sources(
        trajectory,
        observer_view_provenance,
    )
    boundary_index = {
        boundary["boundary_id"]: (boundary, prefix)
        for boundary, prefix in boundaries
    }
    record_index = {}
    private_counts = Counter()
    public_counts = Counter()
    for raw_record in annotation_records:
        record = validate_offline_annotation_record(raw_record)
        boundary_id = record["boundary_id"]
        if boundary_id not in boundary_index:
            raise ValueError("C1 record does not reference an eligible PRE boundary")
        boundary, prefix = boundary_index[boundary_id]
        _validate_c1_lineage(
            record,
            trajectory=trajectory,
            provenance=observer_view_provenance,
            boundary=boundary,
            prefix=prefix,
        )
        key = (record["annotation_task"], boundary_id, record["observer_id"])
        if key in record_index:
            raise ValueError("duplicate C1 task/boundary/observer record")
        record_index[key] = record
        counts = (
            private_counts
            if record["annotation_task"] == PRIVATE_CONDITIONED_SUSPICION_TASK
            else public_counts
        )
        counts[record["status"]] += 1

    tom1_records = []
    tom2_records = []
    dropped_tom1_boundaries = 0
    filtered_tom2_observers = 0
    dropped_tom2_boundaries = 0
    for boundary, prefix in boundaries:
        boundary_id = boundary["boundary_id"]
        speaker_id = boundary["speaker_id"]
        private = record_index.get(
            (PRIVATE_CONDITIONED_SUSPICION_TASK, boundary_id, speaker_id)
        )
        if private is None or private["status"] != STATUS_OK:
            dropped_tom1_boundaries += 1
        else:
            tom1_records.append(
                _make_record(
                    task=OFFLINE_PRIVATE_CONDITIONED_TOM1_TASK,
                    materializer_code_commit=materializer_code_commit,
                    trajectory=trajectory,
                    provenance=observer_view_provenance,
                    boundary=boundary,
                    prefix=prefix,
                    source_records=[private],
                )
            )

        public_candidates = sorted(
            (
                record
                for (task, candidate_boundary, observer_id), record
                in record_index.items()
                if task == PUBLIC_ONLY_SUSPICION_TASK
                and candidate_boundary == boundary_id
                and observer_id != speaker_id
            ),
            key=lambda record: record["observer_id"],
        )
        eligible_public = [
            record for record in public_candidates if record["status"] == STATUS_OK
        ]
        filtered_tom2_observers += len(public_candidates) - len(eligible_public)
        if not eligible_public:
            dropped_tom2_boundaries += 1
        else:
            tom2_records.append(
                _make_record(
                    task=OFFLINE_PUBLIC_ONLY_TOM2_TASK,
                    materializer_code_commit=materializer_code_commit,
                    trajectory=trajectory,
                    provenance=observer_view_provenance,
                    boundary=boundary,
                    prefix=prefix,
                    source_records=eligible_public,
                )
            )

    summary = {
        "private_status_counts": {
            status: private_counts[status] for status in _STATUS_ORDER
        },
        "public_status_counts": {
            status: public_counts[status] for status in _STATUS_ORDER
        },
        "emitted_tom1_rows": len(tom1_records),
        "emitted_tom2_rows": len(tom2_records),
        "dropped_tom1_boundaries": dropped_tom1_boundaries,
        "filtered_tom2_observers": filtered_tom2_observers,
        "dropped_tom2_boundaries": dropped_tom2_boundaries,
    }
    return {
        "tom1_records": tom1_records,
        "tom2_records": tom2_records,
        "summary": summary,
    }


def write_offline_tom_jsonl(
    output_path: str | Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Atomically create one canonical, single-task D JSONL file."""

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence")
    materialized = [dict(record) for record in records]
    if not materialized:
        raise ValueError("records cannot be empty")
    tasks = {record.get("materialization_task") for record in materialized}
    if len(tasks) != 1 or not tasks <= D_MATERIALIZATION_TASKS:
        raise ValueError("one JSONL file must contain one materialization_task")
    ordering = [
        (record.get("game_id"), record.get("step_idx"))
        for record in materialized
    ]
    if ordering != sorted(ordering) or len(ordering) != len(set(ordering)):
        raise ValueError("records are not in unique canonical materialization order")
    for record in materialized:
        validate_offline_tom_training_record(record)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"materialization output already exists: {path}")
    content = "".join(f"{canonical_json(record)}\n" for record in materialized)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.link(temporary_path, path)
    except FileExistsError as exc:
        raise FileExistsError(
            f"materialization output already exists: {path}"
        ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


__all__ = [
    "D_MATERIALIZATION_POLICY_VERSION",
    "D_MATERIALIZATION_TASKS",
    "D_RECORD_FIELDS",
    "D_SCHEMA_VERSION",
    "OBSERVER_ANNOTATION_CONFIDENCE",
    "OFFLINE_PRIVATE_CONDITIONED_TOM1_TASK",
    "OFFLINE_PUBLIC_ONLY_TOM2_TASK",
    "TOM1_MODEL_INPUT_SCOPE",
    "TOM1_OBSERVER_PROVENANCE",
    "TOM1_PRIVATE_FIELDS_USAGE",
    "TOM2_MODEL_INPUT_SCOPE",
    "TOM2_OBSERVER_PROVENANCE",
    "TOM2_PRIVATE_FIELDS_USAGE",
    "materialize_offline_tom_records",
    "validate_offline_tom_training_record",
    "write_offline_tom_jsonl",
]
