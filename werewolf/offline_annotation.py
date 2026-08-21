"""Offline PRE-speech suspicion annotations from frozen A/C0 artifacts."""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from werewolf.models.twd_tom.public_events import (
    PUBLIC_EVENT_SCHEMA_VERSION,
    normalize_public_events,
    public_event_digest,
    public_speech_actions,
    structured_input_digest,
)
from werewolf.models.twd_tom.schema import PLAYER_NAMES
from werewolf.observer_knowledge import (
    derive_observer_hard_knowledge,
    legal_observer_state,
)
from werewolf.speech.private_belief_perceiver import (
    PlayingAgentBeliefReporter,
)
from werewolf.trajectory import (
    OBSERVER_VIEW_PROVENANCE_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    PRE_PUBLIC_SPEECH,
    SIMULATOR_BASELINE,
    TRAJECTORY_SCHEMA_VERSION,
    canonical_digest,
    canonical_json,
)


OFFLINE_ANNOTATION_SCHEMA_VERSION = "classic7_offline_annotation_record_v1"

PRIVATE_CONDITIONED_SUSPICION_TASK = "private_conditioned_suspicion_v1"
PUBLIC_ONLY_SUSPICION_TASK = "public_only_suspicion_v1"
ANNOTATION_TASKS = frozenset(
    {
        PRIVATE_CONDITIONED_SUSPICION_TASK,
        PUBLIC_ONLY_SUSPICION_TASK,
    }
)

PRIVATE_INFORMATION_SCOPE = (
    "canonical_pre_speech_observer_private_plus_public_v1"
)
PUBLIC_INFORMATION_SCOPE = "canonical_pre_speech_public_only_v1"
PRIVATE_PROMPT_VERSION = (
    "classic7_offline_private_conditioned_suspicion_prompt_v1"
)
PUBLIC_PROMPT_VERSION = "classic7_offline_public_only_suspicion_prompt_v1"

STATUS_OK = "ok"
STATUS_PARSE_ERROR = "parse_error"
STATUS_SEMANTIC_ERROR = "semantic_error"
STATUS_REPORTER_ERROR = "reporter_error"
ANNOTATION_STATUSES = frozenset(
    {
        STATUS_OK,
        STATUS_PARSE_ERROR,
        STATUS_SEMANTIC_ERROR,
        STATUS_REPORTER_ERROR,
    }
)

ANNOTATION_TEMPERATURE = 0.0
ANNOTATION_MAX_TOKENS = 96
ANNOTATION_EXTRA_BODY = {"thinking": {"type": "disabled"}}
OFFLINE_SUSPICION_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["suspected_werewolves"],
    "properties": {
        "suspected_werewolves": {
            "type": "array",
            "minItems": 0,
            "maxItems": 7,
            "items": {
                "type": "string",
                "enum": list(PLAYER_NAMES),
            },
        },
    },
}
_GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_COMMON_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "annotation_task",
        "annotation_run_id",
        "annotation_code_commit",
        "game_id",
        "source_trajectory_commit",
        "trajectory_digest",
        "observer_view_artifact_digest",
        "boundary_id",
        "boundary_type",
        "step_idx",
        "observer_id",
        "information_scope",
        "source",
        "prompt_version",
        "prompt",
        "prompt_digest",
        "reporter_backend_id",
        "reporter_model_id",
        "request_parameters",
        "raw_response",
        "status",
        "error",
        "result",
        "record_digest",
    }
)
_PUBLIC_SOURCE_FIELDS = frozenset(
    {
        "public_event_count",
        "public_event_digest",
        "structured_input_digest",
        "public_action_count",
    }
)
_PRIVATE_SOURCE_FIELDS = frozenset(
    {
        *_PUBLIC_SOURCE_FIELDS,
        "observation_digest",
        "derived_hard_knowledge",
    }
)
_TASK_CONTRACTS = {
    PRIVATE_CONDITIONED_SUSPICION_TASK: {
        "information_scope": PRIVATE_INFORMATION_SCOPE,
        "prompt_version": PRIVATE_PROMPT_VERSION,
        "source_fields": _PRIVATE_SOURCE_FIELDS,
    },
    PUBLIC_ONLY_SUSPICION_TASK: {
        "information_scope": PUBLIC_INFORMATION_SCOPE,
        "prompt_version": PUBLIC_PROMPT_VERSION,
        "source_fields": _PUBLIC_SOURCE_FIELDS,
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


def _validate_player_set(value: Any, *, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    if any(player not in PLAYER_NAMES for player in value):
        raise ValueError(f"{field_name} contains a non-canonical player")
    if len(value) != len(set(value)):
        raise ValueError(f"{field_name} contains duplicate players")
    if value != sorted(value, key=PLAYER_NAMES.index):
        raise ValueError(f"{field_name} must use canonical player order")
    return value


def _response_format(*, supports_json_schema: bool) -> dict[str, Any]:
    if not isinstance(supports_json_schema, bool):
        raise TypeError("backend supports_json_schema must be boolean")
    if not supports_json_schema:
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "offline_suspicion_annotation_v1",
            "strict": True,
            "schema": deepcopy(OFFLINE_SUSPICION_JSON_SCHEMA),
        },
    }


def validate_offline_annotation_record(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless one value is an exact canonical C1 V1 record."""

    if not isinstance(record, Mapping):
        raise TypeError("offline annotation record must be a mapping")
    if set(record) != _COMMON_RECORD_FIELDS:
        raise ValueError("offline annotation record fields do not match contract")
    if record.get("schema_version") != OFFLINE_ANNOTATION_SCHEMA_VERSION:
        raise ValueError("unsupported offline annotation schema")
    task = record.get("annotation_task")
    if task not in _TASK_CONTRACTS:
        raise ValueError("unsupported annotation_task")
    contract = _TASK_CONTRACTS[task]
    if record.get("information_scope") != contract["information_scope"]:
        raise ValueError("annotation information_scope does not match task")
    if record.get("prompt_version") != contract["prompt_version"]:
        raise ValueError("annotation prompt_version does not match task")
    if record.get("boundary_type") != PRE_PUBLIC_SPEECH:
        raise ValueError("offline annotation requires PRE_PUBLIC_SPEECH")

    for field_name in (
        "annotation_run_id",
        "game_id",
        "boundary_id",
        "prompt",
        "reporter_backend_id",
        "reporter_model_id",
    ):
        _required_text(record.get(field_name), field_name=field_name)
    _validate_sha(
        record.get("annotation_code_commit"),
        field_name="annotation_code_commit",
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
        "prompt_digest",
        "record_digest",
    ):
        _validate_sha(
            record.get(field_name),
            field_name=field_name,
            pattern=_SHA256_PATTERN,
        )
    for field_name in ("step_idx", "observer_id"):
        value = record.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field_name} must be an integer")
    if record["step_idx"] < 0:
        raise ValueError("step_idx must be non-negative")
    if not 1 <= record["observer_id"] <= 7:
        raise ValueError("observer_id must be in [1, 7]")
    expected_boundary_id = (
        f"{record['game_id']}:step_{record['step_idx']:06d}:"
        f"{PRE_PUBLIC_SPEECH}"
    )
    if record["boundary_id"] != expected_boundary_id:
        raise ValueError("record boundary_id is not canonical")

    source = record.get("source")
    if not isinstance(source, Mapping) or set(source) != contract["source_fields"]:
        raise ValueError("annotation source fields do not match task contract")
    for field_name in (
        "public_event_count",
        "public_action_count",
    ):
        value = source.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"source.{field_name} must be non-negative integer")
    for field_name in ("public_event_digest", "structured_input_digest"):
        _validate_sha(
            source.get(field_name),
            field_name=f"source.{field_name}",
            pattern=_SHA256_PATTERN,
        )
    hard_knowledge = None
    if task == PRIVATE_CONDITIONED_SUSPICION_TASK:
        _validate_sha(
            source.get("observation_digest"),
            field_name="source.observation_digest",
            pattern=_SHA256_PATTERN,
        )
        hard_knowledge = source.get("derived_hard_knowledge")
        if not isinstance(hard_knowledge, Mapping) or set(hard_knowledge) != {
            "known_werewolves",
            "known_non_werewolves",
        }:
            raise ValueError("derived_hard_knowledge fields do not match contract")
        known_werewolves = _validate_player_set(
            hard_knowledge["known_werewolves"],
            field_name="known_werewolves",
        )
        known_non_werewolves = _validate_player_set(
            hard_knowledge["known_non_werewolves"],
            field_name="known_non_werewolves",
        )
        if set(known_werewolves) & set(known_non_werewolves):
            raise ValueError("derived hard knowledge conflicts")

    if record["prompt_digest"] != canonical_digest(record["prompt"]):
        raise ValueError("prompt_digest does not match prompt")
    request_parameters = record.get("request_parameters")
    if not isinstance(request_parameters, Mapping) or set(request_parameters) != {
        "temperature",
        "max_tokens",
        "response_format",
        "extra_body",
    }:
        raise ValueError("request_parameters fields do not match contract")
    if (
        type(request_parameters["temperature"]) is not float
        or request_parameters["temperature"] != ANNOTATION_TEMPERATURE
    ):
        raise ValueError("request temperature does not match V1")
    if (
        type(request_parameters["max_tokens"]) is not int
        or request_parameters["max_tokens"] != ANNOTATION_MAX_TOKENS
    ):
        raise ValueError("request max_tokens does not match V1")
    if request_parameters["extra_body"] != ANNOTATION_EXTRA_BODY:
        raise ValueError("request extra_body does not match V1")
    if request_parameters["response_format"] not in (
        {"type": "json_object"},
        _response_format(supports_json_schema=True),
    ):
        raise ValueError("request response_format does not match V1")

    status = record.get("status")
    if status not in ANNOTATION_STATUSES:
        raise ValueError("unsupported annotation status")
    raw_response = record.get("raw_response")
    error = record.get("error")
    result = record.get("result")
    if status == STATUS_OK:
        if not isinstance(raw_response, str):
            raise TypeError("ok annotation requires text raw_response")
        if error is not None:
            raise ValueError("ok annotation requires null error")
        if not isinstance(result, Mapping) or set(result) != {
            "suspected_werewolves"
        }:
            raise ValueError("ok annotation result fields do not match contract")
        suspected = _validate_player_set(
            result["suspected_werewolves"],
            field_name="suspected_werewolves",
        )
        try:
            parsed_response = PlayingAgentBeliefReporter.parse_response(
                raw_response
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("ok raw_response is not a valid result") from exc
        if parsed_response != suspected:
            raise ValueError("ok raw_response and result do not match")
        if hard_knowledge is not None:
            PlayingAgentBeliefReporter.validate_semantics(
                observer_id=record["observer_id"],
                suspected_werewolves=suspected,
                known_werewolves=hard_knowledge["known_werewolves"],
                known_non_werewolves=hard_knowledge["known_non_werewolves"],
            )
    elif status in {STATUS_PARSE_ERROR, STATUS_SEMANTIC_ERROR}:
        if not isinstance(raw_response, str):
            raise TypeError(f"{status} requires text raw_response")
        if not isinstance(error, str) or not error.strip():
            raise ValueError(f"{status} requires non-empty error")
        if result is not None:
            raise ValueError(f"{status} requires null result")
    else:
        if raw_response is not None and not isinstance(raw_response, str):
            raise TypeError("reporter_error raw_response must be text or null")
        if not isinstance(error, str) or not error.strip():
            raise ValueError("reporter_error requires non-empty error")
        if result is not None:
            raise ValueError("reporter_error requires null result")

    payload = dict(record)
    payload.pop("record_digest")
    if record["record_digest"] != canonical_digest(payload):
        raise ValueError("record_digest does not match record")
    return dict(record)


def _validate_digest(
    artifact: Mapping[str, Any],
    *,
    digest_field: str,
) -> str:
    digest = artifact.get(digest_field)
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{digest_field} must be a SHA256 digest")
    payload = dict(artifact)
    payload.pop(digest_field)
    if canonical_digest(payload) != digest:
        raise ValueError(f"{digest_field} does not match canonical artifact")
    return digest


def _validate_frozen_inputs(
    trajectory: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(trajectory, Mapping):
        raise TypeError("trajectory must be a mapping")
    if not isinstance(provenance, Mapping):
        raise TypeError("observer-view provenance must be a mapping")
    if trajectory.get("schema_version") != TRAJECTORY_SCHEMA_VERSION:
        raise ValueError("unsupported trajectory schema")
    if provenance.get("schema_version") != (
        OBSERVER_VIEW_PROVENANCE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported observer-view provenance schema")
    if trajectory.get("observation_schema_version") != OBSERVATION_SCHEMA_VERSION:
        raise ValueError("unsupported trajectory observation schema")
    if provenance.get("observation_schema_version") != OBSERVATION_SCHEMA_VERSION:
        raise ValueError("unsupported provenance observation schema")
    if trajectory.get("public_event_schema_version") != PUBLIC_EVENT_SCHEMA_VERSION:
        raise ValueError("unsupported public-event schema")
    if trajectory.get("simulator_baseline") != SIMULATOR_BASELINE:
        raise ValueError("unsupported simulator baseline")
    _validate_sha(
        trajectory.get("source_commit"),
        field_name="trajectory.source_commit",
        pattern=_GIT_SHA_PATTERN,
    )
    if provenance.get("source_commit") != trajectory.get("source_commit"):
        raise ValueError("A/C0 source_commit mismatch")
    if provenance.get("simulator_baseline") != trajectory.get(
        "simulator_baseline"
    ):
        raise ValueError("A/C0 simulator baseline mismatch")
    if provenance.get("game_id") != trajectory.get("game_id"):
        raise ValueError("A/C0 game_id mismatch")
    if provenance.get("run_id") != trajectory.get("run_id"):
        raise ValueError("A/C0 run_id mismatch")

    trajectory_digest = _validate_digest(
        trajectory,
        digest_field="trajectory_digest",
    )
    if provenance.get("trajectory_digest") != trajectory_digest:
        raise ValueError("A/C0 trajectory_digest mismatch")
    _validate_digest(provenance, digest_field="artifact_digest")

    initial_events = trajectory.get("initial_public_events")
    transitions = trajectory.get("transitions")
    if isinstance(transitions, (str, bytes)) or not isinstance(
        transitions,
        Sequence,
    ):
        raise TypeError("trajectory transitions must be a sequence")
    reconstructed = list(normalize_public_events(initial_events))
    for expected_step, transition in enumerate(transitions):
        if not isinstance(transition, Mapping):
            raise TypeError("trajectory transition must be a mapping")
        if transition.get("step_idx") != expected_step:
            raise ValueError("trajectory transition step_idx is not canonical")
        if transition.get("public_event_count_before") != len(reconstructed):
            raise ValueError("transition public-event cutoff mismatch")
        appended = transition.get("public_events_appended")
        if isinstance(appended, (str, bytes)) or not isinstance(
            appended,
            Sequence,
        ):
            raise TypeError("transition public-event delta must be a sequence")
        reconstructed.extend(deepcopy(list(appended)))
        reconstructed = normalize_public_events(reconstructed)
    if normalize_public_events(reconstructed) != reconstructed:
        raise ValueError("public-event projection is not normalized")
    if public_event_digest(reconstructed) != trajectory.get("public_event_digest"):
        raise ValueError("trajectory public_event_digest mismatch")
    return reconstructed


def _validated_pre_boundaries(
    trajectory: Mapping[str, Any],
    provenance: Mapping[str, Any],
    public_events: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], list[dict[str, Any]]]]:
    boundaries = provenance.get("boundaries")
    if isinstance(boundaries, (str, bytes)) or not isinstance(
        boundaries,
        Sequence,
    ):
        raise TypeError("provenance boundaries must be a sequence")
    eligible = []
    seen_boundary_ids = set()
    seen_pre_steps = set()
    for boundary in boundaries:
        if not isinstance(boundary, Mapping):
            raise TypeError("provenance boundary must be a mapping")
        boundary_id = boundary.get("boundary_id")
        if not isinstance(boundary_id, str) or boundary_id in seen_boundary_ids:
            raise ValueError("boundary_id must be unique text")
        seen_boundary_ids.add(boundary_id)
        _validate_digest(boundary, digest_field="boundary_digest")
        count = boundary.get("public_event_count_at_materialization")
        step_idx = boundary.get("step_idx")
        if (
            isinstance(step_idx, bool)
            or not isinstance(step_idx, int)
            or step_idx < 0
        ):
            raise ValueError("boundary step_idx must be a non-negative integer")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not 0 <= count <= len(public_events)
        ):
            raise ValueError("boundary public-event count is invalid")
        prefix = normalize_public_events(public_events[:count])
        if public_event_digest(prefix) != boundary.get(
            "public_event_digest_at_materialization"
        ):
            raise ValueError("boundary public-event digest mismatch")
        if boundary.get("boundary_type") == PRE_PUBLIC_SPEECH:
            if boundary.get("speech_event_idx") is not None:
                raise ValueError("PRE boundary cannot have a speech_event_idx")
            expected_boundary_id = (
                f"{trajectory['game_id']}:step_{step_idx:06d}:"
                f"{PRE_PUBLIC_SPEECH}"
            )
            if boundary_id != expected_boundary_id:
                raise ValueError("PRE boundary_id is not canonical")
            speaker_id = boundary.get("speaker_id")
            if (
                isinstance(speaker_id, bool)
                or not isinstance(speaker_id, int)
                or not 1 <= speaker_id <= 7
            ):
                raise ValueError("PRE speaker_id must be an integer in [1, 7]")
            if boundary.get("speech_kind") not in {"speech", "speech_pk"}:
                raise ValueError("PRE speech_kind is not canonical")
            if (
                not prefix
                or prefix[-1].get("event_type") != "turn_start"
                or prefix[-1].get("speaker") != f"player{speaker_id}"
            ):
                raise ValueError(
                    "PRE public prefix must end with the boundary speaker turn_start"
                )
            transitions = trajectory["transitions"]
            if step_idx < len(transitions):
                expected_count = transitions[step_idx][
                    "public_event_count_before"
                ]
                if transitions[step_idx].get("acting_player_id") != speaker_id:
                    raise ValueError(
                        "PRE boundary speaker does not match transition actor"
                    )
            elif step_idx == len(transitions):
                expected_count = len(public_events)
            else:
                raise ValueError(
                    "PRE boundary step_idx exceeds the trajectory prefix"
                )
            if count != expected_count:
                raise ValueError(
                    "PRE boundary cutoff does not match its trajectory step"
                )
            if step_idx in seen_pre_steps:
                raise ValueError("trajectory step has duplicate PRE boundaries")
            seen_pre_steps.add(step_idx)
            eligible.append((boundary, prefix))
    return sorted(
        eligible,
        key=lambda item: item[0]["step_idx"],
    )


def _public_source(
    boundary: Mapping[str, Any],
    prefix: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "public_event_count": boundary[
            "public_event_count_at_materialization"
        ],
        "public_event_digest": public_event_digest(prefix),
        "structured_input_digest": structured_input_digest(prefix),
        "public_action_count": len(public_speech_actions(prefix)),
    }


def validate_offline_annotation_sources(
    trajectory: Mapping[str, Any],
    observer_view_provenance: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[tuple[Mapping[str, Any], list[dict[str, Any]]]],
]:
    """Validate frozen A/C0 and return canonical events plus PRE cutoffs."""

    public_events = _validate_frozen_inputs(
        trajectory,
        observer_view_provenance,
    )
    boundaries = _validated_pre_boundaries(
        trajectory,
        observer_view_provenance,
        public_events,
    )
    return public_events, boundaries


def _private_prompt(
    *,
    boundary: Mapping[str, Any],
    observer_id: int,
    legal_state: Mapping[str, Any],
    hard_knowledge: Mapping[str, Sequence[str]],
    public_events: Sequence[Mapping[str, Any]],
) -> str:
    known_werewolves = canonical_json(hard_knowledge["known_werewolves"])
    known_non_werewolves = canonical_json(
        hard_knowledge["known_non_werewolves"]
    )
    return f"""This is a stateless offline observer-conditioned annotation. It is NOT the historical gameplay agent self-report.
The current PRE speaker player{boundary['speaker_id']} has not yet produced this speech.
Use only the provided legal observer state and frozen PRE public prefix. Do not use any god view, actual role table, other-player private state, or future information.

observer_id: player{observer_id}
boundary_id: {boundary['boundary_id']}
prompt_version: {PRIVATE_PROMPT_VERSION}
legal_pre_speech_observer_state: {canonical_json(legal_state)}
canonical_pre_public_events: {canonical_json(public_events)}

`suspected_werewolves` is a relative suspicion set. Size 0..7 is legal; do not force exactly two. A player being still possible is not automatically the same as being currently suspected.
MUST INCLUDE known_werewolves: {known_werewolves}
MUST EXCLUDE known_non_werewolves: {known_non_werewolves}

Output JSON only as {{"suspected_werewolves":[...]}}. Do not output reasons, reasoning, or chain-of-thought."""


def _public_prompt(
    *,
    boundary: Mapping[str, Any],
    observer_id: int,
    public_events: Sequence[Mapping[str, Any]],
) -> str:
    return f"""This is a stateless offline public-only annotation. Only the frozen PRE public prefix is available.
The current PRE speaker player{boundary['speaker_id']} has not yet produced this speech. Observer ID player{observer_id} is only the reporting perspective identifier.
Do not use any role, private state, team information, Seer information, Witch or night information, private memory, or future information.

observer_id: player{observer_id}
boundary_id: {boundary['boundary_id']}
prompt_version: {PUBLIC_PROMPT_VERSION}
canonical_pre_public_events: {canonical_json(public_events)}

`suspected_werewolves` is a relative suspicion set. Size 0..7 is legal; do not force exactly two. A player being still possible is not automatically the same as being currently suspected.

Output JSON only as {{"suspected_werewolves":[...]}}. Do not output reasons or reasoning."""


def _safe_error(exception: BaseException, *, category: str) -> str:
    return f"{type(exception).__name__}: {category}"


def _run_reporter(
    *,
    backend,
    model_name: str,
    prompt: str,
    request_parameters: Mapping[str, Any],
    hard_knowledge: Mapping[str, Sequence[str]] | None,
) -> tuple[str | None, str, str | None, dict[str, Any] | None]:
    try:
        raw_response = backend.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            temperature=request_parameters["temperature"],
            max_tokens=request_parameters["max_tokens"],
            response_format=deepcopy(request_parameters["response_format"]),
            extra_body=deepcopy(request_parameters["extra_body"]),
        )
    except Exception as exc:
        return None, STATUS_REPORTER_ERROR, _safe_error(
            exc,
            category="reporter request failed",
        ), None
    if not isinstance(raw_response, str):
        return None, STATUS_REPORTER_ERROR, (
            "TypeError: reporter response was not text"
        ), None
    try:
        suspected = PlayingAgentBeliefReporter.parse_response(raw_response)
    except (TypeError, ValueError) as exc:
        return raw_response, STATUS_PARSE_ERROR, _safe_error(
            exc,
            category="invalid reporter response",
        ), None
    if hard_knowledge is not None:
        try:
            PlayingAgentBeliefReporter.validate_semantics(
                observer_id=hard_knowledge["observer_id"],
                suspected_werewolves=suspected,
                known_werewolves=hard_knowledge["known_werewolves"],
                known_non_werewolves=hard_knowledge["known_non_werewolves"],
            )
        except (TypeError, ValueError) as exc:
            return raw_response, STATUS_SEMANTIC_ERROR, _safe_error(
                exc,
                category="response contradicts observer hard knowledge",
            ), None
    return raw_response, STATUS_OK, None, {
        "suspected_werewolves": suspected,
    }


def annotate_pre_speech_suspicion(
    trajectory: Mapping[str, Any],
    observer_view_provenance: Mapping[str, Any],
    *,
    annotation_task: str,
    annotation_run_id: str,
    annotation_code_commit: str,
    backend,
    backend_id: str,
    model_name: str,
) -> list[dict[str, Any]]:
    """Return canonical C1 records without live gameplay dependencies."""

    if annotation_task not in ANNOTATION_TASKS:
        raise ValueError("unsupported annotation_task")
    annotation_run_id = _required_text(
        annotation_run_id,
        field_name="annotation_run_id",
    )
    annotation_code_commit = _required_text(
        annotation_code_commit,
        field_name="annotation_code_commit",
    )
    _validate_sha(
        annotation_code_commit,
        field_name="annotation_code_commit",
        pattern=_GIT_SHA_PATTERN,
    )
    backend_id = _required_text(backend_id, field_name="backend_id")
    model_name = _required_text(model_name, field_name="model_name")
    if backend is None or not hasattr(backend, "chat"):
        raise TypeError("backend must provide chat()")
    supports_json_schema = getattr(backend, "supports_json_schema", False)
    response_format = _response_format(
        supports_json_schema=supports_json_schema
    )
    request_parameters = {
        "temperature": ANNOTATION_TEMPERATURE,
        "max_tokens": ANNOTATION_MAX_TOKENS,
        "response_format": response_format,
        "extra_body": deepcopy(ANNOTATION_EXTRA_BODY),
    }

    public_events, boundaries = validate_offline_annotation_sources(
        trajectory,
        observer_view_provenance,
    )
    is_private = annotation_task == PRIVATE_CONDITIONED_SUSPICION_TASK
    records = []
    for boundary, prefix in boundaries:
        observer_views = boundary.get("observer_views")
        if isinstance(observer_views, (str, bytes)) or not isinstance(
            observer_views,
            Sequence,
        ):
            raise TypeError("boundary observer_views must be a sequence")
        materialized_views = list(observer_views)
        for observer_view in materialized_views:
            if not isinstance(observer_view, Mapping):
                raise TypeError("observer view must be a mapping")
            observer_id = observer_view.get("observer_id")
            if (
                isinstance(observer_id, bool)
                or not isinstance(observer_id, int)
                or not 1 <= observer_id <= 7
            ):
                raise ValueError("observer_id must be an integer in [1, 7]")
        ordered_views = sorted(
            materialized_views,
            key=lambda item: item["observer_id"],
        )
        seen_observers = set()
        for observer_view in ordered_views:
            observer_id = observer_view.get("observer_id")
            if observer_id in seen_observers:
                raise ValueError("boundary observer IDs must be unique")
            seen_observers.add(observer_id)
            public_source = _public_source(boundary, prefix)
            if is_private:
                observation = observer_view.get("observation")
                if not isinstance(observation, Mapping):
                    raise TypeError("private observer view requires an observation")
                observation_digest = observer_view.get("observation_digest")
                if canonical_digest(observation) != observation_digest:
                    raise ValueError("private observation_digest mismatch")
                if observation.get("observer_id") != observer_id:
                    raise ValueError(
                        "private observation observer_id does not match view"
                    )
                if observation.get("current_act_idx") != boundary["speaker_id"]:
                    raise ValueError(
                        "private observation current actor does not match "
                        "PRE speaker"
                    )
                hard = derive_observer_hard_knowledge(
                    observer_id,
                    observation,
                )
                source = {
                    "observation_digest": observation_digest,
                    **public_source,
                    "derived_hard_knowledge": {
                        "known_werewolves": hard["known_werewolves"],
                        "known_non_werewolves": hard[
                            "known_non_werewolves"
                        ],
                    },
                }
                legal_state = legal_observer_state(
                    observer_id,
                    observation,
                )
                prompt = _private_prompt(
                    boundary=boundary,
                    observer_id=observer_id,
                    legal_state=legal_state,
                    hard_knowledge=hard,
                    public_events=prefix,
                )
                validation_knowledge = {
                    "observer_id": observer_id,
                    "known_werewolves": hard["known_werewolves"],
                    "known_non_werewolves": hard["known_non_werewolves"],
                }
                information_scope = PRIVATE_INFORMATION_SCOPE
                prompt_version = PRIVATE_PROMPT_VERSION
            else:
                source = public_source
                prompt = _public_prompt(
                    boundary=boundary,
                    observer_id=observer_id,
                    public_events=prefix,
                )
                validation_knowledge = None
                information_scope = PUBLIC_INFORMATION_SCOPE
                prompt_version = PUBLIC_PROMPT_VERSION

            raw_response, status, error, result = _run_reporter(
                backend=backend,
                model_name=model_name,
                prompt=prompt,
                request_parameters=request_parameters,
                hard_knowledge=validation_knowledge,
            )
            record = {
                "schema_version": OFFLINE_ANNOTATION_SCHEMA_VERSION,
                "annotation_task": annotation_task,
                "annotation_run_id": annotation_run_id,
                "annotation_code_commit": annotation_code_commit,
                "game_id": trajectory["game_id"],
                "source_trajectory_commit": trajectory["source_commit"],
                "trajectory_digest": trajectory["trajectory_digest"],
                "observer_view_artifact_digest": observer_view_provenance[
                    "artifact_digest"
                ],
                "boundary_id": boundary["boundary_id"],
                "boundary_type": boundary["boundary_type"],
                "step_idx": boundary["step_idx"],
                "observer_id": observer_id,
                "information_scope": information_scope,
                "source": source,
                "prompt_version": prompt_version,
                "prompt": prompt,
                "prompt_digest": canonical_digest(prompt),
                "reporter_backend_id": backend_id,
                "reporter_model_id": model_name,
                "request_parameters": deepcopy(request_parameters),
                "raw_response": raw_response,
                "status": status,
                "error": error,
                "result": result,
            }
            record["record_digest"] = canonical_digest(record)
            validate_offline_annotation_record(record)
            records.append(record)
    return records


def write_annotation_jsonl(
    output_path: str | Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Atomically create one canonical, single-task C1 JSONL file."""

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence")
    materialized = [dict(record) for record in records]
    if not materialized:
        raise ValueError("records cannot be empty")
    tasks = {record.get("annotation_task") for record in materialized}
    if len(tasks) != 1 or not tasks <= ANNOTATION_TASKS:
        raise ValueError("one JSONL file must contain one annotation_task")
    ordering = [
        (record.get("step_idx"), record.get("observer_id"))
        for record in materialized
    ]
    if ordering != sorted(ordering):
        raise ValueError("records are not in canonical annotation order")
    for record in materialized:
        validate_offline_annotation_record(record)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"annotation output already exists: {path}")
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
            f"annotation output already exists: {path}"
        ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


__all__ = [
    "ANNOTATION_MAX_TOKENS",
    "ANNOTATION_TEMPERATURE",
    "OFFLINE_ANNOTATION_SCHEMA_VERSION",
    "OFFLINE_SUSPICION_JSON_SCHEMA",
    "PRIVATE_CONDITIONED_SUSPICION_TASK",
    "PRIVATE_INFORMATION_SCOPE",
    "PRIVATE_PROMPT_VERSION",
    "PUBLIC_INFORMATION_SCOPE",
    "PUBLIC_ONLY_SUSPICION_TASK",
    "PUBLIC_PROMPT_VERSION",
    "annotate_pre_speech_suspicion",
    "validate_offline_annotation_record",
    "validate_offline_annotation_sources",
    "write_annotation_jsonl",
]
