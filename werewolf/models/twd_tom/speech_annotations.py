"""Versioned public-speech annotations detached from canonical events."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from werewolf.models.twd_tom.schema import (
    PLAYER_NAMES,
    parse_speech_action,
)


SPEECH_ANNOTATION_SCHEMA_VERSION = "classic7_speech_annotation_v3"
SPEECH_ACTION_ONTOLOGY_VERSION = "classic7_speech_action_v1"
SPEECH_PARSER_PROMPT_VERSION = "classic7_speech_parser_v3"

STATUS_OK = "ok"
STATUS_NO_ACTION = "no_action"
STATUS_ERROR = "error"
ANNOTATION_STATUSES = frozenset({STATUS_OK, STATUS_NO_ACTION, STATUS_ERROR})
ANNOTATION_SOURCES = frozenset({"llm_parser"})

ANNOTATION_FIELDS = frozenset(
    {
        "schema_version",
        "event_idx",
        "speaker",
        "raw_text_digest",
        "ontology_version",
        "parser_prompt_version",
        "parser_model_id",
        "parser_call_id",
        "annotation_source",
        "status",
        "actions",
        "generation_attempts",
        "raw_response",
        "error_type",
        "error_message",
    }
)
GENERATION_ATTEMPT_FIELDS = frozenset(
    {
        "generation_attempt",
        "status",
        "raw_response",
        "error_type",
        "error_message",
    }
)
GENERATION_ATTEMPT_STATUSES = frozenset({"ok", "parser_error"})


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def raw_text_digest(raw_text: Any) -> str:
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be text")
    return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()


def make_speech_annotation(
    *,
    event_idx: int,
    speaker: str,
    raw_text: str,
    parser_model_id: str,
    parser_call_id: str,
    annotation_source: str,
    status: str,
    actions: Sequence[Any],
    raw_response: str | None,
    error_type: str | None,
    error_message: str | None,
    generation_attempts: Sequence[Any] = (),
) -> dict[str, Any]:
    """Build and validate one annotation for an immutable speech event."""

    candidate = {
        "schema_version": SPEECH_ANNOTATION_SCHEMA_VERSION,
        "event_idx": event_idx,
        "speaker": speaker,
        "raw_text_digest": raw_text_digest(raw_text),
        "ontology_version": SPEECH_ACTION_ONTOLOGY_VERSION,
        "parser_prompt_version": SPEECH_PARSER_PROMPT_VERSION,
        "parser_model_id": parser_model_id,
        "parser_call_id": parser_call_id,
        "annotation_source": annotation_source,
        "status": status,
        "actions": list(actions),
        "generation_attempts": list(generation_attempts),
        "raw_response": raw_response,
        "error_type": error_type,
        "error_message": error_message,
    }
    return normalize_speech_annotation(candidate)


def _normalize_generation_attempts(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("speech parser generation_attempts must be a sequence")

    normalized = []
    for expected_index, raw_attempt in enumerate(value, start=1):
        if not isinstance(raw_attempt, Mapping):
            raise TypeError("speech parser generation attempt must be a mapping")
        if set(raw_attempt) != GENERATION_ATTEMPT_FIELDS:
            raise ValueError("speech parser generation attempt field set mismatch")
        if raw_attempt.get("generation_attempt") != expected_index:
            raise ValueError(
                "speech parser generation attempts must be contiguous from one"
            )
        status = raw_attempt.get("status")
        if status not in GENERATION_ATTEMPT_STATUSES:
            raise ValueError("unsupported speech parser generation attempt status")
        raw_response = raw_attempt.get("raw_response")
        if raw_response is not None and not isinstance(raw_response, str):
            raise TypeError(
                "speech parser generation attempt raw_response must be text or null"
            )
        error_type = raw_attempt.get("error_type")
        error_message = raw_attempt.get("error_message")
        if status == "parser_error":
            if not isinstance(error_type, str) or not error_type:
                raise ValueError("failed speech parser attempt requires error_type")
            if not isinstance(error_message, str) or not error_message:
                raise ValueError("failed speech parser attempt requires error_message")
        elif error_type is not None or error_message is not None:
            raise ValueError("successful speech parser attempt requires null errors")
        normalized.append(deepcopy(dict(raw_attempt)))

    successful_indices = [
        index
        for index, attempt in enumerate(normalized, start=1)
        if attempt["status"] == "ok"
    ]
    if successful_indices and successful_indices != [len(normalized)]:
        raise ValueError("speech parser generation must stop after first success")
    return normalized


def normalize_speech_annotation(value: Any) -> dict[str, Any]:
    """Validate one annotation without repairing parser output."""

    if not isinstance(value, Mapping):
        raise TypeError("speech annotation must be a mapping")
    if set(value) != ANNOTATION_FIELDS:
        missing = sorted(ANNOTATION_FIELDS - set(value))
        extra = sorted(set(value) - ANNOTATION_FIELDS)
        raise ValueError(
            "speech annotation field set mismatch; "
            f"missing={missing}, extra={extra}"
        )
    if value.get("schema_version") != SPEECH_ANNOTATION_SCHEMA_VERSION:
        raise ValueError("unsupported speech annotation schema_version")
    event_idx = value.get("event_idx")
    if isinstance(event_idx, bool) or not isinstance(event_idx, int) or event_idx < 0:
        raise ValueError("speech annotation event_idx must be non-negative")
    speaker = value.get("speaker")
    if speaker not in PLAYER_NAMES:
        raise ValueError("speech annotation speaker must be canonical")
    digest = value.get("raw_text_digest")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("speech annotation raw_text_digest must be sha256")
    if value.get("ontology_version") != SPEECH_ACTION_ONTOLOGY_VERSION:
        raise ValueError("unsupported speech action ontology_version")
    if value.get("parser_prompt_version") != SPEECH_PARSER_PROMPT_VERSION:
        raise ValueError("unsupported speech parser_prompt_version")
    for field_name in ("parser_model_id", "parser_call_id"):
        field = value.get(field_name)
        if not isinstance(field, str) or not field.strip():
            raise ValueError(f"speech annotation {field_name} is required")
    source = value.get("annotation_source")
    if source not in ANNOTATION_SOURCES:
        raise ValueError("unsupported speech annotation source")
    status = value.get("status")
    if status not in ANNOTATION_STATUSES:
        raise ValueError("unsupported speech annotation status")
    raw_actions = value.get("actions")
    if isinstance(raw_actions, (str, bytes)) or not isinstance(raw_actions, Sequence):
        raise TypeError("speech annotation actions must be a sequence")
    actions = []
    for raw_action in raw_actions:
        action = parse_speech_action(raw_action)
        if action.subject != speaker:
            raise ValueError("speech action subject must equal annotation speaker")
        actions.append(action.to_list())
    if status == STATUS_OK and not actions:
        raise ValueError("status=ok requires at least one speech action")
    if status in {STATUS_NO_ACTION, STATUS_ERROR} and actions:
        raise ValueError(f"status={status} requires an empty action list")

    generation_attempts = _normalize_generation_attempts(
        value.get("generation_attempts")
    )

    raw_response = value.get("raw_response")
    if raw_response is not None and not isinstance(raw_response, str):
        raise TypeError("speech annotation raw_response must be text or null")
    error_type = value.get("error_type")
    error_message = value.get("error_message")
    if status == STATUS_ERROR:
        if not isinstance(error_type, str) or not error_type:
            raise ValueError("status=error requires error_type")
        if not isinstance(error_message, str) or not error_message:
            raise ValueError("status=error requires error_message")
    elif error_type is not None or error_message is not None:
        raise ValueError("non-error annotation requires null error fields")

    if generation_attempts:
        final_attempt = generation_attempts[-1]
        expected_attempt_status = (
            "parser_error" if status == STATUS_ERROR else "ok"
        )
        if final_attempt["status"] != expected_attempt_status:
            raise ValueError(
                "speech annotation status differs from final generation attempt"
            )
        for field_name in ("raw_response", "error_type", "error_message"):
            if final_attempt[field_name] != value.get(field_name):
                raise ValueError(
                    "speech annotation final result differs from final generation "
                    f"attempt field {field_name}"
                )

    normalized = dict(value)
    normalized["actions"] = actions
    normalized["generation_attempts"] = generation_attempts
    return normalized


def normalize_speech_annotations(
    annotations: Any,
    *,
    public_events: Sequence[Mapping[str, Any]] | None = None,
    require_complete: bool = False,
) -> list[dict[str, Any]]:
    """Validate ordered annotations and optionally bind them to raw events."""

    if isinstance(annotations, (str, bytes)) or not isinstance(annotations, Sequence):
        raise TypeError("speech annotations must be a sequence")
    normalized = [normalize_speech_annotation(item) for item in annotations]
    event_indices = [item["event_idx"] for item in normalized]
    if event_indices != sorted(event_indices) or len(set(event_indices)) != len(event_indices):
        raise ValueError("speech annotation event_idx values must be unique ascending")

    if public_events is not None:
        speeches = {
            event["event_idx"]: event
            for event in public_events
            if event.get("event_type") == "public_speech"
        }
        for item in normalized:
            speech = speeches.get(item["event_idx"])
            if speech is None:
                raise ValueError("speech annotation has no matching public_speech")
            if speech.get("speaker") != item["speaker"]:
                raise ValueError("speech annotation speaker differs from event")
            if raw_text_digest(speech.get("raw_text")) != item["raw_text_digest"]:
                raise ValueError("speech annotation raw_text_digest differs from event")
        if require_complete and set(event_indices) != set(speeches):
            missing = sorted(set(speeches) - set(event_indices))
            extra = sorted(set(event_indices) - set(speeches))
            raise ValueError(
                "speech annotation coverage mismatch; "
                f"missing={missing}, extra={extra}"
            )
    elif require_complete:
        raise ValueError("require_complete needs public_events")
    return normalized


def speech_annotation_digest(annotations: Any) -> str:
    normalized = normalize_speech_annotations(annotations)
    return hashlib.sha256(_canonical_json(normalized).encode("utf-8")).hexdigest()


def speech_annotation_actions(annotations: Any) -> list[list[str | None]]:
    return [
        list(action)
        for annotation in normalize_speech_annotations(annotations)
        if annotation["status"] == STATUS_OK
        for action in annotation["actions"]
    ]


__all__ = [
    "ANNOTATION_FIELDS",
    "ANNOTATION_SOURCES",
    "ANNOTATION_STATUSES",
    "GENERATION_ATTEMPT_FIELDS",
    "GENERATION_ATTEMPT_STATUSES",
    "SPEECH_ACTION_ONTOLOGY_VERSION",
    "SPEECH_ANNOTATION_SCHEMA_VERSION",
    "SPEECH_PARSER_PROMPT_VERSION",
    "STATUS_ERROR",
    "STATUS_NO_ACTION",
    "STATUS_OK",
    "make_speech_annotation",
    "normalize_speech_annotation",
    "normalize_speech_annotations",
    "raw_text_digest",
    "speech_annotation_actions",
    "speech_annotation_digest",
]
