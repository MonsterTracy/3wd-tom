"""Freeze public histories and build playing-agent belief samples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from werewolf.models.twd_tom.public_events import (
    PUBLIC_EVENT_SCHEMA_VERSION,
    copy_public_events,
    parse_public_phase,
    public_event_digest,
    public_speech_actions,
    structured_input_digest,
)
from werewolf.models.twd_tom.schema import (
    LABEL_PROMPT_VERSION,
    LABEL_PROVENANCE,
    NUM_PLAYERS,
    normalize_player,
    parse_speech_action,
    validate_player_suspicion,
)
from werewolf.models.twd_tom.belief_labels import close_hard_knowledge
from werewolf.speech.private_belief_perceiver import (
    STATUS_OK,
    STATUS_PARSE_ERROR,
    STATUS_REPORTER_ERROR,
    STATUS_SEMANTIC_ERROR,
)


SAMPLE_SCHEMA_VERSION = "classic7_pre_speech_player_suspicion_v3"
PUBLIC_SPEECH_EVENTS = {"speech", "speech_pk"}
REPORT_TRIGGERS = {"pre_public_speech", "pre_public_speech_pk"}
SAMPLE_FIELDS = frozenset(
    {
        "schema_version",
        "game_id",
        "step_idx",
        "phase",
        "speaker_id",
        "report_trigger",
        "public_event_schema_version",
        "public_events",
        "public_event_digest",
        "structured_input_digest",
        "observer_ids",
        "suspected_werewolves",
        "known_werewolves",
        "known_non_werewolves",
        "belief_status",
        "belief_errors",
        "label_cutoff_step_idx",
        "public_action_count",
        "label_prompt_version",
        "label_provenance",
        "agent_backend_ids",
    }
)


@dataclass(frozen=True)
class SpeakerPreSpeechBelief:
    """Immutable handoff of the speaker row from collection to cognition."""

    observer_id: str
    suspected_werewolves: tuple[str, ...]
    known_werewolves: tuple[str, ...]
    known_non_werewolves: tuple[str, ...]
    source_schema_version: str
    label_prompt_version: str
    label_provenance: str
    step_idx: int
    structured_input_digest: str

    def prompt_payload(self) -> dict[str, list[str]]:
        return {
            "suspected_werewolves": list(self.suspected_werewolves),
        }


def speaker_pre_speech_belief_from_sample(
    sample: Mapping[str, Any],
    *,
    speaker_id: int,
    step_idx: int,
) -> SpeakerPreSpeechBelief:
    """Extract the exact valid speaker report from one just-written sample."""

    if not isinstance(sample, Mapping):
        raise TypeError("collected belief sample must be a mapping")
    if sample.get("schema_version") != SAMPLE_SCHEMA_VERSION:
        raise ValueError("unsupported speaker belief sample schema_version")
    if sample.get("speaker_id") != speaker_id:
        raise ValueError("speaker belief sample has a different speaker_id")
    if sample.get("step_idx") != step_idx:
        raise ValueError("speaker belief sample has a different step_idx")
    if sample.get("label_cutoff_step_idx") != step_idx:
        raise ValueError("speaker belief cutoff must equal the current step_idx")
    if sample.get("label_prompt_version") != LABEL_PROMPT_VERSION:
        raise ValueError("unsupported speaker belief label_prompt_version")
    if sample.get("label_provenance") != LABEL_PROVENANCE:
        raise ValueError("unsupported speaker belief label_provenance")

    observer = normalize_player(speaker_id)
    status = sample.get("belief_status", {}).get(observer)
    if status != STATUS_OK:
        raise ValueError("speaker PRE belief requires status=ok")
    if sample.get("belief_errors", {}).get(observer) is not None:
        raise ValueError("speaker PRE belief status=ok requires null error")

    known_wolves = sample.get("known_werewolves", {}).get(observer)
    known_non_wolves = sample.get("known_non_werewolves", {}).get(observer)
    closed_wolves, closed_non_wolves = close_hard_knowledge(
        known_wolves,
        known_non_wolves,
    )
    if known_wolves != closed_wolves or known_non_wolves != closed_non_wolves:
        raise ValueError("speaker PRE hard knowledge must already be closed")
    suspected = validate_player_suspicion(
        sample.get("suspected_werewolves", {}).get(observer),
        closed_wolves,
        closed_non_wolves,
        observer_id=observer,
    )
    if sample["suspected_werewolves"][observer] != suspected:
        raise ValueError("speaker PRE suspicion must use canonical order")

    digest = sample.get("structured_input_digest")
    if not isinstance(digest, str) or not digest:
        raise ValueError("speaker PRE belief requires structured_input_digest")
    return SpeakerPreSpeechBelief(
        observer_id=observer,
        suspected_werewolves=tuple(suspected),
        known_werewolves=tuple(closed_wolves),
        known_non_werewolves=tuple(closed_non_wolves),
        source_schema_version=SAMPLE_SCHEMA_VERSION,
        label_prompt_version=LABEL_PROMPT_VERSION,
        label_provenance=LABEL_PROVENANCE,
        step_idx=step_idx,
        structured_input_digest=digest,
    )


@dataclass(frozen=True)
class PublicSnapshot:
    """Immutable model-visible public history shared by all reporters."""

    game_id: str
    step_idx: int
    phase: str
    speaker_id: int
    report_trigger: str
    observer_ids: tuple[int, ...]
    public_events: tuple[Mapping[str, Any], ...]
    public_event_digest: str
    structured_input_digest: str
    sp_actions: tuple[tuple[str, str, str | None], ...]
    label_cutoff_step_idx: int
    public_action_count: int
    label_prompt_version: str

    @property
    def public_history_digest(self) -> str:
        """Audit alias for the frozen structured-input digest."""

        return self.structured_input_digest


def _normalize_observer_ids(observer_ids: Sequence[int]) -> tuple[int, ...]:
    if isinstance(observer_ids, (str, bytes)) or not isinstance(
        observer_ids, Sequence
    ):
        raise TypeError("observer_ids must be a sequence")
    normalized = tuple(observer_ids)
    if not normalized:
        raise ValueError("observer_ids cannot be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("observer_ids cannot contain duplicates")
    for player_id in normalized:
        if isinstance(player_id, bool) or not isinstance(player_id, int):
            raise TypeError("observer IDs must be integers")
        if not 1 <= player_id <= NUM_PLAYERS:
            raise ValueError("observer IDs must be in [1, 7]")
    return normalized


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _freeze_json_value(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def freeze_public_snapshot(
    *,
    game_id: str,
    step_idx: int,
    phase: str,
    speaker_id: int,
    report_trigger: str,
    observer_ids: Sequence[int],
    public_events: Sequence[Any],
) -> PublicSnapshot:
    """Freeze the single public cutoff used by model and all reporters."""

    if not isinstance(game_id, str) or not game_id.strip():
        raise ValueError("game_id must be non-empty text")
    if isinstance(step_idx, bool) or not isinstance(step_idx, int) or step_idx < 0:
        raise ValueError("step_idx must be a non-negative integer")
    if not isinstance(phase, str) or not phase:
        raise ValueError("phase must be non-empty text")
    parse_public_phase(phase)
    if isinstance(speaker_id, bool) or not isinstance(speaker_id, int):
        raise TypeError("speaker_id must be an integer")
    if not 1 <= speaker_id <= NUM_PLAYERS:
        raise ValueError("speaker_id must be in [1, 7]")
    if report_trigger not in REPORT_TRIGGERS:
        raise ValueError("unsupported report_trigger")

    normalized_events = copy_public_events(public_events)
    if not normalized_events:
        raise ValueError("public_events cannot be empty at a speech snapshot")
    last_event = normalized_events[-1]
    expected_speaker = normalize_player(speaker_id)
    if (
        last_event["event_type"] != "turn_start"
        or last_event["speaker"] != expected_speaker
    ):
        raise ValueError(
            "pre-speech public_events must end with matching turn_start"
        )
    latest_phase = next(
        (
            event["phase"]
            for event in reversed(normalized_events)
            if event["event_type"] == "phase_change"
        ),
        None,
    )
    if latest_phase != phase:
        raise ValueError(
            "snapshot phase must match latest public phase_change"
        )
    normalized_actions = tuple(
        tuple(parse_speech_action(action).to_list())
        for action in public_speech_actions(normalized_events)
    )
    return PublicSnapshot(
        game_id=game_id,
        step_idx=step_idx,
        phase=phase,
        speaker_id=speaker_id,
        report_trigger=report_trigger,
        observer_ids=_normalize_observer_ids(observer_ids),
        public_events=tuple(
            _freeze_json_value(event)
            for event in normalized_events
        ),
        public_event_digest=public_event_digest(normalized_events),
        structured_input_digest=structured_input_digest(normalized_events),
        sp_actions=normalized_actions,
        label_cutoff_step_idx=step_idx,
        public_action_count=len(normalized_actions),
        label_prompt_version=LABEL_PROMPT_VERSION,
    )


def make_twd_tom_sample(
    *,
    public_snapshot: PublicSnapshot,
    reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one JSON-safe record without serializing private context."""

    if not isinstance(public_snapshot, PublicSnapshot):
        raise TypeError("public_snapshot must be a PublicSnapshot")
    if not isinstance(reports, Mapping):
        raise TypeError("reports must be a mapping")
    expected_subjects = {
        normalize_player(player_id)
        for player_id in public_snapshot.observer_ids
    }
    if set(reports) != expected_subjects:
        raise ValueError("reports must exactly match public snapshot observers")

    suspicions: dict[str, list[str] | None] = {}
    statuses: dict[str, str] = {}
    errors: dict[str, str | None] = {}
    backend_ids: dict[str, str] = {}
    known_werewolves: dict[str, list[str]] = {}
    known_non_werewolves: dict[str, list[str]] = {}
    for subject in sorted(expected_subjects):
        report = reports[subject]
        if not isinstance(report, Mapping):
            raise TypeError("every report must be a mapping")
        status = report.get("status")
        suspicion = report.get("suspected_werewolves")
        error = report.get("error")
        backend_id = report.get("agent_backend_id")
        known_wolves = report.get("known_werewolves")
        known_non_wolves = report.get("known_non_werewolves")
        closed_wolves, closed_non_wolves = close_hard_knowledge(
            known_wolves,
            known_non_wolves,
        )
        if known_wolves != closed_wolves or known_non_wolves != closed_non_wolves:
            raise ValueError(f"{subject} hard knowledge must already be closed")
        if not isinstance(status, str) or not status:
            raise ValueError(f"{subject} has invalid report status")
        if status not in {
            STATUS_OK,
            STATUS_PARSE_ERROR,
            STATUS_REPORTER_ERROR,
            STATUS_SEMANTIC_ERROR,
        }:
            raise ValueError(f"{subject} has unsupported report status")
        if status == STATUS_OK:
            if not isinstance(suspicion, list):
                raise ValueError(f"{subject} ok report requires a list")
            normalized_suspicion = validate_player_suspicion(
                suspicion,
                closed_wolves,
                closed_non_wolves,
                observer_id=subject,
            )
            if error is not None:
                raise ValueError(f"{subject} ok report cannot contain an error")
        else:
            if suspicion is not None:
                raise ValueError(
                    f"{subject} non-ok report must have no suspicion set"
                )
            if not isinstance(error, str) or not error:
                raise ValueError(f"{subject} non-ok report requires an error")
            normalized_suspicion = None
        if error is not None and not isinstance(error, str):
            raise TypeError(f"{subject} error must be text or None")
        if not isinstance(backend_id, str) or not backend_id.strip():
            raise ValueError(f"{subject} requires agent_backend_id")
        suspicions[subject] = normalized_suspicion
        statuses[subject] = status
        errors[subject] = error
        backend_ids[subject] = backend_id
        known_werewolves[subject] = closed_wolves
        known_non_werewolves[subject] = closed_non_wolves

    public_events = copy_public_events(public_snapshot.public_events)
    if public_event_digest(public_events) != public_snapshot.public_event_digest:
        raise RuntimeError("frozen public event digest changed")
    if structured_input_digest(public_events) != (
        public_snapshot.structured_input_digest
    ):
        raise RuntimeError("frozen structured input digest changed")
    if public_speech_actions(public_events) != [
        list(action) for action in public_snapshot.sp_actions
    ]:
        raise RuntimeError("frozen public speech actions changed")

    return {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "game_id": public_snapshot.game_id,
        "step_idx": public_snapshot.step_idx,
        "phase": public_snapshot.phase,
        "speaker_id": public_snapshot.speaker_id,
        "report_trigger": public_snapshot.report_trigger,
        "public_event_schema_version": PUBLIC_EVENT_SCHEMA_VERSION,
        "public_events": public_events,
        "public_event_digest": public_snapshot.public_event_digest,
        "structured_input_digest": public_snapshot.structured_input_digest,
        "observer_ids": list(public_snapshot.observer_ids),
        "suspected_werewolves": suspicions,
        "known_werewolves": known_werewolves,
        "known_non_werewolves": known_non_werewolves,
        "belief_status": statuses,
        "belief_errors": errors,
        "label_cutoff_step_idx": public_snapshot.label_cutoff_step_idx,
        "public_action_count": public_snapshot.public_action_count,
        "label_prompt_version": public_snapshot.label_prompt_version,
        "label_provenance": LABEL_PROVENANCE,
        "agent_backend_ids": backend_ids,
    }


__all__ = [
    "SAMPLE_SCHEMA_VERSION",
    "PUBLIC_SPEECH_EVENTS",
    "REPORT_TRIGGERS",
    "SAMPLE_FIELDS",
    "SpeakerPreSpeechBelief",
    "speaker_pre_speech_belief_from_sample",
    "PublicSnapshot",
    "freeze_public_snapshot",
    "make_twd_tom_sample",
]
