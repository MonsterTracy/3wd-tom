"""Freeze public histories and build playing-agent belief samples."""

from __future__ import annotations

import hashlib
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
    PLAYER_NAMES,
    normalize_player,
    parse_speech_action,
    validate_player_suspicion,
)
from werewolf.models.twd_tom.belief_labels import close_hard_knowledge
from werewolf.speech.pair_belief_self_reporter import (
    PAIR_BELIEF_PARSER_VERSION,
    PAIR_BELIEF_PROMPT_VERSION,
    PAIR_BELIEF_REPORT_PROVENANCE,
    STATUS_OK,
    STATUS_PARSE_ERROR,
    STATUS_REPORTER_ERROR,
    STATUS_SEMANTIC_ERROR,
    canonical_json_sha256,
    validate_pair_probabilities,
)


SAMPLE_SCHEMA_VERSION = "classic7_pre_speech_player_suspicion_v2"
ACTOR_PAIR_BELIEF_SCHEMA_VERSION = (
    "classic7_fixed_two_wolves_actor_perspective_"
    "direct_pair_belief_self_reports_v1"
)
ACTOR_PAIR_BELIEF_ANNOTATION_VERSION = (
    "classic7_direct_pair_belief_self_reports_v1"
)
ACTOR_PAIR_BELIEF_GENERATOR_NAME = "twd_tom_actor_pair_belief_collector"
ACTOR_PAIR_BELIEF_GENERATOR_VERSION = "1"
PUBLIC_SPEECH_EVENTS = {"speech", "speech_pk"}
REPORT_TRIGGERS = {"pre_public_speech", "pre_public_speech_pk"}
ACTOR_PAIR_BELIEF_SAMPLE_FIELDS = frozenset(
    {
        "schema_version",
        "annotation_version",
        "game_id",
        "event_idx",
        "day",
        "phase",
        "current_speaker",
        "report_trigger",
        "public_event_schema_version",
        "public_events",
        "public_event_digest",
        "public_history_cutoff_event_idx",
        "current_action_used",
        "future_information_used",
        "reasoning_player_id",
        "reasoning_player_private_knowledge",
        "reasoning_input_payload",
        "reasoning_input_payload_sha256",
        "player_reports",
        "provenance",
    }
)
# Historical v2 readers retain the exact diagnostic-data contract.
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
    sp_actions: tuple[tuple[str, str, str], ...]
    label_cutoff_step_idx: int
    public_action_count: int
    label_prompt_version: str
    event_idx: int
    day: int

    @property
    def public_history_digest(self) -> str:
        """Prompt-v2 audit alias; not part of the serialized v2 schema."""

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
    day, _ = parse_public_phase(phase)
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
        label_prompt_version=PAIR_BELIEF_PROMPT_VERSION,
        event_idx=last_event["event_idx"],
        day=day,
    )


def make_twd_tom_sample(
    *,
    public_snapshot: PublicSnapshot,
    reports: Mapping[str, Mapping[str, Any]],
    collection_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one actor-perspective raw record with complete provenance."""

    if not isinstance(public_snapshot, PublicSnapshot):
        raise TypeError("public_snapshot must be a PublicSnapshot")
    if not isinstance(reports, Mapping):
        raise TypeError("reports must be a mapping")
    if not isinstance(collection_provenance, Mapping):
        raise TypeError("collection_provenance must be a mapping")
    expected_subjects = {
        normalize_player(player_id)
        for player_id in public_snapshot.observer_ids
    }
    if set(reports) != expected_subjects:
        raise ValueError("reports must exactly match public snapshot observers")
    reasoning_player = normalize_player(public_snapshot.speaker_id)
    if reasoning_player not in expected_subjects:
        raise ValueError("current_speaker must be eligible for reporting")

    required_provenance = {
        "generator_name",
        "generator_version",
        "git_commit_sha",
        "git_worktree_clean",
        "collection_timestamp_utc",
        "game_seed",
        "source_config_path",
        "source_config_sha256",
        "resolved_runtime_config_sha256",
        "resolved_backend_config_sha256",
    }
    if not required_provenance.issubset(collection_provenance):
        missing = sorted(required_provenance - set(collection_provenance))
        raise ValueError(f"collection provenance missing fields: {missing}")
    if collection_provenance["git_worktree_clean"] is not True:
        raise ValueError(
            "formal actor-perspective sample requires git_worktree_clean=true"
        )
    source_config_path = collection_provenance["source_config_path"]
    if not isinstance(source_config_path, str) or not source_config_path:
        raise ValueError("source_config_path must be non-empty text")
    if source_config_path.startswith("/"):
        raise ValueError("source_config_path must be repository-relative")

    player_reports: list[dict[str, Any]] = []
    for player in PLAYER_NAMES:
        if player not in expected_subjects:
            player_reports.append(
                {
                    "player_id": player,
                    "alive": False,
                    "report_status": "not_collected_dead",
                    "report_error": "player was not alive at the snapshot",
                    "pair_probabilities": None,
                    "known_werewolves": None,
                    "known_non_werewolves": None,
                    "reporter_input_payload": None,
                    "reporter_input_payload_sha256": None,
                    "raw_reporter_output": None,
                    "parsed_output": None,
                    "hard_knowledge_validation": {"status": "not_checked"},
                    "report_provenance": PAIR_BELIEF_REPORT_PROVENANCE,
                    "backend_alias": None,
                    "resolved_model_name": None,
                    "prompt_version": PAIR_BELIEF_PROMPT_VERSION,
                    "prompt_sha256": None,
                    "parser_version": PAIR_BELIEF_PARSER_VERSION,
                    "sampling_parameters": None,
                    "reporter_seed": None,
                }
            )
            continue
        report = reports[player]
        if not isinstance(report, Mapping):
            raise TypeError("every report must be a mapping")
        status = report.get("report_status")
        probabilities = report.get("pair_probabilities")
        error = report.get("report_error")
        known_wolves = report.get("known_werewolves")
        known_non_wolves = report.get("known_non_werewolves")
        if not isinstance(status, str) or not status:
            raise ValueError(f"{player} has invalid report status")
        if status not in {
            STATUS_OK,
            STATUS_PARSE_ERROR,
            STATUS_REPORTER_ERROR,
            STATUS_SEMANTIC_ERROR,
        }:
            raise ValueError(f"{player} has unsupported report status")
        if status == STATUS_OK:
            normalized_probabilities, _ = validate_pair_probabilities(
                probabilities,
                known_werewolves=known_wolves,
                known_non_werewolves=known_non_wolves,
            )
            if error is not None:
                raise ValueError(f"{player} ok report cannot contain an error")
            if normalized_probabilities != probabilities:
                raise ValueError(f"{player} probabilities must not be rewritten")
        else:
            if probabilities is not None:
                raise ValueError(f"{player} non-ok report must remain missing")
            if not isinstance(error, str) or not error:
                raise ValueError(f"{player} non-ok report requires an error")
        if error is not None and not isinstance(error, str):
            raise TypeError(f"{player} error must be text or None")
        if report.get("player_id") != player:
            raise ValueError(f"{player} report has the wrong player_id")
        payload = report.get("reporter_input_payload")
        payload_digest = report.get("reporter_input_payload_sha256")
        if payload is not None and canonical_json_sha256(payload) != payload_digest:
            raise ValueError(f"{player} reporter payload digest mismatch")
        if status == STATUS_OK and payload is None:
            raise ValueError(f"{player} ok report requires the sent payload")
        prompt_digest = report.get("prompt_sha256")
        if payload is not None:
            messages = payload.get("messages") if isinstance(payload, Mapping) else None
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{player} reporter payload requires messages")
            prompt = messages[-1].get("content")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"{player} reporter payload requires a final prompt")
            if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != prompt_digest:
                raise ValueError(f"{player} prompt digest mismatch")
        for route_field in ("backend_alias", "resolved_model_name"):
            route_value = report.get(route_field)
            if not isinstance(route_value, str) or not route_value.strip():
                raise ValueError(f"{player} requires {route_field}")
        serialized_report = dict(report)
        serialized_report["alive"] = True
        player_reports.append(serialized_report)

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

    reasoning_report = next(
        report for report in player_reports if report["player_id"] == reasoning_player
    )
    reasoning_private_knowledge = {
        "known_werewolves": list(reasoning_report["known_werewolves"]),
        "known_non_werewolves": list(reasoning_report["known_non_werewolves"]),
    }
    reasoning_input_payload = {
        "public_history": public_events,
        "reasoning_player_id": reasoning_player,
        "legal_private_knowledge": reasoning_private_knowledge,
    }
    provenance = dict(collection_provenance)
    provenance.update(
        {
            "schema_version": ACTOR_PAIR_BELIEF_SCHEMA_VERSION,
            "annotation_version": ACTOR_PAIR_BELIEF_ANNOTATION_VERSION,
            "prompt_version": PAIR_BELIEF_PROMPT_VERSION,
            "parser_version": PAIR_BELIEF_PARSER_VERSION,
            "public_event_digest": public_snapshot.public_event_digest,
            "reporter_payload_sha256_by_player": {
                report["player_id"]: report["reporter_input_payload_sha256"]
                for report in player_reports
            },
            "prompt_sha256_by_player": {
                report["player_id"]: report["prompt_sha256"]
                for report in player_reports
            },
            "reporter_routes": {
                report["player_id"]: {
                    "backend_alias": report["backend_alias"],
                    "resolved_model_name": report["resolved_model_name"],
                }
                for report in player_reports
            },
        }
    )

    return {
        "schema_version": ACTOR_PAIR_BELIEF_SCHEMA_VERSION,
        "annotation_version": ACTOR_PAIR_BELIEF_ANNOTATION_VERSION,
        "game_id": public_snapshot.game_id,
        "event_idx": public_snapshot.event_idx,
        "day": public_snapshot.day,
        "phase": public_snapshot.phase,
        "current_speaker": reasoning_player,
        "report_trigger": public_snapshot.report_trigger,
        "public_event_schema_version": PUBLIC_EVENT_SCHEMA_VERSION,
        "public_events": public_events,
        "public_event_digest": public_snapshot.public_event_digest,
        "public_history_cutoff_event_idx": public_snapshot.event_idx,
        "current_action_used": False,
        "future_information_used": False,
        "reasoning_player_id": reasoning_player,
        "reasoning_player_private_knowledge": reasoning_private_knowledge,
        "reasoning_input_payload": reasoning_input_payload,
        "reasoning_input_payload_sha256": canonical_json_sha256(
            reasoning_input_payload
        ),
        "player_reports": player_reports,
        "provenance": provenance,
    }


def normalize_historical_suspicion_sample(
    *,
    public_snapshot: PublicSnapshot,
    reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the frozen v2 diagnostic schema without making it formal."""

    expected_subjects = {
        normalize_player(player_id) for player_id in public_snapshot.observer_ids
    }
    if set(reports) != expected_subjects:
        raise ValueError("reports must exactly match public snapshot observers")
    suspicions = {}
    statuses = {}
    errors = {}
    backend_ids = {}
    known_werewolves = {}
    known_non_werewolves = {}
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
            normalized = validate_player_suspicion(
                suspicion,
                closed_wolves,
                closed_non_wolves,
            )
            if error is not None:
                raise ValueError(f"{subject} ok report cannot contain an error")
        else:
            if suspicion is not None:
                raise ValueError(f"{subject} non-ok report must have no suspicion set")
            if not isinstance(error, str) or not error:
                raise ValueError(f"{subject} non-ok report requires an error")
            normalized = None
        if error is not None and not isinstance(error, str):
            raise TypeError(f"{subject} error must be text or None")
        if not isinstance(backend_id, str) or not backend_id.strip():
            raise ValueError(f"{subject} requires agent_backend_id")
        suspicions[subject] = normalized
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
        "label_prompt_version": LABEL_PROMPT_VERSION,
        "label_provenance": LABEL_PROVENANCE,
        "agent_backend_ids": backend_ids,
    }


__all__ = [
    "SAMPLE_SCHEMA_VERSION",
    "ACTOR_PAIR_BELIEF_SCHEMA_VERSION",
    "ACTOR_PAIR_BELIEF_ANNOTATION_VERSION",
    "ACTOR_PAIR_BELIEF_GENERATOR_NAME",
    "ACTOR_PAIR_BELIEF_GENERATOR_VERSION",
    "PUBLIC_SPEECH_EVENTS",
    "REPORT_TRIGGERS",
    "SAMPLE_FIELDS",
    "ACTOR_PAIR_BELIEF_SAMPLE_FIELDS",
    "PublicSnapshot",
    "freeze_public_snapshot",
    "make_twd_tom_sample",
    "normalize_historical_suspicion_sample",
]
