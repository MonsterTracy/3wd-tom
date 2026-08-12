"""Raw symbolic collection for Public Belief Matrix V1."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from werewolf.models.public_belief_matrix.public_prefix import (
    build_public_belief_matrix_visible_prefix,
    render_public_belief_matrix_visible_prefix,
)
from werewolf.models.public_belief_matrix.targets import suspicion_reports_to_matrix_target
from werewolf.models.twd_tom.action_features import PublicEventFeatureBuilder
from werewolf.models.twd_tom.public_events import normalize_public_events
from werewolf.models.twd_tom.schema import CANONICAL_PLAYER_ORDERING, normalize_player

PUBLIC_BELIEF_MATRIX_COLLECTION_MODE = "public_belief_matrix"
PUBLIC_BELIEF_MATRIX_SAMPLE_SCHEMA_VERSION = "classic7_public_belief_matrix_symbolic_v1"
PUBLIC_BELIEF_MATRIX_VISIBLE_PREFIX_SCHEMA_VERSION = "classic7_structured_public_prefix_v1"
PUBLIC_BELIEF_MATRIX_SUPERVISION_BOUNDARY = "post_completed_public_speech_pre_next_action_v1"
PUBLIC_BELIEF_MATRIX_MAX_SEQ_LEN = 256
PUBLIC_BELIEF_MATRIX_REPORT_STATUSES = ("ok", "parse_error", "reporter_error")
PUBLIC_BELIEF_MATRIX_PROVENANCE = {
    "task": "public_belief_matrix",
    "supervision_boundary": PUBLIC_BELIEF_MATRIX_SUPERVISION_BOUNDARY,
    "player_count": 7,
    "model_input_scope": "structured_public_prefix_raw_text_free",
    "reporter_information_scope": "same_structured_prefix_plus_observer_id",
    "private_fields_usage": "none",
    "max_seq_len": PUBLIC_BELIEF_MATRIX_MAX_SEQ_LEN,
    "target_source": "reporter_symbolic_suspected_werewolves",
    "matrix_semantics": "normalized_individual_suspicion_distribution",
    "diagonal_policy": "allowed",
    "empty_set_policy": "uniform_at_materialization",
    "report_status_vocabulary": list(PUBLIC_BELIEF_MATRIX_REPORT_STATUSES),
    "reporter_temperature": 0.0,
}


@dataclass(frozen=True)
class PublicBeliefMatrixCutoff:
    game_id: str
    step_idx: int
    phase: str
    speaker_id: int
    snapshot_id: str
    public_action_count: int
    public_history_digest: str


def _prefix_payload(prefix) -> dict[str, list[Any]]:
    return {field: value.detach().cpu().tolist() for field, value in prefix.items()}


def _public_alive_metadata(public_events) -> tuple[list[str], list[bool]]:
    dead = set()
    for event in public_events:
        if event["event_type"] == "exile_result":
            dead.update(event["exiled_players"])
        elif event["event_type"] == "death_announcement":
            dead.update(event["dead_players"])
    alive_mask = [player not in dead for player in CANONICAL_PLAYER_ORDERING]
    return (
        [
            player
            for player, alive in zip(CANONICAL_PLAYER_ORDERING, alive_mask)
            if alive
        ],
        alive_mask,
    )


def _validated_serialized_prefix(sample) -> dict[str, torch.Tensor]:
    if (
        sample.get("visible_prefix_schema_version")
        != PUBLIC_BELIEF_MATRIX_VISIBLE_PREFIX_SCHEMA_VERSION
    ):
        raise ValueError("unsupported PBM visible-prefix schema")
    payload = sample.get("structured_prefix")
    if not isinstance(payload, Mapping):
        raise TypeError("structured_prefix must be a mapping")
    fields = PublicEventFeatureBuilder.FEATURE_FIELDS
    if set(payload) != set(fields):
        raise ValueError("structured_prefix fields do not match the canonical contract")
    lengths = set()
    prefix = {}
    for field in fields:
        values = payload[field]
        if not isinstance(values, list):
            raise TypeError(f"structured_prefix.{field} must be a JSON list")
        if field == "day_values":
            valid_values = all(
                not isinstance(value, bool) and isinstance(value, (int, float))
                for value in values
            )
        else:
            valid_values = all(
                not isinstance(value, bool) and isinstance(value, int)
                for value in values
            )
        if not valid_values:
            raise TypeError(f"structured_prefix.{field} has invalid values")
        lengths.add(len(values))
        dtype = torch.float32 if field == "day_values" else torch.long
        try:
            prefix[field] = torch.tensor(values, dtype=dtype)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"invalid structured_prefix.{field}") from exc
    if len(lengths) != 1:
        raise ValueError("structured_prefix fields must have equal lengths")
    rendered = render_public_belief_matrix_visible_prefix(prefix)
    digest = sample.get("structured_prefix_digest")
    if not isinstance(digest, str) or hashlib.sha256(
        rendered.encode("utf-8")
    ).hexdigest() != digest:
        raise ValueError("structured_prefix_digest mismatch")
    return prefix


def validate_public_belief_matrix_sample(sample) -> None:
    if not isinstance(sample, dict):
        raise TypeError("PBM sample must be a mapping")
    if sample.get("schema_version") != PUBLIC_BELIEF_MATRIX_SAMPLE_SCHEMA_VERSION:
        raise ValueError("unsupported PBM sample schema")
    for field, expected in PUBLIC_BELIEF_MATRIX_PROVENANCE.items():
        if sample.get(field) != expected:
            raise ValueError(f"PBM provenance mismatch: {field}")
    _validated_serialized_prefix(sample)
    reports = sample.get("observer_reports")
    if not isinstance(reports, list) or len(reports) != 7:
        raise ValueError("PBM sample requires exactly seven reports")
    if [report.get("observer") for report in reports] != list(CANONICAL_PLAYER_ORDERING):
        raise ValueError("PBM reports must use canonical observer order")
    suspicion_reports_to_matrix_target(reports)
    for report in reports:
        status = report.get("status")
        suspicion = report.get("suspected_werewolves")
        if status not in PUBLIC_BELIEF_MATRIX_REPORT_STATUSES:
            raise ValueError("unsupported PBM report status")
        if (status == "ok") != (isinstance(suspicion, list)):
            raise ValueError("PBM report status and suspicion are inconsistent")
        if status != "ok" and suspicion is not None:
            raise ValueError("non-ok PBM suspicion must be None")
    backend_id = sample.get("reporter_backend_id")
    model_id = sample.get("reporter_model_id")
    if not isinstance(backend_id, str) or not backend_id.strip():
        raise ValueError("reporter_backend_id must be non-empty text")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("reporter_model_id must be non-empty text")
    if any(report.get("reporter_backend_id") != backend_id for report in reports):
        raise ValueError("observer report backend provenance mismatch")
    alive = sample.get("publicly_alive_players")
    mask = sample.get("observer_alive_mask")
    if not isinstance(alive, list) or not isinstance(mask, list) or len(mask) != 7:
        raise ValueError("invalid PBM alive audit metadata")
    expected_alive = [player for player, value in zip(CANONICAL_PLAYER_ORDERING, mask) if value]
    if alive != expected_alive or any(type(value) is not bool for value in mask):
        raise ValueError("PBM alive audit metadata is inconsistent")
    forbidden = {"roles", "actual_roles", "true_roles", "raw_text", "matrix_target", "pair_target"}
    if forbidden.intersection(sample):
        raise ValueError("PBM raw sample contains forbidden fields")


class PublicBeliefMatrixSampleCollector:
    """Collect seven symbolic reports after each completed public speech."""

    collection_timing = PUBLIC_BELIEF_MATRIX_SUPERVISION_BOUNDARY

    def __init__(self, *, output_path, game_id, reporter, reporter_dispatch) -> None:
        self.game_id = game_id
        self.reporter = reporter
        self.dispatch = dict(reporter_dispatch)
        if set(self.dispatch) != {"backend", "backend_id", "model_name"}:
            raise ValueError("reporter_dispatch must be one shared explicit dispatch")
        self._output = Path(output_path).open("a", encoding="utf-8")
        self.snapshot_count = 0

    def record(self, env, *, step_idx, trigger, phase, speaker_id, observer_ids=None):
        del observer_ids
        if trigger not in {"speech", "speech_pk"}:
            raise ValueError("PBM collection requires a completed public speech")
        raw_events = getattr(env, "public_events", None)
        if isinstance(raw_events, (str, bytes)) or not isinstance(raw_events, Sequence):
            raise TypeError("environment must provide a public-event sequence")
        expected_speaker = normalize_player(speaker_id)
        speech_indices = [
            index
            for index, event in enumerate(raw_events)
            if isinstance(event, Mapping)
            and event.get("event_type") == "public_speech"
        ]
        if not speech_indices:
            raise RuntimeError("completed speech has no public_speech event")
        speech_index = speech_indices[-1]
        visible_events = normalize_public_events(raw_events[: speech_index + 1])
        speech = visible_events[-1]
        if speech["speaker"] != expected_speaker:
            raise ValueError("completed speech speaker mismatch")
        prefix = build_public_belief_matrix_visible_prefix(
            visible_events, max_seq_len=PUBLIC_BELIEF_MATRIX_MAX_SEQ_LEN
        )
        rendered = render_public_belief_matrix_visible_prefix(prefix)
        digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        self.snapshot_count += 1
        snapshot_id = f"{self.game_id}:pbm:{self.snapshot_count:06d}"
        cutoff = PublicBeliefMatrixCutoff(
            game_id=self.game_id,
            step_idx=step_idx,
            phase=phase,
            speaker_id=speaker_id,
            snapshot_id=snapshot_id,
            public_action_count=sum(
                len(event["sp_actions"])
                for event in visible_events
                if event["event_type"] == "public_speech"
            ),
            public_history_digest=digest,
        )
        reports = [
            self.reporter.report(
                visible_prefix=prefix,
                observer_id=observer,
                cutoff=cutoff,
                **self.dispatch,
            )
            for observer in CANONICAL_PLAYER_ORDERING
        ]
        publicly_alive_players, alive_mask = _public_alive_metadata(visible_events)
        sample = {
            "schema_version": PUBLIC_BELIEF_MATRIX_SAMPLE_SCHEMA_VERSION,
            "visible_prefix_schema_version": PUBLIC_BELIEF_MATRIX_VISIBLE_PREFIX_SCHEMA_VERSION,
            **PUBLIC_BELIEF_MATRIX_PROVENANCE,
            "game_id": self.game_id,
            "snapshot_id": snapshot_id,
            "step_idx": step_idx,
            "phase": phase,
            "speaker": normalize_player(speaker_id),
            "public_speech_event_idx": speech["event_idx"],
            "structured_prefix": _prefix_payload(prefix),
            "structured_prefix_digest": digest,
            "publicly_alive_players": publicly_alive_players,
            "observer_alive_mask": alive_mask,
            "observer_reports": reports,
            "reporter_backend_id": self.dispatch["backend_id"],
            "reporter_model_id": self.dispatch["model_name"],
        }
        self.write(sample)
        return sample

    def write(self, sample) -> None:
        validate_public_belief_matrix_sample(sample)
        self._output.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
        self._output.flush()

    def close(self) -> None:
        self._output.close()


__all__ = [name for name in globals() if name.startswith("PUBLIC_BELIEF_MATRIX_")] + [
    "PublicBeliefMatrixCutoff",
    "PublicBeliefMatrixSampleCollector",
    "validate_public_belief_matrix_sample",
]
