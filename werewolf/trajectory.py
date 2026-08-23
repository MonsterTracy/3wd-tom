"""Canonical read-only capture for one Classic-7 gameplay trajectory."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from werewolf.helper.log_utils import Log
from werewolf.models.twd_tom.public_events import (
    PUBLIC_EVENT_SCHEMA_VERSION,
    normalize_public_events,
    public_event_digest,
)


TRAJECTORY_SCHEMA_VERSION = "classic7_game_interaction_trajectory_v2"
OBSERVATION_SCHEMA_VERSION = "classic7_agent_observation_v2"
OBSERVER_VIEW_PROVENANCE_SCHEMA_VERSION = (
    "classic7_observer_view_provenance_v2"
)
SIMULATOR_BASELINE = "classic7-witch-parity-v1"

PRE_PUBLIC_SPEECH = "PRE_PUBLIC_SPEECH"
POST_PUBLIC_SPEECH = "POST_PUBLIC_SPEECH"
PUBLIC_SPEECH_KINDS = frozenset({"speech", "speech_pk"})

_LOG_FIELDS = (
    "viewer",
    "source",
    "target",
    "content",
    "day",
    "time",
    "event",
)
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?P<key>[A-Za-z][A-Za-z0-9_-]*)"
    r"(?P<separator>\s*[:=]\s*)"
    r"(?P<value>[^\s,;]+)"
)
_BEARER_MESSAGE_PATTERN = re.compile(r"(?i)\bbearer\s+[^\s,;]+")


def serialize_json_value(value: Any) -> Any:
    """Convert one supported runtime value into strict JSON data."""

    if isinstance(value, Log):
        return {
            field: serialize_json_value(getattr(value, field))
            for field in _LOG_FIELDS
        }
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floats are not valid canonical JSON")
        return value
    if isinstance(value, (tuple, list)):
        return [serialize_json_value(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical JSON object keys must be strings")
        return {
            key: serialize_json_value(item)
            for key, item in value.items()
        }
    raise TypeError(
        f"unsupported canonical JSON runtime type: {type(value).__name__}"
    )


def canonical_json(value: Any) -> str:
    return json.dumps(
        serialize_json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def observation_record(observation: Any) -> dict[str, Any]:
    serialized = serialize_json_value(observation)
    if not isinstance(serialized, dict):
        raise TypeError("agent observation must be a mapping")
    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation": serialized,
        "observation_digest": canonical_digest(serialized),
    }


def _normalized_secret_key(key: str) -> str:
    camel_separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    return re.sub(r"[^a-z0-9]+", "_", camel_separated.lower()).strip("_")


def _is_secret_key(key: str) -> bool:
    normalized = _normalized_secret_key(key)
    if normalized.endswith("_env"):
        return False
    secret_suffixes = (
        "api_key",
        "token",
        "secret",
        "password",
        "credential",
        "credentials",
        "authorization",
        "authorization_header",
        "authorization_headers",
    )
    return any(
        normalized == suffix or normalized.endswith(f"_{suffix}")
        for suffix in secret_suffixes
    )


def _sanitize_runtime_config(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("runtime config keys must be strings")
        return {
            key: _sanitize_runtime_config(item)
            for key, item in value.items()
            if not _is_secret_key(key)
        }
    if isinstance(value, (tuple, list)):
        return [_sanitize_runtime_config(item) for item in value]
    return serialize_json_value(value)


def _sanitize_exception_message(exception: BaseException) -> str:
    message = " ".join(str(exception).splitlines()).strip()
    message = _BEARER_MESSAGE_PATTERN.sub("Bearer <redacted>", message)

    def redact_assignment(match: re.Match[str]) -> str:
        if not _is_secret_key(match.group("key")):
            return match.group(0)
        return (
            f"{match.group('key')}"
            f"{match.group('separator')}"
            "<redacted>"
        )

    return _SECRET_ASSIGNMENT_PATTERN.sub(redact_assignment, message)


def _alive_player_ids(env) -> list[int]:
    alive = getattr(env, "alive", None)
    if isinstance(alive, (str, bytes)) or not isinstance(alive, Sequence):
        raise TypeError("environment must provide an alive sequence")
    if len(alive) != 7:
        raise ValueError("Classic-7 environment must provide seven alive flags")
    players = [
        index + 1
        for index, is_alive in enumerate(alive)
        if is_alive == 1
    ]
    if not players:
        raise RuntimeError("environment has no alive players")
    return players


def _normalize_players(players: Any) -> list[dict[str, Any]]:
    if isinstance(players, (str, bytes)) or not isinstance(players, Sequence):
        raise TypeError("players must be a sequence")
    normalized = [serialize_json_value(player) for player in players]
    expected_fields = {
        "player_id",
        "role",
        "profile_name",
        "backend_id",
        "model_name",
    }
    if len(normalized) != 7:
        raise ValueError("Classic-7 trajectory requires exactly seven players")
    for index, player in enumerate(normalized, start=1):
        if not isinstance(player, dict) or set(player) != expected_fields:
            raise ValueError("player metadata fields do not match contract")
        if player["player_id"] != index:
            raise ValueError("players must use ascending player_id order")
        for field in expected_fields - {"player_id"}:
            if not isinstance(player[field], str) or not player[field].strip():
                raise ValueError(f"players.{field} must be non-empty text")
    return normalized


class CanonicalGameInteractionTrajectoryRecorder:
    """Capture one immutable trajectory plus C0 observer-view provenance."""

    def __init__(
        self,
        trajectory_output_path: str | Path,
        observer_view_output_path: str | Path,
        *,
        game_id: str,
        run_id: str,
        source_commit: str,
        environment_seed: int,
        runtime_config: Mapping[str, Any],
        players: Sequence[Mapping[str, Any]],
        simulator_baseline: str = SIMULATOR_BASELINE,
    ) -> None:
        for field_name, value in (
            ("game_id", game_id),
            ("run_id", run_id),
            ("source_commit", source_commit),
            ("simulator_baseline", simulator_baseline),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be non-empty text")
        if isinstance(environment_seed, bool) or not isinstance(
            environment_seed,
            int,
        ):
            raise TypeError("environment_seed must be an integer")
        if not isinstance(runtime_config, Mapping):
            raise TypeError("runtime_config must be a mapping")

        self.trajectory_output_path = Path(trajectory_output_path)
        self.observer_view_output_path = Path(observer_view_output_path)
        if self.trajectory_output_path == self.observer_view_output_path:
            raise ValueError("trajectory and observer-view paths must differ")
        for path in (
            self.trajectory_output_path,
            self.observer_view_output_path,
        ):
            if path.exists():
                raise FileExistsError(f"output already exists: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)

        sanitized_config = _sanitize_runtime_config(runtime_config)
        self._base = {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "game_id": game_id,
            "run_id": run_id,
            "source_commit": source_commit,
            "simulator_baseline": simulator_baseline,
            "environment_seed": environment_seed,
            "runtime_config": sanitized_config,
            "runtime_config_digest": canonical_digest(sanitized_config),
            "players": _normalize_players(players),
            "public_event_schema_version": PUBLIC_EVENT_SCHEMA_VERSION,
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        }
        self._initial_public_events: list[dict[str, Any]] | None = None
        self._transitions: list[dict[str, Any]] = []
        self._boundaries: list[dict[str, Any]] = []
        self._pending: dict[str, Any] | None = None
        self._finished = False

    def start(self, env, *, roles: Sequence[str]) -> None:
        if self._initial_public_events is not None:
            raise RuntimeError("trajectory recorder has already started")
        if list(roles) != [player["role"] for player in self._base["players"]]:
            raise ValueError("trajectory player roles do not match environment roles")
        self._initial_public_events = normalize_public_events(
            getattr(env, "public_events", None)
        )

    def before_agent_act(
        self,
        env,
        *,
        step_idx: int,
        acting_player_id: int,
        delivered_observation: Any,
        speech_kind: str | None,
    ) -> None:
        self._require_active()
        if self._pending is not None:
            raise RuntimeError("previous trajectory step is still pending")
        if isinstance(step_idx, bool) or not isinstance(step_idx, int):
            raise TypeError("step_idx must be an integer")
        if step_idx != len(self._transitions):
            raise ValueError(
                f"step_idx must equal the next committed index "
                f"{len(self._transitions)}"
            )
        delivered = observation_record(delivered_observation)
        phase_before = delivered["observation"].get("phase")
        if not isinstance(phase_before, str):
            raise TypeError("delivered observation phase must be text")
        current_actor = delivered["observation"].get("current_act_idx")
        if current_actor != acting_player_id:
            raise ValueError("delivered observation actor does not match step actor")
        public_events = normalize_public_events(
            getattr(env, "public_events", None)
        )
        reconstructed = self._reconstruct_public_events()
        if public_events != reconstructed:
            raise ValueError("environment public events diverged from trajectory")

        self._pending = {
            "step_idx": step_idx,
            "phase_before": phase_before,
            "acting_player_id": acting_player_id,
            "delivered_observation": delivered["observation"],
            "delivered_observation_digest": delivered["observation_digest"],
            "submitted_action": None,
            "public_event_count_before": len(public_events),
            "speech_kind": speech_kind,
        }
        if speech_kind is not None:
            if speech_kind not in PUBLIC_SPEECH_KINDS:
                raise ValueError("unsupported public speech kind")
            self._materialize_boundary(
                env,
                boundary_type=PRE_PUBLIC_SPEECH,
                speech_event_idx=None,
            )

    def after_agent_act(self, action: Any) -> None:
        pending = self._require_pending()
        if pending["submitted_action"] is not None:
            raise RuntimeError("trajectory step already has a submitted action")
        pending["submitted_action"] = serialize_json_value(action)

    def after_env_step(
        self,
        env,
        *,
        observation_after: Any,
        terminal_after: bool,
    ) -> None:
        pending = self._require_pending()
        if pending["submitted_action"] is None:
            raise RuntimeError("successful environment step has no submitted action")
        serialized_after = serialize_json_value(observation_after)
        if not isinstance(serialized_after, dict):
            raise TypeError("post-transition observation must be a mapping")
        phase_after = serialized_after.get("phase")
        if not isinstance(phase_after, str):
            raise TypeError("post-transition observation phase must be text")

        public_events = normalize_public_events(
            getattr(env, "public_events", None)
        )
        count_before = pending["public_event_count_before"]
        reconstructed = self._reconstruct_public_events()
        if public_events[:count_before] != reconstructed:
            raise ValueError("committed transition changed prior public events")
        appended = public_events[count_before:]
        self._validate_committed_speech(pending, appended)

        transition = {
            "step_idx": pending["step_idx"],
            "phase_before": pending["phase_before"],
            "acting_player_id": pending["acting_player_id"],
            "delivered_observation": pending["delivered_observation"],
            "delivered_observation_digest": pending[
                "delivered_observation_digest"
            ],
            "submitted_action": pending["submitted_action"],
            "public_event_count_before": count_before,
            "public_events_appended": appended,
            "phase_after": phase_after,
            "alive_players_after": _alive_player_ids(env),
            "terminal_after": bool(terminal_after),
        }
        self._transitions.append(transition)

        speech_kind = pending["speech_kind"]
        if speech_kind is not None:
            speech_events = [
                event
                for event in appended
                if event["event_type"] == "public_speech"
            ]
            self._materialize_boundary(
                env,
                boundary_type=POST_PUBLIC_SPEECH,
                speech_event_idx=speech_events[0]["event_idx"],
            )
        self._pending = None

    def fail(self, *, failure_stage: str, exception: BaseException) -> None:
        pending = self._require_pending()
        if failure_stage not in {"agent_act", "env_step"}:
            raise ValueError("unsupported trajectory failure stage")
        submitted_action = pending["submitted_action"]
        if failure_stage == "env_step" and submitted_action is None:
            raise RuntimeError("env_step failure has no submitted action")
        failure_context = {
            "failed_step_idx": pending["step_idx"],
            "failure_stage": failure_stage,
            "acting_player_id": pending["acting_player_id"],
            "phase_before": pending["phase_before"],
            "delivered_observation": pending["delivered_observation"],
            "delivered_observation_digest": pending[
                "delivered_observation_digest"
            ],
            "submitted_action": submitted_action,
            "exception_type": type(exception).__name__,
            "exception_message": _sanitize_exception_message(exception),
        }
        self._pending = None
        self._finalize(
            {
                "completion_status": "FAILED",
                "termination_kind": "exception",
                "failure_context": failure_context,
            }
        )

    def complete(self, env, *, winner: str) -> None:
        self._require_active()
        if self._pending is not None:
            raise RuntimeError("cannot complete with a pending trajectory step")
        terminal_steps = [
            transition
            for transition in self._transitions
            if transition["terminal_after"]
        ]
        if len(terminal_steps) != 1 or terminal_steps[0] is not self._transitions[-1]:
            raise ValueError("complete trajectory requires one final terminal step")
        if winner not in {"Werewolf", "Villager"}:
            raise ValueError("winner must be Werewolf or Villager")
        self._finalize(
            {
                "completion_status": "COMPLETE",
                "termination_kind": "normal_game_end",
                "winner": winner,
                "final_alive_players": _alive_player_ids(env),
            }
        )

    def abort(self) -> None:
        """Finalize an explicitly handled abort at the committed prefix."""

        self._require_active()
        self._pending = None
        self._finalize(
            {
                "completion_status": "ABORTED",
                "termination_kind": "explicit_handled_abort",
            }
        )

    def _materialize_boundary(
        self,
        env,
        *,
        boundary_type: str,
        speech_event_idx: int | None,
    ) -> None:
        pending = self._require_pending()
        public_events = normalize_public_events(
            getattr(env, "public_events", None)
        )
        observer_views = []
        for observer_id in _alive_player_ids(env):
            observation = (
                pending["delivered_observation"]
                if (
                    boundary_type == PRE_PUBLIC_SPEECH
                    and observer_id == pending["acting_player_id"]
                )
                else serialize_json_value(env.get_observation_for(observer_id))
            )
            if not isinstance(observation, dict):
                raise TypeError("observer view must be a mapping")
            observer_views.append(
                {
                    "observer_id": observer_id,
                    "observation": observation,
                    "observation_digest": canonical_digest(observation),
                }
            )
        boundary = {
            "boundary_id": (
                f"{self._base['game_id']}:step_{pending['step_idx']:06d}:"
                f"{boundary_type}"
            ),
            "boundary_type": boundary_type,
            "step_idx": pending["step_idx"],
            "speech_kind": pending["speech_kind"],
            "speaker_id": pending["acting_player_id"],
            "speech_event_idx": speech_event_idx,
            "public_event_count_at_materialization": len(public_events),
            "public_event_digest_at_materialization": public_event_digest(
                public_events
            ),
            "observer_views": observer_views,
        }
        boundary["boundary_digest"] = canonical_digest(boundary)
        self._boundaries.append(boundary)

    @staticmethod
    def _validate_committed_speech(pending, appended) -> None:
        speech_kind = pending["speech_kind"]
        if speech_kind is None:
            return
        speech_events = [
            event
            for event in appended
            if event["event_type"] == "public_speech"
        ]
        if len(speech_events) != 1:
            raise ValueError("speech step must append exactly one public speech")
        speech = speech_events[0]
        if speech["speaker"] != f"player{pending['acting_player_id']}":
            raise ValueError("committed public speech speaker mismatch")
        action = pending["submitted_action"]
        if not isinstance(action, list) or len(action) != 2:
            raise TypeError("submitted speech action must be a pair")
        if action[0] != speech_kind:
            raise ValueError("submitted speech action kind mismatch")
        content = action[1]
        if isinstance(content, dict):
            if set(content) != {"raw_text", "sp_actions"}:
                raise ValueError("strict speech action fields do not match contract")
            if content["raw_text"] != speech["raw_text"]:
                raise ValueError("submitted and committed speech raw_text differ")
        elif isinstance(content, str):
            if content != speech["raw_text"]:
                raise ValueError("submitted and committed speech raw_text differ")
        else:
            raise TypeError("speech action content must be text or a mapping")

    def _reconstruct_public_events(self) -> list[dict[str, Any]]:
        if self._initial_public_events is None:
            raise RuntimeError("trajectory recorder has not started")
        reconstructed = list(self._initial_public_events)
        for transition in self._transitions:
            reconstructed.extend(transition["public_events_appended"])
        return normalize_public_events(reconstructed)

    def _finalize(self, termination: dict[str, Any]) -> None:
        self._require_active()
        reconstructed = self._reconstruct_public_events()
        digest = public_event_digest(reconstructed)
        trajectory = {
            **self._base,
            "initial_public_events": self._initial_public_events,
            "transitions": self._transitions,
            "termination": termination,
            "public_event_digest": digest,
        }
        trajectory["trajectory_digest"] = canonical_digest(trajectory)
        provenance = {
            "schema_version": OBSERVER_VIEW_PROVENANCE_SCHEMA_VERSION,
            "game_id": self._base["game_id"],
            "run_id": self._base["run_id"],
            "source_commit": self._base["source_commit"],
            "simulator_baseline": self._base["simulator_baseline"],
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "trajectory_digest": trajectory["trajectory_digest"],
            "boundaries": self._boundaries,
        }
        provenance["artifact_digest"] = canonical_digest(provenance)
        trajectory_text = canonical_json(trajectory)
        provenance_text = canonical_json(provenance)
        with self.trajectory_output_path.open("x", encoding="utf-8") as output:
            output.write(trajectory_text)
        with self.observer_view_output_path.open("x", encoding="utf-8") as output:
            output.write(provenance_text)
        self._finished = True

    def _require_active(self) -> None:
        if self._finished:
            raise RuntimeError("trajectory recorder is already finalized")
        if self._initial_public_events is None:
            raise RuntimeError("trajectory recorder has not started")

    def _require_pending(self) -> dict[str, Any]:
        self._require_active()
        if self._pending is None:
            raise RuntimeError("trajectory recorder has no pending step")
        return self._pending


__all__ = [
    "CanonicalGameInteractionTrajectoryRecorder",
    "OBSERVER_VIEW_PROVENANCE_SCHEMA_VERSION",
    "OBSERVATION_SCHEMA_VERSION",
    "POST_PUBLIC_SPEECH",
    "PRE_PUBLIC_SPEECH",
    "SIMULATOR_BASELINE",
    "TRAJECTORY_SCHEMA_VERSION",
    "canonical_digest",
    "canonical_json",
    "observation_record",
    "serialize_json_value",
]
