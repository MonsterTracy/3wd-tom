"""Deterministically replay one completed canonical game without model calls."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
from werewolf.models.twd_tom.public_events import (
    normalize_public_events,
    public_event_digest,
)
from werewolf.trajectory import (
    OBSERVER_VIEW_PROVENANCE_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    POST_PUBLIC_SPEECH,
    PRE_PUBLIC_SPEECH,
    PUBLIC_SPEECH_KINDS,
    SIMULATOR_BASELINE,
    TRAJECTORY_SCHEMA_VERSION,
    canonical_digest,
    canonical_json,
    serialize_json_value,
)


REPLAY_RESULT_SCHEMA_VERSION = "classic7_canonical_replay_result_v1"


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact must contain an object: {path}")
    return value


def _alive_player_ids(env: WerewolfTextEnvV0) -> list[int]:
    return [
        player_id
        for player_id, is_alive in enumerate(env.alive, start=1)
        if is_alive == 1
    ]


def _load_boundaries(
    observer_views_path: Path,
    trajectory: Mapping[str, Any],
) -> dict[tuple[int, str], dict[str, Any]]:
    provenance = _load_json_object(observer_views_path)
    if provenance.get("schema_version") != OBSERVER_VIEW_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("observer-view schema version mismatch")
    if provenance.get("observation_schema_version") != OBSERVATION_SCHEMA_VERSION:
        raise ValueError("observer-view observation schema version mismatch")
    if provenance.get("simulator_baseline") != SIMULATOR_BASELINE:
        raise ValueError("observer-view simulator baseline mismatch")
    if provenance.get("trajectory_digest") != trajectory.get("trajectory_digest"):
        raise ValueError("observer-view trajectory digest mismatch")

    payload = deepcopy(provenance)
    recorded_digest = payload.pop("artifact_digest", None)
    if recorded_digest != canonical_digest(payload):
        raise ValueError("observer-view artifact digest mismatch")

    boundaries = provenance.get("boundaries")
    if not isinstance(boundaries, list):
        raise TypeError("observer-view boundaries must be a list")
    indexed: dict[tuple[int, str], dict[str, Any]] = {}
    for boundary in boundaries:
        if not isinstance(boundary, dict):
            raise TypeError("observer-view boundary must be an object")
        key = (boundary.get("step_idx"), boundary.get("boundary_type"))
        if key in indexed:
            raise ValueError("duplicate observer-view boundary")
        indexed[key] = boundary
    return indexed


def _verify_boundary(
    boundary: Mapping[str, Any],
    *,
    env: WerewolfTextEnvV0,
    step_idx: int,
    boundary_type: str,
    speech_kind: str,
    speaker_id: int,
    delivered_observation: Mapping[str, Any] | None,
    speech_event_idx: int | None,
) -> None:
    if boundary.get("step_idx") != step_idx:
        raise ValueError("replayed boundary step mismatch")
    if boundary.get("boundary_type") != boundary_type:
        raise ValueError("replayed boundary type mismatch")
    if boundary.get("speech_kind") != speech_kind:
        raise ValueError("replayed boundary speech kind mismatch")
    if boundary.get("speaker_id") != speaker_id:
        raise ValueError("replayed boundary speaker mismatch")
    if boundary.get("speech_event_idx") != speech_event_idx:
        raise ValueError("replayed boundary speech event mismatch")

    public_events = normalize_public_events(env.public_events)
    if boundary.get("public_event_count_at_materialization") != len(public_events):
        raise ValueError("replayed boundary public event count mismatch")
    if boundary.get("public_event_digest_at_materialization") != public_event_digest(
        public_events
    ):
        raise ValueError("replayed boundary public event digest mismatch")

    expected_views = []
    for observer_id in _alive_player_ids(env):
        if (
            boundary_type == PRE_PUBLIC_SPEECH
            and observer_id == speaker_id
        ):
            observation = delivered_observation
        else:
            observation = env.get_observation_for(observer_id)
        serialized = serialize_json_value(observation)
        expected_views.append(
            {
                "observer_id": observer_id,
                "observation": serialized,
                "observation_digest": canonical_digest(serialized),
            }
        )
    if boundary.get("observer_views") != expected_views:
        raise ValueError("replayed observer views differ from canonical boundary")

    payload = dict(boundary)
    recorded_digest = payload.pop("boundary_digest", None)
    if recorded_digest != canonical_digest(payload):
        raise ValueError("observer-view boundary digest mismatch")


def _runtime_action(
    submitted_action: Any,
    *,
    expected_appended_events: Sequence[Mapping[str, Any]],
) -> tuple[str, Any]:
    if not isinstance(submitted_action, list) or len(submitted_action) != 2:
        raise TypeError("submitted_action must be a JSON action pair")
    action_type, action_content = deepcopy(submitted_action)
    if not isinstance(action_type, str):
        raise TypeError("submitted action type must be text")

    if action_type in PUBLIC_SPEECH_KINDS and isinstance(action_content, str):
        speech_events = [
            event
            for event in expected_appended_events
            if event.get("event_type") == "public_speech"
        ]
        if len(speech_events) != 1:
            raise ValueError("text speech replay requires one recorded speech event")
        action_content = {
            "raw_text": action_content,
            # Replay only needs the immutable public text. Supplying an empty
            # generator contract prevents any parser/model call while leaving
            # the v4 public event byte-for-byte identical. Parser annotations
            # are validated separately against speech_annotations.jsonl.
            "sp_actions": [],
        }
    return action_type, action_content


def replay_canonical_trajectory(
    trajectory_path: str | Path,
    observer_views_path: str | Path | None = None,
) -> dict[str, Any]:
    """Replay stored actions and fail on any environment or view divergence."""

    trajectory_path = Path(trajectory_path).resolve()
    if observer_views_path is None:
        observer_views_path = trajectory_path.with_name("observer_views.json")
    observer_views_path = Path(observer_views_path).resolve()

    trajectory = _load_json_object(trajectory_path)
    if trajectory.get("schema_version") != TRAJECTORY_SCHEMA_VERSION:
        raise ValueError("trajectory schema version mismatch")
    if trajectory.get("simulator_baseline") != SIMULATOR_BASELINE:
        raise ValueError("trajectory simulator baseline mismatch")

    trajectory_payload = deepcopy(trajectory)
    recorded_trajectory_digest = trajectory_payload.pop("trajectory_digest", None)
    if recorded_trajectory_digest != canonical_digest(trajectory_payload):
        raise ValueError("trajectory digest mismatch")
    runtime_config = trajectory.get("runtime_config")
    if not isinstance(runtime_config, Mapping):
        raise TypeError("trajectory runtime_config must be an object")
    if trajectory.get("runtime_config_digest") != canonical_digest(runtime_config):
        raise ValueError("trajectory runtime config digest mismatch")

    environment_seed = trajectory.get("environment_seed")
    if isinstance(environment_seed, bool) or not isinstance(environment_seed, int):
        raise TypeError("trajectory environment_seed must be an integer")
    env_config = runtime_config.get("env_config")
    if not isinstance(env_config, Mapping):
        raise TypeError("trajectory runtime_config.env_config must be an object")

    players = trajectory.get("players")
    if not isinstance(players, list) or len(players) != 7:
        raise ValueError("trajectory must contain seven players")
    if any(not isinstance(player, Mapping) for player in players):
        raise TypeError("trajectory player entries must be objects")
    roles = [player.get("role") for player in players]
    replay_env_config = dict(env_config)
    replay_env_config["log_save_path"] = None
    replay_env_config["random_seed"] = environment_seed
    env = WerewolfTextEnvV0(**replay_env_config)
    observation = env.reset(roles=roles)

    initial_public_events = normalize_public_events(
        trajectory.get("initial_public_events")
    )
    if normalize_public_events(env.public_events) != initial_public_events:
        raise ValueError("replayed initial public events differ")

    boundaries = _load_boundaries(observer_views_path, trajectory)
    used_boundaries: set[tuple[int, str]] = set()
    transitions = trajectory.get("transitions")
    if not isinstance(transitions, list) or not transitions:
        raise ValueError("complete trajectory must contain transitions")

    final_info: Mapping[str, Any] = {}
    for step_idx, transition in enumerate(transitions):
        if not isinstance(transition, Mapping):
            raise TypeError("trajectory transition must be an object")
        if transition.get("step_idx") != step_idx:
            raise ValueError("trajectory step indices are not contiguous")

        delivered = serialize_json_value(observation)
        if delivered != transition.get("delivered_observation"):
            raise ValueError("replayed delivered observation differs")
        if canonical_digest(delivered) != transition.get(
            "delivered_observation_digest"
        ):
            raise ValueError("replayed delivered observation digest differs")
        if delivered.get("phase") != transition.get("phase_before"):
            raise ValueError("replayed pre-transition phase differs")
        if delivered.get("current_act_idx") != transition.get("acting_player_id"):
            raise ValueError("replayed acting player differs")

        raw_expected_appended = transition.get("public_events_appended")
        if not isinstance(raw_expected_appended, list):
            raise TypeError("transition public_events_appended must be a list")
        public_event_count_before = len(env.public_events)
        expected_public_events = normalize_public_events(
            [*env.public_events, *raw_expected_appended]
        )
        expected_appended = expected_public_events[public_event_count_before:]
        action = _runtime_action(
            transition.get("submitted_action"),
            expected_appended_events=expected_appended,
        )
        speech_kind = action[0] if action[0] in PUBLIC_SPEECH_KINDS else None
        if speech_kind is not None:
            key = (step_idx, PRE_PUBLIC_SPEECH)
            if key not in boundaries:
                raise ValueError("missing PRE speech observer-view boundary")
            _verify_boundary(
                boundaries[key],
                env=env,
                step_idx=step_idx,
                boundary_type=PRE_PUBLIC_SPEECH,
                speech_kind=speech_kind,
                speaker_id=transition["acting_player_id"],
                delivered_observation=delivered,
                speech_event_idx=None,
            )
            used_boundaries.add(key)

        if transition.get("public_event_count_before") != public_event_count_before:
            raise ValueError("replayed pre-transition public event count differs")
        observation, _, done, final_info = env.step(action)

        actual_public_events = normalize_public_events(env.public_events)
        actual_appended = actual_public_events[public_event_count_before:]
        if actual_appended != expected_appended:
            raise ValueError("replayed appended public events differ")
        serialized_after = serialize_json_value(observation)
        if serialized_after.get("phase") != transition.get("phase_after"):
            raise ValueError("replayed post-transition phase differs")
        if _alive_player_ids(env) != transition.get("alive_players_after"):
            raise ValueError("replayed alive players differ")
        if bool(done) != transition.get("terminal_after"):
            raise ValueError("replayed terminal flag differs")

        if speech_kind is not None:
            speech_events = [
                event
                for event in actual_appended
                if event["event_type"] == "public_speech"
            ]
            key = (step_idx, POST_PUBLIC_SPEECH)
            if key not in boundaries or len(speech_events) != 1:
                raise ValueError("missing POST speech observer-view boundary")
            _verify_boundary(
                boundaries[key],
                env=env,
                step_idx=step_idx,
                boundary_type=POST_PUBLIC_SPEECH,
                speech_kind=speech_kind,
                speaker_id=transition["acting_player_id"],
                delivered_observation=None,
                speech_event_idx=speech_events[0]["event_idx"],
            )
            used_boundaries.add(key)

    if used_boundaries != set(boundaries):
        raise ValueError("observer-view boundaries do not match replayed speech steps")
    if not transitions[-1].get("terminal_after"):
        raise ValueError("complete replay did not finish on the final transition")

    termination = trajectory.get("termination")
    if not isinstance(termination, Mapping):
        raise TypeError("trajectory termination must be an object")
    if termination.get("completion_status") != "COMPLETE":
        raise ValueError("only complete trajectories can be replayed")
    if termination.get("final_alive_players") != _alive_player_ids(env):
        raise ValueError("replayed final alive players differ")
    winner = termination.get("winner")
    expected_result = 1 if winner == "Werewolf" else -1 if winner == "Villager" else None
    if expected_result is None or final_info.get("Werewolf") != expected_result:
        raise ValueError("replayed winner differs")

    actual_public_events = normalize_public_events(env.public_events)
    if trajectory.get("public_event_digest") != public_event_digest(
        actual_public_events
    ):
        raise ValueError("replayed public event digest differs")

    result = {
        "schema_version": REPLAY_RESULT_SCHEMA_VERSION,
        "status": "MATCH",
        "game_id": trajectory.get("game_id"),
        "simulator_baseline": SIMULATOR_BASELINE,
        "environment_seed": environment_seed,
        "winner": winner,
        "transition_count": len(transitions),
        "public_event_count": len(actual_public_events),
        "observer_view_boundary_count": len(boundaries),
        "trajectory_digest": recorded_trajectory_digest,
    }
    result["replay_result_digest"] = canonical_digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay one canonical Classic-7 trajectory without model calls."
    )
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--observer-views", type=Path)
    args = parser.parse_args()
    print(
        canonical_json(
            replay_canonical_trajectory(
                args.trajectory,
                observer_views_path=args.observer_views,
            )
        )
    )


if __name__ == "__main__":
    main()
