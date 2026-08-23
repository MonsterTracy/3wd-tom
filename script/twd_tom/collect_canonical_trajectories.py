"""Collect one strict batch of canonical Classic-7 A/C0 trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from run_random import (
    build_runtime,
    build_twd_tom_sample_collector,
    eval as run_game,
)
from script.twd_tom.collection_budget import (
    GameCallBudgetAudit,
    audited_backends,
)
from werewolf.backends import load_named_backends
from werewolf.models.twd_tom.dataset import TARGET_CONVERSION
from werewolf.models.twd_tom.public_events import (
    PUBLIC_EVENT_SCHEMA_VERSION,
    normalize_public_events,
    public_event_digest,
    public_speech_actions,
    structured_input_digest,
)
from werewolf.models.twd_tom.samples import (
    SAMPLE_FIELDS,
    SAMPLE_SCHEMA_VERSION,
)
from werewolf.models.twd_tom.schema import (
    LABEL_PROMPT_VERSION,
    LABEL_PROVENANCE,
)
from werewolf.runtime_config import normalize_runtime_config
from werewolf.trajectory import (
    CanonicalGameInteractionTrajectoryRecorder,
    OBSERVER_VIEW_PROVENANCE_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    POST_PUBLIC_SPEECH,
    PRE_PUBLIC_SPEECH,
    SIMULATOR_BASELINE,
    TRAJECTORY_SCHEMA_VERSION,
    canonical_digest,
    canonical_json,
)


BATCH_PLAN_SCHEMA_VERSION = "classic7_canonical_gameplay_batch_plan_v2"
GAME_SUMMARY_SCHEMA_VERSION = "classic7_canonical_gameplay_game_summary_v2"
BATCH_SUMMARY_SCHEMA_VERSION = "classic7_canonical_gameplay_batch_summary_v2"
BATCH_FAILURE_SCHEMA_VERSION = "classic7_canonical_gameplay_batch_failure_v1"
PROJECTED_SCHEMA_VERSION = "classic7_observer_conditioned_belief_matrix_v1"

BACKEND_MAX_RETRIES = 0
STOP_ON_FIRST_FAILURE = True
RERUN_ON_FAILURE = False
REPLACEMENT_SEED_ON_FAILURE = False
BELIEF_SNAPSHOTS_FILENAME = "belief_snapshots.jsonl"

REPO_ROOT = Path(__file__).resolve().parents[2]
_GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

_TRAJECTORY_FIELDS = frozenset(
    {
        "schema_version",
        "game_id",
        "run_id",
        "source_commit",
        "simulator_baseline",
        "environment_seed",
        "runtime_config",
        "runtime_config_digest",
        "players",
        "public_event_schema_version",
        "observation_schema_version",
        "initial_public_events",
        "transitions",
        "termination",
        "public_event_digest",
        "trajectory_digest",
    }
)
_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "game_id",
        "run_id",
        "source_commit",
        "simulator_baseline",
        "observation_schema_version",
        "trajectory_digest",
        "boundaries",
        "artifact_digest",
    }
)
_TRANSITION_FIELDS = frozenset(
    {
        "step_idx",
        "phase_before",
        "acting_player_id",
        "delivered_observation",
        "delivered_observation_digest",
        "submitted_action",
        "public_event_count_before",
        "public_events_appended",
        "phase_after",
        "alive_players_after",
        "terminal_after",
    }
)
_BOUNDARY_FIELDS = frozenset(
    {
        "boundary_id",
        "boundary_type",
        "step_idx",
        "speech_kind",
        "speaker_id",
        "speech_event_idx",
        "public_event_count_at_materialization",
        "public_event_digest_at_materialization",
        "observer_views",
        "boundary_digest",
    }
)
_OBSERVER_VIEW_FIELDS = frozenset(
    {"observer_id", "observation", "observation_digest"}
)
_FORBIDDEN_BELIEF_ARTIFACT_KEYS = frozenset(
    {
        "observation",
        "private_observation",
        "delivered_observation",
        "role",
        "roles",
        "true_role",
        "true_roles",
        "winner",
    }
)


def _positive_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _run_id(value: Any) -> str:
    if not isinstance(value, str) or _RUN_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "run_id must match [A-Za-z0-9][A-Za-z0-9._-]*"
        )
    return value


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"required file not found: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required JSON artifact not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact must be an object: {path}")
    return value


def _load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"required JSONL artifact not found: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            raise ValueError(f"blank JSONL line at {path}:{line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"JSONL record must be an object: {path}:{line_number}")
        records.append(value)
    return records


def _reject_private_belief_artifact_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        forbidden = _FORBIDDEN_BELIEF_ARTIFACT_KEYS & set(value)
        if forbidden:
            raise ValueError(
                "belief snapshot artifact contains forbidden private/truth fields: "
                f"{sorted(forbidden)}"
            )
        for item in value.values():
            _reject_private_belief_artifact_keys(item)
    elif isinstance(value, list):
        for item in value:
            _reject_private_belief_artifact_keys(item)


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically publish one new canonical JSON object without overwrite."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(canonical_json(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"output already exists: {path}")
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _read_code_provenance(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    try:
        top_level = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--short", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"canonical batch collection requires a readable Git worktree: {root}"
        ) from exc
    if Path(top_level).resolve() != root:
        raise RuntimeError("canonical batch collection must resolve the repository root")
    if _GIT_SHA_PATTERN.fullmatch(commit) is None:
        raise RuntimeError("batch code commit must be a lowercase 40-character Git SHA")
    if dirty:
        raise RuntimeError(
            "canonical batch collection requires a clean Git worktree; dirty files:\n"
            + "\n".join(dirty)
        )
    return {"batch_code_commit": commit, "git_worktree_clean": True}


def _validate_classic7_config(normalized: Mapping[str, Any]) -> None:
    env_config = normalized.get("env_config")
    if not isinstance(env_config, Mapping):
        raise TypeError("normalized runtime config env_config must be a mapping")
    expected = {
        "n_player": 7,
        "n_werewolf": 2,
        "n_villager": 3,
        "n_seer": 1,
        "n_witch": 1,
        "n_guard": 0,
        "n_hunter": 0,
    }
    mismatches = {
        field: (env_config.get(field), expected_value)
        for field, expected_value in expected.items()
        if env_config.get(field) != expected_value
    }
    if mismatches:
        raise ValueError(
            "canonical batch requires frozen Classic-7 role counts; "
            f"mismatches={mismatches}"
        )


def _pipeline_collection_contract(
    parsed_yaml: Mapping[str, Any],
    *,
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Validate the frozen pipeline declaration against this invocation."""

    pipeline = parsed_yaml.get("pipeline")
    if not isinstance(pipeline, Mapping):
        raise TypeError("runtime config pipeline must be a mapping")
    expected_versions = {
        "public_event_schema_version": PUBLIC_EVENT_SCHEMA_VERSION,
        "raw_schema_version": SAMPLE_SCHEMA_VERSION,
        "projected_schema_version": PROJECTED_SCHEMA_VERSION,
        "projection_version": TARGET_CONVERSION,
    }
    mismatches = {
        field: (pipeline.get(field), expected)
        for field, expected in expected_versions.items()
        if pipeline.get(field) != expected
    }
    if mismatches:
        raise ValueError(
            "pipeline schema/projection contract mismatch; "
            f"mismatches={mismatches}"
        )

    collection = pipeline.get("collection")
    if not isinstance(collection, Mapping):
        raise TypeError("pipeline.collection must be a mapping")
    configured_game_count = _positive_integer(
        collection.get("game_count"),
        field_name="pipeline.collection.game_count",
    )
    configured_seeds = collection.get("seeds")
    if (
        isinstance(configured_seeds, (str, bytes))
        or not isinstance(configured_seeds, Sequence)
    ):
        raise TypeError("pipeline.collection.seeds must be a sequence")
    configured_seeds = list(configured_seeds)
    if any(
        isinstance(seed, bool) or not isinstance(seed, int)
        for seed in configured_seeds
    ):
        raise TypeError("pipeline.collection.seeds must contain integers")
    if configured_game_count != len(configured_seeds):
        raise ValueError(
            "pipeline.collection.game_count must equal the configured seed count"
        )
    if configured_seeds != list(seeds):
        raise ValueError(
            "CLI seed range/game_count must exactly match pipeline.collection"
        )

    contract = {
        "game_count": configured_game_count,
        "seeds": configured_seeds,
        "max_gameplay_calls_per_game": _positive_integer(
            collection.get("max_gameplay_calls_per_game"),
            field_name="pipeline.collection.max_gameplay_calls_per_game",
        ),
        "max_belief_calls_per_game": _positive_integer(
            collection.get("max_belief_calls_per_game"),
            field_name="pipeline.collection.max_belief_calls_per_game",
        ),
        "max_total_calls_per_game": _positive_integer(
            collection.get("max_total_calls_per_game"),
            field_name="pipeline.collection.max_total_calls_per_game",
        ),
    }
    wall_seconds = collection.get("max_wall_seconds_per_game")
    if (
        isinstance(wall_seconds, bool)
        or not isinstance(wall_seconds, (int, float))
        or wall_seconds <= 0
    ):
        raise ValueError(
            "pipeline.collection.max_wall_seconds_per_game "
            "must be a positive number"
        )
    contract["max_wall_seconds_per_game"] = float(wall_seconds)
    if contract["max_total_calls_per_game"] < max(
        contract["max_gameplay_calls_per_game"],
        contract["max_belief_calls_per_game"],
    ):
        raise ValueError(
            "pipeline.collection max_total_calls_per_game cannot be smaller "
            "than a category budget"
        )
    return contract


def _game_id(run_id: str, game_number: int, seed: int) -> str:
    return f"{run_id}_game_{game_number:04d}_seed_{seed}"


def _build_players(
    *,
    roles: Sequence[str],
    profile_names: Sequence[str],
    agents: Sequence[Any],
) -> list[dict[str, Any]]:
    if len(roles) != 7 or len(profile_names) != 7 or len(agents) != 7:
        raise ValueError("canonical Classic-7 player metadata requires seven entries")
    players = []
    for player_id, (role, profile_name, agent) in enumerate(
        zip(roles, profile_names, agents), start=1
    ):
        for field_name, value in (
            ("role", role),
            ("profile_name", profile_name),
            ("backend_id", getattr(agent, "backend_id", None)),
            ("model_name", getattr(agent, "model_name", None)),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"player{player_id} {field_name} must be non-empty text"
                )
        players.append(
            {
                "player_id": player_id,
                "role": role,
                "profile_name": profile_name,
                "backend_id": agent.backend_id,
                "model_name": agent.model_name,
            }
        )
    return players


def _expected_winner(
    players: Sequence[Mapping[str, Any]], final_alive_players: Sequence[int]
) -> str | None:
    alive = set(final_alive_players)
    alive_roles = [
        player["role"] for player in players if player["player_id"] in alive
    ]
    if "Werewolf" not in alive_roles:
        return "Villager"
    if "Villager" not in alive_roles:
        return "Werewolf"
    if "Seer" not in alive_roles and "Witch" not in alive_roles:
        return "Werewolf"
    return None


def validate_complete_game_artifacts(
    trajectory_path: str | Path,
    observer_views_path: str | Path,
    *,
    expected_game_id: str,
    expected_run_id: str,
    expected_seed: int,
    expected_source_commit: str,
) -> dict[str, Any]:
    """Strictly validate one completed recorder A/C0 pair and return a summary."""

    trajectory_path = Path(trajectory_path)
    observer_views_path = Path(observer_views_path)
    trajectory = _load_json_object(trajectory_path)
    provenance = _load_json_object(observer_views_path)

    if set(trajectory) != _TRAJECTORY_FIELDS:
        raise ValueError("trajectory top-level fields do not match contract")
    if set(provenance) != _PROVENANCE_FIELDS:
        raise ValueError("observer-view top-level fields do not match contract")

    expected_identity = {
        "game_id": expected_game_id,
        "run_id": expected_run_id,
        "source_commit": expected_source_commit,
    }
    for field_name, expected_value in expected_identity.items():
        if trajectory[field_name] != expected_value:
            raise ValueError(f"trajectory {field_name} mismatch")
        if provenance[field_name] != expected_value:
            raise ValueError(f"observer-view {field_name} mismatch")
    if trajectory["environment_seed"] != expected_seed:
        raise ValueError("trajectory environment_seed mismatch")
    if trajectory["schema_version"] != TRAJECTORY_SCHEMA_VERSION:
        raise ValueError("trajectory schema version mismatch")
    if provenance["schema_version"] != OBSERVER_VIEW_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("observer-view schema version mismatch")
    if trajectory["simulator_baseline"] != SIMULATOR_BASELINE:
        raise ValueError("trajectory simulator baseline mismatch")
    if provenance["simulator_baseline"] != SIMULATOR_BASELINE:
        raise ValueError("observer-view simulator baseline mismatch")
    if trajectory["public_event_schema_version"] != PUBLIC_EVENT_SCHEMA_VERSION:
        raise ValueError("trajectory public-event schema mismatch")
    if trajectory["observation_schema_version"] != OBSERVATION_SCHEMA_VERSION:
        raise ValueError("trajectory observation schema mismatch")
    if provenance["observation_schema_version"] != OBSERVATION_SCHEMA_VERSION:
        raise ValueError("observer-view observation schema mismatch")

    runtime_config = trajectory["runtime_config"]
    if not isinstance(runtime_config, Mapping):
        raise TypeError("trajectory runtime_config must be a mapping")
    if trajectory["runtime_config_digest"] != canonical_digest(runtime_config):
        raise ValueError("trajectory runtime_config_digest mismatch")

    trajectory_payload = deepcopy(trajectory)
    recorded_trajectory_digest = trajectory_payload.pop("trajectory_digest")
    if recorded_trajectory_digest != canonical_digest(trajectory_payload):
        raise ValueError("trajectory_digest mismatch")
    if provenance["trajectory_digest"] != recorded_trajectory_digest:
        raise ValueError("observer-view trajectory_digest mismatch")

    provenance_payload = deepcopy(provenance)
    recorded_artifact_digest = provenance_payload.pop("artifact_digest")
    if recorded_artifact_digest != canonical_digest(provenance_payload):
        raise ValueError("observer-view artifact_digest mismatch")

    initial_events = normalize_public_events(trajectory["initial_public_events"])
    reconstructed = list(initial_events)
    transitions = trajectory["transitions"]
    if not isinstance(transitions, list) or not transitions:
        raise ValueError("complete trajectory must contain transitions")
    terminal_indices = []
    speech_steps: list[tuple[int, Mapping[str, Any], list[Mapping[str, Any]]]] = []
    for expected_step, transition in enumerate(transitions):
        if not isinstance(transition, Mapping) or set(transition) != _TRANSITION_FIELDS:
            raise ValueError("trajectory transition fields do not match contract")
        if transition["step_idx"] != expected_step:
            raise ValueError("trajectory step_idx is not contiguous")
        observation = transition["delivered_observation"]
        if not isinstance(observation, Mapping):
            raise TypeError("delivered_observation must be a mapping")
        if observation.get("current_act_idx") != transition["acting_player_id"]:
            raise ValueError("delivered observation actor mismatch")
        if observation.get("phase") != transition["phase_before"]:
            raise ValueError("delivered observation phase mismatch")
        if transition["delivered_observation_digest"] != canonical_digest(observation):
            raise ValueError("delivered_observation_digest mismatch")
        if transition["public_event_count_before"] != len(reconstructed):
            raise ValueError("public_event_count_before mismatch")
        appended = transition["public_events_appended"]
        if not isinstance(appended, list):
            raise TypeError("public_events_appended must be a list")
        action = transition["submitted_action"]
        if (
            isinstance(action, list)
            and len(action) == 2
            and action[0] in {"speech", "speech_pk"}
        ):
            speeches = [
                event
                for event in appended
                if isinstance(event, Mapping)
                and event.get("event_type") == "public_speech"
            ]
            if len(speeches) != 1:
                raise ValueError("speech step must append exactly one public_speech")
            speech = speeches[0]
            if speech.get("speaker") != f"player{transition['acting_player_id']}":
                raise ValueError("committed public speech speaker mismatch")
            content = action[1]
            if isinstance(content, Mapping):
                if set(content) != {"raw_text", "sp_actions"}:
                    raise ValueError("strict speech action fields do not match contract")
                if content["raw_text"] != speech.get("raw_text"):
                    raise ValueError("submitted and committed speech raw_text differ")
                if content["sp_actions"] != speech.get("sp_actions"):
                    raise ValueError("submitted and committed speech sp_actions differ")
            elif isinstance(content, str):
                if content != speech.get("raw_text"):
                    raise ValueError("submitted and committed speech raw_text differ")
            else:
                raise TypeError("speech action content must be text or a mapping")
            speech_steps.append((expected_step, transition, speeches))
        reconstructed = normalize_public_events([*reconstructed, *appended])
        alive = transition["alive_players_after"]
        if (
            not isinstance(alive, list)
            or not alive
            or any(isinstance(player_id, bool) or not isinstance(player_id, int) for player_id in alive)
            or any(not 1 <= player_id <= 7 for player_id in alive)
            or len(set(alive)) != len(alive)
            or alive != sorted(alive)
        ):
            raise ValueError(
                "alive_players_after must be unique ascending Classic-7 player IDs"
            )
        if transition["terminal_after"] is True:
            terminal_indices.append(expected_step)
        elif transition["terminal_after"] is not False:
            raise TypeError("terminal_after must be boolean")

    reconstructed = normalize_public_events(reconstructed)
    if trajectory["public_event_digest"] != public_event_digest(reconstructed):
        raise ValueError("trajectory public_event_digest mismatch")
    if terminal_indices != [len(transitions) - 1]:
        raise ValueError("complete trajectory requires exactly one final terminal step")

    termination = trajectory["termination"]
    if not isinstance(termination, Mapping) or set(termination) != {
        "completion_status",
        "termination_kind",
        "winner",
        "final_alive_players",
    }:
        raise ValueError("complete termination fields do not match contract")
    if termination["completion_status"] != "COMPLETE":
        raise ValueError("production corpus requires COMPLETE games")
    if termination["termination_kind"] != "normal_game_end":
        raise ValueError("production corpus requires normal_game_end")
    if termination["winner"] not in {"Werewolf", "Villager"}:
        raise ValueError("trajectory winner is invalid")
    if transitions[-1]["alive_players_after"] != termination["final_alive_players"]:
        raise ValueError("final alive players disagree with final transition")

    players = trajectory["players"]
    if not isinstance(players, list) or len(players) != 7:
        raise ValueError("trajectory must contain seven player metadata entries")
    expected_player_fields = {
        "player_id", "role", "profile_name", "backend_id", "model_name"
    }
    for player_id, player in enumerate(players, start=1):
        if not isinstance(player, Mapping) or set(player) != expected_player_fields:
            raise ValueError("player metadata fields do not match contract")
        if player["player_id"] != player_id:
            raise ValueError("player metadata must use ascending IDs")
    mechanically_expected = _expected_winner(players, termination["final_alive_players"])
    if mechanically_expected != termination["winner"]:
        raise ValueError("trajectory winner is not mechanically valid")

    boundaries = provenance["boundaries"]
    if not isinstance(boundaries, list):
        raise TypeError("observer-view boundaries must be a list")
    if len(boundaries) != 2 * len(speech_steps):
        raise ValueError(
            "observer-view provenance must contain exactly PRE+POST for each speech step"
        )
    by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
    observer_view_count = 0
    for boundary in boundaries:
        if not isinstance(boundary, Mapping) or set(boundary) != _BOUNDARY_FIELDS:
            raise ValueError("observer-view boundary fields do not match contract")
        boundary_payload = deepcopy(dict(boundary))
        recorded_boundary_digest = boundary_payload.pop("boundary_digest")
        if recorded_boundary_digest != canonical_digest(boundary_payload):
            raise ValueError("boundary_digest mismatch")
        step_idx = boundary["step_idx"]
        boundary_type = boundary["boundary_type"]
        if (
            isinstance(step_idx, bool)
            or not isinstance(step_idx, int)
            or not 0 <= step_idx < len(transitions)
        ):
            raise ValueError("boundary step_idx is invalid")
        if boundary_type not in {PRE_PUBLIC_SPEECH, POST_PUBLIC_SPEECH}:
            raise ValueError("unsupported observer-view boundary type")
        key = (step_idx, boundary_type)
        if key in by_key:
            raise ValueError("duplicate observer-view boundary")
        by_key[key] = boundary
        views = boundary["observer_views"]
        if not isinstance(views, list) or not views:
            raise ValueError("boundary observer_views cannot be empty")
        observer_ids = []
        for view in views:
            if not isinstance(view, Mapping) or set(view) != _OBSERVER_VIEW_FIELDS:
                raise ValueError("observer-view fields do not match contract")
            observer_id = view["observer_id"]
            if (
                isinstance(observer_id, bool)
                or not isinstance(observer_id, int)
                or not 1 <= observer_id <= 7
            ):
                raise ValueError("observer_id must be a Classic-7 player ID")
            observer_ids.append(observer_id)
            observation = view["observation"]
            if not isinstance(observation, Mapping):
                raise TypeError("observer observation must be a mapping")
            if observation.get("observer_id") != observer_id:
                raise ValueError("observer observation identity mismatch")
            if view["observation_digest"] != canonical_digest(observation):
                raise ValueError("observer observation digest mismatch")
            observer_view_count += 1
        if observer_ids != sorted(observer_ids) or len(set(observer_ids)) != len(observer_ids):
            raise ValueError("boundary observer IDs must be unique ascending IDs")

    for step_idx, transition, speech_events in speech_steps:
        action_kind = transition["submitted_action"][0]
        pre = by_key.get((step_idx, PRE_PUBLIC_SPEECH))
        post = by_key.get((step_idx, POST_PUBLIC_SPEECH))
        if pre is None or post is None:
            raise ValueError("speech step is missing PRE/POST observer-view boundaries")
        for boundary in (pre, post):
            if boundary["speech_kind"] != action_kind:
                raise ValueError("boundary speech kind mismatch")
            if boundary["speaker_id"] != transition["acting_player_id"]:
                raise ValueError("boundary speaker mismatch")
            expected_boundary_id = (
                f"{expected_game_id}:step_{step_idx:06d}:{boundary['boundary_type']}"
            )
            if boundary["boundary_id"] != expected_boundary_id:
                raise ValueError("boundary_id mismatch")

        if pre["speech_event_idx"] is not None:
            raise ValueError("PRE boundary speech_event_idx must be null")
        if pre["public_event_count_at_materialization"] != transition["public_event_count_before"]:
            raise ValueError("PRE public event count mismatch")
        pre_count = transition["public_event_count_before"]
        if pre["public_event_digest_at_materialization"] != public_event_digest(
            reconstructed[:pre_count]
        ):
            raise ValueError("PRE public event digest mismatch")
        pre_views = {int(view["observer_id"]): view for view in pre["observer_views"]}
        actor = int(transition["acting_player_id"])
        if actor not in pre_views:
            raise ValueError("PRE boundary is missing acting-player view")
        if pre_views[actor]["observation"] != transition["delivered_observation"]:
            raise ValueError("PRE acting-player observation differs from delivered observation")
        alive_before = (
            list(range(1, 8))
            if step_idx == 0
            else transitions[step_idx - 1]["alive_players_after"]
        )
        if sorted(pre_views) != alive_before:
            raise ValueError("PRE observer set differs from alive players before speech")
        for view in pre_views.values():
            if view["observation"].get("current_act_idx") != transition["acting_player_id"]:
                raise ValueError("PRE observer view current actor mismatch")
            if view["observation"].get("phase") != transition["phase_before"]:
                raise ValueError("PRE observer view phase mismatch")

        post_count = transition["public_event_count_before"] + len(
            transition["public_events_appended"]
        )
        if post["public_event_count_at_materialization"] != post_count:
            raise ValueError("POST public event count mismatch")
        if post["public_event_digest_at_materialization"] != public_event_digest(
            reconstructed[:post_count]
        ):
            raise ValueError("POST public event digest mismatch")
        if post["speech_event_idx"] != speech_events[0]["event_idx"]:
            raise ValueError("POST speech_event_idx mismatch")
        post_views = {int(view["observer_id"]): view for view in post["observer_views"]}
        if sorted(post_views) != transition["alive_players_after"]:
            raise ValueError("POST observer set differs from alive players after speech")

    return {
        "game_id": expected_game_id,
        "run_id": expected_run_id,
        "environment_seed": expected_seed,
        "completion_status": "COMPLETE",
        "winner": termination["winner"],
        "transition_count": len(transitions),
        "speech_transition_count": len(speech_steps),
        "boundary_count": len(boundaries),
        "public_event_count": len(reconstructed),
        "pre_public_speech_boundary_count": len(speech_steps),
        "post_public_speech_boundary_count": len(speech_steps),
        "observer_view_count": observer_view_count,
        "runtime_config_digest": trajectory["runtime_config_digest"],
        "trajectory_digest": recorded_trajectory_digest,
        "observer_view_artifact_digest": recorded_artifact_digest,
        "trajectory_sha256": _sha256(trajectory_path),
        "observer_views_sha256": _sha256(observer_views_path),
    }


def validate_belief_snapshot_artifact(
    belief_snapshots_path: str | Path,
    observer_views_path: str | Path,
    *,
    expected_game_id: str,
) -> dict[str, Any]:
    """Validate raw self-reports against the canonical PRE-speech cutoffs."""

    belief_snapshots_path = Path(belief_snapshots_path)
    provenance = _load_json_object(Path(observer_views_path))
    records = _load_jsonl_objects(belief_snapshots_path)
    pre_boundaries = {
        boundary["step_idx"]: boundary
        for boundary in provenance.get("boundaries", [])
        if isinstance(boundary, Mapping)
        and boundary.get("boundary_type") == PRE_PUBLIC_SPEECH
    }
    if len(records) != len(pre_boundaries):
        raise ValueError(
            "belief snapshot count must equal PRE_PUBLIC_SPEECH boundary count"
        )

    seen_steps: set[int] = set()
    for record in records:
        _reject_private_belief_artifact_keys(record)
        if set(record) != SAMPLE_FIELDS:
            raise ValueError("belief snapshot fields do not match raw sample contract")
        if record["schema_version"] != SAMPLE_SCHEMA_VERSION:
            raise ValueError("belief snapshot schema version mismatch")
        if record["game_id"] != expected_game_id:
            raise ValueError("belief snapshot game_id mismatch")
        if record["label_prompt_version"] != LABEL_PROMPT_VERSION:
            raise ValueError("belief snapshot label prompt version mismatch")
        if record["label_provenance"] != LABEL_PROVENANCE:
            raise ValueError("belief snapshot label provenance mismatch")
        if record["public_event_schema_version"] != PUBLIC_EVENT_SCHEMA_VERSION:
            raise ValueError("belief snapshot public-event schema mismatch")

        step_idx = record["step_idx"]
        if isinstance(step_idx, bool) or not isinstance(step_idx, int):
            raise TypeError("belief snapshot step_idx must be an integer")
        if step_idx in seen_steps:
            raise ValueError("duplicate belief snapshot step_idx")
        seen_steps.add(step_idx)
        if record["label_cutoff_step_idx"] != step_idx:
            raise ValueError("belief snapshot cutoff must equal its PRE-speech step")
        boundary = pre_boundaries.get(step_idx)
        if boundary is None:
            raise ValueError("belief snapshot has no matching PRE-speech boundary")

        expected_trigger = {
            "speech": "pre_public_speech",
            "speech_pk": "pre_public_speech_pk",
        }.get(boundary["speech_kind"])
        if record["report_trigger"] != expected_trigger:
            raise ValueError("belief snapshot report trigger mismatch")
        if record["speaker_id"] != boundary["speaker_id"]:
            raise ValueError("belief snapshot speaker mismatch")
        boundary_views = boundary["observer_views"]
        expected_observer_ids = [view["observer_id"] for view in boundary_views]
        if record["observer_ids"] != expected_observer_ids:
            raise ValueError("belief snapshot observer identities mismatch")
        expected_subjects = {f"player{player_id}" for player_id in expected_observer_ids}
        for field_name in (
            "suspected_werewolves",
            "known_werewolves",
            "known_non_werewolves",
            "belief_status",
            "belief_errors",
            "agent_backend_ids",
        ):
            value = record[field_name]
            if not isinstance(value, Mapping) or set(value) != expected_subjects:
                raise ValueError(f"belief snapshot {field_name} observer set mismatch")
        failed_reports = {
            subject: status
            for subject, status in record["belief_status"].items()
            if status != "ok"
        }
        if failed_reports:
            raise ValueError(
                "canonical belief snapshot requires status=ok for every "
                f"alive observer; failures={failed_reports}"
            )
        if any(
            error is not None
            for error in record["belief_errors"].values()
        ):
            raise ValueError(
                "successful canonical belief reports must have null errors"
            )
        if any(
            not isinstance(suspected, list)
            for suspected in record["suspected_werewolves"].values()
        ):
            raise TypeError(
                "successful canonical suspected_werewolves rows must be lists"
            )

        phases = {view["observation"].get("phase") for view in boundary_views}
        if phases != {record["phase"]}:
            raise ValueError("belief snapshot phase differs from PRE observer views")
        public_events = normalize_public_events(record["public_events"])
        if len(public_events) != boundary["public_event_count_at_materialization"]:
            raise ValueError("belief snapshot public cutoff count mismatch")
        if record["public_event_digest"] != public_event_digest(public_events):
            raise ValueError("belief snapshot public event digest mismatch")
        if record["public_event_digest"] != (
            boundary["public_event_digest_at_materialization"]
        ):
            raise ValueError("belief snapshot public cutoff differs from PRE boundary")
        if record["structured_input_digest"] != structured_input_digest(public_events):
            raise ValueError("belief snapshot structured input digest mismatch")
        if record["public_action_count"] != len(public_speech_actions(public_events)):
            raise ValueError("belief snapshot public action count mismatch")

    if seen_steps != set(pre_boundaries):
        raise ValueError("belief snapshots do not cover every PRE-speech boundary")
    return {
        "belief_snapshot_count": len(records),
        "belief_report_count": sum(
            len(record["observer_ids"])
            for record in records
        ),
        "belief_snapshots_sha256": _sha256(belief_snapshots_path),
    }


def _game_summary(
    validation: Mapping[str, Any], *, summary_path: Path
) -> dict[str, Any]:
    summary = {
        "schema_version": GAME_SUMMARY_SCHEMA_VERSION,
        **dict(validation),
    }
    summary["summary_digest"] = canonical_digest(summary)
    _write_json_new(summary_path, summary)
    return summary


def _batch_plan(
    *,
    run_id: str,
    commit: str,
    config_sha256: str,
    normalized_runtime_config_digest: str,
    seeds: Sequence[int],
    collection_contract: Mapping[str, Any],
) -> dict[str, Any]:
    plan = {
        "schema_version": BATCH_PLAN_SCHEMA_VERSION,
        "batch_code_commit": commit,
        "git_worktree_clean": True,
        "run_id": run_id,
        "simulator_baseline": SIMULATOR_BASELINE,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "observer_view_schema_version": OBSERVER_VIEW_PROVENANCE_SCHEMA_VERSION,
        "public_event_schema_version": PUBLIC_EVENT_SCHEMA_VERSION,
        "config_sha256": config_sha256,
        "normalized_runtime_config_digest": normalized_runtime_config_digest,
        "seeds": list(seeds),
        "planned_game_count": len(seeds),
        "canonical_runtime_builder": "run_random.build_runtime",
        "canonical_game_driver": "run_random.eval",
        "collectors_enabled": True,
        "collection_contract": dict(collection_contract),
        "backend_max_retries": BACKEND_MAX_RETRIES,
        "stop_on_first_failure": STOP_ON_FIRST_FAILURE,
        "rerun_on_failure": RERUN_ON_FAILURE,
        "replacement_seed_on_failure": REPLACEMENT_SEED_ON_FAILURE,
    }
    plan["plan_digest"] = canonical_digest(plan)
    return plan


def _failure_record(
    *,
    run_id: str,
    commit: str,
    failed_seed: int | None,
    failed_game_id: str | None,
    failure_stage: str,
    exception: BaseException,
    completed_games: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    record = {
        "schema_version": BATCH_FAILURE_SCHEMA_VERSION,
        "batch_code_commit": commit,
        "run_id": run_id,
        "failed_seed": failed_seed,
        "failed_game_id": failed_game_id,
        "failure_stage": failure_stage,
        "exception_type": type(exception).__name__,
        "completed_game_count": len(completed_games),
        "completed_game_ids": [game["game_id"] for game in completed_games],
        "stop_on_first_failure": True,
        "rerun_on_failure": False,
        "replacement_seed_on_failure": False,
    }
    record["failure_digest"] = canonical_digest(record)
    return record


def collect_canonical_trajectory_batch(
    *,
    config_path: str | Path,
    run_id: str,
    seed_start: int,
    game_count: int,
    output_root: str | Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Run one immutable no-retry Classic-7 canonical A/C0 batch."""

    run_id = _run_id(run_id)
    if isinstance(seed_start, bool) or not isinstance(seed_start, int):
        raise ValueError("seed_start must be an integer")
    game_count = _positive_integer(game_count, field_name="game_count")
    seeds = list(range(seed_start, seed_start + game_count))

    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"runtime config not found: {config_path}")
    destination = Path(output_root).resolve()
    if destination.exists():
        raise FileExistsError(f"output root already exists: {destination}")

    provenance = _read_code_provenance(Path(repo_root))
    parsed_yaml = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(parsed_yaml, Mapping):
        raise ValueError("runtime config must be a mapping")
    normalized = normalize_runtime_config(deepcopy(parsed_yaml))
    _validate_classic7_config(normalized)
    collection_contract = _pipeline_collection_contract(
        parsed_yaml,
        seeds=seeds,
    )
    normalized_digest = canonical_digest(normalized)

    destination.mkdir(parents=True)
    plan = _batch_plan(
        run_id=run_id,
        commit=provenance["batch_code_commit"],
        config_sha256=_sha256(config_path),
        normalized_runtime_config_digest=normalized_digest,
        seeds=seeds,
        collection_contract=collection_contract,
    )
    _write_json_new(destination / "plan.json", plan)

    completed_games: list[dict[str, Any]] = []
    try:
        try:
            backend_map = load_named_backends(
                normalized,
                env_file=Path(repo_root).resolve() / ".env",
                max_retries=BACKEND_MAX_RETRIES,
            )
        except BaseException as exc:
            failure = _failure_record(
                run_id=run_id,
                commit=provenance["batch_code_commit"],
                failed_seed=None,
                failed_game_id=None,
                failure_stage="backend_load",
                exception=exc,
                completed_games=completed_games,
            )
            _write_json_new(destination / "batch_failure.json", failure)
            raise

        for game_number, seed in enumerate(seeds, start=1):
            game_id = _game_id(run_id, game_number, seed)
            game_dir = destination / "games" / f"game_{game_number:04d}_seed_{seed}"
            log_dir = game_dir / "game_logs"
            game_dir.mkdir(parents=True)
            log_dir.mkdir()
            trajectory_path = game_dir / "trajectory.json"
            observer_views_path = game_dir / "observer_views.json"
            belief_snapshots_path = game_dir / BELIEF_SNAPSHOTS_FILENAME
            call_audit_path = game_dir / "call_audit.json"
            summary_path = game_dir / "summary.json"
            call_audit = GameCallBudgetAudit(
                game_id=game_id,
                max_gameplay_calls=collection_contract[
                    "max_gameplay_calls_per_game"
                ],
                max_belief_calls=collection_contract[
                    "max_belief_calls_per_game"
                ],
                max_total_calls=collection_contract[
                    "max_total_calls_per_game"
                ],
                max_wall_seconds=collection_contract[
                    "max_wall_seconds_per_game"
                ],
            )
            stage = "build_runtime"
            try:
                env, agents, roles, profile_names = build_runtime(
                    deepcopy(parsed_yaml),
                    log_save_path=str(log_dir),
                    random_seed=seed,
                    backends=audited_backends(backend_map, call_audit),
                )
                players = _build_players(
                    roles=roles,
                    profile_names=profile_names,
                    agents=agents,
                )
                stage = "recorder_init"
                recorder = CanonicalGameInteractionTrajectoryRecorder(
                    trajectory_path,
                    observer_views_path,
                    game_id=game_id,
                    run_id=run_id,
                    source_commit=provenance["batch_code_commit"],
                    environment_seed=seed,
                    runtime_config=normalized,
                    players=players,
                )
                stage = "belief_collector_init"
                belief_collector = build_twd_tom_sample_collector(
                    agent_list=agents,
                    output_path=str(belief_snapshots_path),
                    game_id=game_id,
                    report_audit=call_audit,
                )
                try:
                    stage = "gameplay"
                    result = run_game(
                        env,
                        agents,
                        roles,
                        sample_collector=belief_collector,
                        call_audit=call_audit,
                        trajectory_recorder=recorder,
                    )
                    call_audit.assert_wall_budget()
                finally:
                    belief_collector.close()
                call_audit_record = call_audit.snapshot()
                if not call_audit_record["within_budget"]:
                    raise RuntimeError("game call audit finished outside its budget")
                _write_json_new(call_audit_path, call_audit_record)
                stage = "artifact_validation"
                validation = validate_complete_game_artifacts(
                    trajectory_path,
                    observer_views_path,
                    expected_game_id=game_id,
                    expected_run_id=run_id,
                    expected_seed=seed,
                    expected_source_commit=provenance["batch_code_commit"],
                )
                validation.update(
                    validate_belief_snapshot_artifact(
                        belief_snapshots_path,
                        observer_views_path,
                        expected_game_id=game_id,
                    )
                )
                if validation["runtime_config_digest"] != normalized_digest:
                    raise ValueError(
                        "trajectory runtime_config_digest differs from frozen batch config"
                    )
                expected_result = f"{validation['winner']} win"
                if result != expected_result:
                    raise ValueError("run_random result disagrees with trajectory winner")
                validation["run_random_result"] = result
                validation["call_audit"] = call_audit_record
                game_summary = _game_summary(validation, summary_path=summary_path)
                completed_games.append(game_summary)
            except BaseException as exc:
                if not call_audit_path.exists():
                    _write_json_new(call_audit_path, call_audit.snapshot())
                failure = _failure_record(
                    run_id=run_id,
                    commit=provenance["batch_code_commit"],
                    failed_seed=seed,
                    failed_game_id=game_id,
                    failure_stage=stage,
                    exception=exc,
                    completed_games=completed_games,
                )
                _write_json_new(destination / "batch_failure.json", failure)
                raise

        summary = {
            "schema_version": BATCH_SUMMARY_SCHEMA_VERSION,
            "batch_code_commit": provenance["batch_code_commit"],
            "run_id": run_id,
            "plan_digest": plan["plan_digest"],
            "planned_game_count": game_count,
            "completed_game_count": len(completed_games),
            "seeds": seeds,
            "game_ids": [game["game_id"] for game in completed_games],
            "game_summary_digests": {
                game["game_id"]: game["summary_digest"] for game in completed_games
            },
            "winner_counts": {
                "Werewolf": sum(game["winner"] == "Werewolf" for game in completed_games),
                "Villager": sum(game["winner"] == "Villager" for game in completed_games),
            },
            "total_transition_count": sum(
                game["transition_count"] for game in completed_games
            ),
            "total_speech_transition_count": sum(
                game["speech_transition_count"] for game in completed_games
            ),
            "total_pre_public_speech_boundary_count": sum(
                game["pre_public_speech_boundary_count"] for game in completed_games
            ),
            "total_post_public_speech_boundary_count": sum(
                game["post_public_speech_boundary_count"] for game in completed_games
            ),
            "total_observer_view_count": sum(
                game["observer_view_count"] for game in completed_games
            ),
            "total_belief_snapshot_count": sum(
                game["belief_snapshot_count"] for game in completed_games
            ),
            "total_belief_report_count": sum(
                game["belief_report_count"] for game in completed_games
            ),
            "total_gameplay_call_count": sum(
                game["call_audit"]["gameplay_call_count"]
                for game in completed_games
            ),
            "total_belief_call_count": sum(
                game["call_audit"]["belief_call_count"]
                for game in completed_games
            ),
            "total_backend_call_count": sum(
                game["call_audit"]["total_call_count"]
                for game in completed_games
            ),
            "backend_max_retries": BACKEND_MAX_RETRIES,
            "stop_on_first_failure": True,
            "rerun_on_failure": False,
            "replacement_seed_on_failure": False,
        }
        summary["summary_digest"] = canonical_digest(summary)
        _write_json_new(destination / "summary.json", summary)
        return summary
    except BaseException:
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect one strict canonical Classic-7 A/C0 batch."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed-start", required=True, type=int)
    parser.add_argument("--game-count", required=True, type=int)
    parser.add_argument("--output-root", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = collect_canonical_trajectory_batch(
        config_path=args.config,
        run_id=args.run_id,
        seed_start=args.seed_start,
        game_count=args.game_count,
        output_root=args.output_root,
    )
    print(
        "CANONICAL_A_C0_BATCH_PASS "
        f"run_id={summary['run_id']} games={summary['completed_game_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
