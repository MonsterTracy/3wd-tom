"""Compare a detached shadow speech parser with recorded annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from script.twd_tom.collect_canonical_trajectories import (
    _read_code_provenance,
    validate_speech_annotation_artifact,
)
from werewolf.backends.openai_compatible import OpenAICompatibleBackend
from werewolf.models.twd_tom.public_events import (
    normalize_public_events,
    parse_public_phase,
)
from werewolf.models.twd_tom.speech_annotations import (
    STATUS_ERROR,
    STATUS_NO_ACTION,
    STATUS_OK,
    make_speech_annotation,
    normalize_speech_annotations,
)
from werewolf.runtime_config import (
    normalize_backend_config,
    normalize_parser_config,
)
from werewolf.speech.speech_perceiver import (
    SPEECH_PARSER_GENERATION_MAX_ATTEMPTS,
    SPEECH_PARSER_MAX_TOKENS,
    SpeechPerceiver,
)
from werewolf.trajectory import canonical_digest, canonical_json


REPO_ROOT = Path(__file__).resolve().parents[2]
SHADOW_PLAN_SCHEMA_VERSION = "classic7_shadow_speech_parser_plan_v1"
SHADOW_COMPARISON_SCHEMA_VERSION = "classic7_shadow_speech_comparison_v1"
SHADOW_GAME_SUMMARY_SCHEMA_VERSION = "classic7_shadow_speech_game_summary_v1"
SHADOW_SUMMARY_SCHEMA_VERSION = "classic7_shadow_speech_batch_summary_v1"
DEEPSEEK_NON_THINKING_REQUEST = {"thinking": {"type": "disabled"}}


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact must contain an object: {path}")
    return value


def _load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL line at {path}:{line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise TypeError(
                    f"JSONL record must contain an object: {path}:{line_number}"
                )
            records.append(record)
    return records


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(canonical_json(value) + "\n")


def _write_jsonl_new(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(canonical_json(record) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_embedded_digest(
    value: Mapping[str, Any],
    *,
    digest_field: str,
    artifact_name: str,
) -> str:
    payload = deepcopy(dict(value))
    recorded = payload.pop(digest_field, None)
    if not isinstance(recorded, str) or recorded != canonical_digest(payload):
        raise ValueError(f"{artifact_name} digest mismatch")
    return recorded


def _load_shadow_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("shadow parser config must be a mapping")
    if set(raw) != {"backends", "parser"}:
        raise ValueError("shadow parser config must contain only backends and parser")

    raw_backends = raw["backends"]
    if not isinstance(raw_backends, Mapping) or len(raw_backends) != 1:
        raise ValueError("shadow parser config requires exactly one backend")
    backend_name, raw_backend = next(iter(raw_backends.items()))
    if not isinstance(backend_name, str) or not backend_name.strip():
        raise ValueError("shadow backend name must be non-empty text")
    if not isinstance(raw_backend, Mapping):
        raise ValueError("shadow backend config must be a mapping")
    allowed_backend_fields = {
        "type",
        "base_url",
        "api_key_env",
        "default_model",
        "supports_json_schema",
    }
    if set(raw_backend) - allowed_backend_fields:
        raise ValueError("shadow backend config contains unsupported fields")
    backend = normalize_backend_config(raw_backend)
    if not isinstance(backend["base_url"], str) or not backend["base_url"].strip():
        raise ValueError("shadow backend base_url must be non-empty text")
    if not isinstance(backend["api_key_env"], str) or not backend[
        "api_key_env"
    ].strip():
        raise ValueError("shadow backend api_key_env must be non-empty text")

    raw_parser = raw["parser"]
    if not isinstance(raw_parser, Mapping):
        raise ValueError("shadow parser config must be a mapping")
    if set(raw_parser) != {"backend", "model", "model_params"}:
        raise ValueError("shadow parser config field set mismatch")
    parser = normalize_parser_config(
        raw_parser,
        {backend_name: backend},
    )
    model_params = parser["model_params"]
    if set(model_params) != {"temperature", "request_extra_body"}:
        raise ValueError("shadow parser model_params field set mismatch")
    temperature = model_params["temperature"]
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("shadow parser temperature must be numeric")
    if float(temperature) != 0.0:
        raise ValueError("shadow parser temperature must equal 0")
    if model_params["request_extra_body"] != DEEPSEEK_NON_THINKING_REQUEST:
        raise ValueError(
            "shadow parser must explicitly disable DeepSeek thinking"
        )

    return {
        "backends": {backend_name: backend},
        "parser": parser,
    }


def _build_backend(
    normalized: Mapping[str, Any],
    *,
    env_file: Path | None,
) -> OpenAICompatibleBackend:
    if env_file is not None:
        load_dotenv(dotenv_path=env_file, override=False)
    parser_config = normalized["parser"]
    backend_config = normalized["backends"][parser_config["backend"]]
    api_key_env = backend_config["api_key_env"]
    api_key = os.environ.get(api_key_env)
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError(f"API key environment variable {api_key_env} is required")
    return OpenAICompatibleBackend(
        api_key=api_key,
        base_url=backend_config["base_url"],
        default_model=backend_config["default_model"],
        max_retries=0,
        supports_json_schema=backend_config["supports_json_schema"],
    )


def _source_games(
    input_root: Path,
    batch_summary: Mapping[str, Any],
) -> list[tuple[Path, dict[str, Any]]]:
    game_ids = batch_summary.get("game_ids")
    if not isinstance(game_ids, list) or not game_ids:
        raise ValueError("source batch summary requires non-empty game_ids")
    if any(not isinstance(game_id, str) or not game_id for game_id in game_ids):
        raise ValueError("source batch game_ids must be non-empty text")
    if len(game_ids) != len(set(game_ids)):
        raise ValueError("source batch game_ids must be unique")

    games_root = input_root / "games"
    if not games_root.is_dir():
        raise FileNotFoundError(f"source games directory not found: {games_root}")
    games: dict[str, tuple[Path, dict[str, Any]]] = {}
    for game_dir in sorted(games_root.iterdir()):
        if not game_dir.is_dir():
            raise ValueError(f"unexpected non-directory in source games: {game_dir}")
        trajectory_path = game_dir / "trajectory.json"
        if not trajectory_path.is_file():
            raise FileNotFoundError(f"source trajectory not found: {trajectory_path}")
        trajectory = _load_json_object(trajectory_path)
        game_id = trajectory.get("game_id")
        if not isinstance(game_id, str) or not game_id:
            raise ValueError(f"source trajectory has no game_id: {trajectory_path}")
        if game_id in games:
            raise ValueError(f"duplicate source game_id: {game_id}")
        games[game_id] = (game_dir, trajectory)
    if set(games) != set(game_ids):
        raise ValueError("source game directories differ from batch summary game_ids")
    return [games[game_id] for game_id in game_ids]


def _public_events(trajectory: Mapping[str, Any]) -> list[dict[str, Any]]:
    events = list(trajectory.get("initial_public_events", []))
    transitions = trajectory.get("transitions")
    if not isinstance(transitions, list):
        raise TypeError("source trajectory transitions must be a list")
    for transition in transitions:
        if not isinstance(transition, Mapping):
            raise TypeError("source trajectory transition must be a mapping")
        appended = transition.get("public_events_appended")
        if not isinstance(appended, list):
            raise TypeError("public_events_appended must be a list")
        events.extend(appended)
    return normalize_public_events(events)


def _speech_inputs(trajectory: Mapping[str, Any]) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    for transition in trajectory["transitions"]:
        speeches = [
            event
            for event in transition["public_events_appended"]
            if event.get("event_type") == "public_speech"
        ]
        if not speeches:
            continue
        if len(speeches) != 1:
            raise ValueError("one transition may append at most one public speech")
        day, phase_category = parse_public_phase(transition.get("phase_before"))
        if not phase_category.startswith("day_"):
            raise ValueError("public speech transition must occur during daytime")
        event = speeches[0]
        speaker = event["speaker"]
        if not isinstance(speaker, str) or not speaker.startswith("player"):
            raise ValueError("public speech speaker must use canonical player ID")
        inputs.append(
            {
                "event_idx": event["event_idx"],
                "speaker": speaker,
                "speaker_id": int(speaker.removeprefix("player")),
                "raw_text": event["raw_text"],
                "day": day,
                "phase": phase_category.removeprefix("day_"),
            }
        )
    return inputs


def _action_set(actions: Sequence[Any]) -> set[str]:
    return {canonical_json(action) for action in actions}


def _shadow_annotation(
    *,
    perceiver: SpeechPerceiver,
    parser_model_id: str,
    game_id: str,
    speech_input: Mapping[str, Any],
) -> dict[str, Any]:
    audit = perceiver.parse_with_audit(
        speaker=speech_input["speaker_id"],
        speech=speech_input["raw_text"],
        day=speech_input["day"],
        phase=speech_input["phase"],
    )
    if audit.parse_status == "ok":
        actions = audit.normalized_actions
        status = STATUS_OK if actions else STATUS_NO_ACTION
        error_type = None
        error_message = None
    else:
        actions = []
        status = STATUS_ERROR
        error_type = audit.error_type or "SpeechParserError"
        error_message = audit.error_message or "shadow speech parser failed"
    return make_speech_annotation(
        event_idx=speech_input["event_idx"],
        speaker=speech_input["speaker"],
        raw_text=speech_input["raw_text"],
        parser_model_id=parser_model_id,
        parser_call_id=(
            f"shadow_{game_id}_event_{speech_input['event_idx']:06d}"
        ),
        annotation_source="llm_parser",
        status=status,
        actions=actions,
        generation_attempts=audit.generation_attempts,
        raw_response=audit.raw_response,
        error_type=error_type,
        error_message=error_message,
    )


def audit_shadow_speech_parser(
    *,
    input_root: str | Path,
    config_path: str | Path,
    output_root: str | Path,
    env_file: str | Path | None = ".env",
    backend=None,
    code_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a detached parser comparison without mutating source artifacts."""

    input_root = Path(input_root).resolve()
    config_path = Path(config_path).resolve()
    output_root = Path(output_root).resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"source batch root not found: {input_root}")
    if not config_path.is_file():
        raise FileNotFoundError(f"shadow parser config not found: {config_path}")
    if output_root.exists():
        raise FileExistsError(f"shadow output root already exists: {output_root}")
    if input_root == output_root or input_root in output_root.parents:
        raise ValueError("shadow output must be outside the source batch root")

    normalized = _load_shadow_config(config_path)
    batch_summary_path = input_root / "summary.json"
    batch_summary = _load_json_object(batch_summary_path)
    source_summary_digest = _validate_embedded_digest(
        batch_summary,
        digest_field="summary_digest",
        artifact_name="source batch summary",
    )
    source_games = _source_games(input_root, batch_summary)
    provenance = dict(
        code_provenance
        if code_provenance is not None
        else _read_code_provenance(REPO_ROOT)
    )
    if provenance.get("git_worktree_clean") is not True:
        raise ValueError("shadow parser audit requires a clean Git worktree")

    if backend is None:
        backend = _build_backend(
            normalized,
            env_file=None if env_file is None else Path(env_file).resolve(),
        )
    parser_config = normalized["parser"]
    backend_config = normalized["backends"][parser_config["backend"]]
    perceiver = SpeechPerceiver(
        backend=backend,
        model_name=parser_config["model"],
        request_extra_body=parser_config["model_params"]["request_extra_body"],
    )

    plan = {
        "schema_version": SHADOW_PLAN_SCHEMA_VERSION,
        "source_run_id": batch_summary.get("run_id"),
        "source_batch_summary_digest": source_summary_digest,
        "source_batch_summary_sha256": _sha256(batch_summary_path),
        "shadow_code_commit": provenance.get("batch_code_commit"),
        "shadow_config_sha256": _sha256(config_path),
        "parser_backend": parser_config["backend"],
        "parser_base_url": backend_config["base_url"],
        "parser_api_key_env": backend_config["api_key_env"],
        "parser_model_id": parser_config["model"],
        "temperature": 0.0,
        "request_extra_body": deepcopy(
            parser_config["model_params"]["request_extra_body"]
        ),
        "max_tokens": SPEECH_PARSER_MAX_TOKENS,
        "generation_max_attempts": SPEECH_PARSER_GENERATION_MAX_ATTEMPTS,
        "backend_sdk_max_retries": 0,
        "source_artifacts_mutated": False,
    }
    plan["plan_digest"] = canonical_digest(plan)
    output_root.mkdir(parents=True)
    _write_json_new(output_root / "plan.json", plan)

    canonical_status_counts: Counter[str] = Counter()
    shadow_status_counts: Counter[str] = Counter()
    total_attempts = 0
    exact_match_count = 0
    action_set_match_count = 0
    status_match_count = 0
    game_summaries: list[dict[str, Any]] = []

    for game_dir, trajectory in source_games:
        game_id = trajectory["game_id"]
        trajectory_path = game_dir / "trajectory.json"
        canonical_path = game_dir / "speech_annotations.jsonl"
        validate_speech_annotation_artifact(
            canonical_path,
            trajectory_path,
            require_success=False,
        )
        public_events = _public_events(trajectory)
        canonical_annotations = normalize_speech_annotations(
            _load_jsonl_objects(canonical_path),
            public_events=public_events,
            require_complete=True,
        )
        canonical_by_event = {
            annotation["event_idx"]: annotation
            for annotation in canonical_annotations
        }
        speech_inputs = _speech_inputs(trajectory)
        if {item["event_idx"] for item in speech_inputs} != set(canonical_by_event):
            raise ValueError("source speech inputs differ from canonical annotations")

        shadow_annotations: list[dict[str, Any]] = []
        comparisons: list[dict[str, Any]] = []
        game_exact_matches = 0
        game_action_set_matches = 0
        game_status_matches = 0
        game_attempts = 0
        for speech_input in speech_inputs:
            canonical = canonical_by_event[speech_input["event_idx"]]
            shadow = _shadow_annotation(
                perceiver=perceiver,
                parser_model_id=parser_config["model"],
                game_id=game_id,
                speech_input=speech_input,
            )
            shadow_annotations.append(shadow)
            canonical_status_counts[canonical["status"]] += 1
            shadow_status_counts[shadow["status"]] += 1
            attempts = len(shadow["generation_attempts"])
            game_attempts += attempts

            both_valid = (
                canonical["status"] != STATUS_ERROR
                and shadow["status"] != STATUS_ERROR
            )
            exact_match = both_valid and canonical["actions"] == shadow["actions"]
            action_set_match = both_valid and _action_set(
                canonical["actions"]
            ) == _action_set(shadow["actions"])
            status_match = canonical["status"] == shadow["status"]
            game_exact_matches += int(exact_match)
            game_action_set_matches += int(action_set_match)
            game_status_matches += int(status_match)
            comparisons.append(
                {
                    "schema_version": SHADOW_COMPARISON_SCHEMA_VERSION,
                    "game_id": game_id,
                    "event_idx": speech_input["event_idx"],
                    "speaker": speech_input["speaker"],
                    "raw_text": speech_input["raw_text"],
                    "canonical_status": canonical["status"],
                    "canonical_actions": canonical["actions"],
                    "shadow_status": shadow["status"],
                    "shadow_actions": shadow["actions"],
                    "status_match": status_match,
                    "exact_action_order_match": exact_match,
                    "action_set_match": action_set_match,
                }
            )

        relative_game_dir = game_dir.relative_to(input_root)
        output_game_dir = output_root / relative_game_dir
        shadow_path = output_game_dir / "shadow_speech_annotations.jsonl"
        comparison_path = output_game_dir / "shadow_speech_comparisons.jsonl"
        _write_jsonl_new(shadow_path, shadow_annotations)
        _write_jsonl_new(comparison_path, comparisons)
        game_summary = {
            "schema_version": SHADOW_GAME_SUMMARY_SCHEMA_VERSION,
            "game_id": game_id,
            "source_trajectory_sha256": _sha256(trajectory_path),
            "source_speech_annotations_sha256": _sha256(canonical_path),
            "speech_count": len(speech_inputs),
            "shadow_generation_attempt_count": game_attempts,
            "shadow_retry_count": game_attempts - len(speech_inputs),
            "status_match_count": game_status_matches,
            "exact_action_order_match_count": game_exact_matches,
            "action_set_match_count": game_action_set_matches,
            "shadow_speech_annotations_sha256": _sha256(shadow_path),
            "shadow_speech_comparisons_sha256": _sha256(comparison_path),
        }
        game_summary["game_summary_digest"] = canonical_digest(game_summary)
        _write_json_new(output_game_dir / "summary.json", game_summary)
        game_summaries.append(game_summary)
        total_attempts += game_attempts
        exact_match_count += game_exact_matches
        action_set_match_count += game_action_set_matches
        status_match_count += game_status_matches

    total_speech_count = sum(item["speech_count"] for item in game_summaries)
    summary = {
        "schema_version": SHADOW_SUMMARY_SCHEMA_VERSION,
        "completion_status": "COMPLETE",
        "plan_digest": plan["plan_digest"],
        "source_run_id": batch_summary.get("run_id"),
        "source_batch_summary_digest": source_summary_digest,
        "game_count": len(game_summaries),
        "speech_count": total_speech_count,
        "canonical_status_counts": dict(sorted(canonical_status_counts.items())),
        "shadow_status_counts": dict(sorted(shadow_status_counts.items())),
        "shadow_generation_attempt_count": total_attempts,
        "shadow_retry_count": total_attempts - total_speech_count,
        "status_match_count": status_match_count,
        "exact_action_order_match_count": exact_match_count,
        "action_set_match_count": action_set_match_count,
        "disagreement_count": total_speech_count - action_set_match_count,
        "game_summary_digests": {
            item["game_id"]: item["game_summary_digest"]
            for item in game_summaries
        },
        "source_artifacts_mutated": False,
    }
    summary["summary_digest"] = canonical_digest(summary)
    _write_json_new(output_root / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a detached DeepSeek speech parser over an existing batch and "
            "write comparison-only artifacts."
        )
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--env-file", default=".env")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = audit_shadow_speech_parser(
        input_root=args.input_root,
        config_path=args.config,
        output_root=args.output_root,
        env_file=args.env_file,
    )
    print(canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
