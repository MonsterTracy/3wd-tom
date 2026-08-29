"""Collect a fresh, fail-closed Classic7 ONUW-parity pilot batch."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import yaml

from run_random import (
    build_onuw_role_guess_collector,
    build_runtime,
    eval as run_game,
)
from script.twd_tom.collect_canonical_trajectories import (
    BACKEND_SDK_MAX_RETRIES,
    REPO_ROOT,
    _build_players,
    _game_id,
    _read_code_provenance,
    _sha256,
    _validate_classic7_config,
    _write_json_new,
    _write_jsonl_new,
    validate_complete_game_artifacts,
    validate_speech_annotation_artifact,
)
from script.twd_tom.collection_budget import (
    BACKEND_MAX_ATTEMPTS,
    GameCallBudgetAudit,
    audited_backends,
)
from script.twd_tom.replay_canonical_trajectory import (
    replay_canonical_trajectory,
)
from werewolf.backends import load_named_backends
from werewolf.models.twd_tom.onuw_parity_audit import pilot_collection_audit
from werewolf.models.twd_tom.onuw_parity_protocol import (
    CLASSIC7_ONUW_REFERENCE,
    ONUW_ACTION_ONLY,
    ONUW_AGENT_DECLARED_MULTIMODAL,
)
from werewolf.models.twd_tom.onuw_parity_recorder import (
    OnuwParityGameRecorder,
)
from werewolf.models.twd_tom.speech_annotations import (
    normalize_speech_annotations,
)
from werewolf.runtime_config import normalize_runtime_config
from werewolf.trajectory import (
    CanonicalGameInteractionTrajectoryRecorder,
    canonical_digest,
    sanitize_exception_message,
)


PARITY_PILOT_NAMESPACE = "onuw_parity_pilot_v1"
PARITY_PILOT_PLAN_SCHEMA = "classic7_onuw_parity_pilot_plan_v1"
PARITY_PILOT_GAME_SUMMARY_SCHEMA = "classic7_onuw_parity_pilot_game_summary_v1"
PARITY_PILOT_SUMMARY_SCHEMA = "classic7_onuw_parity_pilot_summary_v1"
PARITY_PILOT_FAILURE_SCHEMA = "classic7_onuw_parity_pilot_failure_v1"


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _pilot_contract(
    parsed: Mapping[str, Any],
    *,
    seeds: Sequence[int],
) -> dict[str, Any]:
    pipeline = parsed.get("pipeline")
    if not isinstance(pipeline, Mapping):
        raise TypeError("config pipeline must be a mapping")
    parity = pipeline.get("onuw_parity_pilot")
    if not isinstance(parity, Mapping):
        raise TypeError("pipeline.onuw_parity_pilot must be a mapping")
    required = {
        "data_namespace": PARITY_PILOT_NAMESPACE,
        "protocol_id": CLASSIC7_ONUW_REFERENCE,
        "content_profile": ONUW_ACTION_ONLY,
        "modality_profile": ONUW_AGENT_DECLARED_MULTIMODAL,
        "label_collector": "onuw_style_role_guess",
        "timing": "strict_pre",
        "model_input": "public_only",
        "allow_gameplay_fallback": False,
        "formal_training_eligible": False,
    }
    mismatches = {
        field: (parity.get(field), expected)
        for field, expected in required.items()
        if parity.get(field) != expected
    }
    if mismatches:
        raise ValueError(f"ONUW parity pilot contract mismatch: {mismatches}")
    collection = parity.get("collection")
    if not isinstance(collection, Mapping):
        raise TypeError("onuw parity pilot collection must be a mapping")
    configured_seeds = collection.get("seeds")
    if isinstance(configured_seeds, (str, bytes)) or not isinstance(
        configured_seeds, Sequence
    ):
        raise TypeError("pilot seeds must be a sequence")
    configured_seeds = list(configured_seeds)
    if configured_seeds != list(seeds):
        raise ValueError("CLI seed range must exactly match pilot config")
    game_count = _positive_int(collection.get("game_count"), field="game_count")
    if game_count != len(configured_seeds):
        raise ValueError("pilot game_count must equal seed count")
    contract = {
        **required,
        "game_count": game_count,
        "seeds": configured_seeds,
        "max_gameplay_calls_per_game": _positive_int(
            collection.get("max_gameplay_calls_per_game"),
            field="max_gameplay_calls_per_game",
        ),
        "max_belief_calls_per_game": _positive_int(
            collection.get("max_belief_calls_per_game"),
            field="max_belief_calls_per_game",
        ),
        "max_total_calls_per_game": _positive_int(
            collection.get("max_total_calls_per_game"),
            field="max_total_calls_per_game",
        ),
    }
    wall = collection.get("max_wall_seconds_per_game")
    if isinstance(wall, bool) or not isinstance(wall, (int, float)) or wall <= 0:
        raise ValueError("max_wall_seconds_per_game must be positive")
    contract["max_wall_seconds_per_game"] = float(wall)
    if contract["max_total_calls_per_game"] < max(
        contract["max_gameplay_calls_per_game"],
        contract["max_belief_calls_per_game"],
    ):
        raise ValueError("total call budget is smaller than a category budget")
    return contract


def _validate_agent_profiles(normalized: Mapping[str, Any]) -> None:
    profiles = normalized.get("agent_config", {}).get("all_candidates")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("parity pilot requires at least one agent profile")
    for profile in profiles:
        params = profile.get("model_params")
        if not isinstance(params, Mapping):
            raise TypeError("agent model_params must be a mapping")
        if params.get("gameplay_prompt_profile") != "strict_classic7":
            raise ValueError("parity pilot requires strict_classic7 gameplay")
        if (
            params.get("speech_modality_profile")
            != ONUW_AGENT_DECLARED_MULTIMODAL
        ):
            raise ValueError("parity pilot requires agent-declared face/tone")


def _close_agents(agents: Sequence[Any]) -> None:
    for agent in agents:
        close = getattr(agent, "close", None)
        if callable(close):
            close()


def collect_onuw_parity_pilot(
    *,
    config_path: str | Path,
    run_id: str,
    seed_start: int,
    game_count: int,
    output_root: str | Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Collect one new non-training pilot namespace without fallback."""

    if not isinstance(run_id, str) or not run_id.startswith("onuw_parity_pilot_"):
        raise ValueError("run_id must start with onuw_parity_pilot_")
    if isinstance(seed_start, bool) or not isinstance(seed_start, int):
        raise TypeError("seed_start must be an integer")
    game_count = _positive_int(game_count, field="game_count")
    seeds = list(range(seed_start, seed_start + game_count))
    destination = Path(output_root).resolve()
    if destination.exists():
        raise FileExistsError(f"fresh pilot output already exists: {destination}")
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"pilot config not found: {config_path}")

    provenance = _read_code_provenance(Path(repo_root))
    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(parsed, Mapping):
        raise TypeError("pilot config must be a mapping")
    normalized = normalize_runtime_config(deepcopy(parsed))
    _validate_classic7_config(normalized)
    _validate_agent_profiles(normalized)
    contract = _pilot_contract(parsed, seeds=seeds)
    if contract["game_count"] != game_count:
        raise ValueError("CLI game_count must exactly match pilot config")

    plan = {
        "schema_version": PARITY_PILOT_PLAN_SCHEMA,
        "data_namespace": PARITY_PILOT_NAMESPACE,
        "run_id": run_id,
        "source_commit": provenance["batch_code_commit"],
        "git_worktree_clean": True,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "normalized_runtime_config_digest": canonical_digest(normalized),
        "contract": contract,
    }
    plan["plan_digest"] = canonical_digest(plan)
    destination.mkdir(parents=True)
    _write_json_new(destination / "plan.json", plan)

    backend_map = load_named_backends(
        normalized,
        env_file=Path(repo_root).resolve() / ".env",
        max_retries=BACKEND_SDK_MAX_RETRIES,
    )
    completed_games = []
    completed_audits = []
    game_summaries = []
    failures = []
    for game_number, seed in enumerate(seeds, start=1):
        game_id = _game_id(run_id, game_number, seed)
        directory_name = f"game_{game_number:04d}_seed_{seed}"
        game_dir = destination / "games" / directory_name
        log_dir = game_dir / "game_logs"
        log_dir.mkdir(parents=True)
        call_audit = GameCallBudgetAudit(
            game_id=game_id,
            max_gameplay_calls=contract["max_gameplay_calls_per_game"],
            max_belief_calls=contract["max_belief_calls_per_game"],
            max_total_calls=contract["max_total_calls_per_game"],
            max_wall_seconds=contract["max_wall_seconds_per_game"],
            max_backend_attempts=BACKEND_MAX_ATTEMPTS,
        )
        agents = []
        stage = "build_runtime"
        try:
            env, agents, roles, profile_names = build_runtime(
                deepcopy(parsed),
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
            trajectory_recorder = CanonicalGameInteractionTrajectoryRecorder(
                game_dir / "trajectory.json",
                game_dir / "observer_views.json",
                game_id=game_id,
                run_id=run_id,
                source_commit=provenance["batch_code_commit"],
                environment_seed=seed,
                runtime_config=normalized,
                players=players,
            )
            parity_recorder = OnuwParityGameRecorder(
                game_id=game_id,
                content_profile=contract["content_profile"],
                modality_profile=contract["modality_profile"],
            )
            role_collector = build_onuw_role_guess_collector(
                agent_list=agents,
                report_audit=call_audit,
            )
            stage = "gameplay"
            result = run_game(
                env,
                agents,
                roles,
                call_audit=call_audit,
                trajectory_recorder=trajectory_recorder,
                allow_gameplay_fallback=False,
                onuw_parity_recorder=parity_recorder,
                onuw_role_guess_collector=role_collector,
            )
            call_audit.assert_wall_budget()
            parity_game = parity_recorder.finalize()
            collection_audit = parity_recorder.finalize_collection_audit()
            game_stats = pilot_collection_audit(
                [parity_game], [collection_audit]
            )
            if game_stats["declared_emotion_coverage"] != 1.0:
                raise ValueError("full multimodal pilot requires 100% emotion coverage")

            annotations = normalize_speech_annotations(
                env.speech_annotations,
                public_events=env.public_events,
                require_complete=True,
            )
            _write_json_new(game_dir / "parity_game.json", parity_game)
            _write_json_new(
                game_dir / "parity_collection_audit.json", collection_audit
            )
            _write_jsonl_new(game_dir / "speech_annotations.jsonl", annotations)
            call_record = call_audit.snapshot()
            if not call_record["within_budget"]:
                raise RuntimeError("game completed outside call budget")
            if call_record["gameplay_fallback_count"] != 0:
                raise RuntimeError("parity pilot forbids gameplay fallback")
            _write_json_new(game_dir / "call_audit.json", call_record)

            stage = "artifact_validation"
            trajectory_validation = validate_complete_game_artifacts(
                game_dir / "trajectory.json",
                game_dir / "observer_views.json",
                expected_game_id=game_id,
                expected_run_id=run_id,
                expected_seed=seed,
                expected_source_commit=provenance["batch_code_commit"],
            )
            speech_validation = validate_speech_annotation_artifact(
                game_dir / "speech_annotations.jsonl",
                game_dir / "trajectory.json",
                require_success=True,
            )
            stage = "deterministic_replay"
            replay = replay_canonical_trajectory(
                game_dir / "trajectory.json",
                game_dir / "observer_views.json",
            )
            if result != f"{trajectory_validation['winner']} win":
                raise ValueError("rollout result differs from recorded winner")
            game_summary = {
                "schema_version": PARITY_PILOT_GAME_SUMMARY_SCHEMA,
                "data_namespace": PARITY_PILOT_NAMESPACE,
                "game_id": game_id,
                "run_id": run_id,
                "environment_seed": seed,
                "source_commit": provenance["batch_code_commit"],
                "result": result,
                "trajectory_validation": trajectory_validation,
                "speech_validation": speech_validation,
                "deterministic_replay": replay,
                "call_audit_digest": call_record["audit_digest"],
                "pilot_statistics": game_stats,
                "parity_game_sha256": _sha256(game_dir / "parity_game.json"),
                "parity_collection_audit_sha256": _sha256(
                    game_dir / "parity_collection_audit.json"
                ),
            }
            game_summary["summary_digest"] = canonical_digest(game_summary)
            _write_json_new(game_dir / "summary.json", game_summary)
            completed_games.append(parity_game)
            completed_audits.append(collection_audit)
            game_summaries.append(game_summary)
        except Exception as exc:
            call_path = game_dir / "call_audit.json"
            if not call_path.exists():
                _write_json_new(call_path, call_audit.snapshot())
            failure = {
                "schema_version": PARITY_PILOT_FAILURE_SCHEMA,
                "data_namespace": PARITY_PILOT_NAMESPACE,
                "run_id": run_id,
                "game_id": game_id,
                "environment_seed": seed,
                "source_commit": provenance["batch_code_commit"],
                "failure_stage": stage,
                "exception_type": type(exc).__name__,
                "exception_message": sanitize_exception_message(exc),
            }
            failure["failure_digest"] = canonical_digest(failure)
            _write_json_new(game_dir / "failure.json", failure)
            failures_root = destination / "failures"
            failures_root.mkdir(exist_ok=True)
            game_dir.rename(failures_root / directory_name)
            failures.append(failure)
        finally:
            _close_agents(agents)

    _write_jsonl_new(destination / "parity_games.jsonl", completed_games)
    _write_jsonl_new(
        destination / "parity_collection_audits.jsonl", completed_audits
    )
    aggregate = (
        pilot_collection_audit(completed_games, completed_audits)
        if completed_games
        else None
    )
    summary = {
        "schema_version": PARITY_PILOT_SUMMARY_SCHEMA,
        "data_namespace": PARITY_PILOT_NAMESPACE,
        "run_id": run_id,
        "source_commit": provenance["batch_code_commit"],
        "plan_digest": plan["plan_digest"],
        "planned_game_count": game_count,
        "completed_game_count": len(completed_games),
        "failed_game_count": len(failures),
        "completed_game_ids": [game["game_id"] for game in completed_games],
        "failed_game_ids": [failure["game_id"] for failure in failures],
        "game_summary_digests": {
            summary["game_id"]: summary["summary_digest"]
            for summary in game_summaries
        },
        "failure_digests": {
            failure["game_id"]: failure["failure_digest"]
            for failure in failures
        },
        "pilot_statistics": aggregate,
        "status": "PASS" if len(completed_games) == game_count else "INCOMPLETE",
        "formal_training_eligible": False,
        "training_budget_frozen": False,
        "sealed_test_boundary_created": False,
    }
    summary["summary_digest"] = canonical_digest(summary)
    _write_json_new(destination / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--game-count", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = collect_onuw_parity_pilot(
        config_path=args.config,
        run_id=args.run_id,
        seed_start=args.seed_start,
        game_count=args.game_count,
        output_root=args.output_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
