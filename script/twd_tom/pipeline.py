"""Run one explicit stage of the classic-seven TWD-ToM pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from script.twd_tom import real_backend_dry_run as collection_core
from werewolf.backends import is_local_unauthenticated_backend
from werewolf.runtime_config import normalize_runtime_config
from werewolf.models.twd_tom.collector import require_clean_collection_worktree
from werewolf.models.twd_tom.public_events import PUBLIC_EVENT_SCHEMA_VERSION
from werewolf.models.twd_tom.samples import (
    ACTOR_PAIR_BELIEF_SCHEMA_VERSION,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
STAGES = ("validate", "collect")
COLLECTION_OVERRIDE_STAGES = ("validate", "collect")


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return dict(value)


def _text(mapping: Mapping[str, Any], name: str, prefix: str) -> str:
    value = mapping.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{prefix}.{name} must be non-empty text")
    return value.strip()


def _integer(mapping: Mapping[str, Any], name: str, prefix: str) -> int:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{prefix}.{name} must be an integer")
    return value


def _validated_run_id(run_id: Any) -> str:
    if (
        not isinstance(run_id, str)
        or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}",
            run_id,
        )
        is None
    ):
        raise ValueError(
            "run_id must be 1-64 ASCII letters, digits, hyphens, "
            "or underscores and must start with a letter or digit"
        )
    return run_id


def _run_paths(run_id: str) -> dict[str, dict[str, Path]]:
    logs_run = (
        REPO_ROOT / "logs" / "tom" / run_id
    ).resolve()
    data_run = (
        REPO_ROOT / "data" / "tom" / run_id
    ).resolve()
    outputs_run = (
        REPO_ROOT / "outputs" / "tom" / run_id
    ).resolve()
    return {
        "logs": {
            "run_dir": logs_run,
            "games_dir": logs_run / "games",
            "call_audit_path": logs_run / "call_audit.jsonl",
            "manifest_path": logs_run / "manifest.json",
            "resolved_config_path": logs_run / "resolved_config.yaml",
        },
        "data": {
            "run_dir": data_run,
            "raw_path": data_run / "raw.jsonl",
        },
        "outputs": {
            "run_dir": outputs_run,
        },
    }


def _validated_collection_plan(
    collection: Mapping[str, Any],
    *,
    game_count_override: int | None,
    seeds_override: Sequence[int] | None,
) -> tuple[int, tuple[int, ...]]:
    if (game_count_override is None) != (seeds_override is None):
        raise ValueError(
            "--game-count and --seeds must be provided together"
        )
    game_count = (
        _integer(
            collection,
            "game_count",
            "pipeline.collection",
        )
        if game_count_override is None
        else game_count_override
    )
    raw_seeds = (
        collection.get("seeds")
        if seeds_override is None
        else seeds_override
    )
    if (
        isinstance(game_count, bool)
        or not isinstance(game_count, int)
        or game_count <= 0
    ):
        raise ValueError("game_count must be positive")
    if isinstance(raw_seeds, (str, bytes)) or not isinstance(
        raw_seeds, Sequence
    ):
        raise TypeError("seeds must be a sequence")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int)
        for seed in raw_seeds
    ):
        raise ValueError("seeds must contain integers")
    seeds = tuple(raw_seeds)
    if len(seeds) != game_count:
        raise ValueError(
            "game_count must equal the number of seeds"
        )
    return game_count, seeds


def _reject_path_fields(
    mapping: Mapping[str, Any],
    *,
    prefix: str,
    fields: Sequence[str],
) -> None:
    present = sorted(set(mapping) & set(fields))
    if present:
        raise ValueError(
            f"{prefix} must not define run artifact paths: {present}"
        )


def _string_paths(
    paths: Mapping[str, Mapping[str, Path]],
) -> dict[str, dict[str, str]]:
    return {
        group: {
            name: str(path)
            for name, path in group_paths.items()
        }
        for group, group_paths in paths.items()
    }


def _load_pipeline_config(
    config_path: str | Path,
    *,
    run_id: str,
    game_count_override: int | None,
    seeds_override: Sequence[int] | None,
) -> dict[str, Any]:
    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"pipeline config not found: {path}")
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, Mapping):
        raise TypeError("pipeline config must be a mapping")
    normalized_runtime = normalize_runtime_config(deepcopy(parsed))

    pipeline = _mapping(parsed.get("pipeline"), "pipeline")
    expected_versions = {
        "public_event_schema_version": PUBLIC_EVENT_SCHEMA_VERSION,
        "raw_schema_version": ACTOR_PAIR_BELIEF_SCHEMA_VERSION,
    }
    for field_name, expected in expected_versions.items():
        actual = _text(pipeline, field_name, "pipeline")
        if actual != expected:
            raise ValueError(
                f"pipeline.{field_name} must equal {expected!r}"
            )
    _reject_path_fields(
        pipeline,
        prefix="pipeline",
        fields=("experiment_name", "output_root"),
    )
    paths = _run_paths(run_id)

    collection = _mapping(pipeline.get("collection"), "pipeline.collection")
    _reject_path_fields(
        collection,
        prefix="pipeline.collection",
        fields=("raw_jsonl_path", "output_dir"),
    )
    game_count, seeds = _validated_collection_plan(
        collection,
        game_count_override=game_count_override,
        seeds_override=seeds_override,
    )
    budget = collection_core.DryRunBudget(
        game_id="pipeline_validation",
        max_gameplay_calls=_integer(
            collection, "max_gameplay_calls_per_game", "pipeline.collection"
        ),
        max_belief_calls=_integer(
            collection, "max_belief_calls_per_game", "pipeline.collection"
        ),
        max_total_calls=_integer(
            collection, "max_total_calls_per_game", "pipeline.collection"
        ),
        max_wall_seconds=collection.get("max_wall_seconds_per_game"),
    )

    forbidden_stage_config = {"project", "split"} & set(pipeline)
    if forbidden_stage_config:
        raise ValueError(
            "actor-perspective phase-one pipeline cannot configure project or split"
        )

    resolved_runtime = deepcopy(parsed)
    resolved_pipeline = resolved_runtime["pipeline"]
    resolved_pipeline["collection"]["game_count"] = game_count
    resolved_pipeline["collection"]["seeds"] = list(seeds)
    resolved_pipeline["resolved_run"] = {
        "run_id": run_id,
        "paths": _string_paths(paths),
    }

    return {
        "config_path": path,
        "parsed_runtime": dict(parsed),
        "normalized_runtime": normalized_runtime,
        "resolved_runtime": resolved_runtime,
        "run_id": run_id,
        "versions": expected_versions,
        "paths": paths,
        "collection": {
            "raw_jsonl_path": paths["data"]["raw_path"],
            "game_count": game_count,
            "seeds": seeds,
            "budget": budget,
        },
    }


def _summary(
    config: Mapping[str, Any],
    *,
    stage: str,
    input_path: Any,
    output_path: Any,
    collection_git_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "stage": stage,
        "run_id": config["run_id"],
        "source_commit": (
            collection_git_state["git_commit_sha"]
            if collection_git_state is not None
            else collection_core._runtime_source_commit()
        ),
        "config_path": str(config["config_path"]),
        "game_count": config["collection"]["game_count"],
        "seeds": list(config["collection"]["seeds"]),
        "input_path": input_path,
        "output_path": output_path,
        "status": "ok",
    }
    if collection_git_state is not None:
        result.update(
            git_commit_sha=collection_git_state["git_commit_sha"],
            git_worktree_clean=collection_git_state["git_worktree_clean"],
        )
    return result


def _validate(config: Mapping[str, Any]) -> dict[str, Any]:
    summary = _summary(
        config, stage="validate", input_path=None, output_path=None
    )
    summary["plan"] = {
        "run_id": config["run_id"],
        "versions": dict(config["versions"]),
        **_string_paths(config["paths"]),
        "collection": {
            "game_count": config["collection"]["game_count"],
            "seeds": list(config["collection"]["seeds"]),
            "raw_jsonl_path": str(config["collection"]["raw_jsonl_path"]),
        },
    }
    return summary


def _atomic_yaml_dump(
    value: Mapping[str, Any],
    path: Path,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(
            f"temporary output already exists: {temporary}"
        )
    try:
        temporary.write_text(
            yaml.safe_dump(
                dict(value),
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _check_collect_api_keys(config: Mapping[str, Any]) -> None:
    load_dotenv(dotenv_path=REPO_ROOT / ".env", override=False)
    for backend_name, backend in config["normalized_runtime"]["backends"].items():
        if is_local_unauthenticated_backend(backend):
            continue
        environment_name = backend.get("api_key_env") or "OPENAI_API_KEY"
        if not os.environ.get(environment_name):
            raise ValueError(
                f"API key environment variable {environment_name} "
                f"is required for backend '{backend_name}'"
            )


def _run_collect(config: Mapping[str, Any]) -> dict[str, Any]:
    collection_git_state = require_clean_collection_worktree()
    paths = config["paths"]
    for run_directory in (
        paths["logs"]["run_dir"],
        paths["data"]["run_dir"],
        paths["outputs"]["run_dir"],
    ):
        if run_directory.exists():
            raise FileExistsError(
                f"run directory already exists: {run_directory}"
            )

    _check_collect_api_keys(config)

    collection = config["collection"]
    logs_run = paths["logs"]["run_dir"]
    data_run = paths["data"]["run_dir"]
    games_dir = paths["logs"]["games_dir"]
    audit_path = paths["logs"]["call_audit_path"]
    logs_run.mkdir(parents=True, exist_ok=False)
    data_run.mkdir(parents=True, exist_ok=False)
    games_dir.mkdir(exist_ok=False)
    _atomic_yaml_dump(
        config["resolved_runtime"],
        paths["logs"]["resolved_config_path"],
    )

    with collection_core.PrivacySafeAuditWriter(audit_path) as writer:
        for game_index, seed in enumerate(collection["seeds"], start=1):
            template = collection["budget"]
            game_id = f"game_{game_index:03d}_seed_{seed}"
            game_log_dir = games_dir / game_id
            game_log_dir.mkdir(exist_ok=False)
            budget = collection_core.DryRunBudget(
                game_id=game_id,
                max_gameplay_calls=template.max_gameplay_calls,
                max_belief_calls=template.max_belief_calls,
                max_total_calls=template.max_total_calls,
                max_wall_seconds=template.max_wall_seconds,
            )
            collection_core.run_real_backend_game(
                parsed_yaml=config["parsed_runtime"],
                samples_path=collection["raw_jsonl_path"],
                log_dir=game_log_dir,
                game_id=game_id,
                seed=seed,
                budget=budget,
                writer=writer,
                source_config_path=config["config_path"],
            )

    summary = _summary(
        config,
        stage="collect",
        input_path=None,
        output_path=str(collection["raw_jsonl_path"]),
        collection_git_state=collection_git_state,
    )
    summary["game_count"] = collection["game_count"]
    summary["seeds"] = list(collection["seeds"])
    summary["call_audit_path"] = str(audit_path)
    summary["games_dir"] = str(games_dir)
    summary["resolved_config_path"] = str(
        paths["logs"]["resolved_config_path"]
    )
    collection_core._atomic_json_dump(
        {
            **summary,
            "paths": _string_paths(paths),
        },
        paths["logs"]["manifest_path"],
    )
    summary["manifest_path"] = str(
        paths["logs"]["manifest_path"]
    )
    return summary


def run_pipeline_stage(
    *,
    config_path: str | Path,
    run_id: str,
    stage: str,
    game_count: int | None = None,
    seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"unsupported pipeline stage: {stage}")
    validated_run_id = _validated_run_id(run_id)
    if (
        game_count is not None or seeds is not None
    ) and stage not in COLLECTION_OVERRIDE_STAGES:
        raise ValueError(
            "--game-count and --seeds are only allowed for "
            "validate and collect"
        )
    config = _load_pipeline_config(
        config_path,
        run_id=validated_run_id,
        game_count_override=game_count,
        seeds_override=seeds,
    )

    if stage == "validate":
        return _validate(config)
    if stage == "collect":
        return _run_collect(config)
    raise AssertionError(f"unhandled pipeline stage: {stage}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--game-count", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = run_pipeline_stage(
        config_path=args.config,
        run_id=args.run_id,
        stage=args.stage,
        game_count=args.game_count,
        seeds=args.seeds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
