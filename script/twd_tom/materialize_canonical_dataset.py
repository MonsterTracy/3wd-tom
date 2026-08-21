"""Build one canonical offline ToM dataset from frozen A/C0 games."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from werewolf.backends import load_named_backends, resolve_backend
from werewolf.offline_annotation import (
    OFFLINE_ANNOTATION_SCHEMA_VERSION,
    PRIVATE_CONDITIONED_SUSPICION_TASK,
    PUBLIC_ONLY_SUSPICION_TASK,
    annotate_pre_speech_suspicion,
    validate_offline_annotation_record,
    validate_offline_annotation_sources,
    write_annotation_jsonl,
)
from werewolf.offline_materialization import (
    D_MATERIALIZATION_POLICY_VERSION,
    D_SCHEMA_VERSION,
    OFFLINE_PRIVATE_CONDITIONED_TOM1_TASK,
    OFFLINE_PUBLIC_ONLY_TOM2_TASK,
    materialize_offline_tom_records,
    validate_offline_tom_training_record,
    write_offline_tom_jsonl,
)
from werewolf.trajectory import canonical_digest, canonical_json


CANONICAL_DATASET_MANIFEST_SCHEMA_VERSION = (
    "classic7_canonical_offline_tom_dataset_manifest_v1"
)
REPO_ROOT = Path(__file__).resolve().parents[2]

PRIVATE_ANNOTATION_PATH = Path("annotations/private_conditioned.jsonl")
PUBLIC_ANNOTATION_PATH = Path("annotations/public_only.jsonl")
TOM1_PATH = Path("tom1.jsonl")
TOM2_PATH = Path("tom2.jsonl")
MANIFEST_PATH = Path("manifest.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"canonical artifact not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"canonical artifact is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"canonical artifact must be a JSON object: {path}")
    return value


def _load_games(canonical_root: Path) -> list[dict[str, Any]]:
    if not canonical_root.is_dir():
        raise NotADirectoryError(
            f"canonical root is not a directory: {canonical_root}"
        )
    game_directories = sorted(path for path in canonical_root.iterdir() if path.is_dir())
    if not game_directories:
        raise ValueError("canonical root contains no game directories")

    games = []
    seen_game_ids = set()
    for game_directory in game_directories:
        trajectory_path = game_directory / "trajectory.json"
        provenance_path = game_directory / "observer_views.json"
        trajectory = _load_json_object(trajectory_path)
        provenance = _load_json_object(provenance_path)
        validate_offline_annotation_sources(trajectory, provenance)
        game_id = trajectory.get("game_id")
        if not isinstance(game_id, str) or not game_id:
            raise ValueError("canonical trajectory requires a non-empty game_id")
        if game_id in seen_game_ids:
            raise ValueError(f"duplicate canonical game_id: {game_id}")
        seen_game_ids.add(game_id)
        games.append(
            {
                "game_id": game_id,
                "trajectory": trajectory,
                "provenance": provenance,
                "trajectory_sha256": _sha256(trajectory_path),
                "observer_views_sha256": _sha256(provenance_path),
            }
        )
    return sorted(games, key=lambda game: game["game_id"])


def _read_validated_jsonl(
    path: Path,
    validator: Callable[[Mapping[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}: blank JSONL line at {line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: invalid JSON on line {line_number}"
                ) from exc
            records.append(validator(value))
    if not records:
        raise ValueError(f"canonical output JSONL is empty: {path}")
    return records


def _file_summary(
    path: Path,
    relative_path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    schema_version: str,
    task_field: str,
    task: str,
) -> dict[str, Any]:
    if {record.get(task_field) for record in records} != {task}:
        raise ValueError(f"{relative_path} records do not match task {task}")
    return {
        "path": relative_path.as_posix(),
        "schema_version": schema_version,
        "task": task,
        "row_count": len(records),
        "sha256": _sha256(path),
    }


def materialize_canonical_dataset(
    *,
    canonical_root: str | Path,
    output_dir: str | Path,
    annotation_run_id: str,
    code_commit: str,
    backend,
    backend_id: str,
    model_name: str,
) -> dict[str, Any]:
    """Create one immutable C1/D dataset directory from canonical games."""

    source = Path(canonical_root).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    games = _load_games(source)

    private_annotations = []
    public_annotations = []
    tom1_records = []
    tom2_records = []
    game_summaries = []
    for game in games:
        trajectory = game["trajectory"]
        provenance = game["provenance"]
        private = annotate_pre_speech_suspicion(
            trajectory,
            provenance,
            annotation_task=PRIVATE_CONDITIONED_SUSPICION_TASK,
            annotation_run_id=annotation_run_id,
            annotation_code_commit=code_commit,
            backend=backend,
            backend_id=backend_id,
            model_name=model_name,
        )
        public = annotate_pre_speech_suspicion(
            trajectory,
            provenance,
            annotation_task=PUBLIC_ONLY_SUSPICION_TASK,
            annotation_run_id=annotation_run_id,
            annotation_code_commit=code_commit,
            backend=backend,
            backend_id=backend_id,
            model_name=model_name,
        )
        materialized = materialize_offline_tom_records(
            trajectory,
            provenance,
            private + public,
            materializer_code_commit=code_commit,
        )
        private_annotations.extend(private)
        public_annotations.extend(public)
        tom1_records.extend(materialized["tom1_records"])
        tom2_records.extend(materialized["tom2_records"])
        game_summaries.append(
            {
                "game_id": game["game_id"],
                **materialized["summary"],
            }
        )

    private_annotations.sort(
        key=lambda record: (
            record["step_idx"],
            record["observer_id"],
            record["game_id"],
            record["boundary_id"],
        )
    )
    public_annotations.sort(
        key=lambda record: (
            record["step_idx"],
            record["observer_id"],
            record["game_id"],
            record["boundary_id"],
        )
    )
    tom1_records.sort(key=lambda record: (record["game_id"], record["step_idx"]))
    tom2_records.sort(key=lambda record: (record["game_id"], record["step_idx"]))

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
        )
    )
    try:
        private_path = temporary / PRIVATE_ANNOTATION_PATH
        public_path = temporary / PUBLIC_ANNOTATION_PATH
        tom1_path = temporary / TOM1_PATH
        tom2_path = temporary / TOM2_PATH
        write_annotation_jsonl(private_path, private_annotations)
        write_annotation_jsonl(public_path, public_annotations)
        write_offline_tom_jsonl(tom1_path, tom1_records)
        write_offline_tom_jsonl(tom2_path, tom2_records)

        reread_private = _read_validated_jsonl(
            private_path,
            validate_offline_annotation_record,
        )
        reread_public = _read_validated_jsonl(
            public_path,
            validate_offline_annotation_record,
        )
        reread_tom1 = _read_validated_jsonl(
            tom1_path,
            validate_offline_tom_training_record,
        )
        reread_tom2 = _read_validated_jsonl(
            tom2_path,
            validate_offline_tom_training_record,
        )

        source_games = [
            {
                "game_id": game["game_id"],
                "source_commit": game["trajectory"]["source_commit"],
                "environment_seed": game["trajectory"]["environment_seed"],
                "trajectory_digest": game["trajectory"]["trajectory_digest"],
                "observer_view_artifact_digest": game["provenance"][
                    "artifact_digest"
                ],
                "trajectory_file_sha256": game["trajectory_sha256"],
                "observer_views_file_sha256": game["observer_views_sha256"],
            }
            for game in games
        ]
        manifest = {
            "schema_version": CANONICAL_DATASET_MANIFEST_SCHEMA_VERSION,
            "annotation_code_commit": code_commit,
            "materializer_code_commit": code_commit,
            "annotation_run_id": annotation_run_id,
            "annotation_backend_id": backend_id,
            "annotation_model_id": model_name,
            "annotation_schema_version": OFFLINE_ANNOTATION_SCHEMA_VERSION,
            "d_schema_version": D_SCHEMA_VERSION,
            "d_materialization_policy_version": D_MATERIALIZATION_POLICY_VERSION,
            "source_game_count": len(games),
            "game_ids": [game["game_id"] for game in games],
            "source_games": source_games,
            "files": {
                "private_annotations": _file_summary(
                    private_path,
                    PRIVATE_ANNOTATION_PATH,
                    reread_private,
                    schema_version=OFFLINE_ANNOTATION_SCHEMA_VERSION,
                    task_field="annotation_task",
                    task=PRIVATE_CONDITIONED_SUSPICION_TASK,
                ),
                "public_annotations": _file_summary(
                    public_path,
                    PUBLIC_ANNOTATION_PATH,
                    reread_public,
                    schema_version=OFFLINE_ANNOTATION_SCHEMA_VERSION,
                    task_field="annotation_task",
                    task=PUBLIC_ONLY_SUSPICION_TASK,
                ),
                "tom1": _file_summary(
                    tom1_path,
                    TOM1_PATH,
                    reread_tom1,
                    schema_version=D_SCHEMA_VERSION,
                    task_field="materialization_task",
                    task=OFFLINE_PRIVATE_CONDITIONED_TOM1_TASK,
                ),
                "tom2": _file_summary(
                    tom2_path,
                    TOM2_PATH,
                    reread_tom2,
                    schema_version=D_SCHEMA_VERSION,
                    task_field="materialization_task",
                    task=OFFLINE_PUBLIC_ONLY_TOM2_TASK,
                ),
            },
            "game_materialization_summaries": game_summaries,
        }
        manifest["manifest_digest"] = canonical_digest(manifest)
        (temporary / MANIFEST_PATH).write_text(
            f"{canonical_json(manifest)}\n",
            encoding="utf-8",
        )
        if destination.exists():
            raise FileExistsError(f"output directory already exists: {destination}")
        temporary.rename(destination)
        return manifest
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _code_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot resolve materializer code commit") from exc
    commit = result.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError("materializer code commit must be a lowercase Git SHA")
    return commit


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize canonical offline C1/D ToM data from A/C0 games."
    )
    parser.add_argument("--canonical-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--annotation-run-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--backend-id", required=True)
    parser.add_argument("--model-name", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"backend config not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise TypeError("backend config must be a mapping")
    backends = load_named_backends(
        config,
        env_file=REPO_ROOT / ".env",
        max_retries=0,
    )
    backend = resolve_backend(args.backend_id, backends)
    manifest = materialize_canonical_dataset(
        canonical_root=args.canonical_root,
        output_dir=args.output_dir,
        annotation_run_id=args.annotation_run_id,
        code_commit=_code_commit(),
        backend=backend,
        backend_id=args.backend_id,
        model_name=args.model_name,
    )
    print(canonical_json(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
