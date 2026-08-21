"""Materialize deterministic PBM targets into explicit game-level splits."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from werewolf.models.public_belief_matrix.collection import (
    PUBLIC_BELIEF_MATRIX_MAX_SEQ_LEN,
    PUBLIC_BELIEF_MATRIX_SAMPLE_SCHEMA_VERSION,
    validate_public_belief_matrix_sample,
)
from werewolf.models.public_belief_matrix.dataset import (
    PUBLIC_BELIEF_MATRIX_MATERIALIZATION_VERSION,
    seed_from_formal_game_id,
)
from werewolf.models.public_belief_matrix.targets import (
    suspicion_reports_to_matrix_target,
)


SPLIT_NAMES = ("train", "validation", "test")


def _git_commit() -> str:
    repository_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot resolve the current repository commit") from exc
    commit = result.stdout.strip()
    if re.fullmatch(r"[0-9a-fA-F]{40}", commit) is None:
        raise RuntimeError("current commit must be a 40-character SHA")
    return commit


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"JSON document must be a mapping: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"line {line_number}: blank JSONL rows are not allowed")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise TypeError(f"line {line_number}: sample must be a mapping")
            records.append(value)
    return records


def _validated_seed_tuple(values: Sequence[int], *, split_name: str) -> tuple[int, ...]:
    seeds = tuple(values)
    if not seeds:
        raise ValueError(f"{split_name} seeds cannot be empty")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise TypeError(f"{split_name} seeds must be integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{split_name} seeds contain duplicates")
    return seeds


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_training_data(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    train_seeds: Sequence[int],
    validation_seeds: Sequence[int],
    test_seeds: Sequence[int],
) -> dict[str, Any]:
    """Validate one formal raw batch and materialize explicit seed splits."""

    source = Path(input_path).resolve()
    destination = Path(output_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"input file not found: {source}")
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    source_manifest_path = source.parent / "formal_batch_manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"source manifest not found: {source_manifest_path}")

    split_seeds = {
        "train": _validated_seed_tuple(train_seeds, split_name="train"),
        "validation": _validated_seed_tuple(
            validation_seeds, split_name="validation"
        ),
        "test": _validated_seed_tuple(test_seeds, split_name="test"),
    }
    seed_sets = {name: set(seeds) for name, seeds in split_seeds.items()}
    if any(
        seed_sets[left] & seed_sets[right]
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
    ):
        raise ValueError("train/validation/test seed assignments overlap")
    requested_seeds = set().union(*seed_sets.values())

    source_manifest = _read_json(source_manifest_path)
    if source_manifest.get("schema_version") != PUBLIC_BELIEF_MATRIX_SAMPLE_SCHEMA_VERSION:
        raise ValueError("source manifest schema is not PBM symbolic V1")
    manifest_seeds = source_manifest.get("seeds")
    if not isinstance(manifest_seeds, list) or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in manifest_seeds
    ):
        raise ValueError("source manifest seeds are invalid")
    if len(set(manifest_seeds)) != len(manifest_seeds):
        raise ValueError("source manifest seeds contain duplicates")
    if requested_seeds != set(manifest_seeds):
        missing = sorted(requested_seeds - set(manifest_seeds))
        unspecified = sorted(set(manifest_seeds) - requested_seeds)
        raise ValueError(
            f"split seeds must exactly cover source seeds; missing={missing}, "
            f"unspecified={unspecified}"
        )
    source_commit = source_manifest.get("source_commit")
    if not isinstance(source_commit, str) or re.fullmatch(
        r"[0-9a-fA-F]{40}", source_commit
    ) is None:
        raise ValueError("source manifest commit is invalid")

    samples = _read_jsonl(source)
    if not samples:
        raise ValueError("source sample file is empty")
    seen_snapshots: set[str] = set()
    game_to_seed: dict[str, int] = {}
    materialized_by_split: dict[str, list[dict[str, Any]]] = {
        name: [] for name in SPLIT_NAMES
    }
    for line_number, sample in enumerate(samples, start=1):
        try:
            validate_public_belief_matrix_sample(sample)
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"line {line_number}: {exc}") from exc
        snapshot_id = sample.get("snapshot_id")
        if not isinstance(snapshot_id, str) or not snapshot_id:
            raise ValueError(f"line {line_number}: snapshot_id must be non-empty text")
        if snapshot_id in seen_snapshots:
            raise ValueError(f"duplicate snapshot_id: {snapshot_id}")
        seen_snapshots.add(snapshot_id)
        game_id = sample.get("game_id")
        seed = seed_from_formal_game_id(game_id)
        previous_seed = game_to_seed.setdefault(game_id, seed)
        if previous_seed != seed:
            raise ValueError(f"game_id has inconsistent seeds: {game_id}")
        split_matches = [name for name, seeds in seed_sets.items() if seed in seeds]
        if len(split_matches) != 1:
            raise ValueError(f"sample seed is not assigned to exactly one split: {seed}")
        target = suspicion_reports_to_matrix_target(sample["observer_reports"])
        if not any(target.observer_row_mask):
            raise ValueError(f"snapshot has no valid observer rows: {snapshot_id}")
        materialized_by_split[split_matches[0]].append(
            {
                "materialization_version": PUBLIC_BELIEF_MATRIX_MATERIALIZATION_VERSION,
                "source_schema_version": PUBLIC_BELIEF_MATRIX_SAMPLE_SCHEMA_VERSION,
                "seed": seed,
                "sample": sample,
                "matrix_target": [list(row) for row in target.matrix_target],
                "observer_row_mask": list(target.observer_row_mask),
            }
        )

    if set(game_to_seed.values()) != requested_seeds:
        missing = sorted(requested_seeds - set(game_to_seed.values()))
        raise ValueError(f"requested seeds have no source game: {missing}")
    if len(game_to_seed) != len(requested_seeds):
        raise ValueError("source must contain exactly one game per requested seed")
    expected_game_count = source_manifest.get("requested_game_count")
    completed_game_count = source_manifest.get("completed_game_count")
    if expected_game_count != len(game_to_seed) or completed_game_count != len(game_to_seed):
        raise ValueError(
            "source manifest expected/completed game count does not match source games"
        )

    split_game_ids = {
        name: sorted(
            game_id for game_id, seed in game_to_seed.items() if seed in seed_sets[name]
        )
        for name in SPLIT_NAMES
    }
    split_game_sets = {name: set(ids) for name, ids in split_game_ids.items()}
    if any(
        split_game_sets[left] & split_game_sets[right]
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
    ):
        raise RuntimeError("game IDs overlap across splits")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        split_summaries = {}
        for split_name in SPLIT_NAMES:
            _write_jsonl(
                temporary / f"{split_name}.jsonl",
                materialized_by_split[split_name],
            )
            split_summaries[split_name] = {
                "seeds": list(split_seeds[split_name]),
                "game_ids": split_game_ids[split_name],
                "game_count": len(split_game_ids[split_name]),
                "sample_count": len(materialized_by_split[split_name]),
            }
        manifest = {
            "source_schema_version": PUBLIC_BELIEF_MATRIX_SAMPLE_SCHEMA_VERSION,
            "target_materialization_version": PUBLIC_BELIEF_MATRIX_MATERIALIZATION_VERSION,
            "source_file": str(source),
            "source_manifest": str(source_manifest_path.resolve()),
            "source_commit": source_commit,
            "current_commit": _git_commit(),
            "max_seq_len": PUBLIC_BELIEF_MATRIX_MAX_SEQ_LEN,
            "total_game_count": len(game_to_seed),
            "total_sample_count": len(samples),
            "game_level_split": True,
            "split_overlap": False,
            "splits": split_summaries,
        }
        _write_json(temporary / "manifest.json", manifest)
        if destination.exists():
            raise FileExistsError(f"output directory already exists: {destination}")
        temporary.replace(destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-seeds", required=True, type=int, nargs="+")
    parser.add_argument("--validation-seeds", required=True, type=int, nargs="+")
    parser.add_argument("--test-seeds", required=True, type=int, nargs="+")
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_argument_parser().parse_args(argv)
    manifest = build_training_data(
        input_path=args.input,
        output_dir=args.output_dir,
        train_seeds=args.train_seeds,
        validation_seeds=args.validation_seeds,
        test_seeds=args.test_seeds,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return manifest


if __name__ == "__main__":
    main()
