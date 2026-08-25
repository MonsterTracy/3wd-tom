"""Materialize canonical tom-v2 belief snapshots with a game-level split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from script.twd_tom.collect_canonical_trajectories import (
    validate_canonical_belief_batch,
)
from werewolf.models.twd_tom.dataset import TWDToMDataset, load_twd_tom_jsonl
from werewolf.models.twd_tom.samples import SAMPLE_SCHEMA_VERSION
from werewolf.trajectory import canonical_digest, canonical_json


SPLIT_POLICY_VERSION = "classic7_tom_v2_game_hash_split_v1"
SPLIT_MANIFEST_SCHEMA_VERSION = "classic7_tom_v2_split_manifest_v1"
BELIEF_SNAPSHOTS_FILENAME = "belief_snapshots.jsonl"
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "raw_schema_version",
        "canonical_batch_summary_digest",
        "canonical_batch_summary_sha256",
        "game_summary_digests",
        "split_policy_version",
        "split_seed",
        "game_ids",
        "game_counts",
        "row_counts",
        "output_files",
        "manifest_digest",
    }
)


def _non_negative_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _rank_game_ids(game_ids: list[str], *, split_seed: int) -> list[str]:
    split_seed = _non_negative_integer(split_seed, field_name="split_seed")

    def rank(game_id: str) -> tuple[str, str]:
        payload = f"{SPLIT_POLICY_VERSION}\0{split_seed}\0{game_id}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest(), game_id

    return sorted(game_ids, key=rank)


def _load_canonical_games(
    canonical_root: Path,
    verified_batch: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    games: dict[str, list[dict[str, Any]]] = {}
    for verified_game in verified_batch["games"]:
        path = canonical_root / verified_game["relative_path"]
        records = load_twd_tom_jsonl(path)
        if not records:
            raise ValueError(f"canonical belief snapshot file cannot be empty: {path}")
        if len(records) != verified_game["belief_snapshot_count"]:
            raise ValueError(
                "canonical belief snapshot count differs from game summary: "
                f"{verified_game['game_id']}"
            )
        game_ids = {record.get("game_id") for record in records}
        if len(game_ids) != 1:
            raise ValueError(f"one canonical game file must contain exactly one game_id: {path}")
        game_id = next(iter(game_ids))
        if not isinstance(game_id, str) or not game_id.strip():
            raise ValueError(f"canonical game_id must be non-empty text: {path}")
        if game_id != verified_game["game_id"]:
            raise ValueError("canonical belief file game_id differs from game summary")
        if game_id in games:
            raise ValueError(f"duplicate canonical game_id: {game_id}")
        step_indices = [record.get("step_idx") for record in records]
        if any(isinstance(step, bool) or not isinstance(step, int) for step in step_indices):
            raise TypeError(f"every canonical step_idx must be an integer: {path}")
        if len(step_indices) != len(set(step_indices)):
            raise ValueError(f"duplicate canonical (game_id, step_idx): {game_id}")
        records = sorted(records, key=lambda record: record["step_idx"])
        TWDToMDataset(records)
        games[game_id] = records
    return games


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(canonical_json(value) + "\n")


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"split manifest not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("split manifest must contain one JSON object")
    return value


def _require_sha256(value: Any, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be lowercase SHA-256 text")
    return value


def _stream_jsonl_identity(path: Path) -> tuple[int, set[str]]:
    row_count = 0
    game_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid split JSON on line {line_number}: {path}"
                ) from exc
            if not isinstance(record, Mapping):
                raise TypeError(f"split JSONL line must contain an object: {path}")
            game_id = record.get("game_id")
            if not isinstance(game_id, str) or not game_id.strip():
                raise ValueError(f"split JSONL record has no valid game_id: {path}")
            row_count += 1
            game_ids.add(game_id)
    return row_count, game_ids


def validate_split_manifest(
    manifest_path: str | Path,
    *,
    verify_split_files: tuple[str, ...] = SPLIT_NAMES,
) -> dict[str, Any]:
    """Validate materialized lineage and the selected physical split files."""

    path = Path(manifest_path).resolve()
    if (
        not isinstance(verify_split_files, tuple)
        or not set(verify_split_files).issubset(SPLIT_NAMES)
        or len(verify_split_files) != len(set(verify_split_files))
    ):
        raise ValueError("verify_split_files must be unique declared split names")
    manifest = _load_json_object(path)
    if set(manifest) != SPLIT_MANIFEST_FIELDS:
        raise ValueError("split manifest field set mismatch")
    if manifest["schema_version"] != SPLIT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("split manifest schema version mismatch")
    if manifest["raw_schema_version"] != SAMPLE_SCHEMA_VERSION:
        raise ValueError("split manifest raw schema version mismatch")
    if manifest["split_policy_version"] != SPLIT_POLICY_VERSION:
        raise ValueError("split manifest policy version mismatch")
    _non_negative_integer(manifest["split_seed"], field_name="split_seed")

    payload = dict(manifest)
    manifest_digest = payload.pop("manifest_digest")
    _require_sha256(manifest_digest, field_name="manifest_digest")
    if manifest_digest != canonical_digest(payload):
        raise ValueError("split manifest digest mismatch")
    _require_sha256(
        manifest["canonical_batch_summary_digest"],
        field_name="canonical_batch_summary_digest",
    )
    _require_sha256(
        manifest["canonical_batch_summary_sha256"],
        field_name="canonical_batch_summary_sha256",
    )

    for field_name in ("game_ids", "game_counts", "row_counts", "output_files"):
        value = manifest[field_name]
        if not isinstance(value, Mapping) or set(value) != set(SPLIT_NAMES):
            raise ValueError(f"split manifest {field_name} split set mismatch")

    all_game_ids: set[str] = set()
    normalized_game_ids: dict[str, list[str]] = {}
    for split_name in SPLIT_NAMES:
        game_ids = manifest["game_ids"][split_name]
        if not isinstance(game_ids, list) or not game_ids:
            raise ValueError(f"split manifest {split_name} game_ids must be non-empty")
        if any(not isinstance(game_id, str) or not game_id.strip() for game_id in game_ids):
            raise ValueError("split manifest game_ids must be non-empty text")
        if len(game_ids) != len(set(game_ids)):
            raise ValueError(f"split manifest {split_name} contains duplicate game_ids")
        overlap = all_game_ids & set(game_ids)
        if overlap:
            raise ValueError(
                "split manifest game_ids overlap across splits: "
                f"{sorted(overlap)[:10]}"
            )
        all_game_ids.update(game_ids)
        normalized_game_ids[split_name] = list(game_ids)
        if manifest["game_counts"][split_name] != len(game_ids):
            raise ValueError(f"split manifest {split_name} game count mismatch")
        _non_negative_integer(
            manifest["row_counts"][split_name],
            field_name=f"{split_name}_row_count",
        )

    summary_digests = manifest["game_summary_digests"]
    if not isinstance(summary_digests, Mapping) or set(summary_digests) != all_game_ids:
        raise ValueError("split manifest game summary digest set mismatch")
    for game_id, digest in summary_digests.items():
        _require_sha256(digest, field_name=f"game_summary_digests[{game_id}]")

    for split_name in SPLIT_NAMES:
        descriptor = manifest["output_files"][split_name]
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "relative_path",
            "sha256",
        }:
            raise ValueError(f"split manifest {split_name} output descriptor mismatch")
        relative_path = descriptor["relative_path"]
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or Path(relative_path).is_absolute()
            or len(Path(relative_path).parts) != 1
        ):
            raise ValueError("split output paths must be direct relative filenames")
        output_path = path.parent / relative_path
        expected_sha256 = _require_sha256(
            descriptor["sha256"],
            field_name=f"output_files[{split_name}].sha256",
        )
        if split_name not in verify_split_files:
            continue
        if _sha256(output_path) != expected_sha256:
            raise ValueError(f"split manifest {split_name} output SHA-256 mismatch")
        row_count, output_game_ids = _stream_jsonl_identity(output_path)
        if row_count != manifest["row_counts"][split_name]:
            raise ValueError(f"split manifest {split_name} row count mismatch")
        if output_game_ids != set(normalized_game_ids[split_name]):
            raise ValueError(f"split manifest {split_name} game IDs differ from data")

    return manifest


def validate_materialized_split_path(
    dataset_path: str | Path,
    *,
    split_name: str,
) -> dict[str, Any]:
    """Require one Dataset path to be the named file in its split manifest."""

    if split_name not in SPLIT_NAMES:
        raise ValueError(f"split_name must be one of {SPLIT_NAMES}")
    path = Path(dataset_path).resolve()
    manifest = validate_split_manifest(path.parent / "split_manifest.json")
    expected_path = (
        path.parent / manifest["output_files"][split_name]["relative_path"]
    ).resolve()
    if path != expected_path:
        raise ValueError(f"dataset path is not the manifest {split_name} split")
    return manifest


def materialize_canonical_belief_dataset(
    *,
    canonical_root: str | Path,
    output_dir: str | Path,
    split_seed: int,
    train_game_count: int,
    validation_game_count: int,
    test_game_count: int,
) -> dict[str, Any]:
    """Publish unchanged raw snapshots into deterministic game-level splits."""

    counts = {
        "train": _non_negative_integer(
            train_game_count, field_name="train_game_count"
        ),
        "validation": _non_negative_integer(
            validation_game_count, field_name="validation_game_count"
        ),
        "test": _non_negative_integer(test_game_count, field_name="test_game_count"),
    }
    if any(count == 0 for count in counts.values()):
        raise ValueError("train, validation, and test must each contain at least one game")

    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    canonical_root = Path(canonical_root).resolve()
    verified_batch = validate_canonical_belief_batch(canonical_root)
    games = _load_canonical_games(canonical_root, verified_batch)
    if sum(counts.values()) != len(games):
        raise ValueError("split game counts must sum exactly to the canonical game count")

    ranked_game_ids = _rank_game_ids(list(games), split_seed=split_seed)
    split_game_ids: dict[str, list[str]] = {}
    cursor = 0
    for split_name in SPLIT_NAMES:
        next_cursor = cursor + counts[split_name]
        split_game_ids[split_name] = ranked_game_ids[cursor:next_cursor]
        cursor = next_cursor

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        row_counts: dict[str, int] = {}
        output_files: dict[str, dict[str, Any]] = {}
        for split_name in SPLIT_NAMES:
            records = [
                record
                for game_id in split_game_ids[split_name]
                for record in games[game_id]
            ]
            records.sort(key=lambda record: (record["game_id"], record["step_idx"]))
            output_path = temporary / f"{split_name}.jsonl"
            _write_jsonl(output_path, records)
            reread = load_twd_tom_jsonl(output_path)
            if reread != records:
                raise RuntimeError(f"{split_name} output changed canonical records")
            TWDToMDataset(reread)
            row_counts[split_name] = len(records)
            output_files[split_name] = {
                "relative_path": output_path.name,
                "sha256": _sha256(output_path),
            }
        manifest = {
            "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
            "raw_schema_version": SAMPLE_SCHEMA_VERSION,
            "canonical_batch_summary_digest": verified_batch[
                "batch_summary_digest"
            ],
            "canonical_batch_summary_sha256": verified_batch[
                "batch_summary_sha256"
            ],
            "game_summary_digests": {
                game["game_id"]: game["game_summary_digest"]
                for game in verified_batch["games"]
            },
            "split_policy_version": SPLIT_POLICY_VERSION,
            "split_seed": split_seed,
            "game_ids": split_game_ids,
            "game_counts": counts,
            "row_counts": row_counts,
            "output_files": output_files,
        }
        manifest["manifest_digest"] = canonical_digest(manifest)
        _write_json(temporary / "split_manifest.json", manifest)
        validate_split_manifest(temporary / "split_manifest.json")
        if destination.exists():
            raise FileExistsError(f"output directory already exists: {destination}")
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize canonical tom-v2 belief snapshots by game."
    )
    parser.add_argument("--canonical-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-seed", required=True, type=int)
    parser.add_argument("--train-game-count", required=True, type=int)
    parser.add_argument("--validation-game-count", required=True, type=int)
    parser.add_argument("--test-game-count", required=True, type=int)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = materialize_canonical_belief_dataset(
        canonical_root=args.canonical_root,
        output_dir=args.output_dir,
        split_seed=args.split_seed,
        train_game_count=args.train_game_count,
        validation_game_count=args.validation_game_count,
        test_game_count=args.test_game_count,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
