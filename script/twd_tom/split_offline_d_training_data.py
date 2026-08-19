"""Split canonical D V1 ToM1/ToM2 records by deterministic game hash."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from werewolf.offline_materialization import (
    D_MATERIALIZATION_POLICY_VERSION,
    D_SCHEMA_VERSION,
    OFFLINE_PRIVATE_CONDITIONED_TOM1_TASK,
    OFFLINE_PUBLIC_ONLY_TOM2_TASK,
    validate_offline_tom_training_record,
)
from werewolf.trajectory import canonical_digest, canonical_json


SPLIT_MANIFEST_SCHEMA_VERSION = (
    "classic7_offline_d_training_split_manifest_v1"
)
SPLIT_POLICY_VERSION = "classic7_d_game_hash_split_v1"
SPLIT_NAMES = ("train", "validation", "test")

_GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_code_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot resolve the splitter code commit") from exc
    commit = result.stdout.strip()
    if _GIT_SHA_PATTERN.fullmatch(commit) is None:
        raise RuntimeError("splitter code commit must be a lowercase Git SHA")
    return commit


def _positive_game_count(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _load_d_jsonl(
    path: Path,
    *,
    expected_task: str,
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"input file not found: {path}")

    records: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int]] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}: blank JSONL line at {line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: invalid JSON on line {line_number}: {exc}"
                ) from exc
            try:
                record = validate_offline_tom_training_record(value)
            except (TypeError, ValueError) as exc:
                raise type(exc)(f"{path}: line {line_number}: {exc}") from exc
            if record["materialization_task"] != expected_task:
                raise ValueError(
                    f"{path}: line {line_number}: wrong materialization_task; "
                    f"expected {expected_task}"
                )
            key = (record["game_id"], record["step_idx"])
            if key in seen_keys:
                raise ValueError(
                    f"{path}: duplicate (game_id, step_idx): {key}"
                )
            seen_keys.add(key)
            records.append(record)

    if not records:
        raise ValueError(f"input dataset is empty: {path}")
    return records


def _game_hash(game_id: str, *, split_seed: int) -> str:
    payload = (
        f"{SPLIT_POLICY_VERSION}\0{split_seed}\0{game_id}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _assign_games(
    game_ids: set[str],
    *,
    split_seed: int,
    game_counts: Mapping[str, int],
) -> dict[str, list[str]]:
    ranked = sorted(
        game_ids,
        key=lambda game_id: (
            _game_hash(game_id, split_seed=split_seed),
            game_id,
        ),
    )
    train_end = game_counts["train"]
    validation_end = train_end + game_counts["validation"]
    return {
        "train": ranked[:train_end],
        "validation": ranked[train_end:validation_end],
        "test": ranked[validation_end:],
    }


def _write_jsonl(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(f"{canonical_json(record)}\n")
        handle.flush()
        os.fsync(handle.fileno())


def _source_summary(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    materialization_task: str,
) -> dict[str, Any]:
    return {
        "sha256": _sha256(path),
        "row_count": len(records),
        "game_count": len({record["game_id"] for record in records}),
        "materialization_task": materialization_task,
        "materializer_code_commits": sorted(
            {record["materializer_code_commit"] for record in records}
        ),
    }


def split_offline_d_training_data(
    *,
    tom1_path: str | Path,
    tom2_path: str | Path,
    output_dir: str | Path,
    split_seed: int,
    train_game_count: int,
    validation_game_count: int,
    test_game_count: int,
) -> dict[str, Any]:
    """Create one provenance-complete D V1 game-level split directory."""

    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise ValueError("split_seed must be an integer")
    game_counts = {
        "train": _positive_game_count(
            train_game_count,
            field_name="train_game_count",
        ),
        "validation": _positive_game_count(
            validation_game_count,
            field_name="validation_game_count",
        ),
        "test": _positive_game_count(
            test_game_count,
            field_name="test_game_count",
        ),
    }

    tom1_source = Path(tom1_path).resolve()
    tom2_source = Path(tom2_path).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(
            f"output directory already exists: {destination}"
        )

    tom1_records = _load_d_jsonl(
        tom1_source,
        expected_task=OFFLINE_PRIVATE_CONDITIONED_TOM1_TASK,
    )
    tom2_records = _load_d_jsonl(
        tom2_source,
        expected_task=OFFLINE_PUBLIC_ONLY_TOM2_TASK,
    )
    tom1_game_ids = {record["game_id"] for record in tom1_records}
    tom2_game_ids = {record["game_id"] for record in tom2_records}
    if tom1_game_ids != tom2_game_ids:
        raise ValueError(
            "tom1 and tom2 game_id sets differ; "
            f"only_tom1={sorted(tom1_game_ids - tom2_game_ids)}, "
            f"only_tom2={sorted(tom2_game_ids - tom1_game_ids)}"
        )
    requested_total = sum(game_counts.values())
    if requested_total != len(tom1_game_ids):
        raise ValueError(
            "split game counts must sum to the unique game count; "
            f"requested={requested_total}, actual={len(tom1_game_ids)}"
        )

    split_game_ids = _assign_games(
        tom1_game_ids,
        split_seed=split_seed,
        game_counts=game_counts,
    )
    split_game_sets = {
        split_name: set(game_ids)
        for split_name, game_ids in split_game_ids.items()
    }
    if any(
        split_game_sets[left] & split_game_sets[right]
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
    ):
        raise RuntimeError("internal game split overlap")
    if set().union(*split_game_sets.values()) != tom1_game_ids:
        raise RuntimeError("internal game split does not cover the input")

    source_records = {"tom1": tom1_records, "tom2": tom2_records}
    split_records = {
        tom_order: {
            split_name: sorted(
                (
                    record
                    for record in records
                    if record["game_id"] in split_game_sets[split_name]
                ),
                key=lambda record: (record["game_id"], record["step_idx"]),
            )
            for split_name in SPLIT_NAMES
        }
        for tom_order, records in source_records.items()
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
    )
    try:
        for tom_order in ("tom1", "tom2"):
            (temporary / tom_order).mkdir()

        split_summaries: dict[str, Any] = {}
        for split_name in SPLIT_NAMES:
            tom1_output = temporary / "tom1" / f"{split_name}.jsonl"
            tom2_output = temporary / "tom2" / f"{split_name}.jsonl"
            _write_jsonl(tom1_output, split_records["tom1"][split_name])
            _write_jsonl(tom2_output, split_records["tom2"][split_name])

            reread_tom1 = _load_d_jsonl(
                tom1_output,
                expected_task=OFFLINE_PRIVATE_CONDITIONED_TOM1_TASK,
            )
            reread_tom2 = _load_d_jsonl(
                tom2_output,
                expected_task=OFFLINE_PUBLIC_ONLY_TOM2_TASK,
            )
            if reread_tom1 != split_records["tom1"][split_name]:
                raise RuntimeError("written ToM1 records differ from source D")
            if reread_tom2 != split_records["tom2"][split_name]:
                raise RuntimeError("written ToM2 records differ from source D")

            split_summaries[split_name] = {
                "game_ids": split_game_ids[split_name],
                "game_count": game_counts[split_name],
                "tom1_row_count": len(reread_tom1),
                "tom2_row_count": len(reread_tom2),
                "tom1_file_sha256": _sha256(tom1_output),
                "tom2_file_sha256": _sha256(tom2_output),
            }

        manifest = {
            "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
            "split_policy_version": SPLIT_POLICY_VERSION,
            "split_code_commit": _split_code_commit(),
            "split_seed": split_seed,
            "d_schema_version": D_SCHEMA_VERSION,
            "d_materialization_policy_version": (
                D_MATERIALIZATION_POLICY_VERSION
            ),
            "train_game_count": game_counts["train"],
            "validation_game_count": game_counts["validation"],
            "test_game_count": game_counts["test"],
            "total_game_count": len(tom1_game_ids),
            "tom1_source": _source_summary(
                tom1_source,
                tom1_records,
                materialization_task=(
                    OFFLINE_PRIVATE_CONDITIONED_TOM1_TASK
                ),
            ),
            "tom2_source": _source_summary(
                tom2_source,
                tom2_records,
                materialization_task=OFFLINE_PUBLIC_ONLY_TOM2_TASK,
            ),
            "game_ids": sorted(tom1_game_ids),
            "splits": split_summaries,
            "game_id_sets_equal": True,
            "tom1_step_set_equals_tom2_step_set_required": False,
            "game_overlap": False,
        }
        manifest["manifest_digest"] = canonical_digest(manifest)
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            f"{canonical_json(manifest)}\n",
            encoding="utf-8",
        )

        if destination.exists():
            raise FileExistsError(
                f"output directory already exists: {destination}"
            )
        temporary.rename(destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split strict canonical D V1 ToM1/ToM2 JSONL by game."
    )
    parser.add_argument("--tom1", required=True)
    parser.add_argument("--tom2", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-seed", required=True, type=int)
    parser.add_argument("--train-game-count", required=True, type=int)
    parser.add_argument("--validation-game-count", required=True, type=int)
    parser.add_argument("--test-game-count", required=True, type=int)
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    manifest = split_offline_d_training_data(
        tom1_path=args.tom1,
        tom2_path=args.tom2,
        output_dir=args.output_dir,
        split_seed=args.split_seed,
        train_game_count=args.train_game_count,
        validation_game_count=args.validation_game_count,
        test_game_count=args.test_game_count,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
