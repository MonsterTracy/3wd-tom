from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from script.twd_tom.project_suspicion_to_pairs import (
    PROJECTED_SAMPLE_FIELDS,
    project_suspicion_sample,
)
from werewolf.models.twd_tom.dataset import load_twd_tom_jsonl
from werewolf.models.twd_tom.samples import SAMPLE_FIELDS
from werewolf.models.twd_tom.schema import (
    PROJECTED_SCHEMA_VERSION,
    PROJECTION_VERSION,
)


SPLIT_NAMES = (
    "train",
    "validation",
    "test",
)


def _validate_projected_sample(sample: Any) -> None:
    """Validate a projected split record independently of raw training data."""

    if not isinstance(sample, Mapping):
        raise TypeError("each projected sample must be a mapping")
    if set(sample) != PROJECTED_SAMPLE_FIELDS:
        missing = sorted(PROJECTED_SAMPLE_FIELDS - set(sample))
        extra = sorted(set(sample) - PROJECTED_SAMPLE_FIELDS)
        raise ValueError(
            f"projected sample field set mismatch; missing={missing}, extra={extra}"
        )
    raw_sample = {field: sample[field] for field in SAMPLE_FIELDS}
    raw_sample["schema_version"] = sample["source_schema_version"]
    expected = project_suspicion_sample(raw_sample)
    if sample != expected:
        raise ValueError("projected pair target or projection metadata is invalid")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def _source_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot resolve the splitter source commit") from exc

    source_commit = result.stdout.strip()
    if (
        len(source_commit) != 40
        or any(
            character
            not in "0123456789abcdefABCDEF"
            for character in source_commit
        )
    ):
        raise RuntimeError("splitter source commit must be a 40-character SHA")
    return source_commit


def _positive_game_count(
    value: Any,
    *,
    field_name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _write_jsonl(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    with path.open(
        "x",
        encoding="utf-8",
    ) as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")


def _write_json(
    path: Path,
    value: Mapping[str, Any],
) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def split_projected_dataset(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    seed: int,
    train_game_count: int,
    validation_game_count: int,
    test_game_count: int,
) -> dict[str, Any]:
    """Split one validated projected JSONL by complete game."""

    source = Path(input_path).resolve()
    destination = Path(output_dir).resolve()

    if destination.exists():
        raise FileExistsError(
            f"output directory already exists: {destination}"
        )
    if not source.is_file():
        raise FileNotFoundError(
            f"input file not found: {source}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    requested_counts = {
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

    records = load_twd_tom_jsonl(source)
    if not records:
        raise ValueError("input dataset is empty")

    records_by_game: dict[str, list[dict[str, Any]]] = {}
    for row_number, record in enumerate(
        records,
        start=1,
    ):
        try:
            _validate_projected_sample(record)
        except (TypeError, ValueError) as exc:
            raise type(exc)(
                f"line {row_number}: {exc}"
            ) from exc
        records_by_game.setdefault(
            record["game_id"],
            [],
        ).append(record)

    game_ids = sorted(records_by_game)
    requested_game_count = sum(
        requested_counts.values()
    )
    if requested_game_count != len(game_ids):
        raise ValueError(
            "split game counts must sum to the distinct game_id count; "
            f"requested={requested_game_count}, actual={len(game_ids)}"
        )

    shuffled_game_ids = list(game_ids)
    random.Random(seed).shuffle(
        shuffled_game_ids
    )

    train_end = requested_counts["train"]
    validation_end = (
        train_end
        + requested_counts["validation"]
    )
    split_game_ids = {
        "train": shuffled_game_ids[:train_end],
        "validation": shuffled_game_ids[
            train_end:validation_end
        ],
        "test": shuffled_game_ids[
            validation_end:
        ],
    }
    split_game_id_sets = {
        split_name: set(assigned_game_ids)
        for split_name, assigned_game_ids in split_game_ids.items()
    }

    if (
        split_game_id_sets["train"]
        & split_game_id_sets["validation"]
        or split_game_id_sets["train"]
        & split_game_id_sets["test"]
        or split_game_id_sets["validation"]
        & split_game_id_sets["test"]
    ):
        raise RuntimeError("split game_id assignments overlap")
    if set().union(*split_game_id_sets.values()) != set(game_ids):
        raise RuntimeError("split game_id assignments do not cover the input")

    split_records = {
        split_name: [
            record
            for record in records
            if record["game_id"] in assigned_game_ids
        ]
        for split_name, assigned_game_ids in split_game_id_sets.items()
    }
    if sum(
        len(partition)
        for partition in split_records.values()
    ) != len(records):
        raise RuntimeError("split record count does not match the input")

    source_commit = _source_commit()
    input_sha256 = _sha256(source)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
    )

    try:
        split_summaries: dict[str, Any] = {}
        for split_name in SPLIT_NAMES:
            output_path = (
                temporary
                / f"{split_name}.jsonl"
            )
            _write_jsonl(
                output_path,
                split_records[split_name],
            )
            split_summaries[split_name] = {
                "game_ids": split_game_ids[
                    split_name
                ],
                "game_count": requested_counts[
                    split_name
                ],
                "record_count": len(
                    split_records[split_name]
                ),
                "sha256": _sha256(output_path),
            }

        manifest = {
            "source_commit": source_commit,
            "input_path": str(source),
            "input_sha256": input_sha256,
            "schema_version": PROJECTED_SCHEMA_VERSION,
            "projection_version": PROJECTION_VERSION,
            "split_seed": seed,
            "train_game_count": requested_counts["train"],
            "validation_game_count": requested_counts["validation"],
            "test_game_count": requested_counts["test"],
            "total_game_count": len(game_ids),
            "total_record_count": len(records),
            "splits": split_summaries,
        }
        _write_json(
            temporary / "split_manifest.json",
            manifest,
        )

        if destination.exists():
            raise FileExistsError(
                f"output directory already exists: {destination}"
            )
        temporary.replace(destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return manifest


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Split one projected ToM JSONL into deterministic "
            "game-level train, validation, and test partitions."
        )
    )
    parser.add_argument(
        "--input-path",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    parser.add_argument(
        "--seed",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--train-game-count",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--validation-game-count",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--test-game-count",
        required=True,
        type=int,
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = build_argument_parser().parse_args(
        argv
    )
    manifest = split_projected_dataset(
        input_path=args.input_path,
        output_dir=args.output_dir,
        seed=args.seed,
        train_game_count=(
            args.train_game_count
        ),
        validation_game_count=(
            args.validation_game_count
        ),
        test_game_count=args.test_game_count,
    )
    print(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
