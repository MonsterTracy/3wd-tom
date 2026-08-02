"""Split aligned first- and second-order ToM data by game ID."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SPLIT_GAME_COUNTS = {
    "train": 210,
    "val": 45,
    "test": 45,
}
SNAPSHOT_KEY_FIELDS = (
    "game_id",
    "step_idx",
    "phase",
    "speaker_id",
    "report_trigger",
)

JsonlRecord = tuple[dict[str, Any], str]


def _load_jsonl(path: Path) -> list[JsonlRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"input file not found: {path}")

    records: list[JsonlRecord] = []
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
            if not isinstance(value, dict):
                raise TypeError(f"{path}: line {line_number} must be a JSON object")
            _validate_snapshot_key_fields(value, path=path, line_number=line_number)
            records.append((value, line))

    if not records:
        raise ValueError(f"input dataset is empty: {path}")
    return records


def _validate_snapshot_key_fields(
    record: Mapping[str, Any],
    *,
    path: Path,
    line_number: int,
) -> None:
    missing = [field for field in SNAPSHOT_KEY_FIELDS if field not in record]
    if missing:
        raise ValueError(
            f"{path}: line {line_number} is missing snapshot key fields: {missing}"
        )

    game_id = record["game_id"]
    if not isinstance(game_id, str) or not game_id.strip():
        raise ValueError(f"{path}: line {line_number} has an invalid game_id")
    step_idx = record["step_idx"]
    if isinstance(step_idx, bool) or not isinstance(step_idx, int) or step_idx < 0:
        raise ValueError(f"{path}: line {line_number} has an invalid step_idx")
    phase = record["phase"]
    if not isinstance(phase, str) or not phase.strip():
        raise ValueError(f"{path}: line {line_number} has an invalid phase")
    speaker_id = record["speaker_id"]
    if (
        isinstance(speaker_id, bool)
        or not isinstance(speaker_id, int)
        or not 1 <= speaker_id <= 7
    ):
        raise ValueError(f"{path}: line {line_number} has an invalid speaker_id")
    report_trigger = record["report_trigger"]
    if not isinstance(report_trigger, str) or not report_trigger.strip():
        raise ValueError(f"{path}: line {line_number} has an invalid report_trigger")


def _group_by_game(records: Sequence[JsonlRecord]) -> dict[str, list[JsonlRecord]]:
    grouped: dict[str, list[JsonlRecord]] = defaultdict(list)
    for record in records:
        grouped[record[0]["game_id"]].append(record)
    return dict(grouped)


def _snapshot_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(record[field] for field in SNAPSHOT_KEY_FIELDS)


def _validate_alignment(
    tom1_by_game: Mapping[str, Sequence[JsonlRecord]],
    tom2_by_game: Mapping[str, Sequence[JsonlRecord]],
) -> list[str]:
    tom1_games = set(tom1_by_game)
    tom2_games = set(tom2_by_game)
    if tom1_games != tom2_games:
        only_tom1 = sorted(tom1_games - tom2_games)
        only_tom2 = sorted(tom2_games - tom1_games)
        raise ValueError(
            "tom1 and tom2 game_id sets differ; "
            f"only_tom1={only_tom1[:10]}, only_tom2={only_tom2[:10]}"
        )

    expected_game_count = sum(SPLIT_GAME_COUNTS.values())
    if len(tom1_games) != expected_game_count:
        raise ValueError(
            f"expected {expected_game_count} unique games, got {len(tom1_games)}"
        )

    for game_id in sorted(tom1_games):
        tom1_records = tom1_by_game[game_id]
        tom2_records = tom2_by_game[game_id]
        if len(tom1_records) != len(tom2_records):
            raise ValueError(
                f"snapshot count mismatch for {game_id}: "
                f"tom1={len(tom1_records)}, tom2={len(tom2_records)}"
            )
        tom1_keys = Counter(_snapshot_key(record[0]) for record in tom1_records)
        tom2_keys = Counter(_snapshot_key(record[0]) for record in tom2_records)
        if tom1_keys != tom2_keys:
            raise ValueError(f"corresponding snapshot keys differ for {game_id}")

    return sorted(tom1_games)


def _build_game_splits(game_ids: Sequence[str], seed: int) -> dict[str, set[str]]:
    shuffled = list(game_ids)
    random.Random(seed).shuffle(shuffled)
    train_end = SPLIT_GAME_COUNTS["train"]
    val_end = train_end + SPLIT_GAME_COUNTS["val"]
    return {
        "train": set(shuffled[:train_end]),
        "val": set(shuffled[train_end:val_end]),
        "test": set(shuffled[val_end:]),
    }


def _output_paths(output_dir: Path) -> dict[str, dict[str, Path]]:
    return {
        tom_order: {
            split_name: output_dir / tom_order / f"{split_name}.jsonl"
            for split_name in SPLIT_GAME_COUNTS
        }
        for tom_order in ("tom1", "tom2")
    }


def _write_splits(
    *,
    records_by_order: Mapping[str, Sequence[JsonlRecord]],
    game_splits: Mapping[str, set[str]],
    output_paths: Mapping[str, Mapping[str, Path]],
) -> dict[str, dict[str, int]]:
    for order_paths in output_paths.values():
        next(iter(order_paths.values())).parent.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    counts: dict[str, dict[str, int]] = {}
    try:
        for tom_order, records in records_by_order.items():
            counts[tom_order] = {}
            for split_name, split_games in game_splits.items():
                target = output_paths[tom_order][split_name]
                record_count = 0
                with target.open("x", encoding="utf-8", newline="") as handle:
                    created.append(target)
                    for record, original_line in records:
                        if record["game_id"] not in split_games:
                            continue
                        handle.write(original_line)
                        if not original_line.endswith(("\n", "\r")):
                            handle.write("\n")
                        record_count += 1
                counts[tom_order][split_name] = record_count
    except BaseException:
        for path in created:
            if path.exists():
                path.unlink()
        raise
    return counts


def split_training_data(
    *,
    tom1_path: str | Path,
    tom2_path: str | Path,
    output_dir: str | Path,
    seed: int,
) -> dict[str, Any]:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    tom1_source = Path(tom1_path)
    tom2_source = Path(tom2_path)
    destinations = _output_paths(Path(output_dir))
    existing = [
        str(path)
        for order_paths in destinations.values()
        for path in order_paths.values()
        if path.exists()
    ]
    if existing:
        raise FileExistsError(f"output files already exist: {existing}")

    tom1_records = _load_jsonl(tom1_source)
    tom2_records = _load_jsonl(tom2_source)
    tom1_by_game = _group_by_game(tom1_records)
    tom2_by_game = _group_by_game(tom2_records)
    game_ids = _validate_alignment(tom1_by_game, tom2_by_game)
    game_splits = _build_game_splits(game_ids, seed)
    record_counts = _write_splits(
        records_by_order={"tom1": tom1_records, "tom2": tom2_records},
        game_splits=game_splits,
        output_paths=destinations,
    )

    split_names = tuple(SPLIT_GAME_COUNTS)
    disjoint = all(
        game_splits[left].isdisjoint(game_splits[right])
        for index, left in enumerate(split_names)
        for right in split_names[index + 1 :]
    )
    covers_all_games = set().union(*game_splits.values()) == set(game_ids)
    aligned_game_ids = {
        split_name: (
            {record[0]["game_id"] for record in tom1_records if record[0]["game_id"] in games}
            == {record[0]["game_id"] for record in tom2_records if record[0]["game_id"] in games}
        )
        for split_name, games in game_splits.items()
    }
    if not disjoint or not covers_all_games or not all(aligned_game_ids.values()):
        raise RuntimeError("internal game split verification failed")

    return {
        "total_games": len(game_ids),
        "game_counts": dict(SPLIT_GAME_COUNTS),
        "record_counts": record_counts,
        "disjoint": disjoint,
        "covers_all_games": covers_all_games,
        "aligned_game_ids": aligned_game_ids,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split aligned first- and second-order ToM JSONL by game ID."
    )
    parser.add_argument("--tom1", required=True)
    parser.add_argument("--tom2", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", required=True, type=int)
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    result = split_training_data(
        tom1_path=args.tom1,
        tom2_path=args.tom2,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
