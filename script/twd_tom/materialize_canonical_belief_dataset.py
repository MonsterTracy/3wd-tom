"""Materialize canonical tom-v2 belief snapshots with a game-level split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from werewolf.models.twd_tom.dataset import TWDToMDataset, load_twd_tom_jsonl


SPLIT_POLICY_VERSION = "classic7_tom_v2_game_hash_split_v1"
BELIEF_SNAPSHOTS_FILENAME = "belief_snapshots.jsonl"
SPLIT_NAMES = ("train", "validation", "test")


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


def _load_canonical_games(canonical_root: Path) -> dict[str, list[dict[str, Any]]]:
    games_root = canonical_root / "games"
    if not games_root.is_dir():
        raise FileNotFoundError(f"canonical games directory not found: {games_root}")
    paths = sorted(games_root.glob(f"*/{BELIEF_SNAPSHOTS_FILENAME}"))
    if not paths:
        raise FileNotFoundError(f"no canonical belief snapshots found under: {games_root}")

    games: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        records = load_twd_tom_jsonl(path)
        if not records:
            raise ValueError(f"canonical belief snapshot file cannot be empty: {path}")
        game_ids = {record.get("game_id") for record in records}
        if len(game_ids) != 1:
            raise ValueError(f"one canonical game file must contain exactly one game_id: {path}")
        game_id = next(iter(game_ids))
        if not isinstance(game_id, str) or not game_id.strip():
            raise ValueError(f"canonical game_id must be non-empty text: {path}")
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
    games = _load_canonical_games(Path(canonical_root).resolve())
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
        if destination.exists():
            raise FileExistsError(f"output directory already exists: {destination}")
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return {
        "split_policy_version": SPLIT_POLICY_VERSION,
        "split_seed": split_seed,
        "game_ids": split_game_ids,
        "game_counts": counts,
        "row_counts": row_counts,
    }


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
