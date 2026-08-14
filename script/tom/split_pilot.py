"""Prepare deterministic whole-game splits for a formal ToM pilot."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from werewolf.models.tom.dataset import encode_sample
from werewolf.models.tom.schema import normalize_episode_context, normalize_player


SPLIT_NAMES = ("train", "val", "test")


def _positive_count(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _load_rows(input_dir: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input directory not found: {input_dir}")
    source_files = sorted(input_dir.glob("*.jsonl"))
    if not source_files:
        raise ValueError(f"input directory contains no JSONL files: {input_dir}")
    rows = []
    for path in source_files:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                location = f"{path}:{line_number}"
                if not line.strip():
                    raise ValueError(f"{location}: blank JSONL row")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{location}: invalid JSON") from exc
                if not isinstance(row, dict):
                    raise TypeError(f"{location}: row must be a mapping")
                rows.append(row)
    if not rows:
        raise ValueError("input JSONL files contain no rows")
    return rows, source_files


def _validate_identity(row: Mapping[str, Any]) -> tuple[str, int]:
    game_id = row.get("game_id")
    if not isinstance(game_id, str) or not game_id.strip():
        raise ValueError("game_id must be non-empty text")
    step_idx = row.get("step_idx")
    if isinstance(step_idx, bool) or not isinstance(step_idx, int) or step_idx < 0:
        raise ValueError("step_idx must be a non-negative integer")
    seed = row.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None")
    context = normalize_episode_context(row.get("episode_context"))
    if row.get("episode_context") != context:
        raise ValueError("episode_context must be canonical")
    speaker = normalize_player(row.get("speaker_id"))
    if row.get("speaker_id") != speaker:
        raise ValueError("speaker_id must be canonical")
    round_number = row.get("round")
    if (
        isinstance(round_number, bool)
        or not isinstance(round_number, int)
        or round_number <= 0
    ):
        raise ValueError("round must be a positive integer")
    if row.get("phase") not in {"speech", "speech_pk"}:
        raise ValueError("phase must be speech or speech_pk")
    return game_id, step_idx


def _length_stats(lengths: Sequence[int]) -> dict[str, int | float]:
    if not lengths:
        raise ValueError("sequence length stats require at least one sample")
    return {
        "count": len(lengths),
        "min": min(lengths),
        "median": float(statistics.median(lengths)),
        "mean": float(statistics.fmean(lengths)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "max": max(lengths),
        "count_gt_256": sum(length > 256 for length in lengths),
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def prepare_pilot_split(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    train_games: int,
    val_games: int,
    test_games: int,
    split_seed: int,
) -> dict[str, Any]:
    """Validate raw rows and split effective samples by complete game."""

    counts = {
        "train": _positive_count(train_games, field="train_games"),
        "val": _positive_count(val_games, field="val_games"),
        "test": _positive_count(test_games, field="test_games"),
    }
    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise ValueError("split_seed must be an integer")

    source = Path(input_dir).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise FileExistsError(f"output directory must be absent or empty: {destination}")

    rows, source_files = _load_rows(source)
    seen_identities = set()
    metadata_by_game: dict[str, tuple[int | None, str]] = {}
    rows_by_game: dict[str, list[dict[str, Any]]] = {}
    encoded_by_identity = {}
    for row_number, row in enumerate(rows, start=1):
        try:
            game_id, step_idx = _validate_identity(row)
            identity = (game_id, step_idx)
            if identity in seen_identities:
                raise ValueError(f"duplicate sample identity: {identity}")
            seen_identities.add(identity)
            metadata = (row["seed"], row["episode_context"])
            prior_metadata = metadata_by_game.setdefault(game_id, metadata)
            if prior_metadata[0] != metadata[0]:
                raise ValueError(f"game {game_id!r} has inconsistent seed")
            if prior_metadata[1] != metadata[1]:
                raise ValueError(f"game {game_id!r} has inconsistent episode_context")
            encoded = encode_sample(row)
        except (KeyError, TypeError, ValueError) as exc:
            raise type(exc)(f"raw row {row_number}: {exc}") from exc
        rows_by_game.setdefault(game_id, []).append(row)
        encoded_by_identity[identity] = encoded

    effective_by_game: dict[str, list[dict[str, Any]]] = {}
    excluded_games = []
    excluded_all_invalid_samples = 0
    for game_id in sorted(rows_by_game):
        effective_rows = []
        for row in rows_by_game[game_id]:
            identity = (game_id, row["step_idx"])
            if encoded_by_identity[identity]["observer_mask"].any().item():
                effective_rows.append(row)
            else:
                excluded_all_invalid_samples += 1
        if effective_rows:
            effective_by_game[game_id] = effective_rows
        else:
            excluded_games.append(
                {
                    "game_id": game_id,
                    "reason": "NO_SUPERVISED_SAMPLES",
                    "raw_sample_count": len(rows_by_game[game_id]),
                }
            )

    effective_game_ids = sorted(effective_by_game)
    requested_total = sum(counts.values())
    if len(effective_game_ids) != requested_total:
        raise ValueError(
            "effective game count does not match requested split total; "
            f"requested={requested_total}, effective={len(effective_game_ids)}"
        )
    shuffled = list(effective_game_ids)
    random.Random(split_seed).shuffle(shuffled)
    train_end = counts["train"]
    val_end = train_end + counts["val"]
    split_game_ids = {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }
    split_sets = {name: set(game_ids) for name, game_ids in split_game_ids.items()}
    if any(
        split_sets[left] & split_sets[right]
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
    ):
        raise RuntimeError("game IDs overlap across splits")

    split_rows = {
        name: [
            row
            for game_id in game_ids
            for row in effective_by_game[game_id]
        ]
        for name, game_ids in split_game_ids.items()
    }
    split_summaries = {}
    effective_lengths = []
    for name in SPLIT_NAMES:
        lengths = [
            int(encoded_by_identity[(row["game_id"], row["step_idx"])][
                "sequence_length"
            ].item())
            for row in split_rows[name]
        ]
        effective_lengths.extend(lengths)
        split_summaries[name] = {
            "game_ids": split_game_ids[name],
            "game_count": len(split_game_ids[name]),
            "sample_count": len(split_rows[name]),
            "valid_observer_rows": sum(
                int(
                    encoded_by_identity[(row["game_id"], row["step_idx"])][
                        "observer_mask"
                    ].sum().item()
                )
                for row in split_rows[name]
            ),
        }

    manifest = {
        "split_seed": split_seed,
        **split_summaries,
        "overall": {
            "source_files": [str(path) for path in source_files],
            "raw_sample_count": len(rows),
            "effective_sample_count": sum(len(value) for value in split_rows.values()),
            "effective_games": effective_game_ids,
            "effective_game_count": len(effective_game_ids),
            "excluded_games": excluded_games,
            "excluded_all_invalid_samples": excluded_all_invalid_samples,
            "sequence_length": _length_stats(effective_lengths),
        },
    }

    destination.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        _write_jsonl(destination / f"{name}.jsonl", split_rows[name])
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-games", type=int, default=8)
    parser.add_argument("--val-games", type=int, default=1)
    parser.add_argument("--test-games", type=int, default=1)
    parser.add_argument("--split-seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = prepare_pilot_split(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_games=args.train_games,
        val_games=args.val_games,
        test_games=args.test_games,
        split_seed=args.split_seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
