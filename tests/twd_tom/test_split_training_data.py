"""Tests for paired game-level ToM training-data splitting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from script.twd_tom.split_training_data import SPLIT_GAME_COUNTS, split_training_data


def _records(*, tom_order: int) -> list[dict]:
    records = []
    for game_index in reversed(range(300)):
        game_id = f"game_{game_index:03d}"
        for step_idx in (2, 5):
            records.append(
                {
                    "game_id": game_id,
                    "step_idx": step_idx,
                    "phase": "1_day_speech",
                    "speaker_id": (game_index % 7) + 1,
                    "report_trigger": "pre_public_speech",
                    "tom_order": tom_order,
                }
            )
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _inputs(tmp_path: Path) -> tuple[Path, Path, list[dict], list[dict]]:
    tom1_records = _records(tom_order=1)
    tom2_records = _records(tom_order=2)
    tom1_path = tmp_path / "tom1.jsonl"
    tom2_path = tmp_path / "tom2.jsonl"
    _write_jsonl(tom1_path, tom1_records)
    _write_jsonl(tom2_path, tom2_records)
    return tom1_path, tom2_path, tom1_records, tom2_records


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _output_game_ids(output_dir: Path, tom_order: str, split_name: str) -> set[str]:
    return {
        record["game_id"]
        for record in _read(output_dir / tom_order / f"{split_name}.jsonl")
    }


def test_splits_by_game_id_not_by_line(tmp_path):
    tom1, tom2, _, _ = _inputs(tmp_path)
    output = tmp_path / "output"
    result = split_training_data(
        tom1_path=tom1, tom2_path=tom2, output_dir=output, seed=42
    )

    assert result["game_counts"] == SPLIT_GAME_COUNTS
    for split_name, game_count in SPLIT_GAME_COUNTS.items():
        rows = _read(output / "tom1" / f"{split_name}.jsonl")
        assert len(rows) == game_count * 2
        assert len({row["game_id"] for row in rows}) == game_count


def test_one_game_never_crosses_splits(tmp_path):
    tom1, tom2, _, _ = _inputs(tmp_path)
    output = tmp_path / "output"
    split_training_data(tom1_path=tom1, tom2_path=tom2, output_dir=output, seed=42)

    games = {
        name: _output_game_ids(output, "tom1", name) for name in SPLIT_GAME_COUNTS
    }
    assert games["train"].isdisjoint(games["val"])
    assert games["train"].isdisjoint(games["test"])
    assert games["val"].isdisjoint(games["test"])
    assert len(set().union(*games.values())) == 300


def test_tom_orders_share_the_same_game_split(tmp_path):
    tom1, tom2, _, _ = _inputs(tmp_path)
    output = tmp_path / "output"
    result = split_training_data(
        tom1_path=tom1, tom2_path=tom2, output_dir=output, seed=42
    )

    assert result["aligned_game_ids"] == {"train": True, "val": True, "test": True}
    for split_name in SPLIT_GAME_COUNTS:
        assert _output_game_ids(output, "tom1", split_name) == _output_game_ids(
            output, "tom2", split_name
        )


def test_same_seed_produces_identical_outputs(tmp_path):
    tom1, tom2, _, _ = _inputs(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    split_training_data(tom1_path=tom1, tom2_path=tom2, output_dir=first, seed=42)
    split_training_data(tom1_path=tom1, tom2_path=tom2, output_dir=second, seed=42)

    for tom_order in ("tom1", "tom2"):
        for split_name in SPLIT_GAME_COUNTS:
            relative = Path(tom_order) / f"{split_name}.jsonl"
            assert (first / relative).read_bytes() == (second / relative).read_bytes()


def test_mismatched_game_ids_fail(tmp_path):
    tom1, tom2, _, tom2_records = _inputs(tmp_path)
    tom2_records[-1]["game_id"] = "tom2_only_game"
    _write_jsonl(tom2, tom2_records)

    with pytest.raises(ValueError, match="game_id sets differ"):
        split_training_data(
            tom1_path=tom1,
            tom2_path=tom2,
            output_dir=tmp_path / "output",
            seed=42,
        )


def test_mismatched_snapshot_keys_fail(tmp_path):
    tom1, tom2, _, tom2_records = _inputs(tmp_path)
    tom2_records[0]["phase"] = "2_day_speech"
    _write_jsonl(tom2, tom2_records)

    with pytest.raises(ValueError, match="snapshot keys differ"):
        split_training_data(
            tom1_path=tom1,
            tom2_path=tom2,
            output_dir=tmp_path / "output",
            seed=42,
        )


def test_mismatched_snapshot_counts_fail(tmp_path):
    tom1, tom2, _, tom2_records = _inputs(tmp_path)
    _write_jsonl(tom2, tom2_records[:-1])

    with pytest.raises(ValueError, match="snapshot count mismatch"):
        split_training_data(
            tom1_path=tom1,
            tom2_path=tom2,
            output_dir=tmp_path / "output",
            seed=42,
        )


def test_existing_target_fails_without_overwrite(tmp_path):
    tom1, tom2, _, _ = _inputs(tmp_path)
    target = tmp_path / "output" / "tom1" / "train.jsonl"
    target.parent.mkdir(parents=True)
    target.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output files already exist"):
        split_training_data(
            tom1_path=tom1,
            tom2_path=tom2,
            output_dir=tmp_path / "output",
            seed=42,
        )
    assert target.read_text(encoding="utf-8") == "existing\n"
    assert not (tmp_path / "output" / "tom2").exists()


def test_input_files_are_not_modified(tmp_path):
    tom1, tom2, tom1_records, _ = _inputs(tmp_path)
    before = {tom1: tom1.read_bytes(), tom2: tom2.read_bytes()}
    output = tmp_path / "output"
    split_training_data(tom1_path=tom1, tom2_path=tom2, output_dir=output, seed=42)

    assert tom1.read_bytes() == before[tom1]
    assert tom2.read_bytes() == before[tom2]
    expected_order = [
        record["game_id"]
        for record in tom1_records
        if record["game_id"] in _output_game_ids(output, "tom1", "train")
    ]
    assert [record["game_id"] for record in _read(output / "tom1" / "train.jsonl")] == expected_order
