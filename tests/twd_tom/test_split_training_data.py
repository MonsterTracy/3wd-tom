"""Tests for paired game-level ToM training-data splitting."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from script.twd_tom.materialize_training_data import materialize_training_records
from script.twd_tom.split_training_data import (
    SPLIT_GAME_COUNTS,
    split_training_data,
    split_training_data_from_manifest,
)
from werewolf.models.twd_tom.dataset import TWDToMDataset
from werewolf.models.twd_tom.schema import (
    PROJECTED_SCHEMA_VERSION,
    PROJECTION_VERSION,
)


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


def _write_split_manifest(path: Path, splits: dict[str, list[str]]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": PROJECTED_SCHEMA_VERSION,
                "projection_version": PROJECTION_VERSION,
                "train_game_count": len(splits["train"]),
                "validation_game_count": len(splits["validation"]),
                "test_game_count": len(splits["test"]),
                "total_game_count": sum(len(values) for values in splits.values()),
                "splits": {
                    name: {
                        "game_ids": game_ids,
                        "game_count": len(game_ids),
                    }
                    for name, game_ids in splits.items()
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _all_ok(sample: dict) -> dict:
    sample = deepcopy(sample)
    for subject in sample["belief_status"]:
        sample["belief_status"][subject] = "ok"
        sample["belief_errors"][subject] = None
        sample["suspected_werewolves"][subject] = []
    return sample


def _formal_inputs(tmp_path: Path, suspicion_sample_factory):
    raw_records = [
        _all_ok(
            suspicion_sample_factory(
                game_id=f"game_{game_index:03d}",
                observers=(1, 2, 3),
            )
        )
        for game_index in range(1, 5)
    ]
    unresolved_speaker = raw_records[3]
    unresolved_speaker["belief_status"]["player2"] = "semantic_error"
    unresolved_speaker["belief_errors"]["player2"] = "synthetic invalid report"
    unresolved_speaker["suspected_werewolves"]["player2"] = None
    materialized = materialize_training_records(raw_records)
    tom1 = tmp_path / "formal_tom1.jsonl"
    tom2 = tmp_path / "formal_tom2.jsonl"
    _write_jsonl(tom1, materialized["tom1_records"])
    _write_jsonl(tom2, materialized["tom2_records"])
    return tom1, tom2, materialized


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


def test_manifest_split_reuses_assignment_and_allows_order_row_difference(
    tmp_path,
    suspicion_sample_factory,
):
    tom1, tom2, materialized = _formal_inputs(
        tmp_path,
        suspicion_sample_factory,
    )
    manifest = tmp_path / "split_manifest.json"
    assignments = {
        "train": ["game_004", "game_001"],
        "validation": ["game_002"],
        "test": ["game_003"],
    }
    _write_split_manifest(manifest, assignments)

    first = tmp_path / "first"
    second = tmp_path / "second"
    result = split_training_data_from_manifest(
        tom1_path=tom1,
        tom2_path=tom2,
        split_manifest_path=manifest,
        output_dir=first,
    )
    split_training_data_from_manifest(
        tom1_path=tom1,
        tom2_path=tom2,
        split_manifest_path=manifest,
        output_dir=second,
    )

    assert len(materialized["tom1_records"]) == 3
    assert len(materialized["tom2_records"]) == 4
    assert result["order_split_stats"]["tom1"] == {
        "train": {
            "assigned_game_count": 2,
            "represented_game_count": 1,
            "zero_row_game_ids": ["game_004"],
            "record_count": 1,
        },
        "validation": {
            "assigned_game_count": 1,
            "represented_game_count": 1,
            "zero_row_game_ids": [],
            "record_count": 1,
        },
        "test": {
            "assigned_game_count": 1,
            "represented_game_count": 1,
            "zero_row_game_ids": [],
            "record_count": 1,
        },
    }
    assert result["order_split_stats"]["tom2"]["train"] == {
        "assigned_game_count": 2,
        "represented_game_count": 2,
        "zero_row_game_ids": [],
        "record_count": 2,
    }
    seen = set()
    for tom_order in (1, 2):
        for split_name, assigned_game_ids in assignments.items():
            path = first / f"tom{tom_order}" / f"{split_name}.jsonl"
            dataset = TWDToMDataset.from_jsonl(path, tom_order=tom_order)
            actual_game_ids = {sample["game_id"] for sample in dataset.samples}
            assert actual_game_ids <= set(assigned_game_ids)
            if tom_order == 2:
                assert actual_game_ids == set(assigned_game_ids)
                assert seen.isdisjoint(actual_game_ids)
                seen.update(actual_game_ids)
            relative = path.relative_to(first)
            assert path.read_bytes() == (second / relative).read_bytes()
    assert seen == set().union(*map(set, assignments.values()))


def test_manifest_split_rejects_unknown_games_before_writing(
    tmp_path,
    suspicion_sample_factory,
):
    tom1, tom2, _materialized = _formal_inputs(
        tmp_path,
        suspicion_sample_factory,
    )
    records = _read(tom2)
    records[0]["game_id"] = "unknown_game"
    _write_jsonl(tom2, records)
    manifest = tmp_path / "split_manifest.json"
    _write_split_manifest(
        manifest,
        {
            "train": ["game_001", "game_004"],
            "validation": ["game_002"],
            "test": ["game_003"],
        },
    )
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="absent from the split manifest"):
        split_training_data_from_manifest(
            tom1_path=tom1,
            tom2_path=tom2,
            split_manifest_path=manifest,
            output_dir=output,
        )
    assert not output.exists()


def test_manifest_split_rejects_malformed_manifest_and_formal_rows(
    tmp_path,
    suspicion_sample_factory,
):
    tom1, tom2, _materialized = _formal_inputs(
        tmp_path,
        suspicion_sample_factory,
    )
    manifest = tmp_path / "split_manifest.json"
    _write_split_manifest(
        manifest,
        {
            "train": ["game_001", "game_002"],
            "validation": ["game_002"],
            "test": ["game_003"],
        },
    )
    with pytest.raises(ValueError, match="assignments overlap"):
        split_training_data_from_manifest(
            tom1_path=tom1,
            tom2_path=tom2,
            split_manifest_path=manifest,
            output_dir=tmp_path / "overlap",
        )

    _write_split_manifest(
        manifest,
        {
            "train": ["game_001", "game_004"],
            "validation": ["game_002"],
            "test": ["game_003"],
        },
    )
    malformed = _read(tom1)
    malformed[0]["unexpected"] = True
    _write_jsonl(tom1, malformed)
    with pytest.raises(ValueError, match="sample field set mismatch"):
        split_training_data_from_manifest(
            tom1_path=tom1,
            tom2_path=tom2,
            split_manifest_path=manifest,
            output_dir=tmp_path / "malformed",
        )
