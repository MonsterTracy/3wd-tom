from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from script.twd_tom.build_dev100_training_data import (
    DATASET_ID,
    build_dev100_training_data,
)
from werewolf.models.twd_tom.dataset import (
    TWDToMDataset,
    collate_twd_tom_samples,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PACKAGE = REPOSITORY_ROOT / "datasets" / DATASET_ID


def _copy_source_package(tmp_path, name=DATASET_ID):
    required = (
        "dataset_manifest.json",
        "source_runs.txt",
        "raw.jsonl",
        "projected.jsonl",
    )
    missing = [name for name in required if not (SOURCE_PACKAGE / name).is_file()]
    if missing:
        pytest.skip(f"DEV100 source package is unavailable: {missing}")
    package = tmp_path / name
    package.mkdir()
    for name in required:
        shutil.copy2(SOURCE_PACKAGE / name, package / name)
    return package


def test_dev100_build_has_exact_counts_fixed_split_and_strict_load(tmp_path):
    package = _copy_source_package(tmp_path)
    result = build_dev100_training_data(package)

    assert result["materialization"]["input_snapshot_count"] == 1131
    assert result["materialization"]["semantic_error_observer_count"] == 18
    assert result["materialization"]["hard_knowledge_recovered_count"] == 0
    assert result["materialization"]["unresolved_observer_count"] == 18
    assert result["materialization"]["removed_tom1_snapshot_count"] == 3
    assert result["materialization"]["filtered_tom2_observer_count"] == 18
    assert result["materialization"]["raw_tom_row_count"] == 1128
    assert result["materialization"]["raw_tom2_row_count"] == 1131
    assert result["valid_tom2_target_count"] == 6497
    assert result["effective_tom2_target_count"] == 4401

    split = result["split"]["splits"]
    assert split["train"]["collected_game_count"] == 70
    assert split["train"]["represented_game_count"] == 69
    assert split["val"]["represented_game_count"] == 15
    assert split["test"]["represented_game_count"] == 15
    assert [split[name]["tom1_row_count"] for name in ("train", "val", "test")] == [
        799,
        162,
        167,
    ]
    assert [split[name]["tom2_row_count"] for name in ("train", "val", "test")] == [
        802,
        162,
        167,
    ]

    tom1_keys = set()
    tom2_keys = set()
    split_games = {}
    for tom_order in (1, 2):
        for split_name in ("train", "val", "test"):
            path = package / f"tom{tom_order}" / f"{split_name}.jsonl"
            dataset = TWDToMDataset.from_jsonl(path, tom_order=tom_order)
            loader = DataLoader(
                dataset,
                batch_size=2,
                collate_fn=collate_twd_tom_samples,
            )
            assert next(iter(loader))["pair_targets"].shape[-2:] == (7, 21)
            games = {sample["game_id"] for sample in dataset.samples}
            split_games[(tom_order, split_name)] = games
            keys = {
                (sample["game_id"], sample["step_idx"])
                for sample in dataset.samples
            }
            if tom_order == 1:
                tom1_keys |= keys
            else:
                tom2_keys |= keys

    for split_name in ("train", "val", "test"):
        assert split_games[(1, split_name)] == split_games[(2, split_name)]
    assert tom1_keys < tom2_keys
    assert all(
        split_games[(2, left)].isdisjoint(split_games[(2, right)])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    )
    assert not any(
        game_id.endswith("_seed_589")
        for games in split_games.values()
        for game_id in games
    )

    materialization_manifest = json.loads(
        (package / "materialization_manifest.json").read_text(encoding="utf-8")
    )
    split_manifest = json.loads(
        (package / "split_manifest.json").read_text(encoding="utf-8")
    )
    assert "/Users/" not in json.dumps(materialization_manifest)
    assert "/Users/" not in json.dumps(split_manifest)
    assert split_manifest["tom1_removed_snapshot_keys"] == [
        {
            "game_id": "game_005_seed_599",
            "phase": "1_day_speech",
            "speaker_id": 4,
            "step_idx": 4,
        },
        {
            "game_id": "game_004_seed_608",
            "phase": "1_day_speech",
            "speaker_id": 5,
            "step_idx": 9,
        },
        {
            "game_id": "game_001_seed_640",
            "phase": "2_day_speech",
            "speaker_id": 5,
            "step_idx": 25,
        },
    ]

    with pytest.raises(FileExistsError, match="already exist"):
        build_dev100_training_data(package)


def test_dev100_repeated_build_is_byte_deterministic(tmp_path):
    first = _copy_source_package(tmp_path, "first")
    second = _copy_source_package(tmp_path, "second")
    build_dev100_training_data(first)
    build_dev100_training_data(second)

    derived = (
        "raw_tom.jsonl",
        "raw_tom2.jsonl",
        "materialization_manifest.json",
        "split_manifest.json",
        "tom1/train.jsonl",
        "tom1/val.jsonl",
        "tom1/test.jsonl",
        "tom2/train.jsonl",
        "tom2/val.jsonl",
        "tom2/test.jsonl",
    )
    for relative_path in derived:
        assert (first / relative_path).read_bytes() == (
            second / relative_path
        ).read_bytes()
