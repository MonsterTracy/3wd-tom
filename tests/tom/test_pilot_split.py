import json
import math
from copy import deepcopy

import pytest

from archive.legacy_tom.script.tom.split_pilot import prepare_pilot_split
from archive.legacy_tom.werewolf.models.tom.dataset import encode_sample


def _prepare(
    input_dir,
    output_dir,
    *,
    train_games=8,
    val_games=1,
    test_games=1,
    split_seed=42,
):
    return prepare_pilot_split(
        input_dir=input_dir,
        output_dir=output_dir,
        train_games=train_games,
        val_games=val_games,
        test_games=test_games,
        split_seed=split_seed,
    )


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_split_is_game_disjoint_deterministic_and_keeps_long_rows(
    tmp_path,
    pilot_sample_factory,
    write_jsonl,
):
    raw = tmp_path / "raw"
    samples = [
        pilot_sample_factory(
            f"game-{index:02d}",
            action_count=256 if index == 9 else 1,
        )
        for index in range(10)
    ]
    samples.append(pilot_sample_factory("game-00", step_idx=2, valid=False))
    for index, sample in enumerate(reversed(samples)):
        write_jsonl(raw / f"pilot_{index:03d}.jsonl", [sample])

    first = _prepare(raw, tmp_path / "split-a")
    second = _prepare(raw, tmp_path / "split-b")

    assert first == second
    split_sets = {
        name: set(first[name]["game_ids"])
        for name in ("train", "val", "test")
    }
    assert [first[name]["game_count"] for name in ("train", "val", "test")] == [
        8, 1, 1,
    ]
    assert split_sets["train"].isdisjoint(split_sets["val"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["val"].isdisjoint(split_sets["test"])
    assert set().union(*split_sets.values()) == {
        f"game-{index:02d}" for index in range(10)
    }
    assert sum(first[name]["sample_count"] for name in split_sets) == 10
    assert sum(first[name]["valid_observer_rows"] for name in split_sets) == 10

    overall = first["overall"]
    assert overall["raw_sample_count"] == 11
    assert overall["effective_sample_count"] == 10
    assert overall["excluded_all_invalid_samples"] == 1
    assert overall["excluded_games"] == []
    lengths = overall["sequence_length"]
    assert lengths["count"] == 10
    assert lengths["min"] == 2
    assert lengths["max"] == 257
    assert lengths["count_gt_256"] == 1
    assert math.isfinite(lengths["mean"])
    assert math.isfinite(lengths["p90"])
    assert math.isfinite(lengths["p95"])

    output_rows = [
        row
        for name in split_sets
        for row in _read_jsonl(tmp_path / "split-a" / f"{name}.jsonl")
    ]
    long_row = next(row for row in output_rows if row["game_id"] == "game-09")
    assert encode_sample(long_row)["sequence_length"].item() == 257
    assert json.loads((tmp_path / "split-a" / "manifest.json").read_text()) == first


def test_all_invalid_game_is_excluded_before_exact_count_check(
    tmp_path,
    pilot_sample_factory,
    write_jsonl,
):
    rows = [pilot_sample_factory(f"game-{index:02d}") for index in range(10)]
    rows.append(pilot_sample_factory("empty-game", valid=False))
    write_jsonl(tmp_path / "raw" / "pilot.jsonl", rows)

    manifest = _prepare(tmp_path / "raw", tmp_path / "split")

    assert manifest["overall"]["effective_game_count"] == 10
    assert manifest["overall"]["excluded_all_invalid_samples"] == 1
    assert manifest["overall"]["excluded_games"] == [
        {
            "game_id": "empty-game",
            "reason": "NO_SUPERVISED_SAMPLES",
            "raw_sample_count": 1,
        }
    ]


def test_effective_game_count_must_match_explicit_counts(
    tmp_path,
    pilot_sample_factory,
    write_jsonl,
):
    write_jsonl(
        tmp_path / "raw" / "pilot.jsonl",
        [pilot_sample_factory(f"game-{index}") for index in range(9)],
    )

    with pytest.raises(ValueError, match="requested=10, effective=9"):
        _prepare(tmp_path / "raw", tmp_path / "split")
    with pytest.raises(ValueError, match="train_games must be a positive integer"):
        _prepare(
            tmp_path / "raw",
            tmp_path / "split-zero",
            train_games=0,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate", "duplicate sample identity"),
        ("seed", "inconsistent seed"),
        ("context", "inconsistent episode_context"),
        ("formal", "formal_speech_actions must not be empty"),
    ],
)
def test_split_rejects_duplicate_mismatch_and_malformed_rows(
    tmp_path,
    pilot_sample_factory,
    write_jsonl,
    mutation,
    message,
):
    first = pilot_sample_factory("game-1")
    second = deepcopy(first)
    if mutation == "seed":
        second["step_idx"] = 2
        second["seed"] = 18
    elif mutation == "context":
        second["step_idx"] = 2
        second["episode_context"] = "seer_guard"
    elif mutation == "formal":
        first["formal_speech_actions"] = []
        first["public_events"][-1]["sp_actions"] = []
        second = pilot_sample_factory("game-2")
    write_jsonl(tmp_path / "raw" / "pilot.jsonl", [first, second])

    with pytest.raises((TypeError, ValueError), match=message):
        _prepare(
            tmp_path / "raw",
            tmp_path / "split",
            train_games=1,
            val_games=1,
            test_games=1,
        )


def test_split_rejects_invalid_json_without_dropping_it(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "pilot.jsonl").write_text("{not-json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        _prepare(raw, tmp_path / "split", train_games=1, val_games=1, test_games=1)
