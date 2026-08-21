"""Tests for the canonical D V1 game-level split contract."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

import script.twd_tom.split_offline_d_training_data as splitter_module
from script.twd_tom.split_offline_d_training_data import (
    SPLIT_MANIFEST_SCHEMA_VERSION,
    SPLIT_NAMES,
    SPLIT_POLICY_VERSION,
    split_offline_d_training_data,
)
from tests.twd_tom.test_twd_tom_dataset import d_sample
from werewolf.offline_materialization import (
    D_MATERIALIZATION_POLICY_VERSION,
    D_SCHEMA_VERSION,
    OFFLINE_PRIVATE_CONDITIONED_TOM1_TASK,
    OFFLINE_PUBLIC_ONLY_TOM2_TASK,
    validate_offline_tom_training_record,
)
from werewolf.trajectory import canonical_digest, canonical_json


SPLITTER_COMMIT = "abcdef0123456789abcdef0123456789abcdef01"
MATERIALIZER_COMMITS = (
    "0123456789abcdef0123456789abcdef01234567",
    "123456789abcdef0123456789abcdef012345678",
)


def _record(
    tom_order: int,
    *,
    game_id: str,
    step_idx: int,
    materializer_code_commit: str | None = None,
) -> dict:
    record = d_sample(tom_order)
    record["game_id"] = game_id
    record["step_idx"] = step_idx
    record["label_cutoff_step_idx"] = step_idx
    record["boundary_id"] = (
        f"{game_id}:step_{step_idx:06d}:PRE_PUBLIC_SPEECH"
    )
    if materializer_code_commit is not None:
        record["materializer_code_commit"] = materializer_code_commit
    record.pop("record_digest")
    record["record_digest"] = canonical_digest(record)
    return validate_offline_tom_training_record(record)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(f"{canonical_json(record)}\n" for record in records),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_records() -> tuple[list[dict], list[dict]]:
    game_ids = ("game_charlie", "game_alpha", "game_bravo")
    tom1 = [
        _record(
            1,
            game_id=game_id,
            step_idx=step_idx,
            materializer_code_commit=MATERIALIZER_COMMITS[index % 2],
        )
        for index, (game_id, step_idx) in enumerate(
            zip(game_ids, (7, 2, 9))
        )
    ]
    tom2 = [
        _record(
            2,
            game_id=game_id,
            step_idx=step_idx,
            materializer_code_commit=MATERIALIZER_COMMITS[(index + 1) % 2],
        )
        for index, (game_id, step_idx) in enumerate(
            zip(game_ids, (3, 12, 1))
        )
    ]
    return tom1, tom2


def _write_sources(
    tmp_path: Path,
    *,
    tom1: list[dict] | None = None,
    tom2: list[dict] | None = None,
    prefix: str = "source",
) -> tuple[Path, Path, list[dict], list[dict]]:
    default_tom1, default_tom2 = _source_records()
    tom1_records = default_tom1 if tom1 is None else tom1
    tom2_records = default_tom2 if tom2 is None else tom2
    tom1_path = tmp_path / f"{prefix}_tom1.jsonl"
    tom2_path = tmp_path / f"{prefix}_tom2.jsonl"
    _write_jsonl(tom1_path, tom1_records)
    _write_jsonl(tom2_path, tom2_records)
    return tom1_path, tom2_path, tom1_records, tom2_records


def _split(
    tom1_path: Path,
    tom2_path: Path,
    output_dir: Path,
    *,
    split_seed: int = 402,
) -> dict:
    return split_offline_d_training_data(
        tom1_path=tom1_path,
        tom2_path=tom2_path,
        output_dir=output_dir,
        split_seed=split_seed,
        train_game_count=1,
        validation_game_count=1,
        test_game_count=1,
    )


def _expected_rank(game_ids: set[str], split_seed: int) -> list[str]:
    def game_hash(game_id: str) -> str:
        value = f"{SPLIT_POLICY_VERSION}\0{split_seed}\0{game_id}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    return sorted(game_ids, key=lambda game_id: (game_hash(game_id), game_id))


def test_split_contract_preserves_d_records_and_provenance(
    tmp_path,
    monkeypatch,
):
    tom1_path, tom2_path, tom1_source, tom2_source = _write_sources(tmp_path)
    output_dir = tmp_path / "split"
    validation_calls = []
    strict_validator = validate_offline_tom_training_record

    def tracking_validator(record):
        validation_calls.append((record.get("game_id"), record.get("step_idx")))
        return strict_validator(record)

    monkeypatch.setattr(
        splitter_module,
        "validate_offline_tom_training_record",
        tracking_validator,
    )
    monkeypatch.setattr(
        splitter_module,
        "_split_code_commit",
        lambda: SPLITTER_COMMIT,
    )
    manifest = _split(tom1_path, tom2_path, output_dir)

    assert len(validation_calls) == 2 * (len(tom1_source) + len(tom2_source))
    assert {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file()
    } == {
        "tom1/train.jsonl",
        "tom1/validation.jsonl",
        "tom1/test.jsonl",
        "tom2/train.jsonl",
        "tom2/validation.jsonl",
        "tom2/test.jsonl",
        "manifest.json",
    }

    expected_top_level_fields = {
        "schema_version",
        "split_policy_version",
        "split_code_commit",
        "split_seed",
        "d_schema_version",
        "d_materialization_policy_version",
        "train_game_count",
        "validation_game_count",
        "test_game_count",
        "total_game_count",
        "tom1_source",
        "tom2_source",
        "game_ids",
        "splits",
        "game_id_sets_equal",
        "tom1_step_set_equals_tom2_step_set_required",
        "game_overlap",
        "manifest_digest",
    }
    assert set(manifest) == expected_top_level_fields
    assert manifest["schema_version"] == SPLIT_MANIFEST_SCHEMA_VERSION
    assert manifest["split_policy_version"] == SPLIT_POLICY_VERSION
    assert manifest["split_code_commit"] == SPLITTER_COMMIT
    assert manifest["split_seed"] == 402
    assert manifest["d_schema_version"] == D_SCHEMA_VERSION
    assert manifest["d_materialization_policy_version"] == (
        D_MATERIALIZATION_POLICY_VERSION
    )
    assert set(manifest["game_ids"]) == set(SPLIT_NAMES)
    assert manifest["game_id_sets_equal"] is True
    assert manifest["tom1_step_set_equals_tom2_step_set_required"] is False
    assert manifest["game_overlap"] is False
    assert manifest["tom1_source"] == {
        "sha256": _sha256(tom1_path),
        "row_count": 3,
        "game_count": 3,
        "materialization_task": OFFLINE_PRIVATE_CONDITIONED_TOM1_TASK,
        "materializer_code_commits": sorted(MATERIALIZER_COMMITS),
    }
    assert manifest["tom2_source"] == {
        "sha256": _sha256(tom2_path),
        "row_count": 3,
        "game_count": 3,
        "materialization_task": OFFLINE_PUBLIC_ONLY_TOM2_TASK,
        "materializer_code_commits": sorted(MATERIALIZER_COMMITS),
    }

    source_by_order = {"tom1": tom1_source, "tom2": tom2_source}
    source_game_ids = {record["game_id"] for record in tom1_source}
    seen_games = set()
    for split_name in SPLIT_NAMES:
        summary = manifest["splits"][split_name]
        assert set(summary) == {
            "game_count",
            "tom1_row_count",
            "tom2_row_count",
            "tom1_file_sha256",
            "tom2_file_sha256",
        }
        assert summary["game_count"] == 1
        split_games = set(manifest["game_ids"][split_name])
        assert summary["game_count"] == len(
            manifest["game_ids"][split_name]
        )
        assert not seen_games & split_games
        seen_games.update(split_games)
        for tom_order in ("tom1", "tom2"):
            output_path = output_dir / tom_order / f"{split_name}.jsonl"
            output_records = _read_jsonl(output_path)
            expected_records = sorted(
                (
                    record
                    for record in source_by_order[tom_order]
                    if record["game_id"] in split_games
                ),
                key=lambda record: (record["game_id"], record["step_idx"]),
            )
            assert output_records == expected_records
            assert [record["record_digest"] for record in output_records] == [
                record["record_digest"] for record in expected_records
            ]
            for record in output_records:
                validate_offline_tom_training_record(record)
            assert summary[f"{tom_order}_row_count"] == len(output_records)
            assert summary[f"{tom_order}_file_sha256"] == _sha256(output_path)

    assert seen_games == source_game_ids
    assert json.loads((output_dir / "manifest.json").read_text()) == manifest
    payload = deepcopy(manifest)
    digest = payload.pop("manifest_digest")
    assert digest == canonical_digest(payload)


def test_hash_assignment_and_output_jsonl_are_input_order_independent(
    tmp_path,
    monkeypatch,
):
    tom1, tom2 = _source_records()
    first_tom1, first_tom2, _, _ = _write_sources(
        tmp_path,
        tom1=tom1,
        tom2=tom2,
        prefix="first",
    )
    second_tom1, second_tom2, _, _ = _write_sources(
        tmp_path,
        tom1=list(reversed(tom1)),
        tom2=[tom2[1], tom2[2], tom2[0]],
        prefix="second",
    )
    monkeypatch.setattr(
        splitter_module,
        "_split_code_commit",
        lambda: SPLITTER_COMMIT,
    )
    first_output = tmp_path / "first_split"
    second_output = tmp_path / "second_split"
    first = _split(first_tom1, first_tom2, first_output, split_seed=17)
    second = _split(second_tom1, second_tom2, second_output, split_seed=17)

    all_game_ids = set().union(
        *(set(game_ids) for game_ids in first["game_ids"].values())
    )
    ranked = _expected_rank(all_game_ids, 17)
    assert first["game_ids"]["train"] == ranked[:1]
    assert first["game_ids"]["validation"] == ranked[1:2]
    assert first["game_ids"]["test"] == ranked[2:]
    assert first["game_ids"] == second["game_ids"]
    for tom_order in ("tom1", "tom2"):
        for split_name in SPLIT_NAMES:
            relative = Path(tom_order) / f"{split_name}.jsonl"
            assert (first_output / relative).read_bytes() == (
                second_output / relative
            ).read_bytes()


def test_split_code_commit_runs_git_from_splitter_repository(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return type("Result", (), {"stdout": f"{SPLITTER_COMMIT}\n"})()

    monkeypatch.setattr(splitter_module.subprocess, "run", fake_run)
    assert splitter_module._split_code_commit() == SPLITTER_COMMIT
    assert calls == [
        (
            ["git", "rev-parse", "HEAD"],
            {
                "check": True,
                "capture_output": True,
                "text": True,
                "cwd": Path(splitter_module.__file__).resolve().parents[2],
            },
        )
    ]


def test_split_code_commit_is_independent_of_process_cwd(tmp_path, monkeypatch):
    expected = splitter_module._split_code_commit()
    monkeypatch.chdir(tmp_path)
    assert splitter_module._split_code_commit() == expected


@pytest.mark.parametrize(
    ("wrong_order", "expected_task"),
    [
        ("tom1", OFFLINE_PRIVATE_CONDITIONED_TOM1_TASK),
        ("tom2", OFFLINE_PUBLIC_ONLY_TOM2_TASK),
    ],
)
def test_wrong_materialization_task_is_rejected(
    tmp_path,
    wrong_order,
    expected_task,
):
    tom1, tom2 = _source_records()
    if wrong_order == "tom1":
        tom1[0] = _record(2, game_id="game_charlie", step_idx=7)
    else:
        tom2[0] = _record(1, game_id="game_charlie", step_idx=3)
    tom1_path, tom2_path, _, _ = _write_sources(
        tmp_path,
        tom1=tom1,
        tom2=tom2,
    )

    with pytest.raises(
        ValueError,
        match=f"wrong materialization_task; expected {expected_task}",
    ):
        _split(tom1_path, tom2_path, tmp_path / "split")


def test_unequal_game_id_sets_are_rejected(tmp_path):
    tom1, tom2 = _source_records()
    tom2[0] = _record(2, game_id="game_delta", step_idx=3)
    tom1_path, tom2_path, _, _ = _write_sources(
        tmp_path,
        tom1=tom1,
        tom2=tom2,
    )
    with pytest.raises(ValueError, match="game_id sets differ"):
        _split(tom1_path, tom2_path, tmp_path / "split")


@pytest.mark.parametrize("tom_order", [1, 2])
def test_duplicate_game_step_within_one_source_is_rejected(tmp_path, tom_order):
    tom1, tom2 = _source_records()
    records = tom1 if tom_order == 1 else tom2
    records.append(deepcopy(records[0]))
    tom1_path, tom2_path, _, _ = _write_sources(
        tmp_path,
        tom1=tom1,
        tom2=tom2,
    )
    with pytest.raises(ValueError, match=r"duplicate \(game_id, step_idx\)"):
        _split(tom1_path, tom2_path, tmp_path / "split")


@pytest.mark.parametrize(
    ("train_count", "validation_count", "test_count", "message"),
    [
        (1, 1, 2, "must sum to the unique game count"),
        (0, 1, 2, "train_game_count must be a positive integer"),
        (True, 1, 1, "train_game_count must be a positive integer"),
    ],
)
def test_split_counts_are_nonempty_integers_and_sum_exactly(
    tmp_path,
    train_count,
    validation_count,
    test_count,
    message,
):
    tom1_path, tom2_path, _, _ = _write_sources(tmp_path)
    with pytest.raises(ValueError, match=message):
        split_offline_d_training_data(
            tom1_path=tom1_path,
            tom2_path=tom2_path,
            output_dir=tmp_path / "split",
            split_seed=402,
            train_game_count=train_count,
            validation_game_count=validation_count,
            test_game_count=test_count,
        )


def test_malformed_d_record_is_rejected_by_strict_validator(tmp_path):
    tom1, tom2 = _source_records()
    tom1[0].pop("record_digest")
    tom1_path, tom2_path, _, _ = _write_sources(
        tmp_path,
        tom1=tom1,
        tom2=tom2,
    )
    with pytest.raises(ValueError, match="fields do not match V1"):
        _split(tom1_path, tom2_path, tmp_path / "split")


def test_existing_destination_is_rejected(tmp_path):
    tom1_path, tom2_path, _, _ = _write_sources(tmp_path)
    output_dir = tmp_path / "split"
    output_dir.mkdir()
    sentinel = output_dir / "sentinel"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output directory already exists"):
        _split(tom1_path, tom2_path, output_dir)
    assert sentinel.read_text(encoding="utf-8") == "preserve"
