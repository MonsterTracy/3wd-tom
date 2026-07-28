from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from tests.twd_tom.public_event_fixtures import public_history_fields
from werewolf.models.twd_tom.public_events import public_speech_actions

from script.twd_tom.split_formal_dataset import (
    split_projected_dataset,
)
from werewolf.models.twd_tom.samples import (
    SAMPLE_SCHEMA_VERSION as PLAYER_SUSPICION_SCHEMA_VERSION,
)
from werewolf.models.twd_tom.schema import (
    PROJECTED_SCHEMA_VERSION,
    PROJECTION_VERSION,
)


SPLIT_NAMES = (
    "train",
    "validation",
    "test",
)


def _write_jsonl(
    path: Path,
    records,
) -> None:
    path.write_text(
        "".join(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _read_jsonl(
    path: Path,
) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def _sha256(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def _make_projected_input(
    path: Path,
    projected_sample_factory,
    *,
    game_count: int,
    snapshots_per_game: int = 2,
) -> list[dict]:
    records = []
    for game_index in range(1, game_count + 1):
        for step_idx in range(1, snapshots_per_game + 1):
            row = projected_sample_factory(
                game_id=f"game_{game_index:03d}",
                step_idx=step_idx,
            )
            actions = (
                []
                if step_idx == 1
                else public_speech_actions(row["public_events"])
            )
            row.update(
                public_history_fields(
                    actions, speaker_id=row["speaker_id"]
                )
            )
            records.append(row)
    _write_jsonl(
        path,
        records,
    )
    return records


def _split(
    input_path: Path,
    output_dir: Path,
    *,
    seed: int = 7,
    train_game_count: int = 1,
    validation_game_count: int = 1,
    test_game_count: int = 1,
):
    return split_projected_dataset(
        input_path=input_path,
        output_dir=output_dir,
        seed=seed,
        train_game_count=train_game_count,
        validation_game_count=(
            validation_game_count
        ),
        test_game_count=test_game_count,
    )


def _assert_no_temporary_output(
    output_dir: Path,
) -> None:
    assert not output_dir.exists()
    assert list(
        output_dir.parent.glob(
            f".{output_dir.name}.*"
        )
    ) == []


def test_three_game_split_is_complete_disjoint_and_order_preserving(
    tmp_path,
    projected_sample_factory,
):
    input_path = tmp_path / "projected.jsonl"
    source_records = _make_projected_input(
        input_path,
        projected_sample_factory,
        game_count=3,
        snapshots_per_game=3,
    )
    original_input = input_path.read_bytes()
    output_dir = tmp_path / "split"

    manifest = _split(
        input_path,
        output_dir,
    )

    assert {
        path.name
        for path in output_dir.iterdir()
    } == {
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        "split_manifest.json",
    }
    assert manifest[
        "total_game_count"
    ] == 3
    assert manifest[
        "total_record_count"
    ] == 9
    assert manifest[
        "input_sha256"
    ] == _sha256(input_path)
    assert len(
        manifest["source_commit"]
    ) == 40
    assert manifest[
        "schema_version"
    ] == PROJECTED_SCHEMA_VERSION
    assert manifest[
        "projection_version"
    ] == PROJECTION_VERSION
    assert (
        manifest["train_game_count"],
        manifest["validation_game_count"],
        manifest["test_game_count"],
    ) == (
        1,
        1,
        1,
    )

    seen_game_ids: set[str] = set()
    seen_records: set[tuple[str, int]] = set()
    for split_name in SPLIT_NAMES:
        split = manifest["splits"][
            split_name
        ]
        assert split["game_count"] == 1
        assert split["record_count"] == 3
        assert split["sha256"] == _sha256(
            output_dir
            / f"{split_name}.jsonl"
        )

        game_ids = set(
            split["game_ids"]
        )
        assert len(game_ids) == 1
        assert not seen_game_ids & game_ids
        seen_game_ids.update(game_ids)

        output_records = _read_jsonl(
            output_dir
            / f"{split_name}.jsonl"
        )
        expected_records = [
            record
            for record in source_records
            if record["game_id"] in game_ids
        ]
        assert output_records == expected_records
        seen_records.update(
            (
                record["game_id"],
                record["step_idx"],
            )
            for record in output_records
        )

    assert seen_game_ids == {
        "game_001",
        "game_002",
        "game_003",
    }
    assert len(seen_records) == len(
        source_records
    )
    assert input_path.read_bytes() == original_input


def test_twelve_game_split_supports_explicit_eight_two_two(
    tmp_path,
    projected_sample_factory,
):
    input_path = tmp_path / "projected.jsonl"
    _make_projected_input(
        input_path,
        projected_sample_factory,
        game_count=12,
    )

    manifest = _split(
        input_path,
        tmp_path / "split",
        seed=123,
        train_game_count=8,
        validation_game_count=2,
        test_game_count=2,
    )

    assert {
        split_name: manifest[
            "splits"
        ][split_name]["game_count"]
        for split_name in SPLIT_NAMES
    } == {
        "train": 8,
        "validation": 2,
        "test": 2,
    }
    assert sum(
        manifest["splits"][
            split_name
        ]["record_count"]
        for split_name in SPLIT_NAMES
    ) == 24


def test_assignments_and_outputs_are_deterministic(
    tmp_path,
    projected_sample_factory,
):
    input_path = tmp_path / "projected.jsonl"
    _make_projected_input(
        input_path,
        projected_sample_factory,
        game_count=12,
    )

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    different_dir = tmp_path / "different"
    first = _split(
        input_path,
        first_dir,
        seed=41,
        train_game_count=8,
        validation_game_count=2,
        test_game_count=2,
    )
    second = _split(
        input_path,
        second_dir,
        seed=41,
        train_game_count=8,
        validation_game_count=2,
        test_game_count=2,
    )
    different = _split(
        input_path,
        different_dir,
        seed=42,
        train_game_count=8,
        validation_game_count=2,
        test_game_count=2,
    )

    assert first == second
    for filename in (
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        "split_manifest.json",
    ):
        assert (
            first_dir / filename
        ).read_bytes() == (
            second_dir / filename
        ).read_bytes()
    assert {
        split_name: first["splits"][
            split_name
        ]["game_ids"]
        for split_name in SPLIT_NAMES
    } != {
        split_name: different["splits"][
            split_name
        ]["game_ids"]
        for split_name in SPLIT_NAMES
    }


@pytest.mark.parametrize(
    (
        "train_game_count",
        "validation_game_count",
        "test_game_count",
        "match",
    ),
    [
        (0, 1, 2, "train_game_count"),
        (1, -1, 3, "validation_game_count"),
        (1, 1, False, "test_game_count"),
        (1, 1, 2, "must sum"),
        (1, 1, 4, "must sum"),
    ],
)
def test_invalid_game_counts_fail_before_output(
    tmp_path,
    projected_sample_factory,
    train_game_count,
    validation_game_count,
    test_game_count,
    match,
):
    input_path = tmp_path / "projected.jsonl"
    _make_projected_input(
        input_path,
        projected_sample_factory,
        game_count=3,
    )
    output_dir = tmp_path / "split"

    with pytest.raises(
        ValueError,
        match=match,
    ):
        _split(
            input_path,
            output_dir,
            train_game_count=train_game_count,
            validation_game_count=(
                validation_game_count
            ),
            test_game_count=test_game_count,
        )

    _assert_no_temporary_output(
        output_dir
    )


def test_existing_output_directory_is_rejected(
    tmp_path,
    projected_sample_factory,
):
    input_path = tmp_path / "projected.jsonl"
    _make_projected_input(
        input_path,
        projected_sample_factory,
        game_count=3,
    )
    output_dir = tmp_path / "split"
    output_dir.mkdir()
    marker = output_dir / "marker"
    marker.write_text(
        "unchanged",
        encoding="utf-8",
    )

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        _split(
            input_path,
            output_dir,
        )

    assert marker.read_text(
        encoding="utf-8"
    ) == "unchanged"


def test_game_counts_cannot_leave_unassigned_games(
    tmp_path,
    projected_sample_factory,
):
    input_path = tmp_path / "projected.jsonl"
    _make_projected_input(
        input_path,
        projected_sample_factory,
        game_count=4,
    )
    output_dir = tmp_path / "split"

    with pytest.raises(
        ValueError,
        match="must sum",
    ):
        _split(
            input_path,
            output_dir,
        )

    _assert_no_temporary_output(
        output_dir
    )


def test_empty_input_is_rejected_before_output(
    tmp_path,
):
    input_path = tmp_path / "empty.jsonl"
    input_path.write_text(
        "",
        encoding="utf-8",
    )
    output_dir = tmp_path / "split"

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        _split(
            input_path,
            output_dir,
        )

    _assert_no_temporary_output(
        output_dir
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda row: row.__setitem__(
                "schema_version",
                PLAYER_SUSPICION_SCHEMA_VERSION,
            ),
            "explicit offline pair projection",
        ),
        (
            lambda row: row.__setitem__(
                "schema_version",
                "classic7_pre_speech_pair_belief_v1",
            ),
            "schema_version",
        ),
        (
            lambda row: row.__setitem__(
                "schema_version",
                "unknown_schema",
            ),
            "schema_version",
        ),
        (
            lambda row: row.pop(
                "game_id"
            ),
            "game_id",
        ),
        (
            lambda row: row.__setitem__(
                "game_id",
                7,
            ),
            "game_id",
        ),
        (
            lambda row: row.__setitem__(
                "game_id",
                "",
            ),
            "game_id",
        ),
        (
            lambda row: row.pop(
                "phase"
            ),
            "phase",
        ),
        (
            lambda row: row.__setitem__(
                "plausible_wolf_pairs",
                [],
            ),
            "legacy or truth-derived",
        ),
    ],
    ids=(
        "raw-schema",
        "old-pair-support-schema",
        "unknown-schema",
        "missing-game-id",
        "non-string-game-id",
        "empty-game-id",
        "malformed-row",
        "legacy-field",
    ),
)
def test_invalid_projected_rows_fail_closed(
    tmp_path,
    projected_sample_factory,
    mutation,
    match,
):
    records = [
        projected_sample_factory(
            game_id=f"game_{index:03d}",
        )
        for index in range(1, 4)
    ]
    mutation(records[1])
    input_path = tmp_path / "projected.jsonl"
    _write_jsonl(
        input_path,
        records,
    )
    output_dir = tmp_path / "split"

    with pytest.raises(
        (TypeError, ValueError),
        match=match,
    ):
        _split(
            input_path,
            output_dir,
        )

    _assert_no_temporary_output(
        output_dir
    )


def test_tampered_pair_target_fails_closed(
    tmp_path,
    projected_sample_factory,
):
    records = [
        projected_sample_factory(
            game_id=f"game_{index:03d}",
        )
        for index in range(1, 4)
    ]
    target = records[2][
        "pair_targets"
    ]["player3"]
    target[0] += 0.25
    input_path = tmp_path / "projected.jsonl"
    _write_jsonl(
        input_path,
        records,
    )
    output_dir = tmp_path / "split"

    with pytest.raises(
        ValueError,
        match="pair target",
    ):
        _split(
            input_path,
            output_dir,
        )

    _assert_no_temporary_output(
        output_dir
    )


def test_write_failure_removes_temporary_directory(
    tmp_path,
    projected_sample_factory,
    monkeypatch,
):
    from script.twd_tom import (
        split_formal_dataset as splitter,
    )

    input_path = tmp_path / "projected.jsonl"
    _make_projected_input(
        input_path,
        projected_sample_factory,
        game_count=3,
    )
    output_dir = tmp_path / "split"

    def fail_write(*_args, **_kwargs):
        raise OSError("synthetic write failure")

    monkeypatch.setattr(
        splitter,
        "_write_jsonl",
        fail_write,
    )

    with pytest.raises(
        OSError,
        match="synthetic write failure",
    ):
        _split(
            input_path,
            output_dir,
        )

    _assert_no_temporary_output(
        output_dir
    )


def test_train_ignores_test_and_eval_reads_it_only_when_explicit(
    tmp_path,
    projected_sample_factory,
    monkeypatch,
):
    from script.twd_tom import (
        eval as eval_module,
    )
    from script.twd_tom import (
        train as train_module,
    )

    input_path = tmp_path / "projected.jsonl"
    _make_projected_input(
        input_path,
        projected_sample_factory,
        game_count=3,
    )
    output_dir = tmp_path / "split"
    _split(
        input_path,
        output_dir,
    )

    train_path = (
        output_dir / "train.jsonl"
    )
    validation_path = (
        output_dir / "validation.jsonl"
    )
    test_path = (
        output_dir / "test.jsonl"
    )
    original_test = test_path.read_bytes()
    test_path.write_text(
        "not valid projected data\n",
        encoding="utf-8",
    )

    training_reads: list[Path] = []
    original_training_loader = (
        train_module.load_twd_tom_jsonl
    )

    def tracked_training_loader(path):
        training_reads.append(
            Path(path).resolve()
        )
        return original_training_loader(
            path
        )

    monkeypatch.setattr(
        train_module,
        "load_twd_tom_jsonl",
        tracked_training_loader,
    )
    training_summary = (
        train_module.run_training(
            train_module.TrainingConfig(
                train_dataset_path=str(
                    train_path
                ),
                validation_dataset_path=str(
                    validation_path
                ),
                output_dir=str(
                    tmp_path / "training"
                ),
                epochs=1,
                batch_size=2,
                learning_rate=1e-3,
                weight_decay=0.0,
                seed=3,
                device="cpu",
                d_model=8,
                n_head=2,
                n_layer=1,
                dropout=0.0,
                max_seq_len=8,
                dim_feedforward=16,
            )
        )
    )

    assert set(training_reads) == {
        train_path.resolve(),
        validation_path.resolve(),
    }
    assert test_path.resolve() not in (
        training_reads
    )

    test_path.write_bytes(
        original_test
    )
    evaluation_reads: list[Path] = []
    original_evaluation_loader = (
        eval_module.load_twd_tom_jsonl
    )

    def tracked_evaluation_loader(path):
        evaluation_reads.append(
            Path(path).resolve()
        )
        return original_evaluation_loader(
            path
        )

    monkeypatch.setattr(
        eval_module,
        "load_twd_tom_jsonl",
        tracked_evaluation_loader,
    )
    evaluation_summary = (
        eval_module.evaluate_checkpoint(
            eval_module.EvaluationConfig(
                checkpoint_path=(
                    training_summary[
                        "best_checkpoint"
                    ]
                ),
                dataset_path=str(
                    test_path
                ),
                training_dataset_path=str(
                    train_path
                ),
                output_path=str(
                    tmp_path
                    / "evaluation.json"
                ),
                batch_size=2,
                device="cpu",
            )
        )
    )

    assert evaluation_summary[
        "status"
    ] == "ok"
    assert set(evaluation_reads) == {
        train_path.resolve(),
        test_path.resolve(),
    }
    assert validation_path.resolve() not in (
        evaluation_reads
    )
