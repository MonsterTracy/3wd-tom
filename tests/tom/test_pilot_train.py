import json
import math

import pytest
import torch

import script.tom.train as train_module
from script.tom.split_pilot import build_parser as build_split_parser
from script.tom.split_pilot import prepare_pilot_split
from script.tom.train import build_parser as build_train_parser
from script.tom.train import train_and_evaluate
from werewolf.models.tom.model import BeliefModel


def _make_three_way_split(tmp_path, rows, write_jsonl):
    raw = tmp_path / "raw"
    write_jsonl(raw / "pilot.jsonl", rows)
    split = tmp_path / "split"
    prepare_pilot_split(
        input_dir=raw,
        output_dir=split,
        train_games=1,
        val_games=1,
        test_games=1,
        split_seed=42,
    )
    return split


def _write_manual_split(split, train, val, test, write_jsonl):
    rows = {"train": train, "val": val, "test": test}
    for name, values in rows.items():
        write_jsonl(split / f"{name}.jsonl", values)
    manifest = {
        "split_seed": 42,
        **{
            name: {
                "game_ids": sorted({row["game_id"] for row in values}),
            }
            for name, values in rows.items()
        },
    }
    (split / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def test_pilot_cli_defaults_are_explicit():
    split_args = build_split_parser().parse_args(
        ["--input-dir", "raw", "--output-dir", "split"]
    )
    assert (
        split_args.train_games,
        split_args.val_games,
        split_args.test_games,
        split_args.split_seed,
    ) == (8, 1, 1, 42)

    train_args = build_train_parser().parse_args(
        [
            "--train", "train.jsonl",
            "--val", "val.jsonl",
            "--test", "test.jsonl",
            "--output-dir", "run",
        ]
    )
    assert train_args.learning_rate == 3e-4
    assert train_args.weight_decay == 1e-2
    assert train_args.batch_size == 16
    assert train_args.max_epochs == 30
    assert train_args.seed == 42


def test_one_epoch_saves_reloads_and_evaluates_test_once(
    tmp_path,
    pilot_sample_factory,
    write_jsonl,
    monkeypatch,
):
    split = _make_three_way_split(
        tmp_path,
        [pilot_sample_factory(f"game-{index}") for index in range(3)],
        write_jsonl,
    )
    calls = []
    original_evaluate = train_module._evaluate_model

    def recording_evaluate(model, loader, device):
        calls.append(loader)
        return original_evaluate(model, loader, device)

    monkeypatch.setattr(train_module, "_evaluate_model", recording_evaluate)
    run = tmp_path / "run"
    metrics = train_and_evaluate(
        train_path=split / "train.jsonl",
        val_path=split / "val.jsonl",
        test_path=split / "test.jsonl",
        output_dir=run,
        batch_size=1,
        max_epochs=1,
        seed=42,
        device="cpu",
    )

    assert len(calls) == 2
    assert metrics["best_epoch"] == 1
    for field in (
        "best_val_ce",
        "test_ce",
        "val_uniform_ce",
        "test_uniform_ce",
        "test_improvement_over_uniform",
    ):
        assert math.isfinite(metrics[field])
    assert metrics["val_uniform_ce"] == pytest.approx(math.log(7), abs=1e-6)
    assert metrics["test_uniform_ce"] == pytest.approx(math.log(7), abs=1e-6)
    assert metrics["training_seed"] == 42
    assert metrics["split_seed"] == 42
    assert json.loads((run / "metrics.json").read_text()) == metrics
    history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
    assert len(history) == 1
    assert history[0]["epoch"] == 1
    assert math.isfinite(history[0]["train_ce"])
    assert math.isfinite(history[0]["val_ce"])

    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 1
    assert checkpoint["validation_ce"] == metrics["best_val_ce"]
    assert checkpoint["model_config"]["max_sequence_length"] == 256
    assert checkpoint["training_arguments"]["max_epochs"] == 1
    reloaded = BeliefModel()
    reloaded.load_state_dict(checkpoint["model_state_dict"])


def test_training_rejects_all_mask_false_batch(
    tmp_path,
    pilot_sample_factory,
    write_jsonl,
):
    split = tmp_path / "split"
    _write_manual_split(
        split,
        [pilot_sample_factory("train", valid=False)],
        [pilot_sample_factory("val")],
        [pilot_sample_factory("test")],
        write_jsonl,
    )

    with pytest.raises(ValueError, match="observer_mask must select"):
        train_and_evaluate(
            train_path=split / "train.jsonl",
            val_path=split / "val.jsonl",
            test_path=split / "test.jsonl",
            output_dir=tmp_path / "run",
            batch_size=1,
            max_epochs=1,
            device="cpu",
        )


def test_training_preserves_model_error_above_256(
    tmp_path,
    pilot_sample_factory,
    write_jsonl,
):
    split = tmp_path / "split"
    _write_manual_split(
        split,
        [pilot_sample_factory("train", action_count=256)],
        [pilot_sample_factory("val")],
        [pilot_sample_factory("test")],
        write_jsonl,
    )

    with pytest.raises(ValueError, match="sequence length 257 exceeds.*256"):
        train_and_evaluate(
            train_path=split / "train.jsonl",
            val_path=split / "val.jsonl",
            test_path=split / "test.jsonl",
            output_dir=tmp_path / "run",
            batch_size=1,
            max_epochs=1,
            device="cpu",
        )
