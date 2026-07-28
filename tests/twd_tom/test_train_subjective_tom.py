"""Tests for explicit train/validation subjective ToM training."""

import inspect
import json
from pathlib import Path

import pytest
import torch
from transformers import GPT2Model

import script.twd_tom.train as train_module
from script.twd_tom.project_suspicion_to_pairs import project_suspicion_sample
from script.twd_tom.train import (
    TrainingConfig,
    build_arg_parser,
    build_model,
    load_explicit_dataset_partition,
    resolve_device,
    run_training,
)
from werewolf.models.twd_tom.samples import (
    SAMPLE_SCHEMA_VERSION as PLAYER_SUSPICION_SCHEMA_VERSION,
)
from tests.twd_tom.public_event_fixtures import public_history_fields
from werewolf.models.twd_tom.schema import (
    LABEL_PROVENANCE,
    LABEL_PROMPT_VERSION,
    NUM_WOLF_PAIR_CLASSES,
    PROJECTED_SCHEMA_VERSION as SAMPLE_SCHEMA_VERSION,
    PROJECTION_VERSION,
    TARGET_ENCODING,
)


def make_sample(
    *,
    game_id,
    step_idx,
):
    actions = [
        ["player2", "point_as_seer", "player2"],
        ["player2", "point_as_werewolf", "player7" if step_idx % 2 else "player6"],
    ]
    history_actions = [] if step_idx == 1 else actions
    return project_suspicion_sample({
        "schema_version": PLAYER_SUSPICION_SCHEMA_VERSION,
        "game_id": game_id,
        "step_idx": step_idx,
        "report_trigger": "pre_public_speech",
        "phase": "1_day_speech",
        "speaker_id": 6,
        "observer_ids": [1, 3, 5],
        **public_history_fields(history_actions, speaker_id=6),
        "suspected_werewolves": {
            "player1": None,
            "player3": ["player7"],
            "player5": [],
        },
        "known_werewolves": {
            subject: []
            for subject in ("player1", "player3", "player5")
        },
        "known_non_werewolves": {
            subject: [subject]
            for subject in ("player1", "player3", "player5")
        },
        "belief_status": {
            "player1": "parse_error",
            "player3": "ok",
            "player5": "ok",
        },
        "belief_errors": {
            "player1": "synthetic invalid report",
            "player3": None,
            "player5": None,
        },
        "label_provenance": LABEL_PROVENANCE,
        "agent_backend_ids": {
            subject: "fake_backend"
            for subject in ("player1", "player3", "player5")
        },
        "label_cutoff_step_idx": step_idx,
        "label_prompt_version": LABEL_PROMPT_VERSION,
    })


def write_dataset(
    path: Path,
    *,
    game_ids,
    samples_per_game=2,
):
    samples = []

    for game_id in game_ids:
        for step_index in range(
            samples_per_game
        ):
            samples.append(
                make_sample(
                    game_id=game_id,
                    step_idx=step_index + 1,
                )
            )

    path.write_text(
        "\n".join(
            json.dumps(
                sample,
                ensure_ascii=False,
            )
            for sample in samples
        )
        + "\n",
        encoding="utf-8",
    )

    return samples


def load_checkpoint(path: Path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def metric_record(value: float):
    return {
        "valid_subject_count": 1,
        "mean_loss": value,
        "mean_pair_kl_divergence": value,
        "mean_pair_cross_entropy": value,
        "mean_pair_total_variation": value,
        "mean_marginal_mae": value,
        "mean_marginal_row_sum_error": value,
        "mean_predicted_diagonal_marginal": value,
        "mean_target_diagonal_marginal": value,
    }


def test_explicit_partition_preserves_files_and_rejects_overlap(
    tmp_path,
):
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"

    write_dataset(
        train_path,
        game_ids=("game_a", "game_b"),
    )
    write_dataset(
        validation_path,
        game_ids=("game_c",),
    )

    partition = load_explicit_dataset_partition(
        train_dataset_path=train_path,
        validation_dataset_path=validation_path,
    )

    assert len(partition.train_samples) == 4
    assert len(partition.validation_samples) == 2
    assert partition.train_game_ids == (
        "game_a",
        "game_b",
    )
    assert partition.validation_game_ids == (
        "game_c",
    )

    write_dataset(
        validation_path,
        game_ids=("game_b",),
    )

    with pytest.raises(
        ValueError,
        match="overlapping game_id",
    ):
        load_explicit_dataset_partition(
            train_dataset_path=train_path,
            validation_dataset_path=(
                validation_path
            ),
        )


def test_run_training_writes_complete_artifacts(
    tmp_path,
):
    train_path = tmp_path / "train.jsonl"
    validation_path = (
        tmp_path / "validation.jsonl"
    )
    output_dir = tmp_path / "training"

    write_dataset(
        train_path,
        game_ids=(
            "game_001",
            "game_002",
            "game_003",
        ),
    )
    write_dataset(
        validation_path,
        game_ids=("game_004",),
    )

    config = TrainingConfig(
        train_dataset_path=str(train_path),
        validation_dataset_path=str(
            validation_path
        ),
        output_dir=str(output_dir),
        epochs=2,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        seed=7,
        device="cpu",
        gradient_clip_norm=1.0,
        early_stopping_patience=5,
        d_model=16,
        n_head=4,
        n_layer=1,
        dropout=0.0,
        max_seq_len=8,
        dim_feedforward=32,
    )

    summary = run_training(config)

    assert summary["status"] == "ok"
    assert summary["device"] == "cpu"
    assert summary["requested_epoch_count"] == 2
    assert summary["epoch_count"] == 2
    assert summary["stopped_early"] is False
    assert summary["train_sample_count"] == 6
    assert summary["validation_sample_count"] == 2
    assert summary["train_game_ids"] == [
        "game_001",
        "game_002",
        "game_003",
    ]
    assert summary["validation_game_ids"] == [
        "game_004",
    ]
    assert set(summary["train_game_ids"]).isdisjoint(
        summary["validation_game_ids"]
    )
    assert summary[
        "train_supervised_subject_count"
    ] == 12
    assert summary[
        "validation_supervised_subject_count"
    ] == 4

    best_path = Path(summary["best_checkpoint"])
    last_path = Path(summary["last_checkpoint"])
    history_path = Path(summary["history_path"])
    summary_path = output_dir / "summary.json"

    assert best_path.is_file()
    assert last_path.is_file()
    assert history_path.is_file()
    assert summary_path.is_file()

    checkpoint = load_checkpoint(best_path)

    assert checkpoint["schema_version"] == (
        SAMPLE_SCHEMA_VERSION
    )
    assert checkpoint["target_encoding"] == TARGET_ENCODING
    assert checkpoint["projection_version"] == PROJECTION_VERSION
    assert checkpoint["pair_class_count"] == NUM_WOLF_PAIR_CLASSES
    assert checkpoint[
        "target_distribution_is_reporter_probability"
    ] is False
    assert checkpoint[
        "target_distribution_is_deterministic_encoding"
    ] is True
    assert checkpoint["backbone"] == "gpt2_model"
    assert checkpoint["model_config"]["d_model"] == 16
    assert summary["backbone"] == "gpt2_model"
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert checkpoint["training_config"][
        "train_dataset_path"
    ] == str(train_path)
    assert checkpoint["training_config"][
        "validation_dataset_path"
    ] == str(validation_path)

    history = json.loads(
        history_path.read_text(
            encoding="utf-8"
        )
    )

    assert len(history) == 2
    assert history[-1]["train"][
        "valid_subject_count"
    ] == 12
    assert history[-1]["validation"][
        "valid_subject_count"
    ] == 4
    assert summary["final_train_metrics"][
        "mean_loss"
    ] >= 0.0
    assert summary["final_validation_metrics"][
        "mean_loss"
    ] >= 0.0
    assert "mean_pair_kl_divergence" in history[-1]["validation"]
    assert "mean_kl_divergence" not in history[-1]["validation"]


def test_early_stopping_uses_validation_pair_kl(
    tmp_path,
    monkeypatch,
):
    train_path = tmp_path / "train.jsonl"
    validation_path = (
        tmp_path / "validation.jsonl"
    )

    write_dataset(
        train_path,
        game_ids=("game_train",),
        samples_per_game=1,
    )
    write_dataset(
        validation_path,
        game_ids=("game_validation",),
        samples_per_game=1,
    )

    monkeypatch.setattr(
        train_module,
        "train_one_epoch",
        lambda *args, **kwargs: metric_record(0.5),
    )
    monkeypatch.setattr(
        train_module,
        "evaluate_model",
        lambda *args, **kwargs: metric_record(0.4),
    )

    summary = run_training(
        TrainingConfig(
            train_dataset_path=str(train_path),
            validation_dataset_path=str(
                validation_path
            ),
            output_dir=str(tmp_path / "output"),
            epochs=10,
            batch_size=1,
            device="cpu",
            early_stopping_patience=2,
            d_model=8,
            n_head=2,
            n_layer=1,
            dropout=0.0,
            max_seq_len=4,
            dim_feedforward=16,
        )
    )

    assert summary["stopped_early"] is True
    assert summary["epoch_count"] == 3
    assert summary["best_epoch"] == 1
    assert summary[
        "best_mean_pair_kl_divergence"
    ] == pytest.approx(0.4)
    assert summary["target_encoding"] == TARGET_ENCODING
    assert summary["projection_version"] == PROJECTION_VERSION
    assert summary["pair_class_count"] == NUM_WOLF_PAIR_CLASSES


def test_cli_requires_explicit_train_and_validation_paths():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--train-dataset",
            "train.jsonl",
            "--validation-dataset",
            "validation.jsonl",
            "--output-dir",
            "output",
        ]
    )

    assert args.train_dataset == "train.jsonl"
    assert args.validation_dataset == (
        "validation.jsonl"
    )
    assert not hasattr(args, "backbone_type")
    assert not hasattr(args, "dataset")
    assert not hasattr(
        args,
        "validation_fraction",
    )


def test_model_builder_uses_training_config(
    tmp_path,
):
    config = TrainingConfig(
        train_dataset_path=str(
            tmp_path / "train.jsonl"
        ),
        validation_dataset_path=str(
            tmp_path / "validation.jsonl"
        ),
        output_dir=str(tmp_path / "unused"),
        d_model=24,
        n_head=4,
        n_layer=3,
        dropout=0.0,
        max_seq_len=32,
        dim_feedforward=48,
    )

    model = build_model(config)

    assert isinstance(model.transformer, GPT2Model)
    assert model.config.d_model == 24
    assert model.config.n_head == 4
    assert model.config.n_layer == 3
    assert model.config.max_seq_len == 32
    assert model.config.dim_feedforward == 48


def test_cpu_device_resolution():
    assert resolve_device("cpu") == torch.device("cpu")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"batch_size": 0},
        {"learning_rate": 0.0},
        {"early_stopping_patience": 0},
        {"early_stopping_min_delta": -0.1},
        {"d_model": 15, "n_head": 4},
        {"max_seq_len": 0},
    ],
)
def test_invalid_training_config_is_rejected(
    tmp_path,
    kwargs,
):
    arguments = {
        "train_dataset_path": str(
            tmp_path / "train.jsonl"
        ),
        "validation_dataset_path": str(
            tmp_path / "validation.jsonl"
        ),
        "output_dir": str(tmp_path / "output"),
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError):
        TrainingConfig(**arguments)


def test_same_train_and_validation_path_is_rejected(
    tmp_path,
):
    path = tmp_path / "same.jsonl"

    with pytest.raises(
        ValueError,
        match="must be different files",
    ):
        TrainingConfig(
            train_dataset_path=str(path),
            validation_dataset_path=str(path),
            output_dir=str(tmp_path / "output"),
        )


def test_training_apis_have_no_truth_or_test_inputs():
    forbidden = {
        "roles",
        "true_roles",
        "wolf_labels",
        "truth",
        "actual_wolves",
        "alive_mask",
        "observer_id",
        "test_dataset_path",
        "test_dataset",
    }

    for function in (
        run_training,
        load_explicit_dataset_partition,
        build_model,
    ):
        parameters = inspect.signature(
            function
        ).parameters
        assert forbidden.isdisjoint(parameters)


def test_cli_rejects_removed_backbone_selector():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--train-dataset",
                "train.jsonl",
                "--validation-dataset",
                "validation.jsonl",
                "--output-dir",
                "output",
                "--backbone-type",
                "torch",
            ]
        )


def test_model_builder_creates_gpt2model_backbone(
    tmp_path,
):
    config = TrainingConfig(
        train_dataset_path=str(
            tmp_path / "train.jsonl"
        ),
        validation_dataset_path=str(
            tmp_path / "validation.jsonl"
        ),
        output_dir=str(tmp_path / "unused"),
        d_model=16,
        n_head=4,
        n_layer=1,
        dropout=0.0,
        max_seq_len=8,
        dim_feedforward=32,
    )

    model = build_model(config)

    assert isinstance(model.transformer, GPT2Model)
    assert model.transformer.config.n_embd == 16
    assert model.transformer.config.n_layer == 1
    assert model.transformer.config.n_head == 4


def test_training_writes_fixed_backbone_metadata_to_checkpoint(
    tmp_path,
):
    train_path = tmp_path / "train_gpt2.jsonl"
    validation_path = tmp_path / "validation_gpt2.jsonl"
    output_dir = tmp_path / "training_gpt2"

    write_dataset(
        train_path,
        game_ids=("game_train",),
        samples_per_game=1,
    )
    write_dataset(
        validation_path,
        game_ids=("game_validation",),
        samples_per_game=1,
    )

    summary = run_training(
        TrainingConfig(
            train_dataset_path=str(train_path),
            validation_dataset_path=str(validation_path),
            output_dir=str(output_dir),
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            seed=7,
            device="cpu",
            early_stopping_patience=2,
            d_model=16,
            n_head=4,
            n_layer=1,
            dropout=0.0,
            max_seq_len=8,
            dim_feedforward=32,
        )
    )

    checkpoint = load_checkpoint(
        Path(summary["best_checkpoint"])
    )

    assert summary["backbone"] == "gpt2_model"
    assert checkpoint["backbone"] == "gpt2_model"
    assert "backbone_type" not in summary["model_config"]
    assert "backbone_type" not in checkpoint["model_config"]
