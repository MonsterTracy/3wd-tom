"""Tests for restoring and evaluating the current Qwen2 ToM checkpoint."""

import json
from pathlib import Path

import pytest
import torch
from torch.optim import AdamW
from transformers import Qwen2Model

from script.twd_tom.eval import (
    EvaluationConfig,
    build_arg_parser,
    build_model_from_checkpoint,
    evaluate_checkpoint,
    load_checkpoint,
    resolve_training_dataset_path,
)
from script.twd_tom.train import (
    TrainingConfig,
    build_model,
    checkpoint_payload,
    sha256_file,
)
from werewolf.models.twd_tom.schema import (
    PAIR_ORDERING,
    SECOND_ORDER_OBSERVER_EVENT_CONDITIONING,
    SECOND_ORDER_OBSERVER_READOUT,
    SECOND_ORDER_SUBJECT_SUPERVISION,
    SECOND_ORDER_TARGET_ENCODING,
)
from werewolf.models.twd_tom.dataset import CYCLIC_ROTATION_VERSION
from tests.twd_tom.public_event_fixtures import make_training_sample


REPO_ROOT = Path(__file__).resolve().parents[2]


def make_checkpoint(tmp_path, tom_order=1):
    config = TrainingConfig(
        tom_order=tom_order,
        output_dir=str(tmp_path),
        dataset_path=str(tmp_path / "train.jsonl"),
        validation_dataset_path=str(tmp_path / "validation.jsonl"),
        batch_size=1,
        max_seq_len=64,
    )
    model = build_model(config)
    optimizer = AdamW(model.parameters())
    train_path = Path(config.dataset_path)
    train_sha256 = sha256_file(train_path) if train_path.is_file() else "0" * 64
    validation_path = Path(config.validation_dataset_path)
    validation_sha256 = (
        sha256_file(validation_path) if validation_path.is_file() else "0" * 64
    )
    return checkpoint_payload(
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=1,
        train_metrics={"mean_loss": 1.0, "valid_subject_count": 1},
        validation_metrics={"mean_loss": 0.5, "valid_subject_count": 1},
        best_epoch=1,
        best_validation_mean_loss=0.5,
        run_provenance={
            "git_commit_sha": "1" * 40,
            "git_worktree_clean": True,
            "train_dataset_path": "data/synthetic/train.jsonl",
            "train_dataset_sha256": train_sha256,
            "validation_dataset_path": "data/synthetic/validation.jsonl",
            "validation_dataset_sha256": validation_sha256,
            "output_dir": "outputs/synthetic",
            "python_version": "test",
            "torch_version": str(torch.__version__),
            "transformers_version": "test",
            "platform": "test",
            "requested_device": "auto",
            "resolved_device": "cpu",
            "deterministic_algorithms_enabled": True,
            "seed": 42,
        },
    )


@pytest.mark.parametrize("tom_order", [1, 2])
def test_qwen2_checkpoint_restores_strictly(tmp_path, tom_order):
    checkpoint = make_checkpoint(tmp_path, tom_order=tom_order)
    restored = build_model_from_checkpoint(checkpoint, device=torch.device("cpu"))
    assert isinstance(restored.transformer, Qwen2Model)
    assert restored.config.max_seq_len == 64
    assert restored.tom_order == tom_order
    for name, expected in checkpoint["model_state_dict"].items():
        torch.testing.assert_close(restored.state_dict()[name], expected)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "old"),
        ("tom_order", 3),
        ("model_input_scope", "wrong"),
        ("backbone", "other"),
        ("target_encoding", "wrong"),
    ],
)
def test_checkpoint_contract_mismatch_is_rejected(tmp_path, field, value):
    checkpoint = make_checkpoint(tmp_path)
    checkpoint[field] = value
    with pytest.raises(ValueError, match="checkpoint"):
        build_model_from_checkpoint(checkpoint, device=torch.device("cpu"))


def test_old_architecture_fields_are_rejected(tmp_path):
    checkpoint = make_checkpoint(tmp_path)
    checkpoint["model_config"]["d_model"] = 16
    with pytest.raises(ValueError, match="model_config"):
        build_model_from_checkpoint(checkpoint, device=torch.device("cpu"))


def test_incompatible_state_dict_is_rejected(tmp_path):
    checkpoint = make_checkpoint(tmp_path)
    checkpoint["model_state_dict"].pop("output_projection.bias")
    with pytest.raises(ValueError, match="state_dict"):
        build_model_from_checkpoint(checkpoint, device=torch.device("cpu"))


def test_new_second_order_checkpoint_has_strict_pair_contract(tmp_path):
    checkpoint = make_checkpoint(tmp_path, tom_order=2)
    assert checkpoint["target_encoding"] == SECOND_ORDER_TARGET_ENCODING
    assert checkpoint["output_class_count"] == 21
    assert checkpoint["pair_class_count"] == 21
    assert checkpoint["pair_ordering"] == PAIR_ORDERING
    assert checkpoint["observer_readout"] == SECOND_ORDER_OBSERVER_READOUT
    assert (
        checkpoint["observer_event_conditioning"]
        == SECOND_ORDER_OBSERVER_EVENT_CONDITIONING
    )
    assert (
        checkpoint["second_order_subject_supervision"]
        == SECOND_ORDER_SUBJECT_SUPERVISION
    )
    assert (
        checkpoint["train_player_augmentation"]
        == CYCLIC_ROTATION_VERSION
    )
    assert "projection_version" not in checkpoint
    assert checkpoint["model_config"]["pair_class_count"] == 21


def test_both_result_model_configs_record_pair_count(tmp_path):
    first = make_checkpoint(tmp_path / "first", tom_order=1)
    second = make_checkpoint(tmp_path / "second", tom_order=2)
    assert first["model_config"]["pair_class_count"] == 21
    assert second["output_class_count"] == 21
    assert second["model_config"]["pair_class_count"] == 21


def test_old_second_order_seven_class_checkpoint_is_rejected(tmp_path):
    checkpoint = make_checkpoint(tmp_path, tom_order=2)
    checkpoint["target_encoding"] = "classic7_player_suspicion_distribution_v1"
    checkpoint["output_class_count"] = 7
    with pytest.raises(ValueError, match="target_encoding"):
        build_model_from_checkpoint(checkpoint, device=torch.device("cpu"))


@pytest.mark.parametrize(
    "field",
    [
        "observer_readout",
        "train_player_augmentation",
        "observer_event_conditioning",
        "second_order_subject_supervision",
    ],
)
def test_old_second_order_architecture_checkpoint_is_rejected(tmp_path, field):
    checkpoint = make_checkpoint(tmp_path, tom_order=2)
    checkpoint.pop(field)
    with pytest.raises(ValueError, match=field):
        build_model_from_checkpoint(checkpoint, device=torch.device("cpu"))


@pytest.mark.parametrize(
    "field",
    ["observer_event_conditioning", "second_order_subject_supervision"],
)
def test_mismatched_second_order_evidence_contract_is_rejected(tmp_path, field):
    checkpoint = make_checkpoint(tmp_path, tom_order=2)
    checkpoint[field] = "wrong"
    with pytest.raises(ValueError, match=field):
        build_model_from_checkpoint(checkpoint, device=torch.device("cpu"))


def test_prior_public_action_supervision_checkpoint_is_rejected(tmp_path):
    checkpoint = make_checkpoint(tmp_path, tom_order=2)
    checkpoint["second_order_subject_supervision"] = (
        "prior_public_action_mask_v1"
    )
    with pytest.raises(ValueError, match="second_order_subject_supervision"):
        build_model_from_checkpoint(checkpoint, device=torch.device("cpu"))


def test_one_validation_sample_can_be_evaluated_against_explicit_training_data(
    tmp_path,
):
    train_sample = make_training_sample(1, game_id="synthetic_train")
    validation_sample = make_training_sample(1, game_id="synthetic_validation")
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(json.dumps(train_sample) + "\n", encoding="utf-8")
    dataset_path = tmp_path / "validation.jsonl"
    dataset_path.write_text(json.dumps(validation_sample) + "\n", encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint = make_checkpoint(tmp_path)
    torch.save(checkpoint, checkpoint_path)
    summary = evaluate_checkpoint(
        EvaluationConfig(
            checkpoint_path=str(checkpoint_path),
            dataset_path=str(dataset_path),
            training_dataset_path=str(train_path),
            batch_size=1,
            device="cpu",
        )
    )
    assert summary["status"] == "ok"
    assert summary["tom_order"] == 1
    assert summary["evaluation_sample_count"] == 1
    assert summary["evaluation_supervised_subject_count"] == 1


def test_checkpoint_load_has_no_unsafe_fallback(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"fixture")
    calls = []

    def reject(*args, **kwargs):
        calls.append(kwargs)
        raise TypeError("weights_only unsupported")

    monkeypatch.setattr(torch, "load", reject)
    with pytest.raises(TypeError, match="weights_only unsupported"):
        load_checkpoint(path)
    assert calls == [{"map_location": "cpu", "weights_only": True}]


def test_eval_cli_has_no_overlap_bypass():
    parser = build_arg_parser()
    args = parser.parse_args(
        ["--checkpoint", "best.pt", "--dataset", "val.jsonl"]
    )
    assert not hasattr(args, "allow_game_id_overlap")
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--checkpoint",
                "best.pt",
                "--dataset",
                "val.jsonl",
                "--allow-game-id-overlap",
            ]
        )


def test_missing_training_dataset_identity_is_rejected(tmp_path):
    checkpoint = make_checkpoint(tmp_path)
    checkpoint.pop("run_provenance")
    with pytest.raises(ValueError, match="training dataset identity"):
        resolve_training_dataset_path(checkpoint, override_path=None)


def test_training_dataset_sha_mismatch_is_rejected(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text("different\n", encoding="utf-8")
    checkpoint = make_checkpoint(tmp_path)
    checkpoint["run_provenance"]["train_dataset_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        resolve_training_dataset_path(checkpoint, override_path=str(path))


def test_evaluation_game_overlap_cannot_be_disabled(tmp_path):
    sample = make_training_sample(1, game_id="same_game")
    train_path = tmp_path / "train.jsonl"
    evaluation_path = tmp_path / "evaluation.jsonl"
    for path in (train_path, evaluation_path):
        path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
    checkpoint = make_checkpoint(tmp_path)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match=r"overlap_count=1.*same_game"):
        evaluate_checkpoint(
            EvaluationConfig(
                checkpoint_path=str(checkpoint_path),
                dataset_path=str(evaluation_path),
                training_dataset_path=str(train_path),
                device="cpu",
            )
        )
