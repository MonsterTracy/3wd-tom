"""Tests for restoring and evaluating the current Qwen2 ToM checkpoint."""

import json
from pathlib import Path

import pytest
import torch
from torch.optim import AdamW
from transformers import Qwen2Model

from script.twd_tom.eval import (
    EvaluationConfig,
    build_model_from_checkpoint,
    evaluate_checkpoint,
)
from script.twd_tom.train import TrainingConfig, build_model, checkpoint_payload
from werewolf.models.twd_tom.dataset import TOM_INPUT_SCOPES
from werewolf.models.twd_tom.schema import SUSPICION_TARGET_ENCODING


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
    return checkpoint_payload(
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=1,
        train_metrics={"mean_loss": 1.0, "valid_subject_count": 1},
        validation_metrics={"mean_loss": 0.5, "valid_subject_count": 1},
        best_epoch=1,
        best_validation_mean_loss=0.5,
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


def test_new_second_order_checkpoint_has_strict_suspicion_contract(tmp_path):
    checkpoint = make_checkpoint(tmp_path, tom_order=2)
    assert checkpoint["target_encoding"] == SUSPICION_TARGET_ENCODING
    assert checkpoint["output_class_count"] == 7
    assert tuple(checkpoint["canonical_player_ordering"]) == tuple(
        f"player{index}" for index in range(1, 8)
    )
    assert "pair_class_count" not in checkpoint
    assert "pair_ordering" not in checkpoint
    assert "projection_version" not in checkpoint
    assert "pair_class_count" not in checkpoint["model_config"]


def test_order_specific_result_model_config_excludes_second_order_pair_count(
    tmp_path,
):
    first = make_checkpoint(tmp_path / "first", tom_order=1)
    second = make_checkpoint(tmp_path / "second", tom_order=2)
    assert first["model_config"]["pair_class_count"] == 21
    assert second["output_class_count"] == 7
    assert "pair_class_count" not in second["model_config"]


def test_old_second_order_pair_checkpoint_is_rejected(tmp_path):
    checkpoint = make_checkpoint(tmp_path, tom_order=1)
    checkpoint["tom_order"] = 2
    checkpoint["model_input_scope"] = TOM_INPUT_SCOPES[2]
    with pytest.raises(ValueError, match="target_encoding"):
        build_model_from_checkpoint(checkpoint, device=torch.device("cpu"))


def test_one_validation_sample_can_be_evaluated_against_explicit_training_data(
    tmp_path,
):
    train_source = REPO_ROOT / "data" / "qwen25" / "tom1" / "train.jsonl"
    validation_source = REPO_ROOT / "data" / "qwen25" / "tom1" / "val.jsonl"
    with train_source.open(encoding="utf-8") as handle:
        train_sample = json.loads(next(handle))
    with validation_source.open(encoding="utf-8") as handle:
        validation_sample = json.loads(next(handle))
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
            batch_size=1,
            device="cpu",
        )
    )
    assert summary["status"] == "ok"
    assert summary["tom_order"] == 1
    assert summary["evaluation_sample_count"] == 1
    assert summary["evaluation_supervised_subject_count"] == 1
