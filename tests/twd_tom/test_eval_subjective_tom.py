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


REPO_ROOT = Path(__file__).resolve().parents[2]


def make_checkpoint(tmp_path, tom_order=1):
    config = TrainingConfig(
        tom_order=tom_order,
        output_dir=str(tmp_path),
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
        metrics={"mean_loss": 1.0, "valid_subject_count": 1},
    )


def test_qwen2_checkpoint_restores_strictly(tmp_path):
    checkpoint = make_checkpoint(tmp_path)
    restored = build_model_from_checkpoint(checkpoint, device=torch.device("cpu"))
    assert isinstance(restored.transformer, Qwen2Model)
    assert restored.config.max_seq_len == 64
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


def test_one_raw_sample_can_be_evaluated_without_a_split(tmp_path):
    source = REPO_ROOT / "data" / "qwen25" / "raw_tom.jsonl"
    sample = json.loads(source.read_text(encoding="utf-8").splitlines()[0])
    dataset_path = tmp_path / "one.jsonl"
    dataset_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(make_checkpoint(tmp_path), checkpoint_path)
    summary = evaluate_checkpoint(
        EvaluationConfig(
            checkpoint_path=str(checkpoint_path),
            dataset_path=str(dataset_path),
            batch_size=1,
            device="cpu",
            allow_game_id_overlap=True,
        )
    )
    assert summary["status"] == "ok"
    assert summary["tom_order"] == 1
    assert summary["evaluation_sample_count"] == 1
    assert summary["evaluation_supervised_subject_count"] == 1
