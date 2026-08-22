import json
from copy import deepcopy

import pytest
import torch
from torch.optim import AdamW

from script.twd_tom.eval import (
    EvaluationConfig,
    build_model_from_checkpoint,
    evaluate_checkpoint,
)
from script.twd_tom.train import (
    TrainingConfig,
    build_model,
    checkpoint_payload,
    sha256_file,
)


def write_jsonl(path, sample):
    path.write_text(json.dumps(sample) + "\n", encoding="utf-8")


def make_checkpoint(tmp_path, training_path):
    config = TrainingConfig(
        output_dir=str(tmp_path / "run"),
        dataset_path=str(training_path),
        validation_dataset_path=str(training_path),
        epochs=1,
        batch_size=1,
        backbone="gpt2_block",
    )
    model = build_model(config)
    optimizer = AdamW(model.parameters())
    metrics = {"mean_loss": 1.0, "valid_observer_count": 4}
    payload = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=1,
        train_metrics=metrics,
        validation_metrics=metrics,
        best_epoch=1,
        best_validation_mean_loss=1.0,
        run_provenance={
            "train_dataset_path": "unused.jsonl",
            "train_dataset_sha256": sha256_file(training_path),
            "validation_dataset_path": "unused_validation.jsonl",
            "output_dir": "run",
        },
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(payload, checkpoint_path)
    return checkpoint_path, payload


def test_checkpoint_restores_direct_belief_model(tmp_path, training_sample_factory):
    training_path = tmp_path / "training.jsonl"
    write_jsonl(training_path, training_sample_factory(game_id="train"))
    _, checkpoint = make_checkpoint(tmp_path, training_path)
    model = build_model_from_checkpoint(checkpoint, device=torch.device("cpu"))
    assert model.output_projection.out_features == 7
    assert not hasattr(model, "tom_order")


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_output", "observer_pair_logits"),
        ("output_shape", [7, 21]),
        ("objective", "pair_prediction"),
    ],
)
def test_removed_objective_checkpoint_contract_is_rejected(
    tmp_path, training_sample_factory, field, value
):
    training_path = tmp_path / "training.jsonl"
    write_jsonl(training_path, training_sample_factory(game_id="train"))
    _, checkpoint = make_checkpoint(tmp_path, training_path)
    checkpoint[field] = value
    with pytest.raises(ValueError, match=field):
        build_model_from_checkpoint(checkpoint, device=torch.device("cpu"))


def test_evaluate_checkpoint_uses_observer_belief_targets(
    tmp_path, training_sample_factory
):
    training_path = tmp_path / "training.jsonl"
    evaluation_path = tmp_path / "evaluation.jsonl"
    write_jsonl(training_path, training_sample_factory(game_id="train"))
    evaluation_sample = deepcopy(training_sample_factory(game_id="evaluation"))
    write_jsonl(evaluation_path, evaluation_sample)
    checkpoint_path, _ = make_checkpoint(tmp_path, training_path)
    summary = evaluate_checkpoint(EvaluationConfig(
        checkpoint_path=str(checkpoint_path),
        dataset_path=str(evaluation_path),
        training_dataset_path=str(training_path),
        batch_size=1,
        device="cpu",
    ))
    assert summary["status"] == "ok"
    assert summary["model_output"] == "belief_logits"
    assert summary["evaluation_supervised_observer_count"] == 4
    assert summary["metrics"]["valid_observer_count"] == 4
    serialized = json.dumps(summary)
    assert "tom_order" not in serialized
    assert "pair" not in serialized


def test_evaluation_rejects_game_overlap(tmp_path, training_sample_factory):
    training_path = tmp_path / "training.jsonl"
    evaluation_path = tmp_path / "evaluation.jsonl"
    sample = training_sample_factory(game_id="same")
    write_jsonl(training_path, sample)
    write_jsonl(evaluation_path, sample)
    checkpoint_path, _ = make_checkpoint(tmp_path, training_path)
    with pytest.raises(ValueError, match="disjoint"):
        evaluate_checkpoint(EvaluationConfig(
            checkpoint_path=str(checkpoint_path),
            dataset_path=str(evaluation_path),
            training_dataset_path=str(training_path),
            device="cpu",
        ))
