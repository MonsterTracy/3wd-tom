"""Tests for the explicit train/validation Qwen2 training entry."""

import inspect
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from transformers import Qwen2Model

import script.twd_tom.train as train_module
from script.twd_tom.eval import (
    EvaluationConfig,
    build_model_from_checkpoint,
    evaluate_checkpoint,
)
from script.twd_tom.train import (
    TrainingConfig,
    _forward_batch,
    _move_batch_to_device,
    build_arg_parser,
    build_data_loader,
    build_model,
    build_training_data_loaders,
    evaluate_model,
    run_training,
)
from werewolf.models.twd_tom.losses import masked_distribution_cross_entropy
from werewolf.models.twd_tom.schema import SUSPICION_TARGET_ENCODING


REPO_ROOT = Path(__file__).resolve().parents[2]


def _source_sample(tom_order: int, split: str) -> dict:
    path = REPO_ROOT / "data" / "qwen25" / f"tom{tom_order}" / f"{split}.jsonl"
    with path.open(encoding="utf-8") as handle:
        return json.loads(next(handle))


def _write_sample(path: Path, sample: dict) -> None:
    path.write_text(json.dumps(sample) + "\n", encoding="utf-8")


def _training_config(
    tmp_path: Path,
    tom_order: int,
    *,
    train_sample: dict | None = None,
    validation_sample: dict | None = None,
    epochs: int = 1,
) -> TrainingConfig:
    train_path = tmp_path / f"tom{tom_order}_train.jsonl"
    validation_path = tmp_path / f"tom{tom_order}_validation.jsonl"
    _write_sample(train_path, train_sample or _source_sample(tom_order, "train"))
    _write_sample(
        validation_path,
        validation_sample or _source_sample(tom_order, "val"),
    )
    return TrainingConfig(
        tom_order=tom_order,
        output_dir=str(tmp_path / "outputs"),
        dataset_path=str(train_path),
        validation_dataset_path=str(validation_path),
        epochs=epochs,
        batch_size=1,
        device="cpu",
        max_seq_len=32,
    )


def test_cli_requires_explicit_train_and_validation_datasets():
    parser = build_arg_parser()
    required = ["--tom-order", "1", "--output-dir", "output"]
    with pytest.raises(SystemExit):
        parser.parse_args(
            [*required, "--validation-dataset", "val.jsonl"]
        )
    with pytest.raises(SystemExit):
        parser.parse_args([*required, "--dataset", "train.jsonl"])

    args = parser.parse_args(
        [
            *required,
            "--dataset",
            "train.jsonl",
            "--validation-dataset",
            "val.jsonl",
        ]
    )
    assert args.dataset == "train.jsonl"
    assert args.validation_dataset == "val.jsonl"
    assert not hasattr(args, "test_dataset")
    for old_name in ("d_model", "n_head", "n_layer", "dropout", "dim_feedforward"):
        assert not hasattr(args, old_name)

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                *required,
                "--dataset",
                "train.jsonl",
                "--validation-dataset",
                "val.jsonl",
                "--test-dataset",
                "test.jsonl",
            ]
        )


def test_model_builder_uses_fixed_qwen2_configuration(tmp_path):
    model = build_model(_training_config(tmp_path, 1))
    assert isinstance(model.transformer, Qwen2Model)
    assert model.transformer.config.hidden_size == 256
    assert model.transformer.config.num_hidden_layers == 4


def test_train_and_validation_order_mismatch_is_rejected(tmp_path):
    config = _training_config(
        tmp_path,
        1,
        validation_sample=_source_sample(2, "val"),
    )
    with pytest.raises(ValueError, match="tom_order"):
        build_training_data_loaders(config)


def test_train_and_validation_scope_mismatch_is_rejected(tmp_path):
    validation_sample = deepcopy(_source_sample(1, "val"))
    validation_sample["model_input_scope"] = "public_events_only"
    config = _training_config(
        tmp_path,
        1,
        validation_sample=validation_sample,
    )
    with pytest.raises(ValueError, match="model_input_scope"):
        build_training_data_loaders(config)


def test_train_and_validation_game_overlap_is_rejected(tmp_path):
    train_sample = _source_sample(1, "train")
    config = _training_config(
        tmp_path,
        1,
        train_sample=train_sample,
        validation_sample=deepcopy(train_sample),
    )
    with pytest.raises(ValueError, match=r"overlap: count=1.*game_"):
        build_training_data_loaders(config)


def test_train_and_validation_loader_shuffle_contract(tmp_path):
    config = _training_config(tmp_path, 1)
    train_loader, _, validation_loader, _ = build_training_data_loaders(config)
    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(validation_loader.sampler, SequentialSampler)


@pytest.mark.parametrize("empty_split", ["train", "validation"])
def test_train_and_validation_must_be_non_empty(tmp_path, empty_split):
    config = _training_config(tmp_path, 1)
    path = (
        config.resolved_dataset_path
        if empty_split == "train"
        else config.resolved_validation_dataset_path
    )
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="dataset cannot be empty"):
        build_training_data_loaders(config)


def test_validation_does_not_change_weights_or_create_gradients(tmp_path):
    config = _training_config(tmp_path, 1)
    validation_loader, _ = build_data_loader(
        config,
        dataset_path=config.resolved_validation_dataset_path,
        shuffle=False,
    )
    model = build_model(config)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    metrics = evaluate_model(model, validation_loader, device=torch.device("cpu"))
    assert metrics["valid_subject_count"] == 1
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, expected in before.items():
        torch.testing.assert_close(model.state_dict()[name], expected)


def test_each_epoch_validates_and_equal_loss_keeps_earlier_best(tmp_path, monkeypatch):
    config = _training_config(tmp_path, 1, epochs=3)
    train_epochs = []
    validation_losses = iter((2.0, 1.0, 1.0))
    validation_calls = []

    def fake_train_one_epoch(model, data_loader, optimizer, **kwargs):
        epoch = len(train_epochs) + 1
        train_epochs.append(epoch)
        with torch.no_grad():
            model.output_projection.weight.fill_(float(epoch))
        return {"mean_loss": float(epoch), "valid_subject_count": 1}

    def fake_evaluate_model(model, data_loader, **kwargs):
        loss = next(validation_losses)
        validation_calls.append(loss)
        return {"mean_loss": loss, "valid_subject_count": 1}

    monkeypatch.setattr(train_module, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(train_module, "evaluate_model", fake_evaluate_model)
    summary = run_training(config)

    assert train_epochs == [1, 2, 3]
    assert validation_calls == [2.0, 1.0, 1.0]
    assert summary["best_epoch"] == 2
    assert summary["best_validation_mean_loss"] == 1.0

    best = torch.load(config.run_output_dir / "best.pt", map_location="cpu", weights_only=True)
    last = torch.load(config.run_output_dir / "last.pt", map_location="cpu", weights_only=True)
    assert best["epoch"] == 2
    assert best["selection_metric_value"] == 1.0
    assert last["epoch"] == 3
    assert last["best_epoch"] == 2
    assert torch.all(best["model_state_dict"]["output_projection.weight"] == 2.0)
    assert torch.all(last["model_state_dict"]["output_projection.weight"] == 3.0)

    history = json.loads((config.run_output_dir / "history.json").read_text())
    assert [entry["is_best"] for entry in history] == [True, True, False]
    assert [entry["best_epoch"] for entry in history] == [1, 2, 2]
    assert all("train" in entry and "validation" in entry for entry in history)


@pytest.mark.parametrize("tom_order", [1, 2])
def test_one_batch_train_validation_smoke_and_best_eval(tmp_path, tom_order):
    config = _training_config(tmp_path, tom_order)
    summary = run_training(config)
    output_files = {path.name for path in config.run_output_dir.iterdir()}
    assert output_files == {"best.pt", "last.pt", "history.json", "summary.json"}
    assert summary["best_epoch"] == 1
    assert summary["epochs_completed"] == 1
    assert summary["selection_metric_name"] == "validation_mean_loss"
    saved_summary = json.loads(
        (config.run_output_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert saved_summary["best_epoch"] == 1
    assert saved_summary["best_validation_mean_loss"] == summary[
        "best_validation_mean_loss"
    ]
    assert "final_train_metrics" in saved_summary
    assert "final_validation_metrics" in saved_summary

    best = torch.load(config.run_output_dir / "best.pt", map_location="cpu", weights_only=True)
    last = torch.load(config.run_output_dir / "last.pt", map_location="cpu", weights_only=True)
    assert best["epoch"] == last["epoch"] == 1
    assert best["train_dataset_path"] == str(config.resolved_dataset_path.resolve())
    assert best["validation_dataset_path"] == str(
        config.resolved_validation_dataset_path.resolve()
    )
    assert best["selection_metric_name"] == "validation_mean_loss"
    assert best["validation_metrics"]["mean_loss"] == best["selection_metric_value"]
    for name, expected in best["model_state_dict"].items():
        torch.testing.assert_close(last["model_state_dict"][name], expected)
    restored_last = build_model_from_checkpoint(last, device=torch.device("cpu"))
    assert isinstance(restored_last.transformer, Qwen2Model)

    evaluation = evaluate_checkpoint(
        EvaluationConfig(
            checkpoint_path=str(config.run_output_dir / "best.pt"),
            dataset_path=str(config.resolved_validation_dataset_path),
            batch_size=1,
            device="cpu",
        )
    )
    assert evaluation["status"] == "ok"
    assert evaluation["tom_order"] == tom_order
    assert evaluation["evaluation_sample_count"] == 1
    if tom_order == 1:
        assert best["pair_class_count"] == 21
        assert "output_class_count" not in best
        assert "mean_pair_cross_entropy" in evaluation["metrics"]
    else:
        assert best["target_encoding"] == SUSPICION_TARGET_ENCODING
        assert best["output_class_count"] == 7
        assert tuple(best["canonical_player_ordering"]) == tuple(
            f"player{index}" for index in range(1, 8)
        )
        assert "pair_class_count" not in best
        assert "pair_ordering" not in best
        assert "projection_version" not in best
        assert "pair_class_count" not in best["model_config"]
        assert set(evaluation["metrics"]) >= {
            "mean_suspicion_cross_entropy",
            "mean_suspicion_kl_divergence",
            "mean_suspicion_total_variation",
            "mean_suspicion_mae",
        }
        assert not any("pair" in name for name in evaluation["metrics"])


@pytest.mark.parametrize("tom_order", [1, 2])
def test_one_batch_forward_backward_uses_only_soft_target_cross_entropy(
    tmp_path,
    tom_order,
):
    config = _training_config(tmp_path, tom_order)
    loader, _ = build_data_loader(
        config,
        dataset_path=config.resolved_dataset_path,
        shuffle=False,
    )
    raw_batch = next(iter(loader))
    model = build_model(config)
    batch = _move_batch_to_device(raw_batch, torch.device("cpu"))
    output = _forward_batch(model, batch)
    logits_name = (
        "observer_pair_logits" if tom_order == 1 else "observer_suspicion_logits"
    )
    target_name = "pair_targets" if tom_order == 1 else "suspicion_targets"
    loss = masked_distribution_cross_entropy(
        output[logits_name], batch[target_name], batch["subject_mask"]
    )
    loss.backward()
    assert output[logits_name].shape == (1, 7, 21 if tom_order == 1 else 7)
    assert torch.isfinite(loss)
    assert model.output_projection.weight.grad is not None
    source = inspect.getsource(train_module.train_one_epoch)
    assert source.count("masked_distribution_cross_entropy") == 1
    assert "masked_pair_cross_entropy" not in source


@pytest.mark.parametrize("value", [0, 3, True, None, "1"])
def test_invalid_tom_order_is_rejected(tmp_path, value):
    with pytest.raises(ValueError, match="tom_order"):
        TrainingConfig(
            tom_order=value,
            output_dir=str(tmp_path),
            dataset_path="train.jsonl",
            validation_dataset_path="val.jsonl",
        )
