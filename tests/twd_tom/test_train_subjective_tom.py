"""Tests for the single raw-data Qwen2 training entry."""

from pathlib import Path

import pytest
import torch
from transformers import Qwen2Model

from script.twd_tom.train import (
    RAW_DATASET_PATHS,
    TrainingConfig,
    _forward_batch,
    _move_batch_to_device,
    build_arg_parser,
    build_data_loader,
    build_model,
)
from werewolf.models.twd_tom.losses import masked_pair_cross_entropy


@pytest.mark.parametrize(
    ("tom_order", "filename"), [(1, "raw_tom.jsonl"), (2, "raw_tom2.jsonl")]
)
def test_tom_order_selects_the_current_raw_file(tmp_path, tom_order, filename):
    config = TrainingConfig(tom_order=tom_order, output_dir=str(tmp_path))
    assert config.resolved_dataset_path == RAW_DATASET_PATHS[tom_order]
    assert config.resolved_dataset_path.name == filename
    assert config.run_output_dir.name == f"tom_order_{tom_order}"


def test_cli_has_one_tom_order_entry_and_no_gpt2_architecture_options():
    parser = build_arg_parser()
    args = parser.parse_args(["--tom-order", "1", "--output-dir", "output"])
    assert args.tom_order == 1
    for old_name in ("d_model", "n_head", "n_layer", "dropout", "dim_feedforward"):
        assert not hasattr(args, old_name)


def test_model_builder_uses_fixed_qwen2_configuration(tmp_path):
    model = build_model(TrainingConfig(tom_order=1, output_dir=str(tmp_path)))
    assert isinstance(model.transformer, Qwen2Model)
    assert model.transformer.config.hidden_size == 256
    assert model.transformer.config.num_hidden_layers == 4


@pytest.mark.parametrize("tom_order", [1, 2])
def test_complete_dataset_loads_and_one_batch_forward_loss_backward(
    tmp_path, tom_order
):
    config = TrainingConfig(
        tom_order=tom_order,
        output_dir=str(tmp_path),
        batch_size=1,
        device="cpu",
        max_seq_len=64,
    )
    loader, dataset = build_data_loader(config, shuffle=False)
    assert len(dataset) == 4535
    raw_batch = next(iter(loader))
    if tom_order == 1:
        assert "known_werewolves" in raw_batch
    else:
        assert "known_werewolves" not in raw_batch

    model = build_model(config)
    batch = _move_batch_to_device(raw_batch, torch.device("cpu"))
    output = _forward_batch(model, batch)
    loss = masked_pair_cross_entropy(
        output["pair_logits"], batch["pair_targets"], batch["subject_mask"]
    )
    loss.backward()
    assert output["pair_logits"].shape == (1, 7, 21)
    assert torch.isfinite(loss)
    assert model.output_projection.weight.grad is not None


@pytest.mark.parametrize("value", [0, 3, True, None, "1"])
def test_invalid_tom_order_is_rejected(tmp_path, value):
    with pytest.raises(ValueError, match="tom_order"):
        TrainingConfig(tom_order=value, output_dir=str(tmp_path))


def test_default_dataset_paths_exist():
    assert all(Path(path).is_file() for path in RAW_DATASET_PATHS.values())
