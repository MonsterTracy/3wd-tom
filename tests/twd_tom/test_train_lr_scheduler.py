"""Tests for ToM learning-rate scheduling."""

import pytest
import torch
from torch.optim import AdamW

from script.twd_tom.train import (
    TrainingConfig,
    build_arg_parser,
    build_learning_rate_scheduler,
)


def make_config(**overrides):
    values = {
        "tom_order": 2,
        "output_dir": "outputs/test",
        "dataset_path": "data/train.jsonl",
        "validation_dataset_path": "data/val.jsonl",
        "epochs": 2,
        "batch_size": 1,
        "learning_rate": 3e-4,
    }
    values.update(overrides)
    return TrainingConfig(**values)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"lr_scheduler": "invalid"}, "lr_scheduler"),
        ({"warmup_ratio": -0.1}, "warmup_ratio"),
        ({"warmup_ratio": 1.0}, "warmup_ratio"),
        ({"min_learning_rate": -1e-5}, "min_learning_rate"),
        ({"min_learning_rate": 4e-4}, "min_learning_rate"),
    ],
)
def test_scheduler_config_validation(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_config(**overrides)


def test_scheduler_cli_arguments():
    args = build_arg_parser().parse_args(
        [
            "--tom-order", "2",
            "--output-dir", "outputs/run",
            "--dataset", "data/train.jsonl",
            "--validation-dataset", "data/val.jsonl",
            "--lr-scheduler", "warmup_cosine",
            "--warmup-ratio", "0.05",
            "--min-learning-rate", "3e-5",
        ]
    )

    assert args.lr_scheduler == "warmup_cosine"
    assert args.warmup_ratio == pytest.approx(0.05)
    assert args.min_learning_rate == pytest.approx(3e-5)


def test_constant_scheduler_preserves_learning_rate():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = AdamW([parameter], lr=3e-4)

    scheduler, metadata = build_learning_rate_scheduler(
        optimizer,
        config=make_config(lr_scheduler="constant"),
        steps_per_epoch=5,
    )

    assert scheduler is None
    assert metadata["name"] == "constant"
    assert metadata["total_steps"] == 10
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)


def test_warmup_cosine_reaches_peak_and_minimum():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = AdamW([parameter], lr=3e-4)
    config = make_config(
        epochs=2,
        lr_scheduler="warmup_cosine",
        warmup_ratio=0.2,
        min_learning_rate=3e-5,
    )

    scheduler, metadata = build_learning_rate_scheduler(
        optimizer,
        config=config,
        steps_per_epoch=5,
    )

    observed = [float(optimizer.param_groups[0]["lr"])]

    for _ in range(metadata["total_steps"]):
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        scheduler.step()
        observed.append(float(optimizer.param_groups[0]["lr"]))

    assert metadata["warmup_steps"] == 2
    assert observed[0] < config.learning_rate
    assert max(observed) == pytest.approx(config.learning_rate)
    assert observed[-1] == pytest.approx(
        config.min_learning_rate,
        rel=1e-6,
        abs=1e-10,
    )
