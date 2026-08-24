import json

import pytest
import torch
from torch.optim import AdamW

from script.twd_tom.train import (
    MODEL_OUTPUT,
    OBJECTIVE,
    TrainingConfig,
    _forward_batch,
    _move_batch_to_device,
    build_data_loader,
    build_model,
    checkpoint_payload,
    checkpoint_task_contract,
    evaluate_model,
    train_one_epoch,
)
from werewolf.models.twd_tom.dataset import (
    CYCLIC_ROTATION_VERSION,
    MODEL_INPUT_SCOPE,
    TARGET_CONVERSION,
    TARGET_SEMANTICS,
)


def write_jsonl(path, sample):
    path.write_text(json.dumps(sample) + "\n", encoding="utf-8")


def make_config(tmp_path, dataset_path):
    return TrainingConfig(
        output_dir=str(tmp_path / "run"),
        dataset_path=str(dataset_path),
        validation_dataset_path=str(dataset_path),
        epochs=1,
        batch_size=1,
        max_seq_len=32,
        backbone="gpt2_block",
        device="cpu",
    )


def test_checkpoint_contract_is_single_belief_objective():
    contract = checkpoint_task_contract()
    assert contract == {
        "objective": OBJECTIVE,
        "model_input_scope": MODEL_INPUT_SCOPE,
        "model_output": "belief_logits",
        "output_shape": [7, 7],
        "target_semantics": TARGET_SEMANTICS,
        "target_conversion": TARGET_CONVERSION,
        "train_player_augmentation": CYCLIC_ROTATION_VERSION,
    }
    assert all("pair" not in key for key in contract)
    assert all("order" not in key for key in contract)


def test_training_loader_uses_rotation_only_for_training(
    tmp_path, training_sample_factory
):
    path = tmp_path / "data.jsonl"
    write_jsonl(path, training_sample_factory())
    config = make_config(tmp_path, path)
    _, training = build_data_loader(config, dataset_path=path, shuffle=True)
    _, validation = build_data_loader(config, dataset_path=path, shuffle=False)
    assert training.enable_cyclic_rotation is True
    assert validation.enable_cyclic_rotation is False


def test_one_train_and_evaluation_batch_use_direct_belief_contract(
    tmp_path, training_sample_factory
):
    path = tmp_path / "data.jsonl"
    write_jsonl(path, training_sample_factory())
    config = make_config(tmp_path, path)
    loader, _ = build_data_loader(config, dataset_path=path, shuffle=False)
    model = build_model(config)
    raw_batch = next(iter(loader))
    batch = _move_batch_to_device(raw_batch, torch.device("cpu"))
    output = _forward_batch(model, batch)
    assert output[MODEL_OUTPUT].shape == (1, 7, 7)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    training = train_one_epoch(
        model,
        loader,
        optimizer,
        device=torch.device("cpu"),
        gradient_clip_norm=1.0,
    )
    evaluation = evaluate_model(model, loader, device=torch.device("cpu"))
    assert training["valid_observer_count"] == 4
    assert evaluation["valid_observer_count"] == 4
    assert training["mean_loss"] > 0
    assert evaluation["mean_loss"] > 0


def test_checkpoint_payload_contains_no_removed_objective_fields(tmp_path):
    config = TrainingConfig(
        output_dir=str(tmp_path / "run"),
        dataset_path="train.jsonl",
        validation_dataset_path="validation.jsonl",
        epochs=1,
        batch_size=1,
        backbone="gpt2_block",
    )
    model = build_model(config)
    optimizer = AdamW(model.parameters())
    metrics = {"mean_loss": 1.0, "valid_observer_count": 4}
    provenance = {
        "train_dataset_path": "train.jsonl",
        "validation_dataset_path": "validation.jsonl",
        "output_dir": "run",
    }
    payload = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=1,
        train_metrics=metrics,
        validation_metrics=metrics,
        best_epoch=1,
        best_validation_mean_loss=1.0,
        run_provenance=provenance,
    )
    serialized = json.dumps(
        {key: value for key, value in payload.items() if key not in {
            "model_state_dict", "optimizer_state_dict"
        }}
    )
    for removed in ("tom_order", "pair", "second_order", "observer_pair_logits"):
        assert removed not in serialized


def test_training_config_has_no_order_argument(tmp_path):
    with pytest.raises(TypeError):
        TrainingConfig(
            tom_order=2,
            output_dir=str(tmp_path),
            dataset_path="train.jsonl",
            validation_dataset_path="validation.jsonl",
        )
