"""Tests for the explicit train/validation Qwen2 training entry."""

import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler, Subset
from transformers import Qwen2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

import script.twd_tom.train as train_module
from script.twd_tom.eval import (
    EvaluationConfig,
    build_model_from_checkpoint,
    evaluate_checkpoint,
)
from script.twd_tom.train import (
    CANONICAL_D_TRAINING_INTEGRATION_VERSION,
    TOM2_TARGET_SEMANTICS,
    TOM2_TEMPORAL_SUPERVISION_POLICY,
    TRAINING_MANIFEST_SCHEMA_VERSION,
    TrainingConfig,
    _atomic_json_write,
    _atomic_torch_save,
    _effective_subject_mask,
    _forward_batch,
    _move_batch_to_device,
    _prepare_run_output_dir,
    _targets_for_loss,
    build_arg_parser,
    build_data_loader,
    build_model,
    build_run_provenance,
    load_canonical_d_split_manifest,
    build_training_data_loaders,
    evaluate_model,
    run_training,
    set_random_seed,
    sha256_file,
)
from werewolf.models.twd_tom.losses import masked_distribution_cross_entropy
from werewolf.models.twd_tom.belief_backbone import (
    GPT2_BLOCK_BACKBONE_NAME,
    GPT2BlockStack,
    QWEN2_BACKBONE_NAME,
    ToMBeliefBackbone,
    ToMBeliefBackboneConfig,
)
from werewolf.models.twd_tom.dataset import (
    CYCLIC_ROTATION_VERSION,
    second_order_effective_subject_mask,
)
from werewolf.models.twd_tom.schema import (
    PUBLIC_ONLY_BELIEF_INFORMATION_SCOPE,
    PUBLIC_ONLY_FORMAL_ANNOTATION_SCHEMA_VERSION,
    PUBLIC_ONLY_FORMAL_LABEL_PROVENANCE,
    PUBLIC_ONLY_LABEL_PROVENANCE,
    PUBLIC_ONLY_MODEL_INPUT_SCOPE,
    PUBLIC_ONLY_PRIVATE_FIELDS_USAGE,
    SECOND_ORDER_OBSERVER_EVENT_CONDITIONING,
    SECOND_ORDER_OBSERVER_READOUT,
    SECOND_ORDER_SUBJECT_SUPERVISION,
    SECOND_ORDER_TARGET_ENCODING,
)
from werewolf.models.twd_tom.samples import PUBLIC_ONLY_SAMPLE_SCHEMA_VERSION
from script.twd_tom.split_offline_d_training_data import (
    SPLIT_MANIFEST_SCHEMA_VERSION,
    SPLIT_POLICY_VERSION,
    split_offline_d_training_data,
)
from werewolf.offline_materialization import (
    D_MATERIALIZATION_POLICY_VERSION,
    D_SCHEMA_VERSION,
    OFFLINE_PRIVATE_CONDITIONED_TOM1_TASK,
    OFFLINE_PUBLIC_ONLY_TOM2_TASK,
)
from werewolf.trajectory import canonical_digest, canonical_json
from tests.twd_tom.test_twd_tom_dataset import d_sample
from tests.twd_tom.public_event_fixtures import (
    make_public_only_training_sample,
    make_training_sample,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
COLLAPSE_DIAGNOSTIC_KEYS = {
    "mean_target_pair_entropy",
    "mean_predicted_pair_entropy",
    "mean_target_pair_top1_top2_margin",
    "mean_predicted_pair_top1_top2_margin",
    "mean_target_marginal_spread",
    "mean_predicted_marginal_spread",
    "mean_target_observer_pairwise_tv",
    "mean_predicted_observer_pairwise_tv",
}


def _source_sample(tom_order: int, split: str) -> dict:
    return make_training_sample(
        tom_order,
        game_id=f"synthetic_tom{tom_order}_{split}",
    )


def _source_sample_without_latest_action(split: str) -> dict:
    return make_training_sample(
        2,
        game_id=f"synthetic_tom2_{split}_without_action",
        with_latest_action=False,
    )


@pytest.fixture(autouse=True)
def synthetic_run_provenance(monkeypatch):
    def build(config, *, resolved_device):
        if config.resolved_split_manifest_path is not None:
            manifest_path = config.resolved_split_manifest_path.resolve()
            manifest = load_canonical_d_split_manifest(manifest_path)
            manifest_root = manifest_path.parent
            train_dataset_path = config.resolved_dataset_path.resolve().relative_to(
                manifest_root
            ).as_posix()
            validation_dataset_path = (
                config.resolved_validation_dataset_path.resolve().relative_to(
                    manifest_root
                ).as_posix()
            )
            return {
                "git_commit_sha": "1" * 40,
                "git_worktree_clean": True,
                "train_dataset_path": train_dataset_path,
                "train_dataset_sha256": sha256_file(config.resolved_dataset_path),
                "validation_dataset_path": validation_dataset_path,
                "validation_dataset_sha256": sha256_file(
                    config.resolved_validation_dataset_path
                ),
                "output_dir": str(Path(config.output_dir).resolve()),
                "python_version": "test",
                "torch_version": str(torch.__version__),
                "transformers_version": "test",
                "platform": "test",
                "requested_device": config.device,
                "resolved_device": str(resolved_device),
                "deterministic_algorithms_enabled": True,
                "seed": config.seed,
                "split_manifest_path": "manifest.json",
                "split_manifest_sha256": sha256_file(manifest_path),
                "split_manifest_digest": manifest["manifest_digest"],
                "split_manifest_schema_version": manifest["schema_version"],
                "split_policy_version": manifest["split_policy_version"],
                "split_seed": manifest["split_seed"],
            }
        return {
            "git_commit_sha": "1" * 40,
            "git_worktree_clean": True,
            "train_dataset_path": "tests/fixtures/synthetic_train.jsonl",
            "train_dataset_sha256": sha256_file(config.resolved_dataset_path),
            "validation_dataset_path": "tests/fixtures/synthetic_val.jsonl",
            "validation_dataset_sha256": sha256_file(
                config.resolved_validation_dataset_path
            ),
            "output_dir": "outputs/tests",
            "python_version": "test",
            "torch_version": str(torch.__version__),
            "transformers_version": "test",
            "platform": "test",
            "requested_device": config.device,
            "resolved_device": str(resolved_device),
            "deterministic_algorithms_enabled": True,
            "seed": config.seed,
        }

    monkeypatch.setattr(train_module, "build_run_provenance", build)


def _write_sample(path: Path, sample: dict) -> None:
    path.write_text(json.dumps(sample) + "\n", encoding="utf-8")


def _write_samples(path: Path, samples: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples),
        encoding="utf-8",
    )


def _training_config(
    tmp_path: Path,
    tom_order: int,
    *,
    train_sample: dict | None = None,
    validation_sample: dict | None = None,
    epochs: int = 1,
    backbone: str = QWEN2_BACKBONE_NAME,
    split_manifest_path: str | None = None,
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
        split_manifest_path=split_manifest_path,
        backbone=backbone,
        epochs=epochs,
        batch_size=1,
        device="cpu",
        max_seq_len=32,
    )


def _redigest_d_record(record: dict, *, game_id: str, step_idx: int) -> dict:
    value = deepcopy(record)
    value["game_id"] = game_id
    value["step_idx"] = step_idx
    value["label_cutoff_step_idx"] = step_idx
    value["boundary_id"] = f"{game_id}:step_{step_idx:06d}:PRE_PUBLIC_SPEECH"
    value.pop("record_digest", None)
    value["record_digest"] = canonical_digest(value)
    return value


def _canonical_d_split_fixture(tmp_path: Path):
    game_ids = ["d_game_001", "d_game_002", "d_game_003"]
    tom1_rows = [
        _redigest_d_record(d_sample(1), game_id=game_id, step_idx=10 + index)
        for index, game_id in enumerate(game_ids)
    ]
    tom2_rows = []
    for index, game_id in enumerate(game_ids):
        base_step = 20 + index * 2
        tom2_rows.append(
            _redigest_d_record(
                d_sample(2, with_latest_action=False),
                game_id=game_id,
                step_idx=base_step,
            )
        )
        tom2_rows.append(
            _redigest_d_record(
                d_sample(2, with_latest_action=True),
                game_id=game_id,
                step_idx=base_step + 1,
            )
        )
    tom1_source = tmp_path / "source_tom1.jsonl"
    tom2_source = tmp_path / "source_tom2.jsonl"
    tom1_source.write_text(
        "".join(canonical_json(row) + "\n" for row in tom1_rows),
        encoding="utf-8",
    )
    tom2_source.write_text(
        "".join(canonical_json(row) + "\n" for row in tom2_rows),
        encoding="utf-8",
    )
    split_root = tmp_path / "canonical_split"
    manifest = split_offline_d_training_data(
        tom1_path=tom1_source,
        tom2_path=tom2_source,
        output_dir=split_root,
        split_seed=17,
        train_game_count=1,
        validation_game_count=1,
        test_game_count=1,
    )
    return split_root, manifest


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
    assert args.backbone == QWEN2_BACKBONE_NAME
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

    gpt2_args = parser.parse_args(
        [
            *required,
            "--dataset",
            "train.jsonl",
            "--validation-dataset",
            "val.jsonl",
            "--backbone",
            GPT2_BLOCK_BACKBONE_NAME,
        ]
    )
    assert gpt2_args.backbone == GPT2_BLOCK_BACKBONE_NAME


def test_model_builder_uses_fixed_qwen2_configuration(tmp_path):
    model = build_model(_training_config(tmp_path, 1))
    assert model.backbone_name == QWEN2_BACKBONE_NAME
    assert isinstance(model.transformer, Qwen2Model)
    assert model.transformer.config.hidden_size == 256
    assert model.transformer.config.num_hidden_layers == 4


def test_model_builder_uses_direct_gpt2_block_stack(tmp_path):
    model = build_model(
        _training_config(
            tmp_path,
            1,
            backbone=GPT2_BLOCK_BACKBONE_NAME,
        )
    )
    assert model.backbone_name == GPT2_BLOCK_BACKBONE_NAME
    assert isinstance(model.transformer, GPT2BlockStack)
    assert len(model.transformer.blocks) == 4
    assert all(
        isinstance(block, GPT2Block)
        for block in model.transformer.blocks
    )


def test_gpt2_block_stack_is_causal():
    torch.manual_seed(7)
    stack = GPT2BlockStack(max_seq_len=8).eval()
    hidden = torch.randn(1, 4, 256)
    changed_future = hidden.clone()
    changed_future[:, 3] += 10.0
    mask = torch.ones((1, 4), dtype=torch.bool)
    with torch.no_grad():
        original = stack(hidden, attention_mask=mask)
        changed = stack(changed_future, attention_mask=mask)
    torch.testing.assert_close(original[:, :3], changed[:, :3])


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
    with pytest.raises(ValueError, match=r"overlap: count=1.*synthetic_tom1_train"):
        build_training_data_loaders(config)


def test_train_and_validation_loader_shuffle_contract(tmp_path):
    config = _training_config(tmp_path, 1)
    train_loader, _, validation_loader, _ = build_training_data_loaders(config)
    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(validation_loader.sampler, SequentialSampler)


def test_first_order_loader_does_not_call_second_order_prefilter(
    tmp_path,
    monkeypatch,
):
    config = _training_config(tmp_path, 1)

    def fail_if_called(_dataset):
        raise AssertionError("first-order loader called second-order prefilter")

    monkeypatch.setattr(
        train_module.TWDToMDataset,
        "second_order_supervised_indices",
        fail_if_called,
    )
    loader, _dataset = build_data_loader(
        config,
        dataset_path=config.resolved_dataset_path,
        shuffle=True,
    )
    assert next(iter(loader))["subject_mask"].any()


@pytest.mark.parametrize(
    ("tom_order", "public_only", "expects_private"),
    [(1, False, True), (1, True, False), (2, True, False)],
)
def test_forwarding_respects_validated_information_scope(
    tmp_path,
    tom_order,
    public_only,
    expects_private,
):
    sample = (
        make_public_only_training_sample(tom_order)
        if public_only
        else make_training_sample(tom_order)
    )
    config = _training_config(
        tmp_path,
        tom_order,
        train_sample=sample,
        validation_sample=deepcopy(sample) | {"game_id": "validation-game"},
    )
    loader, _dataset = build_data_loader(
        config,
        dataset_path=config.resolved_dataset_path,
        shuffle=False,
    )

    class CapturingModel:
        def __call__(self, **kwargs):
            self.arguments = kwargs
            return {}

    model = CapturingModel()
    _forward_batch(model, next(iter(loader)))
    assert ("known_werewolves" in model.arguments) is expects_private
    assert ("known_non_werewolves" in model.arguments) is expects_private


def test_second_order_rotation_is_enabled_only_for_training_loader(tmp_path):
    config = _training_config(tmp_path, 2)
    train_loader, train_dataset, validation_loader, validation_dataset = (
        build_training_data_loaders(config)
    )
    assert train_dataset.enable_cyclic_rotation is True
    assert validation_dataset.enable_cyclic_rotation is False
    assert train_dataset.augmentation_seed == config.seed
    assert isinstance(train_loader.dataset, Subset)
    assert isinstance(validation_loader.dataset, Subset)
    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(validation_loader.sampler, SequentialSampler)
    validation_before = validation_dataset[0]
    validation_dataset.set_epoch(5)
    validation_after = validation_dataset[0]
    assert torch.equal(
        validation_before["pair_targets"],
        validation_after["pair_targets"],
    )
    assert validation_before["metadata"] == validation_after["metadata"]


@pytest.mark.parametrize("shuffle", [False, True])
def test_second_order_loader_keeps_only_completed_speech_boundaries(
    tmp_path,
    shuffle,
):
    config = _training_config(tmp_path, 2)
    path = tmp_path / f"filter-{shuffle}.jsonl"
    _write_samples(
        path,
        [
            _source_sample_without_latest_action("train"),
            _source_sample(2, "train"),
        ],
    )
    loader, dataset = build_data_loader(
        config,
        dataset_path=path,
        shuffle=shuffle,
    )
    assert len(dataset) == 2
    assert isinstance(loader.dataset, Subset)
    assert tuple(loader.dataset.indices) == (1,)
    batch = next(iter(loader))
    assert batch["post_completed_public_speech_pre_next_action"].all()
    assert second_order_effective_subject_mask(
        batch["subject_mask"],
        batch["reasoning_player_id"],
    ).any()


@pytest.mark.parametrize(
    ("split", "shuffle"),
    [("train", True), ("val", False)],
)
def test_formal_second_order_loader_reads_first_batch(
    tmp_path,
    split,
    shuffle,
    require_real_twd_tom_data,
):
    formal_path = REPO_ROOT / "data" / "qwen25" / "tom2" / f"{split}.jsonl"
    require_real_twd_tom_data(formal_path)
    config = TrainingConfig(
        tom_order=2,
        output_dir=str(tmp_path / "unused-output"),
        dataset_path=str(formal_path),
        validation_dataset_path=str(formal_path),
        batch_size=32,
        device="cpu",
    )
    loader, dataset = build_data_loader(
        config,
        dataset_path=formal_path,
        shuffle=shuffle,
    )
    batch = next(iter(loader))
    assert len(dataset) > 0
    assert batch["subject_mask"].shape[1:] == (7,)
    assert batch["post_completed_public_speech_pre_next_action"].all()
    assert second_order_effective_subject_mask(
        batch["subject_mask"],
        batch["reasoning_player_id"],
    ).any(dim=1).all()


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
    dataset_epochs = []
    validation_losses = iter((2.0, 1.0, 1.0))
    validation_calls = []
    original_set_epoch = train_module.TWDToMDataset.set_epoch

    def capture_set_epoch(dataset, epoch):
        dataset_epochs.append(epoch)
        original_set_epoch(dataset, epoch)

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
    monkeypatch.setattr(
        train_module.TWDToMDataset,
        "set_epoch",
        capture_set_epoch,
    )
    summary = run_training(config)

    assert train_epochs == [1, 2, 3]
    assert dataset_epochs == [1, 2, 3]
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


def test_public_only_checkpoint_and_summary_record_dataset_lineage(
    tmp_path,
    monkeypatch,
):
    config = _training_config(
        tmp_path,
        1,
        train_sample=make_public_only_training_sample(
            1,
            game_id="public-train",
        ),
        validation_sample=make_public_only_training_sample(
            1,
            game_id="public-validation",
        ),
    )
    monkeypatch.setattr(
        train_module,
        "train_one_epoch",
        lambda *_args, **_kwargs: {"mean_loss": 1.0, "valid_subject_count": 1},
    )
    monkeypatch.setattr(
        train_module,
        "evaluate_model",
        lambda *_args, **_kwargs: {"mean_loss": 1.0, "valid_subject_count": 1},
    )
    summary = run_training(config)
    checkpoint = torch.load(
        config.run_output_dir / "best.pt",
        map_location="cpu",
        weights_only=True,
    )
    expected = {
        "schema_version": PUBLIC_ONLY_SAMPLE_SCHEMA_VERSION,
        "model_input_scope": PUBLIC_ONLY_MODEL_INPUT_SCOPE,
        "private_fields_usage": PUBLIC_ONLY_PRIVATE_FIELDS_USAGE,
        "belief_information_scope": PUBLIC_ONLY_BELIEF_INFORMATION_SCOPE,
        "annotation_schema_version": (
            PUBLIC_ONLY_FORMAL_ANNOTATION_SCHEMA_VERSION
        ),
        "label_provenance": PUBLIC_ONLY_FORMAL_LABEL_PROVENANCE,
        "source_label_provenance": PUBLIC_ONLY_LABEL_PROVENANCE,
    }
    for field_name, value in expected.items():
        assert checkpoint[field_name] == value
        assert summary[field_name] == value
        assert checkpoint["run_provenance"][field_name] == value


@pytest.mark.parametrize("tom_order", [1, 2])
@pytest.mark.parametrize(
    "backbone",
    [QWEN2_BACKBONE_NAME, GPT2_BLOCK_BACKBONE_NAME],
)
def test_one_batch_train_validation_smoke_and_best_eval(
    tmp_path,
    tom_order,
    backbone,
):
    config = _training_config(
        tmp_path,
        tom_order,
        backbone=backbone,
    )
    summary = run_training(config)
    output_files = {path.name for path in config.run_output_dir.iterdir()}
    assert output_files == {"best.pt", "last.pt", "history.json", "summary.json"}
    assert summary["best_epoch"] == 1
    assert summary["epochs_completed"] == 1
    assert summary["backbone"] == backbone
    assert summary["selection_metric_name"] == "validation_mean_loss"
    saved_summary = json.loads(
        (config.run_output_dir / "summary.json").read_text(encoding="utf-8")
    )
    history = json.loads(
        (config.run_output_dir / "history.json").read_text(encoding="utf-8")
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
    assert best["backbone"] == last["backbone"] == backbone
    assert best["training_config"]["backbone"] == backbone
    assert best["train_dataset_path"] == "tests/fixtures/synthetic_train.jsonl"
    assert best["validation_dataset_path"] == "tests/fixtures/synthetic_val.jsonl"
    assert best["run_provenance"]["git_worktree_clean"] is True
    for payload in (best, last, saved_summary, history[0]):
        assert payload["run_provenance"]["git_commit_sha"] == "1" * 40
        assert payload["run_provenance"]["train_dataset_sha256"] == sha256_file(
            config.resolved_dataset_path
        )
        assert str(tmp_path) not in json.dumps(
            payload["run_provenance"],
            default=str,
        )
    assert best["selection_metric_name"] == "validation_mean_loss"
    assert best["validation_metrics"]["mean_loss"] == best["selection_metric_value"]
    for name, expected in best["model_state_dict"].items():
        torch.testing.assert_close(last["model_state_dict"][name], expected)
    restored_last = build_model_from_checkpoint(last, device=torch.device("cpu"))
    assert restored_last.backbone_name == backbone
    if backbone == QWEN2_BACKBONE_NAME:
        assert isinstance(restored_last.transformer, Qwen2Model)
    else:
        assert isinstance(restored_last.transformer, GPT2BlockStack)

    evaluation_path = config.run_output_dir / "val_metrics.json"
    evaluation = evaluate_checkpoint(
        EvaluationConfig(
            checkpoint_path=str(config.run_output_dir / "best.pt"),
            dataset_path=str(config.resolved_validation_dataset_path),
            output_path=str(evaluation_path),
            training_dataset_path=str(config.resolved_dataset_path),
            batch_size=1,
            device="cpu",
        )
    )
    saved_evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert evaluation["status"] == "ok"
    assert evaluation["tom_order"] == tom_order
    assert evaluation["backbone"] == backbone
    assert evaluation["evaluation_sample_count"] == 1
    if tom_order == 1:
        assert best["pair_class_count"] == 21
        assert "output_class_count" not in best
        assert best["model_config"]["pair_class_count"] == 21
        assert saved_summary["pair_class_count"] == 21
        assert saved_summary["model_config"]["pair_class_count"] == 21
        assert history[0]["pair_class_count"] == 21
        assert saved_evaluation["pair_class_count"] == 21
        assert saved_evaluation["model_config"]["pair_class_count"] == 21
        assert "mean_pair_cross_entropy" in evaluation["metrics"]
        for field in (
            "observer_event_conditioning",
            "second_order_subject_supervision",
        ):
            assert field not in best
            assert field not in saved_summary
            assert field not in history[0]
            assert field not in saved_evaluation
    else:
        result_payloads = (best, saved_summary, history[0], saved_evaluation)
        assert all(
            payload["target_encoding"] == SECOND_ORDER_TARGET_ENCODING
            for payload in result_payloads
        )
        assert all(
            payload["output_class_count"] == 21
            for payload in result_payloads
        )
        assert all(
            payload["pair_class_count"] == 21
            for payload in result_payloads
        )
        assert all(
            payload["pair_ordering"]
            == "global_lexicographic_two_player_combinations"
            for payload in result_payloads
        )
        assert "projection_version" not in best
        assert set(evaluation["metrics"]) >= {
            "mean_pair_cross_entropy",
            "mean_pair_kl_divergence",
            "mean_pair_total_variation",
            "mean_marginal_mae",
        }
        assert not any("suspicion" in name for name in evaluation["metrics"])
        assert all(
            payload["observer_readout"] == SECOND_ORDER_OBSERVER_READOUT
            for payload in result_payloads
        )
        assert all(
            payload["train_player_augmentation"]
            == CYCLIC_ROTATION_VERSION
            for payload in result_payloads
        )
        assert all(
            payload["observer_event_conditioning"]
            == SECOND_ORDER_OBSERVER_EVENT_CONDITIONING
            for payload in result_payloads
        )
        assert all(
            payload["second_order_subject_supervision"]
            == SECOND_ORDER_SUBJECT_SUPERVISION
            for payload in result_payloads
        )
        assert COLLAPSE_DIAGNOSTIC_KEYS <= set(evaluation["metrics"])
        assert COLLAPSE_DIAGNOSTIC_KEYS <= set(best["train_metrics"])
        assert COLLAPSE_DIAGNOSTIC_KEYS <= set(best["validation_metrics"])
        for metrics in (
            evaluation["metrics"],
            best["train_metrics"],
            best["validation_metrics"],
        ):
            assert metrics["post_speech_other_player_valid_subject_count"] >= 1
            assert metrics["valid_subject_count"] == metrics[
                "post_speech_other_player_valid_subject_count"
            ]
            assert 0 < metrics[
                "post_speech_supervised_snapshot_fraction"
            ] <= 1


@pytest.mark.parametrize("tom_order", [1, 2])
@pytest.mark.parametrize(
    "backbone",
    [QWEN2_BACKBONE_NAME, GPT2_BLOCK_BACKBONE_NAME],
)
def test_one_batch_forward_backward_uses_only_soft_target_cross_entropy(
    tmp_path,
    tom_order,
    backbone,
):
    config = _training_config(
        tmp_path,
        tom_order,
        backbone=backbone,
    )
    loader, _ = build_data_loader(
        config,
        dataset_path=config.resolved_dataset_path,
        shuffle=False,
    )
    raw_batch = next(iter(loader))
    model = build_model(config)
    batch = _move_batch_to_device(raw_batch, torch.device("cpu"))
    output = _forward_batch(model, batch)
    loss = masked_distribution_cross_entropy(
        output["observer_pair_logits"],
        _targets_for_loss(
            model,
            batch["pair_targets"],
            _effective_subject_mask(model, batch),
        ),
        _effective_subject_mask(model, batch),
    )
    loss.backward()
    assert output["observer_pair_logits"].shape == (1, 7, 21)
    assert "observer_suspicion_logits" not in output
    assert "suspicion_targets" not in batch
    assert torch.isfinite(loss)
    assert model.output_projection.weight.grad is not None
    assert loss.grad_fn is not None


def test_effective_subject_mask_excludes_only_the_reasoning_player():
    first_order = ToMBeliefBackbone(
        ToMBeliefBackboneConfig(max_seq_len=8),
        tom_order=1,
    )
    second_order = ToMBeliefBackbone(
        ToMBeliefBackboneConfig(max_seq_len=8),
        tom_order=2,
    )
    subject_mask = torch.tensor([[True, True, False, True, False, False, False]])
    batch = {
        "subject_mask": subject_mask,
        "reasoning_player_id": torch.tensor([4]),
        "post_completed_public_speech_pre_next_action": torch.tensor([True]),
    }
    assert torch.equal(_effective_subject_mask(first_order, batch), subject_mask)
    assert torch.equal(
        _effective_subject_mask(second_order, batch),
        torch.tensor([[True, True, False, False, False, False, False]]),
    )


def test_effective_subject_mask_rejects_non_speech_boundary():
    model = ToMBeliefBackbone(
        ToMBeliefBackboneConfig(max_seq_len=8),
        tom_order=2,
    )
    batch = {
        "subject_mask": torch.ones((1, 7), dtype=torch.bool),
        "reasoning_player_id": torch.tensor([4]),
        "post_completed_public_speech_pre_next_action": torch.tensor([False]),
    }
    with pytest.raises(ValueError, match="non-speech-boundary"):
        _effective_subject_mask(model, batch)


def test_second_order_loss_aggregates_all_valid_other_player_rows():
    logits = torch.zeros((1, 7, 21))
    targets = torch.zeros_like(logits)
    targets[0, 0, 0] = 1
    targets[0, 1, 1] = 1
    logits[0, 0, 0] = 5
    logits[0, 1, 1] = -5
    original_mask = torch.tensor(
        [[True, True, False, False, False, False, False]]
    )
    second_order = ToMBeliefBackbone(
        ToMBeliefBackboneConfig(max_seq_len=8),
        tom_order=2,
    )
    effective = _effective_subject_mask(
        second_order,
        {
            "subject_mask": original_mask,
            "reasoning_player_id": torch.tensor([1]),
            "post_completed_public_speech_pre_next_action": torch.tensor([True]),
        },
    )
    loss_targets = _targets_for_loss(second_order, targets, effective)
    assert targets[0, 0, 0] == 1
    assert loss_targets[0, 0].count_nonzero() == 0
    actual = masked_distribution_cross_entropy(logits, loss_targets, effective)
    expected = -torch.log_softmax(logits[0, 1], dim=-1)[1]
    torch.testing.assert_close(actual, expected)


def test_canonical_d_loader_requires_split_manifest(tmp_path):
    split_root, _manifest = _canonical_d_split_fixture(tmp_path)
    config = TrainingConfig(
        tom_order=1,
        output_dir=str(tmp_path / "outputs"),
        dataset_path=str(split_root / "tom1" / "train.jsonl"),
        validation_dataset_path=str(split_root / "tom1" / "validation.jsonl"),
        batch_size=1,
        device="cpu",
        max_seq_len=32,
    )
    with pytest.raises(ValueError, match="requires --split-manifest"):
        build_training_data_loaders(config)


def test_canonical_d_split_manifest_tamper_is_rejected(tmp_path):
    split_root, manifest = _canonical_d_split_fixture(tmp_path)
    manifest_path = split_root / "manifest.json"
    tampered = deepcopy(manifest)
    tampered["split_seed"] += 1
    manifest_path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest_digest"):
        load_canonical_d_split_manifest(manifest_path)


def test_canonical_d_training_manifest_and_checkpoint_lineage(tmp_path, monkeypatch):
    split_root, manifest = _canonical_d_split_fixture(tmp_path)
    config = TrainingConfig(
        tom_order=2,
        output_dir=str(tmp_path / "outside_repo_outputs"),
        dataset_path=str(split_root / "tom2" / "train.jsonl"),
        validation_dataset_path=str(split_root / "tom2" / "validation.jsonl"),
        split_manifest_path=str(split_root / "manifest.json"),
        epochs=1,
        batch_size=1,
        device="cpu",
        max_seq_len=32,
    )

    monkeypatch.setattr(
        train_module,
        "train_one_epoch",
        lambda *_args, **_kwargs: {"mean_loss": 1.0, "valid_subject_count": 1},
    )
    monkeypatch.setattr(
        train_module,
        "evaluate_model",
        lambda *_args, **_kwargs: {"mean_loss": 0.5, "valid_subject_count": 1},
    )
    summary = run_training(config)
    training_manifest_path = config.run_output_dir / "training_manifest.json"
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version", "integration_version", "training_code_commit",
        "git_worktree_clean", "tom_order", "split_manifest_schema_version",
        "split_policy_version", "split_seed", "split_manifest_sha256",
        "split_manifest_digest", "d_schema_version",
        "d_materialization_policy_version", "materialization_task",
        "materializer_code_commits", "belief_information_scope",
        "model_input_scope", "private_fields_usage",
        "annotation_schema_version", "label_provenance",
        "source_label_provenance", "train_dataset_relative_path",
        "validation_dataset_relative_path", "train_dataset_sha256",
        "validation_dataset_sha256", "train_game_ids", "validation_game_ids",
        "train_source_row_count", "validation_source_row_count",
        "train_effective_supervised_snapshot_count",
        "validation_effective_supervised_snapshot_count",
        "tom2_target_semantics", "tom2_temporal_supervision_policy",
        "train_cyclic_rotation_enabled", "validation_cyclic_rotation_enabled",
        "cyclic_rotation_version", "augmentation_seed", "training_config",
        "python_version", "torch_version", "transformers_version", "platform",
        "requested_device", "resolved_device", "manifest_digest",
    }
    assert set(training_manifest) == expected_fields
    assert training_manifest["schema_version"] == TRAINING_MANIFEST_SCHEMA_VERSION
    assert training_manifest["integration_version"] == (
        CANONICAL_D_TRAINING_INTEGRATION_VERSION
    )
    assert training_manifest["d_schema_version"] == D_SCHEMA_VERSION
    assert training_manifest["d_materialization_policy_version"] == (
        D_MATERIALIZATION_POLICY_VERSION
    )
    assert training_manifest["materialization_task"] == OFFLINE_PUBLIC_ONLY_TOM2_TASK
    assert training_manifest["split_manifest_schema_version"] == (
        SPLIT_MANIFEST_SCHEMA_VERSION
    )
    assert training_manifest["split_policy_version"] == SPLIT_POLICY_VERSION
    assert training_manifest["split_seed"] == manifest["split_seed"]
    assert training_manifest["tom2_target_semantics"] == TOM2_TARGET_SEMANTICS
    assert training_manifest["tom2_temporal_supervision_policy"] == (
        TOM2_TEMPORAL_SUPERVISION_POLICY
    )
    payload = deepcopy(training_manifest)
    digest = payload.pop("manifest_digest")
    assert canonical_digest(payload) == digest
    assert training_manifest["train_source_row_count"] == 2
    assert training_manifest["validation_source_row_count"] == 2
    assert 0 < training_manifest["train_effective_supervised_snapshot_count"] < 2
    assert 0 < training_manifest[
        "validation_effective_supervised_snapshot_count"
    ] < 2

    checkpoint = torch.load(
        config.run_output_dir / "best.pt", map_location="cpu", weights_only=True
    )
    assert checkpoint["schema_version"] == D_SCHEMA_VERSION
    assert checkpoint["materialization_task"] == OFFLINE_PUBLIC_ONLY_TOM2_TASK
    assert checkpoint["training_manifest_schema_version"] == (
        TRAINING_MANIFEST_SCHEMA_VERSION
    )
    assert checkpoint["training_manifest_sha256"] == summary[
        "training_manifest_sha256"
    ]
    assert checkpoint["split_manifest_sha256"] == training_manifest[
        "split_manifest_sha256"
    ]
    assert checkpoint["tom2_target_semantics"] == TOM2_TARGET_SEMANTICS
    assert build_model_from_checkpoint(
        checkpoint, device=torch.device("cpu")
    ).tom_order == 2


def test_canonical_d_tom1_manifest_uses_null_tom2_and_rotation_fields(
    tmp_path, monkeypatch
):
    split_root, _manifest = _canonical_d_split_fixture(tmp_path)
    config = TrainingConfig(
        tom_order=1,
        output_dir=str(tmp_path / "d_tom1_outputs"),
        dataset_path=str(split_root / "tom1" / "train.jsonl"),
        validation_dataset_path=str(split_root / "tom1" / "validation.jsonl"),
        split_manifest_path=str(split_root / "manifest.json"),
        epochs=1,
        batch_size=1,
        device="cpu",
        max_seq_len=32,
    )
    monkeypatch.setattr(
        train_module,
        "train_one_epoch",
        lambda *_args, **_kwargs: {"mean_loss": 1.0, "valid_subject_count": 1},
    )
    monkeypatch.setattr(
        train_module,
        "evaluate_model",
        lambda *_args, **_kwargs: {"mean_loss": 0.5, "valid_subject_count": 1},
    )
    run_training(config)
    manifest = json.loads(
        (config.run_output_dir / "training_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["materialization_task"] == OFFLINE_PRIVATE_CONDITIONED_TOM1_TASK
    assert manifest["train_source_row_count"] == manifest[
        "train_effective_supervised_snapshot_count"
    ]
    assert manifest["validation_source_row_count"] == manifest[
        "validation_effective_supervised_snapshot_count"
    ]
    assert manifest["tom2_target_semantics"] is None
    assert manifest["tom2_temporal_supervision_policy"] is None
    assert manifest["train_cyclic_rotation_enabled"] is False
    assert manifest["validation_cyclic_rotation_enabled"] is False
    assert manifest["cyclic_rotation_version"] is None
    assert manifest["augmentation_seed"] is None


@pytest.mark.parametrize("value", [0, 3, True, None, "1"])
def test_invalid_tom_order_is_rejected(tmp_path, value):
    with pytest.raises(ValueError, match="tom_order"):
        TrainingConfig(
            tom_order=value,
            output_dir=str(tmp_path),
            dataset_path="train.jsonl",
            validation_dataset_path="val.jsonl",
        )


def _initialize_clean_git_worktree(tmp_path: Path) -> TrainingConfig:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    train_path.write_text('{"game_id":"train"}\n', encoding="utf-8")
    validation_path.write_text('{"game_id":"validation"}\n', encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=ToM Test",
            "-c",
            "user.email=tom-test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return TrainingConfig(
        tom_order=1,
        output_dir=str(tmp_path / "outputs"),
        dataset_path=str(train_path),
        validation_dataset_path=str(validation_path),
        device="cpu",
    )


def test_run_provenance_records_clean_git_data_environment_and_device(tmp_path):
    config = _initialize_clean_git_worktree(tmp_path)
    set_random_seed(config.seed)
    provenance = build_run_provenance(
        config,
        resolved_device=torch.device("cpu"),
        repo_root=tmp_path,
    )
    assert len(provenance["git_commit_sha"]) == 40
    assert provenance["git_worktree_clean"] is True
    assert provenance["train_dataset_path"] == "train.jsonl"
    assert provenance["validation_dataset_path"] == "validation.jsonl"
    assert provenance["output_dir"] == "outputs"
    assert provenance["train_dataset_sha256"] == sha256_file(
        config.resolved_dataset_path
    )
    assert provenance["validation_dataset_sha256"] == sha256_file(
        config.resolved_validation_dataset_path
    )
    assert provenance["requested_device"] == "cpu"
    assert provenance["resolved_device"] == "cpu"
    assert provenance["deterministic_algorithms_enabled"] is True
    for field in (
        "python_version",
        "torch_version",
        "transformers_version",
        "platform",
    ):
        assert provenance[field]
    assert str(tmp_path) not in json.dumps(provenance)


def test_canonical_d_run_provenance_allows_external_data_and_output_roots(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "tracked.txt").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git", "-C", str(repo), "-c", "user.name=ToM Test",
            "-c", "user.email=tom-test@example.invalid", "commit", "-qm",
            "fixture",
        ],
        check=True,
    )
    data_root = tmp_path / "external_data"
    data_root.mkdir()
    split_root, manifest = _canonical_d_split_fixture(data_root)
    config = TrainingConfig(
        tom_order=1,
        output_dir=str(data_root / "outputs"),
        dataset_path=str(split_root / "tom1" / "train.jsonl"),
        validation_dataset_path=str(split_root / "tom1" / "validation.jsonl"),
        split_manifest_path=str(split_root / "manifest.json"),
        device="cpu",
    )
    provenance = build_run_provenance(
        config, resolved_device=torch.device("cpu"), repo_root=repo
    )
    assert provenance["train_dataset_path"] == "tom1/train.jsonl"
    assert provenance["validation_dataset_path"] == "tom1/validation.jsonl"
    assert provenance["split_manifest_path"] == "manifest.json"
    assert provenance["split_manifest_sha256"] == sha256_file(
        split_root / "manifest.json"
    )
    assert provenance["split_manifest_digest"] == manifest["manifest_digest"]
    assert provenance["output_dir"] == str((data_root / "outputs").resolve())
    assert str(split_root.resolve()) not in json.dumps(
        {
            "train_dataset_path": provenance["train_dataset_path"],
            "validation_dataset_path": provenance["validation_dataset_path"],
            "split_manifest_path": provenance["split_manifest_path"],
        }
    )


def test_run_provenance_rejects_dirty_worktree_and_lists_file(tmp_path):
    config = _initialize_clean_git_worktree(tmp_path)
    Path(config.dataset_path).write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"dirty files:[\s\S]*train\.jsonl"):
        build_run_provenance(
            config,
            resolved_device=torch.device("cpu"),
            repo_root=tmp_path,
        )


def test_dataset_sha_changes_with_file_content(tmp_path):
    path = tmp_path / "dataset.jsonl"
    path.write_text("first\n", encoding="utf-8")
    first = sha256_file(path)
    path.write_text("second\n", encoding="utf-8")
    assert sha256_file(path) != first


def test_atomic_checkpoint_failure_preserves_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "best.pt"
    path.write_bytes(b"previous-valid-checkpoint")

    def fail_save(_value, _file):
        raise RuntimeError("synthetic save failure")

    monkeypatch.setattr(train_module.torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="synthetic save failure"):
        _atomic_torch_save({"new": True}, path)
    assert path.read_bytes() == b"previous-valid-checkpoint"
    assert not (tmp_path / ".best.pt.tmp").exists()


def test_atomic_json_write_replaces_complete_document(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text('{"old": true}\n', encoding="utf-8")
    _atomic_json_write({"status": "ok"}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "ok"}
    assert not (tmp_path / ".summary.json.tmp").exists()


def test_non_empty_training_output_directory_is_rejected(tmp_path):
    output_dir = tmp_path / "tom_order_2"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="must be empty"):
        _prepare_run_output_dir(output_dir)
    assert (output_dir / "existing.txt").read_text(encoding="utf-8") == "keep"


def test_run_training_rejects_non_empty_output_before_loading_data(
    tmp_path,
    monkeypatch,
):
    config = _training_config(tmp_path, 1)
    config.run_output_dir.mkdir(parents=True)
    existing = config.run_output_dir / "best.pt"
    existing.write_bytes(b"existing-checkpoint")

    def fail_if_called(_config):
        raise AssertionError("data loading began before output safety check")

    monkeypatch.setattr(
        train_module,
        "build_training_data_loaders",
        fail_if_called,
    )
    with pytest.raises(FileExistsError, match="must be empty"):
        run_training(config)
    assert existing.read_bytes() == b"existing-checkpoint"
