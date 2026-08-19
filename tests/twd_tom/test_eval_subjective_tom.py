"""Tests for restoring and evaluating the current Qwen2 ToM checkpoint."""

import json
from pathlib import Path

import pytest
import torch

import script.twd_tom.eval as eval_module
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
    CANONICAL_D_TRAINING_INTEGRATION_VERSION,
    TOM2_TARGET_SEMANTICS,
    TOM2_TEMPORAL_SUPERVISION_POLICY,
    TRAINING_MANIFEST_SCHEMA_VERSION,
    TrainingConfig,
    build_model,
    checkpoint_payload,
    sha256_file,
)
from werewolf.models.twd_tom.schema import (
    ACTION_NAMES,
    ACTION_TO_ID,
    PAIR_ORDERING,
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
from werewolf.models.twd_tom.dataset import (
    CYCLIC_ROTATION_VERSION,
    D_PUBLIC_ONLY_TOM2_BELIEF_INFORMATION_SCOPE,
    PRIVATE_CONDITIONED_BELIEF_INFORMATION_SCOPE,
)
from script.twd_tom.split_offline_d_training_data import (
    SPLIT_MANIFEST_SCHEMA_VERSION,
    SPLIT_POLICY_VERSION,
    split_offline_d_training_data,
)
from werewolf.offline_annotation import OFFLINE_ANNOTATION_SCHEMA_VERSION
from werewolf.offline_materialization import (
    D_MATERIALIZATION_POLICY_VERSION,
    D_SCHEMA_VERSION,
    OFFLINE_PRIVATE_CONDITIONED_TOM1_TASK,
    OFFLINE_PUBLIC_ONLY_TOM2_TASK,
    TOM1_MODEL_INPUT_SCOPE,
    TOM1_OBSERVER_PROVENANCE,
    TOM1_PRIVATE_FIELDS_USAGE,
    TOM2_MODEL_INPUT_SCOPE,
    TOM2_OBSERVER_PROVENANCE,
    TOM2_PRIVATE_FIELDS_USAGE,
)
from werewolf.trajectory import canonical_digest, canonical_json
from tests.twd_tom.test_twd_tom_dataset import d_sample
from tests.twd_tom.public_event_fixtures import make_training_sample
from werewolf.models.twd_tom.belief_backbone import (
    GPT2_BLOCK_BACKBONE_NAME,
    GPT2BlockStack,
    QWEN2_BACKBONE_NAME,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def make_checkpoint(
    tmp_path,
    tom_order=1,
    backbone=QWEN2_BACKBONE_NAME,
    public_only=False,
    canonical_d=False,
):
    config = TrainingConfig(
        tom_order=tom_order,
        output_dir=str(tmp_path),
        dataset_path=str(tmp_path / "train.jsonl"),
        validation_dataset_path=str(tmp_path / "validation.jsonl"),
        backbone=backbone,
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
    dataset_contract = None
    training_manifest = None
    training_manifest_sha256 = None
    if canonical_d:
        if tom_order == 1:
            dataset_contract = {
                "source_schema_version": D_SCHEMA_VERSION,
                "model_input_scope": TOM1_MODEL_INPUT_SCOPE,
                "belief_information_scope": PRIVATE_CONDITIONED_BELIEF_INFORMATION_SCOPE,
                "private_fields_usage": TOM1_PRIVATE_FIELDS_USAGE,
                "annotation_schema_version": OFFLINE_ANNOTATION_SCHEMA_VERSION,
                "label_provenance": TOM1_OBSERVER_PROVENANCE,
                "source_label_provenance": TOM1_OBSERVER_PROVENANCE,
            }
            task = OFFLINE_PRIVATE_CONDITIONED_TOM1_TASK
        else:
            dataset_contract = {
                "source_schema_version": D_SCHEMA_VERSION,
                "model_input_scope": TOM2_MODEL_INPUT_SCOPE,
                "belief_information_scope": D_PUBLIC_ONLY_TOM2_BELIEF_INFORMATION_SCOPE,
                "private_fields_usage": TOM2_PRIVATE_FIELDS_USAGE,
                "annotation_schema_version": OFFLINE_ANNOTATION_SCHEMA_VERSION,
                "label_provenance": TOM2_OBSERVER_PROVENANCE,
                "source_label_provenance": TOM2_OBSERVER_PROVENANCE,
            }
            task = OFFLINE_PUBLIC_ONLY_TOM2_TASK
        is_tom2 = tom_order == 2
        training_manifest = {
            "schema_version": TRAINING_MANIFEST_SCHEMA_VERSION,
            "integration_version": CANONICAL_D_TRAINING_INTEGRATION_VERSION,
            "training_code_commit": "1" * 40,
            "git_worktree_clean": True,
            "tom_order": tom_order,
            "split_manifest_schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
            "split_policy_version": SPLIT_POLICY_VERSION,
            "split_seed": 17,
            "split_manifest_sha256": "b" * 64,
            "split_manifest_digest": "d" * 64,
            "d_schema_version": D_SCHEMA_VERSION,
            "d_materialization_policy_version": D_MATERIALIZATION_POLICY_VERSION,
            "materialization_task": task,
            "materializer_code_commits": ["2" * 40],
            "belief_information_scope": dataset_contract[
                "belief_information_scope"
            ],
            "model_input_scope": dataset_contract["model_input_scope"],
            "private_fields_usage": dataset_contract["private_fields_usage"],
            "annotation_schema_version": dataset_contract[
                "annotation_schema_version"
            ],
            "label_provenance": dataset_contract["label_provenance"],
            "source_label_provenance": dataset_contract[
                "source_label_provenance"
            ],
            "train_dataset_relative_path": f"tom{tom_order}/train.jsonl",
            "validation_dataset_relative_path": (
                f"tom{tom_order}/validation.jsonl"
            ),
            "train_dataset_sha256": "3" * 64,
            "validation_dataset_sha256": "4" * 64,
            "train_game_ids": ["synthetic_train"],
            "validation_game_ids": ["synthetic_validation"],
            "train_source_row_count": 1,
            "validation_source_row_count": 1,
            "train_effective_supervised_snapshot_count": 1,
            "validation_effective_supervised_snapshot_count": 1,
            "tom2_target_semantics": TOM2_TARGET_SEMANTICS if is_tom2 else None,
            "tom2_temporal_supervision_policy": (
                TOM2_TEMPORAL_SUPERVISION_POLICY if is_tom2 else None
            ),
            "train_cyclic_rotation_enabled": is_tom2,
            "validation_cyclic_rotation_enabled": False,
            "cyclic_rotation_version": CYCLIC_ROTATION_VERSION if is_tom2 else None,
            "augmentation_seed": 42 if is_tom2 else None,
            "training_config": {"fixture": True},
            "python_version": "test",
            "torch_version": str(torch.__version__),
            "transformers_version": "test",
            "platform": "test",
            "requested_device": "auto",
            "resolved_device": "cpu",
        }
        training_manifest["manifest_digest"] = canonical_digest(training_manifest)
        training_manifest_sha256 = "c" * 64
    elif public_only:
        dataset_contract = {
            "source_schema_version": PUBLIC_ONLY_SAMPLE_SCHEMA_VERSION,
            "model_input_scope": PUBLIC_ONLY_MODEL_INPUT_SCOPE,
            "belief_information_scope": PUBLIC_ONLY_BELIEF_INFORMATION_SCOPE,
            "private_fields_usage": PUBLIC_ONLY_PRIVATE_FIELDS_USAGE,
            "annotation_schema_version": (
                PUBLIC_ONLY_FORMAL_ANNOTATION_SCHEMA_VERSION
            ),
            "label_provenance": PUBLIC_ONLY_FORMAL_LABEL_PROVENANCE,
            "source_label_provenance": PUBLIC_ONLY_LABEL_PROVENANCE,
        }
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
        dataset_contract=dataset_contract,
        training_manifest=training_manifest,
        training_manifest_sha256=training_manifest_sha256,
    )


def _redigest_d_record(record: dict, *, game_id: str, step_idx: int) -> dict:
    value = dict(record)
    value["game_id"] = game_id
    value["step_idx"] = step_idx
    value["label_cutoff_step_idx"] = step_idx
    value["boundary_id"] = f"{game_id}:step_{step_idx:06d}:PRE_PUBLIC_SPEECH"
    value.pop("record_digest", None)
    value["record_digest"] = canonical_digest(value)
    return value


def _canonical_d_split_fixture(tmp_path: Path):
    game_ids = ["eval_d_game_001", "eval_d_game_002", "eval_d_game_003"]
    tom1_rows = [
        _redigest_d_record(d_sample(1), game_id=game_id, step_idx=10 + index)
        for index, game_id in enumerate(game_ids)
    ]
    tom2_rows = []
    for index, game_id in enumerate(game_ids):
        base_step = 20 + index * 2
        tom2_rows.extend(
            [
                _redigest_d_record(
                    d_sample(2, with_latest_action=False),
                    game_id=game_id,
                    step_idx=base_step,
                ),
                _redigest_d_record(
                    d_sample(2, with_latest_action=True),
                    game_id=game_id,
                    step_idx=base_step + 1,
                ),
            ]
        )
    tom1_source = tmp_path / "eval_source_tom1.jsonl"
    tom2_source = tmp_path / "eval_source_tom2.jsonl"
    tom1_source.write_text(
        "".join(canonical_json(row) + "\n" for row in tom1_rows),
        encoding="utf-8",
    )
    tom2_source.write_text(
        "".join(canonical_json(row) + "\n" for row in tom2_rows),
        encoding="utf-8",
    )
    split_root = tmp_path / "eval_canonical_split"
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


def _bind_checkpoint_to_split(checkpoint: dict, split_root: Path, manifest: dict) -> dict:
    checkpoint["split_manifest_sha256"] = sha256_file(split_root / "manifest.json")
    checkpoint["split_manifest_schema_version"] = manifest["schema_version"]
    checkpoint["split_policy_version"] = manifest["split_policy_version"]
    checkpoint["split_seed"] = manifest["split_seed"]
    return checkpoint


@pytest.mark.parametrize("tom_order", [1, 2])
def test_qwen2_checkpoint_restores_strictly(tmp_path, tom_order):
    checkpoint = make_checkpoint(tmp_path, tom_order=tom_order)
    restored = build_model_from_checkpoint(checkpoint, device=torch.device("cpu"))
    assert restored.backbone_name == QWEN2_BACKBONE_NAME
    assert isinstance(restored.transformer, Qwen2Model)
    assert restored.config.max_seq_len == 64
    assert restored.tom_order == tom_order
    for name, expected in checkpoint["model_state_dict"].items():
        torch.testing.assert_close(restored.state_dict()[name], expected)


@pytest.mark.parametrize("tom_order", [1, 2])
def test_gpt2_block_checkpoint_restores_strictly(tmp_path, tom_order):
    checkpoint = make_checkpoint(
        tmp_path,
        tom_order=tom_order,
        backbone=GPT2_BLOCK_BACKBONE_NAME,
    )
    restored = build_model_from_checkpoint(
        checkpoint,
        device=torch.device("cpu"),
    )
    assert restored.backbone_name == GPT2_BLOCK_BACKBONE_NAME
    assert isinstance(restored.transformer, GPT2BlockStack)
    assert len(restored.transformer.blocks) == 4
    for name, expected in checkpoint["model_state_dict"].items():
        torch.testing.assert_close(restored.state_dict()[name], expected)


def test_public_only_tom2_checkpoint_restores_strictly(tmp_path):
    checkpoint = make_checkpoint(
        tmp_path,
        tom_order=2,
        public_only=True,
    )

    restored = build_model_from_checkpoint(
        checkpoint,
        device=torch.device("cpu"),
    )

    assert restored.tom_order == 2
    assert restored.state_dict().keys() == checkpoint["model_state_dict"].keys()
    for name, expected in checkpoint["model_state_dict"].items():
        torch.testing.assert_close(restored.state_dict()[name], expected)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("private_fields_usage", "label_construction_only"),
        ("model_input_scope", "structured_public_events_only"),
    ],
)
def test_public_only_checkpoint_with_mixed_lineage_is_rejected(
    tmp_path,
    field,
    value,
):
    checkpoint = make_checkpoint(tmp_path, tom_order=2, public_only=True)
    checkpoint[field] = value

    with pytest.raises(ValueError, match=field):
        build_model_from_checkpoint(checkpoint, device=torch.device("cpu"))


@pytest.mark.parametrize("tom_order", [1, 2])
def test_canonical_d_checkpoint_restores_strictly(tmp_path, tom_order):
    checkpoint = make_checkpoint(tmp_path, tom_order=tom_order, canonical_d=True)
    restored = build_model_from_checkpoint(checkpoint, device=torch.device("cpu"))
    assert restored.tom_order == tom_order
    assert checkpoint["schema_version"] == D_SCHEMA_VERSION
    assert checkpoint["training_manifest_schema_version"] == (
        TRAINING_MANIFEST_SCHEMA_VERSION
    )
    if tom_order == 2:
        assert checkpoint["tom2_target_semantics"] == TOM2_TARGET_SEMANTICS
        assert checkpoint["tom2_temporal_supervision_policy"] == (
            TOM2_TEMPORAL_SUPERVISION_POLICY
        )
    else:
        assert "tom2_target_semantics" not in checkpoint


def test_canonical_d_checkpoint_rejects_wrong_lineage(tmp_path):
    checkpoint = make_checkpoint(tmp_path, tom_order=2, canonical_d=True)
    checkpoint["materialization_task"] = OFFLINE_PRIVATE_CONDITIONED_TOM1_TASK
    with pytest.raises(ValueError, match="materialization_task"):
        build_model_from_checkpoint(checkpoint, device=torch.device("cpu"))


def test_unknown_checkpoint_schema_fails_closed(tmp_path):
    checkpoint = make_checkpoint(tmp_path, tom_order=2)
    checkpoint["schema_version"] = "unknown_formal_schema"

    with pytest.raises(ValueError, match="supported formal lineage"):
        build_model_from_checkpoint(checkpoint, device=torch.device("cpu"))


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


def test_checkpoint_records_the_complete_speech_action_contract(tmp_path):
    checkpoint = make_checkpoint(tmp_path)

    assert checkpoint["speech_action_count"] == len(ACTION_NAMES) == 13
    assert checkpoint["speech_action_to_id"] == dict(ACTION_TO_ID)


@pytest.mark.parametrize(
    "field",
    ["speech_action_count", "speech_action_to_id"],
)
def test_checkpoint_without_current_speech_action_contract_is_rejected(
    tmp_path,
    field,
):
    checkpoint = make_checkpoint(tmp_path)
    checkpoint.pop(field)

    with pytest.raises(ValueError, match=field):
        build_model_from_checkpoint(
            checkpoint,
            device=torch.device("cpu"),
        )


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


def test_checkpoint_backbone_cannot_be_relabelled(tmp_path):
    checkpoint = make_checkpoint(tmp_path)
    checkpoint["backbone"] = GPT2_BLOCK_BACKBONE_NAME
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


@pytest.mark.parametrize("evaluation_split", ["validation", "test"])
def test_canonical_d_evaluation_uses_matching_split_manifest(
    tmp_path, monkeypatch, evaluation_split
):
    split_root, manifest = _canonical_d_split_fixture(tmp_path)
    checkpoint = _bind_checkpoint_to_split(
        make_checkpoint(tmp_path, tom_order=2, canonical_d=True),
        split_root,
        manifest,
    )
    checkpoint_path = tmp_path / "d_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    monkeypatch.setattr(
        eval_module,
        "count_supervised_subjects",
        lambda _loader: 1,
    )
    monkeypatch.setattr(
        eval_module,
        "evaluate_model",
        lambda *_args, **_kwargs: {"mean_loss": 0.5, "valid_subject_count": 1},
    )
    summary = evaluate_checkpoint(
        EvaluationConfig(
            checkpoint_path=str(checkpoint_path),
            dataset_path=str(split_root / "tom2" / f"{evaluation_split}.jsonl"),
            split_manifest_path=str(split_root / "manifest.json"),
            batch_size=1,
            device="cpu",
        )
    )
    assert summary["schema_version"] == D_SCHEMA_VERSION
    assert summary["evaluation_split"] == evaluation_split
    assert summary["split_manifest_sha256"] == sha256_file(
        split_root / "manifest.json"
    )
    assert summary["evaluation_source_sample_count"] == 2
    assert summary["evaluation_effective_supervised_snapshot_count"] == 1
    assert summary["tom2_target_semantics"] == TOM2_TARGET_SEMANTICS
    assert summary["tom2_temporal_supervision_policy"] == (
        TOM2_TEMPORAL_SUPERVISION_POLICY
    )
    assert summary["training_dataset_path"] is None
    assert summary["training_game_ids"] == sorted(manifest["game_ids"]["train"])


def test_canonical_d_evaluation_requires_split_manifest(tmp_path):
    split_root, manifest = _canonical_d_split_fixture(tmp_path)
    checkpoint = _bind_checkpoint_to_split(
        make_checkpoint(tmp_path, tom_order=1, canonical_d=True),
        split_root,
        manifest,
    )
    checkpoint_path = tmp_path / "d_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="requires --split-manifest"):
        evaluate_checkpoint(
            EvaluationConfig(
                checkpoint_path=str(checkpoint_path),
                dataset_path=str(split_root / "tom1" / "validation.jsonl"),
                device="cpu",
            )
        )


def test_canonical_d_evaluation_rejects_wrong_manifest_sha(tmp_path):
    split_root, manifest = _canonical_d_split_fixture(tmp_path)
    checkpoint = _bind_checkpoint_to_split(
        make_checkpoint(tmp_path, tom_order=1, canonical_d=True),
        split_root,
        manifest,
    )
    checkpoint["split_manifest_sha256"] = "f" * 64
    checkpoint_path = tmp_path / "d_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="split manifest SHA-256"):
        evaluate_checkpoint(
            EvaluationConfig(
                checkpoint_path=str(checkpoint_path),
                dataset_path=str(split_root / "tom1" / "validation.jsonl"),
                split_manifest_path=str(split_root / "manifest.json"),
                device="cpu",
            )
        )


def test_canonical_d_train_split_cannot_be_evaluated(tmp_path):
    split_root, manifest = _canonical_d_split_fixture(tmp_path)
    checkpoint = _bind_checkpoint_to_split(
        make_checkpoint(tmp_path, tom_order=1, canonical_d=True),
        split_root,
        manifest,
    )
    checkpoint_path = tmp_path / "d_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="train split cannot"):
        evaluate_checkpoint(
            EvaluationConfig(
                checkpoint_path=str(checkpoint_path),
                dataset_path=str(split_root / "tom1" / "train.jsonl"),
                split_manifest_path=str(split_root / "manifest.json"),
                device="cpu",
            )
        )


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
