import json

import pytest
import torch
from torch.optim import AdamW

from script.twd_tom.eval import (
    EvaluationConfig,
    build_model_from_checkpoint,
    evaluate_checkpoint,
)
from script.twd_tom.materialize_canonical_belief_dataset import (
    SPLIT_MANIFEST_SCHEMA_VERSION,
    SPLIT_POLICY_VERSION,
)
from script.twd_tom.train import (
    TrainingConfig,
    build_model,
    checkpoint_payload,
    sha256_file,
)
from werewolf.models.twd_tom.samples import SAMPLE_SCHEMA_VERSION
from werewolf.trajectory import canonical_digest


def write_jsonl(path, sample):
    path.write_text(json.dumps(sample) + "\n", encoding="utf-8")


def make_materialized_splits(
    tmp_path,
    training_sample_factory,
    *,
    game_ids=("train", "validation", "test"),
):
    paths = {
        split_name: tmp_path / f"{split_name}.jsonl"
        for split_name in ("train", "validation", "test")
    }
    for split_name, game_id in zip(paths, game_ids):
        write_jsonl(
            paths[split_name],
            training_sample_factory(game_id=game_id),
        )
    manifest = {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "raw_schema_version": SAMPLE_SCHEMA_VERSION,
        "canonical_batch_summary_digest": "1" * 64,
        "canonical_batch_summary_sha256": "2" * 64,
        "game_summary_digests": {
            game_id: "3" * 64 for game_id in set(game_ids)
        },
        "split_policy_version": SPLIT_POLICY_VERSION,
        "split_seed": 0,
        "game_ids": {
            split_name: [game_id]
            for split_name, game_id in zip(paths, game_ids)
        },
        "game_counts": {split_name: 1 for split_name in paths},
        "row_counts": {split_name: 1 for split_name in paths},
        "output_files": {
            split_name: {
                "relative_path": path.name,
                "sha256": sha256_file(path),
            }
            for split_name, path in paths.items()
        },
    }
    manifest["manifest_digest"] = canonical_digest(manifest)
    (tmp_path / "split_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return paths, manifest


def make_checkpoint(
    tmp_path,
    training_path,
    *,
    validation_path=None,
    split_manifest_digest=None,
):
    validation_path = validation_path or training_path
    config = TrainingConfig(
        output_dir=str(tmp_path / "run"),
        dataset_path=str(training_path),
        validation_dataset_path=str(validation_path),
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
            "split_manifest_digest": split_manifest_digest,
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
    paths, manifest = make_materialized_splits(
        tmp_path,
        training_sample_factory,
    )
    checkpoint_path, _ = make_checkpoint(
        tmp_path,
        paths["train"],
        validation_path=paths["validation"],
        split_manifest_digest=manifest["manifest_digest"],
    )
    summary = evaluate_checkpoint(EvaluationConfig(
        checkpoint_path=str(checkpoint_path),
        dataset_path=str(paths["test"]),
        training_dataset_path=str(paths["train"]),
        batch_size=1,
        device="cpu",
    ))
    assert summary["status"] == "ok"
    assert summary["model_output"] == "belief_logits"
    assert summary["evaluation_supervised_observer_count"] == 4
    assert summary["metrics"]["valid_observer_count"] == 4
    assert summary["validation_game_ids"] == ["validation"]
    assert summary["split_manifest_digest"] == manifest["manifest_digest"]
    serialized = json.dumps(summary)
    assert "tom_order" not in serialized
    assert "pair" not in serialized


def test_evaluation_rejects_validation_test_game_overlap(
    tmp_path,
    training_sample_factory,
):
    paths, manifest = make_materialized_splits(
        tmp_path,
        training_sample_factory,
        game_ids=("train", "same", "same"),
    )
    checkpoint_path, _ = make_checkpoint(
        tmp_path,
        paths["train"],
        validation_path=paths["validation"],
        split_manifest_digest=manifest["manifest_digest"],
    )
    with pytest.raises(ValueError, match="overlap"):
        evaluate_checkpoint(EvaluationConfig(
            checkpoint_path=str(checkpoint_path),
            dataset_path=str(paths["test"]),
            training_dataset_path=str(paths["train"]),
            device="cpu",
        ))
