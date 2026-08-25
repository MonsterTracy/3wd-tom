import json

import pytest

from script.twd_tom.materialize_canonical_belief_dataset import (
    materialize_canonical_belief_dataset,
)
from script.twd_tom.materialize_development_folds import (
    DEVELOPMENT_FOLD_MANIFEST_SCHEMA_VERSION,
    materialize_development_folds,
    validate_development_fold_paths,
)
from script.twd_tom.train import validate_training_split_lineage
from werewolf.models.twd_tom.dataset import load_twd_tom_jsonl
from werewolf.trajectory import canonical_digest


def _source_split(
    tmp_path,
    suspicion_sample_factory,
    canonical_belief_batch_factory,
):
    canonical_root = tmp_path / "canonical"
    canonical_belief_batch_factory(
        canonical_root,
        {
            f"game_{index:02d}": [
                suspicion_sample_factory(game_id=f"game_{index:02d}")
            ]
            for index in range(10)
        },
    )
    split_dir = tmp_path / "source_split"
    manifest = materialize_canonical_belief_dataset(
        canonical_root=canonical_root,
        output_dir=split_dir,
        split_seed=42,
        train_game_count=6,
        validation_game_count=2,
        test_game_count=2,
    )
    return split_dir, manifest


def test_development_folds_cover_train_plus_validation_and_never_copy_test(
    tmp_path,
    suspicion_sample_factory,
    canonical_belief_batch_factory,
):
    split_dir, source_manifest = _source_split(
        tmp_path,
        suspicion_sample_factory,
        canonical_belief_batch_factory,
    )
    # Fold materialization is allowed to use only manifest metadata for test.
    (split_dir / "test.jsonl").unlink()
    fold_root = tmp_path / "development_folds"

    manifest = materialize_development_folds(
        train_path=split_dir / "train.jsonl",
        validation_path=split_dir / "validation.jsonl",
        output_dir=fold_root,
        fold_count=5,
        fold_seed=17,
    )

    assert manifest["schema_version"] == DEVELOPMENT_FOLD_MANIFEST_SCHEMA_VERSION
    assert manifest["source_split_manifest_digest"] == source_manifest[
        "manifest_digest"
    ]
    assert set(manifest["development_game_ids"]) == (
        set(source_manifest["game_ids"]["train"])
        | set(source_manifest["game_ids"]["validation"])
    )
    assert set(manifest["sealed_test_game_ids"]) == set(
        source_manifest["game_ids"]["test"]
    )
    assert not list(fold_root.rglob("test.jsonl"))
    validation_appearances = []
    for fold_name, descriptor in manifest["folds"].items():
        train_path = fold_root / fold_name / "train.jsonl"
        validation_path = fold_root / fold_name / "validation.jsonl"
        verified = validate_development_fold_paths(train_path, validation_path)
        assert verified["_fold_name"] == fold_name
        assert validate_training_split_lineage(
            train_path, validation_path
        )["_fold_name"] == fold_name
        train_ids = {item["game_id"] for item in load_twd_tom_jsonl(train_path)}
        validation_ids = {
            item["game_id"] for item in load_twd_tom_jsonl(validation_path)
        }
        assert train_ids.isdisjoint(validation_ids)
        assert (train_ids | validation_ids).isdisjoint(
            manifest["sealed_test_game_ids"]
        )
        assert len(validation_ids) in {1, 2}
        assert set(descriptor["validation_game_ids"]) == validation_ids
        validation_appearances.extend(validation_ids)
    assert sorted(validation_appearances) == sorted(
        manifest["development_game_ids"]
    )
    payload = dict(manifest)
    digest = payload.pop("manifest_digest")
    assert digest == canonical_digest(payload)
    assert json.loads(
        (fold_root / "development_folds_manifest.json").read_text(
            encoding="utf-8"
        )
    ) == manifest


def test_development_fold_validator_rejects_test_as_validation(
    tmp_path,
    suspicion_sample_factory,
    canonical_belief_batch_factory,
):
    split_dir, _ = _source_split(
        tmp_path,
        suspicion_sample_factory,
        canonical_belief_batch_factory,
    )
    fold_root = tmp_path / "development_folds"
    materialize_development_folds(
        train_path=split_dir / "train.jsonl",
        validation_path=split_dir / "validation.jsonl",
        output_dir=fold_root,
        fold_count=5,
    )

    with pytest.raises(ValueError, match="siblings"):
        validate_development_fold_paths(
            fold_root / "fold_0" / "train.jsonl",
            split_dir / "test.jsonl",
        )
