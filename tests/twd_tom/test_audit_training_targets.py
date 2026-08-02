"""Tests for the read-only second-order target seat audit."""

import json
from pathlib import Path

import pytest

from script.twd_tom.audit_training_targets import audit_training_targets


REPO_ROOT = Path(__file__).resolve().parents[2]


def _first_sample(split):
    path = REPO_ROOT / "data" / "qwen25" / "tom2" / f"{split}.jsonl"
    with path.open(encoding="utf-8") as handle:
        return json.loads(next(handle))


def test_seven_epoch_rotation_audit_is_seat_symmetric_and_read_only(tmp_path):
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "val.jsonl"
    train_text = json.dumps(_first_sample("train")) + "\n"
    validation_text = json.dumps(_first_sample("val")) + "\n"
    train_path.write_text(train_text, encoding="utf-8")
    validation_path.write_text(validation_text, encoding="utf-8")

    result = audit_training_targets(
        train_path=train_path,
        validation_path=validation_path,
        seed=42,
    )

    assert result["augmentation_epochs"] == list(range(7))
    before = result["train_before_augmentation"]["overall"]
    after = result["train_after_seven_epoch_rotation_orbit"]["overall"]
    validation = result["validation_without_augmentation"]["overall"]
    assert before["snapshot_count"] == validation["snapshot_count"] == 1
    assert after["snapshot_count"] == 7
    assert after["absolute_player_marginal_mean_gap"] == pytest.approx(0.0)
    assert all(
        value == pytest.approx(2.0 / 7.0)
        for value in after["mean_target_marginal_by_player"].values()
    )
    assert train_path.read_text(encoding="utf-8") == train_text
    assert validation_path.read_text(encoding="utf-8") == validation_text
