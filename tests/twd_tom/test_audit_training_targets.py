"""Tests for the read-only second-order target seat audit."""

import json
from pathlib import Path

import pytest

from script.twd_tom.audit_training_targets import audit_training_targets
from werewolf.models.twd_tom.public_events import latest_completed_public_action
from werewolf.models.twd_tom.schema import PLAYER_NAMES


REPO_ROOT = Path(__file__).resolve().parents[2]


def _samples(split):
    path = REPO_ROOT / "data" / "qwen25" / "tom2" / f"{split}.jsonl"
    without_action = None
    with_action = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            sample = json.loads(line)
            actor_ids, _action_type = latest_completed_public_action(
                sample["public_events"]
            )
            if not actor_ids and without_action is None:
                without_action = sample
            if any(
                sample["belief_status"][PLAYER_NAMES[player_id - 1]] == "ok"
                for player_id in actor_ids
            ):
                with_action = sample
            if without_action is not None and with_action is not None:
                return [without_action, with_action]
    raise AssertionError(f"missing latest-action audit fixture in {path}")


def test_seven_epoch_rotation_audit_is_seat_symmetric_and_read_only(tmp_path):
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "val.jsonl"
    train_text = "".join(json.dumps(sample) + "\n" for sample in _samples("train"))
    validation_text = "".join(
        json.dumps(sample) + "\n" for sample in _samples("val")
    )
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
    assert before["snapshot_count"] == validation["snapshot_count"] == 2
    assert after["snapshot_count"] == 14
    for summary in (before, after, validation):
        assert 0 < summary["update_valid_observer_row_count"] <= summary[
            "valid_observer_row_count"
        ]
        assert 0 < summary["latest_action_snapshot_fraction"] < 1
        assert summary["no_latest_completed_public_action_snapshot_count"] > 0
        assert "update_conditioned_target_pair_entropy" in summary
        assert "update_conditioned_target_marginal_spread" in summary
        assert "update_conditioned_target_observer_pairwise_tv" in summary
        assert set(summary["update_rows_by_actor_id"]) == set(
            PLAYER_NAMES
        )
        assert summary["multi_actor_speech_snapshot_count"] == 0
    assert before["valid_observer_count_per_snapshot_distribution"] == {
        "0": 1,
        "1": 1,
    }
    assert after["absolute_player_marginal_mean_gap"] == pytest.approx(0.0)
    assert all(
        value == pytest.approx(2.0 / 7.0)
        for value in after["mean_target_marginal_by_player"].values()
    )
    assert train_path.read_text(encoding="utf-8") == train_text
    assert validation_path.read_text(encoding="utf-8") == validation_text
