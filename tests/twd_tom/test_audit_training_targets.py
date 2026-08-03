"""Tests for the read-only second-order target seat audit."""

import json

import pytest

from script.twd_tom.audit_training_targets import audit_training_targets
from werewolf.models.twd_tom.schema import PLAYER_NAMES
from tests.twd_tom.public_event_fixtures import (
    make_public_events,
    make_training_sample,
)


def _samples(split):
    without_action = make_training_sample(
        2,
        game_id=f"synthetic_{split}_without_action",
        with_latest_action=False,
    )
    events = make_public_events(
        [["player2", "support", "player4"]],
        speaker_id=2,
    )
    with_action = make_training_sample(
        2,
        game_id=f"synthetic_{split}_with_action",
        public_events=events,
    )
    return [without_action, with_action]


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
