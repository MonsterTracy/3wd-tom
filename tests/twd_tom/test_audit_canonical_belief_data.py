import json

import pytest

from script.twd_tom.audit_canonical_belief_data import (
    AUDIT_SCHEMA_VERSION,
    audit_canonical_belief_data,
)
from werewolf.trajectory import canonical_digest


def test_audit_reports_label_and_length_truncation_statistics(
    tmp_path,
    suspicion_sample_factory,
    canonical_belief_batch_factory,
):
    root = tmp_path / "canonical"
    first = suspicion_sample_factory(game_id="game_1")
    second = suspicion_sample_factory(
        game_id="game_2",
        observers=(1, 2, 3),
        suspicions_by_observer={1: [], 2: ["player7"], 3: ["player6", "player7"]},
    )
    canonical_belief_batch_factory(
        root,
        {"game_1": [first], "game_2": [second]},
    )
    output = tmp_path / "audit.json"

    report = audit_canonical_belief_data(
        canonical_root=root,
        max_seq_len=1,
        output_path=output,
    )

    assert AUDIT_SCHEMA_VERSION == "classic7_canonical_belief_data_audit_v3"
    assert report["schema_version"] == AUDIT_SCHEMA_VERSION
    assert report["status"] == "PASS"
    assert report["game_count"] == 2
    assert report["sample_count"] == 2
    assert report["observer_report_count"] == 7
    assert report["status_counts"] == {"ok": 7}
    assert report["collector_raw_structured_token_count"] == {
        "min": 5,
        "max": 5,
        "mean": 5.0,
    }
    assert report["model_input_structured_token_count"] == {
        "min": 4,
        "max": 4,
        "mean": 4.0,
    }
    assert report["retained_structured_token_count"] == {
        "min": 1,
        "max": 1,
        "mean": 1.0,
    }
    assert report["terminal_turn_start_removed_sample_count"] == 2
    assert report["length_truncated_sample_count"] == 2
    assert report["length_truncated_sample_fraction"] == 1.0
    assert "raw_structured_token_count" not in report
    assert "truncated_sample_count" not in report
    assert "truncated_sample_fraction" not in report
    payload = dict(report)
    digest = payload.pop("audit_digest")
    assert digest == canonical_digest(payload)
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_audit_does_not_count_terminal_pre_marker_as_length_truncation(
    tmp_path,
    suspicion_sample_factory,
    canonical_belief_batch_factory,
):
    root = tmp_path / "canonical"
    sample = suspicion_sample_factory(game_id="game_1")
    canonical_belief_batch_factory(root, {"game_1": [sample]})

    report = audit_canonical_belief_data(
        canonical_root=root,
        max_seq_len=256,
    )

    assert report["collector_raw_structured_token_count"] == {
        "min": 5,
        "max": 5,
        "mean": 5.0,
    }
    assert report["model_input_structured_token_count"] == {
        "min": 4,
        "max": 4,
        "mean": 4.0,
    }
    assert report["retained_structured_token_count"] == {
        "min": 4,
        "max": 4,
        "mean": 4.0,
    }
    assert report["terminal_turn_start_removed_sample_count"] == 1
    assert report["length_truncated_sample_count"] == 0
    assert report["length_truncated_sample_fraction"] == 0.0


def test_audit_rejects_failed_observer_before_dataset_materialization(
    tmp_path,
    suspicion_sample_factory,
    canonical_belief_batch_factory,
):
    root = tmp_path / "canonical"
    sample = suspicion_sample_factory(game_id="game_1", failed_observer=2)
    canonical_belief_batch_factory(root, {"game_1": [sample]})

    with pytest.raises(ValueError, match="status=ok.*failed_report_count=1"):
        audit_canonical_belief_data(canonical_root=root)


def test_audit_rejects_failed_batch_marker(
    tmp_path,
    suspicion_sample_factory,
    canonical_belief_batch_factory,
):
    root = tmp_path / "canonical"
    sample = suspicion_sample_factory(game_id="game_1")
    canonical_belief_batch_factory(root, {"game_1": [sample]})
    (root / "batch_failure.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="contains batch_failure.json"):
        audit_canonical_belief_data(canonical_root=root)


def test_audit_rejects_snapshot_changed_after_game_summary(
    tmp_path,
    suspicion_sample_factory,
    canonical_belief_batch_factory,
):
    root = tmp_path / "canonical"
    sample = suspicion_sample_factory(game_id="game_1")
    canonical_belief_batch_factory(root, {"game_1": [sample]})
    belief_path = (
        root
        / "games"
        / "game_0001_seed_1001"
        / "belief_snapshots.jsonl"
    )
    belief_path.write_text(belief_path.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        audit_canonical_belief_data(canonical_root=root)


def test_audit_rejects_missing_success_summary(
    tmp_path,
    suspicion_sample_factory,
    canonical_belief_batch_factory,
):
    root = tmp_path / "canonical"
    sample = suspicion_sample_factory(game_id="game_1")
    canonical_belief_batch_factory(root, {"game_1": [sample]})
    (root / "summary.json").unlink()

    with pytest.raises(FileNotFoundError, match="summary.json"):
        audit_canonical_belief_data(canonical_root=root)


def test_audit_rejects_unlisted_game_directory(
    tmp_path,
    suspicion_sample_factory,
    canonical_belief_batch_factory,
):
    root = tmp_path / "canonical"
    sample = suspicion_sample_factory(game_id="game_1")
    canonical_belief_batch_factory(root, {"game_1": [sample]})
    (root / "games" / "unexpected_game").mkdir()

    with pytest.raises(ValueError, match="game directory count mismatch"):
        audit_canonical_belief_data(canonical_root=root)
