import json

import pytest

from script.twd_tom.audit_canonical_belief_data import (
    AUDIT_SCHEMA_VERSION,
    audit_canonical_belief_data,
)
from werewolf.trajectory import canonical_digest


def _write_game(root, directory_name, sample):
    game_dir = root / "games" / directory_name
    game_dir.mkdir(parents=True)
    (game_dir / "belief_snapshots.jsonl").write_text(
        json.dumps(sample, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_audit_reports_label_and_truncation_statistics(
    tmp_path,
    suspicion_sample_factory,
):
    root = tmp_path / "canonical"
    first = suspicion_sample_factory(game_id="game_1")
    second = suspicion_sample_factory(
        game_id="game_2",
        observers=(1, 2, 3),
        suspicions_by_observer={1: [], 2: ["player7"], 3: ["player6", "player7"]},
    )
    _write_game(root, "game_0001", first)
    _write_game(root, "game_0002", second)
    output = tmp_path / "audit.json"

    report = audit_canonical_belief_data(
        canonical_root=root,
        max_seq_len=1,
        output_path=output,
    )

    assert report["schema_version"] == AUDIT_SCHEMA_VERSION
    assert report["status"] == "PASS"
    assert report["game_count"] == 2
    assert report["sample_count"] == 2
    assert report["observer_report_count"] == 7
    assert report["status_counts"] == {"ok": 7}
    assert report["truncated_sample_count"] == 2
    assert report["truncated_sample_fraction"] == 1.0
    payload = dict(report)
    digest = payload.pop("audit_digest")
    assert digest == canonical_digest(payload)
    assert json.loads(output.read_text()) == report


def test_audit_rejects_failed_observer_before_dataset_materialization(
    tmp_path,
    suspicion_sample_factory,
):
    root = tmp_path / "canonical"
    sample = suspicion_sample_factory(game_id="game_1", failed_observer=2)
    _write_game(root, "game_0001", sample)

    with pytest.raises(ValueError, match="status=ok.*failed_report_count=1"):
        audit_canonical_belief_data(canonical_root=root)


def test_audit_rejects_duplicate_game_ids_across_files(
    tmp_path,
    suspicion_sample_factory,
):
    root = tmp_path / "canonical"
    sample = suspicion_sample_factory(game_id="same_game")
    _write_game(root, "game_0001", sample)
    _write_game(root, "game_0002", sample)

    with pytest.raises(ValueError, match="duplicate canonical game_id"):
        audit_canonical_belief_data(canonical_root=root)
