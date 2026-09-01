import json

import pytest

from script.twd_tom.audit_dense_belief_dataset import (
    DENSE_AUDIT_SCHEMA_VERSION,
    audit_dense_belief_dataset,
)
from script.twd_tom.materialize_canonical_belief_dataset import (
    materialize_canonical_belief_dataset,
)
from werewolf.models.twd_tom.dense_dataset import DENSE_SUPERVISION_VERSION
from werewolf.models.twd_tom.annotation_v2 import (
    V1_EMPTY_UNIFORM_NONSELF_BELIEF_SOURCE,
)
from werewolf.models.twd_tom.dataset import (
    LABEL_OBSERVATION_SEMANTICS,
    TARGET_CONVERSION,
    TARGET_SEMANTICS,
)
from werewolf.trajectory import canonical_digest


def _materialized_three_game_split(
    tmp_path,
    suspicion_sample_factory,
    canonical_belief_batch_factory,
):
    canonical_root = tmp_path / "canonical"
    canonical_belief_batch_factory(
        canonical_root,
        {
            f"game_{index}": [
                suspicion_sample_factory(game_id=f"game_{index}")
            ]
            for index in range(1, 4)
        },
    )
    output_dir = tmp_path / "dataset"
    materialize_canonical_belief_dataset(
        canonical_root=canonical_root,
        output_dir=output_dir,
        split_seed=11,
        train_game_count=1,
        validation_game_count=1,
        test_game_count=1,
    )
    return output_dir


def test_dense_audit_reports_strict_pre_contract(
    tmp_path,
    suspicion_sample_factory,
    canonical_belief_batch_factory,
):
    dataset_dir = _materialized_three_game_split(
        tmp_path,
        suspicion_sample_factory,
        canonical_belief_batch_factory,
    )
    output_path = tmp_path / "dense_audit.json"
    # A development audit must not open the sealed test file.
    (dataset_dir / "test.jsonl").unlink()

    report = audit_dense_belief_dataset(
        dataset_path=dataset_dir / "train.jsonl",
        split_name="train",
        output_path=output_path,
    )

    assert report["schema_version"] == DENSE_AUDIT_SCHEMA_VERSION
    assert report["status"] == "PASS"
    assert report["training_supervision"] == DENSE_SUPERVISION_VERSION
    assert report["target_semantics"] == TARGET_SEMANTICS
    assert report["target_conversion"] == TARGET_CONVERSION
    assert report["label_observation_semantics"] == (
        LABEL_OBSERVATION_SEMANTICS
    )
    assert report["belief_annotation_source"] == (
        V1_EMPTY_UNIFORM_NONSELF_BELIEF_SOURCE
    )
    assert report["game_count"] == 1
    assert report["boundary_count"] == 1
    assert report["causal_contract"] == {
        "target_time": "strict_pre_speech",
        "boundary_index_semantics": "inclusive_last_visible_token",
        "future_tokens_visible": False,
        "terminal_turn_start_visible": False,
        "prefix_relation": "exact_encoded_prefix",
    }
    payload = dict(report)
    digest = payload.pop("audit_digest")
    assert digest == canonical_digest(payload)
    assert json.loads(output_path.read_text(encoding="utf-8")) == report


def test_dense_audit_distinguishes_alive_from_supervised_observers(
    tmp_path,
    suspicion_sample_factory,
    canonical_belief_batch_factory,
):
    dataset_dir = _materialized_three_game_split(
        tmp_path,
        suspicion_sample_factory,
        canonical_belief_batch_factory,
    )
    (dataset_dir / "test.jsonl").unlink()

    report = audit_dense_belief_dataset(
        dataset_path=dataset_dir / "train.jsonl",
        split_name="train",
    )

    # The fixture has four alive observers, including one observed empty report.
    assert report["alive_observer_count"] == 4
    assert report["supervised_observer_count"] == 4
    assert report["alive_observers_per_game"] == {
        "min": 4,
        "max": 4,
        "mean": 4.0,
    }
    assert report["supervised_observers_per_game"] == {
        "min": 4,
        "max": 4,
        "mean": 4.0,
    }


def test_dense_audit_never_accepts_test_split(
    tmp_path,
    suspicion_sample_factory,
    canonical_belief_batch_factory,
):
    dataset_dir = _materialized_three_game_split(
        tmp_path,
        suspicion_sample_factory,
        canonical_belief_batch_factory,
    )

    with pytest.raises(ValueError, match="split_name"):
        audit_dense_belief_dataset(
            dataset_path=dataset_dir / "test.jsonl",
            split_name="test",
        )
