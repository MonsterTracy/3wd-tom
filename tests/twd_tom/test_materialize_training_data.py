from __future__ import annotations

import json
from copy import deepcopy

import pytest
import torch

import script.twd_tom.materialize_training_data as materializer_module
from script.twd_tom.materialize_training_data import (
    materialize_training_data,
    materialize_training_records,
)
from script.twd_tom.project_suspicion_to_pairs import (
    project_suspicion_sample,
)
from werewolf.models.twd_tom.dataset import (
    RAW_TRAINING_SAMPLE_FIELDS,
    TWDToMDataset,
    second_order_effective_subject_mask,
)
from werewolf.models.twd_tom.schema import (
    DETERMINISTIC_HARD_KNOWLEDGE_OBSERVER_PROVENANCE,
    FORMAL_ANNOTATION_SCHEMA_VERSION,
    FORMAL_LABEL_PROVENANCE,
    PUBLIC_ONLY_BELIEF_INFORMATION_SCOPE,
    PUBLIC_ONLY_FORMAL_ANNOTATION_SCHEMA_VERSION,
    PUBLIC_ONLY_FORMAL_LABEL_PROVENANCE,
    PUBLIC_ONLY_FORMALIZATION_POLICY_VERSION,
    PUBLIC_ONLY_LABEL_PROMPT_VERSION,
    PUBLIC_ONLY_LABEL_PROVENANCE,
    PUBLIC_ONLY_MODEL_INPUT_SCOPE,
    PUBLIC_ONLY_PRIVATE_FIELDS_USAGE,
)
from werewolf.models.twd_tom.samples import PUBLIC_ONLY_SAMPLE_SCHEMA_VERSION


def _all_ok(sample):
    sample = deepcopy(sample)
    for subject in sample["belief_status"]:
        sample["belief_status"][subject] = "ok"
        sample["belief_errors"][subject] = None
        sample["suspected_werewolves"][subject] = []
    return sample


def _semantic_error(sample, observer_id, *, unique_pair=False):
    sample = deepcopy(sample)
    subject = f"player{observer_id}"
    sample["belief_status"][subject] = "semantic_error"
    sample["belief_errors"][subject] = (
        "suspected_werewolves cannot contain known_non_werewolves"
    )
    sample["suspected_werewolves"][subject] = None
    if unique_pair:
        pair = ["player3", "player7"]
        sample["known_werewolves"][subject] = pair
        sample["known_non_werewolves"][subject] = [
            player
            for player in (f"player{index}" for index in range(1, 8))
            if player not in pair
        ]
    return sample


def _public_only(sample):
    sample = deepcopy(sample)
    sample["schema_version"] = PUBLIC_ONLY_SAMPLE_SCHEMA_VERSION
    sample["label_prompt_version"] = PUBLIC_ONLY_LABEL_PROMPT_VERSION
    sample["label_provenance"] = PUBLIC_ONLY_LABEL_PROVENANCE
    sample["known_werewolves"] = {
        subject: [] for subject in sample["known_werewolves"]
    }
    sample["known_non_werewolves"] = {
        subject: [] for subject in sample["known_non_werewolves"]
    }
    return sample


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_valid_source_observers_materialize_with_strict_current_schema(
    suspicion_sample_factory,
):
    raw = _all_ok(suspicion_sample_factory(observers=(1, 2, 3, 4)))
    result = materialize_training_records([raw])
    first = result["tom1_records"][0]
    second = result["tom2_records"][0]

    assert first["observer_ids"] == [2]
    assert second["observer_ids"] == [1, 2, 3, 4]
    assert set(first) == RAW_TRAINING_SAMPLE_FIELDS
    assert first["annotation_schema_version"] == FORMAL_ANNOTATION_SCHEMA_VERSION
    assert first["label_provenance"] == FORMAL_LABEL_PROVENANCE
    assert first["suspected_werewolves"]["player2"] == (
        raw["suspected_werewolves"]["player2"]
    )
    assert first["known_werewolves"]["player2"] == (
        raw["known_werewolves"]["player2"]
    )
    assert first["known_non_werewolves"]["player2"] == (
        raw["known_non_werewolves"]["player2"]
    )
    assert set(first["observer_label_provenance"].values()) == {
        "original_self_report"
    }
    TWDToMDataset(result["tom1_records"], tom_order=1)
    TWDToMDataset(result["tom2_records"], tom_order=2)


def test_unique_hard_knowledge_is_retained_as_exact_deterministic_target(
    suspicion_sample_factory,
):
    raw = _all_ok(suspicion_sample_factory(observers=(1, 2, 3)))
    raw = _semantic_error(raw, 3, unique_pair=True)
    result = materialize_training_records([raw])
    sample = result["tom2_records"][0]

    assert result["hard_knowledge_recovered_count"] == 1
    assert result["unresolved_observer_count"] == 0
    assert sample["source_belief_status"]["player3"] == "semantic_error"
    assert sample["belief_status"]["player3"] == "ok"
    assert sample["belief_errors"]["player3"] is None
    assert sample["suspected_werewolves"]["player3"] == ["player3", "player7"]
    assert sample["observer_label_provenance"]["player3"] == (
        DETERMINISTIC_HARD_KNOWLEDGE_OBSERVER_PROVENANCE
    )
    item = TWDToMDataset([sample], tom_order=2)[0]
    target = item["pair_targets"][2]
    assert torch.count_nonzero(target).item() == 1
    assert target.sum().item() == pytest.approx(1.0)


def test_public_only_ok_reports_materialize_with_public_lineage(
    suspicion_sample_factory,
):
    raw = _public_only(
        _all_ok(suspicion_sample_factory(observers=(1, 2, 3)))
    )
    raw["suspected_werewolves"]["player2"] = ["player4", "player6"]
    result = materialize_training_records([raw])
    first = result["tom1_records"][0]
    second = result["tom2_records"][0]
    assert result["policy_version"] == (
        PUBLIC_ONLY_FORMALIZATION_POLICY_VERSION
    )
    assert result["belief_information_scope"] == (
        PUBLIC_ONLY_BELIEF_INFORMATION_SCOPE
    )
    assert first["suspected_werewolves"]["player2"] == [
        "player4",
        "player6",
    ]
    for sample in (first, second):
        assert sample["source_schema_version"] == (
            PUBLIC_ONLY_SAMPLE_SCHEMA_VERSION
        )
        assert sample["source_label_provenance"] == (
            PUBLIC_ONLY_LABEL_PROVENANCE
        )
        assert sample["annotation_schema_version"] == (
            PUBLIC_ONLY_FORMAL_ANNOTATION_SCHEMA_VERSION
        )
        assert sample["label_provenance"] == (
            PUBLIC_ONLY_FORMAL_LABEL_PROVENANCE
        )
        assert sample["model_input_scope"] == PUBLIC_ONLY_MODEL_INPUT_SCOPE
        assert sample["private_fields_usage"] == (
            PUBLIC_ONLY_PRIVATE_FIELDS_USAGE
        )


@pytest.mark.parametrize("status", ["semantic_error", "parse_error", "reporter_error"])
def test_public_only_non_ok_is_unavailable_without_hard_knowledge_recovery(
    suspicion_sample_factory,
    monkeypatch,
    status,
):
    raw = _public_only(
        _all_ok(suspicion_sample_factory(observers=(1, 2, 3)))
    )
    subject = "player2"
    raw["belief_status"][subject] = status
    raw["belief_errors"][subject] = f"synthetic {status}"
    raw["suspected_werewolves"][subject] = None

    def forbidden_recovery(*_args, **_kwargs):
        raise AssertionError("public-only policy entered hard-knowledge recovery")

    monkeypatch.setattr(
        materializer_module,
        "close_hard_knowledge",
        forbidden_recovery,
    )
    result = materialize_training_records([raw])
    assert result["hard_knowledge_recovered_count"] == 0
    assert result["unresolved_observer_count"] == 1
    assert result["tom1_records"] == []
    assert set(result["tom2_records"][0]["observer_ids"]) == {1, 3}


def test_unresolved_speaker_drops_tom1_and_filters_tom2_atomically(
    suspicion_sample_factory,
):
    raw = _all_ok(suspicion_sample_factory(observers=(1, 2, 3, 4)))
    raw = _semantic_error(raw, 2)
    result = materialize_training_records([raw])

    assert result["tom1_records"] == []
    assert len(result["removed_tom1_snapshot_keys"]) == 1
    second = result["tom2_records"][0]
    assert second["observer_ids"] == [1, 3, 4]
    observer_fields = {
        "suspected_werewolves",
        "known_werewolves",
        "known_non_werewolves",
        "belief_status",
        "belief_errors",
        "agent_backend_ids",
        "observer_annotation_confidence",
        "observer_label_provenance",
        "source_belief_status",
        "source_belief_errors",
    }
    expected = {"player1", "player3", "player4"}
    for field_name in observer_fields:
        assert set(second[field_name]) == expected


def test_unresolved_non_speaker_keeps_tom1_and_empty_tom2_is_dropped(
    suspicion_sample_factory,
):
    raw = _all_ok(suspicion_sample_factory(observers=(1, 2, 3)))
    raw = _semantic_error(raw, 3)
    result = materialize_training_records([raw])
    assert len(result["tom1_records"]) == 1
    assert result["tom1_records"][0]["observer_ids"] == [2]
    assert result["tom2_records"][0]["observer_ids"] == [1, 2]

    only_speaker = _all_ok(suspicion_sample_factory(observers=(2,)))
    only_speaker = _semantic_error(only_speaker, 2)
    empty = materialize_training_records([only_speaker])
    assert empty["tom1_records"] == []
    assert empty["tom2_records"] == []


def test_reasoning_player_exclusion_remains_the_canonical_helper():
    subject_mask = torch.tensor([[True, True, False, True, False, False, False]])
    reasoning_player = torch.tensor([2])
    assert second_order_effective_subject_mask(
        subject_mask, reasoning_player
    ).tolist() == [[True, False, False, True, False, False, False]]


def test_projected_schema_is_not_formal(suspicion_sample_factory):
    raw = _all_ok(suspicion_sample_factory(observers=(1, 2, 3)))
    projected = project_suspicion_sample(raw)
    with pytest.raises(ValueError, match="sample field set mismatch"):
        TWDToMDataset([projected], tom_order=2)


def test_materializer_is_deterministic_atomic_and_refuses_overwrite(
    tmp_path,
    suspicion_sample_factory,
):
    raw = _all_ok(suspicion_sample_factory(observers=(1, 2, 3)))
    input_path = tmp_path / "raw.jsonl"
    first_tom1 = tmp_path / "first" / "raw_tom.jsonl"
    first_tom2 = tmp_path / "first" / "raw_tom2.jsonl"
    second_tom1 = tmp_path / "second" / "raw_tom.jsonl"
    second_tom2 = tmp_path / "second" / "raw_tom2.jsonl"
    _write_jsonl(input_path, [raw])

    materialize_training_data(
        raw_path=input_path,
        tom1_output_path=first_tom1,
        tom2_output_path=first_tom2,
    )
    materialize_training_data(
        raw_path=input_path,
        tom1_output_path=second_tom1,
        tom2_output_path=second_tom2,
    )
    assert first_tom1.read_bytes() == second_tom1.read_bytes()
    assert first_tom2.read_bytes() == second_tom2.read_bytes()
    with pytest.raises(FileExistsError, match="already exist"):
        materialize_training_data(
            raw_path=input_path,
            tom1_output_path=first_tom1,
            tom2_output_path=first_tom2,
        )
