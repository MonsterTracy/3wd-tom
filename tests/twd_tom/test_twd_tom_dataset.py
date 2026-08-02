"""Tests for strict current raw first-/second-order data adaptation."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

import werewolf.models.twd_tom.dataset as dataset_module
from werewolf.models.twd_tom.belief_labels import suspicion_set_to_pair_target
from werewolf.models.twd_tom.dataset import TWDToMDataset, collate_twd_tom_samples
from werewolf.models.twd_tom.schema import PLAYER_TO_ID


REPO_ROOT = Path(__file__).resolve().parents[2]


def raw_sample(tom_order):
    name = "raw_tom.jsonl" if tom_order == 1 else "raw_tom2.jsonl"
    with (REPO_ROOT / "data" / "qwen25" / name).open(encoding="utf-8") as file:
        return json.loads(next(file))


def test_first_order_uses_only_current_observer_private_knowledge():
    sample = raw_sample(1)
    item = TWDToMDataset([sample], tom_order=1)[0]
    observer = sample["observer_ids"][0]
    observer_index = observer - 1
    assert sample["observer_ids"] == [sample["speaker_id"]]
    assert item["subject_mask"].sum().item() == 1
    assert item["subject_mask"][observer_index]
    assert item["known_werewolves"].shape == (7, 7)
    assert item["known_non_werewolves"].shape == (7, 7)
    other_rows = torch.ones(7, dtype=torch.bool)
    other_rows[observer_index] = False
    assert item["known_werewolves"][other_rows].count_nonzero().item() == 0
    assert item["known_non_werewolves"][other_rows].count_nonzero().item() == 0
    assert item["pair_targets"].shape == (7, 21)
    assert "suspicion_targets" not in item


def test_second_order_has_multiple_targets_and_no_private_model_inputs():
    sample = raw_sample(2)
    item = TWDToMDataset([sample], tom_order=2)[0]
    assert item["subject_mask"].sum().item() == len(sample["observer_ids"])
    assert len(sample["observer_ids"]) > 1
    assert "known_werewolves" not in item
    assert "known_non_werewolves" not in item
    assert item["suspicion_targets"].shape == (7, 7)
    assert "pair_targets" not in item


def test_first_order_pair_projection_semantics_are_unchanged():
    sample = raw_sample(1)
    item = TWDToMDataset([sample], tom_order=1)[0]
    subject = next(
        name for name, status in sample["belief_status"].items() if status == "ok"
    )
    expected = suspicion_set_to_pair_target(
        sample["suspected_werewolves"][subject],
        sample["known_werewolves"][subject],
        sample["known_non_werewolves"][subject],
    )
    torch.testing.assert_close(
        item["pair_targets"][PLAYER_TO_ID[subject] - 1], expected
    )


def _second_order_target(sample, subject):
    item = TWDToMDataset([sample], tom_order=2)[0]
    return item["suspicion_targets"][PLAYER_TO_ID[subject] - 1]


def test_second_order_non_empty_suspicion_is_uniform_over_reported_players():
    sample = raw_sample(2)
    subject = next(name for name, status in sample["belief_status"].items() if status == "ok")
    sample["suspected_werewolves"][subject] = ["player2", "player6"]
    expected = torch.tensor([0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0])
    torch.testing.assert_close(_second_order_target(sample, subject), expected)


def test_second_order_empty_suspicion_is_uniform_over_all_players():
    sample = raw_sample(2)
    subject = next(name for name, status in sample["belief_status"].items() if status == "ok")
    sample["suspected_werewolves"][subject] = []
    torch.testing.assert_close(
        _second_order_target(sample, subject),
        torch.full((7,), 1.0 / 7.0),
    )


def test_second_order_suspicion_keeps_the_observer_itself():
    sample = raw_sample(2)
    subject = next(name for name, status in sample["belief_status"].items() if status == "ok")
    sample["suspected_werewolves"][subject] = [subject]
    expected = torch.zeros(7)
    expected[PLAYER_TO_ID[subject] - 1] = 1.0
    torch.testing.assert_close(_second_order_target(sample, subject), expected)


@pytest.mark.parametrize(
    ("field_name", "first_value", "second_value"),
    [
        ("known_werewolves", [], ["player1"]),
        ("known_non_werewolves", [], ["player7"]),
    ],
)
def test_second_order_target_does_not_depend_on_private_knowledge(
    field_name,
    first_value,
    second_value,
):
    sample = raw_sample(2)
    subject = next(name for name, status in sample["belief_status"].items() if status == "ok")
    sample["suspected_werewolves"][subject] = ["player2", "player6"]
    sample["known_werewolves"][subject] = []
    sample["known_non_werewolves"][subject] = []
    first = deepcopy(sample)
    second = deepcopy(sample)
    first[field_name][subject] = first_value
    second[field_name][subject] = second_value
    torch.testing.assert_close(
        _second_order_target(first, subject),
        _second_order_target(second, subject),
    )


@pytest.mark.parametrize(
    "invalid_suspicion",
    ["player1", ["player8"], ["player2", "player2"]],
)
def test_second_order_suspicion_ids_are_strict(invalid_suspicion):
    sample = raw_sample(2)
    subject = next(name for name, status in sample["belief_status"].items() if status == "ok")
    sample["suspected_werewolves"][subject] = invalid_suspicion
    with pytest.raises(
        (TypeError, ValueError),
        match="suspicion list|suspected_werewolves|duplicate",
    ):
        TWDToMDataset([sample], tom_order=2)


def test_second_order_never_calls_pair_projection(monkeypatch):
    def fail_pair_projection(*args, **kwargs):
        raise AssertionError("second-order target must not use pair projection")

    monkeypatch.setattr(
        dataset_module,
        "suspicion_set_to_pair_target",
        fail_pair_projection,
    )
    item = TWDToMDataset([raw_sample(2)], tom_order=2)[0]
    assert item["suspicion_targets"].shape == (7, 7)


def test_collate_preserves_tensor_contracts_and_order_specific_private_fields():
    first = TWDToMDataset([raw_sample(1)], tom_order=1)[0]
    first_batch = collate_twd_tom_samples([first])
    assert first_batch["pair_targets"].shape == (1, 7, 21)
    assert first_batch["subject_mask"].shape == (1, 7)
    assert first_batch["known_werewolves"].shape == (1, 7, 7)

    second = TWDToMDataset([raw_sample(2)], tom_order=2)[0]
    second_batch = collate_twd_tom_samples([second])
    assert second_batch["suspicion_targets"].shape == (1, 7, 7)
    assert "pair_targets" not in second_batch
    assert "known_werewolves" not in second_batch
    assert "known_non_werewolves" not in second_batch


def test_tom_order_scope_and_private_usage_mismatches_fail():
    sample = raw_sample(1)
    with pytest.raises(ValueError, match="tom_order"):
        TWDToMDataset([sample], tom_order=2)
    broken = deepcopy(sample)
    broken["model_input_scope"] = "public_events_only"
    with pytest.raises(ValueError, match="model_input_scope"):
        TWDToMDataset([broken], tom_order=1)
    broken = deepcopy(sample)
    broken["private_fields_usage"] = "label_construction_and_audit_only"
    with pytest.raises(ValueError, match="private_fields_usage"):
        TWDToMDataset([broken], tom_order=1)


def test_event_order_and_cutoff_are_strict():
    sample = raw_sample(2)
    broken = deepcopy(sample)
    broken["public_events"][1]["event_idx"] = 9
    with pytest.raises(ValueError, match="event_idx must be continuous"):
        TWDToMDataset([broken], tom_order=2)
    broken = deepcopy(sample)
    broken["label_cutoff_step_idx"] += 1
    with pytest.raises(ValueError, match="label_cutoff_step_idx"):
        TWDToMDataset([broken], tom_order=2)


def test_extra_observer_private_mapping_is_rejected_not_ignored():
    sample = raw_sample(1)
    broken = deepcopy(sample)
    broken["known_werewolves"]["player7"] = []
    with pytest.raises(ValueError, match="subject set mismatch"):
        TWDToMDataset([broken], tom_order=1)


def test_raw_text_never_changes_model_features():
    path = REPO_ROOT / "data" / "qwen25" / "raw_tom2.jsonl"
    with path.open(encoding="utf-8") as file:
        first = next(
            sample
            for sample in (json.loads(line) for line in file)
            if any(
                event["event_type"] == "public_speech"
                for event in sample["public_events"]
            )
        )
    second = deepcopy(first)
    speech = next(e for e in second["public_events"] if e["event_type"] == "public_speech")
    speech["raw_text"] = "changed text that must not enter model features"
    from werewolf.models.twd_tom.public_events import public_event_digest

    second["public_event_digest"] = public_event_digest(second["public_events"])
    first_item = TWDToMDataset([first], tom_order=2)[0]
    second_item = TWDToMDataset([second], tom_order=2)[0]
    for field in (
        "subject_ids", "action_ids", "object_ids", "event_type_ids",
        "phase_ids", "day_values", "attention_mask",
    ):
        assert torch.equal(first_item[field], second_item[field])
