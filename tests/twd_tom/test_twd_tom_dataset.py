"""Tests for strict current raw first-/second-order data adaptation."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

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


def test_second_order_has_multiple_targets_and_no_private_model_inputs():
    sample = raw_sample(2)
    item = TWDToMDataset([sample], tom_order=2)[0]
    assert item["subject_mask"].sum().item() == len(sample["observer_ids"])
    assert len(sample["observer_ids"]) > 1
    assert "known_werewolves" not in item
    assert "known_non_werewolves" not in item
    assert item["pair_targets"].shape == (7, 21)


@pytest.mark.parametrize("tom_order", [1, 2])
def test_pair_projection_semantics_are_unchanged(tom_order):
    sample = raw_sample(tom_order)
    item = TWDToMDataset([sample], tom_order=tom_order)[0]
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


def test_collate_preserves_tensor_contracts_and_order_specific_private_fields():
    first = TWDToMDataset([raw_sample(1)], tom_order=1)[0]
    first_batch = collate_twd_tom_samples([first])
    assert first_batch["pair_targets"].shape == (1, 7, 21)
    assert first_batch["subject_mask"].shape == (1, 7)
    assert first_batch["known_werewolves"].shape == (1, 7, 7)

    second = TWDToMDataset([raw_sample(2)], tom_order=2)[0]
    second_batch = collate_twd_tom_samples([second])
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
