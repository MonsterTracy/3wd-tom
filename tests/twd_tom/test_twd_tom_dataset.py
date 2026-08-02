"""Tests for strict current raw first-/second-order data adaptation."""

import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest
import torch

import werewolf.models.twd_tom.dataset as dataset_module
from werewolf.models.twd_tom.belief_labels import suspicion_set_to_pair_target
from werewolf.models.twd_tom.dataset import (
    SUBJECT_MAPPING_FIELDS,
    TWDToMDataset,
    collate_twd_tom_samples,
    cyclically_rotate_second_order_sample,
    deterministic_cyclic_shift,
)
from werewolf.models.twd_tom.public_events import (
    latest_completed_public_action,
    latest_completed_public_action_mask,
    structured_event_tokens,
)
from werewolf.models.twd_tom.schema import (
    PLAYER_NAMES,
    PLAYER_TO_ID,
    canonical_wolf_pairs,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TOM2_SPLIT_SHA256 = {
    "train": "31b99dc26ba9724c6ee8390a184b01faca50ec830313fcac3e5c873e5fa1e1a8",
    "val": "5b7f9206e4a8a59938a5aba50ba68f0d7c6cd9079b377d1a25aedd349151f0c8",
    "test": "91622732b35dc0a6ca17dcb4b74292514306db128d9a2a9ee1f9538df2c03a6d",
}


def raw_sample(tom_order):
    name = "raw_tom.jsonl" if tom_order == 1 else "raw_tom2.jsonl"
    with (REPO_ROOT / "data" / "qwen25" / name).open(encoding="utf-8") as file:
        return json.loads(next(file))


def second_order_sample(*, with_latest_action):
    path = REPO_ROOT / "data" / "qwen25" / "tom2" / "train.jsonl"
    with path.open(encoding="utf-8") as file:
        for line in file:
            sample = json.loads(line)
            actor_ids, _action_type = latest_completed_public_action(
                sample["public_events"]
            )
            has_valid_actor = any(
                sample["belief_status"][PLAYER_NAMES[player_id - 1]] == "ok"
                for player_id in actor_ids
            )
            if (
                (with_latest_action and has_valid_actor)
                or (not with_latest_action and not actor_ids)
            ):
                return sample
    raise AssertionError("missing requested second-order sample fixture")


def test_second_order_formal_split_files_are_unchanged():
    for split, expected in TOM2_SPLIT_SHA256.items():
        path = REPO_ROOT / "data" / "qwen25" / "tom2" / f"{split}.jsonl"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def full_history_second_order_sample():
    required = {
        "public_speech",
        "vote_result",
        "exile_result",
        "death_announcement",
    }
    path = REPO_ROOT / "data" / "qwen25" / "tom2" / "train.jsonl"
    with path.open(encoding="utf-8") as file:
        return next(
            sample
            for sample in (json.loads(line) for line in file)
            if required
            <= {event["event_type"] for event in sample["public_events"]}
        )


def rotated_player(player, shift):
    if player is None:
        return None
    return PLAYER_NAMES[(PLAYER_TO_ID[player] - 1 + shift) % 7]


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
    assert item["pair_targets"].shape == (7, 21)
    assert item["latest_completed_public_action_mask"].shape == (7,)
    assert "suspicion_targets" not in item


def test_latest_completed_action_mask_is_zero_before_first_speech():
    events = [
        {
            "event_idx": 0,
            "event_type": "phase_change",
            "phase": "1_day_speech",
        },
        {
            "event_idx": 1,
            "event_type": "turn_start",
            "speaker": "player4",
        },
    ]
    assert latest_completed_public_action(events) == ((), None)
    assert latest_completed_public_action_mask(events) == (False,) * 7


def test_system_events_do_not_create_latest_action_actors():
    events = [
        {
            "event_idx": 0,
            "event_type": "death_announcement",
            "dead_players": ["player2"],
        },
        {
            "event_idx": 1,
            "event_type": "exile_result",
            "exiled_players": ["player3"],
        },
        {
            "event_idx": 2,
            "event_type": "phase_change",
            "phase": "2_day_speech",
        },
        {
            "event_idx": 3,
            "event_type": "turn_start",
            "speaker": "player4",
        },
    ]
    assert latest_completed_public_action(events) == ((), None)
    assert latest_completed_public_action_mask(events) == (False,) * 7


def test_latest_speech_actor_replaces_earlier_actor_without_accumulating():
    events = [
        {
            "event_idx": 0,
            "event_type": "phase_change",
            "phase": "1_day_speech",
        },
        {
            "event_idx": 1,
            "event_type": "turn_start",
            "speaker": "player4",
        },
        {
            "event_idx": 2,
            "event_type": "public_speech",
            "speaker": "player4",
            "raw_text": "completed player4 speech",
            "sp_actions": [["player4", "support", "player2"]],
        },
        {
            "event_idx": 3,
            "event_type": "turn_start",
            "speaker": "player5",
        },
    ]
    assert latest_completed_public_action(events) == ((4,), "public_speech")
    assert latest_completed_public_action_mask(events) == (
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    )
    events.extend(
        [
            {
                "event_idx": 4,
                "event_type": "public_speech",
                "speaker": "player5",
                "raw_text": "completed player5 speech",
                "sp_actions": [],
            },
            {
                "event_idx": 5,
                "event_type": "turn_start",
                "speaker": "player6",
            },
        ]
    )
    assert latest_completed_public_action(events) == ((5,), "public_speech")
    assert latest_completed_public_action_mask(events) == (
        False,
        False,
        False,
        False,
        True,
        False,
        False,
    )


def test_latest_speech_uses_speaker_and_action_subject_not_object():
    events = [
        {
            "event_idx": 0,
            "event_type": "phase_change",
            "phase": "1_day_speech",
        },
        {
            "event_idx": 1,
            "event_type": "public_speech",
            "speaker": "player4",
            "raw_text": "completed speech",
            "sp_actions": [["player3", "support", "player2"]],
        },
        {
            "event_idx": 2,
            "event_type": "turn_start",
            "speaker": "player5",
        },
    ]
    assert latest_completed_public_action(events) == (
        (3, 4),
        "public_speech",
    )
    assert latest_completed_public_action_mask(events)[1] is False


def test_latest_vote_block_uses_voters_not_targets_and_skips_system_events():
    events = [
        {
            "event_idx": 0,
            "event_type": "phase_change",
            "phase": "1_day_vote",
        },
        {
            "event_idx": 1,
            "event_type": "vote_result",
            "votes": [
                {"voter": "player1", "target": "player2"},
                {"voter": "player3", "target": "player1"},
            ],
        },
        {
            "event_idx": 2,
            "event_type": "exile_result",
            "exiled_players": ["player2"],
        },
        {
            "event_idx": 3,
            "event_type": "death_announcement",
            "dead_players": ["player4"],
        },
        {
            "event_idx": 4,
            "event_type": "phase_change",
            "phase": "2_day_speech",
        },
        {
            "event_idx": 5,
            "event_type": "turn_start",
            "speaker": "player5",
        },
    ]
    assert latest_completed_public_action(events) == ((1, 3), "vote_result")
    assert latest_completed_public_action_mask(events) == (
        True,
        False,
        True,
        False,
        False,
        False,
        False,
    )


def test_second_order_indices_filter_zero_effective_mask_deterministically():
    without_action = second_order_sample(with_latest_action=False)
    with_action = second_order_sample(with_latest_action=True)
    dataset = TWDToMDataset([without_action, with_action], tom_order=2)
    assert dataset.second_order_supervised_indices() == (1,)
    assert dataset.second_order_supervised_indices() == (1,)


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
    return item["pair_targets"][PLAYER_TO_ID[subject] - 1]


def test_second_order_pair_target_uses_suspicion_and_private_label_knowledge():
    sample = raw_sample(2)
    subject = next(
        name for name, status in sample["belief_status"].items() if status == "ok"
    )
    expected = suspicion_set_to_pair_target(
        sample["suspected_werewolves"][subject],
        sample["known_werewolves"][subject],
        sample["known_non_werewolves"][subject],
    )
    torch.testing.assert_close(_second_order_target(sample, subject), expected)


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


def test_second_order_calls_the_shared_pair_projection(monkeypatch):
    calls = []

    def record_pair_projection(*args, **kwargs):
        calls.append((args, kwargs))
        return suspicion_set_to_pair_target(*args, **kwargs)

    monkeypatch.setattr(
        dataset_module,
        "suspicion_set_to_pair_target",
        record_pair_projection,
    )
    item = TWDToMDataset([raw_sample(2)], tom_order=2)[0]
    assert item["pair_targets"].shape == (7, 21)
    assert calls


def test_collate_preserves_tensor_contracts_and_order_specific_private_fields():
    first = TWDToMDataset([raw_sample(1)], tom_order=1)[0]
    first_batch = collate_twd_tom_samples([first])
    assert first_batch["pair_targets"].shape == (1, 7, 21)
    assert first_batch["subject_mask"].shape == (1, 7)
    assert first_batch["known_werewolves"].shape == (1, 7, 7)

    second = TWDToMDataset([raw_sample(2)], tom_order=2)[0]
    second_batch = collate_twd_tom_samples([second])
    assert second_batch["pair_targets"].shape == (1, 7, 21)
    assert second_batch["latest_completed_public_action_mask"].shape == (1, 7)
    assert "suspicion_targets" not in second_batch
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


def test_cyclic_rotation_identity_composition_and_detachment():
    sample = full_history_second_order_sample()
    original = deepcopy(sample)
    assert cyclically_rotate_second_order_sample(sample, shift=0) == sample
    assert cyclically_rotate_second_order_sample(sample, shift=7) == sample
    sequential = cyclically_rotate_second_order_sample(
        cyclically_rotate_second_order_sample(sample, shift=2),
        shift=4,
    )
    combined = cyclically_rotate_second_order_sample(sample, shift=6)
    assert sequential == combined
    assert sample == original


def test_cyclic_rotation_updates_every_structured_player_field():
    sample = full_history_second_order_sample()
    rotated = cyclically_rotate_second_order_sample(sample, shift=3)
    assert rotated["speaker_id"] == ((sample["speaker_id"] - 1 + 3) % 7) + 1
    assert rotated["observer_ids"] == [
        ((player_id - 1 + 3) % 7) + 1
        for player_id in sample["observer_ids"]
    ]
    for field_name in SUBJECT_MAPPING_FIELDS:
        assert set(rotated[field_name]) == {
            rotated_player(subject, 3) for subject in sample[field_name]
        }
    for field_name in (
        "suspected_werewolves",
        "known_werewolves",
        "known_non_werewolves",
    ):
        for subject, values in sample[field_name].items():
            rotated_values = rotated[field_name][rotated_player(subject, 3)]
            assert set(rotated_values or []) == {
                rotated_player(player, 3) for player in values or []
            }
    original_tokens = structured_event_tokens(sample["public_events"])
    rotated_tokens = structured_event_tokens(rotated["public_events"])
    assert [token["token_type"] for token in rotated_tokens] == [
        token["token_type"] for token in original_tokens
    ]
    expected_tokens = Counter(
        (
            token["token_type"],
            rotated_player(token["subject"], 3),
            token["action"],
            rotated_player(token["object"], 3),
            token["phase"],
            token["day"],
        )
        for token in original_tokens
    )
    actual_tokens = Counter(
        (
            token["token_type"],
            token["subject"],
            token["action"],
            token["object"],
            token["phase"],
            token["day"],
        )
        for token in rotated_tokens
    )
    assert actual_tokens == expected_tokens
    assert [event["event_idx"] for event in rotated["public_events"]] == [
        event["event_idx"] for event in sample["public_events"]
    ]
    assert [event["event_type"] for event in rotated["public_events"]] == [
        event["event_type"] for event in sample["public_events"]
    ]
    assert [
        event.get("raw_text")
        for event in rotated["public_events"]
        if event["event_type"] == "public_speech"
    ] == [
        event.get("raw_text")
        for event in sample["public_events"]
        if event["event_type"] == "public_speech"
    ]


def test_rotated_one_hot_pair_target_uses_canonical_pair_class():
    sample = raw_sample(2)
    subject = next(
        name for name, status in sample["belief_status"].items() if status == "ok"
    )
    pair = ["player1", "player3"]
    sample["known_werewolves"][subject] = pair
    sample["known_non_werewolves"][subject] = [
        player for player in PLAYER_NAMES if player not in pair
    ]
    sample["suspected_werewolves"][subject] = pair
    rotated = cyclically_rotate_second_order_sample(sample, shift=2)
    item = TWDToMDataset([rotated], tom_order=2)[0]
    rotated_subject = rotated_player(subject, 2)
    rotated_pair = tuple(
        sorted(
            (rotated_player(player, 2) for player in pair),
            key=PLAYER_TO_ID.__getitem__,
        )
    )
    pair_index = canonical_wolf_pairs().index(rotated_pair)
    target = item["pair_targets"][PLAYER_TO_ID[rotated_subject] - 1]
    assert target.shape == (21,)
    assert target[pair_index] == 1
    assert target.sum() == 1


def test_train_rotation_is_deterministic_epoch_dependent_and_train_only():
    sample = raw_sample(2)
    original = deepcopy(sample)
    first = TWDToMDataset(
        [sample],
        tom_order=2,
        enable_cyclic_rotation=True,
        augmentation_seed=42,
    )
    second = TWDToMDataset(
        [sample],
        tom_order=2,
        enable_cyclic_rotation=True,
        augmentation_seed=42,
    )
    observed_speakers = set()
    for epoch in range(7):
        first.set_epoch(epoch)
        second.set_epoch(epoch)
        first_item = first[0]
        second_item = second[0]
        assert torch.equal(first_item["pair_targets"], second_item["pair_targets"])
        assert first_item["metadata"] == second_item["metadata"]
        assert first_item["pair_targets"].shape == (7, 21)
        torch.testing.assert_close(
            first_item["pair_targets"][first_item["subject_mask"]].sum(dim=-1),
            torch.ones(int(first_item["subject_mask"].sum().item())),
        )
        assert "known_werewolves" not in first_item
        assert "known_non_werewolves" not in first_item
        observed_speakers.add(first_item["metadata"]["speaker_id"])
        assert deterministic_cyclic_shift(
            seed=42, epoch=epoch, sample_index=0
        ) == deterministic_cyclic_shift(seed=42, epoch=epoch, sample_index=0)
    assert len(observed_speakers) == 7
    validation = TWDToMDataset([sample], tom_order=2)
    unchanged = validation[0]
    validation.set_epoch(6)
    assert torch.equal(unchanged["pair_targets"], validation[0]["pair_targets"])
    assert unchanged["metadata"] == validation[0]["metadata"]
    assert sample == original
    with pytest.raises(ValueError, match="tom_order=2"):
        TWDToMDataset(
            [raw_sample(1)],
            tom_order=1,
            enable_cyclic_rotation=True,
        )


def test_cyclic_rotation_rejects_illegal_player_id():
    sample = raw_sample(2)
    sample["public_events"][-1]["speaker"] = "player8"
    with pytest.raises(ValueError, match="canonical player"):
        cyclically_rotate_second_order_sample(sample, shift=1)
