"""Tests for the sole tom-v2 observer-conditioned belief Dataset."""

import inspect
import json
from copy import deepcopy

import pytest
import torch

from werewolf.models.twd_tom.belief_labels import (
    suspicion_set_to_belief_vector,
)
from werewolf.models.twd_tom.dataset import (
    CYCLIC_ROTATION_VERSION,
    MODEL_INPUT_SCOPE,
    TARGET_CONVERSION,
    TWDToMDataset,
    collate_twd_tom_samples,
    cyclically_rotate_belief_sample,
    deterministic_cyclic_shift,
    load_twd_tom_jsonl,
)
from werewolf.models.twd_tom.public_events import (
    public_event_digest,
    structured_input_digest,
)
from werewolf.models.twd_tom.schema import PLAYER_TO_ID


def test_dataset_consumes_raw_self_report_and_emits_fixed_belief_contract(
    suspicion_sample_factory,
):
    sample = suspicion_sample_factory()
    item = TWDToMDataset([sample])[0]

    assert item["belief_targets"].shape == (7, 7)
    assert item["observer_alive_mask"].shape == (7,)
    assert item["diagonal_target_mask"].shape == (7, 7)
    assert item["belief_targets"].dtype == torch.float32
    assert item["observer_alive_mask"].dtype == torch.bool
    assert item["diagonal_target_mask"].dtype == torch.bool
    assert item["metadata"]["target_conversion"] == TARGET_CONVERSION
    assert "pair_targets" not in item
    assert "known_werewolves" not in item
    assert "known_non_werewolves" not in item
    assert "reasoning_player_id" not in item


def test_each_alive_observer_row_matches_the_deterministic_conversion(
    suspicion_sample_factory,
):
    sample = suspicion_sample_factory(
        suspicions_by_observer={1: ["player3", "player7"], 2: []}
    )
    item = TWDToMDataset([sample], target_dtype=torch.float64)[0]
    for observer_id in sample["observer_ids"]:
        subject = f"player{observer_id}"
        expected = suspicion_set_to_belief_vector(
            sample["suspected_werewolves"][subject],
            observer_id=subject,
            dtype=torch.float64,
        )
        torch.testing.assert_close(item["belief_targets"][observer_id - 1], expected)


def test_observer_alive_mask_uses_observer_ids_only(suspicion_sample_factory):
    sample = suspicion_sample_factory(observers=(2, 4, 7))
    item = TWDToMDataset([sample])[0]
    assert item["observer_alive_mask"].tolist() == [
        False,
        True,
        False,
        True,
        False,
        False,
        True,
    ]
    assert torch.count_nonzero(item["belief_targets"][~item["observer_alive_mask"]]) == 0


def test_diagonal_target_mask_excludes_only_self_targets(
    suspicion_sample_factory,
):
    item = TWDToMDataset([suspicion_sample_factory()])[0]
    mask = item["diagonal_target_mask"]
    assert not mask.diagonal().any()
    assert int(mask.sum().item()) == 42
    for observer_index in range(7):
        assert mask[observer_index].sum().item() == 6


def test_dead_player_columns_remain_valid_targets(suspicion_sample_factory):
    sample = suspicion_sample_factory(
        observers=(1, 2, 3),
        suspicions_by_observer={1: ["player7"], 2: ["player7"], 3: []},
    )
    item = TWDToMDataset([sample])[0]
    assert not item["observer_alive_mask"][6]
    assert item["diagonal_target_mask"][:6, 6].all()
    player1_row = item["belief_targets"][0]
    assert player1_row[6] > player1_row[1]


def test_all_alive_rows_are_normalized_and_have_zero_diagonal(
    suspicion_sample_factory,
):
    item = TWDToMDataset([suspicion_sample_factory()])[0]
    alive_targets = item["belief_targets"][item["observer_alive_mask"]]
    torch.testing.assert_close(
        alive_targets.sum(dim=-1),
        torch.ones(alive_targets.shape[0]),
    )
    assert torch.equal(item["belief_targets"].diagonal(), torch.zeros(7))


def test_dataset_api_contains_no_tom_order(suspicion_sample_factory):
    parameters = inspect.signature(TWDToMDataset).parameters
    from_jsonl_parameters = inspect.signature(TWDToMDataset.from_jsonl).parameters
    assert "tom_order" not in parameters
    assert "tom_order" not in from_jsonl_parameters
    assert "enable_cyclic_rotation" in parameters
    dataset = TWDToMDataset([suspicion_sample_factory()])
    assert not hasattr(dataset, "tom_order")
    assert hasattr(dataset, "set_epoch")
    assert dataset.model_input_scope == MODEL_INPUT_SCOPE


def test_generic_rotation_moves_observers_targets_public_references_and_masks(
    suspicion_sample_factory,
):
    sample = suspicion_sample_factory(
        observers=(1, 2, 5),
        suspicions_by_observer={1: ["player4"], 2: ["player7"], 5: ["player3"]},
    )
    baseline = TWDToMDataset([sample])[0]
    rotated_raw = cyclically_rotate_belief_sample(sample, shift=2)
    assert rotated_raw["observer_ids"] == [3, 4, 7]
    assert rotated_raw["speaker_id"] == 4
    speech = next(
        event for event in rotated_raw["public_events"]
        if event["event_type"] == "public_speech"
    )
    assert speech["speaker"] == "player4"
    assert speech["sp_actions"] == [["player4", "point_as_werewolf", "player2"]]

    rotated = TWDToMDataset([rotated_raw])[0]
    torch.testing.assert_close(
        rotated["belief_targets"],
        torch.roll(baseline["belief_targets"], shifts=(2, 2), dims=(0, 1)),
    )
    assert torch.equal(
        rotated["observer_alive_mask"],
        torch.roll(baseline["observer_alive_mask"], shifts=2, dims=0),
    )
    assert torch.equal(rotated["diagonal_target_mask"], baseline["diagonal_target_mask"])


def test_dataset_rotation_is_deterministic_and_epoch_conditioned(
    suspicion_sample_factory,
):
    sample = suspicion_sample_factory(observers=(1, 2, 3, 5))
    dataset = TWDToMDataset(
        [sample],
        enable_cyclic_rotation=True,
        augmentation_seed=2,
    )
    assert CYCLIC_ROTATION_VERSION == "cyclic_rotation_v1"
    first = dataset[0]
    dataset.set_epoch(1)
    second = dataset[0]
    dataset.set_epoch(1)
    repeated = dataset[0]
    assert deterministic_cyclic_shift(seed=2, epoch=1, sample_index=0) == 3
    assert not torch.equal(first["observer_alive_mask"], second["observer_alive_mask"])
    assert torch.equal(second["belief_targets"], repeated["belief_targets"])


def test_rotation_preserves_targetless_public_action_object(
    suspicion_sample_factory,
):
    sample = suspicion_sample_factory()
    speech = next(
        event for event in sample["public_events"]
        if event["event_type"] == "public_speech"
    )
    speech["sp_actions"] = [
        ["player2", "oppose", "player5"],
        ["player2", "abstain_intent", None],
        ["player2", "no_commitment", None],
    ]
    sample["public_action_count"] = 3
    sample["public_event_digest"] = public_event_digest(sample["public_events"])
    sample["structured_input_digest"] = structured_input_digest(
        sample["public_events"]
    )
    rotated = cyclically_rotate_belief_sample(sample, shift=3)
    rotated_speech = next(
        event for event in rotated["public_events"]
        if event["event_type"] == "public_speech"
    )
    assert rotated_speech["sp_actions"] == [
        ["player5", "oppose", "player1"],
        ["player5", "abstain_intent", None],
        ["player5", "no_commitment", None],
    ]
    item = TWDToMDataset([rotated])[0]
    assert item["object_ids"][item["action_ids"] != 0][-2:].tolist() == [8, 8]


def test_old_materialized_lineage_is_rejected(suspicion_sample_factory):
    sample = suspicion_sample_factory()
    sample["tom_order"] = 2
    with pytest.raises(ValueError, match="field set mismatch"):
        TWDToMDataset([sample])


def test_non_ok_alive_self_report_is_rejected(suspicion_sample_factory):
    sample = suspicion_sample_factory(failed_observer=3)
    with pytest.raises(ValueError, match="status=ok for every alive observer"):
        TWDToMDataset([sample])


@pytest.mark.parametrize(
    "invalid_suspicion",
    ["player7", ["player8"], ["player7", "player7"]],
)
def test_suspicion_set_remains_strict(
    suspicion_sample_factory,
    invalid_suspicion,
):
    sample = suspicion_sample_factory()
    sample["suspected_werewolves"]["player1"] = invalid_suspicion
    with pytest.raises((TypeError, ValueError)):
        TWDToMDataset([sample])


def test_hard_knowledge_is_audit_provenance_not_target_content_constraint(
    suspicion_sample_factory,
):
    sample = suspicion_sample_factory(
        suspicions_by_observer={1: ["player7"]}
    )
    sample["known_werewolves"]["player1"] = ["player3"]
    sample["known_non_werewolves"]["player1"] = ["player7"]

    item = TWDToMDataset([sample])[0]

    assert item["belief_targets"][0].tolist() == pytest.approx(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    )


def test_raw_sample_is_not_mutated(suspicion_sample_factory):
    sample = suspicion_sample_factory()
    original = deepcopy(sample)
    TWDToMDataset([sample])
    assert sample == original


def test_public_cutoff_and_digest_validation_remain_strict(
    suspicion_sample_factory,
):
    sample = suspicion_sample_factory()
    broken = deepcopy(sample)
    broken["label_cutoff_step_idx"] += 1
    with pytest.raises(ValueError, match="label_cutoff_step_idx"):
        TWDToMDataset([broken])

    broken = deepcopy(sample)
    broken["public_events"][-1]["speaker"] = "player3"
    broken["public_event_digest"] = public_event_digest(broken["public_events"])
    broken["structured_input_digest"] = structured_input_digest(
        broken["public_events"]
    )
    with pytest.raises(ValueError, match="matching turn_start"):
        TWDToMDataset([broken])


def test_jsonl_loading_and_dataset_materialization_are_deterministic(
    tmp_path,
    suspicion_sample_factory,
):
    sample = suspicion_sample_factory()
    path = tmp_path / "belief.jsonl"
    path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
    assert load_twd_tom_jsonl(path) == [sample]
    first = TWDToMDataset.from_jsonl(path)[0]
    second = TWDToMDataset.from_jsonl(path)[0]
    torch.testing.assert_close(first["belief_targets"], second["belief_targets"])
    assert first["metadata"] == second["metadata"]


def test_collate_stacks_fixed_contract_and_pads_public_history(
    suspicion_sample_factory,
):
    first_sample = suspicion_sample_factory(game_id="game_a")
    second_sample = suspicion_sample_factory(game_id="game_b")
    first = TWDToMDataset([first_sample])[0]
    second = TWDToMDataset([second_sample])[0]
    batch = collate_twd_tom_samples([first, second])

    assert batch["belief_targets"].shape == (2, 7, 7)
    assert batch["observer_alive_mask"].shape == (2, 7)
    assert batch["diagonal_target_mask"].shape == (2, 7, 7)
    assert batch["metadata"]["game_id"] == ["game_a", "game_b"]
    assert "pair_targets" not in batch
    assert "subject_mask" not in batch


def test_game_identity_is_preserved_for_game_level_splits(
    suspicion_sample_factory,
):
    samples = [
        suspicion_sample_factory(game_id="train_game"),
        suspicion_sample_factory(game_id="validation_game"),
    ]
    dataset = TWDToMDataset(samples)
    assert [sample["game_id"] for sample in dataset.samples] == [
        "train_game",
        "validation_game",
    ]
    assert [dataset[index]["metadata"]["game_id"] for index in range(2)] == [
        "train_game",
        "validation_game",
    ]


def test_model_features_do_not_depend_on_private_knowledge(
    suspicion_sample_factory,
):
    first = suspicion_sample_factory()
    second = deepcopy(first)
    second["known_non_werewolves"]["player1"] = ["player1", "player6"]
    first_item = TWDToMDataset([first])[0]
    second_item = TWDToMDataset([second])[0]
    for field_name in (
        "subject_ids",
        "action_ids",
        "object_ids",
        "event_type_ids",
        "phase_ids",
        "day_values",
        "attention_mask",
    ):
        assert torch.equal(first_item[field_name], second_item[field_name])
    assert "known_werewolves" not in second_item
    assert "known_non_werewolves" not in second_item


def test_target_rows_follow_canonical_player_indices(suspicion_sample_factory):
    sample = suspicion_sample_factory(
        observers=(2, 5),
        suspicions_by_observer={2: ["player6"], 5: ["player1"]},
    )
    item = TWDToMDataset([sample])[0]
    assert item["belief_targets"][PLAYER_TO_ID["player2"] - 1, 5] > 0
    assert item["belief_targets"][PLAYER_TO_ID["player5"] - 1, 0] > 0
