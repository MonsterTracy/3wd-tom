"""Tests for the sole tom-v2 observer-conditioned belief Dataset."""

import inspect
import json
from copy import deepcopy

import pytest
import torch

from werewolf.models.twd_tom.action_features import PublicEventFeatureBuilder
from werewolf.models.twd_tom.annotation_v2 import LEGACY_V1_BELIEF_SOURCE
from werewolf.models.twd_tom.belief_labels import (
    legacy_v1_suspicion_set_to_belief_vector,
    suspicion_set_to_belief_vector,
)
from werewolf.models.twd_tom.dataset import (
    CYCLIC_ROTATION_VERSION,
    MODEL_INPUT_SCOPE,
    PRIVATE_MODEL_INPUT_SCOPE,
    TARGET_CONVERSION,
    TARGET_SEMANTICS,
    TWDToMDataset,
    collate_twd_tom_samples,
    cyclically_rotate_belief_sample,
    deterministic_cyclic_shift,
    load_twd_tom_jsonl,
)
from werewolf.models.twd_tom.dense_dataset import (
    DENSE_SUPERVISION_VERSION,
    DenseTWDToMDataset,
    collate_dense_twd_tom_games,
)
from werewolf.models.twd_tom.public_events import (
    completed_pre_speech_public_events,
    public_event_digest,
    structured_input_digest,
)
from werewolf.models.twd_tom.schema import PLAYER_TO_ID
from werewolf.models.twd_tom.speech_annotations import (
    make_speech_annotation,
    speech_annotation_digest,
)


def _later_dense_snapshot(sample):
    later = deepcopy(sample)
    later["step_idx"] += 1
    later["label_cutoff_step_idx"] = later["step_idx"]
    previous_speaker = f"player{later['speaker_id']}"
    speech_index = len(later["public_events"])
    later["public_events"].extend(
        [
            {
                "event_idx": speech_index,
                "event_type": "public_speech",
                "speaker": previous_speaker,
                "raw_text": "later synthetic speech",
            },
            {
                "event_idx": speech_index + 1,
                "event_type": "turn_start",
                "speaker": "player3",
            },
        ]
    )
    later["speech_annotations"].append(
        make_speech_annotation(
            event_idx=speech_index,
            speaker=previous_speaker,
            raw_text="later synthetic speech",
            parser_model_id="synthetic_parser",
            parser_call_id=f"synthetic_{speech_index:06d}",
            annotation_source="llm_parser",
            status="ok",
            actions=[[previous_speaker, "support", "player4"]],
            raw_response=None,
            error_type=None,
            error_message=None,
        )
    )
    later["speaker_id"] = 3
    later["public_action_count"] += 1
    later["public_event_digest"] = public_event_digest(later["public_events"])
    later["speech_annotation_digest"] = speech_annotation_digest(
        later["speech_annotations"]
    )
    later["structured_input_digest"] = structured_input_digest(
        later["public_events"], later["speech_annotations"]
    )
    return later


def test_dataset_consumes_raw_self_report_and_emits_fixed_belief_contract(
    suspicion_sample_factory,
):
    sample = suspicion_sample_factory()
    item = TWDToMDataset([sample])[0]

    assert item["belief_targets"].shape == (7, 7)
    assert item["observer_alive_mask"].shape == (7,)
    assert item["observer_scope_mask"].shape == (7,)
    assert item["label_observed_mask"].shape == (7,)
    assert item["diagonal_target_mask"].shape == (7, 7)
    assert item["belief_targets"].dtype == torch.float32
    assert item["observer_alive_mask"].dtype == torch.bool
    assert item["diagonal_target_mask"].dtype == torch.bool
    assert item["metadata"]["target_conversion"] == TARGET_CONVERSION
    assert item["metadata"]["target_semantics"] == TARGET_SEMANTICS
    assert "pair_targets" not in item
    assert "known_werewolves" not in item
    assert "known_non_werewolves" not in item
    assert "reasoning_player_id" not in item


def test_each_observed_alive_row_matches_the_deterministic_conversion(
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
            known_werewolves=sample["known_werewolves"][subject],
            known_non_werewolves=sample["known_non_werewolves"][subject],
            dtype=torch.float64,
        )
        assert item["label_observed_mask"][observer_id - 1]
        torch.testing.assert_close(item["belief_targets"][observer_id - 1], expected)


def test_legacy_v1_is_distinct_from_empty_uniform_nonself_v1(
    suspicion_sample_factory,
):
    sample = suspicion_sample_factory(
        suspicions_by_observer={1: ["player3"], 2: []}
    )
    sample["known_non_werewolves"]["player2"] = ["player2", "player4"]
    current = TWDToMDataset([sample], target_dtype=torch.float64)[0]
    legacy = TWDToMDataset(
        [sample],
        target_dtype=torch.float64,
        belief_annotation_source=LEGACY_V1_BELIEF_SOURCE,
    )[0]
    expected = legacy_v1_suspicion_set_to_belief_vector(
        [],
        observer_id="player2",
        known_werewolves=sample["known_werewolves"]["player2"],
        known_non_werewolves=sample["known_non_werewolves"]["player2"],
        dtype=torch.float64,
    )

    assert current["label_observed_mask"][1]
    assert current["belief_targets"][1].tolist() == pytest.approx(
        [1.0 / 6.0, 0.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]
    )
    assert legacy["label_observed_mask"][1]
    torch.testing.assert_close(legacy["belief_targets"][1], expected)
    assert current["metadata"]["belief_annotation_source"] != (
        legacy["metadata"]["belief_annotation_source"]
    )
    assert current["metadata"]["target_conversion"] != (
        legacy["metadata"]["target_conversion"]
    )
    assert current["metadata"]["target_semantics"] != (
        legacy["metadata"]["target_semantics"]
    )


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


def test_all_observed_rows_are_normalized_and_have_zero_diagonal(
    suspicion_sample_factory,
):
    item = TWDToMDataset([suspicion_sample_factory()])[0]
    alive_targets = item["belief_targets"][item["label_observed_mask"]]
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
    assert rotated_raw["speech_annotations"][0]["actions"] == [
        ["player4", "point_as_werewolf", "player2"]
    ]

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


def test_rotation_keeps_supervision_roles_aligned_with_rotated_targets(
    suspicion_sample_factory,
):
    sample = suspicion_sample_factory(observers=(1, 2, 3, 5))
    roles = {
        sample["game_id"]: {
            "player1": "Werewolf",
            "player2": "Werewolf",
            "player3": "Villager",
            "player4": "Villager",
            "player5": "Villager",
            "player6": "Seer",
            "player7": "Witch",
        }
    }
    dataset = TWDToMDataset(
        [sample],
        enable_cyclic_rotation=True,
        augmentation_seed=2,
        observer_roles_by_game=roles,
        supervision_scope="villager_alive",
    )
    dataset.set_epoch(1)

    item = dataset[0]

    assert item["observer_scope_mask"].tolist() == [
        True,
        False,
        False,
        False,
        False,
        True,
        False,
    ]
    assert torch.equal(
        item["observer_supervision_mask"],
        item["observer_scope_mask"] & item["label_observed_mask"],
    )
    assert item["metadata"]["observer_roles"] == [
        "Villager",
        "Seer",
        "Witch",
        "Werewolf",
        "Werewolf",
        "Villager",
        "Villager",
    ]


def test_rotation_preserves_targetless_public_action_object(
    suspicion_sample_factory,
):
    sample = suspicion_sample_factory()
    annotation = sample["speech_annotations"][0]
    annotation["actions"] = [
        ["player2", "oppose", "player5"],
        ["player2", "abstain_intent", None],
        ["player2", "no_commitment", None],
    ]
    annotation["status"] = "ok"
    sample["public_action_count"] = 3
    sample["public_event_digest"] = public_event_digest(sample["public_events"])
    sample["speech_annotation_digest"] = speech_annotation_digest(
        sample["speech_annotations"]
    )
    sample["structured_input_digest"] = structured_input_digest(
        sample["public_events"], sample["speech_annotations"]
    )
    rotated = cyclically_rotate_belief_sample(sample, shift=3)
    assert rotated["speech_annotations"][0]["actions"] == [
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


def test_non_ok_alive_self_report_is_unobserved_without_imputation(
    suspicion_sample_factory,
):
    sample = suspicion_sample_factory(failed_observer=3)
    item = TWDToMDataset([sample])[0]

    assert item["observer_alive_mask"][2]
    assert not item["label_observed_mask"][2]
    assert not item["observer_supervision_mask"][2]
    assert item["belief_targets"][2].tolist() == pytest.approx([0.0] * 7)


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        ("suspected_werewolves", [], "requires null suspicion"),
        ("belief_errors", None, "requires an error"),
    ],
)
def test_failed_report_cannot_fall_back_to_empty_uniform(
    suspicion_sample_factory,
    field_name,
    invalid_value,
    message,
):
    sample = suspicion_sample_factory(failed_observer=3)
    sample[field_name]["player3"] = invalid_value

    with pytest.raises(ValueError, match=message):
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


def test_dataset_rejects_suspicion_that_contradicts_hard_knowledge(
    suspicion_sample_factory,
):
    sample = suspicion_sample_factory(
        suspicions_by_observer={1: ["player7"]}
    )
    sample["known_werewolves"]["player1"] = ["player3"]
    sample["known_non_werewolves"]["player1"] = ["player7"]

    with pytest.raises(ValueError, match="known Werewolves"):
        TWDToMDataset([sample])


def test_dataset_empty_successful_report_is_observed_uniform_nonself(
    suspicion_sample_factory,
):
    sample = suspicion_sample_factory(
        suspicions_by_observer={1: []}
    )
    sample["known_non_werewolves"]["player1"] = ["player1", "player4"]

    item = TWDToMDataset([sample])[0]

    assert item["label_observed_mask"][0]
    assert item["observer_supervision_mask"][0]
    assert item["belief_targets"][0].tolist() == pytest.approx(
        [0.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]
    )
    assert item["metadata"]["raw_support_size"][0] == 0
    assert item["metadata"]["raw_empty"][0]


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
        broken["public_events"], broken["speech_annotations"]
    )
    with pytest.raises(ValueError, match="matching turn_start"):
        TWDToMDataset([broken])


def test_model_features_exclude_only_the_terminal_pre_speech_turn_start(
    suspicion_sample_factory,
):
    sample = suspicion_sample_factory()
    item = TWDToMDataset([sample])[0]
    expected = PublicEventFeatureBuilder().encode_events(
        completed_pre_speech_public_events(
            sample["public_events"],
            speaker_id=sample["speaker_id"],
        ),
        sample["speech_annotations"],
    )
    complete = PublicEventFeatureBuilder().encode_events(
        sample["public_events"],
        sample["speech_annotations"],
    )
    for field_name in PublicEventFeatureBuilder.FEATURE_FIELDS:
        assert torch.equal(item[field_name], expected[field_name])
    assert item["subject_ids"].shape[0] < complete["subject_ids"].shape[0]


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


def test_private_conditioned_dataset_emits_only_observer_knowledge_masks(
    suspicion_sample_factory,
):
    sample = suspicion_sample_factory(
        suspicions_by_observer={1: ["player2"]}
    )
    sample["known_werewolves"]["player1"] = ["player2"]
    item = TWDToMDataset([sample], include_private_features=True)[0]

    assert item["known_werewolf_mask"].shape == (7, 7)
    assert item["known_non_werewolf_mask"].shape == (7, 7)
    assert item["known_werewolf_mask"].dtype == torch.bool
    assert item["known_non_werewolf_mask"].dtype == torch.bool
    assert item["known_werewolf_mask"][0, 1]
    assert item["known_non_werewolf_mask"][0, 0]
    assert not torch.any(
        item["known_werewolf_mask"] & item["known_non_werewolf_mask"]
    )
    assert item["metadata"]["private_feature_fields"] == [
        "known_werewolves",
        "known_non_werewolves",
    ]
    assert TWDToMDataset(
        [sample], include_private_features=True
    ).model_input_scope == PRIVATE_MODEL_INPUT_SCOPE


def test_private_knowledge_masks_follow_cyclic_seat_rotation(
    suspicion_sample_factory,
):
    sample = suspicion_sample_factory(
        observers=(1, 2, 5),
        suspicions_by_observer={1: ["player4"], 2: ["player7"], 5: ["player3"]},
    )
    sample["known_non_werewolves"]["player1"] = ["player1", "player6"]
    baseline = TWDToMDataset([sample], include_private_features=True)[0]
    rotated = TWDToMDataset(
        [cyclically_rotate_belief_sample(sample, shift=2)],
        include_private_features=True,
    )[0]

    for field_name in ("known_werewolf_mask", "known_non_werewolf_mask"):
        assert torch.equal(
            rotated[field_name],
            torch.roll(baseline[field_name], shifts=(2, 2), dims=(0, 1)),
        )


def test_target_rows_follow_canonical_player_indices(suspicion_sample_factory):
    sample = suspicion_sample_factory(
        observers=(2, 5),
        suspicions_by_observer={2: ["player6"], 5: ["player1"]},
    )
    item = TWDToMDataset([sample])[0]
    assert item["belief_targets"][PLAYER_TO_ID["player2"] - 1, 5] > 0
    assert item["belief_targets"][PLAYER_TO_ID["player5"] - 1, 0] > 0


def test_dense_dataset_groups_every_game_and_supervises_all_pre_boundaries(
    suspicion_sample_factory,
):
    first = suspicion_sample_factory(game_id="dense_game", step_idx=1)
    second = _later_dense_snapshot(first)
    dataset = DenseTWDToMDataset([second, first])
    item = dataset[0]

    assert len(dataset) == 1
    assert dataset.boundary_count == 2
    assert item["metadata"]["supervision_version"] == DENSE_SUPERVISION_VERSION
    assert item["metadata"]["step_idx"] == [1, 2]
    assert item["boundary_indices"].shape == (2,)
    assert item["boundary_indices"][0] < item["boundary_indices"][1]
    assert item["boundary_indices"][-1] == item["subject_ids"].shape[0] - 1
    assert item["belief_targets"].shape == (2, 7, 7)
    assert item["observer_alive_mask"].shape == (2, 7)
    assert item["diagonal_target_mask"].shape == (2, 7, 7)
    assert item["boundary_valid_mask"].tolist() == [True, True]


def test_dense_collate_right_pads_timelines_and_boundaries(
    suspicion_sample_factory,
):
    first = suspicion_sample_factory(game_id="dense_a", step_idx=1)
    second = _later_dense_snapshot(first)
    other = suspicion_sample_factory(game_id="dense_b", step_idx=1)
    dataset = DenseTWDToMDataset([first, second, other])
    batch = collate_dense_twd_tom_games([dataset[0], dataset[1]])

    assert batch["belief_targets"].shape == (2, 2, 7, 7)
    assert batch["observer_alive_mask"].shape == (2, 2, 7)
    assert batch["boundary_indices"].shape == (2, 2)
    assert batch["boundary_valid_mask"].tolist() == [
        [True, True],
        [True, False],
    ]
    assert not batch["observer_alive_mask"][1, 1].any()
    assert not batch["belief_targets"][1, 1].any()


def test_dense_private_masks_are_stacked_and_padded_per_boundary(
    suspicion_sample_factory,
):
    first = suspicion_sample_factory(game_id="dense_a", step_idx=1)
    second = _later_dense_snapshot(first)
    other = suspicion_sample_factory(game_id="dense_b", step_idx=1)
    dataset = DenseTWDToMDataset(
        [first, second, other],
        include_private_features=True,
    )
    batch = collate_dense_twd_tom_games([dataset[0], dataset[1]])

    assert batch["known_werewolf_mask"].shape == (2, 2, 7, 7)
    assert batch["known_non_werewolf_mask"].shape == (2, 2, 7, 7)
    assert not batch["known_werewolf_mask"][1, 1].any()
    assert not batch["known_non_werewolf_mask"][1, 1].any()


def test_dense_dataset_rejects_annotation_history_that_is_not_an_exact_prefix(
    suspicion_sample_factory,
):
    first = suspicion_sample_factory(game_id="dense_game", step_idx=1)
    second = _later_dense_snapshot(first)
    second["speech_annotations"][0]["actions"] = [
        ["player2", "oppose", "player4"]
    ]
    second["speech_annotation_digest"] = speech_annotation_digest(
        second["speech_annotations"]
    )
    second["structured_input_digest"] = structured_input_digest(
        second["public_events"], second["speech_annotations"]
    )
    dataset = DenseTWDToMDataset([first, second])
    with pytest.raises(ValueError, match="exact prefixes"):
        dataset[0]
