from copy import deepcopy

import pytest
import torch

from werewolf.models.twd_tom.dataset import TWDToMDataset, collate_twd_tom_samples
from werewolf.models.twd_tom.public_events import public_event_digest
from tests.twd_tom.public_event_fixtures import public_history_fields


def test_dataset_pair_targets_masks_and_dtypes(projected_sample_factory):
    item = TWDToMDataset([projected_sample_factory()])[0]
    assert item["pair_targets"].shape == (7, 21)
    assert item["subject_mask"].tolist() == [False, False, True, False, True, False, False]
    assert torch.count_nonzero(item["pair_targets"][0]).item() == 0
    assert item["pair_targets"][2].sum().item() == pytest.approx(1.0)
    assert item["pair_targets"][4].sum().item() == pytest.approx(1.0)
    assert item["subject_ids"].dtype == torch.long
    assert item["action_ids"].dtype == torch.long
    assert item["object_ids"].dtype == torch.long
    assert item["attention_mask"].dtype == torch.long
    assert item["pair_targets"].dtype == torch.float32
    assert item["subject_mask"].dtype == torch.bool


def test_collate_shapes_and_attention_padding(projected_sample_factory):
    first = projected_sample_factory(step_idx=1)
    second = projected_sample_factory(step_idx=2)
    for key, value in public_history_fields(
        [], speaker_id=first["speaker_id"]
    ).items():
        first[key] = value
    actions = [
        ["player3", "oppose", "player2"],
        ["player2", "point_as_werewolf", "player7"],
    ]
    for key, value in public_history_fields(
        actions, speaker_id=second["speaker_id"]
    ).items():
        second[key] = value
    dataset = TWDToMDataset([first, second])
    batch = collate_twd_tom_samples([dataset[0], dataset[1]])
    assert batch["pair_targets"].shape == (2, 7, 21)
    assert batch["subject_mask"].shape == (2, 7)
    assert batch["attention_mask"].dtype == torch.long
    assert batch["attention_mask"].tolist() == [
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1],
    ]


@pytest.mark.parametrize(
    "field",
    [
        "suspected_werewolves",
        "known_werewolves",
        "known_non_werewolves",
        "belief_mode",
        "agent_backend_ids",
        "public_event_digest",
        "structured_input_digest",
    ],
)
def test_supervision_metadata_never_enters_model_tensor_fields(projected_sample_factory, field):
    item = TWDToMDataset([projected_sample_factory()])[0]
    assert field not in {key for key in item if key != "metadata"}


def test_dataset_rejects_legacy_plausible_wolf_pairs(projected_sample_factory):
    sample = projected_sample_factory()
    sample["plausible_wolf_pairs"] = {}
    with pytest.raises(ValueError, match="legacy"):
        TWDToMDataset([sample])


def test_dataset_rejects_old_schema_identifier(projected_sample_factory):
    sample = projected_sample_factory()
    sample["schema_version"] = "onuw7_playing_agent_belief_set_v1"
    with pytest.raises(ValueError, match="schema_version"):
        TWDToMDataset([sample])


def test_dataset_rejects_projected_v1(projected_sample_factory):
    sample = projected_sample_factory()
    sample["schema_version"] = (
        "classic7_pre_speech_suspicion_pair_distribution_v1"
    )
    with pytest.raises(ValueError, match="schema_version"):
        TWDToMDataset([sample])


def test_dataset_rejects_historical_pair_support_schema(
    projected_sample_factory,
):
    sample = projected_sample_factory()
    sample["schema_version"] = "classic7_pre_speech_pair_belief_v1"
    with pytest.raises(ValueError, match="schema_version"):
        TWDToMDataset([sample])


def test_pair_dataset_requires_explicit_projection_for_raw_suspicion(
    suspicion_sample_factory,
):
    with pytest.raises(
        ValueError,
        match="requires an explicit offline pair projection",
    ):
        TWDToMDataset([suspicion_sample_factory()])


def test_dataset_rejects_contradictory_hard_knowledge(
    projected_sample_factory,
):
    sample = projected_sample_factory()
    sample["known_werewolves"]["player3"] = ["player3"]
    with pytest.raises(ValueError, match="disjoint"):
        TWDToMDataset([sample])


@pytest.mark.parametrize(
    ("status", "error"),
    [
        ("parse_error", "invalid JSON"),
        (
            "semantic_error",
            "non-canonical belief report does not strictly narrow hard support",
        ),
        ("reporter_error", "backend failed"),
    ],
)
def test_invalid_status_row_is_zero_and_unmasked(
    projected_sample_factory,
    status,
    error,
):
    sample = projected_sample_factory()
    sample["belief_status"]["player3"] = status
    sample["belief_errors"]["player3"] = error
    sample["suspected_werewolves"]["player3"] = None
    sample["pair_targets"]["player3"] = None
    item = TWDToMDataset([sample])[0]
    assert not item["subject_mask"][2]
    assert torch.count_nonzero(item["pair_targets"][2]).item() == 0


def test_dataset_rejects_digest_or_cutoff_mismatch(projected_sample_factory):
    sample = projected_sample_factory()
    broken = deepcopy(sample)
    broken["label_cutoff_step_idx"] += 1
    with pytest.raises(ValueError, match="label_cutoff"):
        TWDToMDataset([broken])
    sample["public_event_digest"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        TWDToMDataset([sample])


def test_raw_text_does_not_change_dataset_model_features(
    projected_sample_factory,
):
    first = projected_sample_factory()
    second = deepcopy(first)
    speech = next(
        event
        for event in second["public_events"]
        if event["event_type"] == "public_speech"
    )
    speech["raw_text"] = "different public wording"
    second["public_event_digest"] = public_event_digest(
        second["public_events"]
    )
    first_item = TWDToMDataset([first])[0]
    second_item = TWDToMDataset([second])[0]
    for field in (
        "subject_ids",
        "action_ids",
        "object_ids",
        "event_type_ids",
        "phase_ids",
        "day_values",
        "attention_mask",
    ):
        assert torch.equal(first_item[field], second_item[field])


def test_dataset_rejects_nonmonotonic_game_prefix(projected_sample_factory):
    first = projected_sample_factory(step_idx=1)
    second = projected_sample_factory(step_idx=2)
    second["public_events"][0]["phase"] = "2_day_speech"
    second["phase"] = "2_day_speech"
    from werewolf.models.twd_tom.public_events import (
        public_event_digest,
        structured_input_digest,
    )

    second["public_event_digest"] = public_event_digest(
        second["public_events"]
    )
    second["structured_input_digest"] = structured_input_digest(
        second["public_events"]
    )
    with pytest.raises(ValueError, match="monotonic prefixes"):
        TWDToMDataset([first, second])


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("projection_version", "wrong", "projection_version"),
        ("pair_ordering", "wrong", "pair_ordering"),
        ("source_schema_version", "wrong", "source_schema_version"),
    ],
)
def test_dataset_rejects_wrong_projection_metadata(
    projected_sample_factory,
    field,
    value,
    match,
):
    sample = projected_sample_factory()
    sample[field] = value
    with pytest.raises(ValueError, match=match):
        TWDToMDataset([sample])


def test_dataset_requires_projected_prompt_v2(projected_sample_factory):
    sample = projected_sample_factory()
    assert sample["label_prompt_version"] == (
        "classic7_pre_speech_player_suspicion_prompt_v2"
    )
    sample["label_prompt_version"] = (
        "classic7_pre_speech_player_suspicion_prompt_v1"
    )
    with pytest.raises(ValueError, match="label_prompt_version"):
        TWDToMDataset([sample])


def test_dataset_rejects_tampered_or_malformed_stored_targets(
    projected_sample_factory,
):
    sample = projected_sample_factory()
    target = sample["pair_targets"]["player3"]
    target[0], target[1] = target[1], target[0]
    with pytest.raises(
        ValueError,
        match="hard-legal|hard-illegal|projection_version",
    ):
        TWDToMDataset([sample])

    sample = projected_sample_factory()
    sample["pair_targets"]["player3"][0] = True
    with pytest.raises(TypeError, match="not bool"):
        TWDToMDataset([sample])

    sample = projected_sample_factory()
    sample["pair_targets"]["player3"] = [0.0] * 21
    with pytest.raises(ValueError, match="sum to one"):
        TWDToMDataset([sample])


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda sample: sample["pair_targets"].pop("player3"),
            "subject set mismatch",
        ),
        (
            lambda sample: sample["pair_targets"].update(player3=[1.0] * 20),
            "contain 21",
        ),
        (
            lambda sample: sample["pair_targets"]["player3"].__setitem__(
                0, float("inf")
            ),
            "finite",
        ),
        (
            lambda sample: sample["pair_targets"]["player3"].__setitem__(
                0, -0.1
            ),
            "non-negative",
        ),
    ],
)
def test_dataset_rejects_invalid_pair_target_structure(
    projected_sample_factory,
    mutation,
    match,
):
    sample = projected_sample_factory()
    mutation(sample)
    with pytest.raises((TypeError, ValueError), match=match):
        TWDToMDataset([sample])


def test_dataset_uses_stored_pair_targets_without_online_replacement(
    projected_sample_factory,
):
    sample = projected_sample_factory()
    stored = deepcopy(sample["pair_targets"]["player3"])
    dataset = TWDToMDataset([sample])
    assert dataset[0]["pair_targets"][2].tolist() == pytest.approx(stored)
