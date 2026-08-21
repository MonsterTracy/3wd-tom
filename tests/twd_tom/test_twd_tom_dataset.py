"""Tests for strict current raw first-/second-order data adaptation."""

import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest
import torch

import werewolf.models.twd_tom.dataset as dataset_module
import werewolf.offline_materialization as offline_materialization_module
from werewolf.models.twd_tom.belief_labels import suspicion_set_to_pair_target
from werewolf.models.twd_tom.dataset import (
    D_PUBLIC_ONLY_TOM2_BELIEF_INFORMATION_SCOPE,
    SUBJECT_MAPPING_FIELDS,
    TWDToMDataset,
    collate_twd_tom_samples,
    cyclically_rotate_second_order_sample,
    deterministic_cyclic_shift,
    second_order_effective_subject_mask,
)
from werewolf.models.twd_tom.public_events import (
    is_post_completed_public_speech_pre_next_action,
    normalize_public_events,
    public_event_digest,
    structured_event_tokens,
    structured_input_digest,
)
from werewolf.models.twd_tom.schema import (
    ACTION_TO_ID,
    NONE_TOKEN,
    PLAYER_NAMES,
    PLAYER_TO_ID,
    canonical_wolf_pairs,
)
from werewolf.offline_annotation import (
    OFFLINE_ANNOTATION_SCHEMA_VERSION,
    PRIVATE_CONDITIONED_SUSPICION_TASK,
    PRIVATE_PROMPT_VERSION,
    PUBLIC_ONLY_SUSPICION_TASK,
    PUBLIC_PROMPT_VERSION,
)
from werewolf.offline_materialization import (
    D_MATERIALIZATION_POLICY_VERSION,
    D_SCHEMA_VERSION,
    OFFLINE_PRIVATE_CONDITIONED_TOM1_TASK,
    OFFLINE_PUBLIC_ONLY_TOM2_TASK,
    TOM1_MODEL_INPUT_SCOPE,
    TOM1_OBSERVER_PROVENANCE,
    TOM1_PRIVATE_FIELDS_USAGE,
    TOM2_MODEL_INPUT_SCOPE,
    TOM2_OBSERVER_PROVENANCE,
    TOM2_PRIVATE_FIELDS_USAGE,
)
from werewolf.trajectory import canonical_digest
from tests.twd_tom.public_event_fixtures import (
    make_full_history_training_sample,
    make_public_only_training_sample,
    make_training_sample,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TOM2_SPLIT_SHA256 = {
    "train": "31b99dc26ba9724c6ee8390a184b01faca50ec830313fcac3e5c873e5fa1e1a8",
    "val": "5b7f9206e4a8a59938a5aba50ba68f0d7c6cd9079b377d1a25aedd349151f0c8",
    "test": "91622732b35dc0a6ca17dcb4b74292514306db128d9a2a9ee1f9538df2c03a6d",
}


def raw_sample(tom_order):
    return make_training_sample(tom_order)


def second_order_sample(*, with_latest_action):
    return make_training_sample(2, with_latest_action=with_latest_action)


def second_order_sample_with_sparse_latest_actor_target():
    return make_training_sample(2, observers=(1, 3, 5))


def d_sample(
    tom_order,
    *,
    with_latest_action=True,
    phase="1_day_speech",
):
    speaker_id = 2
    observers = (speaker_id,) if tom_order == 1 else (1, 3, 5)
    legacy = make_training_sample(
        tom_order,
        observers=observers,
        with_latest_action=with_latest_action,
        phase=phase,
    )
    subjects = [f"player{observer_id}" for observer_id in observers]
    is_private = tom_order == 1
    materialization_task = (
        OFFLINE_PRIVATE_CONDITIONED_TOM1_TASK
        if is_private
        else OFFLINE_PUBLIC_ONLY_TOM2_TASK
    )
    source_task = (
        PRIVATE_CONDITIONED_SUSPICION_TASK
        if is_private
        else PUBLIC_ONLY_SUSPICION_TASK
    )
    prompt_version = PRIVATE_PROMPT_VERSION if is_private else PUBLIC_PROMPT_VERSION
    provenance = (
        TOM1_OBSERVER_PROVENANCE if is_private else TOM2_OBSERVER_PROVENANCE
    )
    semantic_phase = "speech_pk" if phase.endswith("speech_pk") else "speech"
    record = {
        "schema_version": D_SCHEMA_VERSION,
        "materialization_task": materialization_task,
        "materialization_policy_version": D_MATERIALIZATION_POLICY_VERSION,
        "materializer_code_commit": "0123456789abcdef0123456789abcdef01234567",
        "game_id": legacy["game_id"],
        "source_trajectory_commit": "1234567890abcdef1234567890abcdef12345678",
        "trajectory_digest": "1" * 64,
        "observer_view_artifact_digest": "2" * 64,
        "boundary_id": (
            f"{legacy['game_id']}:step_{legacy['step_idx']:06d}:"
            "PRE_PUBLIC_SPEECH"
        ),
        "step_idx": legacy["step_idx"],
        "phase": semantic_phase,
        "speaker_id": speaker_id,
        "report_trigger": (
            "pre_public_speech_pk"
            if semantic_phase == "speech_pk"
            else "pre_public_speech"
        ),
        "public_event_schema_version": legacy["public_event_schema_version"],
        "public_events": deepcopy(legacy["public_events"]),
        "public_event_digest": legacy["public_event_digest"],
        "structured_input_digest": legacy["structured_input_digest"],
        "public_action_count": legacy["public_action_count"],
        "label_cutoff_step_idx": legacy["label_cutoff_step_idx"],
        "tom_order": tom_order,
        "model_input_scope": (
            TOM1_MODEL_INPUT_SCOPE if is_private else TOM2_MODEL_INPUT_SCOPE
        ),
        "private_fields_usage": (
            TOM1_PRIVATE_FIELDS_USAGE
            if is_private
            else TOM2_PRIVATE_FIELDS_USAGE
        ),
        "observer_ids": list(observers),
        "suspected_werewolves": deepcopy(legacy["suspected_werewolves"]),
        "known_werewolves": deepcopy(legacy["known_werewolves"]),
        "known_non_werewolves": deepcopy(legacy["known_non_werewolves"]),
        "belief_status": {subject: "ok" for subject in subjects},
        "belief_errors": {subject: None for subject in subjects},
        "source_annotation_schema_version": OFFLINE_ANNOTATION_SCHEMA_VERSION,
        "source_annotation_task": source_task,
        "source_prompt_version": prompt_version,
        "source_annotation_run_ids": {
            subject: "annotation_run" for subject in subjects
        },
        "source_annotation_code_commits": {
            subject: "234567890abcdef1234567890abcdef123456789"
            for subject in subjects
        },
        "source_annotation_record_digests": {
            subject: f"{observer_id}" * 64
            for subject, observer_id in zip(subjects, observers)
        },
        "reporter_backend_ids": {
            subject: "reporter_backend" for subject in subjects
        },
        "reporter_model_ids": {
            subject: "reporter_model" for subject in subjects
        },
        "observer_label_provenance": {
            subject: provenance for subject in subjects
        },
        "observer_annotation_confidence": {
            subject: "model_reported_source" for subject in subjects
        },
        "current_action_used": False,
        "expert_labels_used_as_later_evidence": False,
        "future_information_used": False,
    }
    if not is_private:
        record["known_werewolves"] = {subject: [] for subject in subjects}
        record["known_non_werewolves"] = {subject: [] for subject in subjects}
    record["record_digest"] = canonical_digest(record)
    return record


def test_second_order_formal_split_files_are_unchanged(
    require_real_twd_tom_data,
):
    paths = [
        REPO_ROOT / "data" / "qwen25" / "tom2" / f"{split}.jsonl"
        for split in TOM2_SPLIT_SHA256
    ]
    require_real_twd_tom_data(*paths)
    for split, expected in TOM2_SPLIT_SHA256.items():
        path = REPO_ROOT / "data" / "qwen25" / "tom2" / f"{split}.jsonl"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_real_data_smoke_gate_is_explicit(
    tmp_path,
    monkeypatch,
    require_real_twd_tom_data,
):
    path = tmp_path / "fixture.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    monkeypatch.delenv("RUN_TWD_TOM_REAL_DATA_TESTS", raising=False)
    with pytest.raises(pytest.skip.Exception):
        require_real_twd_tom_data(path)
    monkeypatch.setenv("RUN_TWD_TOM_REAL_DATA_TESTS", "1")
    require_real_twd_tom_data(path)
    with pytest.raises(pytest.skip.Exception):
        require_real_twd_tom_data(tmp_path / "missing.jsonl")


def full_history_second_order_sample():
    return make_full_history_training_sample()


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
    assert item["reasoning_player_id"].shape == ()
    assert isinstance(
        item["post_completed_public_speech_pre_next_action"], bool
    )
    assert "suspicion_targets" not in item


def test_public_only_first_order_omits_private_tensor_keys():
    sample = make_public_only_training_sample(1)
    dataset = TWDToMDataset([sample], tom_order=1)
    item = dataset[0]
    batch = collate_twd_tom_samples([item])
    assert "known_werewolves" not in item
    assert "known_non_werewolves" not in item
    assert "known_werewolves" not in batch
    assert "known_non_werewolves" not in batch
    assert "reasoning_player_id" not in batch
    assert item["pair_targets"].shape == (7, 21)
    assert item["pair_targets"][item["subject_mask"]].sum().item() == (
        pytest.approx(1.0)
    )


def test_public_only_second_order_reuses_existing_public_tensor_contract():
    private_item = TWDToMDataset([raw_sample(2)], tom_order=2)[0]
    public_item = TWDToMDataset(
        [make_public_only_training_sample(2)],
        tom_order=2,
    )[0]
    for item in (private_item, public_item):
        assert "known_werewolves" not in item
        assert "known_non_werewolves" not in item
        assert item["reasoning_player_id"].shape == ()
        assert item["pair_targets"].shape == (7, 21)
        assert torch.allclose(
            item["pair_targets"][item["subject_mask"]].sum(dim=-1),
            torch.ones(int(item["subject_mask"].sum().item())),
        )


def test_first_turn_is_not_a_completed_speech_boundary():
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
    assert not is_post_completed_public_speech_pre_next_action(
        events, reasoning_player_id=4
    )


def test_system_events_do_not_create_a_completed_speech_boundary():
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
    assert not is_post_completed_public_speech_pre_next_action(
        events, reasoning_player_id=4
    )


def test_completed_speech_immediately_before_current_turn_is_the_boundary():
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
    assert is_post_completed_public_speech_pre_next_action(
        events, reasoning_player_id=5
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
    assert is_post_completed_public_speech_pre_next_action(
        events, reasoning_player_id=6
    )


def test_nested_speech_actions_do_not_change_the_synchronized_boundary():
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
            "sp_actions": [["player4", "support", "player2"]],
        },
        {
            "event_idx": 2,
            "event_type": "turn_start",
            "speaker": "player5",
        },
    ]
    assert is_post_completed_public_speech_pre_next_action(
        events, reasoning_player_id=5
    )


def test_vote_and_later_system_events_are_not_speech_boundaries():
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
    assert not is_post_completed_public_speech_pre_next_action(
        events, reasoning_player_id=5
    )


def test_boundary_rejects_a_reasoning_player_other_than_current_turn():
    events = make_full_history_training_sample()["public_events"]
    with pytest.raises(ValueError, match="reasoning player"):
        is_post_completed_public_speech_pre_next_action(
            events,
            reasoning_player_id=7,
        )


def test_second_order_indices_filter_zero_effective_mask_deterministically():
    without_action = second_order_sample(with_latest_action=False)
    with_action = second_order_sample(with_latest_action=True)
    dataset = TWDToMDataset([without_action, with_action], tom_order=2)
    assert dataset.second_order_supervised_indices() == (1,)
    assert dataset.second_order_supervised_indices() == (1,)


def test_second_order_indices_accept_sparse_normalized_pair_targets():
    dataset = TWDToMDataset(
        [second_order_sample_with_sparse_latest_actor_target()],
        tom_order=2,
    )
    item = dataset[0]
    expected = second_order_effective_subject_mask(
        item["subject_mask"],
        item["reasoning_player_id"],
    ).any()
    assert dataset.second_order_supervised_indices() == ((0,) if expected else ())


def test_second_order_indices_filter_when_only_reasoning_player_has_target():
    dataset = TWDToMDataset(
        [second_order_sample(with_latest_action=True)],
        tom_order=2,
    )
    reasoning_player = PLAYER_NAMES[dataset.samples[0]["speaker_id"] - 1]
    for player in PLAYER_NAMES:
        if player != reasoning_player:
            dataset.samples[0]["_pair_targets"].pop(player, None)
    item = dataset[0]
    assert item["post_completed_public_speech_pre_next_action"]
    assert not second_order_effective_subject_mask(
        item["subject_mask"],
        item["reasoning_player_id"],
    ).any()
    assert dataset.second_order_supervised_indices() == ()


def test_second_order_indices_do_not_depend_on_target_probability_values():
    dataset = TWDToMDataset(
        [second_order_sample(with_latest_action=True)],
        tom_order=2,
    )
    for subject, target in dataset.samples[0]["_pair_targets"].items():
        if target is not None:
            dataset.samples[0]["_pair_targets"][subject] = torch.zeros_like(target)
    assert dataset.second_order_supervised_indices() == (0,)


@pytest.mark.parametrize(
    ("subject_mask", "reasoning_player_id", "error", "match"),
    [
        ([True] * 7, torch.tensor(1), TypeError, "torch.Tensor"),
        (torch.ones(7, dtype=torch.bool), 1, TypeError, "torch.Tensor"),
        (torch.ones(7), torch.tensor(1), TypeError, "dtype"),
        (torch.ones(7, dtype=torch.bool), torch.tensor(1.0), TypeError, "integral"),
        (
            torch.ones(1, 7, dtype=torch.bool),
            torch.tensor(1),
            ValueError,
            "batch dimensions",
        ),
        (
            torch.ones(6, dtype=torch.bool),
            torch.tensor(1),
            ValueError,
            "last dimension must be 7",
        ),
        (torch.ones(7, dtype=torch.bool), torch.tensor(0), ValueError, r"\[1, 7\]"),
    ],
)
def test_effective_subject_mask_rejects_non_boolean_or_wrong_shape(
    subject_mask,
    reasoning_player_id,
    error,
    match,
):
    with pytest.raises(error, match=match):
        second_order_effective_subject_mask(subject_mask, reasoning_player_id)


def test_effective_subject_mask_keeps_all_valid_other_players():
    subject_mask = torch.tensor(
        [[True, False, True, True, False, True, True]]
    )
    actual = second_order_effective_subject_mask(
        subject_mask,
        torch.tensor([4]),
    )
    assert torch.equal(
        actual,
        torch.tensor([[True, False, True, False, False, True, True]]),
    )


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
    assert second_batch["reasoning_player_id"].shape == (1,)
    assert second_batch[
        "post_completed_public_speech_pre_next_action"
    ].shape == (1,)
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
    first = make_training_sample(2, with_latest_action=True)
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


def test_cyclic_rotation_preserves_targetless_speech_action_objects():
    sample = make_training_sample(2)
    speech = next(
        event
        for event in sample["public_events"]
        if event["event_type"] == "public_speech"
    )
    speech["sp_actions"] = [
        ["player2", "oppose", "player5"],
        ["player2", "abstain_intent", None],
        ["player2", "no_commitment", None],
    ]
    sample["public_action_count"] = 3
    sample["public_event_digest"] = public_event_digest(
        sample["public_events"]
    )
    sample["structured_input_digest"] = structured_input_digest(
        sample["public_events"]
    )

    rotated = cyclically_rotate_second_order_sample(sample, shift=3)
    rotated_speech = next(
        event
        for event in rotated["public_events"]
        if event["event_type"] == "public_speech"
    )
    assert rotated_speech["sp_actions"] == [
        ["player5", "oppose", "player1"],
        ["player5", "abstain_intent", None],
        ["player5", "no_commitment", None],
    ]
    assert normalize_public_events(rotated["public_events"]) == (
        rotated["public_events"]
    )
    assert rotated["public_event_digest"] == public_event_digest(
        rotated["public_events"]
    )
    assert rotated["structured_input_digest"] == structured_input_digest(
        rotated["public_events"]
    )

    dataset = TWDToMDataset(
        [sample],
        tom_order=2,
        enable_cyclic_rotation=True,
        augmentation_seed=3,
    )
    item = dataset[0]
    abstain_index = item["action_ids"].tolist().index(
        ACTION_TO_ID["abstain_intent"]
    )
    assert PLAYER_TO_ID[NONE_TOKEN] == 8
    assert item["object_ids"][abstain_index].item() == 8


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


def test_legacy_private_and_public_only_inputs_keep_existing_identity_and_behavior():
    private = raw_sample(1)
    public = make_public_only_training_sample(2)
    private_original = deepcopy(private)
    public_original = deepcopy(public)

    private_item = TWDToMDataset([private], tom_order=1)[0]
    public_item = TWDToMDataset([public], tom_order=2)[0]

    assert private == private_original
    assert public == public_original
    assert private_item["metadata"]["schema_version"] == private["schema_version"]
    assert public_item["metadata"]["schema_version"] == public["schema_version"]
    assert "materialization_task" not in private_item["metadata"]
    assert "materialization_task" not in public_item["metadata"]
    assert "known_werewolves" in private_item
    assert "known_werewolves" not in public_item


def test_d_tasks_require_matching_requested_tom_order():
    first = d_sample(1)
    second = d_sample(2)
    assert len(TWDToMDataset([first], tom_order=1)) == 1
    assert len(TWDToMDataset([second], tom_order=2)) == 1
    with pytest.raises(ValueError, match="requires requested tom_order=1"):
        TWDToMDataset([first], tom_order=2)
    with pytest.raises(ValueError, match="requires requested tom_order=2"):
        TWDToMDataset([second], tom_order=1)


@pytest.mark.parametrize("d_first", [False, True])
def test_dataset_rejects_mixed_legacy_and_d_tom1_lineages(d_first):
    legacy = raw_sample(1)
    d_record = d_sample(1)
    rows = [d_record, legacy] if d_first else [legacy, d_record]

    with pytest.raises(
        ValueError,
        match="one Dataset cannot mix legacy and D V1 input lineages",
    ):
        TWDToMDataset(rows, tom_order=1)


def test_dataset_allows_multiple_legacy_private_rows():
    first = raw_sample(1)
    second = raw_sample(1)
    second["game_id"] = "synthetic_game_002"

    dataset = TWDToMDataset([first, second], tom_order=1)
    assert len(dataset) == 2


def test_dataset_allows_multiple_d_tom1_rows():
    first = d_sample(1)
    second = d_sample(1)
    second["game_id"] = "synthetic_game_002"
    second["boundary_id"] = (
        f"{second['game_id']}:step_{second['step_idx']:06d}:"
        "PRE_PUBLIC_SPEECH"
    )
    second.pop("record_digest")
    second["record_digest"] = canonical_digest(second)

    dataset = TWDToMDataset([first, second], tom_order=1)
    assert len(dataset) == 2


def test_d_strict_validator_is_called_and_rejects_redigested_malformed_row(
    monkeypatch,
):
    record = d_sample(1)
    calls = []
    strict_validator = (
        offline_materialization_module.validate_offline_tom_training_record
    )

    def recording_validator(value):
        calls.append(value)
        return strict_validator(value)

    monkeypatch.setattr(
        offline_materialization_module,
        "validate_offline_tom_training_record",
        recording_validator,
    )
    TWDToMDataset([record], tom_order=1)
    assert calls == [record]

    malformed = deepcopy(record)
    malformed["pair_targets"] = []
    malformed.pop("record_digest")
    malformed["record_digest"] = canonical_digest(malformed)
    with pytest.raises(ValueError, match="fields"):
        TWDToMDataset([malformed], tom_order=1)


@pytest.mark.parametrize(
    ("semantic_phase", "full_phase", "report_trigger"),
    [
        ("speech", "1_day_speech", "pre_public_speech"),
        ("speech_pk", "1_day_speech_pk", "pre_public_speech_pk"),
    ],
)
def test_d_phase_maps_to_full_public_phase_and_metadata_is_honest(
    semantic_phase,
    full_phase,
    report_trigger,
):
    record = d_sample(2, phase=full_phase)
    original = deepcopy(record)
    dataset = TWDToMDataset([record], tom_order=2)
    item = dataset[0]
    metadata = item["metadata"]

    assert record == original
    assert record["phase"] == semantic_phase
    assert dataset.samples[0]["phase"] == full_phase
    assert dataset.samples[0]["report_trigger"] == report_trigger
    assert metadata["phase"] == full_phase
    assert metadata["schema_version"] == D_SCHEMA_VERSION
    assert metadata["materialization_task"] == (
        OFFLINE_PUBLIC_ONLY_TOM2_TASK
    )
    assert metadata["materialization_policy_version"] == (
        D_MATERIALIZATION_POLICY_VERSION
    )
    assert metadata["materializer_code_commit"] == record[
        "materializer_code_commit"
    ]
    assert metadata["source_annotation_task"] == PUBLIC_ONLY_SUSPICION_TASK
    assert metadata["model_input_scope"] == TOM2_MODEL_INPUT_SCOPE
    assert metadata["speaker_id"] == record["speaker_id"]
    assert metadata["observer_ids"] == record["observer_ids"]
    assert "record_digest" not in metadata
    assert "record_digest" not in dataset.samples[0]
    assert dataset.source_schema_version == D_SCHEMA_VERSION
    assert dataset.belief_information_scope == (
        D_PUBLIC_ONLY_TOM2_BELIEF_INFORMATION_SCOPE
    )


def test_d_phase_mismatch_is_rejected():
    record = d_sample(2)
    record["phase"] = "speech_pk"
    record["report_trigger"] = "pre_public_speech_pk"
    record.pop("record_digest")
    record["record_digest"] = canonical_digest(record)
    with pytest.raises(ValueError, match="phase"):
        TWDToMDataset([record], tom_order=2)


def test_d_tom1_reuses_pair_projection_and_private_hard_knowledge_tensors():
    record = d_sample(1)
    subject = f"player{record['speaker_id']}"
    item = TWDToMDataset([record], tom_order=1)[0]
    expected = suspicion_set_to_pair_target(
        record["suspected_werewolves"][subject],
        record["known_werewolves"][subject],
        record["known_non_werewolves"][subject],
    )
    subject_index = record["speaker_id"] - 1

    torch.testing.assert_close(item["pair_targets"][subject_index], expected)
    for player in record["known_werewolves"][subject]:
        assert item["known_werewolves"][
            subject_index, PLAYER_TO_ID[player] - 1
        ] == 1
    for player in record["known_non_werewolves"][subject]:
        assert item["known_non_werewolves"][
            subject_index, PLAYER_TO_ID[player] - 1
        ] == 1
    assert item["known_werewolves"].sum().item() == len(
        record["known_werewolves"][subject]
    )
    assert item["known_non_werewolves"].sum().item() == len(
        record["known_non_werewolves"][subject]
    )


def test_d_tom2_keeps_public_only_model_input_reasoning_player_and_mask():
    record = d_sample(2)
    item = TWDToMDataset([record], tom_order=2)[0]

    assert "known_werewolves" not in item
    assert "known_non_werewolves" not in item
    assert item["reasoning_player_id"].item() == record["speaker_id"]
    effective = second_order_effective_subject_mask(
        item["subject_mask"],
        item["reasoning_player_id"],
    )
    assert not effective[record["speaker_id"] - 1]
    assert effective.sum().item() == len(record["observer_ids"])


def test_d_tom2_temporal_gate_stays_in_dataset_supervised_index_selection():
    first_turn = d_sample(2, with_latest_action=False)
    post_speech = d_sample(2, with_latest_action=True)
    first_dataset = TWDToMDataset([first_turn], tom_order=2)
    post_dataset = TWDToMDataset([post_speech], tom_order=2)

    assert not first_dataset[0]["post_completed_public_speech_pre_next_action"]
    assert first_dataset.second_order_supervised_indices() == ()
    assert post_dataset[0]["post_completed_public_speech_pre_next_action"]
    assert post_dataset.second_order_supervised_indices() == (0,)


def test_d_tom2_rotation_happens_after_validation_without_mutating_artifact():
    record = d_sample(2)
    original = deepcopy(record)
    dataset = TWDToMDataset(
        [record],
        tom_order=2,
        enable_cyclic_rotation=True,
        augmentation_seed=3,
    )
    canonical_internal = deepcopy(dataset.samples[0])
    item = dataset[0]
    rotated = cyclically_rotate_second_order_sample(
        canonical_internal,
        shift=3,
    )

    assert record == original
    assert "pair_targets" not in record
    assert "record_digest" not in canonical_internal
    assert item["metadata"]["schema_version"] == D_SCHEMA_VERSION
    assert item["metadata"]["speaker_id"] == 5
    assert item["metadata"]["observer_ids"] == [4, 6, 1]
    for subject, suspicion in rotated["suspected_werewolves"].items():
        expected = suspicion_set_to_pair_target(
            suspicion,
            rotated["known_werewolves"][subject],
            rotated["known_non_werewolves"][subject],
        )
        torch.testing.assert_close(
            item["pair_targets"][PLAYER_TO_ID[subject] - 1],
            expected,
        )
    effective = second_order_effective_subject_mask(
        item["subject_mask"],
        item["reasoning_player_id"],
    )
    assert not effective[4]


def test_d_from_jsonl_and_collate_preserve_existing_output_contract(tmp_path):
    first_path = tmp_path / "tom1.jsonl"
    second_path = tmp_path / "tom2.jsonl"
    first_path.write_text(json.dumps(d_sample(1)) + "\n", encoding="utf-8")
    second_path.write_text(json.dumps(d_sample(2)) + "\n", encoding="utf-8")
    first_item = TWDToMDataset.from_jsonl(first_path, tom_order=1)[0]
    second_item = TWDToMDataset.from_jsonl(second_path, tom_order=2)[0]
    first_batch = collate_twd_tom_samples([first_item])
    second_batch = collate_twd_tom_samples([second_item])

    assert first_batch["pair_targets"].shape == (1, 7, 21)
    assert first_batch["known_werewolves"].shape == (1, 7, 7)
    assert second_batch["pair_targets"].shape == (1, 7, 21)
    assert second_batch["reasoning_player_id"].shape == (1,)
    assert second_batch[
        "post_completed_public_speech_pre_next_action"
    ].shape == (1,)


def test_cyclic_rotation_rejects_illegal_player_id():
    sample = raw_sample(2)
    sample["public_events"][-1]["speaker"] = "player8"
    with pytest.raises(ValueError, match="canonical player"):
        cyclically_rotate_second_order_sample(sample, shift=1)
