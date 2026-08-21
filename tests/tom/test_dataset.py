import json
from copy import deepcopy

import pytest
import torch

import archive.legacy_tom.werewolf.models.tom.dataset as dataset_module
from archive.legacy_tom.werewolf.models.tom.dataset import (
    TomDataset,
    collate_batch,
    encode_sample,
)
from archive.legacy_tom.werewolf.models.tom.schema import (
    ACTION_NAMES,
    ACTION_TO_ID,
    CONFIG_TO_ID,
    EVENT_TO_ID,
    NONE_ACTION_ID,
    NONE_TOKEN,
    PAD_TOKEN,
    PHASE_TO_ID,
    PLAYER_NAMES,
    PLAYER_TO_ID,
)
from archive.legacy_tom.werewolf.models.tom.targets import materialize_target


CURRENT_ACTIONS = [
    ["player5", "oppose", "player6"],
    ["player5", "point_as_guard", "player7"],
]
MODEL_FIELDS = (
    "config_id",
    "event_type_ids",
    "subject_ids",
    "action_ids",
    "object_ids",
    "rounds",
    "phase_ids",
    "dead_players",
    "attention_mask",
    "sequence_length",
)


def raw_sample():
    return {
        "game_id": "game-1",
        "seed": 17,
        "episode_context": "seer_witch",
        "step_idx": 14,
        "speaker_id": "player5",
        "round": 2,
        "phase": "speech",
        "formal_speech_actions": deepcopy(CURRENT_ACTIONS),
        "public_history_cutoff": {"event_idx": 13, "digest": "audit-only"},
        "public_events": [
            {"event_idx": 0, "event_type": "death_announcement", "dead_players": []},
            {"event_idx": 1, "event_type": "phase_change", "phase": "1_day_speech"},
            {"event_idx": 2, "event_type": "turn_start", "speaker": "player1"},
            {
                "event_idx": 3,
                "event_type": "public_speech",
                "speaker": "player1",
                "raw_text": "RAW-FIRST",
                "sp_actions": [["player1", "point_as_werewolf", "player2"]],
            },
            {"event_idx": 4, "event_type": "turn_start", "speaker": "player2"},
            {
                "event_idx": 5,
                "event_type": "public_speech",
                "speaker": "player2",
                "raw_text": "RAW-SECOND",
                "sp_actions": [["player2", "support", "player1"]],
            },
            {"event_idx": 6, "event_type": "phase_change", "phase": "1_day_vote"},
            {
                "event_idx": 7,
                "event_type": "vote_result",
                "votes": [
                    {"voter": "player1", "target": "player2"},
                    {"voter": "player2", "target": None},
                ],
            },
            {"event_idx": 8, "event_type": "exile_result", "exiled_players": ["player2"]},
            {"event_idx": 9, "event_type": "phase_change", "phase": "1_night_skill_wolf"},
            {
                "event_idx": 10,
                "event_type": "death_announcement",
                "dead_players": ["player3", "player4"],
            },
            {"event_idx": 11, "event_type": "phase_change", "phase": "2_day_speech"},
            {"event_idx": 12, "event_type": "turn_start", "speaker": "player5"},
            {
                "event_idx": 13,
                "event_type": "public_speech",
                "speaker": "player5",
                "raw_text": "RAW-CURRENT",
                "sp_actions": deepcopy(CURRENT_ACTIONS),
            },
            {"event_idx": 14, "event_type": "turn_start", "speaker": "player6"},
        ],
        "alive_observers": ["player1", "player5"],
        "observer_reports": [
            {
                "observer_id": "player1",
                "valid": True,
                "suspected_werewolves": [],
                "error": None,
            },
            {
                "observer_id": "player5",
                "valid": False,
                "suspected_werewolves": None,
                "error": "parse_error",
            },
        ],
        "private_observations": "PRIVATE-CANARY",
        "true_roles": "TRUTH-CANARY",
    }


def short_sample(*, context="seer_guard", phase="speech"):
    actions = [["player1", "support", "player2"]]
    sample = raw_sample()
    sample["episode_context"] = context
    sample["formal_speech_actions"] = actions
    sample["public_history_cutoff"] = {"event_idx": 2, "digest": "audit"}
    sample["public_events"] = [
        {"event_idx": 0, "event_type": "death_announcement", "dead_players": []},
        {"event_idx": 1, "event_type": "phase_change", "phase": f"1_day_{phase}"},
        {
            "event_idx": 2,
            "event_type": "public_speech",
            "speaker": "player1",
            "raw_text": "RAW-SHORT",
            "sp_actions": actions,
        },
    ]
    return sample


def assert_same_features(first, second):
    for field in MODEL_FIELDS:
        torch.testing.assert_close(first[field], second[field])


def test_multi_round_prefix_is_complete_chronological_and_ends_at_current_action():
    item = encode_sample(raw_sample())
    assert item["sequence_length"].item() == 9
    assert item["event_type_ids"].tolist() == [
        EVENT_TO_ID["night_result"],
        EVENT_TO_ID["speech_action"],
        EVENT_TO_ID["speech_action"],
        EVENT_TO_ID["vote"],
        EVENT_TO_ID["vote"],
        EVENT_TO_ID["exile"],
        EVENT_TO_ID["night_result"],
        EVENT_TO_ID["speech_action"],
        EVENT_TO_ID["speech_action"],
    ]
    assert item["action_ids"].tolist()[1:3] == [
        ACTION_TO_ID["point_as_werewolf"],
        ACTION_TO_ID["support"],
    ]
    assert item["subject_ids"].tolist()[3:5] == [
        PLAYER_TO_ID["player1"],
        PLAYER_TO_ID["player2"],
    ]
    assert item["object_ids"].tolist()[3:6] == [
        PLAYER_TO_ID["player2"],
        PLAYER_TO_ID[NONE_TOKEN],
        PLAYER_TO_ID["player2"],
    ]
    assert item["rounds"].tolist() == [1, 1, 1, 1, 1, 1, 2, 2, 2]
    assert item["phase_ids"].tolist()[-2:] == [
        PHASE_TO_ID["discussion"],
        PHASE_TO_ID["discussion"],
    ]
    assert item["action_ids"].tolist()[-2:] == [
        ACTION_TO_ID["oppose"],
        ACTION_TO_ID["point_as_guard"],
    ]
    assert item["subject_ids"][-1].item() == PLAYER_TO_ID["player5"]
    assert item["action_ids"][-1].item() == ACTION_TO_ID["point_as_guard"]
    assert item["object_ids"][-1].item() == PLAYER_TO_ID["player7"]
    assert item["attention_mask"].all().item()


def test_all_core_thirteen_speech_actions_encode_with_exact_ids():
    sample = short_sample()
    actions = [
        ["player1", action, "player2"]
        for action in ACTION_NAMES
    ]
    sample["formal_speech_actions"] = deepcopy(actions)
    sample["public_events"][-1]["sp_actions"] = deepcopy(actions)

    item = encode_sample(sample)

    assert item["sequence_length"].item() == 14
    assert item["action_ids"][0].item() == NONE_ACTION_ID
    assert item["action_ids"][1:].tolist() == list(range(1, 14))


def test_night_result_is_lossless_and_peaceful_night_is_a_real_event():
    item = encode_sample(raw_sample())
    assert item["dead_players"][0].tolist() == [False] * 7
    assert item["event_type_ids"][0].item() == EVENT_TO_ID["night_result"]
    assert item["attention_mask"][0].item() is True
    assert item["dead_players"][6].tolist() == [
        False, False, True, True, False, False, False
    ]


def test_static_config_and_pk_phase_have_canonical_ids_not_synthetic_tokens():
    guard = encode_sample(short_sample(context="seer_guard"))
    witch = encode_sample(short_sample(context="seer_witch"))
    pk = encode_sample(short_sample(phase="speech_pk"))
    assert guard["config_id"].item() == CONFIG_TO_ID["seer_guard"]
    assert witch["config_id"].item() == CONFIG_TO_ID["seer_witch"]
    assert guard["config_id"].item() != witch["config_id"].item()
    assert guard["event_type_ids"].tolist() == witch["event_type_ids"].tolist()
    assert guard["sequence_length"].item() == witch["sequence_length"].item()
    assert guard["phase_ids"][-1].item() == PHASE_TO_ID["discussion"]
    assert pk["phase_ids"][-1].item() == PHASE_TO_ID["pk_discussion"]


def test_semantic_none_is_distinct_from_padding_for_vote_exile_and_action():
    item = encode_sample(raw_sample())
    assert PLAYER_TO_ID[NONE_TOKEN] != PLAYER_TO_ID[PAD_TOKEN]
    assert NONE_ACTION_ID != ACTION_TO_ID[PAD_TOKEN]
    assert item["object_ids"][4].item() == PLAYER_TO_ID[NONE_TOKEN]
    assert item["action_ids"][4].item() == NONE_ACTION_ID
    assert [item["action_ids"][index].item() for index in (0, 3, 4, 5, 6)] == [
        NONE_ACTION_ID,
    ] * 5

    no_exile = raw_sample()
    no_exile["public_events"][8]["exiled_players"] = []
    encoded = encode_sample(no_exile)
    assert encoded["object_ids"][5].item() == PLAYER_TO_ID[NONE_TOKEN]


def test_target_and_mask_come_directly_from_phase_three(monkeypatch):
    sample = raw_sample()
    expected = materialize_target(
        alive_observers=sample["alive_observers"],
        observer_reports=sample["observer_reports"],
    )
    calls = []
    original = dataset_module.materialize_target

    def recording_materializer(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(dataset_module, "materialize_target", recording_materializer)
    item = encode_sample(sample)
    assert calls == [
        {
            "alive_observers": sample["alive_observers"],
            "observer_reports": sample["observer_reports"],
        }
    ]
    torch.testing.assert_close(item["target"], torch.tensor(expected[0]))
    assert torch.equal(item["observer_mask"], torch.tensor(expected[1]))
    torch.testing.assert_close(item["target"][0], torch.full((7,), 1.0 / 7.0))
    assert item["observer_mask"].tolist() == [True, False, False, False, False, False, False]


def test_label_side_and_nonformal_fields_cannot_change_model_features():
    first_sample = raw_sample()
    second_sample = deepcopy(first_sample)
    second_sample["alive_observers"] = ["player5", "player1"]
    second_sample["observer_reports"] = list(reversed(second_sample["observer_reports"]))
    second_sample["observer_reports"][0]["error"] = "DIFFERENT-ERROR"
    second_sample["observer_reports"][1]["suspected_werewolves"] = ["player2"]
    second_sample["true_roles"] = "DIFFERENT-TRUTH"
    second_sample["private_observations"] = "DIFFERENT-PRIVATE"
    for event in second_sample["public_events"]:
        if event["event_type"] == "public_speech":
            event["raw_text"] = "DIFFERENT-RAW"
    assert_same_features(encode_sample(first_sample), encode_sample(second_sample))


def test_dictionary_insertion_order_does_not_change_encoding():
    first = raw_sample()
    second = deepcopy(first)
    second["public_events"] = [
        dict(reversed(list(event.items())))
        for event in second["public_events"]
    ]
    assert_same_features(encode_sample(first), encode_sample(second))


def test_dataset_loads_jsonl_and_uses_required_dtypes(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text(json.dumps(raw_sample()) + "\n", encoding="utf-8")
    dataset = TomDataset(path)
    item = dataset[0]
    assert len(dataset) == 1
    for field in (*MODEL_FIELDS[:7], "sequence_length"):
        assert item[field].dtype == torch.long
    assert item["dead_players"].dtype == torch.bool
    assert item["attention_mask"].dtype == torch.bool
    assert item["target"].dtype == torch.float32
    assert item["observer_mask"].dtype == torch.bool
    assert item["target"].shape == (7, 7)
    assert item["observer_mask"].shape == (7,)


def test_collate_right_pads_to_batch_max_without_truncation():
    long_item = encode_sample(raw_sample())
    short_item = encode_sample(short_sample())
    batch = collate_batch([short_item, long_item])
    assert batch["event_type_ids"].shape == (2, 9)
    assert batch["sequence_length"].tolist() == [2, 9]
    assert batch["attention_mask"][0].tolist() == [True, True] + [False] * 7
    assert batch["attention_mask"][1].tolist() == [True] * 9
    for field in (
        "event_type_ids", "subject_ids", "action_ids",
        "object_ids", "rounds", "phase_ids",
    ):
        assert batch[field][0, 2:].tolist() == [0] * 7
        assert torch.equal(batch[field][1], long_item[field])
    assert not batch["dead_players"][0, 2:].any().item()
    assert batch["target"].shape == (2, 7, 7)
    assert batch["observer_mask"].shape == (2, 7)


def test_zero_triplet_and_current_speech_mismatch_are_malformed():
    zero = raw_sample()
    zero["formal_speech_actions"] = []
    with pytest.raises(ValueError, match="must not be empty"):
        encode_sample(zero)

    mismatch = raw_sample()
    mismatch["formal_speech_actions"] = list(reversed(CURRENT_ACTIONS))
    with pytest.raises(ValueError, match="do not match"):
        encode_sample(mismatch)


def test_unknown_event_action_config_and_future_evidence_are_malformed():
    unknown_event = raw_sample()
    unknown_event["public_events"][12]["event_type"] = "unknown"
    with pytest.raises(ValueError, match="unsupported public ledger event"):
        encode_sample(unknown_event)

    unknown_action = raw_sample()
    unknown_action["formal_speech_actions"][0][1] = "unknown"
    unknown_action["public_events"][13]["sp_actions"][0][1] = "unknown"
    with pytest.raises(ValueError, match="unsupported speech action"):
        encode_sample(unknown_action)

    invalid_config = raw_sample()
    invalid_config["episode_context"] = "unknown"
    with pytest.raises(ValueError, match="episode context"):
        encode_sample(invalid_config)

    future = raw_sample()
    future["public_events"][14] = {
        "event_idx": 14,
        "event_type": "vote_result",
        "votes": [],
    }
    with pytest.raises(ValueError, match="after the speech cutoff"):
        encode_sample(future)
