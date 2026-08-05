import torch
import inspect
import pytest

from werewolf.models.twd_tom.action_features import PublicEventFeatureBuilder
from werewolf.models.twd_tom.public_events import STRUCTURED_TOKEN_TO_ID
from werewolf.models.twd_tom.schema import ACTION_TO_ID, PLAYER_TO_ID


def _events(raw_text="", actions=None):
    return [
        {
            "event_idx": 0,
            "event_type": "phase_change",
            "phase": "1_day_speech",
        },
        {
            "event_idx": 1,
            "event_type": "turn_start",
            "speaker": "player1",
        },
        {
            "event_idx": 2,
            "event_type": "public_speech",
            "speaker": "player1",
            "raw_text": raw_text,
            "sp_actions": [] if actions is None else actions,
        },
        {
            "event_idx": 3,
            "event_type": "exile_result",
            "exiled_players": [],
        },
        {
            "event_idx": 4,
            "event_type": "death_announcement",
            "dead_players": [],
        },
    ]


def test_event_encoder_keeps_every_boundary_and_empty_results():
    features = PublicEventFeatureBuilder().encode_events(_events())
    assert features["attention_mask"].tolist() == [1, 1, 1, 1, 1]
    assert features["event_type_ids"].tolist() == [
        STRUCTURED_TOKEN_TO_ID["phase_change"],
        STRUCTURED_TOKEN_TO_ID["turn_start"],
        STRUCTURED_TOKEN_TO_ID["public_speech"],
        STRUCTURED_TOKEN_TO_ID["exile_result"],
        STRUCTURED_TOKEN_TO_ID["death_announcement"],
    ]


def test_event_encoder_excludes_raw_text_but_keeps_multiple_speech_actions():
    actions = [
        ["player1", "support", "player2"],
        ["player1", "oppose", "player3"],
    ]
    builder = PublicEventFeatureBuilder()
    first = builder.encode_events(_events("first", actions))
    second = builder.encode_events(_events("second", actions))
    assert all(torch.equal(first[key], second[key]) for key in first)
    assert first["attention_mask"].sum().item() == 7
    assert first["event_type_ids"].tolist().count(
        STRUCTURED_TOKEN_TO_ID["public_speech"]
    ) == 1
    assert first["event_type_ids"].tolist().count(
        STRUCTURED_TOKEN_TO_ID["speech_action"]
    ) == 2


def test_event_encoder_right_pads_and_preserves_order():
    builder = PublicEventFeatureBuilder()
    batch = builder.encode_batch([_events(), _events(actions=[
        ["player1", "support", "player2"],
    ])])
    assert batch["attention_mask"].shape == (2, 6)
    assert batch["attention_mask"][0].tolist() == [1, 1, 1, 1, 1, 0]
    assert batch["attention_mask"][1].tolist() == [1, 1, 1, 1, 1, 1]


def test_encodes_exact_subject_action_object_ids_and_dtypes():
    features = PublicEventFeatureBuilder().encode_events(
        _events(actions=[["player1", "support", "player2"]])
    )
    index = features["event_type_ids"].tolist().index(
        STRUCTURED_TOKEN_TO_ID["speech_action"]
    )
    assert features["subject_ids"][index].item() == PLAYER_TO_ID["player1"]
    assert features["action_ids"][index].item() == ACTION_TO_ID["support"]
    assert features["object_ids"][index].item() == PLAYER_TO_ID["player2"]
    assert all(
        tensor.dtype == torch.long
        for name, tensor in features.items()
        if name != "day_values"
    )
    assert features["day_values"].dtype == torch.float32


@pytest.mark.parametrize(
    "action_name",
    (
        "check_as_good",
        "check_as_werewolf",
        "save",
        "poison",
        "guard",
        "vote_intent",
    ),
)
def test_extended_actions_encode_to_their_canonical_ids(action_name):
    features = PublicEventFeatureBuilder().encode_events(
        _events(actions=[["player1", action_name, "player2"]])
    )
    index = features["event_type_ids"].tolist().index(
        STRUCTURED_TOKEN_TO_ID["speech_action"]
    )
    assert features["action_ids"][index].item() == ACTION_TO_ID[action_name]


def test_preserves_duplicate_actions_inside_one_speech_boundary():
    action = ["player1", "support", "player2"]
    features = PublicEventFeatureBuilder().encode_events(
        _events(actions=[action, action])
    )
    assert features["event_type_ids"].tolist().count(
        STRUCTURED_TOKEN_TO_ID["speech_action"]
    ) == 2


def test_empty_event_history_has_one_masked_padding_position():
    features = PublicEventFeatureBuilder().encode_events([])
    assert all(tensor.shape == (1,) for tensor in features.values())
    assert features["attention_mask"].tolist() == [0]
    assert all(
        torch.count_nonzero(tensor).item() == 0
        for tensor in features.values()
    )


def test_truncation_keeps_recent_events_without_dropping_their_boundaries():
    actions = [
        ["player1", "support", "player2"],
        ["player1", "oppose", "player3"],
        ["player1", "point_as_werewolf", "player4"],
    ]
    features = PublicEventFeatureBuilder(max_seq_len=3).encode_events(
        _events(actions=actions)
    )
    assert features["event_type_ids"].tolist() == [
        STRUCTURED_TOKEN_TO_ID["exile_result"],
        STRUCTURED_TOKEN_TO_ID["death_announcement"],
    ]

    speech_only = _events(actions=actions)[:3]
    features = PublicEventFeatureBuilder(max_seq_len=3).encode_events(
        speech_only
    )
    assert features["event_type_ids"].tolist() == [
        STRUCTURED_TOKEN_TO_ID["public_speech"],
        STRUCTURED_TOKEN_TO_ID["speech_action"],
        STRUCTURED_TOKEN_TO_ID["speech_action"],
    ]
    assert features["object_ids"][-1].item() == PLAYER_TO_ID["player4"]


@pytest.mark.parametrize(
    "actions",
    [
        [["player1", "unknown", "player2"]],
        [["player8", "support", "player2"]],
        [["player1", "support"]],
    ],
)
def test_invalid_speech_actions_are_rejected(actions):
    with pytest.raises((TypeError, ValueError)):
        PublicEventFeatureBuilder().encode_events(_events(actions=actions))


def test_empty_batch_and_invalid_max_length_are_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        PublicEventFeatureBuilder().encode_batch([])
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            PublicEventFeatureBuilder(max_seq_len=value)


def test_encoder_is_parameter_free_and_api_has_no_private_or_truth_inputs():
    builder = PublicEventFeatureBuilder()
    assert not hasattr(builder, "parameters")
    parameters = set(inspect.signature(builder.encode_events).parameters)
    assert parameters == {"public_events"}
    assert {
        "roles",
        "actual_roles",
        "observer_role",
        "private_observation",
        "raw_text",
        "pair_targets",
    }.isdisjoint(parameters)
