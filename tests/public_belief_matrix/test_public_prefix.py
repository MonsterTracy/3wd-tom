from copy import deepcopy

import torch

from werewolf.models.public_belief_matrix.public_prefix import (
    build_public_belief_matrix_visible_prefix,
)
from werewolf.models.twd_tom.public_events import STRUCTURED_TOKEN_TO_ID
from werewolf.models.twd_tom.schema import ACTION_TO_ID, PLAYER_TO_ID


def _speech_events(*, raw_text="private-looking canary", sp_actions=None):
    if sp_actions is None:
        sp_actions = [["player1", "point_as_werewolf", "player3"]]
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
            "sp_actions": sp_actions,
        },
    ]


def _assert_same_features(first, second):
    assert first.keys() == second.keys()
    for field in first:
        torch.testing.assert_close(first[field], second[field])


def test_visible_prefix_excludes_raw_text_and_retains_structured_actions():
    first_events = _speech_events()
    second_events = deepcopy(first_events)
    second_events[-1]["raw_text"] = "different raw public speech"

    first = build_public_belief_matrix_visible_prefix(first_events)
    second = build_public_belief_matrix_visible_prefix(second_events)

    _assert_same_features(first, second)
    assert ACTION_TO_ID["point_as_werewolf"] in first["action_ids"].tolist()
    assert PLAYER_TO_ID["player3"] in first["object_ids"].tolist()


def test_empty_action_speech_retains_the_public_speech_boundary():
    features = build_public_belief_matrix_visible_prefix(
        _speech_events(sp_actions=[])
    )

    assert STRUCTURED_TOKEN_TO_ID["public_speech"] in (
        features["event_type_ids"].tolist()
    )


def test_default_truncation_is_deterministic_and_bounded_to_256_tokens():
    public_events = [
        {
            "event_idx": index,
            "event_type": "turn_start",
            "speaker": f"player{index % 7 + 1}",
        }
        for index in range(260)
    ]

    first = build_public_belief_matrix_visible_prefix(public_events)
    second = build_public_belief_matrix_visible_prefix(public_events)

    _assert_same_features(first, second)
    assert first["attention_mask"].shape == (256,)
    assert first["attention_mask"].tolist() == [1] * 256
    assert first["subject_ids"][0].item() == PLAYER_TO_ID["player5"]
