from copy import deepcopy

import pytest

from werewolf.models.twd_tom.public_events import (
    normalize_public_events,
    public_event_digest,
    structured_event_tokens,
    structured_input_digest,
)


def _events():
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
            "raw_text": "",
            "sp_actions": [],
        },
        {
            "event_idx": 3,
            "event_type": "vote_result",
            "votes": [
                {"voter": "player1", "target": "player2"},
                {"voter": "player3", "target": None},
            ],
        },
        {
            "event_idx": 4,
            "event_type": "exile_result",
            "exiled_players": [],
        },
        {
            "event_idx": 5,
            "event_type": "death_announcement",
            "dead_players": ["player2", "player7"],
        },
    ]


def test_public_event_schema_accepts_all_canonical_event_types():
    assert normalize_public_events(_events()) == _events()
    token_types = [
        token["token_type"]
        for token in structured_event_tokens(_events())
    ]
    assert token_types == [
        "phase_change",
        "turn_start",
        "public_speech",
        "vote_result",
        "vote",
        "vote",
        "exile_result",
        "death_announcement",
        "dead_player",
        "dead_player",
    ]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: rows[1].pop("event_idx"),
        lambda rows: rows[1].update(event_idx=0),
        lambda rows: rows[1].update(event_type="private_skill"),
        lambda rows: rows[1].update(speaker="Player_1"),
        lambda rows: rows[0].update(phase="speech"),
        lambda rows: rows[5].update(death_reason="wolf"),
        lambda rows: rows[4].update(actual_roles={}),
        lambda rows: rows[2].update(sp_actions=[["bad"]]),
        lambda rows: rows[3].update(
            votes=[
                {"voter": "player3", "target": None},
                {"voter": "player1", "target": "player2"},
            ]
        ),
        lambda rows: rows[5].update(
            dead_players=["player7", "player2"]
        ),
    ],
)
def test_public_event_schema_rejects_noncanonical_or_private_payloads(mutate):
    rows = _events()
    mutate(rows)
    with pytest.raises((TypeError, ValueError)):
        normalize_public_events(rows)


def test_digests_separate_public_text_from_structured_model_input():
    first = _events()
    second = deepcopy(first)
    second[2]["raw_text"] = "same parsed structure, different public words"
    assert public_event_digest(first) != public_event_digest(second)
    assert structured_input_digest(first) == structured_input_digest(second)

    second[2]["sp_actions"] = [
        ["player1", "point_as_werewolf", "player2"]
    ]
    assert public_event_digest(first) != public_event_digest(second)
    assert structured_input_digest(first) != structured_input_digest(second)
    assert public_event_digest(first) == public_event_digest(deepcopy(first))
    assert structured_input_digest(first) == structured_input_digest(
        deepcopy(first)
    )


@pytest.mark.parametrize("event_index", [0, 3, 4, 5])
def test_structured_digest_changes_for_public_system_facts(event_index):
    first = _events()
    second = deepcopy(first)
    event = second[event_index]
    if event["event_type"] == "phase_change":
        event["phase"] = "2_day_speech"
    elif event["event_type"] == "vote_result":
        event["votes"][0]["target"] = "player3"
    elif event["event_type"] == "exile_result":
        event["exiled_players"] = ["player2"]
    else:
        event["dead_players"] = []
    assert structured_input_digest(first) != structured_input_digest(second)
