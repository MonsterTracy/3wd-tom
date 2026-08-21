from copy import deepcopy

import pytest

from archive.legacy_tom.werewolf.models.tom.public_history import build_model_input


def ledger():
    return [
        {"event_idx": 0, "event_type": "death_announcement", "dead_players": []},
        {"event_idx": 1, "event_type": "phase_change", "phase": "1_day_speech"},
        {"event_idx": 2, "event_type": "turn_start", "speaker": "player1"},
        {
            "event_idx": 3,
            "event_type": "public_speech",
            "speaker": "player1",
            "raw_text": "2号是狼人。",
            "sp_actions": [["player1", "point_as_werewolf", "player2"]],
        },
        {"event_idx": 4, "event_type": "turn_start", "speaker": "player2"},
        {
            "event_idx": 5,
            "event_type": "public_speech",
            "speaker": "player2",
            "raw_text": "暂无判断。",
            "sp_actions": [],
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
        {"event_idx": 11, "event_type": "phase_change", "phase": "2_day_speech_pk"},
        {
            "event_idx": 12,
            "event_type": "public_speech",
            "speaker": "player5",
            "raw_text": "我反对6号。",
            "sp_actions": [["player5", "oppose", "player6"]],
        },
        {"event_idx": 13, "event_type": "phase_change", "phase": "2_day_vote_pk"},
        {"event_idx": 14, "event_type": "vote_result", "votes": []},
        {"event_idx": 15, "event_type": "exile_result", "exiled_players": []},
    ]


def test_projection_contains_only_four_public_evidence_types():
    source = ledger()
    before = deepcopy(source)
    result = build_model_input(episode_context="seer_witch", public_events=source)
    assert source == before
    assert {event["type"] for event in result["events"]} == {
        "speech_action",
        "vote",
        "exile",
        "night_result",
    }
    assert all("raw_text" not in event for event in result["events"])
    assert all(
        event["type"] not in {"turn_start", "public_speech", "phase_change"}
        for event in result["events"]
    )


def test_speech_actions_and_round_phase_metadata_are_projected():
    events = build_model_input(
        episode_context="seer_guard",
        public_events=ledger(),
    )["events"]
    speech = [event for event in events if event["type"] == "speech_action"]
    assert speech == [
        {
            "type": "speech_action",
            "subject": "player1",
            "predicate": "point_as_werewolf",
            "object": "player2",
            "round": 1,
            "phase": "discussion",
        },
        {
            "type": "speech_action",
            "subject": "player5",
            "predicate": "oppose",
            "object": "player6",
            "round": 2,
            "phase": "pk_discussion",
        },
    ]


def test_completed_votes_include_abstention():
    votes = [
        event
        for event in build_model_input(
            episode_context="seer_guard", public_events=ledger()
        )["events"]
        if event["type"] == "vote"
    ]
    assert votes == [
        {"type": "vote", "voter": "player1", "target": "player2", "round": 1, "phase": "vote"},
        {"type": "vote", "voter": "player2", "target": None, "round": 1, "phase": "vote"},
    ]


def test_exile_player_and_none_are_represented():
    exiles = [
        event
        for event in build_model_input(
            episode_context="seer_guard", public_events=ledger()
        )["events"]
        if event["type"] == "exile"
    ]
    assert [event["player"] for event in exiles] == ["player2", None]
    assert [event["phase"] for event in exiles] == ["vote", "pk_vote"]


def test_night_deaths_and_peaceful_night_are_single_events():
    nights = [
        event
        for event in build_model_input(
            episode_context="seer_witch", public_events=ledger()
        )["events"]
        if event["type"] == "night_result"
    ]
    assert nights == [
        {"type": "night_result", "dead_players": [], "round": 1, "phase": "night"},
        {
            "type": "night_result",
            "dead_players": ["player3", "player4"],
            "round": 2,
            "phase": "night",
        },
    ]


def test_static_episode_context_distinguishes_both_public_variants():
    assert build_model_input(
        episode_context="seer_guard", public_events=[]
    )["episode_context"] == "seer_guard"
    assert build_model_input(
        episode_context="seer_witch", public_events=[]
    )["episode_context"] == "seer_witch"
    with pytest.raises(ValueError, match="episode context"):
        build_model_input(episode_context="unknown", public_events=[])


def test_private_skill_details_and_bookkeeping_are_not_model_input():
    result = build_model_input(
        episode_context="seer_witch",
        public_events=ledger(),
    )
    serialized = repr(result)
    for forbidden in (
        "skill_wolf",
        "turn_start",
        "public_speech",
        "phase_change",
        "kill_target",
        "seer_check",
        "witch_action",
    ):
        assert forbidden not in serialized


def test_core_thirteen_speech_action_is_projected_as_public_semantics():
    source = ledger()
    source[3]["sp_actions"] = [["player1", "vote_intent", "player2"]]
    events = build_model_input(
        episode_context="seer_guard",
        public_events=source,
    )["events"]
    assert events[1] == {
        "type": "speech_action",
        "subject": "player1",
        "predicate": "vote_intent",
        "object": "player2",
        "round": 1,
        "phase": "discussion",
    }
