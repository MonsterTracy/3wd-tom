import pytest

from werewolf.models.twd_tom.schema import (
    ACTION_NAMES,
    ACTION_TO_ID,
    NONE_TOKEN,
    PLAYER_TO_ID,
    SpeechAction,
    parse_speech_action,
    validate_player_suspicion,
)


def test_action_vocabulary_is_minimal_and_fixed():
    assert ACTION_NAMES == (
        "point_as_werewolf",
        "point_as_villager",
        "point_as_seer",
        "point_as_witch",
        "support",
        "oppose",
        "check_as_good",
        "check_as_werewolf",
        "save",
        "poison",
        "vote_intent",
        "abstain_intent",
        "no_commitment",
    )

    assert "suspect" not in ACTION_NAMES
    assert "certainty" not in ACTION_NAMES
    assert "vote_intention" not in ACTION_NAMES
    assert len(ACTION_NAMES) == 13


def test_padding_ids_are_separate_from_real_values():
    assert PLAYER_TO_ID["<pad>"] == 0
    assert PLAYER_TO_ID["player1"] == 1
    assert PLAYER_TO_ID["player7"] == 7
    assert PLAYER_TO_ID[NONE_TOKEN] == 8
    assert PLAYER_TO_ID[NONE_TOKEN] != PLAYER_TO_ID["<pad>"]

    assert ACTION_TO_ID["<pad>"] == 0
    assert ACTION_TO_ID == {
        "<pad>": 0,
        "point_as_werewolf": 1,
        "point_as_villager": 2,
        "point_as_seer": 3,
        "point_as_witch": 4,
        "support": 5,
        "oppose": 6,
        "check_as_good": 7,
        "check_as_werewolf": 8,
        "save": 9,
        "poison": 10,
        "vote_intent": 11,
        "abstain_intent": 12,
        "no_commitment": 13,
    }
    assert len(set(ACTION_TO_ID.values())) == len(ACTION_TO_ID)


def test_parse_onuw_style_speech_action():
    action = parse_speech_action(
        [3, "point_as_werewolf", "Player_5"]
    )

    assert action == SpeechAction(
        subject="player3",
        action="point_as_werewolf",
        object="player5",
    )

    assert action.to_list() == [
        "player3",
        "point_as_werewolf",
        "player5",
    ]


@pytest.mark.parametrize(
    "action_name",
    (
        "suspect",
        "check_good",
        "checked_good",
        "heal",
        "protected",
        "intend_vote",
    ),
)
def test_unsupported_action_is_rejected(action_name):
    with pytest.raises(ValueError, match="unsupported speech action"):
        parse_speech_action(
            ["player1", action_name, "player2"]
        )


@pytest.mark.parametrize(
    "action_name",
    (
        "check_as_good",
        "check_as_werewolf",
        "save",
        "poison",
        "vote_intent",
    ),
)
def test_extended_actions_use_canonical_validation(action_name):
    assert parse_speech_action(
        ["player1", action_name, "player2"]
    ).to_list() == ["player1", action_name, "player2"]


@pytest.mark.parametrize("action_name", ("abstain_intent", "no_commitment"))
def test_targetless_actions_require_none_object(action_name):
    assert parse_speech_action(
        ["player1", action_name, None]
    ).to_list() == ["player1", action_name, None]

    with pytest.raises(ValueError, match="must use object=None"):
        parse_speech_action(["player1", action_name, "player2"])


def test_targeted_actions_reject_none_object():
    with pytest.raises(ValueError, match="requires a canonical player object"):
        parse_speech_action(["player1", "oppose", None])


@pytest.mark.parametrize("action_name", ("point_as_guard", "guard"))
def test_guard_predicates_are_not_active(action_name):
    with pytest.raises(ValueError, match="unsupported speech action"):
        parse_speech_action(["player1", action_name, "player2"])


@pytest.mark.parametrize(
    ("suspected", "known_wolves", "known_non_wolves", "expected"),
    [
        ([], [], ["player1"], []),
        (["player3"], ["player3"], ["player1"], ["player3"]),
        (
            ["player2", "player6"],
            ["player2", "player6"],
            ["player1", "player3", "player4", "player5", "player7"],
            ["player2", "player6"],
        ),
        (["player2"], [], ["player1"], ["player2"]),
        (
            ["player2", "player4"],
            [],
            ["player1"],
            ["player2", "player4"],
        ),
        (
            ["player3", "player5"],
            ["player3"],
            ["player2", "player6", "player7"],
            ["player3", "player5"],
        ),
        (
            ["player2", "player3", "player4", "player5", "player6", "player7"],
            [],
            ["player1"],
            ["player2", "player3", "player4", "player5", "player6", "player7"],
        ),
        (
            ["player1", "player3", "player4", "player5"],
            ["player3"],
            ["player2", "player6", "player7"],
            ["player1", "player3", "player4", "player5"],
        ),
    ],
)
def test_player_suspicion_canonical_no_extra_forms_are_valid(
    suspected,
    known_wolves,
    known_non_wolves,
    expected,
):
    assert validate_player_suspicion(
        suspected,
        known_wolves,
        known_non_wolves,
    ) == expected


@pytest.mark.parametrize(
    ("suspected", "known_wolves", "known_non_wolves", "match"),
    [
        ([], ["player3"], ["player1"], "contain all known"),
        (["player1"], [], ["player1"], "known_non_werewolves"),
        (["player2", "player2"], [], ["player1"], "duplicate"),
        (["player8"], [], ["player1"], "canonical"),
    ],
)
def test_player_suspicion_invalid_forms_fail_closed(
    suspected,
    known_wolves,
    known_non_wolves,
    match,
):
    with pytest.raises((TypeError, ValueError), match=match):
        validate_player_suspicion(
            suspected,
            known_wolves,
            known_non_wolves,
        )
