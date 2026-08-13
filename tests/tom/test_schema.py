import pytest

from werewolf.models.tom.schema import ACTION_NAMES, ACTION_TO_ID
from werewolf.speech.speech_perceiver import SpeechPerceiver


class FakeBackend:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def parse(speech, response, *, speaker=1):
    perceiver = SpeechPerceiver(FakeBackend(response), "test-model")
    return perceiver.parse(speaker, speech, 1, "speech"), perceiver


def test_formal_action_vocabulary_has_exactly_seven_predicates():
    assert ACTION_NAMES == (
        "point_as_werewolf",
        "point_as_villager",
        "point_as_seer",
        "point_as_witch",
        "point_as_guard",
        "support",
        "oppose",
    )
    assert ACTION_TO_ID == {
        "<pad>": 0,
        **{name: index for index, name in enumerate(ACTION_NAMES, start=1)},
    }


@pytest.mark.parametrize(
    ("speech", "predicate"),
    [
        ("2号是狼人。", "point_as_werewolf"),
        ("2号是平民。", "point_as_villager"),
        ("2号是预言家。", "point_as_seer"),
        ("2号是女巫。", "point_as_witch"),
        ("2号是守卫。", "point_as_guard"),
        ("我支持2号的观点。", "support"),
        ("我反对2号的观点。", "oppose"),
    ],
)
def test_speech_perceiver_accepts_each_formal_predicate(speech, predicate):
    actions, _ = parse(speech, f"player1 | {predicate} | player2")
    assert actions == [["player1", predicate, "player2"]]


@pytest.mark.parametrize("speech", ["2号是好人。", "2号不是狼人。"])
def test_generic_good_or_non_wolf_does_not_become_villager(speech):
    actions, perceiver = parse(speech, "NONE")
    assert actions == []
    prompt = perceiver.backend.calls[0]["messages"][0]["content"]
    assert "好人" in prompt
    assert "非狼" in prompt
    assert "都不能产生该动作" in prompt


@pytest.mark.parametrize(
    "invalid_action",
    [
        "check_as_good",
        "check_as_werewolf",
        "save",
        "poison",
        "guard",
        "vote_intent",
    ],
)
def test_skill_claims_and_vote_intent_are_not_primary_actions(invalid_action):
    actions, perceiver = parse(
        "公开声明。",
        f"player1 | {invalid_action} | player2",
    )
    assert actions == []
    prompt = perceiver.backend.calls[0]["messages"][0]["content"]
    assert f"- {invalid_action}" not in prompt


def test_zero_triplet_speech_is_valid():
    actions, _ = parse("目前没有可抽取的判断。", "NONE")
    assert actions == []


def test_multiple_triplets_preserve_response_order():
    actions, _ = parse(
        "2号是狼人，但我支持3号。",
        "player1 | point_as_werewolf | player2\n"
        "player1 | support | player3",
    )
    assert actions == [
        ["player1", "point_as_werewolf", "player2"],
        ["player1", "support", "player3"],
    ]
