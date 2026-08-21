import pytest

from archive.legacy_tom.werewolf.models.tom.schema import (
    ACTION_NAMES,
    ACTION_TO_ID,
    NONE_ACTION_ID,
    PAD_TOKEN,
)
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


def test_formal_action_vocabulary_has_exact_core_thirteen_ids():
    assert ACTION_NAMES == (
        "point_as_werewolf",
        "point_as_villager",
        "point_as_seer",
        "point_as_witch",
        "point_as_guard",
        "support",
        "oppose",
        "check_as_good",
        "check_as_werewolf",
        "save",
        "poison",
        "guard",
        "vote_intent",
    )
    assert ACTION_TO_ID == {
        "<pad>": 0,
        **{name: index for index, name in enumerate(ACTION_NAMES, start=1)},
    }
    assert ACTION_TO_ID[PAD_TOKEN] == 0
    assert NONE_ACTION_ID == 14
    assert NONE_ACTION_ID != ACTION_TO_ID[PAD_TOKEN]


@pytest.mark.parametrize(
    ("speech", "predicate"),
    [
        ("2号是狼人。", "point_as_werewolf"),
        ("2号是平民。", "point_as_villager"),
        ("2号是预言家。", "point_as_seer"),
        ("2号是女巫。", "point_as_witch"),
        ("我支持2号的观点。", "support"),
        ("我反对2号的观点。", "oppose"),
        ("我查验2号是好人。", "check_as_good"),
        ("我查验2号是狼人。", "check_as_werewolf"),
        ("我救了2号。", "save"),
        ("我毒了2号。", "poison"),
        ("我准备投2号。", "vote_intent"),
    ],
)
def test_speech_perceiver_accepts_each_formal_predicate(speech, predicate):
    actions, _ = parse(speech, f"player1 | {predicate} | player2")
    assert actions == [["player1", predicate, "player2"]]


@pytest.mark.parametrize(
    "speech",
    ["2号是好人。", "2号不是狼人。", "2号值得信任。"],
)
def test_generic_good_or_non_wolf_does_not_become_villager(speech):
    actions, perceiver = parse(speech, "NONE")
    assert actions == []
    prompt = perceiver.backend.calls[0]["messages"][0]["content"]
    assert "好人" in prompt
    assert "非狼" in prompt
    assert "可信" in prompt
    assert "都不能产生该动作" in prompt


@pytest.mark.parametrize(
    ("case", "speech", "response", "expected"),
    [
        (
            "A",
            "我认为3号是狼人",
            "player1 | point_as_werewolf | player3",
            [["player1", "point_as_werewolf", "player3"]],
        ),
        (
            "B",
            "我昨晚查验3号是狼人",
            "player1 | check_as_werewolf | player3",
            [["player1", "check_as_werewolf", "player3"]],
        ),
        (
            "C",
            "我查验2号是好人",
            "player1 | check_as_good | player2",
            [["player1", "check_as_good", "player2"]],
        ),
        (
            "D",
            "我认为2号是村民",
            "player1 | point_as_villager | player2",
            [["player1", "point_as_villager", "player2"]],
        ),
        (
            "F",
            "我支持4号的判断",
            "player1 | support | player4",
            [["player1", "support", "player4"]],
        ),
        (
            "G",
            "我不同意4号的说法",
            "player1 | oppose | player4",
            [["player1", "oppose", "player4"]],
        ),
        (
            "H",
            "我准备投4号",
            "player1 | vote_intent | player4",
            [["player1", "vote_intent", "player4"]],
        ),
        (
            "I",
            "我是女巫，昨晚救了3号",
            "player1 | save | player3",
            [
                ["player1", "point_as_witch", "player1"],
                ["player1", "save", "player3"],
            ],
        ),
        (
            "J",
            "我是女巫，昨晚毒了5号",
            "player1 | poison | player5",
            [
                ["player1", "point_as_witch", "player1"],
                ["player1", "poison", "player5"],
            ],
        ),
    ],
    ids=lambda value: value if isinstance(value, str) and len(value) == 1 else None,
)
def test_core_thirteen_regression_cases(case, speech, response, expected):
    actions, perceiver = parse(speech, response)
    assert actions == expected
    prompt = perceiver.backend.calls[0]["messages"][0]["content"]
    assert "上述可表示类别" in prompt
    if case in {"B", "C", "H"}:
        assert "most-specific-source" in prompt


def test_truth_conflicting_public_skill_claim_is_not_filtered():
    perceiver = SpeechPerceiver(
        FakeBackend("player6 | save | player3"),
        "test-model",
    )
    actions = perceiver.parse(
        6,
        "我是女巫，昨晚救了3号",
        1,
        "speech",
        context={"identity": "Villager"},
    )
    assert actions == [
        ["player6", "point_as_witch", "player6"],
        ["player6", "save", "player3"],
    ]


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
