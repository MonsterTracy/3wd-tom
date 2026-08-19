import unittest

from werewolf.models import SpeechPerceiver


class FakeBackend:
    def __init__(
        self,
        content,
    ):
        self.content = content
        self.calls = []

    def chat(
        self,
        messages,
        model=None,
        temperature=0.7,
        **kwargs,
    ):
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                **kwargs,
            }
        )
        return self.content


class SpeechPerceiverPilotCasesTest(
    unittest.TestCase
):
    def parse_with_response(
        self,
        *,
        speaker,
        speech,
        response,
    ):
        backend = FakeBackend(
            response
        )
        perceiver = SpeechPerceiver(
            backend=backend,
            model_name="test-model",
        )

        actions = perceiver.parse(
            speaker=speaker,
            speech=speech,
            day=1,
            phase="speech",
        )

        prompt = backend.calls[0][
            "messages"
        ][0]["content"]

        return actions, prompt

    def test_explicit_logic_disagreement_is_encoded_as_oppose(
        self,
    ):
        actions, prompt = self.parse_with_response(
            speaker=5,
            speech="我不同意2号的逻辑。",
            response=(
                "player5 | oppose | player2"
            ),
        )

        self.assertEqual(
            actions,
            [
                [
                    "player5",
                    "oppose",
                    "player2",
                ]
            ],
        )
        self.assertIn(
            "明确反对、不信任、质疑",
            prompt,
        )

    def test_real_suspicion_text_stays_oppose_only(
        self,
    ):
        actions, prompt = self.parse_with_response(
            speaker=1,
            speech="同时质疑 player3 的发言。",
            response="player1 | oppose | player3",
        )

        self.assertEqual(
            actions,
            [["player1", "oppose", "player3"]],
        )
        self.assertIn(
            "不得自动升级为 point_as_werewolf",
            prompt,
        )

    def test_endorsing_a_claim_supports_the_speaker(
        self,
    ):
        actions, _ = self.parse_with_response(
            speaker=3,
            speech=(
                "7号跳预言家查杀1号，"
                "我暂时站7号。"
            ),
            response=(
                "player3 | support | player7"
            ),
        )

        self.assertEqual(
            actions,
            [
                [
                    "player3",
                    "support",
                    "player7",
                ]
            ],
        )

    def test_explicit_player6_villager_claim_is_never_dropped(
        self,
    ):
        actions, prompt = self.parse_with_response(
            speaker=6,
            speech=(
                "我是6号玩家，身份是村民。目前听了3、4、5号的发言，"
                "大家都比较中立，没有明显的矛盾或攻击性。"
                "平安夜的情况我也倾向于女巫开了解药，毕竟狼人首夜空刀收益不大。"
                "我会继续观察后面的发言，希望预言家如果有查验可以适当暗示，"
                "女巫也可以提供信息。目前没有明确的怀疑对象，大家先多聊聊，"
                "好人们一起分析找狼。"
            ),
            response="NONE",
        )

        self.assertEqual(
            actions,
            [
                [
                    "player6",
                    "point_as_villager",
                    "player6",
                ]
            ],
        )
        self.assertIn(
            "第一人称明确自报具体身份必须抽取",
            prompt,
        )

    def test_identity_judgement_and_support_can_coexist(
        self,
    ):
        actions, prompt = self.parse_with_response(
            speaker=2,
            speech=(
                "1号大概率是真的预言家，"
                "我建议好人先相信1号。"
            ),
            response=(
                "player2 | point_as_seer | player1\n"
                "player2 | support | player1"
            ),
        )

        self.assertEqual(
            actions,
            [
                [
                    "player2",
                    "point_as_seer",
                    "player1",
                ],
                [
                    "player2",
                    "support",
                    "player1",
                ],
            ],
        )
        self.assertIn(
            "多个不同命题按原文语义顺序输出",
            prompt,
        )

    def test_ambiguous_double_claim_has_no_action(
        self,
    ):
        actions, _ = self.parse_with_response(
            speaker=4,
            speech=(
                "1号和7号都在跳预言家，"
                "我还需要继续听。"
            ),
            response="NONE",
        )

        self.assertEqual(
            actions,
            [],
        )

    def test_multiple_explicit_identity_judgements(
        self,
    ):
        actions, _ = self.parse_with_response(
            speaker=2,
            speech=(
                "我认为1号是狼人，"
                "7号更像预言家。"
            ),
            response=(
                "player2 | point_as_werewolf | player1\n"
                "player2 | point_as_seer | player7"
            ),
        )

        self.assertEqual(
            actions,
            [
                [
                    "player2",
                    "point_as_werewolf",
                    "player1",
                ],
                [
                    "player2",
                    "point_as_seer",
                    "player7",
                ],
            ],
        )

    def test_vote_intent_is_distinct_without_erasing_other_actions(
        self,
    ):
        cases = (
            (
                "这一轮我倾向投4号。",
                "player6 | vote_intent | player4",
                [["player6", "vote_intent", "player4"]],
            ),
            (
                "这一轮我投4号。",
                "player6 | vote_intent | player4",
                [["player6", "vote_intent", "player4"]],
            ),
            (
                "我不信4号，这一轮我倾向投4号。",
                (
                    "player6 | oppose | player4\n"
                    "player6 | vote_intent | player4"
                ),
                [
                    ["player6", "oppose", "player4"],
                    ["player6", "vote_intent", "player4"],
                ],
            ),
            (
                "我不信4号，这一轮我投4号。",
                (
                    "player6 | oppose | player4\n"
                    "player6 | vote_intent | player4"
                ),
                [
                    ["player6", "oppose", "player4"],
                    ["player6", "vote_intent", "player4"],
                ],
            ),
            (
                "我觉得4号是狼，这一轮我倾向投4号。",
                (
                    "player6 | point_as_werewolf | player4\n"
                    "player6 | vote_intent | player4"
                ),
                [
                    ["player6", "point_as_werewolf", "player4"],
                    ["player6", "vote_intent", "player4"],
                ],
            ),
        )

        for speech, response, expected in cases:
            with self.subTest(speech=speech):
                actions, prompt = self.parse_with_response(
                    speaker=6,
                    speech=speech,
                    response=response,
                )
                self.assertEqual(actions, expected)
                self.assertIn(
                    "vote_intent不等于环境实际vote，也不自动产生oppose",
                    prompt,
                )

    def test_specific_claims_coexist_only_with_independent_propositions(
        self,
    ):
        cases = (
            (
                1,
                "我觉得3号是狼。",
                "player1 | point_as_werewolf | player3",
                [["player1", "point_as_werewolf", "player3"]],
            ),
            (
                1,
                "我验3号查杀，今天投3号。",
                (
                    "player1 | check_as_werewolf | player3\n"
                    "player1 | vote_intent | player3"
                ),
                [
                    ["player1", "check_as_werewolf", "player3"],
                    ["player1", "vote_intent", "player3"],
                ],
            ),
            (
                1,
                "我是预言家，验3号查杀。",
                (
                    "player1 | point_as_seer | player1\n"
                    "player1 | check_as_werewolf | player3"
                ),
                [
                    ["player1", "point_as_seer", "player1"],
                    ["player1", "check_as_werewolf", "player3"],
                ],
            ),
            (
                1,
                "3号是好人，但为了统一票型今天投3号。",
                "player1 | vote_intent | player3",
                [["player1", "vote_intent", "player3"]],
            ),
            (
                1,
                "我验3号金水，也同意他的逻辑。",
                (
                    "player1 | check_as_good | player3\n"
                    "player1 | support | player3"
                ),
                [
                    ["player1", "check_as_good", "player3"],
                    ["player1", "support", "player3"],
                ],
            ),
        )

        for speaker, speech, response, expected in cases:
            with self.subTest(speech=speech):
                actions, _ = self.parse_with_response(
                    speaker=speaker,
                    speech=speech,
                    response=response,
                )
                self.assertEqual(actions, expected)

    def test_pure_report_without_endorsement_is_not_a_stance(
        self,
    ):
        actions, prompt = self.parse_with_response(
            speaker=6,
            speech=(
                "4号刚才说自己是村民。"
            ),
            response="NONE",
        )

        self.assertEqual(
            actions,
            [],
        )
        self.assertIn(
            "转述别人的身份声明或立场",
            prompt,
        )

    def test_witch_smoke_failure_texts_follow_primary_ontology(self):
        cases = (
            (
                5,
                "我今天的投票意向是放逐 player3。同时，我公开判断 player4 是村民。",
                (
                    "player5 | vote_intent | player3\n"
                    "player5 | point_as_villager | player4"
                ),
                [
                    ["player5", "vote_intent", "player3"],
                    ["player5", "point_as_villager", "player4"],
                ],
            ),
            (
                6,
                "我认为 player1 是村民。查验 player5 的结果是好人。",
                (
                    "player6 | point_as_villager | player1\n"
                    "player6 | check_as_good | player5"
                ),
                [
                    ["player6", "point_as_villager", "player1"],
                    ["player6", "check_as_good", "player5"],
                ],
            ),
            (
                4,
                "我认为 player1 是村民，player2 也是村民。同时，我判断 player3 是村民，但 player3 也是女巫。",
                (
                    "player4 | point_as_villager | player1\n"
                    "player4 | point_as_villager | player2\n"
                    "player4 | point_as_villager | player3\n"
                    "player4 | point_as_witch | player3"
                ),
                [
                    ["player4", "point_as_villager", "player1"],
                    ["player4", "point_as_villager", "player2"],
                    ["player4", "point_as_villager", "player3"],
                    ["player4", "point_as_witch", "player3"],
                ],
            ),
            (
                7,
                "经过查验，player4 是预言家，而查验 player2 的结果为好人。",
                (
                    "player7 | point_as_seer | player4\n"
                    "player7 | check_as_good | player2"
                ),
                [
                    ["player7", "point_as_seer", "player4"],
                    ["player7", "check_as_good", "player2"],
                ],
            ),
        )

        for speaker, speech, response, expected in cases:
            with self.subTest(speech=speech):
                actions, prompt = self.parse_with_response(
                    speaker=speaker,
                    speech=speech,
                    response=response,
                )
                self.assertEqual(actions, expected)
                self.assertIn("穷尽抽取所有明确属于上述可表示类别的命题", prompt)
                self.assertIn(
                    "vote_intent不等于环境实际vote，也不自动产生oppose",
                    prompt,
                )
                self.assertIn("查验结果单独存在时不得产生 support", prompt)
                self.assertIn("不得产生 point_as_villager", prompt)
                self.assertIn("player4 是预言家", prompt)


if __name__ == "__main__":
    unittest.main()
