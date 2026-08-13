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

    def test_vote_intent_is_excluded_without_erasing_formal_actions(
        self,
    ):
        cases = (
            (
                "这一轮我倾向投4号。",
                "NONE",
                [],
            ),
            (
                "这一轮我投4号。",
                "NONE",
                [],
            ),
            (
                "我不信4号，这一轮我倾向投4号。",
                "player6 | oppose | player4",
                [["player6", "oppose", "player4"]],
            ),
            (
                "我不信4号，这一轮我投4号。",
                "player6 | oppose | player4",
                [["player6", "oppose", "player4"]],
            ),
            (
                "我觉得4号是狼，这一轮我倾向投4号。",
                "player6 | point_as_werewolf | player4",
                [["player6", "point_as_werewolf", "player4"]],
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
                    "投票意图不属于动作，也不等于 support 或 oppose",
                    prompt,
                )

    def test_non_primary_claims_do_not_erase_formal_actions(
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
                [],
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
                ],
            ),
            (
                1,
                "3号是好人，但为了统一票型今天投3号。",
                "NONE",
                [],
            ),
            (
                1,
                "我验3号金水，也同意他的逻辑。",
                (
                    "player1 | check_as_good | player3\n"
                    "player1 | support | player3"
                ),
                [
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


if __name__ == "__main__":
    unittest.main()
