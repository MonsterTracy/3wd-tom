import json
import unittest

from werewolf.models import SpeechPerceiver


class FakeBackend:
    def __init__(
        self,
        content=None,
        error=None,
    ):
        self.content = content
        self.error = error
        self.calls = []

    def chat(
        self,
        messages,
        model=None,
        temperature=0.7,
        max_tokens=None,
        response_format=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": response_format,
                **kwargs,
            }
        )

        if self.error is not None:
            raise self.error

        return self.content


class SpeechPerceiverTest(unittest.TestCase):
    def test_calls_backend_and_returns_onuw_pipe_actions(
        self,
    ):
        speech = "我是预言家，3号是狼人。"
        backend = FakeBackend(
            "\n".join(
                [
                    "player7 | point_as_seer | player2",
                    "player7 | point_as_werewolf | player3",
                ]
            )
        )
        perceiver = SpeechPerceiver(
            backend=backend,
            model_name="test-model",
        )

        actions = perceiver.parse(
            speaker=2,
            speech=speech,
            day=1,
            phase="speech",
        )

        self.assertEqual(
            actions,
            [
                [
                    "player2",
                    "point_as_seer",
                    "player2",
                ],
                [
                    "player2",
                    "point_as_werewolf",
                    "player3",
                ],
            ],
        )
        self.assertEqual(
            len(backend.calls),
            1,
        )

        call = backend.calls[0]
        self.assertEqual(
            call["model"],
            "test-model",
        )
        self.assertEqual(
            call["temperature"],
            0,
        )
        self.assertEqual(
            call["messages"][0]["role"],
            "user",
        )

        prompt = call["messages"][0]["content"]
        self.assertIn(
            "当前发言者：player2",
            prompt,
        )
        self.assertIn(
            "subject | action | object",
            prompt,
        )
        self.assertIn(
            "每个动作单独一行",
            prompt,
        )
        self.assertIn(
            "没有可抽取动作时，只输出：NONE",
            prompt,
        )
        self.assertIn(
            "point_as_werewolf",
            prompt,
        )
        self.assertIn(
            "point_as_guard",
            prompt,
        )
        self.assertIn(
            "support",
            prompt,
        )
        self.assertIn(
            "oppose",
            prompt,
        )
        self.assertIn(
            "好人”“金水”或“非狼",
            prompt,
        )
        self.assertIn(
            "可以同时输出 point_as_* 与 support/oppose",
            prompt,
        )
        self.assertIn(
            "action 不同的动作",
            prompt,
        )
        self.assertIn(
            speech,
            prompt,
        )
        self.assertNotIn(
            "claim_camp",
            prompt,
        )
        self.assertNotIn(
            "certainty",
            prompt,
        )

    def test_preserves_distinct_actions_for_same_object(
        self,
    ):
        backend = FakeBackend(
            "\n".join(
                [
                    "player2 | point_as_seer | player1",
                    "player2 | support | player1",
                ]
            )
        )
        perceiver = SpeechPerceiver(
            backend=backend,
            model_name="test-model",
        )

        actions = perceiver.parse(
            speaker=2,
            speech=(
                "1号大概率是真的预言家，"
                "我建议先相信1号。"
            ),
            day=1,
            phase="speech",
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

    def test_preserves_literal_self_claim_when_backend_returns_none(
        self,
    ):
        backend = FakeBackend("NONE")
        perceiver = SpeechPerceiver(
            backend=backend,
            model_name="test-model",
        )

        actions = perceiver.parse(
            speaker=6,
            speech=(
                "我是6号玩家，身份是村民。"
                "目前信息不足，我暂时没有明确怀疑对象。"
            ),
            day=1,
            phase="speech",
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

        prompt = backend.calls[0][
            "messages"
        ][0]["content"]

        self.assertIn(
            "第一人称明确自报具体身份属于受保护动作",
            prompt,
        )
        self.assertIn(
            "也不能省略已经出现的自报身份动作",
            prompt,
        )

    def test_merges_protected_self_claim_with_other_llm_actions(
        self,
    ):
        backend = FakeBackend(
            "player6 | support | player3"
        )
        perceiver = SpeechPerceiver(
            backend=backend,
            model_name="test-model",
        )

        actions = perceiver.parse(
            speaker=6,
            speech=(
                "我是6号玩家，身份是村民。"
                "我比较认可3号。"
            ),
            day=1,
            phase="speech",
        )

        self.assertEqual(
            actions,
            [
                [
                    "player6",
                    "point_as_villager",
                    "player6",
                ],
                [
                    "player6",
                    "support",
                    "player3",
                ],
            ],
        )

    def test_deduplicates_self_claim_returned_by_backend(
        self,
    ):
        backend = FakeBackend(
            "player6 | point_as_villager | player6"
        )
        perceiver = SpeechPerceiver(
            backend=backend,
            model_name="test-model",
        )

        actions = perceiver.parse(
            speaker=6,
            speech="我是6号玩家，身份是村民。",
            day=1,
            phase="speech",
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

    def test_backend_failure_does_not_drop_literal_self_claim(
        self,
    ):
        backend = FakeBackend(
            error=RuntimeError(
                "backend unavailable"
            )
        )
        perceiver = SpeechPerceiver(
            backend=backend,
            model_name="test-model",
        )

        actions = perceiver.parse(
            speaker=6,
            speech="我是6号玩家，身份是村民。",
            day=1,
            phase="speech",
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

    def test_does_not_protect_non_specific_or_conditional_claims(
        self,
    ):
        for speech in (
            "我是好人。",
            "我不是狼人。",
            "如果我是预言家，我会先验2号。",
            "4号说我是村民。",
        ):
            with self.subTest(
                speech=speech,
            ):
                backend = FakeBackend("NONE")
                perceiver = SpeechPerceiver(
                    backend=backend,
                    model_name="test-model",
                )

                self.assertEqual(
                    perceiver.parse(
                        speaker=6,
                        speech=speech,
                        day=1,
                        phase="speech",
                    ),
                    [],
                )

    def test_parses_pipe_code_fence_bullets_and_fullwidth_separator(
        self,
    ):
        backend = FakeBackend(
            """```text
- player1 | oppose | player2
2. player1｜support｜player3。
```"""
        )
        perceiver = SpeechPerceiver(
            backend=backend,
            model_name="test-model",
        )

        actions = perceiver.parse(
            5,
            "我关注2号，也比较相信3号。",
            1,
            "speech",
        )

        self.assertEqual(
            actions,
            [
                [
                    "player5",
                    "oppose",
                    "player2",
                ],
                [
                    "player5",
                    "support",
                    "player3",
                ],
            ],
        )

    def test_keeps_legacy_json_array_compatibility(
        self,
    ):
        backend = FakeBackend(
            """```json
[
  ["player1", "point_as_seer", "player1"]
]
```"""
        )
        perceiver = SpeechPerceiver(
            backend=backend,
            model_name="test-model",
        )

        actions = perceiver.parse(
            1,
            "我是预言家",
            1,
            "speech",
        )

        self.assertEqual(
            actions,
            [
                [
                    "player1",
                    "point_as_seer",
                    "player1",
                ]
            ],
        )

    def test_extracts_first_legacy_json_array_from_text(
        self,
    ):
        backend = FakeBackend(
            (
                "解析结果："
                '[["player2","support","player4"]]。'
                "备选："
                '[["player2","oppose","player5"]]'
            )
        )
        perceiver = SpeechPerceiver(
            backend=backend,
            model_name="test-model",
        )

        actions = perceiver.parse(
            2,
            "我支持4号",
            1,
            "speech",
        )

        self.assertEqual(
            actions,
            [
                [
                    "player2",
                    "support",
                    "player4",
                ]
            ],
        )

    def test_filters_invalid_actions_and_duplicates(
        self,
    ):
        backend = FakeBackend(
            json.dumps(
                [
                    [
                        "player5",
                        "invented_action",
                        "player1",
                    ],
                    [
                        "player5",
                        "support",
                        "player8",
                    ],
                    [
                        "player7",
                        "oppose",
                        "player3",
                    ],
                    [
                        "player5",
                        "oppose",
                        "player3",
                    ],
                    {
                        "subject": "player1",
                        "action": "point_as_werewolf",
                        "object": "player4",
                    },
                    {
                        "predicate": "suspect",
                        "target": 6,
                    },
                ],
                ensure_ascii=False,
            )
        )
        perceiver = SpeechPerceiver(
            backend=backend,
            model_name="test-model",
        )

        actions = perceiver.parse(
            5,
            "我不信3号，我认为4号是狼人。",
            2,
            "speech_pk",
        )

        self.assertEqual(
            actions,
            [
                [
                    "player5",
                    "oppose",
                    "player3",
                ],
                [
                    "player5",
                    "point_as_werewolf",
                    "player4",
                ],
            ],
        )

    def test_none_is_a_valid_empty_result(
        self,
    ):
        backend = FakeBackend("NONE")
        perceiver = SpeechPerceiver(
            backend=backend,
            model_name="test-model",
        )

        self.assertEqual(
            perceiver.parse(
                2,
                "我还要继续听。",
                1,
                "speech",
            ),
            [],
        )

    def test_returns_empty_without_backend_or_model(
        self,
    ):
        self.assertEqual(
            SpeechPerceiver(
                model_name="test-model"
            ).parse(
                1,
                "发言",
                1,
                "speech",
            ),
            [],
        )

        backend = FakeBackend("NONE")
        self.assertEqual(
            SpeechPerceiver(
                backend=backend
            ).parse(
                1,
                "发言",
                1,
                "speech",
            ),
            [],
        )
        self.assertEqual(
            backend.calls,
            [],
        )

    def test_returns_empty_when_backend_raises(
        self,
    ):
        backend = FakeBackend(
            error=RuntimeError(
                "backend unavailable"
            )
        )
        perceiver = SpeechPerceiver(
            backend=backend,
            model_name="test-model",
        )

        self.assertEqual(
            perceiver.parse(
                1,
                "发言",
                1,
                "speech",
            ),
            [],
        )

    def test_returns_empty_for_malformed_response(
        self,
    ):
        backend = FakeBackend(
            "not a structured response"
        )
        perceiver = SpeechPerceiver(
            backend=backend,
            model_name="test-model",
        )

        self.assertEqual(
            perceiver.parse(
                1,
                "发言",
                1,
                "speech",
            ),
            [],
        )

    def test_rejects_invalid_speaker(
        self,
    ):
        backend = FakeBackend("NONE")
        perceiver = SpeechPerceiver(
            backend=backend,
            model_name="test-model",
        )

        self.assertEqual(
            perceiver.parse(
                0,
                "发言",
                1,
                "speech",
            ),
            [],
        )
        self.assertEqual(
            backend.calls,
            [],
        )

    def test_rejects_empty_speech_without_calling_backend(
        self,
    ):
        backend = FakeBackend("NONE")
        perceiver = SpeechPerceiver(
            backend=backend,
            model_name="test-model",
        )

        self.assertEqual(
            perceiver.parse(
                1,
                " ",
                1,
                "speech",
            ),
            [],
        )
        self.assertEqual(
            backend.calls,
            [],
        )


if __name__ == "__main__":
    unittest.main()
