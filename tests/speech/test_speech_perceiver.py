import json
import unittest

from werewolf.models import SpeechPerceiver
from werewolf.speech.speech_perceiver import (
    SPEECH_PARSER_MAX_TOKENS,
    SpeechActionValidationError,
)
from werewolf.models.twd_tom.schema import SpeechAction, normalize_player


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
            call["max_tokens"],
            SPEECH_PARSER_MAX_TOKENS,
        )
        self.assertIsNone(
            call["response_format"],
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
            "每一个非空行都必须符合上述协议",
            prompt,
        )
        self.assertIn(
            "最多输出7个动作",
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
        for action_name in (
            "check_as_good",
            "check_as_werewolf",
            "save",
            "poison",
            "guard",
            "vote_intent",
        ):
            self.assertIn(action_name, prompt)
        for unsupported_alias in (
            "check_good",
            "checked_good",
            "heal",
            "protected",
            "intend_vote",
        ):
            self.assertNotIn(unsupported_alias, prompt)
        self.assertIn(
            "独立命题共存",
            prompt,
        )
        self.assertIn(
            "action 不同的动作",
            prompt,
        )
        self.assertIn(
            "不得把“player2 至 player4”写入 object",
            prompt,
        )
        self.assertIn(
            "player2 | oppose | player2",
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

    def test_prompt_freezes_a1_semantic_modules_and_non_redundancy(
        self,
    ):
        perceiver = SpeechPerceiver(
            backend=FakeBackend("NONE"),
            model_name="test-model",
        )

        perceiver.parse(1, "公开发言", 1, "speech")
        prompt = perceiver.backend.calls[0]["messages"][0]["content"]

        for module_name in (
            "ROLE_ESTIMATE",
            "SOCIAL_STANCE",
            "CLAIMED_SKILL_REPORT",
            "ACTION_INTENT",
        ):
            self.assertIn(module_name, prompt)
        self.assertIn(
            "A1：最具体、非冗余、显式原子命题编码",
            prompt,
        )
        for principle in (
            "显式性",
            "最具体性",
            "非冗余性",
            "独立命题共存",
        ):
            self.assertIn(principle, prompt)
        self.assertIn(
            "明确认可、同意或支持 target 的发言、逻辑、主张或可信度",
            prompt,
        )
        self.assertIn(
            "明确反对、不认可、不信任或批评 target 的发言、逻辑、主张或可信度",
            prompt,
        )
        for forbidden_expansion in (
            "vote_intent → oppose",
            "point_as_werewolf → oppose",
            "point_as_villager → support",
            "check_as_werewolf → point_as_werewolf 或 oppose",
            "check_as_good → point_as_villager 或 support",
            "poison → oppose",
            "save → support",
            "guard → support",
        ):
            self.assertIn(forbidden_expansion, prompt)
        self.assertIn(
            "投票偏好本身也不等于 support 或 oppose",
            prompt,
        )
        self.assertNotIn("倾向投票/放逐", prompt)

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

    def test_canonicalizes_only_a1_check_expansions(
        self,
    ):
        actions = SpeechPerceiver._canonicalize_a1_actions(
            [
                ["player2", "point_as_villager", "player3"],
                ["player2", "check_as_good", "player3"],
                ["player2", "point_as_werewolf", "player4"],
                ["player2", "check_as_werewolf", "player4"],
                ["player2", "oppose", "player4"],
                ["player2", "vote_intent", "player4"],
            ]
        )

        self.assertEqual(
            actions,
            [
                ["player2", "check_as_good", "player3"],
                ["player2", "check_as_werewolf", "player4"],
                ["player2", "oppose", "player4"],
                ["player2", "vote_intent", "player4"],
            ],
        )

    def test_parse_canonicalizes_redundant_a1_backend_output(
        self,
    ):
        backend = FakeBackend(
            "\n".join(
                [
                    "player2 | check_as_good | player3",
                    "player2 | point_as_villager | player3",
                    "player2 | check_as_werewolf | player4",
                    "player2 | point_as_werewolf | player4",
                ]
            )
        )
        perceiver = SpeechPerceiver(
            backend=backend,
            model_name="test-model",
        )

        self.assertEqual(
            perceiver.parse(
                speaker=2,
                speech=(
                    "我查验3号为好人；经过查验，"
                    "我确认4号是狼人。"
                ),
                day=1,
                phase="speech",
            ),
            [
                ["player2", "check_as_good", "player3"],
                ["player2", "check_as_werewolf", "player4"],
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

    def test_existing_specific_self_claim_actions_remain_unchanged(
        self,
    ):
        expected_actions = {
            "村民": "point_as_villager",
            "预言家": "point_as_seer",
            "女巫": "point_as_witch",
            "守卫": "point_as_guard",
        }

        for role, action_name in expected_actions.items():
            with self.subTest(role=role):
                perceiver = SpeechPerceiver(
                    backend=FakeBackend("NONE"),
                    model_name="test-model",
                )

                self.assertEqual(
                    perceiver.parse(
                        speaker=2,
                        speech=f"我是{role}。",
                        day=1,
                        phase="speech",
                    ),
                    [["player2", action_name, "player2"]],
                )

    def test_vote_intent_prompt_requires_current_unconditional_self_commitment(
        self,
    ):
        perceiver = SpeechPerceiver(
            backend=FakeBackend("NONE"),
            model_name="test-model",
        )

        perceiver.parse(
            speaker=1,
            speech="公开发言",
            day=1,
            phase="speech",
        )
        prompt = perceiver.backend.calls[0]["messages"][0]["content"]

        self.assertIn(
            "对当前待执行投票作出的无条件、明确、自身投票承诺",
            prompt,
        )
        for accepted in (
            "今天我投3号",
            "我的票挂3号",
            "这一轮我会投3号",
        ):
            self.assertIn(accepted, prompt)
        for rejected_category in (
            "条件承诺",
            "可能性表达",
            "未来其他轮次的计划",
            "请求他人投票",
            "转述他人投票意图",
            "已完成的投票",
            "实际投票系统事件",
        ):
            self.assertIn(rejected_category, prompt)
        for rejected_example in (
            "如果3号不解释，我就投他",
            "我可能投3号",
            "明天再考虑投3号",
            "大家投3号",
            "2号说他会投3号",
            "我已经投了3号",
        ):
            self.assertIn(rejected_example, prompt)
        self.assertNotIn("所有未来计划", prompt)

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

    def test_protected_self_role_survives_a1_canonicalization(
        self,
    ):
        backend = FakeBackend(
            "\n".join(
                [
                    "player1 | check_as_good | player1",
                    "player1 | point_as_villager | player1",
                ]
            )
        )
        perceiver = SpeechPerceiver(
            backend=backend,
            model_name="test-model",
        )

        self.assertEqual(
            perceiver.parse(
                speaker=1,
                speech="我是 1 号村民，我查验自己确认为好人。",
                day=1,
                phase="speech",
            ),
            [
                ["player1", "point_as_villager", "player1"],
                ["player1", "check_as_good", "player1"],
            ],
        )

    def test_protected_self_role_accepts_seat_spacing_variants(
        self,
    ):
        for speech in (
            "我是2号村民",
            "我是 2号村民",
            "我是2号 村民",
            "我是 2 号村民",
        ):
            with self.subTest(speech=speech):
                perceiver = SpeechPerceiver(
                    backend=FakeBackend("NONE"),
                    model_name="test-model",
                )

                self.assertEqual(
                    perceiver.parse(
                        speaker=2,
                        speech=speech,
                        day=1,
                        phase="speech",
                    ),
                    [
                        [
                            "player2",
                            "point_as_villager",
                            "player2",
                        ]
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

    def test_strict_accepts_extended_actions_in_pipe_and_complete_json(
        self,
    ):
        expected = [
            ["player1", "check_as_good", "player2"],
            ["player1", "check_as_werewolf", "player3"],
            ["player1", "save", "player4"],
            ["player1", "poison", "player5"],
            ["player1", "guard", "player6"],
            ["player1", "vote_intent", "player7"],
        ]
        responses = [
            "\n".join(" | ".join(action) for action in expected),
            json.dumps(expected),
        ]

        for response in responses:
            with self.subTest(response=response):
                perceiver = SpeechPerceiver(
                    backend=FakeBackend(response),
                    model_name="test-model",
                )

                self.assertEqual(
                    perceiver.parse_strict(
                        1,
                        "公开发言",
                        1,
                        "speech",
                    ),
                    expected,
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

    def test_v27_player_range_expands_to_atomic_actions(self):
        perceiver = SpeechPerceiver(
            backend=FakeBackend(
                "player1 | oppose | player2 至 player7"
            ),
            model_name="test-model",
        )

        self.assertEqual(
            perceiver.parse(1, "player2 至 player7 都有问题", 1, "speech"),
            [
                ["player1", "oppose", f"player{player_id}"]
                for player_id in range(2, 8)
            ],
        )

    def test_range_expansion_reuses_atomic_dedup_and_keeps_other_actions(self):
        perceiver = SpeechPerceiver(
            backend=FakeBackend(
                "\n".join(
                    [
                        "player1 | oppose | player2 至 player4",
                        "player1 | oppose | player3",
                        "player1 | support | player3",
                        "player1 | oppose | player5",
                    ]
                )
            ),
            model_name="test-model",
        )

        self.assertEqual(
            perceiver.parse(1, "发言", 1, "speech"),
            [
                ["player1", "oppose", "player2"],
                ["player1", "oppose", "player3"],
                ["player1", "oppose", "player4"],
                ["player1", "support", "player3"],
                ["player1", "oppose", "player5"],
            ],
        )

    def test_single_player_object_remains_compatible(self):
        perceiver = SpeechPerceiver(
            backend=FakeBackend("player1 | oppose | player3"),
            model_name="test-model",
        )
        self.assertEqual(
            perceiver.parse(1, "我反对3号", 1, "speech"),
            [["player1", "oppose", "player3"]],
        )

    def test_invalid_or_malformed_ranges_fail_closed(self):
        for object_value in (
            "player0 至 player3",
            "player3 至 player8",
            "player4 至 player2",
            "playerX 至 player4",
        ):
            with self.subTest(object_value=object_value):
                perceiver = SpeechPerceiver(
                    backend=FakeBackend(
                        f"player1 | oppose | {object_value}"
                    ),
                    model_name="test-model",
                )
                self.assertEqual(
                    perceiver.parse(1, "发言", 1, "speech"),
                    [],
                )
                with self.assertRaises(
                    (SpeechActionValidationError, ValueError)
                ):
                    perceiver.parse_strict(1, "发言", 1, "speech")

    def test_atomic_player_validators_still_reject_range_strings(self):
        with self.assertRaises(ValueError):
            normalize_player("player2 至 player4")
        with self.assertRaises(ValueError):
            SpeechAction.from_values(
                "player1",
                "oppose",
                "player2 至 player4",
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

    def test_strict_parse_exposes_backend_failure(
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

        with self.assertRaisesRegex(
            RuntimeError,
            "backend unavailable",
        ):
            perceiver.parse_strict(
                1,
                "发言",
                1,
                "speech",
            )

        self.assertEqual(len(backend.calls), 1)

    def test_strict_with_response_preserves_pipe_text(
        self,
    ):
        raw_response = (
            "\nplayer1 | vote_intent | player3\n"
        )
        backend = FakeBackend(raw_response)
        perceiver = SpeechPerceiver(
            backend=backend,
            model_name="test-model",
        )

        actions, returned_response = (
            perceiver.parse_strict_with_response(
                1,
                "发言",
                1,
                "speech",
            )
        )

        self.assertEqual(
            actions,
            [["player1", "vote_intent", "player3"]],
        )
        self.assertEqual(returned_response, raw_response)
        self.assertEqual(len(backend.calls), 1)

    def test_strict_with_response_preserves_json_text(
        self,
    ):
        raw_response = (
            '[ [ "player1", "support", "player2" ] ]'
        )
        backend = FakeBackend(raw_response)
        perceiver = SpeechPerceiver(
            backend=backend,
            model_name="test-model",
        )

        actions, returned_response = (
            perceiver.parse_strict_with_response(
                1,
                "发言",
                1,
                "speech",
            )
        )

        self.assertEqual(
            actions,
            [["player1", "support", "player2"]],
        )
        self.assertEqual(returned_response, raw_response)
        self.assertEqual(len(backend.calls), 1)

    def test_strict_with_response_preserves_none_text(
        self,
    ):
        raw_response = "NONE"
        backend = FakeBackend(raw_response)
        perceiver = SpeechPerceiver(
            backend=backend,
            model_name="test-model",
        )

        actions, returned_response = (
            perceiver.parse_strict_with_response(
                1,
                "发言",
                1,
                "speech",
            )
        )

        self.assertEqual(actions, [])
        self.assertEqual(returned_response, raw_response)
        self.assertEqual(len(backend.calls), 1)

    def test_strict_parse_rejects_malformed_response(
        self,
    ):
        backend = FakeBackend(
            "not a structured response"
        )
        perceiver = SpeechPerceiver(
            backend=backend,
            model_name="test-model",
        )

        with self.assertRaisesRegex(
            ValueError,
            "No structured speech action",
        ):
            perceiver.parse_strict(
                1,
                "发言",
                1,
                "speech",
            )

    def test_strict_rejects_unknown_action_without_changing_parse(
        self,
    ):
        backend = FakeBackend(
            "player1 | invented_action | player2"
        )
        perceiver = SpeechPerceiver(
            backend=backend,
            model_name="test-model",
        )

        self.assertEqual(
            perceiver.parse(1, "发言", 1, "speech"),
            [],
        )

        with self.assertRaises(
            SpeechActionValidationError
        ) as caught:
            perceiver.parse_strict(
                1,
                "发言",
                1,
                "speech",
            )

        self.assertEqual(caught.exception.invalid_count, 1)
        self.assertEqual(
            caught.exception.failures[0]["candidate"],
            ["player1", "invented_action", "player2"],
        )
        self.assertIn(
            "unsupported speech action",
            caught.exception.failures[0]["reason"],
        )

    def test_strict_with_response_attaches_invalid_raw_text(
        self,
    ):
        raw_response = "player1 | invented_action | player2"
        backend = FakeBackend(raw_response)
        perceiver = SpeechPerceiver(
            backend=backend,
            model_name="test-model",
        )

        with self.assertRaises(
            SpeechActionValidationError
        ) as caught:
            perceiver.parse_strict_with_response(
                1,
                "发言",
                1,
                "speech",
            )

        self.assertEqual(
            caught.exception.raw_response,
            raw_response,
        )
        self.assertEqual(len(backend.calls), 1)

    def test_strict_rejects_invalid_player_without_changing_parse(
        self,
    ):
        backend = FakeBackend(
            "player8 | support | player2"
        )
        perceiver = SpeechPerceiver(
            backend=backend,
            model_name="test-model",
        )

        self.assertEqual(
            perceiver.parse(1, "发言", 1, "speech"),
            [],
        )

        with self.assertRaisesRegex(
            SpeechActionValidationError,
            "player8",
        ):
            perceiver.parse_strict(
                1,
                "发言",
                1,
                "speech",
            )

    def test_strict_rejects_malformed_candidate_without_changing_parse(
        self,
    ):
        responses = [
            (
                "player1 | support",
                "pipe triplet protocol",
            ),
            (
                "player1 | support | player2 | extra",
                "pipe triplet protocol",
            ),
            (
                json.dumps(
                    [
                        {
                            "subject": "player1",
                            "action": "support",
                        }
                    ]
                ),
                "three-item sequence",
            ),
        ]

        for response, error_pattern in responses:
            with self.subTest(response=response):
                perceiver = SpeechPerceiver(
                    backend=FakeBackend(response),
                    model_name="test-model",
                )

                self.assertEqual(
                    perceiver.parse(1, "发言", 1, "speech"),
                    [],
                )

                with self.assertRaisesRegex(
                    SpeechActionValidationError,
                    error_pattern,
                ):
                    perceiver.parse_strict(
                        1,
                        "发言",
                        1,
                        "speech",
                    )

    def test_strict_rejects_entire_mixed_valid_invalid_output(
        self,
    ):
        backend = FakeBackend(
            "\n".join(
                [
                    "player1 | support | player2",
                    "player1 | invented_action | player3",
                ]
            )
        )
        perceiver = SpeechPerceiver(
            backend=backend,
            model_name="test-model",
        )

        self.assertEqual(
            perceiver.parse(1, "发言", 1, "speech"),
            [["player1", "support", "player2"]],
        )

        with self.assertRaises(
            SpeechActionValidationError
        ) as caught:
            perceiver.parse_strict(
                1,
                "发言",
                1,
                "speech",
            )

        self.assertEqual(caught.exception.invalid_count, 1)

    def test_strict_none_is_a_valid_empty_result(
        self,
    ):
        backend = FakeBackend("NONE")
        perceiver = SpeechPerceiver(
            backend=backend,
            model_name="test-model",
        )

        self.assertEqual(
            perceiver.parse_strict(
                1,
                "发言",
                1,
                "speech",
            ),
            [],
        )
        self.assertEqual(len(backend.calls), 1)

    def test_strict_accepts_one_triplet_with_blank_lines(
        self,
    ):
        perceiver = SpeechPerceiver(
            backend=FakeBackend(
                "\n\t\nplayer1 | support | player2\n  \n"
            ),
            model_name="test-model",
        )

        self.assertEqual(
            perceiver.parse_strict(
                1,
                "发言",
                1,
                "speech",
            ),
            [["player1", "support", "player2"]],
        )

    def test_strict_rejects_extra_non_protocol_lines(
        self,
    ):
        valid_action = "player1 | support | player2"
        responses = [
            (
                f"{valid_action}\n解释：这是抽取原因",
                "解释：这是抽取原因",
            ),
            (
                f"以下是结果：\n{valid_action}",
                "以下是结果：",
            ),
            (
                f"```\n{valid_action}\n```",
                "```",
            ),
            (
                f"- {valid_action}",
                "- player1",
            ),
        ]

        for response, invalid_line in responses:
            with self.subTest(response=response):
                perceiver = SpeechPerceiver(
                    backend=FakeBackend(response),
                    model_name="test-model",
                )

                self.assertEqual(
                    perceiver.parse(1, "发言", 1, "speech"),
                    [["player1", "support", "player2"]],
                )

                with self.assertRaises(
                    SpeechActionValidationError
                ) as caught:
                    perceiver.parse_strict(
                        1,
                        "发言",
                        1,
                        "speech",
                    )

                self.assertIn(
                    invalid_line,
                    str(caught.exception.failures),
                )

    def test_strict_rejects_none_mixed_with_other_content(
        self,
    ):
        responses = [
            "NONE\nplayer1 | support | player2",
            "NONE\n解释：没有动作",
        ]

        for response in responses:
            with self.subTest(response=response):
                perceiver = SpeechPerceiver(
                    backend=FakeBackend(response),
                    model_name="test-model",
                )

                with self.assertRaises(
                    SpeechActionValidationError
                ):
                    perceiver.parse_strict(
                        1,
                        "发言",
                        1,
                        "speech",
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

    def test_overlong_garbage_fails_closed_without_retry(self):
        backend = FakeBackend(
            "not-an-action " * 10000
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
        self.assertEqual(len(backend.calls), 1)

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
