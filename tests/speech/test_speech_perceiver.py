import unittest

from werewolf.models import SpeechPerceiver
from werewolf.speech.speech_perceiver import (
    SPEECH_PARSER_GENERATION_MAX_ATTEMPTS,
    SPEECH_PARSER_MAX_TOKENS,
    SpeechActionValidationError,
)


class FakeBackend:
    def __init__(self, content=None, error=None):
        self.content = content
        self.contents = list(content) if isinstance(content, tuple) else None
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
        if self.contents is not None:
            return self.contents.pop(0)
        return self.content


class SpeechPerceiverTest(unittest.TestCase):
    def test_calls_backend_and_returns_exact_pipe_actions(self):
        speech = "我是预言家，3号是狼人。"
        backend = FakeBackend(
            "player2 | point_as_seer | player2\n"
            "player2 | point_as_werewolf | player3"
        )
        perceiver = SpeechPerceiver(backend=backend, model_name="test-model")

        self.assertEqual(
            perceiver.parse(2, speech, 1, "speech"),
            [
                ["player2", "point_as_seer", "player2"],
                ["player2", "point_as_werewolf", "player3"],
            ],
        )

        self.assertEqual(len(backend.calls), 1)
        call = backend.calls[0]
        self.assertEqual(call["model"], "test-model")
        self.assertEqual(call["temperature"], 0)
        self.assertEqual(call["max_tokens"], SPEECH_PARSER_MAX_TOKENS)
        self.assertIsNone(call["response_format"])
        self.assertEqual(
            call["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )

        prompt = call["messages"][0]["content"]
        self.assertIn("当前发言者：player2", prompt)
        self.assertIn("subject | action | object", prompt)
        self.assertIn("每个动作单独一行", prompt)
        self.assertIn("没有可抽取动作时，只输出：NONE", prompt)
        self.assertIn("不输出 JSON、解释或 Markdown 代码块", prompt)
        self.assertIn(speech, prompt)

    def test_prompt_freezes_formal_action_semantics(self):
        backend = FakeBackend("NONE")
        perceiver = SpeechPerceiver(backend=backend, model_name="test-model")
        perceiver.parse(1, "公开发言", 1, "speech")
        prompt = backend.calls[0]["messages"][0]["content"]

        for action_name in (
            "point_as_werewolf",
            "point_as_non_werewolf",
            "point_as_villager",
            "point_as_seer",
            "point_as_witch",
            "support",
            "oppose",
            "check_as_non_werewolf",
            "check_as_werewolf",
            "save",
            "poison",
            "vote_intent",
            "abstain_intent",
            "no_commitment",
        ):
            self.assertIn(f"- {action_name}", prompt)
        self.assertNotIn("point_as_guard", prompt)
        self.assertIn("most-specific-source", prompt)
        self.assertIn("不得读取真实角色或环境技能记录", prompt)
        self.assertIn("vote_intent不等于环境实际vote", prompt)
        self.assertIn(
            "泛化的“站边好人阵营”“支持好人阵营”“维护好人阵营”",
            prompt,
        )
        self.assertIn(
            "object=NONE 只允许用于 abstain_intent 和 no_commitment",
            prompt,
        )

    def test_prompt_requires_explicit_target_and_atomic_range_output(self):
        backend = FakeBackend("NONE")
        perceiver = SpeechPerceiver(backend=backend, model_name="test-model")
        perceiver.parse(1, "基于以上判断，我投这一票。", 1, "speech")
        prompt = backend.calls[0]["messages"][0]["content"]

        self.assertIn("同一个明确命题或分句", prompt)
        self.assertIn("不得从代词、省略宾语", prompt)
        self.assertIn("第一人称明确自报具体身份", prompt)
        self.assertIn("连续玩家范围必须按原文顺序展开成多个原子三元组", prompt)
        self.assertIn("player1 | vote_intent | player4", prompt)

    def test_accepts_none_as_the_only_empty_result(self):
        perceiver = SpeechPerceiver(
            backend=FakeBackend("NONE"),
            model_name="test-model",
        )
        self.assertEqual(perceiver.parse(2, "我还要继续听。", 1, "speech"), [])

    def test_targetless_actions_require_uppercase_none(self):
        perceiver = SpeechPerceiver(
            backend=FakeBackend(
                "player1 | abstain_intent | NONE\n"
                "player1 | no_commitment | NONE"
            ),
            model_name="test-model",
        )
        self.assertEqual(
            perceiver.parse(1, "这一轮我弃票，也不作身份表态。", 1, "speech"),
            [
                ["player1", "abstain_intent", None],
                ["player1", "no_commitment", None],
            ],
        )

    def test_targeted_action_rejects_none_without_partial_acceptance(self):
        raw_response = (
            "player5 | point_as_villager | player5\n"
            "player5 | support | NONE\n"
            "player5 | oppose | player4\n"
            "player5 | vote_intent | player4"
        )
        perceiver = SpeechPerceiver(
            backend=FakeBackend(raw_response),
            model_name="test-model",
        )

        with self.assertRaisesRegex(
            SpeechActionValidationError,
            "support requires a canonical player object",
        ) as caught:
            perceiver.parse(
                5,
                "我是平民，明确站边好人阵营，但我质疑并准备投票放逐 player4。",
                1,
                "speech",
            )
        self.assertEqual(caught.exception.raw_response, raw_response)

    def test_deduplicates_only_exact_valid_actions(self):
        perceiver = SpeechPerceiver(
            backend=FakeBackend(
                "player2 | support | player3\n"
                "player2 | support | player3\n"
                "player2 | oppose | player3"
            ),
            model_name="test-model",
        )
        self.assertEqual(
            perceiver.parse(2, "我支持3号，但也质疑3号。", 1, "speech"),
            [
                ["player2", "support", "player3"],
                ["player2", "oppose", "player3"],
            ],
        )

    def test_does_not_add_actions_that_are_absent_from_model_response(self):
        perceiver = SpeechPerceiver(
            backend=FakeBackend("NONE"),
            model_name="test-model",
        )
        self.assertEqual(perceiver.parse(6, "我是6号村民。", 1, "speech"), [])

    def test_rejects_alternate_subject_instead_of_repairing_it(self):
        perceiver = SpeechPerceiver(
            backend=FakeBackend("player7 | support | player3"),
            model_name="test-model",
        )
        with self.assertRaisesRegex(
            SpeechActionValidationError,
            "subject must equal current speaker",
        ):
            perceiver.parse(2, "我支持3号。", 1, "speech")

    def test_rejects_non_protocol_formats(self):
        responses = (
            '[["player1", "support", "player2"]]',
            "```text\nplayer1 | support | player2\n```",
            "- player1 | support | player2",
            "player1｜support｜player2",
            "player1 | support | player2。",
            "player1 | support | player2 至 player4",
            "player1 | no_commitment | null",
            "NONE\n",
        )
        for response in responses:
            with self.subTest(response=response):
                perceiver = SpeechPerceiver(
                    backend=FakeBackend(response),
                    model_name="test-model",
                )
                with self.assertRaises((SpeechActionValidationError, ValueError)):
                    perceiver.parse(1, "公开发言", 1, "speech")

    def test_rejects_extra_or_partial_lines(self):
        responses = (
            "player1 | support",
            "player1 | support | player2 | extra",
            "player1 | support | player2\n解释：这是抽取原因",
            "NONE\nplayer1 | support | player2",
        )
        for response in responses:
            with self.subTest(response=response):
                perceiver = SpeechPerceiver(
                    backend=FakeBackend(response),
                    model_name="test-model",
                )
                with self.assertRaises(SpeechActionValidationError):
                    perceiver.parse(1, "公开发言", 1, "speech")

    def test_rejects_unknown_action_and_invalid_player(self):
        for response in (
            "player1 | invented_action | player2",
            "player1 | support | player8",
        ):
            with self.subTest(response=response):
                perceiver = SpeechPerceiver(
                    backend=FakeBackend(response),
                    model_name="test-model",
                )
                with self.assertRaises(SpeechActionValidationError):
                    perceiver.parse(1, "公开发言", 1, "speech")

    def test_mixed_valid_invalid_output_fails_as_one_response(self):
        perceiver = SpeechPerceiver(
            backend=FakeBackend(
                "player1 | support | player2\n"
                "player1 | invented_action | player3"
            ),
            model_name="test-model",
        )
        with self.assertRaises(SpeechActionValidationError) as caught:
            perceiver.parse(1, "公开发言", 1, "speech")
        self.assertEqual(caught.exception.invalid_count, 1)

    def test_strict_with_response_preserves_backend_text(self):
        raw_response = "player1 | support | player3"
        backend = FakeBackend(raw_response)
        perceiver = SpeechPerceiver(backend=backend, model_name="test-model")

        actions, returned_response = perceiver.parse_strict_with_response(
            1, "发言", 1, "speech"
        )
        self.assertEqual(actions, [["player1", "support", "player3"]])
        self.assertEqual(returned_response, raw_response)

    def test_validation_error_attaches_unchanged_backend_text(self):
        raw_response = "player1 | invented_action | player2"
        perceiver = SpeechPerceiver(
            backend=FakeBackend(raw_response),
            model_name="test-model",
        )
        with self.assertRaises(SpeechActionValidationError) as caught:
            perceiver.parse(1, "发言", 1, "speech")
        self.assertEqual(caught.exception.raw_response, raw_response)

    def test_parse_with_audit_records_success(self):
        raw_response = "player1 | support | player2"
        perceiver = SpeechPerceiver(
            backend=FakeBackend(raw_response),
            model_name="test-model",
        )
        result = perceiver.parse_with_audit(1, "我支持2号。", 1, "speech")
        self.assertEqual(
            result.normalized_actions,
            [["player1", "support", "player2"]],
        )
        self.assertEqual(result.raw_response, raw_response)
        self.assertEqual(result.parse_status, "ok")
        self.assertIsNone(result.error_type)
        self.assertIsNone(result.error_message)
        self.assertEqual(len(result.generation_attempts), 1)
        self.assertEqual(result.generation_attempts[0]["status"], "ok")

    def test_parse_with_audit_fails_closed_without_repair(self):
        raw_response = '[["player1", "support", "player2"]]'
        perceiver = SpeechPerceiver(
            backend=FakeBackend(raw_response),
            model_name="test-model",
        )
        result = perceiver.parse_with_audit(1, "我支持2号。", 1, "speech")
        self.assertEqual(result.normalized_actions, [])
        self.assertEqual(result.raw_response, raw_response)
        self.assertEqual(result.parse_status, "parser_error")
        self.assertEqual(result.error_type, "SpeechActionValidationError")
        self.assertEqual(
            len(result.generation_attempts),
            SPEECH_PARSER_GENERATION_MAX_ATTEMPTS,
        )
        self.assertTrue(
            all(
                attempt["status"] == "parser_error"
                for attempt in result.generation_attempts
            )
        )

    def test_parse_with_audit_retries_the_entire_response_with_feedback(self):
        invalid_response = (
            "player2 | point_as_non_werewolf | player2\n"
            "player2 | support | NONE"
        )
        valid_response = "player2 | point_as_non_werewolf | player2"
        backend = FakeBackend((invalid_response, valid_response))
        perceiver = SpeechPerceiver(backend=backend, model_name="test-model")

        result = perceiver.parse_with_audit(
            2,
            "我是好人玩家 2，今天我必须站ritte。",
            1,
            "speech",
        )

        self.assertEqual(
            result.normalized_actions,
            [["player2", "point_as_non_werewolf", "player2"]],
        )
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(
            [attempt["status"] for attempt in result.generation_attempts],
            ["parser_error", "ok"],
        )
        retry_prompt = backend.calls[1]["messages"][0]["content"]
        self.assertIn("必须重新生成整份输出", retry_prompt)
        self.assertIn("support requires a canonical player object", retry_prompt)
        self.assertIn("不得部分修补", retry_prompt)

    def test_parse_with_audit_records_backend_failure(self):
        perceiver = SpeechPerceiver(
            backend=FakeBackend(error=RuntimeError("backend unavailable")),
            model_name="test-model",
        )
        result = perceiver.parse_with_audit(1, "发言", 1, "speech")
        self.assertEqual(result.normalized_actions, [])
        self.assertIsNone(result.raw_response)
        self.assertEqual(result.parse_status, "parser_error")
        self.assertEqual(result.error_type, "RuntimeError")
        self.assertEqual(result.error_message, "backend unavailable")
        self.assertEqual(
            len(result.generation_attempts),
            SPEECH_PARSER_GENERATION_MAX_ATTEMPTS,
        )

    def test_parse_exposes_backend_and_precondition_failures(self):
        perceiver = SpeechPerceiver(
            backend=FakeBackend(error=RuntimeError("backend unavailable")),
            model_name="test-model",
        )
        with self.assertRaisesRegex(RuntimeError, "backend unavailable"):
            perceiver.parse(1, "发言", 1, "speech")

        for unconfigured in (
            SpeechPerceiver(model_name="test-model"),
            SpeechPerceiver(backend=FakeBackend("NONE")),
        ):
            with self.assertRaisesRegex(RuntimeError, "must be configured"):
                unconfigured.parse(1, "发言", 1, "speech")

        configured = SpeechPerceiver(
            backend=FakeBackend("NONE"),
            model_name="test-model",
        )
        with self.assertRaisesRegex(ValueError, "speaker"):
            configured.parse(0, "发言", 1, "speech")
        with self.assertRaisesRegex(ValueError, "non-empty"):
            configured.parse(1, " ", 1, "speech")


if __name__ == "__main__":
    unittest.main()
