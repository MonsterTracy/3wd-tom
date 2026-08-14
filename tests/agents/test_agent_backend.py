import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from werewolf.agents import agent_registry
from werewolf.agents.gpt_agent import GPTAgent
from werewolf.agents.llm_agent import (
    GameplayActionValidationError,
    GameplaySpeechQualityError,
    validate_gameplay_public_speech,
)
from werewolf.agents.twdm_agent import TWDMStrategyAgent
from werewolf.backends import BackendError
from werewolf.registry import Registry


class RecordingBackend:
    def __init__(self, responses=None):
        self.responses = list(responses or ["response"])
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
        self._record(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            **kwargs,
        )
        return self.responses.pop(0)

    def _record(
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


class MetadataBackend(RecordingBackend):
    def __init__(self, responses=None, metadata=None):
        super().__init__(responses)
        self.metadata = list(
            metadata
            or [{"finish_reason": "stop"}] * len(self.responses)
        )

    def chat_with_metadata(self, **kwargs):
        self._record(**kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response, self.metadata.pop(0)


class AgentBackendTest(unittest.TestCase):
    @staticmethod
    def _strict_observation():
        return {
            "phase": "1_day_speech",
            "identity": "Villager",
            "current_act_idx": 1,
            "game_log": [],
            "valid_action": ("speech", -1),
            "authoritative_public_state": {
                "day": 1,
                "day_or_night": "day",
                "phase": "speech",
                "last_night_result": {"day": 0, "dead_players": []},
                "prior_exiles": [],
                "alive_players": [1, 2, 3, 4, 5, 6, 7],
                "suggestible_exile_targets": [2, 3, 4, 5, 6, 7],
            },
        }

    @staticmethod
    def _dead_player_observation():
        observation = AgentBackendTest._strict_observation()
        observation["current_act_idx"] = 2
        observation["authoritative_public_state"].update({
            "last_night_result": {"day": 0, "dead_players": [3]},
            "alive_players": [1, 2, 4, 5, 6, 7],
            "suggestible_exile_targets": [1, 4, 5, 6, 7],
        })
        return observation

    @staticmethod
    def _player_records(path):
        return [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line
        ]

    def test_direct_speech_is_one_plain_text_call_without_schema(self):
        raw_speech = "  我暂时更怀疑3号。\n"
        backend = MetadataBackend([raw_speech])
        agent = GPTAgent(backend=backend, model_name="agent-model")
        agent.rate_limit = 0

        action = agent.act(self._strict_observation())

        self.assertEqual(action, ("speech", raw_speech))
        self.assertEqual(len(backend.calls), 1)
        self.assertIsNone(backend.calls[0]["response_format"])
        self.assertEqual(
            backend.calls[0]["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )
        prompt = backend.calls[0]["messages"][0]["content"]
        for forbidden in (
            "public_actions",
            "Core-13",
            "Planner",
            "Renderer",
            "reasoning字段",
            "strategy字段",
        ):
            self.assertNotIn(forbidden, prompt)

    def test_direct_speech_backend_failure_is_not_retried(self):
        backend = MetadataBackend([BackendError("speech failed")])
        agent = GPTAgent(backend=backend, model_name="agent-model")
        agent.rate_limit = 0

        with self.assertRaisesRegex(BackendError, "speech failed"):
            agent.act(self._strict_observation())

        self.assertEqual(len(backend.calls), 1)

    def test_gameplay_public_speech_quality_accepts_safe_numeric_context(self):
        for speech in (
            "第2天我会关注player1到player7的发言。",
            "昨夜1人死亡，目前有2票需要重新判断。",
            "我认为3号玩家的逻辑更可信。",
        ):
            with self.subTest(speech=speech):
                self.assertEqual(
                    validate_gameplay_public_speech(
                        speech,
                        finish_reason="stop",
                        player_id=1,
                        phase="2_day_speech",
                    ),
                    speech,
                )

    def test_gameplay_public_speech_quality_rejects_deterministic_failures(self):
        cases = (
            ("", None),
            ("   ", None),
            ("正常发言", "length"),
            ("我怀疑player0", "stop"),
            ("我怀疑player8", "stop"),
            ("我怀疑player12", "stop"),
            ("我怀疑0号玩家", "stop"),
            ("我怀疑12号位", "stop"),
            ('{"speech": "我怀疑3号"}', "stop"),
            ("【权威公共状态】存活玩家如下", "stop"),
            ("current_act_idx=3", "stop"),
        )
        for speech, finish_reason in cases:
            with self.subTest(speech=speech, finish_reason=finish_reason):
                with self.assertRaises(GameplaySpeechQualityError):
                    validate_gameplay_public_speech(
                        speech,
                        finish_reason=finish_reason,
                        player_id=1,
                        phase="2_day_speech",
                    )

    def test_gpt_agent_speech_uses_backend_chat_and_agent_model(self):
        backend = MetadataBackend(["这是发言"])
        agent = GPTAgent(
            backend=backend,
            model_name="agent-model",
            temperature=0.2,
        )
        agent.rate_limit = 0
        observation = self._strict_observation()

        action = agent.act(observation)

        self.assertEqual(action, ("speech", "这是发言"))
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(backend.calls[0]["model"], "agent-model")
        self.assertEqual(backend.calls[0]["temperature"], 0.2)
        self.assertIsNone(backend.calls[0]["max_tokens"])
        self.assertEqual(
            backend.calls[0]["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )

    def test_gameplay_speech_requires_explicit_fresh_metadata(self):
        no_metadata_backend = RecordingBackend(["不会被旧路径读取"])
        agent = GPTAgent(
            backend=no_metadata_backend,
            model_name="agent-model",
        )
        agent.rate_limit = 0
        observation = self._strict_observation()

        with self.assertRaisesRegex(
            BackendError,
            "must support chat_with_metadata",
        ):
            agent.act(observation)
        self.assertEqual(no_metadata_backend.calls, [])

        backend = MetadataBackend(
            ["第一次发言", "第二次发言"],
            metadata=[{"finish_reason": "stop"}, None],
        )
        agent.backend = backend
        self.assertEqual(agent.act(observation), ("speech", "第一次发言"))
        with self.assertRaisesRegex(
            BackendError,
            "requires finish_reason metadata",
        ):
            agent.act(observation)
        self.assertEqual(len(backend.calls), 2)

    def test_gpt_agent_applies_gameplay_max_tokens_to_speech_and_action(self):
        observations = (
            (
                "speech",
                "公开发言",
                AgentBackendTest._strict_observation(),
                ("speech", "公开发言"),
            ),
            (
                "vote",
                "{'投票': '2'}",
                {
                    "phase": "1_day_vote",
                    "identity": "Villager",
                    "current_act_idx": 1,
                    "game_log": [],
                    "valid_action": [("vote", 2)],
                },
                ("vote", 2),
            ),
        )

        for name, response, observation, expected in observations:
            with self.subTest(name=name):
                backend = (
                    MetadataBackend([response])
                    if name == "speech"
                    else RecordingBackend([response])
                )
                agent = GPTAgent(
                    backend=backend,
                    model_name="agent-model",
                    gameplay_max_tokens=512,
                )
                agent.rate_limit = 0

                self.assertEqual(agent.act(observation), expected)
                self.assertEqual(
                    backend.calls[0]["max_tokens"],
                    512,
                )
                if name == "vote":
                    self.assertEqual(
                        backend.calls[0]["temperature"],
                        1.0,
                    )
                    self.assertEqual(
                        backend.calls[0]["extra_body"],
                        {
                            "chat_template_kwargs": {
                                "enable_thinking": False,
                            }
                        },
                    )
                else:
                    self.assertEqual(
                        backend.calls[0]["extra_body"],
                        {"chat_template_kwargs": {"enable_thinking": False}},
                    )

    def test_vote_retries_three_times_then_keeps_existing_fallback(self):
        backend = RecordingBackend(["bad", "still bad", "also bad"])
        agent = GPTAgent(
            backend=backend,
            model_name="agent-model",
            temperature=0.6,
            gameplay_max_tokens=512,
        )
        agent.rate_limit = 0
        fallback_calls = []
        agent.choose_fallback_vote_action = lambda *_args: (
            fallback_calls.append(True) or "{'投票': '2'}"
        )
        observation = {
            "phase": "1_day_vote",
            "identity": "Villager",
            "current_act_idx": 1,
            "game_log": [],
            "valid_action": [("vote", 2), ("vote", 3)],
        }

        self.assertEqual(agent.act(observation), ("vote", 2))
        self.assertEqual(len(backend.calls), 3)
        self.assertEqual(fallback_calls, [True])
        for call in backend.calls:
            self.assertEqual(call["temperature"], 0.6)
            self.assertEqual(call["max_tokens"], 512)
            self.assertEqual(
                call["extra_body"],
                {
                    "chat_template_kwargs": {
                        "enable_thinking": False,
                    }
                },
            )

    def test_invalid_vote_response_retries_then_accepts_valid_response(self):
        backend = RecordingBackend(["bad", "{'投票': '3'}"])
        agent = GPTAgent(
            backend=backend,
            model_name="agent-model",
            gameplay_max_tokens=512,
        )
        agent.rate_limit = 0
        observation = {
            "phase": "1_day_vote_pk",
            "identity": "Villager",
            "current_act_idx": 1,
            "game_log": [],
            "valid_action": [("vote_pk", 2), ("vote_pk", 3)],
        }

        self.assertEqual(agent.act(observation), ("vote_pk", 3))
        self.assertEqual(len(backend.calls), 2)

    def test_v25_structural_matcher_remains_the_membership_boundary(self):
        agent = GPTAgent()
        candidates = (
            "{'查验':'7'}",
            "{'解药': '否', '毒药': '4'}",
        )
        cases = (
            ("{'查验': '7'}", "{'查验':'7'}"),
            ('```json\n{"查验":"7"}\n```', "{'查验':'7'}"),
            (
                "{'毒药': '4', '解药': '否'}",
                "{'解药': '否', '毒药': '4'}",
            ),
        )
        for response, expected in cases:
            with self.subTest(response=response):
                self.assertEqual(
                    agent.match_authoritative_action_response(
                        response,
                        candidates,
                    ),
                    expected,
                )

    def test_v26_night_snapshot_drives_prompt_schema_and_mapping(self):
        cases = (
            (
                "seer",
                "1_night_skill_seer",
                [("check", 1), ("check", 2), ("check", 4), ("check", 5)],
                '{"action_index": 2}',
                ("check", 4),
                "{'查验':'4'}",
            ),
            (
                "werewolf",
                "0_night_skill_wolf",
                [("kill", 3), ("kill", 7)],
                '```json\n{"action_index":0}\n```',
                ("kill", 3),
                "{'杀害':'3'}",
            ),
            (
                "guard",
                "0_night_skill_guard",
                [("guard", 2), ("guard", 4), ("guard", 6)],
                '{"action_index":1}',
                ("guard", 4),
                "{'守卫':'4'}",
            ),
            (
                "witch",
                "0_night_skill_witch",
                [
                    ("witch_pass", 0),
                    ("witch_poison", 1),
                    ("witch_poison", 2),
                    ("witch_heal", 4),
                ],
                '{"action_index":3}',
                ("witch_heal", 4),
                "{'解药': '4', '毒药': '否'}",
            ),
        )
        for name, phase, valid_actions, response, expected, display in cases:
            with self.subTest(name=name):
                backend = MetadataBackend([response])
                backend.supports_json_schema = True
                agent = GPTAgent(
                    backend=backend,
                    model_name="agent-model",
                    gameplay_max_tokens=512,
                )
                agent.rate_limit = 0

                action = agent.act({
                    "phase": phase,
                    "identity": "Villager",
                    "current_act_idx": 1,
                    "game_log": [],
                    "valid_action": valid_actions,
                })

                self.assertEqual(action, expected)
                self.assertEqual(len(backend.calls), 1)
                request = backend.calls[0]
                schema = request["response_format"]["json_schema"]["schema"]
                self.assertEqual(request["response_format"]["type"], "json_schema")
                self.assertEqual(
                    request["response_format"]["json_schema"]["name"],
                    "night_action_selection",
                )
                self.assertTrue(
                    request["response_format"]["json_schema"]["strict"]
                )
                self.assertEqual(request["max_tokens"], 512)
                self.assertEqual(
                    request["extra_body"],
                    {
                        "chat_template_kwargs": {
                            "enable_thinking": False,
                        }
                    },
                )
                self.assertEqual(
                    schema["properties"]["action_index"]["enum"],
                    list(range(len(valid_actions))),
                )
                prompt = request["messages"][0]["content"]
                snapshot = agent.freeze_authoritative_action_candidates(
                    valid_actions
                )
                for index, (candidate_display, env_action) in enumerate(snapshot):
                    self.assertIn(f"{index}: {candidate_display}", prompt)
                    self.assertEqual(valid_actions[index], env_action)
                self.assertEqual(
                    agent.nlp_action_to_env_action[display],
                    expected,
                )
                if name == "seer":
                    self.assertNotIn("{'查验':'3'}", prompt)
                elif name == "werewolf":
                    self.assertNotIn("{'杀害':'2'}", prompt)
                elif name == "guard":
                    self.assertNotIn("{'守卫':'3'}", prompt)
                else:
                    self.assertNotIn(
                        "{'解药': '4', '毒药': '1'}",
                        prompt,
                    )

    def test_v26_seed520_witch_has_no_index_for_poisoned_player3(self):
        valid_actions = [
            ("witch_pass", 0),
            ("witch_poison", 1),
            ("witch_poison", 2),
            ("witch_poison", 4),
            ("witch_poison", 5),
            ("witch_poison", 6),
            ("witch_poison", 7),
        ]
        backend = MetadataBackend(['{"action_index":3}'])
        backend.supports_json_schema = True
        agent = GPTAgent(backend=backend, model_name="agent-model")
        agent.rate_limit = 0

        self.assertEqual(
            agent.act({
                "phase": "1_night_skill_witch",
                "identity": "Witch",
                "current_act_idx": 6,
                "game_log": [],
                "valid_action": valid_actions,
            }),
            ("witch_poison", 4),
        )
        prompt = backend.calls[0]["messages"][0]["content"]
        self.assertNotIn("'毒药': '3'", prompt)
        self.assertEqual(
            backend.calls[0]["response_format"]["json_schema"]["schema"]
            ["properties"]["action_index"]["enum"],
            list(range(len(valid_actions))),
        )

    def test_v26_logs_raw_index_and_canonical_action_without_regeneration(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "Player_5.jsonl"
            backend = MetadataBackend(['{"action_index":1}'])
            backend.supports_json_schema = True
            agent = GPTAgent(
                backend=backend,
                model_name="agent-model",
                log_file=str(log_path),
            )
            agent.rate_limit = 0
            try:
                self.assertEqual(
                    agent.act({
                        "phase": "1_night_skill_seer",
                        "identity": "Seer",
                        "current_act_idx": 5,
                        "game_log": [],
                        "valid_action": [("check", 3), ("check", 7)],
                    }),
                    ("check", 7),
                )
            finally:
                agent.close()

            records = self._player_records(log_path)
            self.assertEqual(len(backend.calls), 1)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["response"], '{"action_index":1}')
            self.assertEqual(records[0]["action"], "{'查验':'7'}")
            self.assertEqual(records[0]["gen_times"], 0)

    def test_v26_rejects_invalid_structured_night_response_once(self):
        invalid_responses = (
            "{}",
            '{"action_index":-1}',
            '{"action_index":999}',
            '{"action_index":true}',
            '{"action_index":1.0}',
            '{"action_index":"1"}',
            '{"action_index":null}',
            '{"action_index":1,"extra":2}',
            "[]",
            "null",
            "not json",
            'Thinking about the choice.\n{"action_index":0}',
        )

        for response in invalid_responses:
            with self.subTest(response=response):
                backend = MetadataBackend([response])
                backend.supports_json_schema = True
                agent = GPTAgent(
                    backend=backend,
                    model_name="agent-model",
                )
                agent.rate_limit = 0
                with self.assertRaisesRegex(
                    GameplayActionValidationError,
                    "invalid night action selection.*phase=.*response=",
                ):
                    agent.act({
                        "phase": "0_night_skill_witch",
                        "identity": "Witch",
                        "current_act_idx": 7,
                        "game_log": [],
                        "valid_action": [
                            ("witch_pass", 0),
                            ("witch_heal", 3),
                            ("witch_poison", 4),
                        ],
                    })
                self.assertEqual(len(backend.calls), 1)

    def test_v26_truncated_night_response_is_fatal_without_retry(self):
        backend = MetadataBackend(
            ['{"action_index":0}'],
            metadata=[{"finish_reason": "length"}],
        )
        backend.supports_json_schema = True
        agent = GPTAgent(
            backend=backend,
            model_name="agent-model",
        )
        agent.rate_limit = 0

        with self.assertRaisesRegex(
            GameplayActionValidationError,
            "phase=.*finish_reason='length'",
        ):
            agent.act({
                "phase": "1_night_skill_seer",
                "identity": "Seer",
                "current_act_idx": 5,
                "game_log": [],
                "valid_action": [("check", 7)],
            })
        self.assertEqual(len(backend.calls), 1)

    def test_v26_night_action_requires_backend_model_and_schema_support(self):
        cases = (
            (None, "agent-model", "backend and model_name are required"),
            (
                MetadataBackend(['{"action_index":0}']),
                None,
                "backend and model_name are required",
            ),
            (
                MetadataBackend(['{"action_index":0}']),
                "agent-model",
                "require backend JSON Schema support",
            ),
        )

        for backend, model_name, error in cases:
            with self.subTest(
                has_backend=backend is not None,
                model_name=model_name,
            ):
                agent = GPTAgent(
                    backend=backend,
                    model_name=model_name,
                )
                agent.rate_limit = 0
                with self.assertRaisesRegex(
                    BackendError,
                    error,
                ):
                    agent.act({
                        "phase": "1_night_skill_seer",
                        "identity": "Seer",
                        "current_act_idx": 5,
                        "game_log": [],
                        "valid_action": [("check", 7)],
                    })
                if backend is not None:
                    self.assertEqual(backend.calls, [])

    def test_gpt_agent_preserves_legacy_o1_limit_when_unconfigured(self):
        backend = MetadataBackend(["公开发言"])
        agent = GPTAgent(
            backend=backend,
            model_name="o1-test-model",
        )
        agent.rate_limit = 0

        agent.act(self._strict_observation())

        self.assertEqual(backend.calls[0]["max_tokens"], 32000)
        self.assertIsNone(backend.calls[0]["temperature"])

    def test_agent_rejects_invalid_gameplay_max_tokens(self):
        for invalid in (True, False, 0, -1, 1.5, "512"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    ValueError,
                    "gameplay_max_tokens",
                ):
                    GPTAgent(
                        backend=RecordingBackend(),
                        model_name="agent-model",
                        gameplay_max_tokens=invalid,
                    )

    def test_twdm_generation_uses_backend_chat_and_agent_model(self):
        backend = RecordingBackend(["  structured response  "])
        agent = TWDMStrategyAgent(
            backend=backend,
            model_name="twdm-model",
            temperature=0.1,
            gameplay_max_tokens=512,
        )

        response = agent._TWDMStrategyAgent__api_generate(
            [{"role": "user", "content": " prompt "}]
        )

        self.assertEqual(response, "structured response")
        self.assertEqual(backend.calls[0]["model"], "twdm-model")
        self.assertEqual(backend.calls[0]["temperature"], 0.1)
        self.assertEqual(backend.calls[0]["max_tokens"], 512)
        self.assertEqual(
            backend.calls[0]["messages"],
            [{"role": "user", "content": "prompt"}],
        )

    def test_registry_injects_backend_and_resolves_model_name(self):
        backend = RecordingBackend()

        agent_type, params = agent_registry.build(
            "gpt",
            backend=backend,
            default_model="default-agent-model",
            temperature=0.3,
            gameplay_max_tokens=512,
        )
        agent = agent_registry.build_agent(
            agent_type,
            player_idx=0,
            agent_param=params,
            env_param={"n_player": 7, "n_role": 4},
            log_file=None,
        )

        self.assertIs(agent.backend, backend)
        self.assertEqual(agent.model_name, "default-agent-model")
        self.assertEqual(agent.temperature, 0.3)
        self.assertNotIn("gameplay_prompt_profile", params)
        self.assertFalse(hasattr(agent, "gameplay_prompt_profile"))
        self.assertEqual(agent.gameplay_max_tokens, 512)

    def test_registry_supports_per_agent_model_override_and_llm_alias(self):
        backend = RecordingBackend()

        _, explicit_params = agent_registry.build(
            "gpt",
            backend=backend,
            default_model="default",
            model_name="explicit",
            temperature=0.3,
        )
        _, alias_params = agent_registry.build(
            "gpt",
            backend=backend,
            default_model="default",
            llm="legacy-alias",
            temperature=0.3,
        )

        self.assertEqual(explicit_params["model_name"], "explicit")
        self.assertEqual(alias_params["model_name"], "legacy-alias")

    def test_registry_has_no_provider_or_credential_responsibility(self):
        source = inspect.getsource(Registry)

        for forbidden in (
            "openai.OpenAI",
            "openai.AzureOpenAI",
            "OPENAI_API_KEY",
            "DEEPSEEK_API_KEY",
            "AZURE_OPENAI_API_KEY",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
