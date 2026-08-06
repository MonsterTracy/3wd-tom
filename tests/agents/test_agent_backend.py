import inspect
import unittest

from werewolf.agents import agent_registry
from werewolf.agents.gpt_agent import GPTAgent
from werewolf.agents.llm_agent import (
    GameplaySpeechQualityError,
    PublicSpeechPlanValidationError,
    validate_gameplay_public_speech,
    validate_public_speech_plan,
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

    def test_strict_speech_uses_plan_then_isolated_renderer(self):
        backend = MetadataBackend([
            '{"public_actions":[{"action":"oppose","target":3}]}',
            "我不认可3号的说法。",
        ])
        backend.supports_json_schema = True
        agent = GPTAgent(
            backend=backend,
            model_name="agent-model",
            gameplay_prompt_profile="strict_classic7",
            gameplay_max_tokens=512,
        )
        agent.rate_limit = 0

        self.assertEqual(
            agent.act(self._strict_observation()),
            ("speech", "我不认可3号的说法。"),
        )
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(backend.calls[0]["response_format"]["type"], "json_schema")
        self.assertIsNone(backend.calls[1]["response_format"])
        self.assertEqual([call["max_tokens"] for call in backend.calls], [512, 512])
        renderer = backend.calls[1]["messages"][0]["content"]
        self.assertIn('"action":"oppose"', renderer)
        self.assertNotIn("【你合法知道的私有信息】", renderer)

    def test_public_speech_plan_validation_contract(self):
        state = self._strict_observation()["authoritative_public_state"]
        valid_cases = (
            [],
            [
                {"action": "point_as_seer", "target": 1},
                {"action": "check_as_werewolf", "target": 3},
                {"action": "vote_intent", "target": 3},
            ],
            [
                {"action": "point_as_werewolf", "target": 3},
                {"action": "oppose", "target": 3},
                {"action": "vote_intent", "target": 3},
            ],
        )
        for actions in valid_cases:
            with self.subTest(valid=actions):
                plan = validate_public_speech_plan(
                    {"public_actions": actions},
                    authoritative_public_state=state,
                    player_id=1,
                    phase="1_day_speech",
                )
                self.assertEqual(plan.as_list(), actions)

        invalid_cases = (
            {"public_actions": [], "notes": "free text"},
            {"public_actions": [{"action": "invented", "target": 3}]},
            {"public_actions": [{"action": "oppose", "target": 3, "why": "x"}]},
            {"public_actions": [{"action": "vote_intent", "target": 1}]},
            {"public_actions": [
                {"action": "oppose", "target": 3},
                {"action": "oppose", "target": 3},
            ]},
            {"public_actions": [
                {"action": "check_as_werewolf", "target": 3},
                {"action": "point_as_werewolf", "target": 3},
            ]},
        )
        for payload in invalid_cases:
            with self.subTest(invalid=payload), self.assertRaises(
                PublicSpeechPlanValidationError
            ):
                validate_public_speech_plan(
                    payload,
                    authoritative_public_state=state,
                    player_id=1,
                    phase="1_day_speech",
                )

    def test_final_speech_must_realize_exact_plan_player_scope(self):
        self.assertEqual(
            validate_gameplay_public_speech(
                "我反对3号。",
                finish_reason="stop",
                player_id=1,
                phase="1_day_speech",
                planned_player_ids={3},
            ),
            "我反对3号。",
        )
        for speech, planned in (("我反对4号。", {3}), ("我继续听。", {3}), ("我观察2号。", set())):
            with self.subTest(speech=speech), self.assertRaises(
                GameplaySpeechQualityError
            ):
                validate_gameplay_public_speech(
                    speech,
                    finish_reason="stop",
                    player_id=1,
                    phase="1_day_speech",
                    planned_player_ids=planned,
                )

    def test_strict_speech_stops_at_failed_stage_without_retry_or_fallback(self):
        cases = (
            ([BackendError("planner failed")], None, 1),
            (["not json"], None, 1),
            (['{"public_actions":[]}'], [{"finish_reason": "length"}], 1),
            (['{"public_actions":[{"action":"vote_intent","target":1}]}'], None, 1),
            (['{"public_actions":[]}'], [None], 1),
            (['{"public_actions":[]}', BackendError("renderer failed")], None, 2),
            (['{"public_actions":[]}', "观望"], [{"finish_reason": "stop"}, None], 2),
            (['{"public_actions":[]}', "观望"], [{"finish_reason": "stop"}, {"finish_reason": "length"}], 2),
            (['{"public_actions":[]}', "提到2号"], None, 2),
        )
        for responses, metadata, call_count in cases:
            with self.subTest(responses=responses):
                backend = MetadataBackend(responses, metadata=metadata)
                backend.supports_json_schema = True
                agent = GPTAgent(
                    backend=backend,
                    model_name="agent-model",
                    gameplay_prompt_profile="strict_classic7",
                )
                agent.rate_limit = 0
                with self.assertRaises((BackendError, PublicSpeechPlanValidationError, GameplaySpeechQualityError)):
                    agent.act(self._strict_observation())
                self.assertEqual(len(backend.calls), call_count)

        backend = MetadataBackend(['{"public_actions":[]}'])
        agent = GPTAgent(
            backend=backend,
            model_name="agent-model",
            gameplay_prompt_profile="strict_classic7",
        )
        agent.rate_limit = 0
        with self.assertRaisesRegex(BackendError, "JSON Schema support"):
            agent.act(self._strict_observation())
        self.assertEqual(backend.calls, [])

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
        observation = {
            "phase": "1_day_speech",
            "identity": "Villager",
            "current_act_idx": 1,
            "game_log": [],
            "valid_action": ("speech", -1),
        }

        action = agent.act(observation)

        self.assertEqual(action, ("speech", "这是发言"))
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(backend.calls[0]["model"], "agent-model")
        self.assertEqual(backend.calls[0]["temperature"], 0.2)
        self.assertIsNone(backend.calls[0]["max_tokens"])

    def test_gameplay_speech_requires_explicit_fresh_metadata(self):
        no_metadata_backend = RecordingBackend(["不会被旧路径读取"])
        agent = GPTAgent(
            backend=no_metadata_backend,
            model_name="agent-model",
        )
        agent.rate_limit = 0
        observation = {
            "phase": "1_day_speech",
            "identity": "Villager",
            "current_act_idx": 1,
            "game_log": [],
            "valid_action": ("speech", -1),
        }

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
                {
                    "phase": "1_day_speech",
                    "identity": "Villager",
                    "current_act_idx": 1,
                    "game_log": [],
                    "valid_action": ("speech", -1),
                },
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

    def test_gpt_agent_preserves_legacy_o1_limit_when_unconfigured(self):
        backend = MetadataBackend(["公开发言"])
        agent = GPTAgent(
            backend=backend,
            model_name="o1-test-model",
        )
        agent.rate_limit = 0

        agent.act(
            {
                "phase": "1_day_speech",
                "identity": "Villager",
                "current_act_idx": 1,
                "game_log": [],
                "valid_action": ("speech", -1),
            }
        )

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
            gameplay_prompt_profile="strict_classic7",
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
        self.assertEqual(
            agent.gameplay_prompt_profile,
            "strict_classic7",
        )
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
