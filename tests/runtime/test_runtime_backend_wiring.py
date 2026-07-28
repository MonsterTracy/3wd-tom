import unittest

from werewolf.backends import BackendSettings

try:
    from run_battle import build_runtime
except ImportError:
    build_runtime = None

try:
    from run_random import build_runtime as build_random_runtime
except ImportError:
    build_random_runtime = None


ROLES = [
    "Werewolf",
    "Werewolf",
    "Seer",
    "Witch",
    "Villager",
    "Villager",
    "Villager",
]
RANDOM_ROLES = [
    "Werewolf",
    "Villager",
    "Seer",
    "Witch",
    "Villager",
    "Villager",
    "Werewolf",
]


class RecordingBackend:
    def __init__(self):
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
        self.calls.append({"messages": messages, "model": model})
        return "[]"


class RuntimeAvailabilityTest(unittest.TestCase):
    def test_battle_runtime_builder_is_available(self):
        self.assertIsNotNone(build_runtime)

    def test_random_runtime_builder_is_available(self):
        self.assertIsNotNone(build_random_runtime)


@unittest.skipIf(build_runtime is None, "runtime builder is not implemented")
class RuntimeBackendWiringTest(unittest.TestCase):
    def test_battle_runtime_injects_backend_and_separate_models(self):
        backend = RecordingBackend()
        settings = BackendSettings(
            backend_type="openai_compatible",
            api_key="dummy",
            base_url="https://example.invalid/v1",
            default_model=None,
            agent_model="agent-model",
            parser_model="parser-model",
        )
        config = {
            "backend": {
                "type": "openai_compatible",
                "agent_model": "agent-model",
                "parser_model": "parser-model",
            },
            "env_config": {
                "n_player": 7,
                "n_role": 4,
                "n_werewolf": 2,
                "n_seer": 1,
                "n_guard": 0,
                "n_witch": 1,
                "n_hunter": 0,
                "n_villager": 3,
            },
            "agent_config": {
                "village_team": {
                    "model_type": "gpt",
                    "model_params": {"temperature": 0.4},
                },
                "werewolf": {
                    "model_type": "twdm_agent",
                    "model_params": {
                        "model_name": "wolf-override-model",
                        "temperature": 0.1,
                        "twdm_config": {
                            "enable_strategy": True,
                            "enable_suspicion": False,
                            "enable_mcts": False,
                        },
                    },
                },
            },
        }

        env, agents, roles = build_runtime(
            parsed_yaml=config,
            log_save_path=None,
            backend=backend,
            backend_settings=settings,
            roles=ROLES,
        )

        self.assertEqual(roles, ROLES)
        self.assertIs(env.speech_perceiver.backend, backend)
        self.assertEqual(env.speech_perceiver.model_name, "parser-model")
        self.assertFalse(hasattr(env, "backend"))
        self.assertFalse(hasattr(env, "model_name"))
        self.assertEqual(len(agents), 7)
        for role, agent in zip(roles, agents):
            self.assertIs(agent.backend, backend)
            expected_model = (
                "wolf-override-model" if role == "Werewolf" else "agent-model"
            )
            self.assertEqual(agent.model_name, expected_model)
        self.assertEqual(backend.calls, [])

    @unittest.skipIf(
        build_random_runtime is None,
        "random runtime builder is not implemented",
    )
    def test_random_runtime_injects_backend_without_starting_game(self):
        backend = RecordingBackend()
        settings = BackendSettings(
            backend_type="openai_compatible",
            api_key="dummy",
            base_url="https://example.invalid/v1",
            default_model=None,
            agent_model="agent-model",
            parser_model="parser-model",
        )
        config = {
            "backend": {"type": "openai_compatible"},
            "env_config": {
                "n_player": 7,
                "n_role": 4,
                "n_werewolf": 2,
                "n_seer": 1,
                "n_guard": 0,
                "n_witch": 1,
                "n_hunter": 0,
                "n_villager": 3,
            },
            "agent_config": {
                "must_include": [],
                "all_candidates": [
                    {
                        "model_type": "gpt",
                        "model_params": {"temperature": 0.2},
                        "sample_ratio": 0.5,
                    },
                    {
                        "model_type": "deepseek",
                        "model_params": {"temperature": 0.2},
                        "sample_ratio": 0.5,
                    },
                ],
            },
        }

        env, agents, runtime_roles, role_models = build_random_runtime(
            parsed_yaml=config,
            log_save_path=None,
            backend=backend,
            backend_settings=settings,
            roles=RANDOM_ROLES,
            random_seed=3,
        )

        self.assertEqual(runtime_roles, RANDOM_ROLES)
        self.assertEqual(len(role_models), 7)
        self.assertEqual(len(agents), 7)
        self.assertIs(env.speech_perceiver.backend, backend)
        self.assertEqual(env.speech_perceiver.model_name, "parser-model")
        for agent in agents:
            self.assertIs(agent.backend, backend)
            self.assertEqual(agent.model_name, "agent-model")
        self.assertEqual(backend.calls, [])


if __name__ == "__main__":
    unittest.main()
