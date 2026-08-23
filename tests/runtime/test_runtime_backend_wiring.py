import unittest

from werewolf.backends import BackendSettings

try:
    from run_random import build_runtime as build_random_runtime
except ImportError:
    build_random_runtime = None


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
    def test_random_runtime_builder_is_available(self):
        self.assertIsNotNone(build_random_runtime)


class RuntimeBackendWiringTest(unittest.TestCase):
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
