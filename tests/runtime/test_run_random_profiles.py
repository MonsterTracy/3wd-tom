import unittest
from unittest.mock import patch

from run_random import build_runtime


ROLES = [
    "Werewolf",
    "Villager",
    "Seer",
    "Witch",
    "Villager",
    "Villager",
    "Werewolf",
]


class RecordingBackend:
    def __init__(self, name):
        self.name = name
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
            }
        )
        return "[]"


def runtime_config():
    return {
        "backends": {
            "parser-api": {
                "type": "openai_compatible",
            },
            "agent-api-a": {
                "type": "openai_compatible",
            },
            "agent-api-b": {
                "type": "openai_compatible",
            },
        },
        "parser": {
            "backend": "parser-api",
            "model": "parser-model",
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
            "must_include": ["profile-b"],
            "all_candidates": [
                {
                    "profile_name": "profile-a",
                    "agent_type": "gpt",
                    "backend": "agent-api-a",
                    "model": "agent-model-a",
                    "model_params": {"temperature": 0.2},
                    "sample_ratio": 1.0,
                },
                {
                    "profile_name": "profile-b",
                    "agent_type": "gpt",
                    "backend": "agent-api-b",
                    "model": "agent-model-b",
                    "model_params": {"temperature": 0.7},
                    "sample_ratio": 1.0,
                },
            ],
        },
    }


class RunRandomProfileTest(unittest.TestCase):
    def setUp(self):
        self.parser_backend = RecordingBackend("parser")
        self.agent_backend_a = RecordingBackend("agent-a")
        self.agent_backend_b = RecordingBackend("agent-b")
        self.backends = {
            "parser-api": self.parser_backend,
            "agent-api-a": self.agent_backend_a,
            "agent-api-b": self.agent_backend_b,
        }

    def test_new_schema_wires_parser_and_profile_backends(self):
        env, agents, roles, profile_names = build_runtime(
            runtime_config(),
            log_save_path=None,
            roles=ROLES,
            random_seed=3,
            backends=self.backends,
        )

        self.assertEqual(roles, ROLES)
        self.assertIs(
            env.speech_perceiver.backend,
            self.parser_backend,
        )
        self.assertEqual(
            env.speech_perceiver.model_name,
            "parser-model",
        )
        self.assertEqual(profile_names[0], "profile-b")
        self.assertIn("profile-a", profile_names)
        for agent, profile_name in zip(agents, profile_names):
            if profile_name == "profile-a":
                self.assertIs(agent.backend, self.agent_backend_a)
                self.assertEqual(agent.model_name, "agent-model-a")
                self.assertEqual(agent.temperature, 0.2)
            else:
                self.assertIs(agent.backend, self.agent_backend_b)
                self.assertEqual(agent.model_name, "agent-model-b")
                self.assertEqual(agent.temperature, 0.7)

    @patch("run_random.load_named_backends")
    def test_default_runtime_path_loads_named_backends(
        self,
        load_named_backends,
    ):
        load_named_backends.return_value = self.backends

        build_runtime(
            runtime_config(),
            log_save_path=None,
            roles=ROLES,
            random_seed=3,
        )

        load_named_backends.assert_called_once()
        normalized = load_named_backends.call_args.args[0]
        self.assertIn("backends", normalized)
        self.assertEqual(
            normalized["parser"]["backend"],
            "parser-api",
        )

    def test_empty_eligible_profiles_raise_value_error(self):
        config = runtime_config()
        config["agent_config"]["must_include"] = []
        config["agent_config"]["all_candidates"] = [
            config["agent_config"]["all_candidates"][0]
        ]

        with self.assertRaises(ValueError) as raised:
            build_runtime(
                config,
                log_save_path=None,
                roles=ROLES,
                random_seed=3,
                backends=self.backends,
            )

        self.assertIn("eligible", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
