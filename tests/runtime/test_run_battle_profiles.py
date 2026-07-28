import unittest
from unittest.mock import patch

from run_battle import build_runtime


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
            "parser-api": {"type": "openai_compatible"},
            "wolf-api": {"type": "openai_compatible"},
            "village-api": {"type": "openai_compatible"},
            "replace-api": {"type": "openai_compatible"},
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
            "werewolf": {
                "profile_name": "wolf-profile",
                "agent_type": "twdm_agent",
                "backend": "wolf-api",
                "model": "wolf-model",
                "model_params": {
                    "temperature": 0.2,
                    "twdm_config": {
                        "enable_strategy": True,
                    },
                },
            },
            "village_team": {
                "profile_name": "village-profile",
                "agent_type": "gpt",
                "backend": "village-api",
                "model": "village-model",
                "model_params": {"temperature": 0.7},
            },
        },
    }


class RunBattleProfileTest(unittest.TestCase):
    def setUp(self):
        self.parser_backend = RecordingBackend("parser")
        self.wolf_backend = RecordingBackend("wolf")
        self.village_backend = RecordingBackend("village")
        self.replace_backend = RecordingBackend("replace")
        self.backends = {
            "parser-api": self.parser_backend,
            "wolf-api": self.wolf_backend,
            "village-api": self.village_backend,
            "replace-api": self.replace_backend,
        }

    def test_new_schema_wires_parser_and_role_profile_backends(self):
        env, agents, roles = build_runtime(
            runtime_config(),
            log_save_path=None,
            roles=ROLES,
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
        for role, agent in zip(roles, agents):
            if role == "Werewolf":
                self.assertIs(agent.backend, self.wolf_backend)
                self.assertEqual(agent.model_name, "wolf-model")
                self.assertEqual(agent.temperature, 0.2)
            else:
                self.assertIs(agent.backend, self.village_backend)
                self.assertEqual(agent.model_name, "village-model")
                self.assertEqual(agent.temperature, 0.7)

    @patch("run_battle.load_named_backends")
    def test_default_runtime_path_loads_named_backends(
        self,
        load_named_backends,
    ):
        load_named_backends.return_value = self.backends

        build_runtime(
            runtime_config(),
            log_save_path=None,
            roles=ROLES,
        )

        load_named_backends.assert_called_once()
        normalized = load_named_backends.call_args.args[0]
        self.assertEqual(
            normalized["parser"]["backend"],
            "parser-api",
        )
        self.assertEqual(
            normalized["agent_config"]["werewolf"][
                "profile_name"
            ],
            "wolf-profile",
        )

    def test_replace_inline_profile_uses_its_backend_and_model(self):
        config = runtime_config()
        config["agent_config"]["replace"] = {
            "profile_name": "replacement-profile",
            "agent_type": "gpt",
            "backend": "replace-api",
            "model": "replacement-model",
            "model_params": {"temperature": 0.4},
            "replace_player": "werewolf_last",
        }

        _, agents, _ = build_runtime(
            config,
            log_save_path=None,
            roles=ROLES,
            backends=self.backends,
        )

        self.assertIs(agents[0].backend, self.wolf_backend)
        self.assertIs(agents[6].backend, self.replace_backend)
        self.assertEqual(
            agents[6].model_name,
            "replacement-model",
        )


if __name__ == "__main__":
    unittest.main()
