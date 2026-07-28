from copy import deepcopy
import os
import unittest
from unittest.mock import call, patch

from werewolf.backends import (
    load_named_backends,
    resolve_backend,
)
from werewolf.runtime_config import normalize_runtime_config


def new_config():
    return {
        "backends": {
            "deepseek": {
                "type": "openai_compatible",
                "base_url": "https://api.deepseek.com",
                "api_key_env": "DEEPSEEK_API_KEY",
                "default_model": "deepseek-chat",
            },
            "openai": {
                "type": "openai_compatible",
                "api_key_env": "OPENAI_API_KEY",
                "default_model": "gpt-4o-mini",
            },
        },
        "parser": {
            "backend": "deepseek",
            "model": "deepseek-chat",
        },
        "agent_config": {},
    }


def legacy_config():
    return {
        "backend": {
            "type": "openai_compatible",
            "base_url": "https://legacy.example/v1",
            "default_model": "legacy-model",
        },
        "agent_config": {},
    }


class NamedBackendFactoryTest(unittest.TestCase):
    @patch("werewolf.backends.factory.OpenAICompatibleBackend")
    def test_multiple_backends_use_different_api_key_envs(
        self,
        backend_class,
    ):
        deepseek_backend = object()
        openai_backend = object()
        backend_class.side_effect = [
            deepseek_backend,
            openai_backend,
        ]
        env = {
            "DEEPSEEK_API_KEY": "deepseek-secret",
            "OPENAI_API_KEY": "openai-secret",
        }

        with patch.dict(os.environ, env, clear=True):
            backends = load_named_backends(
                new_config(),
                env_file=None,
            )

        self.assertEqual(
            backends,
            {
                "deepseek": deepseek_backend,
                "openai": openai_backend,
            },
        )
        self.assertEqual(
            backend_class.call_args_list,
            [
                call(
                    api_key="deepseek-secret",
                    base_url="https://api.deepseek.com",
                    default_model="deepseek-chat",
                ),
                call(
                    api_key="openai-secret",
                    base_url=None,
                    default_model="gpt-4o-mini",
                ),
            ],
        )

    @patch("werewolf.backends.factory.OpenAICompatibleBackend")
    def test_missing_api_key_env_field_defaults_to_openai_key(
        self,
        backend_class,
    ):
        config = new_config()
        config["backends"] = {
            "defaulted": {
                "type": "openai_compatible",
            }
        }
        config["parser"]["backend"] = "defaulted"
        backend = object()
        backend_class.return_value = backend

        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "default-secret"},
            clear=True,
        ):
            result = load_named_backends(
                config,
                env_file=None,
            )

        self.assertIs(result["defaulted"], backend)
        backend_class.assert_called_once_with(
            api_key="default-secret",
            base_url=None,
            default_model=None,
        )

    def test_missing_or_empty_api_key_raises(self):
        for env in ({}, {"DEEPSEEK_API_KEY": "   "}):
            with self.subTest(env=env):
                with patch.dict(os.environ, env, clear=True):
                    with self.assertRaises(ValueError) as raised:
                        load_named_backends(
                            new_config(),
                            env_file=None,
                        )

                self.assertIn(
                    "DEEPSEEK_API_KEY",
                    str(raised.exception),
                )

    @patch("werewolf.backends.factory.OpenAICompatibleBackend")
    def test_error_message_does_not_leak_resolved_key(
        self,
        backend_class,
    ):
        config = new_config()
        config["backends"]["openai"][
            "api_key_env"
        ] = "MISSING_OPENAI_KEY"

        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "do-not-leak-this-secret"},
            clear=True,
        ):
            with self.assertRaises(ValueError) as raised:
                load_named_backends(config, env_file=None)

        message = str(raised.exception)
        self.assertIn("MISSING_OPENAI_KEY", message)
        self.assertNotIn("do-not-leak-this-secret", message)
        self.assertNotIn("deepseek-secret", message)
        self.assertEqual(backend_class.call_count, 1)

    def test_unknown_backend_type_raises(self):
        config = new_config()
        config["backends"]["deepseek"]["type"] = "gemini"

        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "secret"},
            clear=True,
        ):
            with self.assertRaises(ValueError):
                load_named_backends(config, env_file=None)

    def test_resolve_backend_returns_exact_instance(self):
        backend = object()

        resolved = resolve_backend(
            "deepseek",
            {"deepseek": backend},
        )

        self.assertIs(resolved, backend)

    def test_resolve_unknown_backend_lists_available_names(self):
        with self.assertRaises(ValueError) as raised:
            resolve_backend(
                "missing",
                {
                    "qwen": object(),
                    "deepseek": object(),
                },
            )

        message = str(raised.exception)
        self.assertIn("missing", message)
        self.assertIn("deepseek, qwen", message)

    @patch("werewolf.backends.factory.OpenAICompatibleBackend")
    def test_input_config_is_not_modified(self, backend_class):
        config = new_config()
        original = deepcopy(config)

        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "deepseek-secret",
                "OPENAI_API_KEY": "openai-secret",
            },
            clear=True,
        ):
            load_named_backends(config, env_file=None)

        self.assertEqual(config, original)

    @patch("werewolf.backends.factory.OpenAICompatibleBackend")
    def test_normalized_config_is_accepted(self, backend_class):
        config = normalize_runtime_config(new_config())

        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "deepseek-secret",
                "OPENAI_API_KEY": "openai-secret",
            },
            clear=True,
        ):
            backends = load_named_backends(
                config,
                env_file=None,
            )

        self.assertEqual(
            set(backends),
            {"deepseek", "openai"},
        )

    @patch("werewolf.backends.factory.OpenAICompatibleBackend")
    def test_legacy_schema_uses_default_named_backend(
        self,
        backend_class,
    ):
        backend = object()
        backend_class.return_value = backend

        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "legacy-secret"},
            clear=True,
        ):
            backends = load_named_backends(
                legacy_config(),
                env_file=None,
            )

        self.assertEqual(backends, {"default": backend})
        backend_class.assert_called_once_with(
            api_key="legacy-secret",
            base_url="https://legacy.example/v1",
            default_model="legacy-model",
        )


if __name__ == "__main__":
    unittest.main()
