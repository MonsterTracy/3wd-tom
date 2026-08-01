from ipaddress import ip_address
from urllib.parse import urlparse

import httpx
import openai

from werewolf.backends.base import BackendError, LLMBackend


def _is_loopback_base_url(base_url) -> bool:
    if base_url is None:
        return False
    try:
        hostname = urlparse(str(base_url)).hostname
    except (TypeError, ValueError):
        return False
    if hostname is None:
        return False

    normalized = hostname.lower()
    if normalized == "localhost":
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


class OpenAICompatibleBackend(LLMBackend):
    def __init__(
        self,
        api_key=None,
        base_url=None,
        default_model=None,
        client=None,
        max_retries=None,
        supports_json_schema=False,
    ):
        if not isinstance(
            supports_json_schema,
            bool,
        ):
            raise TypeError(
                "supports_json_schema must be boolean"
            )
        if client is None:
            if not api_key:
                raise BackendError(
                    "api_key is required when an OpenAI-compatible client is not injected."
                )
            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            if max_retries is not None:
                client_kwargs["max_retries"] = max_retries
            if _is_loopback_base_url(base_url):
                client_kwargs["http_client"] = httpx.Client(
                    trust_env=False,
                    timeout=openai.DEFAULT_TIMEOUT,
                )
            client = openai.OpenAI(**client_kwargs)

        self.client = client
        self.base_url = base_url
        self.chat_completions_endpoint = (
            f"{base_url.rstrip('/')}/chat/completions"
            if base_url
            else None
        )
        self.default_model = default_model
        self.supports_json_schema = (
            supports_json_schema
        )

    def chat(
        self,
        messages,
        model=None,
        temperature=0.7,
        max_tokens=None,
        response_format=None,
        **kwargs,
    ) -> str:
        content, _usage = self.chat_with_metadata(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            **kwargs,
        )
        return content

    def chat_with_metadata(
        self,
        messages,
        model=None,
        temperature=0.7,
        max_tokens=None,
        response_format=None,
        **kwargs,
    ) -> tuple[str, dict[str, int | str | None] | None]:
        """Return text plus provider-reported token usage when available."""

        resolved_model = model or self.default_model
        if not resolved_model:
            raise BackendError("A model is required for an LLM chat request.")

        request = dict(kwargs)
        request["model"] = resolved_model
        request["messages"] = messages
        if temperature is not None:
            request["temperature"] = temperature
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if response_format is not None:
            request["response_format"] = response_format

        try:
            response = self.client.chat.completions.create(**request)
            content = response.choices[0].message.content
            if not isinstance(content, str):
                raise BackendError(
                    "OpenAI-compatible chat response content must be text."
                )
            metadata = _extract_usage(
                getattr(response, "usage", None)
            )
            finish_reason = getattr(
                response.choices[0],
                "finish_reason",
                None,
            )
            if isinstance(finish_reason, str):
                metadata = dict(metadata or {})
                metadata["finish_reason"] = finish_reason
            return content, metadata
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(
                "OpenAI-compatible chat request failed."
            ) from exc


def _extract_usage(usage):
    """Normalize OpenAI and Responses-style token names without guessing."""

    if usage is None:
        return None

    def read(*names):
        for name in names:
            if isinstance(usage, dict):
                value = usage.get(name)
            else:
                value = getattr(usage, name, None)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        return None

    return {
        "input_tokens": read("prompt_tokens", "input_tokens"),
        "output_tokens": read("completion_tokens", "output_tokens"),
        "total_tokens": read("total_tokens"),
    }
