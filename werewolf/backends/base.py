from abc import ABC, abstractmethod


class BackendError(RuntimeError):
    """Base error raised by an LLM backend."""


class LLMBackend(ABC):
    @abstractmethod
    def chat(
        self,
        messages,
        model=None,
        temperature=0.7,
        max_tokens=None,
        response_format=None,
        **kwargs,
    ) -> str:
        raise NotImplementedError
