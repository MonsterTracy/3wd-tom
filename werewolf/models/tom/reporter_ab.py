"""Audit-only same-observation comparison for the formal belief reporter."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from werewolf.models.tom.reporter import BeliefReporter
from werewolf.models.tom.schema import normalize_player
from werewolf.speech.private_belief_perceiver import (
    PRIVATE_BELIEF_MAX_TOKENS,
    STATUS_SEMANTIC_ERROR,
)


DEEPSEEK_REPORTER_AB_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_REPORTER_AB_MODEL = "deepseek-v4-flash"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class ReporterABAudit:
    """Write one independent Qwen/DeepSeek comparison per observer."""

    def __init__(self, output_path: str | Path, *, deepseek_backend) -> None:
        if deepseek_backend is None or not hasattr(deepseek_backend, "chat"):
            raise TypeError("DeepSeek audit backend must provide chat()")
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._output = path.open("a", encoding="utf-8")
        self.deepseek_backend = deepseek_backend

    @staticmethod
    def _result(
        *,
        valid: bool,
        suspected: list[str] | None = None,
        error: str | None,
    ) -> dict[str, Any]:
        return {
            "valid": valid,
            "suspected_werewolves": suspected,
            "error": error,
        }

    def _deepseek_report(
        self,
        *,
        prompt: str,
        hard_knowledge: Mapping[str, list[str]],
    ) -> dict[str, Any]:
        transport_prompt = (
            "Output the response in JSON format only.\n\n"
            + prompt
        )
        try:
            raw = self.deepseek_backend.chat(
                messages=[{"role": "user", "content": transport_prompt}],
                model=DEEPSEEK_REPORTER_AB_MODEL,
                temperature=0.0,
                max_tokens=PRIVATE_BELIEF_MAX_TOKENS,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception:
            return self._result(valid=False, error="reporter_error")
        try:
            suspected = BeliefReporter.parse(raw)
        except (TypeError, ValueError):
            return self._result(valid=False, error="parse_error")
        try:
            BeliefReporter.validate_hard_knowledge(
                suspected,
                hard_knowledge,
            )
        except (TypeError, ValueError):
            return self._result(valid=False, error=STATUS_SEMANTIC_ERROR)
        return self._result(
            valid=True,
            suspected=suspected,
            error=None,
        )

    def record(
        self,
        *,
        game_id: str,
        step_idx: int,
        speaker_id: int | str,
        observer_id: int | str,
        phase: str,
        observation: Mapping[str, Any],
        qwen_result: Mapping[str, Any],
    ) -> dict[str, Any]:
        observer = normalize_player(observer_id)
        legal_state = BeliefReporter.legal_state(observer, observation)
        hard_knowledge = BeliefReporter.derive_hard_knowledge(
            observer,
            observation,
        )
        prompt = BeliefReporter.build_prompt(observer, observation)
        canonical_state = json.dumps(
            legal_state,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        deepseek_result = self._deepseek_report(
            prompt=prompt,
            hard_knowledge=hard_knowledge,
        )
        row = {
            "game_id": game_id,
            "step_idx": step_idx,
            "speaker_id": normalize_player(speaker_id),
            "observer_id": observer,
            "observer_role": legal_state["self_role"],
            "phase": phase,
            "observation_digest": _digest(canonical_state),
            "prompt_digest": _digest(prompt),
            "hard_knowledge": hard_knowledge,
            "qwen": {
                "valid": qwen_result.get("valid") is True,
                "suspected_werewolves": qwen_result.get(
                    "suspected_werewolves"
                ),
                "error": qwen_result.get("error"),
            },
            "deepseek": deepseek_result,
        }
        self._output.write(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        )
        self._output.flush()
        return row

    def close(self) -> None:
        if not self._output.closed:
            self._output.close()


__all__ = [
    "DEEPSEEK_REPORTER_AB_BASE_URL",
    "DEEPSEEK_REPORTER_AB_MODEL",
    "ReporterABAudit",
]
