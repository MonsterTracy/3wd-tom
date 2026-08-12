"""Stateless reporter over the exact Public Belief Matrix visible prefix."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from werewolf.models.public_belief_matrix.public_prefix import (
    render_public_belief_matrix_visible_prefix,
)
from werewolf.models.twd_tom.schema import PLAYER_NAMES, normalize_player, validate_player_suspicion
from werewolf.speech.private_belief_perceiver import (
    PRIVATE_BELIEF_MAX_TOKENS,
    PlayingAgentBeliefReporter,
    private_belief_response_format,
)

PUBLIC_BELIEF_MATRIX_PROMPT_VERSION = "classic7_public_belief_matrix_structured_prompt_v1"


class PublicBeliefMatrixReporter:
    """Request one symbolic row without accepting raw or private state."""

    def __init__(self, audit_hook=None) -> None:
        self.audit_hook = audit_hook

    @staticmethod
    def build_prompt(*, visible_prefix, observer_id: int | str) -> str:
        observer = normalize_player(observer_id)
        history = render_public_belief_matrix_visible_prefix(visible_prefix)
        return f"""这是独立、无状态的 Public Belief Matrix 标注请求。
仅根据下列公开、结构化且已截断的历史，从 observer={observer} 的视角报告怀疑的狼人。
你没有真实身份、私有角色、狼人队友、查验、女巫/夜间信息、私人记忆、raw speech 或未来事件。
允许空集合、允许任意合法集合大小，也允许怀疑 observer 自己；不要强迫恰好两名狼人。
合法玩家仅为：{', '.join(PLAYER_NAMES)}。不得输出概率、理由、推理或额外字段。
prompt_version: {PUBLIC_BELIEF_MATRIX_PROMPT_VERSION}
structured_public_prefix: {history}
Return only: {{"suspected_werewolves":[...]}}"""

    def report(self, *, visible_prefix, observer_id, cutoff, backend, backend_id, model_name):
        observer = normalize_player(observer_id)
        if backend is None or not hasattr(backend, "chat"):
            raise TypeError("backend must provide chat()")
        if not isinstance(backend_id, str) or not backend_id.strip():
            raise ValueError("backend_id must be non-empty text")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be non-empty text")
        prompt = self.build_prompt(
            visible_prefix=visible_prefix,
            observer_id=observer,
        )
        report_id = None
        try:
            if self.audit_hook is not None:
                report_id, prompt = self.audit_hook.prepare_report(
                    observer_id=observer,
                    public_snapshot=cutoff,
                    agent_backend_id=backend_id,
                    agent_model_id=model_name,
                    report_prompt=prompt,
                    decorate_prompt=False,
                )
            context = self.audit_hook.belief_context(report_id) if self.audit_hook else nullcontext()
            with context:
                raw = backend.chat(
                    messages=[{"role": "user", "content": prompt}],
                    model=model_name,
                    temperature=0.0,
                    max_tokens=PRIVATE_BELIEF_MAX_TOKENS,
                    response_format=private_belief_response_format(
                        supports_json_schema=getattr(backend, "supports_json_schema", False)
                    ),
                    extra_body={"thinking": {"type": "disabled"}},
                )
        except Exception as exc:
            if self.audit_hook is not None and report_id is not None:
                self.audit_hook.complete_report(report_id, None)
            return self._result(observer, "reporter_error", None, str(exc), backend_id)
        if self.audit_hook is not None:
            self.audit_hook.complete_report(report_id, raw)
        try:
            suspected = PlayingAgentBeliefReporter.parse_response(raw)
            suspected = validate_player_suspicion(suspected, [], [])
        except (TypeError, ValueError) as exc:
            return self._result(observer, "parse_error", None, str(exc), backend_id)
        return self._result(observer, "ok", suspected, None, backend_id)

    @staticmethod
    def _result(observer, status, suspicion, error, backend_id) -> dict[str, Any]:
        return {
            "observer": observer,
            "status": status,
            "suspected_werewolves": suspicion,
            "error": error,
            "reporter_backend_id": backend_id,
        }


__all__ = ["PUBLIC_BELIEF_MATRIX_PROMPT_VERSION", "PublicBeliefMatrixReporter"]
