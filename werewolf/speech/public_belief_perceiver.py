"""Stateless public-only Werewolf belief reporting."""

from __future__ import annotations

import json
from contextlib import nullcontext
from typing import Any

from werewolf.models.twd_tom.public_events import copy_public_events
from werewolf.models.twd_tom.schema import (
    PLAYER_NAMES,
    PUBLIC_ONLY_LABEL_PROMPT_VERSION,
    normalize_player,
)
from werewolf.speech.private_belief_perceiver import (
    PRIVATE_BELIEF_MAX_TOKENS,
    PlayingAgentBeliefReporter,
    STATUS_OK,
    STATUS_PARSE_ERROR,
    STATUS_REPORTER_ERROR,
    private_belief_response_format,
)


class PublicOnlyBeliefReporter:
    """Request one belief using only a frozen public snapshot."""

    def __init__(self, audit_hook=None) -> None:
        self.audit_hook = audit_hook

    def report(
        self,
        *,
        public_snapshot,
        observer_id: int | str,
        backend,
        backend_id: str,
        model_name: str,
    ) -> dict[str, Any]:
        observer = normalize_player(observer_id)
        if not isinstance(backend_id, str) or not backend_id.strip():
            raise ValueError("backend_id must be non-empty text")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be non-empty text")
        if backend is None or not hasattr(backend, "chat"):
            raise TypeError("backend must provide chat()")

        report_id = None
        try:
            report_prompt = self.build_prompt(
                public_snapshot=public_snapshot,
                observer_id=observer,
            )
            if self.audit_hook is not None:
                report_id, report_prompt = self.audit_hook.prepare_report(
                    observer_id=observer,
                    public_snapshot=public_snapshot,
                    agent_backend_id=backend_id,
                    agent_model_id=model_name,
                    report_prompt=report_prompt,
                )
            audit_context = (
                self.audit_hook.belief_context(report_id)
                if self.audit_hook is not None
                else nullcontext()
            )
            with audit_context:
                raw_response = backend.chat(
                    messages=[{"role": "user", "content": report_prompt}],
                    model=model_name,
                    temperature=0.0,
                    max_tokens=PRIVATE_BELIEF_MAX_TOKENS,
                    response_format=private_belief_response_format(
                        supports_json_schema=getattr(
                            backend,
                            "supports_json_schema",
                            False,
                        )
                    ),
                    extra_body={"thinking": {"type": "disabled"}},
                )
        except Exception as exc:
            if self.audit_hook is not None and report_id is not None:
                self.audit_hook.complete_report(report_id, None)
            return self._result(
                observer=observer,
                status=STATUS_REPORTER_ERROR,
                error=str(exc),
                backend_id=backend_id,
            )

        if self.audit_hook is not None:
            self.audit_hook.complete_report(report_id, raw_response)

        try:
            suspected = PlayingAgentBeliefReporter.parse_response(raw_response)
        except (TypeError, ValueError) as exc:
            return self._result(
                observer=observer,
                status=STATUS_PARSE_ERROR,
                error=str(exc),
                backend_id=backend_id,
            )
        return self._result(
            observer=observer,
            status=STATUS_OK,
            suspected_werewolves=suspected,
            error=None,
            backend_id=backend_id,
        )

    @staticmethod
    def build_prompt(*, public_snapshot, observer_id: int | str) -> str:
        observer = normalize_player(observer_id)
        public_history = json.dumps(
            copy_public_events(public_snapshot.public_events),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        canonical_players = ", ".join(PLAYER_NAMES)
        return f"""这是一个独立、无状态的 public-only 训练标注请求，不是游戏中的公开发言。
你只能使用下面截至 cutoff 的公开事件。你不知道任何玩家的真实角色，也没有狼人队友、查验、女巫/夜间信息、私人记忆或 hard private knowledge。当前尚未生成的发言和未来事件不可使用。

Your observer identity is exactly: {observer}
Game ID: {public_snapshot.game_id}
Cutoff step: {public_snapshot.step_idx}
Current phase: {public_snapshot.phase}
Current public speaker: player{public_snapshot.speaker_id}
Canonical player IDs: {canonical_players}
Canonical pre-speech public_events:
{public_history}

`suspected_werewolves` 是仅基于公开信息的相对怀疑集合。允许列出 0、1、2 或更多玩家；不要为了凑足两人而补充候选人。
只允许 canonical player IDs，不得重复。不要输出概率、置信度、理由、推理过程或任何额外字段。

observer: {observer}
prompt_version: {PUBLIC_ONLY_LABEL_PROMPT_VERSION}
public_action_count: {public_snapshot.public_action_count}
public_history_digest: {public_snapshot.public_history_digest}

Return only this JSON structure:
{{"suspected_werewolves":[...]}}"""

    @staticmethod
    def _result(
        *,
        observer: str,
        status: str,
        backend_id: str,
        suspected_werewolves: list[str] | None = None,
        error: str | None,
    ) -> dict[str, Any]:
        return {
            "observer": observer,
            "status": status,
            "suspected_werewolves": suspected_werewolves,
            "known_werewolves": [],
            "known_non_werewolves": [],
            "error": error,
            "agent_backend_id": backend_id,
        }


__all__ = ["PublicOnlyBeliefReporter"]
