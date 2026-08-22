"""Readonly playing-agent Werewolf belief self-reporting."""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import nullcontext
from copy import deepcopy
from typing import Any


STATUS_OK = "ok"
STATUS_PARSE_ERROR = "parse_error"
STATUS_SEMANTIC_ERROR = "semantic_error"
STATUS_REPORTER_ERROR = "reporter_error"

from werewolf.models.twd_tom.public_events import (
    copy_public_events,
)
from werewolf.models.twd_tom.schema import (
    LABEL_PROMPT_VERSION,
    PLAYER_NAMES,
    canonicalize_player_set,
    normalize_player,
)


PRIVATE_BELIEF_MAX_TOKENS = 96
PRIVATE_BELIEF_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "suspected_werewolves",
    ],
    "properties": {
        "suspected_werewolves": {
            "type": "array",
            "minItems": 0,
            "maxItems": 7,
            "items": {
                "type": "string",
                "enum": list(
                    PLAYER_NAMES
                ),
            },
        },
    },
}


def private_belief_response_format(
    *,
    supports_json_schema: bool,
) -> dict[str, Any]:
    """Build the provider request format without weakening local validation."""

    if not isinstance(
        supports_json_schema,
        bool,
    ):
        raise TypeError(
            "supports_json_schema must be boolean"
        )
    if not supports_json_schema:
        return {
            "type": "json_object",
        }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "private_belief_report",
            "strict": True,
            "schema": deepcopy(
                PRIVATE_BELIEF_JSON_SCHEMA
            ),
        },
    }


class PlayingAgentBeliefReporter:
    """Request one detached report from the target playing agent."""

    def __init__(self, audit_hook=None) -> None:
        self.audit_hook = audit_hook

    def report(
        self,
        *,
        agent,
        observation: Mapping[str, Any],
        observer_id: int | str,
        public_snapshot,
        agent_backend_id: str,
        known_werewolves: list[str],
        known_non_werewolves: list[str],
    ) -> dict[str, Any]:
        observer = normalize_player(observer_id)
        if not isinstance(observation, Mapping):
            return self._result(
                observer=observer,
                status=STATUS_REPORTER_ERROR,
                error="observation must be a mapping",
                agent_backend_id=agent_backend_id,
                known_werewolves=known_werewolves,
                known_non_werewolves=known_non_werewolves,
            )
        if normalize_player(observation.get("observer_id")) != observer:
            return self._result(
                observer=observer,
                status=STATUS_REPORTER_ERROR,
                error="observation does not belong to the requested observer",
                agent_backend_id=agent_backend_id,
                known_werewolves=known_werewolves,
                known_non_werewolves=known_non_werewolves,
            )
        if observation.get("current_act_idx") != public_snapshot.speaker_id:
            return self._result(
                observer=observer,
                status=STATUS_REPORTER_ERROR,
                error="observation and public snapshot speaker mismatch",
                agent_backend_id=agent_backend_id,
                known_werewolves=known_werewolves,
                known_non_werewolves=known_non_werewolves,
            )
        if observation.get("phase") != public_snapshot.phase:
            return self._result(
                observer=observer,
                status=STATUS_REPORTER_ERROR,
                error="observation and public snapshot phase mismatch",
                agent_backend_id=agent_backend_id,
                known_werewolves=known_werewolves,
                known_non_werewolves=known_non_werewolves,
            )
        if not hasattr(agent, "report_suspected_werewolves_readonly"):
            return self._result(
                observer=observer,
                status=STATUS_REPORTER_ERROR,
                error="playing agent does not support readonly belief reports",
                agent_backend_id=agent_backend_id,
                known_werewolves=known_werewolves,
                known_non_werewolves=known_non_werewolves,
            )

        report_id = None
        try:
            report_prompt = self.build_prompt(
                observer_id=observer,
                public_snapshot=public_snapshot,
                known_werewolves=known_werewolves,
                known_non_werewolves=known_non_werewolves,
            )
            if self.audit_hook is not None:
                report_id, report_prompt = self.audit_hook.prepare_report(
                    observer_id=observer,
                    public_snapshot=public_snapshot,
                    agent_backend_id=agent_backend_id,
                    agent_model_id=getattr(agent, "model_name", None),
                    report_prompt=report_prompt,
                )
            audit_context = (
                self.audit_hook.belief_context(report_id)
                if self.audit_hook is not None
                else nullcontext()
            )
            with audit_context:
                raw_response = agent.report_suspected_werewolves_readonly(
                    observation=observation,
                    report_prompt=report_prompt,
                )
        except Exception as exc:
            if self.audit_hook is not None and report_id is not None:
                self.audit_hook.complete_report(report_id, None)
            return self._result(
                observer=observer,
                status=STATUS_REPORTER_ERROR,
                error=str(exc),
                agent_backend_id=agent_backend_id,
                known_werewolves=known_werewolves,
                known_non_werewolves=known_non_werewolves,
            )

        if self.audit_hook is not None:
            self.audit_hook.complete_report(report_id, raw_response)

        try:
            suspected = self.parse_response(raw_response)
        except (TypeError, ValueError) as exc:
            return self._result(
                observer=observer,
                status=STATUS_PARSE_ERROR,
                error=str(exc),
                agent_backend_id=agent_backend_id,
                known_werewolves=known_werewolves,
                known_non_werewolves=known_non_werewolves,
            )

        if observer in suspected:
            return self._result(
                observer=observer,
                status=STATUS_SEMANTIC_ERROR,
                error="suspected_werewolves cannot contain the observer",
                agent_backend_id=agent_backend_id,
                known_werewolves=known_werewolves,
                known_non_werewolves=known_non_werewolves,
            )

        return self._result(
            observer=observer,
            status=STATUS_OK,
            suspected_werewolves=suspected,
            error=None,
            agent_backend_id=agent_backend_id,
            known_werewolves=known_werewolves,
            known_non_werewolves=known_non_werewolves,
        )

    def record_agent_state(
        self,
        *,
        observer_id: str,
        state_before,
        state_after,
    ) -> None:
        """Forward readonly state snapshots only when auditing is enabled."""

        if self.audit_hook is not None:
            self.audit_hook.record_agent_state(
                observer_id=observer_id,
                state_before=state_before,
                state_after=state_after,
            )

    @staticmethod
    def build_prompt(
        *,
        observer_id: int | str,
        public_snapshot,
        known_werewolves: list[str],
        known_non_werewolves: list[str],
    ) -> str:
        """Render the frozen private self-report instruction."""

        observer = normalize_player(observer_id)
        digest = getattr(public_snapshot, "public_history_digest", None)
        action_count = getattr(public_snapshot, "public_action_count", None)
        if not isinstance(digest, str) or not digest:
            raise ValueError("public snapshot requires a history digest")
        if isinstance(action_count, bool) or not isinstance(action_count, int):
            raise ValueError("public snapshot requires public_action_count")
        canonical_identifiers = list(PLAYER_NAMES)
        canonical_list = ", ".join(canonical_identifiers)
        canonicalize_player_set(
            known_werewolves,
            field_name="known_werewolves",
        )
        canonicalize_player_set(
            known_non_werewolves,
            field_name="known_non_werewolves",
        )
        public_history = json.dumps(
            copy_public_events(
                public_snapshot.public_events
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return f"""这是一个私有的训练标注请求，不是游戏中的公开发言。
其他玩家不会看到你的回答。回答不会写入你的游戏记忆，也不会影响你的下一次发言、投票或技能行动。

请只根据你截至本次公开发言之前合法拥有的公开历史和自己的私人信息，报告你当前内部真实怀疑哪些具体玩家是狼人。不要为了阵营策略欺骗这个私有 reporter。不得使用 god view、actual roles、其他玩家私人信息、当前尚未生成的发言或未来信息。

Your observer identity is exactly: {observer}
Canonical player IDs (complete ordered list):
{canonical_list}
Canonical pre-speech public_events:
{public_history}

`suspected_werewolves` 是玩家级的相对怀疑集合，不是完整双狼人组合约束。只报告你当前认为相对更可疑的其他玩家，不得包含 observer 自己；不要求确定性、不要求完整找到两狼，也不要求恰好两人。不要仅因为某人“仍有可能是狼”就将其列入。允许列出 0、1、2 或更多人；若确实怀疑所有其他玩家，允许全部列出；没有怀疑对象时输出空数组。只允许上面的 canonical player IDs，不得重复。

Before answering, silently verify this checklist:
- 这是我的内部真实怀疑，而不是公开发言策略吗？
- 是否把“仍可能”误当成“当前值得怀疑”？
- 是否错误地强行补足两人？
- 是否只输出允许的 JSON？

Do not output probabilities, confidence, scores, rankings, reasons, the checklist, reasoning, or chain-of-thought. Do not output any extra field.

observer: {observer}
prompt_version: {LABEL_PROMPT_VERSION}
public_action_count: {action_count}
public_history_digest: {digest}

Return only this JSON structure:
{{"suspected_werewolves":[...]}}"""

    @staticmethod
    def parse_response(
        raw_response: Any,
    ) -> list[str]:
        """Parse the sole player-suspicion report protocol."""

        if not isinstance(raw_response, str):
            raise TypeError("reporter response must be text")
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise ValueError("reporter response must be a pure JSON object") from exc
        if not isinstance(parsed, dict):
            raise TypeError("reporter response must be a JSON object")
        if set(parsed) != {"suspected_werewolves"}:
            raise ValueError(
                "reporter response requires only suspected_werewolves"
            )
        return canonicalize_player_set(
            parsed["suspected_werewolves"],
            field_name="suspected_werewolves",
        )

    @staticmethod
    def _result(
        *,
        observer: str,
        status: str,
        agent_backend_id: str,
        suspected_werewolves: list[str] | None = None,
        error: str | None,
        known_werewolves: list[str],
        known_non_werewolves: list[str],
    ) -> dict[str, Any]:
        if not isinstance(agent_backend_id, str) or not agent_backend_id.strip():
            raise ValueError("agent_backend_id must be non-empty text")
        return {
            "observer": observer,
            "status": status,
            "suspected_werewolves": suspected_werewolves,
            "known_werewolves": list(known_werewolves),
            "known_non_werewolves": list(known_non_werewolves),
            "error": error,
            "agent_backend_id": agent_backend_id,
        }


__all__ = [
    "STATUS_OK",
    "STATUS_PARSE_ERROR",
    "STATUS_SEMANTIC_ERROR",
    "STATUS_REPORTER_ERROR",
    "PRIVATE_BELIEF_MAX_TOKENS",
    "PRIVATE_BELIEF_JSON_SCHEMA",
    "private_belief_response_format",
    "PlayingAgentBeliefReporter",
]
