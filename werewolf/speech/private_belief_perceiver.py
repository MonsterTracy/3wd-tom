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
    validate_player_suspicion,
)


PRIVATE_BELIEF_MAX_TOKENS = 96
LABEL_GENERATION_MAX_ATTEMPTS = 3


def _private_belief_json_schema(
    legal_candidates: list[str] | tuple[str, ...],
    required_candidates: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    candidates = canonicalize_player_set(
        list(legal_candidates),
        field_name="legal_candidates",
    )
    required = canonicalize_player_set(
        list(required_candidates),
        field_name="required_candidates",
    )
    if not set(required).issubset(candidates):
        raise ValueError("required_candidates must be legal candidates")
    items: dict[str, Any] = {"type": "string"}
    if candidates:
        items["enum"] = candidates
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["suspected_werewolves"],
        "properties": {
            "suspected_werewolves": {
                "type": "array",
                "minItems": len(required),
                "maxItems": len(candidates),
                "items": items,
            },
        },
    }


PRIVATE_BELIEF_JSON_SCHEMA = _private_belief_json_schema(
    PLAYER_NAMES,
)


def private_belief_response_format(
    *,
    supports_json_schema: bool,
    legal_candidates: list[str] | tuple[str, ...] = PLAYER_NAMES,
    required_candidates: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build the provider request format without weakening local validation."""

    if not isinstance(
        supports_json_schema,
        bool,
    ):
        raise TypeError(
            "supports_json_schema must be boolean"
        )
    schema = _private_belief_json_schema(
        legal_candidates,
        required_candidates,
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
            "schema": deepcopy(schema),
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
            legal_candidates = self.legal_candidates(
                observer_id=observer,
                known_werewolves=known_werewolves,
                known_non_werewolves=known_non_werewolves,
            )
            required_candidates = [
                player
                for player in canonicalize_player_set(
                    known_werewolves,
                    field_name="known_werewolves",
                )
                if player != observer
            ]
            if not set(required_candidates).issubset(legal_candidates):
                raise ValueError(
                    "required known Werewolves must be legal candidates"
                )
            if self.audit_hook is not None:
                report_id, report_prompt = self.audit_hook.prepare_report(
                    observer_id=observer,
                    public_snapshot=public_snapshot,
                    agent_backend_id=agent_backend_id,
                    agent_model_id=getattr(agent, "model_name", None),
                    report_prompt=report_prompt,
                )
        except Exception as exc:
            return self._result(
                observer=observer,
                status=STATUS_REPORTER_ERROR,
                error=str(exc),
                agent_backend_id=agent_backend_id,
                known_werewolves=known_werewolves,
                known_non_werewolves=known_non_werewolves,
            )

        validation_error = None
        for generation_attempt in range(1, LABEL_GENERATION_MAX_ATTEMPTS + 1):
            raw_response = None
            try:
                attempt_prompt = (
                    report_prompt
                    if generation_attempt == 1
                    else self.retry_prompt(
                        report_prompt,
                        validation_error,
                    )
                )
                audit_context = (
                    self.audit_hook.belief_context(report_id)
                    if self.audit_hook is not None
                    else nullcontext()
                )
                with audit_context:
                    raw_response = agent.report_suspected_werewolves_readonly(
                        observation=observation,
                        report_prompt=attempt_prompt,
                        legal_candidates=legal_candidates,
                        required_candidates=required_candidates,
                    )
            except Exception as exc:
                self._record_generation_attempt(
                    report_id=report_id,
                    observer_id=observer,
                    generation_attempt=generation_attempt,
                    status=STATUS_REPORTER_ERROR,
                    error=str(exc),
                    raw_response=None,
                )
                if self.audit_hook is not None and report_id is not None:
                    self.audit_hook.complete_report(report_id, None)
                return self._result(
                    observer=observer,
                    status=STATUS_REPORTER_ERROR,
                    error=str(exc),
                    agent_backend_id=agent_backend_id,
                    known_werewolves=known_werewolves,
                    known_non_werewolves=known_non_werewolves,
                    generation_attempt_count=generation_attempt,
                )

            try:
                suspected = self.parse_response(raw_response)
            except (TypeError, ValueError) as exc:
                result = self._result(
                    observer=observer,
                    status=STATUS_PARSE_ERROR,
                    error=str(exc),
                    agent_backend_id=agent_backend_id,
                    known_werewolves=known_werewolves,
                    known_non_werewolves=known_non_werewolves,
                    generation_attempt_count=generation_attempt,
                )
            else:
                try:
                    suspected = validate_player_suspicion(
                        suspected,
                        known_werewolves,
                        known_non_werewolves,
                        observer_id=observer,
                    )
                except (TypeError, ValueError) as exc:
                    result = self._result(
                        observer=observer,
                        status=STATUS_SEMANTIC_ERROR,
                        error=str(exc),
                        agent_backend_id=agent_backend_id,
                        known_werewolves=known_werewolves,
                        known_non_werewolves=known_non_werewolves,
                        generation_attempt_count=generation_attempt,
                    )
                else:
                    result = self._result(
                        observer=observer,
                        status=STATUS_OK,
                        suspected_werewolves=suspected,
                        error=None,
                        agent_backend_id=agent_backend_id,
                        known_werewolves=known_werewolves,
                        known_non_werewolves=known_non_werewolves,
                        generation_attempt_count=generation_attempt,
                    )

            self._record_generation_attempt(
                report_id=report_id,
                observer_id=observer,
                generation_attempt=generation_attempt,
                status=result["status"],
                error=result["error"],
                raw_response=raw_response,
            )
            if result["status"] == STATUS_OK or (
                generation_attempt == LABEL_GENERATION_MAX_ATTEMPTS
            ):
                if self.audit_hook is not None:
                    self.audit_hook.complete_report(report_id, raw_response)
                return result
            validation_error = result["error"]

        raise AssertionError("unreachable label generation loop")

    @staticmethod
    def retry_prompt(report_prompt: str, validation_error: str | None) -> str:
        """Request a new full response using only the last local error."""

        if not isinstance(validation_error, str) or not validation_error:
            raise ValueError("label retry requires a validation error")
        return f"""{report_prompt}

上一次输出未通过本地严格验证：
{validation_error}

必须重新生成完整 JSON。不得修补、引用或解释上一次响应；不得输出额外字段、推理或 Markdown。"""

    def _record_generation_attempt(self, **event) -> None:
        if self.audit_hook is None:
            return
        recorder = getattr(
            self.audit_hook,
            "record_label_generation_attempt",
            None,
        )
        if callable(recorder):
            recorder(**event)

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
        known_wolves = canonicalize_player_set(
            known_werewolves,
            field_name="known_werewolves",
        )
        known_non_wolves = canonicalize_player_set(
            known_non_werewolves,
            field_name="known_non_werewolves",
        )
        if set(known_wolves) & set(known_non_wolves):
            raise ValueError("hard knowledge sets must be disjoint")
        required = [player for player in known_wolves if player != observer]
        forbidden = [
            player for player in canonical_identifiers
            if player == observer or player in known_non_wolves
        ]
        legal_candidates = [
            player for player in canonical_identifiers
            if player not in forbidden
        ]
        required_text = ", ".join(required) if required else "<none>"
        forbidden_text = ", ".join(forbidden)
        legal_candidates_text = ", ".join(legal_candidates)
        public_history = json.dumps(
            copy_public_events(
                public_snapshot.public_events
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return f"""这是一个私有的训练标注请求，不是游戏中的公开发言。
其他玩家不会看到你的回答，回答也不会写入你的游戏记忆。若你是当前发言者，同一份报告会作为随后 day cognition 的只读输入。

请只根据你截至本次公开发言之前合法拥有的公开历史和自己的私人信息，报告你当前内部真实怀疑哪些具体玩家是狼人。不要为了阵营策略欺骗这个私有 reporter。不得使用 god view、actual roles、其他玩家私人信息、当前尚未生成的发言或未来信息。

Your observer identity is exactly: {observer}
Canonical player IDs (complete ordered list):
{canonical_list}
Canonical pre-speech public_events:
{public_history}

Environment-derived hard knowledge:
MUST INCLUDE: {required_text}
MUST EXCLUDE: {forbidden_text}
Current legal candidates: {legal_candidates_text}

`suspected_werewolves` 是玩家级的相对怀疑集合，不是完整双狼人组合约束。它必须包含 MUST INCLUDE 中的全部玩家，并且不得包含 MUST EXCLUDE 中的任何玩家。在满足硬约束后，只额外报告你当前认为相对更可疑的玩家；不要求确定性、不要求完整找到两狼，也不要求恰好两人。不要仅因为某人“仍有可能是狼”就将其列入。允许列出一个或更多合法玩家；只有 MUST INCLUDE 为空时才允许输出空数组。若确实怀疑所有合法候选人，可以全部列出。只允许上面的 canonical player IDs，不得重复。

Before answering, silently verify this checklist:
- 这是我的内部真实怀疑，而不是公开发言策略吗？
- 是否把“仍可能”误当成“当前值得怀疑”？
- 是否错误地强行补足两人？
- 是否包含全部 MUST INCLUDE，且排除了全部 MUST EXCLUDE？
- 是否只输出允许的 JSON？

Do not output probabilities, confidence, scores, rankings, reasons, the checklist, reasoning, or chain-of-thought. Do not output any extra field.

observer: {observer}
prompt_version: {LABEL_PROMPT_VERSION}
public_action_count: {action_count}
public_history_digest: {digest}

Return only this JSON structure:
{{"suspected_werewolves":[...]}}"""

    @staticmethod
    def legal_candidates(
        *,
        observer_id: int | str,
        known_werewolves: list[str],
        known_non_werewolves: list[str],
    ) -> list[str]:
        observer = normalize_player(observer_id)
        known_wolves = canonicalize_player_set(
            known_werewolves,
            field_name="known_werewolves",
        )
        known_non_wolves = canonicalize_player_set(
            known_non_werewolves,
            field_name="known_non_werewolves",
        )
        if set(known_wolves) & set(known_non_wolves):
            raise ValueError("hard knowledge sets must be disjoint")
        return [
            player
            for player in PLAYER_NAMES
            if player != observer and player not in known_non_wolves
        ]

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
        generation_attempt_count: int = 0,
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
            "generation_attempt_count": generation_attempt_count,
        }


__all__ = [
    "STATUS_OK",
    "STATUS_PARSE_ERROR",
    "STATUS_SEMANTIC_ERROR",
    "STATUS_REPORTER_ERROR",
    "PRIVATE_BELIEF_MAX_TOKENS",
    "LABEL_GENERATION_MAX_ATTEMPTS",
    "PRIVATE_BELIEF_JSON_SCHEMA",
    "private_belief_response_format",
    "PlayingAgentBeliefReporter",
]
