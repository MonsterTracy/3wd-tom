"""Readonly direct self-reports over the canonical two-Werewolf worlds."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from copy import deepcopy
from typing import Any

from werewolf.models.twd_tom.public_events import copy_public_events
from werewolf.models.twd_tom.schema import (
    NUM_WOLF_PAIR_CLASSES,
    PLAYER_NAMES,
    canonical_wolf_pairs,
    normalize_player,
)


STATUS_OK = "ok"
STATUS_PARSE_ERROR = "parse_error"
STATUS_SEMANTIC_ERROR = "semantic_error"
STATUS_REPORTER_ERROR = "reporter_error"

PAIR_BELIEF_PROMPT_VERSION = (
    "classic7_fixed_two_wolves_direct_pair_belief_self_report_prompt_v1"
)
PAIR_BELIEF_PARSER_VERSION = "strict_pair_probability_json_v1"
PAIR_BELIEF_REPORT_PROVENANCE = (
    "playing_agent_readonly_direct_pair_belief_self_report_v1"
)
PAIR_BELIEF_PAYLOAD_VERSION = "readonly_pair_belief_self_report_payload_v1"
PAIR_BELIEF_MAX_TOKENS = 256
PROBABILITY_RTOL = 1e-5
PROBABILITY_ATOL = 1e-6

PAIR_BELIEF_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["pair_probabilities"],
    "properties": {
        "pair_probabilities": {
            "type": "array",
            "minItems": NUM_WOLF_PAIR_CLASSES,
            "maxItems": NUM_WOLF_PAIR_CLASSES,
            "items": {"type": "number", "minimum": 0.0},
        }
    },
}


def canonical_json(value: Any) -> str:
    """Serialize one JSON value deterministically and reject NaN/Inf."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_json_sha256(value: Any) -> str:
    """Return the SHA-256 of the canonical JSON representation."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def pair_belief_response_format(*, supports_json_schema: bool) -> dict[str, Any]:
    if not isinstance(supports_json_schema, bool):
        raise TypeError("supports_json_schema must be boolean")
    if not supports_json_schema:
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "pair_belief_self_report",
            "strict": True,
            "schema": deepcopy(PAIR_BELIEF_JSON_SCHEMA),
        },
    }


def legal_world_mask(
    known_werewolves: Any,
    known_non_werewolves: Any,
) -> tuple[list[str], list[str], list[bool]]:
    """Return the sole canonical support implied by environment knowledge."""

    from werewolf.models.twd_tom.belief_labels import close_hard_knowledge

    closed_wolves, closed_non_wolves = close_hard_knowledge(
        known_werewolves,
        known_non_werewolves,
    )
    wolf_set = set(closed_wolves)
    non_wolf_set = set(closed_non_wolves)
    mask = [
        wolf_set.issubset(pair) and set(pair).isdisjoint(non_wolf_set)
        for pair in canonical_wolf_pairs()
    ]
    if not any(mask):
        raise ValueError("hard knowledge has no legal world")
    return closed_wolves, closed_non_wolves, mask


def validate_pair_probabilities(
    values: Any,
    *,
    known_werewolves: Any,
    known_non_werewolves: Any,
) -> tuple[list[float], dict[str, Any]]:
    """Validate a direct report without projection or normalization."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("pair_probabilities must be a sequence")
    if len(values) != NUM_WOLF_PAIR_CLASSES:
        raise ValueError(
            f"pair_probabilities must contain {NUM_WOLF_PAIR_CLASSES} values"
        )
    probabilities: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"pair_probabilities[{index}] must be a number")
        probability = float(value)
        if not math.isfinite(probability):
            raise ValueError("pair_probabilities must contain only finite values")
        if probability < 0.0:
            raise ValueError("pair_probabilities cannot contain negative values")
        probabilities.append(probability)

    total = math.fsum(probabilities)
    if not math.isclose(
        total,
        1.0,
        rel_tol=PROBABILITY_RTOL,
        abs_tol=PROBABILITY_ATOL,
    ):
        raise ValueError("pair_probabilities must sum to one")

    closed_wolves, closed_non_wolves, mask = legal_world_mask(
        known_werewolves,
        known_non_werewolves,
    )
    illegal_mass = math.fsum(
        probability
        for probability, is_legal in zip(probabilities, mask, strict=True)
        if not is_legal
    )
    if not math.isclose(
        illegal_mass,
        0.0,
        rel_tol=PROBABILITY_RTOL,
        abs_tol=PROBABILITY_ATOL,
    ):
        raise ValueError("pair_probabilities assign mass to an illegal world")

    return probabilities, {
        "status": "valid",
        "legal_world_mask": mask,
        "illegal_probability_mass": illegal_mass,
        "probability_rtol": PROBABILITY_RTOL,
        "probability_atol": PROBABILITY_ATOL,
        "known_werewolves": closed_wolves,
        "known_non_werewolves": closed_non_wolves,
    }


class ReadonlyPairBeliefSelfReporter:
    """Request one direct canonical-pair belief from a detached playing agent."""

    def __init__(self, audit_hook=None) -> None:
        self.audit_hook = audit_hook

    def report(
        self,
        *,
        agent,
        observation: Mapping[str, Any],
        belief_owner_id: int | str,
        public_snapshot,
        backend_alias: str,
        known_werewolves: list[str],
        known_non_werewolves: list[str],
    ) -> dict[str, Any]:
        belief_owner = normalize_player(belief_owner_id)
        closed_wolves, closed_non_wolves, mask = legal_world_mask(
            known_werewolves,
            known_non_werewolves,
        )
        base = {
            "player_id": belief_owner,
            "backend_alias": backend_alias,
            "resolved_model_name": getattr(agent, "model_name", None),
            "known_werewolves": closed_wolves,
            "known_non_werewolves": closed_non_wolves,
            "report_provenance": PAIR_BELIEF_REPORT_PROVENANCE,
            "prompt_version": PAIR_BELIEF_PROMPT_VERSION,
            "parser_version": PAIR_BELIEF_PARSER_VERSION,
        }
        try:
            if not isinstance(observation, Mapping):
                raise TypeError("observation must be a mapping")
            if normalize_player(observation.get("observer_id")) != belief_owner:
                raise ValueError("observation does not belong to belief_owner")
            if observation.get("current_act_idx") != public_snapshot.speaker_id:
                raise ValueError("observation and public snapshot speaker mismatch")
            if observation.get("phase") != public_snapshot.phase:
                raise ValueError("observation and public snapshot phase mismatch")
            if not isinstance(backend_alias, str) or not backend_alias.strip():
                raise ValueError("backend_alias must be non-empty text")
            if not isinstance(base["resolved_model_name"], str) or not base[
                "resolved_model_name"
            ].strip():
                raise ValueError("resolved_model_name must be non-empty text")
            if not hasattr(agent, "build_readonly_pair_belief_payload"):
                raise TypeError("playing agent cannot build a pair-belief payload")
            if not hasattr(agent, "report_pair_belief_self_readonly"):
                raise TypeError("playing agent cannot report a readonly pair belief")

            prompt = self.build_prompt(
                belief_owner_id=belief_owner,
                public_snapshot=public_snapshot,
                known_werewolves=closed_wolves,
                known_non_werewolves=closed_non_wolves,
            )
            report_id = None
            if self.audit_hook is not None:
                report_id, prompt = self.audit_hook.prepare_report(
                    observer_id=belief_owner,
                    public_snapshot=public_snapshot,
                    agent_backend_id=backend_alias,
                    agent_model_id=base["resolved_model_name"],
                    report_prompt=prompt,
                )
            payload = agent.build_readonly_pair_belief_payload(
                observation=observation,
                report_prompt=prompt,
            )
            canonical_json(payload)
            audit_context = (
                self.audit_hook.belief_context(report_id)
                if self.audit_hook is not None
                else nullcontext()
            )
            try:
                with audit_context:
                    raw_output = agent.report_pair_belief_self_readonly(
                        reporter_payload=payload,
                    )
            except Exception:
                if self.audit_hook is not None and report_id is not None:
                    self.audit_hook.complete_report(report_id, None)
                raise
            if self.audit_hook is not None:
                self.audit_hook.complete_report(report_id, raw_output)
        except Exception as exc:
            return self._result(
                **base,
                status=STATUS_REPORTER_ERROR,
                error=str(exc),
                pair_probabilities=None,
                reporter_input_payload=locals().get("payload"),
                prompt=locals().get("prompt"),
                raw_output=None,
                parsed_output=None,
                hard_knowledge_validation={
                    "status": "not_checked",
                    "legal_world_mask": mask,
                },
            )

        try:
            parsed = self.parse_response(raw_output)
        except (TypeError, ValueError) as exc:
            return self._result(
                **base,
                status=STATUS_PARSE_ERROR,
                error=str(exc),
                pair_probabilities=None,
                reporter_input_payload=payload,
                prompt=prompt,
                raw_output=raw_output,
                parsed_output=None,
                hard_knowledge_validation={
                    "status": "not_checked",
                    "legal_world_mask": mask,
                },
            )

        try:
            probabilities, validation = validate_pair_probabilities(
                parsed["pair_probabilities"],
                known_werewolves=closed_wolves,
                known_non_werewolves=closed_non_wolves,
            )
        except (TypeError, ValueError) as exc:
            return self._result(
                **base,
                status=STATUS_SEMANTIC_ERROR,
                error=str(exc),
                pair_probabilities=None,
                reporter_input_payload=payload,
                prompt=prompt,
                raw_output=raw_output,
                parsed_output=parsed,
                hard_knowledge_validation={
                    "status": "invalid",
                    "legal_world_mask": mask,
                    "error": str(exc),
                },
            )

        return self._result(
            **base,
            status=STATUS_OK,
            error=None,
            pair_probabilities=probabilities,
            reporter_input_payload=payload,
            prompt=prompt,
            raw_output=raw_output,
            parsed_output=parsed,
            hard_knowledge_validation=validation,
        )

    def record_agent_state(self, *, observer_id, state_before, state_after) -> None:
        if self.audit_hook is not None:
            self.audit_hook.record_agent_state(
                observer_id=observer_id,
                state_before=state_before,
                state_after=state_after,
            )

    @staticmethod
    def build_prompt(
        *,
        belief_owner_id: int | str,
        public_snapshot,
        known_werewolves: list[str],
        known_non_werewolves: list[str],
    ) -> str:
        belief_owner = normalize_player(belief_owner_id)
        pair_lines = "\n".join(
            f"{index}: {left}, {right}"
            for index, (left, right) in enumerate(canonical_wolf_pairs())
        )
        public_history = canonical_json(
            copy_public_events(public_snapshot.public_events)
        )
        return f"""这是一个私有、只读的训练标注请求，不是公开游戏行动。
请作为 {belief_owner}，只根据截至当前行动角色发言之前你合法拥有的公开历史、角色私有历史、私有记忆和 hard knowledge，直接报告你对固定两狼世界的主观概率分布。不得使用其他玩家私有信息、当前尚未完成的行动或未来事件。

固定 canonical 21-pair ordering 如下；输出数组位置必须严格对应这些索引，不得排序或重排：
{pair_lines}

known_werewolves: {canonical_json(known_werewolves)}
known_non_werewolves: {canonical_json(known_non_werewolves)}
public_history: {public_history}

输出唯一一个 JSON 对象，且只能包含 pair_probabilities。该数组必须恰好 21 个有限非负数，总和为 1，并且不得给违反 hard knowledge 的 pair 分配概率。不要输出解释、Markdown、top-k 或玩家边际概率。"""

    @staticmethod
    def parse_response(raw_output: Any) -> dict[str, list[Any]]:
        if not isinstance(raw_output, str):
            raise TypeError("raw reporter output must be text")
        def reject_nonfinite(token: str):
            raise ValueError(f"non-finite JSON number is not allowed: {token}")
        try:
            parsed = json.loads(raw_output, parse_constant=reject_nonfinite)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid reporter JSON: {exc}") from exc
        if not isinstance(parsed, dict) or set(parsed) != {"pair_probabilities"}:
            raise ValueError("reporter JSON must contain only pair_probabilities")
        values = parsed["pair_probabilities"]
        if not isinstance(values, list):
            raise TypeError("pair_probabilities must be a list")
        return {"pair_probabilities": deepcopy(values)}

    @staticmethod
    def _result(
        *,
        player_id: str,
        backend_alias: str,
        resolved_model_name: str | None,
        known_werewolves: list[str],
        known_non_werewolves: list[str],
        report_provenance: str,
        prompt_version: str,
        parser_version: str,
        status: str,
        error: str | None,
        pair_probabilities: list[float] | None,
        reporter_input_payload: Any,
        prompt: str | None,
        raw_output: Any,
        parsed_output: Any,
        hard_knowledge_validation: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "player_id": player_id,
            "report_status": status,
            "report_error": error,
            "pair_probabilities": pair_probabilities,
            "known_werewolves": known_werewolves,
            "known_non_werewolves": known_non_werewolves,
            "reporter_input_payload": deepcopy(reporter_input_payload),
            "reporter_input_payload_sha256": (
                canonical_json_sha256(reporter_input_payload)
                if reporter_input_payload is not None
                else None
            ),
            "raw_reporter_output": raw_output,
            "parsed_output": deepcopy(parsed_output),
            "hard_knowledge_validation": dict(hard_knowledge_validation),
            "report_provenance": report_provenance,
            "backend_alias": backend_alias,
            "resolved_model_name": resolved_model_name,
            "prompt_version": prompt_version,
            "prompt_sha256": (
                hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                if prompt is not None
                else None
            ),
            "parser_version": parser_version,
            "sampling_parameters": {
                "temperature": 0.0,
                "max_tokens": PAIR_BELIEF_MAX_TOKENS,
                "thinking": "disabled",
            },
            "reporter_seed": None,
        }


__all__ = [
    "STATUS_OK",
    "STATUS_PARSE_ERROR",
    "STATUS_SEMANTIC_ERROR",
    "STATUS_REPORTER_ERROR",
    "PAIR_BELIEF_PROMPT_VERSION",
    "PAIR_BELIEF_PARSER_VERSION",
    "PAIR_BELIEF_REPORT_PROVENANCE",
    "PAIR_BELIEF_PAYLOAD_VERSION",
    "PAIR_BELIEF_MAX_TOKENS",
    "PROBABILITY_RTOL",
    "PROBABILITY_ATOL",
    "PAIR_BELIEF_JSON_SCHEMA",
    "canonical_json",
    "canonical_json_sha256",
    "pair_belief_response_format",
    "legal_world_mask",
    "validate_pair_probabilities",
    "ReadonlyPairBeliefSelfReporter",
]
