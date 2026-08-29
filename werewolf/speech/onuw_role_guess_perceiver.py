"""Readonly ONUW-style full-role-guess label collection.

No function in this module accepts a global role map or a true role-count
repair input. The playing agent receives only its detached legal observation.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from contextlib import nullcontext
from copy import deepcopy
from typing import Any

from werewolf.models.twd_tom.onuw_parity_protocol import ROLE_GUESS_NAMES
from werewolf.models.twd_tom.public_events import copy_public_events
from werewolf.models.twd_tom.schema import PLAYER_NAMES, normalize_player


ROLE_GUESS_MAX_TOKENS = 192
ROLE_GUESS_MAX_ATTEMPTS = 3
_EXTERNAL_AGENT_FIELDS = {
    "backend",
    "handler",
    "logger",
    "tokenizer",
    "strategy",
    "matcher",
}
EXPECTED_CLASSIC7_ROLE_COUNTS = {
    "werewolf": 2,
    "villager": 3,
    "seer": 1,
    "witch": 1,
}


def role_guess_response_format(*, supports_json_schema: bool) -> dict[str, Any]:
    """Return a provider schema; local validation remains authoritative."""

    if not isinstance(supports_json_schema, bool):
        raise TypeError("supports_json_schema must be boolean")
    if not supports_json_schema:
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "onuw_style_role_guess",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["role_guesses"],
                "properties": {
                    "role_guesses": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": list(PLAYER_NAMES),
                        "properties": {
                            player: {
                                "type": "string",
                                "enum": list(ROLE_GUESS_NAMES),
                            }
                            for player in PLAYER_NAMES
                        },
                    }
                },
            },
        },
    }


def validate_role_guesses(value: Any) -> dict[str, str]:
    """Hard-validate exactly seven player fields and the legal vocabulary."""

    if not isinstance(value, Mapping):
        raise TypeError("role_guesses must be an object")
    if set(value) != set(PLAYER_NAMES):
        missing = sorted(set(PLAYER_NAMES) - set(value))
        extra = sorted(set(value) - set(PLAYER_NAMES))
        raise ValueError(
            f"role_guesses field set mismatch; missing={missing}, extra={extra}"
        )
    normalized = {}
    for player in PLAYER_NAMES:
        role = value[player]
        if not isinstance(role, str):
            raise TypeError(f"role guess for {player} must be text")
        role = role.strip().lower()
        if role not in ROLE_GUESS_NAMES:
            raise ValueError(f"unsupported role guess for {player}: {role!r}")
        normalized[player] = role
    return normalized


def parse_role_guess_response(raw_response: Any) -> dict[str, str]:
    if not isinstance(raw_response, str):
        raise TypeError("role-guess response must be text")
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError("role-guess response must be pure JSON") from exc
    if not isinstance(parsed, dict) or set(parsed) != {"role_guesses"}:
        raise ValueError("response requires only role_guesses")
    return validate_role_guesses(parsed["role_guesses"])


def role_guess_audit(role_guesses: Mapping[str, str]) -> dict[str, Any]:
    """Flag count conflicts without rejecting or repairing the report."""

    guesses = validate_role_guesses(role_guesses)
    counts = Counter(guesses.values())
    observed = {role: counts.get(role, 0) for role in ROLE_GUESS_NAMES}
    conflict = any(
        observed[role] != expected
        for role, expected in EXPECTED_CLASSIC7_ROLE_COUNTS.items()
    ) or observed["unknown"] != 0
    return {
        "role_counts": observed,
        "role_count_conflict": conflict,
    }


def role_guesses_to_target(role_guesses: Mapping[str, str]) -> list[float]:
    """Extract guessed-as-werewolf support with ONUW full-uniform empty."""

    guesses = validate_role_guesses(role_guesses)
    support = [player for player in PLAYER_NAMES if guesses[player] == "werewolf"]
    if not support:
        return [1.0 / len(PLAYER_NAMES)] * len(PLAYER_NAMES)
    probability = 1.0 / len(support)
    return [probability if player in support else 0.0 for player in PLAYER_NAMES]


def role_guess_reports_to_matrix(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    observer_ids: list[str] | tuple[str, ...],
) -> list[list[float]]:
    """Build [7, 7], supervising alive rows and retaining all columns."""

    if not isinstance(reports, Mapping):
        raise TypeError("reports must be a mapping")
    alive = list(observer_ids)
    if alive != [player for player in PLAYER_NAMES if player in alive]:
        raise ValueError("observer_ids must be unique canonical order")
    if len(alive) != len(set(alive)):
        raise ValueError("observer_ids cannot contain duplicates")
    if set(reports) != set(alive):
        raise ValueError("reports must contain exactly the alive observers")
    matrix = []
    for observer in PLAYER_NAMES:
        if observer not in reports:
            matrix.append([0.0] * len(PLAYER_NAMES))
            continue
        report = reports[observer]
        if report.get("status") != "ok":
            raise ValueError(f"{observer} role-guess report is not valid")
        matrix.append(role_guesses_to_target(report.get("role_guesses")))
    return matrix


class OnuwStyleRoleGuessReporter:
    """Ask one playing agent for a detached, private-informed role report."""

    def __init__(self, audit_hook=None) -> None:
        self.audit_hook = audit_hook

    @staticmethod
    def build_prompt(*, observer_id: int | str, public_snapshot) -> str:
        observer = normalize_player(observer_id)
        public_history = json.dumps(
            copy_public_events(public_snapshot.public_events),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return f"""这是私有训练标注，不是公开发言，也不会写回游戏记忆。
只使用你在当前发言严格 PRE 边界前依法可见的公开历史、自己的身份及自己的私人信息，猜测全部七名玩家的角色。禁止使用 god view、其他玩家不可见私人状态、真实双狼数量来修复输出、当前尚未生成的发言或未来信息。

observer: {observer}
strict_pre_step_idx: {public_snapshot.step_idx}
public_history_digest: {public_snapshot.public_history_digest}
strict_pre_public_events: {public_history}
canonical_players: {", ".join(PLAYER_NAMES)}
legal_roles: {", ".join(ROLE_GUESS_NAMES)}

必须为全部七名玩家各输出一个角色。允许角色数量冲突，也允许 unknown；不要为了满足真实角色数量而修改你的猜测。只返回：
{{"role_guesses":{{"player1":"...","player2":"...","player3":"...","player4":"...","player5":"...","player6":"...","player7":"..."}}}}"""

    def report(
        self,
        *,
        agent,
        observation: Mapping[str, Any],
        observer_id: int | str,
        public_snapshot,
        agent_backend_id: str,
    ) -> dict[str, Any]:
        observer = normalize_player(observer_id)
        if not isinstance(observation, Mapping):
            raise TypeError("observation must be a mapping")
        if normalize_player(observation.get("observer_id")) != observer:
            raise ValueError("observation does not belong to observer")
        if observation.get("current_act_idx") != public_snapshot.speaker_id:
            raise ValueError("observation and PRE snapshot speaker mismatch")
        if observation.get("phase") != public_snapshot.phase:
            raise ValueError("observation and PRE snapshot phase mismatch")
        if not hasattr(agent, "report_role_guesses_readonly"):
            raise TypeError("playing agent lacks report_role_guesses_readonly()")
        prompt = self.build_prompt(
            observer_id=observer,
            public_snapshot=public_snapshot,
        )
        report_id = None
        if self.audit_hook is not None:
            report_id, prompt = self.audit_hook.prepare_report(
                observer_id=observer,
                public_snapshot=public_snapshot,
                agent_backend_id=agent_backend_id,
                agent_model_id=getattr(agent, "model_name", None),
                report_prompt=prompt,
            )
        last_error = None
        for attempt in range(1, ROLE_GUESS_MAX_ATTEMPTS + 1):
            attempt_prompt = prompt
            if last_error is not None:
                attempt_prompt += (
                    "\n\n上一次输出未通过严格验证："
                    f"{last_error}\n请重新生成完整 JSON，不得解释或修补旧响应。"
                )
            audit_context = (
                self.audit_hook.belief_context(report_id)
                if self.audit_hook is not None
                else nullcontext()
            )
            raw = None
            try:
                with audit_context:
                    raw = agent.report_role_guesses_readonly(
                        observation=deepcopy(observation),
                        report_prompt=attempt_prompt,
                    )
                guesses = parse_role_guess_response(raw)
            except (TypeError, ValueError) as exc:
                last_error = str(exc)
                continue
            audit = role_guess_audit(guesses)
            if self.audit_hook is not None:
                self.audit_hook.complete_report(report_id, raw)
            return {
                "observer": observer,
                "status": "ok",
                "role_guesses": guesses,
                "werewolf_support": [
                    player for player in PLAYER_NAMES
                    if guesses[player] == "werewolf"
                ],
                **audit,
                "agent_backend_id": agent_backend_id,
                "generation_attempt_count": attempt,
            }
        if self.audit_hook is not None:
            self.audit_hook.complete_report(report_id, None)
        return {
            "observer": observer,
            "status": "invalid",
            "error": last_error,
            "agent_backend_id": agent_backend_id,
            "generation_attempt_count": ROLE_GUESS_MAX_ATTEMPTS,
        }


class OnuwRoleGuessSnapshotCollector:
    """Collect all alive rows without accepting an oracle role interface."""

    def __init__(self, reporter, agents) -> None:
        if reporter is None or not hasattr(reporter, "report"):
            raise TypeError("reporter must provide report()")
        if not isinstance(agents, (list, tuple)) or len(agents) != 7:
            raise ValueError("agents must contain exactly seven playing agents")
        self.reporter = reporter
        self.agents = tuple(agents)

    def collect(self, public_snapshot, *, env) -> dict[str, dict[str, Any]]:
        if not hasattr(env, "get_observation_for"):
            raise TypeError("environment must provide get_observation_for()")
        reports = {}
        for player_id in public_snapshot.observer_ids:
            observer = normalize_player(player_id)
            agent = self.agents[player_id - 1]
            backend_id = getattr(agent, "backend_id", None)
            if not isinstance(backend_id, str) or not backend_id:
                raise ValueError(f"{observer} has no agent_backend_id")
            state_before = {
                name: deepcopy(value)
                for name, value in vars(agent).items()
                if name not in _EXTERNAL_AGENT_FIELDS
            }
            result = self.reporter.report(
                agent=agent,
                observation=env.get_observation_for(player_id),
                observer_id=observer,
                public_snapshot=public_snapshot,
                agent_backend_id=backend_id,
            )
            state_after = {
                name: deepcopy(value)
                for name, value in vars(agent).items()
                if name not in _EXTERNAL_AGENT_FIELDS
            }
            if state_after != state_before:
                raise RuntimeError(
                    f"readonly role-guess report mutated {observer} agent state"
                )
            if result.get("status") != "ok":
                raise RuntimeError(
                    f"role-guess collection failed for {observer}: "
                    f"{result.get('error')}"
                )
            reports[observer] = result
        return reports


__all__ = [
    "ROLE_GUESS_MAX_TOKENS",
    "ROLE_GUESS_MAX_ATTEMPTS",
    "EXPECTED_CLASSIC7_ROLE_COUNTS",
    "role_guess_response_format",
    "validate_role_guesses",
    "parse_role_guess_response",
    "role_guess_audit",
    "role_guesses_to_target",
    "role_guess_reports_to_matrix",
    "OnuwStyleRoleGuessReporter",
    "OnuwRoleGuessSnapshotCollector",
]
