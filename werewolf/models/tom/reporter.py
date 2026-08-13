"""Subjective belief reports from one player's legal observation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from werewolf.models.tom.schema import PLAYER_NAMES, normalize_player
from werewolf.speech.private_belief_perceiver import (
    PRIVATE_BELIEF_MAX_TOKENS,
    private_belief_response_format,
)


def _log_payload(log: Any) -> dict[str, Any]:
    fields = ("event", "source", "target", "content", "day", "time")
    if any(not hasattr(log, field) for field in fields):
        raise TypeError("observation game_log entries must be Log objects")
    return {
        field: deepcopy(getattr(log, field))
        for field in fields
    }


class BeliefReporter:
    """Make one stateless backend call from an observer's legal state."""

    def __init__(self, dispatches: Sequence[Mapping[str, Any]]) -> None:
        if isinstance(dispatches, (str, bytes)) or len(dispatches) != 7:
            raise ValueError("dispatches must contain exactly seven entries")
        checked = []
        for dispatch in dispatches:
            if not isinstance(dispatch, Mapping):
                raise TypeError("each reporter dispatch must be a mapping")
            if set(dispatch) != {"backend", "model_name"}:
                raise ValueError("reporter dispatch requires backend and model_name")
            backend = dispatch["backend"]
            model_name = dispatch["model_name"]
            if backend is None or not hasattr(backend, "chat"):
                raise TypeError("reporter backend must provide chat()")
            if not isinstance(model_name, str) or not model_name.strip():
                raise ValueError("reporter model_name must be non-empty text")
            checked.append(dict(dispatch))
        self._dispatches = tuple(checked)

    @staticmethod
    def legal_state(
        observer_id: int | str,
        observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Select the label-side public and private state explicitly."""

        observer = normalize_player(observer_id)
        if not isinstance(observation, Mapping):
            raise TypeError("legal observation must be a mapping")
        if normalize_player(observation.get("observer_id")) != observer:
            raise ValueError("legal observation belongs to another observer")
        identity = observation.get("identity")
        if not isinstance(identity, str) or not identity:
            raise ValueError("legal observation requires observer identity")
        game_log = observation.get("game_log")
        if isinstance(game_log, (str, bytes)) or not isinstance(game_log, Sequence):
            raise TypeError("legal observation requires a game_log sequence")
        public_state = observation.get("authoritative_public_state")
        if not isinstance(public_state, Mapping):
            raise TypeError("legal observation requires authoritative_public_state")
        return {
            "observer_id": observer,
            "self_role": identity,
            "current_phase": observation.get("phase"),
            "current_public_actor": observation.get("current_act_idx"),
            "game_log": [_log_payload(log) for log in game_log],
            "authoritative_public_state": deepcopy(dict(public_state)),
        }

    @classmethod
    def build_prompt(
        cls,
        observer_id: int | str,
        observation: Mapping[str, Any],
    ) -> str:
        observer = normalize_player(observer_id)
        state = cls.legal_state(observer, observation)
        serialized = json.dumps(
            state,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"""这是一个私有、只读的主观狼人怀疑标注请求，不是公开发言或游戏行动。
只根据下面 observer={observer} 当前合法拥有的公开与私人信息回答。当前公开发言已经发生并包含在信息中。不得使用 god view、真实角色表、其他玩家私人信息或未来信息。不要为了公开博弈策略欺骗 reporter。
允许怀疑 0 到 7 名玩家，包括空集合；不要强迫恰好两名。合法 ID 只有：{', '.join(PLAYER_NAMES)}。不得输出概率、理由、推理或额外字段。
legal_post_speech_observation: {serialized}
Return only: {{"suspected_werewolves":[...]}}"""

    @staticmethod
    def parse(raw: Any) -> list[str]:
        if not isinstance(raw, str):
            raise TypeError("reporter response must be text")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("reporter response must be one JSON object") from exc
        if not isinstance(payload, dict) or set(payload) != {"suspected_werewolves"}:
            raise ValueError("reporter response requires only suspected_werewolves")
        values = payload["suspected_werewolves"]
        if isinstance(values, (str, bytes)) or not isinstance(values, list):
            raise TypeError("suspected_werewolves must be a list")
        suspected = [normalize_player(value) for value in values]
        if len(suspected) != len(set(suspected)):
            raise ValueError("suspected_werewolves must not contain duplicates")
        return sorted(suspected, key=PLAYER_NAMES.index)

    def report(
        self,
        observer_id: int | str,
        observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        observer = normalize_player(observer_id)
        try:
            prompt = self.build_prompt(observer, observation)
            dispatch = self._dispatches[PLAYER_NAMES.index(observer)]
            backend = dispatch["backend"]
            raw = backend.chat(
                messages=[{"role": "user", "content": prompt}],
                model=dispatch["model_name"],
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
        except Exception:
            return self._result(observer, valid=False, error="reporter_error")
        try:
            suspected = self.parse(raw)
        except (TypeError, ValueError):
            return self._result(observer, valid=False, error="parse_error")
        return self._result(
            observer,
            valid=True,
            suspected=suspected,
            error=None,
        )

    @staticmethod
    def _result(
        observer: str,
        *,
        valid: bool,
        suspected: list[str] | None = None,
        error: str | None,
    ) -> dict[str, Any]:
        return {
            "observer_id": observer,
            "valid": valid,
            "suspected_werewolves": suspected,
            "error": error,
        }


__all__ = ["BeliefReporter"]
