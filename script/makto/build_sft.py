"""Build causal 7-player Seer-Witch gameplay SFT data from MaKTO raw events."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any


SOURCE_REVISION = "cc761b03e71b8c41407662822f1884a4b1635922"
SOURCE_SETTING = "7_player_game/seer_witch"
SOURCE_RELATIVE_PATH = Path("raw/train/7_player_game/seer_witch")

EXPECTED_GAMES = 20
EXPECTED_TASK_COUNTS = {
    "speech": 144,
    "vote": 144,
    "wolf": 60,
    "seer": 34,
    "witch": 21,
}
EXPECTED_EXCLUSIONS = {
    "witch_after_antidote_spent": 16,
    "later_contaminated_witch_speech": 1,
    "later_contaminated_witch_vote": 1,
}

RAW_ROLE_TO_RUNTIME = {
    "werewolf": "Werewolf",
    "seer": "Seer",
    "witch": "Witch",
    "simple_villager": "Villager",
}
ROLE_ZH = {
    "Werewolf": "狼人",
    "Seer": "预言家",
    "Witch": "女巫",
    "Villager": "村民",
}

_IGNORED_ANNOTATION_EVENTS = {
    "bad_player",
    "review",
    "speech_summary",
}
_KNOWN_EVENTS = _IGNORED_ANNOTATION_EVENTS | {
    "cycle_round",
    "end",
    "end_rule",
    "healed",
    "inquired",
    "poison",
    "roles",
    "speech",
    "vote_results",
    "vote_start",
    "voted",
    "werewolf_kill",
    "werewolf_night_discuss",
}
_MISSING = object()


class DatasetBuildError(ValueError):
    """A pinned-source event violates the frozen gameplay contract."""


@dataclass
class PendingWitchDecision:
    event_index: int
    candidate_snapshot: list[list[Any]]
    prompt: str | None
    eligible: bool
    healed: object = _MISSING
    poison: object = _MISSING


@dataclass(frozen=True)
class CompletedWolfNight:
    round_number: int
    individual_choices: tuple[tuple[int, int | None], ...]
    final_target: int | None


@dataclass
class ReplayState:
    game_id: str
    roles: dict[int, str] = field(default_factory=dict)
    alive: set[int] = field(default_factory=lambda: set(range(1, 8)))
    round_number: int = 0
    phase: str = "setup"
    public_history: list[str] = field(default_factory=list)
    current_wolf_target: int | None = None
    current_wolf_choices: list[tuple[int, int | None]] = field(default_factory=list)
    completed_wolf_history: list[CompletedWolfNight] = field(default_factory=list)
    seer_checks: list[tuple[int, bool]] = field(default_factory=list)
    antidote_available: bool = True
    poison_available: bool = True
    witch_private_clean: bool = True
    witch_history: list[str] = field(default_factory=list)
    pending_witch: PendingWitchDecision | None = None
    pending_votes: dict[int, int | None] = field(default_factory=dict)
    last_heal_target: int | None = None
    last_poison_target: int | None = None

    def fail(self, event_index: int, message: str) -> DatasetBuildError:
        return DatasetBuildError(
            f"{self.game_id} event {event_index}: {message}"
        )

    def require_roles(self, event_index: int) -> None:
        if len(self.roles) != 7:
            raise self.fail(event_index, "behavior occurred before seven roles")

    def role_player(self, role: str, event_index: int) -> int:
        players = [player for player, value in self.roles.items() if value == role]
        if len(players) != 1:
            raise self.fail(event_index, f"expected one {role}, got {players}")
        return players[0]


def vote_candidates(alive: set[int], actor: int) -> list[list[Any]]:
    return [["vote", 0]] + [
        ["vote", player]
        for player in sorted(alive)
        if player != actor
    ]


def wolf_candidates(alive: set[int]) -> list[list[Any]]:
    return [["kill", 0]] + [
        ["kill", player]
        for player in sorted(alive)
    ]


def seer_candidates(
    alive: set[int],
    actor: int,
    checked_players: set[int],
) -> list[list[Any]]:
    return [["check", 0]] + [
        ["check", player]
        for player in sorted(alive)
        if player != actor and player not in checked_players
    ]


def witch_candidates(
    alive: set[int],
    *,
    antidote_available: bool,
    poison_available: bool,
    wolf_target: int | None,
) -> list[list[Any]]:
    candidates: list[list[Any]] = [["witch_pass", 0]]
    if poison_available:
        candidates.extend(
            ["witch_poison", player]
            for player in sorted(alive)
        )
    if antidote_available and wolf_target is not None:
        candidates.append(["witch_heal", wolf_target])
    return candidates


def _player(value: Any, *, state: ReplayState, event_index: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 7:
        raise state.fail(event_index, f"invalid player ID {value!r}")
    return value


def _optional_player(
    value: Any,
    *,
    state: ReplayState,
    event_index: int,
) -> int | None:
    if value is None:
        return None
    return _player(value, state=state, event_index=event_index)


def _content(event: dict[str, Any], state: ReplayState, event_index: int) -> Any:
    if not isinstance(event, dict) or not isinstance(event.get("event"), str):
        raise state.fail(event_index, "event must contain a string event name")
    if event["event"] not in _KNOWN_EVENTS:
        raise state.fail(event_index, f"unknown raw event {event['event']!r}")
    return event.get("content")


def _mapping_content(
    event: dict[str, Any],
    state: ReplayState,
    event_index: int,
) -> dict[str, Any]:
    content = _content(event, state, event_index)
    if not isinstance(content, dict):
        raise state.fail(event_index, "event content must be an object")
    return content


def _validate_roles(state: ReplayState, event_index: int) -> None:
    if len(state.roles) != 7:
        return
    counts = Counter(state.roles.values())
    expected = Counter({"Werewolf": 2, "Seer": 1, "Witch": 1, "Villager": 3})
    if counts != expected:
        raise state.fail(event_index, f"invalid 7P Seer-Witch roles {dict(counts)}")


def _action_text(action: list[Any]) -> str:
    action_type, target = action
    if target == 0:
        return {
            "vote": "abstain",
            "kill": "pass",
            "check": "pass",
            "witch_pass": "pass",
        }[action_type]
    return {
        "vote": f"vote player{target}",
        "kill": f"kill player{target}",
        "check": f"check player{target}",
        "witch_poison": f"poison player{target}",
        "witch_heal": f"heal player{target}",
    }[action_type]


def _wolf_target_text(target: int | None) -> str:
    return "pass" if target in (None, 0) else f"{target}号"


def _action_index(
    candidate_snapshot: list[list[Any]],
    observed_action: list[Any],
    *,
    state: ReplayState,
    event_index: int,
) -> int:
    try:
        return candidate_snapshot.index(observed_action)
    except ValueError as exc:
        raise state.fail(
            event_index,
            f"observed action {observed_action!r} is outside candidate snapshot "
            f"{candidate_snapshot!r}",
        ) from exc


def _private_information(
    state: ReplayState,
    actor: int,
    *,
    prior_wolf_choices: list[tuple[int, int | None]] | None = None,
) -> list[str]:
    role = state.roles[actor]
    if role == "Werewolf":
        team = sorted(
            player for player, player_role in state.roles.items()
            if player_role == "Werewolf"
        )
        lines = ["你知道的狼人队伍：" + "、".join(f"{p}号" for p in team)]
        lines.extend(
            f"第{night.round_number}夜已完成狼人行动："
            + "；".join(
                f"{wolf}号选择{_wolf_target_text(target)}"
                for wolf, target in night.individual_choices
            )
            + f"；狼队最终目标为{_wolf_target_text(night.final_target)}。"
            for night in state.completed_wolf_history
        )
        if prior_wolf_choices:
            lines.append(
                "本夜在你之前的狼人选择：" + "；".join(
                    f"{wolf}号选择{_wolf_target_text(target)}"
                    for wolf, target in prior_wolf_choices
                )
            )
        return lines
    if role == "Seer":
        if not state.seer_checks:
            return ["你尚无已完成的查验结果。"]
        return [
            "你已完成的查验：" + "；".join(
                f"{target}号是{'狼人' if is_wolf else '非狼人'}"
                for target, is_wolf in state.seer_checks
            )
        ]
    if role == "Witch":
        lines = [
            f"解药可用：{'是' if state.antidote_available else '否'}。",
            f"毒药可用：{'是' if state.poison_available else '否'}。",
        ]
        if state.phase == "night" and state.antidote_available:
            lines.append(
                "本夜狼人最终目标："
                + (
                    "无人"
                    if state.current_wolf_target is None
                    else f"{state.current_wolf_target}号"
                )
                + "。"
            )
        if state.witch_history:
            lines.append("你已完成的女巫行动：" + "；".join(state.witch_history))
        return lines
    return ["除自己的身份外，你没有角色私有信息。"]


_RULES = """你正在玩固定的7人预言家-女巫狼人杀。
角色配置：2名狼人、1名预言家、1名女巫、3名村民。
狼人夜间可选择pass或任意存活玩家；预言家可选择pass或尚未查验过的存活非本人玩家；女巫各有一瓶解药和毒药，同一晚不能同时使用；白天存活玩家可弃票或投给其他存活玩家。
胜利条件：所有狼人出局 → 村民阵营获胜；3名普通村民全部出局（即所有普通村民出局），或预言家与女巫全部出局（即所有神职出局） → 狼人阵营获胜。
你的目标是使自己所属阵营达到其胜利条件。"""


def _build_prompt(
    state: ReplayState,
    *,
    actor: int,
    task: str,
    candidate_snapshot: list[list[Any]] | None,
    prior_wolf_choices: list[tuple[int, int | None]] | None = None,
) -> str:
    role = state.roles[actor]
    alive = "、".join(f"{player}号" for player in sorted(state.alive)) or "无"
    dead = "、".join(
        f"{player}号" for player in range(1, 8) if player not in state.alive
    ) or "无"
    period = "夜晚" if state.phase == "night" else "白天"
    history = "\n".join(state.public_history) or "暂无"
    private = "\n".join(
        "- " + line
        for line in _private_information(
            state,
            actor,
            prior_wolf_choices=prior_wolf_choices,
        )
    )
    task_text = {
        "speech": "生成当前轮次要公开说出的自然语言发言。只输出发言正文。",
        "vote": "从权威合法投票动作中选择一项。只输出action_index JSON。",
        "wolf": "从权威合法狼人夜间动作中选择一项。只输出action_index JSON。",
        "seer": "从权威合法预言家夜间动作中选择一项。只输出action_index JSON。",
        "witch": "从权威合法女巫夜间动作中选择一项。只输出action_index JSON。",
    }[task]
    candidate_text = ""
    if candidate_snapshot is not None:
        lines = "\n".join(
            f"{index}: {_action_text(action)}"
            for index, action in enumerate(candidate_snapshot)
        )
        candidate_text = f"\n\n权威合法动作：\n{lines}"

    return f"""{_RULES}

你是{actor}号玩家，你的身份是{ROLE_ZH[role]}。
当前状态：第{state.round_number}{'夜' if period == '夜晚' else '天'}，{period}。
存活玩家：{alive}。
死亡玩家：{dead}。

你合法知道的私有信息：
{private}

按时间顺序公开的历史：
{history}

当前任务：{task_text}{candidate_text}"""


def _record(
    state: ReplayState,
    *,
    event_index: int,
    task: str,
    actor: int,
    candidate_snapshot: list[list[Any]] | None,
    prompt: str,
    response: str,
) -> dict[str, Any]:
    return {
        "source": {
            "dataset": "makto",
            "revision": SOURCE_REVISION,
            "split": "train",
            "setting": SOURCE_SETTING,
            "game_id": state.game_id,
            "event_index": event_index,
        },
        "task": task,
        "actor": actor,
        "role": state.roles[actor],
        "candidate_snapshot": candidate_snapshot,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
    }


def _indexed_record(
    state: ReplayState,
    *,
    event_index: int,
    task: str,
    actor: int,
    candidate_snapshot: list[list[Any]],
    observed_action: list[Any],
    prompt: str | None = None,
    prior_wolf_choices: list[tuple[int, int | None]] | None = None,
) -> dict[str, Any]:
    action_index = _action_index(
        candidate_snapshot,
        observed_action,
        state=state,
        event_index=event_index,
    )
    if prompt is None:
        prompt = _build_prompt(
            state,
            actor=actor,
            task=task,
            candidate_snapshot=candidate_snapshot,
            prior_wolf_choices=prior_wolf_choices,
        )
    return _record(
        state,
        event_index=event_index,
        task=task,
        actor=actor,
        candidate_snapshot=candidate_snapshot,
        prompt=prompt,
        response=json.dumps(
            {"action_index": action_index},
            separators=(",", ":"),
        ),
    )


def _start_witch_decision(
    state: ReplayState,
    *,
    event_index: int,
    exclusions: Counter[str],
) -> None:
    state.require_roles(event_index)
    actor = state.role_player("Witch", event_index)
    if actor not in state.alive:
        raise state.fail(event_index, "dead Witch produced a night decision")
    eligible = state.antidote_available
    candidates = witch_candidates(
        state.alive,
        antidote_available=state.antidote_available,
        poison_available=state.poison_available,
        wolf_target=state.current_wolf_target,
    )
    prompt = None
    if eligible:
        prompt = _build_prompt(
            state,
            actor=actor,
            task="witch",
            candidate_snapshot=candidates,
        )
    else:
        exclusions["witch_after_antidote_spent"] += 1
        state.witch_private_clean = False
    state.pending_witch = PendingWitchDecision(
        event_index=event_index,
        candidate_snapshot=candidates,
        prompt=prompt,
        eligible=eligible,
    )


def _finish_witch_decision(
    state: ReplayState,
    *,
    boundary_index: int,
    records: list[dict[str, Any]],
) -> None:
    pending = state.pending_witch
    if pending is None:
        return
    if pending.poison is _MISSING:
        raise state.fail(boundary_index, "Witch decision is missing poison component")
    if pending.eligible and pending.healed is _MISSING:
        raise state.fail(boundary_index, "eligible Witch decision is missing healed component")
    if not pending.eligible and pending.healed is not _MISSING:
        raise state.fail(boundary_index, "spent-antidote decision unexpectedly has healed event")

    healed = None if pending.healed is _MISSING else pending.healed
    poison = pending.poison
    if healed is not None and poison is not None:
        raise state.fail(boundary_index, "Witch used antidote and poison together")
    if healed is not None:
        observed_action = ["witch_heal", healed]
        history_text = f"第{state.round_number}夜：使用解药救{healed}号"
    elif poison is not None:
        observed_action = ["witch_poison", poison]
        history_text = f"第{state.round_number}夜：使用毒药毒{poison}号"
    else:
        observed_action = ["witch_pass", 0]
        history_text = f"第{state.round_number}夜：pass"

    actor = state.role_player("Witch", pending.event_index)
    _action_index(
        pending.candidate_snapshot,
        observed_action,
        state=state,
        event_index=pending.event_index,
    )
    if pending.eligible:
        records.append(
            _indexed_record(
                state,
                event_index=pending.event_index,
                task="witch",
                actor=actor,
                candidate_snapshot=pending.candidate_snapshot,
                observed_action=observed_action,
                prompt=pending.prompt,
            )
        )

    if healed is not None:
        state.antidote_available = False
    if poison is not None:
        state.poison_available = False
    state.last_heal_target = healed
    state.last_poison_target = poison
    state.witch_history.append(history_text)
    state.pending_witch = None


def _resolve_night(state: ReplayState, event_index: int) -> None:
    dead: set[int] = set()
    if (
        state.current_wolf_target is not None
        and state.current_wolf_target != state.last_heal_target
    ):
        dead.add(state.current_wolf_target)
    if state.last_poison_target is not None:
        dead.add(state.last_poison_target)
    if not dead <= state.alive:
        raise state.fail(event_index, f"night death contains non-alive players {dead}")
    state.alive.difference_update(dead)
    death_text = "、".join(f"{player}号" for player in sorted(dead)) or "无"
    state.public_history.append(
        f"第{state.round_number}天开始，昨夜死亡：{death_text}。"
    )


def _resolve_vote(
    state: ReplayState,
    content: dict[str, Any],
    event_index: int,
) -> None:
    public_votes: dict[int, int] = {}
    for raw_voter, raw_target in content.items():
        try:
            voter = int(raw_voter)
        except (TypeError, ValueError) as exc:
            raise state.fail(event_index, f"invalid vote-result voter {raw_voter!r}") from exc
        voter = _player(voter, state=state, event_index=event_index)
        target = _player(raw_target, state=state, event_index=event_index)
        public_votes[voter] = target
    counts = Counter(public_votes.values())
    exile = None
    if counts:
        highest = max(counts.values())
        leaders = [target for target, count in counts.items() if count == highest]
        if len(leaders) != 1:
            raise state.fail(event_index, f"unsupported tied vote result {dict(counts)}")
        exile = leaders[0]
        if exile not in state.alive:
            raise state.fail(event_index, f"vote exiled non-alive player {exile}")

    ballots = "；".join(
        f"{voter}号→{target}号"
        for voter, target in sorted(public_votes.items())
    )
    abstainers = sorted(
        voter for voter, target in state.pending_votes.items()
        if target is None
    )
    if abstainers:
        abstain_text = "；".join(f"{voter}号→弃票" for voter in abstainers)
        ballots = "；".join(part for part in (ballots, abstain_text) if part)
    state.public_history.append(
        f"第{state.round_number}天投票：{ballots or '无'}。"
        + ("无人被放逐。" if exile is None else f"{exile}号被放逐。")
    )
    if exile is not None:
        state.alive.remove(exile)
    state.pending_votes.clear()


def replay_game(
    events: list[dict[str, Any]],
    *,
    game_id: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Replay one raw event list without exposing future or god-view state."""

    if not isinstance(events, list):
        raise DatasetBuildError(f"{game_id}: event_zh.json root must be a list")
    state = ReplayState(game_id=game_id)
    records: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter({key: 0 for key in EXPECTED_EXCLUSIONS})

    for event_index, event in enumerate(events):
        content = _content(event, state, event_index)
        event_name = event["event"]

        if event_name == "roles":
            if not isinstance(content, dict):
                raise state.fail(event_index, "roles content must be an object")
            player = _player(content.get("player"), state=state, event_index=event_index)
            raw_role = content.get("role")
            if raw_role not in RAW_ROLE_TO_RUNTIME:
                raise state.fail(event_index, f"unsupported role {raw_role!r}")
            if player in state.roles:
                raise state.fail(event_index, f"duplicate role for player {player}")
            state.roles[player] = RAW_ROLE_TO_RUNTIME[raw_role]
            _validate_roles(state, event_index)

        elif event_name == "cycle_round":
            content = _mapping_content(event, state, event_index)
            _finish_witch_decision(
                state,
                boundary_index=event_index,
                records=records,
            )
            round_number = content.get("round")
            status = content.get("status")
            if (
                isinstance(round_number, bool)
                or not isinstance(round_number, int)
                or round_number <= 0
                or status not in {"night", "day"}
            ):
                raise state.fail(event_index, f"invalid cycle_round {content!r}")
            state.round_number = round_number
            state.phase = status
            if status == "night":
                state.current_wolf_target = None
                state.current_wolf_choices.clear()
                state.last_heal_target = None
                state.last_poison_target = None
                state.public_history.append(f"第{round_number}夜开始。")
            else:
                _resolve_night(state, event_index)

        elif event_name == "inquired":
            state.require_roles(event_index)
            content = _mapping_content(event, state, event_index)
            actor = state.role_player("Seer", event_index)
            target = _player(content.get("player"), state=state, event_index=event_index)
            result = content.get("is_werewolf")
            if not isinstance(result, bool):
                raise state.fail(event_index, "Seer result must be boolean")
            expected_result = state.roles[target] == "Werewolf"
            if result != expected_result:
                raise state.fail(event_index, "Seer result conflicts with role truth")
            candidates = seer_candidates(
                state.alive,
                actor,
                {checked for checked, _result in state.seer_checks},
            )
            records.append(
                _indexed_record(
                    state,
                    event_index=event_index,
                    task="seer",
                    actor=actor,
                    candidate_snapshot=candidates,
                    observed_action=["check", target],
                )
            )
            state.seer_checks.append((target, result))

        elif event_name == "werewolf_night_discuss":
            state.require_roles(event_index)
            content = _mapping_content(event, state, event_index)
            decisions = content.get("decision_kill")
            if not isinstance(decisions, dict):
                raise state.fail(event_index, "decision_kill must be an object")
            for raw_actor, raw_target in decisions.items():
                try:
                    actor_value = int(raw_actor)
                except (TypeError, ValueError) as exc:
                    raise state.fail(event_index, f"invalid wolf actor {raw_actor!r}") from exc
                actor = _player(actor_value, state=state, event_index=event_index)
                if state.roles[actor] != "Werewolf" or actor not in state.alive:
                    raise state.fail(event_index, f"illegal acting Werewolf {actor}")
                target = _optional_player(
                    raw_target,
                    state=state,
                    event_index=event_index,
                )
                candidates = wolf_candidates(state.alive)
                prior_choices = list(state.current_wolf_choices)
                records.append(
                    _indexed_record(
                        state,
                        event_index=event_index,
                        task="wolf",
                        actor=actor,
                        candidate_snapshot=candidates,
                        observed_action=["kill", target or 0],
                        prior_wolf_choices=prior_choices,
                    )
                )
                state.current_wolf_choices.append((actor, target))

        elif event_name == "werewolf_kill":
            content = _mapping_content(event, state, event_index)
            final_target = _optional_player(
                content.get("target_player"),
                state=state,
                event_index=event_index,
            )
            if (
                final_target is not None
                and final_target not in state.alive
            ):
                raise state.fail(event_index, "final wolf target is not alive")
            if not state.current_wolf_choices:
                raise state.fail(
                    event_index,
                    "werewolf_kill arrived without current-night individual decisions",
                )
            individual_targets = {
                target for _actor, target in state.current_wolf_choices
            }
            if final_target not in individual_targets:
                raise state.fail(
                    event_index,
                    f"final wolf target {final_target!r} is absent from "
                    f"individual choices {state.current_wolf_choices!r}",
                )
            if any(
                night.round_number == state.round_number
                for night in state.completed_wolf_history
            ):
                raise state.fail(event_index, "duplicate completed wolf night")
            state.current_wolf_target = final_target
            state.completed_wolf_history.append(
                CompletedWolfNight(
                    round_number=state.round_number,
                    individual_choices=tuple(state.current_wolf_choices),
                    final_target=final_target,
                )
            )

        elif event_name in {"healed", "poison"}:
            content = _mapping_content(event, state, event_index)
            if state.pending_witch is None:
                _start_witch_decision(
                    state,
                    event_index=event_index,
                    exclusions=exclusions,
                )
            pending = state.pending_witch
            assert pending is not None
            value = _optional_player(
                content.get("player"),
                state=state,
                event_index=event_index,
            )
            if value is not None and value not in state.alive:
                raise state.fail(event_index, f"Witch targeted dead player {value}")
            attribute = "healed" if event_name == "healed" else "poison"
            if getattr(pending, attribute) is not _MISSING:
                raise state.fail(event_index, f"duplicate Witch {event_name} component")
            setattr(pending, attribute, value)

        elif event_name == "speech":
            state.require_roles(event_index)
            content = _mapping_content(event, state, event_index)
            actor = _player(content.get("player"), state=state, event_index=event_index)
            speech = content.get("context")
            if actor not in state.alive:
                raise state.fail(event_index, f"dead player {actor} produced speech")
            if not isinstance(speech, str) or not speech.strip():
                raise state.fail(event_index, "speech context must be non-empty text")
            if state.roles[actor] == "Witch" and not state.witch_private_clean:
                exclusions["later_contaminated_witch_speech"] += 1
            else:
                records.append(
                    _record(
                        state,
                        event_index=event_index,
                        task="speech",
                        actor=actor,
                        candidate_snapshot=None,
                        prompt=_build_prompt(
                            state,
                            actor=actor,
                            task="speech",
                            candidate_snapshot=None,
                        ),
                        response=speech,
                    )
                )
            state.public_history.append(
                f"第{content.get('day')}轮，{actor}号发言：{speech}"
            )

        elif event_name == "vote_start":
            state.phase = "day"
            state.pending_votes.clear()

        elif event_name == "voted":
            state.require_roles(event_index)
            content = _mapping_content(event, state, event_index)
            actor = _player(content.get("player"), state=state, event_index=event_index)
            if actor not in state.alive:
                raise state.fail(event_index, f"dead player {actor} voted")
            if actor in state.pending_votes:
                raise state.fail(event_index, f"duplicate vote by player {actor}")
            target = _optional_player(
                content.get("voted_to_player"),
                state=state,
                event_index=event_index,
            )
            candidates = vote_candidates(state.alive, actor)
            observed_action = ["vote", target or 0]
            _action_index(
                candidates,
                observed_action,
                state=state,
                event_index=event_index,
            )
            if state.roles[actor] == "Witch" and not state.witch_private_clean:
                exclusions["later_contaminated_witch_vote"] += 1
            else:
                records.append(
                    _indexed_record(
                        state,
                        event_index=event_index,
                        task="vote",
                        actor=actor,
                        candidate_snapshot=candidates,
                        observed_action=observed_action,
                    )
                )
            state.pending_votes[actor] = target

        elif event_name == "vote_results":
            content = _mapping_content(event, state, event_index)
            _resolve_vote(state, content, event_index)

        elif event_name == "end":
            _finish_witch_decision(
                state,
                boundary_index=event_index,
                records=records,
            )

        elif event_name in _IGNORED_ANNOTATION_EVENTS | {"end_rule"}:
            continue

    _finish_witch_decision(
        state,
        boundary_index=len(events),
        records=records,
    )
    if len(state.roles) != 7:
        raise DatasetBuildError(f"{game_id}: incomplete roles")
    return records, dict(exclusions)


def _game_number(path: Path) -> int:
    match = re.fullmatch(r"game_(\d+)", path.parent.name)
    if match is None:
        raise DatasetBuildError(f"invalid game directory {path.parent}")
    return int(match.group(1))


def _manifest(
    *,
    games: int,
    records: list[dict[str, Any]],
    exclusions: Counter[str],
) -> dict[str, Any]:
    task_counts = Counter(record["task"] for record in records)
    samples = {
        task: task_counts[task]
        for task in EXPECTED_TASK_COUNTS
    }
    samples["total"] = len(records)
    return {
        "source_revision": SOURCE_REVISION,
        "source_setting": SOURCE_SETTING,
        "games": games,
        "samples": samples,
        "excluded": {
            reason: exclusions[reason]
            for reason in EXPECTED_EXCLUSIONS
        },
    }


def _validate_expected(manifest: dict[str, Any]) -> None:
    expected_samples = {
        **EXPECTED_TASK_COUNTS,
        "total": sum(EXPECTED_TASK_COUNTS.values()),
    }
    if manifest["games"] != EXPECTED_GAMES:
        raise DatasetBuildError(
            f"expected {EXPECTED_GAMES} games, got {manifest['games']}"
        )
    if manifest["samples"] != expected_samples:
        raise DatasetBuildError(
            f"sample counts differ: expected {expected_samples}, "
            f"got {manifest['samples']}"
        )
    if manifest["excluded"] != EXPECTED_EXCLUSIONS:
        raise DatasetBuildError(
            f"exclusion counts differ: expected {EXPECTED_EXCLUSIONS}, "
            f"got {manifest['excluded']}"
        )


def build_dataset(
    *,
    source_root: Path,
    output: Path,
    enforce_expected: bool = True,
) -> tuple[Path, Path, dict[str, Any]]:
    source_dir = source_root / SOURCE_RELATIVE_PATH
    paths = sorted(source_dir.rglob("event_zh.json"), key=_game_number)
    if not paths:
        raise DatasetBuildError(f"no event_zh.json files under {source_dir}")

    records: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter({key: 0 for key in EXPECTED_EXCLUSIONS})
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            events = json.load(handle)
        game_records, game_exclusions = replay_game(
            events,
            game_id=path.parent.name,
        )
        records.extend(game_records)
        exclusions.update(game_exclusions)

    manifest = _manifest(
        games=len(paths),
        records=records,
        exclusions=exclusions,
    )
    if enforce_expected:
        _validate_expected(manifest)

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(".manifest.json")
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
    return output, manifest_path, manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build pinned MaKTO 7P Seer-Witch gameplay SFT JSONL."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output, manifest_path, manifest = build_dataset(
        source_root=args.source_root,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "manifest": str(manifest_path),
                **manifest,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
