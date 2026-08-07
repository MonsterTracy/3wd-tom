import ast
from copy import copy, deepcopy
from dataclasses import dataclass
import json
import logging
import random
import re
from pathlib import Path
from werewolf.agents.prompt_template_v0 import (
    CON,
    LEGACY_GAMEPLAY_PROMPT_PROFILE,
    STRICT_CLASSIC7_GAMEPLAY_PROMPT_PROFILE,
    build_strict_classic7_speech_plan_prompt,
)
from werewolf.agents.base_agent import Agent
from werewolf.backends import BackendError
from werewolf.helper.log_utils import JsonFormatter, CustomLoggerAdapter
from werewolf.models.twd_tom.schema import ACTION_NAMES
from werewolf.speech.private_belief_perceiver import (
    PRIVATE_BELIEF_MAX_TOKENS,
    private_belief_response_format,
)


_PRIVATE_ROLE_EVENTS = {
    "Werewolf": {
        "werewolf_team_info",
        "skill_wolf",
        "kill_decision",
    },
    "Seer": {
        "skill_seer",
    },
    "Witch": {
        "kill_decision",
        "skill_witch",
    },
    "Guard": {
        "skill_guard",
    },
    "Villager": set(),
}
_OBSERVER_OWNED_PRIVATE_EVENTS = {
    "skill_seer",
    "skill_witch",
    "skill_guard",
}


class GameplaySpeechQualityError(ValueError):
    """A deterministic public-speech response contract violation."""


class PublicSpeechPlanValidationError(ValueError):
    """A private planner response violates the public-plan contract."""


@dataclass(frozen=True)
class PublicSpeechPlan:
    """Validated public claims represented only by formal speech actions."""

    public_actions: tuple[tuple[str, int], ...]

    @property
    def targets(self):
        return frozenset(target for _action, target in self.public_actions)

    def as_list(self):
        return [
            {"action": action, "target": target}
            for action, target in self.public_actions
        ]


def canonical_suggestible_player_ids(authoritative_public_state):
    if not isinstance(authoritative_public_state, dict):
        raise TypeError("authoritative public state is missing")
    candidates = authoritative_public_state.get("suggestible_exile_targets")
    if not isinstance(candidates, list):
        raise TypeError("authoritative candidate set is missing")
    canonical = tuple(candidates)
    if (
        any(
            isinstance(candidate, bool)
            or not isinstance(candidate, int)
            or not 1 <= candidate <= 7
            for candidate in canonical
        )
        or len(canonical) != len(set(canonical))
    ):
        raise ValueError("authoritative candidate set is invalid")
    return canonical


def public_speech_plan_json_schema(*, suggestible_player_ids, speaker_id):
    if not isinstance(suggestible_player_ids, tuple):
        raise TypeError("suggestible_player_ids must be a tuple")
    if isinstance(speaker_id, bool) or not isinstance(speaker_id, int) or not 1 <= speaker_id <= 7:
        raise ValueError("speaker_id must be an integer in [1, 7]")
    other_actions = [
        action
        for action in ACTION_NAMES
        if action not in {"vote_intent", "oppose"}
    ]
    action_branches = []
    if suggestible_player_ids:
        action_branches.append({
            "type": "object",
            "additionalProperties": False,
            "required": ["action", "target"],
            "properties": {
                "action": {"const": "vote_intent"},
                "target": {
                    "type": "integer",
                    "enum": list(suggestible_player_ids),
                },
            },
        })
    action_branches.append({
        "type": "object",
        "additionalProperties": False,
        "required": ["action", "target"],
        "properties": {
            "action": {"const": "oppose"},
            "target": {
                "type": "integer",
                "enum": [
                    player_id
                    for player_id in range(1, 8)
                    if player_id != speaker_id
                ],
            },
        },
    })
    action_branches.append({
        "type": "object",
        "additionalProperties": False,
        "required": ["action", "target"],
        "properties": {
            "action": {"type": "string", "enum": other_actions},
            "target": {"type": "integer", "enum": list(range(1, 8))},
        },
    })
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["public_actions"],
        "properties": {
            "public_actions": {
                "type": "array",
                "items": {"oneOf": action_branches},
            },
        },
    }


def public_speech_plan_response_format(
    *, supports_json_schema, suggestible_player_ids, speaker_id
):
    if supports_json_schema is not True:
        raise BackendError(
            "strict speech planner requires backend JSON Schema support"
        )
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "public_speech_plan",
            "strict": True,
            "schema": public_speech_plan_json_schema(
                suggestible_player_ids=suggestible_player_ids,
                speaker_id=speaker_id,
            ),
        },
    }


def validate_public_speech_plan(
    payload,
    *,
    suggestible_player_ids,
    player_id,
    phase,
    game_context=None,
):
    context = f"game={game_context or 'unavailable'}, player={player_id}, phase={phase}"

    def reject(reason):
        raise PublicSpeechPlanValidationError(f"{reason} ({context})")

    if not isinstance(suggestible_player_ids, tuple):
        reject("suggestible_player_ids must be a tuple")
    if not isinstance(payload, dict) or set(payload) != {"public_actions"}:
        reject("plan must contain only public_actions")
    public_actions = payload["public_actions"]
    if not isinstance(public_actions, list):
        reject("public_actions must be an array")
    validated = []
    seen = set()
    for item in public_actions:
        if not isinstance(item, dict) or set(item) != {"action", "target"}:
            reject("every public action must contain only action and target")
        action = item["action"]
        target = item["target"]
        if action not in ACTION_NAMES:
            reject(f"unsupported public action: {action!r}")
        if isinstance(target, bool) or not isinstance(target, int) or not 1 <= target <= 7:
            reject(f"invalid public action target: {target!r}")
        pair = (action, target)
        if pair in seen:
            reject(f"duplicate public action: {action}/player{target}")
        seen.add(pair)
        validated.append(pair)

    for action, target in validated:
        if action == "vote_intent" and target not in suggestible_player_ids:
            reject(f"vote_intent target player{target} is not currently suggestible")
    if ("oppose", player_id) in seen:
        reject(f"oppose cannot target the current speaker player{player_id}")
    return PublicSpeechPlan(tuple(validated))


_CHINESE_PLAYER_NUMBERS = {
    character: number
    for number, character in enumerate("零一二三四五六七八九")
}
_EXPLICIT_PLAYER_REFERENCE = re.compile(
    r"player\s*(?P<english>\d+)(?!\d)"
    r"|玩家\s*(?P<chinese_prefix>\d+|[零一二三四五六七八九])(?!\d)"
    r"|(?<!第)(?P<suffix>\d+|[零一二三四五六七八九])\s*号(?:玩家|位)?",
    re.IGNORECASE,
)


def _extract_explicit_player_references(content, *, context):
    referenced_players = set()
    for match in _EXPLICIT_PLAYER_REFERENCE.finditer(content):
        value = next(group for group in match.groups() if group is not None)
        player_number = (
            int(value)
            if value.isdigit()
            else _CHINESE_PLAYER_NUMBERS[value]
        )
        if not 1 <= player_number <= 7:
            raise GameplaySpeechQualityError(
                f"invalid player reference {match.group(0)!r} ({context})"
            )
        referenced_players.add(player_number)
    return referenced_players


def validate_gameplay_public_speech(
    content,
    *,
    finish_reason=None,
    player_id=None,
    phase=None,
    planned_player_ids=None,
):
    """Validate only high-confidence gameplay speech failures."""

    context = f"player={player_id}, phase={phase}"
    if not isinstance(content, str) or not content.strip():
        raise GameplaySpeechQualityError(
            f"empty gameplay public speech ({context})"
        )
    if finish_reason == "length":
        raise GameplaySpeechQualityError(
            f"truncated gameplay public speech ({context})"
        )

    referenced_players = _extract_explicit_player_references(
        content,
        context=context,
    )

    stripped = content.lstrip()
    if stripped.startswith(("{", "[", "```", "# ")):
        raise GameplaySpeechQualityError(
            f"structured gameplay public speech output ({context})"
        )
    forbidden_control_text = (
        "【权威公共状态】",
        "【你合法知道的私有信息】",
        "【其他玩家此前的公开主张】",
        "【公开发言要求】",
        "system prompt",
        "系统提示词",
        "current_act_idx",
        "valid_action",
        "game_log",
    )
    if any(marker in content for marker in forbidden_control_text):
        raise GameplaySpeechQualityError(
            f"internal control text in gameplay public speech ({context})"
        )
    if planned_player_ids is not None:
        planned = set(planned_player_ids)
        allowed = planned | {player_id}
        unexpected = referenced_players - allowed
        if unexpected:
            raise GameplaySpeechQualityError(
                f"unplanned player reference(s) {sorted(unexpected)} ({context})"
            )
        missing = planned - referenced_players
        if missing:
            raise GameplaySpeechQualityError(
                f"planned player reference(s) missing {sorted(missing)} ({context})"
            )
    return content

class LLMAgent(Agent):
    def __init__(self,
                 backend=None,
                 model_name=None,
                 tokenizer=None,
                 temperature=1.0,
                 log_file=None,
                 gameplay_prompt_profile=LEGACY_GAMEPLAY_PROMPT_PROFILE,
                 gameplay_max_tokens=None):
        self.backend = backend
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.nlp_action_to_env_action = {}
        self.temperature = temperature
        if (
            gameplay_max_tokens is not None
            and (
                isinstance(gameplay_max_tokens, bool)
                or not isinstance(gameplay_max_tokens, int)
                or gameplay_max_tokens <= 0
            )
        ):
            raise ValueError(
                "gameplay_max_tokens must be a positive integer"
            )
        self.gameplay_max_tokens = gameplay_max_tokens
        if gameplay_prompt_profile not in {
            LEGACY_GAMEPLAY_PROMPT_PROFILE,
            STRICT_CLASSIC7_GAMEPLAY_PROMPT_PROFILE,
        }:
            raise ValueError(
                "unsupported gameplay_prompt_profile: "
                f"{gameplay_prompt_profile}"
            )
        self.gameplay_prompt_profile = (
            gameplay_prompt_profile
        )
        if log_file is not None:
            self.has_log = True
            self.handler = logging.FileHandler(log_file)
            self.handler.setLevel(logging.INFO)
            self.handler.setFormatter(JsonFormatter())
            logger = logging.getLogger(
                str(Path(log_file).resolve())
            )
            logger.setLevel(logging.INFO)
            logger.addHandler(self.handler)
            self.logger = CustomLoggerAdapter(logger, extra={})
        else:
            self.has_log = False

    def close(self):
        """Detach and close this agent's per-game log handler."""

        if not self.has_log:
            return
        self.logger.logger.removeHandler(self.handler)
        self.handler.close()
        self.has_log = False

    def _chat(self, messages, **kwargs):
        if self.backend is None or not self.model_name:
            raise BackendError("Agent backend and model_name are required.")
        return self.backend.chat(
            messages=messages,
            model=self.model_name,
            **kwargs,
        )

    def _chat_with_metadata(
        self,
        messages,
        *,
        player_log_context=None,
        **kwargs,
    ):
        if self.backend is None or not self.model_name:
            raise BackendError("Agent backend and model_name are required.")
        if not hasattr(self.backend, "chat_with_metadata"):
            raise BackendError(
                "gameplay public speech backend must support chat_with_metadata"
            )
        try:
            content, metadata = self.backend.chat_with_metadata(
                messages=messages,
                model=self.model_name,
                **kwargs,
            )
        except Exception as exc:
            self._log_player_backend_call(
                messages=messages,
                content=None,
                metadata=None,
                player_log_context=player_log_context,
                response_format=kwargs.get("response_format"),
                error=exc,
            )
            raise
        self._log_player_backend_call(
            messages=messages,
            content=content,
            metadata=metadata,
            player_log_context=player_log_context,
            response_format=kwargs.get("response_format"),
        )
        if (
            not isinstance(metadata, dict)
            or not isinstance(metadata.get("finish_reason"), str)
        ):
            raise BackendError(
                "gameplay public speech response requires finish_reason metadata"
            )
        return content, metadata

    def _log_player_backend_call(
        self,
        *,
        messages,
        content,
        metadata,
        player_log_context,
        response_format,
        error=None,
    ):
        if player_log_context is None or not self.has_log:
            return
        observation = player_log_context["observation"]
        self.logger.info(
            player_log_context["stage"],
            extra={
                "prompt": messages[0]["content"],
                "messages": messages,
                "response": content,
                "action": content,
                "finish_reason": (
                    metadata.get("finish_reason")
                    if isinstance(metadata, dict)
                    else None
                ),
                "response_format": response_format,
                "model": self.model_name,
                "backend_id": getattr(self, "backend_id", None),
                "game_id": getattr(
                    getattr(self.backend, "session", None),
                    "game_id",
                    None,
                ),
                "dispatch_status": "error" if error else "ok",
                "error_type": type(error).__name__ if error else None,
                "error_message": str(error) if error else None,
                "player_id": observation["current_act_idx"],
                "role": observation["identity"],
                "phase": observation["phase"],
                "gen_times": 0,
            },
        )

    def _build_readonly_belief_context(self, observation):
        """Build detached messages from this player's legal observation."""

        observer_id = observation.get("observer_id")
        if not isinstance(observer_id, int) or not 1 <= observer_id <= 7:
            raise ValueError("readonly belief observation requires observer_id in [1, 7]")
        identity = observation.get("identity")
        if not isinstance(identity, str) or not identity:
            raise ValueError("readonly belief observation requires identity")
        game_log = observation.get("game_log")
        if not isinstance(game_log, list):
            raise TypeError("readonly belief observation requires a game_log list")

        memory = {}
        for field_name in ("notes", "vote_reason"):
            if hasattr(self, field_name):
                memory[field_name] = deepcopy(getattr(self, field_name))

        private_events = _PRIVATE_ROLE_EVENTS.get(
            identity
        )
        if private_events is None:
            raise ValueError(
                "readonly belief observation "
                "has unsupported identity"
            )
        private_logs = [
            deepcopy(log)
            for log in game_log
            if getattr(
                log,
                "event",
                None,
            )
            in private_events
            and (
                getattr(
                    log,
                    "event",
                    None,
                )
                not in _OBSERVER_OWNED_PRIVATE_EVENTS
                or getattr(
                    log,
                    "source",
                    None,
                )
                == observer_id
            )
        ]

        legal_context = {
            "observer_id": f"player{observer_id}",
            "self_role": identity,
            "current_phase": observation.get("phase"),
            "current_public_actor": observation.get("current_act_idx"),
            "private_role_history": self.format_log(
                private_logs
            ),
            "private_agent_memory": memory,
        }
        return [{
            "role": "user",
            "content": (
                "以下是你当前合法拥有的信息状态。它是只读副本，不会写回游戏：\n"
                + json.dumps(legal_context, ensure_ascii=False, sort_keys=True)
            ),
        }]

    def report_suspected_werewolves_readonly(
        self,
        *,
        observation,
        report_prompt,
    ):
        """Run a detached self-report without mutating this agent context."""

        if not isinstance(report_prompt, str) or not report_prompt.strip():
            raise ValueError("report_prompt must be non-empty text")
        detached_agent = copy(self)
        detached_observation = deepcopy(observation)
        messages = detached_agent._build_readonly_belief_context(
            detached_observation
        )
        messages.append({"role": "user", "content": report_prompt})
        return detached_agent._chat(
            messages,
            temperature=0.0,
            max_tokens=(
                PRIVATE_BELIEF_MAX_TOKENS
            ),
            response_format=(
                private_belief_response_format(
                    supports_json_schema=(
                        getattr(
                            detached_agent.backend,
                            "supports_json_schema",
                            False,
                        )
                    ),
                )
            ),
            extra_body={"thinking": {"type": "disabled"}},
        )

    def format_observation(
        self,
        observation,
        *,
        suggestible_player_ids=None,
    ):
        phase = observation['phase']
        if 'skill' in phase or 'vote' in phase:
            valid_actions = observation['valid_action']
            valid_actions_str = self.get_valid_actions_str(valid_actions)
            identity = observation['identity']
            identity_info = CON.player_identity_info.format(player_idx=observation['current_act_idx'],
                                                            identity=CON.identity_chinese[identity],
                                                            identity_ability=CON.identity_abilities[identity])
            logs = self.format_log(observation['game_log'])
            if 'skill' in phase:
                prompt = CON.skill_prompt.format(game_description=CON.game_description,
                                                 player_identity_info=identity_info, logs=logs,
                                                 valid_actions=valid_actions_str)
            else:
                prompt = CON.vote_prompt.format(game_description=CON.game_description,
                                                player_identity_info=identity_info, logs=logs,
                                                valid_actions=valid_actions_str)
        elif 'speech' in phase:
            identity = observation['identity']
            if self.gameplay_prompt_profile == (
                STRICT_CLASSIC7_GAMEPLAY_PROMPT_PROFILE
            ):
                prompt = (
                    build_strict_classic7_speech_plan_prompt(
                        observation,
                        suggestible_player_ids=suggestible_player_ids,
                    )
                )
            else:
                identity_info = CON.player_identity_info.format(
                    player_idx=observation['current_act_idx'],
                    identity=CON.identity_chinese[identity],
                    identity_ability=CON.identity_abilities[identity],
                )
                logs = self.format_log(observation['game_log'])
                prompt = CON.speech_prompt.format(
                    game_description=CON.game_description,
                    player_identity_info=identity_info,
                    logs=logs,
                )
        else:
            raise ValueError
        return prompt

    def _print_log(self, log):
        print("===============")
        print(log.event)
        print(log.viewer)
        print(log.source)
        print(log.target)
        print(log.content)
        print(log.time)
        print("===============\n")


    def format_log(self, game_log):
        logs = ""
        for log in game_log:
            log_tmp=""
            if log.event == 'game_setting':
                log_tmp = '本局游戏各个身份和对应数量如下：\n'
                for key, value in log.content.items():
                    log_tmp += "- {}:{}\n".format(CON.identity_chinese[key], value)
            if log.event == 'skill_wolf':
                log_tmp = "{}号是狼人，他在{}准备猎杀{}号。\n".format(log.source, log.time, log.target)
            elif log.event == 'kill_decision':
                log_tmp = "狼人队伍在{}猎杀了{}号。\n".format(log.time, log.target)
            elif log.event == 'skill_seer':
                log_tmp = "{}号是预言家，你在{}查验了{}号的身份是{}。\n".format(log.source, log.time, log.target,
                                                                              '狼人' if log.content[
                                                                                            'cheked_identity'] == 'bad' else '好人')
            elif log.event == 'skill_guard':
                log_tmp = "{}号是守卫，你在{}守护了{}号。\n".format(log.source, log.time, log.target)
            elif log.event == 'skill_witch':
                if 'heal' in log.content:
                    log_tmp = "{}号是女巫，你在{}使用解药治疗了{}号。\n".format(log.source, log.time, log.target)
                elif 'poison' in log.content:
                    log_tmp = "{}号是女巫，你在{}使用毒药毒害了{}号。\n".format(log.source, log.time, log.target)
            elif log.event == 'speech' or log.event == 'speech_pk':
                if len(log.content['speech_content']) > 0:
                    log_tmp = "{}号在{}发言内容：{}。\n".format(log.source, log.time, log.content['speech_content'])
                else:
                    log_tmp = "{}号在{}发言内容为空。\n".format(log.source, log.time)
            elif log.event == 'vote':
                if log.target > 0:
                    log_tmp = "{}号在{}投票给{}号。\n".format(log.source, log.time, log.target)
                else:
                    log_tmp = "{}号在{}放弃投票。\n".format(log.source, log.time, log.target)
            elif log.event == 'vote_pk':
                if log.target > 0:
                    log_tmp = "{}号在{}pk环节投票给{}号。\n".format(log.source, log.time, log.target)
                else:
                    log_tmp = "{}号在{}pk环节放弃投票。\n".format(log.source, log.time, log.target)
            elif log.event == 'end_game':
                log_tmp = "游戏结束！\n"
            elif log.event == 'end_night':
                dead_list = ""
                for idx in log.content['dead_list']:
                    dead_list += '{}号、'.format(idx)
                if len(dead_list) > 0:
                    dead_list = dead_list[:-1]
                    log_tmp = "{}死亡的玩家是{}。\n".format(log.time, dead_list)
                else:
                    log_tmp = "{}无人死亡。\n".format(log.time)
            elif log.event == 'end_vote':
                if log.content['vote_outcome'] == 'all abstention':
                    log_tmp = "{}所有玩家放弃投票，直接进入夜晚。\n".format(log.time)
                elif log.content['vote_outcome'] == 'all abstention in pk':
                    log_tmp = "{}再次发言，所有玩家放弃投票，直接进入夜晚。\n".format(log.time)
                elif log.content['vote_outcome'] == 'draw':
                    pk_speech_list = ''
                    for idx in log.content['speech_queue']:
                        pk_speech_list += '{}号、'.format(idx)
                    pk_speech_list = pk_speech_list[:-1]

                    pk_vote_list = ''
                    for idx in log.content['vote_queue']:
                        pk_vote_list += '{}号、'.format(idx)
                    pk_vote_list = pk_vote_list[:-1]
                    log_tmp = "{}平票，由{}再次发言，{}进行投票。\n".format(log.time, pk_speech_list, pk_vote_list)
                elif log.content['vote_outcome'] == 'draw in pk':
                    log_tmp = "{}再次平票，直接进入夜晚。\n".format(log.time)
                elif type(log.content['vote_outcome']) == int:
                    log_tmp = "{}通过投票驱逐了{}号。\n".format(log.time, log.content['expelled'])
                else:
                    raise ValueError
            elif log.event == 'werewolf_team_info':
                wolf_team = ''
                for idx in log.content['wolf_team']:
                    wolf_team += '{}号、'.format(idx)
                wolf_team = wolf_team[:-1]
                log_tmp = "狼人队伍的成员是{}。\n".format(wolf_team)
            elif log.event == 'self_identity':
                pass
            logs += log_tmp

        return logs

    def _normalize_vote_target_value(self, value):
        if isinstance(value, int):
            return value if value >= 0 else None

        text = str(value).strip().strip("\"'")
        if text.lower() in ("否", "弃票", "不投", "不投票", "abstain", "0"):
            return 0

        match = re.search(r'\d+', text)
        if match:
            return int(match.group(0))
        return None

    def _extract_json_like(self, raw_text):
        text = str(raw_text).strip().strip("- ").strip()
        fenced = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if fenced:
            text = fenced.group(1).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return None

    def parse_vote_target(self, raw_action):
        if raw_action is None:
            return None

        parsed = self._extract_json_like(raw_action)
        if isinstance(parsed, dict):
            for key in ("投票玩家", "投票"):
                if key in parsed:
                    return self._normalize_vote_target_value(parsed[key])

        text = str(raw_action).strip()
        match = re.search(r'(?:投票玩家|投票)\s*[:：]\s*([^\n,，。；;}]*)', text)
        if match:
            return self._normalize_vote_target_value(match.group(1))

        return None

    def vote_target_to_action_str(self, vote_target):
        if vote_target in (None, -1, 0):
            return "{'投票': '否'}"
        return "{'" + f"投票': '{vote_target}'" + "}"

    def choose_fallback_vote(self, observation, self_player_id=None):
        if self_player_id is None:
            self_player_id = observation.get("current_act_idx")

        positive_candidates = []
        non_self_candidates = []
        for action_name, target in observation.get("valid_action", observation.get("valid_actions", [])):
            if action_name not in ("vote", "vote_pk", "投票"):
                continue
            if not isinstance(target, int) or target <= 0:
                continue

            positive_candidates.append(target)
            if target != self_player_id:
                non_self_candidates.append(target)

        if non_self_candidates:
            return random.choice(non_self_candidates)
        if positive_candidates:
            return random.choice(positive_candidates)
        return 0

    def choose_fallback_vote_action(self, observation, valid_action=None):
        valid_action = list(valid_action or self.nlp_action_to_env_action.keys())
        fallback_target = self.choose_fallback_vote(observation)
        fallback_action = self.vote_target_to_action_str(fallback_target)
        if fallback_action in valid_action:
            return fallback_action

        non_abstain_actions = [
            action for action in valid_action
            if self.parse_vote_target(action) not in (None, 0)
        ]
        if non_abstain_actions:
            return random.choice(non_abstain_actions)

        abstain_action = self.vote_target_to_action_str(0)
        if abstain_action in valid_action:
            return abstain_action
        return valid_action[0] if valid_action else abstain_action

    def parse_vote_action(self, raw_action, observation, valid_action):
        cleaned_action = str(raw_action).strip().strip("- ")
        if cleaned_action in valid_action:
            return cleaned_action

        vote_target = self.parse_vote_target(cleaned_action)
        if vote_target is None:
            return None

        action = self.vote_target_to_action_str(vote_target)
        if action in valid_action:
            return action
        if vote_target == 0:
            return self.choose_fallback_vote_action(observation, valid_action)
        return None

    def get_valid_actions_str(self, valid_actions):
        valid_actions_str = ""
        action_pairs = []
        has_positive_vote_target = any(
            action[0] in ("vote", "vote_pk") and isinstance(action[1], int) and action[1] > 0
            for action in valid_actions
        )
        for action in valid_actions:
            if action[0] == 'kill':
                if action[1] == 0:
                    action_text = "{'杀害':'否'}"
                else:
                    action_text = "{{'杀害':'{0}'}}".format(action[1])
                valid_actions_str += f"- {action_text}\n"
                action_pairs.append((action_text, action))
            elif action[0] == 'check':
                if action[1] == 0:
                    action_text = "{'查验':'否'}"
                else:
                    action_text = "{{'查验':'{0}'}}".format(action[1])
                valid_actions_str += f"- {action_text}\n"
                action_pairs.append((action_text, action))
            elif action[0] == 'guard':
                if action[1] == 0:
                    action_text = "{'守卫':'否'}"
                else:
                    action_text = "{{'守卫':'{0}'}}".format(action[1])
                valid_actions_str += f"- {action_text}\n"
                action_pairs.append((action_text, action))
            elif 'witch' in action[0]:
                if action[0] == 'witch_pass':
                    action_text = "{'解药': '否', '毒药': '否'}"
                elif action[0] == 'witch_poison':
                    action_text = "{{'解药': '否', '毒药': '{0}'}}".format(action[1])
                elif action[0] == 'witch_heal':
                    action_text = "{{'解药': '{0}', '毒药': '否'}}".format(action[1])
                else:
                    continue
                valid_actions_str += f"- {action_text}\n"
                action_pairs.append((action_text, action))
            elif action[0] == 'vote' or action[0] == 'vote_pk':
                if action[1] == 0:
                    if has_positive_vote_target:
                        continue
                    action_text = "{'投票': '否'}"
                else:
                    action_text = "{{'投票': '{0}'}}".format(action[1])
                valid_actions_str += f"- {action_text}\n"
                action_pairs.append((action_text, action))

        self.nlp_action_to_env_action = {}
        for nlp_action, env_action in action_pairs:
            self.nlp_action_to_env_action[nlp_action] = env_action

        return valid_actions_str

    def reset(self):
        return

    def act(self, observation):
        raise NotImplementedError
