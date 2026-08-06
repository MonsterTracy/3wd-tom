import ast
from copy import copy, deepcopy
import json
import logging
import random
import re
from pathlib import Path
from werewolf.agents.prompt_template_v0 import (
    CON,
    LEGACY_GAMEPLAY_PROMPT_PROFILE,
    STRICT_CLASSIC7_GAMEPLAY_PROMPT_PROFILE,
    build_strict_classic7_speech_rules,
)
from werewolf.agents.base_agent import Agent
from werewolf.backends import BackendError
from werewolf.helper.log_utils import JsonFormatter, CustomLoggerAdapter
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

    def format_observation(self, observation):
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
            identity_info = CON.player_identity_info.format(player_idx=observation['current_act_idx'],
                                                            identity=CON.identity_chinese[identity],
                                                            identity_ability=CON.identity_abilities[identity])
            logs = self.format_log(observation['game_log'])

            prompt = CON.speech_prompt.format(game_description=CON.game_description,
                                              player_identity_info=identity_info, logs=logs, )
            if self.gameplay_prompt_profile == (
                STRICT_CLASSIC7_GAMEPLAY_PROMPT_PROFILE
            ):
                strict_rules = (
                    build_strict_classic7_speech_rules(
                        observation
                    )
                )
                prompt = prompt.replace(
                    "** 输出",
                    strict_rules
                    + "\n\n** 输出",
                    1,
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
