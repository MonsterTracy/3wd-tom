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
    DiscussionAct,
    LEGACY_GAMEPLAY_PROMPT_PROFILE,
    STRICT_BELIEF_CONCRETE_ROLES,
    STRICT_CLASSIC7_GAME_DESCRIPTION,
    STRICT_CLASSIC7_GAMEPLAY_PROMPT_PROFILE,
    STRICT_CLASSIC7_ROLE_COUNTS,
    project_discussion_content_indices,
    project_discussion_vote_stances,
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


class GameplaySpeechQualityError(ValueError):
    """A deterministic public-speech response contract violation."""


class GameplayActionValidationError(ValueError):
    """A gameplay action response is not an authoritative candidate."""


class BeliefValidationError(ValueError):
    """A transient gameplay belief response violates its contract."""


class RoleReportValidationError(ValueError):
    """A structured role report violates observer-authoritative semantics."""


BELIEF_ROLES = STRICT_BELIEF_CONCRETE_ROLES + ("unknown",)
COGNITION_TEXT_MIN_LENGTH = 1
BELIEF_TEXT_MAX_LENGTH = 256
CONCISE_TEXT_MAX_LENGTH = 96


@dataclass(frozen=True)
class BeliefReport:
    belief: str
    concise: str
    roles: dict[str, str]

    def as_dict(self):
        return {
            "belief": self.belief,
            "concise": self.concise,
            "roles": dict(self.roles),
        }

    def gameplay_dict(self):
        return {
            "belief": self.belief,
            "concise": self.concise,
        }


@dataclass(frozen=True)
class DayCognitionReportV2:
    belief: str
    concise: str
    roles: dict[str, str]
    public_content_action_indices: tuple[int, ...]
    public_vote_stance_index: int
    evidence_claim_ids: tuple[str, ...]


def belief_response_format(*, supports_json_schema, role_options):
    if supports_json_schema is not True:
        raise BackendError("belief generation requires backend JSON Schema support")
    if not isinstance(role_options, dict) or not role_options:
        raise ValueError("role_options must be a non-empty dictionary")
    unresolved_players = list(role_options)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "belief_report",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["belief", "concise", "roles"],
                "properties": {
                    "belief": {
                        "type": "string",
                        "minLength": COGNITION_TEXT_MIN_LENGTH,
                        "maxLength": BELIEF_TEXT_MAX_LENGTH,
                    },
                    "concise": {
                        "type": "string",
                        "minLength": COGNITION_TEXT_MIN_LENGTH,
                        "maxLength": CONCISE_TEXT_MAX_LENGTH,
                    },
                    "roles": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": unresolved_players,
                        "properties": {
                            player: {
                                "type": "string",
                                "enum": list(role_options[player]),
                            }
                            for player in unresolved_players
                        },
                    },
                },
            },
        },
    }


def day_cognition_response_format_v2(
    *,
    supports_json_schema,
    role_options,
    candidate_snapshot,
    claim_ids,
):
    if supports_json_schema is not True:
        raise BackendError(
            "Day cognition requires backend JSON Schema support"
        )
    if not isinstance(role_options, dict) or not role_options:
        raise ValueError("role_options must be a non-empty dictionary")
    if (
        not isinstance(candidate_snapshot, tuple)
        or not candidate_snapshot
        or any(not isinstance(act, DiscussionAct) for act in candidate_snapshot)
    ):
        raise ValueError(
            "candidate_snapshot must be a non-empty DiscussionAct tuple"
        )
    if not isinstance(claim_ids, tuple) or any(
        not isinstance(claim_id, str) for claim_id in claim_ids
    ):
        raise TypeError("claim_ids must be a tuple of strings")

    unresolved_players = list(role_options)
    content_indices = project_discussion_content_indices(candidate_snapshot)
    vote_stances = project_discussion_vote_stances(candidate_snapshot)
    evidence_items = (
        {"type": "string", "enum": list(claim_ids)}
        if claim_ids
        else {"type": "string"}
    )
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "day_cognition_report_v2",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "belief",
                    "concise",
                    "roles",
                    "public_content_action_indices",
                    "public_vote_stance_index",
                    "evidence_claim_ids",
                ],
                "properties": {
                    "belief": {
                        "type": "string",
                        "minLength": COGNITION_TEXT_MIN_LENGTH,
                        "maxLength": BELIEF_TEXT_MAX_LENGTH,
                    },
                    "concise": {
                        "type": "string",
                        "minLength": COGNITION_TEXT_MIN_LENGTH,
                        "maxLength": CONCISE_TEXT_MAX_LENGTH,
                    },
                    "roles": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": unresolved_players,
                        "properties": {
                            player: {
                                "type": "string",
                                "enum": list(role_options[player]),
                            }
                            for player in unresolved_players
                        },
                    },
                    "public_content_action_indices": {
                        "type": "array",
                        "minItems": 0,
                        "maxItems": 2,
                        "items": {
                            "type": "integer",
                            "enum": list(content_indices),
                        },
                    },
                    "public_vote_stance_index": {
                        "type": "integer",
                        "enum": list(range(len(vote_stances))),
                    },
                    "evidence_claim_ids": {
                        "type": "array",
                        "minItems": 0,
                        "maxItems": min(2, len(claim_ids)),
                        "items": evidence_items,
                    },
                },
            },
        },
    }


def parse_belief_response(
    raw_response,
    *,
    player_id,
    self_role,
    phase,
    exact_roles,
    role_options,
):
    context = f"player={player_id}, phase={phase}"
    try:
        payload = json.loads(raw_response)
    except (TypeError, json.JSONDecodeError) as exc:
        raise BeliefValidationError(
            f"belief response is not valid JSON ({context})"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {"belief", "concise", "roles"}:
        raise BeliefValidationError(f"belief fields do not match contract ({context})")
    if any(not isinstance(payload[field], str) or not payload[field].strip() for field in ("belief", "concise")):
        raise BeliefValidationError(f"belief text fields must be non-empty ({context})")
    for field, max_length in (
        ("belief", BELIEF_TEXT_MAX_LENGTH),
        ("concise", CONCISE_TEXT_MAX_LENGTH),
    ):
        if len(payload[field]) > max_length:
            raise BeliefValidationError(
                f"{field} exceeds {max_length} characters ({context})"
            )
    final_players = [
        f"player{candidate}"
        for candidate in range(1, 8)
        if candidate != player_id
    ]
    expected_players = set(role_options)
    if (
        not isinstance(exact_roles, dict)
        or set(exact_roles) & expected_players
        or set(exact_roles) | expected_players != set(final_players)
    ):
        raise BeliefValidationError(
            f"belief constraints do not cover the other players ({context})"
        )
    roles = payload["roles"]
    if not isinstance(roles, dict) or set(roles) != expected_players:
        raise BeliefValidationError(
            f"roles must contain only unresolved players ({context})"
        )
    if any(not isinstance(role, str) for role in roles.values()):
        raise BeliefValidationError(
            f"role report values must be strings ({context})"
        )
    if self_role not in STRICT_CLASSIC7_ROLE_COUNTS:
        raise BeliefValidationError(f"unsupported self role ({context})")

    final_roles = {
        player: exact_roles[player] if player in exact_roles else roles[player]
        for player in final_players
    }
    return BeliefReport(
        belief=payload["belief"].strip(),
        concise=payload["concise"].strip(),
        roles=final_roles,
    )


def parse_day_cognition_response_v2(
    raw_response,
    *,
    player_id,
    self_role,
    phase,
    exact_roles,
    role_options,
    candidate_snapshot,
    claim_ids,
):
    context = f"player={player_id}, phase={phase}"
    try:
        payload = json.loads(raw_response)
    except (TypeError, json.JSONDecodeError) as exc:
        raise BeliefValidationError(
            f"Day cognition response is not valid JSON ({context})"
        ) from exc
    required_fields = {
        "belief",
        "concise",
        "roles",
        "public_content_action_indices",
        "public_vote_stance_index",
        "evidence_claim_ids",
    }
    if not isinstance(payload, dict) or set(payload) != required_fields:
        raise BeliefValidationError(
            f"Day cognition fields do not match contract ({context})"
        )
    if (
        not isinstance(candidate_snapshot, tuple)
        or not candidate_snapshot
        or any(not isinstance(act, DiscussionAct) for act in candidate_snapshot)
    ):
        raise BeliefValidationError(
            f"invalid discussion candidate snapshot ({context})"
        )
    if not isinstance(claim_ids, tuple) or any(
        not isinstance(claim_id, str) for claim_id in claim_ids
    ):
        raise BeliefValidationError(f"invalid public claim catalog ({context})")

    content_indices = payload["public_content_action_indices"]
    valid_content_indices = set(
        project_discussion_content_indices(candidate_snapshot)
    )
    if (
        not isinstance(content_indices, list)
        or len(content_indices) > 2
        or any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in content_indices
        )
        or len(set(content_indices)) != len(content_indices)
        or any(index not in valid_content_indices for index in content_indices)
    ):
        raise BeliefValidationError(
            f"invalid public_content_action_indices ({context})"
        )

    vote_stance_index = payload["public_vote_stance_index"]
    vote_stances = project_discussion_vote_stances(candidate_snapshot)
    if (
        isinstance(vote_stance_index, bool)
        or not isinstance(vote_stance_index, int)
        or not 0 <= vote_stance_index < len(vote_stances)
    ):
        raise BeliefValidationError(
            f"invalid public_vote_stance_index ({context})"
        )

    evidence_claim_ids = payload["evidence_claim_ids"]
    if (
        not isinstance(evidence_claim_ids, list)
        or len(evidence_claim_ids) > 2
        or any(not isinstance(claim_id, str) for claim_id in evidence_claim_ids)
        or len(set(evidence_claim_ids)) != len(evidence_claim_ids)
        or any(claim_id not in claim_ids for claim_id in evidence_claim_ids)
    ):
        raise BeliefValidationError(
            f"invalid evidence_claim_ids ({context})"
        )

    belief_report = parse_belief_response(
        json.dumps(
            {
                "belief": payload["belief"],
                "concise": payload["concise"],
                "roles": payload["roles"],
            }
        ),
        player_id=player_id,
        self_role=self_role,
        phase=phase,
        exact_roles=exact_roles,
        role_options=role_options,
    )
    return DayCognitionReportV2(
        belief=belief_report.belief,
        concise=belief_report.concise,
        roles=belief_report.roles,
        public_content_action_indices=tuple(content_indices),
        public_vote_stance_index=vote_stance_index,
        evidence_claim_ids=tuple(evidence_claim_ids),
    )


def validate_role_report(
    report,
    *,
    player_id,
    self_role,
    phase,
    exact_roles,
    role_options,
):
    """Validate one role report without judging subjective guesses by truth."""

    context = f"player={player_id}, phase={phase}"
    if not isinstance(report, (BeliefReport, DayCognitionReportV2)):
        raise RoleReportValidationError(f"invalid role report ({context})")
    final_players = {
        f"player{candidate}"
        for candidate in range(1, 8)
        if candidate != player_id
    }
    if set(report.roles) != final_players:
        raise RoleReportValidationError(
            f"role report must contain the other players ({context})"
        )
    for player, role in exact_roles.items():
        if report.roles.get(player) != role:
            raise RoleReportValidationError(
                f"role report contradicts exact-known {player} ({context})"
            )
    for player, options in role_options.items():
        if report.roles.get(player) not in options:
            raise RoleReportValidationError(
                f"role report violates the legal domain for {player} ({context})"
            )

    role_counts = {role: 0 for role in STRICT_BELIEF_CONCRETE_ROLES}
    if self_role not in role_counts:
        raise RoleReportValidationError(f"unsupported self role ({context})")
    role_counts[self_role] += 1
    for role in report.roles.values():
        if role == "unknown":
            continue
        if role not in role_counts:
            raise RoleReportValidationError(
                f"role report contains an unsupported role ({context})"
            )
        role_counts[role] += 1
    for role, count in role_counts.items():
        if count > STRICT_CLASSIC7_ROLE_COUNTS[role]:
            raise RoleReportValidationError(
                f"role report exceeds fixed {role} inventory ({context})"
            )
    return report


def vote_response_format(*, supports_json_schema, legal_targets):
    if supports_json_schema is not True:
        raise BackendError("vote generation requires backend JSON Schema support")
    if not isinstance(legal_targets, tuple) or not legal_targets:
        raise ValueError("legal_targets must be a non-empty tuple")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "vote",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["target"],
                "properties": {
                    "target": {"type": "integer", "enum": list(legal_targets)},
                },
            },
        },
    }


def parse_vote_response(raw_response, *, legal_targets, phase):
    try:
        payload = json.loads(raw_response)
    except (TypeError, json.JSONDecodeError) as exc:
        raise GameplayActionValidationError(
            f"vote response is not valid JSON (phase={phase!r})"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {"target"}:
        raise GameplayActionValidationError(
            f"vote response must contain only target (phase={phase!r})"
        )
    target = payload["target"]
    if isinstance(target, bool) or not isinstance(target, int) or target not in legal_targets:
        raise GameplayActionValidationError(
            f"vote target is not an authoritative candidate (phase={phase!r}, target={target!r})"
        )
    return target


def night_action_response_format(
    *, supports_json_schema, candidate_snapshot
):
    if supports_json_schema is not True:
        raise BackendError(
            "constrained night actions require backend JSON Schema support"
        )
    if not isinstance(candidate_snapshot, tuple) or not candidate_snapshot:
        raise ValueError("candidate_snapshot must be a non-empty tuple")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "night_action_selection",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["action_index"],
                "properties": {
                    "action_index": {
                        "type": "integer",
                        "enum": list(range(len(candidate_snapshot))),
                    },
                },
            },
        },
    }


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
_GROUPED_PLAYER_REFERENCE = re.compile(
    r"(?P<players>(?:\d+|[零一二三四五六七八九])"
    r"(?:\s*[、，,]\s*(?:\d+|[零一二三四五六七八九]))+)"
    r"\s*号(?:玩家|位)?"
)
_GROUPED_PLAYER_NUMBER = re.compile(r"\d+|[零一二三四五六七八九]")
_SPEAKER_SELF_REFERENCE = re.compile(
    r"我(?:自己)?"
    r"|(?<!他)(?<!她)(?<!它)(?<!你)(?<!您)"
    r"(?<!他们)(?<!她们)(?<!它们)(?<!你们)(?<!您们)自己"
)


def _extract_explicit_player_references(content, *, speaker_id, context):
    referenced_players = set()
    for match in _GROUPED_PLAYER_REFERENCE.finditer(content):
        for value in _GROUPED_PLAYER_NUMBER.findall(match.group("players")):
            player_number = (
                int(value)
                if value.isdigit()
                else _CHINESE_PLAYER_NUMBERS[value]
            )
            if not 1 <= player_number <= 7:
                raise GameplaySpeechQualityError(
                    f"invalid player reference {value!r} ({context})"
                )
            referenced_players.add(player_number)
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
    if (
        isinstance(speaker_id, int)
        and not isinstance(speaker_id, bool)
        and 1 <= speaker_id <= 7
        and _SPEAKER_SELF_REFERENCE.search(content)
    ):
        referenced_players.add(speaker_id)
    return referenced_players


def validate_gameplay_public_speech(
    content,
    *,
    finish_reason=None,
    player_id=None,
    phase=None,
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

    _extract_explicit_player_references(
        content,
        speaker_id=player_id,
        context=context,
    )

    stripped = content.lstrip()
    if stripped.startswith(("{", "[", "```", "# ")):
        raise GameplaySpeechQualityError(
            f"structured gameplay public speech output ({context})"
        )
    forbidden_control_text = (
        "GAME / ROLE",
        "KNOWN INFORMATION",
        "AUTHORITATIVE INFORMATION",
        "PUBLIC CONVERSATION",
        "CURRENT PRIVATE BELIEF",
        "FRESH PRIVATE BELIEF",
        "BELIEF OUTPUT",
        "COMMON PUBLIC RULES",
        "PUBLIC AUTHORITATIVE INFORMATION",
        "DISCUSSION INTENT",
        "SELECTED PUBLIC EVIDENCE",
        "Environment authoritative public state",
        "Authoritative public history (chronological)",
        "Private facts legally visible to this player",
        "Actual role supplied by the Environment",
        "【权威公共状态】",
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
                "gen_times": player_log_context.get("gen_times", 0),
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
        action_candidates=None,
    ):
        phase = observation['phase']
        if 'skill' in phase or 'vote' in phase:
            valid_actions = observation['valid_action']
            if action_candidates is None:
                valid_actions_str = self.get_valid_actions_str(valid_actions)
            else:
                valid_actions_str = self.format_authoritative_action_candidates(
                    action_candidates,
                )
            identity = observation['identity']
            identity_info = CON.player_identity_info.format(player_idx=observation['current_act_idx'],
                                                            identity=CON.identity_chinese[identity],
                                                            identity_ability=CON.identity_abilities[identity])
            logs = self.format_log(observation['game_log'])
            if 'skill' in phase:
                template = (
                    CON.constrained_night_skill_prompt
                    if action_candidates is not None
                    else CON.skill_prompt
                )
                game_description = (
                    STRICT_CLASSIC7_GAME_DESCRIPTION
                    if self.gameplay_prompt_profile
                    == STRICT_CLASSIC7_GAMEPLAY_PROMPT_PROFILE
                    else CON.game_description
                )
                prompt = template.format(game_description=game_description,
                                         player_identity_info=identity_info, logs=logs,
                                         valid_actions=valid_actions_str)
            else:
                prompt = CON.vote_prompt.format(game_description=CON.game_description,
                                                player_identity_info=identity_info, logs=logs,
                                                valid_actions=valid_actions_str)
        elif 'speech' in phase:
            identity = observation['identity']
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

    def match_authoritative_action_response(
        self,
        raw_response,
        authoritative_action_strings,
    ):
        parsed_response = self._extract_json_like(raw_response)
        if not isinstance(parsed_response, dict):
            return None

        matches = [
            candidate
            for candidate in authoritative_action_strings
            if self._extract_json_like(candidate) == parsed_response
        ]
        return matches[0] if len(matches) == 1 else None

    def parse_night_action_selection(
        self,
        raw_response,
        candidate_snapshot,
        *,
        phase,
    ):
        def reject(reason):
            raise GameplayActionValidationError(
                f"invalid night action selection: {reason} "
                f"(phase={phase!r}, response={raw_response!r})"
            )

        try:
            payload = json.loads(raw_response)
        except (TypeError, json.JSONDecodeError):
            reject("response must be strict JSON")
        if not isinstance(payload, dict):
            reject("root must be an object")
        if set(payload) != {"action_index"}:
            reject("keys must be exactly {'action_index'}")
        action_index = payload["action_index"]
        if isinstance(action_index, bool) or not isinstance(action_index, int):
            reject("action_index must be an integer")
        if not 0 <= action_index < len(candidate_snapshot):
            reject("action_index is outside the authoritative candidates")

        selected_action, env_action = candidate_snapshot[action_index]
        authoritative_actions = tuple(
            action for action, _env_action in candidate_snapshot
        )
        matched_action = self.match_authoritative_action_response(
            selected_action,
            authoritative_actions,
        )
        if matched_action != selected_action:
            reject("selected action failed authoritative membership")
        return matched_action, env_action

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
            action
            for action in valid_action
            if self.parse_vote_target(action) not in (None, 0)
        ]
        if non_abstain_actions:
            return random.choice(non_abstain_actions)

        abstain_action = self.vote_target_to_action_str(0)
        if abstain_action in valid_action:
            return abstain_action
        return valid_action[0] if valid_action else abstain_action

    def freeze_legal_vote_candidates(self, valid_actions, *, phase):
        expected_action = "vote_pk" if "vote_pk" in phase else "vote"
        candidates = []
        seen_targets = set()
        for action_type, target in valid_actions:
            if action_type != expected_action or isinstance(target, bool):
                continue
            if not isinstance(target, int) or not 0 <= target <= 7:
                continue
            if target in seen_targets:
                raise GameplayActionValidationError(
                    f"duplicate authoritative vote target {target} (phase={phase!r})"
                )
            seen_targets.add(target)
            candidates.append((target, (action_type, target)))
        if not candidates:
            raise GameplayActionValidationError(
                f"no legal vote candidates (phase={phase!r})"
            )
        return tuple(candidates)

    def freeze_authoritative_action_candidates(self, valid_actions):
        self.get_valid_actions_str(valid_actions)
        return tuple(self.nlp_action_to_env_action.items())

    def format_authoritative_action_candidates(self, candidate_snapshot):
        self.nlp_action_to_env_action = dict(candidate_snapshot)
        return "".join(
            f"{index}: {action_text}\n"
            for index, (action_text, _env_action) in enumerate(
                candidate_snapshot
            )
        )

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
