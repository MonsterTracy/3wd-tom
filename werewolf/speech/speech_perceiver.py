"""Parse public Werewolf speech into ONUW-style speech actions."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


SPEECH_PARSER_MAX_TOKENS = 256

_PIPE_TRIPLET_PATTERN = re.compile(
    r"^\s*[\"'`]?\s*"
    r"(?P<subject>(?:player\s*)?[1-7])"
    r"\s*[|｜]\s*"
    r"(?P<action>[a-zA-Z_]+)"
    r"\s*[|｜]\s*"
    r"(?P<object>(?:player\s*[1-7]\s*至\s*player\s*[1-7]|"
    r"(?:player\s*)?[1-7]|NONE|null|<none>))"
    r"\s*[\"'`]?\s*[,;，；。.！!]?\s*$",
    flags=re.IGNORECASE,
)

_BULLET_PREFIX_PATTERN = re.compile(
    r"^\s*(?:[-*•]+\s*|\d+\s*[.)、:：-]\s*)"
)

_EXPLICIT_PLAYER_RANGE_PATTERN = re.compile(
    r"^player\s*(?P<start>[1-7])\s*至\s*player\s*(?P<end>[1-7])$",
    flags=re.IGNORECASE,
)

_EMPTY_RESPONSE_MARKERS = {
    "none",
    "no action",
    "no actions",
    "null",
    "[]",
    "无",
    "无动作",
    "没有动作",
}

_SELF_ROLE_ACTIONS = {
    "狼人": "point_as_werewolf",
    "村民": "point_as_villager",
    "平民": "point_as_villager",
    "预言家": "point_as_seer",
    "女巫": "point_as_witch",
}

# This intentionally covers only literal first-person role declarations.
# It does not infer roles from "好人", negations, conditions, quotations,
# other-player reports, or hidden game state.
_SELF_ROLE_CLAIM_PATTERN = re.compile(
    r"(?:^|[。！？；;\n])\s*"
    r"(?:"
    r"我是\s*(?:[1-7]\s*号(?:\s*玩家)?)?\s*[，,\s]*"
    r"(?:身份(?:是|为|：|:)\s*)?"
    r"|我的身份(?:是|为|：|:)\s*"
    r"|我身份(?:是|为|：|:)\s*"
    r")"
    r"(?:一名|一个|普通的?)?\s*"
    r"(?P<role>狼人|村民|平民|预言家|女巫)"
)


def _load_tom_schema():
    """Load the ToM schema lazily to avoid package import cycles."""

    from werewolf.models.twd_tom.schema import (
        ACTION_NAMES,
        SpeechAction,
    )

    return ACTION_NAMES, SpeechAction


class SpeechActionValidationError(ValueError):
    """Report candidates that violate the speech-action schema."""

    def __init__(
        self,
        failures: list[dict[str, Any]],
        *,
        raw_response: str | None = None,
    ):
        self.failures = failures
        self.invalid_count = len(failures)
        self._raw_response = raw_response
        super().__init__(
            "invalid speech action candidate(s): "
            + json.dumps(
                failures,
                ensure_ascii=False,
                default=repr,
            )
        )

    @property
    def raw_response(self) -> str | None:
        """Return the unchanged backend text when one was produced."""

        return self._raw_response


@dataclass(frozen=True)
class SpeechParseAuditResult:
    """One online parse result with the actual backend response."""

    normalized_actions: list[list[str | None]]
    raw_response: str | None
    parse_status: str
    protected_self_claim_actions: list[list[str | None]]
    error_type: str | None
    error_message: str | None


class SpeechPerceiver:
    """Convert one public speech turn into structured speech actions.

    Returned actions always use the ONUW-compatible format::

        [subject, action, object]

    The preferred LLM response is one pipe-delimited triplet per line::

        player2 | point_as_werewolf | player5
        player2 | oppose | player3

    ``NONE`` represents a valid speech with no extractable action. Legacy
    JSON-array responses remain readable so existing logs and backends do
    not break during the format migration.

    This parser only receives public speech text. It does not construct
    private subjective guesses and never receives the game's true roles.
    """

    def __init__(self, backend=None, model_name=None):
        self.backend = backend
        self.model_name = model_name

    def parse(
        self,
        speaker: int,
        speech: str,
        day: int,
        phase: str,
        context: dict | None = None,
    ) -> list[list[str | None]]:
        """Parse one speech turn without interrupting the game on failure."""

        del context
        return self.parse_with_audit(
            speaker=speaker,
            speech=speech,
            day=day,
            phase=phase,
        ).normalized_actions

    def parse_with_audit(
        self,
        speaker: int,
        speech: str,
        day: int,
        phase: str,
    ) -> SpeechParseAuditResult:
        """Parse once through the online tolerant path and retain audit data."""

        precondition_error = None
        if self.backend is None or not self.model_name:
            precondition_error = RuntimeError(
                "speech parser backend and model must be configured"
            )
        elif type(speaker) is not int or not 1 <= speaker <= 7:
            precondition_error = ValueError(
                "speaker must be an integer in [1, 7]"
            )
        elif not isinstance(speech, str) or not speech.strip():
            precondition_error = ValueError("speech must be non-empty text")
        if precondition_error is not None:
            return SpeechParseAuditResult(
                normalized_actions=[],
                raw_response=None,
                parse_status="parser_error",
                protected_self_claim_actions=[],
                error_type=type(precondition_error).__name__,
                error_message=str(precondition_error),
            )

        protected_actions = self._extract_explicit_self_claim_actions(
            speaker=speaker,
            speech=speech,
        )
        try:
            actions, raw_response = self._parse_configured_with_response(
                speaker=speaker,
                speech=speech,
                day=day,
                phase=phase,
                protected_actions=protected_actions,
            )
            return SpeechParseAuditResult(
                normalized_actions=actions,
                raw_response=raw_response,
                parse_status="ok",
                protected_self_claim_actions=protected_actions,
                error_type=None,
                error_message=None,
            )
        except Exception as exc:
            raw_response = getattr(exc, "raw_response", None)
            if not isinstance(raw_response, str):
                raw_response = None
            return SpeechParseAuditResult(
                normalized_actions=[],
                raw_response=raw_response,
                parse_status="parser_error",
                protected_self_claim_actions=protected_actions,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    def parse_strict(
        self,
        speaker: int,
        speech: str,
        day: int,
        phase: str,
    ) -> list[list[str | None]]:
        """Parse one speech while exposing configuration and parser errors.

        This entry point is for offline audits and reparsing only. The online
        environment continues to call :meth:`parse`, which remains fail-closed
        so a parser outage cannot interrupt a game.
        """

        actions, _raw_response = (
            self.parse_strict_with_response(
                speaker=speaker,
                speech=speech,
                day=day,
                phase=phase,
            )
        )
        return actions

    def parse_strict_with_response(
        self,
        speaker: int,
        speech: str,
        day: int,
        phase: str,
    ) -> tuple[list[list[str | None]], str]:
        """Strictly parse once and return actions plus unchanged response."""

        if self.backend is None or not self.model_name:
            raise RuntimeError(
                "speech parser backend and model must be configured"
            )

        if type(speaker) is not int or not 1 <= speaker <= 7:
            raise ValueError(
                "speaker must be an integer in [1, 7]"
            )

        if not isinstance(speech, str) or not speech.strip():
            raise ValueError(
                "speech must be non-empty text"
            )

        protected_actions = (
            self._extract_explicit_self_claim_actions(
                speaker=speaker,
                speech=speech,
            )
        )

        return self._parse_configured_with_response(
            speaker=speaker,
            speech=speech,
            day=day,
            phase=phase,
            protected_actions=(
                protected_actions
            ),
            strict=True,
        )

    def _parse_configured(
        self,
        *,
        speaker: int,
        speech: str,
        day: int,
        phase: str,
        protected_actions: Sequence[
            Sequence[str]
        ],
        strict: bool = False,
    ) -> list[list[str | None]]:
        actions, _raw_response = (
            self._parse_configured_with_response(
                speaker=speaker,
                speech=speech,
                day=day,
                phase=phase,
                protected_actions=protected_actions,
                strict=strict,
            )
        )
        return actions

    def _parse_configured_with_response(
        self,
        *,
        speaker: int,
        speech: str,
        day: int,
        phase: str,
        protected_actions: Sequence[
            Sequence[str]
        ],
        strict: bool = False,
    ) -> tuple[list[list[str | None]], str]:
        prompt = self._build_prompt(
            speaker=speaker,
            speech=speech,
            day=day,
            phase=phase,
        )

        response_text = self.backend.chat(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model_name,
            temperature=0,
            max_tokens=(
                SPEECH_PARSER_MAX_TOKENS
            ),
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False,
                }
            },
        )

        try:
            parsed = self._extract_response_actions(
                response_text,
                strict=strict,
            )

            llm_actions = self._normalize(
                parsed=parsed,
                speaker=speaker,
                strict=strict,
            )
        except SpeechActionValidationError as exc:
            raise SpeechActionValidationError(
                exc.failures,
                raw_response=response_text,
            ) from exc
        except ValueError as exc:
            exc.raw_response = response_text
            raise
        except Exception as exc:
            exc.raw_response = response_text
            raise

        return (
            self._merge_actions(
                protected_actions,
                llm_actions,
            ),
            response_text,
        )

    @staticmethod
    def _build_prompt(
        speaker: int,
        speech: str,
        day: int,
        phase: str,
    ) -> str:
        """Build the formal public-speech parsing prompt."""

        action_names, _ = _load_tom_schema()
        allowed_actions = "\n".join(
            f"- {action_name}"
            for action_name in action_names
        )

        return f"""你是狼人杀公开发言的结构化动作解析器。

只抽取当前发言者在下面公开发言中明确表达的命题。不要判断真假，不读取隐藏身份，不推测私下想法。

当前发言者：player{speaker}
当前天数：Day {day}
当前阶段：{phase}
玩家范围：player1 到 player7

每个动作必须使用三元组：
subject | action | object

subject 必须是当前发言者 player{speaker}。
带目标动作的 object 必须是 player1 到 player7；无目标动作的 object 必须是 NONE。
允许的 action 只有：
{allowed_actions}

动作语义：
1. point_as_werewolf：明确声称或判断目标是狼人；如果狼人结论来自speaker自己的查验声明，改用check_as_werewolf。
2. point_as_non_werewolf：明确判断目标不是狼人、是泛化“好人”或属于好人阵营，但没有断言其具体角色。
3. point_as_villager：只有明确判断目标的具体身份是 Villager、村民或平民时使用。
4. point_as_seer：明确判断目标是预言家。
5. point_as_witch：明确判断目标是女巫。
6. support：明确支持、认可、站边目标玩家或其观点；不能从“好人”“村民”或查验非狼自动推导。
7. oppose：明确反对、不信任、质疑目标玩家或其观点；不能从狼人判断、查杀或投票意图自动推导。
8. check_as_non_werewolf：speaker明确声称自己通过查验或验人得到目标是好人或非狼的结果。
9. check_as_werewolf：speaker明确声称自己通过查验或验人得到目标是狼人的结果。
10. save：speaker明确公开声称自己救了目标。
11. poison：speaker明确公开声称自己毒了目标。
12. vote_intent：speaker明确表达自己当前准备、打算或决定把票投给目标；实际环境vote是另一类独立public event。
13. abstain_intent：speaker明确表示当前轮次准备弃票；使用 object=NONE。
14. no_commitment：speaker明确表示本轮暂不作身份、查验、技能或投票表态；使用 object=NONE。

抽取规则：
- 只抽取发言直接表达的命题，不得根据常识或其他动作推导。
- 对普通的带目标动作，目标必须在支持该动作的同一个明确命题或分句中，以 playerN 或 N号被明确点名。
- 不得从代词、省略宾语、“这个/那个玩家/他/她”、前一句、前一个action、discourse context或多个可能antecedent之一推断目标；动作含义明确但同一支持命题未显式点名目标时，不输出该动作。
- 第一人称明确自报具体身份（如“我是预言家”）中，“我”明确指向当前发言者，仍必须抽取对当前speaker的 point_as_* ；这不是target推断。
- 原文明示的显式连续编号范围（如“2号到4号”）仍按范围顺序展开每个目标；这不是discourse antecedent推断。
- 穷尽抽取所有明确属于上述可表示类别的命题，多个不同命题按原文语义顺序输出；即使命题彼此冲突、是谎言、不符合speaker真实角色或策略上荒谬，也不得truth-filter或静默漏掉。
- 第一人称明确自报具体身份必须抽取。
- “质疑”“可疑”“狼面大”“需要关注”只支持 oppose，不得自动升级为 point_as_werewolf。
- 对明确目标说“可信”只支持support，不得自动升级为point_as_non_werewolf或具体角色判断。
- most-specific-source：同一个semantic claim只使用最具体predicate，不得从一个specific claim自动派生generic actions。
- 泛化“好人”“非狼”“好人阵营”产生point_as_non_werewolf，不得具体化为point_as_villager、point_as_seer或point_as_witch。
- 查验来源的好人/非狼只产生check_as_non_werewolf，不自动产生point_as_non_werewolf、point_as_villager或support；查验来源的狼人只产生check_as_werewolf，不自动产生point_as_werewolf、oppose或vote_intent。
- 只有原文另外、独立地明确表达第二个formal proposition时，才允许为同一目标输出第二个action。
- save、poison只表示speaker公开声称的技能动作，不自动产生任何身份判断；不得读取真实角色或环境技能记录进行truth validation。
- vote_intent不等于环境实际vote，也不自动产生oppose；“大家应该关注3号”不构成speaker自己的投票意图。
- “查验 player5 是好人”等查验结果单独存在时不得产生 support；只有另有“支持、相信、说得对”等明确认可文本才产生 support。
- 查验结果为“好人”不得产生 point_as_villager、point_as_seer 或 point_as_witch；只有另外明确说出具体角色判断时才抽取对应 point_as_*。
- “player4 是预言家”等明确具体角色判断必须抽取；parser只忠实表示原文，不判断游戏机制是否合理。
- 转述别人的身份声明或立场，不视为当前发言者自己的立场。
- “我是好人”和“我不是狼人”产生针对speaker自己的point_as_non_werewolf，不能产生point_as_villager。
- 连续玩家范围必须按原文顺序展开成多个原子三元组。
- 删除完全重复的动作，不输出真实角色、guesses、置信度或解释。

示例：
输入：我是预言家，3号是狼人。
输出：
player{speaker} | point_as_seer | player{speaker}
player{speaker} | point_as_werewolf | player3

输入：我是6号玩家，身份是村民。
输出：
player{speaker} | point_as_villager | player{speaker}

输入：我不同意3号的逻辑，但支持4号。
输出：
player{speaker} | oppose | player3
player{speaker} | support | player4

输入：昨晚我查验2号是好人，救了3号，今天投4号。
输出：
player{speaker} | check_as_non_werewolf | player2
player{speaker} | save | player3
player{speaker} | vote_intent | player4

输入：我昨晚查验3号是狼人。
输出：
player{speaker} | check_as_werewolf | player3

输入：我是女巫，昨晚毒了5号。
输出：
player{speaker} | point_as_witch | player{speaker}
player{speaker} | poison | player5

输入：3号是好人。
输出：
player{speaker} | point_as_non_werewolf | player3

输入：我反对 player2 至 player4 的发言。
输出：
player{speaker} | oppose | player2
player{speaker} | oppose | player3
player{speaker} | oppose | player4

输入：基于以上判断，我投这一票。我就投他。那我投这个人。
输出：
NONE

输入：player2和player3都在发言。那我投他。
输出：
NONE

输入：这一轮我投4号。
输出：
player{speaker} | vote_intent | player4

输入：这一轮我选择弃票。
输出：
player{speaker} | abstain_intent | NONE

输入：这一轮我暂不作明确表态。
输出：
player{speaker} | no_commitment | NONE

输出协议：
- 每个动作单独一行，格式严格为：subject | action | object
- 穷尽输出所有明确动作，不要重复动作。
- 没有可抽取动作时，只输出：NONE
- 不输出 JSON、解释或 Markdown 代码块。

待解析的玩家发言：
player{speaker}: {speech}"""

    @classmethod
    def _extract_explicit_self_claim_actions(
        cls,
        *,
        speaker: int,
        speech: str,
    ) -> list[list[str | None]]:
        """Protect literal first-person public role declarations.

        This is deliberately narrow. It only preserves an identity that the
        current speaker explicitly states about themself in the public text.
        It never reads the player's real role, observation, or private state.
        """

        _, speech_action_type = (
            _load_tom_schema()
        )

        actions: list[list[str | None]] = []
        seen: set[
            tuple[str | None, str | None, str | None]
        ] = set()

        for match in (
            _SELF_ROLE_CLAIM_PATTERN.finditer(
                speech
            )
        ):
            action_name = _SELF_ROLE_ACTIONS[
                match.group("role")
            ]

            try:
                action = (
                    speech_action_type
                    .from_values(
                        subject=speaker,
                        action=action_name,
                        object_=speaker,
                    )
                )
            except (
                TypeError,
                ValueError,
                KeyError,
            ):
                continue

            normalized = action.to_list()
            key = tuple(normalized)

            if key in seen:
                continue

            seen.add(key)
            actions.append(normalized)

        return actions

    @staticmethod
    def _merge_actions(
        *action_groups: Sequence[
            Sequence[str | None]
        ],
    ) -> list[list[str | None]]:
        """Merge action groups while retaining distinct action triplets."""

        merged: list[list[str | None]] = []
        seen: set[
            tuple[str | None, str | None, str | None]
        ] = set()

        for action_group in action_groups:
            for action in action_group:
                if (
                    isinstance(
                        action,
                        (str, bytes),
                    )
                    or len(action) != 3
                ):
                    continue

                normalized = list(action)
                key = tuple(normalized)

                if key in seen:
                    continue

                seen.add(key)
                merged.append(normalized)

        return merged

    @classmethod
    def _extract_response_actions(
        cls,
        response_text: str,
        *,
        strict: bool = False,
    ) -> list:
        """Read preferred pipe triplets with legacy JSON compatibility."""

        if not isinstance(response_text, str):
            raise ValueError(
                "LLM response content must be text."
            )

        text = response_text.strip()

        if not text:
            raise ValueError(
                "LLM response content is empty."
            )

        if strict:
            meaningful_lines = [
                line.strip()
                for line in text.splitlines()
                if line.strip()
            ]

            if (
                len(meaningful_lines) == 1
                and "`" not in meaningful_lines[0]
                and cls._is_empty_marker(
                    meaningful_lines[0]
                )
            ):
                return []
        elif cls._is_empty_marker(text):
            return []

        # The new ONUW-style format has priority.
        pipe_actions = cls._extract_pipe_triplets(
            text,
            preserve_invalid=strict,
        )

        if pipe_actions:
            return pipe_actions

        # Keep old JSON responses readable during migration.
        try:
            json_actions = cls._extract_json_array(
                text
            )
        except ValueError:
            pass
        else:
            if strict:
                try:
                    exact_json = json.loads(text)
                except json.JSONDecodeError:
                    exact_json = None

                if not isinstance(exact_json, list):
                    raise SpeechActionValidationError(
                        [
                            {
                                "candidate": text,
                                "reason": (
                                    "legacy JSON action output contains "
                                    "extra non-protocol text"
                                ),
                            }
                        ]
                    )

            return json_actions

        if not strict and cls._contains_empty_marker(
            text
        ):
            return []

        raise ValueError(
            "No structured speech action found in LLM response."
        )

    @staticmethod
    def _extract_json_array(
        response_text: str,
    ) -> list:
        """Extract the first JSON array from a legacy response."""

        if not isinstance(response_text, str):
            raise ValueError(
                "LLM response content must be text."
            )

        text = response_text.strip()

        try:
            parsed = json.loads(
                text
            )

            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

        fenced_blocks = re.findall(
            r"```(?:json)?\s*([\s\S]*?)\s*```",
            text,
            flags=re.IGNORECASE,
        )

        for fenced_text in fenced_blocks:
            try:
                parsed = json.loads(
                    fenced_text
                )

                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                continue

        decoder = json.JSONDecoder()

        for index, character in enumerate(
            text
        ):
            if character != "[":
                continue

            try:
                parsed, _ = decoder.raw_decode(
                    text[index:]
                )
            except json.JSONDecodeError:
                continue

            if isinstance(parsed, list):
                return parsed

        raise ValueError(
            "No JSON array found in LLM response."
        )

    @classmethod
    def _extract_pipe_triplets(
        cls,
        response_text: str,
        *,
        preserve_invalid: bool = False,
    ) -> list[list[str | None]]:
        """Extract strict ONUW-style pipe triplets from separate lines."""

        if preserve_invalid:
            cleaned_text = response_text
        else:
            cleaned_text = re.sub(
                r"```(?:[a-zA-Z0-9_-]+)?\s*",
                "",
                response_text,
                flags=re.IGNORECASE,
            ).replace(
                "```",
                "",
            )

        actions: list[list[str | None]] = []
        failures: list[dict[str, Any]] = []
        lines: list[tuple[str, str]] = []

        for raw_line in cleaned_text.splitlines():
            original_line = raw_line.strip()

            if not original_line:
                continue

            if preserve_invalid:
                line = original_line
            else:
                # Tolerate accidental bullets or numbered lists online.
                line = _BULLET_PREFIX_PATTERN.sub(
                    "",
                    original_line,
                    count=1,
                )

            lines.append(
                (original_line, line)
            )

        recognized_protocol = any(
            re.search(r"[|｜]", line)
            or cls._is_empty_marker(line)
            for _, line in lines
        )

        for original_line, line in lines:

            match = _PIPE_TRIPLET_PATTERN.fullmatch(
                line
            )

            if match is None:
                if preserve_invalid and recognized_protocol:
                    reason = (
                        "NONE must be the only non-empty response line"
                        if cls._is_empty_marker(line)
                        else (
                            "non-empty response line does not match "
                            "the pipe triplet protocol"
                        )
                    )
                    failures.append(
                        {
                            "candidate": original_line,
                            "reason": reason,
                        }
                    )
                continue

            actions.append(
                [
                    match.group(
                        "subject"
                    ),
                    match.group(
                        "action"
                    ),
                    match.group(
                        "object"
                    ),
                ]
            )

        if failures:
            raise SpeechActionValidationError(
                failures
            )

        return actions

    @staticmethod
    def _normalized_marker_text(
        text: str,
    ) -> str:
        return re.sub(
            r"[\s.!！。`'\"]+",
            " ",
            text.strip().lower(),
        ).strip()

    @classmethod
    def _is_empty_marker(
        cls,
        text: str,
    ) -> bool:
        return (
            cls._normalized_marker_text(
                text
            )
            in _EMPTY_RESPONSE_MARKERS
        )

    @classmethod
    def _contains_empty_marker(
        cls,
        text: str,
    ) -> bool:
        cleaned_text = re.sub(
            r"```(?:\w+)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        ).replace(
            "```",
            "",
        )

        meaningful_lines = [
            line.strip()
            for line
            in cleaned_text.splitlines()
            if line.strip()
        ]

        return any(
            cls._is_empty_marker(
                line
            )
            for line in meaningful_lines
        )

    @classmethod
    def _normalize(
        cls,
        parsed: list,
        speaker: int,
        *,
        strict: bool = False,
    ) -> list[list[str | None]]:
        """Validate actions and force the subject to the real speaker."""

        _, speech_action_type = (
            _load_tom_schema()
        )

        if not isinstance(
            parsed,
            list,
        ):
            return []

        actions: list[list[str | None]] = []
        failures: list[dict[str, Any]] = []
        seen: set[
            tuple[str | None, str | None, str | None]
        ] = set()

        for item in parsed:
            raw_action = cls._read_raw_action(
                item
            )

            if raw_action is None:
                if strict:
                    failures.append(
                        {
                            "candidate": item,
                            "reason": (
                                "candidate must be a mapping with action/object "
                                "or a three-item sequence"
                            ),
                        }
                    )
                continue

            (
                raw_subject,
                action_name,
                object_player,
            ) = raw_action

            try:
                object_players = (
                    cls._expand_explicit_player_range(
                        object_player
                    )
                )
            except (
                TypeError,
                ValueError,
                KeyError,
            ) as exc:
                if strict:
                    failures.append(
                        {
                            "candidate": item,
                            "reason": (
                                f"{type(exc).__name__}: {exc}"
                            ),
                        }
                    )
                continue

            for atomic_object in object_players:
                try:
                    if strict:
                        strict_action = speech_action_type.from_values(
                            subject=raw_subject,
                            action=action_name,
                            object_=atomic_object,
                        )
                        if strict_action.subject != f"player{speaker}":
                            raise ValueError(
                                "speech action subject must equal current speaker"
                            )

                    action = (
                        speech_action_type
                        .from_values(
                            subject=speaker,
                            action=action_name,
                            object_=atomic_object,
                        )
                    )
                except (
                    TypeError,
                    ValueError,
                    KeyError,
                ) as exc:
                    if strict:
                        failures.append(
                            {
                                "candidate": item,
                                "reason": (
                                    f"{type(exc).__name__}: {exc}"
                                ),
                            }
                        )
                    continue

                normalized = action.to_list()
                key = tuple(normalized)

                if key in seen:
                    continue

                seen.add(key)
                actions.append(
                    normalized
                )

        if failures:
            raise SpeechActionValidationError(
                failures
            )

        return actions

    @staticmethod
    def _expand_explicit_player_range(
        object_player: Any,
    ) -> list[Any]:
        """Expand only the confirmed ascending ``playerN 至 playerM`` form."""

        if not isinstance(object_player, str):
            return [object_player]
        if object_player.strip().lower() in {"none", "null", "<none>"}:
            return [None]
        match = _EXPLICIT_PLAYER_RANGE_PATTERN.fullmatch(
            object_player.strip()
        )
        if match is None:
            return [object_player]
        start = int(match.group("start"))
        end = int(match.group("end"))
        if start > end:
            raise ValueError("player range must be ascending")
        return [
            f"player{player_id}"
            for player_id in range(start, end + 1)
        ]

    @staticmethod
    def _read_raw_action(
        item: Any,
    ) -> tuple[Any, Any, Any] | None:
        """Read a triplet while tolerating the legacy object response."""

        if isinstance(
            item,
            dict,
        ):
            if (
                "action" not in item
                or "object" not in item
            ):
                return None

            return (
                item.get(
                    "subject"
                ),
                item.get(
                    "action"
                ),
                item.get(
                    "object"
                ),
            )

        if (
            isinstance(
                item,
                Sequence,
            )
            and not isinstance(
                item,
                (str, bytes),
            )
            and len(item) == 3
        ):
            return (
                item[0],
                item[1],
                item[2],
            )

        return None


__all__ = [
    "SPEECH_PARSER_MAX_TOKENS",
    "SpeechParseAuditResult",
    "SpeechPerceiver",
]
