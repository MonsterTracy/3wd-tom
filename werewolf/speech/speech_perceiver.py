"""Parse public Werewolf speech into ONUW-style speech actions."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any


SPEECH_PARSER_MAX_TOKENS = 256

_PIPE_TRIPLET_PATTERN = re.compile(
    r"^\s*[\"'`]?\s*"
    r"(?P<subject>(?:player\s*)?[1-7])"
    r"\s*[|｜]\s*"
    r"(?P<action>[a-zA-Z_]+)"
    r"\s*[|｜]\s*"
    r"(?P<object>(?:player\s*)?[1-7])"
    r"\s*[\"'`]?\s*[,;，；。.！!]?\s*$",
    flags=re.IGNORECASE,
)

_BULLET_PREFIX_PATTERN = re.compile(
    r"^\s*(?:[-*•]+\s*|\d+\s*[.)、:：-]\s*)"
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
    "守卫": "point_as_guard",
}

# This intentionally covers only literal first-person role declarations.
# It does not infer roles from "好人", negations, conditions, quotations,
# other-player reports, or hidden game state.
_SELF_ROLE_CLAIM_PATTERN = re.compile(
    r"(?:^|[。！？；;\n])\s*"
    r"(?:"
    r"我是(?:[1-7]号(?:玩家)?)?\s*[，,\s]*"
    r"(?:身份(?:是|为|：|:)\s*)?"
    r"|我的身份(?:是|为|：|:)\s*"
    r"|我身份(?:是|为|：|:)\s*"
    r")"
    r"(?:一名|一个|普通的?)?\s*"
    r"(?P<role>狼人|村民|平民|预言家|女巫|守卫)"
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
    ) -> list[list[str]]:
        """Parse one speech turn without interrupting the game on failure."""

        del context

        if self.backend is None or not self.model_name:
            return []

        if type(speaker) is not int or not 1 <= speaker <= 7:
            return []

        if not isinstance(speech, str) or not speech.strip():
            return []

        protected_actions = (
            self._extract_explicit_self_claim_actions(
                speaker=speaker,
                speech=speech,
            )
        )

        try:
            return self._parse_configured(
                speaker=speaker,
                speech=speech,
                day=day,
                phase=phase,
                protected_actions=(
                    protected_actions
                ),
            )
        except Exception:
            # A literal public self-role declaration is still valid public
            # evidence even if the LLM parser fails. No hidden role or
            # game-state information is used here.
            return protected_actions

    def parse_strict(
        self,
        speaker: int,
        speech: str,
        day: int,
        phase: str,
    ) -> list[list[str]]:
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
    ) -> tuple[list[list[str]], str]:
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
    ) -> list[list[str]]:
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
    ) -> tuple[list[list[str]], str]:
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
        """Build the ONUW-style public-speech parsing prompt."""

        action_names, _ = _load_tom_schema()

        allowed_actions = "\n".join(
            f"- {action_name}"
            for action_name in action_names
        )

        return f"""你是狼人杀公开发言的结构化动作解析器。

你只解析下面这一段公开发言中由当前发言者明确表达的原子命题。你不判断发言真假，不依据游戏规则补全结论，不读取隐藏身份，也不推测玩家没有说出的私下想法。

当前发言者：player{speaker}
当前天数：Day {day}
当前阶段：{phase}
玩家范围：player1 到 player7

每个动作必须使用三元组：
subject | action | object

subject 必须是当前发言者 player{speaker}。
object 必须是 player1 到 player7。
允许的 action 只有：
{allowed_actions}

动作语义按四个语义模块组织。模块只用于理解语义，输出仍是上述平坦三元组。

ROLE_ESTIMATE（角色与阵营判断）：
1. point_as_werewolf：当前发言者明确判断某玩家是狼人。
2. point_as_villager：当前发言者明确判断某玩家是村民、平民或非狼好人阵营。
3. point_as_seer：当前发言者明确判断某玩家是预言家。
4. point_as_witch：当前发言者明确判断某玩家是女巫。
5. point_as_guard：当前发言者明确判断某玩家是守卫。

SOCIAL_STANCE（社会与论述立场）：
6. support：当前发言者明确认可、同意或支持 target 的发言、逻辑、主张或可信度。
7. oppose：当前发言者明确反对、不认可、不信任或批评 target 的发言、逻辑、主张或可信度。

CLAIMED_SKILL_REPORT（公开声称的技能信息）：
8. check_as_good：当前发言者声称自己获得 target 为非狼阵营的查验结果。
9. check_as_werewolf：当前发言者声称自己获得 target 为狼人的查验结果。
10. save：当前发言者声称自己已经对 target 使用解药或完成救人。
11. poison：当前发言者声称自己已经对 target 使用毒药。
12. guard：当前发言者声称自己已经守护 target。

ACTION_INTENT（当前行动意图）：
13. vote_intent：当前发言者在发言时，对当前待执行投票作出的无条件、明确、自身投票承诺。

抽取规则（A1：最具体、非冗余、显式原子命题编码）：
- 显式性：只抽取发言直接表达的命题，不得根据常识、游戏逻辑或其他 action 推导额外 action。
- 最具体性：如果一个具体 action 已经完整表达命题，不再输出它的宽泛版本。
- 非冗余性：同一语义不得因为“通常相关”而重复编码。不得自动扩展 vote_intent → oppose、point_as_werewolf → oppose、point_as_villager → support、check_as_werewolf → point_as_werewolf 或 oppose、check_as_good → point_as_villager 或 support、poison → oppose、save → support 或 guard → support。
- 独立命题共存：同一发言明确表达多个不同命题时，保留全部动作，并按原文顺序输出。
- 第一人称明确自报具体身份属于受保护动作，必须优先且稳定抽取。例如“我是6号玩家，身份是村民”必须输出当前发言者对自己的 point_as_villager。
- 发言前半段已经明确自报具体身份时，即使后文表示“暂时没有怀疑对象”“继续观察”“信息不足”或保持中立，也不能省略已经出现的自报身份动作。
- 修饰明确社会或论述立场的“暂时”“比较”等措辞不转换成数值强度。但“倾向投”“可能投”等不确定投票表达不是 vote_intent，投票偏好本身也不等于 support 或 oppose。
- 角色或阵营判断使用 point_as_*；普通判断不能替代技能查验结果，也不自动追加 support 或 oppose。
- 只有当前发言者明确声称自己的查验结果时，才使用 check_as_good 或 check_as_werewolf。普通认好、普通怀疑、转述他人查验、推测、否定或条件查验不得使用这两个动作。
- 技能 action 只表示公开发言中的主观声明，不表示真实游戏事实，也不重复输出其宽泛结论。
- save、poison 和 guard 只表示当前发言者声称自己已经完成的技能行为。建议、猜测、转述、未来计划、条件或否定表达不得使用。
- vote_intent 只表示当前发言者对当前这一轮待执行投票的无条件自身承诺，不表示怀疑或反对，也不自动产生 oppose。允许“今天我投3号”“我的票挂3号”“这一轮我会投3号”。条件承诺、可能性表达、未来其他轮次的计划、请求他人投票、转述他人投票意图、已完成的投票或实际投票系统事件不得使用，例如“如果3号不解释，我就投他”“我可能投3号”“明天再考虑投3号”“大家投3号”“2号说他会投3号”“我已经投了3号”。
- 单纯转述、复述或引用别人的查验、身份声明或立场，不视为当前发言者自己的立场。
- 如果当前发言者明确表示认同某人的发言，可以输出对该玩家的 support；只有当前发言者本人也明确接受某个具体身份结论时，才输出相应 point_as_*。
- “我是好人”不是具体身份声明，不输出 point_as_villager。
- “我不是狼人”是身份否定，当前动作集合不能准确表达，不输出。
- 不补充发言中没有表达的动作。
- 不输出真实角色，不输出 guesses，不输出置信度、分数、阵营、阶段或其他字段。
- 删除完全重复的动作，但不要删除同一 subject/object 上 action 不同的动作。

示例：
输入：我是预言家，3号是狼人。
输出：
player{speaker} | point_as_seer | player{speaker}
player{speaker} | point_as_werewolf | player3

输入：我是6号玩家，身份是村民。目前信息不足，我暂时没有明确怀疑对象。
输出：
player{speaker} | point_as_villager | player{speaker}

输入：我觉得3号是狼。
输出：
player{speaker} | point_as_werewolf | player3

输入：我不同意3号的逻辑。
输出：
player{speaker} | oppose | player3

输入：7号跳预言家查杀1号，我暂时站7号。
输出：
player{speaker} | support | player7

输入：我是预言家，昨晚查验2号，结果是好人。
输出：
player{speaker} | point_as_seer | player{speaker}
player{speaker} | check_as_good | player2

输入：我是预言家，昨晚验了3号，是查杀。
输出：
player{speaker} | point_as_seer | player{speaker}
player{speaker} | check_as_werewolf | player3

输入：我验3号查杀，今天投3号。
输出：
player{speaker} | check_as_werewolf | player3
player{speaker} | vote_intent | player3

输入：我验3号金水，也同意他的逻辑。
输出：
player{speaker} | check_as_good | player3
player{speaker} | support | player3

输入：第一晚我救了2号，昨晚我毒了3号。
输出：
player{speaker} | save | player2
player{speaker} | poison | player3

输入：昨晚我守了4号。
输出：
player{speaker} | guard | player4

输入：今天我投3号。
输出：
player{speaker} | vote_intent | player3

输入：这一轮我倾向投4号。
输出：
NONE

输入：我不信3号，今天我投3号。
输出：
player{speaker} | oppose | player3
player{speaker} | vote_intent | player3

输入：3号是好人，但为了统一票型今天投3号。
输出：
player{speaker} | point_as_villager | player3
player{speaker} | vote_intent | player3

输入：1号大概率是真的预言家，我建议先相信1号。
输出：
player{speaker} | point_as_seer | player1
player{speaker} | support | player1

输入：1号和7号都在跳预言家，我还需要继续听。
输出：
NONE

输出协议：
- 每个动作单独一行，格式必须严格为：subject | action | object
- 整个回答的每一个非空行都必须符合上述协议；不要混入无法解析的行。
- 最多输出7个动作，不要重复动作，不要输出超长文本。
- 没有可抽取动作时，只输出：NONE
- 不输出 JSON，不输出解释，不输出 Markdown 代码块。

待解析的玩家发言：
player{speaker}: {speech}"""

    @classmethod
    def _extract_explicit_self_claim_actions(
        cls,
        *,
        speaker: int,
        speech: str,
    ) -> list[list[str]]:
        """Protect literal first-person public role declarations.

        This is deliberately narrow. It only preserves an identity that the
        current speaker explicitly states about themself in the public text.
        It never reads the player's real role, observation, or private state.
        """

        _, speech_action_type = (
            _load_tom_schema()
        )

        actions: list[list[str]] = []
        seen: set[
            tuple[str, str, str]
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
            Sequence[str]
        ],
    ) -> list[list[str]]:
        """Merge action groups while retaining distinct action triplets."""

        merged: list[list[str]] = []
        seen: set[
            tuple[str, str, str]
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

                normalized = [
                    str(value)
                    for value in action
                ]
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
    ) -> list[list[str]]:
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

        actions: list[list[str]] = []
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
    ) -> list[list[str]]:
        """Validate actions and force the subject to the real speaker."""

        _, speech_action_type = (
            _load_tom_schema()
        )

        if not isinstance(
            parsed,
            list,
        ):
            return []

        actions: list[list[str]] = []
        failures: list[dict[str, Any]] = []
        seen: set[
            tuple[str, str, str]
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
                if strict:
                    speech_action_type.from_values(
                        subject=raw_subject,
                        action=action_name,
                        object_=object_player,
                    )

                action = (
                    speech_action_type
                    .from_values(
                        subject=speaker,
                        action=action_name,
                        object_=(
                            object_player
                        ),
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
    "SpeechPerceiver",
]
