import json

from werewolf.models.twd_tom.schema import ACTION_NAMES


class Const(object):
    class ConstError(TypeError):
        pass

    class ConstCaseError(ConstError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("Can't change const.%s" % name)
        self.__dict__[name] = value

CON = Const()

LEGACY_GAMEPLAY_PROMPT_PROFILE = "legacy"
STRICT_CLASSIC7_GAMEPLAY_PROMPT_PROFILE = (
    "strict_classic7"
)


def _render_authoritative_public_state(public_state):
    """Render the sole canonical public-state block for strict speech."""

    required_fields = {
        "day",
        "day_or_night",
        "phase",
        "last_night_result",
        "prior_exiles",
        "alive_players",
        "suggestible_exile_targets",
    }
    if not isinstance(public_state, dict) or not required_fields <= public_state.keys():
        raise TypeError("incomplete authoritative public state")
    last_night_result = public_state["last_night_result"]
    prior_exiles = public_state["prior_exiles"]
    alive_players = public_state["alive_players"]
    suggestible_targets = public_state["suggestible_exile_targets"]
    if (
        (
            last_night_result is not None
            and (
                not isinstance(last_night_result, dict)
                or not isinstance(last_night_result.get("dead_players"), list)
            )
        )
        or not isinstance(prior_exiles, list)
        or not isinstance(alive_players, list)
        or not isinstance(suggestible_targets, list)
    ):
        raise TypeError("invalid authoritative public state")

    if last_night_result is None:
        last_night_text = "尚无已公开的昨夜结果"
    else:
        dead_players = last_night_result["dead_players"]
        last_night_text = (
            ", ".join(f"player{player_id}" for player_id in dead_players)
            + " 昨夜死亡"
            if dead_players
            else "昨夜无人死亡"
        )
    exiles_text = "\n".join(
        f"- player{item['player_id']} 已于第{item['day']}天放逐"
        for item in prior_exiles
    ) or "- 此前无人被放逐"
    phase_names = {
        "speech": "公开发言",
        "speech_pk": "平票公开发言",
        "vote": "投票",
        "vote_pk": "平票投票",
        "skill_wolf": "狼人行动",
        "skill_seer": "预言家行动",
        "skill_witch": "女巫行动",
        "skill_guard": "守卫行动",
        "end_game": "游戏结束",
    }
    phase_text = "第{day}天{period}{phase}".format(
        day=public_state["day"],
        period="白天" if public_state["day_or_night"] == "day" else "夜晚",
        phase=phase_names.get(public_state["phase"], str(public_state["phase"])),
    )
    alive_text = ", ".join(f"player{player_id}" for player_id in alive_players) or "(无)"
    targets_text = ", ".join(
        f"player{player_id}" for player_id in suggestible_targets
    ) or "(无)"
    return f"""【权威公共状态】
【当前阶段】{phase_text}
【昨夜公开结果】{last_night_text}
【此前放逐】
{exiles_text}
【当前存活】{alive_text}
【当前可公开建议放逐】{targets_text}"""


def build_strict_classic7_speech_plan_prompt(
    observation,
):
    """Build the private planner prompt from one legally filtered view."""

    if not isinstance(
        observation,
        dict,
    ):
        raise TypeError(
            "strict speech observation "
            "must be a dictionary"
        )
    actor = observation.get(
        "current_act_idx"
    )
    if (
        isinstance(actor, bool)
        or not isinstance(actor, int)
        or not 1 <= actor <= 7
    ):
        raise ValueError(
            "strict speech requires "
            "current_act_idx in [1, 7]"
        )
    identity = observation.get(
        "identity"
    )
    if identity not in {
        "Werewolf",
        "Villager",
        "Seer",
        "Witch",
        "Guard",
    }:
        raise ValueError(
            "strict speech requires "
            "a supported classic7 identity"
        )
    game_log = observation.get(
        "game_log"
    )
    if not isinstance(
        game_log,
        list,
    ):
        raise TypeError(
            "strict speech requires "
            "a game_log list"
        )
    public_state = observation.get(
        "authoritative_public_state"
    )
    if not isinstance(public_state, dict):
        raise TypeError(
            "strict speech requires authoritative_public_state"
        )
    authoritative_state_text = _render_authoritative_public_state(public_state)

    if identity == "Werewolf":
        wolf_team = []
        night_decisions = []
        for log in game_log:
            if getattr(log, "event", None) == "werewolf_team_info":
                wolf_team.extend(
                    getattr(log, "content", {}).get("wolf_team", [])
                )
            elif getattr(log, "event", None) == "kill_decision":
                target = getattr(log, "target", None)
                if isinstance(target, int) and 1 <= target <= 7:
                    night_decisions.append(target)
        team_text = ", ".join(
            f"player{player_id}" for player_id in sorted(set(wolf_team))
        ) or "(无合法可见狼队信息)"
        decision_text = ", ".join(
            f"player{player_id}" for player_id in night_decisions
        ) or "(尚无已完成狼刀决策)"
        role_rules = f"""你的真实私有身份是狼人。
- 真实狼队信息（仅用于内部策略）：{team_text}。
- 已完成的真实夜间刀人决策（仅用于内部策略）：{decision_text}。
- 绝不能公开说自己是狼人，也不能直接公开狼人队友身份。
- 绝不能直接公开狼队夜间讨论、狼刀真实决策或夜间技能信息。
- 不得使用“因为系统告诉我”“我的真实队友是”等内部状态泄露式表达。
- 可以按策略假跳其他身份、隐藏信息或作出虚假公开声明，但不要逐字复制系统内部描述。"""
    elif identity == "Villager":
        role_rules = """你的真实私有身份是普通村民。
- 你没有查验、解药、毒药或守护能力。
- 这是内部真实状态；公开发言时可以按策略假跳身份或作出虚假技能声明。"""
    elif identity == "Seer":
        checks = []
        for log in game_log:
            if getattr(
                log,
                "event",
                None,
            ) != "skill_seer":
                continue
            result = getattr(
                log,
                "content",
                {},
            ).get(
                "cheked_identity"
            )
            target = getattr(
                log,
                "target",
                None,
            )
            if (
                result in {
                    "good",
                    "bad",
                }
                and isinstance(
                    target,
                    int,
                )
                and 1 <= target <= 7
            ):
                checks.append(
                    "player"
                    f"{target}="
                    + (
                        "狼人"
                        if result == "bad"
                        else "好人"
                    )
                )
        check_text = (
            ", ".join(checks)
            if checks
            else "(尚无已完成查验)"
        )
        role_rules = f"""你的真实私有身份是预言家。
- 已真实发生的查验结果：{check_text}
- 这些是内部真实状态；公开时可以披露、隐藏、歪曲或虚构身份和技能声明。
- 你没有真实的解药、毒药或守护能力，也不知道狼人队友。"""
    elif identity == "Witch":
        heal_used = False
        poison_used = False
        night_kill_targets = []
        for log in game_log:
            event = getattr(
                log,
                "event",
                None,
            )
            content = getattr(
                log,
                "content",
                {},
            )
            target = getattr(
                log,
                "target",
                None,
            )
            if event == "skill_witch":
                heal_used = (
                    heal_used
                    or "heal" in content
                )
                poison_used = (
                    poison_used
                    or "poison" in content
                )
            elif (
                event == "kill_decision"
                and isinstance(
                    target,
                    int,
                )
                and 1 <= target <= 7
            ):
                night_kill_targets.append(
                    f"player{target}"
                )
        kill_text = (
            ", ".join(
                night_kill_targets
            )
            if night_kill_targets
            else "(无当前合法可见目标)"
        )
        role_rules = f"""你的合法私有身份是女巫。
- 解药真实状态：{"已使用" if heal_used else "未使用"}。
- 毒药真实状态：{"已使用" if poison_used else "未使用"}。
- 合法可见的历史夜间击杀目标：{kill_text}。
- 这些是内部真实状态；公开时可以披露、隐藏、歪曲或虚构身份和技能声明。
- 女巫没有真实的查验或守护能力，也不知道狼人队友。"""
    else:
        guarded = [
            getattr(
                log,
                "target",
                None,
            )
            for log in game_log
            if getattr(
                log,
                "event",
                None,
            )
            == "skill_guard"
        ]
        guarded_text = ", ".join(
            f"player{target}"
            for target in guarded
            if isinstance(target, int)
            and 1 <= target <= 7
        ) or "(尚无已完成守护)"
        role_rules = f"""你的真实私有身份是守卫。
- 已真实发生的守护目标：{guarded_text}
- 这些是内部真实状态；公开时可以披露、隐藏、歪曲或虚构身份和技能声明。
- 你没有真实的查验、解药或毒药，也不知道狼人队友。"""

    public_claims = []
    for log in game_log:
        if getattr(log, "event", None) not in {"speech", "speech_pk"}:
            continue
        source = getattr(log, "source", None)
        if source == actor or not isinstance(source, int) or not 1 <= source <= 7:
            continue
        speech = getattr(log, "content", {}).get("speech_content")
        if isinstance(speech, str) and speech.strip():
            public_claims.append(f"- player{source}：{speech}")
    claims_text = "\n".join(public_claims) or "- (尚无其他玩家公开主张)"

    action_names = ", ".join(ACTION_NAMES)
    return f"""你是 strict gameplay 的 Private Planner。
当前 speaker：player{actor}

{authoritative_state_text}

【你合法知道的私有信息】
私有信息只用于制定策略。不要逐字复制系统描述、字段名、内部状态或控制元数据。
你可以按策略撒谎、假跳、隐藏信息或虚构公开身份与技能声明。

{role_rules}

【其他玩家此前的公开主张】
这些只是玩家发言，可能是真话、谎言、误解或策略性表达；若与权威公共状态冲突，以权威公共状态为准。
{claims_text}

【计划合同】
- 只规划当前玩家准备公开声称或表达什么；计划不是事实标签。
- 可以假跳、撒谎、隐藏或歪曲，但不要输出最终自然语言发言或解释。
- 只能使用这些正式 action：{action_names}。
- 只输出 public_actions；禁止 reasoning、strategy、notes、summary 或其他自由文本字段。
- vote_intent 只能指向“当前可公开建议放逐”中的玩家，可以不输出 vote_intent。
- 不要把历史旧投票目标直接复制成当前目标；已死亡或放逐玩家不得成为 vote_intent。
- 其他历史技能声明可以指向过去玩家，因为它们只是准备公开表达的声称。
- 只保留少量关键 action，避免复述完整历史；空 public_actions 合法。
- 只输出符合请求 JSON Schema 的对象。"""


def build_strict_classic7_speech_render_prompt(
    *,
    authoritative_public_state,
    actor,
    public_actions,
):
    """Build a public renderer prompt with no private or raw-history input."""

    if isinstance(actor, bool) or not isinstance(actor, int) or not 1 <= actor <= 7:
        raise ValueError("strict speech renderer requires actor in [1, 7]")
    if not isinstance(public_actions, list):
        raise TypeError("strict speech renderer requires validated public_actions")
    state_text = _render_authoritative_public_state(authoritative_public_state)
    plan_text = json.dumps(
        {"public_actions": public_actions},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    empty_plan_rule = (
        "计划为空：生成简短、非目标化的观望发言；不得点名其他玩家，"
        "不得生成身份、技能或投票声明。"
        if not public_actions
        else "计划非空：每个 target 都必须以 playerN、玩家N 或 N号明确出现。"
    )
    vote_targets = [
        item["target"]
        for item in public_actions
        if item["action"] == "vote_intent"
    ]
    vote_rule = (
        "只有计划中的 vote_intent target "
        + ", ".join(f"player{target}" for target in vote_targets)
        + " 才可被表达为投票给、票出、放逐、驱逐、归票或今天出。"
        if vote_targets
        else "计划没有 vote_intent：不得表达投票给、票出、放逐、驱逐、归票或今天出任何玩家。"
    )
    return f"""{state_text}

【当前发言者】player{actor}
【已验证公开计划】{plan_text}
【输出合同】
- 只输出 2–4 句中文公开发言正文，建议不超过 200 个汉字。
- 只能表达已验证计划中的声明、立场、技能声称和投票倾向。
- point_as_villager(X) 只表达将 X 公开判断为村民，不自动产生 support(X)。
- oppose(X) 只表达质疑、反对、不认可或认为其发言可疑；oppose(X) 不等于 vote_intent(X)。
- {vote_rule}
- 不得新增计划外玩家、角色判断、技能结果或投票目标。
- 不得复述完整历史，不输出 JSON、Markdown、标题或分析。
- {empty_plan_rule}"""

CON.game_description = """你现在正在玩一局7人狼人杀游戏。

在这款游戏中，玩家分为两个阵营：狼人阵营和村民阵营。

本局共有1到7号共7名玩家：
- 狼人阵营：2名狼人。
- 村民阵营：5名好人，包括1名预言家、1名额外神职和3名普通村民。

狼人杀游戏中不同角色的玩家有不同目标：
- 村民阵营的目标是识别并投票放逐所有狼人。
- 狼人阵营的目标是隐藏身份、误导好人，并在夜晚猎杀村民阵营玩家。

基本规则：
- 身份：玩家身份秘密分配。狼人彼此知道同伴身份；村民阵营玩家只知道自己的身份。
- 夜晚：狼人秘密选择一名玩家猎杀；预言家可以查验一名玩家是否为狼人；额外神职根据本局配置可能是女巫或守卫。
- 白天：所有存活玩家依次发言，并投票放逐一名最可疑的玩家。
- 平票：若最高票平票，则进入PK发言和再次投票。
- 获胜条件：所有狼人出局，则村民阵营获胜；所有普通村民出局，或所有神职玩家出局，则狼人阵营获胜。
- 玩家编号：本局只有1到7号玩家。
- 角色限制：本项目的7人局只支持“预言家+女巫”或“预言家+守卫”两种配置。
"""

CON.game_description_7p = """你现在正在玩一局7人狼人杀游戏。

在这款游戏中，玩家分为两个阵营：狼人阵营和村民阵营。

本局共有1到7号共7名玩家：
- 狼人阵营：2名狼人。
- 村民阵营：5名好人，包括1名预言家、1名额外神职和3名普通村民。

本局固定包含：
- 2名狼人
- 1名预言家
- 3名普通村民
- 1名额外神职，额外神职由当前配置决定，只能是女巫或守卫之一

基本规则：
- 身份：玩家身份秘密分配。狼人彼此知道同伴身份；村民阵营玩家只知道自己的身份。
- 夜晚：狼人秘密选择一名玩家猎杀；预言家可以查验一名玩家是否为狼人；女巫或守卫根据自身能力行动。
- 白天：所有存活玩家依次发言，并投票放逐一名最可疑的玩家。
- 平票：若最高票平票，则进入PK发言和再次投票。
- 获胜条件：所有狼人出局，则村民阵营获胜；所有普通村民出局，或所有神职玩家出局，则狼人阵营获胜。
- 玩家编号：本局只有1到7号玩家。
- 角色限制：本局不包含猎人。

村民阵营中的特殊角色包括：
- 1位预言家：
    - 目标：帮助村民阵营识别狼人。
    - 能力：每晚可以查验一名玩家，得知该玩家是否为狼人。
{god_description}
村民阵营中的另外3名普通村民没有夜晚技能。"""

CON.guard_description = """- 1位守卫：
    - 阵营：村民阵营。
    - 目标：通过合理守护关键好人，减少村民阵营夜晚损失。
    - 能力：每晚可以选择一名玩家进行守护，被守护玩家可以免受狼人猎杀。
    - 限制：守卫可以守护自己，也可以选择空守；守卫不能连续两个夜晚守护同一名玩家。
"""

CON.witch_description = """- 1位女巫：
    - 阵营：村民阵营。
    - 目标：通过合理使用解药和毒药帮助村民阵营获胜。
    - 能力：女巫拥有一瓶解药和一瓶毒药。
    - 解药：可以救下一名夜晚被狼人猎杀的玩家，使用后不可再次使用。
    - 毒药：可以毒杀一名玩家，使用后不可再次使用。
    - 限制：女巫同一晚不能同时使用解药和毒药。
"""


CON.identity_chinese = {
    "Seer": "预言家",
    "Witch": "女巫",
    "Guard": "守卫",
    "Villager": "村民",
    "Werewolf": "狼人"
}


CON.identity_abilities = {
    "Seer": "你每晚可以查看一名玩家是否为狼人，你的目标是利用这些信息帮助其他人找出并淘汰所有狼人。",
    "Witch": "你拥有一瓶救人的药水和一瓶毒杀的药水，你的目标是策略性利用这些能力来保护村民或淘汰狼人。",
    "Guard": "你每晚可以保护一名玩家不被狼人杀害，你的目标是明智地使用这个能力来减少村民的伤亡。",
    "Villager": "你没有特殊能力，但你的目标是观察、讨论并投票揭示狼人的身份，并努力生存下来。",
    "Werewolf": (
        "你需要和其他狼人合作，每晚选择一个村民猎杀，你的目标是隐藏你的身份，并误导其他玩家，直至狼人获得游戏的胜利。"
        "你可以使用狼人私有信息做内部策略判断，但公开发言必须伪装成普通好人公开视角。"
        "公开发言禁止暴露狼人队友身份、禁止暴露夜晚刀人目标、禁止暴露狼队讨论内容，"
        "也禁止声称谁被刀但没死。禁止声称某人是被女巫救下的人。"
        "怀疑或支持他人的公开理由只能来自白天发言、公开死亡结果、公开投票结果、公开身份声明和公开逻辑矛盾。"
        "不能把私有信息包装成确定公共事实；如果利用夜晚信息，只能转化成公开视角下的模糊怀疑。"
    )
}

CON.werewolf_skill_prompt_v3 = """请综合角色设定、客观信息和主观信息，选择今晚要杀害的玩家。
本局只有1到7号玩家，不允许输出8号、9号或更高编号。
只输出严格JSON，不要输出Markdown，不要输出额外解释。
输出格式：
{
  "原因": "...",
  "杀害": "玩家编号或否"
}"""

CON.seer_skill_prompt_v3 = """请综合角色设定、客观信息和主观信息，选择今晚要查验的玩家。
本局只有1到7号玩家，不允许输出8号、9号或更高编号。
只输出严格JSON，不要输出Markdown，不要输出额外解释。
输出格式：
{
  "原因": "...",
  "查验": "玩家编号"
}"""

CON.guard_skill_prompt_v3 = """请综合角色设定、客观信息和主观信息，选择今晚要守卫的玩家；如果空守则输出“否”。
本局只有1到7号玩家，不允许输出8号、9号或更高编号。
只输出严格JSON，不要输出Markdown，不要输出额外解释。
输出格式：
{
  "原因": "...",
  "守卫": "玩家编号或否"
}"""

CON.witch_skill_prompt_v3 = """请综合角色设定、客观信息和主观信息，决定是否使用解药和毒药。
夜晚信息：{wolf_killed_info}
本局只有1到7号玩家，不允许输出8号、9号或更高编号。
只输出严格JSON，不要输出Markdown，不要输出额外解释。
输出格式：
{{
  "原因": "...",
  "解药": "是或否",
  "毒药": "玩家编号或否"
}}"""

CON.player_identity_info = """
你是{player_idx}号玩家。
你的身份是：{identity}。
{identity_ability}"""


CON.skill_prompt = """
** 游戏说明
{game_description}
{player_identity_info}

** 游戏日志
{logs}

请根据游戏日志，从下列动作列表中选择一个你要执行的动作。
{valid_actions}

请严格按照动作列表中的内容输出，不要改动或者删减内容，也不要选择列表以外的动作。

** 输出
"""

CON.speech_prompt = """
** 游戏说明
{game_description}
{player_identity_info}

** 游戏日志
{logs}

请根据游戏日志，直接输出你本轮的发言。

** 输出 
"""

CON.vote_prompt = """
** 游戏说明
{game_description}
{player_identity_info}

** 游戏日志
* public logs
{logs}

正常白天投票阶段，你必须从当前存活玩家中选择一名玩家投票。
不要投给已经出局的玩家；不要投给自己，除非动作列表明确允许且没有其他合法候选人。
除非动作列表中没有任何合法玩家编号，否则不要选择“否”、弃票或不投票。

投票一致性约束：
- 你的投票必须基于当前可见 observation 中的公开信息。
- 你的投票必须尽量继承你自己白天发言中的怀疑、支持、站边和投票意向。
- 如果你在发言中明确怀疑过某人，优先从这些对象中选择投票目标。
- 如果你明确表示要跟随某个玩家归票，应结合该玩家的发言和当前局势选择目标，而不是投给该玩家本人。
- 不要把“跟随 X 归票”理解成“投 X”。
- 如果你要投给一个自己之前没有怀疑过的人，必须在内部 reasoning 中说明为什么改变目标，但不要改变规定的输出格式。
- 不要无依据随机投票。
- 不要投给自己。
- 不要投已死亡玩家。

请根据游戏日志，从下列动作中选择一个你要执行的动作。
{valid_actions}

优先输出严格JSON，例如：
{{
  "投票玩家": "3"
}}
也可以严格按照动作列表中的内容输出。不要选择列表以外的动作。

** 输出 
"""


CON.skill_prompt_v3 = """在本场7人狼人杀游戏中，你目前已知以下信息：

1. 角色设定：
{player_identity_info}

2. 客观信息：
{objective_info}

3. 主观信息：
{subjective_info}

{your_role}{instruction_prompt}
"""

CON.speech_prompt_v3 = """在本场7人狼人杀游戏中，你目前已知以下信息：

1. 角色设定：
{player_identity_info}

2. 客观信息：
{objective_info}

3. 主观信息：
{subjective_info}

{your_role}请综合角色设定、客观信息和主观信息分析局势并组织本轮发言。
本局只有1到7号玩家，不允许输出8号、9号或更高编号。
本局不包含猎人；不要声称自己或他人是猎人。
只输出严格JSON，不要输出Markdown，不要输出额外解释。
输出格式：
{{
  "想要展示的身份": "你希望展示给其他玩家的身份",
  "身份标签": {{
    "1号玩家": "你的身份判断",
    "2号玩家": "你的身份判断"
  }},
  "归票": "玩家编号或弃票",
  "发言": "你的最终发言文本"
}}
"""

CON.vote_prompt_v3 = """在本场7人狼人杀游戏中，你目前已知以下信息：

1. 角色设定：
{player_identity_info}

2. 客观信息：
{objective_info}

3. 主观信息：
{subjective_info}

{your_role}请综合角色设定、客观信息和主观信息，形成笔记并决定本轮投票。
本局只有1到7号玩家，不允许输出8号、9号或更高编号。
本局不包含猎人。
正常白天投票阶段，你必须从当前存活玩家中选择一名玩家投票。
不要投给已经出局的玩家；不要投给自己，除非valid_actions明确允许且没有其他合法候选人。
除非没有任何合法候选人，否则不要弃票。
投票一致性约束：
- 你的投票必须基于当前可见 observation 中的公开信息。
- 你的投票必须尽量继承你自己白天发言中的怀疑、支持、站边和投票意向。
- 如果你在发言中明确怀疑过某人，优先从这些对象中选择投票目标。
- 如果你明确表示要跟随某个玩家归票，应结合该玩家的发言和当前局势选择目标，而不是投给该玩家本人。
- 不要把“跟随 X 归票”理解成“投 X”。
- 如果你要投给一个自己之前没有怀疑过的人，必须在“投票原因”中说明为什么改变目标。
- 不要无依据随机投票。
- 不要投给自己。
- 不要投已死亡玩家。
“投票玩家”字段必须填写玩家编号，例如 "3"。
不要输出“否”“弃票”“不投票”，除非没有合法候选人。
只输出严格JSON，不要输出Markdown，不要输出额外解释。
输出格式：
{{
  "笔记": "你对本轮夜晚信息、发言、站边和票型的总结",
  "投票原因": "你的投票理由",
  "投票玩家": "3"
}}
"""
