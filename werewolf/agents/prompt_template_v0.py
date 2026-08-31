import json
import re
from dataclasses import dataclass


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
PUBLIC_SPEECH_REALIZATION_PROMPT_VERSION = (
    "classic7_public_speech_realization_prompt_v1"
)
STRICT_CLASSIC7_ROLE_COUNTS = {
    "Werewolf": 2,
    "Seer": 1,
    "Witch": 1,
    "Villager": 3,
}
STRICT_BELIEF_CONCRETE_ROLES = tuple(STRICT_CLASSIC7_ROLE_COUNTS)

DISCUSSION_ACTIONS = (
    "point_as_werewolf",
    "point_as_non_werewolf",
    "point_as_villager",
    "point_as_seer",
    "point_as_witch",
    "support",
    "oppose",
    "check_as_non_werewolf",
    "check_as_werewolf",
    "save",
    "poison",
    "vote_intent",
    "abstain_intent",
    "no_commitment",
)
_DISCUSSION_ACTION_SEMANTICS = {
    "point_as_werewolf": "publicly claim playerX is Werewolf",
    "point_as_non_werewolf": (
        "publicly claim playerX is not a Werewolf / belongs to the good camp; "
        "not a specific Villager-role claim"
    ),
    "point_as_villager": (
        "publicly claim playerX is specifically an ordinary Villager; "
        "not generic good / non-wolf"
    ),
    "point_as_seer": "publicly claim playerX is Seer",
    "point_as_witch": "publicly claim playerX is Witch",
    "support": "publicly support / defend playerX",
    "oppose": "publicly oppose / question playerX",
    "check_as_non_werewolf": (
        "speaker publicly claims a Seer-style check on playerX returned "
        '"not Werewolf"; not necessarily Villager'
    ),
    "check_as_werewolf": (
        "speaker publicly claims a Seer-style check on playerX returned "
        "Werewolf"
    ),
    "save": "speaker publicly claims using the Witch antidote to save playerX",
    "poison": "speaker publicly claims using Witch poison on playerX",
    "vote_intent": "publicly push the current exile vote toward playerX",
    "abstain_intent": "publicly indicate current abstention intent",
    "no_commitment": (
        "make no explicit role/check/skill/vote commitment this turn"
    ),
}
_PUBLIC_SPEECH_ACTION_LABELS = {
    "point_as_werewolf": "指认为狼人",
    "point_as_non_werewolf": "指认为非狼人",
    "point_as_villager": "指认为普通村民",
    "point_as_seer": "指认为预言家",
    "point_as_witch": "指认为女巫",
    "support": "公开支持",
    "oppose": "公开质疑",
    "check_as_non_werewolf": "宣称查验为非狼人",
    "check_as_werewolf": "宣称查验为狼人",
    "save": "宣称使用解药",
    "poison": "宣称使用毒药",
    "vote_intent": "表达放逐投票意图",
    "abstain_intent": "表达弃票意图",
    "no_commitment": "不作明确表态",
}
_PUBLIC_SPEECH_ACTION_SEMANTICS = {
    "point_as_werewolf": "公开声称 playerX 是狼人",
    "point_as_non_werewolf": "公开声称 playerX 不是狼人或属于好人阵营，但不具体声称是普通村民",
    "point_as_villager": "公开声称 playerX 的具体身份是普通村民，而不只是泛称好人",
    "point_as_seer": "公开声称 playerX 是预言家",
    "point_as_witch": "公开声称 playerX 是女巫",
    "support": "公开支持或维护 playerX",
    "oppose": "公开反对或质疑 playerX",
    "check_as_non_werewolf": "公开声称以预言家方式查验 playerX，结果不是狼人，但不等同于普通村民",
    "check_as_werewolf": "公开声称以预言家方式查验 playerX，结果是狼人",
    "save": "公开声称使用女巫解药救了 playerX",
    "poison": "公开声称对 playerX 使用了女巫毒药",
    "vote_intent": "公开推动本轮将放逐票投向 playerX",
    "abstain_intent": "公开表达本轮弃票意图",
    "no_commitment": "本轮不作明确的身份、查验、技能或投票表态",
}
_ALL_PLAYER_TARGET_ACTIONS = DISCUSSION_ACTIONS[:7]
_NON_SELF_TARGET_ACTIONS = DISCUSSION_ACTIONS[7:11]
_PUBLIC_CONTENT_ACTIONS = DISCUSSION_ACTIONS[:11]
_PUBLIC_VOTE_STANCE_ACTIONS = {"vote_intent", "abstain_intent"}
NO_STANCE = "NO_STANCE"


@dataclass(frozen=True)
class DiscussionAct:
    action: str
    target: int | None


@dataclass(frozen=True)
class PublicClaim:
    claim_id: str
    time: str
    event: str
    speaker: int
    raw_text: str


def build_public_claim_catalog(observation):
    """Freeze visible raw public claims in observation order."""

    if not isinstance(observation, dict):
        raise TypeError("discussion observation must be a dictionary")
    game_log = observation.get("game_log")
    if not isinstance(game_log, list):
        raise TypeError("discussion observation requires a game_log list")
    claims = []
    for log in game_log:
        event = getattr(log, "event", None)
        if event not in {"speech", "speech_pk"}:
            continue
        speaker = getattr(log, "source", None)
        time_text = getattr(log, "time", None)
        raw_text = getattr(log, "content", {}).get("speech_content")
        if (
            isinstance(speaker, bool)
            or not isinstance(speaker, int)
            or not 1 <= speaker <= 7
            or not isinstance(time_text, str)
            or not isinstance(raw_text, str)
            or not raw_text.strip()
        ):
            continue
        claims.append(
            PublicClaim(
                claim_id=f"claim_{len(claims):03d}",
                time=time_text,
                event=event,
                speaker=speaker,
                raw_text=raw_text,
            )
        )
    return tuple(claims)


def render_public_claim(claim):
    if not isinstance(claim, PublicClaim):
        raise TypeError("public claim must be a PublicClaim")
    return (
        f"[{claim.claim_id}] [{claim.time} / {claim.event}] "
        f"player{claim.speaker}：{claim.raw_text}"
    )


def derive_discussion_vote_targets(observation):
    """Derive authoritative public vote-intent targets for one speech phase."""

    if not isinstance(observation, dict):
        raise TypeError("discussion observation must be a dictionary")
    actor = observation.get("current_act_idx")
    phase = observation.get("phase")
    public_state = observation.get("authoritative_public_state")
    if (
        isinstance(actor, bool)
        or not isinstance(actor, int)
        or not 1 <= actor <= 7
    ):
        raise ValueError("discussion requires current_act_idx in [1, 7]")
    if not isinstance(phase, str) or "speech" not in phase:
        raise ValueError("discussion vote targets require a speech phase")
    if not isinstance(public_state, dict):
        raise TypeError("discussion requires authoritative_public_state")
    alive_players = public_state.get("alive_players")
    if not isinstance(alive_players, list):
        raise TypeError("discussion requires authoritative alive players")
    alive_set = set(alive_players)

    if "speech_pk" in phase:
        game_log = observation.get("game_log")
        if not isinstance(game_log, list):
            raise TypeError("PK discussion requires a game_log list")
        targets = None
        for log in reversed(game_log):
            if getattr(log, "event", None) != "end_vote":
                continue
            content = getattr(log, "content", {})
            if content.get("vote_outcome") == "draw":
                targets = content.get("speech_queue")
                break
        if not isinstance(targets, list) or not targets:
            raise ValueError(
                "speech_pk requires the latest public draw speech_queue"
            )
    else:
        targets = public_state.get("suggestible_exile_targets")
        if not isinstance(targets, list):
            raise TypeError(
                "normal discussion requires suggestible_exile_targets"
            )

    if (
        any(
            isinstance(target, bool)
            or not isinstance(target, int)
            or not 1 <= target <= 7
            for target in targets
        )
        or len(set(targets)) != len(targets)
        or not set(targets) <= alive_set
    ):
        raise ValueError(
            "discussion vote targets are not authoritative living players"
        )
    return tuple(sorted(target for target in targets if target != actor))


def freeze_discussion_candidates(observation):
    """Build one deterministic Day-only discussion candidate snapshot."""

    actor = (
        observation.get("current_act_idx")
        if isinstance(observation, dict)
        else None
    )
    identity = (
        observation.get("identity")
        if isinstance(observation, dict)
        else None
    )
    if (
        isinstance(actor, bool)
        or not isinstance(actor, int)
        or not 1 <= actor <= 7
    ):
        raise ValueError("discussion requires current_act_idx in [1, 7]")
    if identity not in STRICT_CLASSIC7_ROLE_COUNTS:
        raise ValueError("discussion requires a supported classic7 identity")

    candidates = []
    for action in _ALL_PLAYER_TARGET_ACTIONS:
        for target in range(1, 8):
            if (
                action == "point_as_werewolf"
                and target == actor
            ):
                continue
            candidates.append(DiscussionAct(action, target))
    for action in _NON_SELF_TARGET_ACTIONS:
        for target in range(1, 8):
            if target != actor:
                candidates.append(DiscussionAct(action, target))
    for target in derive_discussion_vote_targets(observation):
        candidates.append(DiscussionAct("vote_intent", target))
    candidates.extend(
        (
            DiscussionAct("abstain_intent", None),
            DiscussionAct("no_commitment", None),
        )
    )
    return tuple(candidates)


def project_discussion_content_indices(candidate_snapshot):
    """Project canonical candidates into the V2 public-content view."""

    return tuple(
        index
        for index, act in enumerate(candidate_snapshot)
        if act.action in _PUBLIC_CONTENT_ACTIONS
    )


def project_discussion_vote_stances(candidate_snapshot):
    """Project canonical candidates into the V2 public vote-stance view."""

    return (NO_STANCE,) + tuple(
        act
        for act in candidate_snapshot
        if act.action in _PUBLIC_VOTE_STANCE_ACTIONS
    )


def compile_discussion_intent_v2(
    candidate_snapshot,
    *,
    public_content_action_indices,
    public_vote_stance_index,
):
    """Compile one parser-valid V2 transport into DiscussionAct V1."""

    discussion_acts = tuple(
        candidate_snapshot[index]
        for index in public_content_action_indices
    )
    vote_stance = project_discussion_vote_stances(candidate_snapshot)[
        public_vote_stance_index
    ]
    if vote_stance != NO_STANCE:
        discussion_acts += (vote_stance,)
    return discussion_acts or (DiscussionAct("no_commitment", None),)


def render_discussion_act(act):
    if not isinstance(act, DiscussionAct):
        raise TypeError("discussion intent must contain DiscussionAct values")
    return act.action if act.target is None else f"{act.action}(player{act.target})"


def _render_discussion_action_glossary():
    targetless_actions = {"abstain_intent", "no_commitment"}
    return "\n".join(
        f"- {action if action in targetless_actions else f'{action}(playerX)'}: "
        f"{_DISCUSSION_ACTION_SEMANTICS[action]}"
        for action in DISCUSSION_ACTIONS
    )


STRICT_CLASSIC7_GAME_DESCRIPTION = """你正在进行一局固定规则的7人多日制狼人杀游戏。

以下规则是所有玩家从游戏开始时就共同知道的确定规则。不要自行加入这里不存在的角色、技能、公开信息规则、投票规则或胜负条件。

【游戏配置】
- 本局共有7名玩家：2名狼人、1名预言家、1名女巫、3名普通村民。
- 本局没有守卫、猎人或其他任何角色。
- 所有玩家的身份整局固定，不会交换或改变。
- 狼人属于狼人阵营；预言家、女巫和普通村民属于好人阵营。

【存活、死亡与公开信息】
- 只有存活玩家可以继续执行夜间行动、白天发言、普通投票和PK投票。
- 玩家夜间死亡或白天被放逐后，退出后续主动游戏流程，但其此前已经公开的发言、投票和其他公开信息继续保留在游戏历史中。
- 夜间死亡和白天放逐都不会公开玩家的真实身份。
- 夜间技能及其私人信息默认不公开；只有游戏环境明确公开的结果才是所有玩家共同知道的事实。

【狼人】
- 两名狼人知道彼此的真实身份。
- 夜晚，存活狼人私下协作，最终形成一个狼队夜间决定。
- 狼队最终可以选择一名合法的存活非狼人玩家作为击杀目标，也可以主动选择当晚不进行击杀。
- 只有狼队最终决定会影响游戏状态；狼人内部的提议和协作过程属于狼人私有信息。
- 如果只剩一名狼人存活，则由该狼人独立作出当晚最终决定。

【预言家】
- 只要预言家存活且存在合法的未查验目标，每个夜晚必须查验一名玩家，不能主动放弃查验。
- 合法查验目标必须当前存活、不是预言家本人，并且此前从未被预言家查验过；已经查验过的玩家不能再次查验。
- 如果已经不存在任何合法且未查验的目标，则该夜不执行查验；这不属于主动放弃查验。
- 查验结果只告诉预言家目标“是狼人”或“不是狼人”。“不是狼人”不代表目标一定是普通村民，也可能是女巫等其他好人身份。
- 查验只提供信息，不会杀死、救活、保护、治疗、毒杀或以其他方式改变目标状态。
- 因此，即使某一夜没有任何玩家死亡，预言家仍然完全可能在该夜正常完成查验。

【女巫】
- 女巫整局拥有1瓶解药和1瓶毒药，每瓶最多使用一次；同一个夜晚不能同时使用两种药，也可以选择当晚不使用任何药。
- 解药尚未使用时，如果狼队当晚选择了击杀目标且女巫仍然存活，女巫会在自己的夜间行动阶段得知该击杀目标。
- 女巫只能用解药救当晚狼队的击杀目标，且不能对自己使用解药，因此不能自救；知道击杀目标并不意味着必须使用解药。
- 解药使用后不能再次使用，之后的夜晚女巫也不再获得狼队当晚击杀目标的信息。
- 毒药尚未使用时，女巫可以毒杀一名合法的存活其他玩家，但不能毒杀自己。
- 毒药目标可以与当晚狼队的击杀目标是同一名玩家；毒药使用后不能再次使用。

【夜间结算】
- 狼队击杀目标如果没有被女巫合法使用解药救下，则当夜死亡；被女巫合法使用毒药命中的玩家也在当夜死亡。
- 如果狼队击杀目标和女巫毒药目标是同一名玩家，该玩家只死亡一次。
- 预言家的查验不会直接导致死亡；因此，女巫使用毒药也不保证当晚一定出现两名不同的死亡玩家。
- 夜间全部行动完成后统一结算死亡结果，并公开死亡玩家名单，但不公开其真实身份。

【白天发言与普通投票】
- 白天只有存活玩家进行公开发言。公开发言只是玩家自己的陈述，可能真实、欺骗、错误或具有策略性；玩家说出的内容不会因此自动成为游戏环境确认的事实。
- 普通投票时，每名存活玩家可以投给一名合法的存活其他玩家，或者弃票。
- 当前投票阶段完成前，已经投出的具体票不会提前公开给之后投票的玩家；投票阶段完成后，票型和结果才成为公开信息。
- 唯一最高票玩家被放逐；如果多名玩家并列最高票，则这些玩家进入PK阶段。
- 如果本轮没有任何玩家获得有效票数，则无人被放逐，也不进入PK阶段。

【PK阶段】
- 只有普通投票中并列最高票的PK候选人进行额外PK发言。
- PK发言结束后，所有存活玩家都参加PK投票，包括PK候选人本人。
- 非PK候选人只能投给一名PK候选人或弃票；PK候选人本人只能投给另一名PK候选人或弃票；任何玩家都不能投给自己。
- 当前PK投票完成前，已经投出的具体票不会提前公开给之后投票的玩家；PK投票完成后，票型和结果才成为公开信息。
- 唯一最高票的PK候选人被放逐；如果最高票再次平票，或没有任何PK候选人获得有效票数，则本轮无人被放逐。

【胜负条件】
- 如果场上不再有任何存活狼人，好人阵营立即获胜。
- 否则，如果存活狼人数达到或超过其他存活玩家总数，狼人阵营立即获胜。

以上规则只定义本局允许发生什么以及信息如何产生。具体玩家当前身份、私人信息、已经发生的历史事实和当前合法行动目标，以游戏环境在当前时刻提供的信息为准。"""


def derive_belief_constraints(observation):
    """Derive one transient belief domain from a legal observation."""

    if not isinstance(observation, dict):
        raise TypeError("belief observation must be a dictionary")
    player_id = observation.get("observer_id")
    if player_id is None:
        player_id = observation.get("current_act_idx")
    if (
        isinstance(player_id, bool)
        or not isinstance(player_id, int)
        or not 1 <= player_id <= 7
    ):
        raise ValueError("belief observation requires an observer in [1, 7]")
    self_role = observation.get("identity")
    if self_role not in STRICT_CLASSIC7_ROLE_COUNTS:
        raise ValueError("belief observation has an unsupported identity")
    game_log = observation.get("game_log")
    if not isinstance(game_log, list):
        raise TypeError("belief observation requires a game_log list")

    other_players = [
        f"player{candidate}"
        for candidate in range(1, 8)
        if candidate != player_id
    ]
    exact_roles = {}
    excluded_roles = {player: set() for player in other_players}

    if self_role == "Werewolf":
        for log in game_log:
            if getattr(log, "event", None) != "werewolf_team_info":
                continue
            wolf_team = getattr(log, "content", {}).get("wolf_team", [])
            if not isinstance(wolf_team, list):
                continue
            for teammate in wolf_team:
                if (
                    isinstance(teammate, int)
                    and not isinstance(teammate, bool)
                    and 1 <= teammate <= 7
                    and teammate != player_id
                ):
                    exact_roles[f"player{teammate}"] = "Werewolf"

    if self_role == "Seer":
        for log in game_log:
            if getattr(log, "event", None) != "skill_seer":
                continue
            target = getattr(log, "target", None)
            result = getattr(log, "content", {}).get("cheked_identity")
            if (
                not isinstance(target, int)
                or isinstance(target, bool)
                or target == player_id
                or not 1 <= target <= 7
            ):
                continue
            player = f"player{target}"
            if result == "bad":
                exact_roles[player] = "Werewolf"
            elif result == "good":
                excluded_roles[player].add("Werewolf")

    if self_role == "Witch":
        for log in game_log:
            if getattr(log, "event", None) != "kill_decision":
                continue
            target = getattr(log, "target", None)
            if (
                isinstance(target, int)
                and not isinstance(target, bool)
                and 1 <= target <= 7
                and target != player_id
            ):
                excluded_roles[f"player{target}"].add("Werewolf")

    known_counts = {role: 0 for role in STRICT_BELIEF_CONCRETE_ROLES}
    known_counts[self_role] += 1
    for role in exact_roles.values():
        known_counts[role] += 1

    role_options = {}
    for player in other_players:
        if player in exact_roles:
            continue
        role_options[player] = tuple(
            role
            for role in STRICT_BELIEF_CONCRETE_ROLES
            if known_counts[role] < STRICT_CLASSIC7_ROLE_COUNTS[role]
            and role not in excluded_roles[player]
        ) + ("unknown",)
    return exact_roles, role_options


def _render_authoritative_public_phase(public_state):
    """Render the player-free phase view from authoritative public state."""

    required_fields = {"day", "day_or_night", "phase"}
    if not isinstance(public_state, dict) or not required_fields <= public_state.keys():
        raise TypeError("incomplete authoritative public phase")
    phase_names = {
        "speech": "公开发言",
        "speech_pk": "平票公开发言",
        "vote": "投票",
        "vote_pk": "平票投票",
        "skill_wolf": "狼人行动",
        "skill_seer": "预言家行动",
        "skill_witch": "女巫行动",
        "end_game": "游戏结束",
    }
    return "第{day}天{period}{phase}".format(
        day=public_state["day"],
        period="白天" if public_state["day_or_night"] == "day" else "夜晚",
        phase=phase_names.get(public_state["phase"], str(public_state["phase"])),
    )


def _render_authoritative_public_state(
    public_state,
    *,
    suggestible_player_ids=None,
):
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
    suggestible_targets = (
        public_state["suggestible_exile_targets"]
        if suggestible_player_ids is None
        else suggestible_player_ids
    )
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
        or not isinstance(suggestible_targets, (list, tuple))
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
    phase_text = _render_authoritative_public_phase(public_state)
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


def _render_authoritative_public_history(game_log):
    """Render legally visible Environment facts in observation order."""

    history = []
    for log in game_log:
        event = getattr(log, "event", None)
        content = getattr(log, "content", {})
        time_text = getattr(log, "time", "(time unavailable)")
        if event == "game_setting":
            counts = ", ".join(
                f"{role}={content.get(role, 0)}"
                for role in ("Werewolf", "Seer", "Witch", "Villager")
            )
            history.append(f"- {time_text} game setting: {counts}")
        elif event == "end_night":
            dead_players = content.get("dead_list", [])
            result = (
                ", ".join(f"player{player_id}" for player_id in dead_players)
                + " died"
                if dead_players
                else "no player died"
            )
            history.append(f"- {time_text} completed night result: {result}")
        elif event in {"vote", "vote_pk"}:
            voter = getattr(log, "source", None)
            target = getattr(log, "target", None)
            phase_name = "PK vote" if event == "vote_pk" else "vote"
            result = (
                "abstained"
                if target == 0
                else f"voted for player{target}"
            )
            history.append(
                f"- {time_text} completed {phase_name}: player{voter} {result}"
            )
        elif event == "end_vote":
            outcome = content.get("vote_outcome")
            expelled = content.get("expelled")
            if isinstance(expelled, int) and not isinstance(expelled, bool):
                result = f"player{expelled} was exiled"
            elif outcome == "draw":
                result = "the vote tied and entered PK speech"
            elif outcome == "draw in pk":
                result = "the PK vote tied; nobody was exiled"
            elif outcome == "all abstention":
                result = "all players abstained; nobody was exiled"
            elif outcome == "all abstention in pk":
                result = "all PK voters abstained; nobody was exiled"
            else:
                result = f"completed vote result: {outcome}"
            history.append(f"- {time_text} {result}")
        elif event == "end_game":
            history.append(f"- {time_text} game ended")
    return "\n".join(history) or "- (no completed public history yet)"


def _build_gameplay_context(observation, *, claim_catalog=None):
    """Render one legally filtered classic-7 observation."""
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
    for log in game_log:
        if getattr(log, "event", None) != "game_setting":
            continue
        role_counts = getattr(log, "content", {})
        if dict(role_counts) != STRICT_CLASSIC7_ROLE_COUNTS:
            raise ValueError(
                "strict gameplay requires exactly 2 Werewolves, 1 Seer, "
                "1 Witch and 3 Villagers"
            )
    public_state = observation.get(
        "authoritative_public_state"
    )
    if not isinstance(public_state, dict):
        raise TypeError(
            "strict speech requires authoritative_public_state"
        )
    authoritative_state_text = _render_authoritative_public_state(
        public_state,
    )

    if identity == "Werewolf":
        wolf_team = []
        night_proposals = []
        night_decisions = []
        for log in game_log:
            if getattr(log, "event", None) == "werewolf_team_info":
                wolf_team.extend(
                    getattr(log, "content", {}).get("wolf_team", [])
                )
            elif getattr(log, "event", None) == "skill_wolf":
                source = getattr(log, "source", None)
                target = getattr(log, "target", None)
                if (
                    isinstance(source, int)
                    and 1 <= source <= 7
                    and isinstance(target, int)
                    and 0 <= target <= 7
                ):
                    choice = "主动空刀" if target == 0 else f"击杀 player{target}"
                    night_proposals.append(
                        f"{log.time}：player{source} 提交{choice}"
                    )
            elif getattr(log, "event", None) == "kill_decision":
                target = getattr(log, "target", None)
                if isinstance(target, int) and 0 <= target <= 7:
                    decision = "主动空刀" if target == 0 else f"击杀 player{target}"
                    night_decisions.append(f"{log.time}：{decision}")
        team_text = ", ".join(
            f"player{player_id}" for player_id in sorted(set(wolf_team))
        ) or "(无合法可见狼队信息)"
        proposal_text = "；".join(night_proposals) or "(尚无已完成狼队夜间选择)"
        decision_text = "；".join(night_decisions) or "(尚无已完成狼刀决策)"
        role_rules = f"""你的真实私有身份是狼人。
- 真实狼队信息（仅用于内部策略）：{team_text}。
- 依法可见的真实狼队夜间选择（仅用于内部策略）：{proposal_text}。
- 已完成的真实夜间刀人决策（仅用于内部策略）：{decision_text}。
- 绝不能公开说自己是狼人，也不能直接公开狼人队友身份。
- 绝不能直接公开狼队夜间讨论、狼刀真实决策或夜间技能信息。
- 不得使用“因为系统告诉我”“我的真实队友是”等内部状态泄露式表达。
- 可以按策略假跳其他身份、隐藏信息或作出虚假公开声明，但不要逐字复制系统内部描述。"""
    elif identity == "Villager":
        role_rules = """你的真实私有身份是普通村民。
- 你没有查验、解药或毒药能力。
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
                    f"{log.time}:player{target}="
                    + (
                        "狼人"
                        if result == "bad"
                        else "不是狼人"
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
- 你没有真实的解药或毒药能力，也不知道狼人队友。"""
    elif identity == "Witch":
        heal_used = False
        poison_used = False
        witch_actions = []
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
                if "heal" in content and isinstance(target, int) and 1 <= target <= 7:
                    witch_actions.append(
                        f"{log.time}：对 player{target} 使用解药"
                    )
                elif "poison" in content and isinstance(target, int) and 1 <= target <= 7:
                    witch_actions.append(
                        f"{log.time}：对 player{target} 使用毒药"
                    )
                elif "pass" in content:
                    witch_actions.append(f"{log.time}：未使用药物")
            elif (
                event == "kill_decision"
                and isinstance(
                    target,
                    int,
                )
                and 0 <= target <= 7
            ):
                target_text = (
                    "狼队主动空刀"
                    if target == 0
                    else f"player{target}"
                )
                night_kill_targets.append(
                    f"{log.time}：{target_text}"
                )
        action_text = "；".join(witch_actions) or "(尚无已执行的女巫夜间行动)"
        kill_text = "；".join(night_kill_targets) or "(无合法可见的历史狼队夜间决定)"
        role_rules = f"""你的合法私有身份是女巫。
- 解药真实状态：{"已使用" if heal_used else "未使用"}。
- 毒药真实状态：{"已使用" if poison_used else "未使用"}。
- 已真实执行的女巫夜间行动：{action_text}。
- 合法可见的历史狼队夜间决定：{kill_text}。
- 这些是内部真实状态；公开时可以披露、隐藏、歪曲或虚构身份和技能声明。
- 女巫没有真实的查验能力，也不知道狼人队友。"""
    if claim_catalog is None:
        public_claims = []
        for log in game_log:
            if getattr(log, "event", None) not in {"speech", "speech_pk"}:
                continue
            source = getattr(log, "source", None)
            if not isinstance(source, int) or not 1 <= source <= 7:
                continue
            speech = getattr(log, "content", {}).get("speech_content")
            if isinstance(speech, str) and speech.strip():
                public_claims.append(
                    f"- [{log.time} / {log.event}] player{source}：{speech}"
                )
    else:
        if not isinstance(claim_catalog, tuple) or any(
            not isinstance(claim, PublicClaim) for claim in claim_catalog
        ):
            raise TypeError("claim_catalog must be a tuple of PublicClaim values")
        public_claims = [
            f"- {render_public_claim(claim)}" for claim in claim_catalog
        ]
    claims_text = "\n".join(public_claims) or "- (尚无玩家公开主张)"
    authoritative_history = _render_authoritative_public_history(game_log)

    faction_objective = (
        "隐藏狼人身份、误导村民阵营并帮助狼人阵营获胜"
        if identity == "Werewolf"
        else "识别并投票放逐所有狼人，帮助村民阵营获胜"
    )
    return f"""GAME / ROLE
Game rules:
{STRICT_CLASSIC7_GAME_DESCRIPTION}
Current player: player{actor}
Actual role supplied by the Environment: {identity}
Faction objective: {faction_objective}

KNOWN INFORMATION
AUTHORITATIVE INFORMATION
Environment authoritative public state:
{authoritative_state_text}

Authoritative public history (chronological):
{authoritative_history}

Private facts legally visible to this player:
{role_rules}

PUBLIC CONVERSATION
The following text is raw chronological public speech visible to this player.
These are player claims; they may be truthful, deceptive, mistaken or strategic.
A claim is not an authoritative fact merely because someone stated it.
{claims_text}"""


def _build_belief_output_instructions(observation, exact_roles, role_options):
    remaining_inventory = dict(STRICT_CLASSIC7_ROLE_COUNTS)
    remaining_inventory[observation["identity"]] -= 1
    for role in exact_roles.values():
        remaining_inventory[role] -= 1
    inventory_text = "\n".join(
        f"{role}: {remaining_inventory[role]}"
        for role in STRICT_BELIEF_CONCRETE_ROLES
    )
    exact_text = json.dumps(exact_roles, ensure_ascii=False, sort_keys=True)
    options_text = json.dumps(
        {
            player: list(options)
            for player, options in role_options.items()
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return f"""BELIEF OUTPUT
Treat the Environment-supplied self role and these exact-known other-player
roles as fixed premises. Do not reinterpret or re-guess them:
{exact_text}
Remaining concrete role inventory for unresolved players:
{inventory_text}
Across all unresolved players, concrete role guesses must not
exceed this remaining inventory. This is one global constraint across the complete roles object.
"unknown" consumes no role slot.
Infer only these unresolved players, using only each player's listed values:
{options_text}
The roles object must contain exactly those unresolved players and no known
player. Use "unknown" when the available information is insufficient.
The roles-object inference rules above apply to the roles field, not to which
players the gameplay belief may discuss.
The belief field is gameplay cognition. Treat the Environment authoritative
current state, self role, exact-known private facts, and exact-known other-player
roles as fixed premises. Use both exact-known facts and relevant unresolved-player
evidence when assessing the current situation. You may mention exact-known
players when they matter to gameplay, but do not reinterpret or re-guess fixed facts.
Keep the reasoning as compact evidence -> current-assessment steps, not a long
hidden-thought transcript, and do not try to explain or enumerate the roles object.
Do not restate the game rules or recount or recompute the fixed 7-player role composition.
Do not repeat the observation or history. Do not mechanically discuss every unresolved player
when there is no useful evidence. Use unknown rather than inventing evidence.
Be as concise as possible and target no more than about 50 words.
The concise field must be a short gameplay conclusion derived from belief, not a
summary of the roles object, and must be no more than 2 short sentences."""


def build_belief_prompt(
    observation,
    *,
    exact_roles=None,
    role_options=None,
):
    """Ask for one transient belief report from a legal observation."""

    if exact_roles is None and role_options is None:
        exact_roles, role_options = derive_belief_constraints(observation)
    elif exact_roles is None or role_options is None:
        raise ValueError("exact_roles and role_options must be supplied together")
    context = _build_gameplay_context(observation)
    belief_instructions = _build_belief_output_instructions(
        observation,
        exact_roles,
        role_options,
    )
    return f"""{context}

{belief_instructions}
Return only the JSON object required by the response schema. The three fields
remain belief, concise and roles."""


def build_day_cognition_prompt(
    observation,
    *,
    candidate_snapshot,
    claim_catalog,
    pre_speech_belief,
):
    """Select public discussion intent from one frozen PRE belief."""

    from werewolf.models.twd_tom.samples import SpeakerPreSpeechBelief

    if not isinstance(candidate_snapshot, tuple) or not candidate_snapshot:
        raise ValueError("candidate_snapshot must be a non-empty tuple")
    if any(not isinstance(act, DiscussionAct) for act in candidate_snapshot):
        raise TypeError("candidate_snapshot must contain DiscussionAct values")
    if not isinstance(pre_speech_belief, SpeakerPreSpeechBelief):
        raise TypeError(
            "day cognition requires immutable SpeakerPreSpeechBelief"
        )
    context = _build_gameplay_context(
        observation,
        claim_catalog=claim_catalog,
    )
    content_indices = project_discussion_content_indices(candidate_snapshot)
    content_candidate_text = "\n".join(
        f"{index}: {render_discussion_act(act)}"
        for index, act in enumerate(candidate_snapshot)
        if index in content_indices
    )
    vote_stance_text = "\n".join(
        f"{index}: {stance if stance == NO_STANCE else render_discussion_act(stance)}"
        for index, stance in enumerate(
            project_discussion_vote_stances(candidate_snapshot)
        )
    )
    claim_ids = [claim.claim_id for claim in claim_catalog]
    prompt_payload = pre_speech_belief.prompt_payload()
    pre_speech_belief_block = f"""PRE-SPEECH PRIVATE BELIEF (IMMUTABLE INPUT)
{json.dumps(prompt_payload, ensure_ascii=False, separators=(",", ":"))}
This is the exact readonly self-report captured at the current PRE boundary.
Use it as the fixed private wolf-suspicion support for this cognition; do not
regenerate, add, delete or reinterpret its entries. Internal belief and public
communication may strategically differ, so it does not force a matching public
claim or vote stance."""
    return f"""{context}

{pre_speech_belief_block}

DISCUSSION ACTION SEMANTICS
{_render_discussion_action_glossary()}
These are communication semantics only. They are never truth labels.

INTERACTION GUIDANCE
When the current table state supports a meaningful response, strongly prefer at
least one existing support(...) or oppose(...) action that states a
current table judgment, instead of emitting only an isolated role or skill
declaration. Keep the total public-content selection within the existing
zero-to-two limit. No new action type is introduced, and point_as_werewolf can
never target the speaker.

DISCUSSION INTENT OUTPUT
Set public_content_selection.mode to none, one or two. For one, first_index is
one absolute index from this deterministic projection of the frozen canonical
candidate snapshot. For two, first_index is the first absolute index and
second_rank is the zero-based rank in the same list after removing first_index:
{content_candidate_text}
Choose exactly one public_vote_stance_index from this deterministic projection:
{vote_stance_text}
NO_STANCE means no publicly stated voting tendency in this speech.
The public vote stance is speech intent, not the authoritative later Vote-phase ballot.
no_commitment is not selectable. When public content is empty and NO_STANCE is
selected, the program represents the empty discussion intent canonically.
These indices describe only what the current speaker intends to communicate
publicly. They are public claim/positioning primitives, not truth labels.
Strategic deception and bluff remain allowed within this frozen candidate space.
Set evidence_selection.mode to none, one or two. For one, first_claim_id is from
the visible public claim catalog. For two, first_claim_id is first and
second_rank is the zero-based rank in the catalog after removing that claim:
{json.dumps(claim_ids)}
Choose claims you consider relevant evidence for the current cognition and
discussion-intent decision. The selected IDs form an internal linkage record for
audit only. They do not authorize or require quoting, paraphrasing or mentioning
those claims in public speech. Selection does not assert truth or falsity and does
not prove causal influence on the belief or action. Do not provide a reason,
confidence, strategy name, expected reaction, or any free-text public plan.
Return only the JSON object required by the response schema. The three fields are
public_content_selection, public_vote_stance_index and evidence_selection."""


def build_public_speech_realization_prompt(
    observation,
    *,
    discussion_acts,
    claim_catalog,
):
    """Realize a frozen private discussion intent as ordinary public speech."""

    if not isinstance(observation, dict):
        raise TypeError("speech realization observation must be a dictionary")
    speaker = observation.get("current_act_idx")
    phase = observation.get("phase")
    day = observation.get("day")
    if (
        isinstance(speaker, bool)
        or not isinstance(speaker, int)
        or not 1 <= speaker <= 7
    ):
        raise ValueError("speech realization requires current_act_idx in [1, 7]")
    if not isinstance(phase, str) or "speech" not in phase:
        raise ValueError("speech realization requires a speech phase")
    if isinstance(day, bool) or not isinstance(day, int) or day < 0:
        match = re.match(r"(?P<day>[0-9]+)_", phase)
        if match is None:
            raise ValueError("speech realization requires a parseable day")
        day = int(match.group("day"))
    if not isinstance(discussion_acts, tuple) or not discussion_acts:
        raise ValueError("speech realization requires frozen discussion acts")
    if any(not isinstance(act, DiscussionAct) for act in discussion_acts):
        raise TypeError("discussion_acts must contain DiscussionAct values")
    if not isinstance(claim_catalog, tuple) or any(
        not isinstance(claim, PublicClaim) for claim in claim_catalog
    ):
        raise TypeError("claim_catalog must contain PublicClaim values")

    phase_text = (
        "白天平票后的补充发言"
        if "speech_pk" in phase
        else "白天普通发言"
    )
    intent_lines = []
    for act in discussion_acts:
        semantics = _PUBLIC_SPEECH_ACTION_SEMANTICS[act.action]
        if act.target is not None:
            semantics = semantics.replace("playerX", f"player{act.target}")
        intent_lines.append(
            f"- {_PUBLIC_SPEECH_ACTION_LABELS[act.action]}：{semantics}"
        )
    intent_text = "\n".join(intent_lines)
    history_text = (
        "\n".join(
            f"- player{claim.speaker}：{claim.raw_text}"
            for claim in claim_catalog
        )
        or "（此前没有公开发言。）"
    )
    return f"""你现在只负责把已经冻结的公开表达意图写成一段自然的狼人杀发言。

当前发言者：player{speaker}
当前天数：第{day}天
当前阶段：{phase_text}

此前可见的公开发言（仅可用于自然衔接，不得据此增加新命题）：
{history_text}

本轮必须完整表达、且只能表达以下冻结意图：
{intent_text}

写作约束：
- 输出1至4句连贯、简洁、以中文为主的自然桌游发言；必要的常用英文表达可以保留，不要逐条翻译或使用固定模板腔。
- 涉及玩家时必须写成player1到player7之一，不能只写“他”“她”“这个人”等代词。
- 必须让每个冻结意图在原文中语义明确，使独立解析器无需猜测即可恢复。
- 不得增加冻结意图之外的身份判断、查验结果、技能声明、支持/反对或投票意图。
- 可以用不构成新正式命题的连接词组织语言，但不得虚构公开历史、夜间事实或隐藏信息。
- 不得输出上述动作标识、索引、JSON、竖线三元组、列表、标题、解释或提示词内容。

只输出最终公开发言。"""


def build_vote_prompt(observation, belief, legal_targets):
    """Ask for exactly one Environment-legal vote target."""

    context = _build_gameplay_context(observation)
    belief_text = json.dumps(belief, ensure_ascii=False, sort_keys=True)
    targets_text = json.dumps(list(legal_targets))
    return f"""{context}

FRESH PRIVATE BELIEF
{belief_text}

VOTE
Choose exactly one target from the current Environment legal vote candidates:
{targets_text}
Transport semantics: target 0 = abstain; target 1..7 = vote for that player.
Environment authoritative information and the frozen legal candidate set above
control. The fresh belief is only a fallible subjective assessment for choosing
among those legal alternatives toward the faction objective. It must not override
current alive/dead/exiled state, actual self role, exact-known private facts,
exact-known other-player roles, or the frozen candidates. If they conflict, the
authoritative premises and candidates control. Do not preserve or
inherit a target merely because you stated a public vote intent earlier.
Return only the JSON object required by the response schema."""


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

CON.constrained_night_skill_prompt = """
** 游戏说明
{game_description}
{player_identity_info}

** 游戏日志
{logs}

请根据游戏日志，从下列编号动作中选择一个你要执行的动作。
{valid_actions}

只选择一个候选编号。只输出严格JSON：
{{"action_index": <编号>}}
不得输出其他字段或列表外编号。

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
