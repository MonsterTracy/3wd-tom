import unittest

from werewolf.agents.llm_agent import LLMAgent
from werewolf.agents.prompt_template_v0 import (
    CON,
    build_strict_classic7_speech_plan_prompt,
    build_strict_classic7_speech_render_prompt,
)
from werewolf.agents.twdm_strategy import TWDMStrategy
from werewolf.helper.log_utils import Log


VOTE_CONSISTENCY_RULES = (
    "基于当前可见 observation 中的公开信息",
    "继承你自己白天发言中的怀疑、支持、站边和投票意向",
    "明确怀疑过某人，优先从这些对象中选择投票目标",
    "不要把“跟随 X 归票”理解成“投 X”",
    "自己之前没有怀疑过的人",
    "不要无依据随机投票",
    "不要投给自己",
    "不要投已死亡玩家",
)


class VotePromptConsistencyTest(unittest.TestCase):
    def test_standard_vote_prompt_contains_consistency_rules(self):
        prompt = CON.vote_prompt.format(
            game_description="game",
            player_identity_info="identity",
            logs="logs",
            valid_actions="actions",
        )

        for rule in VOTE_CONSISTENCY_RULES:
            self.assertIn(rule, prompt)

    def test_twdm_vote_prompt_contains_consistency_rules(self):
        prompt = CON.vote_prompt_v3.format(
            player_identity_info="identity",
            objective_info="objective",
            subjective_info="subjective",
            your_role="role",
        )

        for rule in VOTE_CONSISTENCY_RULES:
            self.assertIn(rule, prompt)
        self.assertIn("投票原因", prompt)
        self.assertIn("说明为什么改变目标", prompt)


class WerewolfPublicPerspectivePromptTest(unittest.TestCase):
    def test_werewolf_speech_prompt_forbids_private_information_leakage(self):
        role_prompt = CON.identity_abilities["Werewolf"]
        strategy_prompt = TWDMStrategy().build_hint(
            {
                "identity": "Werewolf",
                "phase": "1_day_speech",
            }
        )
        combined_prompt = role_prompt + strategy_prompt

        for rule in (
            "公开发言必须伪装成普通好人公开视角",
            "禁止暴露狼人队友身份",
            "禁止暴露夜晚刀人目标",
            "禁止声称某人是被女巫救下的人",
            "不能把私有信息包装成确定公共事实",
            "白天发言、公开死亡结果、公开投票结果、公开身份声明和公开逻辑矛盾",
            "只能转化成公开视角下的模糊怀疑",
        ):
            self.assertIn(rule, combined_prompt)


def _speech_observation(identity, *, game_log=None):
    return {
        "identity": identity,
        "phase": "1_day_speech",
        "current_act_idx": 3,
        "valid_action": [],
        "game_log": list(game_log or []),
        "authoritative_public_state": {
            "day": 1,
            "day_or_night": "day",
            "phase": "speech",
            "last_night_result": {"day": 0, "dead_players": []},
            "prior_exiles": [],
            "alive_players": [1, 2, 3, 4, 5, 6, 7],
            "suggestible_exile_targets": [1, 2, 4, 5, 6, 7],
        },
    }


def _log(*, event, source=3, target=0, content=None):
    return Log(
        viewer=3,
        source=source,
        target=target,
        content=dict(content or {}),
        day=1,
        time="night",
        event=event,
    )


class StrictClassic7GameplayPromptTest(unittest.TestCase):
    def test_profile_is_opt_in_and_legacy_prompt_is_unchanged(self):
        observation = _speech_observation("Villager")
        default_prompt = LLMAgent().format_observation(observation)
        legacy_prompt = LLMAgent(
            gameplay_prompt_profile="legacy"
        ).format_observation(observation)
        strict_prompt = LLMAgent(
            gameplay_prompt_profile="strict_classic7"
        ).format_observation(observation)

        self.assertEqual(default_prompt, legacy_prompt)
        self.assertNotIn("strict_classic7", legacy_prompt)
        self.assertIn("当前 speaker：player3", strict_prompt)
        self.assertIn("Private Planner", strict_prompt)
        self.assertIn("只输出 public_actions", strict_prompt)
        for section in (
            "【权威公共状态】",
            "【你合法知道的私有信息】",
            "【其他玩家此前的公开主张】",
            "【计划合同】",
        ):
            self.assertIn(section, strict_prompt)

        for contract in (
            "不要输出最终自然语言发言或解释",
            "空 public_actions 合法",
            "只输出符合请求 JSON Schema 的对象",
        ):
            self.assertIn(contract, strict_prompt)

    def test_role_rules_only_describe_real_role_capabilities(self):
        werewolf = build_strict_classic7_speech_plan_prompt(
            _speech_observation(
                "Werewolf",
                game_log=[
                    _log(
                        event="werewolf_team_info",
                        content={"wolf_team": [1, 3]},
                    ),
                    _log(event="kill_decision", target=5),
                ],
            )
        )
        villager = build_strict_classic7_speech_plan_prompt(
            _speech_observation("Villager")
        )
        seer = build_strict_classic7_speech_plan_prompt(
            _speech_observation(
                "Seer",
                game_log=[
                    _log(
                        event="skill_seer",
                        target=6,
                        content={"cheked_identity": "bad"},
                    )
                ],
            )
        )
        witch = build_strict_classic7_speech_plan_prompt(
            _speech_observation(
                "Witch",
                game_log=[
                    _log(event="kill_decision", target=5),
                    _log(
                        event="skill_witch",
                        target=5,
                        content={"heal": True},
                    ),
                ],
            )
        )
        guard = build_strict_classic7_speech_plan_prompt(
            _speech_observation(
                "Guard",
                game_log=[
                    _log(event="skill_guard", target=2)
                ],
            )
        )

        self.assertIn("真实狼队信息（仅用于内部策略）：player1, player3", werewolf)
        self.assertIn("真实夜间刀人决策（仅用于内部策略）：player5", werewolf)
        self.assertIn("不能直接公开狼人队友身份", werewolf)
        self.assertIn("不能直接公开狼队夜间讨论、狼刀真实决策", werewolf)
        self.assertIn("你没有查验、解药、毒药或守护能力", villager)
        self.assertIn("可以按策略假跳身份或作出虚假技能声明", villager)
        self.assertNotIn("狼人队伍的成员", villager)
        self.assertIn("player6=狼人", seer)
        self.assertIn("可以披露、隐藏、歪曲或虚构身份和技能声明", seer)
        self.assertIn("解药真实状态：已使用", witch)
        self.assertIn("毒药真实状态：未使用", witch)
        self.assertIn("player5", witch)
        self.assertIn("已真实发生的守护目标：player2", guard)
        for forbidden_contract in (
            "不得声称执行过查验",
            "只能引用上面已经真实发生的查验",
            "只能依据这些真实状态发言",
            "只能引用已经真实发生的守护",
        ):
            self.assertNotIn(forbidden_contract, "\n".join(
                (werewolf, villager, seer, witch, guard)
            ))

    def test_renderer_receives_only_authority_actor_and_validated_plan(self):
        observation = _speech_observation(
            "Werewolf",
            game_log=[
                _log(event="werewolf_team_info", content={"wolf_team": [1, 6]}),
                _log(event="kill_decision", target=3),
                _log(event="speech", source=2, content={"speech_content": "秘密原文"}),
            ],
        )
        observation["current_act_idx"] = 6
        observation["authoritative_public_state"]["suggestible_exile_targets"] = [
            1, 2, 3, 4, 5, 7
        ]
        planner = build_strict_classic7_speech_plan_prompt(observation)
        renderer = build_strict_classic7_speech_render_prompt(
            authoritative_public_state=observation["authoritative_public_state"],
            actor=6,
            public_actions=[{"action": "oppose", "target": 2}],
        )

        self.assertIn("真实狼队信息（仅用于内部策略）：player1, player6", planner)
        self.assertIn("真实夜间刀人决策（仅用于内部策略）：player3", planner)
        self.assertIn("秘密原文", planner)
        self.assertIn('"action":"oppose"', renderer)
        for private_text in ("真实狼队信息", "夜间刀人", "秘密原文", "game_log"):
            self.assertNotIn(private_text, renderer)

        empty_renderer = build_strict_classic7_speech_render_prompt(
            authoritative_public_state=observation["authoritative_public_state"],
            actor=6,
            public_actions=[],
        )
        self.assertIn('{"public_actions":[]}', empty_renderer)
        self.assertIn("不得点名其他玩家", empty_renderer)

    def test_renderer_separates_oppose_from_vote_intent(self):
        observation = _speech_observation("Villager")
        stance_only = build_strict_classic7_speech_render_prompt(
            authoritative_public_state=observation["authoritative_public_state"],
            actor=3,
            public_actions=[
                {"action": "point_as_villager", "target": 3},
                {"action": "oppose", "target": 4},
            ],
        )
        with_vote = build_strict_classic7_speech_render_prompt(
            authoritative_public_state=observation["authoritative_public_state"],
            actor=3,
            public_actions=[
                {"action": "point_as_villager", "target": 3},
                {"action": "oppose", "target": 4},
                {"action": "vote_intent", "target": 4},
            ],
        )

        self.assertIn('"action":"point_as_villager","target":3', stance_only)
        self.assertIn('"action":"oppose","target":4', stance_only)
        self.assertIn("不自动产生 support", stance_only)
        self.assertIn("oppose(X) 不等于 vote_intent(X)", stance_only)
        self.assertIn("计划没有 vote_intent", stance_only)
        self.assertIn("不得表达投票给、票出、放逐、驱逐", stance_only)
        self.assertIn("vote_intent target player4", with_vote)
        self.assertIn("才可被表达为投票给、票出、放逐、驱逐", with_vote)


if __name__ == "__main__":
    unittest.main()
