import unittest

from werewolf.agents.prompt_template_v0 import CON
from werewolf.agents.twdm_strategy import TWDMStrategy


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


if __name__ == "__main__":
    unittest.main()
