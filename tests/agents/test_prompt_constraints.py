import unittest

from werewolf.agents.llm_agent import LLMAgent
from werewolf.agents.prompt_template_v0 import CON
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



def _speech_observation(*, identity="Seer", game_log=None):
    return {
        "identity": identity,
        "phase": "speech",
        "current_act_idx": 3,
        "valid_action": [],
        "game_log": list(game_log or []),
        "authoritative_public_state": {
            "day": 2,
            "day_or_night": "day",
            "phase": "speech",
            "last_night_result": {"day": 1, "dead_players": [7]},
            "prior_exiles": [{"day": 1, "player_id": 6}],
            "alive_players": [1, 2, 3, 4, 5],
            "suggestible_exile_targets": [1, 2, 4, 5],
        },
    }


def _log(*, event, source=3, target=0, content=None, time="第1天白天"):
    return Log(
        viewer=[0, 1, 2, 3, 4, 5, 6],
        source=source,
        target=target,
        content=dict(content or {}),
        day=1,
        time=time,
        event=event,
    )


class DirectPublicSpeechPromptTest(unittest.TestCase):
    def test_prompt_is_minimal_behavior_generation_contract(self):
        agent = LLMAgent()
        prompt = agent.format_observation(_speech_observation())

        for expected in (
            "7人狼人杀",
            "你是3号玩家，你的身份是预言家",
            "第2天白天，公开发言",
            "当前存活玩家：1号、2号、3号、4号、5号",
            "昨夜死亡：7号",
            "6号（第1天）",
            "现在轮到3号玩家公开发言",
            "只输出本轮要公开说出的自然语言发言",
        ):
            self.assertIn(expected, prompt)
        for forbidden in (
            "public_actions",
            "Core-13",
            "JSON Schema",
            "confidence",
            "如何发言",
            "何时公开身份",
            "假跳",
            "狼人队友",
        ):
            self.assertNotIn(forbidden, prompt)

    def test_prompt_preserves_legal_private_history_and_own_public_speech(self):
        own_speech = "我上一轮明确说过，我暂时支持2号。"
        observation = _speech_observation(game_log=[
            _log(
                event="skill_seer",
                target=4,
                content={"cheked_identity": "bad"},
                time="第1天夜晚",
            ),
            _log(
                event="speech",
                source=3,
                content={"speech_content": own_speech},
            ),
        ])
        prompt = LLMAgent().format_observation(observation)

        self.assertIn("查验了4号的身份是狼人", prompt)
        self.assertIn(own_speech, prompt)
        self.assertLess(
            prompt.index("查验了4号的身份是狼人"),
            prompt.index(own_speech),
        )

    def test_prompt_contains_only_the_supplied_legal_observation(self):
        prompt = LLMAgent().format_observation(_speech_observation(identity="Villager"))

        self.assertIn("身份是村民", prompt)
        self.assertNotIn("查验了", prompt)
        self.assertNotIn("狼队信息", prompt)
        self.assertNotIn("Reporter", prompt)
        self.assertNotIn("ToM", prompt)


if __name__ == "__main__":
    unittest.main()
