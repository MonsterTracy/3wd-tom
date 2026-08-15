import unittest

from werewolf.agents.llm_agent import LLMAgent
from werewolf.helper.log_utils import Log


class VotePromptConsistencyTest(unittest.TestCase):
    def test_vote_prompt_is_indexed_and_contains_no_strategy_contract(self):
        agent = LLMAgent()
        observation = {
            "phase": "1_day_vote",
            "identity": "Werewolf",
            "current_act_idx": 1,
            "game_log": [],
            "valid_action": [("vote", 0), ("vote", 2), ("vote", 4)],
        }
        candidates = agent.freeze_authoritative_vote_candidates(
            observation["valid_action"]
        )
        prompt = agent.format_observation(
            observation,
            action_candidates=candidates,
        )

        for required in (
            "当前投票阶段",
            "0: abstain",
            "1: vote player2",
            "2: vote player4",
            '{"action_index": <编号>}',
        ):
            self.assertIn(required, prompt)
        for forbidden in (
            "投票玩家",
            "投票原因",
            "跟随",
            "优先",
            "保护队友",
            "分票",
            "隐藏你的身份",
        ):
            self.assertNotIn(forbidden, prompt)



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
