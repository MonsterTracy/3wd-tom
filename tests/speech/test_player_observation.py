from copy import deepcopy
import unittest

from werewolf.agents.llm_agent import LLMAgent
from werewolf.agents.prompt_template_v0 import (
    build_strict_classic7_speech_rules,
)
from werewolf.envs.werewolf_text_env_v0 import (
    WerewolfTextEnvV0,
)
from werewolf.helper.log_utils import Log


ROLES = [
    "Werewolf",
    "Werewolf",
    "Seer",
    "Witch",
    "Villager",
    "Villager",
    "Villager",
]


def serialize_logs(logs):
    return [
        deepcopy(log.__dict__)
        for log in logs
    ]


class PlayerObservationTest(unittest.TestCase):
    def setUp(self):
        self.env = WerewolfTextEnvV0(
            log_save_path=None,
        )

        self.env.reset(
            roles=ROLES,
        )

    def test_observation_for_arbitrary_player(self):
        observation = self.env.get_observation_for(
            3
        )

        self.assertEqual(
            observation["observer_id"],
            3,
        )

        # The first actual actor is player1, a werewolf.
        self.assertEqual(
            observation["current_act_idx"],
            1,
        )

        self.assertEqual(
            observation["identity"],
            "Seer",
        )

        # Player3 is not currently acting.
        self.assertEqual(
            observation["valid_action"],
            [],
        )

    def test_current_actor_keeps_valid_actions(self):
        current_observation = (
            self.env.get_observation()
        )

        explicit_observation = (
            self.env.get_observation_for(1)
        )

        self.assertEqual(
            current_observation["observer_id"],
            1,
        )

        self.assertEqual(
            current_observation["current_act_idx"],
            1,
        )

        self.assertEqual(
            current_observation["valid_action"],
            explicit_observation["valid_action"],
        )

        self.assertGreater(
            len(current_observation["valid_action"]),
            0,
        )

    def test_private_visibility_is_player_specific(self):
        wolf_observation = (
            self.env.get_observation_for(1)
        )

        seer_observation = (
            self.env.get_observation_for(3)
        )

        wolf_events = {
            log.event
            for log in wolf_observation["game_log"]
        }

        seer_events = {
            log.event
            for log in seer_observation["game_log"]
        }

        self.assertIn(
            "werewolf_team_info",
            wolf_events,
        )

        self.assertNotIn(
            "werewolf_team_info",
            seer_events,
        )

        self.assertNotIn(
            "god_view",
            wolf_events,
        )

        self.assertNotIn(
            "god_view",
            seer_events,
        )

        wolf_team_log = next(
            log
            for log in wolf_observation["game_log"]
            if log.event == "werewolf_team_info"
        )

        self.assertEqual(
            wolf_team_log.content["wolf_team"],
            [1, 2],
        )

    def test_observation_generation_does_not_mutate_env(self):
        before_current_actor = (
            self.env.current_act_idx
        )
        before_phase = self.env.phase
        before_day = self.env.day
        before_logs = serialize_logs(
            self.env.game_log
        )

        self.env.get_observation_for(1)
        self.env.get_observation_for(3)
        self.env.get_observation_for(7)

        self.assertEqual(
            self.env.current_act_idx,
            before_current_actor,
        )

        self.assertEqual(
            self.env.phase,
            before_phase,
        )

        self.assertEqual(
            self.env.day,
            before_day,
        )

        self.assertEqual(
            serialize_logs(self.env.game_log),
            before_logs,
        )

    def test_authoritative_public_state_ignores_player_claims(self):
        self.env.day = 2
        self.env.day_or_night = "day"
        self.env.phase = "speech"
        self.env.current_act_idx = 0
        self.env.alive = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]
        self.env.game_log.extend(
            [
                Log(
                    viewer=list(range(7)),
                    source=-1,
                    target=3,
                    content={"vote_outcome": 3, "expelled": 3},
                    day=1,
                    time="第1天白天",
                    event="end_vote",
                ),
                Log(
                    viewer=list(range(7)),
                    source=4,
                    target=list(range(7)),
                    content={
                        "speech_content": "player3 已死，大家不用管他。",
                        "sp_actions": [],
                    },
                    day=2,
                    time="第2天白天",
                    event="speech",
                ),
            ]
        )

        observations = [
            self.env.get_observation_for(player_id)
            for player_id in (1, 3, 4, 5)
        ]
        public_state = observations[0]["authoritative_public_state"]
        self.assertTrue(
            all(
                observation["authoritative_public_state"] == public_state
                for observation in observations[1:]
            )
        )
        self.assertEqual(public_state["alive_players"], [1, 3, 5, 6, 7])
        self.assertEqual(
            public_state["eliminated_players"],
            [
                {"player_id": 2, "status": "dead"},
                {"player_id": 4, "status": "exiled"},
            ],
        )
        self.assertEqual(public_state["legal_vote_targets"], [])

        prompt = LLMAgent(
            gameplay_prompt_profile="strict_classic7"
        ).format_observation(observations[0])
        authoritative, remainder = prompt.split(
            "【你合法知道的私有信息】",
            1,
        )
        self.assertIn("存活玩家：player1, player3, player5, player6, player7", authoritative)
        self.assertIn("player2：已死亡", authoritative)
        self.assertIn("player4：已放逐", authoritative)
        self.assertIn("本轮合法投票目标：当前无投票动作", authoritative)
        self.assertNotIn("player3 已死", authoritative)
        self.assertNotIn("player0", prompt)
        self.assertIn("【其他玩家此前的公开主张】", remainder)
        self.assertIn("player5：player3 已死，大家不用管他。", remainder)
        self.assertIn("可能是真话、谎言、误解或策略性表达", remainder)

        self.env.phase = "vote"
        vote_state = self.env.get_observation_for(1)[
            "authoritative_public_state"
        ]
        self.assertEqual(
            vote_state["legal_vote_targets"],
            [1, 3, 5, 6, 7],
        )

    def test_invalid_player_id_is_rejected(self):
        with self.assertRaises(ValueError):
            self.env.get_observation_for(0)

        with self.assertRaises(ValueError):
            self.env.get_observation_for(8)

        with self.assertRaises(TypeError):
            self.env.get_observation_for("3")

        with self.assertRaises(TypeError):
            self.env.get_observation_for(True)

    def test_seer_pass_creates_no_completed_investigation(self):
        self.env.step(("kill", 5))
        self.env.step(("kill", 5))
        self.env.step(("check", 0))

        self.assertEqual(self.env.seer_check_target, {})
        self.assertFalse(
            any(
                log.event == "skill_seer"
                for log in self.env.game_log
            )
        )

        for player_id in range(1, 8):
            observation = self.env.get_observation_for(player_id)
            formatted = LLMAgent().format_log(
                observation["game_log"]
            )
            self.assertNotIn("player0", formatted)
            self.assertNotIn("0号", formatted)

        seer_observation = self.env.get_observation_for(3)
        self.assertNotIn(
            "查验了",
            LLMAgent().format_log(
                seer_observation["game_log"]
            ),
        )
        strict_rules = build_strict_classic7_speech_rules(
            seer_observation
        )
        self.assertIn("(尚无已完成查验)", strict_rules)

    def test_seer_player3_check_is_unchanged(self):
        self.env.step(("kill", 5))
        self.env.step(("kill", 5))
        self.env.step(("check", 3))

        self.assertEqual(
            list(self.env.seer_check_target.values()),
            [2],
        )
        seer_observation = self.env.get_observation_for(3)
        check_logs = [
            log
            for log in seer_observation["game_log"]
            if log.event == "skill_seer"
        ]
        self.assertEqual(len(check_logs), 1)
        self.assertEqual(check_logs[0].target, 3)
        self.assertEqual(
            check_logs[0].content["cheked_identity"],
            "good",
        )
        formatted = LLMAgent().format_log(
            seer_observation["game_log"]
        )
        self.assertIn("查验了3号的身份是好人", formatted)
        self.assertNotIn("player0", formatted)
        self.assertNotIn("0号", formatted)
        self.assertIn(
            "player3=好人",
            build_strict_classic7_speech_rules(
                seer_observation
            ),
        )


if __name__ == "__main__":
    unittest.main()
