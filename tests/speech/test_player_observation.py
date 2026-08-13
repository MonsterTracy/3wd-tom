from copy import deepcopy
import unittest

from werewolf.agents.llm_agent import LLMAgent
from werewolf.agents.prompt_template_v0 import (
    build_strict_classic7_speech_plan_prompt,
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

    def test_normal_vote_candidates_are_alive_and_non_self(self):
        self.env.phase = "vote"
        self.env.day = 1
        self.env.day_or_night = "day"
        self.env.current_act_idx = 1
        self.env.alive = [1, 1, 1, 1, 0, 0, 0]

        observation = self.env.get_observation_for(2)

        self.assertEqual(
            observation["valid_action"],
            [("vote", 0), ("vote", 1), ("vote", 3), ("vote", 4)],
        )
        self.assertEqual(
            observation["authoritative_public_state"][
                "suggestible_exile_targets"
            ],
            [1, 3, 4],
        )

    def test_normal_vote_parser_and_environment_reject_self_vote(self):
        self.env.phase = "vote"
        self.env.day = 1
        self.env.day_or_night = "day"
        self.env.current_act_idx = 1
        self.env.alive = [1, 1, 1, 1, 0, 0, 0]
        observation = self.env.get_observation_for(2)
        agent = LLMAgent()
        agent.get_valid_actions_str(observation["valid_action"])
        valid_actions = list(agent.nlp_action_to_env_action)

        self.assertIsNone(
            agent.parse_vote_action(
                "{'投票': '2'}",
                observation,
                valid_actions,
            )
        )
        with self.assertRaisesRegex(ValueError, "invalid normal vote action"):
            self.env.step(("vote", 2))

    def test_vote_pk_candidates_exclude_each_current_voter(self):
        self.env.phase = "vote_pk"
        self.env.day = 1
        self.env.day_or_night = "day"
        self.env.alive = [1, 0, 1, 0, 0, 1, 1]
        self.env.vote_pk_players = [5, 6, 0, 2]

        expected_targets = {
            3: [6, 7, 1],
            1: [6, 7, 3],
            6: [7, 1, 3],
            7: [6, 1, 3],
        }
        for voter, targets in expected_targets.items():
            with self.subTest(voter=voter):
                self.env.current_act_idx = voter - 1
                valid_actions = self.env.get_observation_for(voter)[
                    "valid_action"
                ]
                self.assertEqual(
                    valid_actions,
                    [("vote_pk", 0)] + [
                        ("vote_pk", target)
                        for target in targets
                    ],
                )
                self.assertEqual(self.env.vote_pk_players, [5, 6, 0, 2])

    def test_vote_pk_parser_and_environment_reject_self_vote(self):
        self.env.phase = "vote_pk"
        self.env.day = 1
        self.env.day_or_night = "day"
        self.env.current_act_idx = 2
        self.env.alive = [1, 0, 1, 0, 0, 1, 1]
        self.env.vote_pk_players = [5, 6, 0, 2]
        observation = self.env.get_observation_for(3)
        agent = LLMAgent()
        agent.get_valid_actions_str(observation["valid_action"])
        valid_actions = list(agent.nlp_action_to_env_action)

        self.assertIsNone(
            agent.parse_vote_action(
                "{'投票': '3'}",
                observation,
                valid_actions,
            )
        )
        self.assertEqual(
            agent.parse_vote_action(
                "{'投票': '6'}",
                observation,
                valid_actions,
            ),
            "{'投票': '6'}",
        )
        with self.assertRaisesRegex(ValueError, "invalid PK vote action"):
            self.env.step(("vote_pk", 3))

        self.env.vote_queue = [0]
        self.env.step(("vote_pk", 6))
        self.assertEqual(self.env.game_log[-1].event, "vote_pk")
        self.assertEqual(self.env.game_log[-1].source, 2)
        self.assertEqual(self.env.game_log[-1].target, 5)

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
        self.env.current_act_idx = 1
        self.env.alive = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        self.env.game_log.extend(
            [
                Log(
                    viewer=list(range(7)),
                    source=-1,
                    target=[3],
                    content={"dead_list": [3]},
                    day=1,
                    time="第1天夜晚",
                    event="end_night",
                ),
                Log(
                    viewer=list(range(7)),
                    source=-1,
                    target=2,
                    content={"vote_outcome": 2, "expelled": 2},
                    day=1,
                    time="第1天白天",
                    event="end_vote",
                ),
                Log(
                    viewer=list(range(7)),
                    source=4,
                    target=list(range(7)),
                    content={
                        "speech_content": (
                            "player1 已死亡，player3 仍存活，应该投 player4。"
                        ),
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
            for player_id in (1, 2, 3, 5)
        ]
        public_state = observations[0]["authoritative_public_state"]
        self.assertTrue(
            all(
                observation["authoritative_public_state"] == public_state
                for observation in observations[1:]
            )
        )
        self.assertEqual(public_state["alive_players"], [1, 2, 5, 6, 7])
        self.assertEqual(
            public_state["last_night_result"],
            {"day": 1, "dead_players": [4]},
        )
        self.assertEqual(
            public_state["prior_exiles"],
            [{"player_id": 3, "day": 1}],
        )
        self.assertEqual(
            public_state["suggestible_exile_targets"],
            [1, 5, 6, 7],
        )

        prompt = LLMAgent(
            gameplay_prompt_profile="strict_classic7"
        ).format_observation(
            observations[0],
            suggestible_player_ids=(1, 5, 6, 7),
        )
        before_private, remainder = prompt.split(
            "【你合法知道的私有信息】",
            1,
        )
        authoritative = before_private.split(
            "【权威公共状态】",
            1,
        )[1]
        for label in (
            "【当前阶段】",
            "【昨夜公开结果】",
            "【此前放逐】",
            "【当前存活】",
            "【当前可公开建议放逐】",
        ):
            self.assertEqual(prompt.count(label), 1)
        self.assertIn("【昨夜公开结果】player4 昨夜死亡", authoritative)
        self.assertIn("player3 已于第1天放逐", authoritative)
        self.assertIn("【当前存活】player1, player2, player5, player6, player7", authoritative)
        self.assertIn("【当前可公开建议放逐】player1, player5, player6, player7", authoritative)
        self.assertNotIn("player1 已死亡", authoritative)
        self.assertNotIn("player3 仍存活", authoritative)
        self.assertNotIn("player0", prompt)
        self.assertNotIn("狼人", authoritative)
        self.assertNotIn("查验", authoritative)
        self.assertIn("【所有玩家此前的公开主张】", remainder)
        self.assertIn(
            "player5：player1 已死亡，player3 仍存活，应该投 player4。",
            remainder,
        )
        self.assertIn("可能是真话、谎言、误解或策略性表达", remainder)

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
        strict_rules = build_strict_classic7_speech_plan_prompt(
            seer_observation,
            suggestible_player_ids=tuple(
                seer_observation["authoritative_public_state"][
                    "suggestible_exile_targets"
                ]
            ),
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
            build_strict_classic7_speech_plan_prompt(
                seer_observation,
                suggestible_player_ids=tuple(
                    seer_observation["authoritative_public_state"][
                        "suggestible_exile_targets"
                    ]
                ),
            ),
        )


if __name__ == "__main__":
    unittest.main()
