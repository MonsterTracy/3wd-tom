from copy import deepcopy
import unittest

from werewolf.envs.werewolf_text_env_v0 import (
    WerewolfTextEnvV0,
)


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

    def test_invalid_player_id_is_rejected(self):
        with self.assertRaises(ValueError):
            self.env.get_observation_for(0)

        with self.assertRaises(ValueError):
            self.env.get_observation_for(8)

        with self.assertRaises(TypeError):
            self.env.get_observation_for("3")

        with self.assertRaises(TypeError):
            self.env.get_observation_for(True)


if __name__ == "__main__":
    unittest.main()