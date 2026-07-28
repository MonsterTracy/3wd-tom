import random
import unittest

from werewolf.agents.base_agent import RandomAgent
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


class RecordingPerceiver:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def parse(
        self,
        speaker,
        speech,
        day,
        phase,
        context=None,
    ):
        self.calls.append(
            {
                "speaker": speaker,
                "speech": speech,
                "day": day,
                "phase": phase,
                "context": context,
            }
        )

        return self.result


class RaisingPerceiver:
    def parse(
        self,
        speaker,
        speech,
        day,
        phase,
        context=None,
    ):
        raise RuntimeError("parse failed")


class NonListPerceiver:
    def parse(
        self,
        speaker,
        speech,
        day,
        phase,
        context=None,
    ):
        return {
            "action": "support",
            "object": "player2",
        }


class SpeechPerceiverEnvironmentTest(unittest.TestCase):
    def make_env(self, perceiver=None):
        kwargs = {
            "log_save_path": None,
        }

        if perceiver is not None:
            kwargs["speech_perceiver"] = perceiver

        env = WerewolfTextEnvV0(**kwargs)
        env.reset(roles=ROLES)

        return env

    @staticmethod
    def set_speech_state(
        env,
        phase,
        current_act_idx,
    ):
        env.phase = phase
        env.day = 2
        env.day_or_night = "day"
        env.current_act_idx = current_act_idx
        env.alive = [
            1
            for _ in range(env.n_player)
        ]
        env.speech_queue = [
            (current_act_idx + 1)
            % env.n_player
        ]
        env.vote_queue = []

    def test_visible_observation_contains_sp_actions(self):
        actions = [
            [
                "player2",
                "point_as_werewolf",
                "player3",
            ]
        ]

        perceiver = RecordingPerceiver(actions)
        env = self.make_env(perceiver)

        self.set_speech_state(
            env,
            phase="speech",
            current_act_idx=1,
        )

        observation, _, done, _ = env.step(
            (
                "speech",
                "我认为3号是狼人",
            )
        )

        self.assertFalse(done)
        self.assertEqual(
            observation["current_act_idx"],
            3,
        )

        self.assertEqual(
            perceiver.calls,
            [
                {
                    "speaker": 2,
                    "speech": "我认为3号是狼人",
                    "day": 2,
                    "phase": "speech",
                    "context": None,
                }
            ],
        )

        speech_log = next(
            log
            for log in reversed(env.game_log)
            if log.event == "speech"
        )

        self.assertEqual(
            speech_log.content,
            {
                "speech_content": "我认为3号是狼人",
                "sp_actions": actions,
            },
        )

        observed_log = next(
            log
            for log in reversed(
                observation["game_log"]
            )
            if log.event == "speech"
        )

        self.assertEqual(
            observed_log.source,
            2,
        )
        self.assertEqual(
            observed_log.content["sp_actions"],
            actions,
        )

    def test_speech_pk_log_contains_sp_actions(self):
        actions = [
            [
                "player4",
                "oppose",
                "player2",
            ]
        ]

        perceiver = RecordingPerceiver(actions)
        env = self.make_env(perceiver)

        self.set_speech_state(
            env,
            phase="speech_pk",
            current_act_idx=3,
        )

        env.step(
            (
                "speech_pk",
                "我不信2号",
            )
        )

        self.assertEqual(
            perceiver.calls[0]["speaker"],
            4,
        )
        self.assertEqual(
            perceiver.calls[0]["phase"],
            "speech_pk",
        )

        speech_log = next(
            log
            for log in reversed(env.game_log)
            if log.event == "speech_pk"
        )

        self.assertEqual(
            speech_log.content["sp_actions"],
            actions,
        )

    def test_parser_exception_does_not_interrupt_speech(self):
        env = self.make_env(
            RaisingPerceiver()
        )

        self.set_speech_state(
            env,
            phase="speech",
            current_act_idx=0,
        )

        env.step(
            (
                "speech",
                "发言",
            )
        )

        speech_log = next(
            log
            for log in reversed(env.game_log)
            if log.event == "speech"
        )

        self.assertEqual(
            speech_log.content["sp_actions"],
            [],
        )

    def test_non_list_result_becomes_empty_list(self):
        env = self.make_env(
            NonListPerceiver()
        )

        self.set_speech_state(
            env,
            phase="speech",
            current_act_idx=0,
        )

        env.step(
            (
                "speech",
                "发言",
            )
        )

        speech_log = next(
            log
            for log in reversed(env.game_log)
            if log.event == "speech"
        )

        self.assertEqual(
            speech_log.content["sp_actions"],
            [],
        )

    def test_random_game_logs_sp_actions_lists(self):
        random.seed(7)

        env = self.make_env()
        agent = RandomAgent()

        observation = env.get_observation()
        done = False

        for _ in range(500):
            action = agent.act(observation)
            observation, _, done, _ = env.step(
                action
            )

            if done:
                break

        self.assertTrue(done)

        speech_logs = [
            log
            for log in env.game_log
            if log.event in (
                "speech",
                "speech_pk",
            )
        ]

        self.assertGreater(
            len(speech_logs),
            0,
        )

        for log in speech_logs:
            self.assertIn(
                "sp_actions",
                log.content,
            )
            self.assertIsInstance(
                log.content["sp_actions"],
                list,
            )


if __name__ == "__main__":
    unittest.main()