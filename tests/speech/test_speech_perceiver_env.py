import random
import unittest

from werewolf.agents.base_agent import RandomAgent
from werewolf.envs.werewolf_text_env_v0 import (
    WerewolfTextEnvV0,
)
from werewolf.speech.speech_perceiver import (
    PlannedPublicSpeech,
    PublicSpeechSemanticAlignmentError,
    validate_public_speech_semantic_alignment,
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

    def test_semantic_alignment_uses_deduplicated_orderless_propositions(self):
        planned = (
            ("oppose", 3),
            ("vote_intent", 4),
            ("oppose", 3),
        )

        for parsed in (
            [
                ["player2", "oppose", "player3"],
                ["player2", "vote_intent", "player4"],
            ],
            [
                ["player2", "vote_intent", "player4"],
                ["player2", "oppose", "player3"],
            ],
            [
                ["player2", "oppose", "player3"],
                ["player2", "oppose", "player3"],
                ["player2", "vote_intent", "player4"],
            ],
        ):
            with self.subTest(parsed=parsed):
                validate_public_speech_semantic_alignment(
                    planned,
                    parsed,
                    actual_speaker=2,
                )

    def test_semantic_alignment_requires_actual_speaker_for_every_action(self):
        planned = (("oppose", 3), ("vote_intent", 4))
        correct = [
            ["player2", "oppose", "player3"],
            ["player2", "vote_intent", "player4"],
        ]
        validate_public_speech_semantic_alignment(
            planned,
            correct,
            actual_speaker=2,
        )

        for parsed in (
            [["player7", "oppose", "player3"]],
            [
                ["player2", "oppose", "player3"],
                ["player7", "vote_intent", "player4"],
            ],
        ):
            accepted = (
                (("oppose", 3),)
                if len(parsed) == 1
                else planned
            )
            with self.subTest(parsed=parsed), self.assertRaisesRegex(
                PublicSpeechSemanticAlignmentError,
                "subject",
            ):
                validate_public_speech_semantic_alignment(
                    accepted,
                    parsed,
                    actual_speaker=2,
                )

    def test_semantic_alignment_rejects_every_set_mismatch(self):
        planned = (("oppose", 3),)
        cases = (
            ([], "missing"),
            (
                [
                    ["player2", "oppose", "player3"],
                    ["player2", "vote_intent", "player3"],
                ],
                "extra",
            ),
            (["player2", "support", "player3"], "wrong predicate"),
            (["player2", "oppose", "player4"], "wrong target"),
        )

        for parsed, label in cases:
            if parsed and isinstance(parsed[0], str):
                parsed = [parsed]
            with self.subTest(label=label), self.assertRaises(
                PublicSpeechSemanticAlignmentError
            ):
                validate_public_speech_semantic_alignment(
                    planned,
                    parsed,
                    actual_speaker=2,
                )

    def test_matching_planned_speech_commits_only_perceiver_actions(self):
        parsed_actions = [
            ["player2", "vote_intent", "player4"],
            ["player2", "oppose", "player3"],
        ]
        env = self.make_env(RecordingPerceiver(parsed_actions))
        self.set_speech_state(env, phase="speech", current_act_idx=1)
        speech = PlannedPublicSpeech(
            "我不信3号，今天投4号。",
            (("oppose", 3), ("vote_intent", 4)),
        )

        env.step(("speech", speech))

        event = next(
            event
            for event in reversed(env.public_events)
            if event["event_type"] == "public_speech"
        )
        self.assertEqual(event["raw_text"], str(speech))
        self.assertIs(type(event["raw_text"]), str)
        self.assertEqual(event["sp_actions"], parsed_actions)
        self.assertNotIn("accepted_public_actions", event)

    def test_semantic_mismatch_fails_before_any_speech_commit(self):
        cases = (
            (
                (("oppose", 3),),
                [],
                "missing renderer proposition",
            ),
            (
                (("oppose", 3),),
                [
                    ["player2", "oppose", "player3"],
                    ["player2", "vote_intent", "player3"],
                ],
                "same-target extra renderer semantic",
            ),
            (
                (),
                [["player2", "vote_intent", "player4"]],
                "inferred target parser output",
            ),
        )

        for planned, parsed, label in cases:
            env = self.make_env(RecordingPerceiver(parsed))
            self.set_speech_state(env, phase="speech", current_act_idx=1)
            before_events = list(env.public_events)
            before_log_count = len(env.game_log)
            speech = PlannedPublicSpeech("公开发言", planned)

            with self.subTest(label=label), self.assertRaises(
                PublicSpeechSemanticAlignmentError
            ):
                env.step(("speech", speech))

            self.assertEqual(env.public_events, before_events)
            self.assertEqual(len(env.game_log), before_log_count)
            self.assertEqual(env.phase, "speech")
            self.assertEqual(env.current_act_idx, 1)

    def test_zero_plan_and_zero_parsed_actions_remains_legal(self):
        env = self.make_env(RecordingPerceiver([]))
        self.set_speech_state(env, phase="speech", current_act_idx=1)

        env.step(("speech", PlannedPublicSpeech("我继续听。", ())))

        event = next(
            event
            for event in reversed(env.public_events)
            if event["event_type"] == "public_speech"
        )
        self.assertEqual(event["sp_actions"], [])

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
