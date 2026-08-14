import unittest

from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
from werewolf.speech.speech_perceiver import SpeechActionValidationError


ROLES = [
    "Werewolf", "Werewolf", "Seer", "Witch",
    "Villager", "Villager", "Villager",
]


class RecordingPerceiver:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def parse_strict(self, speaker, speech, day, phase):
        self.calls.append({
            "speaker": speaker,
            "speech": speech,
            "day": day,
            "phase": phase,
        })
        return self.result


class RaisingPerceiver:
    def parse_strict(self, speaker, speech, day, phase):
        raise RuntimeError("parse failed")


class NonListPerceiver:
    def parse_strict(self, speaker, speech, day, phase):
        return {"action": "support", "object": "player2"}


class SpeechPerceiverEnvironmentTest(unittest.TestCase):
    def make_env(self, perceiver):
        env = WerewolfTextEnvV0(
            log_save_path=None,
            speech_perceiver=perceiver,
        )
        env.reset(roles=ROLES)
        return env

    @staticmethod
    def set_speech_state(env, phase="speech", current_act_idx=1):
        env.phase = phase
        env.day = 2
        env.day_or_night = "day"
        env.current_act_idx = current_act_idx
        env.alive = [1] * env.n_player
        env.speech_queue = [(current_act_idx + 1) % env.n_player]
        env.vote_queue = []

    def test_exact_direct_speech_is_parsed_then_committed_with_actions(self):
        actions = [["player2", "point_as_werewolf", "player3"]]
        perceiver = RecordingPerceiver(actions)
        env = self.make_env(perceiver)
        self.set_speech_state(env)
        raw_speech = "我认为3号是狼人。"

        observation, _, done, _ = env.step(("speech", raw_speech))

        self.assertFalse(done)
        self.assertEqual(perceiver.calls, [{
            "speaker": 2,
            "speech": raw_speech,
            "day": 2,
            "phase": "speech",
        }])
        speech_log = next(
            log for log in reversed(env.game_log) if log.event == "speech"
        )
        self.assertEqual(speech_log.content, {
            "speech_content": raw_speech,
            "sp_actions": actions,
        })
        event = next(
            item for item in reversed(env.public_events)
            if item["event_type"] == "public_speech"
        )
        self.assertEqual(event["raw_text"], raw_speech)
        self.assertEqual(event["sp_actions"], actions)
        observed_log = next(
            log for log in reversed(observation["game_log"])
            if log.event == "speech"
        )
        self.assertEqual(observed_log.content["sp_actions"], actions)

    def test_valid_empty_actions_commit_normally(self):
        env = self.make_env(RecordingPerceiver([]))
        self.set_speech_state(env, current_act_idx=0)

        env.step(("speech", "这一轮我暂时没有明确判断，先听后面的发言。"))

        speech_log = next(
            log for log in reversed(env.game_log) if log.event == "speech"
        )
        self.assertEqual(speech_log.content["sp_actions"], [])
        event = next(
            item for item in reversed(env.public_events)
            if item["event_type"] == "public_speech"
        )
        self.assertEqual(event["sp_actions"], [])

    def test_wrong_subject_is_rejected_before_any_commit(self):
        env = self.make_env(RecordingPerceiver([
            ["player7", "oppose", "player3"],
        ]))
        self.set_speech_state(env, current_act_idx=1)
        game_log_before = list(env.game_log)
        public_events_before = list(env.public_events)

        with self.assertRaises(SpeechActionValidationError):
            env.step(("speech", "我反对3号。"))

        self.assertEqual(env.game_log, game_log_before)
        self.assertEqual(env.public_events, public_events_before)

    def test_every_parsed_action_must_use_actual_subject(self):
        env = self.make_env(RecordingPerceiver([
            ["player2", "support", "player4"],
            ["player5", "oppose", "player3"],
        ]))
        self.set_speech_state(env, current_act_idx=1)

        with self.assertRaises(SpeechActionValidationError):
            env.step(("speech", "我支持4号，但反对3号。"))

        self.assertFalse(any(log.event == "speech" for log in env.game_log))
        self.assertFalse(any(
            item["event_type"] == "public_speech"
            for item in env.public_events
        ))

    def test_parser_failure_is_explicit_and_not_committed_as_empty(self):
        env = self.make_env(RaisingPerceiver())
        self.set_speech_state(env, current_act_idx=0)
        game_log_before = list(env.game_log)
        public_events_before = list(env.public_events)

        with self.assertRaisesRegex(RuntimeError, "parse failed"):
            env.step(("speech", "发言"))

        self.assertEqual(env.game_log, game_log_before)
        self.assertEqual(env.public_events, public_events_before)

    def test_non_list_protocol_failure_is_explicit_and_not_committed(self):
        env = self.make_env(NonListPerceiver())
        self.set_speech_state(env, current_act_idx=0)
        game_log_before = list(env.game_log)
        public_events_before = list(env.public_events)

        with self.assertRaises(SpeechActionValidationError):
            env.step(("speech", "发言"))

        self.assertEqual(env.game_log, game_log_before)
        self.assertEqual(env.public_events, public_events_before)

    def test_speech_pk_uses_the_same_strict_direct_path(self):
        actions = [["player4", "oppose", "player2"]]
        perceiver = RecordingPerceiver(actions)
        env = self.make_env(perceiver)
        self.set_speech_state(env, phase="speech_pk", current_act_idx=3)

        env.step(("speech_pk", "我不信2号。"))

        self.assertEqual(perceiver.calls[0]["speaker"], 4)
        self.assertEqual(perceiver.calls[0]["phase"], "speech_pk")
        speech_log = next(
            log for log in reversed(env.game_log) if log.event == "speech_pk"
        )
        self.assertEqual(speech_log.content["sp_actions"], actions)


if __name__ == "__main__":
    unittest.main()
