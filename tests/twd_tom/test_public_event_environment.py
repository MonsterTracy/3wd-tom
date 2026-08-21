import pytest

from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
from werewolf.models.twd_tom.public_events import normalize_public_events


ROLES = [
    "Werewolf",
    "Werewolf",
    "Seer",
    "Witch",
    "Villager",
    "Villager",
    "Villager",
]


class Parser:
    def __init__(self):
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        speaker = f"player{kwargs['speaker']}"
        return [
            [speaker, "support", "player2"],
            [speaker, "oppose", "player3"],
        ]


def _env(parser=None):
    env = WerewolfTextEnvV0(
        log_save_path=None,
        speech_perceiver=Parser() if parser is None else parser,
    )
    env.reset(roles=ROLES)
    return env


def _finish_first_night(env, target):
    env.step(("kill", target))
    env.step(("kill", target))
    env.step(("check", 1))
    env.step(("witch_pass", 0))


def test_first_snapshot_has_death_phase_and_turn_before_speech():
    env = _env()
    _finish_first_night(env, 5)
    speaker = f"player{env.current_act_idx + 1}"
    assert [event["event_type"] for event in env.public_events] == [
        "death_announcement",
        "phase_change",
        "turn_start",
    ]
    assert env.public_events[0]["dead_players"] == ["player5"]
    assert env.public_events[-1]["speaker"] == speaker
    normalize_public_events(env.public_events)

    before = list(env.public_events)
    env.step(("speech", "final public text"))
    assert env.public_events[: len(before)] == before
    speech = env.public_events[len(before)]
    assert speech == {
        "event_idx": len(before),
        "event_type": "public_speech",
        "speaker": speaker,
        "raw_text": "final public text",
        "sp_actions": [
            [speaker, "support", "player2"],
            [speaker, "oppose", "player3"],
        ],
    }
    assert len(
        [
            event
            for event in env.public_events
            if event["event_type"] == "public_speech"
        ]
    ) == 1


def test_structured_speech_bypasses_parser_and_commits_exact_semantics():
    class ForbiddenParser:
        def parse(self, **_kwargs):
            raise AssertionError("structured speech must bypass SpeechPerceiver")

    env = _env(ForbiddenParser())
    _finish_first_night(env, 5)
    speaker = f"player{env.current_act_idx + 1}"
    payload = {
        "raw_text": "deterministic speech",
        "sp_actions": [
            [speaker, "oppose", "player1"],
            [speaker, "vote_intent", "player3"],
        ],
    }

    env.step(("speech", payload))

    public_speech = next(
        event for event in env.public_events
        if event["event_type"] == "public_speech"
    )
    speech_log = next(
        log for log in env.game_log if log.event == "speech"
    )
    assert public_speech["raw_text"] == payload["raw_text"]
    assert public_speech["sp_actions"] == payload["sp_actions"]
    assert speech_log.content == {
        "speech_content": payload["raw_text"],
        "sp_actions": public_speech["sp_actions"],
    }


@pytest.mark.parametrize("action_name", ("abstain_intent", "no_commitment"))
def test_structured_targetless_speech_commits_null(action_name):
    env = _env()
    _finish_first_night(env, 5)
    speaker = f"player{env.current_act_idx + 1}"
    env.step(("speech", {
        "raw_text": "targetless",
        "sp_actions": [[speaker, action_name, None]],
    }))
    assert env.public_events[-2]["sp_actions"] == [
        [speaker, action_name, None]
    ]


def test_structured_speech_rejects_wrong_subject_and_extra_fields():
    env = _env()
    _finish_first_night(env, 5)
    before = list(env.public_events)
    speaker = f"player{env.current_act_idx + 1}"
    wrong_speaker = "player1" if speaker != "player1" else "player2"

    with pytest.raises(ValueError, match="subject must equal event speaker"):
        env.step(("speech", {
            "raw_text": "x",
            "sp_actions": [[wrong_speaker, "oppose", "player2"]],
        }))
    assert env.public_events == before

    with pytest.raises(ValueError, match="field set mismatch"):
        env.step(("speech", {
            "raw_text": "x",
            "sp_actions": [],
            "extra": True,
        }))
    assert env.public_events == before


def test_empty_death_and_empty_speech_remain_explicit_events():
    class EmptyParser:
        def parse(self, **_kwargs):
            return []

    env = _env(EmptyParser())
    _finish_first_night(env, 0)
    assert env.public_events[0] == {
        "event_idx": 0,
        "event_type": "death_announcement",
        "dead_players": [],
    }
    env.step(("speech", ""))
    speech = next(
        event
        for event in env.public_events
        if event["event_type"] == "public_speech"
    )
    assert speech["raw_text"] == ""
    assert speech["sp_actions"] == []


def test_public_vote_records_canonical_ballots_exile_and_night_transition():
    env = _env()
    env.day = 1
    env.day_or_night = "day"
    env.phase = "vote"
    phase_id = env.get_phase(env.day, env.day_or_night, env.phase)
    env.vote_target = [
        {phase_id: 1},
        {phase_id: 1},
        {phase_id: 1},
        {phase_id: -1},
        {phase_id: 1},
        {phase_id: 1},
        {phase_id: 1},
    ]
    env.end_vote()
    vote, exile, night = env.public_events
    assert vote["event_type"] == "vote_result"
    assert [item["voter"] for item in vote["votes"]] == [
        f"player{index}" for index in range(1, 8)
    ]
    assert vote["votes"][3]["target"] is None
    assert exile == {
        "event_idx": 1,
        "event_type": "exile_result",
        "exiled_players": ["player2"],
    }
    assert night == {
        "event_idx": 2,
        "event_type": "phase_change",
        "phase": "1_night_skill_wolf",
    }
    assert all(
        "role" not in event and "death_reason" not in event
        for event in env.public_events
    )


def test_all_abstention_has_explicit_empty_exile():
    env = _env()
    env.day = 1
    env.day_or_night = "day"
    env.phase = "vote"
    phase_id = env.get_phase(env.day, env.day_or_night, env.phase)
    env.vote_target = [{phase_id: -1} for _ in range(7)]
    env.end_vote()
    assert env.public_events[1] == {
        "event_idx": 1,
        "event_type": "exile_result",
        "exiled_players": [],
    }
