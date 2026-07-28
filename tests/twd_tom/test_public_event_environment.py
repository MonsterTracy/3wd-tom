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
    def parse(self, **_kwargs):
        return [
            ["player1", "support", "player2"],
            ["player1", "oppose", "player3"],
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
    env.step(("check", 0))
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
            ["player1", "support", "player2"],
            ["player1", "oppose", "player3"],
        ],
    }
    assert len(
        [
            event
            for event in env.public_events
            if event["event_type"] == "public_speech"
        ]
    ) == 1


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
