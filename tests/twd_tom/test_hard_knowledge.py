import pytest

from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0


ROLES = [
    "Werewolf",
    "Werewolf",
    "Seer",
    "Witch",
    "Villager",
    "Villager",
    "Villager",
]


def _env():
    env = WerewolfTextEnvV0(log_save_path=None)
    env.reset(roles=ROLES)
    return env


def test_villager_and_werewolf_hard_knowledge():
    env = _env()
    assert env.get_twd_tom_hard_knowledge_for(5) == (
        [],
        ["player5"],
    )
    assert env.get_twd_tom_hard_knowledge_for(1) == (
        ["player1", "player2"],
        ["player3", "player4", "player5", "player6", "player7"],
    )


@pytest.mark.parametrize(
    ("checked_player", "expected_wolves", "expected_non_wolves"),
    [
        (1, ["player1"], ["player3"]),
        (5, [], ["player3", "player5"]),
    ],
)
def test_seer_checks_enter_the_correct_hard_set(
    checked_player,
    expected_wolves,
    expected_non_wolves,
):
    env = _env()
    env.step(("kill", 5))
    env.step(("kill", 5))
    env.step(("check", checked_player))
    known_wolves, known_non_wolves = (
        env.get_twd_tom_hard_knowledge_for(3)
    )
    assert known_wolves == expected_wolves
    assert known_non_wolves == expected_non_wolves


def test_witch_keeps_legal_knife_target_and_sees_no_later_target_after_heal():
    env = _env()
    env.step(("kill", 5))
    env.step(("kill", 5))
    assert env.get_twd_tom_hard_knowledge_for(4) == (
        [],
        ["player4", "player5"],
    )
    env.step(("check", 1))
    env.step(("witch_heal", 5))

    env.phase = "skill_wolf"
    env.day_or_night = "night"
    env.current_act_idx = env.WOLF_IDX[0]
    env.step(("kill", 6))
    env.step(("kill", 6))
    known_wolves, known_non_wolves = env.get_twd_tom_hard_knowledge_for(4)
    assert known_wolves == []
    assert known_non_wolves == ["player4", "player5"]


def test_witch_poison_target_does_not_enter_hard_knowledge():
    env = _env()
    env.step(("kill", 5))
    env.step(("kill", 5))
    env.step(("check", 1))
    before = env.get_twd_tom_hard_knowledge_for(4)
    env.step(("witch_poison", 6))

    known_wolves, known_non_wolves = env.get_twd_tom_hard_knowledge_for(4)
    assert (known_wolves, known_non_wolves) == before
    assert "player6" not in known_wolves
    assert "player6" not in known_non_wolves


def test_witch_knife_target_is_isolated_from_other_observers():
    env = _env()
    env.step(("kill", 7))
    env.step(("kill", 7))

    assert env.get_twd_tom_hard_knowledge_for(4) == (
        [],
        ["player4", "player7"],
    )
    assert env.get_twd_tom_hard_knowledge_for(3) == (
        [],
        ["player3"],
    )
    assert env.get_twd_tom_hard_knowledge_for(5) == (
        [],
        ["player5"],
    )

    witch_observation = env.get_observation_for(4)
    seer_observation = env.get_observation_for(3)
    villager_observation = env.get_observation_for(5)
    assert any(
        log.event == "kill_decision"
        for log in witch_observation["game_log"]
    )
    assert all(
        log.event != "kill_decision"
        for observation in (seer_observation, villager_observation)
        for log in observation["game_log"]
    )
    assert all(
        log.event not in {"speech", "speech_pk"}
        for log in seer_observation["game_log"]
    )
    assert all(
        log.event not in {"speech", "speech_pk"}
        for log in villager_observation["game_log"]
    )

    witch_observation["game_log"].clear()
    assert env.get_observation_for(4)["game_log"]
    fresh_seer_observation = env.get_observation_for(3)
    assert [
        log.event for log in fresh_seer_observation["game_log"]
    ] == [
        log.event for log in seer_observation["game_log"]
    ]
    assert all(
        fresh is not original
        for fresh, original in zip(
            fresh_seer_observation["game_log"],
            seer_observation["game_log"],
        )
    )


def test_death_log_does_not_add_identity_knowledge():
    env = _env()
    env.alive[4] = 0
    env.game_log.append(
        type(env.game_log[0])(
            viewer=list(range(7)),
            source=-1,
            target=[4],
            content={"dead_list": [4]},
            day=1,
            time="day",
            event="end_night",
        )
    )
    assert env.get_twd_tom_hard_knowledge_for(6) == (
        [],
        ["player6"],
    )


def test_wolves_cannot_target_themselves_or_teammate():
    env = _env()
    valid = env.get_observation_for(1)["valid_action"]
    assert ("kill", 1) not in valid
    assert ("kill", 2) not in valid
    with pytest.raises(AssertionError):
        env.step(("kill", 2))
