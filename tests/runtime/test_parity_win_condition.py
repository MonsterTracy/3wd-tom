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


@pytest.mark.parametrize(
    ("alive_players", "expected_done", "expected_wolf_result"),
    [
        ({3, 4, 5}, True, -1),
        ({1, 2, 3, 4}, True, 1),
        ({1, 2, 3}, True, 1),
        ({1, 5, 6}, False, None),
        ({1, 3, 4}, False, None),
    ],
)
def test_parity_win_condition(
    alive_players,
    expected_done,
    expected_wolf_result,
):
    env = WerewolfTextEnvV0(log_save_path=None, random_seed=17)
    env.reset(roles=ROLES)
    env.alive = [
        1.0 if player_id in alive_players else 0.0
        for player_id in range(1, 8)
    ]

    _, done, info = env.is_done()

    assert done is expected_done
    if expected_wolf_result is None:
        assert info == {}
    else:
        assert info == {"Werewolf": expected_wolf_result}


def test_environment_seed_must_be_an_integer_or_none():
    with pytest.raises(TypeError, match="random_seed"):
        WerewolfTextEnvV0(log_save_path=None, random_seed=True)
