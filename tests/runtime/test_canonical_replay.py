import json
import subprocess
import sys

from script.twd_tom.replay_canonical_trajectory import (
    replay_canonical_trajectory,
)
from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
from werewolf.trajectory import CanonicalGameInteractionTrajectoryRecorder


ROLES = [
    "Werewolf",
    "Werewolf",
    "Seer",
    "Witch",
    "Villager",
    "Villager",
    "Villager",
]
ENV_CONFIG = {
    "n_player": 7,
    "n_role": 4,
    "n_werewolf": 2,
    "n_seer": 1,
    "n_guard": 0,
    "n_witch": 1,
    "n_hunter": 0,
    "n_villager": 3,
}


def _players():
    return [
        {
            "player_id": player_id,
            "role": role,
            "profile_name": f"profile-{player_id}",
            "backend_id": "test-backend",
            "model_name": "test-model",
        }
        for player_id, role in enumerate(ROLES, start=1)
    ]


def _choose_action(env, observation, step_idx):
    if env.phase in {"speech", "speech_pk"}:
        return env.phase, f"deterministic speech {step_idx}"

    valid_actions = observation["valid_action"]
    if env.phase == "skill_wolf":
        return next(action for action in reversed(valid_actions) if action[1] != 0)
    if env.phase == "skill_seer":
        return next(action for action in valid_actions if action[1] != 0)
    if env.phase == "skill_witch":
        return next(action for action in valid_actions if action[0] == "witch_pass")
    if env.phase in {"vote", "vote_pk"}:
        wolf_targets = [
            player_id
            for player_id, role in enumerate(ROLES, start=1)
            if role == "Werewolf"
            and env.alive[player_id - 1] == 1
            and player_id != observation["current_act_idx"]
        ]
        if wolf_targets:
            desired = (env.phase, wolf_targets[0])
            if desired in valid_actions:
                return desired
        return next(action for action in valid_actions if action[1] != 0)
    raise AssertionError(f"unexpected phase: {env.phase}")


def _record_complete_game(tmp_path, *, seed=23):
    trajectory_path = tmp_path / "trajectory.json"
    observer_views_path = tmp_path / "observer_views.json"
    env = WerewolfTextEnvV0(
        **ENV_CONFIG,
        log_save_path=None,
        random_seed=seed,
    )
    recorder = CanonicalGameInteractionTrajectoryRecorder(
        trajectory_path,
        observer_views_path,
        game_id="replay-game-001",
        run_id="replay-run-001",
        source_commit="1" * 40,
        environment_seed=seed,
        runtime_config={"env_config": ENV_CONFIG},
        players=_players(),
    )

    observation = env.reset(roles=ROLES)
    recorder.start(env, roles=ROLES)
    done = False
    info = {}
    step_idx = 0
    while not done:
        if step_idx >= 200:
            raise AssertionError("deterministic test game did not terminate")
        speech_kind = env.phase if env.phase in {"speech", "speech_pk"} else None
        recorder.before_agent_act(
            env,
            step_idx=step_idx,
            acting_player_id=observation["current_act_idx"],
            delivered_observation=observation,
            speech_kind=speech_kind,
        )
        action = _choose_action(env, observation, step_idx)
        recorder.after_agent_act(action)
        observation, _, done, info = env.step(action)
        recorder.after_env_step(
            env,
            observation_after=observation,
            terminal_after=done,
        )
        step_idx += 1

    winner = "Werewolf" if info["Werewolf"] == 1 else "Villager"
    recorder.complete(env, winner=winner)
    return trajectory_path, observer_views_path


def test_canonical_trajectory_replays_without_model_calls(tmp_path):
    trajectory_path, observer_views_path = _record_complete_game(tmp_path)

    result = replay_canonical_trajectory(
        trajectory_path,
        observer_views_path,
    )

    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    assert result["status"] == "MATCH"
    assert result["winner"] == trajectory["termination"]["winner"]
    assert result["transition_count"] == len(trajectory["transitions"])
    assert result["observer_view_boundary_count"] > 0


def test_replay_rejects_tampered_observer_views(tmp_path):
    trajectory_path, observer_views_path = _record_complete_game(tmp_path)
    observer_views = json.loads(observer_views_path.read_text(encoding="utf-8"))
    observer_views["boundaries"][0]["speaker_id"] = 7
    observer_views_path.write_text(
        json.dumps(observer_views),
        encoding="utf-8",
    )

    try:
        replay_canonical_trajectory(trajectory_path, observer_views_path)
    except ValueError as exc:
        assert "artifact digest" in str(exc)
    else:
        raise AssertionError("tampered observer views were accepted")


def test_replay_cli_prints_match_result(tmp_path):
    trajectory_path, observer_views_path = _record_complete_game(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "script.twd_tom.replay_canonical_trajectory",
            str(trajectory_path),
            "--observer-views",
            str(observer_views_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["status"] == "MATCH"
