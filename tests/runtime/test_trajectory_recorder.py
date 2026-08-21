import inspect
import json
from copy import deepcopy

import pytest

from run_random import eval
from werewolf.helper.log_utils import Log
from werewolf.models.twd_tom.public_events import (
    normalize_public_events,
    public_event_digest,
)
from werewolf.trajectory import (
    OBSERVER_VIEW_PROVENANCE_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    POST_PUBLIC_SPEECH,
    PRE_PUBLIC_SPEECH,
    SIMULATOR_BASELINE,
    TRAJECTORY_SCHEMA_VERSION,
    CanonicalGameInteractionTrajectoryRecorder,
    canonical_digest,
    serialize_json_value,
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
STRICT_SPEECH_ACTION = (
    "speech",
    {
        "raw_text": "player3 looks suspicious",
        "sp_actions": [
            ["player1", "point_as_werewolf", "player3"],
            ["player1", "vote_intent", "player3"],
        ],
    },
)


def _players():
    return [
        {
            "player_id": player_id,
            "role": role,
            "profile_name": f"profile_{player_id}",
            "backend_id": f"backend_{player_id}",
            "model_name": f"model_{player_id}",
        }
        for player_id, role in enumerate(ROLES, start=1)
    ]


def _recorder(
    tmp_path,
    *,
    name="game",
    environment_seed=402,
    runtime_config=None,
):
    return CanonicalGameInteractionTrajectoryRecorder(
        tmp_path / f"{name}.trajectory.json",
        tmp_path / f"{name}.observer_views.json",
        game_id=name,
        run_id="run_001",
        source_commit="35142cb7e8f6175679904d66fa154b72584b8c4e",
        environment_seed=environment_seed,
        runtime_config=runtime_config or {
            "gameplay_max_tokens": 512,
            "api_key_env": "VLLM_API_KEY",
            "token_env": "TOKEN_ENV",
            "api_key": "API-SECRET",
            "token": "TOKEN-SECRET",
            "x-api-key": "X-API-SECRET",
            "x_auth_token": "AUTH-TOKEN-SECRET",
            "service_secret": "SERVICE-SECRET",
            "database_password": "PASSWORD-SECRET",
            "client_credentials": "CREDENTIAL-SECRET",
            "nested": {
                "authorization_headers": {"Authorization": "Bearer SECRET"},
                "safe": True,
            },
        },
        players=_players(),
    )


class ScriptedAgent:
    def __init__(self, actions=(), error=None):
        self.actions = list(actions)
        self.error = error
        self.observations = []
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def act(self, observation):
        self.observations.append(observation)
        if self.error is not None:
            raise self.error
        return self.actions.pop(0)


class TrajectoryEnvironment:
    def __init__(self, *, fail_step=False, start_phase="speech"):
        self.fail_step = fail_step
        self.start_phase = start_phase
        self.phase = start_phase
        self.alive = [1, 1, 0, 1, 1, 0, 1]
        self.public_events = []
        self.steps = 0
        self.order = []
        self.delivered_observation = None

    def _phase_id(self):
        phase = "vote" if self.phase == "end_game" else self.phase
        return f"1_day_{phase}"

    def _observation(self, observer_id, *, marker):
        return {
            "observer_id": observer_id,
            "current_act_idx": 1 if self.steps == 0 else 2,
            "identity": ROLES[observer_id - 1],
            "game_log": [
                Log(
                    viewer=(observer_id,),
                    source=1,
                    target=None,
                    content={"marker": marker, "ordered": ("甲", None)},
                    day=1,
                    time="第1天白天",
                    event=self.phase,
                )
            ],
            "phase": self._phase_id(),
            "valid_action": (
                (self.phase, 0),
            ),
        }

    def reset(self, roles):
        assert list(roles) == ROLES
        self.phase = self.start_phase
        self.steps = 0
        self.public_events = [
            {
                "event_idx": 0,
                "event_type": "phase_change",
                "phase": self._phase_id(),
            }
        ]
        if self.phase in {"speech", "speech_pk"}:
            self.public_events.append(
                {
                    "event_idx": 1,
                    "event_type": "turn_start",
                    "speaker": "player1",
                }
            )
        self.delivered_observation = self._observation(
            1,
            marker="exact-delivered-object",
        )
        return self.delivered_observation

    def get_observation_for(self, observer_id):
        self.order.append(f"view:{self.steps}:{observer_id}")
        return self._observation(
            observer_id,
            marker=f"materialized-after-{self.steps}-steps",
        )

    def step(self, action):
        if self.fail_step:
            self.public_events.append(
                {
                    "event_idx": len(self.public_events),
                    "event_type": "vote_result",
                    "votes": [],
                }
            )
            raise RuntimeError("step failed; api_key=DO-NOT-SAVE")

        if self.steps == 0:
            assert action == STRICT_SPEECH_ACTION
            self.public_events.append(
                {
                    "event_idx": len(self.public_events),
                    "event_type": "public_speech",
                    "speaker": "player1",
                    "raw_text": action[1]["raw_text"],
                    "sp_actions": deepcopy(action[1]["sp_actions"]),
                }
            )
            self.public_events.append(
                {
                    "event_idx": len(self.public_events),
                    "event_type": "phase_change",
                    "phase": "1_day_vote",
                }
            )
            self.steps = 1
            self.phase = "vote"
            return self._observation(2, marker="next-actor"), 0, False, {}

        assert action == ("vote", 1)
        self.public_events.extend(
            [
                {
                    "event_idx": len(self.public_events),
                    "event_type": "vote_result",
                    "votes": [{"voter": "player2", "target": "player1"}],
                },
                {
                    "event_idx": len(self.public_events) + 1,
                    "event_type": "exile_result",
                    "exiled_players": [],
                },
            ]
        )
        self.steps = 2
        self.phase = "end_game"
        return self._observation(2, marker="terminal"), 0, True, {"Werewolf": -1}


class PostSpeechCollector:
    def __init__(self, order):
        self.order = order

    def record(self, env, **kwargs):
        self.order.append("post-speech-collector")


def _agents(*, first=None, second=None):
    agents = [ScriptedAgent() for _ in range(7)]
    agents[0] = first or ScriptedAgent([STRICT_SPEECH_ACTION])
    agents[1] = second or ScriptedAgent([("vote", 1)])
    return agents


def _read_outputs(tmp_path, name="game"):
    trajectory = json.loads(
        (tmp_path / f"{name}.trajectory.json").read_text(encoding="utf-8")
    )
    provenance = json.loads(
        (tmp_path / f"{name}.observer_views.json").read_text(
            encoding="utf-8"
        )
    )
    return trajectory, provenance


def test_strict_serializer_preserves_tuples_and_logs_and_fails_closed():
    log = Log(
        viewer=(1, 3),
        source=1,
        target=None,
        content={"items": ("甲", 2, None)},
        day=1,
        time="夜晚",
        event="skill_seer",
    )
    serialized = serialize_json_value({"logs": (log,)})
    assert serialized == {
        "logs": [
            {
                "viewer": [1, 3],
                "source": 1,
                "target": None,
                "content": {"items": ["甲", 2, None]},
                "day": 1,
                "time": "夜晚",
                "event": "skill_seer",
            }
        ]
    }
    assert canonical_digest({"值": (1, None)}) == canonical_digest(
        {"值": [1, None]}
    )
    with pytest.raises(TypeError, match="unsupported canonical JSON"):
        serialize_json_value(object())


def test_complete_trajectory_and_speech_boundaries_are_canonical(tmp_path):
    env = TrajectoryEnvironment()
    agents = _agents()
    recorder = _recorder(tmp_path)
    collector = PostSpeechCollector(env.order)

    assert eval(
        env,
        agents,
        ROLES,
        tom_collector=collector,
        trajectory_recorder=recorder,
    ) == "Villager win"

    trajectory, provenance = _read_outputs(tmp_path)
    assert trajectory["schema_version"] == TRAJECTORY_SCHEMA_VERSION
    assert trajectory["observation_schema_version"] == OBSERVATION_SCHEMA_VERSION
    assert trajectory["simulator_baseline"] == SIMULATOR_BASELINE
    assert trajectory["environment_seed"] == 402
    assert trajectory["public_event_schema_version"] == (
        "classic7_public_event_sequence_v3"
    )
    assert [player["player_id"] for player in trajectory["players"]] == list(
        range(1, 8)
    )
    assert "api_key" not in trajectory["runtime_config"]
    assert "token" not in trajectory["runtime_config"]
    assert "x-api-key" not in trajectory["runtime_config"]
    assert "x_auth_token" not in trajectory["runtime_config"]
    assert "service_secret" not in trajectory["runtime_config"]
    assert "database_password" not in trajectory["runtime_config"]
    assert "client_credentials" not in trajectory["runtime_config"]
    assert "authorization_headers" not in trajectory["runtime_config"]["nested"]
    assert trajectory["runtime_config"]["gameplay_max_tokens"] == 512
    assert trajectory["runtime_config"]["api_key_env"] == "VLLM_API_KEY"
    assert trajectory["runtime_config"]["token_env"] == "TOKEN_ENV"
    assert trajectory["runtime_config_digest"] == canonical_digest(
        trajectory["runtime_config"]
    )

    assert len(trajectory["transitions"]) == 2
    speech_transition, vote_transition = trajectory["transitions"]
    assert speech_transition["delivered_observation"] == serialize_json_value(
        agents[0].observations[0]
    )
    assert speech_transition["delivered_observation"]["game_log"][0][
        "content"
    ]["marker"] == "exact-delivered-object"
    assert speech_transition["submitted_action"] == serialize_json_value(
        STRICT_SPEECH_ACTION
    )
    committed_speech = next(
        event
        for event in speech_transition["public_events_appended"]
        if event["event_type"] == "public_speech"
    )
    assert committed_speech["raw_text"] == STRICT_SPEECH_ACTION[1]["raw_text"]
    assert committed_speech["sp_actions"] == STRICT_SPEECH_ACTION[1]["sp_actions"]
    assert vote_transition["public_events_appended"][-1]["event_type"] == (
        "exile_result"
    )
    assert trajectory["termination"] == {
        "completion_status": "COMPLETE",
        "termination_kind": "normal_game_end",
        "winner": "Villager",
        "final_alive_players": [1, 2, 4, 5, 7],
    }
    assert [item["terminal_after"] for item in trajectory["transitions"]] == [
        False,
        True,
    ]

    reconstructed = list(trajectory["initial_public_events"])
    for transition in trajectory["transitions"]:
        reconstructed.extend(transition["public_events_appended"])
    assert normalize_public_events(reconstructed) == reconstructed
    assert trajectory["public_event_digest"] == public_event_digest(reconstructed)
    trajectory_without_digest = dict(trajectory)
    trajectory_without_digest.pop("trajectory_digest")
    assert trajectory["trajectory_digest"] == canonical_digest(
        trajectory_without_digest
    )

    assert provenance["schema_version"] == (
        OBSERVER_VIEW_PROVENANCE_SCHEMA_VERSION
    )
    assert provenance["observation_schema_version"] == OBSERVATION_SCHEMA_VERSION
    assert provenance["source_commit"] == trajectory["source_commit"]
    assert provenance["simulator_baseline"] == trajectory["simulator_baseline"]
    assert provenance["trajectory_digest"] == trajectory["trajectory_digest"]
    provenance_without_digest = dict(provenance)
    provenance_without_digest.pop("artifact_digest")
    assert provenance["artifact_digest"] == canonical_digest(
        provenance_without_digest
    )
    assert [boundary["boundary_type"] for boundary in provenance["boundaries"]] == [
        PRE_PUBLIC_SPEECH,
        POST_PUBLIC_SPEECH,
    ]
    pre, post = provenance["boundaries"]
    assert pre["boundary_id"] == "game:step_000000:PRE_PUBLIC_SPEECH"
    assert post["speech_event_idx"] == committed_speech["event_idx"]
    assert [view["observer_id"] for view in pre["observer_views"]] == [
        1,
        2,
        4,
        5,
        7,
    ]
    speaker_pre = next(
        view for view in pre["observer_views"] if view["observer_id"] == 1
    )
    assert speaker_pre["observation"] == speech_transition[
        "delivered_observation"
    ]
    assert speaker_pre["observation_digest"] == speech_transition[
        "delivered_observation_digest"
    ]
    for boundary in provenance["boundaries"]:
        without_digest = dict(boundary)
        without_digest.pop("boundary_digest")
        assert boundary["boundary_digest"] == canonical_digest(without_digest)

    assert env.order[-1] == "post-speech-collector"
    assert all(not item.startswith("view:2:") for item in env.order)
    assert any(item.startswith("view:1:") for item in env.order[:-1])
    assert OBSERVATION_SCHEMA_VERSION == "classic7_agent_observation_v1"


def test_agent_failure_writes_longest_committed_prefix(tmp_path):
    env = TrajectoryEnvironment()
    failure = RuntimeError(
        "agent failed; token=TOKEN-VALUE x-api-key=API-VALUE "
        "x_auth_token=AUTH-VALUE service_secret=SECRET-VALUE "
        "database_password=PASSWORD-VALUE credentials=CREDENTIAL-VALUE "
        "api_key_env=VLLM_API_KEY token_env=TOKEN_ENV"
    )
    recorder = _recorder(tmp_path, name="agent_failure")
    agents = _agents(second=ScriptedAgent(error=failure))

    with pytest.raises(RuntimeError, match="agent failed"):
        eval(
            env,
            agents,
            ROLES,
            trajectory_recorder=recorder,
        )

    trajectory, provenance = _read_outputs(tmp_path, "agent_failure")
    assert len(trajectory["transitions"]) == 1
    assert trajectory["transitions"][0]["submitted_action"] == (
        serialize_json_value(STRICT_SPEECH_ACTION)
    )
    failure_context = trajectory["termination"]["failure_context"]
    assert trajectory["termination"]["completion_status"] == "FAILED"
    assert failure_context["failure_stage"] == "agent_act"
    assert failure_context["failed_step_idx"] == 1
    assert failure_context["delivered_observation"] == serialize_json_value(
        agents[1].observations[0]
    )
    assert failure_context["submitted_action"] is None
    assert failure_context["exception_type"] == "RuntimeError"
    for secret in (
        "TOKEN-VALUE",
        "API-VALUE",
        "AUTH-VALUE",
        "SECRET-VALUE",
        "PASSWORD-VALUE",
        "CREDENTIAL-VALUE",
    ):
        assert secret not in failure_context["exception_message"]
    assert "api_key_env=VLLM_API_KEY" in failure_context["exception_message"]
    assert "token_env=TOKEN_ENV" in failure_context["exception_message"]
    assert len(provenance["boundaries"]) == 2


def test_env_step_failure_does_not_commit_failed_step(tmp_path):
    env = TrajectoryEnvironment(fail_step=True, start_phase="vote")
    recorder = _recorder(tmp_path, name="step_failure")
    action = ("vote", 1)

    with pytest.raises(RuntimeError, match="step failed"):
        eval(
            env,
            _agents(first=ScriptedAgent([action])),
            ROLES,
            trajectory_recorder=recorder,
        )

    trajectory, _ = _read_outputs(tmp_path, "step_failure")
    assert trajectory["transitions"] == []
    assert trajectory["initial_public_events"] == [
        {
            "event_idx": 0,
            "event_type": "phase_change",
            "phase": "1_day_vote",
        }
    ]
    failure_context = trajectory["termination"]["failure_context"]
    assert failure_context["failure_stage"] == "env_step"
    assert failure_context["submitted_action"] == ["vote", 1]
    assert trajectory["public_event_digest"] == public_event_digest(
        trajectory["initial_public_events"]
    )


@pytest.mark.parametrize("existing_output", ["trajectory", "observer_views"])
def test_recorder_refuses_to_overwrite_outputs(tmp_path, existing_output):
    (tmp_path / f"game.{existing_output}.json").write_text("occupied")
    with pytest.raises(FileExistsError, match="already exists"):
        _recorder(tmp_path)


def test_eval_recorder_default_is_none():
    assert inspect.signature(eval).parameters["trajectory_recorder"].default is None


@pytest.mark.parametrize("invalid_seed", [None, True, 402.0, "402"])
def test_recorder_requires_explicit_integer_environment_seed(
    tmp_path,
    invalid_seed,
):
    with pytest.raises(TypeError, match="environment_seed"):
        _recorder(
            tmp_path,
            name=f"invalid_seed_{type(invalid_seed).__name__}",
            environment_seed=invalid_seed,
        )


@pytest.mark.parametrize("invalid_step_idx", [True, -1, 1, 1.0, "0"])
def test_first_step_idx_must_be_integer_zero(tmp_path, invalid_step_idx):
    env = TrajectoryEnvironment(start_phase="vote")
    recorder = _recorder(
        tmp_path,
        name=f"invalid_first_step_{type(invalid_step_idx).__name__}",
    )
    observation = env.reset(ROLES)
    recorder.start(env, roles=ROLES)

    with pytest.raises((TypeError, ValueError), match="step_idx"):
        recorder.before_agent_act(
            env,
            step_idx=invalid_step_idx,
            acting_player_id=1,
            delivered_observation=observation,
            speech_kind=None,
        )


@pytest.mark.parametrize("invalid_step_idx", [0, 2])
def test_duplicate_or_skipped_step_idx_is_rejected(tmp_path, invalid_step_idx):
    env = TrajectoryEnvironment(start_phase="vote")
    recorder = _recorder(tmp_path, name=f"invalid_next_step_{invalid_step_idx}")
    observation = env.reset(ROLES)
    recorder.start(env, roles=ROLES)
    recorder.before_agent_act(
        env,
        step_idx=0,
        acting_player_id=1,
        delivered_observation=observation,
        speech_kind=None,
    )
    recorder.after_agent_act(("vote", 1))
    recorder.after_env_step(
        env,
        observation_after=observation,
        terminal_after=False,
    )

    with pytest.raises(ValueError, match="step_idx"):
        recorder.before_agent_act(
            env,
            step_idx=invalid_step_idx,
            acting_player_id=1,
            delivered_observation=observation,
            speech_kind=None,
        )


def test_submitted_speech_kind_must_equal_pending_kind(tmp_path):
    env = TrajectoryEnvironment()
    recorder = _recorder(tmp_path, name="speech_kind_mismatch")
    observation = env.reset(ROLES)
    recorder.start(env, roles=ROLES)
    recorder.before_agent_act(
        env,
        step_idx=0,
        acting_player_id=1,
        delivered_observation=observation,
        speech_kind="speech_pk",
    )
    recorder.after_agent_act(STRICT_SPEECH_ACTION)
    observation_after, _, done, _ = env.step(STRICT_SPEECH_ACTION)

    with pytest.raises(ValueError, match="kind mismatch"):
        recorder.after_env_step(
            env,
            observation_after=observation_after,
            terminal_after=done,
        )


def test_explicit_abort_preserves_committed_prefix(tmp_path):
    env = TrajectoryEnvironment(start_phase="vote")
    recorder = _recorder(tmp_path, name="aborted")
    env.reset(ROLES)
    recorder.start(env, roles=ROLES)
    recorder.abort()

    trajectory, provenance = _read_outputs(tmp_path, "aborted")
    assert trajectory["transitions"] == []
    assert trajectory["termination"] == {
        "completion_status": "ABORTED",
        "termination_kind": "explicit_handled_abort",
    }
    assert provenance["boundaries"] == []
