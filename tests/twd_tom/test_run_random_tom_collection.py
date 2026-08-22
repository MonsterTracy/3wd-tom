import inspect
from copy import deepcopy

import pytest

from run_random import (
    build_arg_parser,
    build_twd_tom_sample_collector,
    eval,
)
from werewolf.models.twd_tom.collector import (
    TWDToMSampleCollector,
)
from werewolf.models.twd_tom.public_events import public_speech_actions
from tests.twd_tom.public_event_fixtures import public_history_fields
from werewolf.speech.private_belief_perceiver import (
    PlayingAgentBeliefReporter,
)


class ScriptedAgent:
    def __init__(
        self,
        actions,
    ):
        self.actions = list(
            actions
        )
        self.reset_count = 0
        self.observations = []

    def reset(self):
        self.reset_count += 1

    def act(
        self,
        observation,
    ):
        self.observations.append(
            observation
        )

        if not self.actions:
            raise RuntimeError(
                "agent has no scripted action"
            )

        return self.actions.pop(0)


class ScriptedEnvironment:
    def __init__(
        self,
        transitions,
        alive=None,
        start_phase="speech",
        speech_actions=None,
    ):
        self.transitions = list(
            transitions
        )

        self.alive = list(
            alive or [1] * 7
        )

        self.game_log = []
        self.public_events = []
        self.reset_roles = None
        self.step_actions = []
        self.phase = start_phase
        self.speech_actions = dict(
            speech_actions or {}
        )

    def reset(
        self,
        roles,
    ):
        self.reset_roles = list(
            roles
        )
        self.public_events = [
            {
                "event_idx": 0,
                "event_type": "phase_change",
                "phase": f"1_day_{self.phase}",
            },
            {
                "event_idx": 1,
                "event_type": "turn_start",
                "speaker": "player1",
            },
        ]

        return {
            "current_act_idx": 1,
            "phase": f"1_day_{self.phase}",
        }

    def step(
        self,
        action,
    ):
        self.step_actions.append(
            action
        )
        if action[0] in {"speech", "speech_pk"}:
            parsed_actions = self.speech_actions.get(
                action[1]
            )
            if parsed_actions is None:
                self.game_log.append(action[1])
            else:
                self.game_log.append(
                    {
                        "event": action[0],
                        "content": {
                            "speech_content": action[1],
                            "sp_actions": deepcopy(parsed_actions),
                        },
                    }
                )
            self.public_events.append(
                {
                    "event_idx": len(self.public_events),
                    "event_type": "public_speech",
                    "speaker": f"player{len(self.step_actions)}",
                    "raw_text": action[1],
                    "sp_actions": deepcopy(parsed_actions or []),
                }
            )

        if not self.transitions:
            raise RuntimeError(
                "environment has no "
                "scripted transition"
            )

        transition = self.transitions.pop(0)
        phase = transition[0]["phase"]
        self.phase = "speech_pk" if phase.endswith("speech_pk") else phase.rsplit("_", 1)[-1]
        if "speech" in self.phase and not transition[2]:
            self.public_events.append(
                {
                    "event_idx": len(self.public_events),
                    "event_type": "turn_start",
                    "speaker": f"player{transition[0]['current_act_idx']}",
                }
            )
        return transition


class RecordingSampleCollector:
    def __init__(self):
        self.calls = []

    def record(
        self,
        env,
        *,
        step_idx=None,
        trigger=None,
        phase=None,
        speaker_id=None,
        observer_ids=None,
    ):
        self.calls.append(
            {
                "env": env,
                "step_idx": step_idx,
                "trigger": trigger,
                "phase": phase,
                "speaker_id": speaker_id,
                "observer_ids": (
                    observer_ids
                ),
                "environment_step_count": (
                    len(env.step_actions)
                ),
                "public_history": deepcopy(env.public_events),
            }
        )


def test_eval_orders_trajectory_pre_boundary_before_belief_and_action():
    events = []

    class OrderedAgent(ScriptedAgent):
        def act(self, observation):
            events.append("agent.act")
            return super().act(observation)

    class OrderedEnvironment(ScriptedEnvironment):
        def step(self, action):
            events.append("env.step")
            return super().step(action)

    class OrderedCollector(RecordingSampleCollector):
        def record(self, env, **kwargs):
            events.append("belief.record")
            return super().record(env, **kwargs)

    class OrderedRecorder:
        def start(self, env, *, roles):
            events.append("trajectory.start")

        def before_agent_act(self, env, **kwargs):
            events.append("trajectory.before_agent_act")

        def after_agent_act(self, action):
            events.append("trajectory.after_agent_act")

        def after_env_step(self, env, **kwargs):
            events.append("trajectory.after_env_step")

        def complete(self, env, *, winner):
            events.append("trajectory.complete")

    env = OrderedEnvironment(
        transitions=[
            (
                {"current_act_idx": 1, "phase": "1_day_vote"},
                0,
                True,
                {"Werewolf": -1},
            )
        ]
    )
    agents = [OrderedAgent([("speech", "ordered speech")])]

    eval(
        env,
        agents,
        roles_=["Villager"],
        sample_collector=OrderedCollector(),
        trajectory_recorder=OrderedRecorder(),
    )

    assert events == [
        "trajectory.start",
        "trajectory.before_agent_act",
        "belief.record",
        "agent.act",
        "trajectory.after_agent_act",
        "env.step",
        "trajectory.after_env_step",
        "trajectory.complete",
    ]


def test_eval_collects_all_alive_observers_before_public_speech():
    env = ScriptedEnvironment(
        transitions=[
            (
                {
                    "current_act_idx": 2,
                    "phase": "1_day_vote",
                },
                0,
                False,
                {},
            ),
            (
                {
                    "current_act_idx": 2,
                    "phase": "1_day_vote",
                },
                0,
                True,
                {
                    "Werewolf": -1,
                },
            ),
        ],
        alive=[
            1,
            1,
            0,
            1,
            1,
            0,
            1,
        ],
    )

    agents = [
        ScriptedAgent(
            [
                (
                    "speech",
                    "公开发言",
                )
            ]
        ),
        ScriptedAgent(
            [
                (
                    "vote",
                    1,
                )
            ]
        ),
    ]

    collector = (
        RecordingSampleCollector()
    )

    result = eval(
        env,
        agents,
        roles_=[
            "Werewolf",
            "Werewolf",
            "Seer",
            "Witch",
            "Villager",
            "Villager",
            "Villager",
        ],
        sample_collector=collector,
    )

    assert result == "Villager win"

    assert collector.calls == [
        {
            "env": env,
            "step_idx": 0,
            "trigger": "speech",
            "phase": "1_day_speech",
            "speaker_id": 1,
            "observer_ids": [
                1,
                2,
                4,
                5,
                7,
            ],
            "environment_step_count": 0,
                "public_history": [
                    {
                        "event_idx": 0,
                        "event_type": "phase_change",
                        "phase": "1_day_speech",
                    },
                    {
                        "event_idx": 1,
                        "event_type": "turn_start",
                        "speaker": "player1",
                    },
                ],
        }
    ]

    assert all(
        agent.reset_count == 1
        for agent in agents
    )


def test_eval_collects_speech_pk():
    env = ScriptedEnvironment(
        transitions=[
            (
                {
                    "current_act_idx": 1,
                    "phase": "1_day_vote",
                },
                0,
                True,
                {
                    "Werewolf": 1,
                },
            )
        ],
        start_phase="speech_pk",
    )

    agent = ScriptedAgent(
        [
            (
                "speech_pk",
                "PK发言",
            )
        ]
    )

    collector = (
        RecordingSampleCollector()
    )

    result = eval(
        env,
        [agent],
        roles_=[
            "Werewolf",
        ],
        sample_collector=collector,
    )

    assert result == "Werewolf win"

    assert collector.calls[0][
        "trigger"
    ] == "speech_pk"

    assert collector.calls[0][
        "step_idx"
    ] == 0


def test_eval_does_not_collect_non_speech_actions():
    env = ScriptedEnvironment(
        transitions=[
            (
                {
                    "current_act_idx": 1,
                    "phase": "1_day_vote",
                },
                0,
                True,
                {
                    "Werewolf": -1,
                },
            )
        ],
        start_phase="skill_seer",
    )

    collector = (
        RecordingSampleCollector()
    )

    eval(
        env,
        [
            ScriptedAgent(
                [
                    (
                        "check",
                        1,
                    )
                ]
            )
        ],
        roles_=[
            "Seer",
        ],
        sample_collector=collector,
    )

    assert collector.calls == []


def test_current_speech_enters_only_the_next_pre_speech_snapshot():
    env = ScriptedEnvironment(
        transitions=[
            (
                {"current_act_idx": 2, "phase": "1_day_speech"},
                0,
                False,
                {},
            ),
            (
                {"current_act_idx": 2, "phase": "1_day_vote"},
                0,
                True,
                {"Werewolf": -1},
            ),
        ]
    )
    agents = [
        ScriptedAgent([("speech", "first")]),
        ScriptedAgent([("speech", "second")]),
    ]
    collector = RecordingSampleCollector()
    eval(env, agents, roles_=["Villager"] * 7, sample_collector=collector)
    first, second = [
        call["public_history"]
        for call in collector.calls
    ]
    assert first[-1] == {
        "event_idx": 1,
        "event_type": "turn_start",
        "speaker": "player1",
    }
    assert public_speech_actions(first) == []
    assert public_speech_actions(second) == []
    assert second[-2]["raw_text"] == "first"
    assert second[-1] == {
        "event_idx": 3,
        "event_type": "turn_start",
        "speaker": "player2",
    }
    assert [call["environment_step_count"] for call in collector.calls] == [0, 1]


def test_multi_action_speech_has_one_pre_speech_snapshot_and_one_dataset_row():
    first_actions = [
        ["player1", "point_as_werewolf", "player3"],
        ["player1", "support", "player2"],
    ]
    env = ScriptedEnvironment(
        transitions=[
            (
                {"current_act_idx": 2, "phase": "1_day_speech"},
                0,
                False,
                {},
            ),
            (
                {"current_act_idx": 2, "phase": "1_day_vote"},
                0,
                True,
                {"Werewolf": -1},
            ),
        ],
        speech_actions={
            "first": first_actions,
            "second": [["player2", "oppose", "player1"]],
        },
    )
    collector = RecordingSampleCollector()
    eval(
        env,
        [
            ScriptedAgent([("speech", "first")]),
            ScriptedAgent([("speech", "second")]),
        ],
        roles_=["Villager"] * 7,
        sample_collector=collector,
    )

    assert len(collector.calls) == 2
    assert collector.calls[0]["public_history"][-1] == {
        "event_idx": 1,
        "event_type": "turn_start",
        "speaker": "player1",
    }
    assert public_speech_actions(
        collector.calls[1]["public_history"]
    ) == first_actions

    assert collector.calls[1]["public_history"][-1]["event_type"] == "turn_start"


def test_builds_complete_sample_collector(
    tmp_path,
):
    agents = [ScriptedAgent([]) for _ in range(7)]
    for index, agent in enumerate(agents, start=1):
        agent.backend_id = f"backend_{index}"

    collector = (
        build_twd_tom_sample_collector(
            agent_list=agents,
            output_path=str(
                tmp_path
                / "samples.jsonl"
            ),
            game_id="game_001",
        )
    )

    try:
        assert isinstance(
            collector,
            TWDToMSampleCollector,
        )

        assert isinstance(
            collector.snapshot_collector.reporter,
            PlayingAgentBeliefReporter,
        )
        assert collector.snapshot_collector.agents == tuple(agents)

        assert (
            collector.game_id
            == "game_001"
        )
    finally:
        collector.close()


def test_sample_collector_builder_has_no_roles_argument():
    signature = inspect.signature(
        build_twd_tom_sample_collector
    )

    forbidden = {
        "collection_mode",
        "reporter_dispatch",
        "roles",
        "true_roles",
        "wolf_labels",
        "truth",
    }

    assert forbidden.isdisjoint(
        signature.parameters
    )

    collector_signature = inspect.signature(TWDToMSampleCollector)
    assert "sample_builder" not in collector_signature.parameters

    option_strings = {
        option
        for action in build_arg_parser()._actions
        for option in action.option_strings
    }
    assert {
        "--tom_sample_path",
        "--twd_tom_sample_path",
        "--twd_tom2_shadow_checkpoint",
        "--twd_tom2_shadow_device",
        "--twd_tom2_shadow_output_path",
    }.isdisjoint(option_strings)
