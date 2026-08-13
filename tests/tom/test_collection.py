import json
from copy import deepcopy

import pytest

from run_random import build_tom_collector, eval
from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
from werewolf.helper.log_utils import Log
from werewolf.models.tom.collection import Collector
from werewolf.models.tom.reporter import BeliefReporter


WITCH_ROLES = [
    "Werewolf", "Werewolf", "Seer", "Witch",
    "Villager", "Villager", "Villager",
]
GUARD_ROLES = [
    "Werewolf", "Werewolf", "Seer", "Guard",
    "Villager", "Villager", "Villager",
]
FORMAL_ACTION = [["player1", "point_as_werewolf", "player5"]]


class Parser:
    def __init__(self, actions=FORMAL_ACTION):
        self.actions = actions

    def parse(self, **_kwargs):
        return deepcopy(self.actions)


class Backend:
    supports_json_schema = True

    def __init__(self, response='{"suspected_werewolves":[]}'):
        self.response = response
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class Agent:
    def __init__(self, player_id, backend):
        self.backend = backend
        self.model_name = f"model-{player_id}"
        self.memory = [f"persistent-{player_id}"]
        self.actions = []

    def reset(self):
        pass

    def act(self, _observation):
        self.actions.append("gameplay")
        return ("speech", "CURRENT-SPEECH")


def agents(responses=None):
    responses = responses or {}
    result = []
    for player_id in range(1, 8):
        backend = Backend(responses.get(player_id, '{"suspected_werewolves":[]}'))
        result.append(Agent(player_id, backend))
    return result


def ready_env(*, guard=False, actions=FORMAL_ACTION):
    env = WerewolfTextEnvV0(
        n_guard=1 if guard else 0,
        n_witch=0 if guard else 1,
        log_save_path=None,
        speech_perceiver=Parser(actions),
    )
    env.reset(roles=GUARD_ROLES if guard else WITCH_ROLES)
    env.day = 1
    env.day_or_night = "day"
    env.phase = "speech"
    env.current_act_idx = 0
    env.speech_queue = [1]
    env.public_events = [
        {"event_idx": 0, "event_type": "phase_change", "phase": "1_day_speech"},
        {"event_idx": 1, "event_type": "turn_start", "speaker": "player1"},
    ]
    env.game_log.extend(
        [
            Log(
                viewer=[2], source=2, target=4,
                content={"cheked_identity": "bad"}, day=0,
                time="night", event="skill_seer",
            ),
            Log(
                viewer=[3], source=3, target=5,
                content={"protected": 5} if guard else {"poison": 5},
                day=0, time="night",
                event="skill_guard" if guard else "skill_witch",
            ),
        ]
    )
    return env


def make_collector(tmp_path, env, agent_list, *, name="raw.jsonl"):
    return build_tom_collector(
        env=env,
        agent_list=agent_list,
        output_path=tmp_path / name,
        game_id="game-1",
        seed=17,
    )


def prompt_for(agent):
    assert len(agent.backend.calls) == 1
    return agent.backend.calls[0]["messages"][0]["content"]


@pytest.mark.parametrize(("guard", "context"), [(False, "seer_witch"), (True, "seer_guard")])
def test_post_speech_observations_are_private_aware_for_both_configs(
    tmp_path, guard, context
):
    env = ready_env(guard=guard)
    agent_list = agents()
    collector = make_collector(tmp_path, env, agent_list)

    env.step(("speech", "CURRENT-SPEECH"))
    sample = collector.record(
        env, step_idx=9, round_number=1, phase="speech", speaker_id=1
    )
    collector.close()

    assert sample["episode_context"] == context
    assert sample["formal_speech_actions"] == FORMAL_ACTION
    assert "CURRENT-SPEECH" in prompt_for(agent_list[0])
    assert "FUTURE-SPEECH" not in prompt_for(agent_list[0])
    assert "werewolf_team_info" in prompt_for(agent_list[0])
    assert "werewolf_team_info" not in prompt_for(agent_list[2])
    assert "skill_seer" in prompt_for(agent_list[2])
    assert "skill_seer" not in prompt_for(agent_list[4])
    private_event = "skill_guard" if guard else "skill_witch"
    assert private_event in prompt_for(agent_list[3])
    assert private_event not in prompt_for(agent_list[4])
    assert "god_view" not in "".join(prompt_for(agent) for agent in agent_list)
    assert sample["public_events"][-1]["event_type"] == "public_speech"
    assert all(event["event_type"] != "turn_start" for event in sample["public_events"][-1:])


def test_alive_only_queries_post_speech_alive_observers(tmp_path):
    env = ready_env()
    env.alive = [1, 0, 1, 1, 0, 1, 1]
    agent_list = agents()
    collector = make_collector(tmp_path, env, agent_list)
    env.step(("speech", "CURRENT-SPEECH"))

    sample = collector.record(
        env, step_idx=1, round_number=1, phase="speech", speaker_id=1
    )
    collector.close()

    assert sample["alive_observers"] == [
        "player1", "player3", "player4", "player6", "player7"
    ]
    assert [report["observer_id"] for report in sample["observer_reports"]] == (
        sample["alive_observers"]
    )
    assert [len(agent.backend.calls) for agent in agent_list] == [1, 0, 1, 1, 0, 1, 1]


def test_zero_triplet_writes_nothing_and_makes_no_reporter_call(tmp_path):
    env = ready_env(actions=[])
    agent_list = agents()
    output = tmp_path / "zero.jsonl"
    collector = make_collector(tmp_path, env, agent_list, name="zero.jsonl")
    env.step(("speech", "NO-FORMAL-ACTION"))

    result = collector.record(
        env, step_idx=1, round_number=1, phase="speech", speaker_id=1
    )
    collector.close()

    assert result is None
    assert collector.samples_written == 0
    assert output.read_text() == ""
    assert all(agent.backend.calls == [] for agent in agent_list)


def test_failures_are_invalid_without_retry_repair_or_fallback(tmp_path):
    agent_list = agents(
        {
            1: "not-json",
            2: RuntimeError("backend unavailable"),
            3: '{"suspected_werewolves":["player8"]}',
            4: '{"suspected_werewolves":[]}',
        }
    )
    env = ready_env()
    collector = make_collector(tmp_path, env, agent_list)
    env.step(("speech", "CURRENT-SPEECH"))

    sample = collector.record(
        env, step_idx=1, round_number=1, phase="speech", speaker_id=1
    )
    collector.close()

    reports = sample["observer_reports"]
    assert reports[:4] == [
        {
            "observer_id": "player1", "valid": False,
            "suspected_werewolves": None, "error": "parse_error",
        },
        {
            "observer_id": "player2", "valid": False,
            "suspected_werewolves": None, "error": "reporter_error",
        },
        {
            "observer_id": "player3", "valid": False,
            "suspected_werewolves": None, "error": "parse_error",
        },
        {
            "observer_id": "player4", "valid": True,
            "suspected_werewolves": [], "error": None,
        },
    ]
    assert all(len(agent.backend.calls) == 1 for agent in agent_list)


def snapshot_env(env):
    return {
        "alive": deepcopy(env.alive),
        "phase": env.phase,
        "current_act_idx": env.current_act_idx,
        "speech_queue": deepcopy(env.speech_queue),
        "vote_queue": deepcopy(env.vote_queue),
        "public_events": deepcopy(env.public_events),
        "game_log": [deepcopy(log.__dict__) for log in env.game_log],
    }


def test_collection_does_not_mutate_environment_agents_or_gameplay(tmp_path):
    env = ready_env()
    agent_list = agents()
    collector = make_collector(tmp_path, env, agent_list)
    env.step(("speech", "CURRENT-SPEECH"))
    before_env = snapshot_env(env)
    before_memory = [deepcopy(agent.memory) for agent in agent_list]
    before_actions = [list(agent.actions) for agent in agent_list]

    collector.record(
        env, step_idx=1, round_number=1, phase="speech", speaker_id=1
    )
    collector.close()

    assert snapshot_env(env) == before_env
    assert [agent.memory for agent in agent_list] == before_memory
    assert [agent.actions for agent in agent_list] == before_actions


def test_collector_does_not_read_role_truth_outside_legal_observation(tmp_path):
    agent_list = agents()
    dispatches = [
        {"backend": agent.backend, "model_name": agent.model_name}
        for agent in agent_list
    ]
    legal_logs = [
        Log(
            viewer=[5], source=0, target=[1, 2, 3, 4, 5, 6, 7],
            content={"speech_content": "PUBLIC", "sp_actions": FORMAL_ACTION},
            day=1, time="day", event="speech",
        )
    ]

    class BoundaryEnv:
        roles = ["TRUTH-CANARY"] * 7
        alive = [0, 0, 0, 0, 0, 1, 0]
        public_events = [
            {"event_idx": 0, "event_type": "phase_change", "phase": "1_day_speech"},
            {
                "event_idx": 1, "event_type": "public_speech",
                "speaker": "player1", "raw_text": "PUBLIC", "sp_actions": FORMAL_ACTION,
            },
        ]

        def get_observation_for(self, player_id):
            assert player_id == 6
            return {
                "observer_id": 6,
                "current_act_idx": 2,
                "identity": "Villager",
                "game_log": deepcopy(legal_logs),
                "phase": "1_day_speech",
                "authoritative_public_state": {"alive_players": [6]},
            }

    collector = Collector(
        tmp_path / "truth.jsonl", game_id="g", seed=None,
        episode_context="seer_witch", reporter=BeliefReporter(dispatches),
    )
    sample = collector.record(
        BoundaryEnv(), step_idx=1, round_number=1, phase="speech", speaker_id=1
    )
    collector.close()

    assert sample["alive_observers"] == ["player6"]
    assert "TRUTH-CANARY" not in prompt_for(agent_list[5])
    assert "roles" not in sample


def test_runtime_hook_is_after_commit_and_before_next_gameplay_action():
    class RuntimeEnv:
        phase = "speech"
        day = 1
        alive = [1] * 7

        def __init__(self):
            self.committed = False
            self.public_events = []

        def reset(self, roles):
            return {"current_act_idx": 1, "phase": "1_day_speech"}

        def step(self, action):
            assert action == ("speech", "CURRENT-SPEECH")
            self.committed = True
            self.public_events.append(
                {
                    "event_idx": 0, "event_type": "public_speech",
                    "speaker": "player1", "raw_text": action[1],
                    "sp_actions": FORMAL_ACTION,
                }
            )
            return (
                {"current_act_idx": 1, "phase": "1_day_vote"},
                0, True, {"Werewolf": -1},
            )

    class SpyCollector:
        def __init__(self):
            self.calls = []

        def record(self, env, **kwargs):
            assert env.committed is True
            assert env.public_events[-1]["raw_text"] == "CURRENT-SPEECH"
            self.calls.append(kwargs)

    env = RuntimeEnv()
    agent = Agent(1, Backend())
    collector = SpyCollector()
    assert eval(
        env, [agent], ["Villager"], tom_collector=collector
    ) == "Villager win"
    assert len(agent.actions) == 1
    assert collector.calls == [
        {
            "step_idx": 0, "round_number": 1,
            "phase": "speech", "speaker_id": 1,
        }
    ]


def test_saved_sample_has_cutoff_identity_and_no_hidden_dump(tmp_path):
    env = ready_env()
    agent_list = agents({1: '{"suspected_werewolves":["player5","player2"]}'})
    collector = make_collector(tmp_path, env, agent_list)
    env.step(("speech", "CURRENT-SPEECH"))
    sample = collector.record(
        env, step_idx=4, round_number=1, phase="speech", speaker_id=1
    )
    collector.close()
    saved = json.loads((tmp_path / "raw.jsonl").read_text())

    assert saved == sample
    assert len(sample["public_history_cutoff"]["digest"]) == 64
    assert sample["public_history_cutoff"]["event_idx"] == (
        sample["public_events"][-1]["event_idx"]
    )
    assert sample["observer_reports"][0]["suspected_werewolves"] == [
        "player2", "player5"
    ]
    serialized = json.dumps(sample)
    assert "true_roles" not in serialized
    assert "agent internal" not in serialized
