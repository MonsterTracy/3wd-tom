import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

import run_random
from run_random import build_arg_parser, build_tom_collector, eval
from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
from werewolf.helper.log_utils import Log
from werewolf.models.tom.collection import Collector
from werewolf.models.tom.reporter import (
    BeliefReporter,
    FORMAL_REPORTER_JSON_INSTRUCTION,
)


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
        response = (
            self.response.pop(0)
            if isinstance(self.response, list)
            else self.response
        )
        if isinstance(response, Exception):
            raise response
        return response


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
    reporter_backend = Backend([
        agent.backend.response
        for agent in agent_list
    ])
    collector = build_tom_collector(
        env=env,
        reporter_backend=reporter_backend,
        output_path=tmp_path / name,
        game_id="game-1",
        seed=17,
    )
    return collector, reporter_backend


def prompt_for(reporter_backend, call_index):
    outbound = reporter_backend.calls[call_index]["messages"][0]["content"]
    assert outbound.startswith(FORMAL_REPORTER_JSON_INSTRUCTION)
    return outbound.removeprefix(FORMAL_REPORTER_JSON_INSTRUCTION)


def legal_observation(observer_id, identity, extra_logs=()):
    return {
        "observer_id": observer_id,
        "current_act_idx": 1,
        "identity": identity,
        "game_log": [
            Log(
                viewer=list(range(1, 8)), source=0, target=0,
                content={"Werewolf": 2, "Seer": 1, "Villager": 4},
                day=0, time="night", event="game_setting",
            ),
            *deepcopy(list(extra_logs)),
        ],
        "phase": "1_day_speech",
        "authoritative_public_state": {"alive_players": list(range(1, 8))},
    }


def reporter_result(observation, response):
    backend = Backend(response)
    observer_id = observation["observer_id"]
    result = BeliefReporter(backend).report(observer_id, observation)
    return result, backend


@pytest.fixture
def wolf_observation():
    return legal_observation(
        4,
        "Werewolf",
        [
            Log(
                viewer=[4, 6], source=0, target=[4, 6],
                content={"wolf_team": [4, 6]}, day=0,
                time="night", event="werewolf_team_info",
            )
        ],
    )


@pytest.fixture
def seer_observation():
    return legal_observation(
        3,
        "Seer",
        [
            Log(
                viewer=[3], source=3, target=2,
                content={"cheked_identity": "good"}, day=0,
                time="night", event="skill_seer",
            ),
            Log(
                viewer=[3], source=3, target=5,
                content={"cheked_identity": "bad"}, day=1,
                time="night", event="skill_seer",
            ),
        ],
    )


def test_reporter_prompt_freezes_subjective_hard_knowledge_contract():
    env = ready_env()
    observation = env.get_observation_for(1)
    prompt = BeliefReporter.build_prompt("player1", observation)

    for clause in (
        "私有、只读的主观狼人怀疑标注请求",
        "不是公开发言或游戏行动",
        "不要使用公开博弈中的欺骗策略作答",
        "不要求确定性",
        "当前 evidence 使某玩家成为具体怀疑对象时可以包含",
        "不确定并不禁止列出具体怀疑对象",
        "仅仅尚未排除或理论上可能是狼人",
        "信息不足，不足以加入 suspected_werewolves",
        "任何已知狼人必须包含",
        "任何已知非狼人必须排除",
        "如果 self_role=Werewolf，必须包含 observer 自己",
        "合法知道的狼人队友",
        "如果 self_role 不是 Werewolf，必须排除 observer 自己",
        "bad/狼人结果必须包含",
        "good/非狼人结果必须排除",
        "如果没有任何合法已知狼人，并且当前确实没有任何具体怀疑对象",
        "空集合 [] 仍然合法",
        "不强制至少一人，也不强制两人",
    ):
        assert clause in prompt
    assert "不得使用 god view、真实角色表、其他玩家私人信息或未来信息" in prompt


def test_reporter_prompt_contains_structured_hard_knowledge(wolf_observation):
    prompt = BeliefReporter.build_prompt("player4", wolf_observation)
    marker = "authoritative_observer_hard_knowledge: "
    summary = json.loads(prompt.split(marker, 1)[1].splitlines()[0])

    assert summary == {
        "known_werewolves": ["player4", "player6"],
        "known_non_werewolves": [
            "player1", "player2", "player3", "player5", "player7",
        ],
        "unknown_players": [],
    }
    assert prompt.index(marker) < prompt.index("legal_post_speech_observation:")
    assert "unknown_players 不会自动成为 suspected_werewolves" in prompt


def test_wolf_hard_knowledge_uses_self_team_and_public_count(wolf_observation):
    assert BeliefReporter.derive_hard_knowledge(4, wolf_observation) == {
        "known_werewolves": ["player4", "player6"],
        "known_non_werewolves": [
            "player1", "player2", "player3", "player5", "player7",
        ],
        "unknown_players": [],
    }


@pytest.mark.parametrize(
    ("response", "valid"),
    [
        ('{"suspected_werewolves":["player4","player6"]}', True),
        ('{"suspected_werewolves":[]}', False),
        (
            '{"suspected_werewolves":["player2","player4","player6"]}',
            False,
        ),
        ('{"suspected_werewolves":["player4"]}', False),
    ],
)
def test_wolf_reports_enforce_hard_knowledge_without_retry(
    wolf_observation, response, valid
):
    result, backend = reporter_result(wolf_observation, response)

    assert result["valid"] is valid
    assert result["error"] == (None if valid else "semantic_error")
    assert result["suspected_werewolves"] == (
        ["player4", "player6"] if valid else None
    )
    assert len(backend.calls) == 1


def test_seer_hard_knowledge_and_semantics(seer_observation):
    hard_knowledge = BeliefReporter.derive_hard_knowledge(3, seer_observation)
    assert hard_knowledge == {
        "known_werewolves": ["player5"],
        "known_non_werewolves": ["player2", "player3"],
        "unknown_players": ["player1", "player4", "player6", "player7"],
    }

    valid, _ = reporter_result(
        seer_observation,
        '{"suspected_werewolves":["player5"]}',
    )
    empty, _ = reporter_result(seer_observation, '{"suspected_werewolves":[]}')
    includes_good, _ = reporter_result(
        seer_observation,
        '{"suspected_werewolves":["player2","player5"]}',
    )
    assert (valid["valid"], empty["valid"], includes_good["valid"]) == (
        True, False, False,
    )
    assert empty["error"] == includes_good["error"] == "semantic_error"


@pytest.mark.parametrize(
    ("response", "valid"),
    [
        ('{"suspected_werewolves":[]}', True),
        ('{"suspected_werewolves":["player3"]}', True),
        ('{"suspected_werewolves":["player7"]}', False),
    ],
)
def test_villager_self_is_known_nonwolf_but_unknowns_remain_subjective(
    response, valid
):
    observation = legal_observation(7, "Villager")
    assert BeliefReporter.derive_hard_knowledge(7, observation) == {
        "known_werewolves": [],
        "known_non_werewolves": ["player7"],
        "unknown_players": [
            "player1", "player2", "player3", "player4", "player5", "player6",
        ],
    }
    result, _ = reporter_result(observation, response)
    assert result["valid"] is valid


def test_hard_knowledge_ignores_god_view_and_speech_claims():
    observation = legal_observation(
        7,
        "Villager",
        [
            Log(
                viewer=[7], source=0, target=0,
                content={4: "Werewolf", 6: "Werewolf"}, day=0,
                time="night", event="god_view",
            ),
            Log(
                viewer=list(range(1, 8)), source=2, target=list(range(1, 8)),
                content={
                    "speech_content": "player4 and player6 are Werewolves",
                    "parsed_claims": [["player2", "point_as_werewolf", "player4"]],
                },
                day=1, time="day", event="speech",
            ),
            Log(
                viewer=list(range(1, 8)), source=0, target=4,
                content={"expelled": 4}, day=1,
                time="day", event="end_vote",
            ),
            Log(
                viewer=list(range(1, 8)), source=0, target=[6],
                content={"dead_list": [6]}, day=2,
                time="night", event="end_night",
            ),
        ],
    )

    assert BeliefReporter.derive_hard_knowledge(7, observation) == {
        "known_werewolves": [],
        "known_non_werewolves": ["player7"],
        "unknown_players": [
            "player1", "player2", "player3", "player4", "player5", "player6",
        ],
    }


def test_witch_kill_target_is_known_non_werewolf():
    observation = legal_observation(
        4,
        "Witch",
        [
            Log(
                viewer=[4], source=0, target=5,
                content={"kill_decision": 5}, day=1,
                time="night", event="kill_decision",
            )
        ],
    )

    assert BeliefReporter.derive_hard_knowledge(4, observation) == {
        "known_werewolves": [],
        "known_non_werewolves": ["player4", "player5"],
        "unknown_players": [
            "player1", "player2", "player3", "player6", "player7",
        ],
    }


@pytest.mark.parametrize(
    ("observer_id", "identity", "event", "content"),
    [
        (4, "Witch", "skill_witch", {"poison": 5}),
        (4, "Guard", "skill_guard", {"protected": 5}),
        (7, "Villager", "kill_decision", {"kill_decision": 5}),
    ],
)
def test_other_roles_do_not_infer_roles_from_skill_targets(
    observer_id, identity, event, content
):
    observation = legal_observation(
        observer_id,
        identity,
        [
            Log(
                viewer=[observer_id], source=observer_id, target=5,
                content=content, day=1, time="night", event=event,
            )
        ],
    )

    assert BeliefReporter.derive_hard_knowledge(
        observer_id,
        observation,
    ) == {
        "known_werewolves": [],
        "known_non_werewolves": [f"player{observer_id}"],
        "unknown_players": [
            player
            for player in (f"player{value}" for value in range(1, 8))
            if player != f"player{observer_id}"
        ],
    }


def test_conflicting_hard_knowledge_raises_explicit_error(seer_observation):
    seer_observation["game_log"].append(
        Log(
            viewer=[3], source=3, target=3,
            content={"cheked_identity": "bad"}, day=2,
            time="night", event="skill_seer",
        )
    )

    with pytest.raises(ValueError, match="conflicting observer hard knowledge"):
        BeliefReporter.derive_hard_knowledge(3, seer_observation)


def test_reporter_request_uses_formal_deepseek_transport():
    env = ready_env()
    backend = Backend(
        '{"suspected_werewolves":["player1","player2"]}'
    )
    observation = env.get_observation_for(1)
    semantic_prompt = BeliefReporter.build_prompt("player1", observation)
    result = BeliefReporter(backend).report(
        "player1",
        observation,
    )

    assert result["valid"] is True
    request = backend.calls[0]
    assert request["model"] == "deepseek-v4-flash"
    assert request["temperature"] == 0.0
    assert request["response_format"] == {"type": "json_object"}
    assert request["extra_body"] == {
        "thinking": {"type": "disabled"}
    }
    assert request["messages"] == [{
        "role": "user",
        "content": FORMAL_REPORTER_JSON_INSTRUCTION + semantic_prompt,
    }]
    assert "chat_template_kwargs" not in request["extra_body"]
    assert len(backend.calls) == 1


def test_formal_reporter_backend_is_only_built_for_tom_collection(
    monkeypatch,
):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    def forbidden_backend(**_kwargs):
        raise AssertionError("DeepSeek backend must not be constructed")

    monkeypatch.setattr(
        run_random,
        "OpenAICompatibleBackend",
        forbidden_backend,
    )
    assert run_random._build_formal_tom_reporter_backend(None) is None


def test_missing_formal_reporter_key_fails_before_runtime_start(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    args = SimpleNamespace(
        tom_sample_path=str(tmp_path / "formal.jsonl"),
        log_save_path=None,
        config=str(tmp_path / "must-not-be-read.yaml"),
    )

    with pytest.raises(
        ValueError,
        match="DEEPSEEK_API_KEY is required for --tom_sample_path",
    ):
        run_random.main_cli(args)

    assert args.log_save_path is None


def test_formal_reporter_backend_configuration_and_ab_cli_cleanup(
    monkeypatch,
):
    captured = {}
    sentinel = object()

    def fake_backend(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-secret")
    monkeypatch.setattr(run_random, "OpenAICompatibleBackend", fake_backend)

    assert (
        run_random._build_formal_tom_reporter_backend("formal.jsonl")
        is sentinel
    )
    assert captured == {
        "api_key": "environment-secret",
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-v4-flash",
        "max_retries": 0,
        "supports_json_schema": False,
    }
    assert "tom_reporter_ab_path" not in {
        action.dest
        for action in build_arg_parser()._actions
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"suspected_werewolves":[]}', []),
        ('{"suspected_werewolves":["player4"]}', ["player4"]),
        (
            '{"suspected_werewolves":["player7","player1","player4"]}',
            ["player1", "player4", "player7"],
        ),
    ],
)
def test_reporter_valid_suspicion_sets_remain_canonical(raw, expected):
    assert BeliefReporter.parse(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        '{"suspected_werewolves":["player3","player3"]}',
        '{"suspected_werewolves":["player8"]}',
        '{"suspected_werewolves":[],"extra":true}',
    ],
)
def test_reporter_parser_remains_strict(raw):
    with pytest.raises((TypeError, ValueError)):
        BeliefReporter.parse(raw)


@pytest.mark.parametrize(("guard", "context"), [(False, "seer_witch"), (True, "seer_guard")])
def test_post_speech_observations_are_private_aware_for_both_configs(
    tmp_path, guard, context
):
    env = ready_env(guard=guard)
    agent_list = agents()
    collector, reporter_backend = make_collector(tmp_path, env, agent_list)

    env.step(("speech", "CURRENT-SPEECH"))
    sample = collector.record(
        env, step_idx=9, round_number=1, phase="speech", speaker_id=1
    )
    collector.close()

    assert sample["episode_context"] == context
    assert sample["formal_speech_actions"] == FORMAL_ACTION
    assert "CURRENT-SPEECH" in prompt_for(reporter_backend, 0)
    assert "FUTURE-SPEECH" not in prompt_for(reporter_backend, 0)
    assert "werewolf_team_info" in prompt_for(reporter_backend, 0)
    assert "werewolf_team_info" not in prompt_for(reporter_backend, 2)
    assert "skill_seer" in prompt_for(reporter_backend, 2)
    assert "skill_seer" not in prompt_for(reporter_backend, 4)
    private_event = "skill_guard" if guard else "skill_witch"
    assert private_event in prompt_for(reporter_backend, 3)
    assert private_event not in prompt_for(reporter_backend, 4)
    assert "god_view" not in "".join(
        prompt_for(reporter_backend, index)
        for index in range(7)
    )
    assert all(agent.backend.calls == [] for agent in agent_list)
    assert sample["public_events"][-1]["event_type"] == "public_speech"
    assert all(event["event_type"] != "turn_start" for event in sample["public_events"][-1:])


def test_alive_only_queries_post_speech_alive_observers(tmp_path):
    env = ready_env()
    env.alive = [1, 0, 1, 1, 0, 1, 1]
    agent_list = agents()
    collector, reporter_backend = make_collector(tmp_path, env, agent_list)
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
    assert len(reporter_backend.calls) == 5
    assert all(agent.backend.calls == [] for agent in agent_list)


def test_zero_triplet_writes_nothing_and_makes_no_reporter_call(tmp_path):
    env = ready_env(actions=[])
    agent_list = agents()
    output = tmp_path / "zero.jsonl"
    collector, reporter_backend = make_collector(
        tmp_path,
        env,
        agent_list,
        name="zero.jsonl",
    )
    env.step(("speech", "NO-FORMAL-ACTION"))

    result = collector.record(
        env, step_idx=1, round_number=1, phase="speech", speaker_id=1
    )
    collector.close()

    assert result is None
    assert collector.samples_written == 0
    assert output.read_text() == ""
    assert reporter_backend.calls == []
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
    collector, reporter_backend = make_collector(tmp_path, env, agent_list)
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
    assert len(reporter_backend.calls) == 7
    assert all(agent.backend.calls == [] for agent in agent_list)


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
    collector, _reporter_backend = make_collector(tmp_path, env, agent_list)
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
    reporter_backend = Backend()
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
        episode_context="seer_witch", reporter=BeliefReporter(reporter_backend),
    )
    sample = collector.record(
        BoundaryEnv(), step_idx=1, round_number=1, phase="speech", speaker_id=1
    )
    collector.close()

    assert sample["alive_observers"] == ["player6"]
    assert "TRUTH-CANARY" not in prompt_for(reporter_backend, 0)
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
    agent_list = agents({1: '{"suspected_werewolves":["player1","player2"]}'})
    collector, reporter_backend = make_collector(tmp_path, env, agent_list)
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
        "player1", "player2"
    ]
    assert len(reporter_backend.calls) == 7
    assert all(agent.backend.calls == [] for agent in agent_list)
    serialized = json.dumps(sample)
    assert "true_roles" not in serialized
    assert "agent internal" not in serialized
