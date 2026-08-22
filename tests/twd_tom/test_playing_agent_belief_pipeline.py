from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
from werewolf.models.twd_tom.public_events import public_speech_actions
from werewolf.agents.llm_agent import LLMAgent
from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
from werewolf.models.twd_tom.collector import TWDToMSampleCollector
from werewolf.models.twd_tom.dataset import TWDToMDataset
from werewolf.models.twd_tom.belief_snapshot import (
    PlayingAgentBeliefSnapshotCollector,
)
from werewolf.speech.private_belief_perceiver import PlayingAgentBeliefReporter


class FakeBackend:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        return self.response


class ScriptedSpeechPerceiver:
    def __init__(self):
        self.calls = []

    def parse(self, *, speaker, speech, day, phase):
        self.calls.append(
            {
                "speaker": speaker,
                "speech": speech,
                "day": day,
                "phase": phase,
            }
        )
        other_players = [
            f"player{player_id}"
            for player_id in range(1, 8)
            if player_id != speaker
        ]
        if speech == "first synthetic public speech":
            return [
                [f"player{speaker}", "point_as_werewolf", other_players[0]],
                [f"player{speaker}", "support", other_players[1]],
                [f"player{speaker}", "oppose", other_players[2]],
            ]
        return [
            [f"player{speaker}", "point_as_villager", other_players[0]],
        ]


def _log(event, *, source=0, target=0, content=None):
    return SimpleNamespace(
        event=event,
        source=source,
        target=target,
        content={} if content is None else content,
        time="第1天白天",
    )


class SyntheticEnvironment:
    def __init__(self):
        self.game_log = []
        self.public_events = [
            {
                "event_idx": 0,
                "event_type": "phase_change",
                "phase": "1_day_speech",
            },
            {
                "event_idx": 1,
                "event_type": "turn_start",
                "speaker": "player3",
            },
        ]
        self.private_logs = {
            player_id: [
                _log("game_setting", content={"Werewolf": 2, "Villager": 5}),
                _log(
                    "self_identity",
                    target=player_id,
                    content={"identity": "Villager"},
                ),
            ]
            for player_id in range(1, 8)
        }

    def publish_speech(self):
        action = ["player3", "point_as_werewolf", "player6"]
        self.game_log.append({
            "event": "speech",
            "content": {
                "speech_content": "public speech",
                "sp_actions": [action],
            },
        })
        self.public_events.append(
            {
                "event_idx": len(self.public_events),
                "event_type": "public_speech",
                "speaker": "player3",
                "raw_text": "public speech",
                "sp_actions": [action],
            }
        )
        for logs in self.private_logs.values():
            logs.append(
                _log(
                    "speech",
                    source=3,
                    content={"speech_content": "public speech", "sp_actions": [action]},
                )
            )

    def get_observation_for(self, player_id):
        return {
            "observer_id": player_id,
            "current_act_idx": 3,
            "identity": "Villager",
            "phase": "1_day_speech",
            "valid_action": [],
            "game_log": deepcopy(self.private_logs[player_id]),
        }

    def get_twd_tom_hard_knowledge_for(self, player_id):
        return [], [f"player{player_id}"]


def _collect_sample(tmp_path, game_id):
    agents = []
    backends = []
    for player_id in range(1, 8):
        suspicions = {
            1: [],
            2: ["player1"],
            3: ["player1", "player2", "player6"],
            4: ["player1", "player2", "player3", "player5", "player6", "player7"],
            5: [],
            6: ["player1"],
            7: ["player1", "player2", "player3"],
        }[player_id]
        payload = {"suspected_werewolves": suspicions}
        backend = FakeBackend(json.dumps(payload))
        agent = LLMAgent(backend=backend, model_name=f"fake-model-{player_id}")
        agent.backend_id = f"fake-backend-{player_id}"
        agents.append(agent)
        backends.append(backend)

    env = SyntheticEnvironment()
    path = tmp_path / f"{game_id}.jsonl"
    snapshot_collector = PlayingAgentBeliefSnapshotCollector(
        PlayingAgentBeliefReporter(), agents
    )
    with TWDToMSampleCollector(
        str(path), snapshot_collector, game_id=game_id
    ) as collector:
        sample = collector.record(
            env,
            step_idx=1,
            trigger="speech",
            phase="1_day_speech",
            speaker_id=3,
            observer_ids=list(range(1, 8)),
        )
    return sample, path, agents, backends


def test_synthetic_collector_writes_only_player_level_suspicion(tmp_path):
    sample, path, agents, backends = _collect_sample(tmp_path, "game_train")
    assert sample["public_action_count"] == 0
    assert sample["observer_ids"] == list(range(1, 8))
    assert all(len(backend.calls) == 1 for backend in backends)
    assert all(
        backend.calls[0]["model"] == agent.model_name
        for backend, agent in zip(backends, agents)
    )
    serialized = path.read_text(encoding="utf-8")
    assert "belief_mode" not in serialized
    assert "believed_werewolves" not in serialized
    assert not any(
        forbidden in serialized
        for forbidden in (
            "actual_roles",
            "god_view",
            "private_observation",
            "self_identity",
            "private_logs",
        )
    )
    assert sample["schema_version"] == (
        "classic7_pre_speech_player_suspicion_v2"
    )
    assert {
        len(suspicion)
        for suspicion in sample["suspected_werewolves"].values()
        if suspicion is not None
    } >= {0, 1, 3}
    assert sample["belief_status"]["player4"] == "ok"
    assert sample["suspected_werewolves"]["player4"] == [
        "player1", "player2", "player3", "player5", "player6", "player7"
    ]
    assert sample["belief_errors"]["player4"] is None
    assert "pair_target" not in serialized
    assert "pair_support" not in serialized
    dataset = TWDToMDataset.from_jsonl(path)
    item = dataset[0]
    assert item["belief_targets"].shape == (7, 7)
    assert item["observer_alive_mask"].all()
    assert "pair_targets" not in item

def test_two_pre_speech_snapshots_flow_through_real_raw_collector(
    tmp_path,
    monkeypatch,
):
    roles = [
        "Werewolf",
        "Werewolf",
        "Seer",
        "Witch",
        "Villager",
        "Villager",
        "Villager",
    ]
    perceiver = ScriptedSpeechPerceiver()
    env = WerewolfTextEnvV0(
        log_save_path=None,
        speech_perceiver=perceiver,
    )
    monkeypatch.setattr(
        "werewolf.envs.werewolf_text_env_v0.random.randint",
        lambda _start, _end: 0,
    )
    env.reset(roles=roles)
    env.step(("kill", 5))
    env.step(("kill", 5))
    env.step(("check", 1))
    env.step(("witch_pass", 0))
    assert env.phase == "speech"

    agents = []
    backends = []
    responses = {
        1: '{"suspected_werewolves":["player2","player3"]}',
        2: '{"suspected_werewolves":["player1"]}',
        3: '{"suspected_werewolves":["player1"]}',
        4: '{"suspected_werewolves":[]}',
        5: '{"suspected_werewolves":[]}',
        6: '{"suspected_werewolves":["player1","player2","player3"]}',
        7: '{"belief_mode":"no_extra_narrowing"}',
    }
    for player_id in range(1, 8):
        backend = FakeBackend(responses[player_id])
        agent = LLMAgent(
            backend=backend,
            model_name=f"fake-model-{player_id}",
        )
        agent.backend_id = f"fake-backend-{player_id}"
        agents.append(agent)
        backends.append(backend)

    alive_observers = [
        player_id
        for player_id, alive in enumerate(env.alive, start=1)
        if alive
    ]
    assert alive_observers == [1, 2, 3, 4, 6, 7]

    path = tmp_path / "pre_speech_canary.jsonl"
    snapshot_collector = PlayingAgentBeliefSnapshotCollector(
        PlayingAgentBeliefReporter(),
        agents,
    )
    with TWDToMSampleCollector(
        str(path),
        snapshot_collector,
        game_id="synthetic_pre_speech_canary",
    ) as collector:
        first_speaker = env.current_act_idx + 1
        first = collector.record(
            env,
            step_idx=0,
            trigger="speech",
            phase=env.get_observation()["phase"],
            speaker_id=first_speaker,
            observer_ids=alive_observers,
        )
        assert public_speech_actions(first["public_events"]) == []
        assert first["public_action_count"] == 0
        assert [
            len(backend.calls)
            for backend in backends
        ] == [1, 1, 1, 1, 0, 1, 1]

        env.step(("speech", "first synthetic public speech"))
        first_actions = env.game_log[-1].content["sp_actions"]
        assert [action[1] for action in first_actions] == [
            "point_as_werewolf",
            "support",
            "oppose",
        ]

        second_speaker = env.current_act_idx + 1
        second = collector.record(
            env,
            step_idx=1,
            trigger="speech",
            phase=env.get_observation()["phase"],
            speaker_id=second_speaker,
            observer_ids=alive_observers,
        )
        assert public_speech_actions(second["public_events"]) == first_actions
        assert second["public_action_count"] == 3
        assert "second synthetic public speech" not in json.dumps(
            second,
            ensure_ascii=False,
        )
        assert [
            len(backend.calls)
            for backend in backends
        ] == [2, 2, 2, 2, 0, 2, 2]

        env.step(("speech", "second synthetic public speech"))

    raw_samples = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(raw_samples) == 2
    assert {
        sample["schema_version"]
        for sample in raw_samples
    } == {"classic7_pre_speech_player_suspicion_v2"}
    assert public_speech_actions(raw_samples[0]["public_events"]) == []
    assert public_speech_actions(raw_samples[1]["public_events"]) == first_actions
    assert raw_samples[0]["observer_ids"] == alive_observers
    assert raw_samples[1]["observer_ids"] == alive_observers
    assert all(
        sample["belief_status"]["player7"] == "parse_error"
        and sample["suspected_werewolves"]["player7"] is None
        for sample in raw_samples
    )
    assert {
        len(suspicion)
        for sample in raw_samples
        for suspicion in sample["suspected_werewolves"].values()
        if suspicion is not None
    } >= {0, 1, 2, 3}

    assert not any(
        private_term
        in json.dumps(
            public_speech_actions(raw_samples[1]["public_events"])
        )
        for private_term in (
            "known_werewolves",
            "known_non_werewolves",
            "skill_seer",
            "kill_decision",
            "Werewolf",
            "Seer",
            "Witch",
        )
    )

    serialized = path.read_text(encoding="utf-8")
    assert "pair_target" not in serialized
    assert "pair_support" not in serialized
    with pytest.raises(ValueError, match="status=ok for every alive observer"):
        TWDToMDataset.from_jsonl(path)
