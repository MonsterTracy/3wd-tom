from copy import deepcopy

import pytest

from werewolf.agents.llm_agent import LLMAgent
from werewolf.models.twd_tom.belief_snapshot import (
    PlayingAgentBeliefSnapshotCollector,
)
from werewolf.models.twd_tom.samples import freeze_public_snapshot
from werewolf.models.twd_tom.schema import canonical_wolf_pairs
from werewolf.speech.pair_belief_self_reporter import (
    ReadonlyPairBeliefSelfReporter,
)


class FakeAgent:
    def __init__(self, backend_id):
        self.backend_id = backend_id
        self.messages = ["unchanged"]
        self.memory = {"private": backend_id}
        self.state = {"turn": 1}


class FakeEnv:
    def __init__(self):
        self.private = {
            player_id: {
                "observer_id": player_id,
                "identity": "Villager",
                "game_log": [f"private-{player_id}"],
                "current_act_idx": 3,
                "phase": "1_day_speech_pk",
            }
            for player_id in range(1, 8)
        }

    def get_observation_for(self, player_id):
        return deepcopy(self.private[player_id])

    def get_twd_tom_hard_knowledge_for(self, player_id):
        return [], [f"player{player_id}"]


class FakeReporter:
    def __init__(self, mutate=False):
        self.calls = []
        self.mutate = mutate

    def report(self, **kwargs):
        self.calls.append(deepcopy({
            "belief_owner_id": kwargs["belief_owner_id"],
            "observation": kwargs["observation"],
            "backend_id": kwargs["backend_alias"],
            "snapshot_id": id(kwargs["public_snapshot"]),
        }))
        if self.mutate:
            kwargs["agent"].memory["changed"] = True
        return {
            "player_id": kwargs["belief_owner_id"],
            "report_status": "ok",
            "pair_probabilities": [
                1.0 / 15 if f"player{kwargs['observation']['observer_id']}" not in pair else 0.0
                for pair in canonical_wolf_pairs()
            ],
            "known_werewolves": kwargs["known_werewolves"],
            "known_non_werewolves": kwargs["known_non_werewolves"],
            "report_error": None,
            "backend_alias": kwargs["backend_alias"],
        }


def _snapshot(observers=(1, 3, 7)):
    return freeze_public_snapshot(
        game_id="game_001",
        step_idx=4,
        phase="1_day_speech_pk",
        speaker_id=3,
        report_trigger="pre_public_speech_pk",
        observer_ids=observers,
        public_events=[
            {
                "event_idx": 0,
                "event_type": "phase_change",
                "phase": "1_day_speech_pk",
            },
            {
                "event_idx": 1,
                "event_type": "public_speech",
                "speaker": "player7",
                "raw_text": "earlier speech",
                "sp_actions": [["player3", "oppose", "player7"]],
            },
            {
                "event_idx": 2,
                "event_type": "turn_start",
                "speaker": "player3",
            },
        ],
    )


def test_all_observers_use_one_snapshot_and_only_their_private_view():
    agents = [FakeAgent(f"backend_{i}") for i in range(1, 8)]
    before = deepcopy([vars(agent) for agent in agents])
    reporter = FakeReporter()
    reports = PlayingAgentBeliefSnapshotCollector(reporter, agents).collect(
        _snapshot(), env=FakeEnv()
    )

    assert set(reports) == {"player1", "player3", "player7"}
    assert len({call["snapshot_id"] for call in reporter.calls}) == 1
    for call in reporter.calls:
        player_id = int(call["belief_owner_id"][6:])
        assert call["observation"]["game_log"] == [f"private-{player_id}"]
        assert call["backend_id"] == f"backend_{player_id}"
    assert [vars(agent) for agent in agents] == before


def test_readonly_state_mutation_is_detected_and_not_written_as_success():
    agents = [FakeAgent(f"backend_{i}") for i in range(1, 8)]
    collector = PlayingAgentBeliefSnapshotCollector(FakeReporter(mutate=True), agents)
    with pytest.raises(RuntimeError, match="mutated player1 agent state"):
        collector.collect(_snapshot((1,)), env=FakeEnv())


def test_reporter_mutates_only_detached_observation_and_context():
    target = [
        1.0 / 15 if "player1" not in pair else 0.0
        for pair in canonical_wolf_pairs()
    ]
    response = __import__("json").dumps({"pair_probabilities": target})

    class MutatingContextBackend:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(deepcopy(kwargs))
            kwargs["messages"][0]["content"] = "mutated detached context"
            kwargs["messages"].append({"role": "user", "content": "detached"})
            return response

    class MutatingDetachedObservationAgent(LLMAgent):
        def _build_readonly_belief_context(self, observation):
            observation["detached_probe"] = "mutated"
            self.detached_probe = "mutated"
            return super()._build_readonly_belief_context(observation)

    backend = MutatingContextBackend()
    agent = MutatingDetachedObservationAgent(
        backend=backend,
        model_name="fake-model",
    )
    agent.backend_id = "backend_1"
    agent.notes = ["original memory"]
    observation = {
        "observer_id": 1,
        "current_act_idx": 3,
        "identity": "Villager",
        "game_log": [],
        "phase": "1_day_speech_pk",
        "valid_action": [],
    }
    observation_before = deepcopy(observation)
    snapshot = _snapshot((1,))

    result = ReadonlyPairBeliefSelfReporter().report(
        agent=agent,
        observation=observation,
        belief_owner_id="player1",
        public_snapshot=snapshot,
        backend_alias="backend_1",
        known_werewolves=[],
        known_non_werewolves=["player1"],
    )

    assert result["report_status"] == "ok"
    assert observation == observation_before
    assert agent.notes == ["original memory"]
    assert not hasattr(agent, "detached_probe")
    assert snapshot == _snapshot((1,))
    current_speaker_prompt = agent.format_observation(observation)
    assert response not in current_speaker_prompt


def test_snapshot_collector_requires_exactly_seven_playing_agents():
    with pytest.raises(ValueError, match="exactly seven"):
        PlayingAgentBeliefSnapshotCollector(FakeReporter(), [FakeAgent("x")])
