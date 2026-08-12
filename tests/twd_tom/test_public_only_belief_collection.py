import inspect
import json

from run_random import (
    PUBLIC_ONLY_COLLECTION_MODE,
    build_twd_tom_sample_collector,
)
from werewolf.models.twd_tom.belief_snapshot import (
    PlayingAgentBeliefSnapshotCollector,
    PublicOnlyBeliefSnapshotCollector,
)
from werewolf.models.twd_tom.public_events import public_event_digest
from werewolf.models.twd_tom.samples import (
    PUBLIC_ONLY_SAMPLE_SCHEMA_VERSION,
)
from werewolf.models.twd_tom.schema import (
    PUBLIC_ONLY_LABEL_PROMPT_VERSION,
    PUBLIC_ONLY_LABEL_PROVENANCE,
)
from werewolf.speech.private_belief_perceiver import PlayingAgentBeliefReporter
from werewolf.speech.public_belief_perceiver import PublicOnlyBeliefReporter


class CapturingBackend:
    supports_json_schema = True

    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return '{"suspected_werewolves":[]}'


class FakeAgent:
    def __init__(self, player_id, backend):
        self.backend = backend
        self.backend_id = f"backend-{player_id}"
        self.model_name = "public-test-model"
        self.identity_canary = f"ROLE-CANARY-{player_id}"
        self.notes = {"secret": f"NOTES-CANARY-{player_id}"}
        self.vote_reason = {"secret": f"VOTE-CANARY-{player_id}"}

    def report_suspected_werewolves_readonly(self, **_kwargs):
        return '{"suspected_werewolves":[]}'


class CutoffEnv:
    def __init__(self):
        self.public_events = [
            {
                "event_idx": 0,
                "event_type": "phase_change",
                "phase": "1_day_speech",
            },
            {
                "event_idx": 1,
                "event_type": "public_speech",
                "speaker": "player7",
                "raw_text": "PUBLIC-CANARY-SPEECH",
                "sp_actions": [["player7", "oppose", "player2"]],
            },
            {
                "event_idx": 2,
                "event_type": "turn_start",
                "speaker": "player3",
            },
        ]

    def get_observation_for(self, player_id):
        return {
            "observer_id": player_id,
            "current_act_idx": 3,
            "phase": "1_day_speech",
            "identity": "Villager",
            "game_log": [],
        }

    def get_twd_tom_hard_knowledge_for(self, player_id):
        return [], [f"player{player_id}"]


class PrivateAccessForbiddenEnv:
    def get_observation_for(self, _player_id):
        raise AssertionError("public-only collector requested private observation")

    def get_twd_tom_hard_knowledge_for(self, _player_id):
        raise AssertionError("public-only collector requested hard knowledge")


def _agents():
    backends = [CapturingBackend() for _ in range(7)]
    return [
        FakeAgent(player_id, backends[player_id - 1])
        for player_id in range(1, 8)
    ], backends


def _record(collector, env):
    return collector.record(
        env,
        step_idx=11,
        trigger="speech",
        phase="1_day_speech",
        speaker_id=3,
        observer_ids=(1, 3, 7),
    )


def test_default_builder_keeps_original_private_reporter_stack(tmp_path):
    agents, _backends = _agents()
    collector = build_twd_tom_sample_collector(
        agent_list=agents,
        output_path=str(tmp_path / "private.jsonl"),
        game_id="same-game",
    )
    try:
        assert isinstance(
            collector.snapshot_collector,
            PlayingAgentBeliefSnapshotCollector,
        )
        assert isinstance(
            collector.snapshot_collector.reporter,
            PlayingAgentBeliefReporter,
        )
    finally:
        collector.close()


def test_public_reporter_api_and_context_exclude_private_canaries(tmp_path):
    forbidden_parameters = {
        "agent",
        "observation",
        "role",
        "identity",
        "known_werewolves",
        "known_non_werewolves",
        "env",
    }
    assert forbidden_parameters.isdisjoint(
        inspect.signature(PublicOnlyBeliefReporter.report).parameters
    )

    agents, backends = _agents()
    collector = build_twd_tom_sample_collector(
        agent_list=agents,
        output_path=str(tmp_path / "public.jsonl"),
        game_id="same-game",
        collection_mode=PUBLIC_ONLY_COLLECTION_MODE,
    )
    try:
        assert isinstance(
            collector.snapshot_collector,
            PublicOnlyBeliefSnapshotCollector,
        )
        public_env = CutoffEnv()
        sample = _record(collector, public_env)
    finally:
        collector.close()

    request_text = json.dumps(
        [backend.calls for backend in backends],
        ensure_ascii=False,
    )
    assert "PUBLIC-CANARY-SPEECH" in request_text
    for canary in (
        "ROLE-CANARY",
        "NOTES-CANARY",
        "VOTE-CANARY",
    ):
        assert canary not in request_text
    assert all(
        len(backends[player_id - 1].calls) == (1 if player_id in {1, 3, 7} else 0)
        for player_id in range(1, 8)
    )
    assert sample["schema_version"] == PUBLIC_ONLY_SAMPLE_SCHEMA_VERSION
    assert sample["label_prompt_version"] == PUBLIC_ONLY_LABEL_PROMPT_VERSION
    assert sample["label_provenance"] == PUBLIC_ONLY_LABEL_PROVENANCE
    assert all(not values for values in sample["known_werewolves"].values())
    assert all(not values for values in sample["known_non_werewolves"].values())


def test_public_snapshot_collector_never_reads_private_environment():
    agents, _backends = _agents()
    dispatches = [
        {
            "backend": agent.backend,
            "backend_id": agent.backend_id,
            "model_name": agent.model_name,
        }
        for agent in agents
    ]
    collector = PublicOnlyBeliefSnapshotCollector(
        PublicOnlyBeliefReporter(),
        dispatches,
    )
    env = CutoffEnv()
    from werewolf.models.twd_tom.samples import freeze_public_snapshot

    snapshot = freeze_public_snapshot(
        game_id="same-game",
        step_idx=11,
        phase="1_day_speech",
        speaker_id=3,
        report_trigger="pre_public_speech",
        observer_ids=(1, 3, 7),
        public_events=env.public_events,
    )
    reports = collector.collect(snapshot, env=PrivateAccessForbiddenEnv())
    assert set(reports) == {"player1", "player3", "player7"}


def test_private_and_public_lines_freeze_the_identical_cutoff(tmp_path):
    env = CutoffEnv()
    private_agents, _ = _agents()
    public_agents, _ = _agents()
    private = build_twd_tom_sample_collector(
        agent_list=private_agents,
        output_path=str(tmp_path / "private.jsonl"),
        game_id="same-game",
    )
    public = build_twd_tom_sample_collector(
        agent_list=public_agents,
        output_path=str(tmp_path / "public.jsonl"),
        game_id="same-game",
        collection_mode=PUBLIC_ONLY_COLLECTION_MODE,
    )
    try:
        private_sample = _record(private, env)
        public_sample = _record(public, env)
    finally:
        private.close()
        public.close()

    for field in (
        "game_id",
        "step_idx",
        "phase",
        "speaker_id",
        "public_event_digest",
        "structured_input_digest",
    ):
        assert private_sample[field] == public_sample[field]
    assert public_sample["public_event_digest"] == public_event_digest(
        env.public_events
    )
