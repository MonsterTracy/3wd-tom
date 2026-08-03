import json
from copy import deepcopy

from script.twd_tom.collect import _write_audit_manifest
from werewolf.agents.llm_agent import LLMAgent
from werewolf.models.twd_tom.belief_snapshot import (
    PlayingAgentBeliefSnapshotCollector,
)
from werewolf.models.twd_tom.collector import TWDToMSampleCollector
from werewolf.models.twd_tom.schema import canonical_wolf_pairs
from werewolf.speech.pair_belief_self_reporter import (
    ReadonlyPairBeliefSelfReporter,
)


class FakeBackend:
    supports_json_schema = True

    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        return self.response


class SyntheticEnvironment:
    def __init__(self):
        self.public_events = [
            {"event_idx": 0, "event_type": "phase_change", "phase": "1_day_speech"},
            {"event_idx": 1, "event_type": "turn_start", "speaker": "player3"},
        ]

    def get_observation_for(self, player_id):
        return {
            "observer_id": player_id,
            "current_act_idx": 3,
            "identity": "Villager",
            "game_log": [],
            "phase": "1_day_speech",
            "valid_action": [] if player_id != 3 else ("speech", -1),
        }

    def get_twd_tom_hard_knowledge_for(self, player_id):
        return [], [f"player{player_id}"]


def _target(player_id):
    player = f"player{player_id}"
    return [
        1.0 / 15 if player not in pair else 0.0
        for pair in canonical_wolf_pairs()
    ]


def _provenance():
    return {
        "generator_name": "twd_tom_actor_pair_belief_collector",
        "generator_version": "1",
        "git_commit_sha": "a" * 40,
        "git_worktree_clean": True,
        "collection_timestamp_utc": "2026-08-03T00:00:00+00:00",
        "game_seed": 42,
        "source_config_path": "configs/twd_tom_server_qwen25_7b.yaml",
        "source_config_sha256": "b" * 64,
        "resolved_runtime_config_sha256": "c" * 64,
        "resolved_backend_config_sha256": {"backend": "d" * 64},
    }


def test_direct_pair_reports_flow_through_the_only_raw_collector(tmp_path):
    agents = []
    backends = []
    for player_id in range(1, 8):
        backend = FakeBackend(
            json.dumps({"pair_probabilities": _target(player_id)})
        )
        agent = LLMAgent(backend=backend, model_name=f"model-{player_id}")
        agent.backend_id = f"backend-{player_id}"
        agents.append(agent)
        backends.append(backend)

    output = tmp_path / "raw.jsonl"
    stack = PlayingAgentBeliefSnapshotCollector(
        ReadonlyPairBeliefSelfReporter(),
        agents,
    )
    with TWDToMSampleCollector(
        str(output),
        stack,
        game_id="game_seed_42",
        collection_provenance=_provenance(),
    ) as collector:
        sample = collector.record(
            SyntheticEnvironment(),
            step_idx=0,
            trigger="speech",
            phase="1_day_speech",
            speaker_id=3,
            observer_ids=range(1, 8),
        )

    assert sample["reasoning_player_id"] == "player3"
    assert len(sample["player_reports"]) == 7
    assert all(report["report_status"] == "ok" for report in sample["player_reports"])
    assert all(len(report["pair_probabilities"]) == 21 for report in sample["player_reports"])
    assert all(len(backend.calls) == 1 for backend in backends)
    assert "suspected_werewolves" not in output.read_text(encoding="utf-8")


def test_current_unfinished_action_and_future_events_are_not_in_payload(tmp_path):
    env = SyntheticEnvironment()
    backend = FakeBackend(json.dumps({"pair_probabilities": _target(3)}))
    agents = []
    for player_id in range(1, 8):
        agent = LLMAgent(backend=backend, model_name="model")
        agent.backend_id = "backend"
        agents.append(agent)
    stack = PlayingAgentBeliefSnapshotCollector(
        ReadonlyPairBeliefSelfReporter(), agents
    )
    with TWDToMSampleCollector(
        str(tmp_path / "raw.jsonl"),
        stack,
        game_id="g",
        collection_provenance=_provenance(),
    ) as collector:
        sample = collector.record(
            env,
            step_idx=0,
            trigger="speech",
            phase="1_day_speech",
            speaker_id=3,
            observer_ids=[3],
        )
        env.public_events.append(
            {
                "event_idx": 2,
                "event_type": "public_speech",
                "speaker": "player3",
                "raw_text": "future current action",
                "sp_actions": [],
            }
        )

    serialized_payload = json.dumps(
        sample["player_reports"][2]["reporter_input_payload"]
    )
    assert "future current action" not in serialized_payload
    assert sample["public_events"][-1]["event_type"] == "turn_start"
    assert sample["current_action_used"] is False
    assert sample["future_information_used"] is False


def test_collection_manifest_names_direct_actor_pair_contract(tmp_path):
    path = _write_audit_manifest(
        log_save_path=tmp_path,
        game_id="g",
        roles=["Werewolf", "Werewolf", "Seer", "Witch", "Villager", "Villager", "Villager"],
        role2agent_list=["server"] * 7,
        sample_path=tmp_path / "raw.jsonl",
        random_seed=42,
        collection_git_state={
            "git_commit_sha": "a" * 40,
            "git_worktree_clean": True,
        },
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["raw_label_field"] == "pair_probabilities"
    assert manifest["reasoning_perspective"] == "current_speaker"
    assert manifest["expert_completion"] is False
    assert manifest["git_commit_sha"] == "a" * 40
    assert manifest["git_worktree_clean"] is True
