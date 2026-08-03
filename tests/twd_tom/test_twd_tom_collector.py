import inspect
import hashlib
import json

import pytest

from werewolf.models.twd_tom.collector import TWDToMSampleCollector
from werewolf.models.twd_tom.samples import (
    ACTOR_PAIR_BELIEF_SCHEMA_VERSION,
    PublicSnapshot,
)
from werewolf.models.twd_tom.schema import canonical_wolf_pairs
from werewolf.speech.pair_belief_self_reporter import (
    PAIR_BELIEF_PROMPT_VERSION,
    canonical_json_sha256,
)


class Env:
    def __init__(self):
        self.public_events = [
            {"event_idx": 0, "event_type": "phase_change", "phase": "1_day_speech"},
            {
                "event_idx": 1,
                "event_type": "public_speech",
                "speaker": "player1",
                "raw_text": "earlier",
                "sp_actions": [["player3", "point_as_werewolf", "player7"]],
            },
            {"event_idx": 2, "event_type": "turn_start", "speaker": "player3"},
        ]


def _provenance():
    return {
        "generator_name": "twd_tom_actor_pair_belief_collector",
        "generator_version": "1",
        "git_commit_sha": "a" * 40,
        "git_worktree_clean": True,
        "collection_timestamp_utc": "2026-08-03T00:00:00+00:00",
        "game_seed": 1,
        "source_config_path": "configs/twd_tom_server_qwen25_7b.yaml",
        "source_config_sha256": "b" * 64,
        "resolved_runtime_config_sha256": "c" * 64,
        "resolved_backend_config_sha256": {"backend": "e" * 64},
    }


def _report(player, *, status="ok"):
    target = [
        1.0 / 15 if player not in pair else 0.0
        for pair in canonical_wolf_pairs()
    ]
    payload = {
        "payload_version": "readonly_pair_belief_self_report_payload_v1",
        "messages": [{"role": "user", "content": f"private-{player}"}],
        "request": {"temperature": 0.0, "max_tokens": 256},
    }
    return {
        "player_id": player,
        "report_status": status,
        "report_error": None if status == "ok" else "synthetic invalid report",
        "pair_probabilities": target if status == "ok" else None,
        "known_werewolves": [],
        "known_non_werewolves": [player],
        "reporter_input_payload": payload,
        "reporter_input_payload_sha256": canonical_json_sha256(payload),
        "raw_reporter_output": json.dumps({"pair_probabilities": target}),
        "parsed_output": {"pair_probabilities": target},
        "hard_knowledge_validation": {"status": "valid" if status == "ok" else "invalid"},
        "report_provenance": "playing_agent_readonly_direct_pair_belief_self_report_v1",
        "backend_alias": f"backend-{player}",
        "resolved_model_name": f"model-{player}",
        "prompt_version": PAIR_BELIEF_PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(
            payload["messages"][-1]["content"].encode("utf-8")
        ).hexdigest(),
        "parser_version": "strict_pair_probability_json_v1",
        "sampling_parameters": {"temperature": 0.0, "max_tokens": 256},
        "reporter_seed": None,
    }


class SnapshotCollector:
    def __init__(self):
        self.snapshots = []

    def collect(self, snapshot, *, env):
        assert isinstance(snapshot, PublicSnapshot)
        self.snapshots.append(snapshot)
        return {
            "player1": _report("player1", status="parse_error"),
            "player3": _report("player3"),
        }


def _collector(path, snapshot_collector):
    return TWDToMSampleCollector(
        str(path),
        snapshot_collector,
        game_id="g1",
        collection_provenance=_provenance(),
    )


def test_collector_freezes_then_writes_direct_pair_jsonl(tmp_path):
    path = tmp_path / "sample.jsonl"
    snapshot_collector = SnapshotCollector()
    with _collector(path, snapshot_collector) as collector:
        sample = collector.record(
            Env(),
            step_idx=2,
            trigger="speech",
            phase="1_day_speech",
            speaker_id=3,
            observer_ids=[1, 3],
        )
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record == sample
    assert record["schema_version"] == ACTOR_PAIR_BELIEF_SCHEMA_VERSION
    assert record["current_speaker"] == record["reasoning_player_id"] == "player3"
    assert record["event_idx"] == record["public_history_cutoff_event_idx"] == 2
    assert record["player_reports"][0]["pair_probabilities"] is None
    assert record["player_reports"][2]["pair_probabilities"] is not None
    assert "suspected_werewolves" not in json.dumps(record)
    assert len(snapshot_collector.snapshots) == 1


def test_collector_uses_exact_pre_speech_history(tmp_path):
    env = Env()
    snapshots = SnapshotCollector()
    with _collector(tmp_path / "sample.jsonl", snapshots) as collector:
        collector.record(
            env,
            step_idx=1,
            trigger="speech",
            phase="1_day_speech",
            speaker_id=3,
            observer_ids=[1, 3],
        )
        env.public_events[-1] = {
            "event_idx": 2,
            "event_type": "public_speech",
            "speaker": "player3",
            "raw_text": "current speech",
            "sp_actions": [["player4", "oppose", "player3"]],
        }
    assert snapshots.snapshots[0].public_events[-1]["event_type"] == "turn_start"


def test_collector_context_manager_and_closed_write(tmp_path):
    collector = _collector(tmp_path / "x.jsonl", SnapshotCollector())
    collector.close()
    assert collector.closed
    with pytest.raises(RuntimeError, match="closed"):
        collector.write({})


def test_collector_requires_snapshot_collector_game_id_and_provenance(tmp_path):
    with pytest.raises(ValueError, match="required"):
        TWDToMSampleCollector(
            str(tmp_path / "x.jsonl"),
            None,
            game_id="g1",
            collection_provenance=_provenance(),
        )
    with pytest.raises(ValueError, match="game_id"):
        TWDToMSampleCollector(
            str(tmp_path / "x.jsonl"),
            SnapshotCollector(),
            game_id="",
            collection_provenance=_provenance(),
        )


def test_collector_rejects_dirty_provenance_before_opening_output(tmp_path):
    provenance = _provenance()
    provenance["git_worktree_clean"] = False
    output = tmp_path / "not-created" / "samples.jsonl"
    with pytest.raises(RuntimeError, match="git_worktree_clean=true"):
        TWDToMSampleCollector(
            str(output),
            SnapshotCollector(),
            game_id="game_001",
            collection_provenance=provenance,
        )
    assert not output.parent.exists()


def test_snapshot_collector_failure_propagates_without_writing(tmp_path):
    class FailingSnapshotCollector:
        def collect(self, snapshot, *, env):
            raise RuntimeError("report collection failed")

    path = tmp_path / "failed.jsonl"
    with _collector(path, FailingSnapshotCollector()) as collector:
        with pytest.raises(RuntimeError, match="report collection failed"):
            collector.record(
                Env(),
                step_idx=1,
                trigger="speech",
                phase="1_day_speech",
                speaker_id=3,
                observer_ids=[1, 3],
            )
    assert path.read_text(encoding="utf-8") == ""


def test_record_api_has_no_private_or_truth_inputs():
    parameters = inspect.signature(TWDToMSampleCollector.record).parameters
    assert not {
        "roles", "true_roles", "actual_wolves", "observation", "private_observation"
    } & set(parameters)
