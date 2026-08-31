import inspect
import json

import pytest

from werewolf.models.twd_tom.collector import TWDToMSampleCollector
from werewolf.models.twd_tom.public_events import public_speech_actions
from werewolf.models.twd_tom.schema import LABEL_PROMPT_VERSION
from werewolf.models.twd_tom.samples import PublicSnapshot, SAMPLE_SCHEMA_VERSION
from tests.twd_tom.public_event_fixtures import make_speech_annotations


class Env:
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
                "speaker": "player1",
                "raw_text": "earlier",
            },
            {
                "event_idx": 2,
                "event_type": "turn_start",
                "speaker": "player3",
            },
        ]
        self.speech_annotations = make_speech_annotations(
            self.public_events,
            [["player1", "point_as_werewolf", "player7"]],
        )


class SnapshotCollector:
    def __init__(self):
        self.snapshots = []

    def collect(self, snapshot, *, env):
        assert isinstance(snapshot, PublicSnapshot)
        self.snapshots.append(snapshot)
        return {
            "player1": {"status": "parse_error", "suspected_werewolves": None, "known_werewolves": [], "known_non_werewolves": ["player1"], "error": "synthetic invalid report", "agent_backend_id": "a"},
            "player3": {"status": "ok", "suspected_werewolves": ["player1"], "known_werewolves": [], "known_non_werewolves": ["player3"], "error": None, "agent_backend_id": "b"},
        }


def test_collector_freezes_then_writes_player_suspicion_jsonl(tmp_path):
    path = tmp_path / "sample.jsonl"
    snapshot_collector = SnapshotCollector()
    with TWDToMSampleCollector(str(path), snapshot_collector, game_id="g1") as collector:
        sample = collector.record(
            Env(), step_idx=2, trigger="speech", phase="1_day_speech",
            speaker_id=3, observer_ids=[1, 3]
        )
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record == sample
    assert record["schema_version"] == SAMPLE_SCHEMA_VERSION
    assert record["schema_version"] == "classic7_pre_speech_player_suspicion_v7"
    assert record["label_prompt_version"] == LABEL_PROMPT_VERSION
    assert record["label_prompt_version"] == (
        "classic7_pre_speech_player_suspicion_prompt_v6"
    )
    assert "belief_mode" not in json.dumps(record)
    assert "believed_werewolves" not in record
    assert record["suspected_werewolves"]["player1"] is None
    assert "pair_support" not in record
    assert "pair_targets" not in record
    assert record["label_cutoff_step_idx"] == record["step_idx"] == 2
    assert record["public_action_count"] == len(
        public_speech_actions(
            record["public_events"], record["speech_annotations"]
        )
    )
    assert len(snapshot_collector.snapshots) == 1
    assert not {
        "actual_roles", "actual_wolves", "wolf_teammates", "role",
        "private_observation", "seer_result", "witch_action",
        "kill_decision", "god_view", "future_events", "final_result",
    } & set(record)


def test_collector_uses_exact_pre_speech_history(tmp_path):
    env = Env()
    snapshots = SnapshotCollector()
    with TWDToMSampleCollector(str(tmp_path / "sample.jsonl"), snapshots, game_id="g1") as collector:
        collector.record(
            env, step_idx=1, trigger="speech", phase="1_day_speech",
            speaker_id=3, observer_ids=[1, 3]
        )
        env.public_events[-1] = {
            "event_idx": 2,
            "event_type": "public_speech",
            "speaker": "player3",
            "raw_text": "current speech",
        }
        env.public_events.append(
            {
                "event_idx": 3,
                "event_type": "turn_start",
                "speaker": "player4",
            }
        )
    assert len(snapshots.snapshots[0].sp_actions) == 1


def test_collector_context_manager_and_closed_write(tmp_path):
    collector = TWDToMSampleCollector(str(tmp_path / "x.jsonl"), SnapshotCollector(), game_id="g1")
    collector.close()
    assert collector.closed
    with pytest.raises(RuntimeError, match="closed"):
        collector.write({})


def test_collector_requires_snapshot_collector_and_game_id(tmp_path):
    with pytest.raises(ValueError, match="required"):
        TWDToMSampleCollector(str(tmp_path / "x.jsonl"), None, game_id="g1")
    with pytest.raises(ValueError, match="game_id"):
        TWDToMSampleCollector(str(tmp_path / "x.jsonl"), SnapshotCollector(), game_id="")


def test_snapshot_collector_failure_propagates_without_writing(tmp_path):
    class FailingSnapshotCollector:
        def collect(self, snapshot, *, env):
            raise RuntimeError("report collection failed")

    path = tmp_path / "failed.jsonl"
    with TWDToMSampleCollector(
        str(path), FailingSnapshotCollector(), game_id="g1"
    ) as collector:
        with pytest.raises(RuntimeError, match="report collection failed"):
            collector.record(
                Env(), step_idx=1, trigger="speech", phase="1_day_speech",
                speaker_id=3, observer_ids=[1, 3]
            )
    assert path.read_text(encoding="utf-8") == ""


def test_record_api_has_no_private_or_truth_inputs():
    parameters = inspect.signature(TWDToMSampleCollector.record).parameters
    assert not {"roles", "true_roles", "actual_wolves", "observation", "private_observation"} & set(parameters)
