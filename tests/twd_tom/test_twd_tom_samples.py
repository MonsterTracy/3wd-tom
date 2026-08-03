import json
import hashlib

import pytest

from werewolf.models.twd_tom.public_events import public_event_digest
from werewolf.models.twd_tom.samples import (
    ACTOR_PAIR_BELIEF_SCHEMA_VERSION,
    PublicSnapshot,
    freeze_public_snapshot,
    make_twd_tom_sample,
)
from werewolf.models.twd_tom.schema import canonical_wolf_pairs
from werewolf.speech.pair_belief_self_reporter import (
    PAIR_BELIEF_PROMPT_VERSION,
    canonical_json_sha256,
)


def _events(speaker="player2"):
    return [
        {"event_idx": 0, "event_type": "phase_change", "phase": "1_day_speech"},
        {
            "event_idx": 1,
            "event_type": "public_speech",
            "speaker": "player1",
            "raw_text": "earlier speech",
            "sp_actions": [],
        },
        {"event_idx": 2, "event_type": "turn_start", "speaker": speaker},
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
        "resolved_backend_config_sha256": {"backend_a": "e" * 64},
    }


def _report(*, status="ok", error=None):
    target = [
        1.0 / 15 if "player1" not in pair else 0.0
        for pair in canonical_wolf_pairs()
    ]
    payload = {
        "payload_version": "readonly_pair_belief_self_report_payload_v1",
        "messages": [{"role": "user", "content": "private payload"}],
        "request": {"temperature": 0.0, "max_tokens": 256},
    }
    return {
        "player_id": "player1",
        "report_status": status,
        "report_error": error,
        "pair_probabilities": target if status == "ok" else None,
        "known_werewolves": [],
        "known_non_werewolves": ["player1"],
        "reporter_input_payload": payload,
        "reporter_input_payload_sha256": canonical_json_sha256(payload),
        "raw_reporter_output": json.dumps({"pair_probabilities": target}),
        "parsed_output": {"pair_probabilities": target},
        "hard_knowledge_validation": {"status": "valid" if status == "ok" else "invalid"},
        "report_provenance": "playing_agent_readonly_direct_pair_belief_self_report_v1",
        "backend_alias": "backend_a",
        "resolved_model_name": "model-a",
        "prompt_version": PAIR_BELIEF_PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(
            payload["messages"][-1]["content"].encode("utf-8")
        ).hexdigest(),
        "parser_version": "strict_pair_probability_json_v1",
        "sampling_parameters": {"temperature": 0.0, "max_tokens": 256},
        "reporter_seed": None,
    }


def _snapshot():
    return freeze_public_snapshot(
        game_id="game_001",
        step_idx=8,
        phase="1_day_speech",
        speaker_id=1,
        report_trigger="pre_public_speech",
        observer_ids=[1],
        public_events=_events("player1"),
    )


def test_frozen_snapshot_has_exact_pre_speech_boundary_and_digest():
    events = _events("player1")
    snapshot = _snapshot()
    assert isinstance(snapshot, PublicSnapshot)
    assert snapshot.event_idx == 2
    assert snapshot.day == 1
    assert snapshot.public_event_digest == public_event_digest(events)
    assert snapshot.public_events[-1]["event_type"] == "turn_start"
    events[1]["raw_text"] = "mutated"
    assert snapshot.public_events[1]["raw_text"] == "earlier speech"
    with pytest.raises(TypeError):
        snapshot.public_events[1]["raw_text"] = "cannot mutate"


def test_sample_serializes_direct_pair_payload_and_actor_contract():
    sample = make_twd_tom_sample(
        public_snapshot=_snapshot(),
        reports={"player1": _report()},
        collection_provenance=_provenance(),
    )
    assert sample["schema_version"] == ACTOR_PAIR_BELIEF_SCHEMA_VERSION
    assert sample["reasoning_player_id"] == sample["current_speaker"] == "player1"
    assert sample["current_action_used"] is False
    assert sample["future_information_used"] is False
    assert len(sample["player_reports"]) == 7
    assert sample["player_reports"][0]["pair_probabilities"] is not None
    assert all(
        report["pair_probabilities"] is None
        for report in sample["player_reports"][1:]
    )
    assert sample["reasoning_input_payload"] == {
        "public_history": sample["public_events"],
        "reasoning_player_id": "player1",
        "legal_private_knowledge": {
            "known_werewolves": [],
            "known_non_werewolves": ["player1"],
        },
    }
    assert sample["reasoning_input_payload_sha256"] == canonical_json_sha256(
        sample["reasoning_input_payload"]
    )
    serialized = json.dumps(sample)
    assert "suspected_werewolves" not in serialized
    assert "expert" not in serialized.lower()


def test_sample_requires_exact_alive_player_reports():
    with pytest.raises(ValueError, match="exactly match"):
        make_twd_tom_sample(
            public_snapshot=_snapshot(),
            reports={},
            collection_provenance=_provenance(),
        )


def test_failed_pair_report_remains_missing_without_completion():
    error = "synthetic semantic error"
    sample = make_twd_tom_sample(
        public_snapshot=_snapshot(),
        reports={"player1": _report(status="semantic_error", error=error)},
        collection_provenance=_provenance(),
    )
    report = sample["player_reports"][0]
    assert report["report_status"] == "semantic_error"
    assert report["pair_probabilities"] is None
    assert report["report_error"] == error
    assert report["parsed_output"] is not None


def test_formal_sample_rejects_dirty_worktree_provenance():
    provenance = _provenance()
    provenance["git_worktree_clean"] = False
    with pytest.raises(ValueError, match="git_worktree_clean=true"):
        make_twd_tom_sample(
            public_snapshot=_snapshot(),
            reports={"player1": _report()},
            collection_provenance=provenance,
        )
