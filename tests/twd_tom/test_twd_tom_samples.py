import pytest

from werewolf.models.twd_tom.schema import LABEL_PROMPT_VERSION
from werewolf.models.twd_tom.public_events import (
    public_event_digest,
    public_speech_actions,
    structured_input_digest,
)
from werewolf.models.twd_tom.samples import (
    PublicSnapshot,
    freeze_public_snapshot,
    make_twd_tom_sample,
)


def _events(speaker="player2", actions=None):
    return [
        {
            "event_idx": 0,
            "event_type": "phase_change",
            "phase": "1_day_speech",
        },
        {
            "event_idx": 1,
            "event_type": "public_speech",
            "speaker": "player1",
            "raw_text": "earlier speech",
            "sp_actions": [] if actions is None else actions,
        },
        {
            "event_idx": 2,
            "event_type": "turn_start",
            "speaker": speaker,
        },
    ]


def test_frozen_snapshot_has_exact_time_alignment_and_digest():
    actions = [["player2", "point_as_werewolf", "player7"]]
    events = _events(actions=actions)
    snapshot = freeze_public_snapshot(
        game_id="game_001", step_idx=8, phase="1_day_speech",
        speaker_id=2, report_trigger="pre_public_speech",
        observer_ids=[1, 2], public_events=events,
    )
    assert isinstance(snapshot, PublicSnapshot)
    assert snapshot.label_cutoff_step_idx == snapshot.step_idx == 8
    assert snapshot.public_action_count == len(snapshot.sp_actions) == 1
    assert snapshot.public_event_digest == public_event_digest(events)
    assert snapshot.structured_input_digest == structured_input_digest(events)
    actions.append(["player3", "oppose", "player2"])
    assert len(snapshot.sp_actions) == 1
    events[1]["raw_text"] = "mutated"
    assert snapshot.public_events[1]["raw_text"] == "earlier speech"
    with pytest.raises(TypeError):
        snapshot.public_events[1]["raw_text"] = "cannot mutate"


def test_sample_uses_same_frozen_history_and_does_not_save_raw_response():
    snapshot = freeze_public_snapshot(
        game_id="game_001", step_idx=1, phase="1_day_speech",
        speaker_id=1, report_trigger="pre_public_speech",
        observer_ids=[1], public_events=_events(speaker="player1"),
    )
    reports = {
        "player1": {
            "status": "ok",
            "suspected_werewolves": [],
            "known_werewolves": [],
            "known_non_werewolves": ["player1"],
            "error": None,
            "agent_backend_id": "backend_a",
            "raw_response": "private transport detail",
        }
    }
    sample = make_twd_tom_sample(public_snapshot=snapshot, reports=reports)
    assert "sp_actions" not in sample
    assert public_speech_actions(sample["public_events"]) == []
    assert sample["public_event_digest"] == snapshot.public_event_digest
    assert sample["structured_input_digest"] == snapshot.structured_input_digest
    assert sample["label_cutoff_step_idx"] == sample["step_idx"]
    assert "raw_response" not in sample
    assert "private_observation" not in sample
    assert "agent_backend_ids" in sample
    assert sample["suspected_werewolves"]["player1"] == []
    assert "believed_werewolves" not in sample
    assert sample["known_non_werewolves"]["player1"] == ["player1"]
    assert sample["label_prompt_version"] == (
        "classic7_pre_speech_player_suspicion_prompt_v2"
    )
    assert sample["label_prompt_version"] == LABEL_PROMPT_VERSION
    assert "target_distribution_is_reporter_probability" not in sample
    assert "target_distribution_is_deterministic_encoding" not in sample


def test_sample_requires_exact_observer_reports():
    snapshot = freeze_public_snapshot(
        game_id="game_001", step_idx=1, phase="1_day_speech",
        speaker_id=1, report_trigger="pre_public_speech",
        observer_ids=[1], public_events=_events(speaker="player1"),
    )
    with pytest.raises(ValueError, match="exactly match"):
        make_twd_tom_sample(public_snapshot=snapshot, reports={})


def test_sample_persists_full_candidate_semantic_error_without_repair():
    snapshot = freeze_public_snapshot(
        game_id="game_001",
        step_idx=1,
        phase="1_day_speech",
        speaker_id=1,
        report_trigger="pre_public_speech",
        observer_ids=[1],
        public_events=_events(speaker="player1"),
    )
    error = (
        "suspected_werewolves cannot equal all legal candidates unless "
        "hard knowledge already determines the full candidate set"
    )
    sample = make_twd_tom_sample(
        public_snapshot=snapshot,
        reports={
            "player1": {
                "status": "semantic_error",
                "suspected_werewolves": None,
                "known_werewolves": [],
                "known_non_werewolves": ["player1"],
                "error": error,
                "agent_backend_id": "backend_a",
            }
        },
    )
    assert sample["belief_status"]["player1"] == "semantic_error"
    assert sample["suspected_werewolves"]["player1"] is None
    assert sample["belief_errors"]["player1"] == error
