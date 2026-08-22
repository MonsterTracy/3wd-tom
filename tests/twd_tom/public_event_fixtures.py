"""Small canonical tom-v2 fixtures shared by Dataset and training tests."""

from copy import deepcopy

from werewolf.models.twd_tom.public_events import (
    PUBLIC_EVENT_SCHEMA_VERSION,
    public_event_digest,
    structured_input_digest,
)
from werewolf.models.twd_tom.samples import SAMPLE_SCHEMA_VERSION
from werewolf.models.twd_tom.schema import LABEL_PROVENANCE, LABEL_PROMPT_VERSION


def make_public_events(
    sp_actions,
    *,
    speaker_id=2,
    phase="1_day_speech",
    raw_text="synthetic earlier public speech",
):
    events = [
        {"event_idx": 0, "event_type": "phase_change", "phase": phase},
        {"event_idx": 1, "event_type": "turn_start", "speaker": f"player{speaker_id}"},
    ]
    if sp_actions:
        events.extend([
            {
                "event_idx": 2,
                "event_type": "public_speech",
                "speaker": f"player{speaker_id}",
                "raw_text": raw_text,
                "sp_actions": deepcopy(sp_actions),
            },
            {
                "event_idx": 3,
                "event_type": "turn_start",
                "speaker": f"player{speaker_id}",
            },
        ])
    return events


def public_history_fields(
    sp_actions,
    *,
    speaker_id=2,
    phase="1_day_speech",
    raw_text="synthetic earlier public speech",
):
    events = make_public_events(
        sp_actions,
        speaker_id=speaker_id,
        phase=phase,
        raw_text=raw_text,
    )
    return {
        "public_event_schema_version": PUBLIC_EVENT_SCHEMA_VERSION,
        "public_events": events,
        "public_event_digest": public_event_digest(events),
        "structured_input_digest": structured_input_digest(events),
        "public_action_count": len(sp_actions),
    }


def make_training_sample(
    *,
    game_id="synthetic_game_001",
    step_idx=1,
    speaker_id=2,
    observers=(1, 2, 3, 5),
    public_events=None,
    phase="1_day_speech",
):
    if public_events is None:
        public_events = make_public_events(
            [[f"player{speaker_id}", "support", "player4"]],
            speaker_id=speaker_id,
            phase=phase,
        )
    else:
        public_events = deepcopy(public_events)
    subjects = [f"player{observer_id}" for observer_id in observers]
    return {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "game_id": game_id,
        "step_idx": step_idx,
        "report_trigger": "pre_public_speech",
        "phase": phase,
        "speaker_id": speaker_id,
        "observer_ids": list(observers),
        "public_event_schema_version": PUBLIC_EVENT_SCHEMA_VERSION,
        "public_events": public_events,
        "public_event_digest": public_event_digest(public_events),
        "structured_input_digest": structured_input_digest(public_events),
        "suspected_werewolves": {
            subject: ["player7"] if subject != "player7" else ["player6"]
            for subject in subjects
        },
        "known_werewolves": {subject: [] for subject in subjects},
        "known_non_werewolves": {subject: [subject] for subject in subjects},
        "belief_status": {subject: "ok" for subject in subjects},
        "belief_errors": {subject: None for subject in subjects},
        "label_provenance": LABEL_PROVENANCE,
        "agent_backend_ids": {subject: "synthetic_backend" for subject in subjects},
        "label_cutoff_step_idx": step_idx,
        "public_action_count": sum(
            len(event.get("sp_actions", ()))
            for event in public_events
            if event["event_type"] == "public_speech"
        ),
        "label_prompt_version": LABEL_PROMPT_VERSION,
    }


def make_full_history_training_sample(*, game_id="synthetic_full_history"):
    events = [
        {"event_idx": 0, "event_type": "phase_change", "phase": "1_day_speech"},
        {"event_idx": 1, "event_type": "turn_start", "speaker": "player2"},
        {
            "event_idx": 2,
            "event_type": "public_speech",
            "speaker": "player2",
            "raw_text": "synthetic speech",
            "sp_actions": [["player2", "support", "player4"]],
        },
        {"event_idx": 3, "event_type": "phase_change", "phase": "1_day_vote"},
        {
            "event_idx": 4,
            "event_type": "vote_result",
            "votes": [
                {"voter": "player1", "target": "player2"},
                {"voter": "player3", "target": "player2"},
            ],
        },
        {"event_idx": 5, "event_type": "exile_result", "exiled_players": ["player2"]},
        {"event_idx": 6, "event_type": "death_announcement", "dead_players": ["player4"]},
        {"event_idx": 7, "event_type": "phase_change", "phase": "2_day_speech"},
        {"event_idx": 8, "event_type": "turn_start", "speaker": "player5"},
    ]
    return make_training_sample(
        game_id=game_id,
        step_idx=8,
        speaker_id=5,
        public_events=events,
        phase="2_day_speech",
    )
