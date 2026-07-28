from copy import deepcopy

from werewolf.models.twd_tom.public_events import (
    PUBLIC_EVENT_SCHEMA_VERSION,
    public_event_digest,
    structured_input_digest,
)


def make_public_events(
    sp_actions,
    *,
    speaker_id=2,
    phase="1_day_speech",
    raw_text="synthetic earlier public speech",
):
    events = [
        {
            "event_idx": 0,
            "event_type": "phase_change",
            "phase": phase,
        },
        {
            "event_idx": 1,
            "event_type": "turn_start",
            "speaker": f"player{speaker_id}",
        },
    ]
    if sp_actions:
        events.append(
            {
                "event_idx": len(events),
                "event_type": "public_speech",
                "speaker": f"player{speaker_id}",
                "raw_text": raw_text,
                "sp_actions": deepcopy(sp_actions),
            }
        )
        events.append(
            {
                "event_idx": len(events),
                "event_type": "turn_start",
                "speaker": f"player{speaker_id}",
            }
        )
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
