from copy import deepcopy

from werewolf.models.twd_tom.public_events import (
    PUBLIC_EVENT_SCHEMA_VERSION,
    public_event_digest,
    structured_input_digest,
)
from werewolf.models.twd_tom.dataset import (
    ANNOTATED_LABEL_PROVENANCE,
    ANNOTATION_SCHEMA_VERSION,
    PRIVATE_FIELDS_USAGE,
    SOURCE_LABEL_PROVENANCE,
    TOM_INPUT_SCOPES,
)
from werewolf.models.twd_tom.samples import (
    PUBLIC_ONLY_SAMPLE_SCHEMA_VERSION,
    SAMPLE_SCHEMA_VERSION,
)
from werewolf.models.twd_tom.schema import (
    LABEL_PROMPT_VERSION,
    PUBLIC_ONLY_FORMAL_ANNOTATION_SCHEMA_VERSION,
    PUBLIC_ONLY_FORMAL_LABEL_PROVENANCE,
    PUBLIC_ONLY_LABEL_PROMPT_VERSION,
    PUBLIC_ONLY_LABEL_PROVENANCE,
    PUBLIC_ONLY_MODEL_INPUT_SCOPE,
    PUBLIC_ONLY_PRIVATE_FIELDS_USAGE,
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


def make_training_sample(
    tom_order,
    *,
    game_id="synthetic_game_001",
    step_idx=1,
    speaker_id=2,
    observers=None,
    with_latest_action=True,
    public_events=None,
    phase="1_day_speech",
):
    """Build one small sample matching the current strict Dataset contract."""

    if observers is None:
        observers = (speaker_id,) if tom_order == 1 else (1, 2, 3, 5)
    if public_events is None:
        actions = (
            [["player3", "support", "player4"]]
            if with_latest_action
            else []
        )
        public_events = make_public_events(
            actions,
            speaker_id=speaker_id,
            phase=phase,
        )
    else:
        public_events = deepcopy(public_events)

    subjects = [f"player{observer_id}" for observer_id in observers]
    mapping = {subject: [] for subject in subjects}
    return {
        "agent_backend_ids": {
            subject: "synthetic_backend" for subject in subjects
        },
        "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
        "belief_errors": {subject: None for subject in subjects},
        "belief_status": {subject: "ok" for subject in subjects},
        "current_action_used": False,
        "expert_labels_used_as_later_evidence": False,
        "future_information_used": False,
        "game_id": game_id,
        "known_non_werewolves": {
            subject: [subject] for subject in subjects
        },
        "known_werewolves": deepcopy(mapping),
        "label_cutoff_step_idx": step_idx,
        "label_prompt_version": LABEL_PROMPT_VERSION,
        "label_provenance": ANNOTATED_LABEL_PROVENANCE,
        "model_input_scope": TOM_INPUT_SCOPES[tom_order],
        "observer_annotation_confidence": {
            subject: "synthetic_fixture" for subject in subjects
        },
        "observer_ids": list(observers),
        "observer_label_provenance": {
            subject: "synthetic_fixture" for subject in subjects
        },
        "phase": phase,
        "private_fields_usage": PRIVATE_FIELDS_USAGE[tom_order],
        "public_action_count": sum(
            len(event.get("sp_actions", ()))
            for event in public_events
            if event["event_type"] == "public_speech"
        ),
        "public_event_digest": public_event_digest(public_events),
        "public_event_schema_version": PUBLIC_EVENT_SCHEMA_VERSION,
        "public_events": public_events,
        "report_trigger": "pre_public_speech",
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "source_belief_errors": {subject: None for subject in subjects},
        "source_belief_status": {subject: "ok" for subject in subjects},
        "source_label_provenance": SOURCE_LABEL_PROVENANCE,
        "source_schema_version": SAMPLE_SCHEMA_VERSION,
        "speaker_id": speaker_id,
        "step_idx": step_idx,
        "structured_input_digest": structured_input_digest(public_events),
        "suspected_werewolves": {
            subject: ["player7"] if subject != "player7" else ["player6"]
            for subject in subjects
        },
        "tom_order": tom_order,
    }


def make_public_only_training_sample(tom_order, **kwargs):
    sample = make_training_sample(tom_order, **kwargs)
    sample["schema_version"] = PUBLIC_ONLY_SAMPLE_SCHEMA_VERSION
    sample["source_schema_version"] = PUBLIC_ONLY_SAMPLE_SCHEMA_VERSION
    sample["label_prompt_version"] = PUBLIC_ONLY_LABEL_PROMPT_VERSION
    sample["source_label_provenance"] = PUBLIC_ONLY_LABEL_PROVENANCE
    sample["annotation_schema_version"] = (
        PUBLIC_ONLY_FORMAL_ANNOTATION_SCHEMA_VERSION
    )
    sample["label_provenance"] = PUBLIC_ONLY_FORMAL_LABEL_PROVENANCE
    sample["model_input_scope"] = PUBLIC_ONLY_MODEL_INPUT_SCOPE
    sample["private_fields_usage"] = PUBLIC_ONLY_PRIVATE_FIELDS_USAGE
    sample["known_werewolves"] = {
        subject: [] for subject in sample["known_werewolves"]
    }
    sample["known_non_werewolves"] = {
        subject: [] for subject in sample["known_non_werewolves"]
    }
    return sample


def make_full_history_training_sample(*, game_id="synthetic_full_history"):
    events = [
        {"event_idx": 0, "event_type": "phase_change", "phase": "1_day_speech"},
        {"event_idx": 1, "event_type": "turn_start", "speaker": "player2"},
        {
            "event_idx": 2,
            "event_type": "public_speech",
            "speaker": "player2",
            "raw_text": "synthetic speech",
            "sp_actions": [["player3", "support", "player4"]],
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
        2,
        game_id=game_id,
        step_idx=8,
        speaker_id=5,
        public_events=events,
        phase="2_day_speech",
    )
