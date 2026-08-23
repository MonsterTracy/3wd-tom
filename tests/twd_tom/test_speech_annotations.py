from copy import deepcopy

import pytest

from werewolf.models.twd_tom.speech_annotations import (
    make_speech_annotation,
    normalize_speech_annotations,
    speech_annotation_digest,
)


def _events(raw_text="player2 的公开原文"):
    return [
        {
            "event_idx": 0,
            "event_type": "phase_change",
            "phase": "1_day_speech",
        },
        {
            "event_idx": 1,
            "event_type": "turn_start",
            "speaker": "player2",
        },
        {
            "event_idx": 2,
            "event_type": "public_speech",
            "speaker": "player2",
            "raw_text": raw_text,
        },
    ]


def _annotation(raw_text="player2 的公开原文"):
    return make_speech_annotation(
        event_idx=2,
        speaker="player2",
        raw_text=raw_text,
        parser_model_id="parser-model",
        parser_call_id="speech_parser_event_000002",
        annotation_source="llm_parser",
        status="ok",
        actions=[["player2", "oppose", "player4"]],
        raw_response="player2 | oppose | player4",
        error_type=None,
        error_message=None,
    )


def test_complete_annotations_bind_to_immutable_public_speech():
    events = _events()
    annotations = [_annotation()]

    normalized = normalize_speech_annotations(
        annotations,
        public_events=events,
        require_complete=True,
    )

    assert normalized == annotations
    assert speech_annotation_digest(normalized) == speech_annotation_digest(
        annotations
    )


def test_annotation_rejects_stale_raw_text_digest():
    with pytest.raises(ValueError, match="raw_text_digest differs"):
        normalize_speech_annotations(
            [_annotation("旧原文")],
            public_events=_events("新原文"),
            require_complete=True,
        )


def test_complete_coverage_rejects_missing_speech_annotation():
    with pytest.raises(ValueError, match="coverage mismatch"):
        normalize_speech_annotations(
            [],
            public_events=_events(),
            require_complete=True,
        )


def test_annotation_rejects_action_from_a_different_speaker():
    annotation = deepcopy(_annotation())
    annotation["actions"] = [["player3", "oppose", "player4"]]

    with pytest.raises(ValueError, match="subject must equal"):
        normalize_speech_annotations([annotation])
