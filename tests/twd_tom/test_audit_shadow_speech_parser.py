import hashlib
import json
from pathlib import Path

import pytest

from script.twd_tom.audit_shadow_speech_parser import (
    audit_shadow_speech_parser,
)
from werewolf.models.twd_tom.speech_annotations import make_speech_annotation
from werewolf.trajectory import canonical_digest, canonical_json


class FakeBackend:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_source_batch(root: Path):
    game_id = "pilot_test_game_0001_seed_1"
    game_dir = root / "games" / "game_0001_seed_1"
    public_events = [
        {
            "event_idx": 0,
            "event_type": "phase_change",
            "phase": "1_day_speech",
        },
        {
            "event_idx": 1,
            "event_type": "turn_start",
            "speaker": "player1",
        },
        {
            "event_idx": 2,
            "event_type": "public_speech",
            "speaker": "player1",
            "raw_text": "我明确支持 player2。",
        },
    ]
    trajectory = {
        "game_id": game_id,
        "initial_public_events": public_events[:2],
        "transitions": [
            {
                "phase_before": "1_day_speech",
                "public_events_appended": public_events[2:],
            }
        ],
    }
    trajectory_path = game_dir / "trajectory.json"
    _write_json(trajectory_path, trajectory)
    annotation = make_speech_annotation(
        event_idx=2,
        speaker="player1",
        raw_text="我明确支持 player2。",
        parser_model_id="qwen3.5-9b",
        parser_call_id="speech_parser_event_000002",
        annotation_source="llm_parser",
        status="ok",
        actions=[["player1", "support", "player2"]],
        generation_attempts=[
            {
                "generation_attempt": 1,
                "status": "ok",
                "raw_response": "player1 | support | player2",
                "error_type": None,
                "error_message": None,
            }
        ],
        raw_response="player1 | support | player2",
        error_type=None,
        error_message=None,
    )
    annotations_path = game_dir / "speech_annotations.jsonl"
    _write_jsonl(annotations_path, [annotation])
    summary = {
        "run_id": "pilot_test",
        "game_ids": [game_id],
    }
    summary["summary_digest"] = canonical_digest(summary)
    _write_json(root / "summary.json", summary)
    return trajectory_path, annotations_path


def _make_config(path: Path, *, thinking_type="disabled"):
    path.write_text(
        "\n".join(
            [
                "backends:",
                "  deepseek_shadow:",
                "    type: openai_compatible",
                "    base_url: https://api.deepseek.com",
                "    api_key_env: DEEPSEEK_API_KEY",
                "    default_model: deepseek-v4-flash",
                "    supports_json_schema: false",
                "parser:",
                "  backend: deepseek_shadow",
                "  model: deepseek-v4-flash",
                "  model_params:",
                "    temperature: 0.0",
                "    request_extra_body:",
                "      thinking:",
                f"        type: {thinking_type}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _code_provenance():
    return {
        "batch_code_commit": "a" * 40,
        "git_worktree_clean": True,
    }


def test_shadow_parser_writes_detached_agreement_artifacts(tmp_path):
    input_root = tmp_path / "source"
    trajectory_path, annotations_path = _make_source_batch(input_root)
    config_path = tmp_path / "shadow.yaml"
    _make_config(config_path)
    output_root = tmp_path / "shadow-output"
    source_hashes = (_sha256(trajectory_path), _sha256(annotations_path))
    backend = FakeBackend(["player1 | support | player2"])

    summary = audit_shadow_speech_parser(
        input_root=input_root,
        config_path=config_path,
        output_root=output_root,
        backend=backend,
        code_provenance=_code_provenance(),
    )

    assert summary["speech_count"] == 1
    assert summary["shadow_generation_attempt_count"] == 1
    assert summary["exact_action_order_match_count"] == 1
    assert summary["action_set_match_count"] == 1
    assert summary["disagreement_count"] == 0
    assert summary["source_artifacts_mutated"] is False
    assert source_hashes == (_sha256(trajectory_path), _sha256(annotations_path))
    assert backend.calls[0]["temperature"] == 0
    assert backend.calls[0]["extra_body"] == {
        "thinking": {"type": "disabled"}
    }

    game_output = output_root / "games" / "game_0001_seed_1"
    shadow = json.loads(
        (game_output / "shadow_speech_annotations.jsonl").read_text(
            encoding="utf-8"
        )
    )
    assert shadow["parser_model_id"] == "deepseek-v4-flash"
    assert shadow["actions"] == [["player1", "support", "player2"]]
    comparison = json.loads(
        (game_output / "shadow_speech_comparisons.jsonl").read_text(
            encoding="utf-8"
        )
    )
    assert comparison["raw_text"] == "我明确支持 player2。"
    assert comparison["exact_action_order_match"] is True


def test_shadow_parser_records_error_without_fallback(tmp_path):
    input_root = tmp_path / "source"
    _trajectory_path, annotations_path = _make_source_batch(input_root)
    config_path = tmp_path / "shadow.yaml"
    _make_config(config_path)
    output_root = tmp_path / "shadow-output"
    original_annotations = annotations_path.read_bytes()
    backend = FakeBackend(['[["player1", "support", "player2"]]'] * 3)

    summary = audit_shadow_speech_parser(
        input_root=input_root,
        config_path=config_path,
        output_root=output_root,
        backend=backend,
        code_provenance=_code_provenance(),
    )

    assert summary["shadow_status_counts"] == {"error": 1}
    assert summary["shadow_generation_attempt_count"] == 3
    assert summary["shadow_retry_count"] == 2
    assert summary["disagreement_count"] == 1
    assert annotations_path.read_bytes() == original_annotations
    shadow_path = (
        output_root
        / "games"
        / "game_0001_seed_1"
        / "shadow_speech_annotations.jsonl"
    )
    shadow = json.loads(shadow_path.read_text(encoding="utf-8"))
    assert shadow["status"] == "error"
    assert shadow["actions"] == []
    assert len(shadow["generation_attempts"]) == 3


def test_shadow_parser_rejects_enabled_thinking_before_writing(tmp_path):
    input_root = tmp_path / "source"
    _make_source_batch(input_root)
    config_path = tmp_path / "shadow.yaml"
    _make_config(config_path, thinking_type="enabled")
    output_root = tmp_path / "shadow-output"

    with pytest.raises(ValueError, match="disable DeepSeek thinking"):
        audit_shadow_speech_parser(
            input_root=input_root,
            config_path=config_path,
            output_root=output_root,
            backend=FakeBackend([]),
            code_provenance=_code_provenance(),
        )
    assert not output_root.exists()
