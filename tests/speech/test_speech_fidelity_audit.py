import json
from types import SimpleNamespace

import pytest

from script.public_belief_matrix import audit_speech_fidelity
from script.twd_tom import formal_batch_collection
from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
from werewolf.models.public_belief_matrix.collection import (
    PublicBeliefMatrixSampleCollector,
)
from werewolf.speech.speech_fidelity_audit import (
    SPEECH_FIDELITY_AUDIT_SCHEMA_VERSION,
    SpeechFidelityAuditSidecar,
)
from werewolf.speech.speech_perceiver import SpeechPerceiver


ROLES = [
    "Werewolf",
    "Werewolf",
    "Seer",
    "Witch",
    "Villager",
    "Villager",
    "Villager",
]


class Backend:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class SymbolicReporter:
    def report(self, *, observer_id, backend_id, **_kwargs):
        return {
            "observer": observer_id,
            "status": "ok",
            "suspected_werewolves": [],
            "error": None,
            "reporter_backend_id": backend_id,
        }


def _speech_env(perceiver):
    env = WerewolfTextEnvV0(log_save_path=None, speech_perceiver=perceiver)
    env.reset(roles=ROLES)
    env.phase = "speech"
    env.day = 2
    env.day_or_night = "day"
    env.current_act_idx = 1
    env.alive = [1] * 7
    env.speech_queue = [2]
    env.vote_queue = []
    return env


def _run_completed_speech(tmp_path, backend, speech="我认为3号是狼人"):
    audit_path = tmp_path / "speech_fidelity.jsonl"
    with SpeechFidelityAuditSidecar(
        audit_path,
        source_commit="a" * 40,
        backend_id="parser_backend",
        model_id="parser_model",
    ) as sidecar:
        audited = sidecar.audited_perceiver(
            SpeechPerceiver(backend=backend, model_name="parser_model"),
            game_id="audit_game_001_seed_881",
            seed=881,
        )
        env = _speech_env(audited)
        env.step(("speech", speech))
        record = audited.record(
            env,
            step_idx=4,
            trigger="speech",
            phase="2_day_speech",
            speaker_id=2,
        )
    return record, json.loads(audit_path.read_text(encoding="utf-8"))


def test_audit_disabled_has_no_side_effect_or_extra_backend_call(tmp_path):
    backend = Backend("player2 | point_as_werewolf | player3")
    env = _speech_env(SpeechPerceiver(backend=backend, model_name="parser_model"))

    env.step(("speech", "我认为3号是狼人"))

    assert len(backend.calls) == 1
    assert list(tmp_path.iterdir()) == []


def test_enabled_audit_writes_actual_single_parse_and_public_only_fields(tmp_path):
    raw_response = "player2 | point_as_werewolf | player3"
    backend = Backend(raw_response)
    record, saved = _run_completed_speech(tmp_path, backend)

    assert len(backend.calls) == 1
    assert saved == record
    assert record == {
        "audit_schema_version": SPEECH_FIDELITY_AUDIT_SCHEMA_VERSION,
        "game_id": "audit_game_001_seed_881",
        "seed": 881,
        "step_idx": 4,
        "phase": "2_day_speech",
        "speaker": "player2",
        "raw_public_speech": "我认为3号是狼人",
        "speech_perceiver_raw_response": raw_response,
        "normalized_sp_actions": [
            ["player2", "point_as_werewolf", "player3"]
        ],
        "parse_status": "ok",
        "parse_error_type": None,
        "parse_error_message": None,
        "protected_self_claim_actions": [],
        "public_speech_event_idx": 0,
        "source_commit": "a" * 40,
        "backend_id": "parser_backend",
        "model_id": "parser_model",
        "speech_perceiver_temperature": 0,
        "speech_perceiver_max_tokens": 256,
        "enable_thinking": False,
    }
    serialized = json.dumps(record, ensure_ascii=False)
    for forbidden in (
        "true_roles",
        "private_observation",
        "wolf_teammates",
        "belief_reporter",
        "matrix_target",
        "game_outcome",
    ):
        assert forbidden not in serialized


def test_parser_failure_keeps_online_fallback_and_records_error(tmp_path):
    backend = Backend(error=RuntimeError("parser down"))
    record, _saved = _run_completed_speech(
        tmp_path,
        backend,
        speech="我是预言家。",
    )

    assert len(backend.calls) == 1
    assert record["parse_status"] == "parser_error"
    assert record["parse_error_type"] == "RuntimeError"
    assert record["speech_perceiver_raw_response"] is None
    assert record["normalized_sp_actions"] == [
        ["player2", "point_as_seer", "player2"]
    ]
    assert record["protected_self_claim_actions"] == record[
        "normalized_sp_actions"
    ]


def test_parser_side_failure_retains_the_actual_received_response(monkeypatch):
    backend = Backend("ACTUAL-PARSER-RESPONSE")
    perceiver = SpeechPerceiver(backend=backend, model_name="parser_model")
    monkeypatch.setattr(
        perceiver,
        "_extract_response_actions",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("normalization failed")
        ),
    )

    result = perceiver.parse_with_audit(2, "公开发言", 2, "speech")

    assert len(backend.calls) == 1
    assert result.parse_status == "parser_error"
    assert result.raw_response == "ACTUAL-PARSER-RESPONSE"
    assert result.normalized_actions == []


def test_production_parse_does_not_swallow_self_claim_extractor_bug(monkeypatch):
    perceiver = SpeechPerceiver(
        backend=Backend("NONE"),
        model_name="parser_model",
    )
    monkeypatch.setattr(
        perceiver,
        "_extract_explicit_self_claim_actions",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("extractor bug")),
    )

    with pytest.raises(RuntimeError, match="extractor bug"):
        perceiver.parse(2, "公开发言", 2, "speech")


def test_formal_privacy_safe_cli_cannot_enable_raw_speech_audit():
    options = {
        option
        for action in formal_batch_collection.build_arg_parser()._actions
        for option in action.option_strings
    }
    assert "--speech-fidelity-audit-file" not in options
    assert "--audit-file" not in options


def test_raw_speech_stays_out_of_pbm_formal_sample(tmp_path):
    raw_canary = "RAW-FIDELITY-CANARY"
    backend = Backend("NONE")
    env = _speech_env(SpeechPerceiver(backend=backend, model_name="parser_model"))
    env.step(("speech", raw_canary))
    sample_path = tmp_path / "formal_batch_samples.jsonl"
    collector = PublicBeliefMatrixSampleCollector(
        output_path=sample_path,
        game_id="formal_game_001_seed_881",
        reporter=SymbolicReporter(),
        reporter_dispatch={
            "backend": object(),
            "backend_id": "parser_backend",
            "model_name": "parser_model",
        },
    )
    collector.record(
        env,
        step_idx=4,
        trigger="speech",
        phase="2_day_speech",
        speaker_id=2,
    )
    collector.close()

    assert raw_canary not in sample_path.read_text(encoding="utf-8")


def test_dedicated_audit_cli_is_explicit_and_limited_to_two_seeds():
    parser = audit_speech_fidelity.build_argument_parser()
    args = parser.parse_args(
        [
            "--config",
            "/repo/config.yaml",
            "--seeds",
            "881",
            "882",
            "--audit-file",
            "/review/speech_fidelity.jsonl",
        ]
    )
    assert args.seeds == [881, 882]
    assert args.audit_file == "/review/speech_fidelity.jsonl"
    with pytest.raises(ValueError, match="one or two"):
        audit_speech_fidelity._validated_seeds([881, 882, 883])


def test_audit_record_requires_same_completed_public_speech(tmp_path):
    backend = Backend("NONE")
    audit_path = tmp_path / "speech_fidelity.jsonl"
    with SpeechFidelityAuditSidecar(
        audit_path,
        source_commit="a" * 40,
        backend_id="parser_backend",
        model_id="parser_model",
    ) as sidecar:
        audited = sidecar.audited_perceiver(
            SpeechPerceiver(backend=backend, model_name="parser_model"),
            game_id="audit_game_001_seed_881",
            seed=881,
        )
        audited.parse(2, "公开发言", 2, "speech")
        env = SimpleNamespace(
            public_events=[
                {
                    "event_idx": 0,
                    "event_type": "public_speech",
                    "speaker": "player2",
                    "raw_text": "DIFFERENT",
                    "sp_actions": [],
                }
            ]
        )
        with pytest.raises(ValueError, match="do not match"):
            audited.record(
                env,
                step_idx=4,
                trigger="speech",
                phase="2_day_speech",
                speaker_id=2,
            )
