"""Explicit raw-content sidecar for online speech-parser fidelity audits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from werewolf.models.public_belief_matrix.collection import (
    PUBLIC_BELIEF_MATRIX_SUPERVISION_BOUNDARY,
)
from werewolf.models.twd_tom.schema import normalize_player
from werewolf.speech.speech_perceiver import (
    SPEECH_PARSER_MAX_TOKENS,
    SpeechParseAuditResult,
    SpeechPerceiver,
)


SPEECH_FIDELITY_AUDIT_SCHEMA_VERSION = "speech_semantic_fidelity_audit_v1"


class SpeechFidelityAuditSidecar:
    """Write raw speech-parser audit records to one exclusive JSONL file."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        source_commit: str,
        backend_id: str,
        model_id: str,
    ) -> None:
        path = Path(output_path).resolve()
        if not path.parent.is_dir():
            raise FileNotFoundError(f"audit parent directory not found: {path.parent}")
        for name, value in {
            "source_commit": source_commit,
            "backend_id": backend_id,
            "model_id": model_id,
        }.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        self.path = path
        self.source_commit = source_commit
        self.backend_id = backend_id
        self.model_id = model_id
        self._output = path.open("x", encoding="utf-8")

    def audited_perceiver(
        self,
        perceiver: SpeechPerceiver,
        *,
        game_id: str,
        seed: int,
    ) -> "AuditedSpeechPerceiver":
        return AuditedSpeechPerceiver(
            perceiver,
            sidecar=self,
            game_id=game_id,
            seed=seed,
        )

    def write(self, record: dict[str, Any]) -> None:
        self._output.write(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        )
        self._output.flush()

    def close(self) -> None:
        self._output.close()

    def __enter__(self) -> "SpeechFidelityAuditSidecar":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


class AuditedSpeechPerceiver:
    """Preserve online parse behavior and emit one record after completion."""

    collection_timing = PUBLIC_BELIEF_MATRIX_SUPERVISION_BOUNDARY

    def __init__(
        self,
        perceiver: SpeechPerceiver,
        *,
        sidecar: SpeechFidelityAuditSidecar,
        game_id: str,
        seed: int,
    ) -> None:
        if not isinstance(perceiver, SpeechPerceiver):
            raise TypeError("perceiver must be SpeechPerceiver")
        if not isinstance(game_id, str) or not game_id.strip():
            raise ValueError("game_id must be non-empty text")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        self.perceiver = perceiver
        self.sidecar = sidecar
        self.game_id = game_id
        self.seed = seed
        self._pending: tuple[dict[str, Any], SpeechParseAuditResult] | None = None

    def parse(
        self,
        speaker: int,
        speech: str,
        day: int,
        phase: str,
        context: dict | None = None,
    ) -> list[list[str]]:
        del context
        if self._pending is not None:
            raise RuntimeError("previous speech audit record was not consumed")
        try:
            result = self.perceiver.parse_with_audit(
                speaker=speaker,
                speech=speech,
                day=day,
                phase=phase,
            )
        except Exception as exc:
            result = SpeechParseAuditResult(
                normalized_actions=[],
                raw_response=None,
                parse_status="parser_error",
                protected_self_claim_actions=[],
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            self._pending = (
                {
                    "speaker": normalize_player(speaker),
                    "raw_public_speech": speech,
                },
                result,
            )
            raise
        self._pending = (
            {
                "speaker": normalize_player(speaker),
                "raw_public_speech": speech,
            },
            result,
        )
        return result.normalized_actions

    def record(
        self,
        env,
        *,
        step_idx: int,
        trigger: str,
        phase: str,
        speaker_id: int,
        observer_ids=None,
    ) -> dict[str, Any]:
        del observer_ids
        if trigger not in {"speech", "speech_pk"}:
            raise ValueError("speech fidelity audit requires a completed speech")
        if self._pending is None:
            raise RuntimeError("completed speech has no pending parser audit")
        pending, result = self._pending
        speech_events = [
            event
            for event in env.public_events
            if event.get("event_type") == "public_speech"
        ]
        if not speech_events:
            raise RuntimeError("completed speech has no public_speech event")
        event = speech_events[-1]
        speaker = normalize_player(speaker_id)
        if (
            event.get("speaker") != speaker
            or pending["speaker"] != speaker
            or event.get("raw_text") != pending["raw_public_speech"]
            or event.get("sp_actions") != result.normalized_actions
        ):
            raise ValueError("completed speech and parser audit do not match")
        record = {
            "audit_schema_version": SPEECH_FIDELITY_AUDIT_SCHEMA_VERSION,
            "game_id": self.game_id,
            "seed": self.seed,
            "step_idx": step_idx,
            "phase": phase,
            "speaker": speaker,
            "raw_public_speech": pending["raw_public_speech"],
            "speech_perceiver_raw_response": result.raw_response,
            "normalized_sp_actions": result.normalized_actions,
            "parse_status": result.parse_status,
            "parse_error_type": result.error_type,
            "parse_error_message": result.error_message,
            "protected_self_claim_actions": result.protected_self_claim_actions,
            "public_speech_event_idx": event["event_idx"],
            "source_commit": self.sidecar.source_commit,
            "backend_id": self.sidecar.backend_id,
            "model_id": self.sidecar.model_id,
            "speech_perceiver_temperature": 0,
            "speech_perceiver_max_tokens": SPEECH_PARSER_MAX_TOKENS,
            "enable_thinking": False,
        }
        self.sidecar.write(record)
        self._pending = None
        return record


__all__ = [
    "AuditedSpeechPerceiver",
    "SPEECH_FIDELITY_AUDIT_SCHEMA_VERSION",
    "SpeechFidelityAuditSidecar",
]
