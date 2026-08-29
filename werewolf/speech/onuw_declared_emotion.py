"""One-call public speech plus ONUW 8-class face/tone declaration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from werewolf.models.twd_tom.onuw_parity_protocol import EMOTION_NAMES


@dataclass(frozen=True)
class DeclaredMultimodalSpeech:
    speech: str
    face: str
    tone: str

    def as_action_content(self) -> dict[str, str]:
        return {"speech": self.speech, "face": self.face, "tone": self.tone}


def declared_speech_response_format(*, supports_json_schema: bool) -> dict[str, Any]:
    if supports_json_schema is not True:
        raise ValueError("declared multimodal speech requires JSON Schema support")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "onuw_agent_declared_multimodal_speech",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["speech", "face", "tone"],
                "properties": {
                    "speech": {"type": "string", "minLength": 1},
                    "face": {"type": "string", "enum": list(EMOTION_NAMES)},
                    "tone": {"type": "string", "enum": list(EMOTION_NAMES)},
                },
            },
        },
    }


def parse_declared_multimodal_speech(raw: Any) -> DeclaredMultimodalSpeech:
    if not isinstance(raw, str):
        raise TypeError("declared multimodal speech response must be text")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("declared multimodal speech must be strict JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"speech", "face", "tone"}:
        raise ValueError("declared multimodal speech fields do not match contract")
    speech = payload["speech"]
    if not isinstance(speech, str) or not speech.strip():
        raise ValueError("declared speech must be non-empty text")
    for field in ("face", "tone"):
        if payload[field] not in EMOTION_NAMES:
            raise ValueError(f"declared {field} must use the 8-class vocabulary")
    return DeclaredMultimodalSpeech(
        speech=speech.strip(), face=payload["face"], tone=payload["tone"]
    )


__all__ = [
    "DeclaredMultimodalSpeech",
    "declared_speech_response_format",
    "parse_declared_multimodal_speech",
]
