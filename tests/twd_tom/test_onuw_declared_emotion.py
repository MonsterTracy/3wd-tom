import json

import pytest

from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
from werewolf.speech.onuw_declared_emotion import (
    parse_declared_multimodal_speech,
)
from werewolf.speech.speech_perceiver import SpeechParseAuditResult


ROLES = [
    "Werewolf",
    "Werewolf",
    "Seer",
    "Witch",
    "Villager",
    "Villager",
    "Villager",
]


class EmptyParser:
    def parse_with_audit(self, **_kwargs):
        return SpeechParseAuditResult(
            normalized_actions=[],
            raw_response="NONE",
            parse_status="ok",
            error_type=None,
            error_message=None,
        )


def _speech_ready_env():
    env = WerewolfTextEnvV0(
        log_save_path=None, speech_perceiver=EmptyParser()
    )
    env.reset(roles=ROLES)
    env.step(("kill", 5))
    env.step(("kill", 5))
    env.step(("check", 1))
    env.step(("witch_pass", 0))
    return env


def test_declared_emotions_are_generated_fields_not_missing_as_other():
    declared = parse_declared_multimodal_speech(
        json.dumps({"speech": "公开发言", "face": "neutral", "tone": "other"})
    )
    assert declared.speech == "公开发言"
    assert declared.face == "neutral"
    assert declared.tone == "other"
    with pytest.raises(ValueError, match="fields do not match"):
        parse_declared_multimodal_speech(json.dumps({"speech": "公开发言"}))


def test_environment_keeps_declared_emotion_sidecar_without_public_input_leak():
    env = _speech_ready_env()
    speaker = f"player{env.current_act_idx + 1}"
    env.step(
        (
            "speech",
            {"speech": "公开发言", "face": "fear", "tone": "neutral"},
        )
    )
    assert env.public_events[-2]["raw_text"] == "公开发言"
    assert set(env.public_events[-2]) == {
        "event_idx", "event_type", "speaker", "raw_text"
    }
    assert env.speech_emotions == [
        {
            "event_idx": env.public_events[-2]["event_idx"],
            "speaker": speaker,
            "face": "fear",
            "tone": "neutral",
            "source": "agent_declared",
        }
    ]
