"""Fail-fast probe for the two structured generations required by parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from werewolf.backends import load_named_backends, resolve_backend
from werewolf.runtime_config import normalize_runtime_config
from werewolf.speech.onuw_declared_emotion import (
    declared_speech_response_format,
    parse_declared_multimodal_speech,
)
from werewolf.speech.onuw_role_guess_perceiver import (
    parse_role_guess_response,
    role_guess_response_format,
)


def probe(*, config_path: Path, env_file: Path) -> dict:
    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    normalized = normalize_runtime_config(parsed)
    profile = normalized["agent_config"]["all_candidates"][0]
    backends = load_named_backends(normalized, env_file=env_file, max_retries=0)
    backend = resolve_backend(profile["backend"], backends)
    model = profile["model"]

    role_raw = backend.chat(
        messages=[
            {
                "role": "user",
                "content": (
                    "Return only JSON with role_guesses for player1 through "
                    "player7. Use only werewolf, villager, seer, witch, or unknown."
                ),
            }
        ],
        model=model,
        temperature=0.0,
        max_tokens=192,
        response_format=role_guess_response_format(supports_json_schema=True),
        extra_body={"thinking": {"type": "disabled"}},
    )
    role_guesses = parse_role_guess_response(role_raw)

    speech_raw, metadata = backend.chat_with_metadata(
        messages=[
            {
                "role": "user",
                "content": (
                    "Return only JSON. speech must be a short Chinese Werewolf "
                    "statement; face and tone must each use one of sad, anger, "
                    "neutral, happy, surprise, fear, disgust, other."
                ),
            }
        ],
        model=model,
        temperature=0.0,
        max_tokens=128,
        response_format=declared_speech_response_format(
            supports_json_schema=True
        ),
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    speech = parse_declared_multimodal_speech(speech_raw)
    finish_reason = metadata.get("finish_reason") if isinstance(metadata, dict) else None
    if not isinstance(finish_reason, str) or finish_reason == "length":
        raise RuntimeError("declared speech probe did not finish normally")
    return {
        "status": "PASS",
        "model": model,
        "role_guess_player_count": len(role_guesses),
        "declared_face": speech.face,
        "declared_tone": speech.tone,
        "speech_finish_reason": finish_reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            probe(config_path=args.config, env_file=args.env_file),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
