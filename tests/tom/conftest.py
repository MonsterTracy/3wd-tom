import json
from copy import deepcopy

import pytest


@pytest.fixture
def pilot_sample_factory():
    def make_sample(
        game_id,
        *,
        step_idx=1,
        seed=17,
        context="seer_witch",
        valid=True,
        action_count=1,
    ):
        actions = [
            ["player1", "support", "player2"]
            for _ in range(action_count)
        ]
        return {
            "game_id": game_id,
            "seed": seed,
            "episode_context": context,
            "step_idx": step_idx,
            "speaker_id": "player1",
            "round": 1,
            "phase": "speech",
            "formal_speech_actions": deepcopy(actions),
            "public_history_cutoff": {"event_idx": 2, "digest": "test"},
            "public_events": [
                {
                    "event_idx": 0,
                    "event_type": "death_announcement",
                    "dead_players": [],
                },
                {
                    "event_idx": 1,
                    "event_type": "phase_change",
                    "phase": "1_day_speech",
                },
                {
                    "event_idx": 2,
                    "event_type": "public_speech",
                    "speaker": "player1",
                    "raw_text": "synthetic",
                    "sp_actions": deepcopy(actions),
                },
            ],
            "alive_observers": ["player1"],
            "observer_reports": [
                {
                    "observer_id": "player1",
                    "valid": valid,
                    "suspected_werewolves": [] if valid else None,
                    "error": None if valid else "parse_error",
                }
            ],
        }

    return make_sample


@pytest.fixture
def write_jsonl():
    def write(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    return write
