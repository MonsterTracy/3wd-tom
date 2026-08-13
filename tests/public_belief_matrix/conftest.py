import hashlib
from copy import deepcopy

import pytest

from werewolf.models.public_belief_matrix.collection import (
    PUBLIC_BELIEF_MATRIX_PROVENANCE,
    PUBLIC_BELIEF_MATRIX_SAMPLE_SCHEMA_VERSION,
    PUBLIC_BELIEF_MATRIX_VISIBLE_PREFIX_SCHEMA_VERSION,
)
from werewolf.models.public_belief_matrix.public_prefix import (
    build_public_belief_matrix_visible_prefix,
    render_public_belief_matrix_visible_prefix,
)
from werewolf.models.twd_tom.schema import CANONICAL_PLAYER_ORDERING


@pytest.fixture
def pbm_sample_factory():
    def build(*, seed=876, snapshot_number=1, reports=None):
        events = [
            {"event_idx": 0, "event_type": "phase_change", "phase": "1_day_speech"},
            {"event_idx": 1, "event_type": "turn_start", "speaker": "player2"},
            {
                "event_idx": 2,
                "event_type": "public_speech",
                "speaker": "player2",
                "raw_text": "not serialized",
                "sp_actions": [["player2", "point_as_werewolf", "player3"]],
            },
        ]
        prefix = build_public_belief_matrix_visible_prefix(events)
        rendered = render_public_belief_matrix_visible_prefix(prefix)
        game_id = f"formal_pbm_game_{seed - 875:03d}_seed_{seed}"
        snapshot_id = f"{game_id}:pbm:{snapshot_number:06d}"
        if reports is None:
            reports = [
                {
                    "observer": observer,
                    "status": "ok",
                    "suspected_werewolves": [],
                    "error": None,
                    "reporter_backend_id": "shared",
                }
                for observer in CANONICAL_PLAYER_ORDERING
            ]
        return {
            "schema_version": PUBLIC_BELIEF_MATRIX_SAMPLE_SCHEMA_VERSION,
            "visible_prefix_schema_version": (
                PUBLIC_BELIEF_MATRIX_VISIBLE_PREFIX_SCHEMA_VERSION
            ),
            **deepcopy(PUBLIC_BELIEF_MATRIX_PROVENANCE),
            "game_id": game_id,
            "snapshot_id": snapshot_id,
            "step_idx": 3,
            "phase": "speech",
            "speaker": "player2",
            "public_speech_event_idx": 2,
            "structured_prefix": {
                field: value.tolist() for field, value in prefix.items()
            },
            "structured_prefix_digest": hashlib.sha256(
                rendered.encode("utf-8")
            ).hexdigest(),
            "publicly_alive_players": list(CANONICAL_PLAYER_ORDERING),
            "observer_alive_mask": [True] * 7,
            "observer_reports": deepcopy(reports),
            "reporter_backend_id": "shared",
            "reporter_model_id": "model",
        }

    return build
