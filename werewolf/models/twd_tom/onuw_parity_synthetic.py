"""Deterministic synthetic material for P0 end-to-end verification."""

from __future__ import annotations

from copy import deepcopy

from werewolf.models.twd_tom.onuw_parity_dataset import (
    PARITY_GAME_SCHEMA_VERSION,
    bos_token,
)
from werewolf.models.twd_tom.onuw_parity_protocol import (
    CLASSIC7_ONUW_REFERENCE,
    ONUW_ACTION_ONLY,
    ONUW_AGENT_DECLARED_MULTIMODAL,
)
from werewolf.models.twd_tom.schema import PLAYER_NAMES


def _target(*support: int, dead_observers: tuple[int, ...] = ()) -> list[list[float]]:
    if support:
        row = [1.0 / len(support) if index in support else 0.0 for index in range(7)]
    else:
        row = [1.0 / 7.0] * 7
    return [
        [0.0] * 7 if index in dead_observers else list(row)
        for index in range(7)
    ]


def synthetic_parity_games() -> list[dict]:
    """Two variable-length games, including empty support and context collision."""

    first = {
        "schema_version": PARITY_GAME_SCHEMA_VERSION,
        "protocol_id": CLASSIC7_ONUW_REFERENCE,
        "game_id": "synthetic_parity_001",
        "content_profile": ONUW_ACTION_ONLY,
        "modality_profile": ONUW_AGENT_DECLARED_MULTIMODAL,
        "tokens": [
            bos_token(),
            {
                "token_type": "speech_action",
                "subject": "player1",
                "action": "point_as_werewolf",
                "object": "player7",
                "face": "neutral",
                "tone": "other",
                "phase": None,
                "day": 0,
            },
        ],
        "queries": [
            {
                "query_id": "q0",
                "step_idx": 0,
                "speaker": "player1",
                "token_cutoff": 0,
                "observer_ids": list(PLAYER_NAMES),
                "belief_target": _target(),
            },
            {
                "query_id": "q1",
                "step_idx": 1,
                "speaker": "player2",
                "token_cutoff": 1,
                "observer_ids": list(PLAYER_NAMES),
                "belief_target": _target(6),
            },
            {
                "query_id": "q2",
                "step_idx": 2,
                "speaker": "player3",
                "token_cutoff": 1,
                "observer_ids": list(PLAYER_NAMES),
                "belief_target": _target(4, 6),
            },
        ],
        "speech_action_counts": [1, 0],
    }
    second = deepcopy(first)
    second["game_id"] = "synthetic_parity_002"
    second["tokens"].append(
        {
            "token_type": "speech_action",
            "subject": "player2",
            "action": "support",
            "object": "player1",
            "face": "happy",
            "tone": "neutral",
            "phase": None,
            "day": 0,
        }
    )
    second["queries"] = [
        deepcopy(first["queries"][0]),
        deepcopy(first["queries"][1]),
        {
            "query_id": "q2",
            "step_idx": 2,
            "speaker": "player4",
            "token_cutoff": 2,
            "observer_ids": list(PLAYER_NAMES[:-1]),
            "belief_target": _target(0, 6, dead_observers=(6,)),
        },
    ]
    second["speech_action_counts"] = [1, 1]
    return [first, second]


__all__ = ["synthetic_parity_games"]
