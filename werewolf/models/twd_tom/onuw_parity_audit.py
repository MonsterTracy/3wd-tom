"""Pilot audits for sequence capacity and action-only information loss."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections import Counter
from typing import Any

import numpy as np

from werewolf.models.twd_tom.onuw_parity_dataset import validate_parity_game
from werewolf.speech.onuw_role_guess_perceiver import role_guess_audit


def _distribution_distances(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    tv = float(0.5 * np.abs(left - right).sum())
    midpoint = 0.5 * (left + right)
    left_positive = left > 0
    right_positive = right > 0
    left_kl = float(
        np.sum(left[left_positive] * np.log(left[left_positive] / midpoint[left_positive]))
    )
    right_kl = float(
        np.sum(
            right[right_positive]
            * np.log(right[right_positive] / midpoint[right_positive])
        )
    )
    return tv, 0.5 * (left_kl + right_kl)


def action_only_information_loss_audit(game: Mapping[str, Any]) -> dict[str, Any]:
    """Quantify collisions caused by projecting speech to structured actions."""

    normalized = validate_parity_game(game)
    counts = normalized["speech_action_counts"]
    queries = normalized["queries"]
    zero_count = sum(count == 0 for count in counts)
    shared_pairs = 0
    different_pairs = 0
    row_tv = []
    row_js = []
    for previous, current in zip(queries, queries[1:]):
        if previous["token_cutoff"] != current["token_cutoff"]:
            continue
        shared_pairs += 1
        previous_alive = set(previous["observer_ids"])
        current_alive = set(current["observer_ids"])
        pair_different = False
        for row_index, player in enumerate(
            (f"player{index}" for index in range(1, 8))
        ):
            if player not in previous_alive or player not in current_alive:
                continue
            left = np.asarray(previous["belief_target"][row_index], dtype=np.float64)
            right = np.asarray(current["belief_target"][row_index], dtype=np.float64)
            tv, js = _distribution_distances(left, right)
            row_tv.append(tv)
            row_js.append(js)
            pair_different = pair_different or tv > 1e-12
        different_pairs += int(pair_different)
    return {
        "speech_count": len(counts),
        "zero_action_speech_count": zero_count,
        "zero_action_speech_rate": zero_count / len(counts) if counts else 0.0,
        "consecutive_query_pair_count": max(0, len(queries) - 1),
        "consecutive_queries_sharing_token_cutoff_count": shared_pairs,
        "consecutive_queries_sharing_token_cutoff_rate": (
            shared_pairs / (len(queries) - 1) if len(queries) > 1 else 0.0
        ),
        "same_context_different_target_count": different_pairs,
        "same_context_different_target_rate": (
            different_pairs / shared_pairs if shared_pairs else 0.0
        ),
        "shared_context_adjacent_target_tv_mean": (
            float(np.mean(row_tv)) if row_tv else 0.0
        ),
        "shared_context_adjacent_target_js_mean": (
            float(np.mean(row_js)) if row_js else 0.0
        ),
        "shared_context_compared_observer_row_count": len(row_tv),
    }


def sequence_capacity_audit(games: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report lengths without choosing a max capacity or truncating data."""

    if isinstance(games, (str, bytes)) or not isinstance(games, Sequence) or not games:
        raise ValueError("games must be a non-empty sequence")
    normalized = [validate_parity_game(game) for game in games]
    sequence_lengths = np.asarray([len(game["tokens"]) for game in normalized])
    query_prefixes = [
        (query["token_cutoff"] + 1, game["game_id"], query["query_id"])
        for game in normalized
        for query in game["queries"]
    ]
    longest = max(query_prefixes)
    percentiles = np.percentile(sequence_lengths, [50, 90, 95, 99])
    return {
        "game_count": len(normalized),
        "query_count": len(query_prefixes),
        "sequence_length_p50": float(percentiles[0]),
        "sequence_length_p90": float(percentiles[1]),
        "sequence_length_p95": float(percentiles[2]),
        "sequence_length_p99": float(percentiles[3]),
        "sequence_length_max": int(sequence_lengths.max()),
        "longest_query_prefix": int(longest[0]),
        "longest_query_game_id": longest[1],
        "longest_query_id": longest[2],
        "exceeds_256": bool(sequence_lengths.max() > 256 or longest[0] > 256),
        "capacity_decision": "unfrozen",
    }


def pilot_collection_audit(
    games: Sequence[Mapping[str, Any]],
    collection_audits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate the review statistics required before formal collection."""

    normalized = [validate_parity_game(game) for game in games]
    if len(normalized) != len(collection_audits):
        raise ValueError("collection audits must match games one-to-one")
    audit_by_game = {}
    for audit in collection_audits:
        if not isinstance(audit, Mapping):
            raise TypeError("each collection audit must be a mapping")
        game_id = audit.get("game_id")
        if not isinstance(game_id, str) or game_id in audit_by_game:
            raise ValueError("collection audit game_id must be unique")
        if audit.get("model_input") != "public_only":
            raise ValueError("parity collection audit must declare public-only input")
        audit_by_game[game_id] = audit
    if set(audit_by_game) != {game["game_id"] for game in normalized}:
        raise ValueError("collection audit game IDs must exactly match games")

    support_sizes = Counter()
    role_report_count = 0
    empty_count = 0
    conflict_count = 0
    face_counts = Counter()
    tone_counts = Counter()
    speech_count = 0
    emotion_count = 0
    information_loss = []
    for game in normalized:
        audit = audit_by_game[game["game_id"]]
        queries = audit.get("queries")
        if not isinstance(queries, Sequence) or len(queries) != len(game["queries"]):
            raise ValueError("label audit queries must match game PRE queries")
        for query in queries:
            reports = query.get("role_guess_reports")
            if not isinstance(reports, Mapping):
                raise TypeError("role_guess_reports must be a mapping")
            for report in reports.values():
                guesses = report.get("role_guesses")
                report_audit = role_guess_audit(guesses)
                support_size = sum(role == "werewolf" for role in guesses.values())
                support_sizes[support_size] += 1
                role_report_count += 1
                empty_count += int(support_size == 0)
                conflict_count += int(report_audit["role_count_conflict"])
        speech_count += len(game["speech_action_counts"])
        emotions = audit.get("speech_emotions")
        if isinstance(emotions, (str, bytes)) or not isinstance(emotions, Sequence):
            raise TypeError("speech_emotions must be a sequence")
        emotion_count += len(emotions)
        for emotion in emotions:
            face_counts[emotion["face"]] += 1
            tone_counts[emotion["tone"]] += 1
        information_loss.append(action_only_information_loss_audit(game))
    return {
        "game_count": len(normalized),
        "query_count": sum(len(game["queries"]) for game in normalized),
        "role_report_count": role_report_count,
        "support_size_histogram": {
            str(size): count for size, count in sorted(support_sizes.items())
        },
        "empty_support_count": empty_count,
        "empty_support_rate": empty_count / role_report_count,
        "role_count_conflict_count": conflict_count,
        "role_count_conflict_rate": conflict_count / role_report_count,
        "speech_count": speech_count,
        "declared_emotion_count": emotion_count,
        "declared_emotion_coverage": (
            emotion_count / speech_count if speech_count else 0.0
        ),
        "face_histogram": dict(sorted(face_counts.items())),
        "tone_histogram": dict(sorted(tone_counts.items())),
        "sequence_capacity": sequence_capacity_audit(normalized),
        "information_loss_by_game": information_loss,
    }


__all__ = [
    "action_only_information_loss_audit",
    "sequence_capacity_audit",
    "pilot_collection_audit",
]
