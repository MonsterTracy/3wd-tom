"""Map one direct-report snapshot to one current-actor ToM data sample."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from werewolf.models.twd_tom.schema import (
    NUM_WOLF_PAIR_CLASSES,
    PLAYER_NAMES,
    canonical_wolf_pairs,
)
from werewolf.models.twd_tom.samples import ACTOR_PAIR_BELIEF_SCHEMA_VERSION
from werewolf.speech.pair_belief_self_reporter import (
    STATUS_OK,
    canonical_json_sha256,
    validate_pair_probabilities,
)


ACTOR_PERSPECTIVE_TARGET_SCHEMA_VERSION = (
    "classic7_fixed_two_wolves_actor_perspective_pair_targets_v1"
)


def build_actor_perspective_sample(raw_snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Build exactly one reasoning-player sample from one raw snapshot."""

    if not isinstance(raw_snapshot, Mapping):
        raise TypeError("raw_snapshot must be a mapping")
    if raw_snapshot.get("schema_version") != ACTOR_PAIR_BELIEF_SCHEMA_VERSION:
        raise ValueError("unsupported actor pair-belief raw schema")
    reasoning_player = raw_snapshot.get("reasoning_player_id")
    if reasoning_player != raw_snapshot.get("current_speaker"):
        raise ValueError("reasoning_player_id must equal current_speaker")
    if reasoning_player not in PLAYER_NAMES:
        raise ValueError("reasoning_player_id must be canonical")
    reports = raw_snapshot.get("player_reports")
    if not isinstance(reports, list) or len(reports) != len(PLAYER_NAMES):
        raise ValueError("player_reports must contain seven canonical rows")
    if [report.get("player_id") for report in reports] != list(PLAYER_NAMES):
        raise ValueError("player_reports must use canonical player order")

    report_by_player = {report["player_id"]: report for report in reports}
    reasoning_report = report_by_player[reasoning_player]
    self_target = _validated_target_or_none(reasoning_report)
    other_player_ids = [
        player for player in PLAYER_NAMES if player != reasoning_player
    ]
    other_targets: list[list[float]] = []
    other_mask: list[bool] = []
    for player in other_player_ids:
        target = _validated_target_or_none(report_by_player[player])
        is_valid = target is not None
        other_mask.append(is_valid)
        other_targets.append(
            target if target is not None else [0.0] * NUM_WOLF_PAIR_CLASSES
        )

    public_history = deepcopy(raw_snapshot.get("public_events"))
    private_knowledge = deepcopy(
        raw_snapshot.get("reasoning_player_private_knowledge")
    )
    reasoning_input = {
        "public_history": public_history,
        "reasoning_player_id": reasoning_player,
        "legal_private_knowledge": private_knowledge,
    }
    expected_input = raw_snapshot.get("reasoning_input_payload")
    if reasoning_input != expected_input:
        raise ValueError("reasoning input does not match the raw canonical payload")
    if canonical_json_sha256(reasoning_input) != raw_snapshot.get(
        "reasoning_input_payload_sha256"
    ):
        raise ValueError("reasoning input payload digest mismatch")

    return {
        "schema_version": ACTOR_PERSPECTIVE_TARGET_SCHEMA_VERSION,
        "source_schema_version": ACTOR_PAIR_BELIEF_SCHEMA_VERSION,
        "game_id": raw_snapshot.get("game_id"),
        "event_idx": raw_snapshot.get("event_idx"),
        "day": raw_snapshot.get("day"),
        "phase": raw_snapshot.get("phase"),
        "reasoning_player_id": reasoning_player,
        "reasoning_input": reasoning_input,
        "reasoning_input_sha256": canonical_json_sha256(reasoning_input),
        "self_pair_target": self_target,
        "self_target_mask": self_target is not None,
        "other_player_ids": other_player_ids,
        "other_pair_targets": other_targets,
        "other_target_mask": other_mask,
        "pair_ordering": [list(pair) for pair in canonical_wolf_pairs()],
    }


def _validated_target_or_none(report: Mapping[str, Any]) -> list[float] | None:
    if report.get("report_status") != STATUS_OK:
        if report.get("pair_probabilities") is not None:
            raise ValueError("non-ok report must not provide a target")
        return None
    if report.get("alive") is not True:
        raise ValueError("an ok report must belong to an alive player")
    target, _ = validate_pair_probabilities(
        report.get("pair_probabilities"),
        known_werewolves=report.get("known_werewolves"),
        known_non_werewolves=report.get("known_non_werewolves"),
    )
    return target


__all__ = [
    "ACTOR_PERSPECTIVE_TARGET_SCHEMA_VERSION",
    "build_actor_perspective_sample",
]
