"""Deterministic normalized-suspicion targets for Public Belief Matrix V1."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from werewolf.models.twd_tom.schema import (
    CANONICAL_PLAYER_ORDERING,
    canonicalize_player_set,
)


_REPORT_STATUSES = frozenset({"ok", "parse_error", "reporter_error"})


@dataclass(frozen=True)
class PublicBeliefMatrixTarget:
    """One canonical 7x7 target and its valid-observer row mask."""

    matrix_target: tuple[tuple[float, ...], ...]
    observer_row_mask: tuple[bool, ...]


def suspicion_set_to_row_target(
    suspected_werewolves: Any,
) -> tuple[float, ...]:
    """Encode one symbolic suspicion set as a normalized seven-player row."""

    suspected = canonicalize_player_set(
        suspected_werewolves,
        field_name="suspected_werewolves",
    )
    if not suspected:
        probability = 1.0 / len(CANONICAL_PLAYER_ORDERING)
        return tuple(probability for _ in CANONICAL_PLAYER_ORDERING)
    probability = 1.0 / len(suspected)
    suspected_set = set(suspected)
    return tuple(
        probability if player in suspected_set else 0.0
        for player in CANONICAL_PLAYER_ORDERING
    )


def suspicion_reports_to_matrix_target(
    reports: Sequence[Mapping[str, Any]],
) -> PublicBeliefMatrixTarget:
    """Encode exactly one report per canonical observer, failing closed."""

    if isinstance(reports, (str, bytes)) or not isinstance(reports, Sequence):
        raise TypeError("reports must be a sequence")
    by_observer: dict[str, Mapping[str, Any]] = {}
    for report in reports:
        if not isinstance(report, Mapping):
            raise TypeError("each report must be a mapping")
        observer = report.get("observer")
        if (
            not isinstance(observer, str)
            or observer not in CANONICAL_PLAYER_ORDERING
        ):
            raise ValueError("observer must be a canonical player1...player7 name")
        if observer in by_observer:
            raise ValueError(f"duplicate observer report: {observer}")
        by_observer[observer] = report
    missing = [
        observer
        for observer in CANONICAL_PLAYER_ORDERING
        if observer not in by_observer
    ]
    if missing:
        raise ValueError(f"missing observer reports: {missing}")

    zero_row = tuple(0.0 for _ in CANONICAL_PLAYER_ORDERING)
    rows: list[tuple[float, ...]] = []
    mask: list[bool] = []
    for observer in CANONICAL_PLAYER_ORDERING:
        report = by_observer[observer]
        status = report.get("status")
        if not isinstance(status, str):
            raise TypeError("report status must be text")
        if status not in _REPORT_STATUSES:
            raise ValueError(f"unsupported report status: {status}")
        if "suspected_werewolves" not in report:
            raise ValueError("report must contain suspected_werewolves")
        valid = status == "ok"
        suspicion = report["suspected_werewolves"]
        if not valid and suspicion is not None:
            raise ValueError(
                "non-ok report suspected_werewolves must be None"
            )
        rows.append(
            suspicion_set_to_row_target(suspicion)
            if valid
            else zero_row
        )
        mask.append(valid)
    return PublicBeliefMatrixTarget(tuple(rows), tuple(mask))


__all__ = [
    "PublicBeliefMatrixTarget",
    "suspicion_reports_to_matrix_target",
    "suspicion_set_to_row_target",
]
