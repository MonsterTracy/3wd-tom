"""Archived targets from raw subjective belief reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from archive.legacy_tom.werewolf.models.tom.schema import (
    PLAYER_NAMES,
    normalize_player,
)


def _canonical_player(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a canonical player name")
    player = normalize_player(value)
    if player != value:
        raise ValueError(f"{field} must be a canonical player name")
    return player


def suspicion_to_row(suspected_werewolves: Any) -> tuple[float, ...]:
    """Normalize one valid subjective suspicion set over seven players."""

    if (
        isinstance(suspected_werewolves, (str, bytes))
        or not isinstance(suspected_werewolves, Sequence)
    ):
        raise TypeError("suspected_werewolves must be a sequence")
    suspected = [
        _canonical_player(value, field="suspected_werewolves item")
        for value in suspected_werewolves
    ]
    if len(suspected) != len(set(suspected)):
        raise ValueError("suspected_werewolves must not contain duplicates")
    probability = 1.0 / (len(suspected) or len(PLAYER_NAMES))
    suspected_set = set(suspected)
    if not suspected:
        return tuple(probability for _ in PLAYER_NAMES)
    return tuple(
        probability if player in suspected_set else 0.0
        for player in PLAYER_NAMES
    )


def materialize_target(
    *,
    alive_observers: Any,
    observer_reports: Any,
) -> tuple[tuple[tuple[float, ...], ...], tuple[bool, ...]]:
    """Return the canonical 7x7 target and alive-valid observer mask."""

    if (
        isinstance(alive_observers, (str, bytes))
        or not isinstance(alive_observers, Sequence)
    ):
        raise TypeError("alive_observers must be a sequence")
    alive = [
        _canonical_player(value, field="alive observer")
        for value in alive_observers
    ]
    if not alive:
        raise ValueError("alive_observers must not be empty")
    if len(alive) != len(set(alive)):
        raise ValueError("alive_observers must not contain duplicates")
    alive_set = set(alive)

    if (
        isinstance(observer_reports, (str, bytes))
        or not isinstance(observer_reports, Sequence)
    ):
        raise TypeError("observer_reports must be a sequence")
    by_observer: dict[str, tuple[tuple[float, ...], bool]] = {}
    zero_row = tuple(0.0 for _ in PLAYER_NAMES)
    for report in observer_reports:
        if not isinstance(report, Mapping):
            raise TypeError("each observer report must be a mapping")
        observer = _canonical_player(
            report.get("observer_id"),
            field="report observer_id",
        )
        if observer not in alive_set:
            raise ValueError(f"report for dead observer: {observer}")
        if observer in by_observer:
            raise ValueError(f"duplicate observer report: {observer}")
        valid = report.get("valid")
        if type(valid) is not bool:
            raise TypeError("report valid must be boolean")
        if "suspected_werewolves" not in report:
            raise ValueError("report must contain suspected_werewolves")
        suspicion = report["suspected_werewolves"]
        if valid:
            row = suspicion_to_row(suspicion)
        else:
            if suspicion is not None:
                raise ValueError(
                    "invalid report suspected_werewolves must be None"
                )
            row = zero_row
        by_observer[observer] = (row, valid)

    missing = [player for player in alive if player not in by_observer]
    if missing:
        raise ValueError(f"missing alive observer reports: {missing}")

    rows = []
    mask = []
    for observer in PLAYER_NAMES:
        row, valid = by_observer.get(observer, (zero_row, False))
        rows.append(row)
        mask.append(valid)
    return tuple(rows), tuple(mask)


__all__ = ["materialize_target", "suspicion_to_row"]
