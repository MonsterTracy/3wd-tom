"""Per-game backend-call and wall-clock budgets for canonical collection."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Callable, Mapping

from werewolf.backends import BackendError
from werewolf.trajectory import (
    canonical_digest,
    sanitize_exception_message,
    serialize_json_value,
)


CALL_AUDIT_SCHEMA_VERSION = "classic7_collection_call_audit_v2"
BACKEND_MAX_ATTEMPTS = 3


def _positive_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _positive_number(value: Any, *, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value <= 0
    ):
        raise ValueError(f"{field_name} must be a positive number")
    return float(value)


class CollectionBudgetExceeded(RuntimeError):
    """Raised before another call or after elapsed time exceeds the contract."""


class GameCallBudgetAudit:
    """Count every dispatched backend call in one synchronous game."""

    def __init__(
        self,
        *,
        game_id: str,
        max_gameplay_calls: int,
        max_belief_calls: int,
        max_total_calls: int,
        max_wall_seconds: float,
        max_backend_attempts: int = BACKEND_MAX_ATTEMPTS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(game_id, str) or not game_id.strip():
            raise ValueError("game_id must be non-empty text")
        self.game_id = game_id
        self.max_gameplay_calls = _positive_integer(
            max_gameplay_calls,
            field_name="max_gameplay_calls",
        )
        self.max_belief_calls = _positive_integer(
            max_belief_calls,
            field_name="max_belief_calls",
        )
        self.max_total_calls = _positive_integer(
            max_total_calls,
            field_name="max_total_calls",
        )
        if self.max_total_calls < max(
            self.max_gameplay_calls,
            self.max_belief_calls,
        ):
            raise ValueError(
                "max_total_calls cannot be smaller than a category budget"
            )
        self.max_wall_seconds = _positive_number(
            max_wall_seconds,
            field_name="max_wall_seconds",
        )
        self.max_backend_attempts = _positive_integer(
            max_backend_attempts,
            field_name="max_backend_attempts",
        )
        self._clock = clock
        self._started_at = float(clock())
        self._active_category: str | None = None
        self._gameplay_calls = 0
        self._belief_calls = 0
        self._unscoped_gameplay_calls = 0
        self._report_sequence = 0
        self._backend_retry_events: list[dict[str, Any]] = []
        self._gameplay_fallback_events: list[dict[str, Any]] = []

    def _elapsed_seconds(self) -> float:
        return max(0.0, float(self._clock()) - self._started_at)

    def assert_wall_budget(self) -> None:
        elapsed = self._elapsed_seconds()
        if elapsed > self.max_wall_seconds:
            raise CollectionBudgetExceeded(
                "per-game wall-clock budget exceeded: "
                f"elapsed={elapsed:.6f}s max={self.max_wall_seconds:.6f}s"
            )

    @contextmanager
    def _category_context(self, category: str):
        self.assert_wall_budget()
        previous = self._active_category
        self._active_category = category
        try:
            yield
        finally:
            self._active_category = previous
            self.assert_wall_budget()

    def gameplay_context(self, **_context):
        return self._category_context("gameplay")

    def prepare_report(self, *, report_prompt: str, **_context):
        self._report_sequence += 1
        return f"belief_report_{self._report_sequence:06d}", report_prompt

    def belief_context(self, _report_id: str):
        return self._category_context("belief")

    def complete_report(self, _report_id: str, _raw_response) -> None:
        return None

    def record_agent_state(self, **_state) -> None:
        return None

    def before_backend_call(self) -> None:
        self.assert_wall_budget()
        category = self._active_category or "gameplay"
        total_calls = self._gameplay_calls + self._belief_calls
        if total_calls >= self.max_total_calls:
            raise CollectionBudgetExceeded(
                f"per-game total call budget exhausted: {self.max_total_calls}"
            )
        if category == "belief":
            if self._belief_calls >= self.max_belief_calls:
                raise CollectionBudgetExceeded(
                    "per-game belief call budget exhausted: "
                    f"{self.max_belief_calls}"
                )
            self._belief_calls += 1
            return
        if self._gameplay_calls >= self.max_gameplay_calls:
            raise CollectionBudgetExceeded(
                "per-game gameplay call budget exhausted: "
                f"{self.max_gameplay_calls}"
            )
        self._gameplay_calls += 1
        if self._active_category is None:
            self._unscoped_gameplay_calls += 1

    def after_backend_call(self) -> None:
        self.assert_wall_budget()

    def record_backend_retry(
        self,
        *,
        method_name: str,
        failed_attempt: int,
        exception: BackendError,
    ) -> None:
        self._backend_retry_events.append(
            {
                "category": self._active_category or "gameplay",
                "method_name": method_name,
                "failed_attempt": failed_attempt,
                "next_attempt": failed_attempt + 1,
                "exception_type": type(exception).__name__,
                "exception_message": sanitize_exception_message(exception),
            }
        )

    def record_gameplay_fallback(
        self,
        *,
        step_idx: int,
        acting_player_id: int,
        phase: str,
        action: Any,
        exception: Exception,
    ) -> None:
        self._gameplay_fallback_events.append(
            {
                "step_idx": step_idx,
                "acting_player_id": acting_player_id,
                "phase": phase,
                "action": serialize_json_value(action),
                "exception_type": type(exception).__name__,
                "exception_message": sanitize_exception_message(exception),
            }
        )

    def snapshot(self) -> dict[str, Any]:
        elapsed = self._elapsed_seconds()
        total_calls = self._gameplay_calls + self._belief_calls
        snapshot = {
            "schema_version": CALL_AUDIT_SCHEMA_VERSION,
            "game_id": self.game_id,
            "gameplay_call_count": self._gameplay_calls,
            "belief_call_count": self._belief_calls,
            "total_call_count": total_calls,
            "unscoped_gameplay_call_count": self._unscoped_gameplay_calls,
            "elapsed_wall_seconds": elapsed,
            "max_gameplay_calls": self.max_gameplay_calls,
            "max_belief_calls": self.max_belief_calls,
            "max_total_calls": self.max_total_calls,
            "max_wall_seconds": self.max_wall_seconds,
            "max_backend_attempts": self.max_backend_attempts,
            "backend_retry_count": len(self._backend_retry_events),
            "backend_retry_events": list(self._backend_retry_events),
            "gameplay_fallback_count": len(self._gameplay_fallback_events),
            "gameplay_fallback_events": list(self._gameplay_fallback_events),
            "within_budget": (
                self._gameplay_calls <= self.max_gameplay_calls
                and self._belief_calls <= self.max_belief_calls
                and total_calls <= self.max_total_calls
                and elapsed <= self.max_wall_seconds
            ),
        }
        snapshot["audit_digest"] = canonical_digest(snapshot)
        return snapshot


class AuditedBackend:
    """Backend proxy with explicit, counted transient-error attempts."""

    def __init__(self, backend, audit: GameCallBudgetAudit) -> None:
        self._backend = backend
        self._audit = audit

    def __getattr__(self, name: str):
        return getattr(self._backend, name)

    def _dispatch(self, method_name: str, *args, **kwargs):
        for attempt in range(1, self._audit.max_backend_attempts + 1):
            self._audit.before_backend_call()
            try:
                return getattr(self._backend, method_name)(*args, **kwargs)
            except BackendError as exc:
                if attempt >= self._audit.max_backend_attempts:
                    raise
                self._audit.record_backend_retry(
                    method_name=method_name,
                    failed_attempt=attempt,
                    exception=exc,
                )
            finally:
                self._audit.after_backend_call()

    def chat(self, *args, **kwargs):
        return self._dispatch("chat", *args, **kwargs)

    def chat_with_metadata(self, *args, **kwargs):
        return self._dispatch("chat_with_metadata", *args, **kwargs)


def audited_backends(
    backends: Mapping[str, Any],
    audit: GameCallBudgetAudit,
) -> dict[str, AuditedBackend]:
    """Wrap one named backend map without changing the underlying clients."""

    return {
        name: AuditedBackend(backend, audit)
        for name, backend in backends.items()
    }


__all__ = [
    "CALL_AUDIT_SCHEMA_VERSION",
    "BACKEND_MAX_ATTEMPTS",
    "AuditedBackend",
    "CollectionBudgetExceeded",
    "GameCallBudgetAudit",
    "audited_backends",
]
