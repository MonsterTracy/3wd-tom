import pytest

from script.twd_tom.collection_budget import (
    AuditedBackend,
    CollectionBudgetExceeded,
    GameCallBudgetAudit,
)
from werewolf.backends import BackendError
from werewolf.trajectory import canonical_digest


class FakeBackend:
    supports_json_schema = True

    def __init__(self):
        self.calls = []

    def chat(self, *args, **kwargs):
        self.calls.append(("chat", args, kwargs))
        return "ok"

    def chat_with_metadata(self, *args, **kwargs):
        self.calls.append(("chat_with_metadata", args, kwargs))
        return "ok", {"finish_reason": "stop"}


class TransientBackend(FakeBackend):
    def __init__(self, failures):
        super().__init__()
        self.failures = failures

    def chat(self, *args, **kwargs):
        self.calls.append(("chat", args, kwargs))
        if len(self.calls) <= self.failures:
            raise BackendError(f"transient-{len(self.calls)}")
        return "ok"


def make_audit(**overrides):
    values = {
        "game_id": "game-001",
        "max_gameplay_calls": 2,
        "max_belief_calls": 1,
        "max_total_calls": 3,
        "max_wall_seconds": 60.0,
    }
    values.update(overrides)
    return GameCallBudgetAudit(**values)


def test_backend_proxy_counts_actual_dispatches_by_context():
    audit = make_audit()
    backend = FakeBackend()
    proxy = AuditedBackend(backend, audit)

    with audit.gameplay_context():
        assert proxy.chat(messages=[]) == "ok"
    with audit.belief_context("report-1"):
        assert proxy.chat_with_metadata(messages=[])[0] == "ok"
    proxy.chat(messages=[])

    snapshot = audit.snapshot()
    assert snapshot["gameplay_call_count"] == 2
    assert snapshot["belief_call_count"] == 1
    assert snapshot["total_call_count"] == 3
    assert snapshot["unscoped_gameplay_call_count"] == 1
    assert snapshot["backend_retry_count"] == 0
    assert snapshot["gameplay_fallback_count"] == 0
    assert snapshot["within_budget"] is True
    payload = dict(snapshot)
    digest = payload.pop("audit_digest")
    assert digest == canonical_digest(payload)
    assert len(backend.calls) == 3


def test_budget_rejects_before_dispatching_an_excess_call():
    audit = make_audit(max_gameplay_calls=1, max_total_calls=2)
    backend = FakeBackend()
    proxy = AuditedBackend(backend, audit)

    proxy.chat(messages=[])
    with pytest.raises(CollectionBudgetExceeded, match="gameplay"):
        proxy.chat(messages=[])

    assert len(backend.calls) == 1
    assert audit.snapshot()["total_call_count"] == 1


def test_wall_budget_is_checked_before_dispatch():
    ticks = iter((0.0, 2.0, 2.0))
    audit = make_audit(max_wall_seconds=1.0, clock=lambda: next(ticks))
    backend = FakeBackend()

    with pytest.raises(CollectionBudgetExceeded, match="wall-clock"):
        AuditedBackend(backend, audit).chat(messages=[])
    assert backend.calls == []


def test_backend_transient_error_uses_three_counted_attempts():
    audit = make_audit(
        max_gameplay_calls=3,
        max_total_calls=3,
    )
    backend = TransientBackend(failures=2)

    assert AuditedBackend(backend, audit).chat(messages=[]) == "ok"

    snapshot = audit.snapshot()
    assert len(backend.calls) == 3
    assert snapshot["total_call_count"] == 3
    assert snapshot["backend_retry_count"] == 2
    assert [
        event["failed_attempt"]
        for event in snapshot["backend_retry_events"]
    ] == [1, 2]


def test_backend_transient_error_raises_after_third_attempt():
    audit = make_audit(
        max_gameplay_calls=3,
        max_total_calls=3,
    )
    backend = TransientBackend(failures=3)

    with pytest.raises(BackendError, match="transient-3"):
        AuditedBackend(backend, audit).chat(messages=[])

    assert len(backend.calls) == 3
    assert audit.snapshot()["backend_retry_count"] == 2
