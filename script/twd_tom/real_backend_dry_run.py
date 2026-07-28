"""Run exactly two privacy-safe, request-bounded ToM dry-run games.

This thin harness reuses the production runtime and playing-agent sample
collector.  It adds only request-boundary auditing and never enables the
agents' prompt/response/role log files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from run_random import (
    build_runtime,
    build_twd_tom_sample_collector,
    eval as run_game,
)
from werewolf.backends import load_named_backends
from werewolf.models.twd_tom.public_events import (
    public_speech_actions,
    structured_input_digest,
)
from werewolf.models.twd_tom.schema import normalize_player
from werewolf.runtime_config import normalize_runtime_config


PROVIDER_CACHE_NOTE = "PROVIDER_INTERNAL_CACHE_NOT_OBSERVABLE"
_FORBIDDEN_SESSION_KEYS = {
    "conversation",
    "conversation_id",
    "previous_response_id",
    "session",
    "session_id",
    "thread",
    "thread_id",
}
_AUDIT_FIELDS = {
    "game_id",
    "seed",
    "call_index",
    "call_category",
    "observer_id",
    "acting_player_id",
    "backend_id",
    "base_url",
    "endpoint",
    "model_id",
    "start_monotonic_ns",
    "end_monotonic_ns",
    "latency_ms",
    "request_message_count",
    "request_character_count",
    "request_sha256",
    "response_character_count",
    "response_sha256",
    "dispatch_status",
    "error_type",
    "usage_available",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "public_action_count",
    "public_history_digest",
    "state_hash_before",
    "state_hash_after",
    "report_nonce_sha256",
    "report_nonce_absent_from_next_action",
    "report_response_absent_from_next_action",
    "next_action_check_not_reached",
}


class DryRunSafetyViolation(BaseException):
    """Hard-stop signal that broad production error recovery must not swallow."""


class DryRunBudgetExceeded(DryRunSafetyViolation):
    """Raised before a dispatch that would exceed a configured budget."""

    def __init__(
        self,
        *,
        budget_name: str,
        current: int | float,
        limit: int | float,
        game_id: str,
        call_category: str,
    ) -> None:
        super().__init__(
            f"{budget_name} exceeded: current={current}, limit={limit}, "
            f"game_id={game_id}, call_category={call_category}"
        )
        self.budget_name = budget_name
        self.current = current
        self.limit = limit
        self.game_id = game_id
        self.call_category = call_category


class DryRunBudget:
    """Atomically check and increment all per-game dispatch budgets."""

    def __init__(
        self,
        *,
        game_id: str,
        max_gameplay_calls: int,
        max_belief_calls: int,
        max_total_calls: int,
        max_wall_seconds: float,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        global_budget=None,
    ) -> None:
        for name, value in {
            "max_gameplay_calls": max_gameplay_calls,
            "max_belief_calls": max_belief_calls,
            "max_total_calls": max_total_calls,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(max_wall_seconds, bool)
            or not isinstance(max_wall_seconds, (int, float))
            or max_wall_seconds <= 0
        ):
            raise ValueError("max_wall_seconds must be positive")
        self.game_id = game_id
        self.max_gameplay_calls = max_gameplay_calls
        self.max_belief_calls = max_belief_calls
        self.max_total_calls = max_total_calls
        self.max_wall_seconds = float(max_wall_seconds)
        self.gameplay_calls = 0
        self.belief_calls = 0
        self.total_calls = 0
        self._clock_ns = clock_ns
        self._started_ns = clock_ns()
        self._last_category = "gameplay"
        self._lock = threading.Lock()
        self.global_budget = global_budget

    def _raise(self, name, current, limit, category):
        raise DryRunBudgetExceeded(
            budget_name=name,
            current=current,
            limit=limit,
            game_id=self.game_id,
            call_category=category,
        )

    def _check_wall_locked(self, category: str) -> None:
        elapsed = (self._clock_ns() - self._started_ns) / 1_000_000_000
        if elapsed > self.max_wall_seconds:
            self._raise(
                "elapsed_wall_time",
                elapsed,
                self.max_wall_seconds,
                category,
            )

    def before_dispatch(self, category: str) -> int:
        if category not in {"gameplay", "belief"}:
            raise ValueError("call category must be gameplay or belief")
        with self._lock:
            self._last_category = category
            self._check_wall_locked(category)
            category_count = (
                self.gameplay_calls if category == "gameplay" else self.belief_calls
            )
            category_limit = (
                self.max_gameplay_calls
                if category == "gameplay"
                else self.max_belief_calls
            )
            if category_count + 1 > category_limit:
                self._raise(
                    f"{category}_calls",
                    category_count + 1,
                    category_limit,
                    category,
                )
            if self.total_calls + 1 > self.max_total_calls:
                self._raise(
                    "total_calls",
                    self.total_calls + 1,
                    self.max_total_calls,
                    category,
                )
            if self.global_budget is not None:
                self.global_budget.before_dispatch(
                    game_id=self.game_id,
                    category=category,
                )
            if category == "gameplay":
                self.gameplay_calls += 1
            else:
                self.belief_calls += 1
            self.total_calls += 1
            return self.total_calls

    def check_wall_time(self, category: str | None = None) -> None:
        with self._lock:
            checked_category = category or self._last_category
            self._check_wall_locked(checked_category)
            if self.global_budget is not None:
                self.global_budget.check_wall_time(
                    game_id=self.game_id,
                    category=checked_category,
                )

    def wall_timeout_error(self) -> DryRunBudgetExceeded:
        elapsed = (self._clock_ns() - self._started_ns) / 1_000_000_000
        return DryRunBudgetExceeded(
            budget_name="elapsed_wall_time",
            current=elapsed,
            limit=self.max_wall_seconds,
            game_id=self.game_id,
            call_category=self._last_category,
        )

    def summary(self) -> dict[str, int]:
        return {
            "gameplay_calls": self.gameplay_calls,
            "belief_calls": self.belief_calls,
            "total_calls": self.total_calls,
        }


class PrivacySafeAuditWriter:
    """Append whitelisted metadata-only call records."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, record: Mapping[str, Any]) -> None:
        unexpected = set(record) - _AUDIT_FIELDS
        if unexpected:
            raise ValueError(f"unexpected audit fields: {sorted(unexpected)}")
        line = json.dumps(dict(record), ensure_ascii=False, sort_keys=True)
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_request(request: Mapping[str, Any]) -> str:
    return json.dumps(
        request,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=repr,
    )


def _request_text_values(value: Any) -> str:
    """Flatten request string values in memory for contamination checks."""

    values: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, str):
            values.append(item)
        elif isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            for child in item:
                visit(child)

    visit(value)
    return "\n".join(values)


def _reject_remote_session_state(value: Any) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).strip().lower()
            if key in _FORBIDDEN_SESSION_KEYS:
                raise DryRunSafetyViolation(
                    f"remote session field is forbidden: {key}"
                )
            if key == "store" and item is True:
                raise DryRunSafetyViolation("store=true is forbidden")
            _reject_remote_session_state(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _reject_remote_session_state(item)


def backend_semantic_state_fingerprint(backend) -> str:
    """Hash only non-secret fields that can change request semantics."""

    state: dict[str, Any] = {
        "class": f"{type(backend).__module__}.{type(backend).__qualname__}",
    }
    for field in (
        "default_model",
        "base_url",
        "model",
        "model_name",
        "api_version",
        "organization",
        "project",
    ):
        value = getattr(backend, field, None)
        if value is None or isinstance(value, (str, int, float, bool)):
            state[field] = value
    return _sha256_text(_canonical_request(state))


class DryRunAuditSession:
    """Coordinate call metadata, report nonces, and contamination checks."""

    def __init__(
        self,
        *,
        game_id: str,
        seed: int,
        budget: DryRunBudget,
        writer: PrivacySafeAuditWriter,
    ) -> None:
        self.game_id = game_id
        self.seed = seed
        self.budget = budget
        self.writer = writer
        self._context: ContextVar[dict[str, Any] | None] = ContextVar(
            f"dry_run_call_context_{game_id}", default=None
        )
        self._pending: dict[str, dict[str, Any]] = {}

    @contextmanager
    def gameplay_context(
        self,
        *,
        acting_player_id: int,
        observation: Mapping,
        public_events: Sequence[Mapping[str, Any]],
    ):
        actions = public_speech_actions(public_events)
        token = self._context.set(
            {
                "call_category": "gameplay",
                "acting_player_id": normalize_player(acting_player_id),
                "observer_id": None,
                "public_action_count": len(actions),
                "public_history_digest": structured_input_digest(public_events),
                "report_id": None,
            }
        )
        try:
            yield
        finally:
            self._context.reset(token)

    def prepare_report(
        self,
        *,
        observer_id: str,
        public_snapshot,
        agent_backend_id: str,
        agent_model_id: str,
        report_prompt: str,
    ) -> tuple[str, str]:
        observer = normalize_player(observer_id)
        if not isinstance(agent_model_id, str) or not agent_model_id.strip():
            raise ValueError("belief report requires the playing agent model ID")
        report_id = secrets.token_hex(16)
        nonce = "TWD_TOM_REPORT_AUDIT_" + secrets.token_urlsafe(24)
        self._pending[report_id] = {
            "observer_id": observer,
            "expected_backend_id": agent_backend_id,
            "expected_model_id": agent_model_id,
            "nonce": nonce,
            "raw_response": None,
            "records": [],
            "public_action_count": public_snapshot.public_action_count,
            "public_history_digest": public_snapshot.public_history_digest,
            "state_hash_before": None,
            "state_hash_after": None,
        }
        decorated = report_prompt + f"\nreport_audit_nonce: {nonce}"
        return report_id, decorated

    @contextmanager
    def belief_context(self, report_id: str):
        pending = self._pending[report_id]
        token = self._context.set(
            {
                "call_category": "belief",
                "acting_player_id": None,
                "observer_id": pending["observer_id"],
                "public_action_count": pending["public_action_count"],
                "public_history_digest": pending["public_history_digest"],
                "report_id": report_id,
                "expected_backend_id": pending["expected_backend_id"],
                "expected_model_id": pending["expected_model_id"],
            }
        )
        try:
            yield
        finally:
            self._context.reset(token)

    def complete_report(self, report_id: str, raw_response: Any) -> None:
        pending = self._pending[report_id]
        pending["raw_response"] = raw_response if isinstance(raw_response, str) else None

    def record_agent_state(
        self,
        *,
        observer_id: str,
        state_before,
        state_after,
    ) -> None:
        observer = normalize_player(observer_id)
        for pending in reversed(list(self._pending.values())):
            if pending["observer_id"] != observer:
                continue
            if pending["state_hash_before"] is not None:
                continue
            pending["state_hash_before"] = _sha256_text(repr(state_before))
            pending["state_hash_after"] = _sha256_text(repr(state_after))
            for record in pending["records"]:
                record["state_hash_before"] = pending["state_hash_before"]
                record["state_hash_after"] = pending["state_hash_after"]
            return

    def current_context(self) -> dict[str, Any]:
        context = self._context.get()
        if context is None:
            return {
                "call_category": "gameplay",
                "acting_player_id": None,
                "observer_id": None,
                "public_action_count": None,
                "public_history_digest": None,
                "report_id": None,
            }
        return dict(context)

    def inspect_request(
        self,
        *,
        backend_id: str,
        model_id: str | None,
        serialized_request: str,
        request_text: str,
    ) -> dict[str, Any]:
        context = self.current_context()
        if context["call_category"] == "belief":
            if backend_id != context["expected_backend_id"]:
                raise DryRunSafetyViolation(
                    "belief report backend differs from playing agent"
                )
            if model_id != context["expected_model_id"]:
                raise DryRunSafetyViolation(
                    "belief report model differs from playing agent"
                )
        acting_player = context.get("acting_player_id")
        if context["call_category"] == "gameplay" and acting_player is not None:
            self._check_next_action(
                acting_player,
                serialized_request + "\n" + request_text,
            )
        return context

    def record_call(self, record: dict[str, Any], context: Mapping[str, Any]) -> None:
        report_id = context.get("report_id")
        if context["call_category"] == "belief" and report_id is not None:
            pending = self._pending[report_id]
            record["report_nonce_sha256"] = _sha256_text(pending["nonce"])
            record["state_hash_before"] = pending["state_hash_before"]
            record["state_hash_after"] = pending["state_hash_after"]
            pending["records"].append(record)
            return
        self.writer.write(record)

    def _flush_pending(
        self,
        report_id: str,
        *,
        nonce_absent: bool | None,
        response_absent: bool | None,
        not_reached: bool,
    ) -> None:
        pending = self._pending.pop(report_id)
        for record in pending["records"]:
            record["report_nonce_absent_from_next_action"] = nonce_absent
            record["report_response_absent_from_next_action"] = response_absent
            record["next_action_check_not_reached"] = not_reached
            self.writer.write(record)
        pending["nonce"] = None
        pending["raw_response"] = None

    def _check_next_action(self, acting_player: str, serialized_request: str) -> None:
        matches = [
            report_id
            for report_id, pending in self._pending.items()
            if pending["observer_id"] == acting_player
        ]
        contaminated = False
        for report_id in matches:
            pending = self._pending[report_id]
            nonce_absent = pending["nonce"] not in serialized_request
            response = pending["raw_response"]
            response_absent = not response or response not in serialized_request
            contaminated = contaminated or not nonce_absent or not response_absent
            self._flush_pending(
                report_id,
                nonce_absent=nonce_absent,
                response_absent=response_absent,
                not_reached=False,
            )
        if contaminated:
            raise DryRunSafetyViolation(
                f"belief report contamination detected for {acting_player}"
            )

    def finish_game(self) -> None:
        for report_id in list(self._pending):
            self._flush_pending(
                report_id,
                nonce_absent=None,
                response_absent=None,
                not_reached=True,
            )


class AuditedBackend:
    """Enforce stateless requests, hard budgets, timing, and safe audit output."""

    def __init__(
        self,
        *,
        backend,
        backend_id: str,
        session: DryRunAuditSession,
    ) -> None:
        self.backend = backend
        self.backend_id = backend_id
        self.session = session
        self.base_url = getattr(backend, "base_url", None)
        self.endpoint = getattr(
            backend,
            "chat_completions_endpoint",
            None,
        )
        self.default_model = getattr(backend, "default_model", None)

    def chat(
        self,
        messages,
        model=None,
        temperature=0.7,
        max_tokens=None,
        response_format=None,
        **kwargs,
    ) -> str:
        if (
            not isinstance(messages, Sequence)
            or isinstance(messages, (str, bytes))
            or not messages
            or any(
                not isinstance(message, Mapping)
                or "role" not in message
                or "content" not in message
                for message in messages
            )
        ):
            raise TypeError("every backend request must explicitly provide messages")
        request = dict(kwargs)
        request.update({"model": model, "messages": messages})
        if temperature is not None:
            request["temperature"] = temperature
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if response_format is not None:
            request["response_format"] = response_format
        _reject_remote_session_state(request)
        serialized = _canonical_request(request)
        context = self.session.inspect_request(
            backend_id=self.backend_id,
            model_id=model,
            serialized_request=serialized,
            request_text=_request_text_values(request),
        )

        call_index = self.session.budget.before_dispatch(
            context["call_category"]
        )
        state_before = backend_semantic_state_fingerprint(self.backend)
        start_ns = time.monotonic_ns()
        response_text = ""
        usage = None
        raised = None
        try:
            if hasattr(self.backend, "chat_with_metadata"):
                response_text, usage = self.backend.chat_with_metadata(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    **kwargs,
                )
            else:
                response_text = self.backend.chat(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    **kwargs,
                )
            if not isinstance(response_text, str):
                raise TypeError("backend response must be text")
        except Exception as exc:
            raised = exc
        end_ns = time.monotonic_ns()
        state_after = backend_semantic_state_fingerprint(self.backend)

        record = {
            "game_id": self.session.game_id,
            "seed": self.session.seed,
            "call_index": call_index,
            "call_category": context["call_category"],
            "observer_id": context.get("observer_id"),
            "acting_player_id": context.get("acting_player_id"),
            "backend_id": self.backend_id,
            "base_url": self.base_url,
            "endpoint": self.endpoint,
            "model_id": model,
            "start_monotonic_ns": start_ns,
            "end_monotonic_ns": end_ns,
            "latency_ms": (end_ns - start_ns) / 1_000_000,
            "request_message_count": len(messages),
            "request_character_count": len(serialized),
            "request_sha256": _sha256_text(serialized),
            "response_character_count": len(response_text),
            "response_sha256": _sha256_text(response_text),
            "dispatch_status": "ok" if raised is None else "error",
            "error_type": (
                None
                if raised is None
                else type(raised.__cause__ or raised).__name__
            ),
            "usage_available": usage is not None,
            "input_tokens": None if usage is None else usage.get("input_tokens"),
            "output_tokens": None if usage is None else usage.get("output_tokens"),
            "total_tokens": None if usage is None else usage.get("total_tokens"),
            "public_action_count": context.get("public_action_count"),
            "public_history_digest": context.get("public_history_digest"),
            "state_hash_before": None,
            "state_hash_after": None,
            "report_nonce_sha256": None,
            "report_nonce_absent_from_next_action": None,
            "report_response_absent_from_next_action": None,
            "next_action_check_not_reached": None,
        }
        self.session.record_call(record, context)

        if state_after != state_before:
            raise DryRunSafetyViolation(
                "backend semantic state changed during request"
            )
        self.session.budget.check_wall_time(context["call_category"])
        if raised is not None:
            raise raised
        return response_text


@dataclass(frozen=True)
class RealBackendDryRunConfig:
    runtime_config_path: str
    output_dir: str
    game_count: int
    seeds: tuple[int, int]
    max_gameplay_calls_per_game: int
    max_belief_calls_per_game: int
    max_total_calls_per_game: int
    max_wall_seconds_per_game: float
    privacy_safe_logging: bool
    audit_only_metadata: bool

    def __post_init__(self) -> None:
        if self.game_count != 2:
            raise ValueError("game_count must be exactly 2")
        if len(self.seeds) != 2 or self.seeds[0] == self.seeds[1]:
            raise ValueError("seeds must contain exactly two distinct values")
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in self.seeds):
            raise TypeError("seeds must be integers")
        if self.privacy_safe_logging is not True:
            raise ValueError("privacy_safe_logging must be enabled")
        if self.audit_only_metadata is not True:
            raise ValueError("audit_only_metadata must be enabled")
        DryRunBudget(
            game_id="validation",
            max_gameplay_calls=self.max_gameplay_calls_per_game,
            max_belief_calls=self.max_belief_calls_per_game,
            max_total_calls=self.max_total_calls_per_game,
            max_wall_seconds=self.max_wall_seconds_per_game,
        )


def validate_output_dir(
    raw_path: str,
    *,
    cwd: Path | None = None,
    required_name_token: str = "dry_run",
) -> Path:
    root = ((cwd or Path.cwd()) / "logs").resolve()
    output = Path(raw_path).resolve()
    try:
        relative = output.relative_to(root)
    except ValueError as exc:
        raise ValueError("output directory must be inside logs/") from exc
    lowered_name = output.name.lower()
    if required_name_token not in lowered_name:
        raise ValueError(
            f"output directory name must contain {required_name_token}"
        )
    forbidden_components = {"data", "checkpoint", "checkpoints"}
    reserved_filenames = {
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
    }
    for component in relative.parts:
        lowered_component = component.lower()
        if lowered_component in forbidden_components:
            raise ValueError(
                f"output path cannot contain component {component}"
            )
        if lowered_component in reserved_filenames:
            raise ValueError(
                f"output path cannot use reserved filename {component}"
            )
    return output


def _atomic_json_dump(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@contextmanager
def _hard_wall_timeout(budget: DryRunBudget):
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("hard wall timeout requires the main thread")
    previous_handler = signal.getsignal(signal.SIGALRM)

    def raise_timeout(_signum, _frame):
        raise budget.wall_timeout_error()

    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, budget.max_wall_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _run_exactly_two(seeds: Sequence[int], run_one: Callable[[int, int], Any]):
    if len(seeds) != 2 or seeds[0] == seeds[1]:
        raise ValueError("exactly two distinct seeds are required")
    results = []
    for game_index, seed in enumerate(seeds, start=1):
        results.append(run_one(game_index, seed))
    return results


def _runtime_source_commit() -> str:
    """Return the exact checked-out commit or fail without a fallback."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot resolve the dry-run source commit") from exc

    source_commit = result.stdout.strip()
    if re.fullmatch(r"[0-9a-fA-F]{40}", source_commit) is None:
        raise RuntimeError("dry-run source commit must be a 40-character SHA")
    return source_commit


def run_real_backend_game(
    *,
    parsed_yaml: Mapping[str, Any],
    samples_path: Path,
    log_dir: Path | None,
    game_id: str,
    seed: int,
    budget: DryRunBudget,
    writer: PrivacySafeAuditWriter,
    collector_wrapper: Callable[[Any], Any] | None = None,
) -> None:
    """Run one game through the shared audited runtime and collector."""

    normalized = normalize_runtime_config(deepcopy(parsed_yaml))
    raw_backends = load_named_backends(
        normalized,
        env_file=".env",
        max_retries=0,
    )
    session = DryRunAuditSession(
        game_id=game_id,
        seed=seed,
        budget=budget,
        writer=writer,
    )
    audited_backends = {
        backend_id: AuditedBackend(
            backend=backend,
            backend_id=backend_id,
            session=session,
        )
        for backend_id, backend in raw_backends.items()
    }
    env, agents, roles, _profiles = build_runtime(
        parsed_yaml,
        log_save_path=log_dir,
        random_seed=seed,
        backends=audited_backends,
    )
    collector = build_twd_tom_sample_collector(
        agent_list=agents,
        output_path=str(samples_path),
        game_id=game_id,
        report_audit=session,
    )
    if collector_wrapper is not None:
        collector = collector_wrapper(collector)
    try:
        with _hard_wall_timeout(budget):
            run_game(
                env,
                agents,
                roles,
                sample_collector=collector,
                call_audit=session,
            )
    finally:
        session.finish_game()
        collector.close()


def run_dry_run(config: RealBackendDryRunConfig) -> dict[str, Any]:
    """Execute the explicitly approved two-game harness."""

    source_commit = _runtime_source_commit()
    output_dir = validate_output_dir(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    samples_path = output_dir / "dry_run_samples.jsonl"
    audit_path = output_dir / "dry_run_call_audit.jsonl"
    manifest_path = output_dir / "dry_run_manifest.json"
    summary_path = output_dir / "dry_run_summary.json"
    runtime_path = Path(config.runtime_config_path).resolve()
    if not runtime_path.is_file():
        raise FileNotFoundError(f"runtime config not found: {runtime_path}")
    parsed_yaml = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
    if not isinstance(parsed_yaml, dict):
        raise ValueError("runtime config must be a mapping")

    manifest = {
        "dry_run_only": True,
        "formal_training_data": False,
        "source_commit": source_commit,
        "requested_game_count": 2,
        "seeds": list(config.seeds),
        "configured_budgets": {
            "max_gameplay_calls_per_game": config.max_gameplay_calls_per_game,
            "max_belief_calls_per_game": config.max_belief_calls_per_game,
            "max_total_calls_per_game": config.max_total_calls_per_game,
            "max_wall_seconds_per_game": config.max_wall_seconds_per_game,
        },
        "privacy_safe_logging": True,
        "audit_only_metadata": True,
        "raw_prompts_saved": False,
        "raw_responses_saved": False,
        "roles_saved": False,
        "private_observations_saved": False,
        "provider_internal_cache": PROVIDER_CACHE_NOTE,
        "exact_cost": "unavailable",
    }
    _atomic_json_dump(manifest, manifest_path)

    game_summaries = []
    with PrivacySafeAuditWriter(audit_path) as writer:

        def run_one(game_index: int, seed: int):
            game_id = f"dry_run_game_{game_index:03d}_seed_{seed}"
            budget = DryRunBudget(
                game_id=game_id,
                max_gameplay_calls=config.max_gameplay_calls_per_game,
                max_belief_calls=config.max_belief_calls_per_game,
                max_total_calls=config.max_total_calls_per_game,
                max_wall_seconds=config.max_wall_seconds_per_game,
            )
            run_real_backend_game(
                parsed_yaml=parsed_yaml,
                samples_path=samples_path,
                log_dir=None,
                game_id=game_id,
                seed=seed,
                budget=budget,
                writer=writer,
            )
            result = {"game_id": game_id, "seed": seed, "calls": budget.summary()}
            game_summaries.append(result)
            return result

        _run_exactly_two(config.seeds, run_one)

    summary = {
        "status": "ok",
        "dry_run_only": True,
        "game_count": len(game_summaries),
        "games": game_summaries,
        "samples_path": str(samples_path),
        "call_audit_path": str(audit_path),
        "manifest_path": str(manifest_path),
        "provider_internal_cache": PROVIDER_CACHE_NOTE,
        "exact_cost": "unavailable",
    }
    _atomic_json_dump(summary, summary_path)
    summary["summary_path"] = str(summary_path)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--game-count", required=True, type=int)
    parser.add_argument("--seeds", required=True, type=int, nargs=2)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-gameplay-calls-per-game", required=True, type=int)
    parser.add_argument("--max-belief-calls-per-game", required=True, type=int)
    parser.add_argument("--max-total-calls-per-game", required=True, type=int)
    parser.add_argument("--max-wall-seconds-per-game", required=True, type=float)
    parser.add_argument("--privacy-safe-logging", required=True, action="store_true")
    parser.add_argument("--audit-only-metadata", required=True, action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    config = RealBackendDryRunConfig(
        runtime_config_path=args.config,
        output_dir=args.output_dir,
        game_count=args.game_count,
        seeds=tuple(args.seeds),
        max_gameplay_calls_per_game=args.max_gameplay_calls_per_game,
        max_belief_calls_per_game=args.max_belief_calls_per_game,
        max_total_calls_per_game=args.max_total_calls_per_game,
        max_wall_seconds_per_game=args.max_wall_seconds_per_game,
        privacy_safe_logging=args.privacy_safe_logging,
        audit_only_metadata=args.audit_only_metadata,
    )
    return run_dry_run(config)


if __name__ == "__main__":
    main()
