"""Run monitored, privacy-safe classic-seven ToM collection batches."""

from __future__ import annotations

import json
import math
import statistics
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from run_random import (
    PRIVATE_CONDITIONED_COLLECTION_MODE,
    PUBLIC_ONLY_COLLECTION_MODE,
)
from script.twd_tom import real_backend_dry_run as harness
from werewolf.models.twd_tom.samples import (
    PUBLIC_ONLY_SAMPLE_SCHEMA_VERSION,
    SAMPLE_FIELDS,
    SAMPLE_SCHEMA_VERSION,
)
from werewolf.models.twd_tom.public_events import (
    public_event_digest,
    public_speech_actions,
    structured_input_digest,
)
from werewolf.models.twd_tom.schema import (
    LABEL_PROMPT_VERSION,
    PUBLIC_ONLY_LABEL_PROMPT_VERSION,
    normalize_player,
    validate_player_suspicion,
)


_OLD_SCHEMA_FIELDS = {
    "plausible_wolf_pairs",
    "role_guesses",
    "guess_status",
    "guess_errors",
    "guesses_to_belief_targets",
    "belief_mode",
    "believed_werewolves",
    "pair_support",
    "pair_target",
    "pair_targets",
}
_PRIVATE_FIELDS = {
    "actual_roles",
    "true_roles",
    "roles",
    "role",
    "private_observation",
    "observation",
    "messages",
    "memory",
    "system_prompt",
    "raw_response",
    "reasoning",
    "chain_of_thought",
    "god_view",
    "actual_wolves",
}
_OTHER_PLAYER_PRIVATE_FIELDS = {
    "teammate",
    "seer_result",
    "witch_action",
    "other_players_private_information",
}
_VALID_STATUSES = {"ok", "parse_error", "semantic_error", "reporter_error"}
_SAFETY_CHECKS = (
    "state_mutation",
    "contamination",
    "digest_mismatch",
    "private_serialization",
    "other_player_private_leakage",
    "label_integrity",
    "old_schema",
)


class CollectionQualityStop(BaseException):
    """Stop the active game without being swallowed by game recovery."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class CollectionSampleSafetyViolation(harness.DryRunSafetyViolation):
    """Identify a failed sample safety invariant without private values."""

    def __init__(self, check_name: str, message: str) -> None:
        super().__init__(message)
        self.check_name = check_name


class CollectionGlobalBudget:
    """Atomically enforce collection-wide call and wall-time limits."""

    def __init__(
        self,
        *,
        max_total_calls: int,
        max_wall_seconds: float,
        clock_ns=time.monotonic_ns,
    ) -> None:
        if (
            isinstance(max_total_calls, bool)
            or not isinstance(max_total_calls, int)
            or max_total_calls <= 0
        ):
            raise ValueError("max_total_calls must be a positive integer")
        if (
            isinstance(max_wall_seconds, bool)
            or not isinstance(max_wall_seconds, (int, float))
            or max_wall_seconds <= 0
        ):
            raise ValueError("max_wall_seconds must be positive")
        self.max_total_calls = max_total_calls
        self.max_wall_seconds = float(max_wall_seconds)
        self.total_calls = 0
        self._clock_ns = clock_ns
        self._started_ns = clock_ns()
        self._lock = threading.Lock()

    def _check_wall_locked(self, *, game_id: str, category: str) -> None:
        elapsed = (self._clock_ns() - self._started_ns) / 1_000_000_000
        if elapsed > self.max_wall_seconds:
            raise harness.DryRunBudgetExceeded(
                budget_name="global_elapsed_wall_time",
                current=elapsed,
                limit=self.max_wall_seconds,
                game_id=game_id,
                call_category=category,
            )

    def before_dispatch(self, *, game_id: str, category: str) -> int:
        with self._lock:
            self._check_wall_locked(game_id=game_id, category=category)
            if self.total_calls + 1 > self.max_total_calls:
                raise harness.DryRunBudgetExceeded(
                    budget_name="global_total_calls",
                    current=self.total_calls + 1,
                    limit=self.max_total_calls,
                    game_id=game_id,
                    call_category=category,
                )
            self.total_calls += 1
            return self.total_calls

    def check_wall_time(self, *, game_id: str, category: str) -> None:
        with self._lock:
            self._check_wall_locked(game_id=game_id, category=category)

    def summary(self) -> dict[str, int | float]:
        return {
            "total_calls": self.total_calls,
            "max_total_calls": self.max_total_calls,
            "max_wall_seconds": self.max_wall_seconds,
        }


def _error_rule(status: str) -> str:
    return status


class CollectionQualityMonitor:
    """Aggregate validator outcomes and raise only frozen stop conditions."""

    def __init__(
        self,
        *,
        collection_mode: str = PRIVATE_CONDITIONED_COLLECTION_MODE,
    ) -> None:
        self.collection_mode = collection_mode
        self.total_reports = 0
        self.status_counts: Counter[str] = Counter()
        self.error_rule_counts: Counter[str] = Counter()
        self.by_observer: dict[str, Counter[str]] = defaultdict(Counter)
        self.by_game_phase: dict[str, dict[str, Counter[str]]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        self._streaks: dict[str, tuple[str, int]] = {}
        self.stop_reason: str | None = None

    def observe_report(
        self,
        *,
        game_id: str,
        phase: str,
        observer: str,
        status: str,
        error: Any,
    ) -> None:
        if status not in _VALID_STATUSES:
            raise CollectionSampleSafetyViolation(
                "old_schema", "sample contains an unsupported belief status"
            )
        observer = normalize_player(observer)
        self.total_reports += 1
        self.status_counts[status] += 1
        self.by_observer[observer]["total"] += 1
        self.by_observer[observer][status] += 1
        self.by_game_phase[game_id][phase]["total"] += 1
        self.by_game_phase[game_id][phase][status] += 1

        if status == "ok":
            self._streaks.pop(observer, None)
        else:
            rule = _error_rule(status)
            self.error_rule_counts[rule] += 1
            self.by_observer[observer]["invalid"] += 1
            previous_rule, previous_count = self._streaks.get(
                observer, ("", 0)
            )
            streak_count = previous_count + 1 if previous_rule == rule else 1
            self._streaks[observer] = (rule, streak_count)
            if streak_count >= 3 and self.stop_reason is None:
                self.stop_reason = (
                    "observer_consecutive_same_error_rule_gt_or_eq_3"
                )

        observer_counts = self.by_observer[observer]
        if (
            observer_counts["total"] >= 10
            and observer_counts["invalid"] / observer_counts["total"] > 0.20
            and self.stop_reason is None
        ):
            self.stop_reason = "observer_invalid_rate_gt_20_percent"

        if self.total_reports >= 100 and self.stop_reason is None:
            invalid = self.total_reports - self.status_counts["ok"]
            if invalid / self.total_reports > 0.05:
                self.stop_reason = "total_invalid_rate_gt_5_percent"
            elif self.status_counts["parse_error"] / self.total_reports > 0.01:
                self.stop_reason = "parse_error_rate_gt_1_percent"

    def observe_sample(self, sample: Mapping[str, Any]) -> None:
        _validate_sample_safety(
            sample,
            collection_mode=self.collection_mode,
        )
        game_id = sample["game_id"]
        phase = sample["phase"]
        for observer in sorted(sample["belief_status"]):
            self.observe_report(
                game_id=game_id,
                phase=phase,
                observer=observer,
                status=sample["belief_status"][observer],
                error=sample["belief_errors"][observer],
            )

    def summary(self) -> dict[str, Any]:
        invalid = self.total_reports - self.status_counts["ok"]
        return {
            "total_reports": self.total_reports,
            "status_counts": {
                status: self.status_counts[status]
                for status in (
                    "ok",
                    "parse_error",
                    "semantic_error",
                    "reporter_error",
                )
            },
            "invalid_rate": 0.0 if not self.total_reports else invalid / self.total_reports,
            "parse_error_rate": (
                0.0
                if not self.total_reports
                else self.status_counts["parse_error"] / self.total_reports
            ),
            "error_rule_counts": dict(sorted(self.error_rule_counts.items())),
            "by_observer": {
                observer: dict(sorted(counts.items()))
                for observer, counts in sorted(self.by_observer.items())
            },
            "by_game_phase": {
                game_id: {
                    phase: dict(sorted(counts.items()))
                    for phase, counts in sorted(phases.items())
                }
                for game_id, phases in sorted(self.by_game_phase.items())
            },
            "stop_reason": self.stop_reason,
        }


def _walk_field_names(value: Any):
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield str(key).lower()
            yield from _walk_field_names(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_field_names(nested)


def _validate_sample_safety(
    sample: Mapping[str, Any],
    *,
    collection_mode: str = PRIVATE_CONDITIONED_COLLECTION_MODE,
) -> None:
    if not isinstance(sample, Mapping):
        raise CollectionSampleSafetyViolation(
            "old_schema", "collector returned a non-mapping sample"
        )
    fields = set(sample)
    expected_schema = (
        PUBLIC_ONLY_SAMPLE_SCHEMA_VERSION
        if collection_mode == PUBLIC_ONLY_COLLECTION_MODE
        else SAMPLE_SCHEMA_VERSION
    )
    expected_prompt = (
        PUBLIC_ONLY_LABEL_PROMPT_VERSION
        if collection_mode == PUBLIC_ONLY_COLLECTION_MODE
        else LABEL_PROMPT_VERSION
    )
    if (
        fields != SAMPLE_FIELDS
        or fields & _OLD_SCHEMA_FIELDS
        or sample.get("schema_version") != expected_schema
        or sample.get("label_prompt_version") != expected_prompt
    ):
        raise CollectionSampleSafetyViolation(
            "old_schema", "sample contains an unsupported schema"
        )
    nested_fields = set(_walk_field_names(sample))
    if nested_fields & _PRIVATE_FIELDS:
        raise CollectionSampleSafetyViolation(
            "private_serialization", "sample contains a private field"
        )
    if nested_fields & _OTHER_PLAYER_PRIVATE_FIELDS:
        raise CollectionSampleSafetyViolation(
            "other_player_private_leakage",
            "sample contains another-player private field",
        )
    if (
        public_event_digest(sample["public_events"])
        != sample["public_event_digest"]
        or structured_input_digest(sample["public_events"])
        != sample["structured_input_digest"]
        or len(public_speech_actions(sample["public_events"]))
        != sample["public_action_count"]
    ):
        raise CollectionSampleSafetyViolation(
            "digest_mismatch", "sample public history metadata is inconsistent"
        )
    expected = {
        normalize_player(player_id) for player_id in sample["observer_ids"]
    }
    for field in (
        "suspected_werewolves",
        "known_werewolves",
        "known_non_werewolves",
        "belief_status",
        "belief_errors",
        "agent_backend_ids",
    ):
        if set(sample[field]) != expected:
            raise CollectionSampleSafetyViolation(
                "old_schema", "sample observer mappings are misaligned"
            )
    for observer, status in sample["belief_status"].items():
        suspicion = sample["suspected_werewolves"][observer]
        error = sample["belief_errors"][observer]
        backend_id = sample["agent_backend_ids"][observer]
        if collection_mode == PUBLIC_ONLY_COLLECTION_MODE and (
            sample["known_werewolves"][observer]
            or sample["known_non_werewolves"][observer]
        ):
            raise CollectionSampleSafetyViolation(
                "private_serialization",
                "public-only sample contains hard private knowledge",
            )
        if status not in _VALID_STATUSES:
            raise CollectionSampleSafetyViolation(
                "old_schema", "sample contains an unsupported belief status"
            )
        if not isinstance(backend_id, str) or not backend_id.strip():
            raise CollectionSampleSafetyViolation(
                "label_integrity", "sample contains an invalid backend ID"
            )
        if status == "ok":
            if error is not None:
                raise CollectionSampleSafetyViolation(
                    "label_integrity", "ok label contains an error"
                )
            try:
                validate_player_suspicion(
                    suspicion,
                    sample["known_werewolves"][observer],
                    sample["known_non_werewolves"][observer],
                )
            except (TypeError, ValueError) as exc:
                raise CollectionSampleSafetyViolation(
                    "label_integrity", str(exc)
                ) from exc
        else:
            if suspicion is not None:
                raise CollectionSampleSafetyViolation(
                    "label_integrity", "invalid label has a suspicion set"
                )
            if not isinstance(error, str) or not error:
                raise CollectionSampleSafetyViolation(
                    "label_integrity", "invalid label requires an error"
                )


class _MonitoredCollector:
    def __init__(
        self,
        collector,
        monitor: CollectionQualityMonitor,
        *,
        collection_mode: str = PRIVATE_CONDITIONED_COLLECTION_MODE,
    ) -> None:
        self.collector = collector
        self.monitor = monitor
        self.collection_mode = collection_mode
        self._write = collector.write
        collector.write = self._validated_write

    def _validated_write(self, sample) -> None:
        _validate_sample_safety(
            sample,
            collection_mode=self.collection_mode,
        )
        self._write(sample)

    def record(self, *args, **kwargs):
        sample = self.collector.record(*args, **kwargs)
        self.monitor.observe_sample(sample)
        if self.monitor.stop_reason is not None:
            raise CollectionQualityStop(self.monitor.stop_reason)
        return sample

    def close(self) -> None:
        self.collector.close()


@dataclass(frozen=True)
class MonitoredCollectionConfig:
    runtime_config_path: str
    output_dir: str
    game_count: int
    seeds: tuple[int, ...]
    max_gameplay_calls_per_game: int
    max_belief_calls_per_game: int
    max_total_calls_per_game: int
    max_wall_seconds_per_game: float
    max_total_calls: int
    max_wall_seconds: float
    privacy_safe_logging: bool
    audit_only_metadata: bool

    def __post_init__(self) -> None:
        if self.game_count != 10:
            raise ValueError("game_count must be exactly 10")
        if len(self.seeds) != 10 or len(set(self.seeds)) != 10:
            raise ValueError("seeds must contain exactly ten distinct values")
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in self.seeds):
            raise TypeError("seeds must be integers")
        if self.privacy_safe_logging is not True:
            raise ValueError("privacy_safe_logging must be enabled")
        if self.audit_only_metadata is not True:
            raise ValueError("audit_only_metadata must be enabled")
        harness.DryRunBudget(
            game_id="validation",
            max_gameplay_calls=self.max_gameplay_calls_per_game,
            max_belief_calls=self.max_belief_calls_per_game,
            max_total_calls=self.max_total_calls_per_game,
            max_wall_seconds=self.max_wall_seconds_per_game,
        )
        CollectionGlobalBudget(
            max_total_calls=self.max_total_calls,
            max_wall_seconds=self.max_wall_seconds,
        )


def _safety_check_name(exc: BaseException) -> str:
    message = str(exc)
    if "semantic state changed" in message:
        return "state_mutation"
    if "contamination" in message:
        return "contamination"
    return "private_serialization"


def summarize_call_audit(path: Path) -> dict[str, Any]:
    """Aggregate metadata-only request, token, and latency statistics."""

    request_counts: Counter[str] = Counter()
    dispatch_status_counts: Counter[str] = Counter()
    usage_available_count = 0
    token_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    latencies = []
    with path.open(encoding="utf-8") as audit_file:
        for line in audit_file:
            if not line.strip():
                continue
            record = json.loads(line)
            category = record.get("call_category")
            if category not in {"gameplay", "belief"}:
                raise ValueError("audit record has an invalid call category")
            request_counts[category] += 1
            dispatch_status = record.get("dispatch_status")
            if dispatch_status not in {"ok", "error"}:
                raise ValueError("audit record has an invalid dispatch status")
            dispatch_status_counts[dispatch_status] += 1
            latency = record.get("latency_ms")
            if isinstance(latency, bool) or not isinstance(latency, (int, float)):
                raise TypeError("audit latency must be numeric")
            latencies.append(float(latency))
            if record.get("usage_available") is True:
                usage_available_count += 1
            for field in token_totals:
                value = record.get(field)
                if isinstance(value, int) and not isinstance(value, bool):
                    token_totals[field] += value

    sorted_latencies = sorted(latencies)
    p95_index = max(0, math.ceil(0.95 * len(sorted_latencies)) - 1)
    latency_summary = {
        "mean_ms": None if not latencies else statistics.fmean(latencies),
        "median_ms": None if not latencies else statistics.median(latencies),
        "p95_ms": None if not latencies else sorted_latencies[p95_index],
        "max_ms": None if not latencies else max(latencies),
    }
    return {
        "request_count": sum(request_counts.values()),
        "request_count_by_category": {
            category: request_counts[category]
            for category in ("gameplay", "belief")
        },
        "dispatch_status_counts": {
            status: dispatch_status_counts[status]
            for status in ("ok", "error")
        },
        "usage_available_count": usage_available_count,
        "token_totals": token_totals,
        "latency": latency_summary,
    }


def run_monitored_collection(
    config: MonitoredCollectionConfig,
    *,
    artifact_prefix: str,
    required_name_token: str,
    game_id_prefix: str,
    mode_metadata: Mapping[str, Any],
    collection_mode: str = PRIVATE_CONDITIONED_COLLECTION_MODE,
) -> dict[str, Any]:
    """Run one ten-game collection mode through the frozen monitor."""

    if artifact_prefix != "formal_batch":
        raise ValueError("unsupported collection artifact prefix")

    source_commit = harness._runtime_source_commit()
    output_dir = harness.validate_output_dir(
        config.output_dir,
        required_name_token=required_name_token,
    )
    runtime_path = Path(config.runtime_config_path).resolve()
    if not runtime_path.is_file():
        raise FileNotFoundError(f"runtime config not found: {runtime_path}")
    parsed_yaml = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
    if not isinstance(parsed_yaml, dict):
        raise ValueError("runtime config must be a mapping")

    output_dir.mkdir(parents=True, exist_ok=False)
    samples_path = output_dir / f"{artifact_prefix}_samples.jsonl"
    audit_path = output_dir / f"{artifact_prefix}_call_audit.jsonl"
    manifest_path = output_dir / f"{artifact_prefix}_manifest.json"
    summary_path = output_dir / f"{artifact_prefix}_summary.json"
    samples_path.touch(exist_ok=False)

    manifest = {
        **dict(mode_metadata),
        "formal_training_data": False,
        "source_commit": source_commit,
        "requested_game_count": 10,
        "seeds": list(config.seeds),
        "configured_budgets": {
            "max_gameplay_calls_per_game": config.max_gameplay_calls_per_game,
            "max_belief_calls_per_game": config.max_belief_calls_per_game,
            "max_total_calls_per_game": config.max_total_calls_per_game,
            "max_wall_seconds_per_game": config.max_wall_seconds_per_game,
            "max_total_calls": config.max_total_calls,
            "max_wall_seconds": config.max_wall_seconds,
        },
        "privacy_safe_logging": True,
        "audit_only_metadata": True,
        "raw_prompts_saved": False,
        "raw_responses_saved": False,
        "roles_saved": False,
        "private_observations_saved": False,
        "provider_internal_cache": harness.PROVIDER_CACHE_NOTE,
        "exact_cost": "unavailable",
    }
    harness._atomic_json_dump(manifest, manifest_path)

    global_budget = CollectionGlobalBudget(
        max_total_calls=config.max_total_calls,
        max_wall_seconds=config.max_wall_seconds,
    )
    monitor = CollectionQualityMonitor(
        collection_mode=collection_mode,
    )
    safety_checks = {name: 0 for name in _SAFETY_CHECKS}
    game_summaries = []
    status = "ok"
    stop_reason = None

    with harness.PrivacySafeAuditWriter(audit_path) as writer:
        for game_index, seed in enumerate(config.seeds, start=1):
            game_id = f"{game_id_prefix}_{game_index:03d}_seed_{seed}"
            budget = harness.DryRunBudget(
                game_id=game_id,
                max_gameplay_calls=config.max_gameplay_calls_per_game,
                max_belief_calls=config.max_belief_calls_per_game,
                max_total_calls=config.max_total_calls_per_game,
                max_wall_seconds=config.max_wall_seconds_per_game,
                global_budget=global_budget,
            )
            completed = False
            try:
                global_budget.check_wall_time(
                    game_id=game_id, category="gameplay"
                )
                harness.run_real_backend_game(
                    parsed_yaml=parsed_yaml,
                    samples_path=samples_path,
                    log_dir=None,
                    game_id=game_id,
                    seed=seed,
                    budget=budget,
                    writer=writer,
                    collector_wrapper=lambda collector: _MonitoredCollector(
                        collector,
                        monitor,
                        collection_mode=collection_mode,
                    ),
                    collection_mode=collection_mode,
                )
                completed = True
            except CollectionQualityStop as exc:
                status = "quality_stop"
                stop_reason = exc.reason
            except harness.DryRunBudgetExceeded as exc:
                status = "budget_stop"
                stop_reason = exc.budget_name
            except CollectionSampleSafetyViolation as exc:
                status = "safety_stop"
                stop_reason = exc.check_name
                safety_checks[exc.check_name] += 1
            except harness.DryRunSafetyViolation as exc:
                status = "safety_stop"
                stop_reason = _safety_check_name(exc)
                safety_checks[stop_reason] += 1

            game_summaries.append(
                {
                    "game_id": game_id,
                    "seed": seed,
                    "completed": completed,
                    "calls": budget.summary(),
                }
            )
            if status != "ok":
                break

    quality_summary = monitor.summary()
    call_audit_summary = summarize_call_audit(audit_path)
    started_game_count = len(game_summaries)
    completed_game_count = sum(game["completed"] for game in game_summaries)
    partial_game_count = started_game_count - completed_game_count
    usable_valid_report_count = quality_summary["status_counts"]["ok"]
    summary = {
        **dict(mode_metadata),
        "status": status,
        "stop_reason": stop_reason,
        "requested_game_count": 10,
        "started_game_count": started_game_count,
        "completed_game_count": completed_game_count,
        "partial_game_count": partial_game_count,
        "usable_valid_report_count": usable_valid_report_count,
        "games": game_summaries,
        "global_budget": global_budget.summary(),
        "quality": quality_summary,
        "safety_checks": safety_checks,
        "call_audit": call_audit_summary,
        "samples_path": str(samples_path),
        "call_audit_path": str(audit_path),
        "manifest_path": str(manifest_path),
        "provider_internal_cache": harness.PROVIDER_CACHE_NOTE,
        "exact_cost": "unavailable",
    }
    manifest.update(
        {
            "started_game_count": started_game_count,
            "completed_game_count": completed_game_count,
            "partial_game_count": partial_game_count,
            "stop_reason": stop_reason,
            "label_status_counts": quality_summary["status_counts"],
            "invalid_rate": quality_summary["invalid_rate"],
            "usable_valid_report_count": usable_valid_report_count,
        }
    )
    harness._atomic_json_dump(manifest, manifest_path)
    harness._atomic_json_dump(summary, summary_path)
    summary["summary_path"] = str(summary_path)
    return summary
