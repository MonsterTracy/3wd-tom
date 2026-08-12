import json
from dataclasses import fields
from pathlib import Path

import pytest

from script.twd_tom import formal_batch_collection as formal
from script.twd_tom import real_backend_dry_run as harness
from script.twd_tom import monitored_collection as monitored
from run_random import PUBLIC_ONLY_COLLECTION_MODE
from werewolf.models.twd_tom.samples import (
    PUBLIC_ONLY_SAMPLE_SCHEMA_VERSION,
    SAMPLE_SCHEMA_VERSION,
)
from werewolf.models.twd_tom.schema import (
    PUBLIC_ONLY_LABEL_PROMPT_VERSION,
    PUBLIC_ONLY_LABEL_PROVENANCE,
)


def _monitored_config(tmp_path, **overrides):
    runtime_path = tmp_path / "runtime.yaml"
    runtime_path.write_text("fake: true\n", encoding="utf-8")
    values = {
        "runtime_config_path": str(runtime_path),
        "output_dir": str(
            tmp_path / "logs" / "belief_set_formal_batch_batch_0001"
        ),
        "game_count": 10,
        "seeds": tuple(range(2001, 2011)),
        "max_gameplay_calls_per_game": 96,
        "max_belief_calls_per_game": 224,
        "max_total_calls_per_game": 320,
        "max_wall_seconds_per_game": 3600.0,
        "max_total_calls": 2000,
        "max_wall_seconds": 18000.0,
        "privacy_safe_logging": True,
        "audit_only_metadata": True,
    }
    values.update(overrides)
    return monitored.MonitoredCollectionConfig(**values)


def test_formal_batch_requires_safe_batch_id_and_matching_output(tmp_path):
    monitored_config = _monitored_config(tmp_path)
    formal.FormalBatchConfig(batch_id="batch_0001", monitored=monitored_config)
    with pytest.raises(ValueError, match="filesystem-safe"):
        formal.FormalBatchConfig(batch_id="../batch", monitored=monitored_config)
    with pytest.raises(ValueError, match="contain batch_id"):
        formal.FormalBatchConfig(batch_id="batch_0002", monitored=monitored_config)


def test_public_only_formal_mode_records_separate_artifact_provenance(
    tmp_path, monkeypatch, suspicion_sample_factory
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(harness, "_runtime_source_commit", lambda: "a" * 40)

    def fake_game(**kwargs):
        assert kwargs["collection_mode"] == PUBLIC_ONLY_COLLECTION_MODE
        sample = suspicion_sample_factory(
            game_id=kwargs["game_id"], observers=(1,)
        )
        sample["schema_version"] = PUBLIC_ONLY_SAMPLE_SCHEMA_VERSION
        sample["label_prompt_version"] = PUBLIC_ONLY_LABEL_PROMPT_VERSION
        sample["label_provenance"] = PUBLIC_ONLY_LABEL_PROVENANCE
        sample["known_werewolves"]["player1"] = []
        sample["known_non_werewolves"]["player1"] = []
        sample["suspected_werewolves"]["player1"] = []

        class Collector:
            def write(self, _sample):
                return None

            def record(self):
                self.write(sample)
                return sample

            def close(self):
                return None

        wrapped = kwargs["collector_wrapper"](Collector())
        wrapped.record()
        wrapped.close()

    monkeypatch.setattr(harness, "run_real_backend_game", fake_game)
    monitored_config = _monitored_config(tmp_path)
    formal.run_formal_batch(
        formal.FormalBatchConfig(
            batch_id="batch_0001",
            monitored=monitored_config,
            collection_mode=PUBLIC_ONLY_COLLECTION_MODE,
        )
    )

    manifest = json.loads(
        (
            Path(monitored_config.output_dir) / "formal_batch_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == PUBLIC_ONLY_SAMPLE_SCHEMA_VERSION
    assert manifest["belief_information_scope"] == "public_events_only"
    assert manifest["playing_agent_context_reused"] is False
    assert manifest["true_role_visible"] is False
    assert manifest["private_memory_visible"] is False
    assert manifest["prompt_version"] == PUBLIC_ONLY_LABEL_PROMPT_VERSION


def test_formal_cli_and_config_cannot_enable_backend_retry(tmp_path):
    retry_field = "max_backend_" + "retries"
    retry_option = "--max-backend-" + "retries"
    assert retry_field not in {
        field.name for field in fields(monitored.MonitoredCollectionConfig)
    }
    assert retry_option not in {
        option
        for action in formal.build_arg_parser()._actions
        for option in action.option_strings
    }
    with pytest.raises(TypeError, match="unexpected keyword"):
        _monitored_config(tmp_path, **{retry_field: 1})
    valid_argv = [
        "--batch-id",
        "batch_0001",
        "--config",
        str(tmp_path / "runtime.yaml"),
        "--game-count",
        "10",
        "--seeds",
        *(str(seed) for seed in range(2001, 2011)),
        "--output-dir",
        str(tmp_path / "logs" / "belief_set_formal_batch_batch_0001"),
        "--max-gameplay-calls-per-game",
        "96",
        "--max-belief-calls-per-game",
        "224",
        "--max-total-calls-per-game",
        "320",
        "--max-wall-seconds-per-game",
        "3600",
        "--max-total-calls",
        "2000",
        "--max-wall-seconds",
        "18000",
        "--privacy-safe-logging",
        "--audit-only-metadata",
    ]
    with pytest.raises(SystemExit) as exc_info:
        formal.build_arg_parser().parse_args(
            [*valid_argv, retry_option, "1"]
        )
    assert exc_info.value.code == 2


def test_formal_cli_passes_optional_logs_root_to_monitored_config(
    tmp_path, monkeypatch
):
    captured = []

    def fake_run(config):
        captured.append(config)
        return {"status": "ok"}

    monkeypatch.setattr(formal, "run_formal_batch", fake_run)
    logs_root = tmp_path / "data" / "project" / "logs"
    output_dir = logs_root / "formal_batch_batch_0001"
    formal.main(
        [
            "--batch-id",
            "batch_0001",
            "--config",
            str(tmp_path / "runtime.yaml"),
            "--game-count",
            "10",
            "--seeds",
            *(str(seed) for seed in range(2001, 2011)),
            "--logs-root",
            str(logs_root),
            "--output-dir",
            str(output_dir),
            "--max-gameplay-calls-per-game",
            "96",
            "--max-belief-calls-per-game",
            "224",
            "--max-total-calls-per-game",
            "320",
            "--max-wall-seconds-per-game",
            "3600",
            "--max-total-calls",
            "2000",
            "--max-wall-seconds",
            "18000",
            "--privacy-safe-logging",
            "--audit-only-metadata",
        ]
    )
    assert captured[0].monitored.logs_root == str(logs_root)


def test_formal_cli_defaults_logs_root_to_none(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(
        formal,
        "run_formal_batch",
        lambda config: captured.append(config) or {"status": "ok"},
    )
    formal.main(
        [
            "--batch-id", "batch_0001",
            "--config", str(tmp_path / "runtime.yaml"),
            "--game-count", "10",
            "--seeds", *(str(seed) for seed in range(2001, 2011)),
            "--output-dir", str(tmp_path / "logs" / "formal_batch_batch_0001"),
            "--max-gameplay-calls-per-game", "96",
            "--max-belief-calls-per-game", "224",
            "--max-total-calls-per-game", "320",
            "--max-wall-seconds-per-game", "3600",
            "--max-total-calls", "2000",
            "--max-wall-seconds", "18000",
            "--privacy-safe-logging",
            "--audit-only-metadata",
        ]
    )
    assert captured[0].monitored.logs_root is None


def test_fake_formal_batch_reuses_monitor_and_writes_final_acceptance_metadata(
    tmp_path, monkeypatch, suspicion_sample_factory
):
    monkeypatch.chdir(tmp_path)
    runtime_head = "d" * 40
    started = []
    monkeypatch.setattr(harness, "_runtime_source_commit", lambda: runtime_head)

    def fake_game(**kwargs):
        started.append((kwargs["game_id"], kwargs["seed"]))
        kwargs["budget"].before_dispatch("gameplay")
        kwargs["writer"].write(
            {
                "game_id": kwargs["game_id"],
                "call_category": "gameplay",
                "dispatch_status": "ok",
                "latency_ms": 10.0,
                "usage_available": True,
                "input_tokens": 8,
                "output_tokens": 2,
                "total_tokens": 10,
            }
        )
        sample = suspicion_sample_factory(
            game_id=kwargs["game_id"], observers=(1,)
        )
        sample["belief_status"]["player1"] = "ok"
        sample["suspected_werewolves"]["player1"] = []
        sample["belief_errors"]["player1"] = None

        class Collector:
            def write(self, _sample):
                return None

            def record(self):
                self.write(sample)
                return sample

            def close(self):
                return None

        monitored_collector = kwargs["collector_wrapper"](Collector())
        monitored_collector.record()
        monitored_collector.close()

    monkeypatch.setattr(harness, "run_real_backend_game", fake_game)
    monitored_config = _monitored_config(tmp_path)
    config = formal.FormalBatchConfig(
        batch_id="batch_0001", monitored=monitored_config
    )
    summary = formal.run_formal_batch(config)

    assert len(started) == 10
    assert [seed for _game_id, seed in started] == list(range(2001, 2011))
    assert summary["status"] == "ok"
    assert summary["batch_id"] == "batch_0001"
    assert summary["started_game_count"] == 10
    assert summary["completed_game_count"] == 10
    assert summary["partial_game_count"] == 0
    assert summary["usable_valid_report_count"] == 10
    assert summary["quality"]["status_counts"] == {
        "ok": 10,
        "parse_error": 0,
        "semantic_error": 0,
        "reporter_error": 0,
    }
    assert summary["call_audit"]["request_count"] == 10
    assert summary["call_audit"]["dispatch_status_counts"] == {
        "ok": 10,
        "error": 0,
    }
    assert summary["call_audit"]["token_totals"]["total_tokens"] == 100
    assert summary["call_audit"]["latency"] == {
        "mean_ms": 10.0,
        "median_ms": 10.0,
        "p95_ms": 10.0,
        "max_ms": 10.0,
    }

    output = Path(monitored_config.output_dir)
    expected_files = {
        "formal_batch_samples.jsonl",
        "formal_batch_call_audit.jsonl",
        "formal_batch_manifest.json",
        "formal_batch_summary.json",
    }
    assert {path.name for path in output.iterdir()} == expected_files
    assert not {"train.jsonl", "validation.jsonl", "test.jsonl"} & expected_files
    manifest = json.loads(
        (output / "formal_batch_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["batch_id"] == "batch_0001"
    assert manifest["schema_version"] == SAMPLE_SCHEMA_VERSION
    assert manifest["source_commit"] == runtime_head
    assert manifest["seeds"] == list(range(2001, 2011))
    assert manifest["completed_game_count"] == 10
    assert manifest["stop_reason"] is None
    assert manifest["label_status_counts"]["ok"] == 10
    assert manifest["usable_valid_report_count"] == 10


def test_formal_quality_stop_finalizes_partial_batch_without_starting_next_game(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    started = []
    monkeypatch.setattr(harness, "_runtime_source_commit", lambda: "e" * 40)

    def fake_game(**kwargs):
        started.append(kwargs["game_id"])
        raise monitored.CollectionQualityStop(
            "observer_consecutive_same_error_rule_gt_or_eq_3"
        )

    monkeypatch.setattr(harness, "run_real_backend_game", fake_game)
    monitored_config = _monitored_config(tmp_path)
    summary = formal.run_formal_batch(
        formal.FormalBatchConfig(
            batch_id="batch_0001",
            monitored=monitored_config,
        )
    )

    assert started == ["formal_batch_0001_game_001_seed_2001"]
    assert summary["status"] == "quality_stop"
    assert summary["started_game_count"] == 1
    assert summary["completed_game_count"] == 0
    assert summary["partial_game_count"] == 1
    assert summary["stop_reason"] == (
        "observer_consecutive_same_error_rule_gt_or_eq_3"
    )
    manifest = json.loads(
        (
            Path(monitored_config.output_dir) / "formal_batch_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["completed_game_count"] == 0
    assert manifest["partial_game_count"] == 1
    assert manifest["stop_reason"] == summary["stop_reason"]


def test_call_audit_acceptance_summary_uses_metadata_only(tmp_path):
    path = tmp_path / "audit.jsonl"
    records = [
        {
            "call_category": "gameplay",
            "dispatch_status": "ok",
            "latency_ms": 1.0,
            "usage_available": True,
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
        },
        {
            "call_category": "belief",
            "dispatch_status": "error",
            "latency_ms": 9.0,
            "usage_available": False,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        },
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    result = monitored.summarize_call_audit(path)
    assert result["request_count"] == 2
    assert result["request_count_by_category"] == {
        "gameplay": 1,
        "belief": 1,
    }
    assert result["dispatch_status_counts"] == {
        "ok": 1,
        "error": 1,
    }
    assert result["usage_available_count"] == 1
    assert result["token_totals"] == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
    }
    assert result["latency"] == {
        "mean_ms": 5.0,
        "median_ms": 5.0,
        "p95_ms": 9.0,
        "max_ms": 9.0,
    }


def test_quality_monitor_classifies_by_status_not_legacy_error_text():
    for status in (
        "parse_error",
        "semantic_error",
        "reporter_error",
    ):
        monitor = monitored.CollectionQualityMonitor()
        for index, error in enumerate(
            (
                "same text for every status",
                "different text within one status",
                "third text reaches the frozen threshold",
            ),
            start=1,
        ):
            monitor.observe_report(
                game_id="game_001",
                phase="1_day_speech",
                observer="player1",
                status=status,
                error=error,
            )
            if index < 3:
                assert monitor.stop_reason is None

        assert monitor.error_rule_counts == {status: 3}
        assert monitor.stop_reason == (
            "observer_consecutive_same_error_rule_gt_or_eq_3"
        )

    switch_monitor = monitored.CollectionQualityMonitor()
    for status in (
        "parse_error",
        "parse_error",
        "semantic_error",
        "parse_error",
        "reporter_error",
        "reporter_error",
    ):
        switch_monitor.observe_report(
            game_id="game_001",
            phase="1_day_speech",
            observer="player1",
            status=status,
            error="identical legacy error text",
        )
    assert switch_monitor.error_rule_counts == {
        "parse_error": 3,
        "semantic_error": 1,
        "reporter_error": 2,
    }
    assert switch_monitor.stop_reason is None


def test_quality_monitor_validates_samples_for_its_collection_mode(
    suspicion_sample_factory,
):
    private_sample = suspicion_sample_factory(observers=(1, 3))
    monitored.CollectionQualityMonitor().observe_sample(private_sample)

    public_sample = suspicion_sample_factory(observers=(1, 3))
    public_sample["schema_version"] = PUBLIC_ONLY_SAMPLE_SCHEMA_VERSION
    public_sample["label_prompt_version"] = PUBLIC_ONLY_LABEL_PROMPT_VERSION
    public_sample["label_provenance"] = PUBLIC_ONLY_LABEL_PROVENANCE
    for observer in public_sample["observer_ids"]:
        subject = f"player{observer}"
        public_sample["known_werewolves"][subject] = []
        public_sample["known_non_werewolves"][subject] = []

    public_monitor = monitored.CollectionQualityMonitor(
        collection_mode=PUBLIC_ONLY_COLLECTION_MODE,
    )
    public_monitor.observe_sample(public_sample)

    with pytest.raises(
        monitored.CollectionSampleSafetyViolation,
        match="unsupported schema",
    ) as private_mode_error:
        monitored.CollectionQualityMonitor().observe_sample(public_sample)
    assert private_mode_error.value.check_name == "old_schema"

    mixed_sample = dict(public_sample)
    mixed_sample["schema_version"] = SAMPLE_SCHEMA_VERSION
    with pytest.raises(
        monitored.CollectionSampleSafetyViolation,
        match="unsupported schema",
    ) as mixed_schema_error:
        public_monitor.observe_sample(mixed_sample)
    assert mixed_schema_error.value.check_name == "old_schema"
