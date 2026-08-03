from __future__ import annotations

import inspect
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_random
from script.twd_tom import collect
from script.twd_tom import formal_batch_collection as formal
from script.twd_tom import monitored_collection as monitored
from script.twd_tom import pipeline
from script.twd_tom import real_backend_dry_run as dry_run
from werewolf.models.twd_tom import collector
from werewolf.models.twd_tom.samples import (
    ACTOR_PAIR_BELIEF_ANNOTATION_VERSION,
    ACTOR_PAIR_BELIEF_SCHEMA_VERSION,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _clean_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Collection Gate Test")
    tracked = repo / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "initial")
    return repo, tracked


def _state(root: Path) -> dict[str, object]:
    return {
        "repository_root": str(root.resolve()),
        "git_commit_sha": "a" * 40,
        "git_worktree_clean": True,
    }


def _blocked():
    raise RuntimeError("blocked by clean collection gate")


def _monitored_config(tmp_path: Path) -> monitored.MonitoredCollectionConfig:
    return monitored.MonitoredCollectionConfig(
        runtime_config_path=str(tmp_path / "runtime.yaml"),
        output_dir=str(tmp_path / "formal_batch_batch_1"),
        game_count=10,
        seeds=tuple(range(10)),
        max_gameplay_calls_per_game=1,
        max_belief_calls_per_game=1,
        max_total_calls_per_game=2,
        max_wall_seconds_per_game=1.0,
        max_total_calls=20,
        max_wall_seconds=10.0,
        privacy_safe_logging=True,
        audit_only_metadata=True,
    )


def test_clean_collection_gate_succeeds_and_returns_commit(tmp_path):
    repo, _ = _clean_repo(tmp_path)
    result = collector.require_clean_collection_worktree(repo)
    assert result == {
        "repository_root": str(repo.resolve()),
        "git_commit_sha": _git(repo, "rev-parse", "HEAD"),
        "git_worktree_clean": True,
    }
    assert "allow_dirty" not in inspect.signature(
        collector.require_clean_collection_worktree
    ).parameters


@pytest.mark.parametrize("dirty_kind", ["tracked", "staged", "untracked"])
def test_clean_collection_gate_rejects_every_dirty_state(tmp_path, dirty_kind):
    repo, tracked = _clean_repo(tmp_path)
    expected_name = "tracked.txt"
    if dirty_kind == "untracked":
        expected_name = "untracked.txt"
        (repo / expected_name).write_text("dirty\n", encoding="utf-8")
    else:
        tracked.write_text("dirty\n", encoding="utf-8")
        if dirty_kind == "staged":
            _git(repo, "add", "tracked.txt")
    with pytest.raises(RuntimeError) as exc_info:
        collector.require_clean_collection_worktree(repo)
    message = str(exc_info.value)
    assert f"repository_root={repo.resolve()}" in message
    assert "dirty_entry_count=1" in message
    assert expected_name in message


def test_clean_collection_gate_rejects_non_git_directory(tmp_path):
    with pytest.raises(RuntimeError, match="requires a Git worktree"):
        collector.require_clean_collection_worktree(tmp_path)


def test_provenance_reuses_successful_gate_result(tmp_path, monkeypatch):
    repo, _ = _clean_repo(tmp_path)
    config_path = repo / "runtime.yaml"
    config_path.write_text("backends: {}\n", encoding="utf-8")
    _git(repo, "add", "runtime.yaml")
    _git(repo, "commit", "-m", "add config")
    git_state = collector.require_clean_collection_worktree(repo)
    monkeypatch.setattr(
        collector,
        "require_clean_collection_worktree",
        lambda *_args, **_kwargs: pytest.fail("provenance reran the Git gate"),
    )
    provenance = collector.build_collection_provenance(
        source_config_path=config_path,
        resolved_runtime_config={"backends": {}},
        game_seed=42,
        repo_root=repo,
        collection_git_state=git_state,
    )
    assert provenance["git_commit_sha"] == git_state["git_commit_sha"]
    assert provenance["git_worktree_clean"] is True


def test_run_random_collection_gate_precedes_every_side_effect(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(run_random, "require_clean_collection_worktree", _blocked)
    monkeypatch.setattr(run_random.os, "makedirs", lambda *_a, **_k: events.append("mkdir"))
    monkeypatch.setattr(
        run_random, "load_named_backends", lambda *_a, **_k: events.append("backend")
    )
    monkeypatch.setattr(run_random, "build_runtime", lambda *_a, **_k: events.append("runtime"))
    args = SimpleNamespace(
        log_save_path=str(tmp_path / "logs"),
        twd_tom_sample_path=str(tmp_path / "raw.jsonl"),
    )
    with pytest.raises(RuntimeError, match="blocked by clean collection gate"):
        run_random.main_cli(args)
    assert events == []
    assert not Path(args.log_save_path).exists()


def test_run_random_without_collection_does_not_require_gate(monkeypatch, tmp_path):
    gate_calls = []
    monkeypatch.setattr(
        run_random,
        "require_clean_collection_worktree",
        lambda: gate_calls.append("gate"),
    )

    class ReachedOrdinaryGameplaySideEffect(Exception):
        pass

    monkeypatch.setattr(
        run_random.os,
        "makedirs",
        lambda *_a, **_k: (_ for _ in ()).throw(ReachedOrdinaryGameplaySideEffect),
    )
    args = SimpleNamespace(
        log_save_path=str(tmp_path / "logs"),
        twd_tom_sample_path=None,
    )
    with pytest.raises(ReachedOrdinaryGameplaySideEffect):
        run_random.main_cli(args)
    assert gate_calls == []


def test_dedicated_collection_gate_precedes_side_effects(monkeypatch, tmp_path):
    monkeypatch.setattr(collect, "require_clean_collection_worktree", _blocked)
    config = collect.CollectionRunConfig(
        runtime_config_path=str(tmp_path / "runtime.yaml"),
        sample_path=str(tmp_path / "data" / "raw.jsonl"),
        log_save_path=str(tmp_path / "logs"),
    )
    with pytest.raises(RuntimeError, match="blocked by clean collection gate"):
        collect.run_collection(config)
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "logs").exists()


def test_pipeline_collect_gate_precedes_path_access(monkeypatch):
    monkeypatch.setattr(pipeline, "require_clean_collection_worktree", _blocked)
    with pytest.raises(RuntimeError, match="blocked by clean collection gate"):
        pipeline._run_collect({})


def test_dry_run_gate_precedes_backend_and_output(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(dry_run, "require_clean_collection_worktree", _blocked)
    monkeypatch.setattr(
        dry_run, "load_named_backends", lambda *_a, **_k: events.append("backend")
    )
    config = dry_run.RealBackendDryRunConfig(
        runtime_config_path=str(tmp_path / "runtime.yaml"),
        output_dir=str(tmp_path / "dry_run"),
        game_count=2,
        seeds=(1, 2),
        max_gameplay_calls_per_game=1,
        max_belief_calls_per_game=1,
        max_total_calls_per_game=2,
        max_wall_seconds_per_game=1.0,
        privacy_safe_logging=True,
        audit_only_metadata=True,
    )
    with pytest.raises(RuntimeError, match="blocked by clean collection gate"):
        dry_run.run_dry_run(config)
    assert events == []
    assert not (tmp_path / "dry_run").exists()


def test_independent_real_backend_game_cannot_bypass_gate(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(dry_run, "require_clean_collection_worktree", _blocked)
    monkeypatch.setattr(
        dry_run, "load_named_backends", lambda *_a, **_k: events.append("backend")
    )
    with pytest.raises(RuntimeError, match="blocked by clean collection gate"):
        dry_run.run_real_backend_game(
            parsed_yaml={},
            samples_path=tmp_path / "raw.jsonl",
            log_dir=None,
            game_id="g",
            seed=1,
            budget=None,
            writer=None,
            source_config_path=tmp_path / "runtime.yaml",
        )
    assert events == []
    assert not (tmp_path / "raw.jsonl").exists()


def test_monitored_and_formal_batch_gate_precede_output(monkeypatch, tmp_path):
    config = _monitored_config(tmp_path)
    monkeypatch.setattr(monitored, "require_clean_collection_worktree", _blocked)
    with pytest.raises(RuntimeError, match="blocked by clean collection gate"):
        monitored.run_monitored_collection(
            config,
            artifact_prefix="formal_batch",
            required_name_token="formal_batch",
            game_id_prefix="g",
            mode_metadata={},
        )
    assert not Path(config.output_dir).exists()

    monkeypatch.setattr(formal, "require_clean_collection_worktree", _blocked)
    with pytest.raises(RuntimeError, match="blocked by clean collection gate"):
        formal.run_formal_batch(
            formal.FormalBatchConfig(batch_id="batch_1", monitored=config)
        )
    assert not Path(config.output_dir).exists()


def test_formal_batch_uses_new_schema_constants(monkeypatch, tmp_path):
    captured = {}
    config = _monitored_config(tmp_path)
    monkeypatch.setattr(
        formal,
        "require_clean_collection_worktree",
        lambda: _state(tmp_path),
    )

    def fake_run(_config, **kwargs):
        captured.update(kwargs["mode_metadata"])
        return {"status": "ok"}

    monkeypatch.setattr(formal, "run_monitored_collection", fake_run)
    formal.run_formal_batch(
        formal.FormalBatchConfig(batch_id="batch_1", monitored=config)
    )
    assert captured["schema_version"] == ACTOR_PAIR_BELIEF_SCHEMA_VERSION
    assert captured["annotation_version"] == ACTOR_PAIR_BELIEF_ANNOTATION_VERSION
