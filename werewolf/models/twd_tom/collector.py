"""Write synchronized playing-agent belief snapshots to JSONL."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from werewolf.models.twd_tom.samples import (
    ACTOR_PAIR_BELIEF_GENERATOR_NAME,
    ACTOR_PAIR_BELIEF_GENERATOR_VERSION,
    freeze_public_snapshot,
    make_twd_tom_sample,
)
from werewolf.speech.pair_belief_self_reporter import (
    canonical_json_sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def require_clean_collection_worktree(
    repo_path: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Require and describe one clean Git worktree for formal collection."""

    candidate = Path(repo_path).resolve()
    try:
        root_result = subprocess.run(
            ["git", "-C", str(candidate), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"formal collection requires a Git worktree: {candidate}"
        ) from exc
    root = Path(root_result.stdout.strip()).resolve()
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty_entries = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"cannot inspect formal collection Git worktree: {root}"
        ) from exc
    if re.fullmatch(r"[0-9a-fA-F]{40}", commit) is None:
        raise RuntimeError(
            f"formal collection requires a committed HEAD in repository: {root}"
        )
    if dirty_entries:
        dirty_files = "\n".join(dirty_entries)
        raise RuntimeError(
            "formal collection requires a clean Git worktree; "
            f"repository_root={root}; "
            f"dirty_entry_count={len(dirty_entries)}; dirty_files:\n"
            f"{dirty_files}"
        )
    return {
        "repository_root": str(root),
        "git_commit_sha": commit,
        "git_worktree_clean": True,
    }


def _validated_collection_git_state(
    value: Mapping[str, Any],
    *,
    expected_root: Path,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("collection_git_state must be a mapping")
    root = Path(value.get("repository_root", "")).resolve()
    if root != expected_root.resolve():
        raise ValueError("collection Git state repository root mismatch")
    commit = value.get("git_commit_sha")
    if not isinstance(commit, str) or re.fullmatch(
        r"[0-9a-fA-F]{40}", commit
    ) is None:
        raise ValueError("collection Git state has an invalid commit SHA")
    if value.get("git_worktree_clean") is not True:
        raise RuntimeError("formal collection requires git_worktree_clean=true")
    return {
        "repository_root": str(root),
        "git_commit_sha": commit,
        "git_worktree_clean": True,
    }


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_collection_provenance(
    *,
    source_config_path: str | Path,
    resolved_runtime_config: dict[str, Any],
    game_seed: int | None,
    repo_root: str | Path = REPO_ROOT,
    timestamp_utc: str | None = None,
    collection_git_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build secret-free, repository-relative provenance for raw collection."""

    root = Path(repo_root).resolve()
    git_state = _validated_collection_git_state(
        collection_git_state
        if collection_git_state is not None
        else require_clean_collection_worktree(root),
        expected_root=root,
    )
    config_path = Path(source_config_path)
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.absolute()
    if not config_path.is_file():
        raise FileNotFoundError(f"source config not found: {config_path}")
    try:
        relative_config = config_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("source config must be inside the repository") from exc
    if game_seed is not None and (
        isinstance(game_seed, bool) or not isinstance(game_seed, int)
    ):
        raise TypeError("game_seed must be an integer or None")
    if not isinstance(resolved_runtime_config, dict):
        raise TypeError("resolved_runtime_config must be a mapping")

    backends = resolved_runtime_config.get("backends")
    if not isinstance(backends, dict):
        raise ValueError("resolved runtime config must contain backends")
    return {
        "generator_name": ACTOR_PAIR_BELIEF_GENERATOR_NAME,
        "generator_version": ACTOR_PAIR_BELIEF_GENERATOR_VERSION,
        "git_commit_sha": git_state["git_commit_sha"],
        "git_worktree_clean": git_state["git_worktree_clean"],
        "collection_timestamp_utc": timestamp_utc
        or datetime.now(timezone.utc).isoformat(),
        "game_seed": game_seed,
        "source_config_path": relative_config,
        "source_config_sha256": _sha256_file(config_path),
        "resolved_runtime_config_sha256": canonical_json_sha256(
            resolved_runtime_config
        ),
        "resolved_backend_config_sha256": {
            alias: canonical_json_sha256(settings)
            for alias, settings in sorted(backends.items())
        },
    }


class TWDToMSampleCollector:
    """Collect and write raw subjective ToM samples."""

    def __init__(
        self,
        output_path: str,
        snapshot_collector,
        *,
        game_id: str,
        collection_provenance: dict[str, Any],
    ):
        if not isinstance(
            output_path,
            str,
        ) or not output_path.strip():
            raise ValueError(
                "output_path is required"
            )

        if snapshot_collector is None:
            raise ValueError(
                "snapshot_collector is required"
            )

        if not hasattr(
            snapshot_collector,
            "collect",
        ):
            raise TypeError(
                "snapshot_collector must provide collect()"
            )

        if not isinstance(game_id, str) or not game_id.strip():
            raise ValueError("game_id is required")
        self.game_id = game_id
        if not isinstance(collection_provenance, dict):
            raise TypeError("collection_provenance must be a mapping")
        if collection_provenance.get("git_worktree_clean") is not True:
            raise RuntimeError("formal collection requires git_worktree_clean=true")
        self.collection_provenance = dict(collection_provenance)
        self.snapshot_collector = (
            snapshot_collector
        )

        absolute_path = os.path.abspath(
            output_path
        )

        parent_directory = os.path.dirname(
            absolute_path
        )

        os.makedirs(
            parent_directory,
            exist_ok=True,
        )

        self.output_path = absolute_path

        self._file = open(
            absolute_path,
            "a",
            encoding="utf-8",
        )
        self.samples_written = 0

    def record(
        self,
        env,
        *,
        step_idx: int | None = None,
        trigger: str | None = None,
        phase: str | None = None,
        speaker_id: int | None = None,
        observer_ids: (
            Iterable[int] | None
        ) = None,
    ) -> dict[str, Any]:
        """Collect and write one synchronized subjective sample.

        Args:
            env:
                Environment providing the sole ``public_events`` history.

            step_idx:
                Rollout step before the public speech is generated.

            trigger:
                Description of the upcoming public event, normally ``speech`` or
                ``speech_pk``.

            observer_ids:
                Optional selected players. All players are collected
                when omitted.
        """

        if not hasattr(env, "public_events"):
            raise TypeError(
                "environment must provide public_events"
            )

        if isinstance(step_idx, bool) or not isinstance(step_idx, int):
            raise TypeError("step_idx is required")
        normalized_observers = list(observer_ids or [])
        report_trigger = {
            "speech": "pre_public_speech",
            "speech_pk": "pre_public_speech_pk",
        }.get(trigger)
        public_snapshot = freeze_public_snapshot(
            game_id=self.game_id,
            step_idx=step_idx,
            phase=phase,
            speaker_id=speaker_id,
            report_trigger=report_trigger,
            observer_ids=normalized_observers,
            public_events=env.public_events,
        )

        reports = self.snapshot_collector.collect(
            public_snapshot,
            env=env,
        )

        sample = make_twd_tom_sample(
            public_snapshot=public_snapshot,
            reports=reports,
            collection_provenance=self.collection_provenance,
        )

        self.write(
            sample
        )

        return sample

    def write(
        self,
        sample: dict[str, Any],
    ) -> None:
        """Write one already validated sample."""

        if self._file.closed:
            raise RuntimeError(
                "collector is closed"
            )

        line = json.dumps(
            sample,
            ensure_ascii=False,
            sort_keys=False,
            allow_nan=False,
        )

        self._file.write(
            line + "\n"
        )

        self._file.flush()
        self.samples_written += 1

    def close(self) -> None:
        """Close the JSONL file."""

        if not self._file.closed:
            self._file.close()

    @property
    def closed(self) -> bool:
        """Return whether the output file is closed."""

        return self._file.closed

    def __enter__(
        self,
    ) -> "TWDToMSampleCollector":
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()


__all__ = [
    "require_clean_collection_worktree",
    "build_collection_provenance",
    "TWDToMSampleCollector",
]
