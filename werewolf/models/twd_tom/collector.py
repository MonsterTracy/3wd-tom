"""Write synchronized playing-agent belief snapshots to JSONL."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import Any

from werewolf.models.twd_tom.samples import (
    freeze_public_snapshot,
    make_twd_tom_sample,
)


class TWDToMSampleCollector:
    """Collect and write raw subjective ToM samples."""

    def __init__(
        self,
        output_path: str,
        snapshot_collector,
        *,
        game_id: str,
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
        if not isinstance(game_id, str) or not game_id.strip():
            raise ValueError("game_id is required")
        self.game_id = game_id
        self.snapshot_collector = (
            snapshot_collector
        )

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
    "TWDToMSampleCollector",
]
