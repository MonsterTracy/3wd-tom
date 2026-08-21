"""Archived post-speech collection of raw subjective belief reports."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from archive.legacy_tom.werewolf.models.tom.schema import (
    SpeechAction,
    normalize_episode_context,
    normalize_player,
)


class Collector:
    """Write one alive-only sample after a committed formal speech."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        game_id: str,
        seed: int | None,
        episode_context: str,
        reporter,
    ) -> None:
        if not isinstance(game_id, str) or not game_id.strip():
            raise ValueError("game_id must be non-empty text")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("seed must be an integer or None")
        if reporter is None or not hasattr(reporter, "report"):
            raise TypeError("reporter must provide report()")
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.game_id = game_id
        self.seed = seed
        self.episode_context = normalize_episode_context(episode_context)
        self.reporter = reporter
        self._output = path.open("a", encoding="utf-8")
        self.samples_written = 0

    @staticmethod
    def _committed_speech(env, speaker_id: int | str) -> tuple[int, dict[str, Any]]:
        events = getattr(env, "public_events", None)
        if isinstance(events, (str, bytes)) or not isinstance(events, Sequence):
            raise TypeError("environment must provide a public-event sequence")
        speech_index = next(
            (
                index
                for index in range(len(events) - 1, -1, -1)
                if isinstance(events[index], Mapping)
                and events[index].get("event_type") == "public_speech"
            ),
            None,
        )
        if speech_index is None:
            raise RuntimeError("completed speech has no public_speech event")
        speech = events[speech_index]
        if speech.get("event_idx") != speech_index:
            raise ValueError("public speech event_idx does not match ledger position")
        if normalize_player(speech.get("speaker")) != normalize_player(speaker_id):
            raise ValueError("completed speech speaker mismatch")
        return speech_index, dict(speech)

    @staticmethod
    def _alive_observers(env) -> list[int]:
        alive = getattr(env, "alive", None)
        if not isinstance(alive, (list, tuple)) or len(alive) != 7:
            raise TypeError("environment must provide seven alive flags")
        observers = [
            index + 1
            for index, value in enumerate(alive)
            if value == 1
        ]
        if not observers:
            raise RuntimeError("post-speech collection has no alive observers")
        return observers

    def record(
        self,
        env,
        *,
        step_idx: int,
        round_number: int,
        phase: str,
        speaker_id: int,
    ) -> dict[str, Any] | None:
        if isinstance(step_idx, bool) or not isinstance(step_idx, int):
            raise TypeError("step_idx must be an integer")
        if isinstance(round_number, bool) or not isinstance(round_number, int):
            raise TypeError("round_number must be an integer")
        if phase not in {"speech", "speech_pk"}:
            raise ValueError("formal collection requires a speech phase")
        speech_index, speech = self._committed_speech(env, speaker_id)
        raw_actions = speech.get("sp_actions")
        if isinstance(raw_actions, (str, bytes)) or not isinstance(raw_actions, Sequence):
            raise TypeError("committed speech actions must be a sequence")
        if not raw_actions:
            return None
        actions = [
            SpeechAction.from_values(action[0], action[1], action[2]).to_list()
            for action in raw_actions
            if isinstance(action, Sequence) and len(action) == 3
        ]
        if len(actions) != len(raw_actions):
            raise ValueError("every committed speech action must be a triplet")

        observer_ids = self._alive_observers(env)
        reports = []
        for observer_id in observer_ids:
            observation = env.get_observation_for(observer_id)
            reports.append(self.reporter.report(observer_id, observation))

        public_events = deepcopy(list(env.public_events[: speech_index + 1]))
        canonical_prefix = json.dumps(
            public_events,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        sample = {
            "game_id": self.game_id,
            "seed": self.seed,
            "episode_context": self.episode_context,
            "step_idx": step_idx,
            "speaker_id": normalize_player(speaker_id),
            "round": round_number,
            "phase": phase,
            "formal_speech_actions": actions,
            "public_history_cutoff": {
                "event_idx": speech_index,
                "digest": hashlib.sha256(
                    canonical_prefix.encode("utf-8")
                ).hexdigest(),
            },
            "public_events": public_events,
            "alive_observers": [normalize_player(value) for value in observer_ids],
            "observer_reports": reports,
        }
        self._output.write(
            json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n"
        )
        self._output.flush()
        self.samples_written += 1
        return sample

    def close(self) -> None:
        if not self._output.closed:
            self._output.close()


__all__ = ["Collector"]
