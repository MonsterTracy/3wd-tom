"""Online second-order ToM inference with an isolated JSONL audit log."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from script.twd_tom.eval import build_model_from_checkpoint, load_checkpoint
from werewolf.models.twd_tom.action_features import PublicEventFeatureBuilder
from werewolf.models.twd_tom.belief_labels import (
    pair_probabilities_to_belief_marginals,
)
from werewolf.models.twd_tom.dataset import TOM_INPUT_SCOPES
from werewolf.models.twd_tom.public_events import parse_public_phase
from werewolf.models.twd_tom.samples import freeze_public_snapshot
from werewolf.models.twd_tom.schema import (
    CANONICAL_PLAYER_ORDERING,
    NUM_PLAYERS,
    NUM_WOLF_PAIR_CLASSES,
    PAIR_ORDERING,
    SECOND_ORDER_TARGET_ENCODING,
    normalize_player,
)


class SecondOrderToMShadow:
    """Run strict public-only second-order inference at pre-speech cutoffs."""

    def __init__(
        self,
        *,
        checkpoint_path: str,
        device: str,
        output_path: str,
        game_id: str,
    ) -> None:
        for field_name, value in {
            "checkpoint_path": checkpoint_path,
            "device": device,
            "output_path": output_path,
            "game_id": game_id,
        }.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} is required")
        if device == "auto":
            raise ValueError("second-order shadow device must be explicit")

        resolved_output = Path(output_path).resolve()
        if resolved_output.exists():
            raise FileExistsError(
                f"second-order shadow output already exists: {resolved_output}"
            )
        if not resolved_output.parent.is_dir():
            raise FileNotFoundError(
                "second-order shadow output parent does not exist: "
                f"{resolved_output.parent}"
            )

        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint.get("tom_order") != 2:
            raise ValueError("second-order shadow requires a tom_order=2 checkpoint")

        self.device = torch.device(device)
        self.model = build_model_from_checkpoint(
            checkpoint,
            device=self.device,
        )
        if self.model.output_projection.out_features != NUM_WOLF_PAIR_CLASSES:
            raise ValueError("second-order shadow model must have 21 pair outputs")

        self.feature_builder = PublicEventFeatureBuilder(
            max_seq_len=self.model.config.max_seq_len,
            device=self.device,
        )
        self.game_id = game_id
        self.output_path = resolved_output
        self._seen_event_indices: set[int] = set()
        self._file = resolved_output.open("x", encoding="utf-8")

    def record(
        self,
        *,
        step_idx: int,
        phase: str,
        speaker_id: int,
        public_events: Sequence[Any],
    ) -> dict[str, Any]:
        """Infer and synchronously log one validated pre-speech snapshot."""

        if self._file.closed:
            raise RuntimeError("second-order shadow log is closed")
        day, phase_category = parse_public_phase(phase)
        report_trigger = {
            "day_speech": "pre_public_speech",
            "day_speech_pk": "pre_public_speech_pk",
        }.get(phase_category)
        if report_trigger is None:
            raise ValueError("second-order shadow inference requires a speech phase")

        snapshot = freeze_public_snapshot(
            game_id=self.game_id,
            step_idx=step_idx,
            phase=phase,
            speaker_id=speaker_id,
            report_trigger=report_trigger,
            observer_ids=tuple(range(1, NUM_PLAYERS + 1)),
            public_events=public_events,
        )
        event_idx = int(snapshot.public_events[-1]["event_idx"])
        if event_idx in self._seen_event_indices:
            raise RuntimeError(
                f"second-order shadow already recorded event_idx {event_idx}"
            )
        self._seen_event_indices.add(event_idx)

        features = self.feature_builder.encode_batch([snapshot.public_events])
        started_ns = time.perf_counter_ns()
        with torch.no_grad():
            output = self.model(**features)
            logits = output.get("observer_pair_logits")
            if not isinstance(logits, torch.Tensor):
                raise TypeError(
                    "second-order shadow model returned no pair logits"
                )
            if tuple(logits.shape) != (
                1,
                NUM_PLAYERS,
                NUM_WOLF_PAIR_CLASSES,
            ):
                raise ValueError(
                    "second-order shadow logits must have shape [1, 7, 21]"
                )
            pair_probabilities = torch.softmax(logits, dim=-1)
            wolf_marginals = pair_probabilities_to_belief_marginals(
                pair_probabilities
            )
        latency_ms = (time.perf_counter_ns() - started_ns) / 1_000_000

        pair_row_sums = pair_probabilities.sum(dim=-1)
        if not torch.allclose(
            pair_row_sums,
            torch.ones_like(pair_row_sums),
            atol=1e-6,
        ):
            raise ValueError("second-order shadow pair rows must sum to one")
        if tuple(wolf_marginals.shape) != (1, NUM_PLAYERS, NUM_PLAYERS):
            raise ValueError(
                "second-order shadow marginals must have shape [1, 7, 7]"
            )
        if not torch.isfinite(wolf_marginals).all():
            raise ValueError("second-order shadow marginals must be finite")
        if torch.any(wolf_marginals < 0) or torch.any(wolf_marginals > 1):
            raise ValueError("second-order shadow marginals must be in [0, 1]")
        marginal_row_sums = wolf_marginals.sum(dim=-1)
        if not torch.allclose(
            marginal_row_sums,
            torch.full_like(marginal_row_sums, 2.0),
            atol=1e-6,
        ):
            raise ValueError("second-order shadow marginal rows must sum to two")

        record = {
            "game_id": self.game_id,
            "event_idx": event_idx,
            "day": day,
            "phase": phase,
            "current_speaker": normalize_player(speaker_id),
            "tom_order": 2,
            "model_input_scope": TOM_INPUT_SCOPES[2],
            "target_encoding": SECOND_ORDER_TARGET_ENCODING,
            "pair_class_count": NUM_WOLF_PAIR_CLASSES,
            "pair_ordering": PAIR_ORDERING,
            "player_ordering": list(CANONICAL_PLAYER_ORDERING),
            "public_event_count": len(snapshot.public_events),
            "pair_probability_matrix": (
                pair_probabilities[0].detach().cpu().tolist()
            ),
            "wolf_marginal_matrix": wolf_marginals[0].detach().cpu().tolist(),
            "inference_latency_ms": latency_ms,
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()
        return record

    def close(self) -> None:
        """Close the independent shadow JSONL file."""

        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> SecondOrderToMShadow:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = ["SecondOrderToMShadow"]
