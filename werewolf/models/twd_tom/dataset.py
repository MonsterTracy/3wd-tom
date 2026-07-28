"""Dataset for classic-seven pre-speech subjective ToM samples.

Each projected sample contains:

- ``public_events``:
    Complete canonical public-event prefix. Public speech events retain raw
    text for audit, but the feature builder encodes only the shared
    raw-text-free structured projection.

- ``pair_targets``:
    Offline-projected per-observer distributions over the global 21 pairs.

The dataset converts them into:

- subject_ids:                 [T]
- action_ids:                  [T]
- object_ids:                  [T]
- attention_mask:              [T]
- pair_targets:                [7, 21]
- subject_mask:                [7]

Collation right-pads event tensors to ``[B, T]`` and stacks targets
and the subject mask to ``[B, 7, 21]`` and ``[B, 7]`` respectively.

No true role assignment or truth-derived wolf label is accepted.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from werewolf.models.twd_tom.action_features import PublicEventFeatureBuilder
from werewolf.models.twd_tom.belief_labels import (
    close_hard_knowledge,
    suspicion_set_to_pair_target,
)
from werewolf.models.twd_tom.samples import (
    REPORT_TRIGGERS,
    SAMPLE_SCHEMA_VERSION as PLAYER_SUSPICION_SCHEMA_VERSION,
)
from werewolf.models.twd_tom.public_events import (
    PUBLIC_EVENT_SCHEMA_VERSION,
    normalize_public_events,
    parse_public_phase,
    public_event_digest,
    public_speech_actions,
    structured_input_digest,
)
from werewolf.models.twd_tom.schema import (
    LABEL_PROMPT_VERSION,
    LABEL_PROVENANCE,
    NUM_PLAYERS,
    NUM_WOLF_PAIR_CLASSES,
    PAIR_ORDERING,
    PLAYER_NAMES,
    PROJECTED_SCHEMA_VERSION,
    PROJECTION_VERSION,
    PLAYER_TO_ID,
    TARGET_DISTRIBUTION_IS_DETERMINISTIC_ENCODING,
    TARGET_DISTRIBUTION_IS_REPORTER_PROBABILITY,
    canonical_wolf_pairs,
    normalize_player,
    validate_player_suspicion,
)
from werewolf.speech.private_belief_perceiver import (
    STATUS_OK,
    STATUS_PARSE_ERROR,
    STATUS_REPORTER_ERROR,
    STATUS_SEMANTIC_ERROR,
)


PROJECTED_SAMPLE_FIELDS = frozenset(
    {
        "schema_version",
        "source_schema_version",
        "projection_version",
        "pair_ordering",
        "game_id",
        "step_idx",
        "phase",
        "speaker_id",
        "report_trigger",
        "public_event_schema_version",
        "public_events",
        "public_event_digest",
        "structured_input_digest",
        "observer_ids",
        "suspected_werewolves",
        "known_werewolves",
        "known_non_werewolves",
        "belief_status",
        "belief_errors",
        "pair_targets",
        "label_cutoff_step_idx",
        "public_action_count",
        "label_prompt_version",
        "label_provenance",
        "agent_backend_ids",
        "target_distribution_is_reporter_probability",
        "target_distribution_is_deterministic_encoding",
    }
)


FORBIDDEN_LEGACY_FIELDS = {
    "observation",
    "event_tokens",
    "wolf_labels",
    "roles",
    "true_roles",
    "truth",
    "actual_wolves",
    "alive_mask",
    "observer_id",
    "guesses",
    "guess_status",
    "guess_errors",
    "plausible_wolf_pairs",
    "excluded_werewolves",
    "raw_responses",
    "raw_response",
    "god_view",
    "private_observation",
    "observer_role",
    "role",
    "wolf_teammate",
    "teammate",
    "seer_result",
    "witch_target",
    "kill_decision",
    "future_speech",
    "future_vote",
    "future_death",
    "future_events",
    "hidden_action",
    "game_result",
}


def _normalize_observer_ids(
    observer_ids: Any,
) -> list[int]:
    """Validate selected 1-based observer IDs."""

    if (
        isinstance(observer_ids, (str, bytes))
        or not isinstance(observer_ids, Sequence)
    ):
        raise TypeError(
            "observer_ids must be a sequence"
        )

    if not observer_ids:
        raise ValueError(
            "observer_ids cannot be empty"
        )

    normalized: list[int] = []
    seen: set[int] = set()

    for player_id in observer_ids:
        if (
            isinstance(player_id, bool)
            or not isinstance(player_id, int)
        ):
            raise TypeError(
                "observer IDs must be integers"
            )

        if not 1 <= player_id <= 7:
            raise ValueError(
                "observer IDs must be in [1, 7]"
            )

        if player_id in seen:
            raise ValueError(
                f"duplicate observer ID: {player_id}"
            )

        seen.add(player_id)
        normalized.append(player_id)

    return normalized


def _normalize_subject_mapping(
    value: Any,
    *,
    field_name: str,
    expected_subjects: set[str],
) -> dict[str, Any]:
    """Canonicalize subject keys and require an exact set."""

    if not isinstance(value, Mapping):
        raise TypeError(
            f"{field_name} must be a mapping"
        )

    normalized: dict[str, Any] = {}

    for raw_subject, item in value.items():
        subject = normalize_player(
            raw_subject
        )
        if raw_subject != subject:
            raise ValueError(
                f"{field_name} keys must use canonical player IDs"
            )

        if subject in normalized:
            raise ValueError(
                "duplicate subject after normalization "
                f"in {field_name}: {subject}"
            )

        normalized[subject] = item

    actual_subjects = set(normalized)

    if actual_subjects != expected_subjects:
        missing = sorted(
            expected_subjects - actual_subjects
        )
        extra = sorted(
            actual_subjects - expected_subjects
        )

        raise ValueError(
            f"{field_name} subject set mismatch; "
            f"missing={missing}, extra={extra}"
        )

    return normalized


def _validate_pair_target(
    value: Any,
    *,
    subject: str,
    suspected_werewolves: list[str],
    known_werewolves: list[str],
    known_non_werewolves: list[str],
) -> None:
    """Audit one stored projected target without replacing it."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{subject} pair target must be a sequence")
    if len(value) != NUM_WOLF_PAIR_CLASSES:
        raise ValueError(
            f"{subject} pair target must contain {NUM_WOLF_PAIR_CLASSES} values"
        )
    numeric: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(
                f"{subject} pair target values must be int or float, not bool"
            )
        converted = float(item)
        if not math.isfinite(converted):
            raise ValueError(f"{subject} pair target values must be finite")
        if converted < 0.0:
            raise ValueError(f"{subject} pair target values must be non-negative")
        numeric.append(converted)
    if not math.isclose(sum(numeric), 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"{subject} pair target must sum to one")

    known_wolf_set = set(known_werewolves)
    known_non_wolf_set = set(known_non_werewolves)
    for probability, pair in zip(numeric, canonical_wolf_pairs(), strict=True):
        hard_legal = (
            known_wolf_set.issubset(pair)
            and set(pair).isdisjoint(known_non_wolf_set)
        )
        if hard_legal and probability <= 0.0:
            raise ValueError(
                f"{subject} hard-legal pair probability must be positive"
            )
        if not hard_legal and probability != 0.0:
            raise ValueError(
                f"{subject} hard-illegal pair probability must be zero"
            )

    expected = suspicion_set_to_pair_target(
        suspected_werewolves,
        known_werewolves,
        known_non_werewolves,
        dtype=torch.float64,
    )
    stored = torch.tensor(numeric, dtype=torch.float64)
    if not torch.allclose(stored, expected, rtol=1e-9, atol=1e-12):
        raise ValueError(
            f"{subject} pair target does not match projection_version"
        )
    marginal_sum = 0.0
    for player in PLAYER_NAMES:
        marginal_sum += sum(
            probability
            for probability, pair in zip(
                numeric, canonical_wolf_pairs(), strict=True
            )
            if player in pair
        )
    if not math.isclose(marginal_sum, 2.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"{subject} pair marginal sum must equal two")


def _normalize_sample(
    sample: Any,
) -> dict[str, Any]:
    """Validate one raw subjective sample."""

    if not isinstance(sample, Mapping):
        raise TypeError(
            "each dataset sample must be a mapping"
        )
    schema_version = sample.get("schema_version")
    if schema_version == PLAYER_SUSPICION_SCHEMA_VERSION:
        raise ValueError(
            "raw player-suspicion schema requires an explicit offline "
            "pair projection"
        )
    present_legacy_fields = (
        FORBIDDEN_LEGACY_FIELDS
        & set(sample)
    )

    if present_legacy_fields:
        raise ValueError(
            "legacy or truth-derived sample fields "
            "are not supported: "
            f"{sorted(present_legacy_fields)}"
        )
    if set(sample) != PROJECTED_SAMPLE_FIELDS:
        missing = sorted(PROJECTED_SAMPLE_FIELDS - set(sample))
        extra = sorted(set(sample) - PROJECTED_SAMPLE_FIELDS)
        raise ValueError(
            f"sample field set mismatch; missing={missing}, extra={extra}"
        )

    if schema_version != PROJECTED_SCHEMA_VERSION:
        raise ValueError(
            "unsupported sample schema_version: "
            f"{schema_version!r}"
        )

    public_events = normalize_public_events(sample.get("public_events"))
    if not public_events:
        raise ValueError("public_events cannot be empty")
    sp_actions = public_speech_actions(public_events)

    observer_ids = _normalize_observer_ids(
        sample.get("observer_ids")
    )

    expected_subjects = {
        normalize_player(player_id)
        for player_id in observer_ids
    }

    suspected_werewolves = _normalize_subject_mapping(
        sample.get("suspected_werewolves"),
        field_name="suspected_werewolves",
        expected_subjects=expected_subjects,
    )
    known_werewolves = _normalize_subject_mapping(
        sample.get("known_werewolves"),
        field_name="known_werewolves",
        expected_subjects=expected_subjects,
    )
    known_non_werewolves = _normalize_subject_mapping(
        sample.get("known_non_werewolves"),
        field_name="known_non_werewolves",
        expected_subjects=expected_subjects,
    )

    belief_status = _normalize_subject_mapping(
        sample.get("belief_status"),
        field_name="belief_status",
        expected_subjects=expected_subjects,
    )

    belief_errors = _normalize_subject_mapping(
        sample.get("belief_errors"),
        field_name="belief_errors",
        expected_subjects=expected_subjects,
    )
    agent_backend_ids = _normalize_subject_mapping(
        sample.get("agent_backend_ids"),
        field_name="agent_backend_ids",
        expected_subjects=expected_subjects,
    )
    pair_targets = _normalize_subject_mapping(
        sample.get("pair_targets"),
        field_name="pair_targets",
        expected_subjects=expected_subjects,
    )

    for subject in expected_subjects:
        status = belief_status[subject]
        suspicion = suspected_werewolves[subject]
        subject_known_wolves = known_werewolves[subject]
        subject_known_non_wolves = known_non_werewolves[subject]
        closed_wolves, closed_non_wolves = close_hard_knowledge(
            subject_known_wolves,
            subject_known_non_wolves,
        )
        if (
            subject_known_wolves != closed_wolves
            or subject_known_non_wolves != closed_non_wolves
        ):
            raise ValueError(
                f"{subject} hard knowledge must already be closed"
            )
        error = belief_errors[subject]
        backend_id = agent_backend_ids[subject]
        pair_target = pair_targets[subject]

        if not isinstance(status, str) or not status:
            raise ValueError(
                f"{subject} has an invalid belief status"
            )
        if status not in {
            STATUS_OK,
            STATUS_PARSE_ERROR,
            STATUS_REPORTER_ERROR,
            STATUS_SEMANTIC_ERROR,
        }:
            raise ValueError(f"{subject} has unsupported belief status")

        if (
            error is not None
            and not isinstance(error, str)
        ):
            raise TypeError(
                f"{subject} belief error must be "
                "text or None"
            )

        if status == STATUS_OK:
            if error is not None:
                raise ValueError(
                    f"{subject} has status=ok but belief error is not None"
                )
            if not isinstance(suspicion, list):
                raise ValueError(
                    f"{subject} has status=ok but "
                    "no suspected_werewolves list"
                )
            normalized_suspicion = validate_player_suspicion(
                suspicion,
                subject_known_wolves,
                subject_known_non_wolves,
            )
            if suspicion != normalized_suspicion:
                raise ValueError(
                    f"{subject} suspected_werewolves must be canonically ordered"
                )
            _validate_pair_target(
                pair_target,
                subject=subject,
                suspected_werewolves=suspicion,
                known_werewolves=subject_known_wolves,
                known_non_werewolves=subject_known_non_wolves,
            )
        else:
            if suspicion is not None:
                raise ValueError(
                    f"{subject} has non-ok status "
                    "but suspected_werewolves is not None"
                )
            if pair_target is not None:
                raise ValueError(
                    f"{subject} has non-ok status but pair_targets is not null"
                )
            if not isinstance(error, str) or not error:
                raise ValueError(
                    f"{subject} non-ok status requires an error"
                )

        if not isinstance(backend_id, str) or not backend_id.strip():
            raise ValueError(f"{subject} has invalid agent_backend_id")

    step_idx = sample.get("step_idx")
    if isinstance(step_idx, bool) or not isinstance(step_idx, int) or step_idx < 0:
        raise ValueError("step_idx must be a non-negative integer")
    game_id = sample.get("game_id")
    if not isinstance(game_id, str) or not game_id.strip():
        raise ValueError("game_id must be non-empty text")
    if sample.get("report_trigger") not in REPORT_TRIGGERS:
        raise ValueError("unsupported report_trigger")
    if not isinstance(sample.get("phase"), str) or not sample["phase"]:
        raise ValueError("phase must be non-empty text")
    parse_public_phase(sample["phase"])
    speaker_id = sample.get("speaker_id")
    if isinstance(speaker_id, bool) or not isinstance(speaker_id, int):
        raise TypeError("speaker_id must be an integer")
    if not 1 <= speaker_id <= NUM_PLAYERS:
        raise ValueError("speaker_id must be in [1, 7]")
    if (
        public_events[-1]["event_type"] != "turn_start"
        or public_events[-1]["speaker"] != normalize_player(speaker_id)
    ):
        raise ValueError(
            "pre-speech public_events must end with matching turn_start"
        )
    latest_public_phase = next(
        (
            event["phase"]
            for event in reversed(public_events)
            if event["event_type"] == "phase_change"
        ),
        None,
    )
    if latest_public_phase != sample["phase"]:
        raise ValueError(
            "sample phase must match latest public phase_change"
        )
    label_cutoff = sample.get("label_cutoff_step_idx")
    if (
        isinstance(label_cutoff, bool)
        or not isinstance(label_cutoff, int)
        or label_cutoff != step_idx
    ):
        raise ValueError("label_cutoff_step_idx must equal step_idx")
    public_action_count = sample.get("public_action_count")
    if (
        isinstance(public_action_count, bool)
        or not isinstance(public_action_count, int)
        or public_action_count != len(sp_actions)
    ):
        raise ValueError("public_action_count must equal len(sp_actions)")
    if sample.get("source_schema_version") != PLAYER_SUSPICION_SCHEMA_VERSION:
        raise ValueError("unsupported source_schema_version")
    if sample.get("projection_version") != PROJECTION_VERSION:
        raise ValueError("unsupported projection_version")
    if sample.get("pair_ordering") != PAIR_ORDERING:
        raise ValueError("unsupported pair_ordering")
    if sample.get("label_prompt_version") != LABEL_PROMPT_VERSION:
        raise ValueError("unsupported label_prompt_version")
    if sample.get("label_provenance") != LABEL_PROVENANCE:
        raise ValueError("unsupported label_provenance")
    if (
        sample.get("public_event_schema_version")
        != PUBLIC_EVENT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported public_event_schema_version")
    if sample.get("public_event_digest") != public_event_digest(public_events):
        raise ValueError("public_event_digest does not match public_events")
    if sample.get("structured_input_digest") != (
        structured_input_digest(public_events)
    ):
        raise ValueError(
            "structured_input_digest does not match public_events"
        )
    if sample.get("target_distribution_is_reporter_probability") is not (
        TARGET_DISTRIBUTION_IS_REPORTER_PROBABILITY
    ):
        raise ValueError(
            "target_distribution_is_reporter_probability must be false"
        )
    if (
        sample.get("target_distribution_is_deterministic_encoding")
        is not TARGET_DISTRIBUTION_IS_DETERMINISTIC_ENCODING
    ):
        raise ValueError(
            "target_distribution_is_deterministic_encoding must be true"
        )

    normalized = deepcopy(
        dict(sample)
    )

    normalized["observer_ids"] = (
        observer_ids
    )
    normalized["suspected_werewolves"] = suspected_werewolves
    normalized["known_werewolves"] = known_werewolves
    normalized["known_non_werewolves"] = known_non_werewolves
    normalized["belief_status"] = (
        belief_status
    )
    normalized["belief_errors"] = (
        belief_errors
    )
    normalized["agent_backend_ids"] = agent_backend_ids
    normalized["pair_targets"] = pair_targets
    normalized["public_events"] = public_events
    return normalized


def load_twd_tom_jsonl(
    path: str | Path,
) -> list[dict[str, Any]]:
    """Load projected subjective samples from a JSONL file."""

    input_path = Path(path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"dataset file not found: {input_path}"
        )

    samples: list[dict[str, Any]] = []

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                sample = json.loads(
                    stripped
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "invalid JSON on line "
                    f"{line_number}: {exc}"
                ) from exc

            if not isinstance(sample, dict):
                raise TypeError(
                    "JSONL line "
                    f"{line_number} must contain an object"
                )

            samples.append(sample)

    return samples


class TWDToMDataset(Dataset):
    """PyTorch dataset for subjective ToM samples."""

    def __init__(
        self,
        samples: Sequence[Mapping[str, Any]],
        feature_builder: (
            PublicEventFeatureBuilder | None
        ) = None,
        target_dtype: torch.dtype = torch.float32,
    ):
        if (
            isinstance(samples, (str, bytes))
            or not isinstance(samples, Sequence)
        ):
            raise TypeError(
                "samples must be a sequence"
            )

        self.samples = [
            _normalize_sample(sample)
            for sample in samples
        ]
        last_prefix_by_game: dict[str, list[dict[str, Any]]] = {}
        for sample in self.samples:
            game_id = sample["game_id"]
            current = sample["public_events"]
            previous = last_prefix_by_game.get(game_id)
            if previous is not None and (
                len(current) <= len(previous)
                or current[: len(previous)] != previous
            ):
                raise ValueError(
                    "public_events snapshots must be strictly monotonic prefixes"
                )
            last_prefix_by_game[game_id] = current

        self.feature_builder = (
            PublicEventFeatureBuilder()
            if feature_builder is None
            else feature_builder
        )

        if (
            not isinstance(target_dtype, torch.dtype)
            or not target_dtype.is_floating_point
        ):
            raise TypeError("target_dtype must be a floating-point dtype")
        self.target_dtype = target_dtype

        if not hasattr(
            self.feature_builder,
            "encode_events",
        ):
            raise TypeError(
                "feature_builder must provide "
                "encode_events()"
            )

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        feature_builder: (
            PublicEventFeatureBuilder | None
        ) = None,
        target_dtype: torch.dtype = torch.float32,
    ) -> "TWDToMDataset":
        """Construct a dataset from a JSONL path."""

        return cls(
            load_twd_tom_jsonl(path),
            feature_builder=feature_builder,
            target_dtype=target_dtype,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        sample = self.samples[index]

        features = (
            self.feature_builder.encode_events(
                sample["public_events"]
            )
        )

        pair_targets = torch.zeros(
            (NUM_PLAYERS, NUM_WOLF_PAIR_CLASSES),
            dtype=self.target_dtype,
        )
        subject_mask = torch.zeros(
            NUM_PLAYERS,
            dtype=torch.bool,
        )

        for subject, status in sample[
            "belief_status"
        ].items():
            if status != STATUS_OK:
                continue
            subject_index = PLAYER_TO_ID[subject] - 1
            target = torch.tensor(
                sample["pair_targets"][subject],
                dtype=self.target_dtype,
            )
            pair_targets[subject_index] = target
            subject_mask[subject_index] = True

        metadata = {
            "schema_version": sample[
                "schema_version"
            ],
            "game_id": sample.get(
                "game_id"
            ),
            "step_idx": sample.get(
                "step_idx"
            ),
            "report_trigger": sample.get(
                "report_trigger"
            ),
            "phase": sample.get(
                "phase"
            ),
            "speaker_id": sample.get(
                "speaker_id"
            ),
            "observer_ids": deepcopy(
                sample["observer_ids"]
            ),
            "belief_status": deepcopy(
                sample["belief_status"]
            ),
            "belief_errors": deepcopy(
                sample["belief_errors"]
            ),
            "known_werewolves": deepcopy(sample["known_werewolves"]),
            "known_non_werewolves": deepcopy(sample["known_non_werewolves"]),
            "source_schema_version": sample["source_schema_version"],
            "projection_version": sample["projection_version"],
            "pair_ordering": sample["pair_ordering"],
            "label_provenance": sample["label_provenance"],
            "agent_backend_ids": deepcopy(sample["agent_backend_ids"]),
            "label_cutoff_step_idx": sample[
                "label_cutoff_step_idx"
            ],
            "public_action_count": sample[
                "public_action_count"
            ],
            "public_event_schema_version": sample[
                "public_event_schema_version"
            ],
            "public_event_digest": sample["public_event_digest"],
            "structured_input_digest": sample[
                "structured_input_digest"
            ],
            "label_prompt_version": sample[
                "label_prompt_version"
            ],
        }

        return {
            "subject_ids": features[
                "subject_ids"
            ],
            "action_ids": features[
                "action_ids"
            ],
            "object_ids": features[
                "object_ids"
            ],
            "event_type_ids": features["event_type_ids"],
            "phase_ids": features["phase_ids"],
            "day_values": features["day_values"],
            "attention_mask": features[
                "attention_mask"
            ],
            "pair_targets": (
                pair_targets
            ),
            "subject_mask": subject_mask,
            "metadata": metadata,
        }


def collate_twd_tom_samples(
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Right-pad action sequences and stack labels."""

    if (
        isinstance(batch, (str, bytes))
        or not isinstance(batch, Sequence)
    ):
        raise TypeError(
            "batch must be a sequence"
        )

    if not batch:
        raise ValueError(
            "batch cannot be empty"
        )

    batch_size = len(batch)

    max_length = max(
        item["subject_ids"].shape[0]
        for item in batch
    )

    first = batch[0]

    feature_fields = (
        "subject_ids",
        "action_ids",
        "object_ids",
        "event_type_ids",
        "phase_ids",
        "day_values",
        "attention_mask",
    )
    padded_features = {
        field_name: first[field_name].new_zeros((batch_size, max_length))
        for field_name in feature_fields
    }

    for batch_index, item in enumerate(
        batch
    ):
        length = item[
            "subject_ids"
        ].shape[0]

        expected_shape = (
            length,
        )

        for field_name in feature_fields[1:]:
            if (
                item[field_name].shape
                != expected_shape
            ):
                raise ValueError(
                    "action feature lengths do not "
                    f"match for {field_name}"
                )

        for field_name in feature_fields:
            padded_features[field_name][batch_index, :length] = item[
                field_name
            ]

    metadata_fields = (
        "schema_version",
        "game_id",
        "step_idx",
        "report_trigger",
        "phase",
        "speaker_id",
        "observer_ids",
        "belief_status",
        "belief_errors",
        "known_werewolves",
        "known_non_werewolves",
    )

    metadata = {
        field_name: [
            deepcopy(
                item["metadata"][
                    field_name
                ]
            )
            for item in batch
        ]
        for field_name in metadata_fields
    }

    return {
        **padded_features,
        "pair_targets": torch.stack(
            [
                item["pair_targets"]
                for item in batch
            ]
        ),
        "subject_mask": torch.stack(
            [
                item["subject_mask"]
                for item in batch
            ]
        ),
        "metadata": metadata,
    }


__all__ = [
    "FORBIDDEN_LEGACY_FIELDS",
    "PROJECTED_SAMPLE_FIELDS",
    "load_twd_tom_jsonl",
    "TWDToMDataset",
    "collate_twd_tom_samples",
]
