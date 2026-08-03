from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from werewolf.models.twd_tom.belief_labels import (
    suspicion_set_to_pair_target,
)
from werewolf.models.twd_tom.samples import (
    SAMPLE_FIELDS,
    SAMPLE_SCHEMA_VERSION as PLAYER_SUSPICION_SCHEMA_VERSION,
    freeze_public_snapshot,
    make_twd_tom_sample,
)
from werewolf.models.twd_tom.schema import (
    LABEL_PROMPT_VERSION,
    LABEL_PROVENANCE,
    PAIR_ORDERING,
    PROJECTED_SCHEMA_VERSION,
    PROJECTION_VERSION,
    TARGET_DISTRIBUTION_IS_DETERMINISTIC_ENCODING,
    TARGET_DISTRIBUTION_IS_REPORTER_PROBABILITY,
    normalize_player,
)


PROJECTED_SAMPLE_FIELDS = frozenset(
    SAMPLE_FIELDS
    | {
        "source_schema_version",
        "projection_version",
        "pair_ordering",
        "pair_targets",
        "target_distribution_is_reporter_probability",
        "target_distribution_is_deterministic_encoding",
    }
)


def _require_subject_mapping(
    sample: Mapping[str, Any],
    field_name: str,
    expected_subjects: set[str],
) -> Mapping[str, Any]:
    value = sample.get(field_name)
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    if set(value) != expected_subjects:
        raise ValueError(f"{field_name} observer keys must match observer_ids")
    return value


def validate_raw_suspicion_sample(sample: Any) -> dict[str, Any]:
    """Strictly validate one serialized online suspicion sample."""

    if not isinstance(sample, Mapping):
        raise TypeError("each raw sample must be a mapping")
    if set(sample) != SAMPLE_FIELDS:
        missing = sorted(SAMPLE_FIELDS - set(sample))
        extra = sorted(set(sample) - SAMPLE_FIELDS)
        raise ValueError(
            f"raw sample field set mismatch; missing={missing}, extra={extra}"
        )
    if sample.get("schema_version") != PLAYER_SUSPICION_SCHEMA_VERSION:
        raise ValueError("unsupported raw schema_version")
    if sample.get("label_prompt_version") != LABEL_PROMPT_VERSION:
        raise ValueError("unsupported label_prompt_version")
    if sample.get("label_provenance") != LABEL_PROVENANCE:
        raise ValueError("unsupported label_provenance")

    snapshot = freeze_public_snapshot(
        game_id=sample.get("game_id"),
        step_idx=sample.get("step_idx"),
        phase=sample.get("phase"),
        speaker_id=sample.get("speaker_id"),
        report_trigger=sample.get("report_trigger"),
        observer_ids=sample.get("observer_ids"),
        public_events=sample.get("public_events"),
    )
    label_cutoff = sample.get("label_cutoff_step_idx")
    if (
        isinstance(label_cutoff, bool)
        or not isinstance(label_cutoff, int)
        or label_cutoff != snapshot.label_cutoff_step_idx
    ):
        raise ValueError("label_cutoff_step_idx must equal step_idx")
    public_action_count = sample.get("public_action_count")
    if (
        isinstance(public_action_count, bool)
        or not isinstance(public_action_count, int)
        or public_action_count != snapshot.public_action_count
    ):
        raise ValueError("public_action_count must equal len(sp_actions)")
    if sample.get("public_event_digest") != snapshot.public_event_digest:
        raise ValueError("public_event_digest does not match public_events")
    if sample.get("structured_input_digest") != (
        snapshot.structured_input_digest
    ):
        raise ValueError(
            "structured_input_digest does not match public_events"
        )

    expected_subjects = {
        normalize_player(observer)
        for observer in snapshot.observer_ids
    }
    suspicions = _require_subject_mapping(
        sample, "suspected_werewolves", expected_subjects
    )
    known_werewolves = _require_subject_mapping(
        sample, "known_werewolves", expected_subjects
    )
    known_non_werewolves = _require_subject_mapping(
        sample, "known_non_werewolves", expected_subjects
    )
    statuses = _require_subject_mapping(
        sample, "belief_status", expected_subjects
    )
    errors = _require_subject_mapping(
        sample, "belief_errors", expected_subjects
    )
    backend_ids = _require_subject_mapping(
        sample, "agent_backend_ids", expected_subjects
    )

    reports = {
        subject: {
            "status": statuses[subject],
            "suspected_werewolves": suspicions[subject],
            "known_werewolves": known_werewolves[subject],
            "known_non_werewolves": known_non_werewolves[subject],
            "error": errors[subject],
            "agent_backend_id": backend_ids[subject],
        }
        for subject in expected_subjects
    }
    normalized = make_twd_tom_sample(
        public_snapshot=snapshot,
        reports=reports,
    )
    if normalized != dict(sample):
        raise ValueError("raw sample must use the canonical serialized form")
    return deepcopy(normalized)


def project_suspicion_sample(sample: Any) -> dict[str, Any]:
    """Project one validated raw row without changing its public metadata."""

    projected = validate_raw_suspicion_sample(sample)
    projected["source_schema_version"] = projected["schema_version"]
    projected["schema_version"] = PROJECTED_SCHEMA_VERSION
    projected["projection_version"] = PROJECTION_VERSION
    projected["pair_ordering"] = PAIR_ORDERING
    projected["pair_targets"] = {}
    for subject, status in projected["belief_status"].items():
        if status != "ok":
            projected["pair_targets"][subject] = None
            continue
        target = suspicion_set_to_pair_target(
            projected["suspected_werewolves"][subject],
            projected["known_werewolves"][subject],
            projected["known_non_werewolves"][subject],
            dtype=torch.float64,
        )
        projected["pair_targets"][subject] = target.tolist()
    projected["target_distribution_is_reporter_probability"] = (
        TARGET_DISTRIBUTION_IS_REPORTER_PROBABILITY
    )
    projected["target_distribution_is_deterministic_encoding"] = (
        TARGET_DISTRIBUTION_IS_DETERMINISTIC_ENCODING
    )
    if set(projected) != PROJECTED_SAMPLE_FIELDS:
        raise RuntimeError("projected sample field set is inconsistent")
    return projected


def project_jsonl(input_path: str | Path, output_path: str | Path) -> int:
    """Project a raw JSONL file atomically, preserving row order."""

    source = Path(input_path)
    destination = Path(output_path)
    if source.resolve() == destination.resolve():
        raise ValueError("input and output paths must differ")
    if not source.is_file():
        raise FileNotFoundError(f"input file not found: {source}")
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")

    projected_rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL line at {line_number}")
            try:
                raw_sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            try:
                projected_rows.append(project_suspicion_sample(raw_sample))
            except (TypeError, ValueError) as exc:
                raise type(exc)(f"line {line_number}: {exc}") from exc
    if not projected_rows:
        raise ValueError("input dataset is empty")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in projected_rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                )
                handle.write("\n")
        temporary.replace(destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return len(projected_rows)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Project raw classic7 suspicion JSONL into 21-pair targets."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    count = project_jsonl(args.input, args.output)
    print(
        json.dumps(
            {
                "status": "ok",
                "input": str(Path(args.input)),
                "output": str(Path(args.output)),
                "record_count": count,
                "schema_version": PROJECTED_SCHEMA_VERSION,
                "projection_version": PROJECTION_VERSION,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
