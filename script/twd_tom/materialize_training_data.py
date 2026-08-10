"""Materialize strict V2.7 ToM1/ToM2 samples from raw belief snapshots."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from script.twd_tom.project_suspicion_to_pairs import (
    validate_raw_suspicion_sample,
)
from werewolf.models.twd_tom.belief_labels import (
    close_hard_knowledge,
    suspicion_set_to_pair_target,
)
from werewolf.models.twd_tom.dataset import (
    PRIVATE_FIELDS_USAGE,
    TOM_INPUT_SCOPES,
    TWDToMDataset,
)
from werewolf.models.twd_tom.schema import (
    DETERMINISTIC_HARD_KNOWLEDGE_ANNOTATION_CONFIDENCE,
    DETERMINISTIC_HARD_KNOWLEDGE_OBSERVER_PROVENANCE,
    FORMAL_ANNOTATION_SCHEMA_VERSION,
    FORMAL_LABEL_PROVENANCE,
    FORMALIZATION_POLICY_VERSION,
    SOURCE_REPORT_ANNOTATION_CONFIDENCE,
    SOURCE_REPORT_OBSERVER_PROVENANCE,
)
from werewolf.speech.private_belief_perceiver import (
    STATUS_OK,
    STATUS_SEMANTIC_ERROR,
)


RAW_SUBJECT_MAPPING_FIELDS = (
    "suspected_werewolves",
    "known_werewolves",
    "known_non_werewolves",
    "belief_status",
    "belief_errors",
    "agent_backend_ids",
)


def _load_raw_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"raw input not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL line at {line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON on line {line_number}: {exc}"
                ) from exc
            try:
                records.append(validate_raw_suspicion_sample(value))
            except (TypeError, ValueError) as exc:
                raise type(exc)(f"line {line_number}: {exc}") from exc
    if not records:
        raise ValueError("raw input cannot be empty")
    return records


def _observer_decision(
    sample: dict[str, Any], subject: str
) -> dict[str, Any]:
    status = sample["belief_status"][subject]
    if status == STATUS_OK:
        return {
            "kind": "source",
            "suspected_werewolves": deepcopy(
                sample["suspected_werewolves"][subject]
            ),
            "provenance": SOURCE_REPORT_OBSERVER_PROVENANCE,
            "confidence": SOURCE_REPORT_ANNOTATION_CONFIDENCE,
            "compatible_pair_count": None,
        }

    compatible_pair_count = None
    closed_wolves: list[str] | None = None
    if status == STATUS_SEMANTIC_ERROR:
        known_wolves = sample["known_werewolves"][subject]
        known_non_wolves = sample["known_non_werewolves"][subject]
        closed_wolves, _closed_non_wolves = close_hard_knowledge(
            known_wolves,
            known_non_wolves,
        )
        target = suspicion_set_to_pair_target(
            closed_wolves,
            known_wolves,
            known_non_wolves,
            dtype=torch.float64,
        )
        compatible_pair_count = int(torch.count_nonzero(target).item())

    if compatible_pair_count == 1:
        return {
            "kind": "deterministic_hard_knowledge",
            "suspected_werewolves": closed_wolves,
            "provenance": (
                DETERMINISTIC_HARD_KNOWLEDGE_OBSERVER_PROVENANCE
            ),
            "confidence": (
                DETERMINISTIC_HARD_KNOWLEDGE_ANNOTATION_CONFIDENCE
            ),
            "compatible_pair_count": 1,
        }
    return {
        "kind": "unavailable",
        "suspected_werewolves": None,
        "provenance": None,
        "confidence": None,
        "compatible_pair_count": compatible_pair_count,
    }


def _formal_sample(
    raw: dict[str, Any],
    *,
    tom_order: int,
    observer_ids: list[int],
    decisions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    subjects = [f"player{observer_id}" for observer_id in observer_ids]
    formal = deepcopy(raw)
    formal["observer_ids"] = list(observer_ids)
    for field_name in RAW_SUBJECT_MAPPING_FIELDS:
        formal[field_name] = {
            subject: deepcopy(raw[field_name][subject])
            for subject in subjects
        }

    formal["source_belief_status"] = {
        subject: deepcopy(raw["belief_status"][subject])
        for subject in subjects
    }
    formal["source_belief_errors"] = {
        subject: deepcopy(raw["belief_errors"][subject])
        for subject in subjects
    }
    formal["observer_label_provenance"] = {
        subject: decisions[subject]["provenance"] for subject in subjects
    }
    formal["observer_annotation_confidence"] = {
        subject: decisions[subject]["confidence"] for subject in subjects
    }
    for subject in subjects:
        if decisions[subject]["kind"] == "deterministic_hard_knowledge":
            formal["belief_status"][subject] = STATUS_OK
            formal["belief_errors"][subject] = None
            formal["suspected_werewolves"][subject] = deepcopy(
                decisions[subject]["suspected_werewolves"]
            )

    formal["source_schema_version"] = raw["schema_version"]
    formal["annotation_schema_version"] = FORMAL_ANNOTATION_SCHEMA_VERSION
    formal["label_provenance"] = FORMAL_LABEL_PROVENANCE
    formal["source_label_provenance"] = raw["label_provenance"]
    formal["tom_order"] = tom_order
    formal["model_input_scope"] = TOM_INPUT_SCOPES[tom_order]
    formal["private_fields_usage"] = PRIVATE_FIELDS_USAGE[tom_order]
    formal["current_action_used"] = False
    formal["expert_labels_used_as_later_evidence"] = False
    formal["future_information_used"] = False
    return formal


def materialize_training_records(
    raw_records: list[dict[str, Any]],
) -> dict[str, Any]:
    tom1_records: list[dict[str, Any]] = []
    tom2_records: list[dict[str, Any]] = []
    removed_tom1_snapshot_keys: list[dict[str, Any]] = []
    filtered_tom2_observer_keys: list[dict[str, Any]] = []
    semantic_errors = 0
    hard_knowledge_recovered = 0
    unresolved = 0

    for raw_value in raw_records:
        raw = validate_raw_suspicion_sample(raw_value)
        decisions: dict[str, dict[str, Any]] = {}
        for observer_id in raw["observer_ids"]:
            subject = f"player{observer_id}"
            decision = _observer_decision(raw, subject)
            decisions[subject] = decision
            if raw["belief_status"][subject] == STATUS_SEMANTIC_ERROR:
                semantic_errors += 1
            if decision["kind"] == "deterministic_hard_knowledge":
                hard_knowledge_recovered += 1
            elif decision["kind"] == "unavailable":
                unresolved += 1
                filtered_tom2_observer_keys.append(
                    {
                        "game_id": raw["game_id"],
                        "step_idx": raw["step_idx"],
                        "phase": raw["phase"],
                        "observer_id": observer_id,
                    }
                )

        speaker_id = raw["speaker_id"]
        speaker_subject = f"player{speaker_id}"
        if decisions[speaker_subject]["kind"] == "unavailable":
            removed_tom1_snapshot_keys.append(
                {
                    "game_id": raw["game_id"],
                    "step_idx": raw["step_idx"],
                    "phase": raw["phase"],
                    "speaker_id": speaker_id,
                }
            )
        else:
            tom1_records.append(
                _formal_sample(
                    raw,
                    tom_order=1,
                    observer_ids=[speaker_id],
                    decisions=decisions,
                )
            )

        tom2_observers = [
            observer_id
            for observer_id in raw["observer_ids"]
            if decisions[f"player{observer_id}"]["kind"] != "unavailable"
        ]
        if tom2_observers:
            tom2_records.append(
                _formal_sample(
                    raw,
                    tom_order=2,
                    observer_ids=tom2_observers,
                    decisions=decisions,
                )
            )

    TWDToMDataset(tom1_records, tom_order=1)
    TWDToMDataset(tom2_records, tom_order=2)
    return {
        "policy_version": FORMALIZATION_POLICY_VERSION,
        "input_snapshot_count": len(raw_records),
        "semantic_error_observer_count": semantic_errors,
        "hard_knowledge_recovered_count": hard_knowledge_recovered,
        "unresolved_observer_count": unresolved,
        "removed_tom1_snapshot_keys": removed_tom1_snapshot_keys,
        "filtered_tom2_observer_keys": filtered_tom2_observer_keys,
        "tom1_records": tom1_records,
        "tom2_records": tom2_records,
    }


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"output already exists: {path}") from exc
        temporary.unlink()
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def materialize_training_data(
    *,
    raw_path: str | Path,
    tom1_output_path: str | Path,
    tom2_output_path: str | Path,
) -> dict[str, Any]:
    source = Path(raw_path)
    tom1_output = Path(tom1_output_path)
    tom2_output = Path(tom2_output_path)
    if tom1_output.resolve() == tom2_output.resolve():
        raise ValueError("ToM1 and ToM2 outputs must differ")
    existing = [str(path) for path in (tom1_output, tom2_output) if path.exists()]
    if existing:
        raise FileExistsError(f"output files already exist: {existing}")

    raw_records = _load_raw_jsonl(source)
    result = materialize_training_records(raw_records)
    created: list[Path] = []
    try:
        write_jsonl_atomic(tom1_output, result["tom1_records"])
        created.append(tom1_output)
        write_jsonl_atomic(tom2_output, result["tom2_records"])
        created.append(tom2_output)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return {
        key: value
        for key, value in result.items()
        if key not in {"tom1_records", "tom2_records"}
    } | {
        "raw_tom_row_count": len(result["tom1_records"]),
        "raw_tom2_row_count": len(result["tom2_records"]),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize strict V2.7 ToM1 and ToM2 JSONL files."
    )
    parser.add_argument("--raw", required=True)
    parser.add_argument("--tom1-output", required=True)
    parser.add_argument("--tom2-output", required=True)
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    result = materialize_training_data(
        raw_path=args.raw,
        tom1_output_path=args.tom1_output,
        tom2_output_path=args.tom2_output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
