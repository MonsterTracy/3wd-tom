"""Export causally bound validation rows with the largest prediction errors."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from script.twd_tom.train import (
    TrainingConfig,
    _forward_batch,
    _move_batch_to_device,
    build_data_loader,
    resolve_device,
)
from werewolf.models.twd_tom.annotation_v2 import (
    V2_ANNOTATION_SOURCE,
    apply_speech_v2_to_sample,
    load_speech_v2_annotations,
)
from werewolf.models.twd_tom.checkpoint import (
    build_model_from_checkpoint,
    load_checkpoint,
)
from werewolf.models.twd_tom.dataset import load_twd_tom_jsonl
from werewolf.models.twd_tom.losses import masked_belief_probabilities
from werewolf.models.twd_tom.public_events import (
    completed_pre_speech_public_events,
)
from werewolf.models.twd_tom.schema import NUM_PLAYERS


WORST_CASE_SCHEMA_VERSION = "classic7_tom_v2_worst_case_v2"
_JSON_FIELDS = {
    "public_history",
    "speech_annotations",
    "legacy_v1_target",
    "v1_empty_unobserved_target",
    "v2_target",
    "model_prediction",
    "previous_boundary_target",
    "next_boundary_target",
}


def _atomic_jsonl_write(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_csv_write(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("worst-case CSV requires at least one row")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    key: (
                        json.dumps(value, ensure_ascii=False, sort_keys=True)
                        if key in _JSON_FIELDS
                        else value
                    )
                    for key, value in row.items()
                })
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _selected_public_samples(
    config: TrainingConfig,
) -> dict[tuple[str, int], dict[str, Any]]:
    samples = load_twd_tom_jsonl(config.resolved_validation_dataset_path)
    if config.speech_annotation_source == V2_ANNOTATION_SOURCE:
        path = config.resolved_speech_v2_annotation_path
        if path is None:
            raise ValueError("V2 speech worst-case export requires its sidecar")
        records = load_speech_v2_annotations(path)
        samples = [apply_speech_v2_to_sample(sample, records) for sample in samples]
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for sample in samples:
        key = (sample["game_id"], sample["step_idx"])
        if key in result:
            raise ValueError(f"duplicate validation PRE sample: {key}")
        result[key] = sample
    return result


def _row_kl(target: torch.Tensor, prediction: torch.Tensor) -> float:
    positive = target > 0
    return float(
        (
            target[positive]
            * (target[positive].log() - prediction[positive].log())
        ).sum().item()
    )


@torch.no_grad()
def export_belief_worst_cases(
    *,
    config: TrainingConfig,
    checkpoint_path: str | Path,
    output_jsonl: str | Path,
    output_csv: str | Path,
    limit: int = 50,
) -> dict[str, Any]:
    """Rank supervised validation observer-rows without changing evaluation."""

    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if not config.dense_supervision:
        raise ValueError("worst-case export currently requires dense supervision")
    device = resolve_device(config.device)
    checkpoint = load_checkpoint(checkpoint_path)
    model = build_model_from_checkpoint(checkpoint, device=device).eval()
    loader, _ = build_data_loader(
        config,
        dataset_path=config.resolved_validation_dataset_path,
        shuffle=False,
    )
    public_samples = _selected_public_samples(config)
    rows: list[dict[str, Any]] = []
    for raw_batch in loader:
        batch = _move_batch_to_device(raw_batch, device)
        logits = _forward_batch(model, batch)["belief_logits"]
        probabilities = masked_belief_probabilities(
            logits.flatten(0, 1),
            batch["diagonal_target_mask"].flatten(0, 1),
        ).view_as(logits)
        for batch_index, metadata in enumerate(raw_batch["metadata"]):
            valid_boundaries = int(
                raw_batch["boundary_valid_mask"][batch_index].sum().item()
            )
            roles = metadata["observer_roles"]
            for boundary_index in range(valid_boundaries):
                game_id = metadata["game_id"]
                step_idx = metadata["step_idx"][boundary_index]
                sample = public_samples[(game_id, step_idx)]
                selected_target = raw_batch["belief_targets"][
                    batch_index, boundary_index
                ]
                legacy_v1_target = raw_batch["legacy_v1_belief_targets"][
                    batch_index, boundary_index
                ]
                v1_empty_unobserved_target = raw_batch[
                    "v1_empty_unobserved_belief_targets"
                ][
                    batch_index, boundary_index
                ]
                v2_targets = raw_batch.get("v2_belief_targets")
                for observer_index in range(NUM_PLAYERS):
                    alive = bool(raw_batch["observer_alive_mask"][
                        batch_index, boundary_index, observer_index
                    ])
                    if not alive:
                        continue
                    prediction = probabilities[
                        batch_index, boundary_index, observer_index
                    ].detach().cpu()
                    target = selected_target[observer_index]
                    observed = bool(raw_batch["label_observed_mask"][
                        batch_index, boundary_index, observer_index
                    ])
                    supervised = bool(raw_batch["observer_supervision_mask"][
                        batch_index, boundary_index, observer_index
                    ])
                    max_error = (
                        float(torch.max(torch.abs(target - prediction)).item())
                        if observed
                        else None
                    )
                    row = {
                        "schema_version": WORST_CASE_SCHEMA_VERSION,
                        "game_id": game_id,
                        "step_idx": step_idx,
                        "boundary_index": boundary_index,
                        "phase": metadata["phase"][boundary_index],
                        "day": metadata["day"][boundary_index],
                        "public_action_count": metadata[
                            "public_action_count"
                        ][boundary_index],
                        "observer": f"player{observer_index + 1}",
                        "observer_role": (
                            None if roles is None else roles[observer_index]
                        ),
                        "current_speaker": (
                            f"player{metadata['speaker_id'][boundary_index]}"
                        ),
                        "is_current_speaker": bool(metadata[
                            "speaker_vs_non_speaker"
                        ][boundary_index][observer_index]),
                        "supervision_scope": config.supervision_scope,
                        "speech_annotation_source": (
                            config.speech_annotation_source
                        ),
                        "belief_annotation_source": (
                            config.belief_annotation_source
                        ),
                        "label_observed": observed,
                        "supervised": supervised,
                        "support_size": int((target > 0).sum().item()),
                        "max_probability_error": max_error,
                        "kl_divergence": (
                            _row_kl(target, prediction) if observed else None
                        ),
                        "legacy_v1_target": (
                            legacy_v1_target[observer_index].tolist()
                        ),
                        "v1_empty_unobserved_target": (
                            v1_empty_unobserved_target[
                                observer_index
                            ].tolist()
                        ),
                        "v2_target": (
                            None
                            if v2_targets is None
                            else v2_targets[
                                batch_index, boundary_index, observer_index
                            ].tolist()
                        ),
                        "model_prediction": prediction.tolist(),
                        "previous_boundary_target": None,
                        "previous_boundary_label_observed": None,
                        "next_boundary_target": None,
                        "next_boundary_label_observed": None,
                        "public_history": deepcopy(
                            completed_pre_speech_public_events(
                                sample["public_events"],
                                speaker_id=sample["speaker_id"],
                            )
                        ),
                        "speech_annotations": deepcopy(
                            sample["speech_annotations"]
                        ),
                        "_selected_target": target.tolist(),
                    }
                    rows.append(row)

    by_observer: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_observer[(row["game_id"], row["observer"])].append(row)
    for group in by_observer.values():
        group.sort(key=lambda row: (row["step_idx"], row["boundary_index"]))
        for index, row in enumerate(group):
            if index > 0:
                previous = group[index - 1]
                row["previous_boundary_target"] = previous["_selected_target"]
                row["previous_boundary_label_observed"] = previous[
                    "label_observed"
                ]
            if index + 1 < len(group):
                following = group[index + 1]
                row["next_boundary_target"] = following["_selected_target"]
                row["next_boundary_label_observed"] = following[
                    "label_observed"
                ]
    candidates = [row for row in rows if row["supervised"]]
    candidates.sort(
        key=lambda row: (
            -float(row["max_probability_error"]),
            row["game_id"],
            row["step_idx"],
            row["observer"],
        )
    )
    selected = candidates[:limit]
    for rank, row in enumerate(selected, start=1):
        row.pop("_selected_target")
        row["error_rank"] = rank
    if not selected or not math.isfinite(
        float(selected[0]["max_probability_error"])
    ):
        raise ValueError("worst-case export found no finite supervised errors")
    jsonl_path = Path(output_jsonl)
    csv_path = Path(output_csv)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl_write(selected, jsonl_path)
    _atomic_csv_write(selected, csv_path)
    return {
        "schema_version": WORST_CASE_SCHEMA_VERSION,
        "candidate_row_count": len(candidates),
        "exported_row_count": len(selected),
        "max_probability_error": selected[0]["max_probability_error"],
        "output_jsonl": str(jsonl_path),
        "output_csv": str(csv_path),
    }


def aggregate_worst_case_exports(
    *,
    input_jsonl_paths: Sequence[str | Path],
    output_jsonl: str | Path,
    output_csv: str | Path,
    limit: int = 50,
) -> dict[str, Any]:
    """Merge independently ranked fold exports into one deterministic OOF list."""

    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    rows = []
    for raw_path in input_jsonl_paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"fold worst-case export not found: {path}")
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                not isinstance(row, dict)
                or row.get("schema_version") != WORST_CASE_SCHEMA_VERSION
            ):
                raise ValueError(
                    f"invalid worst-case row at {path}:{line_number}"
                )
            rows.append(row)
    rows.sort(key=lambda row: (
        -float(row["max_probability_error"]),
        row["game_id"],
        row["step_idx"],
        row["observer"],
    ))
    selected = rows[:limit]
    for rank, row in enumerate(selected, start=1):
        row["error_rank"] = rank
    if not selected:
        raise ValueError("cannot aggregate empty worst-case exports")
    jsonl_path = Path(output_jsonl)
    csv_path = Path(output_csv)
    _atomic_jsonl_write(selected, jsonl_path)
    _atomic_csv_write(selected, csv_path)
    return {
        "schema_version": WORST_CASE_SCHEMA_VERSION,
        "fold_candidate_row_count": len(rows),
        "exported_row_count": len(selected),
        "max_probability_error": selected[0]["max_probability_error"],
        "output_jsonl": str(jsonl_path),
        "output_csv": str(csv_path),
    }


__all__ = [
    "WORST_CASE_SCHEMA_VERSION",
    "aggregate_worst_case_exports",
    "export_belief_worst_cases",
]
