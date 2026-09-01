"""Audit game-level dense PRE supervision before model training."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from script.twd_tom.materialize_canonical_belief_dataset import (
    validate_split_manifest,
)
from werewolf.models.twd_tom.action_features import PublicEventFeatureBuilder
from werewolf.models.twd_tom.dense_dataset import (
    DENSE_SUPERVISION_VERSION,
    DenseTWDToMDataset,
)
from werewolf.trajectory import canonical_digest, canonical_json


DENSE_AUDIT_SCHEMA_VERSION = "classic7_tom_v2_dense_pre_audit_v2"
AUDITABLE_SPLITS = ("train", "validation")


def _positive_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _statistics(values: list[int]) -> dict[str, int | float]:
    if not values:
        raise ValueError("dense audit statistics require at least one value")
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(canonical_json(value) + "\n")


def audit_dense_belief_dataset(
    *,
    dataset_path: str | Path,
    split_name: str,
    max_seq_len: int = 256,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Prove that every materialized target has one causal PRE boundary."""

    if split_name not in AUDITABLE_SPLITS:
        raise ValueError(f"split_name must be one of {AUDITABLE_SPLITS}")
    max_seq_len = _positive_integer(max_seq_len, field_name="max_seq_len")
    resolved_path = Path(dataset_path).resolve()
    manifest = validate_split_manifest(
        resolved_path.parent / "split_manifest.json",
        verify_split_files=(split_name,),
    )
    expected_path = (
        resolved_path.parent
        / manifest["output_files"][split_name]["relative_path"]
    ).resolve()
    if resolved_path != expected_path:
        raise ValueError(f"dataset path is not the manifest {split_name} split")
    dataset = DenseTWDToMDataset.from_jsonl(
        resolved_path,
        feature_builder=PublicEventFeatureBuilder(max_seq_len=max_seq_len),
    )

    boundary_counts: list[int] = []
    final_sequence_lengths: list[int] = []
    alive_observer_counts: list[int] = []
    supervised_observer_counts: list[int] = []
    phase_counts: Counter[str] = Counter()
    game_ids: list[str] = []
    for item_index in range(len(dataset)):
        item = dataset[item_index]
        metadata = item["metadata"]
        boundary_count = int(item["boundary_valid_mask"].sum().item())
        sequence_length = int(item["attention_mask"].sum().item())
        if boundary_count != len(metadata["step_idx"]):
            raise RuntimeError("dense boundary and step counts differ")
        if item["boundary_indices"][-1].item() != sequence_length - 1:
            raise RuntimeError("final dense boundary must end at the final PRE token")
        if boundary_count != item["belief_targets"].shape[0]:
            raise RuntimeError("dense boundary and target counts differ")
        game_ids.append(metadata["game_id"])
        boundary_counts.append(boundary_count)
        final_sequence_lengths.append(sequence_length)
        alive_observer_counts.append(
            int(item["observer_alive_mask"].sum().item())
        )
        supervised_observer_counts.append(
            int(item["observer_supervision_mask"].sum().item())
        )
        phase_counts.update(metadata["phase"])

    report = {
        "schema_version": DENSE_AUDIT_SCHEMA_VERSION,
        "status": "PASS",
        "split_name": split_name,
        "dataset_path": str(resolved_path),
        "dataset_sha256": manifest["output_files"][split_name]["sha256"],
        "split_manifest_digest": manifest["manifest_digest"],
        "canonical_batch_summary_digest": manifest[
            "canonical_batch_summary_digest"
        ],
        "training_supervision": DENSE_SUPERVISION_VERSION,
        "target_semantics": dataset.target_semantics,
        "target_conversion": dataset.target_conversion,
        "label_observation_semantics": dataset.label_observation_semantics,
        "belief_annotation_source": dataset.belief_annotation_source,
        "max_seq_len": max_seq_len,
        "game_count": len(dataset),
        "boundary_count": dataset.boundary_count,
        "alive_observer_count": sum(alive_observer_counts),
        "supervised_observer_count": sum(supervised_observer_counts),
        "boundaries_per_game": _statistics(boundary_counts),
        "final_sequence_length": _statistics(final_sequence_lengths),
        "alive_observers_per_game": _statistics(alive_observer_counts),
        "supervised_observers_per_game": _statistics(
            supervised_observer_counts
        ),
        "phase_boundary_counts": dict(sorted(phase_counts.items())),
        "game_ids": game_ids,
        "causal_contract": {
            "target_time": "strict_pre_speech",
            "boundary_index_semantics": "inclusive_last_visible_token",
            "future_tokens_visible": False,
            "terminal_turn_start_visible": False,
            "prefix_relation": "exact_encoded_prefix",
        },
    }
    report["audit_digest"] = canonical_digest(report)
    if output_path is not None:
        _write_json_new(Path(output_path).resolve(), report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit dense strict-PRE belief supervision."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-name", required=True, choices=AUDITABLE_SPLITS)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--output")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    report = audit_dense_belief_dataset(
        dataset_path=args.dataset,
        split_name=args.split_name,
        max_seq_len=args.max_seq_len,
        output_path=args.output,
    )
    print(canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
