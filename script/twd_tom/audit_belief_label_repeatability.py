"""Audit repeated Annotation V2 labels on exactly identical frozen states."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from werewolf.models.twd_tom.annotation_v2 import (
    annotation_set_digest,
    load_belief_v2_annotations,
)


REPEATABILITY_SCHEMA_VERSION = "classic7_belief_v2_repeatability_audit_v1"
_FROZEN_STATE_FIELDS = (
    "game_id",
    "step_idx",
    "observer",
    "observer_role",
    "current_speaker",
    "is_current_speaker",
    "phase",
    "day",
    "public_action_count",
    "public_event_digest",
    "hard_knowledge",
    "information_boundary",
)


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_state(record: Mapping[str, Any]) -> dict[str, Any]:
    return {field: record[field] for field in _FROZEN_STATE_FIELDS}


def _label(record: Mapping[str, Any]) -> tuple[bool, set[str], list[float]]:
    recommendation = record["training_recommendation"]
    distribution = recommendation["compat_relative_suspicion_distribution"]
    return (
        recommendation["distribution_loss_mask"],
        set(recommendation["compat_suspected_werewolves"]),
        [distribution[f"player{i}"] for i in range(1, 8)],
    )


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def _kl(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(
        probability * math.log(probability / right[index])
        for index, probability in enumerate(left)
        if probability > 0.0
    )


def _js(left: Sequence[float], right: Sequence[float]) -> float:
    midpoint = [(a + b) / 2.0 for a, b in zip(left, right, strict=True)]
    return 0.5 * _kl(left, midpoint) + 0.5 * _kl(right, midpoint)


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("repeatability stratum cannot be empty")
    pair_count = sum(row["pair_count"] for row in rows)
    observed_pair_count = sum(row["observed_pair_count"] for row in rows)
    return {
        "state_count": len(rows),
        "pair_count": pair_count,
        "observed_pair_count": observed_pair_count,
        "all_replicates_exact_support_agreement_rate": mean(
            float(row["all_replicates_exact_support_agreement"]) for row in rows
        ),
        "pairwise_observation_mask_agreement_rate": (
            sum(row["observation_mask_agreement_sum"] for row in rows)
            / pair_count
        ),
        "pairwise_exact_support_agreement_rate": (
            sum(row["exact_support_agreement_sum"] for row in rows)
            / pair_count
        ),
        "mean_pairwise_jaccard": (
            sum(row["jaccard_sum"] for row in rows) / pair_count
        ),
        "mean_pairwise_total_variation_observed_pairs": (
            sum(row["total_variation_sum"] for row in rows)
            / observed_pair_count
            if observed_pair_count
            else None
        ),
        "mean_pairwise_jensen_shannon_observed_pairs": (
            sum(row["jensen_shannon_sum"] for row in rows)
            / observed_pair_count
            if observed_pair_count
            else None
        ),
    }


def _atomic_json_write(value: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def audit_belief_label_repeatability(
    *,
    replicate_paths: Sequence[str | Path],
    output_path: str | Path,
    per_state_jsonl_path: str | Path,
    per_state_csv_path: str | Path,
) -> dict[str, Any]:
    """Measure label stability without treating abstention as a distribution."""

    if not 3 <= len(replicate_paths) <= 5:
        raise ValueError("repeatability audit requires 3 to 5 replicates")
    paths = [Path(path).resolve() for path in replicate_paths]
    if len(set(paths)) != len(paths):
        raise ValueError("replicate paths must be distinct")
    replicates = [load_belief_v2_annotations(path) for path in paths]
    reference_keys = set(replicates[0])
    if not reference_keys:
        raise ValueError("repeatability replicate cannot be empty")
    for index, replicate in enumerate(replicates[1:], start=2):
        if set(replicate) != reference_keys:
            raise ValueError(
                f"replicate {index} does not contain the same frozen-state keys"
            )

    per_state = []
    pair_indices = list(combinations(range(len(replicates)), 2))
    for key in sorted(reference_keys):
        records = [replicate[key] for replicate in replicates]
        state = _frozen_state(records[0])
        if any(_frozen_state(record) != state for record in records[1:]):
            raise ValueError(f"replicate frozen-state mismatch for key={key}")
        labels = [_label(record) for record in records]
        supports = [label[1] for label in labels]
        pair_metrics = []
        for left_index, right_index in pair_indices:
            left_observed, left_support, left_distribution = labels[left_index]
            right_observed, right_support, right_distribution = labels[right_index]
            both_observed = left_observed and right_observed
            pair_metrics.append({
                "observation_mask_agreement": left_observed == right_observed,
                "exact_support_agreement": left_support == right_support,
                "jaccard": _jaccard(left_support, right_support),
                "total_variation": (
                    0.5 * sum(
                        abs(a - b)
                        for a, b in zip(
                            left_distribution,
                            right_distribution,
                            strict=True,
                        )
                    )
                    if both_observed
                    else None
                ),
                "jensen_shannon": (
                    _js(left_distribution, right_distribution)
                    if both_observed
                    else None
                ),
            })
        observed_pairs = [
            metric for metric in pair_metrics
            if metric["total_variation"] is not None
        ]
        reference_support_size = len(supports[0])
        per_state.append({
            "schema_version": REPEATABILITY_SCHEMA_VERSION,
            "frozen_state_digest": _canonical_digest(state),
            "game_id": state["game_id"],
            "step_idx": state["step_idx"],
            "observer": state["observer"],
            "observer_role": state["observer_role"],
            "day": state["day"],
            "phase": state["phase"],
            "reference_support_size": reference_support_size,
            "replicate_count": len(replicates),
            "pair_count": len(pair_metrics),
            "observed_pair_count": len(observed_pairs),
            "all_replicates_exact_support_agreement": all(
                support == supports[0] for support in supports[1:]
            ),
            "observation_mask_agreement_sum": sum(
                metric["observation_mask_agreement"] for metric in pair_metrics
            ),
            "exact_support_agreement_sum": sum(
                metric["exact_support_agreement"] for metric in pair_metrics
            ),
            "jaccard_sum": sum(metric["jaccard"] for metric in pair_metrics),
            "total_variation_sum": sum(
                metric["total_variation"] for metric in observed_pairs
            ),
            "jensen_shannon_sum": sum(
                metric["jensen_shannon"] for metric in observed_pairs
            ),
        })

    strata: dict[str, dict[str, Any]] = {}
    for field_name in ("observer_role", "day", "reference_support_size"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in per_state:
            groups[str(row[field_name])].append(row)
        strata[field_name] = {
            value: _summarize(rows) for value, rows in sorted(groups.items())
        }
    result = {
        "schema_version": REPEATABILITY_SCHEMA_VERSION,
        "status": "PASS",
        "replicate_count": len(replicates),
        "state_count": len(per_state),
        "distribution_metric_policy": (
            "TV_and_JS_only_when_both_replicates_have_distribution_loss_mask_true"
        ),
        "empty_label_policy": "abstain_is_unobserved_never_uniform_imputed",
        "replicates": [
            {
                "replicate_index": index,
                "path": str(path),
                "sha256": _sha256(path),
                "annotation_set_digest": annotation_set_digest(records),
            }
            for index, (path, records) in enumerate(
                zip(paths, replicates, strict=True),
                start=1,
            )
        ],
        "overall": _summarize(per_state),
        "stratified": strata,
        "per_state_jsonl_path": str(Path(per_state_jsonl_path)),
        "per_state_csv_path": str(Path(per_state_csv_path)),
    }
    output = Path(output_path)
    jsonl = Path(per_state_jsonl_path)
    csv_path = Path(per_state_csv_path)
    for path in (output, jsonl, csv_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json_write(result, output)
    _atomic_jsonl_write(per_state, jsonl)
    _atomic_csv_write(per_state, csv_path)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit 3-5 independent V2 labels of identical frozen states."
    )
    parser.add_argument("--replicate", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-state-jsonl", required=True)
    parser.add_argument("--per-state-csv", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = audit_belief_label_repeatability(
        replicate_paths=args.replicate,
        output_path=args.output,
        per_state_jsonl_path=args.per_state_jsonl,
        per_state_csv_path=args.per_state_csv,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
