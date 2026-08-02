"""Read-only seat-bias and public-evidence audit for second-order targets."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

from werewolf.models.twd_tom.belief_labels import (
    pair_probabilities_to_belief_marginals,
)
from werewolf.models.twd_tom.dataset import TWDToMDataset
from werewolf.models.twd_tom.public_events import parse_public_phase
from werewolf.models.twd_tom.schema import PLAYER_NAMES


def _distribution_summary(values: list[float]) -> dict[str, int | float]:
    if not values:
        return {"count": 0, "min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


class _TargetAuditAccumulator:
    def __init__(self) -> None:
        self.valid_row_count = 0
        self.public_evidence_valid_row_count = 0
        self.snapshot_count = 0
        self.valid_rows_by_observer = torch.zeros(7, dtype=torch.int64)
        self.evidence_rows_by_observer = torch.zeros(7, dtype=torch.int64)
        self.marginal_sums = torch.zeros(7, dtype=torch.float64)
        self.top_counts = torch.zeros(7, dtype=torch.int64)
        self.pair_entropies: list[float] = []
        self.marginal_spreads: list[float] = []
        self.observer_pairwise_tv: list[float] = []
        self.evidence_pair_entropies: list[float] = []
        self.evidence_marginal_spreads: list[float] = []
        self.evidence_observer_pairwise_tv: list[float] = []

    def update(self, item: Mapping[str, Any]) -> None:
        targets = item["pair_targets"].to(dtype=torch.float64)
        mask = item["subject_mask"].to(dtype=torch.bool)
        evidence_mask = item["public_evidence_mask"].to(dtype=torch.bool)
        effective_mask = mask & evidence_mask
        valid_targets = targets[mask]
        evidence_targets = targets[effective_mask]
        self.snapshot_count += 1
        self.valid_row_count += valid_targets.shape[0]
        self.public_evidence_valid_row_count += evidence_targets.shape[0]
        self.valid_rows_by_observer += mask.to(dtype=torch.int64)
        self.evidence_rows_by_observer += effective_mask.to(dtype=torch.int64)
        if valid_targets.numel() == 0:
            return
        marginals = pair_probabilities_to_belief_marginals(targets)[mask]
        self.marginal_sums += marginals.sum(dim=0)
        maxima = marginals.max(dim=-1, keepdim=True).values
        self.top_counts += torch.isclose(marginals, maxima).sum(dim=0)
        entropies = -(
            valid_targets
            * valid_targets.clamp_min(torch.finfo(torch.float64).tiny).log()
        ).sum(dim=-1)
        self.pair_entropies.extend(entropies.tolist())
        self.marginal_spreads.extend(
            (marginals.max(dim=-1).values - marginals.min(dim=-1).values).tolist()
        )
        if valid_targets.shape[0] >= 2:
            pairwise_tv = 0.5 * torch.pdist(valid_targets, p=1)
            self.observer_pairwise_tv.append(float(pairwise_tv.mean().item()))
        if evidence_targets.numel():
            evidence_marginals = pair_probabilities_to_belief_marginals(
                targets
            )[effective_mask]
            evidence_entropies = -(
                evidence_targets
                * evidence_targets.clamp_min(
                    torch.finfo(torch.float64).tiny
                ).log()
            ).sum(dim=-1)
            self.evidence_pair_entropies.extend(evidence_entropies.tolist())
            self.evidence_marginal_spreads.extend(
                (
                    evidence_marginals.max(dim=-1).values
                    - evidence_marginals.min(dim=-1).values
                ).tolist()
            )
        if evidence_targets.shape[0] >= 2:
            evidence_pairwise_tv = 0.5 * torch.pdist(evidence_targets, p=1)
            self.evidence_observer_pairwise_tv.append(
                float(evidence_pairwise_tv.mean().item())
            )

    def finalize(self) -> dict[str, Any]:
        if self.valid_row_count:
            marginal_means = self.marginal_sums / self.valid_row_count
        else:
            marginal_means = torch.zeros_like(self.marginal_sums)
        silent_count = (
            self.valid_row_count - self.public_evidence_valid_row_count
        )
        evidence_fraction = (
            self.public_evidence_valid_row_count / self.valid_row_count
            if self.valid_row_count
            else 0.0
        )
        return {
            "snapshot_count": self.snapshot_count,
            "valid_observer_row_count": self.valid_row_count,
            "public_evidence_valid_observer_row_count": (
                self.public_evidence_valid_row_count
            ),
            "silent_observer_row_count": silent_count,
            "silent_observer_row_fraction": 1.0 - evidence_fraction,
            "public_evidence_subject_fraction": evidence_fraction,
            "public_evidence_coverage_by_observer": {
                player: (
                    float(self.evidence_rows_by_observer[index].item())
                    / int(self.valid_rows_by_observer[index].item())
                    if int(self.valid_rows_by_observer[index].item()) > 0
                    else 0.0
                )
                for index, player in enumerate(PLAYER_NAMES)
            },
            "mean_target_marginal_by_player": {
                player: float(marginal_means[index].item())
                for index, player in enumerate(PLAYER_NAMES)
            },
            "target_marginal_max_tie_aware_count_by_player": {
                player: int(self.top_counts[index].item())
                for index, player in enumerate(PLAYER_NAMES)
            },
            "target_pair_entropy": _distribution_summary(self.pair_entropies),
            "target_marginal_spread": _distribution_summary(
                self.marginal_spreads
            ),
            "target_observer_pairwise_tv": _distribution_summary(
                self.observer_pairwise_tv
            ),
            "evidence_conditioned_target_pair_entropy": (
                _distribution_summary(self.evidence_pair_entropies)
            ),
            "evidence_conditioned_target_marginal_spread": (
                _distribution_summary(self.evidence_marginal_spreads)
            ),
            "evidence_conditioned_target_observer_pairwise_tv": (
                _distribution_summary(self.evidence_observer_pairwise_tv)
            ),
            "absolute_player_marginal_mean_gap": float(
                (marginal_means.max() - marginal_means.min()).item()
            ),
        }


def summarize_items(items: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    overall = _TargetAuditAccumulator()
    grouped: dict[str, _TargetAuditAccumulator] = {}
    for item in items:
        overall.update(item)
        phase = item["metadata"]["phase"]
        day, _ = parse_public_phase(phase)
        group_name = f"day={day}|phase={phase}"
        grouped.setdefault(group_name, _TargetAuditAccumulator()).update(item)
    return {
        "overall": overall.finalize(),
        "by_day_and_phase": {
            name: accumulator.finalize()
            for name, accumulator in sorted(grouped.items())
        },
    }


def audit_training_targets(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    seed: int,
) -> dict[str, Any]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    train = TWDToMDataset.from_jsonl(train_path, tom_order=2)
    validation = TWDToMDataset.from_jsonl(validation_path, tom_order=2)
    augmented = TWDToMDataset.from_jsonl(
        train_path,
        tom_order=2,
        enable_cyclic_rotation=True,
        augmentation_seed=seed,
    )

    def augmented_orbit() -> Iterable[Mapping[str, Any]]:
        for epoch in range(7):
            augmented.set_epoch(epoch)
            for index in range(len(augmented)):
                yield augmented[index]

    result = {
        "tom_order": 2,
        "seed": seed,
        "augmentation_epochs": list(range(7)),
        "train_before_augmentation": summarize_items(
            train[index] for index in range(len(train))
        ),
        "train_after_seven_epoch_rotation_orbit": summarize_items(
            augmented_orbit()
        ),
        "validation_without_augmentation": summarize_items(
            validation[index] for index in range(len(validation))
        ),
    }
    if not all(
        math.isfinite(value)
        for section in (
            result["train_before_augmentation"],
            result["train_after_seven_epoch_rotation_orbit"],
            result["validation_without_augmentation"],
        )
        for value in section["overall"]["mean_target_marginal_by_player"].values()
    ):
        raise RuntimeError("target audit produced non-finite player marginals")
    for split_name in (
        "train_before_augmentation",
        "validation_without_augmentation",
    ):
        fraction = result[split_name]["overall"][
            "public_evidence_subject_fraction"
        ]
        if fraction < 0.05:
            raise RuntimeError(
                f"{split_name} public-evidence coverage is abnormally low: "
                f"{fraction:.6f}"
            )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit second-order ToM target seat bias without writing data."
    )
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = audit_training_targets(
        train_path=args.train,
        validation_path=args.validation,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
