"""Read-only seat-bias and latest-public-action audit for ToM2 targets."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

from werewolf.models.twd_tom.belief_labels import (
    pair_probabilities_to_belief_marginals,
)
from werewolf.models.twd_tom.dataset import (
    TWDToMDataset,
    second_order_effective_subject_mask,
)
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
        self.update_valid_row_count = 0
        self.snapshot_count = 0
        self.latest_action_snapshot_count = 0
        self.update_valid_snapshot_count = 0
        self.update_rows_by_actor = torch.zeros(7, dtype=torch.int64)
        self.valid_observer_count_distribution: Counter[int] = Counter()
        self.latest_action_type_counts: Counter[str] = Counter()
        self.multi_actor_speech_examples: list[dict[str, Any]] = []
        self.marginal_sums = torch.zeros(7, dtype=torch.float64)
        self.top_counts = torch.zeros(7, dtype=torch.int64)
        self.pair_entropies: list[float] = []
        self.marginal_spreads: list[float] = []
        self.observer_pairwise_tv: list[float] = []
        self.update_pair_entropies: list[float] = []
        self.update_marginal_spreads: list[float] = []
        self.update_observer_pairwise_tv: list[float] = []

    def update(self, item: Mapping[str, Any]) -> None:
        targets = item["pair_targets"].to(dtype=torch.float64)
        mask = item["subject_mask"]
        update_mask = item["latest_completed_public_action_mask"]
        effective_mask = second_order_effective_subject_mask(
            mask,
            update_mask,
        )
        valid_targets = targets[mask]
        update_targets = targets[effective_mask]
        actor_ids = item["metadata"][
            "latest_completed_public_action_actor_ids"
        ]
        action_type = item["metadata"][
            "latest_completed_public_action_type"
        ]
        self.snapshot_count += 1
        self.valid_row_count += valid_targets.shape[0]
        self.update_valid_row_count += update_targets.shape[0]
        effective_count = int(effective_mask.sum().item())
        self.valid_observer_count_distribution[effective_count] += 1
        self.update_rows_by_actor += effective_mask.to(dtype=torch.int64)
        if actor_ids:
            self.latest_action_snapshot_count += 1
        if effective_count:
            self.update_valid_snapshot_count += 1
        self.latest_action_type_counts[action_type or "none"] += 1
        if action_type == "public_speech" and len(actor_ids) > 1:
            self.multi_actor_speech_examples.append(
                {
                    "game_id": item["metadata"]["game_id"],
                    "step_idx": item["metadata"]["step_idx"],
                    "actor_ids": list(actor_ids),
                    "reason": "speech_action_subject_differs_from_speaker",
                }
            )
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
        if update_targets.numel():
            update_marginals = pair_probabilities_to_belief_marginals(
                targets
            )[effective_mask]
            update_entropies = -(
                update_targets
                * update_targets.clamp_min(
                    torch.finfo(torch.float64).tiny
                ).log()
            ).sum(dim=-1)
            self.update_pair_entropies.extend(update_entropies.tolist())
            self.update_marginal_spreads.extend(
                (
                    update_marginals.max(dim=-1).values
                    - update_marginals.min(dim=-1).values
                ).tolist()
            )
        if update_targets.shape[0] >= 2:
            update_pairwise_tv = 0.5 * torch.pdist(update_targets, p=1)
            self.update_observer_pairwise_tv.append(
                float(update_pairwise_tv.mean().item())
            )

    def finalize(self) -> dict[str, Any]:
        if self.valid_row_count:
            marginal_means = self.marginal_sums / self.valid_row_count
        else:
            marginal_means = torch.zeros_like(self.marginal_sums)
        latest_action_fraction = (
            self.latest_action_snapshot_count / self.snapshot_count
            if self.snapshot_count
            else 0.0
        )
        update_valid_fraction = (
            self.update_valid_snapshot_count / self.snapshot_count
            if self.snapshot_count
            else 0.0
        )
        return {
            "snapshot_count": self.snapshot_count,
            "valid_observer_row_count": self.valid_row_count,
            "latest_completed_public_action_snapshot_count": (
                self.latest_action_snapshot_count
            ),
            "no_latest_completed_public_action_snapshot_count": (
                self.snapshot_count - self.latest_action_snapshot_count
            ),
            "latest_completed_public_action_snapshot_fraction": (
                latest_action_fraction
            ),
            "update_valid_snapshot_count": self.update_valid_snapshot_count,
            "filtered_snapshot_count": (
                self.snapshot_count - self.update_valid_snapshot_count
            ),
            "latest_action_snapshot_fraction": update_valid_fraction,
            "update_valid_observer_row_count": self.update_valid_row_count,
            "valid_observer_count_per_snapshot_distribution": {
                str(count): frequency
                for count, frequency in sorted(
                    self.valid_observer_count_distribution.items()
                )
            },
            "latest_completed_public_action_type_counts": dict(
                sorted(self.latest_action_type_counts.items())
            ),
            "update_rows_by_actor_id": {
                player: int(self.update_rows_by_actor[index].item())
                for index, player in enumerate(PLAYER_NAMES)
            },
            "multi_actor_speech_snapshot_count": len(
                self.multi_actor_speech_examples
            ),
            "multi_actor_speech_examples": self.multi_actor_speech_examples,
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
            "update_conditioned_target_pair_entropy": (
                _distribution_summary(self.update_pair_entropies)
            ),
            "update_conditioned_target_marginal_spread": (
                _distribution_summary(self.update_marginal_spreads)
            ),
            "update_conditioned_target_observer_pairwise_tv": (
                _distribution_summary(self.update_observer_pairwise_tv)
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
            "latest_action_snapshot_fraction"
        ]
        if fraction < 0.05:
            raise RuntimeError(
                f"{split_name} latest-action coverage is abnormally low: "
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
