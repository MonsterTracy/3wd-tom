"""Evaluate fixed non-training baselines for subjective ToM beliefs.

Only explicit, disjoint train and validation datasets are accepted. The
script never reads a test split, role truth, or speech actions when forming
predictions. All computations run on CPU.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from script.twd_tom.train import (
    MetricAccumulator,
    load_explicit_dataset_partition,
)
from werewolf.models.twd_tom.dataset import (
    TWDToMDataset,
    collate_twd_tom_samples,
)
from werewolf.models.twd_tom.belief_labels import (
    pair_probabilities_to_belief_marginals,
)
from werewolf.models.twd_tom.losses import (
    masked_pair_kl_divergence,
)
from werewolf.models.twd_tom.schema import (
    MARGINAL_SEMANTICS,
    NUMERIC_ANNOTATION_PRESENT,
    NUM_WOLF_PAIR_CLASSES,
    NUM_PLAYERS,
    PAIR_ORDERING,
    PROJECTED_SCHEMA_VERSION as SAMPLE_SCHEMA_VERSION,
    PROJECTION_VERSION,
    RAW_LABEL_FIELD,
    RAW_LABEL_SEMANTICS,
    RAW_LABEL_TYPE,
    TARGET_ENCODING,
    TARGET_INTERPRETATION,
)


BASELINE_DTYPE = torch.float64
PROBABILITY_FLOOR = torch.finfo(BASELINE_DTYPE).tiny


@dataclass(frozen=True)
class BaselineConfig:
    """Configuration for fixed train/validation baseline evaluation."""

    train_dataset_path: str
    validation_dataset_path: str
    output_path: str
    batch_size: int = 32
    num_workers: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "train_dataset_path",
            "validation_dataset_path",
            "output_path",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{field_name} must be a non-empty string"
                )

        if (
            Path(self.train_dataset_path).resolve()
            == Path(self.validation_dataset_path).resolve()
        ):
            raise ValueError(
                "train_dataset_path and validation_dataset_path "
                "must be different files"
            )

        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")

        if (
            isinstance(self.num_workers, bool)
            or not isinstance(self.num_workers, int)
            or self.num_workers < 0
        ):
            raise ValueError(
                "num_workers must be a non-negative integer"
            )


def build_uniform_pair_probabilities() -> torch.Tensor:
    """Return the fixed ``[7, 21]`` global-world uniform baseline."""

    return torch.full(
        (NUM_PLAYERS, NUM_WOLF_PAIR_CLASSES),
        1.0 / NUM_WOLF_PAIR_CLASSES,
        dtype=BASELINE_DTYPE,
    )


def probabilities_to_finite_logits(
    probabilities: torch.Tensor,
) -> torch.Tensor:
    """Convert normalized probabilities to finite CPU logits."""

    if not isinstance(probabilities, torch.Tensor):
        raise TypeError("probabilities must be a tensor")

    expected_shape = (
        NUM_PLAYERS,
        NUM_WOLF_PAIR_CLASSES,
    )
    if tuple(probabilities.shape) != expected_shape:
        raise ValueError(
            f"probabilities must have shape {expected_shape}"
        )

    normalized = probabilities.to(
        device="cpu",
        dtype=BASELINE_DTYPE,
    )

    if not torch.isfinite(normalized).all():
        raise ValueError("probabilities must be finite")
    if torch.any(normalized < 0).item():
        raise ValueError("probabilities must be non-negative")

    row_sums = normalized.sum(dim=-1)
    if not torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("probability rows must sum to one")

    logits = normalized.clamp_min(
        PROBABILITY_FLOOR
    ).log()

    if not torch.isfinite(logits).all():
        raise RuntimeError("baseline logits must be finite")

    recovered = torch.softmax(logits, dim=-1)
    if not torch.allclose(
        recovered,
        normalized,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError(
            "finite logits did not recover baseline probabilities"
        )

    return logits


@torch.no_grad()
def fit_observer_empirical_pair_prior(
    train_loader: DataLoader,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average every observer's valid training targets."""

    target_sums = torch.zeros(
        (NUM_PLAYERS, NUM_WOLF_PAIR_CLASSES),
        dtype=BASELINE_DTYPE,
    )
    valid_counts = torch.zeros(
        NUM_PLAYERS,
        dtype=torch.long,
    )

    for batch in train_loader:
        targets = batch["pair_targets"].to(
            dtype=BASELINE_DTYPE
        )
        valid_mask = batch["subject_mask"]

        target_sums += (
            targets
            * valid_mask.unsqueeze(-1).to(
                dtype=BASELINE_DTYPE
            )
        ).sum(dim=0)
        valid_counts += valid_mask.sum(dim=0)

    missing_observers = torch.nonzero(
        valid_counts == 0,
        as_tuple=False,
    ).flatten()
    if missing_observers.numel() > 0:
        player_ids = [
            int(index.item()) + 1
            for index in missing_observers
        ]
        raise ValueError(
            "every observer requires at least one valid training row; "
            f"missing observers: {player_ids}"
        )

    prior = target_sums / valid_counts.to(
        dtype=BASELINE_DTYPE
    ).unsqueeze(-1)
    prior = prior / prior.sum(
        dim=-1,
        keepdim=True,
    )

    return prior, valid_counts


@torch.no_grad()
def evaluate_fixed_pair_baseline(
    data_loader: DataLoader,
    probabilities: torch.Tensor,
) -> dict[str, int | float]:
    """Evaluate one fixed pair distribution with existing metrics."""

    baseline_logits = probabilities_to_finite_logits(
        probabilities
    )
    accumulator = MetricAccumulator()

    for batch in data_loader:
        batch_size = batch["pair_targets"].shape[0]
        pair_logits = baseline_logits.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )
        pair_targets = batch["pair_targets"].to(
            dtype=BASELINE_DTYPE
        )
        subject_mask = batch["subject_mask"]

        loss = masked_pair_kl_divergence(
            pair_logits,
            pair_targets,
            subject_mask,
        )
        accumulator.update(
            loss=loss,
            pair_logits=pair_logits,
            pair_targets=pair_targets,
            subject_mask=subject_mask,
        )

    return accumulator.finalize()


def _build_loader(
    samples: list[Mapping[str, Any]],
    *,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = TWDToMDataset(
        samples,
        target_dtype=BASELINE_DTYPE,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_twd_tom_samples,
    )


def _atomic_json_dump(
    value: Mapping[str, Any],
    path: Path,
) -> None:
    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )
    temporary_path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _dataset_audit(samples: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize valid rows and raw suspicion-set sizes without fitting."""

    histogram = {str(size): 0 for size in range(7)}
    valid = 0
    for sample in samples:
        for observer in sample["observer_ids"]:
            subject = f"player{observer}"
            if sample["belief_status"][subject] != "ok":
                continue
            suspicion = sample["suspected_werewolves"][subject]
            suspicion_size = len(suspicion)
            valid += 1
            histogram[str(suspicion_size)] += 1
    return {
        "raw_record_count": len(samples),
        "valid_subject_count": valid,
        "suspicion_set_size_histogram": histogram,
    }


def run_baselines(
    config: BaselineConfig,
) -> dict[str, Any]:
    """Fit train-only priors and evaluate both fixed baselines."""

    train_path = Path(
        config.train_dataset_path
    ).resolve()
    validation_path = Path(
        config.validation_dataset_path
    ).resolve()

    partition = load_explicit_dataset_partition(
        train_dataset_path=train_path,
        validation_dataset_path=validation_path,
    )

    train_loader = _build_loader(
        partition.train_samples,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    validation_loader = _build_loader(
        partition.validation_samples,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    uniform = build_uniform_pair_probabilities()
    prior, fit_counts = fit_observer_empirical_pair_prior(
        train_loader
    )
    uniform_marginals = pair_probabilities_to_belief_marginals(
        uniform
    )
    prior_marginals = pair_probabilities_to_belief_marginals(
        prior
    )

    summary: dict[str, Any] = {
        "status": "ok",
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "target_encoding": TARGET_ENCODING,
        "projection_version": PROJECTION_VERSION,
        "pair_class_count": NUM_WOLF_PAIR_CLASSES,
        "pair_ordering": PAIR_ORDERING,
        "raw_label_field": RAW_LABEL_FIELD,
        "raw_label_type": RAW_LABEL_TYPE,
        "numeric_annotation_present": NUMERIC_ANNOTATION_PRESENT,
        "raw_label_semantics": RAW_LABEL_SEMANTICS,
        "target_interpretation": TARGET_INTERPRETATION,
        "marginal_semantics": MARGINAL_SEMANTICS,
        "train_dataset_path": str(train_path),
        "validation_dataset_path": str(validation_path),
        "train_sample_count": len(partition.train_samples),
        "validation_sample_count": len(
            partition.validation_samples
        ),
        "train_game_ids": list(partition.train_game_ids),
        "validation_game_ids": list(
            partition.validation_game_ids
        ),
        "game_id_overlap": [],
        "probability_floor": PROBABILITY_FLOOR,
        "train_data_audit": _dataset_audit(partition.train_samples),
        "validation_data_audit": _dataset_audit(
            partition.validation_samples
        ),
        "baselines": {
            "uniform_pair": {
                "pair_distribution": uniform.tolist(),
                "derived_marginal_matrix": uniform_marginals.tolist(),
                "train_metrics": evaluate_fixed_pair_baseline(
                    train_loader,
                    uniform,
                ),
                "validation_metrics": evaluate_fixed_pair_baseline(
                    validation_loader,
                    uniform,
                ),
            },
            "observer_empirical_pair_prior": {
                "fit_valid_subject_counts": fit_counts.tolist(),
                "pair_prior_distribution": prior.tolist(),
                "derived_marginal_matrix": prior_marginals.tolist(),
                "train_metrics": evaluate_fixed_pair_baseline(
                    train_loader,
                    prior,
                ),
                "validation_metrics": evaluate_fixed_pair_baseline(
                    validation_loader,
                    prior,
                ),
            },
        },
    }

    _atomic_json_dump(
        summary,
        Path(config.output_path).resolve(),
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate fixed subjective ToM baselines on explicit "
            "train and validation datasets."
        )
    )
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--validation-dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = run_baselines(
        BaselineConfig(
            train_dataset_path=args.train_dataset,
            validation_dataset_path=args.validation_dataset,
            output_path=args.output,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    )
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
