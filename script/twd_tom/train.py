"""Train the classic-seven observer-specific ToM belief backbone.

The formal training path is intentionally limited to:

    audited train JSONL + audited validation JSONL
        -> TWDToMDataset
        -> subject/action/object sequences
        -> ToMBeliefBackbone
        -> masked subjective-pair KL divergence
        -> subjective-pair metrics

Training and validation files are supplied explicitly. The trainer never
creates another random split and rejects any overlapping ``game_id`` values.
The test split is deliberately absent from this entry point.

The trainer never accepts true roles or truth-derived Werewolf labels.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from werewolf.models.twd_tom.action_features import (
    PublicEventFeatureBuilder,
)
from werewolf.models.twd_tom.belief_backbone import (
    BACKBONE_NAME,
    ToMBeliefBackbone,
    ToMBeliefBackboneConfig,
)
from werewolf.models.twd_tom.dataset import (
    TWDToMDataset,
    collate_twd_tom_samples,
    load_twd_tom_jsonl,
)
from werewolf.models.twd_tom.losses import (
    masked_pair_kl_divergence,
)
from werewolf.models.twd_tom.metrics import (
    compute_subjective_pair_metrics,
)
from werewolf.models.twd_tom.public_events import (
    PHASE_TO_ID,
    PUBLIC_EVENT_SCHEMA_VERSION,
    STRUCTURED_TOKEN_TO_ID,
)
from werewolf.models.twd_tom.schema import (
    GLOBAL_TRUTH_INJECTED,
    LABEL_CONTEXT_SCOPE,
    LABEL_SOURCE,
    MARGINAL_SEMANTICS,
    MODEL_OUTPUT,
    MODEL_INPUT_SCOPE,
    NUMERIC_ANNOTATION_PRESENT,
    NUM_WOLF_PAIR_CLASSES,
    OTHER_PLAYERS_PRIVATE_INFORMATION_VISIBLE,
    OBSERVER_SELECTION,
    OUTPUT_ACTIVATION,
    PAIR_ORDERING,
    PROJECTED_SCHEMA_VERSION as SAMPLE_SCHEMA_VERSION,
    PROJECTION_VERSION,
    PRIVATE_CONTEXT_SERIALIZED,
    REPORT_CONTEXT_MODE,
    REPORT_SIDE_EFFECT_FREE,
    REPORT_TIMING,
    RAW_LABEL_FIELD,
    RAW_LABEL_SEMANTICS,
    RAW_LABEL_TYPE,
    SUPERVISION_SCOPE,
    TARGET_ENCODING,
    TARGET_INTERPRETATION,
    TRUTH_BASED_OBSERVER_SELECTION,
)


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for one explicit-split subjective ToM training run."""

    output_dir: str = ""
    train_dataset_path: str | None = None
    validation_dataset_path: str | None = None

    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.0

    d_model: int = 128
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1
    max_seq_len: int = 256
    dim_feedforward: int | None = None

    def __post_init__(self) -> None:
        train_supplied = self.train_dataset_path is not None
        validation_supplied = (
            self.validation_dataset_path is not None
        )

        if train_supplied and validation_supplied:
            _require_non_empty_string(
                self.train_dataset_path,
                field_name="train_dataset_path",
            )
            _require_non_empty_string(
                self.validation_dataset_path,
                field_name="validation_dataset_path",
            )

            if (
                Path(self.train_dataset_path).resolve()
                == Path(self.validation_dataset_path).resolve()
            ):
                raise ValueError(
                    "train_dataset_path and validation_dataset_path "
                    "must be different files"
                )
        else:
            raise ValueError(
                "provide both train_dataset_path and validation_dataset_path"
            )

        _require_non_empty_string(
            self.output_dir,
            field_name="output_dir",
        )

        _require_positive_integer(
            self.epochs,
            field_name="epochs",
        )
        _require_positive_integer(
            self.batch_size,
            field_name="batch_size",
        )
        _require_positive_number(
            self.learning_rate,
            field_name="learning_rate",
        )
        _require_non_negative_number(
            self.weight_decay,
            field_name="weight_decay",
        )

        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError(
                "seed must be a non-negative integer"
            )

        _require_non_empty_string(
            self.device,
            field_name="device",
        )

        if (
            isinstance(self.num_workers, bool)
            or not isinstance(self.num_workers, int)
            or self.num_workers < 0
        ):
            raise ValueError(
                "num_workers must be a non-negative integer"
            )

        _require_non_negative_number(
            self.gradient_clip_norm,
            field_name="gradient_clip_norm",
        )
        _require_positive_integer(
            self.early_stopping_patience,
            field_name="early_stopping_patience",
        )
        _require_non_negative_number(
            self.early_stopping_min_delta,
            field_name="early_stopping_min_delta",
        )

        _require_positive_integer(
            self.d_model,
            field_name="d_model",
        )
        _require_positive_integer(
            self.n_head,
            field_name="n_head",
        )
        _require_positive_integer(
            self.n_layer,
            field_name="n_layer",
        )
        _require_positive_integer(
            self.max_seq_len,
            field_name="max_seq_len",
        )

        if self.d_model % self.n_head != 0:
            raise ValueError(
                "d_model must be divisible by n_head"
            )

        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not 0.0 <= float(self.dropout) < 1.0
        ):
            raise ValueError(
                "dropout must be in [0, 1)"
            )

        if self.dim_feedforward is not None:
            _require_positive_integer(
                self.dim_feedforward,
                field_name="dim_feedforward",
            )


@dataclass(frozen=True)
class ExplicitDatasetPartition:
    """Explicit train/validation samples and their disjoint game IDs."""

    train_samples: list[dict[str, Any]]
    validation_samples: list[dict[str, Any]]
    train_game_ids: tuple[str, ...]
    validation_game_ids: tuple[str, ...]


def _require_non_empty_string(
    value: Any,
    *,
    field_name: str,
) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string"
        )


def _require_positive_integer(
    value: Any,
    *,
    field_name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(
            f"{field_name} must be a positive integer"
        )


def _require_positive_number(
    value: Any,
    *,
    field_name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or float(value) <= 0.0
    ):
        raise ValueError(
            f"{field_name} must be positive"
        )


def _require_non_negative_number(
    value: Any,
    *,
    field_name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or float(value) < 0.0
    ):
        raise ValueError(
            f"{field_name} cannot be negative"
        )


def set_random_seed(seed: int) -> None:
    """Seed Python and PyTorch random generators."""

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(
    requested_device: str,
) -> torch.device:
    """Resolve ``auto``, CPU, CUDA or MPS devices."""

    normalized = requested_device.strip().lower()

    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        mps_backend = getattr(
            torch.backends,
            "mps",
            None,
        )

        if (
            mps_backend is not None
            and mps_backend.is_available()
        ):
            return torch.device("mps")

        return torch.device("cpu")

    device = torch.device(requested_device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available"
        )

    if device.type == "mps":
        mps_backend = getattr(
            torch.backends,
            "mps",
            None,
        )

        if (
            mps_backend is None
            or not mps_backend.is_available()
        ):
            raise RuntimeError(
                "MPS was requested but is not available"
            )

    return device


def _require_game_id(
    sample: Mapping[str, Any],
    *,
    dataset_name: str,
    sample_index: int,
) -> str:
    """Return a validated non-empty game ID."""

    game_id = sample.get("game_id")

    if not isinstance(game_id, str) or not game_id.strip():
        raise ValueError(
            f"every {dataset_name} sample must contain a "
            "non-empty game_id; "
            f"sample index {sample_index} is invalid"
        )

    return game_id


def _collect_game_ids(
    samples: Sequence[Mapping[str, Any]],
    *,
    dataset_name: str,
) -> tuple[str, ...]:
    if (
        isinstance(samples, (str, bytes))
        or not isinstance(samples, Sequence)
    ):
        raise TypeError(
            f"{dataset_name} samples must be a sequence"
        )

    if not samples:
        raise ValueError(
            f"{dataset_name} dataset cannot be empty"
        )

    return tuple(
        sorted(
            {
                _require_game_id(
                    sample,
                    dataset_name=dataset_name,
                    sample_index=index,
                )
                for index, sample in enumerate(samples)
            }
        )
    )


def load_explicit_dataset_partition(
    *,
    train_dataset_path: str | Path,
    validation_dataset_path: str | Path,
) -> ExplicitDatasetPartition:
    """Load fixed train/validation JSONL files and reject game leakage."""

    train_samples = [
        dict(sample)
        for sample in load_twd_tom_jsonl(
            train_dataset_path
        )
    ]
    validation_samples = [
        dict(sample)
        for sample in load_twd_tom_jsonl(
            validation_dataset_path
        )
    ]

    train_game_ids = _collect_game_ids(
        train_samples,
        dataset_name="training",
    )
    validation_game_ids = _collect_game_ids(
        validation_samples,
        dataset_name="validation",
    )

    overlap = set(train_game_ids) & set(validation_game_ids)

    if overlap:
        raise ValueError(
            "train and validation datasets contain overlapping "
            f"game_id values: {sorted(overlap)}"
        )

    return ExplicitDatasetPartition(
        train_samples=train_samples,
        validation_samples=validation_samples,
        train_game_ids=train_game_ids,
        validation_game_ids=validation_game_ids,
    )


def build_model(
    config: TrainingConfig,
) -> ToMBeliefBackbone:
    """Construct the subjective ToM model."""

    backbone_config = ToMBeliefBackboneConfig(
        d_model=config.d_model,
        n_head=config.n_head,
        n_layer=config.n_layer,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len,
        dim_feedforward=config.dim_feedforward,
    )

    return ToMBeliefBackbone(backbone_config)


def build_data_loaders(
    config: TrainingConfig,
) -> tuple[
    DataLoader,
    DataLoader,
    ExplicitDatasetPartition,
]:
    """Construct loaders from explicit, already-split JSONL files."""

    partition = load_explicit_dataset_partition(
        train_dataset_path=config.train_dataset_path,
        validation_dataset_path=config.validation_dataset_path,
    )

    feature_builder = PublicEventFeatureBuilder(
        max_seq_len=config.max_seq_len
    )

    train_dataset = TWDToMDataset(
        partition.train_samples,
        feature_builder=feature_builder,
    )
    validation_dataset = TWDToMDataset(
        partition.validation_samples,
        feature_builder=feature_builder,
    )

    generator = torch.Generator().manual_seed(
        config.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_twd_tom_samples,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_twd_tom_samples,
    )

    return (
        train_loader,
        validation_loader,
        partition,
    )


def count_supervised_subjects(
    data_loader: DataLoader,
) -> int:
    """Count all valid subjective target rows."""

    total = 0

    for batch in data_loader:
        total += int(
            batch["subject_mask"].sum().item()
        )

    return total


def _move_batch_to_device(
    batch: Mapping[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move only model and label tensors to a device."""

    tensor_fields = (
        "subject_ids",
        "action_ids",
        "object_ids",
        "event_type_ids",
        "phase_ids",
        "day_values",
        "attention_mask",
        "pair_targets",
        "subject_mask",
    )

    return {
        field_name: batch[field_name].to(device)
        for field_name in tensor_fields
    }


class MetricAccumulator:
    """Aggregate row-weighted metrics over batches."""

    def __init__(self) -> None:
        self.valid_subject_count = 0
        self.loss_sum = 0.0
        self.valid_weighted_sums = {
            "mean_pair_kl_divergence": 0.0,
            "mean_pair_cross_entropy": 0.0,
            "mean_pair_total_variation": 0.0,
            "mean_marginal_mae": 0.0,
            "mean_marginal_row_sum_error": 0.0,
            "mean_predicted_diagonal_marginal": 0.0,
            "mean_target_diagonal_marginal": 0.0,
        }

    def update(
        self,
        *,
        loss: torch.Tensor,
        pair_logits: torch.Tensor,
        pair_targets: torch.Tensor,
        subject_mask: torch.Tensor,
    ) -> None:
        metrics = compute_subjective_pair_metrics(
            pair_logits,
            pair_targets,
            subject_mask,
        )

        valid_count = int(
            metrics["valid_subject_count"]
        )

        self.valid_subject_count += valid_count
        self.loss_sum += (
            float(loss.detach().item())
            * valid_count
        )

        for field_name in self.valid_weighted_sums:
            self.valid_weighted_sums[field_name] += (
                float(metrics[field_name])
                * valid_count
            )


    def finalize(self) -> dict[str, int | float]:
        valid_count = self.valid_subject_count

        result: dict[str, int | float] = {
            "valid_subject_count": valid_count,
            "mean_loss": (
                self.loss_sum / valid_count
                if valid_count
                else 0.0
            ),
        }

        for field_name, weighted_sum in (
            self.valid_weighted_sums.items()
        ):
            result[field_name] = (
                weighted_sum / valid_count
                if valid_count
                else 0.0
            )

        return result


def train_one_epoch(
    model: ToMBeliefBackbone,
    data_loader: DataLoader,
    optimizer: AdamW,
    *,
    device: torch.device,
    gradient_clip_norm: float,
) -> dict[str, int | float]:
    """Run one training epoch."""

    model.train()
    accumulator = MetricAccumulator()

    for raw_batch in data_loader:
        batch = _move_batch_to_device(
            raw_batch,
            device,
        )
        valid_subject_count = int(
            batch["subject_mask"].sum().item()
        )

        if valid_subject_count == 0:
            continue

        optimizer.zero_grad(set_to_none=True)

        output = model(
            subject_ids=batch["subject_ids"],
            action_ids=batch["action_ids"],
            object_ids=batch["object_ids"],
            attention_mask=batch["attention_mask"],
            event_type_ids=batch["event_type_ids"],
            phase_ids=batch["phase_ids"],
            day_values=batch["day_values"],
        )

        loss = masked_pair_kl_divergence(
            output["pair_logits"],
            batch["pair_targets"],
            batch["subject_mask"],
        )
        loss.backward()

        if gradient_clip_norm > 0.0:
            clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_norm,
            )

        optimizer.step()

        accumulator.update(
            loss=loss,
            pair_logits=(
                output["pair_logits"].detach()
            ),
            pair_targets=batch["pair_targets"],
            subject_mask=batch["subject_mask"],
        )

    return accumulator.finalize()


@torch.no_grad()
def evaluate_model(
    model: ToMBeliefBackbone,
    data_loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, int | float]:
    """Evaluate subjective beliefs without gradients."""

    model.eval()
    accumulator = MetricAccumulator()

    for raw_batch in data_loader:
        batch = _move_batch_to_device(
            raw_batch,
            device,
        )
        valid_subject_count = int(
            batch["subject_mask"].sum().item()
        )

        if valid_subject_count == 0:
            continue

        output = model(
            subject_ids=batch["subject_ids"],
            action_ids=batch["action_ids"],
            object_ids=batch["object_ids"],
            attention_mask=batch["attention_mask"],
            event_type_ids=batch["event_type_ids"],
            phase_ids=batch["phase_ids"],
            day_values=batch["day_values"],
        )

        loss = masked_pair_kl_divergence(
            output["pair_logits"],
            batch["pair_targets"],
            batch["subject_mask"],
        )

        accumulator.update(
            loss=loss,
            pair_logits=output["pair_logits"],
            pair_targets=batch["pair_targets"],
            subject_mask=batch["subject_mask"],
        )

    return accumulator.finalize()


def _atomic_json_dump(
    value: Any,
    path: Path,
) -> None:
    """Write JSON by replacing a temporary file."""

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )
    temporary_path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _atomic_torch_save(
    value: Any,
    path: Path,
) -> None:
    """Write a PyTorch checkpoint atomically."""

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )
    torch.save(value, temporary_path)
    temporary_path.replace(path)


def _checkpoint_payload(
    *,
    model: ToMBeliefBackbone,
    optimizer: AdamW,
    config: TrainingConfig,
    epoch: int,
    train_metrics: Mapping[str, int | float],
    validation_metrics: Mapping[str, int | float],
) -> dict[str, Any]:
    """Build one reproducible checkpoint payload."""

    return {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "public_event_schema_version": PUBLIC_EVENT_SCHEMA_VERSION,
        "structured_token_to_id": dict(STRUCTURED_TOKEN_TO_ID),
        "public_phase_to_id": dict(PHASE_TO_ID),
        "target_encoding": TARGET_ENCODING,
        "projection_version": PROJECTION_VERSION,
        "pair_class_count": NUM_WOLF_PAIR_CLASSES,
        "pair_ordering": PAIR_ORDERING,
        "raw_label_field": RAW_LABEL_FIELD,
        "raw_label_type": RAW_LABEL_TYPE,
        "numeric_annotation_present": NUMERIC_ANNOTATION_PRESENT,
        "raw_label_semantics": RAW_LABEL_SEMANTICS,
        "target_interpretation": TARGET_INTERPRETATION,
        "target_distribution_is_reporter_probability": False,
        "target_distribution_is_deterministic_encoding": True,
        "supervision_scope": SUPERVISION_SCOPE,
        "label_source": LABEL_SOURCE,
        "label_context_scope": LABEL_CONTEXT_SCOPE,
        "model_input_scope": MODEL_INPUT_SCOPE,
        "report_context_mode": REPORT_CONTEXT_MODE,
        "report_side_effect_free": REPORT_SIDE_EFFECT_FREE,
        "global_truth_injected": GLOBAL_TRUTH_INJECTED,
        "other_players_private_information_visible": (
            OTHER_PLAYERS_PRIVATE_INFORMATION_VISIBLE
        ),
        "private_context_serialized": PRIVATE_CONTEXT_SERIALIZED,
        "report_timing": REPORT_TIMING,
        "observer_selection": OBSERVER_SELECTION,
        "truth_based_observer_selection": TRUTH_BASED_OBSERVER_SELECTION,
        "marginal_semantics": MARGINAL_SEMANTICS,
        "model_output": MODEL_OUTPUT,
        "output_activation": OUTPUT_ACTIVATION,
        "backbone": BACKBONE_NAME,
        "epoch": epoch,
        "training_config": asdict(config),
        "model_config": asdict(model.config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": dict(train_metrics),
        "validation_metrics": dict(validation_metrics),
    }


def _print_epoch_progress(
    *,
    epoch: int,
    train_metrics: Mapping[str, int | float],
    validation_metrics: Mapping[str, int | float],
    best_metric: float,
    epochs_without_improvement: int,
    patience: int,
) -> None:
    print(
        " ".join(
            (
                f"epoch={epoch}",
                "train_pair_kl="
                f"{float(train_metrics['mean_pair_kl_divergence']):.6f}",
                "validation_pair_kl="
                f"{float(validation_metrics['mean_pair_kl_divergence']):.6f}",
                f"best_validation_pair_kl={best_metric:.6f}",
                "epochs_without_improvement="
                f"{epochs_without_improvement}/{patience}",
            )
        ),
        flush=True,
    )


def run_training(
    config: TrainingConfig,
) -> dict[str, Any]:
    """Train with explicit train/validation sets and early stopping."""

    set_random_seed(config.seed)
    device = resolve_device(config.device)

    (
        train_loader,
        validation_loader,
        partition,
    ) = build_data_loaders(config)

    train_subject_count = count_supervised_subjects(
        train_loader
    )
    validation_subject_count = count_supervised_subjects(
        validation_loader
    )

    if train_subject_count == 0:
        raise ValueError(
            "training dataset contains no valid subjective target rows"
        )

    if validation_subject_count == 0:
        raise ValueError(
            "validation dataset contains no valid subjective target rows"
        )

    # Counting traversed the shuffled train loader. Rebuild it after resetting
    # the seed so the first optimization epoch remains reproducible.
    set_random_seed(config.seed)
    (
        train_loader,
        validation_loader,
        partition,
    ) = build_data_loaders(config)

    model = build_model(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )

    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history: list[dict[str, Any]] = []
    best_metric = math.inf
    best_epoch = 0
    best_validation_metrics: dict[str, int | float] | None = None
    epochs_without_improvement = 0
    stopped_early = False

    last_checkpoint_path = (
        output_dir / "checkpoint_last.pt"
    )
    best_checkpoint_path = (
        output_dir / "checkpoint_best.pt"
    )
    history_path = output_dir / "history.json"

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            gradient_clip_norm=float(
                config.gradient_clip_norm
            ),
        )
        validation_metrics = evaluate_model(
            model,
            validation_loader,
            device=device,
        )

        monitored_value = float(
            validation_metrics[
                "mean_pair_kl_divergence"
            ]
        )
        improved = (
            best_epoch == 0
            or monitored_value
            < best_metric
            - float(
                config.early_stopping_min_delta
            )
        )

        if improved:
            best_metric = monitored_value
            best_epoch = epoch
            best_validation_metrics = dict(
                validation_metrics
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "is_best": improved,
            "epochs_without_improvement": (
                epochs_without_improvement
            ),
        }
        history.append(epoch_record)

        checkpoint = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=epoch,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
        )
        _atomic_torch_save(
            checkpoint,
            last_checkpoint_path,
        )

        if improved:
            _atomic_torch_save(
                checkpoint,
                best_checkpoint_path,
            )

        _atomic_json_dump(
            history,
            history_path,
        )

        _print_epoch_progress(
            epoch=epoch,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            best_metric=best_metric,
            epochs_without_improvement=(
                epochs_without_improvement
            ),
            patience=config.early_stopping_patience,
        )

        if (
            epochs_without_improvement
            >= config.early_stopping_patience
        ):
            stopped_early = True
            break

    if best_validation_metrics is None:
        raise RuntimeError(
            "training did not produce a best validation checkpoint"
        )

    final_record = history[-1]

    summary = {
        "status": "ok",
        "device": str(device),
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "public_event_schema_version": PUBLIC_EVENT_SCHEMA_VERSION,
        "target_encoding": TARGET_ENCODING,
        "projection_version": PROJECTION_VERSION,
        "pair_class_count": NUM_WOLF_PAIR_CLASSES,
        "pair_ordering": PAIR_ORDERING,
        "raw_label_field": RAW_LABEL_FIELD,
        "raw_label_type": RAW_LABEL_TYPE,
        "numeric_annotation_present": NUMERIC_ANNOTATION_PRESENT,
        "raw_label_semantics": RAW_LABEL_SEMANTICS,
        "target_interpretation": TARGET_INTERPRETATION,
        "supervision_scope": SUPERVISION_SCOPE,
        "label_source": LABEL_SOURCE,
        "label_context_scope": LABEL_CONTEXT_SCOPE,
        "model_input_scope": MODEL_INPUT_SCOPE,
        "report_context_mode": REPORT_CONTEXT_MODE,
        "report_side_effect_free": REPORT_SIDE_EFFECT_FREE,
        "global_truth_injected": GLOBAL_TRUTH_INJECTED,
        "other_players_private_information_visible": (
            OTHER_PLAYERS_PRIVATE_INFORMATION_VISIBLE
        ),
        "private_context_serialized": PRIVATE_CONTEXT_SERIALIZED,
        "report_timing": REPORT_TIMING,
        "observer_selection": OBSERVER_SELECTION,
        "truth_based_observer_selection": TRUTH_BASED_OBSERVER_SELECTION,
        "model_selection_metric": "all_valid_mean_pair_kl_divergence",
        "marginal_semantics": MARGINAL_SEMANTICS,
        "model_output": MODEL_OUTPUT,
        "output_activation": OUTPUT_ACTIVATION,
        "backbone": BACKBONE_NAME,
        "model_config": asdict(model.config),
        "dataset_mode": "explicit",
        "train_dataset_path": (
            str(
                Path(
                    config.train_dataset_path
                ).resolve()
            )
        ),
        "validation_dataset_path": (
            str(
                Path(
                    config.validation_dataset_path
                ).resolve()
            )
        ),
        "output_dir": str(output_dir),
        "requested_epoch_count": config.epochs,
        "epoch_count": len(history),
        "stopped_early": stopped_early,
        "early_stopping_patience": (
            config.early_stopping_patience
        ),
        "early_stopping_min_delta": float(
            config.early_stopping_min_delta
        ),
        "train_sample_count": len(
            partition.train_samples
        ),
        "validation_sample_count": len(
            partition.validation_samples
        ),
        "train_game_ids": list(
            partition.train_game_ids
        ),
        "validation_game_ids": list(
            partition.validation_game_ids
        ),
        "train_supervised_subject_count": (
            train_subject_count
        ),
        "validation_supervised_subject_count": (
            validation_subject_count
        ),
        "best_epoch": best_epoch,
        "best_mean_pair_kl_divergence": best_metric,
        "best_validation_metrics": (
            best_validation_metrics
        ),
        "best_checkpoint": str(
            best_checkpoint_path
        ),
        "last_checkpoint": str(
            last_checkpoint_path
        ),
        "history_path": str(history_path),
        "final_train_metrics": final_record[
            "train"
        ],
        "final_validation_metrics": final_record[
            "validation"
        ],
    }

    _atomic_json_dump(
        summary,
        output_dir / "summary.json",
    )

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the formal-training command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Train the subjective seven-player ToM belief backbone "
            "from explicit game-level train and validation files."
        )
    )

    parser.add_argument(
        "--train-dataset",
        required=True,
        help="Path to the fixed training JSONL file.",
    )
    parser.add_argument(
        "--validation-dataset",
        required=True,
        help="Path to the fixed validation JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for checkpoints and metrics.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:N or mps",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--n-head",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--n-layer",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--dim-feedforward",
        type=int,
        default=None,
    )

    return parser


def main() -> int:
    """CLI entry point."""

    args = build_arg_parser().parse_args()

    config = TrainingConfig(
        train_dataset_path=args.train_dataset,
        validation_dataset_path=(
            args.validation_dataset
        ),
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        gradient_clip_norm=(
            args.gradient_clip_norm
        ),
        early_stopping_patience=(
            args.early_stopping_patience
        ),
        early_stopping_min_delta=(
            args.early_stopping_min_delta
        ),
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        dim_feedforward=args.dim_feedforward,
    )

    summary = run_training(config)

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
