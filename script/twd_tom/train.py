"""Train the Qwen2 ToM backbone with explicit training and validation data."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from werewolf.models.twd_tom.action_features import PublicEventFeatureBuilder
from werewolf.models.twd_tom.belief_backbone import (
    BACKBONE_NAME,
    ToMBeliefBackbone,
    ToMBeliefBackboneConfig,
)
from werewolf.models.twd_tom.dataset import (
    TOM_INPUT_SCOPES,
    TWDToMDataset,
    collate_twd_tom_samples,
)
from werewolf.models.twd_tom.losses import masked_pair_cross_entropy
from werewolf.models.twd_tom.metrics import compute_subjective_pair_metrics
from werewolf.models.twd_tom.public_events import (
    PHASE_TO_ID,
    PUBLIC_EVENT_SCHEMA_VERSION,
    STRUCTURED_TOKEN_TO_ID,
)
from werewolf.models.twd_tom.samples import SAMPLE_SCHEMA_VERSION
from werewolf.models.twd_tom.schema import (
    NUM_WOLF_PAIR_CLASSES,
    PAIR_ORDERING,
    PROJECTION_VERSION,
    TARGET_ENCODING,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DATASET_PATHS = {
    1: REPO_ROOT / "data" / "qwen25" / "raw_tom.jsonl",
    2: REPO_ROOT / "data" / "qwen25" / "raw_tom2.jsonl",
}


def _positive_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _tom_order(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (1, 2):
        raise ValueError("tom_order must be 1 or 2")
    return value


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for one first- or second-order training run."""

    tom_order: int
    output_dir: str
    dataset_path: str
    validation_dataset_path: str
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    gradient_clip_norm: float = 1.0
    max_seq_len: int = 256

    def __post_init__(self) -> None:
        _tom_order(self.tom_order)
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("output_dir must be non-empty text")
        for field_name in ("dataset_path", "validation_dataset_path"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be non-empty text")
        _positive_integer(self.epochs, field_name="epochs")
        _positive_integer(self.batch_size, field_name="batch_size")
        _positive_integer(self.max_seq_len, field_name="max_seq_len")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be positive")
        if (
            isinstance(self.weight_decay, bool)
            or not isinstance(self.weight_decay, (int, float))
            or self.weight_decay < 0
        ):
            raise ValueError("weight_decay cannot be negative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be non-empty text")
        if (
            isinstance(self.num_workers, bool)
            or not isinstance(self.num_workers, int)
            or self.num_workers < 0
        ):
            raise ValueError("num_workers must be a non-negative integer")
        if (
            isinstance(self.gradient_clip_norm, bool)
            or not isinstance(self.gradient_clip_norm, (int, float))
            or self.gradient_clip_norm < 0
        ):
            raise ValueError("gradient_clip_norm cannot be negative")

    @property
    def resolved_dataset_path(self) -> Path:
        return Path(self.dataset_path)

    @property
    def resolved_validation_dataset_path(self) -> Path:
        return Path(self.validation_dataset_path)

    @property
    def run_output_dir(self) -> Path:
        return Path(self.output_dir) / f"tom_order_{self.tom_order}"


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: str) -> torch.device:
    normalized = requested_device.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
    return device


def build_model(config: TrainingConfig) -> ToMBeliefBackbone:
    """Build the single fixed Qwen2 backbone."""

    return ToMBeliefBackbone(
        ToMBeliefBackboneConfig(max_seq_len=config.max_seq_len)
    )


def build_data_loader(
    config: TrainingConfig,
    *,
    dataset_path: str | Path,
    shuffle: bool,
) -> tuple[DataLoader, TWDToMDataset]:
    """Load and strictly validate one complete file for one ToM order."""

    dataset = TWDToMDataset.from_jsonl(
        dataset_path,
        tom_order=config.tom_order,
        feature_builder=PublicEventFeatureBuilder(max_seq_len=config.max_seq_len),
    )
    if len(dataset) == 0:
        raise ValueError(f"dataset cannot be empty: {Path(dataset_path).resolve()}")
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=collate_twd_tom_samples,
        generator=generator if shuffle else None,
    )
    return loader, dataset


def build_training_data_loaders(
    config: TrainingConfig,
) -> tuple[DataLoader, TWDToMDataset, DataLoader, TWDToMDataset]:
    """Build order-matched loaders and reject train/validation game overlap."""

    train_loader, train_dataset = build_data_loader(
        config,
        dataset_path=config.resolved_dataset_path,
        shuffle=True,
    )
    validation_loader, validation_dataset = build_data_loader(
        config,
        dataset_path=config.resolved_validation_dataset_path,
        shuffle=False,
    )
    train_game_ids = {sample["game_id"] for sample in train_dataset.samples}
    validation_game_ids = {
        sample["game_id"] for sample in validation_dataset.samples
    }
    overlapping_game_ids = sorted(train_game_ids & validation_game_ids)
    if overlapping_game_ids:
        raise ValueError(
            "train and validation game_id values overlap: "
            f"count={len(overlapping_game_ids)}, "
            f"examples={overlapping_game_ids[:10]}"
        )
    return train_loader, train_dataset, validation_loader, validation_dataset


def count_supervised_subjects(data_loader: DataLoader) -> int:
    return sum(int(batch["subject_mask"].sum().item()) for batch in data_loader)


def _move_batch_to_device(
    batch: Mapping[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    fields = (
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
    moved = {field: batch[field].to(device) for field in fields}
    for field in ("known_werewolves", "known_non_werewolves"):
        if field in batch:
            moved[field] = batch[field].to(device)
    return moved


def _forward_batch(
    model: ToMBeliefBackbone,
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    arguments = {
        field: batch[field]
        for field in (
            "subject_ids",
            "action_ids",
            "object_ids",
            "event_type_ids",
            "phase_ids",
            "day_values",
            "attention_mask",
        )
    }
    if "known_werewolves" in batch:
        arguments["known_werewolves"] = batch["known_werewolves"]
        arguments["known_non_werewolves"] = batch["known_non_werewolves"]
    return model(**arguments)


class MetricAccumulator:
    """Aggregate row-weighted training metrics."""

    def __init__(self) -> None:
        self.valid_subject_count = 0
        self.loss_sum = 0.0
        self.metric_sums: dict[str, float] = {}

    def update(
        self,
        *,
        loss: torch.Tensor,
        pair_logits: torch.Tensor,
        pair_targets: torch.Tensor,
        subject_mask: torch.Tensor,
    ) -> None:
        metrics = compute_subjective_pair_metrics(
            pair_logits, pair_targets, subject_mask
        )
        valid_count = int(metrics["valid_subject_count"])
        self.valid_subject_count += valid_count
        self.loss_sum += float(loss.detach().item()) * valid_count
        for name, value in metrics.items():
            if name != "valid_subject_count":
                self.metric_sums[name] = self.metric_sums.get(name, 0.0) + (
                    float(value) * valid_count
                )

    def finalize(self) -> dict[str, int | float]:
        if self.valid_subject_count == 0:
            raise ValueError("dataset contains no valid observer targets")
        result: dict[str, int | float] = {
            "valid_subject_count": self.valid_subject_count,
            "mean_loss": self.loss_sum / self.valid_subject_count,
        }
        result.update(
            {
                name: value / self.valid_subject_count
                for name, value in self.metric_sums.items()
            }
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
    model.train()
    accumulator = MetricAccumulator()
    for raw_batch in data_loader:
        batch = _move_batch_to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = _forward_batch(model, batch)
        loss = masked_pair_cross_entropy(
            output["pair_logits"], batch["pair_targets"], batch["subject_mask"]
        )
        loss.backward()
        if gradient_clip_norm > 0:
            clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        accumulator.update(
            loss=loss,
            pair_logits=output["pair_logits"],
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
    model.eval()
    accumulator = MetricAccumulator()
    for raw_batch in data_loader:
        batch = _move_batch_to_device(raw_batch, device)
        output = _forward_batch(model, batch)
        loss = masked_pair_cross_entropy(
            output["pair_logits"], batch["pair_targets"], batch["subject_mask"]
        )
        accumulator.update(
            loss=loss,
            pair_logits=output["pair_logits"],
            pair_targets=batch["pair_targets"],
            subject_mask=batch["subject_mask"],
        )
    return accumulator.finalize()


def checkpoint_payload(
    *,
    model: ToMBeliefBackbone,
    optimizer: AdamW,
    config: TrainingConfig,
    epoch: int,
    train_metrics: Mapping[str, int | float],
    validation_metrics: Mapping[str, int | float],
    best_epoch: int,
    best_validation_mean_loss: float,
) -> dict[str, Any]:
    selection_metric_value = float(validation_metrics["mean_loss"])
    return {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "tom_order": config.tom_order,
        "model_input_scope": TOM_INPUT_SCOPES[config.tom_order],
        "public_event_schema_version": PUBLIC_EVENT_SCHEMA_VERSION,
        "structured_token_to_id": dict(STRUCTURED_TOKEN_TO_ID),
        "public_phase_to_id": dict(PHASE_TO_ID),
        "target_encoding": TARGET_ENCODING,
        "projection_version": PROJECTION_VERSION,
        "target_distribution_is_reporter_probability": False,
        "target_distribution_is_deterministic_encoding": True,
        "pair_class_count": NUM_WOLF_PAIR_CLASSES,
        "pair_ordering": PAIR_ORDERING,
        "backbone": BACKBONE_NAME,
        "epoch": epoch,
        "train_dataset_path": str(config.resolved_dataset_path.resolve()),
        "validation_dataset_path": str(
            config.resolved_validation_dataset_path.resolve()
        ),
        "training_config": asdict(config),
        "model_config": asdict(model.config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": dict(train_metrics),
        "validation_metrics": dict(validation_metrics),
        "selection_metric_name": "validation_mean_loss",
        "selection_metric_value": selection_metric_value,
        "best_epoch": best_epoch,
        "best_validation_mean_loss": best_validation_mean_loss,
    }


def run_training(config: TrainingConfig) -> dict[str, Any]:
    """Run one order-specific training job and write its independent checkpoint."""

    set_random_seed(config.seed)
    device = resolve_device(config.device)
    (
        train_loader,
        train_dataset,
        validation_loader,
        validation_dataset,
    ) = build_training_data_loaders(config)
    model = build_model(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    output_dir = config.run_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = output_dir / "best.pt"
    last_checkpoint_path = output_dir / "last.pt"
    history = []
    best_epoch = 0
    best_validation_mean_loss = float("inf")
    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        validation_metrics = evaluate_model(
            model,
            validation_loader,
            device=device,
        )
        validation_mean_loss = float(validation_metrics["mean_loss"])
        is_best = epoch == 1 or validation_mean_loss < best_validation_mean_loss
        if is_best:
            best_epoch = epoch
            best_validation_mean_loss = validation_mean_loss
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
                "is_best": is_best,
                "best_epoch": best_epoch,
                "best_validation_mean_loss": best_validation_mean_loss,
            }
        )
        if is_best:
            torch.save(
                checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    epoch=epoch,
                    train_metrics=train_metrics,
                    validation_metrics=validation_metrics,
                    best_epoch=best_epoch,
                    best_validation_mean_loss=best_validation_mean_loss,
                ),
                best_checkpoint_path,
            )

    final_record = history[-1]
    torch.save(
        checkpoint_payload(
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=config.epochs,
            train_metrics=final_record["train"],
            validation_metrics=final_record["validation"],
            best_epoch=best_epoch,
            best_validation_mean_loss=best_validation_mean_loss,
        ),
        last_checkpoint_path,
    )
    history_path = output_dir / "history.json"
    history_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "status": "ok",
        "tom_order": config.tom_order,
        "model_input_scope": TOM_INPUT_SCOPES[config.tom_order],
        "train_dataset": str(config.resolved_dataset_path.resolve()),
        "validation_dataset": str(
            config.resolved_validation_dataset_path.resolve()
        ),
        "train_sample_count": len(train_dataset),
        "validation_sample_count": len(validation_dataset),
        "epochs_completed": config.epochs,
        "best_epoch": best_epoch,
        "best_validation_mean_loss": best_validation_mean_loss,
        "device": str(device),
        "backbone": BACKBONE_NAME,
        "model_config": asdict(model.config),
        "best_checkpoint": str(best_checkpoint_path.resolve()),
        "last_checkpoint": str(last_checkpoint_path.resolve()),
        "history_path": str(history_path.resolve()),
        "final_train_metrics": final_record["train"],
        "final_validation_metrics": final_record["validation"],
        "selection_metric_name": "validation_mean_loss",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the fixed Qwen2 ToM backbone with explicit train/validation data."
    )
    parser.add_argument("--tom-order", required=True, type=int, choices=(1, 2))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--validation-dataset", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--max-seq-len", type=int, default=256)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = run_training(
        TrainingConfig(
            tom_order=args.tom_order,
            output_dir=args.output_dir,
            dataset_path=args.dataset,
            validation_dataset_path=args.validation_dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=args.device,
            num_workers=args.num_workers,
            gradient_clip_norm=args.gradient_clip_norm,
            max_seq_len=args.max_seq_len,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
