"""Train the tom-v2 observer-conditioned belief model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import transformers
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from script.twd_tom.materialize_canonical_belief_dataset import (
    validate_materialized_split_path,
)
from werewolf.models.twd_tom.action_features import PublicEventFeatureBuilder
from werewolf.models.twd_tom.belief_backbone import (
    QWEN2_BACKBONE_NAME,
    SUPPORTED_BACKBONE_NAMES,
    ToMBeliefBackbone,
    ToMBeliefBackboneConfig,
)
from werewolf.models.twd_tom.checkpoint import (
    MODEL_OUTPUT,
    OBJECTIVE,
    checkpoint_task_contract,
    result_model_config,
)
from werewolf.models.twd_tom.dataset import (
    MODEL_INPUT_SCOPE,
    TARGET_CONVERSION,
    TARGET_SEMANTICS,
    TWDToMDataset,
    collate_twd_tom_samples,
)
from werewolf.models.twd_tom.losses import masked_belief_distribution_loss
from werewolf.models.twd_tom.metrics import compute_belief_metrics
from werewolf.models.twd_tom.public_events import (
    PHASE_TO_ID,
    PUBLIC_EVENT_SCHEMA_VERSION,
    STRUCTURED_TOKEN_TO_ID,
)
from werewolf.models.twd_tom.samples import SAMPLE_SCHEMA_VERSION
from werewolf.models.twd_tom.schema import ACTION_NAMES, ACTION_TO_ID


REPO_ROOT = Path(__file__).resolve().parents[2]


def _positive_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for one observer-conditioned belief training run."""

    output_dir: str
    dataset_path: str
    validation_dataset_path: str
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 3e-4
    lr_scheduler: str = "constant"
    warmup_ratio: float = 0.05
    min_learning_rate: float = 0.0
    weight_decay: float = 1e-2
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    gradient_clip_norm: float = 1.0
    max_seq_len: int = 256
    backbone: str = QWEN2_BACKBONE_NAME

    def __post_init__(self) -> None:
        for field_name in ("output_dir", "dataset_path", "validation_dataset_path"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be non-empty text")
        if self.backbone not in SUPPORTED_BACKBONE_NAMES:
            raise ValueError(f"backbone must be one of {SUPPORTED_BACKBONE_NAMES}")
        _positive_integer(self.epochs, field_name="epochs")
        _positive_integer(self.batch_size, field_name="batch_size")
        _positive_integer(self.max_seq_len, field_name="max_seq_len")
        if isinstance(self.learning_rate, bool) or not isinstance(
            self.learning_rate, (int, float)
        ) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.lr_scheduler not in {"constant", "warmup_cosine"}:
            raise ValueError("lr_scheduler must be 'constant' or 'warmup_cosine'")
        if isinstance(self.warmup_ratio, bool) or not isinstance(
            self.warmup_ratio, (int, float)
        ) or not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if isinstance(self.min_learning_rate, bool) or not isinstance(
            self.min_learning_rate, (int, float)
        ) or self.min_learning_rate < 0.0:
            raise ValueError("min_learning_rate cannot be negative")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate cannot exceed learning_rate")
        if isinstance(self.weight_decay, bool) or not isinstance(
            self.weight_decay, (int, float)
        ) or self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be non-empty text")
        if isinstance(self.num_workers, bool) or not isinstance(
            self.num_workers, int
        ) or self.num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")
        if isinstance(self.gradient_clip_norm, bool) or not isinstance(
            self.gradient_clip_norm, (int, float)
        ) or self.gradient_clip_norm < 0:
            raise ValueError("gradient_clip_norm cannot be negative")

    @property
    def resolved_dataset_path(self) -> Path:
        return Path(self.dataset_path)

    @property
    def resolved_validation_dataset_path(self) -> Path:
        return Path(self.validation_dataset_path)

    @property
    def run_output_dir(self) -> Path:
        return Path(self.output_dir)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    cudnn = getattr(torch.backends, "cudnn", None)
    if cudnn is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"required file not found: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_relative_path(path: str | Path, *, repo_root: Path = REPO_ROOT) -> str:
    root = Path(os.path.abspath(repo_root))
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = Path(os.path.abspath(candidate))
    try:
        return candidate.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"path must be inside the Git worktree: {candidate}") from exc


def validate_training_split_lineage(
    train_path: str | Path,
    validation_path: str | Path,
) -> dict[str, Any]:
    """Require train and validation to be siblings from one split manifest."""

    resolved_train = Path(train_path).resolve()
    resolved_validation = Path(validation_path).resolve()
    if resolved_train.parent != resolved_validation.parent:
        raise ValueError("train and validation must share one split manifest directory")
    manifest = validate_materialized_split_path(
        resolved_train,
        split_name="train",
    )
    expected_validation = (
        resolved_train.parent
        / manifest["output_files"]["validation"]["relative_path"]
    ).resolve()
    if resolved_validation != expected_validation:
        raise ValueError("validation path is not the manifest validation split")
    return manifest


def build_run_provenance(
    config: TrainingConfig,
    *,
    resolved_device: torch.device,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    root = Path(repo_root)
    try:
        top_level = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty_lines = subprocess.run(
            ["git", "-C", str(root), "status", "--short", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"formal training requires a readable Git worktree: {root}") from exc
    if Path(os.path.abspath(top_level)) != Path(os.path.abspath(root)):
        raise RuntimeError(f"formal training must run from the repository root: {root}")
    if not commit:
        raise RuntimeError("formal training requires a committed Git HEAD")
    if dirty_lines:
        raise RuntimeError(
            "formal training requires a clean Git worktree; dirty files:\n"
            + "\n".join(dirty_lines)
        )
    train_path = config.resolved_dataset_path
    validation_path = config.resolved_validation_dataset_path
    split_manifest = validate_training_split_lineage(train_path, validation_path)
    split_manifest_path = train_path.parent / "split_manifest.json"
    return {
        "git_commit_sha": commit,
        "git_worktree_clean": True,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
        "platform": platform.platform(),
        "requested_device": config.device,
        "resolved_device": str(resolved_device),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "seed": config.seed,
        "train_dataset_path": _repository_relative_path(train_path, repo_root=root),
        "train_dataset_sha256": sha256_file(train_path),
        "validation_dataset_path": _repository_relative_path(
            validation_path, repo_root=root
        ),
        "validation_dataset_sha256": sha256_file(validation_path),
        "split_manifest_path": _repository_relative_path(
            split_manifest_path,
            repo_root=root,
        ),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "split_manifest_digest": split_manifest["manifest_digest"],
        "canonical_batch_summary_digest": split_manifest[
            "canonical_batch_summary_digest"
        ],
        "output_dir": _repository_relative_path(config.output_dir, repo_root=root),
    }


def _prepare_run_output_dir(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"training output path is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(f"training output directory must be empty: {path}")
        return
    path.mkdir(parents=True)


def _atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as file:
            torch.save(value, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json_write(value: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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
    return ToMBeliefBackbone(
        ToMBeliefBackboneConfig(max_seq_len=config.max_seq_len),
        backbone_name=config.backbone,
    )


def build_data_loader(
    config: TrainingConfig,
    *,
    dataset_path: str | Path,
    shuffle: bool,
) -> tuple[DataLoader, TWDToMDataset]:
    dataset = TWDToMDataset.from_jsonl(
        dataset_path,
        feature_builder=PublicEventFeatureBuilder(max_seq_len=config.max_seq_len),
        enable_cyclic_rotation=shuffle,
        augmentation_seed=config.seed,
    )
    if len(dataset) == 0:
        raise ValueError(f"dataset cannot be empty: {Path(dataset_path).resolve()}")
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=collate_twd_tom_samples,
        generator=generator if shuffle else None,
    ), dataset


def _training_dataset_contract(
    train_dataset: TWDToMDataset,
    validation_dataset: TWDToMDataset,
) -> dict[str, str]:
    for field_name in (
        "model_input_scope",
        "target_semantics",
        "target_conversion",
    ):
        if getattr(train_dataset, field_name) != getattr(validation_dataset, field_name):
            raise ValueError(f"train and validation {field_name} values differ")
    return {
        "source_schema_version": SAMPLE_SCHEMA_VERSION,
        "model_input_scope": train_dataset.model_input_scope,
        "target_semantics": train_dataset.target_semantics,
        "target_conversion": train_dataset.target_conversion,
    }


def build_training_data_loaders(
    config: TrainingConfig,
) -> tuple[DataLoader, TWDToMDataset, DataLoader, TWDToMDataset]:
    validate_training_split_lineage(
        config.resolved_dataset_path,
        config.resolved_validation_dataset_path,
    )
    train_loader, train_dataset = build_data_loader(
        config, dataset_path=config.resolved_dataset_path, shuffle=True
    )
    validation_loader, validation_dataset = build_data_loader(
        config, dataset_path=config.resolved_validation_dataset_path, shuffle=False
    )
    train_game_ids = {sample["game_id"] for sample in train_dataset.samples}
    validation_game_ids = {sample["game_id"] for sample in validation_dataset.samples}
    overlap = sorted(train_game_ids & validation_game_ids)
    if overlap:
        raise ValueError(
            "train and validation game_id values overlap: "
            f"count={len(overlap)}, examples={overlap[:10]}"
        )
    _training_dataset_contract(train_dataset, validation_dataset)
    return train_loader, train_dataset, validation_loader, validation_dataset


_BATCH_TENSOR_FIELDS = (
    "subject_ids",
    "action_ids",
    "object_ids",
    "event_type_ids",
    "phase_ids",
    "day_values",
    "attention_mask",
    "belief_targets",
    "observer_alive_mask",
    "diagonal_target_mask",
)


def _move_batch_to_device(
    batch: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    missing = [field for field in _BATCH_TENSOR_FIELDS if field not in batch]
    if missing:
        raise ValueError(f"batch is missing required fields: {missing}")
    return {field: batch[field].to(device) for field in _BATCH_TENSOR_FIELDS}


def _forward_batch(
    model: ToMBeliefBackbone, batch: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return model(**{
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
    })


class MetricAccumulator:
    """Aggregate metrics weighted by the number of supervised observers."""

    def __init__(self) -> None:
        self.valid_observer_count = 0
        self.loss_sum = 0.0
        self.metric_sums: dict[str, float] = {}

    def update(
        self,
        *,
        loss: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        observer_alive_mask: torch.Tensor,
        diagonal_target_mask: torch.Tensor,
    ) -> None:
        metrics = compute_belief_metrics(
            logits,
            targets,
            observer_alive_mask,
            diagonal_target_mask,
        )
        count = int(metrics["valid_observer_count"])
        self.valid_observer_count += count
        self.loss_sum += float(loss.detach().item()) * count
        for name, value in metrics.items():
            if name != "valid_observer_count":
                self.metric_sums[name] = self.metric_sums.get(name, 0.0) + float(value) * count

    def finalize(self) -> dict[str, int | float]:
        if self.valid_observer_count == 0:
            raise ValueError("dataset contains no valid observer targets")
        return {
            "valid_observer_count": self.valid_observer_count,
            "mean_loss": self.loss_sum / self.valid_observer_count,
            **{
                name: value / self.valid_observer_count
                for name, value in self.metric_sums.items()
            },
        }


def count_supervised_observers(data_loader: DataLoader) -> int:
    return sum(int(batch["observer_alive_mask"].sum().item()) for batch in data_loader)


def build_learning_rate_scheduler(
    optimizer: AdamW,
    *,
    config: TrainingConfig,
    steps_per_epoch: int,
) -> tuple[Any | None, dict[str, Any]]:
    _positive_integer(steps_per_epoch, field_name="steps_per_epoch")
    total_steps = config.epochs * steps_per_epoch
    if config.lr_scheduler == "constant":
        return None, {
            "name": "constant",
            "step_unit": "optimizer_step",
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "warmup_steps": 0,
            "decay_steps": 0,
            "peak_learning_rate": float(config.learning_rate),
            "min_learning_rate": float(config.learning_rate),
        }
    warmup_steps = min(
        int(round(total_steps * config.warmup_ratio)), max(total_steps - 1, 0)
    )
    decay_steps = total_steps - warmup_steps
    minimum_factor = float(config.min_learning_rate) / float(config.learning_rate)

    def lr_multiplier(step_index: int) -> float:
        if warmup_steps > 0 and step_index < warmup_steps:
            return float(step_index + 1) / float(warmup_steps)
        progress = min(
            max(float(step_index - warmup_steps) / float(decay_steps), 0.0), 1.0
        )
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_factor + (1.0 - minimum_factor) * cosine_factor

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
    return scheduler, {
        "name": "warmup_cosine",
        "step_unit": "optimizer_step",
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "decay_steps": decay_steps,
        "warmup_ratio": float(config.warmup_ratio),
        "peak_learning_rate": float(config.learning_rate),
        "min_learning_rate": float(config.min_learning_rate),
    }


def _loss_and_update(
    model: ToMBeliefBackbone,
    batch: Mapping[str, torch.Tensor],
    accumulator: MetricAccumulator,
) -> torch.Tensor:
    logits = _forward_batch(model, batch)[MODEL_OUTPUT]
    loss = masked_belief_distribution_loss(
        logits,
        batch["belief_targets"],
        batch["observer_alive_mask"],
        batch["diagonal_target_mask"],
    )
    accumulator.update(
        loss=loss,
        logits=logits,
        targets=batch["belief_targets"],
        observer_alive_mask=batch["observer_alive_mask"],
        diagonal_target_mask=batch["diagonal_target_mask"],
    )
    return loss


def train_one_epoch(
    model: ToMBeliefBackbone,
    data_loader: DataLoader,
    optimizer: AdamW,
    *,
    device: torch.device,
    gradient_clip_norm: float,
    lr_scheduler: Any | None = None,
) -> dict[str, int | float]:
    model.train()
    accumulator = MetricAccumulator()
    learning_rate_start = float(optimizer.param_groups[0]["lr"])
    for raw_batch in data_loader:
        batch = _move_batch_to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = _loss_and_update(model, batch, accumulator)
        loss.backward()
        if gradient_clip_norm > 0:
            clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
    metrics = accumulator.finalize()
    metrics["learning_rate_start"] = learning_rate_start
    metrics["learning_rate_end"] = float(optimizer.param_groups[0]["lr"])
    return metrics


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
        _loss_and_update(model, _move_batch_to_device(raw_batch, device), accumulator)
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
    run_provenance: Mapping[str, Any],
    dataset_contract: Mapping[str, Any] | None = None,
    learning_rate_schedule: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_contract = dict(dataset_contract or {
        "source_schema_version": SAMPLE_SCHEMA_VERSION,
        "model_input_scope": MODEL_INPUT_SCOPE,
        "target_semantics": TARGET_SEMANTICS,
        "target_conversion": TARGET_CONVERSION,
    })
    serialized_config = asdict(config)
    return {
        "schema_version": dataset_contract["source_schema_version"],
        **checkpoint_task_contract(),
        "public_event_schema_version": PUBLIC_EVENT_SCHEMA_VERSION,
        "speech_action_count": len(ACTION_NAMES),
        "speech_action_to_id": dict(ACTION_TO_ID),
        "structured_token_to_id": dict(STRUCTURED_TOKEN_TO_ID),
        "public_phase_to_id": dict(PHASE_TO_ID),
        "backbone": model.backbone_name,
        "epoch": epoch,
        "train_dataset_path": run_provenance["train_dataset_path"],
        "validation_dataset_path": run_provenance["validation_dataset_path"],
        "training_config": {
            **serialized_config,
            "output_dir": run_provenance["output_dir"],
            "dataset_path": run_provenance["train_dataset_path"],
            "validation_dataset_path": run_provenance["validation_dataset_path"],
        },
        "run_provenance": dict(run_provenance),
        "model_config": result_model_config(model),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "learning_rate_schedule": dict(learning_rate_schedule or {"name": "constant"}),
        "train_metrics": dict(train_metrics),
        "validation_metrics": dict(validation_metrics),
        "selection_metric_name": "validation_mean_loss",
        "selection_metric_value": float(validation_metrics["mean_loss"]),
        "best_epoch": best_epoch,
        "best_validation_mean_loss": best_validation_mean_loss,
    }


def run_training(config: TrainingConfig) -> dict[str, Any]:
    set_random_seed(config.seed)
    device = resolve_device(config.device)
    run_provenance = build_run_provenance(config, resolved_device=device)
    output_dir = config.run_output_dir
    _prepare_run_output_dir(output_dir)
    train_loader, train_dataset, validation_loader, validation_dataset = (
        build_training_data_loaders(config)
    )
    dataset_contract = _training_dataset_contract(train_dataset, validation_dataset)
    model = build_model(config).to(device)
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    lr_scheduler, schedule = build_learning_rate_scheduler(
        optimizer, config=config, steps_per_epoch=len(train_loader)
    )
    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_loss = float("inf")
    best_path = output_dir / "best.pt"
    for epoch in range(1, config.epochs + 1):
        train_dataset.set_epoch(epoch)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            gradient_clip_norm=config.gradient_clip_norm,
            lr_scheduler=lr_scheduler,
        )
        validation_metrics = evaluate_model(model, validation_loader, device=device)
        current_loss = float(validation_metrics["mean_loss"])
        is_best = epoch == 1 or current_loss < best_loss
        if is_best:
            best_epoch, best_loss = epoch, current_loss
        record = {
            "epoch": epoch,
            **checkpoint_task_contract(),
            "train": train_metrics,
            "validation": validation_metrics,
            "is_best": is_best,
            "best_epoch": best_epoch,
            "best_validation_mean_loss": best_loss,
            "learning_rate_schedule": dict(schedule),
            "run_provenance": dict(run_provenance),
        }
        history.append(record)
        if is_best:
            _atomic_torch_save(checkpoint_payload(
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                best_epoch=best_epoch,
                best_validation_mean_loss=best_loss,
                run_provenance=run_provenance,
                dataset_contract=dataset_contract,
                learning_rate_schedule=schedule,
            ), best_path)
    final = history[-1]
    last_path = output_dir / "last.pt"
    _atomic_torch_save(checkpoint_payload(
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=config.epochs,
        train_metrics=final["train"],
        validation_metrics=final["validation"],
        best_epoch=best_epoch,
        best_validation_mean_loss=best_loss,
        run_provenance=run_provenance,
        dataset_contract=dataset_contract,
        learning_rate_schedule=schedule,
    ), last_path)
    history_path = output_dir / "history.json"
    _atomic_json_write(history, history_path)
    logical_output = Path(run_provenance["output_dir"])
    summary = {
        "status": "ok",
        **checkpoint_task_contract(),
        "train_dataset": run_provenance["train_dataset_path"],
        "validation_dataset": run_provenance["validation_dataset_path"],
        "train_sample_count": len(train_dataset),
        "validation_sample_count": len(validation_dataset),
        "epochs_completed": config.epochs,
        "best_epoch": best_epoch,
        "best_validation_mean_loss": best_loss,
        "device": str(device),
        "backbone": model.backbone_name,
        "model_config": result_model_config(model),
        "final_train_metrics": final["train"],
        "final_validation_metrics": final["validation"],
        "selection_metric_name": "validation_mean_loss",
        "learning_rate_schedule": dict(schedule),
        "run_provenance": dict(run_provenance),
        "best_checkpoint": (logical_output / "best.pt").as_posix(),
        "last_checkpoint": (logical_output / "last.pt").as_posix(),
        "history_path": (logical_output / "history.json").as_posix(),
    }
    _atomic_json_write(summary, output_dir / "summary.json")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the tom-v2 belief model.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--validation-dataset", required=True)
    parser.add_argument("--backbone", choices=SUPPORTED_BACKBONE_NAMES, default=QWEN2_BACKBONE_NAME)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lr-scheduler", choices=("constant", "warmup_cosine"), default="constant")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-learning-rate", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--max-seq-len", type=int, default=256)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = run_training(TrainingConfig(
        output_dir=args.output_dir,
        dataset_path=args.dataset,
        validation_dataset_path=args.validation_dataset,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        gradient_clip_norm=args.gradient_clip_norm,
        max_seq_len=args.max_seq_len,
    ))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
