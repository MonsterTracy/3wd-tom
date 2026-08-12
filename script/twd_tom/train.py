"""Train a selected ToM backbone with explicit training and validation data."""

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
from torch.utils.data import DataLoader, Subset

from werewolf.models.twd_tom.action_features import PublicEventFeatureBuilder
from werewolf.models.twd_tom.belief_backbone import (
    QWEN2_BACKBONE_NAME,
    SUPPORTED_BACKBONE_NAMES,
    ToMBeliefBackbone,
    ToMBeliefBackboneConfig,
)
from werewolf.models.twd_tom.dataset import (
    CYCLIC_ROTATION_VERSION,
    TOM_INPUT_SCOPES,
    TWDToMDataset,
    collate_twd_tom_samples,
    second_order_effective_subject_mask,
)
from werewolf.models.twd_tom.losses import (
    masked_distribution_cross_entropy,
    masked_suspicion_binary_cross_entropy,
)
from werewolf.models.twd_tom.metrics import (
    compute_subjective_pair_diagnostics,
    compute_subjective_pair_metrics,
)
from werewolf.models.twd_tom.public_events import (
    PHASE_TO_ID,
    PUBLIC_EVENT_SCHEMA_VERSION,
    STRUCTURED_TOKEN_TO_ID,
)
from werewolf.models.twd_tom.samples import SAMPLE_SCHEMA_VERSION
from werewolf.models.twd_tom.schema import (
    ACTION_NAMES,
    ACTION_TO_ID,
    NUM_WOLF_PAIR_CLASSES,
    PAIR_ORDERING,
    PROJECTION_VERSION,
    PUBLIC_ONLY_BELIEF_INFORMATION_SCOPE,
    SECOND_ORDER_TARGET_ENCODING,
    SECOND_ORDER_OBSERVER_READOUT,
    SECOND_ORDER_OBSERVER_EVENT_CONDITIONING,
    SECOND_ORDER_SUBJECT_SUPERVISION,
    TARGET_ENCODING,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


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
    enable_suspicion_aux: bool = False
    enable_factorized_pair_head: bool = False

    def __post_init__(self) -> None:
        _tom_order(self.tom_order)
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("output_dir must be non-empty text")
        if self.backbone not in SUPPORTED_BACKBONE_NAMES:
            raise ValueError(
                f"backbone must be one of {SUPPORTED_BACKBONE_NAMES}"
            )
        if not isinstance(self.enable_suspicion_aux, bool):
            raise TypeError("enable_suspicion_aux must be a boolean")
        if self.enable_suspicion_aux and self.tom_order != 2:
            raise ValueError("suspicion auxiliary supervision requires tom_order=2")
        if not isinstance(self.enable_factorized_pair_head, bool):
            raise TypeError("enable_factorized_pair_head must be a boolean")
        if self.enable_factorized_pair_head and self.tom_order != 2:
            raise ValueError("factorized pair head requires tom_order=2")
        if self.enable_factorized_pair_head and self.enable_suspicion_aux:
            raise ValueError(
                "factorized pair head and suspicion auxiliary supervision "
                "cannot be enabled together"
            )
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
            not isinstance(self.lr_scheduler, str)
            or self.lr_scheduler not in {"constant", "warmup_cosine"}
        ):
            raise ValueError(
                "lr_scheduler must be 'constant' or 'warmup_cosine'"
            )
        if (
            isinstance(self.warmup_ratio, bool)
            or not isinstance(self.warmup_ratio, (int, float))
            or not 0.0 <= self.warmup_ratio < 1.0
        ):
            raise ValueError("warmup_ratio must be in [0, 1)")
        if (
            isinstance(self.min_learning_rate, bool)
            or not isinstance(self.min_learning_rate, (int, float))
            or self.min_learning_rate < 0.0
        ):
            raise ValueError("min_learning_rate cannot be negative")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError(
                "min_learning_rate cannot exceed learning_rate"
            )
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
    torch.use_deterministic_algorithms(True)
    cudnn = getattr(torch.backends, "cudnn", None)
    if cudnn is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one required file."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"required file not found: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_relative_path(
    path: str | Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> str:
    """Return a logical repository-relative path without resolving symlinks."""

    root = Path(os.path.abspath(repo_root))
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = Path(os.path.abspath(candidate))
    try:
        return candidate.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"path must be inside the Git worktree: {candidate}"
        ) from exc


def build_run_provenance(
    config: TrainingConfig,
    *,
    resolved_device: torch.device,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Build the minimal immutable provenance for one clean-worktree run."""

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
            [
                "git",
                "-C",
                str(root),
                "status",
                "--short",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"formal training requires a readable Git worktree: {root}"
        ) from exc
    if Path(os.path.abspath(top_level)) != Path(os.path.abspath(root)):
        raise RuntimeError(
            f"formal training must run from the repository root: {root}"
        )
    if not commit:
        raise RuntimeError("formal training requires a committed Git HEAD")
    if dirty_lines:
        dirty = "\n".join(dirty_lines)
        raise RuntimeError(
            "formal training requires a clean Git worktree; dirty files:\n"
            f"{dirty}"
        )

    train_path = config.resolved_dataset_path
    validation_path = config.resolved_validation_dataset_path
    return {
        "git_commit_sha": commit,
        "git_worktree_clean": True,
        "train_dataset_path": _repository_relative_path(
            train_path, repo_root=root
        ),
        "train_dataset_sha256": sha256_file(train_path),
        "validation_dataset_path": _repository_relative_path(
            validation_path, repo_root=root
        ),
        "validation_dataset_sha256": sha256_file(validation_path),
        "output_dir": _repository_relative_path(
            config.output_dir, repo_root=root
        ),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
        "platform": platform.platform(),
        "requested_device": config.device,
        "resolved_device": str(resolved_device),
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "seed": config.seed,
    }


def _prepare_run_output_dir(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"training output path is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(
                f"training output directory must be empty: {path}"
            )
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
    """Build the explicitly selected causal backbone."""

    return ToMBeliefBackbone(
        ToMBeliefBackboneConfig(
            max_seq_len=config.max_seq_len,
            enable_suspicion_aux=config.enable_suspicion_aux,
            enable_factorized_pair_head=config.enable_factorized_pair_head,
        ),
        tom_order=config.tom_order,
        backbone_name=config.backbone,
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
        enable_cyclic_rotation=config.tom_order == 2 and shuffle,
        augmentation_seed=config.seed,
    )
    if len(dataset) == 0:
        raise ValueError(f"dataset cannot be empty: {Path(dataset_path).resolve()}")
    generator = torch.Generator().manual_seed(config.seed)
    loader_dataset = dataset
    if config.tom_order == 2:
        eligible_indices = dataset.second_order_supervised_indices()
        if not eligible_indices:
            raise ValueError(
                "dataset has no post-speech other-player observer targets"
            )
        loader_dataset = Subset(
            dataset,
            eligible_indices,
        )
    loader = DataLoader(
        loader_dataset,
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
    _training_dataset_contract(train_dataset, validation_dataset)
    return train_loader, train_dataset, validation_loader, validation_dataset


def _training_dataset_contract(
    train_dataset: TWDToMDataset,
    validation_dataset: TWDToMDataset,
) -> dict[str, Any]:
    """Return one homogeneous contract already validated by both Datasets."""

    fields = (
        "belief_information_scope",
        "model_input_scope",
        "private_fields_usage",
        "source_schema_version",
        "annotation_schema_version",
        "label_provenance",
        "source_label_provenance",
    )
    contract = {}
    for field_name in fields:
        train_value = getattr(train_dataset, field_name)
        validation_value = getattr(validation_dataset, field_name)
        if train_value != validation_value:
            raise ValueError(
                f"train and validation {field_name} values differ"
            )
        contract[field_name] = train_value
    return contract


def _public_only_lineage_metadata(
    dataset_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if dataset_contract["belief_information_scope"] != (
        PUBLIC_ONLY_BELIEF_INFORMATION_SCOPE
    ):
        return {}
    return {
        "schema_version": dataset_contract["source_schema_version"],
        "belief_information_scope": dataset_contract[
            "belief_information_scope"
        ],
        "private_fields_usage": dataset_contract["private_fields_usage"],
        "annotation_schema_version": dataset_contract[
            "annotation_schema_version"
        ],
        "label_provenance": dataset_contract["label_provenance"],
        "source_label_provenance": dataset_contract[
            "source_label_provenance"
        ],
    }


def _second_order_batch_subject_mask(
    batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    reasoning_player_id = batch.get("reasoning_player_id")
    boundary = batch.get("post_completed_public_speech_pre_next_action")
    if reasoning_player_id is None or boundary is None:
        raise ValueError("second-order batch requires formal supervision fields")
    if not torch.all(boundary):
        raise ValueError(
            "second-order batch contains a non-speech-boundary snapshot"
        )
    return second_order_effective_subject_mask(
        batch["subject_mask"],
        reasoning_player_id,
    )


def count_supervised_subjects(data_loader: DataLoader) -> int:
    total = 0
    for batch in data_loader:
        mask = batch["subject_mask"]
        has_second_order_fields = (
            "reasoning_player_id" in batch
            or "post_completed_public_speech_pre_next_action" in batch
        )
        if has_second_order_fields:
            mask = _second_order_batch_subject_mask(batch)
        total += int(mask.sum().item())
    return total


def _move_batch_to_device(
    batch: Mapping[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    fields = [
        "subject_ids",
        "action_ids",
        "object_ids",
        "event_type_ids",
        "phase_ids",
        "day_values",
        "attention_mask",
        "subject_mask",
    ]
    if "pair_targets" not in batch:
        raise ValueError("batch must contain pair_targets")
    fields.append("pair_targets")
    if "suspicion_targets" in batch:
        fields.append("suspicion_targets")
    moved = {field: batch[field].to(device) for field in fields}
    for field in ("known_werewolves", "known_non_werewolves"):
        if field in batch:
            moved[field] = batch[field].to(device)
    for field in (
        "reasoning_player_id",
        "post_completed_public_speech_pre_next_action",
    ):
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

    def __init__(self, *, tom_order: int, source_snapshot_count: int) -> None:
        self.include_collapse_diagnostics = _tom_order(tom_order) == 2
        if (
            isinstance(source_snapshot_count, bool)
            or not isinstance(source_snapshot_count, int)
            or source_snapshot_count <= 0
        ):
            raise ValueError("source_snapshot_count must be a positive integer")
        self.source_snapshot_count = source_snapshot_count
        self.processed_snapshot_count = 0
        self.valid_subject_count = 0
        self.loss_sum = 0.0
        self.suspicion_aux_loss_sum = 0.0
        self.optimization_loss_sum = 0.0
        self.has_suspicion_aux_loss = False
        self.metric_sums: dict[str, float] = {}
        self.metric_weights: dict[str, int] = {}

    def update(
        self,
        *,
        loss: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        subject_mask: torch.Tensor,
        suspicion_aux_loss: torch.Tensor | None = None,
        optimization_loss: torch.Tensor | None = None,
    ) -> None:
        if (suspicion_aux_loss is None) != (optimization_loss is None):
            raise ValueError(
                "suspicion_aux_loss and optimization_loss must be provided together"
            )
        metrics = compute_subjective_pair_metrics(
            logits,
            targets,
            subject_mask,
        )
        if self.include_collapse_diagnostics:
            metrics.update(
                compute_subjective_pair_diagnostics(
                    logits,
                    targets,
                    subject_mask,
                )
            )
        valid_count = int(metrics["valid_subject_count"])
        self.processed_snapshot_count += int(subject_mask.shape[0])
        pairwise_snapshot_count = int(
            (subject_mask.to(dtype=torch.int64).sum(dim=-1) >= 2).sum().item()
        )
        self.valid_subject_count += valid_count
        self.loss_sum += float(loss.detach().item()) * valid_count
        if suspicion_aux_loss is not None and optimization_loss is not None:
            self.has_suspicion_aux_loss = True
            self.suspicion_aux_loss_sum += (
                float(suspicion_aux_loss.detach().item()) * valid_count
            )
            self.optimization_loss_sum += (
                float(optimization_loss.detach().item()) * valid_count
            )
        for name, value in metrics.items():
            if name != "valid_subject_count":
                weight = (
                    pairwise_snapshot_count
                    if name.endswith("observer_pairwise_tv")
                    else valid_count
                )
                if weight == 0:
                    continue
                self.metric_sums[name] = self.metric_sums.get(name, 0.0) + (
                    float(value) * weight
                )
                self.metric_weights[name] = (
                    self.metric_weights.get(name, 0) + weight
                )

    def finalize(self) -> dict[str, int | float]:
        if self.valid_subject_count == 0:
            raise ValueError("dataset contains no valid observer targets")
        result: dict[str, int | float] = {
            "valid_subject_count": self.valid_subject_count,
            "mean_loss": self.loss_sum / self.valid_subject_count,
        }
        result.update({
            name: value / self.metric_weights[name]
            for name, value in self.metric_sums.items()
        })
        if self.has_suspicion_aux_loss:
            result["mean_suspicion_aux_loss"] = (
                self.suspicion_aux_loss_sum / self.valid_subject_count
            )
            result["mean_optimization_loss"] = (
                self.optimization_loss_sum / self.valid_subject_count
            )
        if self.include_collapse_diagnostics:
            for name in (
                "mean_target_observer_pairwise_tv",
                "mean_predicted_observer_pairwise_tv",
            ):
                result.setdefault(name, 0.0)
            result["post_speech_other_player_valid_subject_count"] = (
                self.valid_subject_count
            )
            result["post_speech_supervised_snapshot_fraction"] = (
                self.processed_snapshot_count / self.source_snapshot_count
            )
        return result


def _effective_subject_mask(
    model: ToMBeliefBackbone,
    batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    subject_mask = batch["subject_mask"]
    if model.tom_order == 1:
        return subject_mask.to(dtype=torch.bool)
    return _second_order_batch_subject_mask(batch)


def _targets_for_loss(
    model: ToMBeliefBackbone,
    targets: torch.Tensor,
    effective_subject_mask: torch.Tensor,
) -> torch.Tensor:
    """Keep the first-order loss input unchanged; hide excluded ToM2 rows."""

    if model.tom_order == 1:
        return targets
    return targets * effective_subject_mask.unsqueeze(-1).to(
        dtype=targets.dtype
    )


def _pair_distribution(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    return output["observer_pair_logits"], batch["pair_targets"]


def _source_snapshot_count(data_loader: DataLoader) -> int:
    dataset = data_loader.dataset
    if isinstance(dataset, Subset):
        return len(dataset.dataset)
    return len(dataset)


def build_learning_rate_scheduler(
    optimizer: AdamW,
    *,
    config: TrainingConfig,
    steps_per_epoch: int,
) -> tuple[Any | None, dict[str, Any]]:
    """Build an optimizer-step learning-rate scheduler."""

    _positive_integer(
        steps_per_epoch,
        field_name="steps_per_epoch",
    )
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

    requested_warmup_steps = int(
        round(total_steps * config.warmup_ratio)
    )
    warmup_steps = min(
        requested_warmup_steps,
        max(total_steps - 1, 0),
    )
    decay_steps = total_steps - warmup_steps

    minimum_factor = (
        float(config.min_learning_rate)
        / float(config.learning_rate)
    )

    def lr_multiplier(step_index: int) -> float:
        if warmup_steps > 0 and step_index < warmup_steps:
            return float(step_index + 1) / float(warmup_steps)

        progress = (
            float(step_index - warmup_steps)
            / float(decay_steps)
        )
        progress = min(max(progress, 0.0), 1.0)

        cosine_factor = 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
        return minimum_factor + (
            1.0 - minimum_factor
        ) * cosine_factor

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_multiplier,
    )

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
    learning_rate_start = float(
        optimizer.param_groups[0]["lr"]
    )
    accumulator = MetricAccumulator(
        tom_order=model.tom_order,
        source_snapshot_count=_source_snapshot_count(data_loader),
    )
    for raw_batch in data_loader:
        batch = _move_batch_to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = _forward_batch(model, batch)
        logits, targets = _pair_distribution(output, batch)
        effective_subject_mask = _effective_subject_mask(model, batch)
        effective_targets = _targets_for_loss(
            model,
            targets,
            effective_subject_mask,
        )
        loss = masked_distribution_cross_entropy(
            logits,
            effective_targets,
            effective_subject_mask,
        )
        suspicion_aux_loss = None
        optimization_loss = loss
        if model.config.enable_suspicion_aux:
            suspicion_aux_loss = masked_suspicion_binary_cross_entropy(
                output["observer_suspicion_logits"],
                batch["suspicion_targets"],
                effective_subject_mask,
            )
            optimization_loss = loss + suspicion_aux_loss
        optimization_loss.backward()
        if gradient_clip_norm > 0:
            clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        accumulator.update(
            loss=loss,
            logits=logits,
            targets=effective_targets,
            subject_mask=effective_subject_mask,
            suspicion_aux_loss=suspicion_aux_loss,
            optimization_loss=(
                optimization_loss if suspicion_aux_loss is not None else None
            ),
        )
    metrics = accumulator.finalize()
    metrics["learning_rate_start"] = learning_rate_start
    metrics["learning_rate_end"] = float(
        optimizer.param_groups[0]["lr"]
    )
    return metrics


@torch.no_grad()
def evaluate_model(
    model: ToMBeliefBackbone,
    data_loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, int | float]:
    model.eval()
    accumulator = MetricAccumulator(
        tom_order=model.tom_order,
        source_snapshot_count=_source_snapshot_count(data_loader),
    )
    for raw_batch in data_loader:
        batch = _move_batch_to_device(raw_batch, device)
        output = _forward_batch(model, batch)
        logits, targets = _pair_distribution(output, batch)
        effective_subject_mask = _effective_subject_mask(model, batch)
        effective_targets = _targets_for_loss(
            model,
            targets,
            effective_subject_mask,
        )
        loss = masked_distribution_cross_entropy(
            logits,
            effective_targets,
            effective_subject_mask,
        )
        suspicion_aux_loss = None
        optimization_loss = loss
        if model.config.enable_suspicion_aux:
            suspicion_aux_loss = masked_suspicion_binary_cross_entropy(
                output["observer_suspicion_logits"],
                batch["suspicion_targets"],
                effective_subject_mask,
            )
            optimization_loss = loss + suspicion_aux_loss
        accumulator.update(
            loss=loss,
            logits=logits,
            targets=effective_targets,
            subject_mask=effective_subject_mask,
            suspicion_aux_loss=suspicion_aux_loss,
            optimization_loss=(
                optimization_loss if suspicion_aux_loss is not None else None
            ),
        )
    return accumulator.finalize()


def checkpoint_task_contract(tom_order: int) -> dict[str, Any]:
    """Return the strict order-specific target and output contract."""

    if _tom_order(tom_order) == 1:
        return {
            "target_encoding": TARGET_ENCODING,
            "projection_version": PROJECTION_VERSION,
            "target_distribution_is_reporter_probability": False,
            "target_distribution_is_deterministic_encoding": True,
            "pair_class_count": NUM_WOLF_PAIR_CLASSES,
            "pair_ordering": PAIR_ORDERING,
        }
    return {
        "target_encoding": SECOND_ORDER_TARGET_ENCODING,
        "output_class_count": NUM_WOLF_PAIR_CLASSES,
        "pair_class_count": NUM_WOLF_PAIR_CLASSES,
        "pair_ordering": PAIR_ORDERING,
        "observer_readout": SECOND_ORDER_OBSERVER_READOUT,
        "train_player_augmentation": CYCLIC_ROTATION_VERSION,
        "observer_event_conditioning": SECOND_ORDER_OBSERVER_EVENT_CONDITIONING,
        "second_order_subject_supervision": SECOND_ORDER_SUBJECT_SUPERVISION,
    }


def result_model_config(model: ToMBeliefBackbone) -> dict[str, Any]:
    """Serialize the shared 21-pair model construction fields."""

    return asdict(model.config)


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
    if dataset_contract is None:
        dataset_contract = {
            "belief_information_scope": "private_conditioned",
            "source_schema_version": SAMPLE_SCHEMA_VERSION,
            "model_input_scope": TOM_INPUT_SCOPES[config.tom_order],
        }
    selection_metric_value = float(validation_metrics["mean_loss"])
    return {
        "schema_version": dataset_contract["source_schema_version"],
        "tom_order": config.tom_order,
        "model_input_scope": dataset_contract["model_input_scope"],
        **_public_only_lineage_metadata(dataset_contract),
        "public_event_schema_version": PUBLIC_EVENT_SCHEMA_VERSION,
        "speech_action_count": len(ACTION_NAMES),
        "speech_action_to_id": dict(ACTION_TO_ID),
        "structured_token_to_id": dict(STRUCTURED_TOKEN_TO_ID),
        "public_phase_to_id": dict(PHASE_TO_ID),
        **checkpoint_task_contract(config.tom_order),
        "backbone": model.backbone_name,
        "epoch": epoch,
        "train_dataset_path": run_provenance["train_dataset_path"],
        "validation_dataset_path": run_provenance[
            "validation_dataset_path"
        ],
        "training_config": {
            **asdict(config),
            "output_dir": run_provenance["output_dir"],
            "dataset_path": run_provenance["train_dataset_path"],
            "validation_dataset_path": run_provenance[
                "validation_dataset_path"
            ],
        },
        "run_provenance": dict(run_provenance),
        "model_config": result_model_config(model),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "learning_rate_schedule": dict(
            learning_rate_schedule or {"name": "constant"}
        ),
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
    run_provenance = build_run_provenance(config, resolved_device=device)
    output_dir = config.run_output_dir
    _prepare_run_output_dir(output_dir)
    (
        train_loader,
        train_dataset,
        validation_loader,
        validation_dataset,
    ) = build_training_data_loaders(config)
    dataset_contract = _training_dataset_contract(
        train_dataset,
        validation_dataset,
    )
    public_only_metadata = _public_only_lineage_metadata(dataset_contract)
    if public_only_metadata:
        run_provenance = {
            **run_provenance,
            "schema_version": dataset_contract["source_schema_version"],
            "model_input_scope": dataset_contract["model_input_scope"],
            **public_only_metadata,
        }
    model = build_model(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    lr_scheduler, learning_rate_schedule = (
        build_learning_rate_scheduler(
            optimizer,
            config=config,
            steps_per_epoch=len(train_loader),
        )
    )
    best_checkpoint_path = output_dir / "best.pt"
    last_checkpoint_path = output_dir / "last.pt"
    history = []
    task_contract = checkpoint_task_contract(config.tom_order)
    best_epoch = 0
    best_validation_mean_loss = float("inf")
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
                **task_contract,
                **public_only_metadata,
                "train": train_metrics,
                "validation": validation_metrics,
                "is_best": is_best,
                "best_epoch": best_epoch,
                "best_validation_mean_loss": best_validation_mean_loss,
                "learning_rate_schedule": dict(
                    learning_rate_schedule
                ),
                "run_provenance": dict(run_provenance),
            }
        )
        if is_best:
            _atomic_torch_save(
                checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    epoch=epoch,
                    train_metrics=train_metrics,
                    validation_metrics=validation_metrics,
                    best_epoch=best_epoch,
                    best_validation_mean_loss=best_validation_mean_loss,
                    run_provenance=run_provenance,
                    dataset_contract=dataset_contract,
                    learning_rate_schedule=learning_rate_schedule,
                ),
                best_checkpoint_path,
            )

    final_record = history[-1]
    _atomic_torch_save(
        checkpoint_payload(
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=config.epochs,
            train_metrics=final_record["train"],
            validation_metrics=final_record["validation"],
            best_epoch=best_epoch,
            best_validation_mean_loss=best_validation_mean_loss,
            run_provenance=run_provenance,
            dataset_contract=dataset_contract,
            learning_rate_schedule=learning_rate_schedule,
        ),
        last_checkpoint_path,
    )
    history_path = output_dir / "history.json"
    _atomic_json_write(history, history_path)
    summary = {
        "status": "ok",
        "tom_order": config.tom_order,
        "model_input_scope": dataset_contract["model_input_scope"],
        **public_only_metadata,
        **task_contract,
        "train_dataset": run_provenance["train_dataset_path"],
        "validation_dataset": run_provenance["validation_dataset_path"],
        "train_sample_count": len(train_dataset),
        "validation_sample_count": len(validation_dataset),
        "epochs_completed": config.epochs,
        "best_epoch": best_epoch,
        "best_validation_mean_loss": best_validation_mean_loss,
        "device": str(device),
        "backbone": model.backbone_name,
        "model_config": result_model_config(model),
        "final_train_metrics": final_record["train"],
        "final_validation_metrics": final_record["validation"],
        "selection_metric_name": "validation_mean_loss",
        "learning_rate_schedule": dict(
            learning_rate_schedule
        ),
        "run_provenance": dict(run_provenance),
    }
    run_dir = (
        Path(run_provenance["output_dir"])
        / f"tom_order_{config.tom_order}"
    )
    summary["best_checkpoint"] = (run_dir / "best.pt").as_posix()
    summary["last_checkpoint"] = (run_dir / "last.pt").as_posix()
    summary["history_path"] = (run_dir / "history.json").as_posix()
    _atomic_json_write(summary, output_dir / "summary.json")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an explicitly selected ToM backbone with train/validation data."
    )
    parser.add_argument(
        "--tom-order", required=True, type=int, choices=(1, 2),
        help="ToM order to train (1 or 2).",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Repository-local root for a new, empty training run directory.",
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Repository-local training JSONL file.",
    )
    parser.add_argument(
        "--validation-dataset", required=True,
        help="Repository-local validation JSONL file with disjoint game IDs.",
    )
    parser.add_argument(
        "--backbone",
        choices=SUPPORTED_BACKBONE_NAMES,
        default=QWEN2_BACKBONE_NAME,
        help="Causal backbone: qwen2_model or direct gpt2_block stack.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--lr-scheduler",
        choices=("constant", "warmup_cosine"),
        default="constant",
        help="Learning-rate schedule updated after optimizer steps.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
        help="Fraction of optimizer steps used for linear warmup.",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=0.0,
        help="Final learning rate used by warmup_cosine.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Python, Torch, loader, and rotation random seed (default: 42).",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Torch device or auto (CUDA, then MPS, then CPU).",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument(
        "--tom2-suspicion-aux",
        action="store_true",
        help="Add native suspicion-set BCE to ToM2 optimization only.",
    )
    parser.add_argument(
        "--tom2-factorized-pair-head",
        action="store_true",
        help="Use additive seven-player pair logits for ToM2.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = run_training(
        TrainingConfig(
            tom_order=args.tom_order,
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
            enable_suspicion_aux=args.tom2_suspicion_aux,
            enable_factorized_pair_head=args.tom2_factorized_pair_head,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
