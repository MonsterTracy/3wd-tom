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
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import transformers
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from script.twd_tom.materialize_canonical_belief_dataset import (
    validate_materialized_split_path,
)
from script.twd_tom.materialize_development_folds import (
    DEVELOPMENT_FOLD_MANIFEST_FILENAME,
    DEVELOPMENT_FOLD_MANIFEST_SCHEMA_VERSION,
    validate_development_fold_paths,
)
from werewolf.models.twd_tom.action_features import PublicEventFeatureBuilder
from werewolf.models.twd_tom.belief_backbone import (
    FULL_INPUT_FEATURE_PROFILE,
    QWEN2_BACKBONE_NAME,
    SUPPORTED_BACKBONE_NAMES,
    SUPPORTED_INPUT_FEATURE_PROFILES,
    ToMBeliefBackbone,
    ToMBeliefBackboneConfig,
)
from werewolf.models.twd_tom.baselines import (
    evaluate_dense_empirical_priors,
    fit_dense_empirical_priors,
)
from werewolf.models.twd_tom.checkpoint import (
    MODEL_OUTPUT,
    OBJECTIVE,
    checkpoint_task_contract,
    result_model_config,
)
from werewolf.models.twd_tom.dataset import (
    MODEL_INPUT_SCOPE,
    PRIVATE_MODEL_INPUT_SCOPE,
    TARGET_CONVERSION,
    TARGET_SEMANTICS,
    TWDToMDataset,
    collate_twd_tom_samples,
)
from werewolf.models.twd_tom.dense_dataset import (
    DenseTWDToMDataset,
    collate_dense_twd_tom_games,
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
from werewolf.models.twd_tom.supervision import (
    ALL_ALIVE_SCOPE,
    SUPERVISION_SCOPES,
    load_role_sidecar,
    load_role_sidecar_report,
)


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
    input_feature_profile: str = FULL_INPUT_FEATURE_PROFILE
    dense_supervision: bool = False
    private_conditioning: bool = False
    role_sidecar_path: str | None = None
    supervision_scope: str = ALL_ALIVE_SCOPE
    game_bootstrap_samples: int = 2000
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0

    def __post_init__(self) -> None:
        for field_name in ("output_dir", "dataset_path", "validation_dataset_path"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be non-empty text")
        if self.backbone not in SUPPORTED_BACKBONE_NAMES:
            raise ValueError(f"backbone must be one of {SUPPORTED_BACKBONE_NAMES}")
        if self.input_feature_profile not in SUPPORTED_INPUT_FEATURE_PROFILES:
            raise ValueError(
                "input_feature_profile must be one of "
                f"{SUPPORTED_INPUT_FEATURE_PROFILES}"
            )
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
        if not isinstance(self.dense_supervision, bool):
            raise TypeError("dense_supervision must be bool")
        if not isinstance(self.private_conditioning, bool):
            raise TypeError("private_conditioning must be bool")
        if self.role_sidecar_path is not None and (
            not isinstance(self.role_sidecar_path, str)
            or not self.role_sidecar_path.strip()
        ):
            raise ValueError("role_sidecar_path must be non-empty text or None")
        if self.supervision_scope not in SUPERVISION_SCOPES:
            raise ValueError(
                f"supervision_scope must be one of {SUPERVISION_SCOPES}"
            )
        if (
            self.supervision_scope != ALL_ALIVE_SCOPE
            and self.role_sidecar_path is None
        ):
            raise ValueError(
                "role-based supervision requires role_sidecar_path"
            )
        _positive_integer(
            self.game_bootstrap_samples,
            field_name="game_bootstrap_samples",
        )
        if (
            isinstance(self.early_stopping_patience, bool)
            or not isinstance(self.early_stopping_patience, int)
            or self.early_stopping_patience < 0
        ):
            raise ValueError("early_stopping_patience must be non-negative")
        if (
            isinstance(self.early_stopping_min_delta, bool)
            or not isinstance(self.early_stopping_min_delta, (int, float))
            or self.early_stopping_min_delta < 0
        ):
            raise ValueError("early_stopping_min_delta must be non-negative")

    @property
    def resolved_dataset_path(self) -> Path:
        return Path(self.dataset_path)

    @property
    def resolved_validation_dataset_path(self) -> Path:
        return Path(self.validation_dataset_path)

    @property
    def run_output_dir(self) -> Path:
        return Path(self.output_dir)

    @property
    def resolved_role_sidecar_path(self) -> Path | None:
        return (
            None
            if self.role_sidecar_path is None
            else Path(self.role_sidecar_path)
        )


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
    standard_manifest_path = resolved_train.parent / "split_manifest.json"
    if standard_manifest_path.is_file():
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
    development_manifest_path = (
        resolved_train.parent.parent / DEVELOPMENT_FOLD_MANIFEST_FILENAME
    )
    if development_manifest_path.is_file():
        return validate_development_fold_paths(
            resolved_train,
            resolved_validation,
        )
    raise FileNotFoundError(
        "training data has neither a split manifest nor a development fold manifest"
    )


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
    lineage = validate_training_split_lineage(train_path, validation_path)
    is_development_fold = (
        lineage.get("schema_version") == DEVELOPMENT_FOLD_MANIFEST_SCHEMA_VERSION
    )
    lexical_development_root = train_path.parent.parent
    split_manifest_path = (
        lexical_development_root
        / lineage["source_split_manifest_relative_path"]
        if is_development_fold
        else train_path.parent / "split_manifest.json"
    )
    result = {
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
        "split_manifest_digest": (
            lineage["source_split_manifest_digest"]
            if is_development_fold
            else lineage["manifest_digest"]
        ),
        "canonical_batch_summary_digest": lineage[
            "canonical_batch_summary_digest"
        ],
        "output_dir": _repository_relative_path(config.output_dir, repo_root=root),
    }
    if is_development_fold:
        lineage_manifest_path = (
            lexical_development_root / DEVELOPMENT_FOLD_MANIFEST_FILENAME
        )
        result.update({
            "data_lineage_type": "development_fold",
            "development_fold_name": lineage["_fold_name"],
            "development_fold_manifest_path": _repository_relative_path(
                lineage_manifest_path,
                repo_root=root,
            ),
            "development_fold_manifest_sha256": sha256_file(
                lineage_manifest_path
            ),
            "development_fold_manifest_digest": lineage["manifest_digest"],
        })
    else:
        result["data_lineage_type"] = "original_split"
    role_sidecar_path = config.resolved_role_sidecar_path
    if role_sidecar_path is not None:
        role_sidecar = load_role_sidecar_report(role_sidecar_path)
        expected_split_digest = (
            lineage["source_split_manifest_digest"]
            if is_development_fold
            else lineage["manifest_digest"]
        )
        if role_sidecar["split_manifest_digest"] != expected_split_digest:
            raise ValueError(
                "role sidecar and training split manifest digests differ"
            )
        if role_sidecar["canonical_batch_summary_digest"] != lineage[
            "canonical_batch_summary_digest"
        ]:
            raise ValueError(
                "role sidecar and training canonical batch digests differ"
            )
        result.update({
            "role_sidecar_path": _repository_relative_path(
                role_sidecar_path,
                repo_root=root,
            ),
            "role_sidecar_sha256": sha256_file(role_sidecar_path),
            "role_sidecar_digest": role_sidecar["sidecar_digest"],
            "role_sidecar_usage": "supervision_metadata_only",
        })
    result["supervision_scope"] = config.supervision_scope
    return result


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
        ToMBeliefBackboneConfig(
            max_seq_len=config.max_seq_len,
            private_conditioning=config.private_conditioning,
            input_feature_profile=config.input_feature_profile,
        ),
        backbone_name=config.backbone,
    )


def build_data_loader(
    config: TrainingConfig,
    *,
    dataset_path: str | Path,
    shuffle: bool,
) -> tuple[DataLoader, TWDToMDataset | DenseTWDToMDataset]:
    dataset_class = DenseTWDToMDataset if config.dense_supervision else TWDToMDataset
    role_sidecar_path = config.resolved_role_sidecar_path
    observer_roles_by_game = (
        None
        if role_sidecar_path is None
        else load_role_sidecar(role_sidecar_path)
    )
    dataset = dataset_class.from_jsonl(
        dataset_path,
        feature_builder=PublicEventFeatureBuilder(max_seq_len=config.max_seq_len),
        enable_cyclic_rotation=shuffle,
        augmentation_seed=config.seed,
        include_private_features=config.private_conditioning,
        observer_roles_by_game=observer_roles_by_game,
        supervision_scope=config.supervision_scope,
    )
    if len(dataset) == 0:
        raise ValueError(f"dataset cannot be empty: {Path(dataset_path).resolve()}")
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=(
            collate_dense_twd_tom_games
            if config.dense_supervision
            else collate_twd_tom_samples
        ),
        generator=generator if shuffle else None,
    ), dataset


def _training_dataset_contract(
    train_dataset: TWDToMDataset | DenseTWDToMDataset,
    validation_dataset: TWDToMDataset | DenseTWDToMDataset,
) -> dict[str, str]:
    for field_name in (
        "model_input_scope",
        "target_semantics",
        "target_conversion",
        "supervision_version",
        "supervision_scope",
    ):
        train_value = getattr(
            train_dataset,
            field_name,
            "independent_pre_boundary_v1",
        )
        validation_value = getattr(
            validation_dataset,
            field_name,
            "independent_pre_boundary_v1",
        )
        if train_value != validation_value:
            raise ValueError(f"train and validation {field_name} values differ")
    return {
        "source_schema_version": SAMPLE_SCHEMA_VERSION,
        "model_input_scope": train_dataset.model_input_scope,
        "target_semantics": train_dataset.target_semantics,
        "target_conversion": train_dataset.target_conversion,
        "training_supervision": getattr(
            train_dataset,
            "supervision_version",
            "independent_pre_boundary_v1",
        ),
        "supervision_scope": train_dataset.supervision_scope,
        "role_metadata_usage": "supervision_metadata_only",
    }


def build_training_data_loaders(
    config: TrainingConfig,
) -> tuple[
    DataLoader,
    TWDToMDataset | DenseTWDToMDataset,
    DataLoader,
    TWDToMDataset | DenseTWDToMDataset,
]:
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
    "observer_supervision_mask",
    "diagonal_target_mask",
    "supervision_known_non_werewolf_mask",
)
_DENSE_BATCH_TENSOR_FIELDS = (
    "boundary_indices",
    "boundary_valid_mask",
)
_PRIVATE_BATCH_TENSOR_FIELDS = (
    "known_werewolf_mask",
    "known_non_werewolf_mask",
)


def _move_batch_to_device(
    batch: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    fields = list(_BATCH_TENSOR_FIELDS)
    dense_fields_present = [field in batch for field in _DENSE_BATCH_TENSOR_FIELDS]
    if any(dense_fields_present) and not all(dense_fields_present):
        raise ValueError("dense batch boundary fields must be supplied together")
    if all(dense_fields_present):
        fields.extend(_DENSE_BATCH_TENSOR_FIELDS)
    private_fields_present = [field in batch for field in _PRIVATE_BATCH_TENSOR_FIELDS]
    if any(private_fields_present) and not all(private_fields_present):
        raise ValueError("private knowledge masks must be supplied together")
    if all(private_fields_present):
        fields.extend(_PRIVATE_BATCH_TENSOR_FIELDS)
    missing = [field for field in fields if field not in batch]
    if missing:
        raise ValueError(f"batch is missing required fields: {missing}")
    return {field: batch[field].to(device) for field in fields}


def _forward_batch(
    model: ToMBeliefBackbone, batch: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    fields = [
        "subject_ids",
        "action_ids",
        "object_ids",
        "event_type_ids",
        "phase_ids",
        "day_values",
        "attention_mask",
    ]
    if "boundary_indices" in batch:
        fields.extend(_DENSE_BATCH_TENSOR_FIELDS)
    if "known_werewolf_mask" in batch:
        fields.extend(_PRIVATE_BATCH_TENSOR_FIELDS)
    return model(**{
        field: batch[field]
        for field in fields
    })


class MetricAccumulator:
    """Aggregate metrics weighted by the number of supervised observers."""

    def __init__(self) -> None:
        self.valid_observer_count = 0
        self.loss_sum = 0.0
        self.metric_sums: dict[str, float] = {}
        self.count_sums: dict[str, int] = {}
        self.direct_sums: dict[str, float] = {}
        self.max_values: dict[str, float] = {}

    def update(
        self,
        *,
        loss: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        observer_alive_mask: torch.Tensor,
        observer_supervision_mask: torch.Tensor,
        diagonal_target_mask: torch.Tensor,
        known_non_werewolf_mask: torch.Tensor | None = None,
    ) -> None:
        metrics = compute_belief_metrics(
            logits,
            targets,
            observer_alive_mask,
            diagonal_target_mask,
            observer_supervision_mask=observer_supervision_mask,
            known_non_werewolf_mask=known_non_werewolf_mask,
        )
        count = int(metrics["valid_observer_count"])
        self.valid_observer_count += count
        self.loss_sum += float(loss.detach().item()) * count
        count_fields = {
            "positive_uniform_baseline_gap_row_count",
            "zero_uniform_baseline_gap_row_count",
            "positive_private_admissible_baseline_gap_row_count",
            "zero_private_admissible_baseline_gap_row_count",
        }
        sum_fields = {
            "model_kl_sum",
            "uniform_non_self_baseline_kl_sum",
            "private_admissible_uniform_baseline_kl_sum",
        }
        derived_fields = {
            "normalized_reducible_gap_improvement",
            "private_admissible_normalized_reducible_gap_improvement",
            "uniform_non_self_baseline_mean_kl_divergence",
            "private_admissible_uniform_baseline_mean_kl_divergence",
        }
        for name, value in metrics.items():
            if name in {"valid_observer_count", "total_row_count"}:
                continue
            if name in count_fields:
                self.count_sums[name] = self.count_sums.get(name, 0) + int(value)
            elif name in sum_fields:
                self.direct_sums[name] = self.direct_sums.get(name, 0.0) + float(value)
            elif name.startswith("max_"):
                self.max_values[name] = max(
                    self.max_values.get(name, float("-inf")),
                    float(value),
                )
            elif name not in derived_fields:
                self.metric_sums[name] = self.metric_sums.get(name, 0.0) + float(value) * count

    def finalize(self) -> dict[str, int | float]:
        if self.valid_observer_count == 0:
            raise ValueError("dataset contains no valid observer targets")
        result = {
            "total_row_count": self.valid_observer_count,
            "valid_observer_count": self.valid_observer_count,
            **self.count_sums,
            **self.direct_sums,
            **self.max_values,
            "mean_loss": self.loss_sum / self.valid_observer_count,
            **{
                name: value / self.valid_observer_count
                for name, value in self.metric_sums.items()
            },
        }
        uniform_kl_sum = result["uniform_non_self_baseline_kl_sum"]
        uniform_kl = uniform_kl_sum / self.valid_observer_count
        result["uniform_non_self_baseline_mean_kl_divergence"] = uniform_kl
        result["normalized_reducible_gap_improvement"] = (
            1.0 - result["model_kl_sum"] / uniform_kl_sum
            if uniform_kl_sum > 0.0
            else 0.0
        )
        private_cross_entropy = result.get(
            "private_admissible_uniform_baseline_mean_cross_entropy"
        )
        if private_cross_entropy is not None:
            private_kl_sum = result[
                "private_admissible_uniform_baseline_kl_sum"
            ]
            private_kl = private_kl_sum / self.valid_observer_count
            result[
                "private_admissible_uniform_baseline_mean_kl_divergence"
            ] = private_kl
            result[
                "private_admissible_normalized_reducible_gap_improvement"
            ] = (
                1.0 - result["model_kl_sum"] / private_kl_sum
                if private_kl_sum > 0.0
                else 0.0
            )
        return result


def _normalized_batch_metadata(
    raw_batch: Mapping[str, Any],
    *,
    batch_size: int,
) -> list[Mapping[str, Any]]:
    metadata = raw_batch.get("metadata")
    if isinstance(metadata, Mapping):
        game_ids = metadata.get("game_id")
        if (
            isinstance(game_ids, (str, bytes))
            or not isinstance(game_ids, Sequence)
            or len(game_ids) != batch_size
        ):
            raise ValueError("evaluation metadata requires batched game_id values")
        metadata = [
            {
                field_name: field_values[index]
                for field_name, field_values in metadata.items()
            }
            for index in range(batch_size)
        ]
    if (
        isinstance(metadata, (str, bytes))
        or not isinstance(metadata, Sequence)
        or len(metadata) != batch_size
    ):
        raise ValueError("evaluation metadata must identify every batch item")
    if any(not isinstance(item, Mapping) for item in metadata):
        raise TypeError("evaluation metadata items must be mappings")
    return list(metadata)


def _item_metric_tensors(
    *,
    batch_index: int,
    logits: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    result = {
        "logits": logits[batch_index],
        "targets": batch["belief_targets"][batch_index],
        "alive": batch["observer_alive_mask"][batch_index],
        "supervision": batch["observer_supervision_mask"][batch_index],
        "diagonal": batch["diagonal_target_mask"][batch_index],
        "known_non_wolf": batch[
            "supervision_known_non_werewolf_mask"
        ][batch_index],
    }
    if logits.ndim == 4:
        valid_boundaries = batch["boundary_valid_mask"][batch_index]
        result = {
            name: value[valid_boundaries]
            for name, value in result.items()
        }
    else:
        result = {
            name: value.unsqueeze(0)
            for name, value in result.items()
        }
    return result


def _metadata_rows(
    metadata: Mapping[str, Any],
    *,
    boundary_count: int,
) -> dict[str, list[list[Any]]]:
    def observer_rows(field_name: str) -> list[list[Any]]:
        value = metadata.get(field_name)
        if boundary_count == 1 and (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) == 7
        ):
            rows = [value]
        else:
            rows = value
        if (
            isinstance(rows, (str, bytes))
            or not isinstance(rows, Sequence)
            or len(rows) != boundary_count
        ):
            raise ValueError(
                f"metadata {field_name} must align with PRE boundaries"
            )
        normalized: list[list[Any]] = []
        for row in rows:
            if (
                isinstance(row, (str, bytes))
                or not isinstance(row, Sequence)
                or len(row) != 7
            ):
                raise ValueError(f"metadata {field_name} rows must have length 7")
            normalized.append(list(row))
        return normalized

    def boundary_rows(field_name: str) -> list[list[Any]]:
        value = metadata.get(field_name)
        if boundary_count == 1 and not (
            isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        ):
            values = [value]
        else:
            values = value
        if (
            isinstance(values, (str, bytes))
            or not isinstance(values, Sequence)
            or len(values) != boundary_count
        ):
            raise ValueError(
                f"metadata {field_name} must align with PRE boundaries"
            )
        return [[value] * 7 for value in values]

    roles = metadata.get("observer_roles")
    if roles is not None and (
        isinstance(roles, (str, bytes))
        or not isinstance(roles, Sequence)
        or len(roles) != 7
    ):
        raise ValueError("metadata observer_roles must have length 7")
    role_rows = (
        []
        if roles is None
        else [list(roles)] * boundary_count
    )
    game_id = metadata.get("game_id")
    if not isinstance(game_id, str) or not game_id.strip():
        raise ValueError("evaluation metadata requires a game_id")
    result = {
        "raw_support_size": observer_rows("raw_support_size"),
        "raw_empty": observer_rows("raw_empty"),
        "hard_knowledge_count": observer_rows("hard_knowledge_count"),
        "day": boundary_rows("day"),
        "public_action_count": boundary_rows("public_action_count"),
        "speaker_vs_non_speaker": observer_rows("speaker_vs_non_speaker"),
        "alive_count": boundary_rows("alive_count"),
        "game_id": [[game_id] * 7 for _ in range(boundary_count)],
    }
    if role_rows:
        result["observer_role"] = role_rows
    return result


def _stratum_name(dimension: str, value: Any) -> str:
    if dimension == "raw_empty":
        return "true" if value is True else "false"
    if dimension == "speaker_vs_non_speaker":
        return "speaker" if value is True else "non_speaker"
    return str(value)


class StratifiedMetricAccumulator:
    """Aggregate the frozen supervision-metadata strata from shared logits."""

    def __init__(self) -> None:
        self._groups: dict[str, dict[str, MetricAccumulator]] = {}
        self._game_groups: dict[
            str, dict[str, dict[str, MetricAccumulator]]
        ] = {}

    def update(
        self,
        *,
        raw_batch: Mapping[str, Any],
        logits: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> None:
        metadata_items = _normalized_batch_metadata(
            raw_batch,
            batch_size=logits.shape[0],
        )
        for batch_index, metadata in enumerate(metadata_items):
            game_id = metadata.get("game_id")
            if not isinstance(game_id, str) or not game_id.strip():
                raise ValueError("stratified metadata requires a game_id")
            tensors = _item_metric_tensors(
                batch_index=batch_index,
                logits=logits,
                batch=batch,
            )
            boundary_count = tensors["supervision"].shape[0]
            for dimension, rows in _metadata_rows(
                metadata,
                boundary_count=boundary_count,
            ).items():
                values = {
                    value
                    for row in rows
                    for value in row
                    if value is not None
                }
                for value in values:
                    stratum_mask = torch.tensor(
                        [
                            [item == value for item in row]
                            for row in rows
                        ],
                        dtype=torch.bool,
                        device=tensors["supervision"].device,
                    )
                    supervision = tensors["supervision"] & stratum_mask
                    if not torch.any(supervision):
                        continue
                    loss = masked_belief_distribution_loss(
                        tensors["logits"],
                        tensors["targets"],
                        tensors["alive"],
                        tensors["diagonal"],
                        observer_supervision_mask=supervision,
                    )
                    accumulator = self._groups.setdefault(
                        dimension, {}
                    ).setdefault(
                        _stratum_name(dimension, value),
                        MetricAccumulator(),
                    )
                    accumulator.update(
                        loss=loss,
                        logits=tensors["logits"],
                        targets=tensors["targets"],
                        observer_alive_mask=tensors["alive"],
                        observer_supervision_mask=supervision,
                        diagonal_target_mask=tensors["diagonal"],
                        known_non_werewolf_mask=tensors["known_non_wolf"],
                    )
                    game_accumulator = self._game_groups.setdefault(
                        game_id, {}
                    ).setdefault(
                        dimension, {}
                    ).setdefault(
                        _stratum_name(dimension, value),
                        MetricAccumulator(),
                    )
                    game_accumulator.update(
                        loss=loss,
                        logits=tensors["logits"],
                        targets=tensors["targets"],
                        observer_alive_mask=tensors["alive"],
                        observer_supervision_mask=supervision,
                        diagonal_target_mask=tensors["diagonal"],
                        known_non_werewolf_mask=tensors["known_non_wolf"],
                    )

    def finalize(self) -> dict[str, dict[str, dict[str, int | float]]]:
        return {
            dimension: {
                stratum: groups[stratum].finalize()
                for stratum in sorted(groups)
            }
            for dimension, groups in sorted(self._groups.items())
        }

    def finalize_by_game(
        self,
    ) -> dict[
        str,
        dict[str, dict[str, dict[str, int | float]]],
    ]:
        return {
            game_id: {
                dimension: {
                    stratum: groups[stratum].finalize()
                    for stratum in sorted(groups)
                }
                for dimension, groups in sorted(dimensions.items())
            }
            for game_id, dimensions in sorted(self._game_groups.items())
        }


def game_macro_metrics(
    by_game: Mapping[str, Mapping[str, int | float]],
) -> dict[str, float]:
    """Average per-game mean metrics with every game weighted equally."""

    if not by_game:
        raise ValueError("game macro aggregation requires at least one game")
    names = set.intersection(*(
        {
            name
            for name, value in metrics.items()
            if isinstance(value, (int, float))
            and name not in {"total_row_count", "valid_observer_count"}
            and not name.endswith("_row_count")
            and not name.endswith("_sum")
            and not name.startswith("max_")
        }
        for metrics in by_game.values()
    ))
    return {
        name: sum(float(metrics[name]) for metrics in by_game.values())
        / len(by_game)
        for name in sorted(names)
    }


def stratified_game_macro_metrics(
    by_game: Mapping[
        str,
        Mapping[str, Mapping[str, Mapping[str, int | float]]],
    ],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compute game-macro metrics independently inside every stratum."""

    dimensions = sorted({
        dimension
        for game in by_game.values()
        for dimension in game
    })
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for dimension in dimensions:
        strata = sorted({
            stratum
            for game in by_game.values()
            for stratum in game.get(dimension, {})
        })
        result[dimension] = {}
        for stratum in strata:
            reports = {
                game_id: game[dimension][stratum]
                for game_id, game in by_game.items()
                if stratum in game.get(dimension, {})
            }
            result[dimension][stratum] = {
                "game_count": len(reports),
                "metrics": game_macro_metrics(reports),
            }
    return result


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_game_macro_metric(
    by_game: Mapping[str, Mapping[str, int | float]],
    *,
    metric_name: str,
    samples: int,
    seed: int,
) -> dict[str, int | float | str]:
    """Bootstrap one game-macro metric using games as the only sampling unit."""

    _positive_integer(samples, field_name="samples")
    values = [
        float(by_game[game_id][metric_name])
        for game_id in sorted(by_game)
    ]
    if not values:
        raise ValueError("game bootstrap requires at least one game")
    rng = random.Random(seed)
    means = [
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(samples)
    ]
    return {
        "unit": "game",
        "metric": metric_name,
        "game_count": len(values),
        "bootstrap_samples": samples,
        "seed": seed,
        "point_estimate": sum(values) / len(values),
        "ci95_lower": _percentile(means, 0.025),
        "ci95_upper": _percentile(means, 0.975),
    }


def game_bootstrap_metrics(
    by_game: Mapping[str, Mapping[str, int | float]],
    *,
    samples: int,
    seed: int,
) -> dict[str, dict[str, int | float | str]]:
    names = (
        "normalized_reducible_gap_improvement",
        "private_admissible_normalized_reducible_gap_improvement",
    )
    common_names = set.intersection(*(set(value) for value in by_game.values()))
    return {
        name: bootstrap_game_macro_metric(
            by_game,
            metric_name=name,
            samples=samples,
            seed=seed,
        )
        for name in names
        if name in common_names
    }


def count_supervised_observers(data_loader: DataLoader) -> int:
    return sum(
        int(batch["observer_supervision_mask"].sum().item())
        for batch in data_loader
    )


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
    targets = batch["belief_targets"]
    observer_alive_mask = batch["observer_alive_mask"]
    observer_supervision_mask = batch["observer_supervision_mask"]
    diagonal_target_mask = batch["diagonal_target_mask"]
    known_non_werewolf_mask = batch["supervision_known_non_werewolf_mask"]
    if logits.ndim == 4:
        if targets.shape != logits.shape:
            raise ValueError("dense belief targets must match dense logits")
        logits = logits.flatten(0, 1)
        targets = targets.flatten(0, 1)
        observer_alive_mask = observer_alive_mask.flatten(0, 1)
        observer_supervision_mask = observer_supervision_mask.flatten(0, 1)
        diagonal_target_mask = diagonal_target_mask.flatten(0, 1)
        known_non_werewolf_mask = known_non_werewolf_mask.flatten(0, 1)
    loss = masked_belief_distribution_loss(
        logits,
        targets,
        observer_alive_mask,
        diagonal_target_mask,
        observer_supervision_mask=observer_supervision_mask,
    )
    accumulator.update(
        loss=loss,
        logits=logits,
        targets=targets,
        observer_alive_mask=observer_alive_mask,
        observer_supervision_mask=observer_supervision_mask,
        diagonal_target_mask=diagonal_target_mask,
        known_non_werewolf_mask=known_non_werewolf_mask,
    )
    return loss


def _update_per_game_metrics(
    *,
    raw_batch: Mapping[str, Any],
    logits: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    accumulators: dict[str, MetricAccumulator],
) -> None:
    metadata_items = _normalized_batch_metadata(
        raw_batch,
        batch_size=logits.shape[0],
    )
    for batch_index, item_metadata in enumerate(metadata_items):
        game_id = item_metadata.get("game_id")
        if not isinstance(game_id, str) or not game_id.strip():
            raise ValueError("evaluation metadata requires a game_id")
        accumulator = accumulators.setdefault(game_id, MetricAccumulator())
        tensors = _item_metric_tensors(
            batch_index=batch_index,
            logits=logits,
            batch=batch,
        )
        loss = masked_belief_distribution_loss(
            tensors["logits"],
            tensors["targets"],
            tensors["alive"],
            tensors["diagonal"],
            observer_supervision_mask=tensors["supervision"],
        )
        accumulator.update(
            loss=loss,
            logits=tensors["logits"],
            targets=tensors["targets"],
            observer_alive_mask=tensors["alive"],
            observer_supervision_mask=tensors["supervision"],
            diagonal_target_mask=tensors["diagonal"],
            known_non_werewolf_mask=tensors["known_non_wolf"],
        )


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
    metrics, _ = evaluate_model_with_games(
        model,
        data_loader,
        device=device,
    )
    return metrics


@torch.no_grad()
def evaluate_model_with_games(
    model: ToMBeliefBackbone,
    data_loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[dict[str, int | float], dict[str, dict[str, int | float]]]:
    """Evaluate globally and retain observer-weighted metrics per game."""

    metrics, by_game, _, _ = evaluate_model_with_games_and_strata(
        model,
        data_loader,
        device=device,
    )
    return metrics, by_game


@torch.no_grad()
def evaluate_model_with_games_and_strata(
    model: ToMBeliefBackbone,
    data_loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[
    dict[str, int | float],
    dict[str, dict[str, int | float]],
    dict[str, dict[str, dict[str, int | float]]],
    dict[str, dict[str, dict[str, dict[str, int | float]]]],
]:
    """Evaluate micro, per-game, and frozen supervision-metadata strata."""

    model.eval()
    accumulator = MetricAccumulator()
    game_accumulators: dict[str, MetricAccumulator] = {}
    stratified_accumulator = StratifiedMetricAccumulator()
    for raw_batch in data_loader:
        batch = _move_batch_to_device(raw_batch, device)
        logits = _forward_batch(model, batch)[MODEL_OUTPUT]
        _update_per_game_metrics(
            raw_batch=raw_batch,
            logits=logits,
            batch=batch,
            accumulators=game_accumulators,
        )
        stratified_accumulator.update(
            raw_batch=raw_batch,
            logits=logits,
            batch=batch,
        )
        targets = batch["belief_targets"]
        observer_alive_mask = batch["observer_alive_mask"]
        observer_supervision_mask = batch["observer_supervision_mask"]
        diagonal_target_mask = batch["diagonal_target_mask"]
        known_non_werewolf_mask = batch[
            "supervision_known_non_werewolf_mask"
        ]
        if logits.ndim == 4:
            logits = logits.flatten(0, 1)
            targets = targets.flatten(0, 1)
            observer_alive_mask = observer_alive_mask.flatten(0, 1)
            observer_supervision_mask = observer_supervision_mask.flatten(0, 1)
            diagonal_target_mask = diagonal_target_mask.flatten(0, 1)
            known_non_werewolf_mask = known_non_werewolf_mask.flatten(0, 1)
        loss = masked_belief_distribution_loss(
            logits,
            targets,
            observer_alive_mask,
            diagonal_target_mask,
            observer_supervision_mask=observer_supervision_mask,
        )
        accumulator.update(
            loss=loss,
            logits=logits,
            targets=targets,
            observer_alive_mask=observer_alive_mask,
            observer_supervision_mask=observer_supervision_mask,
            diagonal_target_mask=diagonal_target_mask,
            known_non_werewolf_mask=known_non_werewolf_mask,
        )
    return (
        accumulator.finalize(),
        {
            game_id: game_accumulators[game_id].finalize()
            for game_id in sorted(game_accumulators)
        },
        stratified_accumulator.finalize(),
        stratified_accumulator.finalize_by_game(),
    )


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
    validation_by_game: Mapping[str, Any] | None = None,
    validation_stratified_metrics: Mapping[str, Any] | None = None,
    validation_stratified_by_game: Mapping[str, Any] | None = None,
    validation_stratified_game_macro: Mapping[str, Any] | None = None,
    validation_game_macro_metrics: Mapping[str, Any] | None = None,
    validation_game_bootstrap: Mapping[str, Any] | None = None,
    dataset_contract: Mapping[str, Any] | None = None,
    learning_rate_schedule: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_contract = dict(dataset_contract or {
        "source_schema_version": SAMPLE_SCHEMA_VERSION,
        "model_input_scope": (
            PRIVATE_MODEL_INPUT_SCOPE
            if model.config.private_conditioning
            else MODEL_INPUT_SCOPE
        ),
        "target_semantics": TARGET_SEMANTICS,
        "target_conversion": TARGET_CONVERSION,
    })
    task_contract = checkpoint_task_contract(model.config.private_conditioning)
    if dataset_contract["model_input_scope"] != task_contract["model_input_scope"]:
        raise ValueError("checkpoint model and Dataset input scopes differ")
    serialized_config = asdict(config)
    return {
        "schema_version": dataset_contract["source_schema_version"],
        **task_contract,
        "training_supervision": dataset_contract.get(
            "training_supervision",
            "independent_pre_boundary_v1",
        ),
        "supervision_scope": dataset_contract.get(
            "supervision_scope",
            ALL_ALIVE_SCOPE,
        ),
        "role_metadata_usage": dataset_contract.get(
            "role_metadata_usage",
            "supervision_metadata_only",
        ),
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
            "role_sidecar_path": run_provenance.get("role_sidecar_path"),
        },
        "run_provenance": dict(run_provenance),
        "model_config": result_model_config(model),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "learning_rate_schedule": dict(learning_rate_schedule or {"name": "constant"}),
        "train_metrics": dict(train_metrics),
        "validation_metrics": dict(validation_metrics),
        "validation_by_game": dict(validation_by_game or {}),
        "validation_stratified_metrics": dict(
            validation_stratified_metrics or {}
        ),
        "validation_stratified_by_game": dict(
            validation_stratified_by_game or {}
        ),
        "validation_stratified_game_macro": dict(
            validation_stratified_game_macro or {}
        ),
        "validation_game_macro_metrics": dict(
            validation_game_macro_metrics or {}
        ),
        "validation_game_bootstrap": dict(
            validation_game_bootstrap or {}
        ),
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
    validation_baselines: dict[str, Any] = {}
    if config.dense_supervision:
        _, unaugmented_train_dataset = build_data_loader(
            config,
            dataset_path=config.resolved_dataset_path,
            shuffle=False,
        )
        if not isinstance(unaugmented_train_dataset, DenseTWDToMDataset):
            raise RuntimeError("dense training requires a dense baseline dataset")
        if not isinstance(validation_dataset, DenseTWDToMDataset):
            raise RuntimeError("dense training requires a dense validation dataset")
        validation_baselines = evaluate_dense_empirical_priors(
            validation_dataset,
            fit_dense_empirical_priors(unaugmented_train_dataset),
        )
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
    best_validation_metrics: dict[str, int | float] | None = None
    best_validation_by_game: dict[str, dict[str, int | float]] | None = None
    best_validation_stratified: dict[str, Any] | None = None
    best_validation_stratified_by_game: dict[str, Any] | None = None
    best_validation_stratified_game_macro: dict[str, Any] | None = None
    best_validation_game_macro: dict[str, float] | None = None
    best_validation_game_bootstrap: dict[str, Any] | None = None
    epochs_without_improvement = 0
    stopped_early = False
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
        (
            validation_metrics,
            validation_by_game,
            validation_stratified,
            validation_stratified_by_game,
        ) = evaluate_model_with_games_and_strata(
            model,
            validation_loader,
            device=device,
        )
        validation_game_macro = game_macro_metrics(validation_by_game)
        validation_stratified_game_macro = stratified_game_macro_metrics(
            validation_stratified_by_game
        )
        validation_game_bootstrap = game_bootstrap_metrics(
            validation_by_game,
            samples=config.game_bootstrap_samples,
            seed=config.seed,
        )
        current_loss = float(validation_metrics["mean_loss"])
        if not math.isfinite(current_loss):
            raise RuntimeError("validation mean loss must remain finite")
        is_best = epoch == 1 or current_loss < (
            best_loss - float(config.early_stopping_min_delta)
        )
        if is_best:
            best_epoch, best_loss = epoch, current_loss
            best_validation_metrics = dict(validation_metrics)
            best_validation_by_game = {
                game_id: dict(metrics)
                for game_id, metrics in validation_by_game.items()
            }
            best_validation_stratified = deepcopy(validation_stratified)
            best_validation_stratified_by_game = deepcopy(
                validation_stratified_by_game
            )
            best_validation_stratified_game_macro = deepcopy(
                validation_stratified_game_macro
            )
            best_validation_game_macro = dict(validation_game_macro)
            best_validation_game_bootstrap = deepcopy(
                validation_game_bootstrap
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        stopped_early = (
            config.early_stopping_patience > 0
            and epochs_without_improvement >= config.early_stopping_patience
        )
        record = {
            "epoch": epoch,
            **checkpoint_task_contract(config.private_conditioning),
            "train": train_metrics,
            "validation": validation_metrics,
            "validation_by_game": validation_by_game,
            "validation_stratified_metrics": validation_stratified,
            "validation_stratified_by_game": validation_stratified_by_game,
            "validation_stratified_game_macro": (
                validation_stratified_game_macro
            ),
            "validation_game_macro_metrics": validation_game_macro,
            "validation_game_bootstrap": validation_game_bootstrap,
            "validation_baselines": validation_baselines,
            "is_best": is_best,
            "best_epoch": best_epoch,
            "best_validation_mean_loss": best_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "stopped_early": stopped_early,
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
                validation_by_game=validation_by_game,
                validation_stratified_metrics=validation_stratified,
                validation_stratified_by_game=validation_stratified_by_game,
                validation_stratified_game_macro=(
                    validation_stratified_game_macro
                ),
                validation_game_macro_metrics=validation_game_macro,
                validation_game_bootstrap=validation_game_bootstrap,
                best_epoch=best_epoch,
                best_validation_mean_loss=best_loss,
                run_provenance=run_provenance,
                dataset_contract=dataset_contract,
                learning_rate_schedule=schedule,
            ), best_path)
        if stopped_early:
            break
    final = history[-1]
    last_path = output_dir / "last.pt"
    _atomic_torch_save(checkpoint_payload(
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=int(final["epoch"]),
        train_metrics=final["train"],
        validation_metrics=final["validation"],
        validation_by_game=final["validation_by_game"],
        validation_stratified_metrics=final[
            "validation_stratified_metrics"
        ],
        validation_stratified_by_game=final[
            "validation_stratified_by_game"
        ],
        validation_stratified_game_macro=final[
            "validation_stratified_game_macro"
        ],
        validation_game_macro_metrics=final[
            "validation_game_macro_metrics"
        ],
        validation_game_bootstrap=final["validation_game_bootstrap"],
        best_epoch=best_epoch,
        best_validation_mean_loss=best_loss,
        run_provenance=run_provenance,
        dataset_contract=dataset_contract,
        learning_rate_schedule=schedule,
    ), last_path)
    history_path = output_dir / "history.json"
    _atomic_json_write(history, history_path)
    logical_output = Path(run_provenance["output_dir"])
    if (
        best_validation_metrics is None
        or best_validation_by_game is None
        or best_validation_stratified is None
        or best_validation_stratified_by_game is None
        or best_validation_stratified_game_macro is None
        or best_validation_game_macro is None
        or best_validation_game_bootstrap is None
    ):
        raise RuntimeError("training completed without a best validation epoch")
    summary = {
        "status": "ok",
        **checkpoint_task_contract(config.private_conditioning),
        "train_dataset": run_provenance["train_dataset_path"],
        "validation_dataset": run_provenance["validation_dataset_path"],
        "train_sample_count": len(train_dataset),
        "validation_sample_count": len(validation_dataset),
        "train_boundary_count": getattr(
            train_dataset, "boundary_count", len(train_dataset)
        ),
        "validation_boundary_count": getattr(
            validation_dataset, "boundary_count", len(validation_dataset)
        ),
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "early_stopping": {
            "patience": config.early_stopping_patience,
            "min_delta": float(config.early_stopping_min_delta),
        },
        "training_supervision": dataset_contract["training_supervision"],
        "supervision_scope": dataset_contract["supervision_scope"],
        "role_metadata_usage": dataset_contract["role_metadata_usage"],
        "training_config": asdict(config),
        "best_epoch": best_epoch,
        "best_validation_mean_loss": best_loss,
        "best_validation_metrics": best_validation_metrics,
        "best_validation_by_game": best_validation_by_game,
        "best_validation_stratified_metrics": best_validation_stratified,
        "best_validation_stratified_by_game": (
            best_validation_stratified_by_game
        ),
        "best_validation_stratified_game_macro": (
            best_validation_stratified_game_macro
        ),
        "best_validation_game_macro_metrics": best_validation_game_macro,
        "best_validation_game_bootstrap": best_validation_game_bootstrap,
        "device": str(device),
        "backbone": model.backbone_name,
        "model_config": result_model_config(model),
        "final_train_metrics": final["train"],
        "final_validation_metrics": final["validation"],
        "final_validation_by_game": final["validation_by_game"],
        "final_validation_stratified_metrics": final[
            "validation_stratified_metrics"
        ],
        "final_validation_stratified_by_game": final[
            "validation_stratified_by_game"
        ],
        "final_validation_stratified_game_macro": final[
            "validation_stratified_game_macro"
        ],
        "final_validation_game_macro_metrics": final[
            "validation_game_macro_metrics"
        ],
        "final_validation_game_bootstrap": final[
            "validation_game_bootstrap"
        ],
        "validation_baselines": validation_baselines,
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
    parser.add_argument(
        "--input-feature-profile",
        choices=SUPPORTED_INPUT_FEATURE_PROFILES,
        default=FULL_INPUT_FEATURE_PROFILE,
    )
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
    parser.add_argument("--dense-supervision", action="store_true")
    parser.add_argument("--private-conditioning", action="store_true")
    parser.add_argument("--role-sidecar")
    parser.add_argument(
        "--supervision-scope",
        choices=SUPERVISION_SCOPES,
        default=ALL_ALIVE_SCOPE,
    )
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--game-bootstrap-samples", type=int, default=2000)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = run_training(TrainingConfig(
        output_dir=args.output_dir,
        dataset_path=args.dataset,
        validation_dataset_path=args.validation_dataset,
        backbone=args.backbone,
        input_feature_profile=args.input_feature_profile,
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
        dense_supervision=args.dense_supervision,
        private_conditioning=args.private_conditioning,
        role_sidecar_path=args.role_sidecar,
        supervision_scope=args.supervision_scope,
        game_bootstrap_samples=args.game_bootstrap_samples,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    ))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
