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
    D_PUBLIC_ONLY_TOM2_BELIEF_INFORMATION_SCOPE,
    PRIVATE_CONDITIONED_BELIEF_INFORMATION_SCOPE,
    TOM_INPUT_SCOPES,
    TWDToMDataset,
    collate_twd_tom_samples,
    second_order_effective_subject_mask,
)
from werewolf.models.twd_tom.losses import masked_distribution_cross_entropy
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
from werewolf.offline_annotation import OFFLINE_ANNOTATION_SCHEMA_VERSION
from werewolf.offline_materialization import (
    D_MATERIALIZATION_POLICY_VERSION,
    D_SCHEMA_VERSION,
    OFFLINE_PRIVATE_CONDITIONED_TOM1_TASK,
    OFFLINE_PUBLIC_ONLY_TOM2_TASK,
    TOM1_MODEL_INPUT_SCOPE,
    TOM1_OBSERVER_PROVENANCE,
    TOM1_PRIVATE_FIELDS_USAGE,
    TOM2_MODEL_INPUT_SCOPE,
    TOM2_OBSERVER_PROVENANCE,
    TOM2_PRIVATE_FIELDS_USAGE,
)
from werewolf.trajectory import canonical_digest
from script.twd_tom.split_offline_d_training_data import (
    SPLIT_MANIFEST_SCHEMA_VERSION,
    SPLIT_NAMES,
    SPLIT_POLICY_VERSION,
)
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

TRAINING_MANIFEST_SCHEMA_VERSION = "classic7_twd_tom_training_manifest_v1"
CANONICAL_D_TRAINING_INTEGRATION_VERSION = (
    "classic7_canonical_d_training_integration_v1"
)
TOM2_TARGET_SEMANTICS = "public_only_observer_suspicion_compatibility_v1"
TOM2_TEMPORAL_SUPERVISION_POLICY = (
    "post_completed_public_speech_pre_next_action_v1"
)

_SPLIT_MANIFEST_FIELDS = frozenset(
    {
        "schema_version", "split_policy_version", "split_code_commit",
        "split_seed", "d_schema_version", "d_materialization_policy_version",
        "train_game_count", "validation_game_count", "test_game_count",
        "total_game_count", "tom1_source", "tom2_source", "game_ids",
        "splits", "game_id_sets_equal",
        "tom1_step_set_equals_tom2_step_set_required", "game_overlap",
        "manifest_digest",
    }
)
_SPLIT_SOURCE_FIELDS = frozenset(
    {
        "sha256", "row_count", "game_count", "materialization_task",
        "materializer_code_commits",
    }
)
_SPLIT_SUMMARY_FIELDS = frozenset(
    {
        "game_count", "tom1_row_count", "tom2_row_count",
        "tom1_file_sha256", "tom2_file_sha256",
    }
)
_TRAINING_MANIFEST_FIELDS = frozenset(
    {
        "schema_version", "integration_version", "training_code_commit",
        "git_worktree_clean", "tom_order", "split_manifest_schema_version",
        "split_policy_version", "split_seed", "split_manifest_sha256",
        "split_manifest_digest", "d_schema_version",
        "d_materialization_policy_version", "materialization_task",
        "materializer_code_commits", "belief_information_scope",
        "model_input_scope", "private_fields_usage",
        "annotation_schema_version", "label_provenance",
        "source_label_provenance", "train_dataset_relative_path",
        "validation_dataset_relative_path", "train_dataset_sha256",
        "validation_dataset_sha256", "train_game_ids", "validation_game_ids",
        "train_source_row_count", "validation_source_row_count",
        "train_effective_supervised_snapshot_count",
        "validation_effective_supervised_snapshot_count",
        "tom2_target_semantics", "tom2_temporal_supervision_policy",
        "train_cyclic_rotation_enabled", "validation_cyclic_rotation_enabled",
        "cyclic_rotation_version", "augmentation_seed", "training_config",
        "python_version", "torch_version", "transformers_version", "platform",
        "requested_device", "resolved_device", "manifest_digest",
    }
)


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
    split_manifest_path: str | None = None

    def __post_init__(self) -> None:
        _tom_order(self.tom_order)
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("output_dir must be non-empty text")
        if self.backbone not in SUPPORTED_BACKBONE_NAMES:
            raise ValueError(
                f"backbone must be one of {SUPPORTED_BACKBONE_NAMES}"
            )
        for field_name in ("dataset_path", "validation_dataset_path"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be non-empty text")
        if self.split_manifest_path is not None and (
            not isinstance(self.split_manifest_path, str)
            or not self.split_manifest_path.strip()
        ):
            raise ValueError(
                "split_manifest_path must be non-empty text or None"
            )
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
    def resolved_split_manifest_path(self) -> Path | None:
        if self.split_manifest_path is None:
            return None
        return Path(self.split_manifest_path)

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


def _lower_hex(value: Any, *, length: int, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            f"{field_name} must be a lowercase hexadecimal value of length {length}"
        )
    return value


def _manifest_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _positive_manifest_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def canonical_d_expected_task(tom_order: int) -> str:
    return {
        1: OFFLINE_PRIVATE_CONDITIONED_TOM1_TASK,
        2: OFFLINE_PUBLIC_ONLY_TOM2_TASK,
    }[_tom_order(tom_order)]


def canonical_d_order_name(tom_order: int) -> str:
    return f"tom{_tom_order(tom_order)}"


def load_canonical_d_split_manifest(path: str | Path) -> dict[str, Any]:
    """Read and strictly validate one frozen Canonical D Split V1 manifest."""

    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"split manifest not found: {manifest_path}")
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid split manifest JSON: {manifest_path}") from exc
    if not isinstance(value, dict) or set(value) != _SPLIT_MANIFEST_FIELDS:
        raise ValueError("split manifest fields do not match Canonical D Split V1")
    if value["schema_version"] != SPLIT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported split manifest schema_version")
    if value["split_policy_version"] != SPLIT_POLICY_VERSION:
        raise ValueError("unsupported split policy version")
    _lower_hex(value["split_code_commit"], length=40, field_name="split_code_commit")
    _manifest_integer(value["split_seed"], field_name="split_seed")
    if value["d_schema_version"] != D_SCHEMA_VERSION:
        raise ValueError("split manifest d_schema_version mismatch")
    if value["d_materialization_policy_version"] != D_MATERIALIZATION_POLICY_VERSION:
        raise ValueError("split manifest materialization policy mismatch")
    if value["game_id_sets_equal"] is not True:
        raise ValueError("split manifest game_id_sets_equal must be true")
    if value["tom1_step_set_equals_tom2_step_set_required"] is not False:
        raise ValueError(
            "split manifest must not require ToM1/ToM2 step-set equality"
        )
    if value["game_overlap"] is not False:
        raise ValueError("split manifest game_overlap must be false")

    counts = {
        "train": _positive_manifest_integer(
            value["train_game_count"], field_name="train_game_count"
        ),
        "validation": _positive_manifest_integer(
            value["validation_game_count"], field_name="validation_game_count"
        ),
        "test": _positive_manifest_integer(
            value["test_game_count"], field_name="test_game_count"
        ),
    }
    total = _positive_manifest_integer(
        value["total_game_count"], field_name="total_game_count"
    )
    if sum(counts.values()) != total:
        raise ValueError("split manifest game counts do not sum to total_game_count")

    game_ids = value["game_ids"]
    if not isinstance(game_ids, dict) or set(game_ids) != set(SPLIT_NAMES):
        raise ValueError("split manifest game_ids must contain train/validation/test")
    seen: set[str] = set()
    for split_name in SPLIT_NAMES:
        ids = game_ids[split_name]
        if not isinstance(ids, list) or any(
            not isinstance(game_id, str) or not game_id.strip() for game_id in ids
        ):
            raise ValueError(f"split manifest {split_name} game_ids are invalid")
        if len(ids) != len(set(ids)) or len(ids) != counts[split_name]:
            raise ValueError(f"split manifest {split_name} game_ids count mismatch")
        if seen & set(ids):
            raise ValueError("split manifest game IDs overlap")
        seen.update(ids)
    if len(seen) != total:
        raise ValueError("split manifest game IDs do not cover total_game_count")

    splits = value["splits"]
    if not isinstance(splits, dict) or set(splits) != set(SPLIT_NAMES):
        raise ValueError("split manifest splits must contain train/validation/test")
    for split_name in SPLIT_NAMES:
        summary = splits[split_name]
        if not isinstance(summary, dict) or set(summary) != _SPLIT_SUMMARY_FIELDS:
            raise ValueError(f"split manifest {split_name} summary fields mismatch")
        summary_game_count = _positive_manifest_integer(
            summary["game_count"],
            field_name=f"{split_name}.game_count",
        )
        if summary_game_count != counts[split_name]:
            raise ValueError(f"split manifest {split_name} game_count mismatch")
        for field_name in ("tom1_row_count", "tom2_row_count"):
            _positive_manifest_integer(
                summary[field_name], field_name=f"{split_name}.{field_name}"
            )
        for field_name in ("tom1_file_sha256", "tom2_file_sha256"):
            _lower_hex(
                summary[field_name], length=64,
                field_name=f"{split_name}.{field_name}",
            )

    expected_tasks = {
        "tom1_source": OFFLINE_PRIVATE_CONDITIONED_TOM1_TASK,
        "tom2_source": OFFLINE_PUBLIC_ONLY_TOM2_TASK,
    }
    for source_name, expected_task in expected_tasks.items():
        source = value[source_name]
        if not isinstance(source, dict) or set(source) != _SPLIT_SOURCE_FIELDS:
            raise ValueError(f"split manifest {source_name} fields mismatch")
        _lower_hex(
            source["sha256"],
            length=64,
            field_name=f"{source_name}.sha256",
        )
        source_row_count = _positive_manifest_integer(
            source["row_count"], field_name=f"{source_name}.row_count"
        )
        order_name = source_name.removesuffix("_source")
        split_row_count = sum(
            value["splits"][split_name][f"{order_name}_row_count"]
            for split_name in SPLIT_NAMES
        )
        if source_row_count != split_row_count:
            raise ValueError(
                f"split manifest {source_name}.row_count does not match splits"
            )
        source_game_count = _positive_manifest_integer(
            source["game_count"],
            field_name=f"{source_name}.game_count",
        )
        if source_game_count != total:
            raise ValueError(f"split manifest {source_name}.game_count mismatch")
        if source["materialization_task"] != expected_task:
            raise ValueError(
                f"split manifest {source_name} materialization_task mismatch"
            )
        commits = source["materializer_code_commits"]
        if not isinstance(commits, list) or not commits or commits != sorted(set(commits)):
            raise ValueError(
                f"split manifest {source_name}.materializer_code_commits invalid"
            )
        for commit in commits:
            _lower_hex(
                commit, length=40,
                field_name=f"{source_name}.materializer_code_commits",
            )

    manifest_digest = _lower_hex(
        value["manifest_digest"], length=64, field_name="manifest_digest"
    )
    payload = dict(value)
    payload.pop("manifest_digest")
    if canonical_digest(payload) != manifest_digest:
        raise ValueError("split manifest manifest_digest mismatch")
    return value


def canonical_d_split_binding(
    config: TrainingConfig,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind configured D train/validation paths to one exact split artifact."""

    manifest_path = config.resolved_split_manifest_path
    if manifest_path is None:
        raise ValueError("canonical D training requires --split-manifest")
    manifest_path = manifest_path.resolve()
    manifest_root = manifest_path.parent
    order_name = canonical_d_order_name(config.tom_order)
    expected_train = (manifest_root / order_name / "train.jsonl").resolve()
    expected_validation = (manifest_root / order_name / "validation.jsonl").resolve()
    configured_train = config.resolved_dataset_path.resolve()
    configured_validation = config.resolved_validation_dataset_path.resolve()
    if configured_train != expected_train:
        raise ValueError(
            "canonical D training dataset must be the manifest train JSONL"
        )
    if configured_validation != expected_validation:
        raise ValueError(
            "canonical D validation dataset must be the manifest validation JSONL"
        )
    for path, split_name in (
        (expected_train, "train"),
        (expected_validation, "validation"),
    ):
        expected_sha = manifest["splits"][split_name][f"{order_name}_file_sha256"]
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"canonical D {split_name} dataset SHA-256 mismatch: "
                f"expected {expected_sha}, got {actual_sha}"
            )
    return {
        "manifest_path": manifest_path,
        "manifest_root": manifest_root,
        "train_path": expected_train,
        "validation_path": expected_validation,
        "order_name": order_name,
    }


def canonical_d_dataset_metadata(dataset: TWDToMDataset) -> dict[str, Any]:
    """Return strict uniform D-source metadata from one already-valid Dataset."""

    if dataset.source_schema_version != D_SCHEMA_VERSION or not dataset.samples:
        raise ValueError("dataset is not canonical D V1")
    metadata_rows = []
    for sample in dataset.samples:
        if sample.get("_dataset_input_kind") != "d_v1":
            raise ValueError("canonical D Dataset contains a non-D row")
        metadata = sample.get("_dataset_source_metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("canonical D Dataset row has no source metadata")
        metadata_rows.append(metadata)
    fields = (
        "schema_version", "materialization_task",
        "materialization_policy_version", "source_annotation_task",
        "model_input_scope",
    )
    result: dict[str, Any] = {}
    for field_name in fields:
        values = {metadata[field_name] for metadata in metadata_rows}
        if len(values) != 1:
            raise ValueError(f"canonical D Dataset {field_name} is not uniform")
        result[field_name] = next(iter(values))
    result["materializer_code_commits"] = sorted(
        {metadata["materializer_code_commit"] for metadata in metadata_rows}
    )
    return result


def validate_canonical_d_training_contract(
    config: TrainingConfig,
    train_dataset: TWDToMDataset,
    validation_dataset: TWDToMDataset,
) -> dict[str, Any]:
    """Validate one D train/validation pair against its frozen split manifest."""

    manifest_path = config.resolved_split_manifest_path
    if manifest_path is None:
        raise ValueError("canonical D training requires --split-manifest")
    manifest = load_canonical_d_split_manifest(manifest_path)
    binding = canonical_d_split_binding(config, manifest)
    expected_task = canonical_d_expected_task(config.tom_order)
    train_metadata = canonical_d_dataset_metadata(train_dataset)
    validation_metadata = canonical_d_dataset_metadata(validation_dataset)
    for metadata in (train_metadata, validation_metadata):
        if metadata["schema_version"] != D_SCHEMA_VERSION:
            raise ValueError("canonical D Dataset schema mismatch")
        if metadata["materialization_task"] != expected_task:
            raise ValueError("canonical D materialization_task mismatch")
        if metadata["materialization_policy_version"] != D_MATERIALIZATION_POLICY_VERSION:
            raise ValueError("canonical D materialization policy mismatch")
    if train_metadata["source_annotation_task"] != validation_metadata["source_annotation_task"]:
        raise ValueError("train and validation source annotation tasks differ")
    dataset_contract = _training_dataset_contract(train_dataset, validation_dataset)
    expected_dataset_contract = (
        {
            "belief_information_scope": PRIVATE_CONDITIONED_BELIEF_INFORMATION_SCOPE,
            "model_input_scope": TOM1_MODEL_INPUT_SCOPE,
            "private_fields_usage": TOM1_PRIVATE_FIELDS_USAGE,
            "source_schema_version": D_SCHEMA_VERSION,
            "annotation_schema_version": OFFLINE_ANNOTATION_SCHEMA_VERSION,
            "label_provenance": TOM1_OBSERVER_PROVENANCE,
            "source_label_provenance": TOM1_OBSERVER_PROVENANCE,
        }
        if config.tom_order == 1
        else {
            "belief_information_scope": D_PUBLIC_ONLY_TOM2_BELIEF_INFORMATION_SCOPE,
            "model_input_scope": TOM2_MODEL_INPUT_SCOPE,
            "private_fields_usage": TOM2_PRIVATE_FIELDS_USAGE,
            "source_schema_version": D_SCHEMA_VERSION,
            "annotation_schema_version": OFFLINE_ANNOTATION_SCHEMA_VERSION,
            "label_provenance": TOM2_OBSERVER_PROVENANCE,
            "source_label_provenance": TOM2_OBSERVER_PROVENANCE,
        }
    )
    for field_name, expected_value in expected_dataset_contract.items():
        if dataset_contract[field_name] != expected_value:
            raise ValueError(
                f"canonical D Dataset {field_name} mismatch: "
                f"expected {expected_value!r}, got {dataset_contract[field_name]!r}"
            )

    source_name = f"{binding['order_name']}_source"
    source_summary = manifest[source_name]
    if source_summary["materialization_task"] != expected_task:
        raise ValueError("split manifest materialization_task mismatch")
    allowed_commits = set(source_summary["materializer_code_commits"])
    observed_commits = set(train_metadata["materializer_code_commits"]) | set(
        validation_metadata["materializer_code_commits"]
    )
    if not observed_commits <= allowed_commits:
        raise ValueError("Dataset materializer commit is absent from split provenance")

    train_dataset_game_ids = {
        sample["game_id"] for sample in train_dataset.samples
    }
    validation_dataset_game_ids = {
        sample["game_id"] for sample in validation_dataset.samples
    }
    if train_dataset_game_ids != set(manifest["game_ids"]["train"]):
        raise ValueError("training Dataset game IDs differ from split manifest")
    if validation_dataset_game_ids != set(manifest["game_ids"]["validation"]):
        raise ValueError("validation Dataset game IDs differ from split manifest")
    train_game_ids = list(manifest["game_ids"]["train"])
    validation_game_ids = list(manifest["game_ids"]["validation"])
    order_name = binding["order_name"]
    if len(train_dataset) != manifest["splits"]["train"][f"{order_name}_row_count"]:
        raise ValueError("training Dataset row count differs from split manifest")
    if len(validation_dataset) != manifest["splits"]["validation"][f"{order_name}_row_count"]:
        raise ValueError("validation Dataset row count differs from split manifest")

    expected_train_rotation = config.tom_order == 2
    if train_dataset.enable_cyclic_rotation is not expected_train_rotation:
        raise ValueError("canonical D training rotation configuration mismatch")
    if validation_dataset.enable_cyclic_rotation is not False:
        raise ValueError("canonical D validation rotation must be disabled")
    if expected_train_rotation and train_dataset.augmentation_seed != config.seed:
        raise ValueError("canonical D augmentation seed mismatch")

    train_effective = len(train_dataset) if config.tom_order == 1 else len(
        train_dataset.second_order_supervised_indices()
    )
    validation_effective = len(validation_dataset) if config.tom_order == 1 else len(
        validation_dataset.second_order_supervised_indices()
    )
    if train_effective <= 0 or validation_effective <= 0:
        raise ValueError("canonical D training requires non-empty effective supervision")
    return {
        "manifest": manifest,
        **binding,
        "materialization_task": expected_task,
        "materializer_code_commits": list(source_summary["materializer_code_commits"]),
        "train_game_ids": train_game_ids,
        "validation_game_ids": validation_game_ids,
        "train_source_row_count": len(train_dataset),
        "validation_source_row_count": len(validation_dataset),
        "train_effective_supervised_snapshot_count": train_effective,
        "validation_effective_supervised_snapshot_count": validation_effective,
    }


def validate_training_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless one value is the exact training manifest V1."""

    if not isinstance(value, Mapping) or set(value) != _TRAINING_MANIFEST_FIELDS:
        raise ValueError("training manifest fields do not match V1")
    manifest = dict(value)
    if manifest["schema_version"] != TRAINING_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported training manifest schema_version")
    if manifest["integration_version"] != CANONICAL_D_TRAINING_INTEGRATION_VERSION:
        raise ValueError("unsupported canonical D training integration version")
    _lower_hex(
        manifest["training_code_commit"],
        length=40,
        field_name="training_code_commit",
    )
    if manifest["git_worktree_clean"] is not True:
        raise ValueError("training manifest requires a clean Git worktree")
    tom_order = _tom_order(manifest["tom_order"])
    if manifest["split_manifest_schema_version"] != SPLIT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("training manifest split schema mismatch")
    if manifest["split_policy_version"] != SPLIT_POLICY_VERSION:
        raise ValueError("training manifest split policy mismatch")
    _manifest_integer(manifest["split_seed"], field_name="split_seed")
    for field_name in (
        "split_manifest_sha256",
        "split_manifest_digest",
        "train_dataset_sha256",
        "validation_dataset_sha256",
        "manifest_digest",
    ):
        _lower_hex(manifest[field_name], length=64, field_name=field_name)
    if manifest["d_schema_version"] != D_SCHEMA_VERSION:
        raise ValueError("training manifest D schema mismatch")
    if manifest["d_materialization_policy_version"] != D_MATERIALIZATION_POLICY_VERSION:
        raise ValueError("training manifest D materialization policy mismatch")
    if manifest["materialization_task"] != canonical_d_expected_task(tom_order):
        raise ValueError("training manifest materialization_task mismatch")
    expected_lineage = (
        {
            "belief_information_scope": PRIVATE_CONDITIONED_BELIEF_INFORMATION_SCOPE,
            "model_input_scope": TOM1_MODEL_INPUT_SCOPE,
            "private_fields_usage": TOM1_PRIVATE_FIELDS_USAGE,
            "annotation_schema_version": OFFLINE_ANNOTATION_SCHEMA_VERSION,
            "label_provenance": TOM1_OBSERVER_PROVENANCE,
            "source_label_provenance": TOM1_OBSERVER_PROVENANCE,
        }
        if tom_order == 1
        else {
            "belief_information_scope": D_PUBLIC_ONLY_TOM2_BELIEF_INFORMATION_SCOPE,
            "model_input_scope": TOM2_MODEL_INPUT_SCOPE,
            "private_fields_usage": TOM2_PRIVATE_FIELDS_USAGE,
            "annotation_schema_version": OFFLINE_ANNOTATION_SCHEMA_VERSION,
            "label_provenance": TOM2_OBSERVER_PROVENANCE,
            "source_label_provenance": TOM2_OBSERVER_PROVENANCE,
        }
    )
    for field_name, expected_value in expected_lineage.items():
        if manifest[field_name] != expected_value:
            raise ValueError(
                f"training manifest {field_name} mismatch: "
                f"expected {expected_value!r}, got {manifest[field_name]!r}"
            )
    commits = manifest["materializer_code_commits"]
    if not isinstance(commits, list) or not commits or commits != sorted(set(commits)):
        raise ValueError("training manifest materializer_code_commits invalid")
    for commit in commits:
        _lower_hex(commit, length=40, field_name="materializer_code_commits")
    for field_name in (
        "train_dataset_relative_path",
        "validation_dataset_relative_path",
    ):
        path = manifest[field_name]
        if (
            not isinstance(path, str)
            or not path.strip()
            or Path(path).is_absolute()
            or ".." in Path(path).parts
        ):
            raise ValueError(f"training manifest {field_name} must be safe relative")
    train_games = manifest["train_game_ids"]
    validation_games = manifest["validation_game_ids"]
    for field_name, game_ids in (
        ("train_game_ids", train_games),
        ("validation_game_ids", validation_games),
    ):
        if (
            not isinstance(game_ids, list)
            or not game_ids
            or len(game_ids) != len(set(game_ids))
            or any(not isinstance(game_id, str) or not game_id.strip() for game_id in game_ids)
        ):
            raise ValueError(f"training manifest {field_name} invalid")
    if set(train_games) & set(validation_games):
        raise ValueError("training manifest train/validation game IDs overlap")
    source_counts = {}
    effective_counts = {}
    for split_name in ("train", "validation"):
        source_counts[split_name] = _positive_manifest_integer(
            manifest[f"{split_name}_source_row_count"],
            field_name=f"{split_name}_source_row_count",
        )
        effective_counts[split_name] = _positive_manifest_integer(
            manifest[f"{split_name}_effective_supervised_snapshot_count"],
            field_name=f"{split_name}_effective_supervised_snapshot_count",
        )
        if effective_counts[split_name] > source_counts[split_name]:
            raise ValueError("effective supervision count exceeds source rows")
    if not isinstance(manifest["training_config"], Mapping):
        raise TypeError("training manifest training_config must be a mapping")
    if tom_order == 1:
        if any(
            manifest[field_name] is not None
            for field_name in (
                "tom2_target_semantics",
                "tom2_temporal_supervision_policy",
                "cyclic_rotation_version",
                "augmentation_seed",
            )
        ):
            raise ValueError("ToM1 training manifest has non-null ToM2 fields")
        if manifest["train_cyclic_rotation_enabled"] is not False:
            raise ValueError("ToM1 train rotation must be false")
        if source_counts != effective_counts:
            raise ValueError("ToM1 effective counts must equal source counts")
    else:
        if manifest["tom2_target_semantics"] != TOM2_TARGET_SEMANTICS:
            raise ValueError("training manifest ToM2 target semantics mismatch")
        if manifest["tom2_temporal_supervision_policy"] != (
            TOM2_TEMPORAL_SUPERVISION_POLICY
        ):
            raise ValueError("training manifest ToM2 temporal policy mismatch")
        if manifest["train_cyclic_rotation_enabled"] is not True:
            raise ValueError("ToM2 train rotation must be true")
        if manifest["cyclic_rotation_version"] != CYCLIC_ROTATION_VERSION:
            raise ValueError("training manifest cyclic rotation version mismatch")
        if (
            isinstance(manifest["augmentation_seed"], bool)
            or not isinstance(manifest["augmentation_seed"], int)
            or manifest["augmentation_seed"] < 0
        ):
            raise ValueError("training manifest augmentation_seed invalid")
    if manifest["validation_cyclic_rotation_enabled"] is not False:
        raise ValueError("validation cyclic rotation must be false")
    payload = dict(manifest)
    expected_digest = payload.pop("manifest_digest")
    if canonical_digest(payload) != expected_digest:
        raise ValueError("training manifest manifest_digest mismatch")
    return manifest


def build_training_manifest(
    config: TrainingConfig,
    *,
    resolved_device: torch.device,
    run_provenance: Mapping[str, Any],
    dataset_contract: Mapping[str, Any],
    canonical_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact immutable Canonical D training manifest V1."""

    split_manifest = canonical_context["manifest"]
    is_tom2 = config.tom_order == 2
    training_config = asdict(config)
    training_config["output_dir"] = str(Path(config.output_dir).resolve())
    training_config["dataset_path"] = run_provenance["train_dataset_path"]
    training_config["validation_dataset_path"] = run_provenance["validation_dataset_path"]
    training_config["split_manifest_path"] = "manifest.json"
    result = {
        "schema_version": TRAINING_MANIFEST_SCHEMA_VERSION,
        "integration_version": CANONICAL_D_TRAINING_INTEGRATION_VERSION,
        "training_code_commit": run_provenance["git_commit_sha"],
        "git_worktree_clean": run_provenance["git_worktree_clean"],
        "tom_order": config.tom_order,
        "split_manifest_schema_version": split_manifest["schema_version"],
        "split_policy_version": split_manifest["split_policy_version"],
        "split_seed": split_manifest["split_seed"],
        "split_manifest_sha256": sha256_file(canonical_context["manifest_path"]),
        "split_manifest_digest": split_manifest["manifest_digest"],
        "d_schema_version": D_SCHEMA_VERSION,
        "d_materialization_policy_version": D_MATERIALIZATION_POLICY_VERSION,
        "materialization_task": canonical_context["materialization_task"],
        "materializer_code_commits": canonical_context["materializer_code_commits"],
        "belief_information_scope": dataset_contract["belief_information_scope"],
        "model_input_scope": dataset_contract["model_input_scope"],
        "private_fields_usage": dataset_contract["private_fields_usage"],
        "annotation_schema_version": dataset_contract["annotation_schema_version"],
        "label_provenance": dataset_contract["label_provenance"],
        "source_label_provenance": dataset_contract["source_label_provenance"],
        "train_dataset_relative_path": run_provenance["train_dataset_path"],
        "validation_dataset_relative_path": run_provenance["validation_dataset_path"],
        "train_dataset_sha256": run_provenance["train_dataset_sha256"],
        "validation_dataset_sha256": run_provenance["validation_dataset_sha256"],
        "train_game_ids": canonical_context["train_game_ids"],
        "validation_game_ids": canonical_context["validation_game_ids"],
        "train_source_row_count": canonical_context["train_source_row_count"],
        "validation_source_row_count": canonical_context["validation_source_row_count"],
        "train_effective_supervised_snapshot_count": canonical_context[
            "train_effective_supervised_snapshot_count"
        ],
        "validation_effective_supervised_snapshot_count": canonical_context[
            "validation_effective_supervised_snapshot_count"
        ],
        "tom2_target_semantics": TOM2_TARGET_SEMANTICS if is_tom2 else None,
        "tom2_temporal_supervision_policy": TOM2_TEMPORAL_SUPERVISION_POLICY if is_tom2 else None,
        "train_cyclic_rotation_enabled": is_tom2,
        "validation_cyclic_rotation_enabled": False,
        "cyclic_rotation_version": CYCLIC_ROTATION_VERSION if is_tom2 else None,
        "augmentation_seed": config.seed if is_tom2 else None,
        "training_config": training_config,
        "python_version": run_provenance["python_version"],
        "torch_version": run_provenance["torch_version"],
        "transformers_version": run_provenance["transformers_version"],
        "platform": run_provenance["platform"],
        "requested_device": config.device,
        "resolved_device": str(resolved_device),
    }
    result["manifest_digest"] = canonical_digest(result)
    return validate_training_manifest(result)


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
    base = {
        "git_commit_sha": commit,
        "git_worktree_clean": True,
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
    if config.resolved_split_manifest_path is None:
        return {
            **base,
            "train_dataset_path": _repository_relative_path(train_path, repo_root=root),
            "train_dataset_sha256": sha256_file(train_path),
            "validation_dataset_path": _repository_relative_path(
                validation_path, repo_root=root
            ),
            "validation_dataset_sha256": sha256_file(validation_path),
            "output_dir": _repository_relative_path(config.output_dir, repo_root=root),
        }

    manifest = load_canonical_d_split_manifest(config.resolved_split_manifest_path)
    binding = canonical_d_split_binding(config, manifest)
    manifest_root = binding["manifest_root"]
    return {
        **base,
        "train_dataset_path": binding["train_path"].relative_to(manifest_root).as_posix(),
        "train_dataset_sha256": sha256_file(binding["train_path"]),
        "validation_dataset_path": binding["validation_path"].relative_to(manifest_root).as_posix(),
        "validation_dataset_sha256": sha256_file(binding["validation_path"]),
        "output_dir": str(Path(config.output_dir).resolve()),
        "split_manifest_path": "manifest.json",
        "split_manifest_sha256": sha256_file(binding["manifest_path"]),
        "split_manifest_digest": manifest["manifest_digest"],
        "split_manifest_schema_version": manifest["schema_version"],
        "split_policy_version": manifest["split_policy_version"],
        "split_seed": manifest["split_seed"],
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
        ToMBeliefBackboneConfig(max_seq_len=config.max_seq_len),
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
    dataset_contract = _training_dataset_contract(train_dataset, validation_dataset)
    is_canonical_d = dataset_contract["source_schema_version"] == D_SCHEMA_VERSION
    if is_canonical_d:
        if config.resolved_split_manifest_path is None:
            raise ValueError("canonical D training requires --split-manifest")
        validate_canonical_d_training_contract(
            config, train_dataset, validation_dataset
        )
    elif config.resolved_split_manifest_path is not None:
        raise ValueError("--split-manifest is restricted to canonical D V1 training")
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
    if dataset_contract["source_schema_version"] == D_SCHEMA_VERSION:
        return {
            "schema_version": D_SCHEMA_VERSION,
            "belief_information_scope": dataset_contract["belief_information_scope"],
            "private_fields_usage": dataset_contract["private_fields_usage"],
            "annotation_schema_version": dataset_contract["annotation_schema_version"],
            "label_provenance": dataset_contract["label_provenance"],
            "source_label_provenance": dataset_contract["source_label_provenance"],
        }
    if dataset_contract["belief_information_scope"] != (
        PUBLIC_ONLY_BELIEF_INFORMATION_SCOPE
    ):
        return {}
    return {
        "schema_version": dataset_contract["source_schema_version"],
        "belief_information_scope": dataset_contract["belief_information_scope"],
        "private_fields_usage": dataset_contract["private_fields_usage"],
        "annotation_schema_version": dataset_contract["annotation_schema_version"],
        "label_provenance": dataset_contract["label_provenance"],
        "source_label_provenance": dataset_contract["source_label_provenance"],
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
        self.metric_sums: dict[str, float] = {}
        self.metric_weights: dict[str, int] = {}

    def update(
        self,
        *,
        loss: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        subject_mask: torch.Tensor,
    ) -> None:
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
        loss.backward()
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
        accumulator.update(
            loss=loss,
            logits=logits,
            targets=effective_targets,
            subject_mask=effective_subject_mask,
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
    training_manifest: Mapping[str, Any] | None = None,
    training_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    if dataset_contract is None:
        dataset_contract = {
            "belief_information_scope": "private_conditioned",
            "source_schema_version": SAMPLE_SCHEMA_VERSION,
            "model_input_scope": TOM_INPUT_SCOPES[config.tom_order],
        }
    serialized_training_config = asdict(config)
    if config.split_manifest_path is None:
        serialized_training_config.pop("split_manifest_path", None)
    else:
        serialized_training_config["split_manifest_path"] = "manifest.json"
    selection_metric_value = float(validation_metrics["mean_loss"])
    canonical_d_metadata: dict[str, Any] = {}
    if dataset_contract["source_schema_version"] == D_SCHEMA_VERSION:
        if training_manifest is None or training_manifest_sha256 is None:
            raise ValueError(
                "canonical D checkpoint requires the written training manifest identity"
            )
        training_manifest = validate_training_manifest(training_manifest)
        _lower_hex(
            training_manifest_sha256, length=64,
            field_name="training_manifest_sha256",
        )
        if training_manifest.get("schema_version") != TRAINING_MANIFEST_SCHEMA_VERSION:
            raise ValueError("canonical D checkpoint training manifest schema mismatch")
        canonical_d_metadata = {
            "materialization_task": training_manifest["materialization_task"],
            "d_materialization_policy_version": training_manifest[
                "d_materialization_policy_version"
            ],
            "belief_information_scope": dataset_contract["belief_information_scope"],
            "private_fields_usage": dataset_contract["private_fields_usage"],
            "annotation_schema_version": dataset_contract["annotation_schema_version"],
            "label_provenance": dataset_contract["label_provenance"],
            "source_label_provenance": dataset_contract["source_label_provenance"],
            "training_manifest_schema_version": TRAINING_MANIFEST_SCHEMA_VERSION,
            "training_manifest_sha256": training_manifest_sha256,
            "training_manifest_digest": training_manifest["manifest_digest"],
            "split_manifest_schema_version": training_manifest["split_manifest_schema_version"],
            "split_manifest_sha256": training_manifest["split_manifest_sha256"],
            "split_policy_version": training_manifest["split_policy_version"],
            "split_seed": training_manifest["split_seed"],
        }
        if config.tom_order == 2:
            canonical_d_metadata.update(
                {
                    "tom2_target_semantics": TOM2_TARGET_SEMANTICS,
                    "tom2_temporal_supervision_policy": TOM2_TEMPORAL_SUPERVISION_POLICY,
                }
            )
    return {
        "schema_version": dataset_contract["source_schema_version"],
        "tom_order": config.tom_order,
        "model_input_scope": dataset_contract["model_input_scope"],
        **_public_only_lineage_metadata(dataset_contract),
        **canonical_d_metadata,
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
            **serialized_training_config,
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
    is_canonical_d = dataset_contract["source_schema_version"] == D_SCHEMA_VERSION
    if is_canonical_d and config.resolved_split_manifest_path is None:
        raise ValueError("canonical D training requires --split-manifest")
    if not is_canonical_d and config.resolved_split_manifest_path is not None:
        raise ValueError("--split-manifest is restricted to canonical D V1 training")

    canonical_context = None
    training_manifest = None
    training_manifest_sha256 = None
    if is_canonical_d:
        canonical_context = validate_canonical_d_training_contract(
            config, train_dataset, validation_dataset
        )
        training_manifest = build_training_manifest(
            config,
            resolved_device=device,
            run_provenance=run_provenance,
            dataset_contract=dataset_contract,
            canonical_context=canonical_context,
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
    if training_manifest is not None:
        training_manifest_path = output_dir / "training_manifest.json"
        _atomic_json_write(training_manifest, training_manifest_path)
        written_training_manifest = validate_training_manifest(
            json.loads(training_manifest_path.read_text(encoding="utf-8"))
        )
        if written_training_manifest != training_manifest:
            raise RuntimeError(
                "written training manifest differs from source manifest"
            )
        training_manifest_sha256 = sha256_file(training_manifest_path)
        run_provenance = {
            **run_provenance,
            "training_manifest_schema_version": TRAINING_MANIFEST_SCHEMA_VERSION,
            "training_manifest_sha256": training_manifest_sha256,
            "training_manifest_digest": training_manifest["manifest_digest"],
            "materialization_task": training_manifest["materialization_task"],
            "d_materialization_policy_version": D_MATERIALIZATION_POLICY_VERSION,
            "train_source_row_count": training_manifest["train_source_row_count"],
            "validation_source_row_count": training_manifest[
                "validation_source_row_count"
            ],
            "train_effective_supervised_snapshot_count": training_manifest[
                "train_effective_supervised_snapshot_count"
            ],
            "validation_effective_supervised_snapshot_count": training_manifest[
                "validation_effective_supervised_snapshot_count"
            ],
        }
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
                    training_manifest=training_manifest,
                    training_manifest_sha256=training_manifest_sha256,
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
            training_manifest=training_manifest,
            training_manifest_sha256=training_manifest_sha256,
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
    if training_manifest is not None:
        summary.update(
            {
                "training_manifest_schema_version": TRAINING_MANIFEST_SCHEMA_VERSION,
                "training_manifest_sha256": training_manifest_sha256,
                "training_manifest_digest": training_manifest["manifest_digest"],
                "split_manifest_sha256": training_manifest["split_manifest_sha256"],
                "split_manifest_schema_version": training_manifest["split_manifest_schema_version"],
                "split_policy_version": training_manifest["split_policy_version"],
                "split_seed": training_manifest["split_seed"],
                "materialization_task": training_manifest["materialization_task"],
                "d_materialization_policy_version": D_MATERIALIZATION_POLICY_VERSION,
                "train_source_row_count": training_manifest["train_source_row_count"],
                "validation_source_row_count": training_manifest["validation_source_row_count"],
                "train_effective_supervised_snapshot_count": training_manifest[
                    "train_effective_supervised_snapshot_count"
                ],
                "validation_effective_supervised_snapshot_count": training_manifest[
                    "validation_effective_supervised_snapshot_count"
                ],
                "tom2_target_semantics": training_manifest[
                    "tom2_target_semantics"
                ],
                "tom2_temporal_supervision_policy": training_manifest[
                    "tom2_temporal_supervision_policy"
                ],
            }
        )
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
        help=(
            "Root for a new, empty training run directory; Canonical D mode "
            "may place it outside the Git worktree."
        ),
    )
    parser.add_argument(
        "--dataset", required=True,
        help=(
            "Training JSONL file; legacy mode requires a repository-local path, "
            "Canonical D mode binds it to --split-manifest."
        ),
    )
    parser.add_argument(
        "--validation-dataset", required=True,
        help="Validation JSONL file with disjoint game IDs.",
    )
    parser.add_argument(
        "--split-manifest",
        default=None,
        help=(
            "Required Canonical D Split V1 manifest for canonical D training; "
            "legacy training leaves this unset."
        ),
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
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = run_training(
        TrainingConfig(
            tom_order=args.tom_order,
            output_dir=args.output_dir,
            dataset_path=args.dataset,
            validation_dataset_path=args.validation_dataset,
            split_manifest_path=args.split_manifest,
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
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
