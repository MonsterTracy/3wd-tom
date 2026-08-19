"""Evaluate an order-specific ToM checkpoint on current raw data."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from script.twd_tom.train import (
    REPO_ROOT,
    TOM2_TARGET_SEMANTICS,
    TOM2_TEMPORAL_SUPERVISION_POLICY,
    TRAINING_MANIFEST_SCHEMA_VERSION,
    canonical_d_dataset_metadata,
    canonical_d_order_name,
    checkpoint_task_contract,
    count_supervised_subjects,
    evaluate_model,
    load_canonical_d_split_manifest,
    result_model_config,
    resolve_device,
    sha256_file,
)
from werewolf.models.twd_tom.action_features import PublicEventFeatureBuilder
from werewolf.models.twd_tom.belief_backbone import (
    SUPPORTED_BACKBONE_NAMES,
    ToMBeliefBackbone,
    ToMBeliefBackboneConfig,
)
from werewolf.models.twd_tom.dataset import (
    D_PUBLIC_ONLY_TOM2_BELIEF_INFORMATION_SCOPE,
    PRIVATE_CONDITIONED_BELIEF_INFORMATION_SCOPE,
    TOM_INPUT_SCOPES,
    TWDToMDataset,
    collate_twd_tom_samples,
    load_twd_tom_jsonl,
)
from werewolf.models.twd_tom.public_events import (
    PHASE_TO_ID,
    PUBLIC_EVENT_SCHEMA_VERSION,
    STRUCTURED_TOKEN_TO_ID,
)
from werewolf.models.twd_tom.samples import (
    PUBLIC_ONLY_SAMPLE_SCHEMA_VERSION,
    SAMPLE_SCHEMA_VERSION,
)
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
from script.twd_tom.split_offline_d_training_data import (
    SPLIT_MANIFEST_SCHEMA_VERSION,
    SPLIT_POLICY_VERSION,
)
from werewolf.models.twd_tom.schema import (
    ACTION_NAMES,
    ACTION_TO_ID,
    PUBLIC_ONLY_BELIEF_INFORMATION_SCOPE,
    PUBLIC_ONLY_FORMAL_ANNOTATION_SCHEMA_VERSION,
    PUBLIC_ONLY_FORMAL_LABEL_PROVENANCE,
    PUBLIC_ONLY_LABEL_PROVENANCE,
    PUBLIC_ONLY_MODEL_INPUT_SCOPE,
    PUBLIC_ONLY_PRIVATE_FIELDS_USAGE,
)


@dataclass(frozen=True)
class EvaluationConfig:
    checkpoint_path: str
    dataset_path: str
    output_path: str | None = None
    training_dataset_path: str | None = None
    batch_size: int = 32
    device: str = "auto"
    num_workers: int = 0
    split_manifest_path: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("checkpoint_path", "dataset_path"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} is required")
        for field_name in (
            "output_path",
            "training_dataset_path",
            "split_manifest_path",
        ):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{field_name} must be non-empty text or None")
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be non-empty text")
        if (
            isinstance(self.num_workers, bool)
            or not isinstance(self.num_workers, int)
            or self.num_workers < 0
        ):
            raise ValueError("num_workers must be a non-negative integer")


def load_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain a dictionary")
    return checkpoint


def _checkpoint_tom_order(checkpoint: Mapping[str, Any]) -> int:
    value = checkpoint.get("tom_order")
    if isinstance(value, bool) or not isinstance(value, int) or value not in (1, 2):
        raise ValueError("checkpoint tom_order must be 1 or 2")
    return value


def _require_sha256(value: Any, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"checkpoint {field_name} must be a lowercase SHA-256")
    return value


def _canonical_d_checkpoint_contract(tom_order: int) -> dict[str, Any]:
    if tom_order == 1:
        return {
            "schema_version": D_SCHEMA_VERSION,
            "model_input_scope": TOM1_MODEL_INPUT_SCOPE,
            "belief_information_scope": PRIVATE_CONDITIONED_BELIEF_INFORMATION_SCOPE,
            "private_fields_usage": TOM1_PRIVATE_FIELDS_USAGE,
            "annotation_schema_version": OFFLINE_ANNOTATION_SCHEMA_VERSION,
            "label_provenance": TOM1_OBSERVER_PROVENANCE,
            "source_label_provenance": TOM1_OBSERVER_PROVENANCE,
            "materialization_task": OFFLINE_PRIVATE_CONDITIONED_TOM1_TASK,
            "d_materialization_policy_version": D_MATERIALIZATION_POLICY_VERSION,
            "training_manifest_schema_version": TRAINING_MANIFEST_SCHEMA_VERSION,
            "split_manifest_schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
            "split_policy_version": SPLIT_POLICY_VERSION,
        }
    return {
        "schema_version": D_SCHEMA_VERSION,
        "model_input_scope": TOM2_MODEL_INPUT_SCOPE,
        "belief_information_scope": D_PUBLIC_ONLY_TOM2_BELIEF_INFORMATION_SCOPE,
        "private_fields_usage": TOM2_PRIVATE_FIELDS_USAGE,
        "annotation_schema_version": OFFLINE_ANNOTATION_SCHEMA_VERSION,
        "label_provenance": TOM2_OBSERVER_PROVENANCE,
        "source_label_provenance": TOM2_OBSERVER_PROVENANCE,
        "materialization_task": OFFLINE_PUBLIC_ONLY_TOM2_TASK,
        "d_materialization_policy_version": D_MATERIALIZATION_POLICY_VERSION,
        "training_manifest_schema_version": TRAINING_MANIFEST_SCHEMA_VERSION,
        "split_manifest_schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "split_policy_version": SPLIT_POLICY_VERSION,
        "tom2_target_semantics": TOM2_TARGET_SEMANTICS,
        "tom2_temporal_supervision_policy": TOM2_TEMPORAL_SUPERVISION_POLICY,
    }


def _dataset_lineage(dataset: TWDToMDataset) -> dict[str, Any]:
    return {
        "schema_version": dataset.source_schema_version,
        "belief_information_scope": dataset.belief_information_scope,
        "model_input_scope": dataset.model_input_scope,
        "private_fields_usage": dataset.private_fields_usage,
        "annotation_schema_version": dataset.annotation_schema_version,
        "label_provenance": dataset.label_provenance,
        "source_label_provenance": dataset.source_label_provenance,
    }


def _validate_evaluation_dataset_lineage(
    checkpoint: Mapping[str, Any],
    dataset: TWDToMDataset,
) -> dict[str, Any]:
    lineage = _dataset_lineage(dataset)
    schema_version = checkpoint.get("schema_version")
    if lineage["schema_version"] != schema_version:
        raise ValueError(
            "evaluation Dataset schema_version does not match checkpoint lineage"
        )
    fields = ["model_input_scope"]
    if schema_version in {PUBLIC_ONLY_SAMPLE_SCHEMA_VERSION, D_SCHEMA_VERSION}:
        fields.extend(
            [
                "belief_information_scope",
                "private_fields_usage",
                "annotation_schema_version",
                "label_provenance",
                "source_label_provenance",
            ]
        )
    for field_name in fields:
        if lineage[field_name] != checkpoint.get(field_name):
            raise ValueError(
                f"evaluation Dataset {field_name} does not match checkpoint lineage"
            )
    if schema_version == D_SCHEMA_VERSION:
        metadata = canonical_d_dataset_metadata(dataset)
        if metadata["materialization_task"] != checkpoint.get("materialization_task"):
            raise ValueError(
                "evaluation Dataset materialization_task does not match checkpoint lineage"
            )
        if metadata["materialization_policy_version"] != checkpoint.get(
            "d_materialization_policy_version"
        ):
            raise ValueError(
                "evaluation Dataset materialization policy does not match checkpoint lineage"
            )
        lineage["materialization_task"] = metadata["materialization_task"]
        lineage["d_materialization_policy_version"] = metadata[
            "materialization_policy_version"
        ]
    return lineage


def _resolve_canonical_d_evaluation(
    config: EvaluationConfig,
    checkpoint: Mapping[str, Any],
    *,
    tom_order: int,
) -> dict[str, Any]:
    if config.split_manifest_path is None:
        raise ValueError("canonical D evaluation requires --split-manifest")
    if config.training_dataset_path is not None:
        raise ValueError(
            "canonical D evaluation uses split provenance, not --training-dataset"
        )
    manifest_path = Path(config.split_manifest_path).resolve()
    manifest = load_canonical_d_split_manifest(manifest_path)
    actual_manifest_sha = sha256_file(manifest_path)
    if actual_manifest_sha != checkpoint.get("split_manifest_sha256"):
        raise ValueError("split manifest SHA-256 does not match checkpoint")
    for field_name, actual in (
        ("split_manifest_schema_version", manifest["schema_version"]),
        ("split_policy_version", manifest["split_policy_version"]),
        ("split_seed", manifest["split_seed"]),
    ):
        if checkpoint.get(field_name) != actual:
            raise ValueError(f"checkpoint {field_name} does not match split manifest")

    order_name = canonical_d_order_name(tom_order)
    dataset_path = Path(config.dataset_path).resolve()
    manifest_root = manifest_path.parent
    candidates = {
        split_name: (manifest_root / order_name / f"{split_name}.jsonl").resolve()
        for split_name in ("validation", "test")
    }
    evaluation_split = next(
        (name for name, path in candidates.items() if path == dataset_path),
        None,
    )
    train_path = (manifest_root / order_name / "train.jsonl").resolve()
    if dataset_path == train_path:
        raise ValueError("canonical D train split cannot be used for evaluation")
    if evaluation_split is None:
        raise ValueError(
            "canonical D evaluation dataset must be the manifest validation or test JSONL"
        )
    expected_sha = manifest["splits"][evaluation_split][f"{order_name}_file_sha256"]
    actual_sha = sha256_file(dataset_path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"canonical D {evaluation_split} dataset SHA-256 mismatch: "
            f"expected {expected_sha}, got {actual_sha}"
        )
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": actual_manifest_sha,
        "evaluation_split": evaluation_split,
        "training_game_ids": tuple(sorted(manifest["game_ids"]["train"])),
        "expected_evaluation_game_ids": tuple(
            sorted(manifest["game_ids"][evaluation_split])
        ),
    }


def build_model_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    device: torch.device,
) -> ToMBeliefBackbone:
    """Strictly reconstruct the explicitly recorded backbone."""

    tom_order = _checkpoint_tom_order(checkpoint)
    backbone_name = checkpoint.get("backbone")
    if backbone_name not in SUPPORTED_BACKBONE_NAMES:
        raise ValueError(
            "checkpoint backbone mismatch: expected one of "
            f"{SUPPORTED_BACKBONE_NAMES!r}, got {backbone_name!r}"
        )
    schema_version = checkpoint.get("schema_version")
    if schema_version == SAMPLE_SCHEMA_VERSION:
        lineage_contract = {
            "schema_version": SAMPLE_SCHEMA_VERSION,
            "model_input_scope": TOM_INPUT_SCOPES[tom_order],
        }
    elif schema_version == PUBLIC_ONLY_SAMPLE_SCHEMA_VERSION:
        lineage_contract = {
            "schema_version": PUBLIC_ONLY_SAMPLE_SCHEMA_VERSION,
            "model_input_scope": PUBLIC_ONLY_MODEL_INPUT_SCOPE,
            "belief_information_scope": PUBLIC_ONLY_BELIEF_INFORMATION_SCOPE,
            "private_fields_usage": PUBLIC_ONLY_PRIVATE_FIELDS_USAGE,
            "annotation_schema_version": (
                PUBLIC_ONLY_FORMAL_ANNOTATION_SCHEMA_VERSION
            ),
            "label_provenance": PUBLIC_ONLY_FORMAL_LABEL_PROVENANCE,
            "source_label_provenance": PUBLIC_ONLY_LABEL_PROVENANCE,
        }
    elif schema_version == D_SCHEMA_VERSION:
        lineage_contract = _canonical_d_checkpoint_contract(tom_order)
        for field_name in (
            "training_manifest_sha256",
            "training_manifest_digest",
            "split_manifest_sha256",
        ):
            _require_sha256(checkpoint.get(field_name), field_name=field_name)
        split_seed = checkpoint.get("split_seed")
        if isinstance(split_seed, bool) or not isinstance(split_seed, int):
            raise ValueError("checkpoint split_seed must be an integer")
    else:
        raise ValueError(
            "checkpoint schema_version is not a supported formal lineage: "
            f"{schema_version!r}"
        )
    expected = {
        **lineage_contract,
        "public_event_schema_version": PUBLIC_EVENT_SCHEMA_VERSION,
        "speech_action_count": len(ACTION_NAMES),
        "speech_action_to_id": dict(ACTION_TO_ID),
        "structured_token_to_id": dict(STRUCTURED_TOKEN_TO_ID),
        "public_phase_to_id": dict(PHASE_TO_ID),
        **checkpoint_task_contract(tom_order),
    }
    for field_name, expected_value in expected.items():
        if checkpoint.get(field_name) != expected_value:
            raise ValueError(
                f"checkpoint {field_name} mismatch: expected {expected_value!r}, "
                f"got {checkpoint.get(field_name)!r}"
            )
    raw_model_config = checkpoint.get("model_config")
    if not isinstance(raw_model_config, Mapping):
        raise TypeError("checkpoint has no valid model_config")
    try:
        model_config = ToMBeliefBackboneConfig(**dict(raw_model_config))
    except TypeError as exc:
        raise ValueError("checkpoint model_config is incompatible") from exc
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise TypeError("checkpoint has no valid model_state_dict")
    model = ToMBeliefBackbone(
        model_config,
        tom_order=tom_order,
        backbone_name=backbone_name,
    )
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError("checkpoint state_dict is incompatible") from exc
    return model.to(device).eval()


def collect_game_ids(samples: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise TypeError("samples must be a sequence")
    result = set()
    for sample in samples:
        game_id = sample.get("game_id")
        if not isinstance(game_id, str) or not game_id.strip():
            raise ValueError("every sample requires a non-empty game_id")
        result.add(game_id)
    return tuple(sorted(result))


def resolve_training_dataset_path(
    checkpoint: Mapping[str, Any],
    *,
    override_path: str | None,
) -> Path:
    provenance = checkpoint.get("run_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(
            "checkpoint run_provenance must record the training dataset identity"
        )
    recorded_path = provenance.get("train_dataset_path")
    expected_sha256 = provenance.get("train_dataset_sha256")
    if (
        not isinstance(recorded_path, str)
        or not recorded_path.strip()
        or Path(recorded_path).is_absolute()
        or ".." in Path(recorded_path).parts
    ):
        raise ValueError(
            "checkpoint run_provenance.train_dataset_path must be a safe "
            "repository-relative path"
        )
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError(
            "checkpoint run_provenance.train_dataset_sha256 must be a "
            "lowercase SHA-256 digest"
        )
    if override_path is not None:
        path = Path(override_path).resolve()
    else:
        path = (REPO_ROOT / recorded_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"training dataset not found: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "training dataset SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    return path


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def evaluate_checkpoint(config: EvaluationConfig) -> dict[str, Any]:
    device = resolve_device(config.device)
    checkpoint_path = Path(config.checkpoint_path).resolve()
    dataset_path = Path(config.dataset_path).resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    tom_order = _checkpoint_tom_order(checkpoint)
    model = build_model_from_checkpoint(checkpoint, device=device)
    schema_version = checkpoint.get("schema_version")

    canonical_eval = None
    if schema_version == D_SCHEMA_VERSION:
        canonical_eval = _resolve_canonical_d_evaluation(
            config, checkpoint, tom_order=tom_order
        )
    elif config.split_manifest_path is not None:
        raise ValueError("--split-manifest is restricted to canonical D checkpoints")

    samples = load_twd_tom_jsonl(dataset_path)
    evaluation_game_ids = collect_game_ids(samples)
    if canonical_eval is not None:
        if evaluation_game_ids != canonical_eval["expected_evaluation_game_ids"]:
            raise ValueError(
                "evaluation Dataset game IDs differ from split manifest assignment"
            )
        training_path = None
        training_game_ids = canonical_eval["training_game_ids"]
    else:
        training_path = resolve_training_dataset_path(
            checkpoint, override_path=config.training_dataset_path
        )
        training_game_ids = collect_game_ids(load_twd_tom_jsonl(training_path))

    overlap = tuple(sorted(set(evaluation_game_ids) & set(training_game_ids)))
    if overlap:
        raise ValueError(
            "evaluation game IDs must be disjoint from training data; "
            f"overlap_count={len(overlap)}, examples={list(overlap[:5])}"
        )

    dataset = TWDToMDataset(
        samples,
        tom_order=tom_order,
        feature_builder=PublicEventFeatureBuilder(max_seq_len=model.config.max_seq_len),
    )
    lineage = _validate_evaluation_dataset_lineage(checkpoint, dataset)
    loader_dataset = dataset
    effective_snapshot_count = len(dataset)
    if tom_order == 2:
        supervised_indices = dataset.second_order_supervised_indices()
        if not supervised_indices:
            raise ValueError(
                "evaluation dataset contains no post-speech other-player targets"
            )
        effective_snapshot_count = len(supervised_indices)
        loader_dataset = Subset(dataset, supervised_indices)
    loader = DataLoader(
        loader_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_twd_tom_samples,
    )
    supervised = count_supervised_subjects(loader)
    if supervised == 0:
        raise ValueError("evaluation dataset contains no valid observer targets")
    metrics = evaluate_model(model, loader, device=device)
    epoch = checkpoint.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise ValueError("checkpoint has an invalid epoch")
    summary: dict[str, Any] = {
        "status": "ok",
        "schema_version": lineage["schema_version"],
        "tom_order": tom_order,
        "model_input_scope": lineage["model_input_scope"],
        **checkpoint_task_contract(tom_order),
        "backbone": model.backbone_name,
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": epoch,
        "evaluation_dataset_path": str(dataset_path),
        "evaluation_sample_count": len(dataset),
        "evaluation_game_ids": list(evaluation_game_ids),
        "evaluation_supervised_subject_count": supervised,
        "training_dataset_path": str(training_path) if training_path is not None else None,
        "training_game_ids": list(training_game_ids),
        "overlapping_game_ids": list(overlap),
        "model_config": result_model_config(model),
        "metrics": metrics,
    }
    if schema_version in {PUBLIC_ONLY_SAMPLE_SCHEMA_VERSION, D_SCHEMA_VERSION}:
        for field_name in (
            "belief_information_scope",
            "private_fields_usage",
            "annotation_schema_version",
            "label_provenance",
            "source_label_provenance",
        ):
            summary[field_name] = lineage[field_name]
    if canonical_eval is not None:
        summary.update(
            {
                "materialization_task": lineage["materialization_task"],
                "d_materialization_policy_version": lineage[
                    "d_materialization_policy_version"
                ],
                "split_manifest_sha256": canonical_eval["manifest_sha256"],
                "evaluation_split": canonical_eval["evaluation_split"],
                "evaluation_source_sample_count": len(dataset),
                "evaluation_effective_supervised_snapshot_count": (
                    effective_snapshot_count
                ),
                "tom2_target_semantics": (
                    TOM2_TARGET_SEMANTICS if tom_order == 2 else None
                ),
                "tom2_temporal_supervision_policy": (
                    TOM2_TEMPORAL_SUPERVISION_POLICY if tom_order == 2 else None
                ),
            }
        )
    if config.output_path is not None:
        output_path = Path(config.output_path).resolve()
        _write_json(summary, output_path)
        summary["output_path"] = str(output_path)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a ToM checkpoint.")
    parser.add_argument(
        "--checkpoint", required=True,
        help="Order-specific best.pt or last.pt checkpoint.",
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Validation or test JSONL file to evaluate.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--training-dataset", default=None,
        help="Optional location of the exact hashed legacy training JSONL.",
    )
    parser.add_argument(
        "--split-manifest",
        default=None,
        help="Required matching Canonical D Split V1 manifest for D checkpoints.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device", default="auto",
        help="Torch device or auto (CUDA, then MPS, then CPU).",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = evaluate_checkpoint(
        EvaluationConfig(
            checkpoint_path=args.checkpoint,
            dataset_path=args.dataset,
            output_path=args.output,
            training_dataset_path=args.training_dataset,
            split_manifest_path=args.split_manifest,
            batch_size=args.batch_size,
            device=args.device,
            num_workers=args.num_workers,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
