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
    checkpoint_task_contract,
    count_supervised_subjects,
    evaluate_model,
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
from werewolf.models.twd_tom.samples import SAMPLE_SCHEMA_VERSION


@dataclass(frozen=True)
class EvaluationConfig:
    checkpoint_path: str
    dataset_path: str
    output_path: str | None = None
    training_dataset_path: str | None = None
    batch_size: int = 32
    device: str = "auto"
    num_workers: int = 0

    def __post_init__(self) -> None:
        for field_name in ("checkpoint_path", "dataset_path"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} is required")
        for field_name in ("output_path", "training_dataset_path"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{field_name} must be non-empty text or None")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be non-empty text")
        if isinstance(self.num_workers, bool) or not isinstance(self.num_workers, int) or self.num_workers < 0:
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
    expected = {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "model_input_scope": TOM_INPUT_SCOPES[tom_order],
        "public_event_schema_version": PUBLIC_EVENT_SCHEMA_VERSION,
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
    samples = load_twd_tom_jsonl(dataset_path)
    evaluation_game_ids = collect_game_ids(samples)
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
    loader_dataset = dataset
    if tom_order == 2:
        supervised_indices = dataset.second_order_supervised_indices()
        if not supervised_indices:
            raise ValueError(
                "evaluation dataset contains no post-speech other-player targets"
            )
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
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "tom_order": tom_order,
        "model_input_scope": TOM_INPUT_SCOPES[tom_order],
        **checkpoint_task_contract(tom_order),
        "backbone": model.backbone_name,
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": epoch,
        "evaluation_dataset_path": str(dataset_path),
        "evaluation_sample_count": len(dataset),
        "evaluation_game_ids": list(evaluation_game_ids),
        "evaluation_supervised_subject_count": supervised,
        "training_dataset_path": str(training_path),
        "training_game_ids": list(training_game_ids),
        "overlapping_game_ids": list(overlap),
        "model_config": result_model_config(model),
        "metrics": metrics,
    }
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
        help="Optional location of the exact hashed training JSONL.",
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
            batch_size=args.batch_size,
            device=args.device,
            num_workers=args.num_workers,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
