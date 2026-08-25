"""Evaluate one tom-v2 observer-conditioned belief checkpoint."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from script.twd_tom.materialize_canonical_belief_dataset import (
    validate_materialized_split_path,
)
from script.twd_tom.train import (
    REPO_ROOT,
    bootstrap_game_macro_metric,
    count_supervised_observers,
    evaluate_model_with_games_and_strata,
    game_macro_metrics,
    resolve_device,
    sha256_file,
    stratified_game_macro_metrics,
)
from werewolf.models.twd_tom.action_features import PublicEventFeatureBuilder
from werewolf.models.twd_tom.checkpoint import (
    build_model_from_checkpoint,
    checkpoint_task_contract,
    load_checkpoint,
    result_model_config,
)
from werewolf.models.twd_tom.dataset import (
    TWDToMDataset,
    collate_twd_tom_samples,
    load_twd_tom_jsonl,
)
from werewolf.models.twd_tom.samples import SAMPLE_SCHEMA_VERSION
from werewolf.models.twd_tom.supervision import (
    ALL_ALIVE_SCOPE,
    load_role_sidecar,
    load_role_sidecar_report,
)


@dataclass(frozen=True)
class EvaluationConfig:
    checkpoint_path: str
    dataset_path: str
    output_path: str | None = None
    training_dataset_path: str | None = None
    role_sidecar_path: str | None = None
    batch_size: int = 32
    device: str = "auto"
    num_workers: int = 0

    def __post_init__(self) -> None:
        for field_name in ("checkpoint_path", "dataset_path"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} is required")
        for field_name in (
            "output_path",
            "training_dataset_path",
            "role_sidecar_path",
        ):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} must be non-empty text or None")
        if isinstance(self.batch_size, bool) or not isinstance(
            self.batch_size, int
        ) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be non-empty text")
        if isinstance(self.num_workers, bool) or not isinstance(
            self.num_workers, int
        ) or self.num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")


def collect_game_ids(samples: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise TypeError("samples must be a sequence")
    result: set[str] = set()
    for sample in samples:
        game_id = sample.get("game_id")
        if not isinstance(game_id, str) or not game_id.strip():
            raise ValueError("every sample requires a non-empty game_id")
        result.add(game_id)
    return tuple(sorted(result))


def resolve_training_dataset_path(
    checkpoint: Mapping[str, Any], *, override_path: str | None
) -> Path:
    provenance = checkpoint.get("run_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("checkpoint must record the training dataset identity")
    recorded_path = provenance.get("train_dataset_path")
    expected_sha256 = provenance.get("train_dataset_sha256")
    if not isinstance(recorded_path, str) or not recorded_path.strip() or (
        Path(recorded_path).is_absolute() or ".." in Path(recorded_path).parts
    ):
        raise ValueError("training dataset path must be repository-relative")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError("training dataset SHA-256 must be lowercase hexadecimal")
    path = (
        Path(override_path).resolve()
        if override_path is not None
        else (REPO_ROOT / recorded_path).resolve()
    )
    if not path.is_file():
        raise FileNotFoundError(f"training dataset not found: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"training dataset SHA-256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
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


def resolve_role_sidecar_path(
    checkpoint: Mapping[str, Any],
    *,
    override_path: str | None,
) -> Path | None:
    provenance = checkpoint.get("run_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("checkpoint must record run_provenance")
    recorded_path = provenance.get("role_sidecar_path")
    expected_sha256 = provenance.get("role_sidecar_sha256")
    if recorded_path is None:
        if override_path is not None:
            raise ValueError("checkpoint did not use role sidecar metadata")
        return None
    if (
        not isinstance(recorded_path, str)
        or Path(recorded_path).is_absolute()
        or ".." in Path(recorded_path).parts
    ):
        raise ValueError("role sidecar path must be repository-relative")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("checkpoint role sidecar SHA-256 is invalid")
    path = (
        Path(override_path).resolve()
        if override_path is not None
        else (REPO_ROOT / recorded_path).resolve()
    )
    if sha256_file(path) != expected_sha256:
        raise ValueError("role sidecar SHA-256 mismatch")
    return path


def evaluate_checkpoint(config: EvaluationConfig) -> dict[str, Any]:
    device = resolve_device(config.device)
    checkpoint_path = Path(config.checkpoint_path).resolve()
    dataset_path = Path(config.dataset_path).resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    model = build_model_from_checkpoint(checkpoint, device=device)
    evaluation_manifest = validate_materialized_split_path(
        dataset_path,
        split_name="test",
    )
    provenance = checkpoint.get("run_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("checkpoint must record run_provenance")
    if provenance.get("split_manifest_digest") != evaluation_manifest[
        "manifest_digest"
    ]:
        raise ValueError(
            "evaluation test split does not share the checkpoint split manifest"
        )
    samples = load_twd_tom_jsonl(dataset_path)
    evaluation_game_ids = collect_game_ids(samples)
    training_path = resolve_training_dataset_path(
        checkpoint, override_path=config.training_dataset_path
    )
    training_manifest = validate_materialized_split_path(
        training_path,
        split_name="train",
    )
    if training_manifest["manifest_digest"] != evaluation_manifest["manifest_digest"]:
        raise ValueError("training and evaluation data use different split manifests")
    training_game_ids = collect_game_ids(load_twd_tom_jsonl(training_path))
    validation_game_ids = tuple(
        evaluation_manifest["game_ids"]["validation"]
    )
    if set(evaluation_game_ids) != set(evaluation_manifest["game_ids"]["test"]):
        raise ValueError("evaluation dataset game IDs differ from manifest test split")
    if set(training_game_ids) != set(evaluation_manifest["game_ids"]["train"]):
        raise ValueError("training dataset game IDs differ from manifest train split")
    overlap = tuple(
        sorted(
            set(evaluation_game_ids)
            & (set(training_game_ids) | set(validation_game_ids))
        )
    )
    if overlap:
        raise ValueError(
            "evaluation game IDs must be disjoint from train and validation data; "
            f"overlap_count={len(overlap)}, examples={list(overlap[:5])}"
        )
    supervision_scope = checkpoint.get("supervision_scope", ALL_ALIVE_SCOPE)
    role_sidecar_path = resolve_role_sidecar_path(
        checkpoint,
        override_path=config.role_sidecar_path,
    )
    observer_roles = None
    if role_sidecar_path is not None:
        role_report = load_role_sidecar_report(role_sidecar_path)
        if role_report["split_manifest_digest"] != evaluation_manifest[
            "manifest_digest"
        ]:
            raise ValueError("role sidecar and evaluation split digests differ")
        observer_roles = load_role_sidecar(role_sidecar_path)
    dataset = TWDToMDataset(
        samples,
        feature_builder=PublicEventFeatureBuilder(max_seq_len=model.config.max_seq_len),
        include_private_features=model.config.private_conditioning,
        observer_roles_by_game=observer_roles,
        supervision_scope=supervision_scope,
    )
    if dataset.model_input_scope != checkpoint.get("model_input_scope"):
        raise ValueError("evaluation Dataset model_input_scope mismatch")
    if dataset.target_semantics != checkpoint.get("target_semantics"):
        raise ValueError("evaluation Dataset target_semantics mismatch")
    if dataset.target_conversion != checkpoint.get("target_conversion"):
        raise ValueError("evaluation Dataset target_conversion mismatch")
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_twd_tom_samples,
    )
    supervised = count_supervised_observers(loader)
    if supervised == 0:
        raise ValueError("evaluation dataset contains no valid observer targets")
    (
        metrics,
        by_game,
        stratified,
        stratified_by_game,
    ) = evaluate_model_with_games_and_strata(
        model,
        loader,
        device=device,
    )
    game_macro = game_macro_metrics(by_game)
    game_bootstrap = {
        name: bootstrap_game_macro_metric(
            by_game,
            metric_name=name,
            samples=2000,
            seed=42,
        )
        for name in (
            "normalized_reducible_gap_improvement",
            "private_admissible_normalized_reducible_gap_improvement",
        )
        if all(name in game_metrics for game_metrics in by_game.values())
    }
    epoch = checkpoint.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise ValueError("checkpoint has an invalid epoch")
    summary: dict[str, Any] = {
        "status": "ok",
        "schema_version": SAMPLE_SCHEMA_VERSION,
        **checkpoint_task_contract(model.config.private_conditioning),
        "backbone": model.backbone_name,
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": epoch,
        "evaluation_dataset_path": str(dataset_path),
        "evaluation_sample_count": len(dataset),
        "evaluation_game_ids": list(evaluation_game_ids),
        "evaluation_supervised_observer_count": supervised,
        "supervision_scope": supervision_scope,
        "role_metadata_usage": "supervision_metadata_only",
        "training_dataset_path": str(training_path),
        "training_game_ids": list(training_game_ids),
        "validation_game_ids": list(validation_game_ids),
        "overlapping_game_ids": list(overlap),
        "split_manifest_digest": evaluation_manifest["manifest_digest"],
        "model_config": result_model_config(model),
        "metrics": metrics,
        "metrics_by_game": by_game,
        "stratified_metrics": stratified,
        "stratified_by_game": stratified_by_game,
        "stratified_game_macro": stratified_game_macro_metrics(
            stratified_by_game
        ),
        "game_macro_metrics": game_macro,
        "game_bootstrap": game_bootstrap,
    }
    if config.output_path is not None:
        output_path = Path(config.output_path).resolve()
        _write_json(summary, output_path)
        summary["output_path"] = str(output_path)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a tom-v2 checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--training-dataset", default=None)
    parser.add_argument("--role-sidecar", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = evaluate_checkpoint(EvaluationConfig(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        output_path=args.output,
        training_dataset_path=args.training_dataset,
        role_sidecar_path=args.role_sidecar,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
    ))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
