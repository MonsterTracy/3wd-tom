"""Leakage-safe final fit on the frozen 54-game development set."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch
import transformers
from torch.optim import AdamW
from torch.utils.data import DataLoader

from script.twd_tom.materialize_development_folds import (
    DEVELOPMENT_FOLD_MANIFEST_FILENAME,
    validate_development_fold_paths,
)
from script.twd_tom.run_development_oof import OOF_SUMMARY_SCHEMA_VERSION
from script.twd_tom.train import (
    REPO_ROOT,
    _atomic_json_write,
    _atomic_torch_save,
    _prepare_run_output_dir,
    _repository_relative_path,
    build_learning_rate_scheduler,
    resolve_device,
    set_random_seed,
    sha256_file,
    train_one_epoch,
)
from werewolf.models.twd_tom.action_features import PublicEventFeatureBuilder
from werewolf.models.twd_tom.annotation_v2 import (
    V1_ANNOTATION_SOURCE,
    V1_EMPTY_UNOBSERVED_BELIEF_SOURCE,
)
from werewolf.models.twd_tom.belief_backbone import (
    NO_PHASE_DAY_INPUT_FEATURE_PROFILE,
    QWEN2_BACKBONE_NAME,
    ToMBeliefBackbone,
    ToMBeliefBackboneConfig,
)
from werewolf.models.twd_tom.checkpoint import (
    checkpoint_task_contract,
    result_model_config,
)
from werewolf.models.twd_tom.dense_dataset import (
    DENSE_SUPERVISION_VERSION,
    DenseTWDToMDataset,
    collate_dense_twd_tom_games,
)
from werewolf.models.twd_tom.dataset import load_twd_tom_jsonl
from werewolf.models.twd_tom.public_events import (
    PHASE_TO_ID,
    PUBLIC_EVENT_SCHEMA_VERSION,
    STRUCTURED_TOKEN_TO_ID,
)
from werewolf.models.twd_tom.samples import SAMPLE_SCHEMA_VERSION
from werewolf.models.twd_tom.schema import ACTION_NAMES, ACTION_TO_ID
from werewolf.models.twd_tom.supervision import (
    NON_WOLF_ALIVE_SCOPE,
    load_role_sidecar_report,
)
from werewolf.trajectory import canonical_digest


FINAL_PROTOCOL_SCHEMA_VERSION = "classic7_tom_v2_final_fit_protocol_v1"
FINAL_CHECKPOINT_TYPE = "development_final_fit_v1"
FINAL_CHECKPOINT_FILENAME = "final.pt"
FINAL_PROTOCOL_FILENAME = "final_protocol.json"
FINAL_SUMMARY_FILENAME = "final_summary.json"
FINAL_HISTORY_FILENAME = "history.json"
EXPECTED_FOLD_BEST_EPOCHS = (38, 30, 42, 21, 26)
EXPECTED_DEVELOPMENT_GAME_COUNT = 54
EXPECTED_SEALED_GAME_COUNT = 6
EPOCH_SELECTION_RULE = "median_of_five_oof_fold_best_epochs"


@dataclass(frozen=True)
class FinalFitConfig:
    """Paths and runtime device for the otherwise immutable final protocol."""

    fold_root: str
    role_sidecar_path: str
    oof_summary_path: str
    output_dir: str
    device: str = "auto"

    backbone: ClassVar[str] = QWEN2_BACKBONE_NAME
    input_feature_profile: ClassVar[str] = NO_PHASE_DAY_INPUT_FEATURE_PROFILE
    speech_annotation_source: ClassVar[str] = V1_ANNOTATION_SOURCE
    belief_annotation_source: ClassVar[str] = V1_EMPTY_UNOBSERVED_BELIEF_SOURCE
    supervision_scope: ClassVar[str] = NON_WOLF_ALIVE_SCOPE
    dense_supervision: ClassVar[bool] = True
    private_conditioning: ClassVar[bool] = False
    epochs: ClassVar[int] = 30
    scheduler_horizon_epochs: ClassVar[int] = 80
    batch_size: ClassVar[int] = 8
    learning_rate: ClassVar[float] = 1e-4
    min_learning_rate: ClassVar[float] = 1e-5
    warmup_ratio: ClassVar[float] = 0.05
    lr_scheduler: ClassVar[str] = "warmup_cosine"
    weight_decay: ClassVar[float] = 1e-2
    gradient_clip_norm: ClassVar[float] = 1.0
    max_seq_len: ClassVar[int] = 256
    seed: ClassVar[int] = 42
    num_workers: ClassVar[int] = 0

    def __post_init__(self) -> None:
        for field_name in (
            "fold_root",
            "role_sidecar_path",
            "oof_summary_path",
            "output_dir",
            "device",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be non-empty text")


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required JSON file not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON file must contain one object: {path}")
    return value


def _clean_git_commit(repo_root: Path) -> str:
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
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--short", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"final fit requires a readable Git worktree: {root}"
        ) from exc
    if Path(os.path.abspath(top_level)) != Path(os.path.abspath(root)):
        raise RuntimeError("final fit must use the repository root")
    if not commit:
        raise RuntimeError("final fit requires a committed Git HEAD")
    if dirty:
        raise RuntimeError(
            "final fit requires a clean Git worktree; dirty files:\n"
            + "\n".join(dirty)
        )
    return commit


def _frozen_training_config(config: FinalFitConfig) -> dict[str, Any]:
    return {
        "backbone": config.backbone,
        "input_feature_profile": config.input_feature_profile,
        "speech_annotation_source": config.speech_annotation_source,
        "belief_annotation_source": config.belief_annotation_source,
        "supervision_scope": config.supervision_scope,
        "dense_supervision": config.dense_supervision,
        "private_conditioning": config.private_conditioning,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "lr_scheduler": config.lr_scheduler,
        "min_learning_rate": config.min_learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "gradient_clip_norm": config.gradient_clip_norm,
        "max_seq_len": config.max_seq_len,
        "fit_epochs": config.epochs,
        "scheduler_horizon_epochs": config.scheduler_horizon_epochs,
        "seed": config.seed,
        "validation_dataset_used": False,
        "early_stopping_enabled": False,
    }


def _validate_oof_summary(
    oof_summary: Mapping[str, Any],
    *,
    fold_manifest: Mapping[str, Any],
    config: FinalFitConfig,
) -> list[dict[str, Any]]:
    if oof_summary.get("schema_version") != OOF_SUMMARY_SCHEMA_VERSION:
        raise ValueError("OOF summary schema mismatch")
    if oof_summary.get("status") != "ok":
        raise ValueError("OOF summary is not complete")
    if oof_summary.get("evaluation_scope") != "development_oof_only":
        raise ValueError("OOF summary is not development-only")
    if oof_summary.get("test_evaluated") is not False:
        raise ValueError("OOF summary must record test_evaluated=false")
    if oof_summary.get("oof_game_count") != EXPECTED_DEVELOPMENT_GAME_COUNT:
        raise ValueError("OOF summary must cover exactly 54 development games")
    if oof_summary.get("sealed_test_game_count") != EXPECTED_SEALED_GAME_COUNT:
        raise ValueError("OOF summary must declare exactly 6 sealed games")
    if oof_summary.get("source_split_manifest_digest") != fold_manifest.get(
        "source_split_manifest_digest"
    ):
        raise ValueError("OOF and source split manifest digests differ")
    if oof_summary.get("development_fold_manifest_digest") != fold_manifest.get(
        "manifest_digest"
    ):
        raise ValueError("OOF and development fold manifest digests differ")

    requested = oof_summary.get("training_config")
    if not isinstance(requested, Mapping):
        raise ValueError("OOF summary has no training_config")
    expected = {
        "epochs": config.scheduler_horizon_epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "lr_scheduler": config.lr_scheduler,
        "warmup_ratio": config.warmup_ratio,
        "min_learning_rate": config.min_learning_rate,
        "seed": config.seed,
        "max_seq_len": config.max_seq_len,
        "backbone": config.backbone,
        "input_feature_profile": config.input_feature_profile,
        "dense_supervision": config.dense_supervision,
        "supervision_scope": config.supervision_scope,
        "speech_annotation_source": config.speech_annotation_source,
        "belief_annotation_source": config.belief_annotation_source,
    }
    for field_name, expected_value in expected.items():
        if requested.get(field_name) != expected_value:
            raise ValueError(
                f"OOF training config mismatch for {field_name}: "
                f"{requested.get(field_name)!r} != {expected_value!r}"
            )
    if requested.get("private_conditioning", False) is not False:
        raise ValueError("OOF summary must be public-only")
    raw_folds = oof_summary.get("folds")
    manifest_folds = fold_manifest.get("folds")
    if not isinstance(raw_folds, Mapping) or not isinstance(manifest_folds, Mapping):
        raise ValueError("OOF or development manifest has no fold mapping")
    if set(raw_folds) != set(manifest_folds):
        raise ValueError("OOF fold set differs from the development manifest")
    fold_names = sorted(
        manifest_folds,
        key=lambda name: manifest_folds[name]["fold_index"],
    )
    fold_records = [
        {
            "fold_name": fold_name,
            "fold_index": manifest_folds[fold_name]["fold_index"],
            "best_epoch": raw_folds[fold_name].get("best_epoch"),
        }
        for fold_name in fold_names
    ]
    best_epochs = tuple(record["best_epoch"] for record in fold_records)
    if best_epochs != EXPECTED_FOLD_BEST_EPOCHS:
        raise ValueError(
            "OOF fold best epochs differ from the frozen selection source: "
            f"{best_epochs!r}"
        )
    sorted_epochs = sorted(EXPECTED_FOLD_BEST_EPOCHS)
    median_epoch = sorted_epochs[len(sorted_epochs) // 2]
    if median_epoch != config.epochs:
        raise AssertionError("frozen fit epoch is not the OOF median")
    return fold_records


def _load_development_records(
    fold_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    manifest_path = fold_root / DEVELOPMENT_FOLD_MANIFEST_FILENAME
    manifest = _load_json_object(manifest_path)
    raw_folds = manifest.get("folds")
    if not isinstance(raw_folds, Mapping) or not raw_folds:
        raise ValueError("development manifest has no folds")
    fold_names = sorted(
        raw_folds,
        key=lambda name: raw_folds[name]["fold_index"],
    )
    source_fold_name = fold_names[0]
    source_fold = raw_folds[source_fold_name]
    train_path = fold_root / source_fold["train_file"]["relative_path"]
    heldout_path = fold_root / source_fold["validation_file"]["relative_path"]
    validated = validate_development_fold_paths(train_path, heldout_path)
    if validated.get("manifest_digest") != manifest.get("manifest_digest"):
        raise ValueError("validated development manifest digest mismatch")

    shards = [
        {
            "source_fold": source_fold_name,
            "source_role": "development_shard_a",
            "path": train_path,
            "sha256": sha256_file(train_path),
            "row_count": source_fold["train_row_count"],
        },
        {
            "source_fold": source_fold_name,
            "source_role": "development_shard_b",
            "path": heldout_path,
            "sha256": sha256_file(heldout_path),
            "row_count": source_fold["validation_row_count"],
        },
    ]
    records = sorted(
        load_twd_tom_jsonl(train_path) + load_twd_tom_jsonl(heldout_path),
        key=lambda item: (item["game_id"], item["step_idx"]),
    )
    development_ids = {record["game_id"] for record in records}
    expected_development_ids = set(manifest.get("development_game_ids", ()))
    sealed_ids = set(manifest.get("sealed_test_game_ids", ()))
    if len(expected_development_ids) != EXPECTED_DEVELOPMENT_GAME_COUNT:
        raise ValueError("development manifest must declare exactly 54 games")
    if len(sealed_ids) != EXPECTED_SEALED_GAME_COUNT:
        raise ValueError("development manifest must declare exactly 6 sealed games")
    if development_ids != expected_development_ids:
        raise ValueError("combined final-fit records do not cover all development games")
    if development_ids & sealed_ids:
        raise ValueError("combined final-fit records contain a sealed game")
    return records, manifest, shards


def _build_final_dataset(
    records: list[dict[str, Any]],
    *,
    role_report: Mapping[str, Any],
    config: FinalFitConfig,
) -> DenseTWDToMDataset:
    development_ids = {record["game_id"] for record in records}
    role_games = role_report.get("games")
    if not isinstance(role_games, Mapping):
        raise ValueError("role sidecar has no games mapping")
    missing = sorted(development_ids - set(role_games))
    if missing:
        raise ValueError(f"role sidecar is missing development games: {missing[:3]}")
    development_roles = {
        game_id: role_games[game_id]["observer_roles"]
        for game_id in sorted(development_ids)
    }
    return DenseTWDToMDataset(
        records,
        feature_builder=PublicEventFeatureBuilder(max_seq_len=config.max_seq_len),
        enable_cyclic_rotation=True,
        augmentation_seed=config.seed,
        include_private_features=False,
        observer_roles_by_game=development_roles,
        supervision_scope=config.supervision_scope,
        speech_annotation_source=config.speech_annotation_source,
        belief_annotation_source=config.belief_annotation_source,
    )


def _dataset_contract(dataset: DenseTWDToMDataset) -> dict[str, Any]:
    return {
        "source_schema_version": SAMPLE_SCHEMA_VERSION,
        "model_input_scope": dataset.model_input_scope,
        "target_semantics": dataset.target_semantics,
        "target_conversion": dataset.target_conversion,
        "label_observation_semantics": dataset.label_observation_semantics,
        "training_supervision": DENSE_SUPERVISION_VERSION,
        "supervision_scope": dataset.supervision_scope,
        "speech_annotation_source": dataset.speech_annotation_source,
        "belief_annotation_source": dataset.belief_annotation_source,
        "role_metadata_usage": "development_supervision_metadata_only",
    }


def _build_model(config: FinalFitConfig) -> ToMBeliefBackbone:
    return ToMBeliefBackbone(
        ToMBeliefBackboneConfig(
            max_seq_len=config.max_seq_len,
            private_conditioning=False,
            input_feature_profile=config.input_feature_profile,
        ),
        backbone_name=config.backbone,
    )


def _final_checkpoint_payload(
    *,
    model: ToMBeliefBackbone,
    optimizer: AdamW,
    config: FinalFitConfig,
    train_metrics: Mapping[str, Any],
    run_provenance: Mapping[str, Any],
    dataset_contract: Mapping[str, Any],
    learning_rate_schedule: Mapping[str, Any],
) -> dict[str, Any]:
    task_contract = checkpoint_task_contract(
        False,
        target_semantics=dataset_contract["target_semantics"],
        target_conversion=dataset_contract["target_conversion"],
    )
    if dataset_contract["model_input_scope"] != task_contract["model_input_scope"]:
        raise ValueError("final checkpoint model and dataset input scopes differ")
    return {
        "checkpoint_type": FINAL_CHECKPOINT_TYPE,
        "schema_version": SAMPLE_SCHEMA_VERSION,
        **task_contract,
        "training_supervision": dataset_contract["training_supervision"],
        "supervision_scope": dataset_contract["supervision_scope"],
        "speech_annotation_source": dataset_contract["speech_annotation_source"],
        "belief_annotation_source": dataset_contract["belief_annotation_source"],
        "label_observation_semantics": dataset_contract[
            "label_observation_semantics"
        ],
        "role_metadata_usage": dataset_contract["role_metadata_usage"],
        "public_event_schema_version": PUBLIC_EVENT_SCHEMA_VERSION,
        "speech_action_count": len(ACTION_NAMES),
        "speech_action_to_id": dict(ACTION_TO_ID),
        "structured_token_to_id": dict(STRUCTURED_TOKEN_TO_ID),
        "public_phase_to_id": dict(PHASE_TO_ID),
        "backbone": model.backbone_name,
        "epoch": config.epochs,
        "training_config": _frozen_training_config(config),
        "run_provenance": dict(run_provenance),
        "model_config": result_model_config(model),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "learning_rate_schedule": dict(learning_rate_schedule),
        "train_metrics": dict(train_metrics),
        "selection_rule": EPOCH_SELECTION_RULE,
        "validation_dataset_used": False,
        "early_stopping_enabled": False,
        "sealed_test_evaluated": False,
    }


def run_final_fit(
    config: FinalFitConfig,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Fit exactly once on all development games without validation or test access."""

    set_random_seed(config.seed)
    device = resolve_device(config.device)
    root = Path(repo_root)
    git_commit = _clean_git_commit(root)
    fold_root = Path(os.path.abspath(config.fold_root))
    oof_path = Path(os.path.abspath(config.oof_summary_path))
    role_path = Path(os.path.abspath(config.role_sidecar_path))
    output_dir = Path(os.path.abspath(config.output_dir))

    records, fold_manifest, shards = _load_development_records(fold_root)
    fold_records = _validate_oof_summary(
        _load_json_object(oof_path),
        fold_manifest=fold_manifest,
        config=config,
    )
    role_report = load_role_sidecar_report(role_path)
    if role_report["split_manifest_digest"] != fold_manifest[
        "source_split_manifest_digest"
    ]:
        raise ValueError("role sidecar and source split manifest digests differ")
    if role_report["canonical_batch_summary_digest"] != fold_manifest[
        "canonical_batch_summary_digest"
    ]:
        raise ValueError("role sidecar and canonical batch digests differ")

    dataset = _build_final_dataset(records, role_report=role_report, config=config)
    if len(dataset) != EXPECTED_DEVELOPMENT_GAME_COUNT:
        raise ValueError("final-fit dataset must contain exactly 54 games")
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_dense_twd_tom_games,
        generator=generator,
    )
    dataset_contract = _dataset_contract(dataset)

    manifest_path = fold_root / DEVELOPMENT_FOLD_MANIFEST_FILENAME
    source_manifest_path = (
        fold_root / fold_manifest["source_split_manifest_relative_path"]
    )
    development_ids = sorted(fold_manifest["development_game_ids"])
    shard_provenance = [
        {
            **{key: value for key, value in shard.items() if key != "path"},
            "path": _repository_relative_path(shard["path"], repo_root=root),
        }
        for shard in shards
    ]
    data_digest = canonical_digest({
        "development_game_ids": development_ids,
        "shards": [
            {key: value for key, value in shard.items() if key != "path"}
            for shard in shard_provenance
        ],
    })
    oof_summary = _load_json_object(oof_path)
    protocol_payload = {
        "schema_version": FINAL_PROTOCOL_SCHEMA_VERSION,
        "status": "frozen_before_fit",
        "git_commit_sha": git_commit,
        "seed": config.seed,
        "frozen_training_config": _frozen_training_config(config),
        "epoch_selection": {
            "rule": EPOCH_SELECTION_RULE,
            "folds": fold_records,
            "fold_best_epochs": list(EXPECTED_FOLD_BEST_EPOCHS),
            "fit_epochs": config.epochs,
            "scheduler_horizon_epochs": config.scheduler_horizon_epochs,
        },
        "oof_summary": {
            "path": _repository_relative_path(oof_path, repo_root=root),
            "sha256": sha256_file(oof_path),
            "canonical_digest": canonical_digest(oof_summary),
        },
        "data_lineage": {
            "development_game_count": len(development_ids),
            "development_game_ids_digest": canonical_digest(development_ids),
            "development_data_digest": data_digest,
            "development_data_shards": shard_provenance,
            "development_fold_manifest_path": _repository_relative_path(
                manifest_path, repo_root=root
            ),
            "development_fold_manifest_sha256": sha256_file(manifest_path),
            "development_fold_manifest_digest": fold_manifest["manifest_digest"],
            "source_split_manifest_path": _repository_relative_path(
                source_manifest_path, repo_root=root
            ),
            "source_split_manifest_sha256": sha256_file(source_manifest_path),
            "source_split_manifest_digest": fold_manifest[
                "source_split_manifest_digest"
            ],
            "canonical_batch_summary_digest": fold_manifest[
                "canonical_batch_summary_digest"
            ],
            "role_sidecar_path": _repository_relative_path(role_path, repo_root=root),
            "role_sidecar_sha256": sha256_file(role_path),
            "role_sidecar_digest": role_report["sidecar_digest"],
            "role_sidecar_development_subset_digest": canonical_digest({
                game_id: role_report["games"][game_id]
                for game_id in development_ids
            }),
            "sealed_test_game_count": len(fold_manifest["sealed_test_game_ids"]),
            "sealed_test_dataset_opened": False,
            "sealed_test_labels_used": False,
            "sealed_test_evaluated": False,
        },
        "runtime": {
            "requested_device": config.device,
            "resolved_device": str(device),
            "num_workers": config.num_workers,
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "transformers_version": transformers.__version__,
            "deterministic_algorithms_enabled": (
                torch.are_deterministic_algorithms_enabled()
            ),
        },
        "checkpoint_policy": {
            "initialization": "random_seeded_no_checkpoint_load",
            "checkpoint_count": 1,
            "checkpoint_filename": FINAL_CHECKPOINT_FILENAME,
            "validation_dataset_used": False,
            "early_stopping_enabled": False,
            "fold_checkpoint_reused": False,
        },
    }
    protocol = {
        **protocol_payload,
        "protocol_digest": canonical_digest(protocol_payload),
    }

    _prepare_run_output_dir(output_dir)
    _atomic_json_write(protocol, output_dir / FINAL_PROTOCOL_FILENAME)
    run_provenance = {
        "git_commit_sha": git_commit,
        "git_worktree_clean": True,
        "seed": config.seed,
        "output_dir": _repository_relative_path(output_dir, repo_root=root),
        "final_protocol_path": _repository_relative_path(
            output_dir / FINAL_PROTOCOL_FILENAME,
            repo_root=root,
        ),
        "final_protocol_digest": protocol["protocol_digest"],
        "development_data_digest": data_digest,
        "development_fold_manifest_digest": fold_manifest["manifest_digest"],
        "source_split_manifest_digest": fold_manifest[
            "source_split_manifest_digest"
        ],
        "role_sidecar_development_subset_digest": protocol["data_lineage"][
            "role_sidecar_development_subset_digest"
        ],
        "sealed_test_dataset_opened": False,
        "sealed_test_labels_used": False,
        "sealed_test_evaluated": False,
    }

    model = _build_model(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler, schedule = build_learning_rate_scheduler(
        optimizer,
        config=config,  # type: ignore[arg-type]
        steps_per_epoch=len(loader),
        scheduler_horizon_epochs=config.scheduler_horizon_epochs,
    )
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        dataset.set_epoch(epoch)
        train_metrics = train_one_epoch(
            model,
            loader,
            optimizer,
            device=device,
            gradient_clip_norm=config.gradient_clip_norm,
            lr_scheduler=scheduler,
        )
        mean_loss = float(train_metrics["mean_loss"])
        if not math.isfinite(mean_loss):
            raise RuntimeError("final-fit training mean loss must remain finite")
        history.append({
            "epoch": epoch,
            "train": dict(train_metrics),
            "final_protocol_digest": protocol["protocol_digest"],
        })
    if len(history) != config.epochs:
        raise AssertionError("final fit did not run exactly 30 epochs")

    final_metrics = history[-1]["train"]
    checkpoint_path = output_dir / FINAL_CHECKPOINT_FILENAME
    _atomic_torch_save(
        _final_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            config=config,
            train_metrics=final_metrics,
            run_provenance=run_provenance,
            dataset_contract=dataset_contract,
            learning_rate_schedule=schedule,
        ),
        checkpoint_path,
    )
    if len(list(output_dir.glob("*.pt"))) != 1:
        raise RuntimeError("final fit must produce exactly one checkpoint")
    _atomic_json_write(history, output_dir / FINAL_HISTORY_FILENAME)
    summary = {
        "status": "ok",
        "checkpoint_type": FINAL_CHECKPOINT_TYPE,
        "epochs_completed": len(history),
        "development_game_count": len(dataset),
        "development_boundary_count": dataset.boundary_count,
        "validation_dataset_used": False,
        "early_stopping_enabled": False,
        "sealed_test_evaluated": False,
        "training_config": _frozen_training_config(config),
        "learning_rate_schedule": schedule,
        "final_train_metrics": final_metrics,
        "final_protocol_digest": protocol["protocol_digest"],
        "final_protocol": (
            Path(run_provenance["output_dir"]) / FINAL_PROTOCOL_FILENAME
        ).as_posix(),
        "final_checkpoint": (
            Path(run_provenance["output_dir"]) / FINAL_CHECKPOINT_FILENAME
        ).as_posix(),
        "history": (
            Path(run_provenance["output_dir"]) / FINAL_HISTORY_FILENAME
        ).as_posix(),
        "run_provenance": run_provenance,
    }
    _atomic_json_write(summary, output_dir / FINAL_SUMMARY_FILENAME)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen leakage-safe 54-game ToM final fit."
    )
    parser.add_argument("--fold-root", required=True)
    parser.add_argument("--role-sidecar", required=True)
    parser.add_argument("--oof-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = run_final_fit(FinalFitConfig(
        fold_root=args.fold_root,
        role_sidecar_path=args.role_sidecar,
        oof_summary_path=args.oof_summary,
        output_dir=args.output_dir,
        device=args.device,
    ))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
