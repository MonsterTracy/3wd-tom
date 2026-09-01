"""One-shot, fail-closed evaluation of the frozen final ToM checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch.utils.data import DataLoader

from script.twd_tom.materialize_canonical_belief_dataset import (
    validate_split_manifest,
)
from script.twd_tom.run_final_fit import (
    FINAL_CHECKPOINT_TYPE,
    FINAL_PROTOCOL_SCHEMA_VERSION,
    _clean_git_commit,
)
from script.twd_tom.train import (
    REPO_ROOT,
    _atomic_json_write,
    bootstrap_game_macro_metric,
    evaluate_model_with_games_and_strata,
    resolve_device,
    sha256_file,
)
from werewolf.models.twd_tom.action_features import PublicEventFeatureBuilder
from werewolf.models.twd_tom.annotation_v2 import (
    V1_ANNOTATION_SOURCE,
    V1_EMPTY_UNIFORM_NONSELF_BELIEF_SOURCE,
)
from werewolf.models.twd_tom.belief_backbone import (
    NO_PHASE_DAY_INPUT_FEATURE_PROFILE,
    QWEN2_BACKBONE_NAME,
)
from werewolf.models.twd_tom.checkpoint import (
    build_model_from_checkpoint,
    load_checkpoint,
)
from werewolf.models.twd_tom.dense_dataset import (
    DenseTWDToMDataset,
    collate_dense_twd_tom_games,
)
from werewolf.models.twd_tom.samples import SAMPLE_SCHEMA_VERSION
from werewolf.models.twd_tom.supervision import ALL_ALIVE_SCOPE
from werewolf.trajectory import canonical_digest, canonical_json


SEALED_PROTOCOL_SCHEMA_VERSION = "classic7_tom_v2_sealed_test_protocol_v2"
SEALED_SUMMARY_SCHEMA_VERSION = "classic7_tom_v2_sealed_test_summary_v2"
SEALED_PER_GAME_SCHEMA_VERSION = "classic7_tom_v2_sealed_test_per_game_v2"
SEALED_PROVENANCE_SCHEMA_VERSION = "classic7_tom_v2_sealed_test_provenance_v2"
SEALED_MARKER_SCHEMA_VERSION = "classic7_tom_v2_sealed_test_marker_v2"

SEALED_PROTOCOL_FILENAME = "sealed_test_protocol.json"
SEALED_SUMMARY_FILENAME = "sealed_test_summary.json"
SEALED_PER_GAME_FILENAME = "sealed_test_per_game.json"
SEALED_PROVENANCE_FILENAME = "sealed_test_provenance.json"

FROZEN_CHECKPOINT_SHA256 = ""
FROZEN_FINAL_PROTOCOL_DIGEST = ""
FROZEN_CHECKPOINT_GIT_COMMIT = ""
FROZEN_EPOCH = 0
FROZEN_SEALED_GAME_COUNT = 6
FROZEN_DEVELOPMENT_GAME_COUNT = 54
FROZEN_BOOTSTRAP_SAMPLES = 2000
FROZEN_BOOTSTRAP_SEED = 42
FROZEN_BATCH_SIZE = 8


@dataclass(frozen=True)
class SealedEvalConfig:
    """Only paths and the runtime device are configurable."""

    checkpoint_path: str
    final_protocol_path: str
    manifest_path: str
    output_dir: str
    device: str = "auto"

    checkpoint_type: ClassVar[str] = FINAL_CHECKPOINT_TYPE
    backbone: ClassVar[str] = QWEN2_BACKBONE_NAME
    input_feature_profile: ClassVar[str] = NO_PHASE_DAY_INPUT_FEATURE_PROFILE
    speech_annotation_source: ClassVar[str] = V1_ANNOTATION_SOURCE
    belief_annotation_source: ClassVar[str] = V1_EMPTY_UNIFORM_NONSELF_BELIEF_SOURCE
    supervision_scope: ClassVar[str] = ALL_ALIVE_SCOPE
    epoch: ClassVar[int] = FROZEN_EPOCH
    seed: ClassVar[int] = FROZEN_BOOTSTRAP_SEED
    batch_size: ClassVar[int] = FROZEN_BATCH_SIZE

    def __post_init__(self) -> None:
        for field_name in (
            "checkpoint_path",
            "final_protocol_path",
            "manifest_path",
            "output_dir",
            "device",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be non-empty text")


@dataclass(frozen=True)
class SealedPreflight:
    checkpoint_path: Path
    final_protocol_path: Path
    manifest_path: Path
    output_dir: Path
    marker_path: Path
    sealed_dataset_path: Path
    evaluator_git_commit: str
    sealed_game_ids: tuple[str, ...]
    development_game_ids: tuple[str, ...]
    manifest: dict[str, Any]
    final_protocol: dict[str, Any]
    checkpoint: dict[str, Any]
    model: torch.nn.Module


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{description} must contain one JSON object")
    return value


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_run_id() -> str:
    return f"sealed_{uuid.uuid4().hex}"


def _require_frozen_bindings() -> None:
    if (
        len(FROZEN_CHECKPOINT_SHA256) != 64
        or len(FROZEN_FINAL_PROTOCOL_DIGEST) != 64
        or len(FROZEN_CHECKPOINT_GIT_COMMIT) != 40
        or FROZEN_EPOCH <= 0
    ):
        raise RuntimeError(
            "all-alive sealed bindings have not been frozen from a new final fit"
        )


def _marker_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(checkpoint_path.name + ".sealed_test_evaluated.json")


def _validate_final_protocol(
    protocol: Mapping[str, Any], *, path: Path
) -> None:
    payload = dict(protocol)
    recorded_digest = payload.pop("protocol_digest", None)
    if recorded_digest != canonical_digest(payload):
        raise ValueError("final protocol canonical digest mismatch")
    if recorded_digest != FROZEN_FINAL_PROTOCOL_DIGEST:
        raise ValueError("final protocol differs from the frozen final-fit protocol")
    if protocol.get("schema_version") != FINAL_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("final protocol schema mismatch")
    if protocol.get("git_commit_sha") != FROZEN_CHECKPOINT_GIT_COMMIT:
        raise ValueError("final protocol Git commit mismatch")
    data_lineage = protocol.get("data_lineage")
    if not isinstance(data_lineage, Mapping):
        raise ValueError("final protocol has no data lineage")
    for field_name in (
        "sealed_test_dataset_opened",
        "sealed_test_labels_used",
        "sealed_test_evaluated",
    ):
        if data_lineage.get(field_name) is not False:
            raise ValueError(f"final protocol must record {field_name}=false")
    checkpoint_policy = protocol.get("checkpoint_policy")
    if not isinstance(checkpoint_policy, Mapping):
        raise ValueError("final protocol has no checkpoint policy")
    if checkpoint_policy.get("validation_dataset_used") is not False:
        raise ValueError("final protocol used validation data")
    if checkpoint_policy.get("early_stopping_enabled") is not False:
        raise ValueError("final protocol enabled early stopping")
    if not path.is_file():
        raise FileNotFoundError(path)


def _validate_checkpoint(checkpoint: Mapping[str, Any]) -> torch.nn.Module:
    expected_top_level = {
        "checkpoint_type": FINAL_CHECKPOINT_TYPE,
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "backbone": QWEN2_BACKBONE_NAME,
        "speech_annotation_source": V1_ANNOTATION_SOURCE,
        "belief_annotation_source": V1_EMPTY_UNIFORM_NONSELF_BELIEF_SOURCE,
        "supervision_scope": ALL_ALIVE_SCOPE,
        "epoch": FROZEN_EPOCH,
        "validation_dataset_used": False,
        "early_stopping_enabled": False,
        "sealed_test_evaluated": False,
    }
    for field_name, expected in expected_top_level.items():
        if checkpoint.get(field_name) != expected:
            raise ValueError(
                f"checkpoint {field_name} mismatch: "
                f"{checkpoint.get(field_name)!r} != {expected!r}"
            )
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("checkpoint has no model_config")
    if model_config.get("input_feature_profile") != NO_PHASE_DAY_INPUT_FEATURE_PROFILE:
        raise ValueError("checkpoint input feature profile mismatch")
    if model_config.get("private_conditioning", False) is not False:
        raise ValueError("sealed checkpoint must be public-only")
    training_config = checkpoint.get("training_config")
    if not isinstance(training_config, Mapping):
        raise ValueError("checkpoint has no frozen training config")
    expected_training = {
        "backbone": QWEN2_BACKBONE_NAME,
        "input_feature_profile": NO_PHASE_DAY_INPUT_FEATURE_PROFILE,
        "speech_annotation_source": V1_ANNOTATION_SOURCE,
        "belief_annotation_source": V1_EMPTY_UNIFORM_NONSELF_BELIEF_SOURCE,
        "supervision_scope": ALL_ALIVE_SCOPE,
        "fit_epochs": FROZEN_EPOCH,
        "seed": FROZEN_BOOTSTRAP_SEED,
        "validation_dataset_used": False,
        "early_stopping_enabled": False,
    }
    for field_name, expected in expected_training.items():
        if training_config.get(field_name) != expected:
            raise ValueError(f"checkpoint training config mismatch: {field_name}")
    provenance = checkpoint.get("run_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("checkpoint has no run provenance")
    if provenance.get("final_protocol_digest") != FROZEN_FINAL_PROTOCOL_DIGEST:
        raise ValueError("checkpoint final protocol digest mismatch")
    if provenance.get("git_commit_sha") != FROZEN_CHECKPOINT_GIT_COMMIT:
        raise ValueError("checkpoint Git commit mismatch")
    for field_name in (
        "sealed_test_dataset_opened",
        "sealed_test_labels_used",
        "sealed_test_evaluated",
    ):
        if provenance.get(field_name) is not False:
            raise ValueError(f"checkpoint provenance must record {field_name}=false")

    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint has no model state")
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError("checkpoint model state must contain only tensors")

    def validate_tensors(value: Any, *, path: str) -> None:
        if isinstance(value, torch.Tensor):
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"checkpoint contains a non-finite tensor: {path}")
        elif isinstance(value, Mapping):
            for key, nested in value.items():
                validate_tensors(nested, path=f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                validate_tensors(nested, path=f"{path}[{index}]")

    validate_tensors(checkpoint, path="checkpoint")
    return build_model_from_checkpoint(checkpoint, device=torch.device("cpu"))


def _resolve_manifest_relative_file(
    manifest_path: Path, descriptor: Mapping[str, Any]
) -> Path:
    relative_path = descriptor.get("relative_path")
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or Path(relative_path).is_absolute()
        or len(Path(relative_path).parts) != 1
    ):
        raise ValueError("split manifest test path is not a direct relative filename")
    return manifest_path.parent / relative_path


def preflight_sealed_evaluation(
    config: SealedEvalConfig,
    *,
    repo_root: Path = REPO_ROOT,
) -> SealedPreflight:
    """Validate every frozen binding without opening the sealed Dataset file."""

    _require_frozen_bindings()
    checkpoint_path = Path(os.path.abspath(config.checkpoint_path))
    final_protocol_path = Path(os.path.abspath(config.final_protocol_path))
    manifest_path = Path(os.path.abspath(config.manifest_path))
    output_dir = Path(os.path.abspath(config.output_dir))
    if sha256_file(checkpoint_path) != FROZEN_CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA-256 differs from the frozen checkpoint")
    checkpoint = load_checkpoint(checkpoint_path)
    model = _validate_checkpoint(checkpoint)
    final_protocol = _load_json_object(
        final_protocol_path, description="final protocol"
    )
    _validate_final_protocol(final_protocol, path=final_protocol_path)

    manifest = validate_split_manifest(manifest_path, verify_split_files=())
    data_lineage = final_protocol["data_lineage"]
    if sha256_file(manifest_path) != data_lineage.get(
        "source_split_manifest_sha256"
    ):
        raise ValueError("source split manifest SHA-256 mismatch")
    if manifest.get("manifest_digest") != data_lineage.get(
        "source_split_manifest_digest"
    ):
        raise ValueError("source split manifest digest mismatch")
    if manifest.get("canonical_batch_summary_digest") != data_lineage.get(
        "canonical_batch_summary_digest"
    ):
        raise ValueError("canonical batch summary digest mismatch")

    raw_test_ids = manifest["game_ids"]["test"]
    raw_dev_ids = manifest["game_ids"]["train"] + manifest["game_ids"]["validation"]
    sealed_ids = tuple(raw_test_ids)
    development_ids = tuple(raw_dev_ids)
    if len(sealed_ids) != FROZEN_SEALED_GAME_COUNT:
        raise ValueError("sealed manifest must contain exactly 6 games")
    if len(set(sealed_ids)) != FROZEN_SEALED_GAME_COUNT:
        raise ValueError("sealed manifest game IDs must be unique")
    if len(development_ids) != FROZEN_DEVELOPMENT_GAME_COUNT:
        raise ValueError("development manifest must contain exactly 54 games")
    if len(set(development_ids)) != FROZEN_DEVELOPMENT_GAME_COUNT:
        raise ValueError("development manifest game IDs must be unique")
    overlap = set(sealed_ids) & set(development_ids)
    if overlap:
        raise ValueError(f"sealed and development games overlap: {sorted(overlap)}")
    if data_lineage.get("development_game_ids_digest") != canonical_digest(
        sorted(development_ids)
    ):
        raise ValueError("development game IDs differ from final-fit lineage")
    if data_lineage.get("sealed_test_game_count") != FROZEN_SEALED_GAME_COUNT:
        raise ValueError("final protocol sealed game count mismatch")

    marker_path = _marker_path(checkpoint_path)
    if marker_path.exists():
        raise FileExistsError(
            f"sealed evaluation is permanently locked for this checkpoint: {marker_path}"
        )
    if output_dir.exists():
        raise FileExistsError(f"sealed output directory already exists: {output_dir}")
    evaluator_git_commit = _clean_git_commit(Path(repo_root))
    test_descriptor = manifest["output_files"]["test"]
    sealed_dataset_path = _resolve_manifest_relative_file(
        manifest_path, test_descriptor
    )
    return SealedPreflight(
        checkpoint_path=checkpoint_path,
        final_protocol_path=final_protocol_path,
        manifest_path=manifest_path,
        output_dir=output_dir,
        marker_path=marker_path,
        sealed_dataset_path=sealed_dataset_path,
        evaluator_git_commit=evaluator_git_commit,
        sealed_game_ids=sealed_ids,
        development_game_ids=development_ids,
        manifest=manifest,
        final_protocol=final_protocol,
        checkpoint=checkpoint,
        model=model,
    )


def _metric_record(game_id: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    observed_rows = int(metrics.get("valid_observer_count", 0))
    if observed_rows == 0:
        return {
            "game_id": game_id,
            "status": "unscored_no_observed_labels",
            "observed_rows": 0,
        }
    model_sum = float(metrics["model_kl_sum"])
    uniform_sum = float(metrics["uniform_non_self_baseline_kl_sum"])
    gap_closed = 1.0 - model_sum / uniform_sum if uniform_sum > 0.0 else 0.0
    return {
        "game_id": game_id,
        "status": "scored",
        "observed_rows": observed_rows,
        "model_kl_sum": model_sum,
        "model_kl_mean": model_sum / observed_rows,
        "uniform_kl_sum": uniform_sum,
        "uniform_kl_mean": uniform_sum / observed_rows,
        "gap_closed": gap_closed,
        "total_variation_mean": float(metrics["mean_belief_total_variation"]),
    }


def summarize_metric_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze both game-macro and observer-weighted sealed estimands."""

    scored = [record for record in records if record.get("status") == "scored"]
    if not scored:
        raise ValueError("sealed evaluation contains no scored games")
    rows = sum(int(record["observed_rows"]) for record in scored)
    model_sum = sum(float(record["model_kl_sum"]) for record in scored)
    uniform_sum = sum(float(record["uniform_kl_sum"]) for record in scored)
    tv_sum = sum(
        float(record["total_variation_mean"]) * int(record["observed_rows"])
        for record in scored
    )
    game_macro_gap = sum(float(record["gap_closed"]) for record in scored) / len(
        scored
    )
    bootstrap_input = {
        str(record["game_id"]): {
            "valid_observer_count": int(record["observed_rows"]),
            "gap_closed": float(record["gap_closed"]),
        }
        for record in scored
    }
    return {
        "scored_game_count": len(scored),
        "unscored_game_count": len(records) - len(scored),
        "observed_rows": rows,
        "primary_game_macro_gap_closed": game_macro_gap,
        "observer_weighted_gap_closed": (
            1.0 - model_sum / uniform_sum if uniform_sum > 0.0 else 0.0
        ),
        "model_kl_sum": model_sum,
        "model_kl_mean": model_sum / rows,
        "uniform_kl_sum": uniform_sum,
        "uniform_kl_mean": uniform_sum / rows,
        "total_variation_mean": tv_sum / rows,
        "bootstrap_ci95": bootstrap_game_macro_metric(
            bootstrap_input,
            metric_name="gap_closed",
            samples=FROZEN_BOOTSTRAP_SAMPLES,
            seed=FROZEN_BOOTSTRAP_SEED,
        ),
    }


def _metric_definitions() -> dict[str, Any]:
    return {
        "primary": {
            "name": "all_alive_common_sealed_game_macro_gap_closed",
            "unit": "game",
            "formula": "mean_game(1 - game_model_kl_sum / game_uniform_kl_sum)",
            "zero_uniform_denominator": "gap_closed=0.0 (corrected OOF convention)",
        },
        "secondary": {
            "observer_weighted_gap_closed": (
                "1 - sum_observer(model_kl) / sum_observer(uniform_kl)"
            ),
            "model_kl_mean": "model_kl_sum / observed_rows",
            "uniform_kl_mean": "uniform_kl_sum / observed_rows",
            "total_variation_mean": "observer-weighted mean TV",
        },
        "mask": (
            "observer_alive & v1_empty_uniform_nonself_label_observed"
        ),
        "unobserved_target": "all-zero target with distribution_loss_mask=false",
        "bootstrap": {
            "unit": "game",
            "samples": FROZEN_BOOTSTRAP_SAMPLES,
            "seed": FROZEN_BOOTSTRAP_SEED,
            "interval": "percentile_95",
            "precision_claim": False,
        },
    }


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(canonical_json(value) + "\n")


def _protocol_payload(
    plan: SealedPreflight, *, run_id: str, started_at: str, device: str
) -> dict[str, Any]:
    return {
        "schema_version": SEALED_PROTOCOL_SCHEMA_VERSION,
        "status": "frozen_before_sealed_label_open",
        "run_id": run_id,
        "started_at_utc": started_at,
        "checkpoint": {
            "path": str(plan.checkpoint_path),
            "sha256": FROZEN_CHECKPOINT_SHA256,
            "checkpoint_type": FINAL_CHECKPOINT_TYPE,
            "epoch": FROZEN_EPOCH,
            "final_protocol_digest": FROZEN_FINAL_PROTOCOL_DIGEST,
            "training_git_commit_sha": FROZEN_CHECKPOINT_GIT_COMMIT,
        },
        "evaluator": {
            "git_commit_sha": plan.evaluator_git_commit,
            "git_worktree_clean": True,
            "requested_device": device,
        },
        "frozen_task": {
            "backbone": QWEN2_BACKBONE_NAME,
            "input_feature_profile": NO_PHASE_DAY_INPUT_FEATURE_PROFILE,
            "speech_annotation_source": V1_ANNOTATION_SOURCE,
            "belief_annotation_source": V1_EMPTY_UNIFORM_NONSELF_BELIEF_SOURCE,
            "supervision_scope": ALL_ALIVE_SCOPE,
            "batch_size": FROZEN_BATCH_SIZE,
        },
        "sealed_data": {
            "split_manifest_path": str(plan.manifest_path),
            "split_manifest_sha256": sha256_file(plan.manifest_path),
            "split_manifest_digest": plan.manifest["manifest_digest"],
            "sealed_game_count": len(plan.sealed_game_ids),
            "sealed_game_ids_digest": canonical_digest(sorted(plan.sealed_game_ids)),
            "sealed_dataset_expected_sha256": plan.manifest["output_files"][
                "test"
            ]["sha256"],
            "development_game_ids_digest": canonical_digest(
                sorted(plan.development_game_ids)
            ),
            "sealed_and_development_disjoint": True,
            "sealed_dataset_opened_at_protocol_freeze": False,
        },
        "metric_definitions": _metric_definitions(),
        "one_shot_policy": {
            "marker_path": str(plan.marker_path),
            "rerun_default": "reject",
            "new_model_forward_for_audit": False,
            "audit_source": SEALED_PER_GAME_FILENAME,
        },
    }


def run_sealed_evaluation(
    config: SealedEvalConfig,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Perform the sole sealed forward pass after an entirely label-blind preflight."""

    plan = preflight_sealed_evaluation(config, repo_root=repo_root)
    resolved_device = resolve_device(config.device)
    run_id = _new_run_id()
    started_at = _utc_timestamp()
    protocol_payload = _protocol_payload(
        plan, run_id=run_id, started_at=started_at, device=config.device
    )
    protocol = {
        **protocol_payload,
        "protocol_digest": canonical_digest(protocol_payload),
    }
    marker_started = {
        "schema_version": SEALED_MARKER_SCHEMA_VERSION,
        "status": "started_irreversible_one_shot",
        "sealed_test_evaluated": False,
        "run_id": run_id,
        "sealed_protocol_digest": protocol["protocol_digest"],
        "started_at_utc": started_at,
    }
    _write_exclusive_json(plan.marker_path, marker_started)
    plan.output_dir.mkdir(parents=True, exist_ok=False)
    protocol_path = plan.output_dir / SEALED_PROTOCOL_FILENAME
    _write_exclusive_json(protocol_path, protocol)

    # No sealed label file is touched before the complete preflight, protocol
    # freeze, and one-shot lock above.
    validated_manifest = validate_split_manifest(
        plan.manifest_path, verify_split_files=("test",)
    )
    if validated_manifest["manifest_digest"] != plan.manifest["manifest_digest"]:
        raise ValueError("split manifest changed after preflight")
    dataset = DenseTWDToMDataset.from_jsonl(
        plan.sealed_dataset_path,
        feature_builder=PublicEventFeatureBuilder(
            max_seq_len=plan.model.config.max_seq_len  # type: ignore[attr-defined]
        ),
        enable_cyclic_rotation=False,
        include_private_features=False,
        supervision_scope=ALL_ALIVE_SCOPE,
        speech_annotation_source=V1_ANNOTATION_SOURCE,
        belief_annotation_source=V1_EMPTY_UNIFORM_NONSELF_BELIEF_SOURCE,
    )
    expected_dataset_contract = {
        "model_input_scope": plan.checkpoint.get("model_input_scope"),
        "target_semantics": plan.checkpoint.get("target_semantics"),
        "target_conversion": plan.checkpoint.get("target_conversion"),
        "label_observation_semantics": plan.checkpoint.get(
            "label_observation_semantics"
        ),
        "supervision_scope": ALL_ALIVE_SCOPE,
        "speech_annotation_source": V1_ANNOTATION_SOURCE,
        "belief_annotation_source": V1_EMPTY_UNIFORM_NONSELF_BELIEF_SOURCE,
    }
    for field_name, expected in expected_dataset_contract.items():
        if getattr(dataset, field_name) != expected:
            raise ValueError(f"sealed dataset contract mismatch: {field_name}")
    actual_ids = tuple(sorted(sample["game_id"] for sample in dataset.samples))
    if actual_ids != tuple(sorted(plan.sealed_game_ids)):
        raise ValueError("sealed dataset game IDs differ from the frozen manifest")
    if len(dataset) != FROZEN_SEALED_GAME_COUNT:
        raise ValueError("sealed dataset must contain exactly 6 games")
    loader = DataLoader(
        dataset,
        batch_size=FROZEN_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_dense_twd_tom_games,
    )
    model = plan.model.to(resolved_device).eval()
    with torch.inference_mode():
        _, by_game, _, _ = evaluate_model_with_games_and_strata(
            model,
            loader,
            device=resolved_device,
        )
    forward_completed_at = _utc_timestamp()
    marker_evaluated = {
        **marker_started,
        "status": "forward_completed_artifacts_pending",
        "sealed_test_evaluated": True,
        "forward_completed_at_utc": forward_completed_at,
    }
    _atomic_json_write(marker_evaluated, plan.marker_path)

    per_game_rows = [
        _metric_record(game_id, by_game[game_id])
        for game_id in sorted(plan.sealed_game_ids)
    ]
    primary = summarize_metric_records(per_game_rows)
    completed_at = _utc_timestamp()
    per_game_artifact = {
        "schema_version": SEALED_PER_GAME_SCHEMA_VERSION,
        "status": "ok",
        "run_id": run_id,
        "sealed_protocol_digest": protocol["protocol_digest"],
        "all_alive": per_game_rows,
        "metric_definitions": _metric_definitions(),
    }
    summary = {
        "schema_version": SEALED_SUMMARY_SCHEMA_VERSION,
        "status": "ok",
        "run_id": run_id,
        "sealed_test_evaluated": True,
        "sealed_protocol_digest": protocol["protocol_digest"],
        "checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
        "final_protocol_digest": FROZEN_FINAL_PROTOCOL_DIGEST,
        "sealed_game_count": len(plan.sealed_game_ids),
        "primary_task": "all_alive",
        "primary": primary,
        "metric_definitions": _metric_definitions(),
        "selection_or_tuning_performed": False,
        "checkpoint_updated": False,
    }
    per_game_path = plan.output_dir / SEALED_PER_GAME_FILENAME
    summary_path = plan.output_dir / SEALED_SUMMARY_FILENAME
    _atomic_json_write(per_game_artifact, per_game_path)
    _atomic_json_write(summary, summary_path)
    provenance = {
        "schema_version": SEALED_PROVENANCE_SCHEMA_VERSION,
        "status": "ok",
        "run_id": run_id,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "sealed_test_evaluated": True,
        "sealed_labels_opened_only_after_preflight_and_lock": True,
        "pure_forward": {
            "model_eval": True,
            "torch_inference_mode": True,
            "optimizer_created": False,
            "scheduler_created": False,
            "backward_called": False,
            "checkpoint_written": False,
            "selection_or_tuning_performed": False,
        },
        "checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
        "final_protocol_digest": FROZEN_FINAL_PROTOCOL_DIGEST,
        "sealed_protocol_digest": protocol["protocol_digest"],
        "training_git_commit_sha": FROZEN_CHECKPOINT_GIT_COMMIT,
        "evaluator_git_commit_sha": plan.evaluator_git_commit,
        "split_manifest_sha256": sha256_file(plan.manifest_path),
        "split_manifest_digest": plan.manifest["manifest_digest"],
        "sealed_dataset_sha256": sha256_file(plan.sealed_dataset_path),
        "artifacts": {
            SEALED_PROTOCOL_FILENAME: sha256_file(protocol_path),
            SEALED_SUMMARY_FILENAME: sha256_file(summary_path),
            SEALED_PER_GAME_FILENAME: sha256_file(per_game_path),
        },
        "metric_recomputation_source": SEALED_PER_GAME_FILENAME,
    }
    provenance_path = plan.output_dir / SEALED_PROVENANCE_FILENAME
    _atomic_json_write(provenance, provenance_path)
    marker_complete = {
        **marker_evaluated,
        "status": "completed",
        "completed_at_utc": completed_at,
        "output_dir": str(plan.output_dir),
        "sealed_test_provenance_sha256": sha256_file(provenance_path),
    }
    _atomic_json_write(marker_complete, plan.marker_path)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen one-shot sealed ToM evaluation."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--final-protocol", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = run_sealed_evaluation(SealedEvalConfig(
        checkpoint_path=args.checkpoint,
        final_protocol_path=args.final_protocol,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        device=args.device,
    ))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
