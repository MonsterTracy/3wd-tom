"""Run the strict Speech V1/V2 × Belief V1/V2 OOF attribution matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from script.twd_tom.run_development_oof import run_diagnostic_oof
from script.twd_tom.audit_belief_label_repeatability import (
    REPEATABILITY_SCHEMA_VERSION,
)
from werewolf.models.twd_tom.belief_backbone import (
    NO_PHASE_DAY_INPUT_FEATURE_PROFILE,
    QWEN2_BACKBONE_NAME,
)
from werewolf.models.twd_tom.annotation_v2 import (
    V1_EMPTY_UNIFORM_NONSELF_BELIEF_SOURCE,
)


ABLATION_SCHEMA_VERSION = "classic7_annotation_v2_oof_ablation_v4"
_EXPERIMENTS = (
    (
        "speech_v1_belief_v1_empty_uniform_nonself",
        "v1",
        V1_EMPTY_UNIFORM_NONSELF_BELIEF_SOURCE,
    ),
    (
        "speech_v2_belief_v1_empty_uniform_nonself",
        "v2",
        V1_EMPTY_UNIFORM_NONSELF_BELIEF_SOURCE,
    ),
    ("speech_v1_belief_v2", "v1", "v2"),
    ("speech_v2_belief_v2", "v2", "v2"),
)
_SCOPES = ("non_wolf_alive", "villager_alive")


def _atomic_json_write(value: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_markdown_write(value: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run_annotation_v2_ablation(
    *,
    fold_root: str | Path,
    output_dir: str | Path,
    role_sidecar_path: str | Path,
    speech_v2_annotation_path: str | Path,
    belief_v2_annotation_path: str | Path,
    repeatability_audit_path: str | Path | None = None,
    epochs: int = 80,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    min_learning_rate: float = 1e-5,
    warmup_ratio: float = 0.05,
    early_stopping_patience: int = 12,
    early_stopping_min_delta: float = 1e-4,
    seed: int = 42,
    device: str = "auto",
    bootstrap_samples: int = 2000,
    worst_case_limit: int = 50,
) -> dict[str, Any]:
    """Run eight resumable OOF jobs while holding optimization fixed."""

    output = Path(os.path.abspath(output_dir))
    repeatability_report: dict[str, Any] = {
        "status": "not_provided",
        "purpose": "diagnostic_ceiling_only",
        "used_as_acceptance_ceiling": False,
    }
    if repeatability_audit_path is not None:
        repeatability_path = Path(repeatability_audit_path).resolve()
        if not repeatability_path.is_file():
            raise FileNotFoundError(
                f"repeatability audit not found: {repeatability_path}"
            )
        repeatability = json.loads(
            repeatability_path.read_text(encoding="utf-8")
        )
        if (
            not isinstance(repeatability, dict)
            or repeatability.get("schema_version")
            != REPEATABILITY_SCHEMA_VERSION
            or repeatability.get("status") != "PASS"
        ):
            raise ValueError("repeatability audit must be a passing V2 report")
        repeatability_report = {
            "status": "verified",
            "path": str(repeatability_path),
            "sha256": hashlib.sha256(
                repeatability_path.read_bytes()
            ).hexdigest(),
            "state_count": repeatability["state_count"],
            "replicate_count": repeatability["replicate_count"],
            "purpose": "diagnostic_ceiling_only",
            "used_as_acceptance_ceiling": False,
        }
    output.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    for experiment_name, speech_source, belief_source in _EXPERIMENTS:
        reports[experiment_name] = {}
        for scope in _SCOPES:
            run_output = output / experiment_name / scope
            report = run_diagnostic_oof(
                fold_root=fold_root,
                output_dir=run_output,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                min_learning_rate=min_learning_rate,
                warmup_ratio=warmup_ratio,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=early_stopping_min_delta,
                seed=seed,
                device=device,
                bootstrap_samples=bootstrap_samples,
                private_conditioning=False,
                backbone=QWEN2_BACKBONE_NAME,
                input_feature_profile=NO_PHASE_DAY_INPUT_FEATURE_PROFILE,
                role_sidecar_path=role_sidecar_path,
                supervision_scope=scope,
                speech_annotation_source=speech_source,
                belief_annotation_source=belief_source,
                speech_v2_annotation_path=speech_v2_annotation_path,
                belief_v2_annotation_path=belief_v2_annotation_path,
                worst_case_limit=worst_case_limit,
            )
            reports[experiment_name][scope] = {
                "output_dir": str(run_output),
                "oof_game_count": report["oof_game_count"],
                "oof_scored_game_count": report["oof_scored_game_count"],
                "oof_unscored_game_count": report[
                    "oof_unscored_game_count"
                ],
                "oof_unscored_game_ids": report["oof_unscored_game_ids"],
                "observer_weighted_normalized_reducible_gap_improvement": (
                    report["oof_observer_weighted_metrics"][
                        "normalized_reducible_gap_improvement"
                    ]
                ),
                "game_macro_normalized_reducible_gap_improvement": report[
                    "oof_game_macro_metrics"
                ]["normalized_reducible_gap_improvement"],
                "game_bootstrap_ci": report["oof_game_bootstrap_ci"],
            }
    result = {
        "schema_version": ABLATION_SCHEMA_VERSION,
        "status": "ok",
        "benchmark_status": "exploratory_diagnostic",
        "task": "public_only_subjective_suspicion",
        "backbone": QWEN2_BACKBONE_NAME,
        "input_feature_profile": NO_PHASE_DAY_INPUT_FEATURE_PROFILE,
        "shared_training_parameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "min_learning_rate": min_learning_rate,
            "warmup_ratio": warmup_ratio,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
            "seed": seed,
        },
        "repeatability_audit": repeatability_report,
        "experiments": reports,
    }
    _atomic_json_write(result, output / "annotation_v2_ablation_summary.json")
    lines = [
        "| Speech | Belief | Non-wolf OOF | Villager-only OOF |",
        "|---|---:|---:|---:|",
    ]
    for experiment_name, speech_source, belief_source in _EXPERIMENTS:
        non_wolf = reports[experiment_name]["non_wolf_alive"][
            "observer_weighted_normalized_reducible_gap_improvement"
        ]
        villager = reports[experiment_name]["villager_alive"][
            "observer_weighted_normalized_reducible_gap_improvement"
        ]
        lines.append(
            f"| {speech_source.upper()} | {belief_source.upper()} | "
            f"{100.0 * non_wolf:.2f}% | {100.0 * villager:.2f}% |"
        )
    _atomic_markdown_write(
        "\n".join(lines) + "\n",
        output / "annotation_v2_ablation_table.md",
    )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the strict Annotation V2 2x2 OOF attribution matrix."
    )
    parser.add_argument("--fold-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--role-sidecar", required=True)
    parser.add_argument("--speech-v2-annotations", required=True)
    parser.add_argument("--belief-v2-annotations", required=True)
    parser.add_argument("--repeatability-audit")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--worst-case-limit", type=int, default=50)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = run_annotation_v2_ablation(
        fold_root=args.fold_root,
        output_dir=args.output_dir,
        role_sidecar_path=args.role_sidecar,
        speech_v2_annotation_path=args.speech_v2_annotations,
        belief_v2_annotation_path=args.belief_v2_annotations,
        repeatability_audit_path=args.repeatability_audit,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_ratio=args.warmup_ratio,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        seed=args.seed,
        device=args.device,
        bootstrap_samples=args.bootstrap_samples,
        worst_case_limit=args.worst_case_limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
