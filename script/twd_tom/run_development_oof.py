"""Run and aggregate development-only dense ToM cross-validation."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from script.twd_tom.materialize_development_folds import (
    DEVELOPMENT_FOLD_MANIFEST_FILENAME,
    validate_development_fold_paths,
)
from script.twd_tom.train import TrainingConfig, run_training
from werewolf.models.twd_tom.belief_backbone import (
    FULL_INPUT_FEATURE_PROFILE,
    QWEN2_BACKBONE_NAME,
    SUPPORTED_BACKBONE_NAMES,
    SUPPORTED_INPUT_FEATURE_PROFILES,
)


OOF_SUMMARY_SCHEMA_VERSION = "classic7_tom_v2_dense_oof_summary_v2"
DEFAULT_TARGET_IMPROVEMENT = 0.50


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON file must contain one object: {path}")
    return value


def _atomic_json_write(value: Mapping[str, Any], path: Path) -> None:
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


def _weighted_metrics(
    by_game: Mapping[str, Mapping[str, int | float]],
) -> dict[str, int | float]:
    total_count = sum(int(metrics["valid_observer_count"]) for metrics in by_game.values())
    if total_count <= 0:
        raise ValueError("OOF metrics contain no supervised observers")
    names = set.intersection(
        *(
            {name for name in metrics if name != "valid_observer_count"}
            for metrics in by_game.values()
        )
    )
    result: dict[str, int | float] = {"valid_observer_count": total_count}
    for name in sorted(names):
        result[name] = sum(
            float(metrics[name]) * int(metrics["valid_observer_count"])
            for metrics in by_game.values()
        ) / total_count
    uniform_kl = float(
        result["uniform_non_self_baseline_mean_cross_entropy"]
    ) - float(result["mean_belief_target_entropy"])
    result["uniform_non_self_baseline_mean_kl_divergence"] = uniform_kl
    result["normalized_reducible_gap_improvement"] = (
        1.0 - float(result["mean_belief_kl_divergence"]) / uniform_kl
        if uniform_kl > 0.0
        else 0.0
    )
    private_cross_entropy = result.get(
        "private_admissible_uniform_baseline_mean_cross_entropy"
    )
    if private_cross_entropy is not None:
        private_kl = float(private_cross_entropy) - float(
            result["mean_belief_target_entropy"]
        )
        result[
            "private_admissible_uniform_baseline_mean_kl_divergence"
        ] = private_kl
        result[
            "private_admissible_normalized_reducible_gap_improvement"
        ] = (
            1.0 - float(result["mean_belief_kl_divergence"]) / private_kl
            if private_kl > 0.0
            else 0.0
        )
    return result


def _macro_metrics(
    by_game: Mapping[str, Mapping[str, int | float]],
) -> dict[str, float]:
    names = set.intersection(
        *(
            {
                name
                for name in metrics
                if name != "valid_observer_count"
                and isinstance(metrics[name], (int, float))
            }
            for metrics in by_game.values()
        )
    )
    return {
        name: sum(float(metrics[name]) for metrics in by_game.values())
        / len(by_game)
        for name in sorted(names)
    }


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_game_macro(
    by_game: Mapping[str, Mapping[str, int | float]],
    *,
    metric_name: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    values = [float(by_game[game_id][metric_name]) for game_id in sorted(by_game)]
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


def _validate_completed_fold_summary(
    summary: Mapping[str, Any],
    *,
    fold_name: str,
    requested: Mapping[str, Any],
) -> None:
    if summary.get("status") != "ok":
        raise ValueError(f"existing {fold_name} summary is not complete")
    provenance = summary.get("run_provenance")
    if not isinstance(provenance, Mapping) or provenance.get(
        "development_fold_name"
    ) != fold_name:
        raise ValueError(f"existing {fold_name} summary has wrong lineage")
    config = summary.get("training_config")
    if not isinstance(config, Mapping):
        raise ValueError(f"existing {fold_name} summary has no training config")
    for name, value in requested.items():
        if config.get(name) != value:
            raise ValueError(
                f"existing {fold_name} config differs for {name}: "
                f"{config.get(name)!r} != {value!r}"
            )


def run_development_oof(
    *,
    fold_root: str | Path,
    output_dir: str | Path,
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
    target_improvement: float = DEFAULT_TARGET_IMPROVEMENT,
    private_conditioning: bool = False,
    backbone: str = QWEN2_BACKBONE_NAME,
    input_feature_profile: str = FULL_INPUT_FEATURE_PROFILE,
) -> dict[str, Any]:
    """Train every fold, resume completed folds, and aggregate OOF games."""

    if not isinstance(private_conditioning, bool):
        raise TypeError("private_conditioning must be bool")
    if backbone not in SUPPORTED_BACKBONE_NAMES:
        raise ValueError(f"backbone must be one of {SUPPORTED_BACKBONE_NAMES}")
    if input_feature_profile not in SUPPORTED_INPUT_FEATURE_PROFILES:
        raise ValueError(
            "input_feature_profile must be one of "
            f"{SUPPORTED_INPUT_FEATURE_PROFILES}"
        )

    # Keep the lexical project path so repository symlinks remain valid
    # provenance paths; validators resolve their physical targets separately.
    resolved_fold_root = Path(os.path.abspath(fold_root))
    fold_manifest = _load_json(
        resolved_fold_root / DEVELOPMENT_FOLD_MANIFEST_FILENAME
    )
    fold_names = sorted(
        fold_manifest["folds"],
        key=lambda name: fold_manifest["folds"][name]["fold_index"],
    )
    resolved_output = Path(os.path.abspath(output_dir))
    resolved_output.mkdir(parents=True, exist_ok=True)
    requested_config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "lr_scheduler": "warmup_cosine",
        "warmup_ratio": warmup_ratio,
        "min_learning_rate": min_learning_rate,
        "seed": seed,
        "device": device,
        "max_seq_len": 256,
        "backbone": backbone,
        "input_feature_profile": input_feature_profile,
        "dense_supervision": True,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
    }
    if private_conditioning:
        requested_config["private_conditioning"] = True
    fold_summaries: dict[str, dict[str, Any]] = {}
    for fold_name in fold_names:
        fold_dir = resolved_fold_root / fold_name
        train_path = fold_dir / "train.jsonl"
        validation_path = fold_dir / "validation.jsonl"
        validate_development_fold_paths(train_path, validation_path)
        fold_output = resolved_output / fold_name
        summary_path = fold_output / "summary.json"
        if summary_path.is_file():
            summary = _load_json(summary_path)
            _validate_completed_fold_summary(
                summary,
                fold_name=fold_name,
                requested=requested_config,
            )
        else:
            if fold_output.exists() and any(fold_output.iterdir()):
                raise FileExistsError(
                    f"incomplete fold output must be inspected before retry: {fold_output}"
                )
            summary = run_training(TrainingConfig(
                output_dir=str(fold_output),
                dataset_path=str(train_path),
                validation_dataset_path=str(validation_path),
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                lr_scheduler="warmup_cosine",
                warmup_ratio=warmup_ratio,
                min_learning_rate=min_learning_rate,
                seed=seed,
                device=device,
                max_seq_len=256,
                backbone=backbone,
                input_feature_profile=input_feature_profile,
                dense_supervision=True,
                private_conditioning=private_conditioning,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=early_stopping_min_delta,
            ))
        fold_summaries[fold_name] = summary

    oof_by_game: dict[str, dict[str, int | float]] = {}
    baseline_by_name: dict[str, dict[str, dict[str, int | float]]] = {}
    for fold_name, summary in fold_summaries.items():
        for game_id, metrics in summary["best_validation_by_game"].items():
            if game_id in oof_by_game:
                raise ValueError(f"OOF game appears in multiple folds: {game_id}")
            oof_by_game[game_id] = metrics
        for baseline_name, baseline_report in summary[
            "validation_baselines"
        ].items():
            if (
                not isinstance(baseline_report, Mapping)
                or "by_game" not in baseline_report
            ):
                continue
            target = baseline_by_name.setdefault(baseline_name, {})
            by_game = baseline_report["by_game"]
            overlap = set(target) & set(by_game)
            if overlap:
                raise ValueError(
                    f"baseline OOF games repeat across folds: {sorted(overlap)[:3]}"
                )
            target.update(by_game)
    if set(oof_by_game) != set(fold_manifest["development_game_ids"]):
        raise ValueError("OOF results do not cover every development game exactly once")

    weighted = _weighted_metrics(oof_by_game)
    baselines = {
        name: {
            "observer_weighted": _weighted_metrics(by_game),
            "game_macro": _macro_metrics(by_game),
        }
        for name, by_game in baseline_by_name.items()
    }
    primary_metric = (
        "private_admissible_normalized_reducible_gap_improvement"
        if private_conditioning
        else "normalized_reducible_gap_improvement"
    )
    achieved = float(weighted[primary_metric])
    result = {
        "schema_version": OOF_SUMMARY_SCHEMA_VERSION,
        "status": "ok",
        "evaluation_scope": "development_oof_only",
        "test_evaluated": False,
        "sealed_test_game_count": len(fold_manifest["sealed_test_game_ids"]),
        "source_split_manifest_digest": fold_manifest[
            "source_split_manifest_digest"
        ],
        "development_fold_manifest_digest": fold_manifest["manifest_digest"],
        "training_config": requested_config,
        "folds": {
            fold_name: {
                "best_epoch": summary["best_epoch"],
                "epochs_completed": summary["epochs_completed"],
                "best_validation_mean_loss": summary[
                    "best_validation_mean_loss"
                ],
                "summary_path": str(
                    resolved_output / fold_name / "summary.json"
                ),
            }
            for fold_name, summary in fold_summaries.items()
        },
        "oof_game_count": len(oof_by_game),
        "oof_observer_weighted_metrics": weighted,
        "oof_game_macro_metrics": _macro_metrics(oof_by_game),
        "oof_game_bootstrap_ci": _bootstrap_game_macro(
            oof_by_game,
            metric_name=primary_metric,
            samples=bootstrap_samples,
            seed=seed,
        ),
        "oof_baselines": baselines,
        "acceptance_gate": {
            "metric": f"observer_weighted_{primary_metric}",
            "threshold": target_improvement,
            "achieved": achieved,
            "passed": achieved >= target_improvement,
        },
        "oof_by_game": dict(sorted(oof_by_game.items())),
    }
    _atomic_json_write(result, resolved_output / "oof_summary.json")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run dense ToM development-only 5-fold OOF training."
    )
    parser.add_argument("--fold-root", required=True)
    parser.add_argument("--output-dir", required=True)
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
    parser.add_argument("--target-improvement", type=float, default=0.50)
    parser.add_argument("--private-conditioning", action="store_true")
    parser.add_argument(
        "--backbone",
        choices=SUPPORTED_BACKBONE_NAMES,
        default=QWEN2_BACKBONE_NAME,
    )
    parser.add_argument(
        "--input-feature-profile",
        choices=SUPPORTED_INPUT_FEATURE_PROFILES,
        default=FULL_INPUT_FEATURE_PROFILE,
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = run_development_oof(
        fold_root=args.fold_root,
        output_dir=args.output_dir,
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
        target_improvement=args.target_improvement,
        private_conditioning=args.private_conditioning,
        backbone=args.backbone,
        input_feature_profile=args.input_feature_profile,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
