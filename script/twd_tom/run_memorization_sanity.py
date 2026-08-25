"""Run M1/M2 train-equals-eval memorization diagnostics for the ToM pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from script.twd_tom.train import (
    TrainingConfig,
    _atomic_json_write,
    _prepare_run_output_dir,
    build_learning_rate_scheduler,
    build_model,
    build_run_provenance,
    evaluate_model_with_games_and_strata,
    resolve_device,
    set_random_seed,
    train_one_epoch,
)
from werewolf.models.twd_tom.action_features import PublicEventFeatureBuilder
from werewolf.models.twd_tom.belief_backbone import (
    FULL_INPUT_FEATURE_PROFILE,
    QWEN2_BACKBONE_NAME,
    SUPPORTED_BACKBONE_NAMES,
    SUPPORTED_INPUT_FEATURE_PROFILES,
)
from werewolf.models.twd_tom.dataset import load_twd_tom_jsonl
from werewolf.models.twd_tom.dense_dataset import (
    DenseTWDToMDataset,
    collate_dense_twd_tom_games,
)
from werewolf.models.twd_tom.supervision import (
    ALL_ALIVE_SCOPE,
    SUPERVISION_SCOPES,
    load_role_sidecar,
)


MEMORIZATION_REPORT_SCHEMA_VERSION = "classic7_tom_v2_memorization_sanity_v1"


def _selected_game_samples(
    dataset_path: Path,
    *,
    game_count: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    samples = load_twd_tom_jsonl(dataset_path)
    game_ids = sorted({sample["game_id"] for sample in samples})
    if len(game_ids) < game_count:
        raise ValueError(
            f"dataset has only {len(game_ids)} games, requested {game_count}"
        )
    selected = game_ids[:game_count]
    selected_set = set(selected)
    return (
        [sample for sample in samples if sample["game_id"] in selected_set],
        selected,
    )


def _disable_dropout(model: torch.nn.Module) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
            count += 1
    return count


def _sanity_metrics(metrics: dict[str, int | float]) -> dict[str, float]:
    cross_entropy = float(metrics["mean_belief_cross_entropy"])
    target_entropy = float(metrics["mean_belief_target_entropy"])
    return {
        "kl": float(metrics["mean_belief_kl_divergence"]),
        "cross_entropy_minus_target_entropy": cross_entropy - target_entropy,
        "max_probability_error": float(
            metrics["max_belief_probability_error"]
        ),
        "mean_total_variation": float(
            metrics["mean_belief_total_variation"]
        ),
        "gap_closed": float(metrics["normalized_reducible_gap_improvement"]),
    }


def run_memorization_sanity(
    *,
    dataset_path: str | Path,
    output_dir: str | Path,
    game_count: int,
    epochs: int = 500,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.0,
    seed: int = 42,
    device: str = "auto",
    max_seq_len: int = 256,
    backbone: str = QWEN2_BACKBONE_NAME,
    input_feature_profile: str = FULL_INPUT_FEATURE_PROFILE,
    role_sidecar_path: str | Path | None = None,
    supervision_scope: str = ALL_ALIVE_SCOPE,
    disable_dropout: bool = True,
    kl_threshold: float = 1e-3,
) -> dict[str, Any]:
    """Fit and evaluate the exact same 1–4 games with rotation disabled."""

    if isinstance(game_count, bool) or game_count not in {1, 2, 3, 4}:
        raise ValueError("game_count must be one of 1, 2, 3, or 4")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be positive")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if isinstance(kl_threshold, bool) or not isinstance(
        kl_threshold, (int, float)
    ) or kl_threshold <= 0:
        raise ValueError("kl_threshold must be positive")
    if backbone not in SUPPORTED_BACKBONE_NAMES:
        raise ValueError(f"backbone must be one of {SUPPORTED_BACKBONE_NAMES}")
    if input_feature_profile not in SUPPORTED_INPUT_FEATURE_PROFILES:
        raise ValueError(
            "input_feature_profile must be one of "
            f"{SUPPORTED_INPUT_FEATURE_PROFILES}"
        )
    if supervision_scope not in SUPERVISION_SCOPES:
        raise ValueError(f"supervision_scope must be one of {SUPERVISION_SCOPES}")
    if role_sidecar_path is None:
        raise ValueError(
            "memorization diagnostics require a role sidecar for role strata"
        )

    dataset_path = Path(dataset_path)
    lineage_validation_path = dataset_path.parent / "validation.jsonl"
    output_dir = Path(output_dir)
    config = TrainingConfig(
        output_dir=str(output_dir),
        dataset_path=str(dataset_path),
        validation_dataset_path=str(lineage_validation_path),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        device=device,
        max_seq_len=max_seq_len,
        backbone=backbone,
        input_feature_profile=input_feature_profile,
        dense_supervision=True,
        role_sidecar_path=(
            None if role_sidecar_path is None else str(role_sidecar_path)
        ),
        supervision_scope=supervision_scope,
    )
    set_random_seed(seed)
    resolved_device = resolve_device(device)
    provenance = build_run_provenance(
        config,
        resolved_device=resolved_device,
    )
    raw_samples, selected_game_ids = _selected_game_samples(
        dataset_path,
        game_count=game_count,
    )
    observer_roles = (
        None
        if role_sidecar_path is None
        else load_role_sidecar(role_sidecar_path)
    )
    dataset = DenseTWDToMDataset(
        raw_samples,
        feature_builder=PublicEventFeatureBuilder(max_seq_len=max_seq_len),
        enable_cyclic_rotation=False,
        include_private_features=False,
        observer_roles_by_game=observer_roles,
        supervision_scope=supervision_scope,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        collate_fn=collate_dense_twd_tom_games,
        generator=generator,
    )
    model = build_model(config).to(resolved_device)
    disabled_dropout_module_count = _disable_dropout(model) if disable_dropout else 0
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler, schedule = build_learning_rate_scheduler(
        optimizer,
        config=config,
        steps_per_epoch=len(loader),
    )
    _prepare_run_output_dir(output_dir)
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            loader,
            optimizer,
            device=resolved_device,
            gradient_clip_norm=config.gradient_clip_norm,
            lr_scheduler=scheduler,
        )
        (
            evaluation,
            by_game,
            strata,
            strata_by_game,
        ) = evaluate_model_with_games_and_strata(
            model,
            loader,
            device=resolved_device,
        )
        sanity = _sanity_metrics(evaluation)
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "evaluation": evaluation,
            "sanity": sanity,
        }
        history.append(record)
        if best is None or sanity["kl"] < best["sanity"]["kl"]:
            best = record
        if sanity["kl"] < kl_threshold:
            break
    if best is None:
        raise RuntimeError("memorization diagnostic completed no epoch")
    final = history[-1]
    report = {
        "schema_version": MEMORIZATION_REPORT_SCHEMA_VERSION,
        "status": "ok",
        "diagnostic": f"M{1 if game_count == 1 else 2}",
        "train_equals_eval": True,
        "cyclic_rotation_enabled": False,
        "dropout_disabled": disable_dropout,
        "disabled_dropout_module_count": disabled_dropout_module_count,
        "game_count": game_count,
        "selected_game_ids": selected_game_ids,
        "supervision_scope": supervision_scope,
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "kl_threshold": float(kl_threshold),
        "acceptance_gate": {
            "metric": "mean_belief_kl_divergence",
            "threshold": float(kl_threshold),
            "achieved": best["sanity"]["kl"],
            "passed": best["sanity"]["kl"] < kl_threshold,
        },
        "best_epoch": best["epoch"],
        "best_sanity_metrics": best["sanity"],
        "final_sanity_metrics": final["sanity"],
        "final_evaluation_metrics": final["evaluation"],
        "final_by_game": by_game,
        "final_stratified_metrics": strata,
        "final_stratified_by_game": strata_by_game,
        "learning_rate_schedule": schedule,
        "run_provenance": {
            **provenance,
            "lineage_validation_dataset_usage": "lineage_check_only",
            "memorization_source_dataset_sha256": provenance[
                "train_dataset_sha256"
            ],
        },
        "history": history,
    }
    _atomic_json_write(report, output_dir / "memorization_summary.json")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a 1–4 game train-equals-eval ToM sanity check."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--game-count", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--backbone", choices=SUPPORTED_BACKBONE_NAMES, default=QWEN2_BACKBONE_NAME)
    parser.add_argument(
        "--input-feature-profile",
        choices=SUPPORTED_INPUT_FEATURE_PROFILES,
        default=FULL_INPUT_FEATURE_PROFILE,
    )
    parser.add_argument("--role-sidecar", required=True)
    parser.add_argument(
        "--supervision-scope",
        choices=SUPERVISION_SCOPES,
        default=ALL_ALIVE_SCOPE,
    )
    parser.add_argument(
        "--keep-dropout",
        action="store_true",
        help="Keep configured dropout instead of disabling it for memorization.",
    )
    parser.add_argument("--kl-threshold", type=float, default=1e-3)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = run_memorization_sanity(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        game_count=args.game_count,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        max_seq_len=args.max_seq_len,
        backbone=args.backbone,
        input_feature_profile=args.input_feature_profile,
        role_sidecar_path=args.role_sidecar,
        supervision_scope=args.supervision_scope,
        disable_dropout=not args.keep_dropout,
        kl_threshold=args.kl_threshold,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
