"""Run the single public-only non-wolf-alive development OOF diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from script.twd_tom.materialize_development_folds import (
    DEVELOPMENT_FOLD_MANIFEST_FILENAME,
)
from script.twd_tom.materialize_role_sidecar import (
    validate_development_role_sidecar,
)
from script.twd_tom.run_development_oof import (
    DEFAULT_REFERENCE_IMPROVEMENT,
    run_diagnostic_oof,
)
from werewolf.models.twd_tom.annotation_v2 import (
    V1_ANNOTATION_SOURCE,
    V1_EMPTY_UNOBSERVED_BELIEF_SOURCE,
)
from werewolf.models.twd_tom.belief_backbone import (
    NO_PHASE_DAY_INPUT_FEATURE_PROFILE,
    QWEN2_BACKBONE_NAME,
)
from werewolf.models.twd_tom.supervision import NON_WOLF_ALIVE_SCOPE


def run_non_wolf_oof_diagnostic(
    *,
    fold_root: str | Path,
    role_sidecar_path: str | Path,
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
    reference_improvement: float = DEFAULT_REFERENCE_IMPROVEMENT,
) -> dict[str, Any]:
    """Run exactly the controlled non-wolf-alive diagnostic condition."""

    fold_root = Path(fold_root)
    role_sidecar_path = Path(role_sidecar_path)
    validate_development_role_sidecar(
        role_sidecar_path=role_sidecar_path,
        development_fold_manifest_path=(
            fold_root / DEVELOPMENT_FOLD_MANIFEST_FILENAME
        ),
    )
    return run_diagnostic_oof(
        fold_root=fold_root,
        role_sidecar_path=role_sidecar_path,
        output_dir=output_dir,
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
        reference_improvement=reference_improvement,
        private_conditioning=False,
        backbone=QWEN2_BACKBONE_NAME,
        input_feature_profile=NO_PHASE_DAY_INPUT_FEATURE_PROFILE,
        supervision_scope=NON_WOLF_ALIVE_SCOPE,
        speech_annotation_source=V1_ANNOTATION_SOURCE,
        belief_annotation_source=V1_EMPTY_UNOBSERVED_BELIEF_SOURCE,
        speech_v2_annotation_path=None,
        belief_v2_annotation_path=None,
        worst_case_limit=50,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the single public-only non-wolf-alive development OOF diagnostic."
        )
    )
    parser.add_argument("--fold-root", required=True)
    parser.add_argument("--role-sidecar", required=True)
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
    parser.add_argument(
        "--reference-improvement",
        type=float,
        default=DEFAULT_REFERENCE_IMPROVEMENT,
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = run_non_wolf_oof_diagnostic(
        fold_root=args.fold_root,
        role_sidecar_path=args.role_sidecar,
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
        reference_improvement=args.reference_improvement,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
