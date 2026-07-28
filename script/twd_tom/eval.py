"""Evaluate a trained classic-seven observer-specific ToM checkpoint.

Evaluation path:

    checkpoint
        -> restore ToMBeliefBackbone
        -> subjective JSONL dataset
        -> masked subjective-belief metrics

By default, evaluation game IDs must be disjoint from all game IDs in
the dataset used to create the checkpoint. This prevents cumulative
histories from one game being evaluated as independent held-out data.

The evaluator never reads true roles or truth-derived Werewolf labels.
Only locally trusted checkpoint files should be loaded.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from script.twd_tom.train import (
    count_supervised_subjects,
    evaluate_model,
    resolve_device,
)
from werewolf.models.twd_tom.action_features import (
    PublicEventFeatureBuilder,
)
from werewolf.models.twd_tom.belief_backbone import (
    BACKBONE_NAME,
    ToMBeliefBackbone,
    ToMBeliefBackboneConfig,
)
from werewolf.models.twd_tom.dataset import (
    TWDToMDataset,
    collate_twd_tom_samples,
    load_twd_tom_jsonl,
)
from werewolf.models.twd_tom.public_events import (
    PHASE_TO_ID,
    PUBLIC_EVENT_SCHEMA_VERSION,
    STRUCTURED_TOKEN_TO_ID,
)
from werewolf.models.twd_tom.schema import (
    GLOBAL_TRUTH_INJECTED,
    LABEL_CONTEXT_SCOPE,
    LABEL_SOURCE,
    MARGINAL_SEMANTICS,
    MODEL_OUTPUT,
    MODEL_INPUT_SCOPE,
    NUMERIC_ANNOTATION_PRESENT,
    NUM_WOLF_PAIR_CLASSES,
    OTHER_PLAYERS_PRIVATE_INFORMATION_VISIBLE,
    OBSERVER_SELECTION,
    OUTPUT_ACTIVATION,
    PAIR_ORDERING,
    PROJECTED_SCHEMA_VERSION as SAMPLE_SCHEMA_VERSION,
    PROJECTION_VERSION,
    PRIVATE_CONTEXT_SERIALIZED,
    REPORT_CONTEXT_MODE,
    REPORT_SIDE_EFFECT_FREE,
    REPORT_TIMING,
    RAW_LABEL_FIELD,
    RAW_LABEL_SEMANTICS,
    RAW_LABEL_TYPE,
    SUPERVISION_SCOPE,
    TARGET_ENCODING,
    TARGET_INTERPRETATION,
    TRUTH_BASED_OBSERVER_SELECTION,
)


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for checkpoint evaluation."""

    checkpoint_path: str
    dataset_path: str

    output_path: str | None = None
    training_dataset_path: str | None = None

    batch_size: int = 32
    device: str = "auto"
    num_workers: int = 0
    allow_game_id_overlap: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(
                self.checkpoint_path,
                str,
            )
            or not self.checkpoint_path.strip()
        ):
            raise ValueError(
                "checkpoint_path is required"
            )

        if (
            not isinstance(
                self.dataset_path,
                str,
            )
            or not self.dataset_path.strip()
        ):
            raise ValueError(
                "dataset_path is required"
            )

        if (
            self.output_path is not None
            and (
                not isinstance(
                    self.output_path,
                    str,
                )
                or not self.output_path.strip()
            )
        ):
            raise ValueError(
                "output_path must be non-empty "
                "text or None"
            )

        if (
            self.training_dataset_path
            is not None
            and (
                not isinstance(
                    self.training_dataset_path,
                    str,
                )
                or not self.training_dataset_path.strip()
            )
        ):
            raise ValueError(
                "training_dataset_path must be "
                "non-empty text or None"
            )

        if (
            isinstance(self.batch_size, bool)
            or not isinstance(
                self.batch_size,
                int,
            )
            or self.batch_size <= 0
        ):
            raise ValueError(
                "batch_size must be a "
                "positive integer"
            )

        if (
            not isinstance(self.device, str)
            or not self.device.strip()
        ):
            raise ValueError(
                "device must be a non-empty string"
            )

        if (
            isinstance(self.num_workers, bool)
            or not isinstance(
                self.num_workers,
                int,
            )
            or self.num_workers < 0
        ):
            raise ValueError(
                "num_workers must be a "
                "non-negative integer"
            )

        if not isinstance(
            self.allow_game_id_overlap,
            bool,
        ):
            raise TypeError(
                "allow_game_id_overlap must "
                "be boolean"
            )


def load_checkpoint(
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Load one locally trusted PyTorch checkpoint on CPU."""

    path = Path(
        checkpoint_path
    ).resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {path}"
        )

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch versions that do not expose
        # the weights_only argument.
        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise TypeError(
            "checkpoint must contain a dictionary"
        )

    return checkpoint


def build_model_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    device: torch.device,
) -> ToMBeliefBackbone:
    """Reconstruct and strictly restore the belief backbone."""

    schema_version = checkpoint.get(
        "schema_version"
    )

    if schema_version != SAMPLE_SCHEMA_VERSION:
        raise ValueError(
            "checkpoint schema mismatch: "
            f"expected {SAMPLE_SCHEMA_VERSION!r}, "
            f"got {schema_version!r}"
        )

    target_encoding = checkpoint.get(
        "target_encoding"
    )
    if target_encoding != TARGET_ENCODING:
        raise ValueError(
            "checkpoint target_encoding mismatch: "
            f"expected {TARGET_ENCODING!r}, "
            f"got {target_encoding!r}"
        )

    pair_class_count = checkpoint.get(
        "pair_class_count"
    )
    if pair_class_count != NUM_WOLF_PAIR_CLASSES:
        raise ValueError(
            "checkpoint pair_class_count mismatch: "
            f"expected {NUM_WOLF_PAIR_CLASSES}, "
            f"got {pair_class_count!r}"
        )

    expected_metadata = {
        "public_event_schema_version": PUBLIC_EVENT_SCHEMA_VERSION,
        "structured_token_to_id": dict(STRUCTURED_TOKEN_TO_ID),
        "public_phase_to_id": dict(PHASE_TO_ID),
        "projection_version": PROJECTION_VERSION,
        "raw_label_field": RAW_LABEL_FIELD,
        "raw_label_type": RAW_LABEL_TYPE,
        "numeric_annotation_present": NUMERIC_ANNOTATION_PRESENT,
        "raw_label_semantics": RAW_LABEL_SEMANTICS,
        "target_interpretation": TARGET_INTERPRETATION,
        "target_distribution_is_reporter_probability": False,
        "target_distribution_is_deterministic_encoding": True,
        "supervision_scope": SUPERVISION_SCOPE,
        "label_source": LABEL_SOURCE,
        "label_context_scope": LABEL_CONTEXT_SCOPE,
        "model_input_scope": MODEL_INPUT_SCOPE,
        "report_context_mode": REPORT_CONTEXT_MODE,
        "report_side_effect_free": REPORT_SIDE_EFFECT_FREE,
        "global_truth_injected": GLOBAL_TRUTH_INJECTED,
        "other_players_private_information_visible": (
            OTHER_PLAYERS_PRIVATE_INFORMATION_VISIBLE
        ),
        "private_context_serialized": PRIVATE_CONTEXT_SERIALIZED,
        "report_timing": REPORT_TIMING,
        "observer_selection": OBSERVER_SELECTION,
        "truth_based_observer_selection": TRUTH_BASED_OBSERVER_SELECTION,
        "pair_ordering": PAIR_ORDERING,
        "model_output": MODEL_OUTPUT,
        "output_activation": OUTPUT_ACTIVATION,
        "backbone": BACKBONE_NAME,
    }
    for field_name, expected_value in expected_metadata.items():
        actual_value = checkpoint.get(field_name)
        if isinstance(expected_value, bool):
            matches_expected = (
                type(actual_value) is bool
                and actual_value is expected_value
            )
        else:
            matches_expected = actual_value == expected_value
        if not matches_expected:
            raise ValueError(
                f"checkpoint {field_name} mismatch: "
                f"expected {expected_value!r}, got {actual_value!r}"
            )

    raw_model_config = checkpoint.get(
        "model_config"
    )

    if not isinstance(
        raw_model_config,
        Mapping,
    ):
        raise TypeError(
            "checkpoint has no valid model_config"
        )

    try:
        model_config = (
            ToMBeliefBackboneConfig(
                **dict(raw_model_config)
            )
        )
    except TypeError as exc:
        raise ValueError(
            "checkpoint model_config is "
            "incompatible with the current model"
        ) from exc

    state_dict = checkpoint.get(
        "model_state_dict"
    )

    if not isinstance(
        state_dict,
        Mapping,
    ):
        raise TypeError(
            "checkpoint has no valid "
            "model_state_dict"
        )

    model = ToMBeliefBackbone(
        model_config
    )

    try:
        model.load_state_dict(
            state_dict,
            strict=True,
        )
    except RuntimeError as exc:
        raise ValueError(
            "checkpoint state_dict is "
            "incompatible with model_config"
        ) from exc

    model.to(device)
    model.eval()

    return model


def collect_game_ids(
    samples: Sequence[
        Mapping[str, Any]
    ],
) -> tuple[str, ...]:
    """Collect sorted unique game IDs from raw samples."""

    if (
        isinstance(samples, (str, bytes))
        or not isinstance(samples, Sequence)
    ):
        raise TypeError(
            "samples must be a sequence"
        )

    game_ids: set[str] = set()

    for sample_index, sample in enumerate(
        samples
    ):
        if not isinstance(
            sample,
            Mapping,
        ):
            raise TypeError(
                "sample "
                f"{sample_index} must be a mapping"
            )

        game_id = sample.get(
            "game_id"
        )

        if (
            not isinstance(game_id, str)
            or not game_id.strip()
        ):
            raise ValueError(
                "every evaluation sample must "
                "contain a non-empty game_id; "
                f"sample index {sample_index} "
                "is invalid"
            )

        game_ids.add(
            game_id
        )

    if not game_ids:
        raise ValueError(
            "evaluation dataset cannot be empty"
        )

    return tuple(
        sorted(game_ids)
    )


def resolve_training_dataset_path(
    checkpoint: Mapping[str, Any],
    *,
    override_path: str | None,
) -> Path:
    """Resolve the dataset used to create the checkpoint."""

    if override_path is not None:
        path = Path(
            override_path
        ).resolve()

        if not path.is_file():
            raise FileNotFoundError(
                "training dataset override "
                f"not found: {path}"
            )

        return path

    training_config = checkpoint.get(
        "training_config"
    )

    if not isinstance(
        training_config,
        Mapping,
    ):
        raise ValueError(
            "checkpoint has no training_config; "
            "provide --training-dataset or "
            "--allow-game-id-overlap"
        )

    raw_dataset_path = training_config.get(
        "train_dataset_path"
    )

    if (
        not isinstance(raw_dataset_path, str)
        or not raw_dataset_path.strip()
    ):
        raise ValueError(
            "checkpoint training_config has no valid "
            "train_dataset_path"
        )

    path = Path(
        raw_dataset_path
    ).resolve()

    if not path.is_file():
        raise FileNotFoundError(
            "checkpoint training dataset "
            f"not found: {path}; provide "
            "--training-dataset or use "
            "--allow-game-id-overlap only "
            "for an intentional smoke test"
        )

    return path


def _atomic_json_dump(
    value: Any,
    path: Path,
) -> None:
    """Write JSON through a temporary file."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(
        path
    )


def evaluate_checkpoint(
    config: EvaluationConfig,
) -> dict[str, Any]:
    """Restore a checkpoint and evaluate subjective beliefs."""

    device = resolve_device(
        config.device
    )

    checkpoint_path = Path(
        config.checkpoint_path
    ).resolve()

    evaluation_dataset_path = Path(
        config.dataset_path
    ).resolve()

    checkpoint = load_checkpoint(
        checkpoint_path
    )

    model = build_model_from_checkpoint(
        checkpoint,
        device=device,
    )

    raw_evaluation_samples = (
        load_twd_tom_jsonl(
            evaluation_dataset_path
        )
    )

    evaluation_game_ids = collect_game_ids(
        raw_evaluation_samples
    )

    training_dataset_path = None
    training_game_ids: tuple[
        str,
        ...
    ] = ()

    overlapping_game_ids: tuple[
        str,
        ...
    ] = ()

    if not config.allow_game_id_overlap:
        training_dataset_path = (
            resolve_training_dataset_path(
                checkpoint,
                override_path=(
                    config.training_dataset_path
                ),
            )
        )

        raw_training_samples = (
            load_twd_tom_jsonl(
                training_dataset_path
            )
        )

        training_game_ids = collect_game_ids(
            raw_training_samples
        )

        overlapping_game_ids = tuple(
            sorted(
                set(evaluation_game_ids)
                & set(training_game_ids)
            )
        )

        if overlapping_game_ids:
            raise ValueError(
                "evaluation game_id values overlap "
                "with the checkpoint training "
                "dataset: "
                f"{list(overlapping_game_ids)}"
            )

    feature_builder = (
        PublicEventFeatureBuilder(
            max_seq_len=(
                model.config.max_seq_len
            )
        )
    )

    evaluation_dataset = TWDToMDataset(
        raw_evaluation_samples,
        feature_builder=feature_builder,
    )

    evaluation_loader = DataLoader(
        evaluation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=(
            collate_twd_tom_samples
        ),
    )

    supervised_subject_count = (
        count_supervised_subjects(
            evaluation_loader
        )
    )

    if supervised_subject_count == 0:
        raise ValueError(
            "evaluation dataset contains "
            "no valid subjective target rows"
        )

    metrics = evaluate_model(
        model,
        evaluation_loader,
        device=device,
    )

    checkpoint_epoch = checkpoint.get(
        "epoch"
    )

    if (
        isinstance(checkpoint_epoch, bool)
        or not isinstance(
            checkpoint_epoch,
            int,
        )
        or checkpoint_epoch <= 0
    ):
        raise ValueError(
            "checkpoint has an invalid epoch"
        )

    summary: dict[str, Any] = {
        "status": "ok",
        "schema_version": (
            SAMPLE_SCHEMA_VERSION
        ),
        "target_encoding": TARGET_ENCODING,
        "projection_version": PROJECTION_VERSION,
        "pair_class_count": NUM_WOLF_PAIR_CLASSES,
        "pair_ordering": PAIR_ORDERING,
        "raw_label_field": RAW_LABEL_FIELD,
        "raw_label_type": RAW_LABEL_TYPE,
        "numeric_annotation_present": NUMERIC_ANNOTATION_PRESENT,
        "raw_label_semantics": RAW_LABEL_SEMANTICS,
        "target_interpretation": TARGET_INTERPRETATION,
        "supervision_scope": SUPERVISION_SCOPE,
        "label_source": LABEL_SOURCE,
        "label_context_scope": LABEL_CONTEXT_SCOPE,
        "model_input_scope": MODEL_INPUT_SCOPE,
        "report_context_mode": REPORT_CONTEXT_MODE,
        "report_side_effect_free": REPORT_SIDE_EFFECT_FREE,
        "global_truth_injected": GLOBAL_TRUTH_INJECTED,
        "other_players_private_information_visible": (
            OTHER_PLAYERS_PRIVATE_INFORMATION_VISIBLE
        ),
        "private_context_serialized": PRIVATE_CONTEXT_SERIALIZED,
        "report_timing": REPORT_TIMING,
        "observer_selection": OBSERVER_SELECTION,
        "truth_based_observer_selection": TRUTH_BASED_OBSERVER_SELECTION,
        "marginal_semantics": MARGINAL_SEMANTICS,
        "model_output": MODEL_OUTPUT,
        "output_activation": OUTPUT_ACTIVATION,
        "backbone": BACKBONE_NAME,
        "device": str(device),
        "checkpoint_path": str(
            checkpoint_path
        ),
        "checkpoint_epoch": (
            checkpoint_epoch
        ),
        "evaluation_dataset_path": str(
            evaluation_dataset_path
        ),
        "evaluation_sample_count": len(
            evaluation_dataset
        ),
        "evaluation_game_ids": list(
            evaluation_game_ids
        ),
        "evaluation_supervised_subject_count": (
            supervised_subject_count
        ),
        "game_id_overlap_check_enabled": (
            not config.allow_game_id_overlap
        ),
        "training_dataset_path": (
            None
            if training_dataset_path is None
            else str(
                training_dataset_path
            )
        ),
        "training_game_ids": list(
            training_game_ids
        ),
        "overlapping_game_ids": list(
            overlapping_game_ids
        ),
        "model_config": {
            "num_players": (
                model.config.num_players
            ),
            "pair_class_count": (
                model.config.pair_class_count
            ),
            "d_model": (
                model.config.d_model
            ),
            "n_head": (
                model.config.n_head
            ),
            "n_layer": (
                model.config.n_layer
            ),
            "dropout": (
                model.config.dropout
            ),
            "max_seq_len": (
                model.config.max_seq_len
            ),
            "dim_feedforward": (
                model.config.dim_feedforward
            ),
        },
        "metrics": metrics,
    }

    if config.output_path is not None:
        output_path = Path(
            config.output_path
        ).resolve()

        _atomic_json_dump(
            summary,
            output_path,
        )

        summary["output_path"] = str(
            output_path
        )

    return summary


def build_arg_parser() -> (
    argparse.ArgumentParser
):
    """Build the checkpoint evaluator CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a subjective seven-player "
            "ToM checkpoint."
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Path to checkpoint_best.pt or "
            "checkpoint_last.pt."
        ),
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "Path to subjective evaluation JSONL."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional JSON evaluation summary path."
        ),
    )

    parser.add_argument(
        "--training-dataset",
        default=None,
        help=(
            "Optional override for the dataset "
            "used to create the checkpoint."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "auto, cpu, cuda, cuda:N or mps"
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--allow-game-id-overlap",
        action="store_true",
        help=(
            "Allow evaluation game IDs to overlap "
            "with training data. Use only for an "
            "intentional pipeline smoke test."
        ),
    )

    return parser


def main() -> int:
    """CLI entry point."""

    args = (
        build_arg_parser()
        .parse_args()
    )

    config = EvaluationConfig(
        checkpoint_path=(
            args.checkpoint
        ),
        dataset_path=args.dataset,
        output_path=args.output,
        training_dataset_path=(
            args.training_dataset
        ),
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
        allow_game_id_overlap=(
            args.allow_game_id_overlap
        ),
    )

    summary = evaluate_checkpoint(
        config
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
