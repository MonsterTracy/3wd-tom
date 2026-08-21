"""Train and evaluate the independent Public Belief Matrix V1 backbone."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from werewolf.models.public_belief_matrix.backbone import (
    PublicBeliefMatrixBackbone,
    PublicBeliefMatrixBackboneConfig,
)
from werewolf.models.public_belief_matrix.collection import (
    PUBLIC_BELIEF_MATRIX_MAX_SEQ_LEN,
    PUBLIC_BELIEF_MATRIX_SAMPLE_SCHEMA_VERSION,
)
from werewolf.models.public_belief_matrix.dataset import (
    PUBLIC_BELIEF_MATRIX_MATERIALIZATION_VERSION,
    PublicBeliefMatrixDataset,
    collate_public_belief_matrix_batch,
)
from werewolf.models.public_belief_matrix.losses import (
    masked_row_soft_target_cross_entropy,
)
from werewolf.models.public_belief_matrix.metrics import (
    masked_mean_row_cross_entropy,
    masked_mean_row_entropy,
    mean_observer_pairwise_tv,
    mean_prediction_diagonal_mass,
)
from werewolf.models.twd_tom.action_features import PublicEventFeatureBuilder


PUBLIC_BELIEF_MATRIX_CHECKPOINT_VERSION = "public_belief_matrix_checkpoint_v1"


@dataclass(frozen=True)
class PublicBeliefMatrixTrainingConfig:
    epochs: int = 30
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be non-empty text")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_inputs(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        field: batch[field].to(device)
        for field in PublicEventFeatureBuilder.FEATURE_FIELDS
    }


def run_epoch(
    model: PublicBeliefMatrixBackbone,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, Any]:
    """Run one train or read-only evaluation epoch using existing PBM semantics."""

    training = optimizer is not None
    model.train(training)
    all_logits = []
    all_probabilities = []
    all_targets = []
    all_masks = []
    for batch in loader:
        targets = batch["matrix_target"].to(device)
        observer_mask = batch["observer_row_mask"].to(device)
        if not observer_mask.any():
            raise ValueError("batch has no valid observer rows")
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(**_model_inputs(batch, device))
            loss = masked_row_soft_target_cross_entropy(
                output["matrix_logits"], targets, observer_mask
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("PBM loss must be finite")
            if training:
                loss.backward()
                if any(
                    parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                ):
                    raise FloatingPointError("model gradients must be finite")
                optimizer.step()
        all_logits.append(output["matrix_logits"].detach().cpu())
        all_probabilities.append(output["matrix_probabilities"].detach().cpu())
        all_targets.append(targets.detach().cpu())
        all_masks.append(observer_mask.detach().cpu())
    if not all_logits:
        raise ValueError("data loader produced no batches")

    logits = torch.cat(all_logits)
    probabilities = torch.cat(all_probabilities)
    targets = torch.cat(all_targets)
    observer_mask = torch.cat(all_masks)
    return {
        "loss": masked_mean_row_cross_entropy(logits, targets, observer_mask),
        "target_entropy": masked_mean_row_entropy(targets, observer_mask),
        "prediction_entropy": masked_mean_row_entropy(
            probabilities, observer_mask
        ),
        "observer_pairwise_tv": mean_observer_pairwise_tv(
            probabilities, observer_mask
        ),
        "target_observer_pairwise_tv": mean_observer_pairwise_tv(
            targets, observer_mask
        ),
        "diagonal_mass": mean_prediction_diagonal_mass(
            probabilities, observer_mask
        ),
        "sample_count": int(logits.shape[0]),
        "valid_row_count": int(observer_mask.sum().item()),
    }


def save_checkpoint(
    path: str | Path,
    *,
    model: PublicBeliefMatrixBackbone,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    training_config: PublicBeliefMatrixTrainingConfig,
    split_manifest_path: str | Path,
) -> None:
    checkpoint = {
        "checkpoint_version": PUBLIC_BELIEF_MATRIX_CHECKPOINT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "backbone_config": asdict(model.config),
        "training_config": asdict(training_config),
        "split_manifest_path": str(Path(split_manifest_path).resolve()),
    }
    torch.save(checkpoint, Path(path))


def build_model_from_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[PublicBeliefMatrixBackbone, dict[str, Any]]:
    """Strictly restore one native PBM checkpoint without compatibility fallback."""

    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a mapping")
    required = {
        "checkpoint_version",
        "model_state_dict",
        "optimizer_state_dict",
        "epoch",
        "backbone_config",
        "training_config",
        "split_manifest_path",
    }
    if set(checkpoint) != required:
        raise ValueError("checkpoint fields do not match the PBM contract")
    if checkpoint["checkpoint_version"] != PUBLIC_BELIEF_MATRIX_CHECKPOINT_VERSION:
        raise ValueError("unsupported PBM checkpoint version")
    try:
        config = PublicBeliefMatrixBackboneConfig(**checkpoint["backbone_config"])
        PublicBeliefMatrixTrainingConfig(**checkpoint["training_config"])
    except TypeError as exc:
        raise ValueError("invalid checkpoint configuration") from exc
    model = PublicBeliefMatrixBackbone(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, checkpoint


def _read_data_manifest(data_dir: Path) -> dict[str, Any]:
    manifest_path = data_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"data manifest not found: {manifest_path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError("data manifest is invalid JSON") from exc
    if not isinstance(manifest, dict):
        raise TypeError("data manifest must be a mapping")
    if manifest.get("source_schema_version") != PUBLIC_BELIEF_MATRIX_SAMPLE_SCHEMA_VERSION:
        raise ValueError("data manifest source schema mismatch")
    if (
        manifest.get("target_materialization_version")
        != PUBLIC_BELIEF_MATRIX_MATERIALIZATION_VERSION
    ):
        raise ValueError("data manifest materialization version mismatch")
    if manifest.get("max_seq_len") != PUBLIC_BELIEF_MATRIX_MAX_SEQ_LEN:
        raise ValueError("data manifest max_seq_len mismatch")
    if manifest.get("game_level_split") is not True or manifest.get("split_overlap") is not False:
        raise ValueError("data manifest is not a disjoint game-level split")
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}:
        raise ValueError("data manifest split contract mismatch")
    game_sets = []
    for split_name in ("train", "validation", "test"):
        summary = splits[split_name]
        if not isinstance(summary, dict):
            raise TypeError(f"{split_name} summary must be a mapping")
        game_ids = summary.get("game_ids")
        if not isinstance(game_ids, list) or len(game_ids) != summary.get("game_count"):
            raise ValueError(f"{split_name} game summary is inconsistent")
        game_sets.append(set(game_ids))
    if game_sets[0] & game_sets[1] or game_sets[0] & game_sets[2] or game_sets[1] & game_sets[2]:
        raise ValueError("data manifest game IDs overlap")
    return manifest


def _make_loaders(
    data_dir: Path,
    *,
    config: PublicBeliefMatrixTrainingConfig,
) -> tuple[dict[str, DataLoader], dict[str, PublicBeliefMatrixDataset]]:
    manifest = _read_data_manifest(data_dir)
    datasets = {
        name: PublicBeliefMatrixDataset(data_dir / f"{name}.jsonl")
        for name in ("train", "validation", "test")
    }
    for name, dataset in datasets.items():
        summary = manifest["splits"][name]
        if len(dataset) != summary.get("sample_count"):
            raise ValueError(f"{name} sample count does not match manifest")
        metadata = [dataset[index]["metadata"] for index in range(len(dataset))]
        if {item["game_id"] for item in metadata} != set(summary["game_ids"]):
            raise ValueError(f"{name} game IDs do not match manifest")
        if {item["seed"] for item in metadata} != set(summary.get("seeds", [])):
            raise ValueError(f"{name} seeds do not match manifest")
    generator = torch.Generator().manual_seed(config.seed)
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=name == "train",
            generator=generator if name == "train" else None,
            collate_fn=collate_public_belief_matrix_batch,
        )
        for name, dataset in datasets.items()
    }
    return loaders, datasets


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def train_public_belief_matrix(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    config: PublicBeliefMatrixTrainingConfig,
) -> dict[str, Any]:
    """Train on train, select on validation, and evaluate test exactly once."""

    data_path = Path(data_dir).resolve()
    output_path = Path(output_dir).resolve()
    if output_path.exists():
        raise FileExistsError(f"output directory already exists: {output_path}")
    data_manifest = _read_data_manifest(data_path)
    loaders, _ = _make_loaders(data_path, config=config)
    device = torch.device(config.device)
    _set_seed(config.seed)
    model = PublicBeliefMatrixBackbone(
        PublicBeliefMatrixBackboneConfig(
            max_seq_len=PUBLIC_BELIEF_MATRIX_MAX_SEQ_LEN
        )
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    output_path.mkdir(parents=True, exist_ok=False)
    manifest_path = data_path / "manifest.json"
    history = []
    best_validation_loss = float("inf")
    best_epoch = None
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(
            model, loaders["train"], device=device, optimizer=optimizer
        )
        with torch.no_grad():
            validation_metrics = run_epoch(
                model, loaders["validation"], device=device, optimizer=None
            )
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "validation_loss": validation_metrics["loss"],
            "train_observer_pairwise_tv": train_metrics["observer_pairwise_tv"],
            "validation_observer_pairwise_tv": validation_metrics[
                "observer_pairwise_tv"
            ],
            "train_diagonal_mass": train_metrics["diagonal_mass"],
            "validation_diagonal_mass": validation_metrics["diagonal_mass"],
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=False, sort_keys=True))
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
            save_checkpoint(
                output_path / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                training_config=config,
                split_manifest_path=manifest_path,
            )
        save_checkpoint(
            output_path / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            training_config=config,
            split_manifest_path=manifest_path,
        )

    best_model, _ = build_model_from_checkpoint(
        output_path / "best.pt", map_location=device
    )
    best_model.to(device)
    with torch.no_grad():
        test_metrics = run_epoch(
            best_model, loaders["test"], device=device, optimizer=None
        )
    run_manifest = {
        "checkpoint_version": PUBLIC_BELIEF_MATRIX_CHECKPOINT_VERSION,
        "training_config": asdict(config),
        "backbone_config": asdict(model.config),
        "data_manifest_path": str(manifest_path),
        "data_manifest": data_manifest,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "test_evaluated_once_after_training": True,
        "test_used_for_model_selection": False,
    }
    _write_json(output_path / "training_history.json", history)
    _write_json(output_path / "test_metrics.json", test_metrics)
    _write_json(output_path / "run_manifest.json", run_manifest)
    return run_manifest


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_argument_parser().parse_args(argv)
    return train_public_belief_matrix(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config=PublicBeliefMatrixTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=args.device,
        ),
    )


if __name__ == "__main__":
    main()
