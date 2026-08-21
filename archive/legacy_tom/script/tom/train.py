"""Archived training entry point for the formal ToM pilot."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from archive.legacy_tom.werewolf.models.tom.dataset import TomDataset, collate_batch
from archive.legacy_tom.werewolf.models.tom.losses import (
    masked_soft_target_cross_entropy,
)
from archive.legacy_tom.werewolf.models.tom.model import (
    DROPOUT,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    MAX_SEQUENCE_LENGTH,
    NUM_HEADS,
    NUM_LAYERS,
    BeliefModel,
)


MODEL_INPUT_FIELDS = (
    "event_type_ids",
    "subject_ids",
    "action_ids",
    "object_ids",
    "phase_ids",
    "rounds",
    "dead_players",
    "config_id",
    "attention_mask",
)


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def set_training_seed(seed: int) -> None:
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < 2**32
    ):
        raise ValueError("seed must be an integer in [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if not isinstance(requested, str) or not requested.strip():
        raise ValueError("device must be non-empty text")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _prepare_output_dir(path: Path) -> None:
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise FileExistsError(f"output directory must be absent or empty: {path}")
    else:
        path.mkdir(parents=True)


def _game_ids(path: Path) -> list[str]:
    game_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL row")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, Mapping):
                raise TypeError(f"{path}:{line_number}: row must be a mapping")
            game_id = row.get("game_id")
            if not isinstance(game_id, str) or not game_id.strip():
                raise ValueError(f"{path}:{line_number}: invalid game_id")
            game_ids.add(game_id)
    return sorted(game_ids)


def _load_split_manifest(paths: Mapping[str, Path]) -> dict[str, Any]:
    parents = {path.resolve().parent for path in paths.values()}
    if len(parents) != 1:
        raise ValueError("train, val, and test files must share one split directory")
    manifest_path = next(iter(parents)) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"split manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("split manifest is invalid JSON") from exc
    if not isinstance(manifest, dict):
        raise TypeError("split manifest must be a mapping")
    split_seed = manifest.get("split_seed")
    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise ValueError("split manifest requires an integer split_seed")
    seen = set()
    for name, path in paths.items():
        summary = manifest.get(name)
        if not isinstance(summary, Mapping):
            raise TypeError(f"split manifest requires {name} summary")
        expected = summary.get("game_ids")
        if not isinstance(expected, list) or any(
            not isinstance(game_id, str) or not game_id
            for game_id in expected
        ):
            raise ValueError(f"split manifest {name}.game_ids is invalid")
        actual = _game_ids(path)
        if actual != sorted(expected):
            raise ValueError(f"{name} JSONL game IDs do not match manifest")
        overlap = seen & set(actual)
        if overlap:
            raise ValueError(f"game IDs overlap across splits: {sorted(overlap)}")
        seen.update(actual)
    return manifest


def _loader(
    dataset: TomDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch,
        generator=generator,
        num_workers=0,
    )


def _move_batch(batch: Mapping[str, torch.Tensor], device: torch.device):
    return {name: tensor.to(device) for name, tensor in batch.items()}


def _model_loss(
    model: BeliefModel,
    batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    logits = model(**{name: batch[name] for name in MODEL_INPUT_FIELDS})
    return masked_soft_target_cross_entropy(
        logits,
        batch["target"],
        batch["observer_mask"],
    )


def _train_epoch(
    model: BeliefModel,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
) -> float:
    model.train()
    weighted_loss = 0.0
    valid_rows = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        optimizer.zero_grad()
        loss = _model_loss(model, batch)
        loss.backward()
        optimizer.step()
        batch_valid_rows = int(batch["observer_mask"].sum().item())
        weighted_loss += float(loss.detach().item()) * batch_valid_rows
        valid_rows += batch_valid_rows
    if valid_rows == 0:
        raise ValueError("training data contains no valid observer rows")
    return weighted_loss / valid_rows


def _evaluate_model(
    model: BeliefModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    weighted_loss = 0.0
    valid_rows = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            loss = _model_loss(model, batch)
            batch_valid_rows = int(batch["observer_mask"].sum().item())
            weighted_loss += float(loss.item()) * batch_valid_rows
            valid_rows += batch_valid_rows
    if valid_rows == 0:
        raise ValueError("evaluation data contains no valid observer rows")
    return weighted_loss / valid_rows


def _uniform_ce(loader: DataLoader, device: torch.device) -> float:
    weighted_loss = 0.0
    valid_rows = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        uniform_logits = torch.zeros_like(batch["target"])
        loss = masked_soft_target_cross_entropy(
            uniform_logits,
            batch["target"],
            batch["observer_mask"],
        )
        batch_valid_rows = int(batch["observer_mask"].sum().item())
        weighted_loss += float(loss.item()) * batch_valid_rows
        valid_rows += batch_valid_rows
    if valid_rows == 0:
        raise ValueError("baseline data contains no valid observer rows")
    return weighted_loss / valid_rows


def _model_config() -> dict[str, int | float | str]:
    return {
        "class": "BeliefModel",
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "dropout": DROPOUT,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
    }


def train_and_evaluate(
    *,
    train_path: str | Path,
    val_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-2,
    batch_size: int = 16,
    max_epochs: int = 30,
    seed: int = 42,
    device: str = "auto",
) -> dict[str, Any]:
    """Train by validation CE, reload the best checkpoint, and test once."""

    _positive_int(batch_size, field="batch_size")
    _positive_int(max_epochs, field="max_epochs")
    if isinstance(learning_rate, bool) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if isinstance(weight_decay, bool) or weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")
    set_training_seed(seed)
    resolved_device = resolve_device(device)
    paths = {
        "train": Path(train_path).resolve(),
        "val": Path(val_path).resolve(),
        "test": Path(test_path).resolve(),
    }
    if any(not path.is_file() for path in paths.values()):
        missing = [str(path) for path in paths.values() if not path.is_file()]
        raise FileNotFoundError(f"split JSONL file(s) not found: {missing}")
    manifest = _load_split_manifest(paths)
    destination = Path(output_dir).resolve()
    _prepare_output_dir(destination)

    datasets = {name: TomDataset(path) for name, path in paths.items()}
    loaders = {
        "train": _loader(
            datasets["train"], batch_size=batch_size, shuffle=True, seed=seed
        ),
        "val": _loader(
            datasets["val"], batch_size=batch_size, shuffle=False, seed=seed
        ),
        "test": _loader(
            datasets["test"], batch_size=batch_size, shuffle=False, seed=seed
        ),
    }
    model = BeliefModel().to(resolved_device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    training_arguments = {
        "train": str(paths["train"]),
        "val": str(paths["val"]),
        "test": str(paths["test"]),
        "output_dir": str(destination),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "seed": seed,
        "device": str(resolved_device),
        "split_seed": manifest["split_seed"],
    }

    best_val_ce = math.inf
    best_epoch = None
    best_path = destination / "best.pt"
    history_path = destination / "history.jsonl"
    with history_path.open("x", encoding="utf-8") as history_file:
        for epoch in range(1, max_epochs + 1):
            train_ce = _train_epoch(
                model,
                loaders["train"],
                optimizer,
                resolved_device,
            )
            val_ce = _evaluate_model(model, loaders["val"], resolved_device)
            history_file.write(
                json.dumps(
                    {"epoch": epoch, "train_ce": train_ce, "val_ce": val_ce},
                    sort_keys=True,
                )
                + "\n"
            )
            history_file.flush()
            if val_ce < best_val_ce:
                best_val_ce = val_ce
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "validation_ce": val_ce,
                        "model_config": _model_config(),
                        "training_arguments": training_arguments,
                    },
                    best_path,
                )

    if best_epoch is None or not best_path.is_file():
        raise RuntimeError("training did not produce a best checkpoint")
    checkpoint = torch.load(
        best_path,
        map_location=resolved_device,
        weights_only=True,
    )
    best_model = BeliefModel().to(resolved_device)
    best_model.load_state_dict(checkpoint["model_state_dict"])
    test_ce = _evaluate_model(best_model, loaders["test"], resolved_device)
    val_uniform_ce = _uniform_ce(loaders["val"], resolved_device)
    test_uniform_ce = _uniform_ce(loaders["test"], resolved_device)

    metrics = {
        "train_games": manifest["train"]["game_ids"],
        "val_games": manifest["val"]["game_ids"],
        "test_games": manifest["test"]["game_ids"],
        "best_epoch": best_epoch,
        "best_val_ce": best_val_ce,
        "test_ce": test_ce,
        "val_uniform_ce": val_uniform_ce,
        "test_uniform_ce": test_uniform_ce,
        "test_improvement_over_uniform": test_uniform_ce - test_ce,
        "training_seed": seed,
        "split_seed": manifest["split_seed"],
        "device": str(resolved_device),
    }
    (destination / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = train_and_evaluate(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
