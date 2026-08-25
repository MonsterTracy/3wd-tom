"""Materialize deterministic development-only folds without copying test data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from script.twd_tom.materialize_canonical_belief_dataset import (
    validate_split_manifest,
)
from werewolf.models.twd_tom.dataset import load_twd_tom_jsonl
from werewolf.trajectory import canonical_digest, canonical_json


DEVELOPMENT_FOLD_POLICY_VERSION = "classic7_tom_v2_development_hash_kfold_v1"
DEVELOPMENT_FOLD_MANIFEST_SCHEMA_VERSION = (
    "classic7_tom_v2_development_folds_manifest_v1"
)
DEVELOPMENT_FOLD_MANIFEST_FILENAME = "development_folds_manifest.json"


def _positive_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _non_negative_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(canonical_json(value) + "\n")


def _rank_game_ids(game_ids: list[str], *, fold_seed: int) -> list[str]:
    fold_seed = _non_negative_integer(fold_seed, field_name="fold_seed")

    def rank(game_id: str) -> tuple[str, str]:
        payload = f"{DEVELOPMENT_FOLD_POLICY_VERSION}\0{fold_seed}\0{game_id}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest(), game_id

    return sorted(game_ids, key=rank)


def _source_split_manifest(
    train_path: Path,
    validation_path: Path,
) -> tuple[dict[str, Any], Path]:
    if train_path.parent != validation_path.parent:
        raise ValueError("source train and validation paths must be siblings")
    manifest_path = train_path.parent / "split_manifest.json"
    manifest = validate_split_manifest(
        manifest_path,
        verify_split_files=("train", "validation"),
    )
    expected_train = (
        train_path.parent / manifest["output_files"]["train"]["relative_path"]
    ).resolve()
    if train_path != expected_train:
        raise ValueError("source train path differs from split manifest")
    expected_validation = (
        train_path.parent
        / manifest["output_files"]["validation"]["relative_path"]
    ).resolve()
    if validation_path != expected_validation:
        raise ValueError("source validation path differs from split manifest")
    return manifest, manifest_path


def materialize_development_folds(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    output_dir: str | Path,
    fold_count: int = 5,
    fold_seed: int = 42,
) -> dict[str, Any]:
    """Combine original train+validation and create disjoint OOF folds."""

    fold_count = _positive_integer(fold_count, field_name="fold_count")
    fold_seed = _non_negative_integer(fold_seed, field_name="fold_seed")
    resolved_train = Path(train_path).resolve()
    resolved_validation = Path(validation_path).resolve()
    manifest, source_manifest_path = _source_split_manifest(
        resolved_train,
        resolved_validation,
    )
    train_records = load_twd_tom_jsonl(resolved_train)
    validation_records = load_twd_tom_jsonl(resolved_validation)
    records = sorted(
        train_records + validation_records,
        key=lambda record: (record["game_id"], record["step_idx"]),
    )
    by_game: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_game.setdefault(record["game_id"], []).append(record)
    development_ids = set(manifest["game_ids"]["train"]) | set(
        manifest["game_ids"]["validation"]
    )
    sealed_test_ids = set(manifest["game_ids"]["test"])
    if set(by_game) != development_ids:
        raise ValueError("development records differ from source train+validation IDs")
    if development_ids & sealed_test_ids:
        raise ValueError("source development and test IDs overlap")
    if fold_count > len(development_ids):
        raise ValueError("fold_count cannot exceed development game count")

    ranked_ids = _rank_game_ids(sorted(development_ids), fold_seed=fold_seed)
    fold_validation_ids = [ranked_ids[index::fold_count] for index in range(fold_count)]
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"development fold output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp.",
            dir=destination.parent,
        )
    )
    try:
        folds: dict[str, Any] = {}
        for fold_index, validation_ids in enumerate(fold_validation_ids):
            fold_name = f"fold_{fold_index}"
            validation_set = set(validation_ids)
            training_ids = [
                game_id for game_id in ranked_ids if game_id not in validation_set
            ]
            fold_dir = temporary / fold_name
            fold_dir.mkdir()
            fold_train_records = [
                record for game_id in training_ids for record in by_game[game_id]
            ]
            fold_validation_records = [
                record for game_id in validation_ids for record in by_game[game_id]
            ]
            train_output = fold_dir / "train.jsonl"
            validation_output = fold_dir / "validation.jsonl"
            _write_jsonl(train_output, fold_train_records)
            _write_jsonl(validation_output, fold_validation_records)
            folds[fold_name] = {
                "fold_index": fold_index,
                "train_game_ids": training_ids,
                "validation_game_ids": validation_ids,
                "train_row_count": len(fold_train_records),
                "validation_row_count": len(fold_validation_records),
                "train_file": {
                    "relative_path": f"{fold_name}/train.jsonl",
                    "sha256": _sha256(train_output),
                },
                "validation_file": {
                    "relative_path": f"{fold_name}/validation.jsonl",
                    "sha256": _sha256(validation_output),
                },
            }

        relative_source_manifest = os.path.relpath(
            source_manifest_path.resolve(),
            start=destination,
        )
        result: dict[str, Any] = {
            "schema_version": DEVELOPMENT_FOLD_MANIFEST_SCHEMA_VERSION,
            "policy_version": DEVELOPMENT_FOLD_POLICY_VERSION,
            "fold_count": fold_count,
            "fold_seed": fold_seed,
            "source_split_manifest_relative_path": relative_source_manifest,
            "source_split_manifest_sha256": _sha256(source_manifest_path),
            "source_split_manifest_digest": manifest["manifest_digest"],
            "canonical_batch_summary_digest": manifest[
                "canonical_batch_summary_digest"
            ],
            "development_game_ids": ranked_ids,
            "sealed_test_game_ids": sorted(sealed_test_ids),
            "sealed_test_file_sha256": manifest["output_files"]["test"]["sha256"],
            "folds": folds,
        }
        result["manifest_digest"] = canonical_digest(result)
        _write_json(temporary / DEVELOPMENT_FOLD_MANIFEST_FILENAME, result)
        os.replace(temporary, destination)
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"development fold manifest not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("development fold manifest must contain one object")
    return value


def _jsonl_identity(path: Path) -> tuple[int, set[str]]:
    records = load_twd_tom_jsonl(path)
    return len(records), {record["game_id"] for record in records}


def validate_development_fold_paths(
    train_path: str | Path,
    validation_path: str | Path,
) -> dict[str, Any]:
    """Validate one development fold and prove sealed-test disjointness."""

    resolved_train = Path(train_path).resolve()
    resolved_validation = Path(validation_path).resolve()
    if resolved_train.parent != resolved_validation.parent:
        raise ValueError("fold train and validation paths must be siblings")
    root = resolved_train.parent.parent
    manifest_path = root / DEVELOPMENT_FOLD_MANIFEST_FILENAME
    manifest = _load_manifest(manifest_path)
    payload = dict(manifest)
    digest = payload.pop("manifest_digest", None)
    if digest != canonical_digest(payload):
        raise ValueError("development fold manifest digest mismatch")
    if manifest.get("schema_version") != DEVELOPMENT_FOLD_MANIFEST_SCHEMA_VERSION:
        raise ValueError("development fold manifest schema mismatch")
    if manifest.get("policy_version") != DEVELOPMENT_FOLD_POLICY_VERSION:
        raise ValueError("development fold policy mismatch")
    fold_name = resolved_train.parent.name
    descriptor = manifest.get("folds", {}).get(fold_name)
    if not isinstance(descriptor, Mapping):
        raise ValueError("paths do not identify a declared development fold")
    expected_train = (root / descriptor["train_file"]["relative_path"]).resolve()
    expected_validation = (
        root / descriptor["validation_file"]["relative_path"]
    ).resolve()
    if resolved_train != expected_train or resolved_validation != expected_validation:
        raise ValueError("development fold paths differ from manifest")
    if _sha256(resolved_train) != descriptor["train_file"]["sha256"]:
        raise ValueError("development fold train SHA-256 mismatch")
    if _sha256(resolved_validation) != descriptor["validation_file"]["sha256"]:
        raise ValueError("development fold validation SHA-256 mismatch")
    train_rows, train_ids = _jsonl_identity(resolved_train)
    validation_rows, validation_ids = _jsonl_identity(resolved_validation)
    if train_rows != descriptor["train_row_count"] or train_ids != set(
        descriptor["train_game_ids"]
    ):
        raise ValueError("development fold train identity mismatch")
    if validation_rows != descriptor["validation_row_count"] or validation_ids != set(
        descriptor["validation_game_ids"]
    ):
        raise ValueError("development fold validation identity mismatch")
    sealed_ids = set(manifest["sealed_test_game_ids"])
    if train_ids & validation_ids:
        raise ValueError("development fold train and validation overlap")
    if (train_ids | validation_ids) & sealed_ids:
        raise ValueError("development fold contains a sealed test game")
    if train_ids | validation_ids != set(manifest["development_game_ids"]):
        raise ValueError("development fold does not cover the development set")

    source_manifest_path = (
        root / manifest["source_split_manifest_relative_path"]
    ).resolve()
    source_manifest = validate_split_manifest(
        source_manifest_path,
        verify_split_files=("train", "validation"),
    )
    if source_manifest["manifest_digest"] != manifest["source_split_manifest_digest"]:
        raise ValueError("source split manifest digest changed")
    if _sha256(source_manifest_path) != manifest["source_split_manifest_sha256"]:
        raise ValueError("source split manifest SHA-256 changed")
    if set(source_manifest["game_ids"]["test"]) != sealed_ids:
        raise ValueError("sealed test IDs differ from source split")
    result = dict(manifest)
    result["_lineage_manifest_path"] = str(manifest_path)
    result["_source_split_manifest_path"] = str(source_manifest_path)
    result["_fold_name"] = fold_name
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create development-only K-fold datasets from train+validation."
    )
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = materialize_development_folds(
        train_path=args.train,
        validation_path=args.validation,
        output_dir=args.output_dir,
        fold_count=args.fold_count,
        fold_seed=args.fold_seed,
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
