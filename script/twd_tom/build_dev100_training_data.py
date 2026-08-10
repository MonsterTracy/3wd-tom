"""Build the fixed V2.7 DEV100 formal dataset package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from script.twd_tom.materialize_training_data import (
    materialize_training_data,
    write_jsonl_atomic,
)
from werewolf.models.twd_tom.dataset import (
    TWDToMDataset,
    load_twd_tom_jsonl,
    second_order_effective_subject_mask,
)
from werewolf.models.twd_tom.schema import (
    FORMAL_ANNOTATION_SCHEMA_VERSION,
    FORMAL_LABEL_PROVENANCE,
    FORMALIZATION_POLICY_VERSION,
)


DATASET_ID = "qwen35-v27-dev100-s575-674"
SOURCE_COLLECTION_COMMIT = "844784d7232af7e40633b62c450f72e4c35edb8e"
DEV100_SPLIT_POLICY_VERSION = "v27_dev100_fixed_seed_ranges_v1"
SEED_RANGES = {
    "train": (575, 644),
    "val": (645, 659),
    "test": (660, 674),
}
EXPECTED = {
    "input_snapshots": 1131,
    "collected_games": 100,
    "represented_games": 99,
    "semantic_errors": 18,
    "hard_recovered": 0,
    "unresolved": 18,
    "tom1_rows": 1128,
    "tom2_rows": 1131,
    "valid_tom2_targets": 6497,
    "effective_tom2_targets": 4401,
}
EXPECTED_SPLIT_ROWS = {
    "tom1": {"train": 799, "val": 162, "test": 167},
    "tom2": {"train": 802, "val": 162, "test": 167},
}
EXPECTED_TOM2_LOADER_ROWS = {"train": 649, "val": 132, "test": 134}
EXPECTED_TOM2_EFFECTIVE_TARGETS = {"train": 3129, "val": 629, "test": 643}
SEED_PATTERN = re.compile(r"_seed_(\d+)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required package file not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON file must contain an object: {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"output already exists: {path}") from exc
        temporary.unlink()
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _seed(game_id: str) -> int:
    match = SEED_PATTERN.search(game_id)
    if match is None:
        raise ValueError(f"cannot parse seed from game_id: {game_id!r}")
    return int(match.group(1))


def _split_name(game_id: str) -> str:
    seed = _seed(game_id)
    for split_name, (first_seed, last_seed) in SEED_RANGES.items():
        if first_seed <= seed <= last_seed:
            return split_name
    raise ValueError(f"game seed is outside DEV100 ranges: {game_id}")


def _derived_paths(package_dir: Path) -> dict[str, Path]:
    paths = {
        "raw_tom": package_dir / "raw_tom.jsonl",
        "raw_tom2": package_dir / "raw_tom2.jsonl",
        "materialization_manifest": package_dir / "materialization_manifest.json",
        "split_manifest": package_dir / "split_manifest.json",
    }
    for tom_order in ("tom1", "tom2"):
        for split_name in SEED_RANGES:
            paths[f"{tom_order}_{split_name}"] = (
                package_dir / tom_order / f"{split_name}.jsonl"
            )
    return paths


def _validate_source_package(package_dir: Path) -> tuple[dict[str, Any], Path]:
    manifest = _read_json(package_dir / "dataset_manifest.json")
    raw_path = package_dir / "raw.jsonl"
    if not raw_path.is_file():
        raise FileNotFoundError(f"required package file not found: {raw_path}")
    expected_fields = {
        "dataset_id": DATASET_ID,
        "source_commit": SOURCE_COLLECTION_COMMIT,
        "game_count": EXPECTED["collected_games"],
        "raw_distinct_game_count": EXPECTED["represented_games"],
        "raw_row_count": EXPECTED["input_snapshots"],
        "seed_min": 575,
        "seed_max": 674,
    }
    for field_name, expected in expected_fields.items():
        if manifest.get(field_name) != expected:
            raise ValueError(
                f"dataset manifest {field_name} mismatch: "
                f"expected={expected!r}, actual={manifest.get(field_name)!r}"
            )
    actual_sha256 = _sha256(raw_path)
    if manifest.get("raw_sha256") != actual_sha256:
        raise ValueError("dataset manifest raw_sha256 does not match raw.jsonl")
    if manifest.get("zero_speech_games") != [
        {"game_id": "game_005_seed_589", "seed": 589}
    ]:
        raise ValueError("dataset manifest zero-speech game contract mismatch")
    return manifest, raw_path


def _partition_records(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    result = {split_name: [] for split_name in SEED_RANGES}
    for record in records:
        result[_split_name(record["game_id"])].append(record)
    return result


def _tom2_supervision_counts(path: Path) -> tuple[int, int, int]:
    dataset = TWDToMDataset.from_jsonl(path, tom_order=2)
    valid_targets = sum(
        int(dataset[index]["subject_mask"].sum().item())
        for index in range(len(dataset))
    )
    supervised_indices = dataset.second_order_supervised_indices()
    effective_targets = 0
    for index in supervised_indices:
        item = dataset[index]
        effective_targets += int(
            second_order_effective_subject_mask(
                item["subject_mask"], item["reasoning_player_id"]
            ).sum().item()
        )
    return valid_targets, len(supervised_indices), effective_targets


def build_dev100_training_data(package_dir: str | Path) -> dict[str, Any]:
    package = Path(package_dir)
    manifest, raw_path = _validate_source_package(package)
    destinations = _derived_paths(package)
    existing = [str(path) for path in destinations.values() if path.exists()]
    if existing:
        raise FileExistsError(f"derived outputs already exist: {existing}")

    created_files: list[Path] = []
    created_dirs: list[Path] = []
    try:
        materialization = materialize_training_data(
            raw_path=raw_path,
            tom1_output_path=destinations["raw_tom"],
            tom2_output_path=destinations["raw_tom2"],
        )
        created_files.extend([destinations["raw_tom"], destinations["raw_tom2"]])

        actual_materialization = {
            "input_snapshots": materialization["input_snapshot_count"],
            "semantic_errors": materialization["semantic_error_observer_count"],
            "hard_recovered": materialization["hard_knowledge_recovered_count"],
            "unresolved": materialization["unresolved_observer_count"],
            "tom1_rows": materialization["raw_tom_row_count"],
            "tom2_rows": materialization["raw_tom2_row_count"],
        }
        for field_name, actual in actual_materialization.items():
            if actual != EXPECTED[field_name]:
                raise RuntimeError(
                    f"DEV100 {field_name} mismatch: "
                    f"expected={EXPECTED[field_name]}, actual={actual}"
                )

        tom1_records = load_twd_tom_jsonl(destinations["raw_tom"])
        tom2_records = load_twd_tom_jsonl(destinations["raw_tom2"])
        partitions = {
            "tom1": _partition_records(tom1_records),
            "tom2": _partition_records(tom2_records),
        }
        for tom_order, split_records in partitions.items():
            directory = package / tom_order
            directory.mkdir()
            created_dirs.append(directory)
            for split_name, records in split_records.items():
                if len(records) != EXPECTED_SPLIT_ROWS[tom_order][split_name]:
                    raise RuntimeError(
                        f"DEV100 {tom_order}/{split_name} row mismatch: "
                        f"expected={EXPECTED_SPLIT_ROWS[tom_order][split_name]}, "
                        f"actual={len(records)}"
                    )
                path = destinations[f"{tom_order}_{split_name}"]
                write_jsonl_atomic(path, records)
                created_files.append(path)

        represented_games = {
            record["game_id"] for record in tom2_records
        }
        if len(represented_games) != EXPECTED["represented_games"]:
            raise RuntimeError("DEV100 represented game count mismatch")
        if any(_seed(game_id) == 589 for game_id in represented_games):
            raise RuntimeError("zero-speech seed 589 must not have formal samples")

        split_stats: dict[str, dict[str, Any]] = {}
        total_valid_targets = 0
        total_effective_targets = 0
        for split_name, (first_seed, last_seed) in SEED_RANGES.items():
            tom1_path = destinations[f"tom1_{split_name}"]
            tom2_path = destinations[f"tom2_{split_name}"]
            TWDToMDataset.from_jsonl(tom1_path, tom_order=1)
            valid, loader_rows, effective = _tom2_supervision_counts(tom2_path)
            if loader_rows != EXPECTED_TOM2_LOADER_ROWS[split_name]:
                raise RuntimeError(f"DEV100 {split_name} loader row count mismatch")
            if effective != EXPECTED_TOM2_EFFECTIVE_TARGETS[split_name]:
                raise RuntimeError(
                    f"DEV100 {split_name} effective target count mismatch"
                )
            total_valid_targets += valid
            total_effective_targets += effective
            tom1_games = {record["game_id"] for record in partitions["tom1"][split_name]}
            tom2_games = {record["game_id"] for record in partitions["tom2"][split_name]}
            if tom1_games != tom2_games:
                raise RuntimeError(f"DEV100 {split_name} game assignment mismatch")
            split_stats[split_name] = {
                "seed_first": first_seed,
                "seed_last": last_seed,
                "collected_game_count": last_seed - first_seed + 1,
                "represented_game_count": len(tom2_games),
                "tom1_row_count": len(partitions["tom1"][split_name]),
                "tom2_row_count": len(partitions["tom2"][split_name]),
                "tom2_loader_retained_snapshot_count": loader_rows,
                "tom2_effective_supervised_target_count": effective,
            }
        if total_valid_targets != EXPECTED["valid_tom2_targets"]:
            raise RuntimeError("DEV100 valid ToM2 target count mismatch")
        if total_effective_targets != EXPECTED["effective_tom2_targets"]:
            raise RuntimeError("DEV100 effective ToM2 target count mismatch")

        materialization_manifest = {
            "dataset_id": DATASET_ID,
            "policy_version": FORMALIZATION_POLICY_VERSION,
            "annotation_schema_version": FORMAL_ANNOTATION_SCHEMA_VERSION,
            "label_provenance": FORMAL_LABEL_PROVENANCE,
            "source_collection_commit": SOURCE_COLLECTION_COMMIT,
            "source_raw_sha256": _sha256(raw_path),
            "input_snapshot_count": materialization["input_snapshot_count"],
            "collected_game_count": manifest["game_count"],
            "represented_game_count": manifest["raw_distinct_game_count"],
            "semantic_error_observer_count": materialization[
                "semantic_error_observer_count"
            ],
            "hard_knowledge_recovered_count": materialization[
                "hard_knowledge_recovered_count"
            ],
            "unresolved_observer_count": materialization[
                "unresolved_observer_count"
            ],
            "removed_tom1_snapshot_count": len(
                materialization["removed_tom1_snapshot_keys"]
            ),
            "filtered_tom2_observer_count": len(
                materialization["filtered_tom2_observer_keys"]
            ),
            "raw_tom_row_count": materialization["raw_tom_row_count"],
            "raw_tom2_row_count": materialization["raw_tom2_row_count"],
            "output_sha256": {
                "raw_tom.jsonl": _sha256(destinations["raw_tom"]),
                "raw_tom2.jsonl": _sha256(destinations["raw_tom2"]),
            },
        }
        _write_json_atomic(
            destinations["materialization_manifest"], materialization_manifest
        )
        created_files.append(destinations["materialization_manifest"])

        split_outputs = {
            f"{tom_order}/{split_name}.jsonl": _sha256(
                destinations[f"{tom_order}_{split_name}"]
            )
            for tom_order in ("tom1", "tom2")
            for split_name in SEED_RANGES
        }
        split_manifest = {
            "dataset_id": DATASET_ID,
            "split_policy_version": DEV100_SPLIT_POLICY_VERSION,
            "splits": split_stats,
            "collected_game_count": manifest["game_count"],
            "represented_game_count": len(represented_games),
            "zero_speech_games": manifest["zero_speech_games"],
            "tom1_removed_snapshot_keys": materialization[
                "removed_tom1_snapshot_keys"
            ],
            "output_sha256": split_outputs,
        }
        _write_json_atomic(destinations["split_manifest"], split_manifest)
        created_files.append(destinations["split_manifest"])
    except BaseException:
        for path in reversed(created_files):
            path.unlink(missing_ok=True)
        for path in reversed(created_dirs):
            try:
                path.rmdir()
            except OSError:
                pass
        raise

    return {
        "dataset_id": DATASET_ID,
        "package_dir": str(package),
        "materialization": materialization_manifest,
        "split": split_manifest,
        "valid_tom2_target_count": total_valid_targets,
        "effective_tom2_target_count": total_effective_targets,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the fixed V2.7 DEV100 formal training package."
    )
    parser.add_argument("--package-dir", required=True)
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    result = build_dev100_training_data(args.package_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
