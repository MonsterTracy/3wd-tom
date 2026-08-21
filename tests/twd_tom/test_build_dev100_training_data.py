from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from archive.legacy_tom.script.twd_tom.build_dev100_training_data import (
    DATASET_ID,
    build_dev100_training_data,
)
from werewolf.models.twd_tom.dataset import (
    TWDToMDataset,
    collate_twd_tom_samples,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PACKAGE = REPOSITORY_ROOT / "datasets" / DATASET_ID


def _copy_source_package(tmp_path, name=DATASET_ID):
    required = (
        "dataset_manifest.json",
        "source_runs.txt",
        "raw.jsonl",
        "projected.jsonl",
    )
    missing = [name for name in required if not (SOURCE_PACKAGE / name).is_file()]
    if missing:
        pytest.skip(f"DEV100 source package is unavailable: {missing}")
    package = tmp_path / name
    package.mkdir()
    for name in required:
        shutil.copy2(SOURCE_PACKAGE / name, package / name)
    return package


def test_dev100_v2_source_package_fails_v3_strict_load(tmp_path):
    package = _copy_source_package(tmp_path)
    with pytest.raises(ValueError, match="canonical serialized form"):
        build_dev100_training_data(package)


def test_dev100_v2_rejection_is_deterministic(tmp_path):
    first = _copy_source_package(tmp_path, "first")
    second = _copy_source_package(tmp_path, "second")
    for package in (first, second):
        with pytest.raises(ValueError, match="canonical serialized form"):
            build_dev100_training_data(package)
