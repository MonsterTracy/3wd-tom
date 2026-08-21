import json

import pytest

from script.public_belief_matrix import build_training_data as builder
from werewolf.models.public_belief_matrix.collection import (
    PUBLIC_BELIEF_MATRIX_SAMPLE_SCHEMA_VERSION,
)


def _write_source(tmp_path, samples, seeds=(876, 877, 878, 879, 880)):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    input_path = source_dir / "formal_batch_samples.jsonl"
    input_path.write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples),
        encoding="utf-8",
    )
    (source_dir / "formal_batch_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": PUBLIC_BELIEF_MATRIX_SAMPLE_SCHEMA_VERSION,
                "source_commit": "a" * 40,
                "seeds": list(seeds),
                "requested_game_count": len(seeds),
                "completed_game_count": len(seeds),
            }
        ),
        encoding="utf-8",
    )
    return input_path


def _build(tmp_path, input_path, **overrides):
    arguments = {
        "input_path": input_path,
        "output_dir": tmp_path / "output",
        "train_seeds": (876, 877, 878),
        "validation_seeds": (879,),
        "test_seeds": (880,),
    }
    arguments.update(overrides)
    return builder.build_training_data(**arguments)


def test_fixed_seed_split_is_game_level_and_preserves_symbolic_lineage(
    tmp_path, monkeypatch, pbm_sample_factory
):
    samples = [pbm_sample_factory(seed=seed) for seed in range(876, 881)]
    input_path = _write_source(tmp_path, samples)
    monkeypatch.setattr(builder, "_git_commit", lambda: "b" * 40)
    manifest = _build(tmp_path, input_path)

    assert manifest["splits"]["train"]["seeds"] == [876, 877, 878]
    assert manifest["splits"]["validation"]["seeds"] == [879]
    assert manifest["splits"]["test"]["seeds"] == [880]
    assert [manifest["splits"][name]["game_count"] for name in ("train", "validation", "test")] == [3, 1, 1]
    game_sets = [set(manifest["splits"][name]["game_ids"]) for name in ("train", "validation", "test")]
    assert not game_sets[0] & game_sets[1]
    assert not game_sets[0] & game_sets[2]
    assert not game_sets[1] & game_sets[2]
    row = json.loads((tmp_path / "output" / "train.jsonl").read_text().splitlines()[0])
    assert row["sample"]["observer_reports"] == samples[0]["observer_reports"]
    assert row["sample"]["structured_prefix"] == samples[0]["structured_prefix"]
    assert row["matrix_target"][0] == pytest.approx([1.0 / 7.0] * 7)


def test_seed_overlap_missing_seed_and_unspecified_seed_fail_closed(
    tmp_path, monkeypatch, pbm_sample_factory
):
    monkeypatch.setattr(builder, "_git_commit", lambda: "b" * 40)
    samples = [pbm_sample_factory(seed=seed) for seed in range(876, 881)]
    input_path = _write_source(tmp_path, samples)
    with pytest.raises(ValueError, match="overlap"):
        _build(tmp_path, input_path, validation_seeds=(878, 879))
    with pytest.raises(ValueError, match="exactly cover"):
        _build(tmp_path, input_path, test_seeds=(881,))
    with pytest.raises(ValueError, match="cannot be empty"):
        _build(tmp_path, input_path, test_seeds=())


def test_duplicate_snapshot_and_game_count_mismatch_fail_closed(
    tmp_path, monkeypatch, pbm_sample_factory
):
    monkeypatch.setattr(builder, "_git_commit", lambda: "b" * 40)
    samples = [pbm_sample_factory(seed=seed) for seed in range(876, 881)]
    samples.append(samples[0])
    input_path = _write_source(tmp_path, samples)
    with pytest.raises(ValueError, match="duplicate snapshot_id"):
        _build(tmp_path, input_path)

    other = tmp_path / "other"
    other.mkdir()
    input_path = _write_source(other, samples[:-1])
    manifest_path = input_path.parent / "formal_batch_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["completed_game_count"] = 4
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="game count"):
        _build(other, input_path)
