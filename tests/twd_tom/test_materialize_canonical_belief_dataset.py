import hashlib
import json
from copy import deepcopy

import pytest

from script.twd_tom.materialize_canonical_belief_dataset import (
    SPLIT_NAMES,
    SPLIT_MANIFEST_SCHEMA_VERSION,
    SPLIT_POLICY_VERSION,
    _rank_game_ids,
    materialize_canonical_belief_dataset,
    validate_materialized_split_path,
    validate_split_manifest,
)
from script.twd_tom.train import validate_training_split_lineage
from werewolf.models.twd_tom.dataset import load_twd_tom_jsonl
from werewolf.models.twd_tom.public_events import (
    public_event_digest,
    structured_input_digest,
)
from werewolf.models.twd_tom.speech_annotations import (
    make_speech_annotation,
    speech_annotation_digest,
)
from werewolf.trajectory import canonical_digest


def _later_snapshot(sample):
    later = deepcopy(sample)
    later["step_idx"] += 1
    later["label_cutoff_step_idx"] = later["step_idx"]
    speaker = f"player{later['speaker_id']}"
    later["public_events"].extend(
        [
            {
                "event_idx": len(later["public_events"]),
                "event_type": "public_speech",
                "speaker": speaker,
                "raw_text": "later synthetic speech",
            },
            {
                "event_idx": len(later["public_events"]) + 1,
                "event_type": "turn_start",
                "speaker": "player3",
            },
        ]
    )
    speech_event = later["public_events"][-2]
    later["speech_annotations"].append(
        make_speech_annotation(
            event_idx=speech_event["event_idx"],
            speaker=speaker,
            raw_text=speech_event["raw_text"],
            parser_model_id="synthetic_parser",
            parser_call_id=f"synthetic_{speech_event['event_idx']:06d}",
            annotation_source="llm_parser",
            status="ok",
            actions=[[speaker, "support", "player4"]],
            raw_response=None,
            error_type=None,
            error_message=None,
        )
    )
    later["speaker_id"] = 3
    later["public_event_digest"] = public_event_digest(later["public_events"])
    later["speech_annotation_digest"] = speech_annotation_digest(
        later["speech_annotations"]
    )
    later["structured_input_digest"] = structured_input_digest(
        later["public_events"], later["speech_annotations"]
    )
    later["public_action_count"] += 1
    return later


def test_materialization_is_deterministic_and_never_splits_one_game(
    tmp_path,
    suspicion_sample_factory,
    canonical_belief_batch_factory,
):
    game_ids = [f"game_{index}" for index in range(1, 6)]
    samples_by_game = {
        game_id: [suspicion_sample_factory(game_id=game_id)]
        for game_id in game_ids
    }
    samples_by_game["game_1"].append(
        _later_snapshot(samples_by_game["game_1"][0])
    )
    first_root = tmp_path / "canonical_a"
    second_root = tmp_path / "canonical_b"
    canonical_belief_batch_factory(first_root, samples_by_game)
    canonical_belief_batch_factory(second_root, samples_by_game, reverse=True)

    first_output = tmp_path / "dataset_a"
    second_output = tmp_path / "dataset_b"
    first_summary = materialize_canonical_belief_dataset(
        canonical_root=first_root,
        output_dir=first_output,
        split_seed=17,
        train_game_count=2,
        validation_game_count=1,
        test_game_count=2,
    )
    second_summary = materialize_canonical_belief_dataset(
        canonical_root=second_root,
        output_dir=second_output,
        split_seed=17,
        train_game_count=2,
        validation_game_count=1,
        test_game_count=2,
    )

    assert first_summary["schema_version"] == SPLIT_MANIFEST_SCHEMA_VERSION
    assert first_summary["split_policy_version"] == SPLIT_POLICY_VERSION
    expected_rank = _rank_game_ids(game_ids, split_seed=17)
    assert first_summary["game_ids"] == {
        "train": expected_rank[:2],
        "validation": expected_rank[2:3],
        "test": expected_rank[3:],
    }
    assert second_summary["game_ids"] == first_summary["game_ids"]
    assert second_summary["game_counts"] == first_summary["game_counts"]
    assert second_summary["row_counts"] == first_summary["row_counts"]
    assert (
        second_summary["canonical_batch_summary_digest"]
        != first_summary["canonical_batch_summary_digest"]
    )
    for split_name in SPLIT_NAMES:
        assert (first_output / f"{split_name}.jsonl").read_bytes() == (
            second_output / f"{split_name}.jsonl"
        ).read_bytes()
        digest = hashlib.sha256(
            (first_output / f"{split_name}.jsonl").read_bytes()
        ).hexdigest()
        assert first_summary["output_files"][split_name]["sha256"] == digest
        assert second_summary["output_files"][split_name]["sha256"] == digest
    manifest = json.loads(
        (first_output / "split_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest == first_summary
    manifest_payload = dict(manifest)
    manifest_digest = manifest_payload.pop("manifest_digest")
    assert manifest_digest == canonical_digest(manifest_payload)
    assert validate_split_manifest(
        first_output / "split_manifest.json"
    ) == first_summary
    assert validate_materialized_split_path(
        first_output / "train.jsonl",
        split_name="train",
    ) == first_summary
    assert validate_training_split_lineage(
        first_output / "train.jsonl",
        first_output / "validation.jsonl",
    ) == first_summary
    with pytest.raises(ValueError, match="validation path"):
        validate_training_split_lineage(
            first_output / "train.jsonl",
            first_output / "test.jsonl",
        )

    output_records = {
        split_name: load_twd_tom_jsonl(first_output / f"{split_name}.jsonl")
        for split_name in SPLIT_NAMES
    }
    split_sets = [
        {record["game_id"] for record in output_records[split_name]}
        for split_name in SPLIT_NAMES
    ]
    assert all(
        split_sets[left].isdisjoint(split_sets[right])
        for left in range(3)
        for right in range(left + 1, 3)
    )
    assert set.union(*split_sets) == set(game_ids)
    assert sum(
        record["game_id"] == "game_1"
        for records in output_records.values()
        for record in records
    ) == 2
    source_records = sorted(
        (record for records in samples_by_game.values() for record in records),
        key=lambda record: (record["game_id"], record["step_idx"]),
    )
    materialized_records = sorted(
        (record for records in output_records.values() for record in records),
        key=lambda record: (record["game_id"], record["step_idx"]),
    )
    assert materialized_records == source_records

    train_path = first_output / "train.jsonl"
    train_path.write_text(
        train_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="train output SHA-256 mismatch"):
        validate_split_manifest(first_output / "split_manifest.json")


def test_materialization_rejects_bad_counts_and_existing_destination(
    tmp_path,
    suspicion_sample_factory,
    canonical_belief_batch_factory,
):
    canonical_root = tmp_path / "canonical"
    samples = {
        f"game_{index}": [suspicion_sample_factory(game_id=f"game_{index}")]
        for index in range(1, 4)
    }
    canonical_belief_batch_factory(canonical_root, samples)

    with pytest.raises(ValueError, match="sum exactly"):
        materialize_canonical_belief_dataset(
            canonical_root=canonical_root,
            output_dir=tmp_path / "bad_counts",
            split_seed=0,
            train_game_count=1,
            validation_game_count=1,
            test_game_count=2,
        )

    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        materialize_canonical_belief_dataset(
            canonical_root=canonical_root,
            output_dir=destination,
            split_seed=0,
            train_game_count=1,
            validation_game_count=1,
            test_game_count=1,
        )
