import json
from copy import deepcopy
from pathlib import Path

import pytest

from script.twd_tom.materialize_canonical_dataset import (
    CANONICAL_DATASET_MANIFEST_SCHEMA_VERSION,
    materialize_canonical_dataset,
)
from werewolf.models.twd_tom.dataset import TWDToMDataset
from werewolf.models.twd_tom.public_events import (
    PUBLIC_EVENT_SCHEMA_VERSION,
    public_event_digest,
)
from werewolf.offline_annotation import validate_offline_annotation_record
from werewolf.offline_materialization import validate_offline_tom_training_record
from werewolf.trajectory import (
    OBSERVER_VIEW_PROVENANCE_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    PRE_PUBLIC_SPEECH,
    SIMULATOR_BASELINE,
    TRAJECTORY_SCHEMA_VERSION,
    canonical_digest,
    canonical_json,
)


SOURCE_COMMIT = "bded8b52bbf51d0107e86f8d469eb8a7d621036b"
CODE_COMMIT = "0123456789abcdef0123456789abcdef01234567"


class Backend:
    supports_json_schema = True

    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        return '{"suspected_werewolves":[]}'


def _log(observer_id):
    return {
        "viewer": [observer_id],
        "source": observer_id,
        "target": 0,
        "content": {"Werewolf": 2},
        "day": 1,
        "time": "day one",
        "event": "game_setting",
    }


def _observation(observer_id):
    return {
        "observer_id": observer_id,
        "current_act_idx": 2,
        "identity": "Villager",
        "game_log": [_log(observer_id)],
        "phase": "1_day_speech",
        "valid_action": [],
        "authoritative_public_state": {
            "day": 1,
            "day_or_night": "day",
            "phase": "speech",
            "last_night_result": None,
            "prior_exiles": [],
            "alive_players": [1, 2],
            "suggestible_exile_targets": [1, 2],
        },
    }


@pytest.fixture
def canonical_game_root(tmp_path):
    events = [
        {
            "event_idx": 0,
            "event_type": "phase_change",
            "phase": "1_day_speech",
        },
        {
            "event_idx": 1,
            "event_type": "turn_start",
            "speaker": "player2",
        },
    ]
    trajectory = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "game_id": "game_001",
        "run_id": "run_001",
        "source_commit": SOURCE_COMMIT,
        "simulator_baseline": SIMULATOR_BASELINE,
        "environment_seed": 402,
        "runtime_config": {},
        "runtime_config_digest": canonical_digest({}),
        "players": [
            {
                "player_id": player_id,
                "role": "Villager",
                "profile_name": "profile",
                "backend_id": "backend",
                "model_name": "model",
            }
            for player_id in range(1, 8)
        ],
        "public_event_schema_version": PUBLIC_EVENT_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "initial_public_events": events,
        "transitions": [],
        "termination": {
            "completion_status": "ABORTED",
            "termination_kind": "explicit_handled_abort",
        },
        "public_event_digest": public_event_digest(events),
    }
    trajectory["trajectory_digest"] = canonical_digest(trajectory)

    observer_views = []
    for observer_id in (1, 2):
        observation = _observation(observer_id)
        observer_views.append(
            {
                "observer_id": observer_id,
                "observation": observation,
                "observation_digest": canonical_digest(observation),
            }
        )
    boundary = {
        "boundary_id": "game_001:step_000000:PRE_PUBLIC_SPEECH",
        "boundary_type": PRE_PUBLIC_SPEECH,
        "step_idx": 0,
        "speech_kind": "speech",
        "speaker_id": 2,
        "speech_event_idx": None,
        "public_event_count_at_materialization": len(events),
        "public_event_digest_at_materialization": public_event_digest(events),
        "observer_views": observer_views,
    }
    boundary["boundary_digest"] = canonical_digest(boundary)
    provenance = {
        "schema_version": OBSERVER_VIEW_PROVENANCE_SCHEMA_VERSION,
        "game_id": trajectory["game_id"],
        "run_id": trajectory["run_id"],
        "source_commit": trajectory["source_commit"],
        "simulator_baseline": trajectory["simulator_baseline"],
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "trajectory_digest": trajectory["trajectory_digest"],
        "boundaries": [boundary],
    }
    provenance["artifact_digest"] = canonical_digest(provenance)

    root = tmp_path / "canonical"
    game = root / "games" / "game_0001_seed_402"
    game.mkdir(parents=True)
    (game / "trajectory.json").write_text(
        f"{canonical_json(trajectory)}\n",
        encoding="utf-8",
    )
    (game / "observer_views.json").write_text(
        f"{canonical_json(provenance)}\n",
        encoding="utf-8",
    )
    return root


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _build(canonical_root, output_dir):
    backend = Backend()
    manifest = materialize_canonical_dataset(
        canonical_root=canonical_root,
        output_dir=output_dir,
        annotation_run_id="annotation_run_001",
        code_commit=CODE_COMMIT,
        backend=backend,
        backend_id="annotation_backend",
        model_name="annotation_model",
    )
    assert len(backend.calls) == 4
    return manifest


def test_one_canonical_game_materializes_deterministic_valid_dataset(
    canonical_game_root,
    tmp_path,
):
    first = tmp_path / "dataset_first"
    second = tmp_path / "dataset_second"
    first_manifest = _build(canonical_game_root, first)
    second_manifest = _build(canonical_game_root, second)

    relative_files = {
        path.relative_to(first)
        for path in first.rglob("*")
        if path.is_file()
    }
    assert relative_files == {
        Path("annotations/private_conditioned.jsonl"),
        Path("annotations/public_only.jsonl"),
        Path("tom1.jsonl"),
        Path("tom2.jsonl"),
        Path("manifest.json"),
    }
    assert first_manifest == second_manifest
    for relative_path in relative_files:
        assert (first / relative_path).read_bytes() == (
            second / relative_path
        ).read_bytes()

    private = _read_jsonl(first / "annotations/private_conditioned.jsonl")
    public = _read_jsonl(first / "annotations/public_only.jsonl")
    tom1 = _read_jsonl(first / "tom1.jsonl")
    tom2 = _read_jsonl(first / "tom2.jsonl")
    assert len(private) == len(public) == 2
    assert len(tom1) == len(tom2) == 1
    assert [validate_offline_annotation_record(row) for row in private] == private
    assert [validate_offline_annotation_record(row) for row in public] == public
    assert [validate_offline_tom_training_record(row) for row in tom1] == tom1
    assert [validate_offline_tom_training_record(row) for row in tom2] == tom2
    assert len(TWDToMDataset(tom1, tom_order=1)) == 1
    assert len(TWDToMDataset(tom2, tom_order=2)) == 1

    saved_manifest = json.loads((first / "manifest.json").read_text())
    assert saved_manifest == first_manifest
    assert saved_manifest["schema_version"] == (
        CANONICAL_DATASET_MANIFEST_SCHEMA_VERSION
    )
    recorded_digest = saved_manifest.pop("manifest_digest")
    assert recorded_digest == canonical_digest(saved_manifest)


def test_existing_destination_is_rejected(canonical_game_root, tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        _build(canonical_game_root, destination)


def test_each_game_directory_requires_paired_artifacts(
    canonical_game_root,
    tmp_path,
):
    observer_views = (
        canonical_game_root
        / "games"
        / "game_0001_seed_402"
        / "observer_views.json"
    )
    observer_views.unlink()
    with pytest.raises(FileNotFoundError, match="observer_views.json"):
        _build(canonical_game_root, tmp_path / "dataset")
