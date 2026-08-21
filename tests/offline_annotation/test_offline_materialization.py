import json
from copy import deepcopy

import pytest

from werewolf.models.twd_tom.public_events import (
    PUBLIC_EVENT_SCHEMA_VERSION,
    public_event_digest,
    structured_input_digest,
)
from werewolf.offline_annotation import (
    PRIVATE_CONDITIONED_SUSPICION_TASK,
    PUBLIC_ONLY_SUSPICION_TASK,
    annotate_pre_speech_suspicion,
)
from werewolf.offline_materialization import (
    D_MATERIALIZATION_POLICY_VERSION,
    D_RECORD_FIELDS,
    D_SCHEMA_VERSION,
    OBSERVER_ANNOTATION_CONFIDENCE,
    OFFLINE_PRIVATE_CONDITIONED_TOM1_TASK,
    OFFLINE_PUBLIC_ONLY_TOM2_TASK,
    TOM1_MODEL_INPUT_SCOPE,
    TOM1_OBSERVER_PROVENANCE,
    TOM1_PRIVATE_FIELDS_USAGE,
    TOM2_MODEL_INPUT_SCOPE,
    TOM2_OBSERVER_PROVENANCE,
    TOM2_PRIVATE_FIELDS_USAGE,
    materialize_offline_tom_records,
    validate_offline_tom_training_record,
    write_offline_tom_jsonl,
)
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
ANNOTATION_COMMIT = "b99c07eda9222fa0b67bcfb20d49fd4c9c80f552"
MATERIALIZER_COMMIT = "0123456789abcdef0123456789abcdef01234567"


class Backend:
    supports_json_schema = True

    def __init__(self, responses):
        self.responses = list(responses)

    def chat(self, **_kwargs):
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


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


def _observation(observer_id, speaker_id):
    return {
        "observer_id": observer_id,
        "current_act_idx": speaker_id,
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
            "alive_players": [1, 2, 3],
            "suggestible_exile_targets": [1, 2, 3],
        },
    }


def _boundary(*, step_idx, speaker_id, events):
    observer_views = []
    for observer_id in (1, 2, 3):
        observation = _observation(observer_id, speaker_id)
        observer_views.append(
            {
                "observer_id": observer_id,
                "observation": observation,
                "observation_digest": canonical_digest(observation),
            }
        )
    boundary = {
        "boundary_id": (
            f"game_001:step_{step_idx:06d}:{PRE_PUBLIC_SPEECH}"
        ),
        "boundary_type": PRE_PUBLIC_SPEECH,
        "step_idx": step_idx,
        "speech_kind": "speech",
        "speaker_id": speaker_id,
        "speech_event_idx": None,
        "public_event_count_at_materialization": len(events),
        "public_event_digest_at_materialization": public_event_digest(events),
        "observer_views": observer_views,
    }
    boundary["boundary_digest"] = canonical_digest(boundary)
    return boundary


def _artifacts():
    first_prefix = [
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
    appended = [
        {
            "event_idx": 2,
            "event_type": "public_speech",
            "speaker": "player2",
            "raw_text": "player5 looks suspicious",
            "sp_actions": [["player2", "oppose", "player5"]],
        },
        {
            "event_idx": 3,
            "event_type": "turn_start",
            "speaker": "player3",
        },
    ]
    complete_events = first_prefix + appended
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
        "initial_public_events": first_prefix,
        "transitions": [
            {
                "step_idx": 0,
                "acting_player_id": 2,
                "public_event_count_before": 2,
                "public_events_appended": appended,
            }
        ],
        "termination": {
            "completion_status": "ABORTED",
            "termination_kind": "explicit_handled_abort",
        },
        "public_event_digest": public_event_digest(complete_events),
    }
    trajectory["trajectory_digest"] = canonical_digest(trajectory)
    provenance = {
        "schema_version": OBSERVER_VIEW_PROVENANCE_SCHEMA_VERSION,
        "game_id": trajectory["game_id"],
        "run_id": trajectory["run_id"],
        "source_commit": trajectory["source_commit"],
        "simulator_baseline": trajectory["simulator_baseline"],
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "trajectory_digest": trajectory["trajectory_digest"],
        "boundaries": [
            _boundary(step_idx=1, speaker_id=3, events=complete_events),
            _boundary(step_idx=0, speaker_id=2, events=first_prefix),
        ],
    }
    provenance["artifact_digest"] = canonical_digest(provenance)
    return trajectory, provenance


def _annotations(*, private_responses=None, public_responses=None):
    trajectory, provenance = _artifacts()
    ok = ['{"suspected_werewolves":[]}'] * 6
    private = annotate_pre_speech_suspicion(
        trajectory,
        provenance,
        annotation_task=PRIVATE_CONDITIONED_SUSPICION_TASK,
        annotation_run_id="private_run",
        annotation_code_commit=ANNOTATION_COMMIT,
        backend=Backend(private_responses or ok),
        backend_id="private_backend",
        model_name="private_model",
    )
    public = annotate_pre_speech_suspicion(
        trajectory,
        provenance,
        annotation_task=PUBLIC_ONLY_SUSPICION_TASK,
        annotation_run_id="public_run",
        annotation_code_commit=ANNOTATION_COMMIT,
        backend=Backend(public_responses or ok),
        backend_id="public_backend",
        model_name="public_model",
    )
    return trajectory, provenance, private + public


def _materialize(records=None):
    trajectory, provenance, generated = _annotations()
    return materialize_offline_tom_records(
        trajectory,
        provenance,
        generated if records is None else records,
        materializer_code_commit=MATERIALIZER_COMMIT,
    )


def _redigest(record):
    record.pop("record_digest", None)
    record["record_digest"] = canonical_digest(record)


def _set_non_ok(record, status):
    record["status"] = status
    record["result"] = None
    record["error"] = f"synthetic {status}"
    record["raw_response"] = None if status == "reporter_error" else "invalid"
    _redigest(record)


def test_exact_d_contract_tasks_order_and_public_prefixes():
    trajectory, provenance, records = _annotations()
    result = materialize_offline_tom_records(
        trajectory,
        provenance,
        list(reversed(records)),
        materializer_code_commit=MATERIALIZER_COMMIT,
    )
    tom1 = result["tom1_records"]
    tom2 = result["tom2_records"]

    assert [row["step_idx"] for row in tom1] == [0, 1]
    assert [row["step_idx"] for row in tom2] == [0, 1]
    assert all(set(row) == D_RECORD_FIELDS for row in tom1 + tom2)
    assert all(row["schema_version"] == D_SCHEMA_VERSION for row in tom1 + tom2)
    assert all(
        row["materialization_policy_version"]
        == D_MATERIALIZATION_POLICY_VERSION
        for row in tom1 + tom2
    )
    assert [row["observer_ids"] for row in tom1] == [[2], [3]]
    assert [row["observer_ids"] for row in tom2] == [[1, 3], [1, 2]]
    assert tom1[0]["public_events"] == trajectory["initial_public_events"]
    assert tom2[1]["public_events"] == (
        trajectory["initial_public_events"]
        + trajectory["transitions"][0]["public_events_appended"]
    )
    assert tom1[0]["public_action_count"] == 0
    assert tom1[1]["public_action_count"] == 1
    assert tom1[0]["public_event_digest"] == (
        provenance["boundaries"][1]["public_event_digest_at_materialization"]
    )
    assert tom1[1]["structured_input_digest"] == structured_input_digest(
        tom1[1]["public_events"]
    )
    assert result["summary"] == {
        "private_status_counts": {
            "ok": 6,
            "parse_error": 0,
            "semantic_error": 0,
            "reporter_error": 0,
        },
        "public_status_counts": {
            "ok": 6,
            "parse_error": 0,
            "semantic_error": 0,
            "reporter_error": 0,
        },
        "emitted_tom1_rows": 2,
        "emitted_tom2_rows": 2,
        "dropped_tom1_boundaries": 0,
        "filtered_tom2_observers": 0,
        "dropped_tom2_boundaries": 0,
    }


def test_speech_pk_maps_to_exact_phase_and_report_trigger():
    trajectory, provenance = _artifacts()
    trajectory["initial_public_events"][0]["phase"] = "1_day_speech_pk"
    complete_events = (
        trajectory["initial_public_events"]
        + trajectory["transitions"][0]["public_events_appended"]
    )
    trajectory["public_event_digest"] = public_event_digest(complete_events)
    trajectory.pop("trajectory_digest")
    trajectory["trajectory_digest"] = canonical_digest(trajectory)

    boundary = provenance["boundaries"][1]
    boundary["speech_kind"] = "speech_pk"
    boundary["public_event_digest_at_materialization"] = public_event_digest(
        trajectory["initial_public_events"]
    )
    for view in boundary["observer_views"]:
        view["observation"]["phase"] = "1_day_speech_pk"
        view["observation"]["authoritative_public_state"]["phase"] = (
            "speech_pk"
        )
        view["observation_digest"] = canonical_digest(view["observation"])
    boundary.pop("boundary_digest")
    boundary["boundary_digest"] = canonical_digest(boundary)
    provenance["boundaries"] = [boundary]
    provenance["trajectory_digest"] = trajectory["trajectory_digest"]
    provenance.pop("artifact_digest")
    provenance["artifact_digest"] = canonical_digest(provenance)

    private = annotate_pre_speech_suspicion(
        trajectory,
        provenance,
        annotation_task=PRIVATE_CONDITIONED_SUSPICION_TASK,
        annotation_run_id="private_run",
        annotation_code_commit=ANNOTATION_COMMIT,
        backend=Backend(['{"suspected_werewolves":[]}'] * 3),
        backend_id="private_backend",
        model_name="private_model",
    )
    public = annotate_pre_speech_suspicion(
        trajectory,
        provenance,
        annotation_task=PUBLIC_ONLY_SUSPICION_TASK,
        annotation_run_id="public_run",
        annotation_code_commit=ANNOTATION_COMMIT,
        backend=Backend(['{"suspected_werewolves":[]}'] * 3),
        backend_id="public_backend",
        model_name="public_model",
    )
    result = materialize_offline_tom_records(
        trajectory,
        provenance,
        private + public,
        materializer_code_commit=MATERIALIZER_COMMIT,
    )

    for row in result["tom1_records"] + result["tom2_records"]:
        assert row["phase"] == "speech_pk"
        assert row["report_trigger"] == "pre_public_speech_pk"


def test_tom1_uses_only_speaker_private_c1_and_exact_hard_knowledge():
    trajectory, provenance, records = _annotations()
    speaker_source = next(
        record
        for record in records
        if record["annotation_task"] == PRIVATE_CONDITIONED_SUSPICION_TASK
        and record["step_idx"] == 0
        and record["observer_id"] == 2
    )
    speaker_source["result"] = {"suspected_werewolves": ["player5"]}
    speaker_source["raw_response"] = (
        '{"suspected_werewolves":["player5"]}'
    )
    _redigest(speaker_source)
    result = materialize_offline_tom_records(
        trajectory,
        provenance,
        records,
        materializer_code_commit=MATERIALIZER_COMMIT,
    )
    row = result["tom1_records"][0]
    subject = "player2"
    hard = speaker_source["source"]["derived_hard_knowledge"]

    assert row["materialization_task"] == OFFLINE_PRIVATE_CONDITIONED_TOM1_TASK
    assert row["tom_order"] == 1
    assert row["model_input_scope"] == TOM1_MODEL_INPUT_SCOPE
    assert row["private_fields_usage"] == TOM1_PRIVATE_FIELDS_USAGE
    assert row["suspected_werewolves"] == {subject: ["player5"]}
    assert row["known_werewolves"] == {subject: hard["known_werewolves"]}
    assert row["known_non_werewolves"] == {
        subject: hard["known_non_werewolves"]
    }
    assert row["observer_label_provenance"] == {
        subject: TOM1_OBSERVER_PROVENANCE
    }
    assert row["observer_annotation_confidence"] == {
        subject: OBSERVER_ANNOTATION_CONFIDENCE
    }
    assert row["source_annotation_record_digests"] == {
        subject: speaker_source["record_digest"]
    }


@pytest.mark.parametrize("status", ["semantic_error", "parse_error", "reporter_error"])
def test_every_non_ok_speaker_drops_complete_tom1_without_recovery(status):
    trajectory, provenance, records = _annotations()
    speaker = next(
        record
        for record in records
        if record["annotation_task"] == PRIVATE_CONDITIONED_SUSPICION_TASK
        and record["step_idx"] == 0
        and record["observer_id"] == 2
    )
    _set_non_ok(speaker, status)
    if status == "semantic_error":
        speaker["source"]["derived_hard_knowledge"] = {
            "known_werewolves": ["player4", "player5"],
            "known_non_werewolves": [
                "player1",
                "player2",
                "player3",
                "player6",
                "player7",
            ],
        }
        _redigest(speaker)
    result = materialize_offline_tom_records(
        trajectory,
        provenance,
        records,
        materializer_code_commit=MATERIALIZER_COMMIT,
    )

    assert [row["step_idx"] for row in result["tom1_records"]] == [1]
    assert result["summary"]["dropped_tom1_boundaries"] == 1


def test_tom2_uses_only_ok_public_non_speakers_and_no_private_semantics():
    trajectory, provenance, records = _annotations()
    public_step_zero = [
        record
        for record in records
        if record["annotation_task"] == PUBLIC_ONLY_SUSPICION_TASK
        and record["step_idx"] == 0
    ]
    by_observer = {record["observer_id"]: record for record in public_step_zero}
    by_observer[1]["result"] = {"suspected_werewolves": ["player5"]}
    by_observer[1]["raw_response"] = (
        '{"suspected_werewolves":["player5"]}'
    )
    _redigest(by_observer[1])
    _set_non_ok(by_observer[3], "parse_error")
    result = materialize_offline_tom_records(
        trajectory,
        provenance,
        records,
        materializer_code_commit=MATERIALIZER_COMMIT,
    )
    row = result["tom2_records"][0]

    assert row["materialization_task"] == OFFLINE_PUBLIC_ONLY_TOM2_TASK
    assert row["tom_order"] == 2
    assert row["model_input_scope"] == TOM2_MODEL_INPUT_SCOPE
    assert row["private_fields_usage"] == TOM2_PRIVATE_FIELDS_USAGE
    assert row["observer_ids"] == [1]
    assert row["suspected_werewolves"] == {"player1": ["player5"]}
    assert row["known_werewolves"] == {"player1": []}
    assert row["known_non_werewolves"] == {"player1": []}
    assert row["observer_label_provenance"] == {
        "player1": TOM2_OBSERVER_PROVENANCE
    }
    assert row["source_annotation_record_digests"] == {
        "player1": by_observer[1]["record_digest"]
    }
    assert "observation_digest" not in canonical_json(row)
    assert "derived_hard_knowledge" not in canonical_json(row)
    assert result["summary"]["filtered_tom2_observers"] == 1


def test_no_eligible_non_speaker_drops_complete_tom2_boundary():
    trajectory, provenance, records = _annotations()
    for record in records:
        if (
            record["annotation_task"] == PUBLIC_ONLY_SUSPICION_TASK
            and record["step_idx"] == 0
            and record["observer_id"] != 2
        ):
            _set_non_ok(record, "reporter_error")
    result = materialize_offline_tom_records(
        trajectory,
        provenance,
        records,
        materializer_code_commit=MATERIALIZER_COMMIT,
    )

    assert [row["step_idx"] for row in result["tom2_records"]] == [1]
    assert result["summary"]["filtered_tom2_observers"] == 2
    assert result["summary"]["dropped_tom2_boundaries"] == 1


@pytest.mark.parametrize(
    ("field_name", "bad_value", "message"),
    [
        ("trajectory_digest", "a" * 64, "trajectory_digest"),
        ("observer_view_artifact_digest", "b" * 64, "artifact digest"),
    ],
)
def test_a_c0_c1_digest_lineage_is_exact(field_name, bad_value, message):
    trajectory, provenance, records = _annotations()
    records[0][field_name] = bad_value
    _redigest(records[0])
    with pytest.raises(ValueError, match=message):
        materialize_offline_tom_records(
            trajectory,
            provenance,
            records,
            materializer_code_commit=MATERIALIZER_COMMIT,
        )


@pytest.mark.parametrize(
    ("source_field", "bad_value"),
    [
        ("public_event_count", 99),
        ("public_event_digest", "a" * 64),
        ("structured_input_digest", "b" * 64),
        ("public_action_count", 99),
    ],
)
def test_c1_public_cutoff_digests_and_action_count_are_exact(
    source_field,
    bad_value,
):
    trajectory, provenance, records = _annotations()
    records[0]["source"][source_field] = bad_value
    _redigest(records[0])
    with pytest.raises(ValueError, match=source_field.replace("_", ".*")):
        materialize_offline_tom_records(
            trajectory,
            provenance,
            records,
            materializer_code_commit=MATERIALIZER_COMMIT,
        )


def test_c1_strict_validator_is_enforced_before_materialization():
    trajectory, provenance, records = _annotations()
    records[0]["unexpected"] = True
    _redigest(records[0])
    with pytest.raises(ValueError, match="fields"):
        materialize_offline_tom_records(
            trajectory,
            provenance,
            records,
            materializer_code_commit=MATERIALIZER_COMMIT,
        )


def test_d_digest_validation_and_redigested_malformed_record_rejection():
    row = _materialize()["tom1_records"][0]
    payload = deepcopy(row)
    digest = payload.pop("record_digest")
    assert digest == canonical_digest(payload)
    assert validate_offline_tom_training_record(row) == row

    malformed = deepcopy(row)
    malformed["pair_targets"] = []
    _redigest(malformed)
    with pytest.raises(ValueError, match="fields"):
        validate_offline_tom_training_record(malformed)


def test_materializer_commit_must_be_lowercase_40_hex():
    trajectory, provenance, records = _annotations()
    with pytest.raises(ValueError, match="materializer_code_commit"):
        materialize_offline_tom_records(
            trajectory,
            provenance,
            records,
            materializer_code_commit="invalid",
        )


def test_output_has_no_pair_targets_or_legacy_raw_provenance():
    result = _materialize()
    serialized = canonical_json(result)
    assert "pair_targets" not in serialized
    assert "original_self_report" not in serialized
    assert "alive_observer_readonly_pre_speech_report_v1" not in serialized
    assert all(
        row["belief_status"]
        == {f"player{observer_id}": "ok" for observer_id in row["observer_ids"]}
        and row["belief_errors"]
        == {f"player{observer_id}": None for observer_id in row["observer_ids"]}
        and row["current_action_used"] is False
        and row["expert_labels_used_as_later_evidence"] is False
        and row["future_information_used"] is False
        for row in result["tom1_records"] + result["tom2_records"]
    )


def test_input_c1_order_does_not_change_records_or_observer_order():
    trajectory, provenance, records = _annotations()
    forward = materialize_offline_tom_records(
        trajectory,
        provenance,
        records,
        materializer_code_commit=MATERIALIZER_COMMIT,
    )
    reverse = materialize_offline_tom_records(
        trajectory,
        provenance,
        list(reversed(records)),
        materializer_code_commit=MATERIALIZER_COMMIT,
    )
    assert reverse == forward


def test_atomic_writer_refuses_overwrite_and_mixed_tasks(tmp_path):
    result = _materialize()
    path = tmp_path / "tom1.jsonl"
    write_offline_tom_jsonl(path, result["tom1_records"])
    persisted = [json.loads(line) for line in path.read_text().splitlines()]
    assert persisted == result["tom1_records"]
    with pytest.raises(FileExistsError):
        write_offline_tom_jsonl(path, result["tom1_records"])
    with pytest.raises(ValueError, match="one materialization_task"):
        write_offline_tom_jsonl(
            tmp_path / "mixed.jsonl",
            [result["tom1_records"][0], result["tom2_records"][0]],
        )
