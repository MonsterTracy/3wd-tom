import json
from copy import deepcopy

import pytest

from werewolf.helper.log_utils import Log
from werewolf.models.twd_tom.public_events import (
    PUBLIC_EVENT_SCHEMA_VERSION,
    public_event_digest,
    structured_input_digest,
)
from werewolf.models.twd_tom.schema import (
    LABEL_PROMPT_VERSION,
    PUBLIC_ONLY_LABEL_PROMPT_VERSION,
)
from werewolf.observer_knowledge import (
    derive_observer_hard_knowledge,
    legal_observer_state,
)
from werewolf.offline_annotation import (
    ANNOTATION_MAX_TOKENS,
    ANNOTATION_TEMPERATURE,
    OFFLINE_ANNOTATION_SCHEMA_VERSION,
    PRIVATE_CONDITIONED_SUSPICION_TASK,
    PRIVATE_INFORMATION_SCOPE,
    PRIVATE_PROMPT_VERSION,
    PUBLIC_INFORMATION_SCOPE,
    PUBLIC_ONLY_SUSPICION_TASK,
    PUBLIC_PROMPT_VERSION,
    annotate_pre_speech_suspicion,
    validate_offline_annotation_record,
    write_annotation_jsonl,
)
from werewolf.trajectory import (
    OBSERVER_VIEW_PROVENANCE_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    POST_PUBLIC_SPEECH,
    PRE_PUBLIC_SPEECH,
    SIMULATOR_BASELINE,
    TRAJECTORY_SCHEMA_VERSION,
    canonical_digest,
)


SOURCE_COMMIT = "bded8b52bbf51d0107e86f8d469eb8a7d621036b"
ANNOTATION_CODE_COMMIT = "d257d827dfd465c57bd33a1a710df9bb857b3280"


class Backend:
    def __init__(self, responses, *, supports_json_schema=True):
        self.responses = list(responses)
        self.supports_json_schema = supports_json_schema
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _serialized_log(*, viewer, event, target, content):
    return {
        "viewer": [viewer],
        "source": viewer,
        "target": target,
        "content": content,
        "day": 1,
        "time": "第1天白天",
        "event": event,
    }


def _observation(observer_id):
    identity = "Seer" if observer_id == 1 else "Witch"
    private_log = (
        _serialized_log(
            viewer=1,
            event="skill_seer",
            target=3,
            content={"cheked_identity": "bad"},
        )
        if observer_id == 1
        else _serialized_log(
            viewer=2,
            event="kill_decision",
            target=4,
            content={"kill_decision": 4},
        )
    )
    return {
        "observer_id": observer_id,
        "current_act_idx": 2,
        "identity": identity,
        "game_log": [
            _serialized_log(
                viewer=observer_id,
                event="game_setting",
                target=0,
                content={"Werewolf": 2},
            ),
            private_log,
        ],
        "phase": "1_day_speech",
        "valid_action": [],
        "authoritative_public_state": {
            "day": 1,
            "day_or_night": "day",
            "phase": "speech",
            "last_night_result": None,
            "prior_exiles": [],
            "alive_players": [1, 2],
            "suggestible_exile_targets": [1],
        },
    }


def _boundary(boundary_type, *, step_idx=0):
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
    views = []
    for observer_id in (1, 2):
        observation = _observation(observer_id)
        views.append(
            {
                "observer_id": observer_id,
                "observation": observation,
                "observation_digest": canonical_digest(observation),
            }
        )
    boundary = {
        "boundary_id": f"game_001:step_{step_idx:06d}:{boundary_type}",
        "boundary_type": boundary_type,
        "step_idx": step_idx,
        "speech_kind": "speech",
        "speaker_id": 2,
        "speech_event_idx": None if boundary_type == PRE_PUBLIC_SPEECH else 2,
        "public_event_count_at_materialization": len(events),
        "public_event_digest_at_materialization": public_event_digest(events),
        "observer_views": views,
    }
    boundary["boundary_digest"] = canonical_digest(boundary)
    return boundary


def _artifacts():
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
        "runtime_config": {"gameplay_max_tokens": 512},
        "runtime_config_digest": canonical_digest(
            {"gameplay_max_tokens": 512}
        ),
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
    provenance = {
        "schema_version": OBSERVER_VIEW_PROVENANCE_SCHEMA_VERSION,
        "game_id": trajectory["game_id"],
        "run_id": trajectory["run_id"],
        "source_commit": trajectory["source_commit"],
        "simulator_baseline": trajectory["simulator_baseline"],
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "trajectory_digest": trajectory["trajectory_digest"],
        "boundaries": [
            _boundary(POST_PUBLIC_SPEECH),
            _boundary(PRE_PUBLIC_SPEECH),
        ],
    }
    provenance["artifact_digest"] = canonical_digest(provenance)
    return trajectory, provenance


def _redigest_provenance(provenance, *, boundaries=False):
    if boundaries:
        for boundary in provenance["boundaries"]:
            boundary.pop("boundary_digest", None)
            boundary["boundary_digest"] = canonical_digest(boundary)
    provenance.pop("artifact_digest", None)
    provenance["artifact_digest"] = canonical_digest(provenance)


def _redigest_record(record):
    record.pop("record_digest", None)
    record["record_digest"] = canonical_digest(record)


def _annotate(task, backend, *, artifacts=None):
    trajectory, provenance = artifacts or _artifacts()
    return annotate_pre_speech_suspicion(
        trajectory,
        provenance,
        annotation_task=task,
        annotation_run_id="annotation_run_001",
        annotation_code_commit=ANNOTATION_CODE_COMMIT,
        backend=backend,
        backend_id="reporter_backend",
        model_name="reporter_model",
    )


def test_private_annotations_are_pre_only_ordered_and_lineage_closed():
    backend = Backend(
        [
            '{"suspected_werewolves":["player3"]}',
            '{"suspected_werewolves":[]}',
        ]
    )
    trajectory, provenance = _artifacts()
    records = _annotate(
        PRIVATE_CONDITIONED_SUSPICION_TASK,
        backend,
        artifacts=(trajectory, provenance),
    )

    assert [(record["step_idx"], record["observer_id"]) for record in records] == [
        (0, 1),
        (0, 2),
    ]
    assert {record["boundary_type"] for record in records} == {
        PRE_PUBLIC_SPEECH
    }
    first = records[0]
    assert set(first) == {
        "schema_version",
        "annotation_task",
        "annotation_run_id",
        "annotation_code_commit",
        "game_id",
        "source_trajectory_commit",
        "trajectory_digest",
        "observer_view_artifact_digest",
        "boundary_id",
        "boundary_type",
        "step_idx",
        "observer_id",
        "information_scope",
        "source",
        "prompt_version",
        "prompt",
        "prompt_digest",
        "reporter_backend_id",
        "reporter_model_id",
        "request_parameters",
        "raw_response",
        "status",
        "error",
        "result",
        "record_digest",
    }
    assert first["schema_version"] == OFFLINE_ANNOTATION_SCHEMA_VERSION
    assert first["annotation_task"] == PRIVATE_CONDITIONED_SUSPICION_TASK
    assert first["information_scope"] == PRIVATE_INFORMATION_SCOPE
    assert first["prompt_version"] == PRIVATE_PROMPT_VERSION
    assert first["source_trajectory_commit"] == trajectory["source_commit"]
    assert first["trajectory_digest"] == trajectory["trajectory_digest"]
    assert first["observer_view_artifact_digest"] == provenance[
        "artifact_digest"
    ]
    assert set(first["source"]) == {
        "observation_digest",
        "public_event_count",
        "public_event_digest",
        "structured_input_digest",
        "public_action_count",
        "derived_hard_knowledge",
    }
    assert first["source"]["derived_hard_knowledge"] == {
        "known_werewolves": ["player3"],
        "known_non_werewolves": ["player1"],
    }
    assert first["result"] == {"suspected_werewolves": ["player3"]}
    assert "derived_hard_knowledge" not in first["result"]
    assert first["raw_response"] == (
        '{"suspected_werewolves":["player3"]}'
    )
    assert "role" not in first


def test_multiple_pre_boundaries_sort_by_step_then_observer():
    trajectory, provenance = _artifacts()
    trajectory["transitions"] = [
        {
            "step_idx": 0,
            "acting_player_id": 2,
            "public_event_count_before": 2,
            "public_events_appended": [],
        }
    ]
    trajectory.pop("trajectory_digest")
    trajectory["trajectory_digest"] = canonical_digest(trajectory)

    first_pre = next(
        boundary
        for boundary in provenance["boundaries"]
        if boundary["boundary_type"] == PRE_PUBLIC_SPEECH
    )
    second_pre = deepcopy(first_pre)
    second_pre["step_idx"] = 1
    second_pre["boundary_id"] = "game_001:step_000001:PRE_PUBLIC_SPEECH"
    second_pre.pop("boundary_digest")
    second_pre["boundary_digest"] = canonical_digest(second_pre)
    provenance["boundaries"] = [second_pre, *provenance["boundaries"]]
    provenance["trajectory_digest"] = trajectory["trajectory_digest"]
    _redigest_provenance(provenance)

    records = _annotate(
        PUBLIC_ONLY_SUSPICION_TASK,
        Backend(['{"suspected_werewolves":[]}'] * 4),
        artifacts=(trajectory, provenance),
    )
    assert [(record["step_idx"], record["observer_id"]) for record in records] == [
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
    ]


def test_private_prompt_and_request_are_exact_and_recomputable():
    raw = '{"suspected_werewolves":["player3"]}'
    backend = Backend([raw, '{"suspected_werewolves":[]}'])
    records = _annotate(PRIVATE_CONDITIONED_SUSPICION_TASK, backend)
    record = records[0]
    call = backend.calls[0]

    assert record["raw_response"] == raw
    assert record["status"] == "ok"
    assert record["error"] is None
    assert call["messages"] == [{"role": "user", "content": record["prompt"]}]
    assert "stateless offline observer-conditioned annotation" in record["prompt"]
    assert "NOT the historical gameplay agent self-report" in record["prompt"]
    assert "has not yet produced this speech" in record["prompt"]
    assert "MUST INCLUDE known_werewolves" in record["prompt"]
    assert "MUST EXCLUDE known_non_werewolves" in record["prompt"]
    assert "Size 0..7 is legal" in record["prompt"]
    assert "do not force exactly two" in record["prompt"]
    assert "still possible is not automatically" in record["prompt"]
    assert "god view" in record["prompt"]
    assert "chain-of-thought" in record["prompt"]
    assert record["prompt_digest"] == canonical_digest(record["prompt"])
    without_digest = dict(record)
    without_digest.pop("record_digest")
    assert record["record_digest"] == canonical_digest(without_digest)

    assert record["request_parameters"] == {
        "temperature": ANNOTATION_TEMPERATURE,
        "max_tokens": ANNOTATION_MAX_TOKENS,
        "response_format": call["response_format"],
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    assert call["temperature"] == 0.0
    assert call["max_tokens"] == 96
    assert call["response_format"]["type"] == "json_schema"
    assert call["response_format"]["json_schema"]["name"] == (
        "offline_suspicion_annotation_v1"
    )
    assert call["extra_body"] == {"thinking": {"type": "disabled"}}


def test_public_only_has_exact_public_source_and_no_private_prompt_material():
    backend = Backend(
        [
            '{"suspected_werewolves":["player5"]}',
            '{"suspected_werewolves":[]}',
        ],
        supports_json_schema=False,
    )
    trajectory, _ = _artifacts()
    records = _annotate(PUBLIC_ONLY_SUSPICION_TASK, backend)
    record = records[0]
    events = trajectory["initial_public_events"]

    assert record["information_scope"] == PUBLIC_INFORMATION_SCOPE
    assert record["prompt_version"] == PUBLIC_PROMPT_VERSION
    assert record["source"] == {
        "public_event_count": 2,
        "public_event_digest": public_event_digest(events),
        "structured_input_digest": structured_input_digest(events),
        "public_action_count": 0,
    }
    assert "observation" not in record["source"]
    assert "derived_hard_knowledge" not in record["source"]
    assert "legal_pre_speech_observer_state" not in record["prompt"]
    assert "only the reporting perspective identifier" in record["prompt"]
    assert "private state" in record["prompt"]
    assert "future information" in record["prompt"]
    assert backend.calls[0]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    ("response", "expected_status"),
    [
        ("not-json", "parse_error"),
        ('{"suspected_werewolves":["player1"]}', "semantic_error"),
    ],
)
def test_parse_and_semantic_errors_preserve_raw_response(
    response,
    expected_status,
):
    backend = Backend([response, response])
    records = _annotate(PRIVATE_CONDITIONED_SUSPICION_TASK, backend)
    assert records[0]["status"] == expected_status
    assert records[0]["raw_response"] == response
    assert records[0]["error"]
    assert records[0]["result"] is None


def test_reporter_error_is_safe_and_has_null_result():
    backend = Backend(
        [
            RuntimeError("api_key=DO-NOT-SAVE"),
            RuntimeError("token=DO-NOT-SAVE"),
        ]
    )
    records = _annotate(PUBLIC_ONLY_SUSPICION_TASK, backend)
    assert all(record["status"] == "reporter_error" for record in records)
    assert all(record["raw_response"] is None for record in records)
    assert all(record["result"] is None for record in records)
    assert all("DO-NOT-SAVE" not in record["error"] for record in records)


def test_frozen_digest_linkage_and_nested_digests_are_required():
    trajectory, provenance = _artifacts()
    trajectory["environment_seed"] = 999
    with pytest.raises(ValueError, match="trajectory_digest"):
        _annotate(PUBLIC_ONLY_SUSPICION_TASK, Backend([]), artifacts=(trajectory, provenance))

    trajectory, provenance = _artifacts()
    provenance["trajectory_digest"] = "0" * 64
    _redigest_provenance(provenance)
    with pytest.raises(ValueError, match="trajectory_digest mismatch"):
        _annotate(PUBLIC_ONLY_SUSPICION_TASK, Backend([]), artifacts=(trajectory, provenance))

    trajectory, provenance = _artifacts()
    provenance["game_id"] = "tampered"
    with pytest.raises(ValueError, match="game_id mismatch"):
        _annotate(PUBLIC_ONLY_SUSPICION_TASK, Backend([]), artifacts=(trajectory, provenance))

    trajectory, provenance = _artifacts()
    provenance["tampered"] = True
    with pytest.raises(ValueError, match="artifact_digest"):
        _annotate(PUBLIC_ONLY_SUSPICION_TASK, Backend([]), artifacts=(trajectory, provenance))

    trajectory, provenance = _artifacts()
    provenance["boundaries"][0]["speaker_id"] = 7
    _redigest_provenance(provenance)
    with pytest.raises(ValueError, match="boundary_digest"):
        _annotate(PUBLIC_ONLY_SUSPICION_TASK, Backend([]), artifacts=(trajectory, provenance))


def test_private_observation_digest_is_independently_validated():
    trajectory, provenance = _artifacts()
    pre = next(
        boundary
        for boundary in provenance["boundaries"]
        if boundary["boundary_type"] == PRE_PUBLIC_SPEECH
    )
    pre["observer_views"][0]["observation"]["identity"] = "Villager"
    _redigest_provenance(provenance, boundaries=True)

    with pytest.raises(ValueError, match="observation_digest"):
        _annotate(
            PRIVATE_CONDITIONED_SUSPICION_TASK,
            Backend([]),
            artifacts=(trajectory, provenance),
        )
    public_records = _annotate(
        PUBLIC_ONLY_SUSPICION_TASK,
        Backend([
            '{"suspected_werewolves":[]}',
            '{"suspected_werewolves":[]}',
        ]),
        artifacts=(trajectory, provenance),
    )
    assert len(public_records) == 2


def test_serialized_and_legacy_logs_derive_identical_hard_knowledge():
    serialized = _observation(1)
    legacy = deepcopy(serialized)
    legacy["game_log"] = [
        Log(**entry)
        for entry in serialized["game_log"]
    ]
    assert legal_observer_state(1, serialized) == legal_observer_state(
        1,
        legacy,
    )
    assert derive_observer_hard_knowledge(
        1,
        serialized,
    ) == derive_observer_hard_knowledge(1, legacy)


@pytest.mark.parametrize(
    "task",
    [PRIVATE_CONDITIONED_SUSPICION_TASK, PUBLIC_ONLY_SUSPICION_TASK],
)
def test_omniscient_role_metadata_cannot_change_prompt(task):
    trajectory, provenance = _artifacts()
    responses = [
        '{"suspected_werewolves":["player3"]}',
        '{"suspected_werewolves":[]}',
    ] if task == PRIVATE_CONDITIONED_SUSPICION_TASK else [
        '{"suspected_werewolves":[]}',
        '{"suspected_werewolves":[]}',
    ]
    baseline_backend = Backend(responses)
    _annotate(task, baseline_backend, artifacts=(trajectory, provenance))

    mutated_trajectory = deepcopy(trajectory)
    mutated_provenance = deepcopy(provenance)
    for player in mutated_trajectory["players"]:
        player["role"] = "TRUTH-MUTATED"
    mutated_trajectory.pop("trajectory_digest")
    mutated_trajectory["trajectory_digest"] = canonical_digest(
        mutated_trajectory
    )
    mutated_provenance["trajectory_digest"] = mutated_trajectory[
        "trajectory_digest"
    ]
    _redigest_provenance(mutated_provenance)
    mutated_backend = Backend(responses)
    _annotate(
        task,
        mutated_backend,
        artifacts=(mutated_trajectory, mutated_provenance),
    )
    assert [call["messages"] for call in baseline_backend.calls] == [
        call["messages"] for call in mutated_backend.calls
    ]


def test_offline_identities_do_not_impersonate_legacy_self_reports():
    assert OFFLINE_ANNOTATION_SCHEMA_VERSION not in {
        TRAJECTORY_SCHEMA_VERSION,
        OBSERVER_VIEW_PROVENANCE_SCHEMA_VERSION,
    }
    assert PRIVATE_PROMPT_VERSION != LABEL_PROMPT_VERSION
    assert PUBLIC_PROMPT_VERSION != PUBLIC_ONLY_LABEL_PROMPT_VERSION
    assert PRIVATE_CONDITIONED_SUSPICION_TASK != LABEL_PROMPT_VERSION
    assert PUBLIC_ONLY_SUSPICION_TASK != PUBLIC_ONLY_LABEL_PROMPT_VERSION


def test_atomic_jsonl_writer_preserves_order_and_refuses_overwrite(tmp_path):
    records = _annotate(
        PUBLIC_ONLY_SUSPICION_TASK,
        Backend([
            '{"suspected_werewolves":[]}',
            '{"suspected_werewolves":[]}',
        ]),
    )
    path = tmp_path / "annotations.jsonl"
    write_annotation_jsonl(path, records)
    saved = [json.loads(line) for line in path.read_text().splitlines()]
    assert saved == records
    with pytest.raises(FileExistsError, match="already exists"):
        write_annotation_jsonl(path, records)

    mixed = deepcopy(records)
    mixed[1]["annotation_task"] = PRIVATE_CONDITIONED_SUSPICION_TASK
    mixed[1].pop("record_digest")
    mixed[1]["record_digest"] = canonical_digest(mixed[1])
    with pytest.raises(ValueError, match="one annotation_task"):
        write_annotation_jsonl(tmp_path / "mixed.jsonl", mixed)


def test_strict_validator_rejects_redigested_malformed_records(tmp_path):
    records = _annotate(
        PUBLIC_ONLY_SUSPICION_TASK,
        Backend([
            '{"suspected_werewolves":[]}',
            '{"suspected_werewolves":[]}',
        ]),
    )
    baseline = records[0]
    malformed = []

    missing = deepcopy(baseline)
    missing.pop("prompt_version")
    malformed.append(missing)

    extra = deepcopy(baseline)
    extra["unexpected"] = True
    malformed.append(extra)

    wrong_scope = deepcopy(baseline)
    wrong_scope["information_scope"] = PRIVATE_INFORMATION_SCOPE
    malformed.append(wrong_scope)

    wrong_prompt_digest = deepcopy(baseline)
    wrong_prompt_digest["prompt_digest"] = "0" * 64
    malformed.append(wrong_prompt_digest)

    wrong_request = deepcopy(baseline)
    wrong_request["request_parameters"]["max_tokens"] = 97
    malformed.append(wrong_request)

    status_mismatch = deepcopy(baseline)
    status_mismatch["status"] = "parse_error"
    status_mismatch["error"] = "invalid"
    malformed.append(status_mismatch)

    private_injection = deepcopy(baseline)
    private_injection["source"]["observation_digest"] = "0" * 64
    malformed.append(private_injection)

    invalid_commit = deepcopy(baseline)
    invalid_commit["annotation_code_commit"] = "not-a-git-sha"
    malformed.append(invalid_commit)

    for index, record in enumerate(malformed):
        with pytest.raises((TypeError, ValueError)):
            _redigest_record(record)
            validate_offline_annotation_record(record)
        with pytest.raises((TypeError, ValueError)):
            write_annotation_jsonl(tmp_path / f"malformed_{index}.jsonl", [record])


def test_private_ok_result_must_satisfy_derived_hard_knowledge():
    record = _annotate(
        PRIVATE_CONDITIONED_SUSPICION_TASK,
        Backend([
            '{"suspected_werewolves":["player3"]}',
            '{"suspected_werewolves":[]}',
        ]),
    )[0]
    record["result"] = {"suspected_werewolves": []}
    record["raw_response"] = '{"suspected_werewolves":[]}'
    _redigest_record(record)
    with pytest.raises(ValueError, match="known_werewolves"):
        validate_offline_annotation_record(record)


def test_annotation_code_commit_must_be_lowercase_40_hex():
    trajectory, provenance = _artifacts()
    with pytest.raises(ValueError, match="annotation_code_commit"):
        annotate_pre_speech_suspicion(
            trajectory,
            provenance,
            annotation_task=PUBLIC_ONLY_SUSPICION_TASK,
            annotation_run_id="annotation_run_001",
            annotation_code_commit="invalid",
            backend=Backend([]),
            backend_id="reporter_backend",
            model_name="reporter_model",
        )


def test_pre_boundary_id_must_be_canonical():
    trajectory, provenance = _artifacts()
    pre = next(
        boundary
        for boundary in provenance["boundaries"]
        if boundary["boundary_type"] == PRE_PUBLIC_SPEECH
    )
    pre["boundary_id"] = "wrong"
    _redigest_provenance(provenance, boundaries=True)
    with pytest.raises(ValueError, match="boundary_id"):
        _annotate(
            PUBLIC_ONLY_SUSPICION_TASK,
            Backend([]),
            artifacts=(trajectory, provenance),
        )


def test_pre_speaker_must_match_final_turn_start():
    trajectory, provenance = _artifacts()
    pre = next(
        boundary
        for boundary in provenance["boundaries"]
        if boundary["boundary_type"] == PRE_PUBLIC_SPEECH
    )
    pre["speaker_id"] = 3
    _redigest_provenance(provenance, boundaries=True)
    with pytest.raises(ValueError, match="turn_start"):
        _annotate(
            PUBLIC_ONLY_SUSPICION_TASK,
            Backend([]),
            artifacts=(trajectory, provenance),
        )


def test_committed_transition_actor_must_match_pre_speaker():
    trajectory, provenance = _artifacts()
    trajectory["transitions"] = [
        {
            "step_idx": 0,
            "acting_player_id": 1,
            "public_event_count_before": 2,
            "public_events_appended": [],
        }
    ]
    trajectory.pop("trajectory_digest")
    trajectory["trajectory_digest"] = canonical_digest(trajectory)
    provenance["trajectory_digest"] = trajectory["trajectory_digest"]
    _redigest_provenance(provenance)
    with pytest.raises(ValueError, match="transition actor"):
        _annotate(
            PUBLIC_ONLY_SUSPICION_TASK,
            Backend([]),
            artifacts=(trajectory, provenance),
        )


def test_private_observation_actor_must_match_pre_speaker_but_public_ignores_it():
    trajectory, provenance = _artifacts()
    pre = next(
        boundary
        for boundary in provenance["boundaries"]
        if boundary["boundary_type"] == PRE_PUBLIC_SPEECH
    )
    view = pre["observer_views"][0]
    view["observation"]["current_act_idx"] = 7
    view["observation_digest"] = canonical_digest(view["observation"])
    _redigest_provenance(provenance, boundaries=True)

    with pytest.raises(ValueError, match="current actor"):
        _annotate(
            PRIVATE_CONDITIONED_SUSPICION_TASK,
            Backend([]),
            artifacts=(trajectory, provenance),
        )
    records = _annotate(
        PUBLIC_ONLY_SUSPICION_TASK,
        Backend([
            '{"suspected_werewolves":[]}',
            '{"suspected_werewolves":[]}',
        ]),
        artifacts=(trajectory, provenance),
    )
    assert len(records) == 2
