import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import script.twd_tom.collect_canonical_trajectories as batch_module
from werewolf.models.twd_tom.public_events import (
    public_event_digest,
    structured_input_digest,
)
from werewolf.models.twd_tom.samples import SAMPLE_SCHEMA_VERSION
from werewolf.models.twd_tom.schema import LABEL_PROMPT_VERSION, LABEL_PROVENANCE
from werewolf.models.twd_tom.speech_annotations import (
    SPEECH_ACTION_ONTOLOGY_VERSION,
    SPEECH_ANNOTATION_SCHEMA_VERSION,
    make_speech_annotation,
    speech_annotation_digest,
)
from werewolf.trajectory import canonical_digest, canonical_json


COMMIT = "90fd287e7ca50a0aabcadea8c0388fd3a4ba37f7"
ROLES = [
    "Werewolf",
    "Villager",
    "Seer",
    "Witch",
    "Villager",
    "Villager",
    "Werewolf",
]
PROFILES = [f"profile-{index}" for index in range(1, 8)]


def runtime_config():
    return {
        "backends": {
            "parser-api": {
                "type": "openai_compatible",
            },
            "agent-api": {
                "type": "openai_compatible",
            },
        },
        "parser": {
            "backend": "parser-api",
            "model": "parser-model",
        },
        "env_config": {
            "n_player": 7,
            "n_role": 4,
            "n_werewolf": 2,
            "n_seer": 1,
            "n_guard": 0,
            "n_witch": 1,
            "n_hunter": 0,
            "n_villager": 3,
        },
        "agent_config": {
            "allow_cross_team_profiles": True,
            "all_candidates": [
                {
                    "profile_name": "profile",
                    "agent_type": "gpt",
                    "backend": "agent-api",
                    "model": "agent-model",
                    "model_params": {"temperature": 0.0},
                    "sample_ratio": 1.0,
                }
            ],
        },
        "pipeline": {
            "public_event_schema_version": batch_module.PUBLIC_EVENT_SCHEMA_VERSION,
            "speech_annotation_schema_version": SPEECH_ANNOTATION_SCHEMA_VERSION,
            "speech_action_ontology_version": SPEECH_ACTION_ONTOLOGY_VERSION,
            "speech_parser_prompt_version": batch_module.SPEECH_PARSER_PROMPT_VERSION,
            "public_speech_realization_prompt_version": (
                batch_module.PUBLIC_SPEECH_REALIZATION_PROMPT_VERSION
            ),
            "raw_schema_version": batch_module.SAMPLE_SCHEMA_VERSION,
            "projected_schema_version": batch_module.PROJECTED_SCHEMA_VERSION,
            "projection_version": batch_module.TARGET_CONVERSION,
            "collection": {
                "game_count": 3,
                "target_game_count": 3,
                "seeds": [1001, 1002, 1003],
                "max_gameplay_calls_per_game": 10,
                "max_belief_calls_per_game": 10,
                "max_total_calls_per_game": 20,
                "max_wall_seconds_per_game": 60.0,
            },
        },
    }


def _write_config(tmp_path, config=None):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(config or runtime_config(), sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_canonical_50_server_config_freezes_collection_and_split():
    project_root = Path(__file__).resolve().parents[2]
    config_path = (
        project_root
        / "configs"
        / "twd_tom_server_qwen35_9b_canonical_50.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    pilot_config = yaml.safe_load(
        (project_root / "configs" / "twd_tom_server_qwen35_9b.yaml").read_text(
            encoding="utf-8"
        )
    )
    seeds = list(range(4201, 4261))

    contract = batch_module._pipeline_collection_contract(
        config,
        seeds=seeds,
    )

    assert contract["game_count"] == 60
    assert contract["target_game_count"] == 50
    assert contract["seeds"] == seeds
    assert config["pipeline"]["split"] == {
        "seed": 42,
        "train_game_count": 40,
        "validation_game_count": 5,
        "test_game_count": 5,
    }
    canonical_runtime = deepcopy(config)
    pilot_runtime = deepcopy(pilot_config)
    for runtime in (canonical_runtime, pilot_runtime):
        runtime["pipeline"].pop("collection")
        runtime["pipeline"].pop("split")
    assert canonical_runtime == pilot_runtime
    assert config["parser"]["model_params"]["temperature"] == 0.0
    candidate = config["agent_config"]["all_candidates"][0]
    assert candidate["model_params"]["temperature"] == 1.0
    assert candidate["backend"] == config["parser"]["backend"]


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")


def _complete_artifacts(recorder, *, winner="Villager"):
    base = deepcopy(recorder._base)
    speaker_id = 2
    initial_events = [
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
    speech = {
        "event_idx": 2,
        "event_type": "public_speech",
        "speaker": "player2",
        "raw_text": "synthetic speech",
    }
    all_events = initial_events + [speech]
    if winner == "Villager":
        final_alive = [2, 3, 4, 5, 6]
    else:
        final_alive = [1, 3, 4, 7]

    delivered_observation = {
        "observer_id": speaker_id,
        "current_act_idx": speaker_id,
        "phase": "1_day_speech",
        "valid_action": [],
    }
    transition = {
        "step_idx": 0,
        "phase_before": "1_day_speech",
        "acting_player_id": speaker_id,
        "delivered_observation": delivered_observation,
        "delivered_observation_digest": canonical_digest(delivered_observation),
        "submitted_action": [
            "speech",
            "synthetic speech",
        ],
        "public_event_count_before": len(initial_events),
        "public_events_appended": [speech],
        "phase_after": "1_day_vote",
        "alive_players_after": final_alive,
        "terminal_after": True,
    }
    trajectory = {
        **base,
        "initial_public_events": initial_events,
        "transitions": [transition],
        "termination": {
            "completion_status": "COMPLETE",
            "termination_kind": "normal_game_end",
            "winner": winner,
            "final_alive_players": final_alive,
        },
        "public_event_digest": public_event_digest(all_events),
    }
    trajectory["trajectory_digest"] = canonical_digest(trajectory)

    def observation(observer_id, *, current_actor):
        return {
            "observer_id": observer_id,
            "current_act_idx": current_actor,
            "phase": "1_day_speech",
            "valid_action": [],
        }

    pre_views = []
    for observer_id in range(1, 8):
        obs = observation(observer_id, current_actor=speaker_id)
        pre_views.append(
            {
                "observer_id": observer_id,
                "observation": obs,
                "observation_digest": canonical_digest(obs),
            }
        )
    post_views = []
    for observer_id in final_alive:
        obs = observation(observer_id, current_actor=speaker_id)
        post_views.append(
            {
                "observer_id": observer_id,
                "observation": obs,
                "observation_digest": canonical_digest(obs),
            }
        )

    pre = {
        "boundary_id": f"{base['game_id']}:step_000000:PRE_PUBLIC_SPEECH",
        "boundary_type": "PRE_PUBLIC_SPEECH",
        "step_idx": 0,
        "speech_kind": "speech",
        "speaker_id": speaker_id,
        "speech_event_idx": None,
        "public_event_count_at_materialization": 2,
        "public_event_digest_at_materialization": public_event_digest(initial_events),
        "observer_views": pre_views,
    }
    pre["boundary_digest"] = canonical_digest(pre)
    post = {
        "boundary_id": f"{base['game_id']}:step_000000:POST_PUBLIC_SPEECH",
        "boundary_type": "POST_PUBLIC_SPEECH",
        "step_idx": 0,
        "speech_kind": "speech",
        "speaker_id": speaker_id,
        "speech_event_idx": 2,
        "public_event_count_at_materialization": 3,
        "public_event_digest_at_materialization": public_event_digest(all_events),
        "observer_views": post_views,
    }
    post["boundary_digest"] = canonical_digest(post)
    provenance = {
        "schema_version": batch_module.OBSERVER_VIEW_PROVENANCE_SCHEMA_VERSION,
        "game_id": base["game_id"],
        "run_id": base["run_id"],
        "source_commit": base["source_commit"],
        "simulator_baseline": base["simulator_baseline"],
        "observation_schema_version": batch_module.OBSERVATION_SCHEMA_VERSION,
        "trajectory_digest": trajectory["trajectory_digest"],
        "boundaries": [pre, post],
    }
    provenance["artifact_digest"] = canonical_digest(provenance)

    _write_json(recorder.trajectory_output_path, trajectory)
    _write_json(recorder.observer_view_output_path, provenance)


def _belief_snapshot(game_id):
    public_events = [
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
    subjects = [f"player{player_id}" for player_id in range(1, 8)]
    speech_annotations = []
    return {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "game_id": game_id,
        "step_idx": 0,
        "phase": "1_day_speech",
        "speaker_id": 2,
        "report_trigger": "pre_public_speech",
        "public_event_schema_version": batch_module.PUBLIC_EVENT_SCHEMA_VERSION,
        "public_events": public_events,
        "public_event_digest": public_event_digest(public_events),
        "speech_annotation_schema_version": SPEECH_ANNOTATION_SCHEMA_VERSION,
        "speech_action_ontology_version": SPEECH_ACTION_ONTOLOGY_VERSION,
        "speech_annotations": speech_annotations,
        "speech_annotation_digest": speech_annotation_digest(speech_annotations),
        "structured_input_digest": structured_input_digest(
            public_events, speech_annotations
        ),
        "observer_ids": list(range(1, 8)),
        "suspected_werewolves": {subject: [] for subject in subjects},
        "known_werewolves": {subject: [] for subject in subjects},
        "known_non_werewolves": {subject: [] for subject in subjects},
        "belief_status": {subject: "ok" for subject in subjects},
        "belief_errors": {subject: None for subject in subjects},
        "label_cutoff_step_idx": 0,
        "public_action_count": 0,
        "label_prompt_version": LABEL_PROMPT_VERSION,
        "label_provenance": LABEL_PROVENANCE,
        "agent_backend_ids": {
            subject: f"backend-{player_id}"
            for player_id, subject in enumerate(subjects, start=1)
        },
    }


def _install_fake_runtime(
    monkeypatch,
    output_root,
    *,
    fail_seed=None,
    interrupt_seed=None,
    speech_error=False,
):
    calls = {
        "backend": [],
        "build": [],
        "game": [],
        "allow_gameplay_fallback": [],
    }

    def fake_provenance(_repo_root):
        return {"batch_code_commit": COMMIT, "git_worktree_clean": True}

    def fake_load_backends(config, env_file, *, max_retries):
        assert (output_root / "plan.json").is_file()
        calls["backend"].append(
            {
                "config": deepcopy(config),
                "env_file": Path(env_file),
                "max_retries": max_retries,
            }
        )
        return {"fake": object()}

    def fake_build_runtime(parsed_yaml, log_save_path, random_seed, backends):
        calls["build"].append(
            {
                "seed": random_seed,
                "log_save_path": Path(log_save_path),
                "backends": backends,
            }
        )
        agents = [
            SimpleNamespace(
                backend_id=f"backend-{index}",
                model_name=f"model-{index}",
            )
            for index in range(1, 8)
        ]
        env = SimpleNamespace(public_events=[], speech_annotations=[])
        return env, agents, list(ROLES), list(PROFILES)

    def fake_run_game(
        env,
        agents,
        roles,
        *,
        sample_collector,
        call_audit,
        trajectory_recorder,
        allow_gameplay_fallback,
    ):
        assert call_audit.game_id == trajectory_recorder._base["game_id"]
        seed = trajectory_recorder._base["environment_seed"]
        calls["game"].append(seed)
        calls["allow_gameplay_fallback"].append(allow_gameplay_fallback)
        if seed == interrupt_seed:
            raise KeyboardInterrupt("synthetic interrupted process")
        if seed == fail_seed:
            raise RuntimeError("synthetic gameplay failure")
        winner = "Villager" if seed % 2 else "Werewolf"
        sample_collector.write(_belief_snapshot(trajectory_recorder._base["game_id"]))
        _complete_artifacts(trajectory_recorder, winner=winner)
        trajectory = json.loads(
            trajectory_recorder.trajectory_output_path.read_text(encoding="utf-8")
        )
        env.public_events = list(trajectory["initial_public_events"])
        for transition in trajectory["transitions"]:
            env.public_events.extend(transition["public_events_appended"])
        speech_event = next(
            event
            for event in env.public_events
            if event["event_type"] == "public_speech"
        )
        if speech_error:
            raw_response = "player2 | support | NONE"
            error_type = "SpeechActionValidationError"
            error_message = "support requires a canonical player object"
            generation_attempts = [
                {
                    "generation_attempt": attempt,
                    "status": "parser_error",
                    "raw_response": raw_response,
                    "error_type": error_type,
                    "error_message": error_message,
                }
                for attempt in range(1, 4)
            ]
            status = "error"
        else:
            raw_response = "NONE"
            error_type = None
            error_message = None
            generation_attempts = [
                {
                    "generation_attempt": 1,
                    "status": "ok",
                    "raw_response": raw_response,
                    "error_type": None,
                    "error_message": None,
                }
            ]
            status = "no_action"
        env.speech_annotations = [
            make_speech_annotation(
                event_idx=speech_event["event_idx"],
                speaker=speech_event["speaker"],
                raw_text=speech_event["raw_text"],
                parser_model_id="synthetic-parser",
                parser_call_id="synthetic-parser-call-000001",
                annotation_source="llm_parser",
                status=status,
                actions=[],
                generation_attempts=generation_attempts,
                raw_response=raw_response,
                error_type=error_type,
                error_message=error_message,
            )
        ]
        return f"{winner} win"

    monkeypatch.setattr(batch_module, "_read_code_provenance", fake_provenance)
    monkeypatch.setattr(batch_module, "load_named_backends", fake_load_backends)
    monkeypatch.setattr(batch_module, "build_runtime", fake_build_runtime)
    monkeypatch.setattr(batch_module, "run_game", fake_run_game)
    monkeypatch.setattr(
        batch_module,
        "replay_canonical_trajectory",
        lambda trajectory_path, observer_views_path: {
            "status": "MATCH",
            "trajectory_path": str(trajectory_path),
            "observer_views_path": str(observer_views_path),
        },
    )
    return calls


def test_successful_batch_freezes_plan_before_first_game_and_preserves_seed_contract(
    tmp_path, monkeypatch
):
    config_path = _write_config(tmp_path)
    output_root = tmp_path / "canonical-pilot-001"
    calls = _install_fake_runtime(monkeypatch, output_root)

    summary = batch_module.collect_canonical_trajectory_batch(
        config_path=config_path,
        run_id="canonical-pilot-001",
        seed_start=1001,
        game_count=3,
        output_root=output_root,
        repo_root=tmp_path,
    )

    assert calls["game"] == [1001, 1002, 1003]
    assert calls["allow_gameplay_fallback"] == [False, False, False]
    assert [call["seed"] for call in calls["build"]] == [1001, 1002, 1003]
    assert calls["backend"][0]["max_retries"] == 0
    assert calls["backend"][0]["env_file"] == tmp_path / ".env"

    plan = json.loads((output_root / "plan.json").read_text())
    assert plan["schema_version"] == batch_module.BATCH_PLAN_SCHEMA_VERSION
    assert plan["batch_code_commit"] == COMMIT
    assert plan["seeds"] == [1001, 1002, 1003]
    assert plan["planned_game_count"] == 3
    assert plan["target_game_count"] == 3
    assert plan["collection_contract"]["seeds"] == [1001, 1002, 1003]
    assert plan["collectors_enabled"] is True
    assert plan["collection_mode"] == "canonical"
    assert plan["canonical_eligible"] is True
    assert plan["backend_max_attempts"] == 3
    assert plan["backend_sdk_max_retries"] == 0
    assert plan["gameplay_generation_max_attempts"] == 3
    assert plan["speech_parser_generation_max_attempts"] == 3
    assert plan["speech_parser_retry_policy"] == (
        "full_response_strict_validation_feedback_v1"
    )
    assert plan["speech_parser_failure_policy"] == (
        "canonical_fail_closed_pilot_record_error"
    )
    assert plan["label_generation_max_attempts"] == 3
    assert plan["stop_on_first_failure"] is False
    assert plan["continue_on_game_failure"] is True
    assert plan["resume_supported"] is True
    assert plan["resume_requires_exact_plan"] is True
    assert plan["rerun_on_failure"] is False
    assert plan["replacement_seed_on_failure"] is False
    assert plan["deterministic_replay_required"] is True
    assert plan["canonical_replay_validator"].endswith(
        ".replay_canonical_trajectory"
    )
    payload = deepcopy(plan)
    digest = payload.pop("plan_digest")
    assert digest == canonical_digest(payload)

    assert summary["schema_version"] == batch_module.BATCH_SUMMARY_SCHEMA_VERSION
    assert summary["completed_game_count"] == 3
    assert summary["target_game_count"] == 3
    assert summary["attempted_game_count"] == 3
    assert summary["failed_game_count"] == 0
    assert summary["unattempted_seeds"] == []
    assert summary["collection_mode"] == "canonical"
    assert summary["canonical_eligible"] is True
    assert summary["total_gameplay_fallback_count"] == 0
    assert summary["total_missing_pre_belief_snapshot_count"] == 0
    assert summary["total_label_snapshot_failure_count"] == 0
    assert summary["total_speech_annotation_error_count"] == 0
    assert summary["total_speech_parser_generation_attempt_count"] == 3
    assert summary["total_belief_snapshot_count"] == 3
    assert summary["deterministic_replay_match_count"] == 3
    assert summary["game_ids"] == [
        "canonical-pilot-001_game_0001_seed_1001",
        "canonical-pilot-001_game_0002_seed_1002",
        "canonical-pilot-001_game_0003_seed_1003",
    ]
    payload = deepcopy(summary)
    digest = payload.pop("summary_digest")
    assert digest == canonical_digest(payload)
    assert not (output_root / "batch_failure.json").exists()

    for game_number, seed in enumerate((1001, 1002, 1003), start=1):
        game_dir = output_root / "games" / f"game_{game_number:04d}_seed_{seed}"
        assert (game_dir / "trajectory.json").is_file()
        assert (game_dir / "observer_views.json").is_file()
        belief_path = game_dir / batch_module.BELIEF_SNAPSHOTS_FILENAME
        annotation_path = game_dir / batch_module.SPEECH_ANNOTATIONS_FILENAME
        assert belief_path.is_file()
        assert annotation_path.is_file()
        belief_record = json.loads(belief_path.read_text())
        assert belief_record["label_provenance"] == LABEL_PROVENANCE
        assert belief_record["label_cutoff_step_idx"] == 0
        assert belief_record["observer_ids"] == list(range(1, 8))
        serialized_belief = belief_path.read_text()
        assert '"observation"' not in serialized_belief
        assert '"true_roles"' not in serialized_belief
        assert '"winner"' not in serialized_belief
        game_summary = json.loads((game_dir / "summary.json").read_text())
        assert game_summary["environment_seed"] == seed
        assert game_summary["collection_mode"] == "canonical"
        assert game_summary["canonical_eligible"] is True
        assert game_summary["completion_status"] == "COMPLETE"
        assert game_summary["belief_snapshot_count"] == 1
        assert game_summary["speech_annotation_count"] == 1
        assert game_summary["speech_annotation_error_count"] == 0
        assert game_summary["speech_parser_generation_attempt_count"] == 1
        assert game_summary["call_audit"]["within_budget"] is True
        assert game_summary["call_audit"]["gameplay_fallback_count"] == 0
        assert (game_dir / "call_audit.json").is_file()
        assert game_summary["belief_snapshots_sha256"] == batch_module._sha256(
            belief_path
        )
        game_payload = deepcopy(game_summary)
        game_digest = game_payload.pop("summary_digest")
        assert game_digest == canonical_digest(game_payload)


def test_game_failure_is_recorded_and_collection_continues_to_target(
    tmp_path,
    monkeypatch,
):
    config = runtime_config()
    config["pipeline"]["collection"]["seeds"] = [2001, 2002, 2003]
    config["pipeline"]["collection"]["target_game_count"] = 2
    config_path = _write_config(tmp_path, config)
    output_root = tmp_path / "failure-run"
    calls = _install_fake_runtime(monkeypatch, output_root, fail_seed=2002)

    summary = batch_module.collect_canonical_trajectory_batch(
        config_path=config_path,
        run_id="failure-run",
        seed_start=2001,
        game_count=3,
        output_root=output_root,
        repo_root=tmp_path,
    )

    assert calls["game"] == [2001, 2002, 2003]
    assert [call["seed"] for call in calls["build"]] == [2001, 2002, 2003]
    assert (output_root / "games" / "game_0003_seed_2003").is_dir()
    assert (output_root / "summary.json").is_file()
    failure_path = (
        output_root
        / "failures"
        / "game_0002_seed_2002"
        / "failure.json"
    )
    failure = json.loads(failure_path.read_text())
    assert failure["schema_version"] == batch_module.BATCH_FAILURE_SCHEMA_VERSION
    assert failure["failed_seed"] == 2002
    assert failure["failed_game_id"] == "failure-run_game_0002_seed_2002"
    assert failure["failure_stage"] == "gameplay"
    assert failure["exception_message"] == "synthetic gameplay failure"
    assert failure["stop_on_first_failure"] is False
    assert failure["rerun_on_failure"] is False
    assert failure["replacement_seed_on_failure"] is False
    assert (
        output_root / "failures" / "game_0002_seed_2002" / "call_audit.json"
    ).is_file()
    payload = deepcopy(failure)
    digest = payload.pop("failure_digest")
    assert digest == canonical_digest(payload)
    assert summary["canonical_eligible"] is True
    assert summary["target_game_count"] == 2
    assert summary["attempted_game_count"] == 3
    assert summary["completed_game_count"] == 2
    assert summary["failed_game_count"] == 1
    assert summary["completed_seeds"] == [2001, 2003]
    assert summary["failed_seeds"] == [2002]
    assert summary["unattempted_seeds"] == []
    assert summary["failure_digests"] == {
        "failure-run_game_0002_seed_2002": failure["failure_digest"]
    }
    assert not (output_root / "batch_failure.json").exists()
    verified = batch_module.validate_canonical_belief_batch(output_root)
    assert verified["failed_seeds"] == [2002]


def test_canonical_validator_rejects_tampered_failed_game_record(
    tmp_path,
    monkeypatch,
):
    config = runtime_config()
    config["pipeline"]["collection"]["seeds"] = [2001, 2002, 2003]
    config["pipeline"]["collection"]["target_game_count"] = 2
    config_path = _write_config(tmp_path, config)
    output_root = tmp_path / "tampered-failure-run"
    _install_fake_runtime(monkeypatch, output_root, fail_seed=2002)
    batch_module.collect_canonical_trajectory_batch(
        config_path=config_path,
        run_id="tampered-failure-run",
        seed_start=2001,
        game_count=3,
        output_root=output_root,
        repo_root=tmp_path,
    )

    failure_path = (
        output_root
        / "failures"
        / "game_0002_seed_2002"
        / "failure.json"
    )
    failure = json.loads(failure_path.read_text())
    failure["exception_message"] = "tampered"
    failure_path.write_text(canonical_json(failure) + "\n")

    with pytest.raises(ValueError, match="failure_digest mismatch"):
        batch_module.validate_canonical_belief_batch(output_root)


def test_target_completion_leaves_unused_reserve_seeds(tmp_path, monkeypatch):
    config = runtime_config()
    config["pipeline"]["collection"]["target_game_count"] = 2
    config_path = _write_config(tmp_path, config)
    output_root = tmp_path / "reserve-run"
    calls = _install_fake_runtime(monkeypatch, output_root)

    summary = batch_module.collect_canonical_trajectory_batch(
        config_path=config_path,
        run_id="reserve-run",
        seed_start=1001,
        game_count=3,
        output_root=output_root,
        repo_root=tmp_path,
    )

    assert calls["game"] == [1001, 1002]
    assert summary["completed_seeds"] == [1001, 1002]
    assert summary["failed_seeds"] == []
    assert summary["unattempted_seeds"] == [1003]
    assert summary["canonical_eligible"] is True


def test_resume_marks_interrupted_attempt_and_skips_processed_seeds(
    tmp_path,
    monkeypatch,
):
    config = runtime_config()
    config["pipeline"]["collection"]["game_count"] = 4
    config["pipeline"]["collection"]["target_game_count"] = 3
    config["pipeline"]["collection"]["seeds"] = [1001, 1002, 1003, 1004]
    config_path = _write_config(tmp_path, config)
    output_root = tmp_path / "resume-run"
    first_calls = _install_fake_runtime(
        monkeypatch,
        output_root,
        interrupt_seed=1002,
    )

    with pytest.raises(KeyboardInterrupt, match="synthetic interrupted process"):
        batch_module.collect_canonical_trajectory_batch(
            config_path=config_path,
            run_id="resume-run",
            seed_start=1001,
            game_count=4,
            output_root=output_root,
            repo_root=tmp_path,
        )

    assert first_calls["game"] == [1001, 1002]
    second_calls = _install_fake_runtime(monkeypatch, output_root)
    summary = batch_module.collect_canonical_trajectory_batch(
        config_path=config_path,
        run_id="resume-run",
        seed_start=1001,
        game_count=4,
        output_root=output_root,
        repo_root=tmp_path,
        resume=True,
    )

    assert second_calls["game"] == [1003, 1004]
    assert summary["completed_seeds"] == [1001, 1003, 1004]
    assert summary["failed_seeds"] == [1002]
    assert summary["completed_game_count"] == 3
    assert summary["canonical_eligible"] is True
    interrupted_failure = json.loads(
        (
            output_root
            / "failures"
            / "game_0002_seed_1002"
            / "failure.json"
        ).read_text()
    )
    assert interrupted_failure["failure_stage"] == "interrupted_previous_process"
    assert interrupted_failure["exception_type"] == "InterruptedError"


def test_resume_rejects_any_changed_frozen_plan(tmp_path, monkeypatch):
    config = runtime_config()
    config["pipeline"]["collection"]["game_count"] = 4
    config["pipeline"]["collection"]["target_game_count"] = 3
    config["pipeline"]["collection"]["seeds"] = [1001, 1002, 1003, 1004]
    config_path = _write_config(tmp_path, config)
    output_root = tmp_path / "changed-plan-resume"
    _install_fake_runtime(monkeypatch, output_root, interrupt_seed=1002)

    with pytest.raises(KeyboardInterrupt):
        batch_module.collect_canonical_trajectory_batch(
            config_path=config_path,
            run_id="changed-plan-resume",
            seed_start=1001,
            game_count=4,
            output_root=output_root,
            repo_root=tmp_path,
        )

    changed = deepcopy(config)
    changed["pipeline"]["collection"]["max_total_calls_per_game"] += 1
    _write_config(tmp_path, changed)
    second_calls = _install_fake_runtime(monkeypatch, output_root)
    with pytest.raises(ValueError, match="exact same commit, config"):
        batch_module.collect_canonical_trajectory_batch(
            config_path=config_path,
            run_id="changed-plan-resume",
            seed_start=1001,
            game_count=4,
            output_root=output_root,
            repo_root=tmp_path,
            resume=True,
        )
    assert second_calls["game"] == []


def test_exhausted_seed_pool_is_explicitly_incomplete(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path)
    output_root = tmp_path / "incomplete-run"
    _install_fake_runtime(monkeypatch, output_root, fail_seed=1002)

    summary = batch_module.collect_canonical_trajectory_batch(
        config_path=config_path,
        run_id="incomplete-run",
        seed_start=1001,
        game_count=3,
        output_root=output_root,
        repo_root=tmp_path,
    )

    assert summary["completed_game_count"] == 2
    assert summary["failed_game_count"] == 1
    assert summary["target_reached"] is False
    assert summary["canonical_eligible"] is False
    with pytest.raises(ValueError, match="not canonical-eligible"):
        batch_module.validate_canonical_belief_batch(output_root)


def test_pilot_mode_is_explicitly_noncanonical_and_cannot_materialize(
    tmp_path,
    monkeypatch,
):
    config_path = _write_config(tmp_path)
    output_root = tmp_path / "diagnostic-pilot"
    calls = _install_fake_runtime(monkeypatch, output_root)

    summary = batch_module.collect_canonical_trajectory_batch(
        config_path=config_path,
        run_id="diagnostic-pilot",
        seed_start=1001,
        game_count=3,
        output_root=output_root,
        repo_root=tmp_path,
        collection_mode="pilot",
    )

    assert calls["allow_gameplay_fallback"] == [True, True, True]
    assert summary["collection_mode"] == "pilot"
    assert summary["canonical_eligible"] is False
    plan = json.loads((output_root / "plan.json").read_text())
    assert plan["collection_mode"] == "pilot"
    assert plan["canonical_eligible"] is False
    with pytest.raises(ValueError, match="not in canonical mode"):
        batch_module.validate_canonical_belief_batch(output_root)


def test_pilot_records_failed_speech_annotations_and_completes_batch(
    tmp_path,
    monkeypatch,
):
    config_path = _write_config(tmp_path)
    output_root = tmp_path / "speech-error-pilot"
    calls = _install_fake_runtime(
        monkeypatch,
        output_root,
        speech_error=True,
    )

    summary = batch_module.collect_canonical_trajectory_batch(
        config_path=config_path,
        run_id="speech-error-pilot",
        seed_start=1001,
        game_count=3,
        output_root=output_root,
        repo_root=tmp_path,
        collection_mode="pilot",
    )

    assert calls["game"] == [1001, 1002, 1003]
    assert summary["completed_game_count"] == 3
    assert summary["canonical_eligible"] is False
    assert summary["total_speech_annotation_error_count"] == 3
    assert summary["total_speech_parser_generation_attempt_count"] == 9
    assert not (output_root / "batch_failure.json").exists()


def test_canonical_validator_rejects_embedded_gameplay_fallback(
    tmp_path,
    monkeypatch,
):
    config_path = _write_config(tmp_path)
    output_root = tmp_path / "canonical-with-fallback"
    _install_fake_runtime(monkeypatch, output_root)
    batch_module.collect_canonical_trajectory_batch(
        config_path=config_path,
        run_id="canonical-with-fallback",
        seed_start=1001,
        game_count=3,
        output_root=output_root,
        repo_root=tmp_path,
    )

    game_summary_path = (
        output_root / "games" / "game_0001_seed_1001" / "summary.json"
    )
    game_summary = json.loads(game_summary_path.read_text())
    game_summary["call_audit"]["gameplay_fallback_count"] = 1
    game_summary.pop("summary_digest")
    game_summary["summary_digest"] = canonical_digest(game_summary)
    game_summary_path.write_text(canonical_json(game_summary) + "\n")

    summary_path = output_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["game_summary_digests"][game_summary["game_id"]] = (
        game_summary["summary_digest"]
    )
    summary.pop("summary_digest")
    summary["summary_digest"] = canonical_digest(summary)
    summary_path.write_text(canonical_json(summary) + "\n")

    with pytest.raises(ValueError, match="contains gameplay fallback"):
        batch_module.validate_canonical_belief_batch(output_root)


@pytest.mark.parametrize(
    ("field_name", "error_pattern"),
    [
        (
            "total_missing_pre_belief_snapshot_count",
            "missing PRE belief snapshots",
        ),
        (
            "total_label_snapshot_failure_count",
            "failed label snapshots",
        ),
    ],
)
def test_canonical_validator_rejects_incomplete_label_coverage(
    tmp_path,
    monkeypatch,
    field_name,
    error_pattern,
):
    config_path = _write_config(tmp_path)
    output_root = tmp_path / f"canonical-{field_name}"
    _install_fake_runtime(monkeypatch, output_root)
    batch_module.collect_canonical_trajectory_batch(
        config_path=config_path,
        run_id=f"canonical-{field_name}",
        seed_start=1001,
        game_count=3,
        output_root=output_root,
        repo_root=tmp_path,
    )
    summary_path = output_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary[field_name] = 1
    summary.pop("summary_digest")
    summary["summary_digest"] = canonical_digest(summary)
    summary_path.write_text(canonical_json(summary) + "\n")

    with pytest.raises(ValueError, match=error_pattern):
        batch_module.validate_canonical_belief_batch(output_root)


def test_batch_failure_message_redacts_secrets():
    failure = batch_module._failure_record(
        run_id="failed-run",
        commit=COMMIT,
        failed_seed=1,
        failed_game_id="failed-game",
        failure_stage="gameplay",
        exception=RuntimeError("request failed api_key=raw-test-secret"),
        completed_games=[],
    )

    assert failure["exception_message"] == "request failed api_key=<redacted>"


def test_existing_destination_is_rejected_before_any_runtime_call(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path)
    output_root = tmp_path / "already-exists"
    output_root.mkdir()

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("runtime path must not be reached")

    monkeypatch.setattr(batch_module, "_read_code_provenance", should_not_run)
    with pytest.raises(FileExistsError, match="output root already exists"):
        batch_module.collect_canonical_trajectory_batch(
            config_path=config_path,
            run_id="run-001",
            seed_start=1,
            game_count=1,
            output_root=output_root,
            repo_root=tmp_path,
        )


def test_non_classic7_config_is_rejected_before_output_publication(tmp_path, monkeypatch):
    config = runtime_config()
    config["env_config"]["n_villager"] = 2
    config_path = _write_config(tmp_path, config)
    output_root = tmp_path / "bad-config"
    monkeypatch.setattr(
        batch_module,
        "_read_code_provenance",
        lambda _root: {"batch_code_commit": COMMIT, "git_worktree_clean": True},
    )

    with pytest.raises(ValueError, match="frozen Classic-7 role counts"):
        batch_module.collect_canonical_trajectory_batch(
            config_path=config_path,
            run_id="bad-config",
            seed_start=1,
            game_count=1,
            output_root=output_root,
            repo_root=tmp_path,
        )
    assert not output_root.exists()


def test_pipeline_collection_must_match_cli_before_output_publication(
    tmp_path,
    monkeypatch,
):
    config = runtime_config()
    config["pipeline"]["collection"]["seeds"] = [1, 2, 4]
    config_path = _write_config(tmp_path, config)
    output_root = tmp_path / "bad-pipeline"
    monkeypatch.setattr(
        batch_module,
        "_read_code_provenance",
        lambda _root: {"batch_code_commit": COMMIT, "git_worktree_clean": True},
    )

    with pytest.raises(ValueError, match="exactly match"):
        batch_module.collect_canonical_trajectory_batch(
            config_path=config_path,
            run_id="bad-pipeline",
            seed_start=1,
            game_count=3,
            output_root=output_root,
            repo_root=tmp_path,
        )
    assert not output_root.exists()


def test_run_id_and_game_count_fail_closed_before_output(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path)
    monkeypatch.setattr(
        batch_module,
        "_read_code_provenance",
        lambda _root: {"batch_code_commit": COMMIT, "git_worktree_clean": True},
    )

    with pytest.raises(ValueError, match="run_id"):
        batch_module.collect_canonical_trajectory_batch(
            config_path=config_path,
            run_id="../escape",
            seed_start=1,
            game_count=1,
            output_root=tmp_path / "x",
            repo_root=tmp_path,
        )
    with pytest.raises(ValueError, match="game_count"):
        batch_module.collect_canonical_trajectory_batch(
            config_path=config_path,
            run_id="run-001",
            seed_start=1,
            game_count=0,
            output_root=tmp_path / "y",
            repo_root=tmp_path,
        )


def test_complete_artifact_validator_rejects_wrong_winner(tmp_path):
    players = [
        {
            "player_id": index,
            "role": ROLES[index - 1],
            "profile_name": PROFILES[index - 1],
            "backend_id": f"backend-{index}",
            "model_name": f"model-{index}",
        }
        for index in range(1, 8)
    ]
    recorder = batch_module.CanonicalGameInteractionTrajectoryRecorder(
        tmp_path / "trajectory.json",
        tmp_path / "observer_views.json",
        game_id="game_001",
        run_id="run-001",
        source_commit=COMMIT,
        environment_seed=11,
        runtime_config={"x": 1},
        players=players,
    )
    _complete_artifacts(recorder, winner="Villager")

    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    trajectory["termination"]["winner"] = "Werewolf"
    trajectory.pop("trajectory_digest")
    trajectory["trajectory_digest"] = canonical_digest(trajectory)
    _write_json(tmp_path / "tampered_trajectory.json", trajectory)

    provenance = json.loads((tmp_path / "observer_views.json").read_text())
    provenance["trajectory_digest"] = trajectory["trajectory_digest"]
    provenance.pop("artifact_digest")
    provenance["artifact_digest"] = canonical_digest(provenance)
    _write_json(tmp_path / "tampered_views.json", provenance)

    with pytest.raises(ValueError, match="winner is not mechanically valid"):
        batch_module.validate_complete_game_artifacts(
            tmp_path / "tampered_trajectory.json",
            tmp_path / "tampered_views.json",
            expected_game_id="game_001",
            expected_run_id="run-001",
            expected_seed=11,
            expected_source_commit=COMMIT,
        )


def test_complete_artifact_validator_checks_boundary_and_record_digests(tmp_path):
    players = [
        {
            "player_id": index,
            "role": ROLES[index - 1],
            "profile_name": PROFILES[index - 1],
            "backend_id": f"backend-{index}",
            "model_name": f"model-{index}",
        }
        for index in range(1, 8)
    ]
    recorder = batch_module.CanonicalGameInteractionTrajectoryRecorder(
        tmp_path / "trajectory.json",
        tmp_path / "observer_views.json",
        game_id="game_001",
        run_id="run-001",
        source_commit=COMMIT,
        environment_seed=11,
        runtime_config={"x": 1},
        players=players,
    )
    _complete_artifacts(recorder, winner="Villager")

    validated = batch_module.validate_complete_game_artifacts(
        tmp_path / "trajectory.json",
        tmp_path / "observer_views.json",
        expected_game_id="game_001",
        expected_run_id="run-001",
        expected_seed=11,
        expected_source_commit=COMMIT,
    )
    assert validated["transition_count"] == 1
    assert validated["pre_public_speech_boundary_count"] == 1
    assert validated["post_public_speech_boundary_count"] == 1
    assert validated["observer_view_count"] == 12

    provenance = json.loads((tmp_path / "observer_views.json").read_text())
    provenance["boundaries"][0]["observer_views"][0]["observation"]["phase"] = "tampered"
    provenance["boundaries"][0]["observer_views"][0]["observation_digest"] = canonical_digest(
        provenance["boundaries"][0]["observer_views"][0]["observation"]
    )
    provenance["boundaries"][0].pop("boundary_digest")
    provenance["boundaries"][0]["boundary_digest"] = canonical_digest(
        {k: v for k, v in provenance["boundaries"][0].items() if k != "boundary_digest"}
    )
    provenance.pop("artifact_digest")
    provenance["artifact_digest"] = canonical_digest(provenance)
    _write_json(tmp_path / "tampered_views.json", provenance)

    with pytest.raises(ValueError, match="PRE observer view (current actor|phase) mismatch|boundary"):
        batch_module.validate_complete_game_artifacts(
            tmp_path / "trajectory.json",
            tmp_path / "tampered_views.json",
            expected_game_id="game_001",
            expected_run_id="run-001",
            expected_seed=11,
            expected_source_commit=COMMIT,
        )


def test_belief_artifact_rejects_any_failed_alive_observer(tmp_path):
    players = [
        {
            "player_id": index,
            "role": ROLES[index - 1],
            "profile_name": PROFILES[index - 1],
            "backend_id": f"backend-{index}",
            "model_name": f"model-{index}",
        }
        for index in range(1, 8)
    ]
    recorder = batch_module.CanonicalGameInteractionTrajectoryRecorder(
        tmp_path / "trajectory.json",
        tmp_path / "observer_views.json",
        game_id="game_001",
        run_id="run-001",
        source_commit=COMMIT,
        environment_seed=11,
        runtime_config={"x": 1},
        players=players,
    )
    _complete_artifacts(recorder)
    snapshot = _belief_snapshot("game_001")
    snapshot["belief_status"]["player3"] = "parse_error"
    snapshot["belief_errors"]["player3"] = "synthetic parse failure"
    snapshot["suspected_werewolves"]["player3"] = None
    belief_path = tmp_path / "belief_snapshots.jsonl"
    belief_path.write_text(json.dumps(snapshot) + "\n", encoding="utf-8")
    speech_path = tmp_path / "speech_annotations.jsonl"
    speech_path.write_text(
        "".join(
            json.dumps(annotation) + "\n"
            for annotation in snapshot["speech_annotations"]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="status=ok.*player3"):
        batch_module.validate_belief_snapshot_artifact(
            belief_path,
            tmp_path / "observer_views.json",
            speech_path,
            expected_game_id="game_001",
        )


def test_pilot_belief_artifact_allows_missing_failed_pre_snapshot(tmp_path):
    players = [
        {
            "player_id": index,
            "role": ROLES[index - 1],
            "profile_name": PROFILES[index - 1],
            "backend_id": f"backend-{index}",
            "model_name": f"model-{index}",
        }
        for index in range(1, 8)
    ]
    recorder = batch_module.CanonicalGameInteractionTrajectoryRecorder(
        tmp_path / "trajectory.json",
        tmp_path / "observer_views.json",
        game_id="game_001",
        run_id="run-001",
        source_commit=COMMIT,
        environment_seed=11,
        runtime_config={"x": 1},
        players=players,
    )
    _complete_artifacts(recorder)
    snapshot = _belief_snapshot("game_001")
    belief_path = tmp_path / "belief_snapshots.jsonl"
    belief_path.write_text("", encoding="utf-8")
    speech_path = tmp_path / "speech_annotations.jsonl"
    speech_path.write_text(
        "".join(
            json.dumps(annotation) + "\n"
            for annotation in snapshot["speech_annotations"]
        ),
        encoding="utf-8",
    )

    validation = batch_module.validate_belief_snapshot_artifact(
        belief_path,
        tmp_path / "observer_views.json",
        speech_path,
        expected_game_id="game_001",
        require_complete=False,
    )

    assert validation["belief_snapshot_count"] == 0
    assert validation["belief_snapshot_complete"] is False
    assert validation["belief_snapshot_missing_pre_boundary_count"] == 1
    with pytest.raises(ValueError, match="count must equal"):
        batch_module.validate_belief_snapshot_artifact(
            belief_path,
            tmp_path / "observer_views.json",
            speech_path,
            expected_game_id="game_001",
        )


def test_belief_artifact_annotations_must_match_canonical_sidecar_prefix(tmp_path):
    snapshot = _belief_snapshot("game_001")
    public_events = [
        snapshot["public_events"][0],
        snapshot["public_events"][1],
        {
            "event_idx": 2,
            "event_type": "public_speech",
            "speaker": "player2",
            "raw_text": "synthetic speech",
        },
        {
            "event_idx": 3,
            "event_type": "turn_start",
            "speaker": "player3",
        },
    ]
    snapshot_annotation = make_speech_annotation(
        event_idx=2,
        speaker="player2",
        raw_text="synthetic speech",
        parser_model_id="synthetic-parser",
        parser_call_id="snapshot-parser-call",
        annotation_source="llm_parser",
        status="ok",
        actions=[["player2", "support", "player3"]],
        raw_response=None,
        error_type=None,
        error_message=None,
    )
    canonical_annotation = make_speech_annotation(
        event_idx=2,
        speaker="player2",
        raw_text="synthetic speech",
        parser_model_id="synthetic-parser",
        parser_call_id="canonical-parser-call",
        annotation_source="llm_parser",
        status="ok",
        actions=[["player2", "oppose", "player3"]],
        raw_response=None,
        error_type=None,
        error_message=None,
    )
    snapshot["step_idx"] = 1
    snapshot["label_cutoff_step_idx"] = 1
    snapshot["speaker_id"] = 3
    snapshot["public_events"] = public_events
    snapshot["speech_annotations"] = [snapshot_annotation]
    snapshot["public_event_digest"] = public_event_digest(public_events)
    snapshot["speech_annotation_digest"] = speech_annotation_digest(
        snapshot["speech_annotations"]
    )
    snapshot["structured_input_digest"] = structured_input_digest(
        public_events,
        snapshot["speech_annotations"],
    )
    snapshot["public_action_count"] = 1

    observer_views = {
        "boundaries": [
            {
                "boundary_type": batch_module.PRE_PUBLIC_SPEECH,
                "step_idx": 1,
                "speech_kind": "speech",
                "speaker_id": 3,
                "public_event_count_at_materialization": len(public_events),
                "public_event_digest_at_materialization": public_event_digest(
                    public_events
                ),
                "observer_views": [
                    {
                        "observer_id": observer_id,
                        "observation": {"phase": "1_day_speech"},
                    }
                    for observer_id in range(1, 8)
                ],
            }
        ]
    }
    belief_path = tmp_path / "belief_snapshots.jsonl"
    observer_views_path = tmp_path / "observer_views.json"
    speech_path = tmp_path / "speech_annotations.jsonl"
    belief_path.write_text(json.dumps(snapshot) + "\n", encoding="utf-8")
    observer_views_path.write_text(
        json.dumps(observer_views) + "\n",
        encoding="utf-8",
    )
    speech_path.write_text(
        json.dumps(canonical_annotation) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="differ from canonical sidecar"):
        batch_module.validate_belief_snapshot_artifact(
            belief_path,
            observer_views_path,
            speech_path,
            expected_game_id="game_001",
        )


def test_canonical_speech_annotation_artifact_fails_closed(
    tmp_path,
):
    players = [
        {
            "player_id": index,
            "role": ROLES[index - 1],
            "profile_name": PROFILES[index - 1],
            "backend_id": f"backend-{index}",
            "model_name": f"model-{index}",
        }
        for index in range(1, 8)
    ]
    recorder = batch_module.CanonicalGameInteractionTrajectoryRecorder(
        tmp_path / "trajectory.json",
        tmp_path / "observer_views.json",
        game_id="game_001",
        run_id="run-001",
        source_commit=COMMIT,
        environment_seed=11,
        runtime_config={"x": 1},
        players=players,
    )
    _complete_artifacts(recorder)
    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    speech = next(
        event
        for transition in trajectory["transitions"]
        for event in transition["public_events_appended"]
        if event["event_type"] == "public_speech"
    )
    annotation = make_speech_annotation(
        event_idx=speech["event_idx"],
        speaker=speech["speaker"],
        raw_text=speech["raw_text"],
        parser_model_id="synthetic-parser",
        parser_call_id="synthetic-parser-call-000001",
        annotation_source="llm_parser",
        status="error",
        actions=[],
        generation_attempts=[
            {
                "generation_attempt": 1,
                "status": "parser_error",
                "raw_response": "malformed",
                "error_type": "SyntheticParserError",
                "error_message": "synthetic failure",
            }
        ],
        raw_response="malformed",
        error_type="SyntheticParserError",
        error_message="synthetic failure",
    )
    annotation_path = tmp_path / "speech_annotations.jsonl"
    annotation_path.write_text(
        json.dumps(annotation, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="successful parsing"):
        batch_module.validate_speech_annotation_artifact(
            annotation_path,
            tmp_path / "trajectory.json",
        )

    pilot_validation = batch_module.validate_speech_annotation_artifact(
        annotation_path,
        tmp_path / "trajectory.json",
        require_success=False,
    )
    assert pilot_validation["speech_annotation_error_count"] == 1
    assert pilot_validation["speech_annotation_error_event_indices"] == [
        speech["event_idx"]
    ]
