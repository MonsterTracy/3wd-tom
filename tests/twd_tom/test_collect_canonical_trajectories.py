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
            "raw_schema_version": batch_module.SAMPLE_SCHEMA_VERSION,
            "projected_schema_version": batch_module.PROJECTED_SCHEMA_VERSION,
            "projection_version": batch_module.TARGET_CONVERSION,
            "collection": {
                "game_count": 3,
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
        "sp_actions": [],
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
            {"raw_text": "synthetic speech", "sp_actions": []},
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
        "structured_input_digest": structured_input_digest(public_events),
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


def _install_fake_runtime(monkeypatch, output_root, *, fail_seed=None):
    calls = {
        "backend": [],
        "build": [],
        "game": [],
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
        return object(), agents, list(ROLES), list(PROFILES)

    def fake_run_game(
        env,
        agents,
        roles,
        *,
        sample_collector,
        call_audit,
        trajectory_recorder,
    ):
        assert call_audit.game_id == trajectory_recorder._base["game_id"]
        seed = trajectory_recorder._base["environment_seed"]
        calls["game"].append(seed)
        if seed == fail_seed:
            raise RuntimeError("synthetic gameplay failure")
        winner = "Villager" if seed % 2 else "Werewolf"
        sample_collector.write(_belief_snapshot(trajectory_recorder._base["game_id"]))
        _complete_artifacts(trajectory_recorder, winner=winner)
        return f"{winner} win"

    monkeypatch.setattr(batch_module, "_read_code_provenance", fake_provenance)
    monkeypatch.setattr(batch_module, "load_named_backends", fake_load_backends)
    monkeypatch.setattr(batch_module, "build_runtime", fake_build_runtime)
    monkeypatch.setattr(batch_module, "run_game", fake_run_game)
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
    assert [call["seed"] for call in calls["build"]] == [1001, 1002, 1003]
    assert calls["backend"][0]["max_retries"] == 0
    assert calls["backend"][0]["env_file"] == tmp_path / ".env"

    plan = json.loads((output_root / "plan.json").read_text())
    assert plan["schema_version"] == batch_module.BATCH_PLAN_SCHEMA_VERSION
    assert plan["batch_code_commit"] == COMMIT
    assert plan["seeds"] == [1001, 1002, 1003]
    assert plan["planned_game_count"] == 3
    assert plan["collection_contract"]["seeds"] == [1001, 1002, 1003]
    assert plan["collectors_enabled"] is True
    assert plan["backend_max_retries"] == 0
    assert plan["stop_on_first_failure"] is True
    assert plan["rerun_on_failure"] is False
    assert plan["replacement_seed_on_failure"] is False
    payload = deepcopy(plan)
    digest = payload.pop("plan_digest")
    assert digest == canonical_digest(payload)

    assert summary["schema_version"] == batch_module.BATCH_SUMMARY_SCHEMA_VERSION
    assert summary["completed_game_count"] == 3
    assert summary["total_belief_snapshot_count"] == 3
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
        assert belief_path.is_file()
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
        assert game_summary["completion_status"] == "COMPLETE"
        assert game_summary["belief_snapshot_count"] == 1
        assert game_summary["call_audit"]["within_budget"] is True
        assert (game_dir / "call_audit.json").is_file()
        assert game_summary["belief_snapshots_sha256"] == batch_module._sha256(
            belief_path
        )
        game_payload = deepcopy(game_summary)
        game_digest = game_payload.pop("summary_digest")
        assert game_digest == canonical_digest(game_payload)


def test_failure_stops_batch_without_retry_or_replacement(tmp_path, monkeypatch):
    config = runtime_config()
    config["pipeline"]["collection"]["seeds"] = [2001, 2002, 2003]
    config_path = _write_config(tmp_path, config)
    output_root = tmp_path / "failure-run"
    calls = _install_fake_runtime(monkeypatch, output_root, fail_seed=2002)

    with pytest.raises(RuntimeError, match="synthetic gameplay failure"):
        batch_module.collect_canonical_trajectory_batch(
            config_path=config_path,
            run_id="failure-run",
            seed_start=2001,
            game_count=3,
            output_root=output_root,
            repo_root=tmp_path,
        )

    assert calls["game"] == [2001, 2002]
    assert [call["seed"] for call in calls["build"]] == [2001, 2002]
    assert not (output_root / "games" / "game_0003_seed_2003").exists()
    assert not (output_root / "summary.json").exists()
    failure = json.loads((output_root / "batch_failure.json").read_text())
    assert failure["schema_version"] == batch_module.BATCH_FAILURE_SCHEMA_VERSION
    assert failure["failed_seed"] == 2002
    assert failure["failed_game_id"] == "failure-run_game_0002_seed_2002"
    assert failure["failure_stage"] == "gameplay"
    assert failure["completed_game_count"] == 1
    assert failure["completed_game_ids"] == ["failure-run_game_0001_seed_2001"]
    assert failure["stop_on_first_failure"] is True
    assert failure["rerun_on_failure"] is False
    assert failure["replacement_seed_on_failure"] is False
    assert (
        output_root / "games" / "game_0002_seed_2002" / "call_audit.json"
    ).is_file()
    payload = deepcopy(failure)
    digest = payload.pop("failure_digest")
    assert digest == canonical_digest(payload)


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

    with pytest.raises(ValueError, match="status=ok.*player3"):
        batch_module.validate_belief_snapshot_artifact(
            belief_path,
            tmp_path / "observer_views.json",
            expected_game_id="game_001",
        )
