from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from script.twd_tom import pipeline
from script.twd_tom.eval import (
    build_model_from_checkpoint,
    load_checkpoint,
)
from werewolf.models.twd_tom.dataset import (
    TWDToMDataset,
    collate_twd_tom_samples,
)
from werewolf.models.twd_tom.losses import (
    masked_pair_kl_divergence,
)
from werewolf.models.twd_tom.samples import (
    SAMPLE_SCHEMA_VERSION,
)
from werewolf.models.twd_tom.schema import (
    LABEL_PROMPT_VERSION,
    PROJECTED_SCHEMA_VERSION,
    PROJECTION_VERSION,
)
from werewolf.models.twd_tom.public_events import PUBLIC_EVENT_SCHEMA_VERSION


def _pipeline_config(
    tmp_path: Path,
    *,
    seeds: tuple[int, ...] = (
        101,
        102,
        103,
    ),
) -> dict:
    return {
        "backends": {
            "fake": {
                "type": (
                    "openai_compatible"
                ),
                "base_url": (
                    "https://example.invalid"
                ),
                "api_key_env": (
                    "FAKE_API_KEY"
                ),
            },
        },
        "parser": {
            "backend": "fake",
            "model": "fake-model",
            "model_params": {
                "temperature": 0.0,
            },
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
            "allow_cross_team_profiles": (
                True
            ),
            "must_include": [],
            "all_candidates": [
                {
                    "profile_name": (
                        "fake_agent"
                    ),
                    "agent_type": "deepseek",
                    "backend": "fake",
                    "model": "fake-model",
                    "model_params": {
                        "temperature": 1.0,
                    },
                    "sample_ratio": 1.0,
                },
            ],
        },
        "pipeline": {
            "public_event_schema_version": PUBLIC_EVENT_SCHEMA_VERSION,
            "raw_schema_version": SAMPLE_SCHEMA_VERSION,
            "projected_schema_version": PROJECTED_SCHEMA_VERSION,
            "projection_version": PROJECTION_VERSION,
            "collection": {
                "game_count": len(seeds),
                "seeds": list(seeds),
                "max_gameplay_calls_per_game": 10,
                "max_belief_calls_per_game": 20,
                "max_total_calls_per_game": 30,
                "max_wall_seconds_per_game": 60.0,
            },
            "project": {},
            "split": {
                "seed": 11,
                "train_game_count": 1,
                "validation_game_count": 1,
                "test_game_count": 1,
            },
            "train": {
                "epochs": 1,
                "batch_size": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "seed": 3,
                "device": "cpu",
                "num_workers": 0,
                "gradient_clip_norm": 1.0,
                "early_stopping_patience": 2,
                "early_stopping_min_delta": 0.0,
                "d_model": 8,
                "n_head": 2,
                "n_layer": 1,
                "dropout": 0.0,
                "max_seq_len": 8,
                "dim_feedforward": 16,
            },
            "eval": {
                "batch_size": 1,
                "device": "cpu",
                "num_workers": 0,
                "allow_game_id_overlap": False,
            },
        },
    }


def _write_config(
    tmp_path: Path,
    *,
    seeds: tuple[int, ...] = (
        101,
        102,
        103,
    ),
) -> Path:
    config_path = (
        tmp_path / "pipeline.yaml"
    )
    config_path.write_text(
        yaml.safe_dump(
            _pipeline_config(
                tmp_path,
                seeds=seeds,
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


def _use_temporary_repo(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        pipeline,
        "REPO_ROOT",
        tmp_path,
    )


def _install_fake_collection(
    monkeypatch,
    suspicion_sample_factory,
    *,
    games_dir: Path,
):
    calls = []

    def fake_game(
        *,
        parsed_yaml,
        samples_path,
        log_dir,
        game_id,
        seed,
        budget,
        writer,
        **_kwargs,
    ):
        assert parsed_yaml[
            "backends"
        ]["fake"]["api_key_env"] == (
            "FAKE_API_KEY"
        )
        assert parsed_yaml[
            "agent_config"
        ]["all_candidates"][0][
            "profile_name"
        ] == "fake_agent"
        assert writer is not None
        assert log_dir.parent == games_dir
        assert log_dir.name == game_id
        (log_dir / "game_log.json").write_text(
            "{}\n",
            encoding="utf-8",
        )
        calls.append(
            (
                game_id,
                seed,
                budget.game_id,
            )
        )
        sample = (
            suspicion_sample_factory(
                game_id=game_id,
            )
        )
        with samples_path.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(sample)
                + "\n"
            )

    monkeypatch.setattr(
        pipeline.collection_core,
        "run_real_backend_game",
        fake_game,
    )
    monkeypatch.setattr(
        pipeline,
        "load_dotenv",
        lambda **_kwargs: False,
    )
    return calls


def test_debug_config_validates_without_api_key_or_artifacts(
    monkeypatch,
):
    config_path = (
        Path(pipeline.__file__)
        .resolve()
        .parents[2]
        / "configs"
        / "twd_tom_pipeline_debug.yaml"
    )
    monkeypatch.delenv(
        "DEEPSEEK_API_KEY",
        raising=False,
    )
    monkeypatch.setattr(
        pipeline,
        "load_dotenv",
        lambda **_kwargs: pytest.fail(
            "validate must not load .env"
        ),
    )
    monkeypatch.setattr(
        pipeline.collection_core,
        "run_real_backend_game",
        lambda **_kwargs: pytest.fail(
            "validate must not run a game"
        ),
    )

    summary = pipeline.run_pipeline_stage(
        config_path=config_path,
        run_id="debug4101",
        stage="validate",
    )

    assert summary["status"] == "ok"
    assert summary["stage"] == "validate"
    assert summary["input_path"] is None
    assert summary["output_path"] is None
    assert summary["run_id"] == "debug4101"
    assert summary["game_count"] == 3
    assert summary["seeds"] == [3101, 3102, 3103]
    plan = summary["plan"]
    assert plan["logs"]["run_dir"].endswith(
        "logs/tom/debug4101"
    )
    assert plan["data"]["run_dir"].endswith(
        "data/tom/debug4101"
    )
    assert plan["outputs"]["run_dir"].endswith(
        "outputs/tom/debug4101"
    )
    assert not Path(plan["logs"]["run_dir"]).exists()
    assert not Path(plan["data"]["run_dir"]).exists()
    assert not Path(plan["outputs"]["run_dir"]).exists()
    serialized = json.dumps(summary)
    assert "DEEPSEEK_API_KEY" not in serialized
    assert "secret" not in serialized


def test_runtime_contract_is_tom_only_and_keeps_module_names():
    repo_root = (
        Path(pipeline.__file__)
        .resolve()
        .parents[2]
    )
    checked_files = {
        *(
            repo_root / "script"
        ).rglob("*.py"),
        *(
            repo_root / "werewolf"
        ).rglob("*.py"),
        *(
            repo_root / "configs"
        ).rglob("*.yaml"),
        *(
            repo_root / "docs"
        ).rglob("*.md"),
        *repo_root.glob("run_*.py"),
        repo_root / "README.md",
        repo_root / ".gitignore",
        repo_root / "setup.py",
        repo_root / "run_batch.sh",
    }
    legacy_runtime_name = "_".join(
        ("twd", "tom")
    )
    forbidden_paths = tuple(
        f"{root}/{legacy_runtime_name}/"
        for root in (
            "data",
            "logs",
            "outputs",
        )
    )

    for path in checked_files:
        text = path.read_text(
            encoding="utf-8"
        )
        assert all(
            forbidden not in text
            for forbidden in forbidden_paths
        ), path

    for module_path in (
        repo_root / "script" / "twd_tom",
        repo_root / "werewolf" / "models" / "twd_tom",
        repo_root / "tests" / "twd_tom",
    ):
        assert module_path.is_dir()


def test_runtime_paths_are_scoped_to_tom_roots(
    tmp_path,
    monkeypatch,
):
    _use_temporary_repo(
        tmp_path,
        monkeypatch,
    )
    paths = pipeline._run_paths(
        "test_run"
    )
    expected_roots = {
        "data": (
            tmp_path / "data" / "tom"
        ).resolve(),
        "logs": (
            tmp_path / "logs" / "tom"
        ).resolve(),
        "outputs": (
            tmp_path / "outputs" / "tom"
        ).resolve(),
    }

    for group, root in expected_roots.items():
        run_dir = (
            root / "test_run"
        ).resolve()
        assert paths[group][
            "run_dir"
        ] == run_dir
        assert all(
            path == run_dir
            or run_dir in path.parents
            for path in paths[
                group
            ].values()
        )
        assert not root.exists()


def test_validate_rejects_non_v2_pipeline_schema(tmp_path):
    config = _pipeline_config(tmp_path)
    config["pipeline"]["raw_schema_version"] = (
        "classic7_pre_speech_player_suspicion_v1"
    )
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="raw_schema_version"):
        pipeline.run_pipeline_stage(
            config_path=path,
            run_id="test_run",
            stage="validate",
        )


def test_cli_exposes_only_run_and_collection_instance_parameters():
    parser = (
        pipeline.build_argument_parser()
    )
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option != "-h"
        and option != "--help"
    }
    assert option_strings == {
        "--config",
        "--run-id",
        "--stage",
        "--game-count",
        "--seeds",
    }
    stage_action = next(
        action
        for action in parser._actions
        if "--stage"
        in action.option_strings
    )
    assert tuple(
        stage_action.choices
    ) == pipeline.STAGES
    assert pipeline.STAGES == (
        "validate",
        "collect",
        "project",
        "split",
        "train",
        "eval",
    )


def test_validate_cli_override_is_in_memory_only(
    tmp_path,
    monkeypatch,
    capsys,
):
    _use_temporary_repo(
        tmp_path,
        monkeypatch,
    )
    config_path = _write_config(
        tmp_path,
    )
    original_yaml = config_path.read_bytes()
    monkeypatch.setattr(
        pipeline,
        "load_dotenv",
        lambda **_kwargs: pytest.fail(
            "validate must not load .env"
        ),
    )

    assert pipeline.main(
        [
            "--config",
            str(config_path),
            "--run-id",
            "debug4101",
            "--stage",
            "validate",
            "--game-count",
            "1",
            "--seeds",
            "4101",
        ]
    ) == 0

    summary = json.loads(
        capsys.readouterr().out
    )
    assert summary["run_id"] == "debug4101"
    assert summary["game_count"] == 1
    assert summary["seeds"] == [4101]
    assert config_path.read_bytes() == original_yaml
    assert not (tmp_path / "logs").exists()
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize(
    ("game_count", "seeds", "message"),
    [
        (
            1,
            None,
            "--game-count and --seeds must be provided together",
        ),
        (
            None,
            (4101,),
            "--game-count and --seeds must be provided together",
        ),
        (
            2,
            (4101,),
            "number of seeds",
        ),
        (
            0,
            (),
            "game_count must be positive",
        ),
    ],
)
def test_collection_overrides_fail_closed(
    tmp_path,
    monkeypatch,
    game_count,
    seeds,
    message,
):
    _use_temporary_repo(
        tmp_path,
        monkeypatch,
    )
    config_path = _write_config(
        tmp_path,
    )
    with pytest.raises(
        ValueError,
        match=message,
    ):
        pipeline.run_pipeline_stage(
            config_path=config_path,
            run_id="test_run",
            stage="validate",
            game_count=game_count,
            seeds=seeds,
        )


@pytest.mark.parametrize(
    "run_id",
    (
        "",
        "..",
        "../escape",
        "/absolute",
        "contains space",
        "非ascii",
    ),
)
def test_invalid_run_id_fails_closed(
    tmp_path,
    run_id,
):
    config_path = _write_config(
        tmp_path,
    )
    with pytest.raises(
        ValueError,
        match="run_id",
    ):
        pipeline.run_pipeline_stage(
            config_path=config_path,
            run_id=run_id,
            stage="validate",
        )


@pytest.mark.parametrize(
    "stage",
    ("project", "split", "train", "eval"),
)
def test_collection_overrides_are_rejected_for_later_stages(
    tmp_path,
    stage,
):
    config_path = _write_config(
        tmp_path,
    )
    with pytest.raises(
        ValueError,
        match="only allowed for validate and collect",
    ):
        pipeline.run_pipeline_stage(
            config_path=config_path,
            run_id="test_run",
            stage=stage,
            game_count=1,
            seeds=(4101,),
        )


@pytest.mark.parametrize(
    "seeds",
    [
        (211,),
        (311, 312, 313),
    ],
)
def test_collect_calls_existing_core_sequentially(
    tmp_path,
    monkeypatch,
    suspicion_sample_factory,
    seeds,
):
    _use_temporary_repo(
        tmp_path,
        monkeypatch,
    )
    config_path = _write_config(
        tmp_path,
        seeds=seeds,
    )
    calls = _install_fake_collection(
        monkeypatch,
        suspicion_sample_factory,
        games_dir=(
            tmp_path
            / "logs"
            / "tom"
            / "test_run"
            / "games"
        ),
    )
    monkeypatch.setenv(
        "FAKE_API_KEY",
        "test-secret",
    )

    summary = pipeline.run_pipeline_stage(
        config_path=config_path,
        run_id="test_run",
        stage="collect",
    )

    assert [
        seed
        for _game_id, seed, _budget_id
        in calls
    ] == list(seeds)
    assert all(
        game_id == budget_id
        for game_id, _seed, budget_id
        in calls
    )
    assert summary["game_count"] == len(
        seeds
    )
    assert summary["seeds"] == list(seeds)
    raw_path = Path(
        summary["output_path"]
    )
    assert raw_path == (
        tmp_path
        / "data"
        / "tom"
        / "test_run"
        / "raw.jsonl"
    )
    records = [
        json.loads(line)
        for line in raw_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(records) == len(seeds)
    assert {
        record["schema_version"]
        for record in records
    } == {
        SAMPLE_SCHEMA_VERSION,
    }
    assert {
        record["label_prompt_version"]
        for record in records
    } == {
        LABEL_PROMPT_VERSION,
    }
    assert all(
        "pair_targets" not in record
        for record in records
    )
    assert "test-secret" not in (
        json.dumps(summary)
    )
    logs_run = (
        tmp_path / "logs" / "tom" / "test_run"
    )
    assert (logs_run / "call_audit.jsonl").is_file()
    assert (logs_run / "manifest.json").is_file()
    assert (logs_run / "resolved_config.yaml").is_file()
    assert (logs_run / "games").is_dir()


@pytest.mark.parametrize(
    "seeds",
    (
        (4101,),
        (4101, 4101),
    ),
)
def test_collect_override_preserves_seed_order_and_yaml(
    tmp_path,
    monkeypatch,
    suspicion_sample_factory,
    seeds,
):
    _use_temporary_repo(
        tmp_path,
        monkeypatch,
    )
    config_path = _write_config(
        tmp_path,
    )
    original_yaml = config_path.read_bytes()
    games_dir = (
        tmp_path
        / "logs"
        / "tom"
        / "test_run"
        / "games"
    )
    calls = _install_fake_collection(
        monkeypatch,
        suspicion_sample_factory,
        games_dir=games_dir,
    )
    monkeypatch.setenv(
        "FAKE_API_KEY",
        "test-secret",
    )

    summary = pipeline.run_pipeline_stage(
        config_path=config_path,
        run_id="test_run",
        stage="collect",
        game_count=len(seeds),
        seeds=seeds,
    )

    assert [
        seed
        for _game_id, seed, _budget_id
        in calls
    ] == list(seeds)
    assert summary["game_count"] == len(seeds)
    assert summary["seeds"] == list(seeds)
    assert config_path.read_bytes() == original_yaml
    manifest = json.loads(
        (
            tmp_path
            / "logs"
            / "tom"
            / "test_run"
            / "manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    assert manifest["run_id"] == "test_run"
    assert manifest["game_count"] == len(seeds)
    assert manifest["seeds"] == list(seeds)
    assert manifest["config_path"] == str(
        config_path.resolve()
    )
    assert manifest["source_commit"]
    assert manifest["output_path"] == str(
        tmp_path
        / "data"
        / "tom"
        / "test_run"
        / "raw.jsonl"
    )
    expected_run_dirs = {
        "logs": (
            tmp_path
            / "logs"
            / "tom"
            / "test_run"
        ),
        "data": (
            tmp_path
            / "data"
            / "tom"
            / "test_run"
        ),
        "outputs": (
            tmp_path
            / "outputs"
            / "tom"
            / "test_run"
        ),
    }
    assert {
        group: group_paths[
            "run_dir"
        ]
        for group, group_paths
        in manifest["paths"].items()
    } == {
        group: str(path)
        for group, path
        in expected_run_dirs.items()
    }
    resolved_path = (
        tmp_path
        / "logs"
        / "tom"
        / "test_run"
        / "resolved_config.yaml"
    )
    resolved = yaml.safe_load(
        resolved_path.read_text(
            encoding="utf-8"
        )
    )
    assert resolved["pipeline"]["collection"][
        "game_count"
    ] == len(seeds)
    assert resolved["pipeline"]["collection"][
        "seeds"
    ] == list(seeds)
    assert {
        group: group_paths[
            "run_dir"
        ]
        for group, group_paths
        in resolved["pipeline"][
            "resolved_run"
        ]["paths"].items()
    } == {
        group: str(path)
        for group, path
        in expected_run_dirs.items()
    }
    assert "test-secret" not in resolved_path.read_text(
        encoding="utf-8"
    )
    assert len(
        list(games_dir.iterdir())
    ) == len(seeds)


@pytest.mark.parametrize(
    "existing_relative",
    (
        "logs/tom/test_run",
        "data/tom/test_run",
        "outputs/tom/test_run",
    ),
)
def test_collect_refuses_existing_run_id(
    tmp_path,
    monkeypatch,
    existing_relative,
):
    _use_temporary_repo(
        tmp_path,
        monkeypatch,
    )
    config_path = _write_config(
        tmp_path,
    )
    existing = tmp_path / existing_relative
    existing.mkdir(
        parents=True,
    )
    monkeypatch.setenv(
        "FAKE_API_KEY",
        "test-secret",
    )

    with pytest.raises(
        FileExistsError,
        match="run directory already exists",
    ):
        pipeline.run_pipeline_stage(
            config_path=config_path,
            run_id="test_run",
            stage="collect",
        )


def test_collect_requires_api_key_but_validate_does_not(
    tmp_path,
    monkeypatch,
):
    _use_temporary_repo(
        tmp_path,
        monkeypatch,
    )
    config_path = _write_config(
        tmp_path,
    )
    monkeypatch.delenv(
        "FAKE_API_KEY",
        raising=False,
    )
    monkeypatch.setattr(
        pipeline,
        "load_dotenv",
        lambda **_kwargs: False,
    )

    assert pipeline.run_pipeline_stage(
        config_path=config_path,
        run_id="test_run",
        stage="validate",
    )["status"] == "ok"
    with pytest.raises(
        ValueError,
        match="FAKE_API_KEY",
    ):
        pipeline.run_pipeline_stage(
            config_path=config_path,
            run_id="test_run",
            stage="collect",
        )
    assert not (
        tmp_path / "logs"
    ).exists()
    assert not (
        tmp_path / "data"
    ).exists()
    assert not (
        tmp_path / "outputs"
    ).exists()


def test_explicit_staged_synthetic_flow(
    tmp_path,
    monkeypatch,
    suspicion_sample_factory,
):
    _use_temporary_repo(
        tmp_path,
        monkeypatch,
    )
    config_path = _write_config(
        tmp_path,
    )
    original_yaml = config_path.read_bytes()
    calls = _install_fake_collection(
        monkeypatch,
        suspicion_sample_factory,
        games_dir=(
            tmp_path
            / "logs"
            / "tom"
            / "test_run"
            / "games"
        ),
    )
    monkeypatch.setenv(
        "FAKE_API_KEY",
        "test-secret",
    )

    validate_summary = (
        pipeline.run_pipeline_stage(
            config_path=config_path,
            run_id="test_run",
            stage="validate",
        )
    )
    assert validate_summary[
        "status"
    ] == "ok"
    assert not (
        tmp_path / "logs"
    ).exists()

    collect_summary = (
        pipeline.run_pipeline_stage(
            config_path=config_path,
            run_id="test_run",
            stage="collect",
        )
    )
    assert len(calls) == 3
    raw_path = Path(
        collect_summary["output_path"]
    )
    assert raw_path.is_file()
    logs_run = (
        tmp_path / "logs" / "tom" / "test_run"
    )
    data_run = (
        tmp_path / "data" / "tom" / "test_run"
    )
    outputs_run = (
        tmp_path / "outputs" / "tom" / "test_run"
    )
    assert {
        path.name
        for path in logs_run.iterdir()
    } == {
        "games",
        "call_audit.jsonl",
        "manifest.json",
        "resolved_config.yaml",
    }
    assert {
        path.name
        for path in data_run.iterdir()
    } == {
        "raw.jsonl",
    }
    assert not outputs_run.exists()

    monkeypatch.delenv(
        "FAKE_API_KEY",
        raising=False,
    )
    project_summary = (
        pipeline.run_pipeline_stage(
            config_path=config_path,
            run_id="test_run",
            stage="project",
        )
    )
    projected_path = Path(
        project_summary["output_path"]
    )
    assert project_summary[
        "record_count"
    ] == 3
    projected_records = [
        json.loads(line)
        for line in projected_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert {
        record["schema_version"]
        for record in projected_records
    } == {
        PROJECTED_SCHEMA_VERSION,
    }
    assert {
        record["projection_version"]
        for record in projected_records
    } == {
        PROJECTION_VERSION,
    }
    assert {
        path.name
        for path in data_run.iterdir()
    } == {
        "raw.jsonl",
        "projected.jsonl",
    }

    split_summary = (
        pipeline.run_pipeline_stage(
            config_path=config_path,
            run_id="test_run",
            stage="split",
        )
    )
    split_dir = Path(
        split_summary["output_path"]
    )
    manifest = json.loads(
        (
            split_dir
            / "split_manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    assert {
        split_name: manifest[
            "splits"
        ][split_name]["game_count"]
        for split_name in (
            "train",
            "validation",
            "test",
        )
    } == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    assert manifest["split_seed"] == 11
    game_sets = [
        set(
            manifest["splits"][
                split_name
            ]["game_ids"]
        )
        for split_name in (
            "train",
            "validation",
            "test",
        )
    ]
    assert not (
        game_sets[0] & game_sets[1]
        or game_sets[0] & game_sets[2]
        or game_sets[1] & game_sets[2]
    )
    assert {
        path.name
        for path in data_run.iterdir()
    } == {
        "raw.jsonl",
        "projected.jsonl",
        "split",
    }

    train_summary = (
        pipeline.run_pipeline_stage(
            config_path=config_path,
            run_id="test_run",
            stage="train",
        )
    )
    checkpoint_path = Path(
        train_summary["output_path"]
    )
    assert checkpoint_path.is_file()
    checkpoint_before_eval = (
        checkpoint_path.read_bytes()
    )
    assert checkpoint_path == (
        outputs_run / "checkpoint_best.pt"
    )
    assert (
        outputs_run / "training_metrics.json"
    ).is_file()
    assert not (
        outputs_run / "evaluation.json"
    ).exists()

    eval_summary = (
        pipeline.run_pipeline_stage(
            config_path=config_path,
            run_id="test_run",
            stage="eval",
        )
    )
    assert eval_summary["status"] == "ok"
    assert Path(
        eval_summary["output_path"]
    ) == (
        outputs_run / "evaluation.json"
    )
    assert (
        outputs_run / "evaluation.json"
    ).is_file()
    assert (
        checkpoint_path.read_bytes()
        == checkpoint_before_eval
    )

    dataset = (
        TWDToMDataset.from_jsonl(
            split_dir / "test.jsonl"
        )
    )
    batch = next(
        iter(
            DataLoader(
                dataset,
                batch_size=1,
                collate_fn=(
                    collate_twd_tom_samples
                ),
            )
        )
    )
    for private_field in (
        "suspected_werewolves",
        "known_werewolves",
        "known_non_werewolves",
    ):
        assert private_field not in {
            key
            for key in batch
            if key != "metadata"
        }
    checkpoint = load_checkpoint(
        checkpoint_path
    )
    model = build_model_from_checkpoint(
        checkpoint,
        device=torch.device("cpu"),
    )
    output = model(
        batch["subject_ids"],
        batch["action_ids"],
        batch["object_ids"],
        batch["attention_mask"],
        event_type_ids=batch["event_type_ids"],
        phase_ids=batch["phase_ids"],
        day_values=batch["day_values"],
    )
    assert output[
        "pair_logits"
    ].shape == (
        1,
        7,
        21,
    )
    assert output[
        "belief_matrix"
    ].shape == (
        1,
        7,
        7,
    )
    assert torch.allclose(
        output[
            "belief_matrix"
        ].sum(-1),
        torch.full(
            (1, 7),
            2.0,
        ),
        atol=1e-6,
    )
    loss = masked_pair_kl_divergence(
        output["pair_logits"],
        batch["pair_targets"],
        batch["subject_mask"],
    )
    assert torch.isfinite(loss)

    serialized_summaries = json.dumps(
        [
            validate_summary,
            collect_summary,
            project_summary,
            split_summary,
            train_summary,
            eval_summary,
        ]
    )
    assert "test-secret" not in (
        serialized_summaries
    )
    assert config_path.read_bytes() == original_yaml
