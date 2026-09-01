from dataclasses import asdict

import pytest

import script.twd_tom.run_development_oof as oof_module
from script.twd_tom.run_development_oof import (
    _bootstrap_game_macro,
    _weighted_metrics,
    build_arg_parser,
    run_development_oof,
)
from werewolf.models.twd_tom.belief_backbone import (
    NO_PHASE_DAY_INPUT_FEATURE_PROFILE,
    QWEN2_BACKBONE_NAME,
)
from werewolf.models.twd_tom.samples import SAMPLE_SCHEMA_VERSION
from werewolf.models.twd_tom.supervision import ALL_ALIVE_SCOPE


def _game_metrics(*, count, model_kl, target_entropy=0.5, uniform_ce=1.8):
    uniform_kl = uniform_ce - target_entropy
    return {
        "valid_observer_count": count,
        "mean_loss": model_kl + target_entropy,
        "mean_belief_cross_entropy": model_kl + target_entropy,
        "mean_belief_target_entropy": target_entropy,
        "mean_belief_kl_divergence": model_kl,
        "uniform_non_self_baseline_mean_cross_entropy": uniform_ce,
        "uniform_non_self_baseline_mean_kl_divergence": uniform_kl,
        "normalized_reducible_gap_improvement": 1.0 - model_kl / uniform_kl,
    }


def _formal_completed_fold_config():
    return asdict(oof_module.TrainingConfig(
        output_dir="/formal/oof/fold_0",
        dataset_path="/formal/folds/fold_0/train.jsonl",
        validation_dataset_path="/formal/folds/fold_0/validation.jsonl",
        epochs=80,
        batch_size=8,
        learning_rate=1e-4,
        lr_scheduler="warmup_cosine",
        min_learning_rate=1e-5,
        input_feature_profile=NO_PHASE_DAY_INPUT_FEATURE_PROFILE,
        dense_supervision=True,
        early_stopping_patience=12,
        early_stopping_min_delta=1e-4,
    ))


def _completed_fold_summary(training_config):
    return {
        "status": "ok",
        "source_schema_version": SAMPLE_SCHEMA_VERSION,
        "run_provenance": {
            "development_fold_name": "fold_0",
            "development_fold_manifest_sha256": "a" * 64,
            "development_fold_manifest_digest": "b" * 64,
        },
        "training_config": training_config,
    }


def _formal_resume_provenance():
    return {
        "development_fold_manifest_sha256": "a" * 64,
        "development_fold_manifest_digest": "b" * 64,
    }


def _diagnostic_resume_provenance():
    return {
        **_formal_resume_provenance(),
        "role_sidecar_sha256": "c" * 64,
        "role_sidecar_digest": "d" * 64,
    }


def test_oof_weighting_recomputes_reducible_gap_from_all_observers():
    by_game = {
        "game_a": _game_metrics(count=3, model_kl=0.4),
        "game_b": _game_metrics(count=1, model_kl=0.8),
    }

    result = _weighted_metrics(by_game)

    assert result["valid_observer_count"] == 4
    assert result["mean_belief_kl_divergence"] == pytest.approx(0.5)
    assert result["uniform_non_self_baseline_mean_kl_divergence"] == pytest.approx(
        1.3
    )
    assert result["normalized_reducible_gap_improvement"] == pytest.approx(
        1.0 - 0.5 / 1.3
    )


def test_oof_weighting_excludes_explicit_unscored_game():
    scored = _game_metrics(count=3, model_kl=0.4)
    result = _weighted_metrics({
        "scored_game": scored,
        "unscored_game": {
            "status": "unscored_no_supervised_observers",
            "total_row_count": 0,
            "valid_observer_count": 0,
            "scope_observer_count": 4,
            "observed_label_row_count_in_scope": 0,
            "unobserved_label_row_count_in_scope": 4,
        },
    })

    assert result == _weighted_metrics({"scored_game": scored})


def test_oof_gap_closed_uses_additive_kl_sums_and_preserves_row_counts():
    by_game = {
        "game_a": {
            **_game_metrics(count=3, model_kl=0.4),
            "total_row_count": 3,
            "positive_uniform_baseline_gap_row_count": 2,
            "zero_uniform_baseline_gap_row_count": 1,
            "model_kl_sum": 1.2,
            "uniform_non_self_baseline_kl_sum": 2.0,
        },
        "game_b": {
            **_game_metrics(count=1, model_kl=0.8),
            "total_row_count": 1,
            "positive_uniform_baseline_gap_row_count": 1,
            "zero_uniform_baseline_gap_row_count": 0,
            "model_kl_sum": 0.8,
            "uniform_non_self_baseline_kl_sum": 1.0,
        },
    }

    result = _weighted_metrics(by_game)

    assert result["total_row_count"] == 4
    assert result["positive_uniform_baseline_gap_row_count"] == 3
    assert result["zero_uniform_baseline_gap_row_count"] == 1
    assert result["model_kl_sum"] == pytest.approx(2.0)
    assert result["uniform_non_self_baseline_kl_sum"] == pytest.approx(3.0)
    assert result["normalized_reducible_gap_improvement"] == pytest.approx(
        1.0 - 2.0 / 3.0
    )


def test_game_bootstrap_is_deterministic_and_uses_games_as_units():
    by_game = {
        "game_a": _game_metrics(count=100, model_kl=0.4),
        "game_b": _game_metrics(count=1, model_kl=0.8),
    }

    first = _bootstrap_game_macro(
        by_game,
        metric_name="normalized_reducible_gap_improvement",
        samples=100,
        seed=42,
    )
    second = _bootstrap_game_macro(
        by_game,
        metric_name="normalized_reducible_gap_improvement",
        samples=100,
        seed=42,
    )

    assert first == second
    assert first["unit"] == "game"
    assert first["game_count"] == 2
    assert first["ci95_lower"] <= first["point_estimate"] <= first["ci95_upper"]


def test_oof_weighting_recomputes_private_admissible_reducible_gap():
    by_game = {
        "game_a": {
            **_game_metrics(count=3, model_kl=0.4),
            "private_admissible_uniform_baseline_mean_cross_entropy": 1.5,
        },
        "game_b": {
            **_game_metrics(count=1, model_kl=0.8),
            "private_admissible_uniform_baseline_mean_cross_entropy": 1.5,
        },
    }

    result = _weighted_metrics(by_game)

    assert result[
        "private_admissible_uniform_baseline_mean_kl_divergence"
    ] == pytest.approx(1.0)
    assert result[
        "private_admissible_normalized_reducible_gap_improvement"
    ] == pytest.approx(0.5)


def test_formal_oof_cli_exposes_no_role_or_experiment_switches():
    args = build_arg_parser().parse_args([
        "--fold-root",
        "folds",
        "--output-dir",
        "output",
    ])

    assert "role_sidecar" not in vars(args)
    assert "supervision_scope" not in vars(args)
    assert "private_conditioning" not in vars(args)
    assert "speech_annotation_source" not in vars(args)
    assert "belief_annotation_source" not in vars(args)


def test_formal_oof_fixes_public_all_alive_contract_without_role_sidecar(
    tmp_path,
    monkeypatch,
):
    fold_root = tmp_path / "folds"
    (fold_root / "fold_0").mkdir(parents=True)
    manifest = {
        "folds": {"fold_0": {"fold_index": 0}},
        "development_game_ids": ["game_a"],
        "sealed_test_game_ids": ["game_test"],
        "source_split_manifest_digest": "1" * 64,
        "manifest_digest": "2" * 64,
    }
    captured = {}
    game_metrics = {
        **_game_metrics(count=3, model_kl=0.4),
        "private_admissible_uniform_baseline_mean_cross_entropy": 1.5,
        "private_admissible_uniform_baseline_mean_kl_divergence": 1.0,
        "private_admissible_normalized_reducible_gap_improvement": 0.6,
    }

    monkeypatch.setattr(oof_module, "_load_json", lambda _: manifest)
    monkeypatch.setattr(oof_module, "sha256_file", lambda _: "a" * 64)
    monkeypatch.setattr(
        oof_module,
        "validate_development_fold_paths",
        lambda *_: None,
    )

    def fake_run_training(config):
        captured["config"] = config
        return {
            "status": "ok",
            "source_schema_version": SAMPLE_SCHEMA_VERSION,
            "run_provenance": {"development_fold_name": "fold_0"},
            "training_config": asdict(config),
            "best_epoch": 1,
            "epochs_completed": 1,
            "best_validation_mean_loss": 1.0,
            "best_validation_by_game": {"game_a": game_metrics},
            "best_validation_stratified_by_game": {
                "game_a": {
                    "game_id": {"game_a": game_metrics},
                }
            },
            "validation_baselines": {},
        }

    monkeypatch.setattr(oof_module, "run_training", fake_run_training)
    monkeypatch.setattr(
        oof_module,
        "export_belief_worst_cases",
        lambda **kwargs: {
            "output_jsonl": str(kwargs["output_jsonl"]),
            "output_csv": str(kwargs["output_csv"]),
        },
    )
    monkeypatch.setattr(
        oof_module,
        "aggregate_worst_case_exports",
        lambda **kwargs: {
            "output_jsonl": str(kwargs["output_jsonl"]),
            "output_csv": str(kwargs["output_csv"]),
        },
    )

    result = run_development_oof(
        fold_root=fold_root,
        output_dir=tmp_path / "output",
        bootstrap_samples=10,
    )

    assert captured["config"].backbone == QWEN2_BACKBONE_NAME
    assert captured["config"].private_conditioning is False
    assert captured["config"].role_sidecar_path is None
    assert captured["config"].supervision_scope == ALL_ALIVE_SCOPE
    assert (
        captured["config"].input_feature_profile
        == NO_PHASE_DAY_INPUT_FEATURE_PROFILE
    )
    assert result["training_config"]["backbone"] == QWEN2_BACKBONE_NAME
    assert (
        result["training_config"]["input_feature_profile"]
        == NO_PHASE_DAY_INPUT_FEATURE_PROFILE
    )
    assert result["training_config"]["supervision_scope"] == ALL_ALIVE_SCOPE
    assert "role_sidecar_path" not in result["training_config"]
    assert result["descriptive_reference_target"]["is_acceptance_gate"] is False
    assert result["oof_scored_game_count"] == 1
    assert result["oof_unscored_game_count"] == 0
    assert result["oof_unscored_game_ids"] == []

    with pytest.raises(ValueError, match="wrong source schema"):
        oof_module._validate_completed_fold_summary(
            {
                "status": "ok",
                "run_provenance": {"development_fold_name": "fold_0"},
                "training_config": {},
            },
            fold_name="fold_0",
            requested={},
            expected_run_provenance=_formal_resume_provenance(),
        )


def test_formal_completed_fold_exact_config_can_resume():
    requested = _formal_completed_fold_config()

    oof_module._validate_completed_fold_summary(
        _completed_fold_summary(dict(requested)),
        fold_name="fold_0",
        requested=requested,
        expected_run_provenance=_formal_resume_provenance(),
    )


def test_diagnostic_completed_fold_exact_provenance_can_resume():
    requested = _formal_completed_fold_config()
    requested["role_sidecar_path"] = "/diagnostic/roles.json"
    summary = _completed_fold_summary(dict(requested))
    summary["run_provenance"].update(_diagnostic_resume_provenance())

    oof_module._validate_completed_fold_summary(
        summary,
        fold_name="fold_0",
        requested=requested,
        expected_run_provenance=_diagnostic_resume_provenance(),
    )


@pytest.mark.parametrize(
    "field_name",
    [
        "role_sidecar_sha256",
        "role_sidecar_digest",
        "development_fold_manifest_sha256",
        "development_fold_manifest_digest",
    ],
)
def test_diagnostic_completed_fold_rejects_changed_provenance(field_name):
    requested = _formal_completed_fold_config()
    requested["role_sidecar_path"] = "/diagnostic/roles.json"
    summary = _completed_fold_summary(dict(requested))
    summary["run_provenance"].update(_diagnostic_resume_provenance())
    summary["run_provenance"][field_name] = "e" * 64

    with pytest.raises(ValueError, match=field_name):
        oof_module._validate_completed_fold_summary(
            summary,
            fold_name="fold_0",
            requested=requested,
            expected_run_provenance=_diagnostic_resume_provenance(),
        )


def test_diagnostic_completed_fold_rejects_missing_provenance():
    requested = _formal_completed_fold_config()
    requested["role_sidecar_path"] = "/diagnostic/roles.json"
    summary = _completed_fold_summary(dict(requested))
    summary["run_provenance"].update(_diagnostic_resume_provenance())
    summary["run_provenance"].pop("role_sidecar_digest")

    with pytest.raises(ValueError, match="role_sidecar_digest"):
        oof_module._validate_completed_fold_summary(
            summary,
            fold_name="fold_0",
            requested=requested,
            expected_run_provenance=_diagnostic_resume_provenance(),
        )


@pytest.mark.parametrize(
    ("field_name", "diagnostic_value"),
    [
        ("private_conditioning", True),
        ("role_sidecar_path", "/diagnostic/roles.json"),
        ("speech_v2_annotation_path", "/diagnostic/speech_v2.jsonl"),
        ("belief_v2_annotation_path", "/diagnostic/belief_v2.jsonl"),
    ],
)
def test_formal_completed_fold_rejects_diagnostic_config(
    field_name,
    diagnostic_value,
):
    requested = _formal_completed_fold_config()
    existing = dict(requested)
    existing[field_name] = diagnostic_value

    with pytest.raises(ValueError, match=field_name):
        oof_module._validate_completed_fold_summary(
            _completed_fold_summary(existing),
            fold_name="fold_0",
            requested=requested,
            expected_run_provenance=_formal_resume_provenance(),
        )


def test_formal_completed_fold_rejects_unexpected_config_field():
    requested = _formal_completed_fold_config()
    existing = dict(requested)
    existing["diagnostic_only_extra"] = True

    with pytest.raises(ValueError, match="unexpected=.*diagnostic_only_extra"):
        oof_module._validate_completed_fold_summary(
            _completed_fold_summary(existing),
            fold_name="fold_0",
            requested=requested,
            expected_run_provenance=_formal_resume_provenance(),
        )
