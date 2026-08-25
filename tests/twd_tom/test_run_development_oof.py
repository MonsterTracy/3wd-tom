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
    GPT2_BLOCK_BACKBONE_NAME,
    NO_DAY_INPUT_FEATURE_PROFILE,
)
from werewolf.models.twd_tom.supervision import VILLAGER_ALIVE_SCOPE


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


def test_oof_cli_accepts_backbone_and_input_feature_profile():
    args = build_arg_parser().parse_args([
        "--fold-root",
        "folds",
        "--output-dir",
        "output",
        "--role-sidecar",
        "roles.json",
        "--backbone",
        GPT2_BLOCK_BACKBONE_NAME,
        "--input-feature-profile",
        NO_DAY_INPUT_FEATURE_PROFILE,
    ])

    assert args.backbone == GPT2_BLOCK_BACKBONE_NAME
    assert args.input_feature_profile == NO_DAY_INPUT_FEATURE_PROFILE


def test_oof_cli_accepts_supervision_sidecar_and_scope():
    args = build_arg_parser().parse_args([
        "--fold-root",
        "folds",
        "--output-dir",
        "output",
        "--role-sidecar",
        "roles.json",
        "--supervision-scope",
        VILLAGER_ALIVE_SCOPE,
    ])

    assert args.role_sidecar == "roles.json"
    assert args.supervision_scope == VILLAGER_ALIVE_SCOPE


def test_oof_threads_backbone_and_input_profile_into_every_fold(
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
    monkeypatch.setattr(
        oof_module,
        "validate_development_fold_paths",
        lambda *_: None,
    )

    def fake_run_training(config):
        captured["config"] = config
        return {
            "status": "ok",
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
        private_conditioning=True,
        backbone=GPT2_BLOCK_BACKBONE_NAME,
        input_feature_profile=NO_DAY_INPUT_FEATURE_PROFILE,
        role_sidecar_path=tmp_path / "roles.json",
        bootstrap_samples=10,
    )

    assert captured["config"].backbone == GPT2_BLOCK_BACKBONE_NAME
    assert (
        captured["config"].input_feature_profile
        == NO_DAY_INPUT_FEATURE_PROFILE
    )
    assert result["training_config"]["backbone"] == GPT2_BLOCK_BACKBONE_NAME
    assert (
        result["training_config"]["input_feature_profile"]
        == NO_DAY_INPUT_FEATURE_PROFILE
    )
    assert result["descriptive_reference_target"]["is_acceptance_gate"] is False
