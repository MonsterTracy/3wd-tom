import pytest

from script.twd_tom.run_development_oof import (
    _bootstrap_game_macro,
    _weighted_metrics,
)


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
