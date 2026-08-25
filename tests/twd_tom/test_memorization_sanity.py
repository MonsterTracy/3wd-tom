import json

import pytest

from script.twd_tom.run_memorization_sanity import (
    _sanity_metrics,
    _selected_game_samples,
    build_arg_parser,
)


def test_memorization_selection_is_game_level_and_deterministic(
    tmp_path,
    training_sample_factory,
):
    path = tmp_path / "train.jsonl"
    samples = [
        training_sample_factory(game_id="game_b"),
        training_sample_factory(game_id="game_a"),
        training_sample_factory(game_id="game_b"),
    ]
    path.write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples),
        encoding="utf-8",
    )

    selected, game_ids = _selected_game_samples(path, game_count=1)

    assert game_ids == ["game_a"]
    assert {sample["game_id"] for sample in selected} == {"game_a"}


def test_memorization_report_exposes_direct_pipeline_sanity_metrics():
    result = _sanity_metrics({
        "mean_belief_cross_entropy": 0.75,
        "mean_belief_target_entropy": 0.5,
        "mean_belief_kl_divergence": 0.25,
        "mean_belief_max_probability_error": 0.1,
        "max_belief_probability_error": 0.15,
        "mean_belief_total_variation": 0.2,
        "normalized_reducible_gap_improvement": 0.9,
    })

    assert result == {
        "kl": pytest.approx(0.25),
        "cross_entropy_minus_target_entropy": pytest.approx(0.25),
        "max_probability_error": pytest.approx(0.15),
        "mean_total_variation": pytest.approx(0.2),
        "gap_closed": pytest.approx(0.9),
    }


def test_memorization_cli_freezes_train_equals_eval_controls():
    args = build_arg_parser().parse_args([
        "--dataset",
        "fold_0/train.jsonl",
        "--output-dir",
        "outputs/m1",
        "--game-count",
        "1",
        "--role-sidecar",
        "roles.json",
    ])

    assert args.game_count == 1
    assert args.keep_dropout is False
    assert args.kl_threshold == pytest.approx(1e-3)
