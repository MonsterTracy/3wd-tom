import json

import script.twd_tom.run_annotation_v2_ablation as ablation_module
from script.twd_tom.audit_belief_label_repeatability import (
    REPEATABILITY_SCHEMA_VERSION,
)
from werewolf.models.twd_tom.belief_backbone import (
    NO_PHASE_DAY_INPUT_FEATURE_PROFILE,
)
from werewolf.models.twd_tom.annotation_v2 import (
    V1_EMPTY_UNIFORM_NONSELF_BELIEF_SOURCE,
)


def _audit(path):
    path.write_text(json.dumps({
        "schema_version": REPEATABILITY_SCHEMA_VERSION,
        "status": "PASS",
        "state_count": 30,
        "replicate_count": 3,
    }), encoding="utf-8")


def test_ablation_runs_fixed_2x2_for_nonwolf_and_villager(
    tmp_path,
    monkeypatch,
):
    calls = []

    def fake_oof(**kwargs):
        calls.append(kwargs)
        improvement = (
            0.1 if kwargs["supervision_scope"] == "non_wolf_alive" else 0.2
        )
        return {
            "oof_game_count": 54,
            "oof_scored_game_count": 53,
            "oof_unscored_game_count": 1,
            "oof_unscored_game_ids": ["game_unscored"],
            "oof_observer_weighted_metrics": {
                "normalized_reducible_gap_improvement": improvement,
            },
            "oof_game_macro_metrics": {
                "normalized_reducible_gap_improvement": improvement / 2,
            },
            "oof_game_bootstrap_ci": {"point_estimate": improvement / 2},
        }

    monkeypatch.setattr(ablation_module, "run_diagnostic_oof", fake_oof)
    result = ablation_module.run_annotation_v2_ablation(
        fold_root=tmp_path / "folds",
        output_dir=tmp_path / "output",
        role_sidecar_path=tmp_path / "roles.json",
        speech_v2_annotation_path=tmp_path / "speech.jsonl",
        belief_v2_annotation_path=tmp_path / "belief.jsonl",
    )

    assert len(calls) == 8
    assert {call["speech_annotation_source"] for call in calls} == {"v1", "v2"}
    assert {call["belief_annotation_source"] for call in calls} == {
        V1_EMPTY_UNIFORM_NONSELF_BELIEF_SOURCE,
        "v2",
    }
    assert {call["supervision_scope"] for call in calls} == {
        "non_wolf_alive",
        "villager_alive",
    }
    assert all(
        call["input_feature_profile"] == NO_PHASE_DAY_INPUT_FEATURE_PROFILE
        for call in calls
    )
    assert result["input_feature_profile"] == NO_PHASE_DAY_INPUT_FEATURE_PROFILE
    assert result["benchmark_status"] == "exploratory_diagnostic"
    assert result["repeatability_audit"]["status"] == "not_provided"
    assert result["experiments"][
        "speech_v1_belief_v1_empty_uniform_nonself"
    ]["non_wolf_alive"]["oof_unscored_game_count"] == 1
    assert (tmp_path / "output" / "annotation_v2_ablation_table.md").is_file()


def test_passing_repeatability_is_recorded_as_diagnostic_only(
    tmp_path,
    monkeypatch,
):
    repeatability = tmp_path / "repeatability.json"
    _audit(repeatability)

    def fake_oof(**kwargs):
        return {
            "oof_game_count": 54,
            "oof_scored_game_count": 54,
            "oof_unscored_game_count": 0,
            "oof_unscored_game_ids": [],
            "oof_observer_weighted_metrics": {
                "normalized_reducible_gap_improvement": 0.1,
            },
            "oof_game_macro_metrics": {
                "normalized_reducible_gap_improvement": 0.05,
            },
            "oof_game_bootstrap_ci": {"point_estimate": 0.05},
        }

    monkeypatch.setattr(ablation_module, "run_diagnostic_oof", fake_oof)
    result = ablation_module.run_annotation_v2_ablation(
        fold_root=tmp_path / "folds",
        output_dir=tmp_path / "output",
        role_sidecar_path=tmp_path / "roles.json",
        speech_v2_annotation_path=tmp_path / "speech.jsonl",
        belief_v2_annotation_path=tmp_path / "belief.jsonl",
        repeatability_audit_path=repeatability,
    )

    assert result["benchmark_status"] == "exploratory_diagnostic"
    assert result["repeatability_audit"]["status"] == "verified"
    assert result["repeatability_audit"]["purpose"] == "diagnostic_ceiling_only"
