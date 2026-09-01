from pathlib import Path

import pytest

import script.twd_tom.run_non_wolf_oof_diagnostic as diagnostic_module
from werewolf.models.twd_tom.annotation_v2 import (
    V1_ANNOTATION_SOURCE,
    V1_EMPTY_UNOBSERVED_BELIEF_SOURCE,
)
from werewolf.models.twd_tom.belief_backbone import (
    NO_PHASE_DAY_INPUT_FEATURE_PROFILE,
    QWEN2_BACKBONE_NAME,
)
from werewolf.models.twd_tom.supervision import NON_WOLF_ALIVE_SCOPE


def test_non_wolf_diagnostic_validates_sidecar_and_hard_locks_condition(
    tmp_path,
    monkeypatch,
):
    fold_root = tmp_path / "folds"
    role_sidecar = tmp_path / "roles.json"
    output_dir = tmp_path / "diagnostic"
    captured = {}

    def fake_validate(**kwargs):
        captured["validation"] = kwargs
        return {"status": "ok"}

    def fake_run(**kwargs):
        captured["run"] = kwargs
        return {"status": "ok"}

    monkeypatch.setattr(
        diagnostic_module,
        "validate_development_role_sidecar",
        fake_validate,
    )
    monkeypatch.setattr(diagnostic_module, "run_diagnostic_oof", fake_run)

    result = diagnostic_module.run_non_wolf_oof_diagnostic(
        fold_root=fold_root,
        role_sidecar_path=role_sidecar,
        output_dir=output_dir,
        epochs=80,
        batch_size=8,
        learning_rate=1e-4,
        min_learning_rate=1e-5,
        warmup_ratio=0.05,
        early_stopping_patience=12,
        early_stopping_min_delta=1e-4,
        seed=42,
        device="cuda",
        bootstrap_samples=2000,
        reference_improvement=0.50,
    )

    resolved_fold_root = fold_root.resolve()
    resolved_sidecar = role_sidecar.resolve()
    assert result == {"status": "ok"}
    assert captured["validation"] == {
        "role_sidecar_path": resolved_sidecar,
        "development_fold_manifest_path": (
            resolved_fold_root / "development_folds_manifest.json"
        ),
    }
    assert captured["run"] == {
        "fold_root": resolved_fold_root,
        "role_sidecar_path": resolved_sidecar,
        "output_dir": output_dir,
        "epochs": 80,
        "batch_size": 8,
        "learning_rate": 1e-4,
        "min_learning_rate": 1e-5,
        "warmup_ratio": 0.05,
        "early_stopping_patience": 12,
        "early_stopping_min_delta": 1e-4,
        "seed": 42,
        "device": "cuda",
        "bootstrap_samples": 2000,
        "reference_improvement": 0.50,
        "private_conditioning": False,
        "backbone": QWEN2_BACKBONE_NAME,
        "input_feature_profile": NO_PHASE_DAY_INPUT_FEATURE_PROFILE,
        "supervision_scope": NON_WOLF_ALIVE_SCOPE,
        "speech_annotation_source": V1_ANNOTATION_SOURCE,
        "belief_annotation_source": V1_EMPTY_UNOBSERVED_BELIEF_SOURCE,
        "speech_v2_annotation_path": None,
        "belief_v2_annotation_path": None,
        "worst_case_limit": 50,
    }


def test_non_wolf_diagnostic_cli_exposes_no_condition_or_private_switches():
    parser = diagnostic_module.build_arg_parser()
    args = parser.parse_args([
        "--fold-root",
        "folds",
        "--role-sidecar",
        "roles.json",
        "--output-dir",
        "output",
    ])
    forbidden_destinations = {
        "supervision_scope",
        "private_conditioning",
        "backbone",
        "input_feature_profile",
        "speech_annotation_source",
        "belief_annotation_source",
        "speech_v2_annotation_path",
        "belief_v2_annotation_path",
    }

    assert forbidden_destinations.isdisjoint(vars(args))
    assert args.epochs == 80
    assert args.batch_size == 8
    assert args.learning_rate == 1e-4
    assert args.min_learning_rate == 1e-5
    assert args.warmup_ratio == 0.05
    assert args.early_stopping_patience == 12
    assert args.early_stopping_min_delta == 1e-4
    assert args.seed == 42
    assert args.bootstrap_samples == 2000


@pytest.mark.parametrize(
    "forbidden_arguments",
    [
        ["--supervision-scope", "all_alive"],
        ["--private-conditioning"],
        ["--speech-v2-annotation-path", "speech.jsonl"],
        ["--belief-v2-annotation-path", "belief.jsonl"],
    ],
)
def test_non_wolf_diagnostic_cli_rejects_forbidden_switches(
    forbidden_arguments,
):
    with pytest.raises(SystemExit):
        diagnostic_module.build_arg_parser().parse_args([
            "--fold-root",
            "folds",
            "--role-sidecar",
            "roles.json",
            "--output-dir",
            "output",
            *forbidden_arguments,
        ])


def test_non_wolf_diagnostic_does_not_run_after_sidecar_validation_failure(
    tmp_path,
    monkeypatch,
):
    def fail_validation(**_):
        raise ValueError("role sidecar lineage mismatch")

    def forbidden_run(**_):
        pytest.fail("diagnostic OOF ran after failed role-sidecar validation")

    monkeypatch.setattr(
        diagnostic_module,
        "validate_development_role_sidecar",
        fail_validation,
    )
    monkeypatch.setattr(
        diagnostic_module,
        "run_diagnostic_oof",
        forbidden_run,
    )

    with pytest.raises(ValueError, match="lineage mismatch"):
        diagnostic_module.run_non_wolf_oof_diagnostic(
            fold_root=tmp_path / "folds",
            role_sidecar_path=tmp_path / "missing_roles.json",
            output_dir=tmp_path / "output",
        )


def test_non_wolf_diagnostic_module_has_one_oof_delegate():
    source = Path(diagnostic_module.__file__).read_text(encoding="utf-8")

    assert "run_diagnostic_oof(" in source
    assert "def _run_development_oof" not in source
    assert "def run_development_oof" not in source
