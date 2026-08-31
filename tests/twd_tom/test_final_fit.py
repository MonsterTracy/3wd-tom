import json

import pytest
import torch

import script.twd_tom.run_final_fit as final_fit_module
from script.twd_tom.materialize_canonical_belief_dataset import (
    materialize_canonical_belief_dataset,
)
from script.twd_tom.materialize_development_folds import (
    materialize_development_folds,
)
from script.twd_tom.run_final_fit import (
    FINAL_CHECKPOINT_FILENAME,
    FINAL_PROTOCOL_FILENAME,
    FinalFitConfig,
    _validate_oof_summary,
    build_arg_parser,
    run_final_fit,
)
from script.twd_tom.run_development_oof import OOF_SUMMARY_SCHEMA_VERSION
from script.twd_tom.train import TrainingConfig, build_learning_rate_scheduler
from werewolf.models.twd_tom.belief_backbone import ToMBeliefBackboneConfig
from werewolf.trajectory import canonical_digest, canonical_json


SYNTHETIC_FOLD_BEST_EPOCHS = (38, 30, 42, 21, 26)


def _oof_summary(manifest):
    fold_names = sorted(
        manifest["folds"],
        key=lambda name: manifest["folds"][name]["fold_index"],
    )
    return {
        "schema_version": OOF_SUMMARY_SCHEMA_VERSION,
        "status": "ok",
        "evaluation_scope": "development_oof_only",
        "test_evaluated": False,
        "source_split_manifest_digest": manifest[
            "source_split_manifest_digest"
        ],
        "development_fold_manifest_digest": manifest["manifest_digest"],
        "sealed_test_game_count": 6,
        "training_config": {
            "epochs": 80,
            "batch_size": 8,
            "learning_rate": 1e-4,
            "lr_scheduler": "warmup_cosine",
            "warmup_ratio": 0.05,
            "min_learning_rate": 1e-5,
            "seed": 42,
            "max_seq_len": 256,
            "backbone": "qwen2_model",
            "input_feature_profile": "no_phase_day",
            "dense_supervision": True,
            "supervision_scope": "all_alive",
            "speech_annotation_source": "v1",
            "belief_annotation_source": "v1_empty_unobserved",
        },
        "folds": {
            fold_name: {"best_epoch": best_epoch}
            for fold_name, best_epoch in zip(
                fold_names,
                SYNTHETIC_FOLD_BEST_EPOCHS,
                strict=True,
            )
        },
        "oof_game_count": 54,
    }


def test_final_fit_cli_exposes_paths_and_runtime_only():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--fold-root",
        "folds",
        "--oof-summary",
        "oof.json",
        "--output-dir",
        "output",
        "--device",
        "cuda",
    ])

    assert vars(args) == {
        "fold_root": "folds",
        "oof_summary": "oof.json",
        "output_dir": "output",
        "device": "cuda",
    }
    config = FinalFitConfig(
        fold_root="folds",
        oof_summary_path="oof.json",
        output_dir="output",
    )
    assert config.epochs == 0
    assert config.scheduler_horizon_epochs == 80
    assert config.backbone == "qwen2_model"
    assert config.input_feature_profile == "no_phase_day"
    assert config.supervision_scope == "all_alive"
    assert config.speech_annotation_source == "v1"
    assert config.belief_annotation_source == "v1_empty_unobserved"


def test_final_fit_is_disabled_until_new_all_alive_oof_epochs_are_frozen(
    tmp_path,
):
    config = FinalFitConfig(
        fold_root="folds",
        oof_summary_path="oof.json",
        output_dir="output",
    )

    with pytest.raises(RuntimeError, match="all-alive final-fit epoch selection"):
        run_final_fit(config, repo_root=tmp_path)


def test_final_fit_scheduler_prefix_matches_an_80_epoch_schedule(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(FinalFitConfig, "epochs", 30)
    model_a = torch.nn.Linear(1, 1)
    model_b = torch.nn.Linear(1, 1)
    optimizer_a = torch.optim.AdamW(model_a.parameters(), lr=1e-4)
    optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=1e-4)
    final_config = FinalFitConfig(
        fold_root="folds",
        oof_summary_path="oof.json",
        output_dir="output",
    )
    oof_config = TrainingConfig(
        output_dir=str(tmp_path / "oof"),
        dataset_path=str(tmp_path / "train.jsonl"),
        validation_dataset_path=str(tmp_path / "validation.jsonl"),
        epochs=80,
        batch_size=8,
        learning_rate=1e-4,
        lr_scheduler="warmup_cosine",
        warmup_ratio=0.05,
        min_learning_rate=1e-5,
    )
    final_scheduler, final_schedule = build_learning_rate_scheduler(
        optimizer_a,
        config=final_config,  # type: ignore[arg-type]
        steps_per_epoch=7,
        scheduler_horizon_epochs=80,
    )
    oof_scheduler, oof_schedule = build_learning_rate_scheduler(
        optimizer_b,
        config=oof_config,
        steps_per_epoch=7,
    )
    assert final_schedule == oof_schedule
    assert final_schedule["total_steps"] == 80 * 7
    assert final_schedule["warmup_steps"] == 28
    final_lrs = []
    oof_lrs = []
    for _ in range(30 * 7):
        optimizer_a.step()
        final_scheduler.step()
        final_lrs.append(optimizer_a.param_groups[0]["lr"])
        optimizer_b.step()
        oof_scheduler.step()
        oof_lrs.append(optimizer_b.param_groups[0]["lr"])
    assert final_lrs == pytest.approx(oof_lrs, rel=0.0, abs=0.0)


def test_oof_validation_rejects_a_changed_fold_epoch(monkeypatch):
    monkeypatch.setattr(
        final_fit_module,
        "EXPECTED_FOLD_BEST_EPOCHS",
        SYNTHETIC_FOLD_BEST_EPOCHS,
    )
    monkeypatch.setattr(FinalFitConfig, "epochs", 30)
    manifest = {
        "source_split_manifest_digest": "1" * 64,
        "manifest_digest": "2" * 64,
        "folds": {
            f"fold_{index}": {"fold_index": index}
            for index in range(5)
        },
    }
    summary = _oof_summary(manifest)
    config = FinalFitConfig(
        fold_root="folds",
        oof_summary_path="oof.json",
        output_dir="output",
    )
    records = _validate_oof_summary(summary, fold_manifest=manifest, config=config)
    assert [record["best_epoch"] for record in records] == list(
        SYNTHETIC_FOLD_BEST_EPOCHS
    )

    changed = json.loads(json.dumps(summary))
    changed["folds"]["fold_2"]["best_epoch"] = 41
    with pytest.raises(ValueError, match="best epochs"):
        _validate_oof_summary(changed, fold_manifest=manifest, config=config)


class _TinyFinalModel(torch.nn.Linear):
    def __init__(self):
        super().__init__(1, 1)
        self.backbone_name = "qwen2_model"
        self.config = ToMBeliefBackboneConfig(
            max_seq_len=256,
            private_conditioning=False,
            input_feature_profile="no_phase_day",
        )


def test_final_fit_uses_all_54_development_games_and_one_checkpoint(
    tmp_path,
    monkeypatch,
    suspicion_sample_factory,
    canonical_belief_batch_factory,
):
    canonical_root = tmp_path / "canonical"
    samples = {
        f"game_{index:03d}": [
            suspicion_sample_factory(game_id=f"game_{index:03d}")
        ]
        for index in range(60)
    }
    canonical_belief_batch_factory(canonical_root, samples)
    split_root = tmp_path / "datasets" / "split"
    materialize_canonical_belief_dataset(
        canonical_root=canonical_root,
        output_dir=split_root,
        split_seed=42,
        train_game_count=48,
        validation_game_count=6,
        test_game_count=6,
    )
    fold_root = tmp_path / "datasets" / "folds"
    fold_manifest = materialize_development_folds(
        train_path=split_root / "train.jsonl",
        validation_path=split_root / "validation.jsonl",
        output_dir=fold_root,
        fold_count=5,
        fold_seed=42,
    )
    # The final-fit path must remain valid when sealed labels are unavailable.
    (split_root / "test.jsonl").unlink()
    oof_path = tmp_path / "outputs" / "oof" / "oof_summary.json"
    oof_path.parent.mkdir(parents=True)
    oof_path.write_text(
        canonical_json(_oof_summary(fold_manifest)) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "outputs" / "final"

    monkeypatch.setattr(
        final_fit_module,
        "_clean_git_commit",
        lambda _: "c" * 40,
    )
    monkeypatch.setattr(
        final_fit_module,
        "EXPECTED_FOLD_BEST_EPOCHS",
        SYNTHETIC_FOLD_BEST_EPOCHS,
    )
    monkeypatch.setattr(FinalFitConfig, "epochs", 30)
    monkeypatch.setattr(final_fit_module, "_build_model", lambda _: _TinyFinalModel())
    observed_epochs = []

    def fake_train_one_epoch(
        model,
        data_loader,
        optimizer,
        *,
        device,
        gradient_clip_norm,
        lr_scheduler,
    ):
        observed_epochs.append(data_loader.dataset._epoch)
        for _ in range(len(data_loader)):
            optimizer.step()
            lr_scheduler.step()
        return {
            "mean_loss": 1.0,
            "valid_observer_count": 1,
            "learning_rate_start": optimizer.param_groups[0]["lr"],
            "learning_rate_end": optimizer.param_groups[0]["lr"],
        }

    monkeypatch.setattr(final_fit_module, "train_one_epoch", fake_train_one_epoch)
    summary = run_final_fit(
        FinalFitConfig(
            fold_root=str(fold_root),
            oof_summary_path=str(oof_path),
            output_dir=str(output_dir),
            device="cpu",
        ),
        repo_root=tmp_path,
    )

    assert observed_epochs == list(range(1, 31))
    assert summary["epochs_completed"] == 30
    assert summary["development_game_count"] == 54
    assert summary["validation_dataset_used"] is False
    assert summary["early_stopping_enabled"] is False
    assert summary["sealed_test_evaluated"] is False
    assert summary["learning_rate_schedule"]["total_steps"] == 80 * 7
    assert sorted(path.name for path in output_dir.glob("*.pt")) == [
        FINAL_CHECKPOINT_FILENAME
    ]
    protocol = json.loads(
        (output_dir / FINAL_PROTOCOL_FILENAME).read_text(encoding="utf-8")
    )
    protocol_digest = protocol.pop("protocol_digest")
    assert protocol_digest == canonical_digest(protocol)
    assert protocol["epoch_selection"]["fold_best_epochs"] == list(
        SYNTHETIC_FOLD_BEST_EPOCHS
    )
    assert protocol["data_lineage"]["sealed_test_dataset_opened"] is False
    checkpoint = torch.load(
        output_dir / FINAL_CHECKPOINT_FILENAME,
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["epoch"] == 30
    assert checkpoint["validation_dataset_used"] is False
    assert checkpoint["early_stopping_enabled"] is False
    assert checkpoint["sealed_test_evaluated"] is False
    assert checkpoint["run_provenance"]["final_protocol_digest"] == protocol_digest
    assert "validation_metrics" not in checkpoint
