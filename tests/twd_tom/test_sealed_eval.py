import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import script.twd_tom.run_sealed_eval as sealed_module
from script.twd_tom.run_sealed_eval import (
    SEALED_PER_GAME_FILENAME,
    SEALED_PROTOCOL_FILENAME,
    SEALED_PROVENANCE_FILENAME,
    SEALED_SUMMARY_FILENAME,
    SealedEvalConfig,
    SealedPreflight,
    _metric_record,
    _validate_checkpoint,
    _validate_final_protocol,
    build_arg_parser,
    preflight_sealed_evaluation,
    run_sealed_evaluation,
    summarize_metric_records,
)
from werewolf.models.twd_tom.dense_dataset import DenseTWDToMDataset
from werewolf.trajectory import canonical_digest, canonical_json


def _valid_checkpoint():
    return {
        "checkpoint_type": sealed_module.FINAL_CHECKPOINT_TYPE,
        "schema_version": sealed_module.SAMPLE_SCHEMA_VERSION,
        "backbone": "qwen2_model",
        "speech_annotation_source": "v1",
        "belief_annotation_source": "v1_empty_unobserved",
        "supervision_scope": "all_alive",
        "epoch": sealed_module.FROZEN_EPOCH,
        "validation_dataset_used": False,
        "early_stopping_enabled": False,
        "sealed_test_evaluated": False,
        "model_config": {
            "input_feature_profile": "no_phase_day",
            "private_conditioning": False,
        },
        "training_config": {
            "backbone": "qwen2_model",
            "input_feature_profile": "no_phase_day",
            "speech_annotation_source": "v1",
            "belief_annotation_source": "v1_empty_unobserved",
            "supervision_scope": "all_alive",
            "fit_epochs": sealed_module.FROZEN_EPOCH,
            "seed": 42,
            "validation_dataset_used": False,
            "early_stopping_enabled": False,
        },
        "run_provenance": {
            "final_protocol_digest": sealed_module.FROZEN_FINAL_PROTOCOL_DIGEST,
            "git_commit_sha": sealed_module.FROZEN_CHECKPOINT_GIT_COMMIT,
            "sealed_test_dataset_opened": False,
            "sealed_test_labels_used": False,
            "sealed_test_evaluated": False,
        },
        "model_state_dict": {"weight": torch.tensor([1.0])},
    }


def _valid_manifest():
    train = [f"train_{index:02d}" for index in range(48)]
    validation = [f"validation_{index:02d}" for index in range(6)]
    test = [f"sealed_{index:02d}" for index in range(6)]
    return {
        "manifest_digest": "a" * 64,
        "canonical_batch_summary_digest": "b" * 64,
        "game_ids": {"train": train, "validation": validation, "test": test},
        "output_files": {
            "test": {"relative_path": "test.jsonl", "sha256": "c" * 64}
        },
    }


def _protocol_payload(manifest, manifest_sha):
    return {
        "schema_version": sealed_module.FINAL_PROTOCOL_SCHEMA_VERSION,
        "status": "frozen_before_fit",
        "git_commit_sha": sealed_module.FROZEN_CHECKPOINT_GIT_COMMIT,
        "data_lineage": {
            "source_split_manifest_sha256": manifest_sha,
            "source_split_manifest_digest": manifest["manifest_digest"],
            "canonical_batch_summary_digest": manifest[
                "canonical_batch_summary_digest"
            ],
            "development_game_ids_digest": canonical_digest(
                sorted(
                    manifest["game_ids"]["train"]
                    + manifest["game_ids"]["validation"]
                )
            ),
            "sealed_test_game_count": 6,
            "sealed_test_dataset_opened": False,
            "sealed_test_labels_used": False,
            "sealed_test_evaluated": False,
        },
        "checkpoint_policy": {
            "validation_dataset_used": False,
            "early_stopping_enabled": False,
        },
    }


def _write_protocol(path, payload):
    protocol = {**payload, "protocol_digest": canonical_digest(payload)}
    path.write_text(canonical_json(protocol) + "\n", encoding="utf-8")
    return protocol


def _preflight_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(sealed_module, "FROZEN_CHECKPOINT_GIT_COMMIT", "f" * 40)
    monkeypatch.setattr(sealed_module, "FROZEN_EPOCH", 30)
    checkpoint_path = tmp_path / "final.pt"
    checkpoint_path.write_bytes(b"synthetic checkpoint")
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text("synthetic manifest\n", encoding="utf-8")
    manifest = _valid_manifest()
    protocol_path = tmp_path / "final_protocol.json"
    payload = _protocol_payload(
        manifest, sealed_module.sha256_file(manifest_path)
    )
    protocol = _write_protocol(protocol_path, payload)
    monkeypatch.setattr(
        sealed_module, "FROZEN_CHECKPOINT_SHA256", sealed_module.sha256_file(checkpoint_path)
    )
    monkeypatch.setattr(
        sealed_module, "FROZEN_FINAL_PROTOCOL_DIGEST", protocol["protocol_digest"]
    )
    monkeypatch.setattr(sealed_module, "load_checkpoint", lambda _: {})
    model = SimpleNamespace(config=SimpleNamespace(max_seq_len=256))
    monkeypatch.setattr(sealed_module, "_validate_checkpoint", lambda _: model)
    monkeypatch.setattr(
        sealed_module,
        "validate_split_manifest",
        lambda path, verify_split_files=(): manifest,
    )
    monkeypatch.setattr(
        sealed_module,
        "_clean_git_commit",
        lambda _: "f" * 40,
    )
    config = SealedEvalConfig(
        checkpoint_path=str(checkpoint_path),
        final_protocol_path=str(protocol_path),
        manifest_path=str(manifest_path),
        output_dir=str(tmp_path / "output"),
        device="cpu",
    )
    return config, manifest, protocol


def test_sealed_cli_exposes_only_frozen_paths_and_device():
    args = build_arg_parser().parse_args([
        "--checkpoint",
        "final.pt",
        "--final-protocol",
        "final_protocol.json",
        "--manifest",
        "split_manifest.json",
        "--output-dir",
        "sealed",
        "--device",
        "cuda",
    ])
    assert vars(args) == {
        "checkpoint": "final.pt",
        "final_protocol": "final_protocol.json",
        "manifest": "split_manifest.json",
        "output_dir": "sealed",
        "device": "cuda",
    }


def test_sealed_preflight_is_disabled_until_new_all_alive_artifacts_are_frozen(
    tmp_path,
):
    config = SealedEvalConfig(
        checkpoint_path=str(tmp_path / "missing.pt"),
        final_protocol_path=str(tmp_path / "missing_protocol.json"),
        manifest_path=str(tmp_path / "missing_manifest.json"),
        output_dir=str(tmp_path / "sealed"),
    )

    with pytest.raises(RuntimeError, match="all-alive sealed bindings"):
        preflight_sealed_evaluation(config, repo_root=tmp_path)


def test_checkpoint_sha_mismatch_aborts_before_checkpoint_load(tmp_path, monkeypatch):
    config, _, _ = _preflight_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(sealed_module, "FROZEN_CHECKPOINT_SHA256", "0" * 64)
    monkeypatch.setattr(
        sealed_module,
        "load_checkpoint",
        lambda _: pytest.fail("checkpoint must not load after hash mismatch"),
    )
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        preflight_sealed_evaluation(config, repo_root=tmp_path)


def test_final_protocol_digest_mismatch_aborts(tmp_path):
    payload = {
        "git_commit_sha": sealed_module.FROZEN_CHECKPOINT_GIT_COMMIT,
        "data_lineage": {},
    }
    protocol = {**payload, "protocol_digest": "0" * 64}
    with pytest.raises(ValueError, match="canonical digest"):
        _validate_final_protocol(protocol, path=tmp_path / "missing.json")


@pytest.mark.parametrize(
    ("field_path", "wrong", "message"),
    [
        (("backbone",), "gpt2_block", "backbone"),
        (("model_config", "input_feature_profile"), "full", "input feature"),
        (("supervision_scope",), "non_wolf_alive", "supervision_scope"),
    ],
)
def test_checkpoint_rejects_wrong_model_profile_or_scope(
    monkeypatch, field_path, wrong, message
):
    checkpoint = _valid_checkpoint()
    target = checkpoint
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = wrong
    monkeypatch.setattr(
        sealed_module,
        "build_model_from_checkpoint",
        lambda *args, **kwargs: object(),
    )
    with pytest.raises(ValueError, match=message):
        _validate_checkpoint(checkpoint)


def test_checkpoint_rejects_prior_sealed_evaluation(monkeypatch):
    checkpoint = _valid_checkpoint()
    checkpoint["sealed_test_evaluated"] = True
    monkeypatch.setattr(
        sealed_module,
        "build_model_from_checkpoint",
        lambda *args, **kwargs: object(),
    )
    with pytest.raises(ValueError, match="sealed_test_evaluated"):
        _validate_checkpoint(checkpoint)


def test_checkpoint_rejects_nonfinite_tensor(monkeypatch):
    checkpoint = _valid_checkpoint()
    checkpoint["model_state_dict"]["weight"] = torch.tensor([float("nan")])
    monkeypatch.setattr(
        sealed_module,
        "build_model_from_checkpoint",
        lambda *args, **kwargs: pytest.fail("invalid tensor must fail first"),
    )
    with pytest.raises(ValueError, match="non-finite"):
        _validate_checkpoint(checkpoint)


def test_dirty_git_aborts_preflight(tmp_path, monkeypatch):
    config, _, _ = _preflight_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        sealed_module,
        "_clean_git_commit",
        lambda _: (_ for _ in ()).throw(RuntimeError("dirty files")),
    )
    with pytest.raises(RuntimeError, match="dirty"):
        preflight_sealed_evaluation(config, repo_root=tmp_path)


@pytest.mark.parametrize("mutation", ["overlap", "count", "duplicate"])
def test_manifest_rejects_overlap_wrong_count_and_duplicates(
    tmp_path, monkeypatch, mutation
):
    config, manifest, _ = _preflight_fixture(tmp_path, monkeypatch)
    changed = json.loads(json.dumps(manifest))
    if mutation == "overlap":
        changed["game_ids"]["test"][0] = changed["game_ids"]["train"][0]
    elif mutation == "count":
        changed["game_ids"]["test"].pop()
    else:
        changed["game_ids"]["test"][1] = changed["game_ids"]["test"][0]
    monkeypatch.setattr(
        sealed_module,
        "validate_split_manifest",
        lambda path, verify_split_files=(): changed,
    )
    message = {
        "overlap": "overlap",
        "count": "exactly 6",
        "duplicate": "unique",
    }[mutation]
    with pytest.raises(ValueError, match=message):
        preflight_sealed_evaluation(config, repo_root=tmp_path)


def test_label_blind_preflight_succeeds_when_sealed_file_is_absent(
    tmp_path, monkeypatch
):
    config, _, _ = _preflight_fixture(tmp_path, monkeypatch)
    assert not (tmp_path / "test.jsonl").exists()
    plan = preflight_sealed_evaluation(config, repo_root=tmp_path)
    assert plan.sealed_dataset_path == tmp_path / "test.jsonl"
    assert not plan.sealed_dataset_path.exists()


def test_existing_checkpoint_marker_rejects_another_preflight(tmp_path, monkeypatch):
    config, _, _ = _preflight_fixture(tmp_path, monkeypatch)
    marker = sealed_module._marker_path(Path(config.checkpoint_path))
    marker.write_text('{"sealed_test_evaluated":true}\n', encoding="utf-8")
    with pytest.raises(FileExistsError, match="permanently locked"):
        preflight_sealed_evaluation(config, repo_root=tmp_path)


def test_formal_all_alive_mask_includes_an_observed_living_wolf(
    suspicion_sample_factory,
):
    sample = suspicion_sample_factory(
        observers=(1, 2, 3, 4, 5, 6, 7),
        suspicions_by_observer={5: []},
    )
    ground_truth_roles = {"player1": "Werewolf", "player2": "Werewolf"}
    dataset = DenseTWDToMDataset(
        [sample],
        supervision_scope="all_alive",
        speech_annotation_source="v1",
        belief_annotation_source="v1_empty_unobserved",
    )
    row = dataset[0]
    assert torch.equal(
        row["observer_supervision_mask"],
        row["observer_alive_mask"] & row["label_observed_mask"],
    )
    assert ground_truth_roles["player1"] == "Werewolf"
    # Formal population construction never receives ground_truth_roles.
    assert bool(row["observer_alive_mask"][0, 0])
    assert bool(row["label_observed_mask"][0, 0])
    assert bool(row["observer_supervision_mask"][0, 0])
    assert torch.equal(row["belief_targets"][0, 4], torch.zeros(7))
    assert not bool(row["label_observed_mask"][0, 4])
    assert not bool(row["observer_supervision_mask"][0, 4])


def test_metric_formula_matches_corrected_oof_synthetic_case():
    record = _metric_record(
        "game_1",
        {
            "valid_observer_count": 4,
            "model_kl_sum": 2.0,
            "uniform_non_self_baseline_kl_sum": 5.0,
            "mean_belief_total_variation": 0.25,
        },
    )
    assert record["model_kl_mean"] == pytest.approx(0.5)
    assert record["uniform_kl_mean"] == pytest.approx(1.25)
    assert record["gap_closed"] == pytest.approx(0.6)


def test_game_macro_and_observer_weighted_estimands_are_distinct(monkeypatch):
    monkeypatch.setattr(sealed_module, "FROZEN_BOOTSTRAP_SAMPLES", 10)
    records = [
        {
            "game_id": "short",
            "status": "scored",
            "observed_rows": 1,
            "model_kl_sum": 0.0,
            "uniform_kl_sum": 1.0,
            "gap_closed": 1.0,
            "total_variation_mean": 0.1,
        },
        {
            "game_id": "long",
            "status": "scored",
            "observed_rows": 9,
            "model_kl_sum": 9.0,
            "uniform_kl_sum": 9.0,
            "gap_closed": 0.0,
            "total_variation_mean": 0.3,
        },
    ]
    summary = summarize_metric_records(records)
    assert summary["primary_game_macro_gap_closed"] == pytest.approx(0.5)
    assert summary["observer_weighted_gap_closed"] == pytest.approx(0.1)
    assert summary["total_variation_mean"] == pytest.approx(0.28)


def test_bootstrap_is_frozen_to_game_unit(monkeypatch):
    monkeypatch.setattr(sealed_module, "FROZEN_BOOTSTRAP_SAMPLES", 25)
    records = [
        {
            "game_id": f"game_{index}",
            "status": "scored",
            "observed_rows": index + 1,
            "model_kl_sum": 1.0,
            "uniform_kl_sum": 2.0,
            "gap_closed": 0.5,
            "total_variation_mean": 0.2,
        }
        for index in range(6)
    ]
    bootstrap = summarize_metric_records(records)["bootstrap_ci95"]
    assert bootstrap["unit"] == "game"
    assert bootstrap["game_count"] == 6
    assert bootstrap["bootstrap_samples"] == 25
    assert bootstrap["seed"] == 42


def test_sealed_path_contains_no_training_or_checkpoint_write_calls():
    source = inspect.getsource(run_sealed_evaluation)
    names = set(run_sealed_evaluation.__code__.co_names)
    assert "AdamW" not in names
    assert "build_learning_rate_scheduler" not in names
    assert ".backward(" not in source
    assert "_atomic_torch_save" not in source
    assert "torch.inference_mode()" in source


def _synthetic_metric(scale):
    return {
        "valid_observer_count": 2,
        "model_kl_sum": 0.5 * scale,
        "uniform_non_self_baseline_kl_sum": 1.0 * scale,
        "mean_belief_total_variation": 0.2,
    }


def test_synthetic_six_game_one_shot_run_writes_only_audit_artifacts(
    tmp_path, monkeypatch
):
    checkpoint_path = tmp_path / "final.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    protocol_path = tmp_path / "final_protocol.json"
    protocol_path.write_text("{}\n", encoding="utf-8")
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    test_path = tmp_path / "test.jsonl"
    test_path.write_text("synthetic sealed labels\n", encoding="utf-8")
    sealed_ids = tuple(f"sealed_{index}" for index in range(6))
    model = SimpleNamespace(
        config=SimpleNamespace(max_seq_len=256),
        to=lambda device: model,
        eval=lambda: model,
    )
    manifest = {
        "manifest_digest": "a" * 64,
        "output_files": {"test": {"sha256": sealed_module.sha256_file(test_path)}},
    }
    final_protocol = {"data_lineage": {}}
    plan = SealedPreflight(
        checkpoint_path=checkpoint_path,
        final_protocol_path=protocol_path,
        manifest_path=manifest_path,
        output_dir=tmp_path / "sealed_output",
        marker_path=sealed_module._marker_path(checkpoint_path),
        sealed_dataset_path=test_path,
        evaluator_git_commit="f" * 40,
        sealed_game_ids=sealed_ids,
        development_game_ids=tuple(f"dev_{index}" for index in range(54)),
        manifest=manifest,
        final_protocol=final_protocol,
        checkpoint={},
        model=model,
    )
    monkeypatch.setattr(
        sealed_module, "preflight_sealed_evaluation", lambda *args, **kwargs: plan
    )
    monkeypatch.setattr(sealed_module, "resolve_device", lambda _: torch.device("cpu"))
    monkeypatch.setattr(
        sealed_module,
        "validate_split_manifest",
        lambda *args, **kwargs: manifest,
    )
    class FakeDataset(torch.utils.data.Dataset):
        samples = [{"game_id": game_id} for game_id in sealed_ids]
        model_input_scope = None
        target_semantics = None
        target_conversion = None
        label_observation_semantics = None
        supervision_scope = "all_alive"
        speech_annotation_source = "v1"
        belief_annotation_source = "v1_empty_unobserved"

        def __len__(self):
            return 6

        def __getitem__(self, index):
            return {"index": index}

    monkeypatch.setattr(
        sealed_module.DenseTWDToMDataset,
        "from_jsonl",
        lambda *args, **kwargs: FakeDataset(),
    )
    monkeypatch.setattr(sealed_module, "collate_dense_twd_tom_games", lambda x: x)
    by_game = {
        game_id: _synthetic_metric(index + 1)
        for index, game_id in enumerate(sealed_ids)
    }
    monkeypatch.setattr(
        sealed_module,
        "evaluate_model_with_games_and_strata",
        lambda *args, **kwargs: ({}, by_game, {}, {}),
    )
    monkeypatch.setattr(sealed_module, "FROZEN_BOOTSTRAP_SAMPLES", 10)
    monkeypatch.setattr(sealed_module, "_new_run_id", lambda: "sealed_test_run")
    monkeypatch.setattr(
        sealed_module, "_utc_timestamp", lambda: "2026-08-28T00:00:00+00:00"
    )
    config = SealedEvalConfig(
        checkpoint_path=str(checkpoint_path),
        final_protocol_path=str(protocol_path),
        manifest_path=str(manifest_path),
        output_dir=str(plan.output_dir),
        device="cpu",
    )
    result = run_sealed_evaluation(config, repo_root=tmp_path)
    assert result["sealed_test_evaluated"] is True
    assert sorted(path.name for path in plan.output_dir.iterdir()) == sorted([
        SEALED_PROTOCOL_FILENAME,
        SEALED_SUMMARY_FILENAME,
        SEALED_PER_GAME_FILENAME,
        SEALED_PROVENANCE_FILENAME,
    ])
    assert not list(plan.output_dir.glob("*.pt"))
    marker = json.loads(plan.marker_path.read_text(encoding="utf-8"))
    assert marker["status"] == "completed"
    assert marker["sealed_test_evaluated"] is True
    with pytest.raises(FileExistsError):
        run_sealed_evaluation(config, repo_root=tmp_path)
