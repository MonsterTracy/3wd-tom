import json

import pytest
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from script.twd_tom.train import (
    MODEL_OUTPUT,
    OBJECTIVE,
    TrainingConfig,
    _repository_relative_path,
    _forward_batch,
    _move_batch_to_device,
    build_data_loader,
    build_model,
    checkpoint_payload,
    checkpoint_task_contract,
    evaluate_model,
    evaluate_model_with_games,
    evaluate_model_with_games_and_strata,
    game_macro_metrics,
    train_one_epoch,
)
from werewolf.models.twd_tom.belief_backbone import (
    NO_PHASE_DAY_INPUT_FEATURE_PROFILE,
)
from werewolf.models.twd_tom.annotation_v2 import (
    LEGACY_V1_BELIEF_SOURCE,
    V1_ANNOTATION_SOURCE,
)
from werewolf.models.twd_tom.checkpoint import build_model_from_checkpoint
from werewolf.models.twd_tom.dataset import (
    CYCLIC_ROTATION_VERSION,
    MODEL_INPUT_SCOPE,
    PRIVATE_MODEL_INPUT_SCOPE,
    LEGACY_V1_LABEL_OBSERVATION_SEMANTICS,
    LEGACY_V1_TARGET_CONVERSION,
    LEGACY_V1_TARGET_SEMANTICS,
    TARGET_CONVERSION,
    TARGET_SEMANTICS,
    V2_TARGET_CONVERSION,
    V2_TARGET_SEMANTICS,
    TWDToMDataset,
    collate_twd_tom_samples,
)
from werewolf.models.twd_tom.dense_dataset import (
    DENSE_SUPERVISION_VERSION,
    DenseTWDToMDataset,
    collate_dense_twd_tom_games,
)


def write_jsonl(path, sample):
    path.write_text(json.dumps(sample) + "\n", encoding="utf-8")


def test_provenance_preserves_repository_symlink_path(tmp_path):
    repository = tmp_path / "repository"
    physical_data = tmp_path / "physical_data"
    repository.mkdir()
    physical_data.mkdir()
    (repository / "datasets").symlink_to(physical_data, target_is_directory=True)
    dataset = repository / "datasets" / "fold_0" / "train.jsonl"

    assert _repository_relative_path(dataset, repo_root=repository) == (
        "datasets/fold_0/train.jsonl"
    )


def make_config(tmp_path, dataset_path):
    return TrainingConfig(
        output_dir=str(tmp_path / "run"),
        dataset_path=str(dataset_path),
        validation_dataset_path=str(dataset_path),
        epochs=1,
        batch_size=1,
        max_seq_len=32,
        backbone="gpt2_block",
        device="cpu",
    )


def test_checkpoint_contract_is_single_belief_objective():
    contract = checkpoint_task_contract()
    assert contract == {
        "objective": OBJECTIVE,
        "model_input_scope": MODEL_INPUT_SCOPE,
        "model_output": "belief_logits",
        "output_shape": [7, 7],
        "target_semantics": TARGET_SEMANTICS,
        "target_conversion": TARGET_CONVERSION,
        "train_player_augmentation": CYCLIC_ROTATION_VERSION,
    }
    assert all("pair" not in key for key in contract)
    assert all("order" not in key for key in contract)


def test_private_checkpoint_contract_is_explicit_and_parallel():
    contract = checkpoint_task_contract(private_conditioning=True)
    assert contract["objective"] == OBJECTIVE
    assert contract["model_input_scope"] == PRIVATE_MODEL_INPUT_SCOPE
    assert contract["private_conditioning"] is True
    assert contract["model_output"] == "belief_logits"


def test_v2_checkpoint_contract_records_the_selected_target_pair():
    contract = checkpoint_task_contract(
        target_semantics=V2_TARGET_SEMANTICS,
        target_conversion=V2_TARGET_CONVERSION,
    )
    assert contract["target_semantics"] == V2_TARGET_SEMANTICS
    assert contract["target_conversion"] == V2_TARGET_CONVERSION
    with pytest.raises(ValueError, match="unsupported target"):
        checkpoint_task_contract(
            target_semantics=V2_TARGET_SEMANTICS,
            target_conversion=TARGET_CONVERSION,
        )


def test_legacy_v1_checkpoint_contract_remains_explicitly_supported():
    contract = checkpoint_task_contract(
        target_semantics=LEGACY_V1_TARGET_SEMANTICS,
        target_conversion=LEGACY_V1_TARGET_CONVERSION,
    )
    assert contract["target_conversion"] == LEGACY_V1_TARGET_CONVERSION
    assert contract["target_conversion"] != TARGET_CONVERSION


def test_training_loader_uses_rotation_only_for_training(
    tmp_path, training_sample_factory
):
    path = tmp_path / "data.jsonl"
    write_jsonl(path, training_sample_factory())
    config = make_config(tmp_path, path)
    _, training = build_data_loader(config, dataset_path=path, shuffle=True)
    _, validation = build_data_loader(config, dataset_path=path, shuffle=False)
    assert training.enable_cyclic_rotation is True
    assert validation.enable_cyclic_rotation is False


def test_supervision_role_metadata_never_enters_model_forward(
    training_sample_factory,
):
    sample = training_sample_factory()
    roles = {
        sample["game_id"]: {
            "player1": "Werewolf",
            "player2": "Werewolf",
            "player3": "Villager",
            "player4": "Villager",
            "player5": "Villager",
            "player6": "Seer",
            "player7": "Witch",
        }
    }
    dataset = TWDToMDataset(
        [sample],
        observer_roles_by_game=roles,
        supervision_scope="villager_alive",
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_twd_tom_samples,
    )
    batch = _move_batch_to_device(next(iter(loader)), torch.device("cpu"))
    captured = {}

    class RecordingModel:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return {MODEL_OUTPUT: torch.zeros(1, 7, 7)}

    _forward_batch(RecordingModel(), batch)

    assert set(captured) == {
        "subject_ids",
        "action_ids",
        "object_ids",
        "event_type_ids",
        "phase_ids",
        "day_values",
        "attention_mask",
    }
    assert not any(
        marker in name
        for name in captured
        for marker in ("role", "is_werewolf", "is_villager")
    )


def test_private_training_path_threads_knowledge_masks_and_metrics(
    tmp_path, training_sample_factory
):
    path = tmp_path / "private.jsonl"
    write_jsonl(path, training_sample_factory())
    config = TrainingConfig(
        output_dir=str(tmp_path / "run"),
        dataset_path=str(path),
        validation_dataset_path=str(path),
        epochs=1,
        batch_size=1,
        max_seq_len=32,
        backbone="gpt2_block",
        device="cpu",
        dense_supervision=True,
        private_conditioning=True,
    )
    loader, dataset = build_data_loader(config, dataset_path=path, shuffle=False)
    batch = _move_batch_to_device(next(iter(loader)), torch.device("cpu"))
    model = build_model(config)

    assert dataset.include_private_features is True
    assert model.config.private_conditioning is True
    assert _forward_batch(model, batch)[MODEL_OUTPUT].shape == (1, 1, 7, 7)
    metrics = evaluate_model(model, loader, device=torch.device("cpu"))
    assert "private_admissible_normalized_reducible_gap_improvement" in metrics


def test_one_train_and_evaluation_batch_use_direct_belief_contract(
    tmp_path, training_sample_factory
):
    path = tmp_path / "data.jsonl"
    write_jsonl(path, training_sample_factory())
    config = make_config(tmp_path, path)
    loader, _ = build_data_loader(config, dataset_path=path, shuffle=False)
    model = build_model(config)
    raw_batch = next(iter(loader))
    batch = _move_batch_to_device(raw_batch, torch.device("cpu"))
    output = _forward_batch(model, batch)
    assert output[MODEL_OUTPUT].shape == (1, 7, 7)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    training = train_one_epoch(
        model,
        loader,
        optimizer,
        device=torch.device("cpu"),
        gradient_clip_norm=1.0,
    )
    evaluation = evaluate_model(model, loader, device=torch.device("cpu"))
    assert training["valid_observer_count"] == 4
    assert evaluation["valid_observer_count"] == 4
    assert training["mean_loss"] > 0
    assert evaluation["mean_loss"] > 0
    assert evaluation["mean_belief_target_entropy"] >= 0
    assert evaluation["mean_belief_kl_divergence"] == pytest.approx(
        evaluation["mean_belief_cross_entropy"]
        - evaluation["mean_belief_target_entropy"]
    )
    assert evaluation["normalized_reducible_gap_improvement"] == pytest.approx(
        1.0
        - evaluation["mean_belief_kl_divergence"]
        / evaluation["uniform_non_self_baseline_mean_kl_divergence"]
    )


def test_dense_game_batch_trains_every_pre_boundary(
    tmp_path, training_sample_factory
):
    path = tmp_path / "dense.jsonl"
    write_jsonl(path, training_sample_factory())
    config = TrainingConfig(
        output_dir=str(tmp_path / "run"),
        dataset_path=str(path),
        validation_dataset_path=str(path),
        epochs=1,
        batch_size=8,
        max_seq_len=32,
        backbone="gpt2_block",
        device="cpu",
        dense_supervision=True,
    )
    loader, dataset = build_data_loader(
        config,
        dataset_path=path,
        shuffle=False,
    )
    assert isinstance(dataset, DenseTWDToMDataset)
    assert dataset.supervision_version == DENSE_SUPERVISION_VERSION
    model = build_model(config)
    batch = _move_batch_to_device(next(iter(loader)), torch.device("cpu"))
    assert _forward_batch(model, batch)[MODEL_OUTPUT].shape == (1, 1, 7, 7)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    metrics = train_one_epoch(
        model,
        loader,
        optimizer,
        device=torch.device("cpu"),
        gradient_clip_norm=1.0,
    )
    assert metrics["valid_observer_count"] == 4
    assert metrics["mean_loss"] > 0
    aggregate, by_game = evaluate_model_with_games(
        model,
        loader,
        device=torch.device("cpu"),
    )
    assert set(by_game) == {"synthetic_game_001"}
    assert by_game["synthetic_game_001"]["valid_observer_count"] == 4
    assert aggregate["mean_loss"] == pytest.approx(
        by_game["synthetic_game_001"]["mean_loss"]
    )
    _, _, strata, strata_by_game = evaluate_model_with_games_and_strata(
        model,
        loader,
        device=torch.device("cpu"),
    )
    assert "true" not in strata["raw_empty"]
    assert strata["raw_empty"]["false"]["total_row_count"] == 4
    assert strata["game_id"]["synthetic_game_001"][
        "total_row_count"
    ] == 4
    assert strata["speaker_vs_non_speaker"]["speaker"][
        "total_row_count"
    ] == 1
    assert strata_by_game["synthetic_game_001"]["raw_empty"]["false"][
        "total_row_count"
    ] == 4


def test_dense_evaluation_excludes_unscored_game_without_losing_lineage(
    tmp_path,
    training_sample_factory,
):
    scored = training_sample_factory(game_id="scored_game")
    unscored = training_sample_factory(game_id="unscored_game")
    unscored["suspected_werewolves"] = {
        observer: [] for observer in unscored["suspected_werewolves"]
    }
    dataset = DenseTWDToMDataset([scored, unscored])
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_dense_twd_tom_games,
    )
    config = TrainingConfig(
        output_dir=str(tmp_path / "run"),
        dataset_path=str(tmp_path / "unused.jsonl"),
        validation_dataset_path=str(tmp_path / "unused.jsonl"),
        epochs=1,
        batch_size=2,
        max_seq_len=32,
        backbone="gpt2_block",
        device="cpu",
        dense_supervision=True,
    )
    model = build_model(config)

    aggregate, by_game, _, strata_by_game = (
        evaluate_model_with_games_and_strata(
            model,
            loader,
            device=torch.device("cpu"),
        )
    )

    assert aggregate["valid_observer_count"] == 4
    assert set(by_game) == {"scored_game", "unscored_game"}
    assert by_game["unscored_game"]["status"] == (
        "unscored_no_supervised_observers"
    )
    assert strata_by_game["unscored_game"] == {}
    assert game_macro_metrics(by_game) == game_macro_metrics({
        "scored_game": by_game["scored_game"],
    })


def test_checkpoint_payload_contains_no_removed_objective_fields(tmp_path):
    config = TrainingConfig(
        output_dir=str(tmp_path / "run"),
        dataset_path="train.jsonl",
        validation_dataset_path="validation.jsonl",
        epochs=1,
        batch_size=1,
        backbone="gpt2_block",
    )
    model = build_model(config)
    optimizer = AdamW(model.parameters())
    metrics = {"mean_loss": 1.0, "valid_observer_count": 4}
    provenance = {
        "train_dataset_path": "train.jsonl",
        "validation_dataset_path": "validation.jsonl",
        "output_dir": "run",
    }
    payload = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=1,
        train_metrics=metrics,
        validation_metrics=metrics,
        best_epoch=1,
        best_validation_mean_loss=1.0,
        run_provenance=provenance,
    )
    serialized = json.dumps(
        {key: value for key, value in payload.items() if key not in {
            "model_state_dict", "optimizer_state_dict"
        }}
    )
    for removed in ("tom_order", "pair", "second_order", "observer_pair_logits"):
        assert removed not in serialized


def test_checkpoint_payload_uses_dataset_v2_target_contract(tmp_path):
    from werewolf.models.twd_tom.samples import SAMPLE_SCHEMA_VERSION

    config = TrainingConfig(
        output_dir=str(tmp_path / "run"),
        dataset_path="train.jsonl",
        validation_dataset_path="validation.jsonl",
        epochs=1,
        batch_size=1,
        backbone="gpt2_block",
    )
    model = build_model(config)
    optimizer = AdamW(model.parameters())
    metrics = {"mean_loss": 1.0, "valid_observer_count": 4}
    payload = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=1,
        train_metrics=metrics,
        validation_metrics=metrics,
        best_epoch=1,
        best_validation_mean_loss=1.0,
        run_provenance={
            "train_dataset_path": "train.jsonl",
            "validation_dataset_path": "validation.jsonl",
            "output_dir": "run",
        },
        dataset_contract={
            "source_schema_version": SAMPLE_SCHEMA_VERSION,
            "model_input_scope": MODEL_INPUT_SCOPE,
            "target_semantics": V2_TARGET_SEMANTICS,
            "target_conversion": V2_TARGET_CONVERSION,
        },
    )
    assert payload["target_semantics"] == V2_TARGET_SEMANTICS
    assert payload["target_conversion"] == V2_TARGET_CONVERSION
    restored = build_model_from_checkpoint(payload, device=torch.device("cpu"))
    assert restored.output_projection.out_features == 7


def test_checkpoint_payload_keeps_legacy_v1_provenance_distinct(tmp_path):
    from werewolf.models.twd_tom.samples import SAMPLE_SCHEMA_VERSION

    config = TrainingConfig(
        output_dir=str(tmp_path / "run"),
        dataset_path="train.jsonl",
        validation_dataset_path="validation.jsonl",
        epochs=1,
        batch_size=1,
        backbone="gpt2_block",
        belief_annotation_source=LEGACY_V1_BELIEF_SOURCE,
    )
    model = build_model(config)
    payload = checkpoint_payload(
        model=model,
        optimizer=AdamW(model.parameters()),
        config=config,
        epoch=1,
        train_metrics={"mean_loss": 1.0},
        validation_metrics={"mean_loss": 1.0},
        best_epoch=1,
        best_validation_mean_loss=1.0,
        run_provenance={
            "train_dataset_path": "train.jsonl",
            "validation_dataset_path": "validation.jsonl",
            "output_dir": "run",
        },
        dataset_contract={
            "source_schema_version": SAMPLE_SCHEMA_VERSION,
            "model_input_scope": MODEL_INPUT_SCOPE,
            "target_semantics": LEGACY_V1_TARGET_SEMANTICS,
            "target_conversion": LEGACY_V1_TARGET_CONVERSION,
            "label_observation_semantics": (
                LEGACY_V1_LABEL_OBSERVATION_SEMANTICS
            ),
            "speech_annotation_source": V1_ANNOTATION_SOURCE,
            "belief_annotation_source": LEGACY_V1_BELIEF_SOURCE,
        },
    )

    assert payload["belief_annotation_source"] == LEGACY_V1_BELIEF_SOURCE
    assert payload["target_conversion"] == LEGACY_V1_TARGET_CONVERSION
    assert payload["label_observation_semantics"] == (
        LEGACY_V1_LABEL_OBSERVATION_SEMANTICS
    )


def test_training_config_has_no_order_argument(tmp_path):
    with pytest.raises(TypeError):
        TrainingConfig(
            tom_order=2,
            output_dir=str(tmp_path),
            dataset_path="train.jsonl",
            validation_dataset_path="validation.jsonl",
        )


def test_role_scope_requires_explicit_supervision_sidecar(tmp_path):
    with pytest.raises(ValueError, match="role_sidecar_path"):
        TrainingConfig(
            output_dir=str(tmp_path / "run"),
            dataset_path="train.jsonl",
            validation_dataset_path="validation.jsonl",
            supervision_scope="villager_alive",
        )


def test_training_config_threads_input_profile_into_model(tmp_path):
    config = TrainingConfig(
        output_dir=str(tmp_path / "run"),
        dataset_path="train.jsonl",
        validation_dataset_path="validation.jsonl",
        backbone="gpt2_block",
        input_feature_profile=NO_PHASE_DAY_INPUT_FEATURE_PROFILE,
    )

    model = build_model(config)

    assert (
        model.config.input_feature_profile
        == NO_PHASE_DAY_INPUT_FEATURE_PROFILE
    )
