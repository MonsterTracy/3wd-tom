"""Tests for subjective ToM checkpoint evaluation."""

import inspect
import json
from pathlib import Path

import pytest
import torch
from transformers import GPT2Model
import werewolf.models.twd_tom.schema as tom_schema

from script.twd_tom.eval import (
    EvaluationConfig,
    build_model_from_checkpoint,
    evaluate_checkpoint,
    load_checkpoint,
    resolve_training_dataset_path,
)
from script.twd_tom.project_suspicion_to_pairs import project_suspicion_sample
from script.twd_tom.train import (
    TrainingConfig,
    run_training,
)
from werewolf.models.twd_tom.belief_backbone import (
    ToMBeliefBackbone,
    ToMBeliefBackboneConfig,
)
from werewolf.models.twd_tom.public_events import (
    PHASE_TO_ID,
    PUBLIC_EVENT_SCHEMA_VERSION,
    STRUCTURED_TOKEN_TO_ID,
)
from werewolf.models.twd_tom.samples import (
    SAMPLE_SCHEMA_VERSION as PLAYER_SUSPICION_SCHEMA_VERSION,
)
from tests.twd_tom.public_event_fixtures import public_history_fields
from werewolf.models.twd_tom.schema import (
    LABEL_PROVENANCE,
    LABEL_PROMPT_VERSION,
    NUM_WOLF_PAIR_CLASSES,
    PROJECTED_SCHEMA_VERSION as SAMPLE_SCHEMA_VERSION,
    TARGET_ENCODING,
)


def make_sample(
    *,
    game_id,
    step_idx,
    accused_player="player7",
):
    actions = [
        ["player6", "point_as_seer", "player6"],
        ["player6", "point_as_werewolf", accused_player],
    ]
    history_actions = [] if step_idx == 1 else actions
    return project_suspicion_sample({
        "schema_version": (
            PLAYER_SUSPICION_SCHEMA_VERSION
        ),
        "game_id": game_id,
        "step_idx": step_idx,
        "report_trigger": "pre_public_speech",
        "phase": "1_day_speech",
        "speaker_id": 6,
        "observer_ids": [
            1,
            3,
            5,
        ],
        **public_history_fields(history_actions, speaker_id=6),
        "suspected_werewolves": {
            "player1": None,
            "player3": ["player7"],
            "player5": [],
        },
        "known_werewolves": {
            subject: []
            for subject in ("player1", "player3", "player5")
        },
        "known_non_werewolves": {
            subject: [subject]
            for subject in ("player1", "player3", "player5")
        },
        "belief_status": {
            "player1": "parse_error",
            "player3": "ok",
            "player5": "ok",
        },
        "belief_errors": {
            "player1": "synthetic invalid report",
            "player3": None,
            "player5": None,
        },
        "label_provenance": LABEL_PROVENANCE,
        "agent_backend_ids": {
            subject: "fake_backend"
            for subject in ("player1", "player3", "player5")
        },
        "label_cutoff_step_idx": step_idx,
        "label_prompt_version": LABEL_PROMPT_VERSION,
    })


def write_samples(
    path: Path,
    samples,
):
    path.write_text(
        "\n".join(
            json.dumps(
                sample,
                ensure_ascii=False,
            )
            for sample in samples
        )
        + "\n",
        encoding="utf-8",
    )


def build_test_checkpoint(
    tmp_path: Path,
):
    training_dataset = (
        tmp_path
        / "training.jsonl"
    )
    validation_dataset = tmp_path / "validation.jsonl"

    output_dir = (
        tmp_path
        / "training_output"
    )

    write_samples(
        training_dataset,
        [
            make_sample(
                game_id="train_game_a",
                step_idx=1,
            ),
            make_sample(
                game_id="train_game_a",
                step_idx=2,
            ),
        ],
    )
    write_samples(
        validation_dataset,
        [
            make_sample(
                game_id="validation_game",
                step_idx=1,
                accused_player="player6",
            ),
            make_sample(
                game_id="validation_game",
                step_idx=2,
                accused_player="player6",
            ),
        ],
    )

    summary = run_training(
        TrainingConfig(
            train_dataset_path=str(
                training_dataset
            ),
            validation_dataset_path=str(validation_dataset),
            output_dir=str(
                output_dir
            ),
            epochs=1,
            batch_size=2,
            seed=3,
            device="cpu",
            learning_rate=1e-3,
            weight_decay=0.0,
            d_model=8,
            n_head=2,
            n_layer=1,
            dropout=0.0,
            max_seq_len=8,
            dim_feedforward=16,
        )
    )

    return (
        training_dataset,
        Path(
            summary[
                "best_checkpoint"
            ]
        ),
    )


def save_checkpoint(
    value,
    path: Path,
):
    torch.save(
        value,
        path,
    )


def test_evaluates_disjoint_game_ids(
    tmp_path,
):
    (
        training_dataset,
        checkpoint_path,
    ) = build_test_checkpoint(
        tmp_path
    )

    evaluation_dataset = (
        tmp_path
        / "evaluation.jsonl"
    )

    output_path = (
        tmp_path
        / "evaluation_summary.json"
    )

    write_samples(
        evaluation_dataset,
        [
            make_sample(
                game_id="evaluation_game",
                step_idx=1,
            ),
            make_sample(
                game_id="evaluation_game",
                step_idx=2,
            ),
        ],
    )

    summary = evaluate_checkpoint(
        EvaluationConfig(
            checkpoint_path=str(
                checkpoint_path
            ),
            dataset_path=str(
                evaluation_dataset
            ),
            training_dataset_path=str(
                training_dataset
            ),
            output_path=str(
                output_path
            ),
            batch_size=2,
            device="cpu",
        )
    )

    assert summary[
        "status"
    ] == "ok"

    assert summary[
        "checkpoint_epoch"
    ] == 1

    assert summary[
        "evaluation_sample_count"
    ] == 2

    assert summary[
        "evaluation_game_ids"
    ] == [
        "evaluation_game",
    ]

    assert summary[
        "evaluation_supervised_subject_count"
    ] == 4

    assert summary[
        "overlapping_game_ids"
    ] == []

    assert summary[
        "game_id_overlap_check_enabled"
    ] is True

    assert output_path.is_file()

    stored_summary = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    assert stored_summary["backbone"] == "gpt2_model"
    assert "backbone_type" not in stored_summary["model_config"]

    assert stored_summary[
        "metrics"
    ][
        "valid_subject_count"
    ] == 4

    assert (
        stored_summary[
            "metrics"
        ][
            "mean_pair_kl_divergence"
        ]
        >= 0.0
    )


def test_overlapping_game_id_is_rejected(
    tmp_path,
):
    (
        training_dataset,
        checkpoint_path,
    ) = build_test_checkpoint(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match="overlap",
    ):
        evaluate_checkpoint(
            EvaluationConfig(
                checkpoint_path=str(
                    checkpoint_path
                ),
                dataset_path=str(
                    training_dataset
                ),
                device="cpu",
            )
        )


def test_overlap_can_be_allowed_for_smoke_test(
    tmp_path,
):
    (
        training_dataset,
        checkpoint_path,
    ) = build_test_checkpoint(
        tmp_path
    )

    summary = evaluate_checkpoint(
        EvaluationConfig(
            checkpoint_path=str(
                checkpoint_path
            ),
            dataset_path=str(
                training_dataset
            ),
            device="cpu",
            allow_game_id_overlap=True,
        )
    )

    assert summary[
        "status"
    ] == "ok"

    assert summary[
        "game_id_overlap_check_enabled"
    ] is False

    assert summary[
        "training_dataset_path"
    ] is None

    assert summary[
        "overlapping_game_ids"
    ] == []


def test_checkpoint_restores_model_config(
    tmp_path,
):
    (
        _training_dataset,
        checkpoint_path,
    ) = build_test_checkpoint(
        tmp_path
    )

    checkpoint = load_checkpoint(
        checkpoint_path
    )

    assert checkpoint[
        "target_distribution_is_reporter_probability"
    ] is False
    assert checkpoint[
        "target_distribution_is_deterministic_encoding"
    ] is True

    model = build_model_from_checkpoint(
        checkpoint,
        device=torch.device("cpu"),
    )

    assert checkpoint["backbone"] == "gpt2_model"
    assert isinstance(model.transformer, GPT2Model)
    assert model.config.d_model == 8
    assert model.config.n_head == 2
    assert model.config.n_layer == 1

    assert (
        model.config.dim_feedforward
        == 16
    )

    assert model.training is False
    restored_state = model.state_dict()
    for field_name, expected_tensor in checkpoint[
        "model_state_dict"
    ].items():
        torch.testing.assert_close(
            restored_state[field_name],
            expected_tensor,
        )


def test_wrong_checkpoint_schema_is_rejected(
    tmp_path,
):
    (
        _training_dataset,
        checkpoint_path,
    ) = build_test_checkpoint(
        tmp_path
    )

    checkpoint = load_checkpoint(
        checkpoint_path
    )

    checkpoint[
        "schema_version"
    ] = "onuw7_public_pair_support_v1"

    invalid_path = (
        tmp_path
        / "invalid_schema.pt"
    )

    save_checkpoint(
        checkpoint,
        invalid_path,
    )

    invalid_checkpoint = (
        load_checkpoint(
            invalid_path
        )
    )

    with pytest.raises(
        ValueError,
        match="schema mismatch",
    ):
        build_model_from_checkpoint(
            invalid_checkpoint,
            device=torch.device(
                "cpu"
            ),
        )


def test_wrong_checkpoint_target_encoding_is_rejected(tmp_path):
    _, checkpoint_path = build_test_checkpoint(tmp_path)
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint["target_encoding"] = "legacy_player_distribution"
    with pytest.raises(ValueError, match="target_encoding mismatch"):
        build_model_from_checkpoint(
            checkpoint,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize(
    "field_name, invalid_value, remove_field",
    [
        pytest.param(
            "projection_version",
            "wrong",
            False,
            id="wrong-projection-version",
        ),
        pytest.param(
            "target_distribution_is_reporter_probability",
            None,
            True,
            id="missing-reporter-probability",
        ),
        pytest.param(
            "target_distribution_is_deterministic_encoding",
            None,
            True,
            id="missing-deterministic-encoding",
        ),
        pytest.param(
            "target_distribution_is_reporter_probability",
            True,
            False,
            id="reporter-probability-true",
        ),
        pytest.param(
            "target_distribution_is_deterministic_encoding",
            False,
            False,
            id="deterministic-encoding-false",
        ),
        pytest.param(
            "target_distribution_is_reporter_probability",
            "false",
            False,
            id="reporter-probability-string",
        ),
        pytest.param(
            "target_distribution_is_deterministic_encoding",
            "true",
            False,
            id="deterministic-encoding-string",
        ),
    ],
)
def test_checkpoint_target_distribution_metadata_is_strict(
    tmp_path,
    field_name,
    invalid_value,
    remove_field,
):
    _, checkpoint_path = build_test_checkpoint(tmp_path)
    checkpoint = load_checkpoint(checkpoint_path)
    if remove_field:
        checkpoint.pop(field_name)
    else:
        checkpoint[field_name] = invalid_value

    with pytest.raises(
        ValueError,
        match=f"checkpoint {field_name} mismatch",
    ):
        build_model_from_checkpoint(
            checkpoint,
            device=torch.device("cpu"),
        )


def test_incompatible_state_dict_is_rejected(
    tmp_path,
):
    (
        _training_dataset,
        checkpoint_path,
    ) = build_test_checkpoint(
        tmp_path
    )

    checkpoint = load_checkpoint(
        checkpoint_path
    )

    checkpoint[
        "model_state_dict"
    ].pop(
        "output_projection.bias"
    )

    invalid_path = (
        tmp_path
        / "invalid_state.pt"
    )

    save_checkpoint(
        checkpoint,
        invalid_path,
    )

    invalid_checkpoint = (
        load_checkpoint(
            invalid_path
        )
    )

    with pytest.raises(
        ValueError,
        match="state_dict",
    ):
        build_model_from_checkpoint(
            invalid_checkpoint,
            device=torch.device(
                "cpu"
            ),
        )


def test_missing_training_dataset_is_rejected(
    tmp_path,
):
    (
        training_dataset,
        checkpoint_path,
    ) = build_test_checkpoint(
        tmp_path
    )

    evaluation_dataset = (
        tmp_path
        / "evaluation.jsonl"
    )

    write_samples(
        evaluation_dataset,
        [
            make_sample(
                game_id="independent_game",
                step_idx=1,
            )
        ],
    )

    training_dataset.unlink()

    with pytest.raises(
        FileNotFoundError,
        match="training dataset",
    ):
        evaluate_checkpoint(
            EvaluationConfig(
                checkpoint_path=str(
                    checkpoint_path
                ),
                dataset_path=str(
                    evaluation_dataset
                ),
                device="cpu",
            )
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "checkpoint_path": "",
        },
        {
            "dataset_path": "",
        },
        {
            "batch_size": 0,
        },
        {
            "num_workers": -1,
        },
    ],
)
def test_invalid_evaluation_config(
    tmp_path,
    kwargs,
):
    arguments = {
        "checkpoint_path": str(
            tmp_path / "checkpoint.pt"
        ),
        "dataset_path": str(
            tmp_path / "evaluation.jsonl"
        ),
    }

    arguments.update(
        kwargs
    )

    with pytest.raises(
        (TypeError, ValueError),
    ):
        EvaluationConfig(
            **arguments
        )


def test_evaluation_apis_have_no_truth_inputs():
    forbidden = {
        "roles",
        "true_roles",
        "wolf_labels",
        "truth",
        "actual_wolves",
        "alive_mask",
        "observer_id",
    }

    for function in (
        evaluate_checkpoint,
        build_model_from_checkpoint,
        load_checkpoint,
    ):
        parameters = inspect.signature(
            function
        ).parameters

        assert forbidden.isdisjoint(
            parameters
        )



def test_checkpoint_without_pair_target_encoding_is_rejected(
    tmp_path,
):
    (
        _training_dataset,
        checkpoint_path,
    ) = build_test_checkpoint(tmp_path)

    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint.pop("target_encoding")

    with pytest.raises(ValueError, match="target_encoding mismatch"):
        build_model_from_checkpoint(
            checkpoint,
            device=torch.device("cpu"),
        )


def test_gpt2model_checkpoint_restores_strictly_and_rejects_torch():
    config = ToMBeliefBackboneConfig(
        d_model=16,
        n_head=4,
        n_layer=1,
        dropout=0.0,
        max_seq_len=8,
        dim_feedforward=32,
    )
    source_model = ToMBeliefBackbone(config)
    checkpoint = {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "public_event_schema_version": PUBLIC_EVENT_SCHEMA_VERSION,
        "structured_token_to_id": dict(STRUCTURED_TOKEN_TO_ID),
        "public_phase_to_id": dict(PHASE_TO_ID),
        "target_encoding": TARGET_ENCODING,
        "projection_version": tom_schema.PROJECTION_VERSION,
        "pair_class_count": NUM_WOLF_PAIR_CLASSES,
        "raw_label_field": tom_schema.RAW_LABEL_FIELD,
        "raw_label_type": tom_schema.RAW_LABEL_TYPE,
        "numeric_annotation_present": tom_schema.NUMERIC_ANNOTATION_PRESENT,
        "raw_label_semantics": tom_schema.RAW_LABEL_SEMANTICS,
        "target_interpretation": tom_schema.TARGET_INTERPRETATION,
        "target_distribution_is_reporter_probability": False,
        "target_distribution_is_deterministic_encoding": True,
        "supervision_scope": tom_schema.SUPERVISION_SCOPE,
        "label_source": tom_schema.LABEL_SOURCE,
        "label_context_scope": tom_schema.LABEL_CONTEXT_SCOPE,
        "model_input_scope": tom_schema.MODEL_INPUT_SCOPE,
        "report_context_mode": tom_schema.REPORT_CONTEXT_MODE,
        "report_side_effect_free": tom_schema.REPORT_SIDE_EFFECT_FREE,
        "global_truth_injected": tom_schema.GLOBAL_TRUTH_INJECTED,
        "other_players_private_information_visible": tom_schema.OTHER_PLAYERS_PRIVATE_INFORMATION_VISIBLE,
        "private_context_serialized": tom_schema.PRIVATE_CONTEXT_SERIALIZED,
        "report_timing": tom_schema.REPORT_TIMING,
        "observer_selection": tom_schema.OBSERVER_SELECTION,
        "truth_based_observer_selection": tom_schema.TRUTH_BASED_OBSERVER_SELECTION,
        "pair_ordering": tom_schema.PAIR_ORDERING,
        "model_output": tom_schema.MODEL_OUTPUT,
        "output_activation": tom_schema.OUTPUT_ACTIVATION,
        "backbone": "gpt2_model",
        "epoch": 1,
        "model_config": {
            "num_players": config.num_players,
            "pair_class_count": config.pair_class_count,
            "d_model": config.d_model,
            "n_head": config.n_head,
            "n_layer": config.n_layer,
            "dropout": config.dropout,
            "max_seq_len": config.max_seq_len,
            "dim_feedforward": config.dim_feedforward,
        },
        "model_state_dict": source_model.state_dict(),
    }

    restored = build_model_from_checkpoint(
        checkpoint,
        device=torch.device("cpu"),
    )

    assert isinstance(restored.transformer, GPT2Model)
    assert restored.training is False

    source_state = source_model.state_dict()
    restored_state = restored.state_dict()
    assert source_state.keys() == restored_state.keys()

    for key in source_state:
        torch.testing.assert_close(
            source_state[key],
            restored_state[key],
        )

    torch_checkpoint = dict(checkpoint)
    torch_checkpoint["backbone"] = "torch"
    with pytest.raises(
        ValueError,
        match="checkpoint backbone mismatch",
    ):
        build_model_from_checkpoint(
            torch_checkpoint,
            device=torch.device("cpu"),
        )


def test_explicit_checkpoint_training_path_is_resolved(
    tmp_path,
):
    train_path = tmp_path / "formal_train.jsonl"
    train_path.write_text("{}\n", encoding="utf-8")

    checkpoint = {
        "training_config": {
            "dataset_path": None,
            "train_dataset_path": str(train_path),
        }
    }

    resolved = resolve_training_dataset_path(
        checkpoint,
        override_path=None,
    )

    assert resolved == train_path.resolve()
