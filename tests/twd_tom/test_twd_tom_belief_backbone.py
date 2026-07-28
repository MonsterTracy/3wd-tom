"""Tests for the ONUW-style subjective belief backbone."""

import builtins
import inspect

import pytest
import torch
from torch import nn
from transformers import GPT2Model

import werewolf.models.twd_tom.belief_backbone as backbone_module
from werewolf.models.twd_tom.belief_backbone import (
    ToMBeliefBackbone,
    ToMBeliefBackboneConfig,
)
from werewolf.models.twd_tom.schema import (
    ACTION_TO_ID,
    NUM_PLAYERS,
    NUM_WOLF_PAIR_CLASSES,
    PLAYER_TO_ID,
)
from werewolf.models.twd_tom.public_events import STRUCTURED_TOKEN_TO_ID


def make_config():
    return ToMBeliefBackboneConfig(
        d_model=16,
        n_head=4,
        n_layer=2,
        dropout=0.0,
        max_seq_len=8,
        dim_feedforward=32,
    )


def make_features(
    *,
    batch_size=2,
    seq_len=4,
):
    subject_ids = torch.tensor(
        [
            [
                PLAYER_TO_ID["player1"],
                PLAYER_TO_ID["player2"],
                PLAYER_TO_ID["player3"],
                PLAYER_TO_ID["player4"],
            ],
            [
                PLAYER_TO_ID["player5"],
                PLAYER_TO_ID["player5"],
                PLAYER_TO_ID["player6"],
                PLAYER_TO_ID["player7"],
            ],
        ],
        dtype=torch.long,
    )[
        :batch_size,
        :seq_len,
    ]

    action_ids = torch.tensor(
        [
            [
                ACTION_TO_ID["support"],
                ACTION_TO_ID["oppose"],
                ACTION_TO_ID[
                    "point_as_werewolf"
                ],
                ACTION_TO_ID[
                    "point_as_villager"
                ],
            ],
            [
                ACTION_TO_ID[
                    "point_as_seer"
                ],
                ACTION_TO_ID["support"],
                ACTION_TO_ID["oppose"],
                ACTION_TO_ID[
                    "point_as_werewolf"
                ],
            ],
        ],
        dtype=torch.long,
    )[
        :batch_size,
        :seq_len,
    ]

    object_ids = torch.tensor(
        [
            [
                PLAYER_TO_ID["player2"],
                PLAYER_TO_ID["player3"],
                PLAYER_TO_ID["player7"],
                PLAYER_TO_ID["player1"],
            ],
            [
                PLAYER_TO_ID["player5"],
                PLAYER_TO_ID["player3"],
                PLAYER_TO_ID["player5"],
                PLAYER_TO_ID["player2"],
            ],
        ],
        dtype=torch.long,
    )[
        :batch_size,
        :seq_len,
    ]

    attention_mask = torch.ones(
        (
            batch_size,
            seq_len,
        ),
        dtype=torch.long,
    )
    event_type_ids = torch.full(
        (batch_size, seq_len),
        STRUCTURED_TOKEN_TO_ID["speech_action"],
        dtype=torch.long,
    )
    phase_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
    day_values = torch.zeros((batch_size, seq_len), dtype=torch.float32)

    return {
        "subject_ids": subject_ids,
        "action_ids": action_ids,
        "object_ids": object_ids,
        "event_type_ids": event_type_ids,
        "phase_ids": phase_ids,
        "day_values": day_values,
        "attention_mask": attention_mask,
    }


def test_forward_returns_pair_logits_and_belief_marginals():
    model = ToMBeliefBackbone(
        make_config()
    )

    output = model(
        **make_features()
    )

    assert output[
        "hidden_states"
    ].shape == (
        2,
        4,
        16,
    )

    assert output[
        "pooled_hidden_state"
    ].shape == (
        2,
        16,
    )

    assert output[
        "pair_logits"
    ].shape == (
        2,
        7,
        NUM_WOLF_PAIR_CLASSES,
    )
    assert output["pair_probabilities"].shape == (
        2, 7, NUM_WOLF_PAIR_CLASSES
    )

    assert output[
        "belief_matrix"
    ].shape == (
        2,
        7,
        7,
    )

    torch.testing.assert_close(
        output[
            "belief_matrix"
        ].sum(dim=-1),
        torch.full(
            (
                2,
                7,
            ),
            2.0,
        ),
    )

    assert torch.all(
        output["belief_matrix"].diagonal(dim1=-2, dim2=-1) > 0.0
    )

    assert torch.all(
        output["belief_matrix"] >= 0.0
    )

    assert torch.all(
        output["belief_matrix"] <= 1.0
    )


def test_embedding_vocabulary_uses_schema_ids():
    model = ToMBeliefBackbone(
        make_config()
    )

    assert (
        model.subject_embedding
        .num_embeddings
        == max(
            PLAYER_TO_ID.values()
        )
        + 1
    )

    assert (
        model.object_embedding
        .num_embeddings
        == max(
            PLAYER_TO_ID.values()
        )
        + 1
    )

    assert (
        model.action_embedding
        .num_embeddings
        == max(
            ACTION_TO_ID.values()
        )
        + 1
    )

    assert (
        model.subject_embedding
        .padding_idx
        == 0
    )

    assert (
        model.action_embedding
        .padding_idx
        == 0
    )

    assert (
        model.object_embedding
        .padding_idx
        == 0
    )


def test_padding_positions_are_zeroed():
    model = ToMBeliefBackbone(
        make_config()
    ).eval()

    features = make_features()

    features["subject_ids"][
        0,
        2:,
    ] = 0

    features["action_ids"][
        0,
        2:,
    ] = 0

    features["object_ids"][
        0,
        2:,
    ] = 0
    features["event_type_ids"][0, 2:] = 0

    features["attention_mask"][
        0,
        2:,
    ] = 0

    with torch.no_grad():
        output = model(
            **features
        )

    assert (
        output[
            "hidden_states"
        ][
            0,
            2:,
        ]
        .count_nonzero()
        .item()
        == 0
    )

    torch.testing.assert_close(
        output[
            "pooled_hidden_state"
        ][0],
        output[
            "hidden_states"
        ][
            0,
            1,
        ],
    )


def test_empty_history_is_supported():
    model = ToMBeliefBackbone(
        make_config()
    )

    zeros = torch.zeros(
        (
            2,
            1,
        ),
        dtype=torch.long,
    )

    output = model(
        subject_ids=zeros,
        action_ids=zeros,
        object_ids=zeros,
        attention_mask=zeros,
        event_type_ids=zeros,
        phase_ids=zeros,
        day_values=zeros.float(),
    )

    assert torch.isfinite(
        output["hidden_states"]
    ).all()

    assert torch.isfinite(
        output["belief_matrix"]
    ).all()

    torch.testing.assert_close(
        output[
            "belief_matrix"
        ].sum(dim=-1),
        torch.full(
            (
                2,
                NUM_PLAYERS,
            ),
            2.0,
        ),
    )


def test_attention_mask_can_be_inferred():
    model = ToMBeliefBackbone(
        make_config()
    )

    features = make_features()

    features.pop(
        "attention_mask"
    )

    output = model(
        **features
    )

    assert output[
        "belief_matrix"
    ].shape == (
        2,
        7,
        7,
    )


def test_backbone_is_causal():
    model = ToMBeliefBackbone(
        make_config()
    ).eval()

    features = make_features(
        batch_size=1,
    )

    changed = {
        key: value.clone()
        for key, value
        in features.items()
    }

    changed[
        "object_ids"
    ][
        0,
        3,
    ] = PLAYER_TO_ID[
        "player6"
    ]

    with torch.no_grad():
        original_hidden = model(
            **features
        )[
            "hidden_states"
        ]

        changed_hidden = model(
            **changed
        )[
            "hidden_states"
        ]

    torch.testing.assert_close(
        original_hidden[
            :,
            :3,
        ],
        changed_hidden[
            :,
            :3,
        ],
    )


def test_gradient_reaches_output_projection():
    model = ToMBeliefBackbone(
        make_config()
    )

    output = model(
        **make_features()
    )

    loss = (
        output[
            "pair_logits"
        ][
            :,
            0,
            0,
        ]
        .sum()
    )

    loss.backward()

    assert (
        model.output_projection
        .weight.grad
        is not None
    )

    assert torch.isfinite(
        model.output_projection
        .weight.grad
    ).all()


def test_wrong_input_rank_is_rejected():
    model = ToMBeliefBackbone(
        make_config()
    )

    with pytest.raises(
        ValueError,
        match=r"shape \[B, T\]",
    ):
        model(
            subject_ids=torch.zeros(
                (
                    2,
                    3,
                    1,
                ),
                dtype=torch.long,
            ),
            action_ids=torch.zeros(
                (
                    2,
                    3,
                ),
                dtype=torch.long,
            ),
            object_ids=torch.zeros(
                (
                    2,
                    3,
                ),
                dtype=torch.long,
            ),
        )


def test_mismatched_shapes_are_rejected():
    model = ToMBeliefBackbone(
        make_config()
    )

    with pytest.raises(
        ValueError,
        match="same shape",
    ):
        model(
            subject_ids=torch.ones(
                (
                    2,
                    3,
                ),
                dtype=torch.long,
            ),
            action_ids=torch.ones(
                (
                    2,
                    2,
                ),
                dtype=torch.long,
            ),
            object_ids=torch.ones(
                (
                    2,
                    3,
                ),
                dtype=torch.long,
            ),
        )


def test_partial_triplet_is_rejected():
    model = ToMBeliefBackbone(
        make_config()
    )

    with pytest.raises(
        ValueError,
        match="complete subject",
    ):
        model(
            subject_ids=torch.tensor(
                [
                    [
                        PLAYER_TO_ID[
                            "player1"
                        ],
                    ]
                ]
            ),
            action_ids=torch.tensor(
                [
                    [
                        0,
                    ]
                ]
            ),
            object_ids=torch.tensor(
                [
                    [
                        PLAYER_TO_ID[
                            "player2"
                        ],
                    ]
                ]
            ),
            event_type_ids=torch.tensor(
                [[STRUCTURED_TOKEN_TO_ID["speech_action"]]]
            ),
            phase_ids=torch.zeros((1, 1), dtype=torch.long),
            day_values=torch.zeros((1, 1), dtype=torch.float32),
            attention_mask=torch.ones((1, 1), dtype=torch.long),
        )


def test_non_right_padding_is_rejected():
    model = ToMBeliefBackbone(
        make_config()
    )

    features = make_features(
        batch_size=1,
    )

    features[
        "subject_ids"
    ][
        0,
        1,
    ] = 0

    features[
        "action_ids"
    ][
        0,
        1,
    ] = 0

    features[
        "object_ids"
    ][
        0,
        1,
    ] = 0
    features["event_type_ids"][0, 1] = 0

    features[
        "attention_mask"
    ][
        0,
    ] = torch.tensor(
        [
            1,
            0,
            1,
            1,
        ]
    )

    with pytest.raises(
        ValueError,
        match="right padding",
    ):
        model(
            **features
        )


def test_out_of_range_ids_are_rejected():
    model = ToMBeliefBackbone(
        make_config()
    )

    features = make_features()

    features[
        "action_ids"
    ][
        0,
        0,
    ] = (
        model.action_embedding
        .num_embeddings
    )

    with pytest.raises(
        ValueError,
        match="outside",
    ):
        model(
            **features
        )


def test_sequence_longer_than_limit_is_rejected():
    model = ToMBeliefBackbone(
        ToMBeliefBackboneConfig(
            d_model=16,
            n_head=4,
            n_layer=1,
            dropout=0.0,
            max_seq_len=2,
        )
    )

    ids = torch.ones(
        (
            1,
            3,
        ),
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="exceeds max_seq_len",
    ):
        model(
            subject_ids=ids,
            action_ids=ids,
            object_ids=ids,
            event_type_ids=torch.full_like(
                ids, STRUCTURED_TOKEN_TO_ID["speech_action"]
            ),
            phase_ids=torch.zeros_like(ids),
            day_values=torch.zeros_like(ids, dtype=torch.float32),
        )


def test_invalid_configuration_is_rejected():
    with pytest.raises(
        ValueError,
        match="divisible",
    ):
        ToMBeliefBackboneConfig(
            d_model=15,
            n_head=4,
        )

    with pytest.raises(
        ValueError,
        match="num_players",
    ):
        ToMBeliefBackboneConfig(
            num_players=5
        )

    with pytest.raises(
        ValueError,
        match="dropout",
    ):
        ToMBeliefBackboneConfig(
            dropout=1.0
        )

def test_no_legacy_manually_defined_embeddings():
    model = ToMBeliefBackbone(
        make_config()
    )

    forbidden_attributes = {
        "event_type_embedding",
        "observer_emb",
        "predicate_embedding",
        "role_embedding",
        "camp_embedding",
        "polarity_embedding",
        "certainty_embedding",
        "phase_embedding",
        "day_embedding",
        "wolf_head",
    }

    assert forbidden_attributes.isdisjoint(
        vars(model)
    )

    assert isinstance(
        model.subject_embedding,
        nn.Embedding,
    )

    assert isinstance(
        model.action_embedding,
        nn.Embedding,
    )

    assert isinstance(
        model.object_embedding,
        nn.Embedding,
    )


def test_forward_api_contains_only_action_features():
    parameters = tuple(
        inspect.signature(
            ToMBeliefBackbone.forward
        ).parameters
    )

    assert parameters == (
        "self",
        "subject_ids",
        "action_ids",
        "object_ids",
        "attention_mask",
        "event_type_ids",
        "phase_ids",
        "day_values",
    )

    forbidden = {
        "event_tokens",
        "roles",
        "true_roles",
        "wolf_labels",
        "truth",
        "observer_id",
        "alive_mask",
        "observation",
    }

    assert forbidden.isdisjoint(
        parameters
    )



def test_gpt2model_is_the_only_backbone():
    model = ToMBeliefBackbone(
        ToMBeliefBackboneConfig(
            d_model=16,
            n_head=4,
            n_layer=1,
            dropout=0.0,
            max_seq_len=8,
            dim_feedforward=32,
        )
    )

    assert "backbone_type" not in inspect.signature(
        ToMBeliefBackboneConfig
    ).parameters
    assert isinstance(model.transformer, GPT2Model)
    assert len(model.transformer.h) == model.config.n_layer

    source = inspect.getsource(backbone_module)
    assert "nn.TransformerEncoder" not in source
    assert "nn.TransformerEncoderLayer" not in source


def test_removed_backbone_selector_is_rejected():
    with pytest.raises(
        TypeError,
        match="backbone_type",
    ):
        ToMBeliefBackboneConfig(
            backbone_type="torch"  # type: ignore[call-arg]
        )


def test_missing_transformers_fails_without_fallback(
    monkeypatch,
):
    real_import = builtins.__import__

    def blocked_transformers_import(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("transformers unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(
        builtins,
        "__import__",
        blocked_transformers_import,
    )

    with pytest.raises(
        RuntimeError,
        match="requires the 'transformers' package",
    ):
        ToMBeliefBackbone(make_config())


def test_forward_returns_pair_logits_and_belief_marginals():
    model = ToMBeliefBackbone(
        make_config()
    ).eval()

    with torch.no_grad():
        output = model(
            **make_features()
        )

    assert output["hidden_states"].shape == (
        2,
        4,
        16,
    )
    assert output["pair_logits"].shape == (
        2,
        NUM_PLAYERS,
        NUM_WOLF_PAIR_CLASSES,
    )
    assert output["belief_matrix"].shape == (
        2,
        NUM_PLAYERS,
        NUM_PLAYERS,
    )

    torch.testing.assert_close(
        output["belief_matrix"].sum(dim=-1),
        torch.full((2, NUM_PLAYERS), 2.0),
    )


def test_model_is_causal_and_respects_right_padding():
    model = ToMBeliefBackbone(
        make_config()
    ).eval()

    features = make_features(
        batch_size=1,
    )
    changed = {
        key: value.clone()
        for key, value in features.items()
    }
    changed["object_ids"][0, 3] = (
        PLAYER_TO_ID["player6"]
    )

    with torch.no_grad():
        original = model(
            **features
        )["hidden_states"]
        modified = model(
            **changed
        )["hidden_states"]

    torch.testing.assert_close(
        original[:, :3],
        modified[:, :3],
    )

    padded = make_features()
    for field_name in (
        "subject_ids",
        "action_ids",
        "object_ids",
        "event_type_ids",
        "attention_mask",
    ):
        padded[field_name][0, 2:] = 0

    with torch.no_grad():
        padded_output = model(
            **padded
        )

    assert (
        padded_output["hidden_states"][0, 2:]
        .count_nonzero()
        .item()
        == 0
    )


def test_empty_history_is_supported():
    model = ToMBeliefBackbone(
        make_config()
    ).eval()
    zeros = torch.zeros(
        (2, 1),
        dtype=torch.long,
    )

    with torch.no_grad():
        output = model(
            subject_ids=zeros,
            action_ids=zeros,
                object_ids=zeros,
                attention_mask=zeros,
                event_type_ids=zeros,
                phase_ids=zeros,
                day_values=zeros.float(),
        )

    assert torch.isfinite(
        output["hidden_states"]
    ).all()
    assert torch.isfinite(
        output["belief_matrix"]
    ).all()
