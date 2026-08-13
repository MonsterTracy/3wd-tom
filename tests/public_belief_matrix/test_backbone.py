import pytest
import torch

from werewolf.models.public_belief_matrix.backbone import (
    HIDDEN_SIZE,
    NUM_ATTENTION_HEADS,
    NUM_HIDDEN_LAYERS,
    PublicBeliefMatrixBackbone,
    PublicBeliefMatrixBackboneConfig,
)
from werewolf.models.public_belief_matrix.public_prefix import (
    build_public_belief_matrix_visible_prefix,
)
from werewolf.models.twd_tom.belief_backbone import GPT2BlockStack
from werewolf.models.twd_tom.public_events import PHASE_TO_ID, STRUCTURED_TOKEN_TO_ID
from werewolf.models.twd_tom.schema import ACTION_TO_ID, PLAYER_TO_ID


def _features(attention_mask=None):
    if attention_mask is None:
        attention_mask = [[1, 1, 0]]
    shape = (len(attention_mask), len(attention_mask[0]))
    return {
        "subject_ids": torch.full(
            shape, PLAYER_TO_ID["player1"], dtype=torch.long
        ),
        "action_ids": torch.zeros(shape, dtype=torch.long),
        "object_ids": torch.zeros(shape, dtype=torch.long),
        "event_type_ids": torch.full(
            shape,
            STRUCTURED_TOKEN_TO_ID["turn_start"],
            dtype=torch.long,
        ),
        "phase_ids": torch.zeros(shape, dtype=torch.long),
        "day_values": torch.zeros(shape),
        "attention_mask": torch.tensor(attention_mask),
    }


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(11)
    return PublicBeliefMatrixBackbone(
        PublicBeliefMatrixBackboneConfig(max_seq_len=8)
    ).eval()


def test_backbone_has_fixed_gpt2_capacity_and_matrix_output(model):
    assert isinstance(model.transformer, GPT2BlockStack)
    assert len(model.transformer.blocks) == NUM_HIDDEN_LAYERS == 4
    assert model.transformer.blocks[0].attn.num_heads == NUM_ATTENTION_HEADS == 8
    assert model.matrix_projection.in_features == HIDDEN_SIZE == 256
    assert model.matrix_projection.out_features == 49

    with torch.no_grad():
        output = model(**_features())

    assert output["matrix_logits"].shape == (1, 7, 7)
    assert output["matrix_probabilities"].shape == (1, 7, 7)
    torch.testing.assert_close(
        output["matrix_probabilities"].sum(dim=-1),
        torch.ones(1, 7),
    )
    assert {
        "observer_pair_logits",
        "pair_probabilities",
        "observer_hidden_states",
        "relative_public_hidden_states",
    }.isdisjoint(output)


def test_embedding_cardinalities_cover_canonical_id_spaces(model):
    assert (
        model.subject_embedding.num_embeddings
        == max(PLAYER_TO_ID.values()) + 1
        == 8
    )
    assert (
        model.action_embedding.num_embeddings
        == max(ACTION_TO_ID.values()) + 1
        == 14
    )
    assert (
        model.object_embedding.num_embeddings
        == max(PLAYER_TO_ID.values()) + 1
        == 8
    )
    assert model.event_type_embedding.num_embeddings == max(
        STRUCTURED_TOKEN_TO_ID.values()
    ) + 1
    assert model.event_type_embedding.num_embeddings == 11
    assert (
        model.phase_embedding.num_embeddings
        == max(PHASE_TO_ID.values()) + 1
        == 6
    )
    assert all(
        embedding.padding_idx == 0
        for embedding in (
            model.subject_embedding,
            model.action_embedding,
            model.object_embedding,
            model.event_type_embedding,
            model.phase_embedding,
        )
    )


def test_maximum_canonical_ids_and_padding_can_forward(model):
    features = _features([[1, 1]])
    features.update(
        {
            "subject_ids": torch.tensor([[max(PLAYER_TO_ID.values()), 0]]),
            "action_ids": torch.tensor([[max(ACTION_TO_ID.values()), 0]]),
            "object_ids": torch.tensor([[max(PLAYER_TO_ID.values()), 0]]),
            "event_type_ids": torch.tensor(
                [[max(STRUCTURED_TOKEN_TO_ID.values()), 0]]
            ),
            "phase_ids": torch.tensor([[max(PHASE_TO_ID.values()), 0]]),
        }
    )

    with torch.no_grad():
        output = model(**features)

    assert output["matrix_logits"].shape == (1, 7, 7)
    assert output["matrix_probabilities"].shape == (1, 7, 7)


@pytest.mark.parametrize(
    ("field", "invalid_id"),
    [
        ("event_type_ids", max(STRUCTURED_TOKEN_TO_ID.values()) + 1),
        ("phase_ids", max(PHASE_TO_ID.values()) + 1),
    ],
)
def test_id_above_canonical_maximum_fails_closed(model, field, invalid_id):
    features = _features([[1]])
    features[field] = torch.tensor([[invalid_id]])

    with pytest.raises(ValueError, match=rf"{field} contains an out-of-range ID"):
        model(**features)


def test_stage4_batch_style_max_event_and_phase_ids_can_forward(model):
    features = _features([[1, 1, 1], [1, 1, 0]])
    features["event_type_ids"] = torch.tensor(
        [[1, 10, 3], [10, 2, 0]], dtype=torch.long
    )
    features["phase_ids"] = torch.tensor(
        [[1, 5, 1], [5, 2, 0]], dtype=torch.long
    )

    with torch.no_grad():
        output = model(**features)

    assert output["matrix_logits"].shape == (2, 7, 7)
    assert output["matrix_probabilities"].shape == (2, 7, 7)


def test_stage1_visible_prefix_feeds_stage2_backbone_directly(model):
    public_events = [
        {
            "event_idx": 0,
            "event_type": "phase_change",
            "phase": "1_day_speech",
        },
        {
            "event_idx": 1,
            "event_type": "turn_start",
            "speaker": "player1",
        },
        {
            "event_idx": 2,
            "event_type": "public_speech",
            "speaker": "player1",
            "raw_text": "player3 looks suspicious",
            "sp_actions": [
                ["player1", "point_as_werewolf", "player3"]
            ],
        },
    ]
    prefix = build_public_belief_matrix_visible_prefix(
        public_events,
        max_seq_len=8,
    )
    batched = {key: value.unsqueeze(0) for key, value in prefix.items()}

    with torch.no_grad():
        output = model(**batched)

    assert output["matrix_logits"].shape == (1, 7, 7)
    assert output["matrix_probabilities"].shape == (1, 7, 7)
    torch.testing.assert_close(
        output["matrix_probabilities"].sum(dim=-1),
        torch.ones(1, 7),
    )


def test_diagonal_is_a_normal_output_position():
    torch.manual_seed(12)
    model = PublicBeliefMatrixBackbone(
        PublicBeliefMatrixBackboneConfig(max_seq_len=8)
    ).eval()
    with torch.no_grad():
        model.matrix_projection.weight.zero_()
        model.matrix_projection.bias.zero_()
        for player_index in range(7):
            model.matrix_projection.bias[player_index * 7 + player_index] = 10
        output = model(**_features())

    assert torch.all(output["matrix_probabilities"].diagonal(dim1=1, dim2=2) > 0.99)


def test_padding_pooling_uses_last_valid_not_physical_last_position(model):
    with torch.no_grad():
        output = model(**_features([[1, 1, 0, 0]]))

    torch.testing.assert_close(
        output["pooled_hidden_state"], output["hidden_states"][:, 1]
    )
    assert output["hidden_states"][:, 2:].count_nonzero().item() == 0


def test_empty_history_fails_closed(model):
    with pytest.raises(ValueError, match="valid token"):
        model(**_features([[0, 0, 0]]))


def test_same_seed_produces_identical_parameters():
    torch.manual_seed(1234)
    first = PublicBeliefMatrixBackbone(
        PublicBeliefMatrixBackboneConfig(max_seq_len=8)
    )
    torch.manual_seed(1234)
    second = PublicBeliefMatrixBackbone(
        PublicBeliefMatrixBackboneConfig(max_seq_len=8)
    )

    assert first.state_dict().keys() == second.state_dict().keys()
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name])


def test_eval_forward_is_deterministic(model):
    features = _features()
    with torch.no_grad():
        first = model(**features)["matrix_logits"]
        second = model(**features)["matrix_logits"]
    torch.testing.assert_close(first, second)
