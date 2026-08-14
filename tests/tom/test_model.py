from copy import deepcopy

import pytest
import torch

from werewolf.models.tom.dataset import collate_batch, encode_sample
from werewolf.models.tom.losses import masked_soft_target_cross_entropy
from werewolf.models.tom.model import (
    BeliefModel,
    DROPOUT,
    HIDDEN_SIZE,
    MAX_SEQUENCE_LENGTH,
    NUM_HEADS,
    NUM_LAYERS,
)
from werewolf.models.tom.schema import (
    ACTION_TO_ID,
    CONFIG_TO_ID,
    EVENT_TO_ID,
    NONE_ACTION_ID,
    PHASE_TO_ID,
    PLAYER_TO_ID,
)


MODEL_FIELDS = (
    "event_type_ids",
    "subject_ids",
    "action_ids",
    "object_ids",
    "phase_ids",
    "rounds",
    "dead_players",
    "config_id",
    "attention_mask",
)


def _raw_sample(*, context="seer_guard", include_prior_speech=False):
    current_actions = [["player2", "oppose", "player3"]]
    public_events = [
        {"event_idx": 0, "event_type": "death_announcement", "dead_players": []},
        {"event_idx": 1, "event_type": "phase_change", "phase": "1_day_speech"},
    ]
    if include_prior_speech:
        public_events.extend(
            [
                {
                    "event_idx": 2,
                    "event_type": "public_speech",
                    "speaker": "player1",
                    "raw_text": "not model input",
                    "sp_actions": [["player1", "support", "player2"]],
                },
                {"event_idx": 3, "event_type": "turn_start", "speaker": "player2"},
            ]
        )
    cutoff_index = len(public_events)
    public_events.append(
        {
            "event_idx": cutoff_index,
            "event_type": "public_speech",
            "speaker": "player2",
            "raw_text": "not model input",
            "sp_actions": deepcopy(current_actions),
        }
    )
    return {
        "game_id": "audit-only",
        "seed": 3,
        "episode_context": context,
        "formal_speech_actions": current_actions,
        "public_history_cutoff": {"event_idx": cutoff_index, "digest": "audit"},
        "public_events": public_events,
        "alive_observers": ["player1"],
        "observer_reports": [
            {
                "observer_id": "player1",
                "valid": True,
                "suspected_werewolves": [],
                "error": None,
            }
        ],
    }


def _model_inputs(batch):
    return {field: batch[field] for field in MODEL_FIELDS}


def _tensor_inputs(length):
    shape = (1, length)
    return {
        "event_type_ids": torch.full(
            shape, EVENT_TO_ID["speech_action"], dtype=torch.long
        ),
        "subject_ids": torch.full(
            shape, PLAYER_TO_ID["player1"], dtype=torch.long
        ),
        "action_ids": torch.full(
            shape, ACTION_TO_ID["support"], dtype=torch.long
        ),
        "object_ids": torch.full(
            shape, PLAYER_TO_ID["player2"], dtype=torch.long
        ),
        "phase_ids": torch.full(
            shape, PHASE_TO_ID["discussion"], dtype=torch.long
        ),
        "rounds": torch.ones(shape, dtype=torch.long),
        "dead_players": torch.zeros((1, length, 7), dtype=torch.bool),
        "config_id": torch.tensor([CONFIG_TO_ID["seer_guard"]]),
        "attention_mask": torch.ones(shape, dtype=torch.bool),
    }


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(41)
    return BeliefModel().eval()


def test_fixed_architecture_and_schema_cardinalities(model):
    assert HIDDEN_SIZE == 256
    assert NUM_LAYERS == len(model.layers) == 4
    assert NUM_HEADS == model.layers[0].self_attn.num_heads == 8
    assert DROPOUT == model.layers[0].dropout.p == 0.1
    assert MAX_SEQUENCE_LENGTH == model.position_embedding.num_embeddings == 256
    assert model.event_type_embedding.num_embeddings == max(EVENT_TO_ID.values()) + 1
    assert model.subject_embedding.num_embeddings == max(PLAYER_TO_ID.values()) + 1
    assert model.object_embedding.num_embeddings == max(PLAYER_TO_ID.values()) + 1
    assert model.action_embedding.num_embeddings == NONE_ACTION_ID + 1
    assert model.phase_embedding.num_embeddings == max(PHASE_TO_ID.values()) + 1
    assert model.config_embedding.num_embeddings == max(CONFIG_TO_ID.values()) + 1
    assert model.round_projection.in_features == 1
    assert model.dead_set_projection.in_features == 7
    assert model.output_projection.out_features == 49


def test_core_thirteen_none_and_padding_action_ids_are_valid(model):
    assert ACTION_TO_ID["<pad>"] == 0
    assert ACTION_TO_ID["vote_intent"] == 13
    assert NONE_ACTION_ID == 14
    assert model.action_embedding.num_embeddings == 15

    for action_id in (0, 13, 14):
        inputs = _tensor_inputs(2)
        inputs["action_ids"].fill_(action_id)
        with torch.no_grad():
            logits = model(**inputs)
            probabilities = torch.softmax(logits, dim=-1)
        assert logits.shape == (1, 7, 7)
        torch.testing.assert_close(
            probabilities.sum(dim=-1),
            torch.ones((1, 7)),
        )


def test_dataset_collate_feeds_complete_matrix_forward(model):
    batch = collate_batch([encode_sample(_raw_sample())])
    with torch.no_grad():
        logits = model(**_model_inputs(batch))
        probabilities = torch.softmax(logits, dim=-1)

    assert logits.shape == (1, 7, 7)
    assert probabilities.diagonal(dim1=1, dim2=2).shape == (1, 7)
    assert torch.all(probabilities > 0)
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones((1, 7)))
    assert batch["observer_mask"].tolist() == [
        [True, False, False, False, False, False, False]
    ]
    assert torch.isfinite(logits[:, 1:]).all()


def test_static_config_changes_logits_without_adding_a_token(model):
    guard = encode_sample(_raw_sample(context="seer_guard"))
    witch = encode_sample(_raw_sample(context="seer_witch"))
    batch = collate_batch([guard, witch])
    assert batch["sequence_length"].tolist() == [2, 2]

    with torch.no_grad():
        logits = model(**_model_inputs(batch))

    assert not torch.allclose(logits[0], logits[1])


def test_right_padding_and_its_contents_do_not_change_prediction(model):
    short = encode_sample(_raw_sample())
    long = encode_sample(_raw_sample(include_prior_speech=True))
    alone = collate_batch([short])
    padded = collate_batch([short, long])

    with torch.no_grad():
        alone_logits = model(**_model_inputs(alone))[0]
        padded_logits = model(**_model_inputs(padded))[0]
    torch.testing.assert_close(alone_logits, padded_logits, rtol=1e-5, atol=1e-6)

    changed = _model_inputs(padded)
    changed = {name: value.clone() for name, value in changed.items()}
    padding = ~changed["attention_mask"][0]
    changed["event_type_ids"][0, padding] = max(EVENT_TO_ID.values())
    changed["subject_ids"][0, padding] = max(PLAYER_TO_ID.values())
    changed["action_ids"][0, padding] = NONE_ACTION_ID
    changed["object_ids"][0, padding] = max(PLAYER_TO_ID.values())
    changed["phase_ids"][0, padding] = max(PHASE_TO_ID.values())
    changed["rounds"][0, padding] = 99
    changed["dead_players"][0, padding] = True
    with torch.no_grad():
        changed_logits = model(**changed)[0]
    torch.testing.assert_close(padded_logits, changed_logits, rtol=1e-5, atol=1e-6)


def test_position_capacity_allows_256_and_rejects_257(model):
    with torch.no_grad():
        assert model(**_tensor_inputs(256)).shape == (1, 7, 7)
    with pytest.raises(ValueError, match="257.*maximum.*256"):
        model(**_tensor_inputs(257))


def test_dataset_forward_and_loss_backpropagate_to_model_parameters():
    torch.manual_seed(42)
    model = BeliefModel().train()
    batch = collate_batch([encode_sample(_raw_sample())])
    logits = model(**_model_inputs(batch))
    loss = masked_soft_target_cross_entropy(
        logits, batch["target"], batch["observer_mask"]
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert model.output_projection.weight.grad is not None
    assert torch.isfinite(model.output_projection.weight.grad).all()
