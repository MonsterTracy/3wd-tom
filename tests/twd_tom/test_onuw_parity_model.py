import pytest
import torch

from werewolf.models.twd_tom.onuw_parity_dataset import (
    OnuwParityGameDataset,
    collate_onuw_parity_games,
)
from werewolf.models.twd_tom.onuw_parity_model import (
    HIDDEN_SIZE,
    NUM_HEADS,
    NUM_LAYERS,
    OnuwParityBeliefModel,
    OnuwParityModelConfig,
)
from werewolf.models.twd_tom.onuw_parity_protocol import (
    ONUW_ACTION_ONLY,
    ONUW_AGENT_DECLARED_MULTIMODAL,
)
from werewolf.models.twd_tom.onuw_parity_synthetic import synthetic_parity_games


def _features(batch):
    return {
        key: batch[key]
        for key in (
            "subject_ids",
            "action_ids",
            "object_ids",
            "token_type_ids",
            "face_ids",
            "tone_ids",
            "phase_ids",
            "day_values",
            "token_attention_mask",
            "query_positions",
            "query_valid_mask",
            "observer_alive_mask",
        )
    }


def test_reference_shape_is_512_8_8_and_direct_full_matrix():
    assert (HIDDEN_SIZE, NUM_LAYERS, NUM_HEADS) == (512, 8, 8)
    dataset = OnuwParityGameDataset(synthetic_parity_games())
    batch = collate_onuw_parity_games([dataset[0], dataset[1]])
    model = OnuwParityBeliefModel(
        OnuwParityModelConfig(
            max_positions=3,
            content_profile=ONUW_ACTION_ONLY,
            modality_profile=ONUW_AGENT_DECLARED_MULTIMODAL,
        )
    )
    model.eval()
    with torch.no_grad():
        logits = model(**_features(batch))
    assert logits.shape == (2, 3, 7, 7)
    assert model.matrix_head.out_features == 49
    assert not hasattr(model, "observer_query")


def test_model_refuses_capacity_overflow_instead_of_truncating():
    dataset = OnuwParityGameDataset(synthetic_parity_games())
    batch = collate_onuw_parity_games([dataset[1]])
    model = OnuwParityBeliefModel(
        OnuwParityModelConfig(
            max_positions=2,
            content_profile=ONUW_ACTION_ONLY,
            modality_profile=ONUW_AGENT_DECLARED_MULTIMODAL,
        )
    )
    with pytest.raises(ValueError, match="silent truncation is forbidden"):
        model(**_features(batch))
