import inspect

import pytest
import torch

from werewolf.models.twd_tom.belief_backbone import (
    GPT2_BLOCK_BACKBONE_NAME,
    ToMBeliefBackbone,
    ToMBeliefBackboneConfig,
    relative_player_indices,
)


def make_features(batch_size=2, sequence_length=4):
    subject_ids = torch.tensor([[1, 2, 2, 3]]).expand(batch_size, -1).clone()
    return {
        "subject_ids": subject_ids,
        "action_ids": torch.tensor([[0, 0, 5, 0]]).expand(batch_size, -1).clone(),
        "object_ids": torch.tensor([[0, 0, 4, 0]]).expand(batch_size, -1).clone(),
        "event_type_ids": torch.tensor([[1, 2, 4, 2]]).expand(batch_size, -1).clone(),
        "phase_ids": torch.ones((batch_size, sequence_length), dtype=torch.long),
        "day_values": torch.ones((batch_size, sequence_length)),
        "attention_mask": torch.ones((batch_size, sequence_length), dtype=torch.bool),
    }


def make_model():
    return ToMBeliefBackbone(
        ToMBeliefBackboneConfig(max_seq_len=16),
        backbone_name=GPT2_BLOCK_BACKBONE_NAME,
    )


def test_model_outputs_direct_observer_target_logits():
    model = make_model().eval()
    with torch.no_grad():
        output = model(**make_features())
    assert output["belief_logits"].shape == (2, 7, 7)
    assert output["observer_hidden_states"].shape == (2, 7, 256)
    assert output["relative_public_hidden_states"].shape == (2, 7, 4, 256)
    assert set(output) == {
        "hidden_states",
        "pooled_hidden_state",
        "observer_hidden_states",
        "relative_public_hidden_states",
        "belief_logits",
    }


def test_model_outputs_every_dense_pre_boundary_without_future_attention():
    model = make_model().eval()
    features = make_features()
    boundaries = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    boundary_mask = torch.ones((2, 2), dtype=torch.bool)
    with torch.no_grad():
        output = model(
            **features,
            boundary_indices=boundaries,
            boundary_valid_mask=boundary_mask,
        )
    assert output["belief_logits"].shape == (2, 2, 7, 7)
    assert output["observer_hidden_states"].shape == (2, 2, 7, 256)

    changed = make_features()
    changed["object_ids"][:, 3] = 6
    with torch.no_grad():
        changed_output = model(
            **changed,
            boundary_indices=boundaries,
            boundary_valid_mask=boundary_mask,
        )
    torch.testing.assert_close(
        output["belief_logits"][0, 0],
        changed_output["belief_logits"][0, 0],
    )


def test_dense_boundary_contract_rejects_future_or_non_monotonic_indices():
    model = make_model()
    features = make_features(batch_size=1)
    with pytest.raises(ValueError, match="right padding|event sequence"):
        model(
            **features,
            boundary_indices=torch.tensor([[0, 4]]),
            boundary_valid_mask=torch.ones((1, 2), dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        model(
            **features,
            boundary_indices=torch.tensor([[2, 1]]),
            boundary_valid_mask=torch.ones((1, 2), dtype=torch.bool),
        )


def test_model_has_no_legacy_objective_or_private_input_api():
    model = make_model()
    parameters = inspect.signature(model.forward).parameters
    assert "known_werewolves" not in parameters
    assert "known_non_werewolves" not in parameters
    assert model.output_projection.out_features == 7
    assert not hasattr(model, "tom_order")
    assert not hasattr(model.config, "pair_class_count")


def test_observer_query_uses_one_shared_attention_module():
    model = make_model()
    attentions = [module for module in model.modules() if isinstance(module, torch.nn.MultiheadAttention)]
    assert attentions == [model.observer_query_attention]


def test_observer_relative_player_indices_are_cyclic():
    ids = torch.tensor([[1, 2, 8, 0]])
    relative = relative_player_indices(ids)
    assert relative.shape == (1, 7, 4)
    assert relative[0, 0].tolist() == [0, 1, 7, 7]
    assert relative[0, 1, :2].tolist() == [6, 0]


def test_empty_public_history_is_rejected():
    model = make_model()
    features = make_features(batch_size=1)
    for value in features.values():
        value.zero_()
    with pytest.raises(ValueError, match="non-empty public history"):
        model(**features)


def test_config_rejects_non_classic_player_count():
    with pytest.raises(ValueError, match="num_players"):
        ToMBeliefBackboneConfig(num_players=8)
