"""Tests for the single fixed Qwen2 ToM backbone."""

import inspect

import pytest
import torch
from transformers import Qwen2Model

import werewolf.models.twd_tom.belief_backbone as backbone_module
from werewolf.models.twd_tom.belief_backbone import (
    HIDDEN_SIZE,
    ToMBeliefBackbone,
    ToMBeliefBackboneConfig,
)
from werewolf.models.twd_tom.public_events import STRUCTURED_TOKEN_TO_ID
from werewolf.models.twd_tom.schema import NUM_WOLF_PAIR_CLASSES, PLAYER_TO_ID


def make_features():
    return {
        "subject_ids": torch.tensor([[PLAYER_TO_ID["player1"], 0, 0]]),
        "action_ids": torch.zeros((1, 3), dtype=torch.long),
        "object_ids": torch.zeros((1, 3), dtype=torch.long),
        "event_type_ids": torch.tensor(
            [[STRUCTURED_TOKEN_TO_ID["turn_start"],
              STRUCTURED_TOKEN_TO_ID["public_speech"], 0]]
        ),
        "phase_ids": torch.zeros((1, 3), dtype=torch.long),
        "day_values": torch.zeros((1, 3), dtype=torch.float32),
        "attention_mask": torch.tensor([[1, 1, 0]]),
    }


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(1)
    return ToMBeliefBackbone(ToMBeliefBackboneConfig(max_seq_len=8)).eval()


def test_qwen2model_is_the_only_decoder_with_fixed_configuration(model):
    assert isinstance(model.transformer, Qwen2Model)
    config = model.transformer.config
    assert config.vocab_size == 1
    assert config.hidden_size == 256
    assert config.intermediate_size == 768
    assert config.num_hidden_layers == 4
    assert config.num_attention_heads == 8
    assert config.num_key_value_heads == 4
    assert config.hidden_act == "silu"
    assert config.attention_dropout == pytest.approx(0.1)
    assert config.rms_norm_eps == pytest.approx(1e-6)
    assert config.use_cache is False
    assert config.max_position_embeddings == 8

    source = inspect.getsource(backbone_module)
    assert "Qwen2DecoderLayer" not in source
    assert "Qwen2ForCausalLM" not in source
    assert "from_pretrained" not in source


def test_qwen2_receives_inputs_embeds_and_attention_mask(model):
    observed = {}

    def capture(_module, _args, kwargs):
        observed.update(kwargs)

    handle = model.transformer.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        with torch.no_grad():
            model(**make_features())
    finally:
        handle.remove()
    assert observed["inputs_embeds"].shape == (1, 3, HIDDEN_SIZE)
    assert observed["attention_mask"].shape == (1, 3)
    assert "input_ids" not in observed


def test_output_contract_and_last_non_padding_pooling(model):
    with torch.no_grad():
        output = model(**make_features())
    assert output["hidden_states"].shape == (1, 3, HIDDEN_SIZE)
    assert output["observer_hidden_states"].shape == (1, 7, HIDDEN_SIZE)
    assert output["observer_pair_logits"].shape == (1, 7, NUM_WOLF_PAIR_CLASSES)
    assert output["observer_pair_logits"] is output["pair_logits"]
    assert output["pair_logits"].shape == (1, 7, NUM_WOLF_PAIR_CLASSES)
    assert output["pair_probabilities"].shape == (1, 7, 21)
    assert output["belief_matrix"].shape == (1, 7, 7)
    torch.testing.assert_close(
        output["pooled_hidden_state"], output["hidden_states"][:, 1]
    )
    assert output["hidden_states"][:, 2].count_nonzero().item() == 0


def test_second_order_uses_the_same_single_pair_output_projection():
    second_order = ToMBeliefBackbone(
        ToMBeliefBackboneConfig(max_seq_len=8),
        tom_order=2,
    ).eval()
    with torch.no_grad():
        output = second_order(**make_features())
    assert second_order.output_projection.out_features == NUM_WOLF_PAIR_CLASSES
    assert output["observer_pair_logits"].shape == (1, 7, 21)
    assert output["observer_pair_logits"] is output["pair_logits"]
    assert output["pair_probabilities"].shape == (1, 7, 21)
    assert output["belief_matrix"].shape == (1, 7, 7)
    torch.testing.assert_close(
        output["pair_probabilities"].sum(dim=-1),
        torch.ones((1, 7)),
    )
    torch.testing.assert_close(
        output["belief_matrix"].sum(dim=-1),
        torch.full((1, 7), 2.0),
    )
    assert "observer_suspicion_logits" not in output
    assert not hasattr(second_order, "pair_output_projection")
    assert not hasattr(second_order, "suspicion_output_projection")


def test_second_order_observer_query_attention_shapes_and_padding_mask():
    second_order = ToMBeliefBackbone(
        ToMBeliefBackboneConfig(max_seq_len=8),
        tom_order=2,
    ).eval()
    observed = {}

    def capture(_module, _args, kwargs):
        observed.update(kwargs)

    handle = second_order.second_order_observer_query_attention.register_forward_pre_hook(
        capture,
        with_kwargs=True,
    )
    try:
        with torch.no_grad():
            output = second_order(**make_features())
    finally:
        handle.remove()
    assert observed["query"].shape == (1, 7, HIDDEN_SIZE)
    assert observed["key"].shape == (1, 3, HIDDEN_SIZE)
    assert observed["value"].shape == (1, 3, HIDDEN_SIZE)
    assert torch.equal(
        observed["key_padding_mask"],
        torch.tensor([[False, False, True]]),
    )
    assert observed["need_weights"] is False
    assert output["observer_hidden_states"].shape == (1, 7, HIDDEN_SIZE)
    assert output["observer_pair_logits"].shape == (1, 7, 21)


def test_second_order_uses_one_shared_attention_and_head():
    second_order = ToMBeliefBackbone(
        ToMBeliefBackboneConfig(max_seq_len=8),
        tom_order=2,
    )
    attentions = [
        module
        for module in second_order.modules()
        if isinstance(module, torch.nn.MultiheadAttention)
    ]
    assert attentions == [second_order.second_order_observer_query_attention]
    assert second_order.output_projection.out_features == 21
    assert not hasattr(model, "second_order_observer_query_attention")


def test_second_order_rejects_all_padding_before_query_attention(model):
    second_order = ToMBeliefBackbone(
        ToMBeliefBackboneConfig(max_seq_len=8),
        tom_order=2,
    )
    features = {
        name: torch.zeros_like(value)
        for name, value in make_features().items()
    }
    with pytest.raises(ValueError, match="non-empty public history"):
        second_order(**features)
    with torch.no_grad():
        first_order = model(**features)
    assert first_order["observer_pair_logits"].shape == (1, 7, 21)


def test_first_order_private_projection_changes_only_its_observer(model):
    features = make_features()
    wolves = torch.zeros((1, 7, 7))
    non_wolves = torch.zeros_like(wolves)
    wolves[0, 2, 6] = 1
    non_wolves[0, 2, 2] = 1
    with torch.no_grad():
        public_only = model(**features)["observer_hidden_states"]
        first_order = model(
            **features,
            known_werewolves=wolves,
            known_non_werewolves=non_wolves,
        )["observer_hidden_states"]
    assert not torch.equal(public_only[:, 2], first_order[:, 2])
    torch.testing.assert_close(public_only[:, :2], first_order[:, :2])
    torch.testing.assert_close(public_only[:, 3:], first_order[:, 3:])


def test_private_inputs_are_strict_and_optional(model):
    with pytest.raises(ValueError, match="provided together"):
        model(**make_features(), known_werewolves=torch.zeros((1, 7, 7)))
    bad = torch.zeros((1, 7, 7))
    bad[0, 0, 0] = 0.5
    with pytest.raises(ValueError, match="only 0 or 1"):
        model(
            **make_features(),
            known_werewolves=bad,
            known_non_werewolves=torch.zeros_like(bad),
        )


def test_second_order_rejects_private_inputs():
    model = ToMBeliefBackbone(
        ToMBeliefBackboneConfig(max_seq_len=8),
        tom_order=2,
    )
    private = torch.zeros((1, 7, 7))
    with pytest.raises(ValueError, match="does not accept private knowledge"):
        model(
            **make_features(),
            known_werewolves=private,
            known_non_werewolves=private,
        )


def test_gpt2_configuration_fields_are_removed():
    parameters = inspect.signature(ToMBeliefBackboneConfig).parameters
    assert set(parameters) == {"num_players", "pair_class_count", "max_seq_len"}
    with pytest.raises(TypeError):
        ToMBeliefBackboneConfig(d_model=16)  # type: ignore[call-arg]


def test_backbone_is_causal(model):
    features = make_features()
    changed = {name: value.clone() for name, value in features.items()}
    changed["subject_ids"][0, 1] = PLAYER_TO_ID["player7"]
    with torch.no_grad():
        first = model(**features)["hidden_states"]
        second = model(**changed)["hidden_states"]
    torch.testing.assert_close(first[:, 0], second[:, 0])


def test_sequence_limit_and_right_padding_are_strict(model):
    features = make_features()
    features["event_type_ids"] = torch.tensor(
        [[STRUCTURED_TOKEN_TO_ID["turn_start"], 0,
          STRUCTURED_TOKEN_TO_ID["public_speech"]]]
    )
    features["attention_mask"] = torch.tensor([[1, 0, 1]])
    with pytest.raises(ValueError, match="right padding"):
        model(**features)
    short = ToMBeliefBackbone(ToMBeliefBackboneConfig(max_seq_len=2))
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        short(**make_features())
