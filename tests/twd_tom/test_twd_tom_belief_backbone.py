"""Tests for the single fixed Qwen2 ToM backbone."""

import inspect

import pytest
import torch
from transformers import Qwen2Model

from werewolf.models.twd_tom.belief_backbone import (
    HIDDEN_SIZE,
    NONE_RELATIVE_PLAYER_INDEX,
    ToMBeliefBackbone,
    ToMBeliefBackboneConfig,
    relative_player_indices,
)
from werewolf.models.twd_tom.public_events import STRUCTURED_TOKEN_TO_ID
from werewolf.models.twd_tom.schema import (
    ACTION_TO_ID,
    NUM_WOLF_PAIR_CLASSES,
    PLAYER_TO_ID,
)


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

    assert not hasattr(model, "tokenizer")


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


def test_qwen2_forward_accepts_all_extended_speech_actions(model):
    action_names = (
        "check_as_good",
        "check_as_werewolf",
        "save",
        "poison",
        "guard",
        "vote_intent",
    )
    features = {
        "subject_ids": torch.ones((1, 6), dtype=torch.long),
        "action_ids": torch.tensor(
            [[ACTION_TO_ID[name] for name in action_names]]
        ),
        "object_ids": torch.tensor([[2, 3, 4, 5, 6, 7]]),
        "event_type_ids": torch.full(
            (1, 6),
            STRUCTURED_TOKEN_TO_ID["speech_action"],
            dtype=torch.long,
        ),
        "phase_ids": torch.zeros((1, 6), dtype=torch.long),
        "day_values": torch.ones((1, 6), dtype=torch.float32),
        "attention_mask": torch.ones((1, 6), dtype=torch.long),
    }

    with torch.no_grad():
        output = model(**features)

    assert model.action_embedding.num_embeddings == 14
    assert output["observer_pair_logits"].shape == (1, 7, 21)


def test_output_contract_and_last_non_padding_pooling(model):
    with torch.no_grad():
        output = model(**make_features())
    assert output["hidden_states"].shape == (1, 3, HIDDEN_SIZE)
    assert output["observer_hidden_states"].shape == (1, 7, HIDDEN_SIZE)
    assert output["observer_pair_logits"].shape == (1, 7, NUM_WOLF_PAIR_CLASSES)
    assert output["pair_probabilities"].shape == (1, 7, 21)
    assert output["wolf_marginals"].shape == (1, 7, 7)
    assert "pair_logits" not in output
    assert "belief_matrix" not in output
    torch.testing.assert_close(
        output["pooled_hidden_state"], output["hidden_states"][:, 1]
    )
    assert output["hidden_states"][:, 2].count_nonzero().item() == 0
    assert "relative_public_hidden_states" not in output


def test_relative_player_indices_use_self_cyclic_and_distinct_none_indices():
    relative = relative_player_indices(
        torch.tensor([[PLAYER_TO_ID["player1"], PLAYER_TO_ID["player7"], 0]])
    )
    assert relative.shape == (1, 7, 3)
    assert relative[0, 0].tolist() == [0, 6, NONE_RELATIVE_PLAYER_INDEX]
    assert relative[0, 6].tolist() == [1, 0, NONE_RELATIVE_PLAYER_INDEX]
    assert NONE_RELATIVE_PLAYER_INDEX not in range(7)


def test_relative_player_indices_are_equivariant_to_cyclic_rotation():
    player_ids = torch.tensor([[1, 3, 7, 0]])
    shift = 4
    rotated_player_ids = torch.where(
        player_ids == 0,
        player_ids,
        ((player_ids - 1 + shift) % 7) + 1,
    )
    original = relative_player_indices(player_ids)
    rotated = relative_player_indices(rotated_player_ids)
    rotated_observer_rows = torch.roll(rotated, shifts=-shift, dims=1)
    torch.testing.assert_close(rotated_observer_rows, original)


def test_second_order_routes_speaker_subject_and_object_relations():
    second_order = ToMBeliefBackbone(
        ToMBeliefBackboneConfig(max_seq_len=8),
        tom_order=2,
    ).eval()
    observed = {}

    def capture(name):
        def hook(_module, args):
            observed[name] = args[0].detach().clone()

        return hook

    handles = [
        second_order.second_order_speaker_relative_embedding.register_forward_pre_hook(
            capture("speaker")
        ),
        second_order.second_order_subject_relative_embedding.register_forward_pre_hook(
            capture("subject")
        ),
        second_order.second_order_object_relative_embedding.register_forward_pre_hook(
            capture("object")
        ),
    ]
    features = {
        "subject_ids": torch.tensor([[1, 2, 3]]),
        "action_ids": torch.tensor([[0, ACTION_TO_ID["support"], 0]]),
        "object_ids": torch.tensor([[0, 4, 5]]),
        "event_type_ids": torch.tensor(
            [[
                STRUCTURED_TOKEN_TO_ID["turn_start"],
                STRUCTURED_TOKEN_TO_ID["speech_action"],
                STRUCTURED_TOKEN_TO_ID["vote"],
            ]]
        ),
        "phase_ids": torch.zeros((1, 3), dtype=torch.long),
        "day_values": torch.zeros((1, 3), dtype=torch.float32),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
    }
    try:
        with torch.no_grad():
            output = second_order(**features)
    finally:
        for handle in handles:
            handle.remove()

    assert observed["speaker"][0, 0].tolist() == [0, 7, 7]
    assert observed["subject"][0, 0].tolist() == [7, 1, 2]
    assert observed["object"][0, 0].tolist() == [7, 3, 4]
    assert output["relative_public_hidden_states"].shape == (
        1,
        7,
        3,
        HIDDEN_SIZE,
    )


def test_second_order_uses_the_same_single_pair_output_projection():
    second_order = ToMBeliefBackbone(
        ToMBeliefBackboneConfig(max_seq_len=8),
        tom_order=2,
    ).eval()
    with torch.no_grad():
        output = second_order(**make_features())
    assert second_order.output_projection.out_features == NUM_WOLF_PAIR_CLASSES
    assert output["observer_pair_logits"].shape == (1, 7, 21)
    assert output["pair_probabilities"].shape == (1, 7, 21)
    assert output["wolf_marginals"].shape == (1, 7, 7)
    assert "pair_logits" not in output
    assert "belief_matrix" not in output
    torch.testing.assert_close(
        output["pair_probabilities"].sum(dim=-1),
        torch.ones((1, 7)),
    )
    torch.testing.assert_close(
        output["wolf_marginals"].sum(dim=-1),
        torch.full((1, 7), 2.0),
    )
    assert "observer_suspicion_logits" not in output
    assert second_order.config.enable_suspicion_aux is False
    assert not hasattr(second_order, "suspicion_projection")
    assert not hasattr(second_order, "pair_output_projection")
    assert not hasattr(second_order, "suspicion_output_projection")


def test_second_order_optional_suspicion_head_reuses_observer_hidden_states():
    second_order = ToMBeliefBackbone(
        ToMBeliefBackboneConfig(
            max_seq_len=8,
            enable_suspicion_aux=True,
        ),
        tom_order=2,
    ).eval()
    captured = {}

    def capture(_module, args):
        captured["input"] = args[0].detach().clone()

    handle = second_order.suspicion_projection.register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            output = second_order(**make_features())
    finally:
        handle.remove()

    assert output["observer_suspicion_logits"].shape == (1, 7, 7)
    torch.testing.assert_close(captured["input"], output["observer_hidden_states"])
    assert second_order.suspicion_projection.in_features == HIDDEN_SIZE
    assert second_order.suspicion_projection.out_features == 7


def test_suspicion_auxiliary_head_rejects_first_order_model():
    with pytest.raises(ValueError, match="requires tom_order=2"):
        ToMBeliefBackbone(
            ToMBeliefBackboneConfig(
                max_seq_len=8,
                enable_suspicion_aux=True,
            ),
            tom_order=1,
        )


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
    assert observed["query"].shape == (7, 1, HIDDEN_SIZE)
    assert observed["key"].shape == (7, 3, HIDDEN_SIZE)
    assert observed["value"].shape == (7, 3, HIDDEN_SIZE)
    assert torch.equal(
        observed["key_padding_mask"],
        torch.tensor([[False, False, True]]).expand(7, -1),
    )
    assert observed["need_weights"] is False
    assert output["relative_public_hidden_states"].shape == (
        1,
        7,
        3,
        HIDDEN_SIZE,
    )
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
    for field in (
        "second_order_speaker_relative_embedding",
        "second_order_subject_relative_embedding",
        "second_order_object_relative_embedding",
        "second_order_relation_flag_projection",
    ):
        assert hasattr(second_order, field)
        assert not hasattr(model, field)


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
    assert set(parameters) == {
        "num_players",
        "pair_class_count",
        "max_seq_len",
        "enable_suspicion_aux",
    }
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
