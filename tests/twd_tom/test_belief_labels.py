import pytest
import torch

from werewolf.models.twd_tom.belief_labels import (
    close_hard_knowledge,
    pair_probabilities_to_belief_marginals,
    suspicion_set_to_pair_target,
)
from werewolf.models.twd_tom.schema import canonical_wolf_pairs


def _mass(target, pair):
    return target[canonical_wolf_pairs().index(tuple(pair))].item()


def _positive_pairs(target):
    pairs = canonical_wolf_pairs()
    return {pairs[index] for index in torch.nonzero(target).flatten().tolist()}


def _assert_distribution(target):
    assert target.shape == (21,)
    assert torch.isfinite(target).all()
    assert torch.all(target >= 0)
    assert target.sum().item() == pytest.approx(1.0)
    marginals = pair_probabilities_to_belief_marginals(target.repeat(7, 1))
    assert marginals.shape == (7, 7)
    assert marginals[0].sum().item() == pytest.approx(2.0)


def test_global_pair_order_is_stable_unique_and_complete():
    pairs = canonical_wolf_pairs()
    assert len(pairs) == len(set(pairs)) == 21
    assert pairs[0] == ("player1", "player2")
    assert pairs[-1] == ("player6", "player7")


def test_exactly_two_wolves_closure_and_contradiction():
    assert close_hard_knowledge(
        ["player1"], ["player3", "player4", "player5", "player6", "player7"]
    ) == (
        ["player1", "player2"],
        ["player3", "player4", "player5", "player6", "player7"],
    )
    with pytest.raises(ValueError, match="no legal"):
        close_hard_knowledge(["player1", "player2", "player3"], [])


def test_empty_suspicion_is_uniform_on_hard_legal_support():
    target = suspicion_set_to_pair_target([], [], ["player7"])
    legal_pairs = {
        pair for pair in canonical_wolf_pairs() if "player7" not in pair
    }
    assert _positive_pairs(target) == legal_pairs
    assert target[target > 0].tolist() == pytest.approx([1 / 15] * 15)
    _assert_distribution(target)


def test_one_soft_suspect_has_two_to_one_weight_without_hard_exclusion():
    target = suspicion_set_to_pair_target(["player3"], [], ["player7"])
    containing = _mass(target, ("player1", "player3"))
    excluding = _mass(target, ("player1", "player2"))
    assert containing / excluding == pytest.approx(2.0)
    assert excluding > 0.0
    assert torch.count_nonzero(target).item() == 15
    _assert_distribution(target)


def test_two_soft_suspects_have_one_two_four_weights():
    target = suspicion_set_to_pair_target(
        ["player3", "player5"], [], ["player7"]
    )
    zero_hit = _mass(target, ("player1", "player2"))
    one_hit = _mass(target, ("player1", "player3"))
    two_hit = _mass(target, ("player3", "player5"))
    assert one_hit / zero_hit == pytest.approx(2.0)
    assert two_hit / zero_hit == pytest.approx(4.0)
    assert torch.count_nonzero(target).item() == 15
    _assert_distribution(target)


def test_three_suspects_are_weighted_only_by_pair_hit_count():
    target = suspicion_set_to_pair_target(
        ["player2", "player3", "player5"], [], ["player7"]
    )
    assert _mass(target, ("player2", "player3")) == pytest.approx(
        _mass(target, ("player2", "player5"))
    )
    assert _mass(target, ("player1", "player2")) == pytest.approx(
        _mass(target, ("player1", "player3"))
    )
    assert _mass(target, ("player2", "player3")) / _mass(
        target, ("player1", "player2")
    ) == pytest.approx(2.0)
    _assert_distribution(target)


def test_full_legal_candidate_suspicion_is_rejected_before_projection():
    with pytest.raises(ValueError, match="cannot equal all legal candidates"):
        suspicion_set_to_pair_target(
            [f"player{i}" for i in range(1, 7)], [], ["player7"]
        )


def test_known_wolf_is_hard_and_not_double_weighted():
    target = suspicion_set_to_pair_target(
        ["player1", "player3"], ["player1"], ["player7"]
    )
    positive = _positive_pairs(target)
    assert all("player1" in pair and "player7" not in pair for pair in positive)
    assert _mass(target, ("player1", "player3")) / _mass(
        target, ("player1", "player2")
    ) == pytest.approx(2.0)
    _assert_distribution(target)


def test_two_known_wolves_are_one_hot():
    target = suspicion_set_to_pair_target(
        ["player2", "player5"],
        ["player2", "player5"],
        ["player1", "player3", "player4", "player6", "player7"],
    )
    assert _positive_pairs(target) == {("player2", "player5")}
    assert _mass(target, ("player2", "player5")) == pytest.approx(1.0)
    _assert_distribution(target)


def test_known_non_wolves_are_hard_zeros():
    target = suspicion_set_to_pair_target(
        ["player3"], [], ["player6", "player7"]
    )
    assert all(
        _mass(target, pair) == 0.0
        for pair in canonical_wolf_pairs()
        if "player6" in pair or "player7" in pair
    )
    assert all(
        _mass(target, pair) > 0.0
        for pair in canonical_wolf_pairs()
        if "player6" not in pair and "player7" not in pair
    )
    _assert_distribution(target)


def test_canonical_input_order_does_not_change_target_and_duplicates_fail():
    assert torch.equal(
        suspicion_set_to_pair_target(
            ["player5", "player2"], [], ["player7"]
        ),
        suspicion_set_to_pair_target(
            ["player2", "player5"], [], ["player7"]
        ),
    )
    with pytest.raises(ValueError, match="duplicate"):
        suspicion_set_to_pair_target(
            ["player2", "player2"], [], ["player7"]
        )


def test_hard_knowledge_contract_fails_closed():
    with pytest.raises(ValueError, match="contain all known"):
        suspicion_set_to_pair_target([], ["player1"], ["player7"])
    with pytest.raises(ValueError, match="known_non_werewolves"):
        suspicion_set_to_pair_target(["player7"], [], ["player7"])
    with pytest.raises(ValueError, match="no legal"):
        suspicion_set_to_pair_target(
            [],
            [],
            [
                "player2",
                "player3",
                "player4",
                "player5",
                "player6",
                "player7",
            ],
        )


def test_marginals_have_shape_and_valid_rows_sum_to_two():
    probabilities = torch.full((2, 7, 21), 1 / 21)
    marginals = pair_probabilities_to_belief_marginals(probabilities)
    assert marginals.shape == (2, 7, 7)
    assert torch.allclose(marginals.sum(-1), torch.full((2, 7), 2.0))
