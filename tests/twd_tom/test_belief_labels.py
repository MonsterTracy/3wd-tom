import pytest
import torch

from werewolf.models.twd_tom.belief_labels import (
    close_hard_knowledge,
    suspicion_set_to_belief_vector,
)


def test_empty_suspicion_is_uniform_over_six_non_self_targets():
    target = suspicion_set_to_belief_vector([], observer_id="player7")
    assert target.shape == (7,)
    assert target[6].item() == 0.0
    assert target[:6].tolist() == pytest.approx([1.0 / 6.0] * 6)
    assert target.sum().item() == pytest.approx(1.0)


def test_non_empty_suspicion_is_uniform_only_over_suspected_players():
    target = suspicion_set_to_belief_vector(
        ["player3", "player5"],
        observer_id="player1",
        dtype=torch.float64,
    )
    assert target.tolist() == pytest.approx([0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0])
    assert target.sum().item() == pytest.approx(1.0)


def test_three_suspected_players_receive_exact_sparse_thirds():
    target = suspicion_set_to_belief_vector(
        ["player2", "player5", "player6"],
        observer_id="player1",
        dtype=torch.float64,
    )
    assert target.tolist() == pytest.approx(
        [0.0, 1.0 / 3.0, 0.0, 0.0, 1.0 / 3.0, 1.0 / 3.0, 0.0]
    )


def test_conversion_is_canonical_and_deterministic():
    first = suspicion_set_to_belief_vector(
        ["player5", "player2"], observer_id=3
    )
    second = suspicion_set_to_belief_vector(
        ["player2", "player5"], observer_id="player3"
    )
    torch.testing.assert_close(first, second)


@pytest.mark.parametrize(
    "value",
    ["player2", ["player8"], ["player2", "player2"], [2]],
)
def test_conversion_rejects_non_set_or_non_canonical_members(value):
    with pytest.raises((TypeError, ValueError)):
        suspicion_set_to_belief_vector(value, observer_id="player1")


def test_self_suspicion_is_rejected_by_the_observer_legality_contract():
    with pytest.raises(ValueError, match="cannot contain the observer"):
        suspicion_set_to_belief_vector(["player4"], observer_id="player4")


def test_conversion_requires_floating_dtype():
    with pytest.raises(TypeError, match="floating-point"):
        suspicion_set_to_belief_vector(
            [], observer_id="player1", dtype=torch.int64
        )


def test_hard_knowledge_closure_remains_available_to_validate_raw_reports():
    known_wolves, known_non_wolves = close_hard_knowledge(
        ["player1", "player2"],
        [],
    )
    assert known_wolves == ["player1", "player2"]
    assert known_non_wolves == [
        "player3",
        "player4",
        "player5",
        "player6",
        "player7",
    ]
