import inspect

import pytest

import werewolf.models.tom.targets as targets_module
from werewolf.models.tom.schema import PLAYER_NAMES
from werewolf.models.tom.targets import materialize_target, suspicion_to_row


ZERO_ROW = (0.0,) * 7
UNIFORM_ROW = (1.0 / 7.0,) * 7


def report(observer, suspicion=None, *, valid=True):
    return {
        "observer_id": observer,
        "valid": valid,
        "suspected_werewolves": (
            [] if valid and suspicion is None else suspicion
        ),
        "error": None if valid else "parse_error",
    }


@pytest.mark.parametrize(
    ("suspects", "expected"),
    [
        (["player3"], (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
        (
            ["player2", "player5"],
            (0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0),
        ),
        (
            ["player1", "player4", "player7"],
            (1.0 / 3.0, 0.0, 0.0, 1.0 / 3.0, 0.0, 0.0, 1.0 / 3.0),
        ),
        (list(PLAYER_NAMES), UNIFORM_ROW),
        ([], UNIFORM_ROW),
    ],
)
def test_valid_suspicion_rows_are_normalized(suspects, expected):
    row = suspicion_to_row(suspects)
    assert row == expected
    assert sum(row) == pytest.approx(1.0)


def test_self_suspicion_preserves_diagonal_mass():
    target, mask = materialize_target(
        alive_observers=["player4"],
        observer_reports=[report("player4", ["player4"])],
    )
    assert target[3][3] == 1.0
    assert mask[3] is True


def test_valid_empty_invalid_and_dead_rows_are_distinct():
    target, mask = materialize_target(
        alive_observers=["player1", "player2"],
        observer_reports=[
            report("player1", []),
            report("player2", None, valid=False),
        ],
    )
    assert target[0] == UNIFORM_ROW
    assert mask[0] is True
    assert target[1] == ZERO_ROW
    assert mask[1] is False
    assert target[2] == ZERO_ROW
    assert mask[2] is False


def test_shape_and_canonical_row_column_order_ignore_input_order():
    alive = ["player7", "player1", "player4"]
    reports = [
        report("player4", ["player2"]),
        report("player7", ["player6"]),
        report("player1", ["player5"]),
    ]
    first = materialize_target(
        alive_observers=alive,
        observer_reports=reports,
    )
    second = materialize_target(
        alive_observers=list(reversed(alive)),
        observer_reports=list(reversed(reports)),
    )
    target, mask = first
    assert first == second
    assert len(target) == 7
    assert all(len(row) == 7 for row in target)
    assert len(mask) == 7
    assert target[0][4] == 1.0
    assert target[3][1] == 1.0
    assert target[6][5] == 1.0
    assert mask == (True, False, False, True, False, False, True)


def test_materializer_has_no_role_truth_or_hard_knowledge_input():
    assert tuple(inspect.signature(materialize_target).parameters) == (
        "alive_observers",
        "observer_reports",
    )
    inputs = {
        "alive_observers": ["player1"],
        "observer_reports": [report("player1", ["player2"])],
    }
    role_truth = {"player1": "Werewolf", "player2": "Villager"}
    before = materialize_target(**inputs)
    role_truth.update({"player1": "Villager", "player2": "Werewolf"})
    assert materialize_target(**inputs) == before
    source = inspect.getsource(targets_module)
    assert "get_twd_tom_hard_knowledge_for" not in source
    assert "known_wolves" not in source


def test_duplicate_observer_report_is_malformed():
    with pytest.raises(ValueError, match="duplicate observer"):
        materialize_target(
            alive_observers=["player1"],
            observer_reports=[report("player1"), report("player1")],
        )


def test_report_for_dead_observer_is_malformed():
    with pytest.raises(ValueError, match="dead observer"):
        materialize_target(
            alive_observers=["player1"],
            observer_reports=[report("player1"), report("player2")],
        )


def test_missing_alive_report_is_malformed_not_automatically_masked():
    with pytest.raises(ValueError, match="missing alive observer"):
        materialize_target(
            alive_observers=["player1", "player2"],
            observer_reports=[report("player1")],
        )


@pytest.mark.parametrize(
    ("alive", "reports"),
    [
        (["player8"], []),
        (["player1"], [report("player8")]),
        (["player1"], [report("player1", ["player8"])]),
    ],
)
def test_illegal_player_ids_are_malformed(alive, reports):
    with pytest.raises(ValueError):
        materialize_target(
            alive_observers=alive,
            observer_reports=reports,
        )


def test_duplicate_suspicion_ids_are_malformed():
    with pytest.raises(ValueError, match="duplicates"):
        suspicion_to_row(["player2", "player2"])


@pytest.mark.parametrize("suspicion", [[], ["player2"]])
def test_invalid_report_cannot_carry_a_suspicion(suspicion):
    with pytest.raises(ValueError, match="must be None"):
        materialize_target(
            alive_observers=["player1"],
            observer_reports=[report("player1", suspicion, valid=False)],
        )


def test_noncanonical_player_alias_is_not_silently_repaired():
    with pytest.raises(ValueError, match="canonical"):
        materialize_target(
            alive_observers=["Player_1"],
            observer_reports=[],
        )
