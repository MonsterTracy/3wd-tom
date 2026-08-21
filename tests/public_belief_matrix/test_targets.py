import pytest

from werewolf.models.public_belief_matrix.targets import (
    suspicion_reports_to_matrix_target,
    suspicion_set_to_row_target,
)
from werewolf.models.twd_tom.schema import CANONICAL_PLAYER_ORDERING


def _reports():
    return [
        {
            "observer": observer,
            "status": "ok",
            "suspected_werewolves": [],
        }
        for observer in CANONICAL_PLAYER_ORDERING
    ]


def test_empty_suspicion_is_exactly_uniform():
    assert suspicion_set_to_row_target([]) == (1.0 / 7.0,) * 7


@pytest.mark.parametrize(
    ("suspects", "expected"),
    [
        (["player3"], (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
        (
            ["player3", "player5"],
            (0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0),
        ),
        (
            ["player2", "player4", "player7"],
            (0.0, 1.0 / 3.0, 0.0, 1.0 / 3.0, 0.0, 0.0, 1.0 / 3.0),
        ),
    ],
)
def test_nonempty_suspicion_is_uniform_over_exact_members(suspects, expected):
    assert suspicion_set_to_row_target(suspects) == expected
    assert sum(expected) == pytest.approx(1.0)


def test_self_suspicion_keeps_the_diagonal():
    reports = _reports()
    reports[2]["suspected_werewolves"] = ["player3"]

    result = suspicion_reports_to_matrix_target(reports)

    assert result.matrix_target[2][2] == 1.0
    assert result.observer_row_mask[2] is True


@pytest.mark.parametrize(
    "suspects",
    [["player8"], ["player3", "player3"]],
)
def test_invalid_or_duplicate_suspects_fail_closed(suspects):
    with pytest.raises(ValueError):
        suspicion_set_to_row_target(suspects)


def test_complete_valid_reports_make_canonical_seven_by_seven_matrix():
    reports = list(reversed(_reports()))

    result = suspicion_reports_to_matrix_target(reports)

    assert len(result.matrix_target) == 7
    assert all(len(row) == 7 for row in result.matrix_target)
    assert result.observer_row_mask == (True,) * 7
    assert result.matrix_target == ((1.0 / 7.0,) * 7,) * 7


@pytest.mark.parametrize("status", ["parse_error", "reporter_error"])
def test_supported_non_ok_report_has_masked_zero_placeholder(status):
    reports = _reports()
    reports[3].update(status=status, suspected_werewolves=None)

    result = suspicion_reports_to_matrix_target(reports)

    assert result.observer_row_mask[3] is False
    assert result.matrix_target[3] == (0.0,) * 7


def test_unknown_report_status_fails_closed():
    reports = _reports()
    reports[3].update(status="okk", suspected_werewolves=None)

    with pytest.raises(ValueError, match="unsupported report status"):
        suspicion_reports_to_matrix_target(reports)


@pytest.mark.parametrize("suspicion", [[], ["player3"]])
def test_non_ok_report_with_non_null_suspicion_fails_closed(suspicion):
    reports = _reports()
    reports[3].update(status="parse_error", suspected_werewolves=suspicion)

    with pytest.raises(ValueError, match="must be None"):
        suspicion_reports_to_matrix_target(reports)


@pytest.mark.parametrize("status", ["ok", "parse_error"])
def test_missing_suspected_werewolves_field_fails_closed(status):
    reports = _reports()
    reports[3]["status"] = status
    del reports[3]["suspected_werewolves"]

    with pytest.raises(ValueError, match="must contain suspected_werewolves"):
        suspicion_reports_to_matrix_target(reports)


def test_missing_observer_fails_closed():
    with pytest.raises(ValueError, match="missing observer"):
        suspicion_reports_to_matrix_target(_reports()[:-1])


def test_duplicate_observer_fails_closed():
    reports = _reports()
    reports[-1]["observer"] = "player1"
    with pytest.raises(ValueError, match="duplicate observer"):
        suspicion_reports_to_matrix_target(reports)
