import inspect
import json

import pytest

from werewolf.models.twd_tom.samples import PublicSnapshot
from werewolf.models.twd_tom.schema import PLAYER_NAMES
from werewolf.speech.onuw_role_guess_perceiver import (
    OnuwRoleGuessSnapshotCollector,
    OnuwStyleRoleGuessReporter,
    parse_role_guess_response,
    role_guess_audit,
    role_guess_reports_to_matrix,
    role_guesses_to_target,
)


def guesses(*wolves):
    result = {player: "unknown" for player in PLAYER_NAMES}
    for player in wolves:
        result[player] = "werewolf"
    return result


def test_role_guess_hard_fields_and_vocab_only():
    parsed = parse_role_guess_response(
        json.dumps({"role_guesses": guesses("player1", "player2", "player3")})
    )
    audit = role_guess_audit(parsed)
    assert audit["role_count_conflict"] is True
    assert role_guesses_to_target(parsed) == pytest.approx(
        [1 / 3, 1 / 3, 1 / 3, 0, 0, 0, 0]
    )
    broken = guesses()
    broken.pop("player7")
    with pytest.raises(ValueError, match="field set mismatch"):
        parse_role_guess_response(json.dumps({"role_guesses": broken}))


def test_empty_is_full_uniform_and_self_is_not_masked():
    target = role_guesses_to_target(guesses())
    assert target == pytest.approx([1 / 7] * 7)
    self_target = role_guesses_to_target(guesses("player1"))
    assert self_target == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_matrix_keeps_dead_target_columns_but_zeros_dead_observer_rows():
    alive = list(PLAYER_NAMES[:-1])
    reports = {
        observer: {"status": "ok", "role_guesses": guesses("player7")}
        for observer in alive
    }
    matrix = role_guess_reports_to_matrix(reports, observer_ids=alive)
    assert all(row[6] == 1.0 for row in matrix[:6])
    assert matrix[6] == [0.0] * 7


def test_collector_has_no_oracle_role_parameter_and_uses_legal_observation():
    parameters = inspect.signature(OnuwRoleGuessSnapshotCollector.collect).parameters
    assert "actual_roles" not in parameters
    assert "known_werewolves" not in parameters

    class Agent:
        backend_id = "fake"

    class Env:
        def get_observation_for(self, player_id):
            return {
                "observer_id": player_id,
                "identity": "Villager",
                "current_act_idx": 1,
                "phase": "1_day_speech",
            }

    class Reporter:
        def report(self, **kwargs):
            assert kwargs["observation"]["observer_id"] == int(
                kwargs["observer_id"].removeprefix("player")
            )
            return {
                "observer": kwargs["observer_id"],
                "status": "ok",
                "role_guesses": guesses(),
            }

    snapshot = PublicSnapshot(
        game_id="g",
        step_idx=0,
        phase="1_day_speech",
        speaker_id=1,
        report_trigger="pre_public_speech",
        observer_ids=(1, 2),
        public_events=(),
        public_event_digest="e",
        speech_annotations=(),
        speech_annotation_digest="a",
        structured_input_digest="s",
        sp_actions=(),
        label_cutoff_step_idx=0,
        public_action_count=0,
        label_prompt_version="test",
    )
    collected = OnuwRoleGuessSnapshotCollector(
        Reporter(), [Agent() for _ in range(7)]
    ).collect(snapshot, env=Env())
    assert set(collected) == {"player1", "player2"}


def test_role_guess_prompt_prohibits_oracle_repair():
    class Snapshot:
        step_idx = 4
        public_history_digest = "digest"
        public_events = ()

    prompt = OnuwStyleRoleGuessReporter.build_prompt(
        observer_id=1, public_snapshot=Snapshot()
    )
    assert "自己的私人信息" in prompt
    assert "其他玩家不可见私人状态" in prompt
    assert "真实双狼数量来修复" in prompt
