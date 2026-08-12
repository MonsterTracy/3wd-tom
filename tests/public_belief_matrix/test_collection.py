import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from werewolf.models.public_belief_matrix.collection import (
    PUBLIC_BELIEF_MATRIX_SUPERVISION_BOUNDARY,
    PublicBeliefMatrixSampleCollector,
    validate_public_belief_matrix_sample,
)
from werewolf.models.public_belief_matrix.public_prefix import (
    build_public_belief_matrix_visible_prefix,
)
from werewolf.models.public_belief_matrix.reporter import PublicBeliefMatrixReporter
from werewolf.models.twd_tom.schema import CANONICAL_PLAYER_ORDERING


def _events(raw_text="RAW-CANARY", actions=None):
    if actions is None:
        actions = [["player2", "point_as_werewolf", "player3"]]
    return [
        {"event_idx": 0, "event_type": "phase_change", "phase": "1_day_speech"},
        {"event_idx": 1, "event_type": "turn_start", "speaker": "player2"},
        {
            "event_idx": 2,
            "event_type": "public_speech",
            "speaker": "player2",
            "raw_text": raw_text,
            "sp_actions": actions,
        },
    ]


class Backend:
    supports_json_schema = True

    def __init__(self, response='{"suspected_werewolves":[]}'):
        self.response = response
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _cutoff():
    return SimpleNamespace(
        game_id="g", step_idx=3, phase="speech", speaker_id=2,
        public_action_count=1, public_history_digest="d",
    )


def test_reporter_prompt_is_raw_text_free_and_observer_is_only_row_difference():
    first = build_public_belief_matrix_visible_prefix(_events("AAA"))
    second = build_public_belief_matrix_visible_prefix(_events("BBB"))
    prompt1 = PublicBeliefMatrixReporter.build_prompt(
        visible_prefix=first, observer_id="player1"
    )
    prompt1_again = PublicBeliefMatrixReporter.build_prompt(
        visible_prefix=second, observer_id="player1"
    )
    prompt2 = PublicBeliefMatrixReporter.build_prompt(
        visible_prefix=first, observer_id="player2"
    )
    assert prompt1 == prompt1_again
    assert "AAA" not in prompt1 and "BBB" not in prompt1
    assert prompt1.replace("observer=player1", "observer=ROW") == (
        prompt2.replace("observer=player2", "observer=ROW")
    )


def test_reporter_accepts_self_and_empty_and_preserves_failure_statuses():
    prefix = build_public_belief_matrix_visible_prefix(_events())
    backend = Backend('{"suspected_werewolves":["player1"]}')
    reporter = PublicBeliefMatrixReporter()
    result = reporter.report(
        visible_prefix=prefix, observer_id="player1", cutoff=_cutoff(),
        backend=backend, backend_id="shared", model_name="model",
    )
    assert result["status"] == "ok"
    assert result["suspected_werewolves"] == ["player1"]
    backend.response = "bad"
    assert reporter.report(
        visible_prefix=prefix, observer_id="player1", cutoff=_cutoff(),
        backend=backend, backend_id="shared", model_name="model",
    )["status"] == "parse_error"
    backend.chat = lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("down"))
    assert reporter.report(
        visible_prefix=prefix, observer_id="player1", cutoff=_cutoff(),
        backend=backend, backend_id="shared", model_name="model",
    )["status"] == "reporter_error"


def test_collector_writes_seven_symbolic_rows_including_dead_without_future(tmp_path):
    backend = Backend()
    collector = PublicBeliefMatrixSampleCollector(
        output_path=tmp_path / "raw.jsonl",
        game_id="game",
        reporter=PublicBeliefMatrixReporter(),
        reporter_dispatch={"backend": backend, "backend_id": "shared", "model_name": "m"},
    )
    env = SimpleNamespace(
        alive=[1, 0, 1, 1, 1, 1, 1],
        public_events=[
            *_events(actions=[
                ["player2", "point_as_werewolf", "player3"],
                ["player2", "point_as_werewolf", "player4"],
            ]),
            {"event_idx": 3, "event_type": "turn_start", "speaker": "player3"},
        ],
    )
    sample = collector.record(
        env, step_idx=3, trigger="speech", phase="speech", speaker_id=2
    )
    collector.close()
    validate_public_belief_matrix_sample(sample)
    assert collector.collection_timing == PUBLIC_BELIEF_MATRIX_SUPERVISION_BOUNDARY
    assert len(backend.calls) == 7
    assert [row["observer"] for row in sample["observer_reports"]] == list(CANONICAL_PLAYER_ORDERING)
    assert sample["observer_alive_mask"] == [True] * 7
    assert "RAW-CANARY" not in json.dumps(sample)
    assert "matrix_target" not in sample
    assert all('"subject":"player2"' in call["messages"][0]["content"] for call in backend.calls)
    assert all('"subject":"player3"' not in call["messages"][0]["content"] for call in backend.calls)
    saved = json.loads((tmp_path / "raw.jsonl").read_text().strip())
    assert saved["observer_reports"] == sample["observer_reports"]


def test_alive_metadata_uses_only_public_exile_and_death_events(tmp_path):
    backend = Backend()
    collector = PublicBeliefMatrixSampleCollector(
        output_path=tmp_path / "raw.jsonl",
        game_id="game",
        reporter=PublicBeliefMatrixReporter(),
        reporter_dispatch={"backend": backend, "backend_id": "shared", "model_name": "m"},
    )
    env = SimpleNamespace(
        alive=[0, 1, 1, 1, 1, 1, 1],
        public_events=[
            {"event_idx": 0, "event_type": "phase_change", "phase": "1_day_speech"},
            {"event_idx": 1, "event_type": "exile_result", "exiled_players": ["player3"]},
            {"event_idx": 2, "event_type": "death_announcement", "dead_players": ["player5"]},
            {"event_idx": 3, "event_type": "turn_start", "speaker": "player2"},
            {"event_idx": 4, "event_type": "public_speech", "speaker": "player2", "raw_text": "x", "sp_actions": []},
        ],
    )
    sample = collector.record(
        env, step_idx=4, trigger="speech", phase="speech", speaker_id=2
    )
    collector.close()
    assert sample["observer_alive_mask"] == [True, True, False, True, False, True, True]
    assert sample["publicly_alive_players"] == [
        "player1", "player2", "player4", "player6", "player7"
    ]
    assert len(sample["observer_reports"]) == 7


def test_malformed_future_event_is_not_normalized_or_collected(tmp_path):
    def collect(path, events):
        backend = Backend()
        collector = PublicBeliefMatrixSampleCollector(
            output_path=path,
            game_id="game",
            reporter=PublicBeliefMatrixReporter(),
            reporter_dispatch={"backend": backend, "backend_id": "shared", "model_name": "m"},
        )
        sample = collector.record(
            SimpleNamespace(public_events=events, alive=[0] * 7),
            step_idx=3,
            trigger="speech",
            phase="speech",
            speaker_id=2,
        )
        collector.close()
        return sample, [call["messages"] for call in backend.calls]

    events = _events()
    first, first_prompts = collect(tmp_path / "first.jsonl", events)
    second, second_prompts = collect(
        tmp_path / "second.jsonl",
        [*events, {"event_idx": 3, "event_type": "intentionally_invalid"}],
    )
    assert first["structured_prefix"] == second["structured_prefix"]
    assert first["structured_prefix_digest"] == second["structured_prefix_digest"]
    assert first_prompts == second_prompts


def test_serialized_prefix_and_reporter_provenance_fail_closed(tmp_path):
    backend = Backend()
    collector = PublicBeliefMatrixSampleCollector(
        output_path=tmp_path / "raw.jsonl",
        game_id="game",
        reporter=PublicBeliefMatrixReporter(),
        reporter_dispatch={"backend": backend, "backend_id": "shared", "model_name": "m"},
    )
    sample = collector.record(
        SimpleNamespace(public_events=_events(), alive=[1] * 7),
        step_idx=3,
        trigger="speech",
        phase="speech",
        speaker_id=2,
    )
    collector.close()

    changed_payload = deepcopy(sample)
    changed_payload["structured_prefix"]["subject_ids"][0] = 7
    with pytest.raises(ValueError, match="digest"):
        validate_public_belief_matrix_sample(changed_payload)
    changed_digest = deepcopy(sample)
    changed_digest["structured_prefix_digest"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        validate_public_belief_matrix_sample(changed_digest)
    changed_backend = deepcopy(sample)
    changed_backend["observer_reports"][0]["reporter_backend_id"] = "other"
    with pytest.raises(ValueError, match="backend provenance"):
        validate_public_belief_matrix_sample(changed_backend)
