from werewolf.models.twd_tom.onuw_parity_recorder import OnuwParityGameRecorder
from werewolf.models.twd_tom.onuw_parity_audit import pilot_collection_audit
from werewolf.models.twd_tom.schema import PLAYER_NAMES
from tests.twd_tom.public_event_fixtures import (
    make_public_events,
    make_speech_annotations,
)


def _reports(support=()):
    result = {}
    for observer in PLAYER_NAMES:
        guesses = {player: "unknown" for player in PLAYER_NAMES}
        for player in support:
            guesses[player] = "werewolf"
        result[observer] = {
            "observer": observer,
            "status": "ok",
            "role_guesses": guesses,
        }
    return result


def test_recorder_keeps_strict_pre_cutoff_before_post_speech_tokens():
    recorder = OnuwParityGameRecorder(
        game_id="new_parity_pilot_001",
        content_profile="onuw_action_only",
        modality_profile="onuw_agent_declared_multimodal",
    )
    recorder.record_pre(
        step_idx=0,
        speaker_id=1,
        observer_ids=range(1, 8),
        role_guess_reports=_reports(),
    )
    actions = [["player1", "point_as_werewolf", "player7"]]
    events = make_public_events(actions, speaker_id=1)
    annotations = make_speech_annotations(events, actions)
    recorder.sync_post_public_history(
        public_events=events,
        speech_annotations=annotations,
        speech_emotions=[
            {
                "event_idx": 2,
                "speaker": "player1",
                "face": "neutral",
                "tone": "other",
                "source": "agent_declared",
            }
        ],
    )
    recorder.record_pre(
        step_idx=1,
        speaker_id=2,
        observer_ids=range(1, 8),
        role_guess_reports=_reports(("player7",)),
    )
    game = recorder.finalize()
    assert [query["token_cutoff"] for query in game["queries"]] == [0, 1]
    assert len(game["tokens"]) == 2
    assert game["speech_action_counts"] == [1]
    audit = recorder.finalize_collection_audit()
    assert audit["model_input"] == "public_only"
    assert len(audit["queries"]) == 2
    assert "role_guess_reports" not in game
    assert audit["queries"][0]["role_guess_reports"]["player1"][
        "role_guesses"
    ]["player1"] == "unknown"
    stats = pilot_collection_audit([game], [audit])
    assert stats["query_count"] == 2
    assert stats["role_report_count"] == 14
    assert stats["support_size_histogram"] == {"0": 7, "1": 7}
    assert stats["declared_emotion_coverage"] == 1.0
