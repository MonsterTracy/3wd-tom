from copy import deepcopy

import pytest
import torch

from werewolf.models.twd_tom.onuw_parity_dataset import (
    OnuwParityGameDataset,
    collate_onuw_parity_games,
    materialize_public_tokens,
    validate_parity_game,
)
from werewolf.models.twd_tom.onuw_parity_protocol import (
    ONUW_NO_FACE_NO_TONE,
)
from werewolf.models.twd_tom.onuw_parity_synthetic import synthetic_parity_games
from tests.twd_tom.public_event_fixtures import (
    make_public_events,
    make_speech_annotations,
)


def test_canonical_unit_is_game_with_multiple_nondecreasing_pre_queries():
    games = synthetic_parity_games()
    normalized = validate_parity_game(games[0])
    assert [query["token_cutoff"] for query in normalized["queries"]] == [0, 1, 1]
    dataset = OnuwParityGameDataset(games)
    assert len(dataset) == 2
    assert dataset[0]["belief_targets"].shape == (3, 7, 7)


def test_collate_uses_three_explicit_independent_masks():
    dataset = OnuwParityGameDataset(synthetic_parity_games())
    batch = collate_onuw_parity_games([dataset[0], dataset[1]])
    assert batch["token_attention_mask"].shape == (2, 3)
    assert batch["query_valid_mask"].shape == (2, 3)
    assert batch["observer_alive_mask"].shape == (2, 3, 7)
    assert batch["token_attention_mask"].data_ptr() != batch[
        "query_valid_mask"
    ].data_ptr()
    assert batch["query_valid_mask"].data_ptr() != batch[
        "observer_alive_mask"
    ].data_ptr()


def test_no_face_no_tone_zeroes_contribution_ids_not_other_class():
    game = deepcopy(synthetic_parity_games()[0])
    game["modality_profile"] = ONUW_NO_FACE_NO_TONE
    for token in game["tokens"]:
        token["face"] = None
        token["tone"] = None
    item = OnuwParityGameDataset([game])[0]
    assert torch.equal(item["face_ids"], torch.zeros_like(item["face_ids"]))
    assert torch.equal(item["tone_ids"], torch.zeros_like(item["tone_ids"]))


def test_full_multimodal_rejects_missing_emotion_and_never_truncates():
    game = deepcopy(synthetic_parity_games()[0])
    game["tokens"][1]["face"] = None
    with pytest.raises(ValueError, match="requires 8-class face"):
        validate_parity_game(game)

    long_game = deepcopy(synthetic_parity_games()[0])
    long_game["tokens"] = [long_game["tokens"][0]] + [
        deepcopy(long_game["tokens"][1]) for _ in range(300)
    ]
    long_game["queries"][-1]["token_cutoff"] = 300
    item = OnuwParityGameDataset([long_game])[0]
    assert item["token_attention_mask"].shape[0] == 301


def test_no_diagonal_or_dead_target_column_mask_is_materialized():
    item = OnuwParityGameDataset(synthetic_parity_games())[1]
    assert "diagonal_target_mask" not in item
    assert "target_candidate_mask" not in item
    assert item["observer_alive_mask"][-1, -1].item() is False
    assert item["belief_targets"][-1, 0, -1].item() > 0.0


def test_action_only_materialization_keeps_actions_and_declared_emotion_only():
    actions = [["player2", "point_as_werewolf", "player7"]]
    events = make_public_events(actions)
    annotations = make_speech_annotations(events, actions)
    tokens, counts = materialize_public_tokens(
        public_events=events,
        speech_annotations=annotations,
        speech_emotions=[
            {
                "event_idx": 2,
                "speaker": "player2",
                "face": "fear",
                "tone": "neutral",
                "source": "agent_declared",
            }
        ],
        content_profile="onuw_action_only",
        modality_profile="onuw_agent_declared_multimodal",
    )
    assert counts == [1]
    assert len(tokens) == 2
    assert tokens[1] == {
        "token_type": "speech_action",
        "subject": "player2",
        "action": "point_as_werewolf",
        "object": "player7",
        "phase": None,
        "day": 0,
        "face": "fear",
        "tone": "neutral",
    }
