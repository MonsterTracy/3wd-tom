import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from werewolf.agents.llm_agent import LLMAgent
from werewolf.models.twd_tom.actor_perspective import (
    ACTOR_PERSPECTIVE_TARGET_SCHEMA_VERSION,
    build_actor_perspective_sample,
)
from werewolf.models.twd_tom.collector import build_collection_provenance
from werewolf.models.twd_tom.samples import (
    ACTOR_PAIR_BELIEF_SCHEMA_VERSION,
    freeze_public_snapshot,
    make_twd_tom_sample,
)
from werewolf.models.twd_tom.schema import (
    PLAYER_NAMES,
    canonical_wolf_pairs,
)
from werewolf.runtime_config import normalize_runtime_config
from werewolf.speech.pair_belief_self_reporter import (
    PAIR_BELIEF_PROMPT_VERSION,
    PROBABILITY_ATOL,
    ReadonlyPairBeliefSelfReporter,
    canonical_json_sha256,
    validate_pair_probabilities,
)


OLD_TOM2_SHA256 = {
    "data/qwen25/raw_tom2.jsonl": (
        "380004dfef3f731accbd8a1bc83fdbcb290c7bce50e30030b31d29e6b2e3256c"
    ),
    "data/qwen25/tom2/train.jsonl": (
        "31b99dc26ba9724c6ee8390a184b01faca50ec830313fcac3e5c873e5fa1e1a8"
    ),
    "data/qwen25/tom2/val.jsonl": (
        "5b7f9206e4a8a59938a5aba50ba68f0d7c6cd9079b377d1a25aedd349151f0c8"
    ),
    "data/qwen25/tom2/test.jsonl": (
        "91622732b35dc0a6ca17dcb4b74292514306db128d9a2a9ee1f9538df2c03a6d"
    ),
}
REPO_ROOT = Path(__file__).resolve().parents[2]


def _events(speaker="player3"):
    return [
        {"event_idx": 0, "event_type": "phase_change", "phase": "1_day_speech"},
        {"event_idx": 1, "event_type": "turn_start", "speaker": "player1"},
        {
            "event_idx": 2,
            "event_type": "public_speech",
            "speaker": "player1",
            "raw_text": "earlier public speech",
            "sp_actions": [["player1", "oppose", "player2"]],
        },
        {"event_idx": 3, "event_type": "turn_start", "speaker": speaker},
    ]


def _snapshot(*, speaker=3, alive=range(1, 8)):
    return freeze_public_snapshot(
        game_id="synthetic_game_seed_42",
        step_idx=7,
        phase="1_day_speech",
        speaker_id=speaker,
        report_trigger="pre_public_speech",
        observer_ids=list(alive),
        public_events=_events(f"player{speaker}"),
    )


def _target(*, known_wolves=(), known_non_wolves=(), weights=None):
    legal = [
        set(known_wolves).issubset(pair)
        and set(pair).isdisjoint(known_non_wolves)
        for pair in canonical_wolf_pairs()
    ]
    if weights is None:
        count = sum(legal)
        return [1.0 / count if allowed else 0.0 for allowed in legal]
    values = [0.0] * 21
    for pair, value in weights.items():
        values[canonical_wolf_pairs().index(tuple(pair))] = value
    return values


class FakeBackend:
    supports_json_schema = True

    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        return self.response


def _observation(player_id, *, current=3, identity="Villager"):
    return {
        "observer_id": player_id,
        "current_act_idx": current,
        "identity": identity,
        "game_log": [],
        "phase": "1_day_speech",
        "valid_action": [],
    }


def _successful_report(player, target, known_wolves, known_non_wolves):
    payload = {
        "payload_version": "readonly_pair_belief_self_report_payload_v1",
        "messages": [{"role": "user", "content": f"private-{player}"}],
        "request": {"temperature": 0.0, "max_tokens": 256},
    }
    return {
        "player_id": player,
        "alive": True,
        "report_status": "ok",
        "report_error": None,
        "pair_probabilities": target,
        "known_werewolves": list(known_wolves),
        "known_non_werewolves": list(known_non_wolves),
        "reporter_input_payload": payload,
        "reporter_input_payload_sha256": canonical_json_sha256(payload),
        "raw_reporter_output": json.dumps({"pair_probabilities": target}),
        "parsed_output": {"pair_probabilities": target},
        "hard_knowledge_validation": {"status": "valid"},
        "report_provenance": "playing_agent_readonly_direct_pair_belief_self_report_v1",
        "backend_alias": "synthetic_backend",
        "resolved_model_name": "synthetic-model",
        "prompt_version": PAIR_BELIEF_PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(
            payload["messages"][-1]["content"].encode("utf-8")
        ).hexdigest(),
        "parser_version": "strict_pair_probability_json_v1",
        "sampling_parameters": {"temperature": 0.0, "max_tokens": 256},
        "reporter_seed": None,
    }


def _provenance():
    return {
        "generator_name": "twd_tom_actor_pair_belief_collector",
        "generator_version": "1",
        "git_commit_sha": "a" * 40,
        "git_worktree_clean": True,
        "collection_timestamp_utc": "2026-08-03T00:00:00+00:00",
        "game_seed": 42,
        "source_config_path": "configs/twd_tom_server_qwen25_7b.yaml",
        "source_config_sha256": "b" * 64,
        "resolved_runtime_config_sha256": "c" * 64,
        "resolved_backend_config_sha256": {"synthetic_backend": "d" * 64},
    }


def test_canonical_pair_order_has_exactly_twenty_one_worlds():
    pairs = canonical_wolf_pairs()
    assert len(pairs) == 21
    assert pairs[0] == ("player1", "player2")
    assert pairs[-1] == ("player6", "player7")


@pytest.mark.parametrize(
    "values,error",
    [
        ([1.0], "21"),
        ([float("nan")] + [0.05] * 20, "finite"),
        ([float("inf")] + [0.05] * 20, "finite"),
        ([-0.1] + [0.055] * 20, "negative"),
        ([0.1] * 21, "sum to one"),
        ([True] + [0.05] * 20, "number"),
        (["0.0"] + [0.05] * 20, "number"),
    ],
)
def test_direct_pair_report_rejects_invalid_values_without_normalizing(values, error):
    original = deepcopy(values)
    with pytest.raises((TypeError, ValueError), match=error):
        validate_pair_probabilities(
            values,
            known_werewolves=[],
            known_non_werewolves=[],
        )
    for before, after in zip(original, values):
        if isinstance(before, float) and before != before:
            assert isinstance(after, float) and after != after
        else:
            assert after == before


def test_hard_knowledge_support_rejects_illegal_mass_for_good_and_wolf():
    good = _target(known_non_wolves=("player3",))
    illegal_index = canonical_wolf_pairs().index(("player3", "player4"))
    good[illegal_index] = PROBABILITY_ATOL * 2
    good[canonical_wolf_pairs().index(("player1", "player2"))] -= (
        PROBABILITY_ATOL * 2
    )
    with pytest.raises(ValueError, match="illegal world"):
        validate_pair_probabilities(
            good,
            known_werewolves=[],
            known_non_werewolves=["player3"],
        )

    wolf = _target(
        known_wolves=("player1", "player2"),
        known_non_wolves=("player3", "player4", "player5", "player6", "player7"),
    )
    validated, result = validate_pair_probabilities(
        wolf,
        known_werewolves=["player1", "player2"],
        known_non_werewolves=[
            "player3", "player4", "player5", "player6", "player7"
        ],
    )
    assert validated[0] == 1.0
    assert sum(result["legal_world_mask"]) == 1


def test_reporter_sends_and_serializes_the_identical_canonical_payload():
    target = _target(known_non_wolves=("player3",))
    backend = FakeBackend(json.dumps({"pair_probabilities": target}))
    agent = LLMAgent(backend=backend, model_name="resolved-model")
    agent.backend_id = "server"
    agent.notes = ["private note"]
    reporter = ReadonlyPairBeliefSelfReporter()
    result = reporter.report(
        agent=agent,
        observation=_observation(3),
        belief_owner_id="player3",
        public_snapshot=_snapshot(),
        backend_alias="server",
        known_werewolves=[],
        known_non_werewolves=["player3"],
    )
    assert result["report_status"] == "ok"
    assert backend.calls[0]["messages"] == result["reporter_input_payload"]["messages"]
    assert {
        key: value for key, value in backend.calls[0].items() if key != "model"
    } == {
        "messages": result["reporter_input_payload"]["messages"],
        **result["reporter_input_payload"]["request"],
    }
    assert result["reporter_input_payload_sha256"] == canonical_json_sha256(
        result["reporter_input_payload"]
    )
    repeated_prompt = reporter.build_prompt(
        belief_owner_id="player3",
        public_snapshot=_snapshot(),
        known_werewolves=[],
        known_non_werewolves=["player3"],
    )
    assert result["prompt_sha256"] == hashlib.sha256(
        repeated_prompt.encode("utf-8")
    ).hexdigest()
    assert result["resolved_model_name"] == "resolved-model"
    assert result["sampling_parameters"]["temperature"] == 0.0


def test_failed_report_stays_missing_and_preserves_raw_and_parsed_output():
    target = _target(known_non_wolves=("player3",))
    illegal_index = canonical_wolf_pairs().index(("player3", "player4"))
    legal_index = canonical_wolf_pairs().index(("player1", "player2"))
    target[illegal_index] = 0.2
    target[legal_index] -= 0.2
    raw = json.dumps({"pair_probabilities": target})
    reporter = ReadonlyPairBeliefSelfReporter()
    agent = LLMAgent(backend=FakeBackend(raw), model_name="resolved-model")
    result = reporter.report(
        agent=agent,
        observation=_observation(3),
        belief_owner_id=3,
        public_snapshot=_snapshot(),
        backend_alias="server",
        known_werewolves=[],
        known_non_werewolves=["player3"],
    )
    assert result["report_status"] == "semantic_error"
    assert result["pair_probabilities"] is None
    assert result["raw_reporter_output"] == raw
    assert result["parsed_output"] == {"pair_probabilities": target}
    assert result["hard_knowledge_validation"]["status"] == "invalid"


def test_actor_mapping_builds_only_current_speaker_perspective_and_masks_missing():
    reports = {}
    for player in PLAYER_NAMES:
        player_id = int(player[6:])
        known_non = [player]
        if player == "player1":
            reports[player] = _successful_report(
                player,
                _target(
                    known_wolves=("player1", "player2"),
                    known_non_wolves=(
                        "player3", "player4", "player5", "player6", "player7"
                    ),
                ),
                ["player1", "player2"],
                ["player3", "player4", "player5", "player6", "player7"],
            )
        else:
            reports[player] = _successful_report(
                player,
                _target(known_non_wolves=known_non),
                [],
                known_non,
            )
    reports.pop("player6")
    raw = make_twd_tom_sample(
        public_snapshot=_snapshot(alive=(1, 2, 3, 4, 5, 7)),
        reports=reports,
        collection_provenance=_provenance(),
    )
    mapped = build_actor_perspective_sample(raw)

    assert raw["schema_version"] == ACTOR_PAIR_BELIEF_SCHEMA_VERSION
    assert raw["reasoning_player_id"] == raw["current_speaker"] == "player3"
    assert mapped["schema_version"] == ACTOR_PERSPECTIVE_TARGET_SCHEMA_VERSION
    assert mapped["self_pair_target"] == reports["player3"]["pair_probabilities"]
    assert mapped["other_player_ids"] == [
        "player1", "player2", "player4", "player5", "player6", "player7"
    ]
    assert len(mapped["other_pair_targets"]) == 6
    assert all(len(row) == 21 for row in mapped["other_pair_targets"])
    assert mapped["other_target_mask"] == [True, True, True, True, False, True]
    assert mapped["other_pair_targets"][4] == [0.0] * 21
    # A ToM2 row may include the reasoning player (player3).
    assert mapped["other_pair_targets"][1][
        canonical_wolf_pairs().index(("player3", "player4"))
    ] > 0.0
    # A Werewolf belief owner's legal world includes that owner (player1).
    assert mapped["other_pair_targets"][0][0] == 1.0
    assert "player_reports" not in mapped["reasoning_input"]
    assert "other_pair_targets" not in mapped["reasoning_input"]
    assert set(mapped["reasoning_input"]) == {
        "public_history", "reasoning_player_id", "legal_private_knowledge"
    }
    assert raw["current_action_used"] is False
    assert raw["future_information_used"] is False
    assert "expert" not in json.dumps(raw).lower()


def test_collection_provenance_records_git_config_and_clean_state(tmp_path):
    repo = tmp_path / "repo"
    config_path = repo / "configs" / "twd_tom_server_qwen25_7b.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        (REPO_ROOT / "configs" / "twd_tom_server_qwen25_7b.yaml").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    for command in (
        ("init",),
        ("config", "user.email", "test@example.invalid"),
        ("config", "user.name", "Collection Provenance Test"),
        ("add", "configs/twd_tom_server_qwen25_7b.yaml"),
        ("commit", "-m", "initial"),
    ):
        subprocess.run(
            ["git", "-C", str(repo), *command],
            check=True,
            capture_output=True,
            text=True,
        )
    import yaml

    parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    provenance = build_collection_provenance(
        source_config_path=config_path,
        resolved_runtime_config=normalize_runtime_config(parsed),
        game_seed=42,
        repo_root=repo,
        timestamp_utc="2026-08-03T00:00:00+00:00",
    )
    assert provenance["source_config_path"] == (
        "configs/twd_tom_server_qwen25_7b.yaml"
    )
    assert len(provenance["git_commit_sha"]) == 40
    assert provenance["git_worktree_clean"] is True
    assert len(provenance["source_config_sha256"]) == 64
    assert len(provenance["resolved_runtime_config_sha256"]) == 64
    assert provenance["game_seed"] == 42


def test_historical_tom2_files_are_byte_identical():
    for relative_path, expected in OLD_TOM2_SHA256.items():
        digest = hashlib.sha256((REPO_ROOT / relative_path).read_bytes()).hexdigest()
        assert digest == expected
