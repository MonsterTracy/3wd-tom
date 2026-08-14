import hashlib
import json

import pytest

import run_random
from run_random import build_arg_parser, build_tom_collector
from script.tom.summarize_reporter_ab import summarize_reporter_ab
from tests.tom.test_collection import agents, ready_env
from werewolf.models.tom.reporter import BeliefReporter


def _single_villager_checkpoint(tmp_path, deepseek_response):
    env = ready_env()
    agent_list = agents()
    deepseek = type(agent_list[0].backend)(deepseek_response)
    sidecar_path = tmp_path / "reporter-ab.jsonl"
    collector = build_tom_collector(
        env=env,
        agent_list=agent_list,
        output_path=tmp_path / "formal.jsonl",
        game_id="game-ab",
        seed=17,
        reporter_ab_path=sidecar_path,
        reporter_ab_backend=deepseek,
    )
    env.step(("speech", "CURRENT-SPEECH"))
    env.alive = [0, 0, 0, 0, 1, 0, 0]
    observation = env.get_observation_for(5)
    sample = collector.record(
        env,
        step_idx=9,
        round_number=1,
        phase="speech",
        speaker_id=1,
    )
    collector.close()
    row = json.loads(sidecar_path.read_text(encoding="utf-8"))
    return env, agent_list, deepseek, observation, sample, row, sidecar_path


def test_formal_collector_without_ab_path_keeps_primary_only(tmp_path):
    env = ready_env()
    agent_list = agents()
    collector = build_tom_collector(
        env=env,
        agent_list=agent_list,
        output_path=tmp_path / "formal.jsonl",
        game_id="game-primary",
        seed=17,
    )
    assert collector.reporter_ab is None

    env.step(("speech", "CURRENT-SPEECH"))
    sample = collector.record(
        env,
        step_idx=1,
        round_number=1,
        phase="speech",
        speaker_id=1,
    )
    collector.close()

    assert "qwen" not in sample
    assert "deepseek" not in sample
    assert all(len(agent.backend.calls) == 1 for agent in agent_list)


def test_build_tom_collector_rejects_same_formal_and_ab_paths(tmp_path):
    env = ready_env()
    agent_list = agents()
    formal_path = tmp_path / "a.jsonl"
    equivalent_paths = (
        formal_path,
        f"{tmp_path}/./a.jsonl",
    )

    for reporter_ab_path in equivalent_paths:
        with pytest.raises(
            ValueError,
            match="output_path and reporter_ab_path must be different",
        ):
            build_tom_collector(
                env=env,
                agent_list=agent_list,
                output_path=formal_path,
                game_id="game-same-path",
                seed=17,
                reporter_ab_path=reporter_ab_path,
                reporter_ab_backend=agent_list[0].backend,
            )

    assert not formal_path.exists()


def test_same_observation_ab_uses_shared_prompt_and_provider_transports(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "API-KEY-CANARY")
    (
        _env,
        agent_list,
        deepseek,
        observation,
        sample,
        row,
        _sidecar_path,
    ) = _single_villager_checkpoint(
        tmp_path,
        '{"suspected_werewolves":[]}',
    )

    qwen_request = agent_list[4].backend.calls[0]
    deepseek_request = deepseek.calls[0]
    qwen_prompt = qwen_request["messages"][0]["content"]
    deepseek_prompt = deepseek_request["messages"][0]["content"]
    assert qwen_prompt == deepseek_prompt
    assert row["prompt_digest"] == hashlib.sha256(
        qwen_prompt.encode("utf-8")
    ).hexdigest()
    legal_state = BeliefReporter.legal_state(5, observation)
    canonical_state = json.dumps(
        legal_state,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert row["observation_digest"] == hashlib.sha256(
        canonical_state.encode("utf-8")
    ).hexdigest()

    assert qwen_request["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert "json_schema" in qwen_request["response_format"]
    assert deepseek_request["model"] == "deepseek-v4-flash"
    assert deepseek_request["response_format"] == {"type": "json_object"}
    assert deepseek_request["extra_body"] == {
        "thinking": {"type": "disabled"}
    }
    assert len(agent_list[4].backend.calls) == 1
    assert len(deepseek.calls) == 1
    assert sample["observer_reports"] == [row["qwen"] | {"observer_id": "player5"}]
    assert "qwen" not in sample and "deepseek" not in sample
    assert row["observer_role"] == "Villager"
    serialized = json.dumps(row)
    assert "API-KEY-CANARY" not in serialized
    assert "roles" not in row


@pytest.mark.parametrize(
    ("deepseek_response", "expected_error"),
    [
        ('{"suspected_werewolves":["player5"]}', "semantic_error"),
        ("not-json", "parse_error"),
        (RuntimeError("backend unavailable"), "reporter_error"),
    ],
)
def test_deepseek_failure_is_sidecar_only(
    tmp_path,
    deepseek_response,
    expected_error,
):
    (
        _env,
        agent_list,
        deepseek,
        _observation,
        sample,
        row,
        _sidecar_path,
    ) = _single_villager_checkpoint(tmp_path, deepseek_response)

    assert sample["observer_reports"] == [
        {
            "observer_id": "player5",
            "valid": True,
            "suspected_werewolves": [],
            "error": None,
        }
    ]
    assert row["qwen"] == {
        "valid": True,
        "suspected_werewolves": [],
        "error": None,
    }
    assert row["deepseek"] == {
        "valid": False,
        "suspected_werewolves": None,
        "error": expected_error,
    }
    assert len(agent_list[4].backend.calls) == 1
    assert len(deepseek.calls) == 1


def test_ab_backend_uses_only_deepseek_environment_key(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_backend(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-secret")
    monkeypatch.setattr(run_random, "OpenAICompatibleBackend", fake_backend)

    assert run_random._build_tom_reporter_ab_backend() is sentinel
    assert captured == {
        "api_key": "environment-secret",
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-v4-flash",
        "max_retries": 0,
        "supports_json_schema": False,
    }


def test_ab_cli_and_readonly_summary(tmp_path):
    args = build_arg_parser().parse_args(
        ["--tom_reporter_ab_path", "audit.jsonl"]
    )
    assert args.tom_reporter_ab_path == "audit.jsonl"

    path = tmp_path / "audit.jsonl"
    rows = [
        {
            "observer_role": "Villager",
            "qwen": {
                "valid": True,
                "suspected_werewolves": [],
                "error": None,
            },
            "deepseek": {
                "valid": False,
                "suspected_werewolves": None,
                "error": "semantic_error",
            },
        },
        {
            "observer_role": "Seer",
            "qwen": {
                "valid": False,
                "suspected_werewolves": None,
                "error": "parse_error",
            },
            "deepseek": {
                "valid": True,
                "suspected_werewolves": ["player2"],
                "error": None,
            },
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    summary = summarize_reporter_ab(path)
    assert summary["overall"] == {
        "qwen": {"valid": 1, "attempts": 2},
        "deepseek": {"valid": 1, "attempts": 2},
    }
    assert summary["error_type_counts"] == {
        "qwen": {"parse_error": 1},
        "deepseek": {"semantic_error": 1},
    }
    assert summary["valid_set_counts"] == {
        "qwen": {"empty": 1, "nonempty": 0},
        "deepseek": {"empty": 0, "nonempty": 1},
    }
    assert summary["paired_checkpoint_counts"] == {
        "both_valid": 0,
        "qwen_only_valid": 1,
        "deepseek_only_valid": 1,
        "both_invalid": 0,
    }
