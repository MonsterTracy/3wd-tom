import json
from pathlib import Path

import httpx
import pytest
import yaml

import script.twd_tom.real_backend_dry_run as dry_run
import werewolf.backends.openai_compatible as openai_compatible
from script.twd_tom import pipeline
from werewolf.agents import agent_registry
from werewolf.agents.llm_agent import LLMAgent
from werewolf.backends import BackendError, load_named_backends
from werewolf.runtime_config import normalize_runtime_config
from werewolf.speech.private_belief_perceiver import (
    PRIVATE_BELIEF_JSON_SCHEMA,
    PRIVATE_BELIEF_MAX_TOKENS,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "twd_tom_local_mlx.yaml"
QWEN_CONFIG_PATH = (
    REPO_ROOT / "configs" / "twd_tom_local_qwen25_7b.yaml"
)
SERVER_QWEN_CONFIG_PATH = (
    REPO_ROOT / "configs" / "twd_tom_server_qwen25_7b.yaml"
)
SERVER_QWEN35_CONFIG_PATH = (
    REPO_ROOT / "configs" / "twd_tom_server_qwen35_9b.yaml"
)
LOCAL_MODELS = {
    "local_qwen25_7b": (
        "http://127.0.0.1:8080/v1",
        "mlx-community/Qwen2.5-7B-Instruct-4bit",
    ),
    "local_llama31_8b": (
        "http://127.0.0.1:8081/v1",
        "mlx-community/Llama-3.1-8B-Instruct-4bit",
    ),
    "local_mistral7b_v03": (
        "http://127.0.0.1:8082/v1",
        "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    ),
}


def _config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _qwen_config():
    return yaml.safe_load(
        QWEN_CONFIG_PATH.read_text(encoding="utf-8")
    )


def _server_qwen_config():
    return yaml.safe_load(
        SERVER_QWEN_CONFIG_PATH.read_text(encoding="utf-8")
    )


def _server_qwen35_config():
    return yaml.safe_load(
        SERVER_QWEN35_CONFIG_PATH.read_text(encoding="utf-8")
    )


def _mock_openai_clients(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    original_client = openai_compatible.openai.OpenAI

    def client_factory(**kwargs):
        assert kwargs["api_key"] == "local-mlx"
        assert kwargs["max_retries"] == 0
        kwargs.pop("http_client").close()
        return original_client(
            **kwargs,
            http_client=httpx.Client(transport=transport),
        )

    monkeypatch.setattr(
        openai_compatible.openai,
        "OpenAI",
        client_factory,
    )


def _success_response(
    request,
    *,
    content='{"suspected_werewolves":[]}',
):
    payload = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            "id": "fake-local-response",
            "object": "chat.completion",
            "created": 0,
            "model": payload["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        },
    )


def _new_audit_session(tmp_path):
    writer = dry_run.PrivacySafeAuditWriter(tmp_path / "audit.jsonl")
    budget = dry_run.DryRunBudget(
        game_id="local_mlx_test",
        max_gameplay_calls=10,
        max_belief_calls=10,
        max_total_calls=20,
        max_wall_seconds=60.0,
    )
    session = dry_run.DryRunAuditSession(
        game_id="local_mlx_test",
        seed=4101,
        budget=budget,
        writer=writer,
    )
    return session, writer


def test_local_mlx_config_resolves_exact_routes_without_secrets():
    parsed = _config()
    normalized = normalize_runtime_config(parsed)

    assert set(normalized["backends"]) == set(LOCAL_MODELS)
    for alias, (base_url, model) in LOCAL_MODELS.items():
        backend = normalized["backends"][alias]
        assert backend["base_url"] == base_url
        assert backend["default_model"] == model
        assert backend["api_key_env"] is None
        assert backend["supports_json_schema"] is False

    profiles = {
        profile["profile_name"]: profile
        for profile in normalized["agent_config"]["all_candidates"]
    }
    assert set(profiles) == set(LOCAL_MODELS)
    for alias, (_base_url, model) in LOCAL_MODELS.items():
        assert profiles[alias]["backend"] == alias
        assert profiles[alias]["model"] == model
        assert profiles[alias]["sample_ratio"] == 1.0
    assert normalized["agent_config"]["allow_cross_team_profiles"] is True
    assert normalized["agent_config"]["must_include"] == []
    assert normalized["parser"]["backend"] == "local_qwen25_7b"
    assert normalized["parser"]["model"] == LOCAL_MODELS[
        "local_qwen25_7b"
    ][1]

    result = pipeline.run_pipeline_stage(
        config_path=CONFIG_PATH,
        run_id="local_mlx_validation",
        stage="validate",
    )
    assert result["plan"]["collection"]["game_count"] == 3
    assert result["plan"]["collection"]["seeds"] == [4101, 4102, 4103]

    serialized = json.dumps(parsed, sort_keys=True)
    for base_url, model in LOCAL_MODELS.values():
        assert base_url in serialized
        assert model in serialized
    assert "api_key" not in serialized.lower()
    assert "local-mlx" not in serialized


def test_single_qwen_config_resolves_without_network_or_artifacts(
    monkeypatch,
):
    raw_text = QWEN_CONFIG_PATH.read_text(encoding="utf-8")
    parsed = _qwen_config()
    normalized = normalize_runtime_config(parsed)
    alias = "local_qwen25_7b"
    base_url, model = LOCAL_MODELS[alias]

    assert parsed["pipeline"] == _config()["pipeline"]
    assert set(normalized["backends"]) == {alias}
    assert normalized["backends"][alias] == {
        "type": "openai_compatible",
        "base_url": base_url,
        "api_key_env": None,
        "default_model": model,
        "supports_json_schema": False,
    }
    assert normalized["agent_config"]["all_candidates"] == [
        {
            "profile_name": alias,
            "agent_type": "gpt",
            "backend": alias,
            "model": model,
            "model_params": {"temperature": 1.0},
            "sample_ratio": 1.0,
        }
    ]
    assert normalized["parser"]["backend"] == alias
    assert normalized["parser"]["model"] == model
    for forbidden in (
        "local_llama31_8b",
        "local_mistral7b_v03",
        "8081",
        "8082",
    ):
        assert forbidden not in raw_text

    monkeypatch.setattr(
        openai_compatible.openai,
        "OpenAI",
        lambda **_kwargs: pytest.fail("validate accessed the network client"),
    )
    run_id = "single_qwen_validate_test"
    paths = pipeline._run_paths(run_id)
    assert all(
        not path.exists()
        for group in paths.values()
        for path in group.values()
    )
    result = pipeline.run_pipeline_stage(
        config_path=QWEN_CONFIG_PATH,
        run_id=run_id,
        stage="validate",
    )
    assert result["plan"]["collection"]["game_count"] == 3
    assert result["plan"]["collection"]["seeds"] == [4101, 4102, 4103]
    assert all(
        not path.exists()
        for group in paths.values()
        for path in group.values()
    )

    resolved = pipeline._load_pipeline_config(
        QWEN_CONFIG_PATH,
        run_id=run_id,
        game_count_override=None,
        seeds_override=None,
    )
    serialized = json.dumps(
        resolved["resolved_runtime"],
        sort_keys=True,
        default=str,
    )
    assert "api_key" not in serialized.lower()
    assert "local-mlx" not in serialized


def test_single_qwen_gameplay_and_belief_use_same_route(
    monkeypatch,
):
    calls = []

    def handler(request):
        calls.append((str(request.url), json.loads(request.content)))
        return _success_response(request)

    _mock_openai_clients(monkeypatch, handler)
    backends = load_named_backends(
        _qwen_config(),
        env_file=None,
        max_retries=0,
    )
    alias = "local_qwen25_7b"
    base_url, model = LOCAL_MODELS[alias]
    agent = LLMAgent(
        backend=backends[alias],
        model_name=model,
    )

    agent._chat(
        [{"role": "user", "content": "gameplay"}],
        temperature=0.2,
        max_tokens=8,
    )
    agent.report_suspected_werewolves_readonly(
        observation={
            "observer_id": 1,
            "identity": "Villager",
            "phase": "1_day_speech",
            "current_act_idx": 1,
            "game_log": [],
        },
        report_prompt="Return the required JSON object.",
    )

    assert len(calls) == 2
    for url, payload in calls:
        assert url == f"{base_url}/chat/completions"
        assert payload["model"] == model
    assert calls[1][1]["max_tokens"] == PRIVATE_BELIEF_MAX_TOKENS
    assert calls[1][1]["response_format"] == {
        "type": "json_object"
    }


def test_server_qwen_config_is_offline_and_uses_one_loopback_route(
    monkeypatch,
):
    parsed = _server_qwen_config()
    baseline = _qwen_config()
    normalized = normalize_runtime_config(parsed)
    alias = "server_qwen25_7b"
    base_url = "http://127.0.0.1:8000/v1"
    model = "qwen2.5-7b-instruct"

    assert parsed["env_config"] == baseline["env_config"]
    assert parsed["pipeline"] == baseline["pipeline"]
    assert normalized["backends"] == {
        alias: {
            "type": "openai_compatible",
            "base_url": base_url,
            "api_key_env": None,
            "default_model": model,
            "supports_json_schema": True,
        }
    }
    assert normalized["agent_config"]["all_candidates"] == [
        {
            "profile_name": alias,
            "agent_type": "gpt",
            "backend": alias,
            "model": model,
            "model_params": {
                "temperature": 1.0,
                "gameplay_prompt_profile": "strict_classic7",
                "gameplay_max_tokens": 512,
            },
            "sample_ratio": 1.0,
        }
    ]
    assert normalized["parser"] == {
        "backend": alias,
        "model": model,
        "model_params": {"temperature": 0.0},
    }
    assert "belief" not in parsed

    monkeypatch.setattr(
        openai_compatible.openai,
        "OpenAI",
        lambda **_kwargs: pytest.fail(
            "validate accessed the network client"
        ),
    )
    run_id = "server_qwen_validate_test"
    paths = pipeline._run_paths(run_id)
    expected_run_dirs = {
        group: (
            REPO_ROOT
            / group
            / "tom"
            / run_id
        ).resolve()
        for group in (
            "data",
            "logs",
            "outputs",
        )
    }
    assert {
        group: group_paths["run_dir"]
        for group, group_paths in paths.items()
    } == expected_run_dirs
    assert all(
        not path.exists()
        for group in paths.values()
        for path in group.values()
    )

    result = pipeline.run_pipeline_stage(
        config_path=SERVER_QWEN_CONFIG_PATH,
        run_id=run_id,
        stage="validate",
    )

    assert {
        group: Path(
            result["plan"][group]["run_dir"]
        ).resolve()
        for group in expected_run_dirs
    } == expected_run_dirs
    assert all(
        not path.exists()
        for group in paths.values()
        for path in group.values()
    )

    captured = {}

    def fake_openai(**kwargs):
        captured.update(kwargs)
        kwargs["http_client"].close()
        return object()

    monkeypatch.setattr(
        openai_compatible.openai,
        "OpenAI",
        fake_openai,
    )
    backends = load_named_backends(
        parsed,
        env_file=None,
        max_retries=0,
    )
    assert set(backends) == {alias}
    assert captured["api_key"] == "local-mlx"
    assert captured["base_url"] == base_url
    assert captured["max_retries"] == 0
    assert "http_client" in captured
    assert backends[alias].default_model == model


def test_server_qwen35_config_validates_five_game_collection_plan(
    monkeypatch,
):
    parsed = _server_qwen35_config()
    normalized = normalize_runtime_config(parsed)
    alias = "server_qwen35_9b"
    base_url = "http://127.0.0.1:8000/v1"
    model = "qwen3.5-9b"

    assert normalized["backends"] == {
        alias: {
            "type": "openai_compatible",
            "base_url": base_url,
            "api_key_env": None,
            "default_model": model,
            "supports_json_schema": True,
        }
    }
    assert normalized["parser"] == {
        "backend": alias,
        "model": model,
        "model_params": {"temperature": 0.0},
    }
    assert normalized["agent_config"]["all_candidates"] == [
        {
            "profile_name": alias,
            "agent_type": "gpt",
            "backend": alias,
            "model": model,
            "model_params": {
                "temperature": 1.0,
                "gameplay_prompt_profile": "strict_classic7",
                "gameplay_max_tokens": 512,
            },
            "sample_ratio": 1.0,
        }
    ]
    assert parsed["pipeline"]["public_event_schema_version"] == (
        "classic7_public_event_sequence_v2"
    )
    assert parsed["pipeline"]["collection"] == {
        "game_count": 3,
        "seeds": [4101, 4102, 4103],
        "max_gameplay_calls_per_game": 192,
        "max_belief_calls_per_game": 448,
        "max_total_calls_per_game": 640,
        "max_wall_seconds_per_game": 3600.0,
    }
    assert "resolved_run" not in parsed["pipeline"]
    assert "output_root" not in parsed["pipeline"]

    monkeypatch.setattr(
        openai_compatible.openai,
        "OpenAI",
        lambda **_kwargs: pytest.fail(
            "validate accessed the network client"
        ),
    )
    run_id = "server-qwen35-a1-v2-s343-347-5g-r1"
    result = pipeline.run_pipeline_stage(
        config_path=SERVER_QWEN35_CONFIG_PATH,
        run_id=run_id,
        stage="validate",
        game_count=5,
        seeds=(343, 344, 345, 346, 347),
    )

    assert result["game_count"] == 5
    assert result["seeds"] == [343, 344, 345, 346, 347]
    assert result["plan"]["versions"][
        "public_event_schema_version"
    ] == "classic7_public_event_sequence_v2"
    assert {
        group: Path(result["plan"][group]["run_dir"]).resolve()
        for group in ("data", "logs", "outputs")
    } == {
        group: (REPO_ROOT / group / "tom" / run_id).resolve()
        for group in ("data", "logs", "outputs")
    }
    assert all(
        not path.exists()
        for group in pipeline._run_paths(run_id).values()
        for path in group.values()
    )


def test_server_qwen_gameplay_limit_reaches_chat_completions(
    monkeypatch,
):
    calls = []

    def handler(request):
        calls.append((str(request.url), json.loads(request.content)))
        return _success_response(request, content="这是公开发言。")

    _mock_openai_clients(monkeypatch, handler)
    normalized = normalize_runtime_config(_server_qwen_config())
    profile = normalized["agent_config"]["all_candidates"][0]
    backends = load_named_backends(
        normalized,
        env_file=None,
        max_retries=0,
    )
    agent_type, agent_params = agent_registry.build(
        profile["agent_type"],
        backend=backends[profile["backend"]],
        model_name=profile["model"],
        **profile["model_params"],
    )
    agent = agent_registry.build_agent(
        agent_type,
        player_idx=0,
        agent_param=agent_params,
        env_param={"n_player": 7, "n_role": 4},
        log_file=None,
    )
    agent.rate_limit = 0

    agent.act(
        {
            "phase": "1_day_speech",
            "identity": "Villager",
            "current_act_idx": 1,
            "game_log": [],
            "valid_action": ("speech", -1),
                "authoritative_public_state": {
                    "day": 1,
                    "day_or_night": "day",
                    "phase": "speech",
                    "last_night_result": {
                        "day": 0,
                        "dead_players": [],
                    },
                    "prior_exiles": [],
                    "alive_players": [1, 2, 3, 4, 5, 6, 7],
                    "suggestible_exile_targets": [2, 3, 4, 5, 6, 7],
                },
        }
    )

    assert len(calls) == 1
    url, payload = calls[0]
    assert url == "http://127.0.0.1:8000/v1/chat/completions"
    assert payload["max_tokens"] == 512
    assert payload["model"] == "qwen2.5-7b-instruct"


@pytest.mark.parametrize(
    "config_name",
    [
        "deepseek_vs_twdm.yaml",
        "gpt_vs_twdm.yaml",
        "random_models.yaml",
        "twd_tom_multi_api.yaml",
    ],
)
def test_existing_cloud_configs_still_normalize(config_name):
    parsed = yaml.safe_load(
        (REPO_ROOT / "configs" / config_name).read_text(
            encoding="utf-8"
        )
    )

    normalized = normalize_runtime_config(parsed)

    assert normalized["backends"]
    assert all(
        backend["supports_json_schema"] is False
        for backend in normalized["backends"].values()
    )
    assert normalized["parser"]["backend"] in normalized["backends"]


def test_local_routes_preserve_alias_and_model_for_gameplay_and_belief(
    monkeypatch,
):
    calls = []

    def handler(request):
        calls.append((str(request.url), json.loads(request.content)))
        return _success_response(request)

    _mock_openai_clients(monkeypatch, handler)
    backends = load_named_backends(
        _config(),
        env_file=None,
        max_retries=0,
    )

    for alias, (_base_url, model) in LOCAL_MODELS.items():
        agent = LLMAgent(
            backend=backends[alias],
            model_name=model,
        )
        agent._chat(
            [{"role": "user", "content": "gameplay"}],
            temperature=0.2,
            max_tokens=8,
        )
        agent.report_suspected_werewolves_readonly(
            observation={
                "observer_id": 1,
                "identity": "Villager",
                "phase": "1_day_speech",
                "current_act_idx": 1,
                "game_log": [],
            },
            report_prompt="Return the required JSON object.",
        )

    assert len(calls) == 6
    for index, (_alias, (base_url, model)) in enumerate(
        LOCAL_MODELS.items()
    ):
        gameplay_url, gameplay_payload = calls[index * 2]
        belief_url, belief_payload = calls[index * 2 + 1]
        assert gameplay_url == f"{base_url}/chat/completions"
        assert belief_url == f"{base_url}/chat/completions"
        assert gameplay_payload["model"] == model
        assert belief_payload["model"] == model
        assert belief_payload["response_format"] == {
            "type": "json_object"
        }
        assert (
            belief_payload["max_tokens"]
            == PRIVATE_BELIEF_MAX_TOKENS
        )
        assert belief_payload["thinking"] == {"type": "disabled"}


def test_server_qwen_belief_uses_strict_schema_without_network(
    monkeypatch,
):
    calls = []

    def handler(request):
        calls.append((str(request.url), json.loads(request.content)))
        return _success_response(request)

    _mock_openai_clients(monkeypatch, handler)
    backends = load_named_backends(
        _server_qwen_config(),
        env_file=None,
        max_retries=0,
    )
    alias = "server_qwen25_7b"
    model = "qwen2.5-7b-instruct"
    agent = LLMAgent(
        backend=backends[alias],
        model_name=model,
    )

    agent.report_suspected_werewolves_readonly(
        observation={
            "observer_id": 1,
            "identity": "Villager",
            "phase": "1_day_speech",
            "current_act_idx": 1,
            "game_log": [],
        },
        report_prompt="Return the required JSON object.",
    )

    assert len(calls) == 1
    url, payload = calls[0]
    assert url == "http://127.0.0.1:8000/v1/chat/completions"
    assert payload["model"] == model
    assert payload["max_tokens"] == PRIVATE_BELIEF_MAX_TOKENS
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "private_belief_report",
            "strict": True,
            "schema": PRIVATE_BELIEF_JSON_SCHEMA,
        },
    }
    transport_schema = payload["response_format"][
        "json_schema"
    ]["schema"]
    array_schema = transport_schema["properties"][
        "suspected_werewolves"
    ]
    assert "uniqueItems" not in array_schema
    assert "contains" not in array_schema
    assert "minContains" not in array_schema
    assert "maxContains" not in array_schema


def test_collect_key_preflight_accepts_only_implicit_loopback_auth(
    monkeypatch,
):
    resolved = pipeline._load_pipeline_config(
        CONFIG_PATH,
        run_id="local_mlx_preflight",
        game_count_override=None,
        seeds_override=None,
    )
    monkeypatch.setattr(pipeline, "load_dotenv", lambda **_kwargs: None)

    pipeline._check_collect_api_keys(resolved)

    resolved["normalized_runtime"]["backends"]["local_qwen25_7b"][
        "api_key_env"
    ] = "LOCAL_QWEN_API_KEY"
    monkeypatch.delenv("LOCAL_QWEN_API_KEY", raising=False)
    with pytest.raises(ValueError, match="LOCAL_QWEN_API_KEY"):
        pipeline._check_collect_api_keys(resolved)


def test_audit_distinguishes_local_alias_url_model_and_error(
    tmp_path,
    monkeypatch,
):
    requests = []

    def handler(request):
        requests.append(str(request.url))
        if request.url.port == 8081:
            return httpx.Response(503, json={"error": "fake unavailable"})
        return _success_response(request)

    _mock_openai_clients(monkeypatch, handler)
    backends = load_named_backends(
        _config(),
        env_file=None,
        max_retries=0,
    )
    session, writer = _new_audit_session(tmp_path)

    for alias, (_base_url, model) in LOCAL_MODELS.items():
        audited = dry_run.AuditedBackend(
            backend=backends[alias],
            backend_id=alias,
            session=session,
        )
        with session.gameplay_context(
            acting_player_id=1,
            observation={},
            public_events=[],
        ):
            if alias == "local_llama31_8b":
                with pytest.raises(BackendError):
                    audited.chat(
                        messages=[{"role": "user", "content": "test"}],
                        model=model,
                        max_tokens=8,
                    )
            else:
                audited.chat(
                    messages=[{"role": "user", "content": "test"}],
                    model=model,
                    max_tokens=8,
                )
    writer.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "audit.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(requests) == 3
    assert len(records) == 3
    assert {
        (
            record["backend_id"],
            record["base_url"],
            record["endpoint"],
            record["model_id"],
        )
        for record in records
    } == {
        (
            alias,
            base_url,
            f"{base_url}/chat/completions",
            model,
        )
        for alias, (base_url, model) in LOCAL_MODELS.items()
    }
    by_alias = {record["backend_id"]: record for record in records}
    assert by_alias["local_llama31_8b"]["dispatch_status"] == "error"
    assert (
        by_alias["local_llama31_8b"]["error_type"]
        == "InternalServerError"
    )
    assert by_alias["local_qwen25_7b"]["dispatch_status"] == "ok"
    assert by_alias["local_qwen25_7b"]["error_type"] is None
    assert requests.count(
        "http://127.0.0.1:8081/v1/chat/completions"
    ) == 1

    serialized = json.dumps(records, sort_keys=True)
    assert "local-mlx" not in serialized
    assert "api_key" not in serialized.lower()


def test_explicit_local_api_key_env_remains_fail_closed(monkeypatch):
    config = _config()
    config["backends"]["local_qwen25_7b"][
        "api_key_env"
    ] = "LOCAL_QWEN_API_KEY"
    monkeypatch.delenv("LOCAL_QWEN_API_KEY", raising=False)

    with pytest.raises(ValueError, match="LOCAL_QWEN_API_KEY"):
        load_named_backends(
            config,
            env_file=None,
            max_retries=0,
        )
