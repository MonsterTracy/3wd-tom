from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

import script.twd_tom.real_backend_dry_run as dry_run
from werewolf.agents.llm_agent import LLMAgent
from werewolf.models import SpeechPerceiver
from werewolf.speech.private_belief_perceiver import PlayingAgentBeliefReporter


class FakeBackend:
    def __init__(self, responses=None, *, usage=None, failures=0):
        self.default_model = "fake-model"
        self.responses = list(responses or ["fake response"])
        self.usage = usage
        self.failures = failures
        self.calls = []

    def chat_with_metadata(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("fake provider failure")
        response = self.responses.pop(0) if self.responses else "fake response"
        return response, self.usage


class MutatingBackend(FakeBackend):
    def chat_with_metadata(self, **kwargs):
        result = super().chat_with_metadata(**kwargs)
        self.default_model = "mutated-model"
        return result


def _new_session(
    tmp_path,
    *,
    gameplay=10,
    belief=10,
    total=20,
    wall=60.0,
    clock_ns=None,
):
    writer = dry_run.PrivacySafeAuditWriter(tmp_path / "audit.jsonl")
    budget_kwargs = {}
    if clock_ns is not None:
        budget_kwargs["clock_ns"] = clock_ns
    budget = dry_run.DryRunBudget(
        game_id="dry_run_game_001",
        max_gameplay_calls=gameplay,
        max_belief_calls=belief,
        max_total_calls=total,
        max_wall_seconds=wall,
        **budget_kwargs,
    )
    session = dry_run.DryRunAuditSession(
        game_id="dry_run_game_001",
        seed=42,
        budget=budget,
        writer=writer,
    )
    return session, writer


def _backend(fake, session):
    return dry_run.AuditedBackend(
        backend=fake,
        backend_id="fake-backend",
        session=session,
    )


def _observation(player_id=1):
    return {
        "observer_id": player_id,
        "identity": "Villager",
        "phase": "1_day_speech",
        "current_act_idx": player_id,
        "game_log": [],
    }


def _public_events(player_id=1):
    return [
        {
            "event_idx": 0,
            "event_type": "phase_change",
            "phase": "1_day_speech",
        },
        {
            "event_idx": 1,
            "event_type": "turn_start",
            "speaker": f"player{player_id}",
        },
    ]


def _snapshot():
    return SimpleNamespace(
        public_action_count=0,
        public_history_digest="0" * 64,
        sp_actions=(),
        speaker_id=1,
        phase="1_day_speech",
    )


def _records(path):
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line], text


def _perform_report(
    tmp_path,
    *,
    response='{"suspected_werewolves":[]}',
):
    session, writer = _new_session(tmp_path)
    fake = FakeBackend([response, "next action"])
    audited = _backend(fake, session)
    agent = LLMAgent(backend=audited, model_name="fake-model")
    agent.backend_id = "fake-backend"
    reporter = PlayingAgentBeliefReporter(audit_hook=session)
    result = reporter.report(
        agent=agent,
        observation=_observation(),
        observer_id="player1",
        public_snapshot=_snapshot(),
        agent_backend_id="fake-backend",
        known_werewolves=[],
        known_non_werewolves=["player1"],
    )
    reporter.record_agent_state(
        observer_id="player1",
        state_before={"notes": []},
        state_after={"notes": []},
    )
    return session, writer, fake, audited, result


def test_privacy_safe_call_record_has_usage_hashes_but_no_raw_text(tmp_path):
    session, writer = _new_session(tmp_path)
    fake = FakeBackend(
        ["PRIVATE_RAW_RESPONSE"],
        usage={"input_tokens": 12, "output_tokens": 3, "total_tokens": 15},
    )
    audited = _backend(fake, session)
    private_prompt = "PRIVATE_PROMPT secret-key role teammate private observation"
    with session.gameplay_context(
        acting_player_id=1,
        observation=_observation(),
        public_events=_public_events(),
    ):
        assert audited.chat(
            messages=[{"role": "user", "content": private_prompt}],
            model="fake-model",
        ) == "PRIVATE_RAW_RESPONSE"
    writer.close()

    records, serialized = _records(tmp_path / "audit.jsonl")
    assert len(records) == 1
    assert set(records[0]) <= dry_run._AUDIT_FIELDS
    assert records[0]["usage_available"] is True
    assert records[0]["input_tokens"] == 12
    assert records[0]["output_tokens"] == 3
    assert records[0]["total_tokens"] == 15
    assert records[0]["request_character_count"] > 0
    assert records[0]["response_character_count"] == len("PRIVATE_RAW_RESPONSE")
    for forbidden in (
        private_prompt,
        "PRIVATE_RAW_RESPONSE",
        "secret-key",
        "teammate",
        "private observation",
        "Villager",
    ):
        assert forbidden not in serialized
    assert fake.calls[0]["messages"] == [
        {"role": "user", "content": private_prompt}
    ]


@pytest.mark.parametrize(
    "forbidden_kwargs",
    [
        {"conversation_id": "remote"},
        {"thread_id": "remote"},
        {"session_id": "remote"},
        {"previous_response_id": "remote"},
        {"extra_body": {"store": True}},
    ],
)
def test_remote_session_payload_is_rejected_before_dispatch(
    tmp_path, forbidden_kwargs
):
    session, writer = _new_session(tmp_path)
    fake = FakeBackend()
    audited = _backend(fake, session)
    with pytest.raises(dry_run.DryRunSafetyViolation):
        audited.chat(
            messages=[{"role": "user", "content": "full messages"}],
            model="fake-model",
            **forbidden_kwargs,
        )
    writer.close()
    assert fake.calls == []
    assert session.budget.total_calls == 0


def test_backend_fingerprint_and_playing_agent_backend_model_are_enforced(tmp_path):
    session, writer = _new_session(tmp_path)
    fake = FakeBackend(['{"suspected_werewolves":[]}'])
    audited = _backend(fake, session)
    before = dry_run.backend_semantic_state_fingerprint(fake)
    report_id, _prompt = session.prepare_report(
        observer_id="player1",
        public_snapshot=_snapshot(),
        agent_backend_id="fake-backend",
        agent_model_id="fake-model",
        report_prompt="private report",
    )
    with session.belief_context(report_id):
        audited.chat(
            messages=[{"role": "user", "content": "complete context"}],
            model="fake-model",
        )
    after = dry_run.backend_semantic_state_fingerprint(fake)
    session.finish_game()
    writer.close()
    assert before == after
    assert fake.calls[0]["messages"][0]["content"] == "complete context"
    records, serialized = _records(tmp_path / "audit.jsonl")
    assert records[0]["backend_id"] == "fake-backend"
    assert records[0]["model_id"] == "fake-model"
    assert "response_id" not in serialized


def test_backend_semantic_state_mutation_fails(tmp_path):
    session, writer = _new_session(tmp_path)
    fake = MutatingBackend()
    audited = _backend(fake, session)
    with pytest.raises(dry_run.DryRunSafetyViolation, match="semantic state changed"):
        audited.chat(
            messages=[{"role": "user", "content": "prompt"}],
            model="fake-model",
        )
    writer.close()


def test_report_nonce_and_response_are_absent_from_next_action(tmp_path):
    session, writer, fake, audited, result = _perform_report(tmp_path)
    assert result["status"] == "ok"
    report_request = json.dumps(fake.calls[0]["messages"], ensure_ascii=False)
    assert "TWD_TOM_REPORT_AUDIT_" in report_request

    with session.gameplay_context(
        acting_player_id=1,
        observation=_observation(),
        public_events=_public_events(),
    ):
        audited.chat(
            messages=[{"role": "user", "content": "clean next action"}],
            model="fake-model",
        )
    writer.close()
    records, serialized = _records(tmp_path / "audit.jsonl")
    belief = next(record for record in records if record["call_category"] == "belief")
    assert belief["report_nonce_absent_from_next_action"] is True
    assert belief["report_response_absent_from_next_action"] is True
    assert belief["next_action_check_not_reached"] is False
    assert belief["state_hash_before"] == belief["state_hash_after"]
    assert "TWD_TOM_REPORT_AUDIT_" not in serialized
    assert '{"suspected_werewolves":[]}' not in serialized


@pytest.mark.parametrize("contamination", ["nonce", "response"])
def test_forced_report_contamination_fails_before_next_dispatch(
    tmp_path, contamination
):
    session, writer, fake, audited, _result = _perform_report(tmp_path)
    report_request = json.dumps(fake.calls[0]["messages"], ensure_ascii=False)
    marker = (
        report_request[report_request.index("TWD_TOM_REPORT_AUDIT_") :].split('"')[0]
        if contamination == "nonce"
        else '{"suspected_werewolves":[]}'
    )
    with session.gameplay_context(
        acting_player_id=1,
        observation=_observation(),
        public_events=_public_events(),
    ):
        with pytest.raises(
            dry_run.DryRunSafetyViolation,
            match="contamination detected",
        ):
            audited.chat(
                messages=[{"role": "user", "content": marker}],
                model="fake-model",
            )
    writer.close()
    assert len(fake.calls) == 1
    _records_list, serialized = _records(tmp_path / "audit.jsonl")
    assert marker not in serialized
    assert "TWD_TOM_REPORT_AUDIT_" not in serialized


def test_report_without_next_action_is_marked_not_reached(tmp_path):
    session, writer, _fake, _audited, _result = _perform_report(tmp_path)
    session.finish_game()
    writer.close()
    records, serialized = _records(tmp_path / "audit.jsonl")
    assert records[0]["next_action_check_not_reached"] is True
    assert records[0]["report_nonce_absent_from_next_action"] is None
    assert records[0]["report_response_absent_from_next_action"] is None
    assert "TWD_TOM_REPORT_AUDIT_" not in serialized
    assert '{"suspected_werewolves":[]}' not in serialized


def test_gameplay_budget_stops_before_extra_dispatch(tmp_path):
    session, writer = _new_session(tmp_path, gameplay=1)
    fake = FakeBackend(["one", "two"])
    audited = _backend(fake, session)
    audited.chat(messages=[{"role": "user", "content": "one"}], model="fake-model")
    with pytest.raises(dry_run.DryRunBudgetExceeded, match="gameplay_calls"):
        audited.chat(messages=[{"role": "user", "content": "two"}], model="fake-model")
    writer.close()
    assert len(fake.calls) == 1


def test_parser_error_recovery_cannot_swallow_hard_budget_stop(tmp_path):
    session, writer = _new_session(tmp_path, gameplay=1)
    fake = FakeBackend(["first"])
    audited = _backend(fake, session)
    audited.chat(
        messages=[{"role": "user", "content": "first"}],
        model="fake-model",
    )
    parser = SpeechPerceiver(backend=audited, model_name="fake-model")
    with pytest.raises(dry_run.DryRunBudgetExceeded, match="gameplay_calls"):
        parser.parse(
            speaker=1,
            speech="public speech",
            day=1,
            phase="1_day_speech",
        )
    writer.close()
    assert len(fake.calls) == 1


def test_belief_budget_stops_before_extra_dispatch(tmp_path):
    session, writer = _new_session(tmp_path, belief=1, total=3)
    fake = FakeBackend(
        ['{"suspected_werewolves":[]}', "extra"]
    )
    audited = _backend(fake, session)
    report_id, _ = session.prepare_report(
        observer_id="player1",
        public_snapshot=_snapshot(),
        agent_backend_id="fake-backend",
        agent_model_id="fake-model",
        report_prompt="report",
    )
    with session.belief_context(report_id):
        audited.chat(messages=[{"role": "user", "content": "one"}], model="fake-model")
    second_id, _ = session.prepare_report(
        observer_id="player2",
        public_snapshot=_snapshot(),
        agent_backend_id="fake-backend",
        agent_model_id="fake-model",
        report_prompt="report",
    )
    with session.belief_context(second_id):
        with pytest.raises(dry_run.DryRunBudgetExceeded, match="belief_calls"):
            audited.chat(
                messages=[{"role": "user", "content": "two"}],
                model="fake-model",
            )
    session.finish_game()
    writer.close()
    assert len(fake.calls) == 1


def test_total_budget_stops_before_extra_dispatch(tmp_path):
    session, writer = _new_session(tmp_path, gameplay=2, total=1)
    fake = FakeBackend(["one", "two"])
    audited = _backend(fake, session)
    audited.chat(messages=[{"role": "user", "content": "one"}], model="fake-model")
    with pytest.raises(dry_run.DryRunBudgetExceeded, match="total_calls"):
        audited.chat(messages=[{"role": "user", "content": "two"}], model="fake-model")
    writer.close()
    assert len(fake.calls) == 1


def test_wall_budget_is_checked_before_dispatch(tmp_path):
    now = [0]
    session, writer = _new_session(
        tmp_path,
        wall=1.0,
        clock_ns=lambda: now[0],
    )
    now[0] = 2_000_000_000
    fake = FakeBackend()
    with pytest.raises(dry_run.DryRunBudgetExceeded, match="elapsed_wall_time"):
        _backend(fake, session).chat(
            messages=[{"role": "user", "content": "late"}],
            model="fake-model",
        )
    writer.close()
    assert fake.calls == []


def test_hard_wall_timer_interrupts_in_flight_work():
    budget = dry_run.DryRunBudget(
        game_id="dry_run_game_001",
        max_gameplay_calls=1,
        max_belief_calls=1,
        max_total_calls=2,
        max_wall_seconds=0.01,
    )
    with pytest.raises(dry_run.DryRunBudgetExceeded, match="elapsed_wall_time"):
        with dry_run._hard_wall_timeout(budget):
            time.sleep(0.1)


def test_belief_backend_failure_dispatches_once_and_fails_closed(
    tmp_path,
):
    session, writer = _new_session(tmp_path)
    fake = FakeBackend(
        responses=['{"suspected_werewolves":[]}'],
        failures=1,
    )
    audited = _backend(fake, session)
    agent = LLMAgent(backend=audited, model_name="fake-model")
    agent.backend_id = "fake-backend"
    agent.notes = ["unchanged gameplay memory"]
    observation = _observation()
    observation_before = deepcopy(observation)
    result = PlayingAgentBeliefReporter(audit_hook=session).report(
        agent=agent,
        observation=observation,
        observer_id="player1",
        public_snapshot=_snapshot(),
        agent_backend_id="fake-backend",
        known_werewolves=[],
        known_non_werewolves=["player1"],
    )
    session.finish_game()
    writer.close()

    assert result["status"] == "reporter_error"
    assert result["suspected_werewolves"] is None
    assert len(fake.calls) == 1
    assert fake.responses == ['{"suspected_werewolves":[]}']
    assert session.budget.total_calls == 1
    assert agent.notes == ["unchanged gameplay memory"]
    assert observation == observation_before
    assert result["error"] not in agent.format_observation(observation)
    records, _serialized = _records(tmp_path / "audit.jsonl")
    assert records[0]["dispatch_status"] == "error"



def test_exactly_two_scheduler_stops_on_first_failure_and_never_adds_third():
    completed = []
    assert dry_run._run_exactly_two(
        (42, 43), lambda index, seed: completed.append((index, seed))
    ) == [None, None]
    assert completed == [(1, 42), (2, 43)]

    started = []

    def fail_first(index, seed):
        started.append((index, seed))
        raise dry_run.DryRunBudgetExceeded(
            budget_name="total_calls",
            current=2,
            limit=1,
            game_id="dry_run_game_001",
            call_category="gameplay",
        )

    with pytest.raises(dry_run.DryRunBudgetExceeded, match="total_calls"):
        dry_run._run_exactly_two((42, 43), fail_first)
    assert started == [(1, 42)]


def test_output_dir_and_config_require_explicit_safe_two_game_values(tmp_path):
    safe = tmp_path / "logs" / "belief_dry_run"
    assert dry_run.validate_output_dir(str(safe), cwd=tmp_path) == safe.resolve()
    for allowed_name in ("constraint_dry_run", "trainingless_dry_run"):
        allowed = tmp_path / "logs" / allowed_name
        assert dry_run.validate_output_dir(str(allowed), cwd=tmp_path) == (
            allowed.resolve()
        )
    for unsafe in (
        tmp_path / "data" / "belief_dry_run",
        tmp_path / "logs" / "belief",
        tmp_path / "logs" / "data" / "belief_dry_run",
        tmp_path / "logs" / "train.jsonl" / "belief_dry_run",
        tmp_path / "logs" / "validation.jsonl" / "belief_dry_run",
        tmp_path / "logs" / "test.jsonl" / "belief_dry_run",
        tmp_path / "logs" / "checkpoint" / "belief_dry_run",
    ):
        with pytest.raises(ValueError):
            dry_run.validate_output_dir(str(unsafe), cwd=tmp_path)

    base = dict(
        runtime_config_path="config.yaml",
        output_dir=str(safe),
        game_count=2,
        seeds=(42, 43),
        max_gameplay_calls_per_game=1,
        max_belief_calls_per_game=1,
        max_total_calls_per_game=2,
        max_wall_seconds_per_game=1.0,
        privacy_safe_logging=True,
        audit_only_metadata=True,
    )
    dry_run.RealBackendDryRunConfig(**base)
    with pytest.raises(ValueError, match="game_count"):
        dry_run.RealBackendDryRunConfig(**{**base, "game_count": 1})
    with pytest.raises(ValueError, match="distinct"):
        dry_run.RealBackendDryRunConfig(**{**base, "seeds": (42, 42)})

    required_cli = [
        "--config",
        "missing.yaml",
        "--game-count",
        "1",
        "--seeds",
        "42",
        "43",
        "--output-dir",
        str(safe),
        "--max-gameplay-calls-per-game",
        "1",
        "--max-belief-calls-per-game",
        "1",
        "--max-total-calls-per-game",
        "2",
        "--max-wall-seconds-per-game",
        "1",
        "--privacy-safe-logging",
        "--audit-only-metadata",
    ]
    with pytest.raises(ValueError, match="game_count"):
        dry_run.main(required_cli)
    retry_option = "--max-backend-" + "retries"
    assert retry_option not in {
        option
        for action in dry_run.build_arg_parser()._actions
        for option in action.option_strings
    }


def test_fake_two_game_harness_writes_manifest_and_separate_files(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text("fake: true\n", encoding="utf-8")
    calls = {"runtime": 0, "games": 0}
    runtime_head = "a" * 40

    def fake_git(command, **kwargs):
        assert command == ["git", "rev-parse", "HEAD"]
        assert kwargs == {
            "check": True,
            "capture_output": True,
            "text": True,
        }
        return SimpleNamespace(stdout=runtime_head + "\n")

    monkeypatch.setattr(dry_run.subprocess, "run", fake_git)
    monkeypatch.setattr(dry_run, "normalize_runtime_config", lambda value: value)

    def fake_backend_loader(*_args, **kwargs):
        assert kwargs["max_retries"] == 0
        return {"fake-backend": FakeBackend()}

    monkeypatch.setattr(dry_run, "load_named_backends", fake_backend_loader)

    def fake_runtime(_parsed, *, log_save_path, random_seed, backends):
        assert log_save_path is None
        calls["runtime"] += 1
        agents = [SimpleNamespace(backend=backends["fake-backend"])] * 7
        return object(), agents, ["hidden"] * 7, ["profile"] * 7

    class Collector:
        def __init__(self, path):
            Path(path).touch()

        def close(self):
            return None

    def fake_collector(*, output_path, **_kwargs):
        return Collector(output_path)

    def fake_game(_env, _agents, _roles, *, sample_collector, call_audit):
        assert sample_collector is not None
        assert call_audit is not None
        calls["games"] += 1
        return "finished"

    monkeypatch.setattr(dry_run, "build_runtime", fake_runtime)
    monkeypatch.setattr(dry_run, "build_twd_tom_sample_collector", fake_collector)
    monkeypatch.setattr(dry_run, "run_game", fake_game)

    output = tmp_path / "logs" / "safe_dry_run"
    summary = dry_run.run_dry_run(
        dry_run.RealBackendDryRunConfig(
            runtime_config_path=str(config_path),
            output_dir=str(output),
            game_count=2,
            seeds=(42, 43),
            max_gameplay_calls_per_game=1,
            max_belief_calls_per_game=1,
            max_total_calls_per_game=2,
            max_wall_seconds_per_game=2.0,
            privacy_safe_logging=True,
            audit_only_metadata=True,
        )
    )
    assert calls == {"runtime": 2, "games": 2}
    assert summary["game_count"] == 2
    assert (output / "dry_run_samples.jsonl").is_file()
    assert (output / "dry_run_call_audit.jsonl").is_file()
    manifest = json.loads(
        (output / "dry_run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["dry_run_only"] is True
    assert manifest["formal_training_data"] is False
    assert manifest["source_commit"] == runtime_head
    assert manifest["requested_game_count"] == 2
    assert manifest["seeds"] == [42, 43]
    assert manifest["privacy_safe_logging"] is True
    assert manifest["raw_prompts_saved"] is False
    assert manifest["raw_responses_saved"] is False
    assert manifest["roles_saved"] is False
    assert manifest["private_observations_saved"] is False
    serialized = json.dumps(manifest)
    assert "hidden" not in serialized


@pytest.mark.parametrize(
    "git_result",
    [
        dry_run.subprocess.CalledProcessError(1, ["git", "rev-parse", "HEAD"]),
        "",
        "abc1234",
        "g" * 40,
    ],
    ids=["command-failure", "empty", "short-sha", "non-hex"],
)
def test_invalid_runtime_commit_fails_before_output_or_backend(
    tmp_path, monkeypatch, git_result
):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "logs" / "safe_dry_run"
    backend_loads = []

    def fake_git(*_args, **_kwargs):
        if isinstance(git_result, BaseException):
            raise git_result
        return SimpleNamespace(stdout=git_result)

    monkeypatch.setattr(dry_run.subprocess, "run", fake_git)
    monkeypatch.setattr(
        dry_run,
        "load_named_backends",
        lambda *_args, **_kwargs: backend_loads.append(True),
    )

    with pytest.raises(RuntimeError, match="source commit"):
        dry_run.run_dry_run(
            dry_run.RealBackendDryRunConfig(
                runtime_config_path="missing.yaml",
                output_dir=str(output),
                game_count=2,
                seeds=(42, 43),
                max_gameplay_calls_per_game=1,
                max_belief_calls_per_game=1,
                max_total_calls_per_game=2,
                max_wall_seconds_per_game=1.0,
                privacy_safe_logging=True,
                audit_only_metadata=True,
            )
        )

    assert not output.exists()
    assert backend_loads == []


def test_old_hardcoded_source_commit_is_absent():
    source = Path(dry_run.__file__).read_text(encoding="utf-8")
    old_commit = "9047c12f40a092598e6227f20272e60e9" + "46fdbc7"
    assert old_commit not in source
