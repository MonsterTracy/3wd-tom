from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import time

import httpx
import openai
import pytest

import script.twd_tom.real_backend_dry_run as dry_run
from werewolf.agents.gpt_agent import GPTAgent
from werewolf.agents.llm_agent import (
    GameplaySpeechQualityError,
    LLMAgent,
)
from werewolf.models import SpeechPerceiver
from werewolf.models.public_belief_matrix.public_prefix import (
    build_public_belief_matrix_visible_prefix,
)
from werewolf.models.public_belief_matrix.reporter import PublicBeliefMatrixReporter
from werewolf.speech.private_belief_perceiver import (
    PRIVATE_BELIEF_MAX_TOKENS,
    PlayingAgentBeliefReporter,
)
from werewolf.speech.speech_perceiver import (
    SPEECH_PARSER_MAX_TOKENS,
)


class FakeBackend:
    def __init__(
        self,
        responses=None,
        *,
        usage=None,
        failures=0,
        error=None,
        supports_json_schema=False,
    ):
        self.default_model = "fake-model"
        self.supports_json_schema = supports_json_schema
        self.responses = list(responses or ["fake response"])
        self.usage = usage
        self.failures = failures
        self.error = error
        self.calls = []

    def chat_with_metadata(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        if self.error is not None:
            raise self.error
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
        public_events=tuple(_public_events()),
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
    supports_json_schema=False,
):
    session, writer = _new_session(tmp_path)
    fake = FakeBackend(
        [response, "next action"],
        supports_json_schema=supports_json_schema,
    )
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
    assert records[0]["request_max_tokens"] is None
    assert records[0]["response_format_type"] is None
    assert records[0]["error_message"] is None
    assert records[0]["finish_reason"] is None
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
    "finish_reason",
    ["stop", "length"],
)
def test_gameplay_limit_and_finish_reason_are_audited(
    tmp_path,
    finish_reason,
):
    session, writer = _new_session(tmp_path)
    fake = FakeBackend(
        responses=["public speech"],
        usage={"finish_reason": finish_reason},
    )
    agent = GPTAgent(
        backend=_backend(fake, session),
        model_name="fake-model",
        gameplay_max_tokens=512,
    )
    agent.rate_limit = 0

    with session.gameplay_context(
        acting_player_id=1,
        observation=_observation(),
        public_events=_public_events(),
    ):
        if finish_reason == "length":
            with pytest.raises(
                GameplaySpeechQualityError,
                match="truncated gameplay public speech",
            ):
                agent.act(
                    {
                        **_observation(),
                        "valid_action": ("speech", -1),
                    }
                )
        else:
            assert agent.act(
                {
                    **_observation(),
                    "valid_action": ("speech", -1),
                }
            ) == ("speech", "public speech")
    writer.close()

    records, _serialized = _records(tmp_path / "audit.jsonl")
    assert len(records) == 1
    assert records[0]["call_category"] == "gameplay"
    assert records[0]["request_max_tokens"] == 512
    assert records[0]["finish_reason"] == finish_reason
    assert records[0]["usage_available"] is True
    assert records[0]["response_character_count"] == len("public speech")
    assert records[0]["response_sha256"] == dry_run._sha256_text(
        "public speech"
    )


def test_strict_day_cognition_is_one_audited_gameplay_call(
    tmp_path,
):
    session, writer = _new_session(tmp_path)
    fake = FakeBackend(
        responses=[
            json.dumps({
                "belief": "当前信息有限。",
                "concise": "继续观察。",
                "roles": {
                    f"player{player_id}": "unknown"
                    for player_id in range(2, 8)
                },
                "public_content_action_indices": [],
                "public_vote_stance_index": 0,
                "evidence_claim_ids": [],
            }, ensure_ascii=False),
        ],
        usage={"finish_reason": "stop"},
        supports_json_schema=True,
    )
    agent = GPTAgent(
        backend=_backend(fake, session),
        model_name="fake-model",
        gameplay_prompt_profile="strict_classic7",
        gameplay_max_tokens=512,
    )
    agent.rate_limit = 0
    observation = {
        **_observation(),
        "valid_action": ("speech", -1),
        "authoritative_public_state": {
            "day": 1,
            "day_or_night": "day",
            "phase": "speech",
            "last_night_result": {"day": 0, "dead_players": []},
            "prior_exiles": [],
            "alive_players": [1, 2, 3, 4, 5, 6, 7],
            "suggestible_exile_targets": [2, 3, 4, 5, 6, 7],
        },
    }

    with session.gameplay_context(
        acting_player_id=1,
        observation=observation,
        public_events=_public_events(),
    ):
        assert agent.act(observation) == (
            "speech",
            "这一轮我暂不作明确的身份、查验、技能或投票表态。",
        )
    writer.close()

    records, _serialized = _records(tmp_path / "audit.jsonl")
    assert [record["call_category"] for record in records] == ["gameplay"]
    assert [record["response_format_type"] for record in records] == [
        "json_schema",
    ]


def test_bad_request_error_message_is_capped_and_privacy_safe(
    tmp_path,
):
    session, writer = _new_session(tmp_path)
    error_body = (
        "vLLM rejected unsupported keyword uniqueItems: "
        + "x" * 1200
    )
    request = httpx.Request(
        "POST",
        "http://127.0.0.1:8000/v1/chat/completions",
        headers={"Authorization": "Bearer audit-secret"},
    )
    response = httpx.Response(
        400,
        request=request,
    )
    bad_request = openai.BadRequestError(
        message=error_body,
        response=response,
        body={"error": {"message": error_body}},
    )
    fake = FakeBackend(error=bad_request)
    audited = _backend(fake, session)
    private_prompt = "PRIVATE_PROMPT do-not-audit"

    with session.gameplay_context(
        acting_player_id=1,
        observation=_observation(),
        public_events=_public_events(),
    ):
        with pytest.raises(openai.BadRequestError):
            audited.chat(
                messages=[
                    {
                        "role": "user",
                        "content": private_prompt,
                    }
                ],
                model="fake-model",
            )
    writer.close()

    records, serialized = _records(tmp_path / "audit.jsonl")
    assert len(records) == 1
    assert records[0]["dispatch_status"] == "error"
    assert records[0]["error_type"] == "BadRequestError"
    assert (
        "vLLM rejected unsupported keyword uniqueItems"
        in records[0]["error_message"]
    )
    assert len(records[0]["error_message"]) == 1000
    for forbidden in (
        private_prompt,
        "do-not-audit",
        "audit-secret",
        "Authorization",
    ):
        assert forbidden not in serialized


def test_audit_writer_accepts_legacy_record_without_error_message(
    tmp_path,
):
    writer = dry_run.PrivacySafeAuditWriter(
        tmp_path / "legacy_audit.jsonl"
    )
    writer.write({"game_id": "legacy"})
    writer.close()

    records, _serialized = _records(
        tmp_path / "legacy_audit.jsonl"
    )
    assert records == [{"game_id": "legacy"}]


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
    assert belief["request_max_tokens"] == PRIVATE_BELIEF_MAX_TOKENS
    assert belief["response_format_type"] == "json_object"
    assert belief["report_nonce_absent_from_next_action"] is True
    assert belief["report_response_absent_from_next_action"] is True
    assert belief["next_action_check_not_reached"] is False
    assert belief["state_hash_before"] == belief["state_hash_after"]
    assert "TWD_TOM_REPORT_AUDIT_" not in serialized
    assert '{"suspected_werewolves":[]}' not in serialized


def test_audited_pbm_request_has_no_nonce_or_snapshot_metadata(tmp_path):
    session, writer = _new_session(tmp_path)
    fake = FakeBackend(
        ['{"suspected_werewolves":[]}', '{"suspected_werewolves":[]}']
    )
    audited = _backend(fake, session)
    prefix = build_public_belief_matrix_visible_prefix(
        [
            {"event_idx": 0, "event_type": "phase_change", "phase": "1_day_speech"},
            {"event_idx": 1, "event_type": "turn_start", "speaker": "player1"},
            {
                "event_idx": 2,
                "event_type": "public_speech",
                "speaker": "player1",
                "raw_text": "RAW-TEXT-CANARY",
                "sp_actions": [],
            },
        ]
    )
    cutoff = SimpleNamespace(
        game_id="GAME-ID-CANARY",
        step_idx=987654,
        phase="PHASE-CANARY",
        speaker_id=7,
        public_action_count=0,
        public_history_digest="d" * 64,
    )
    reporter = PublicBeliefMatrixReporter(audit_hook=session)
    for observer in ("player1", "player2"):
        assert reporter.report(
            visible_prefix=prefix,
            observer_id=observer,
            cutoff=cutoff,
            backend=audited,
            backend_id="fake-backend",
            model_name="fake-model",
        )["status"] == "ok"
    session.finish_game()
    writer.close()

    prompts = [call["messages"][0]["content"] for call in fake.calls]
    for prompt in prompts:
        assert "GAME-ID-CANARY" not in prompt
        assert "987654" not in prompt
        assert "PHASE-CANARY" not in prompt
        assert "TWD_TOM_REPORT_AUDIT_" not in prompt
        assert "RAW-TEXT-CANARY" not in prompt
    assert prompts[0].replace("observer=player1", "observer=ROW") == (
        prompts[1].replace("observer=player2", "observer=ROW")
    )


def test_audit_records_strict_belief_and_pipe_parser_contracts(
    tmp_path,
):
    session, writer, fake, audited, result = _perform_report(
        tmp_path,
        supports_json_schema=True,
    )
    assert result["status"] == "ok"
    parser = SpeechPerceiver(
        backend=audited,
        model_name="fake-model",
    )
    with session.gameplay_context(
        acting_player_id=1,
        observation=_observation(),
        public_events=_public_events(),
    ):
        assert parser.parse(
            speaker=1,
            speech="public speech",
            day=1,
            phase="1_day_speech",
        ) == []
    session.finish_game()
    writer.close()

    records, _serialized = _records(tmp_path / "audit.jsonl")
    belief = next(
        record
        for record in records
        if record["call_category"] == "belief"
    )
    parser_record = next(
        record
        for record in records
        if record["request_max_tokens"] == SPEECH_PARSER_MAX_TOKENS
    )
    assert belief["request_max_tokens"] == PRIVATE_BELIEF_MAX_TOKENS
    assert belief["response_format_type"] == "json_schema"
    assert parser_record["response_format_type"] is None
    assert fake.calls[1]["response_format"] is None


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


def test_player_logs_are_isolated_and_closed_between_games(
    tmp_path,
    monkeypatch,
):
    game_action_counts = {
        "game_001_seed_343": 2,
        "game_002_seed_344": 3,
        "game_003_seed_345": 1,
    }
    closed_agents = []

    class FakeCollector:
        def close(self):
            return None

    def fake_build_runtime(
        _parsed_yaml,
        *,
        log_save_path,
        **_kwargs,
    ):
        agents = [LLMAgent(
            log_file=str(log_save_path / "Player_1.jsonl")
        )]
        closed_agents.extend(agents)
        return (
            SimpleNamespace(log_dir=log_save_path),
            agents,
            ["Villager"],
            [],
        )

    def fake_run_game(env, agents, _roles, **_kwargs):
        game_id = env.log_dir.name
        game_log = [
            {"source": 1, "event": "speech"}
            for _index in range(game_action_counts[game_id])
        ]
        for action_index in range(game_action_counts[game_id]):
            agents[0].logger.info(
                "speech",
                extra={
                    "game_id": game_id,
                    "player_id": 1,
                    "action_index": action_index,
                },
            )
        if game_id == "game_003_seed_345":
            raise RuntimeError("mock game failure")
        (env.log_dir / "game_log.json").write_text(
            json.dumps(game_log),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        dry_run,
        "normalize_runtime_config",
        lambda parsed: parsed,
    )
    monkeypatch.setattr(
        dry_run,
        "load_named_backends",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        dry_run,
        "build_runtime",
        fake_build_runtime,
    )
    monkeypatch.setattr(
        dry_run,
        "build_twd_tom_sample_collector",
        lambda **_kwargs: FakeCollector(),
    )
    monkeypatch.setattr(
        dry_run,
        "run_game",
        fake_run_game,
    )

    audit_path = tmp_path / "call_audit.jsonl"
    samples_path = tmp_path / "raw.jsonl"

    def run_mock_game(game_id, seed, writer):
        game_dir = tmp_path / game_id
        game_dir.mkdir()
        dry_run.run_real_backend_game(
            parsed_yaml={},
            samples_path=samples_path,
            log_dir=game_dir,
            game_id=game_id,
            seed=seed,
            budget=dry_run.DryRunBudget(
                game_id=game_id,
                max_gameplay_calls=10,
                max_belief_calls=10,
                max_total_calls=20,
                max_wall_seconds=60.0,
            ),
            writer=writer,
        )
        return game_dir

    with dry_run.PrivacySafeAuditWriter(audit_path) as writer:
        first_dir = run_mock_game(
            "game_001_seed_343", 343, writer
        )
        first_contents = (
            first_dir / "Player_1.jsonl"
        ).read_bytes()
        second_dir = run_mock_game(
            "game_002_seed_344", 344, writer
        )
        failed_dir = tmp_path / "game_003_seed_345"
        with pytest.raises(RuntimeError, match="mock game failure"):
            run_mock_game(
                "game_003_seed_345", 345, writer
            )

    assert (first_dir / "Player_1.jsonl").read_bytes() == first_contents
    assert all(not agent.has_log for agent in closed_agents)
    assert len(
        (failed_dir / "Player_1.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ) == 1

    for game_dir in (first_dir, second_dir):
        game_log = json.loads(
            (game_dir / "game_log.json").read_text(encoding="utf-8")
        )
        player_records = [
            json.loads(line)
            for line in (game_dir / "Player_1.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line
        ]
        assert len(player_records) == len(game_log)
        assert {
            record["game_id"] for record in player_records
        } == {game_dir.name}


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


def test_explicit_logs_root_is_bounded_and_checks_only_relative_components(
    tmp_path,
):
    logs_root = tmp_path / "data" / "project" / "logs"
    safe = logs_root / "formal_batch_test"
    assert dry_run.validate_output_dir(
        str(safe),
        logs_root=str(logs_root),
        required_name_token="formal_batch",
    ) == safe.resolve()

    for unsafe in (
        logs_root.parent / "formal_batch_test",
        logs_root / ".." / "formal_batch_test",
        logs_root / "data" / "formal_batch_test",
    ):
        with pytest.raises(ValueError):
            dry_run.validate_output_dir(
                str(unsafe),
                logs_root=str(logs_root),
                required_name_token="formal_batch",
            )

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
