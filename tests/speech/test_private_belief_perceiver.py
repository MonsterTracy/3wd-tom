from copy import deepcopy
import json

import pytest

from werewolf.agents.llm_agent import LLMAgent
from werewolf.helper.log_utils import Log
from werewolf.models.twd_tom.public_events import copy_public_events
from werewolf.models.twd_tom.schema import LABEL_PROMPT_VERSION
from werewolf.models.twd_tom.samples import freeze_public_snapshot
from werewolf.speech.private_belief_perceiver import (
    PRIVATE_BELIEF_JSON_SCHEMA,
    PRIVATE_BELIEF_MAX_TOKENS,
    PlayingAgentBeliefReporter,
    STATUS_OK,
    STATUS_PARSE_ERROR,
    STATUS_REPORTER_ERROR,
    STATUS_SEMANTIC_ERROR,
)


class CapturingBackend:
    def __init__(self, response, *, supports_json_schema=False):
        self.response = response
        self.calls = []
        self.supports_json_schema = supports_json_schema

    def chat(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        return self.response


def _snapshot():
    return freeze_public_snapshot(
        game_id="game_001",
        step_idx=2,
        phase="1_day_speech",
        speaker_id=3,
        report_trigger="pre_public_speech",
        observer_ids=[1, 2, 3],
        public_events=[
            {
                "event_idx": 0,
                "event_type": "phase_change",
                "phase": "1_day_speech",
            },
            {
                "event_idx": 1,
                "event_type": "turn_start",
                "speaker": "player2",
            },
            {
                "event_idx": 2,
                "event_type": "public_speech",
                "speaker": "player2",
                "raw_text": "earlier public speech",
                "sp_actions": [
                    ["player2", "point_as_werewolf", "player6"]
                ],
            },
            {
                "event_idx": 3,
                "event_type": "turn_start",
                "speaker": "player3",
            },
        ],
    )


def _observation(player_id=1, identity="Villager"):
    return {
        "observer_id": player_id,
        "current_act_idx": 3,
        "identity": identity,
        "phase": "1_day_speech",
        "valid_action": [],
        "game_log": [],
    }


def _report(
    response,
    *,
    player_id=1,
    identity="Villager",
    known_werewolves=None,
    known_non_werewolves=None,
    supports_json_schema=False,
    observation=None,
):
    backend = CapturingBackend(
        response,
        supports_json_schema=supports_json_schema,
    )
    agent = LLMAgent(backend=backend, model_name="fake")
    result = PlayingAgentBeliefReporter().report(
        agent=agent,
        observation=(
            observation
            if observation is not None
            else _observation(player_id, identity)
        ),
        observer_id=player_id,
        public_snapshot=_snapshot(),
        agent_backend_id="backend_a",
        known_werewolves=list(known_werewolves or []),
        known_non_werewolves=list(
            known_non_werewolves or [f"player{player_id}"]
        ),
    )
    return result, backend, agent


def test_prompt_defines_player_suspicion_without_pair_support_semantics():
    prompt = PlayingAgentBeliefReporter.build_prompt(
        observer_id="player3",
        public_snapshot=_snapshot(),
        known_werewolves=["player1"],
        known_non_werewolves=["player3", "player6"],
    )
    for required in (
        "本次公开发言之前",
        "内部真实怀疑",
        "不要为了阵营策略欺骗",
        "相对更可疑",
        "不要求确定性",
        "不要求完整找到两狼",
        "不要求恰好两人",
        "仍有可能",
        "输出空数组",
        "允许全部列出",
        "HARD CONSTRAINTS",
        'MUST INCLUDE: ["player1"]',
        'MUST EXCLUDE: ["player3","player6"]',
        "Your output is invalid if it omits any MUST INCLUDE player",
        "没有额外软怀疑时，精确输出 MUST INCLUDE",
        "Your observer identity is exactly: player3",
        'known_werewolves: ["player1"]',
        'known_non_werewolves: ["player3","player6"]',
        "Current legal_candidates: player1, player2, player4, player5, player7",
        "player1, player2, player3, player4, player5, player6, player7",
        '"raw_text":"earlier public speech"',
        '"sp_actions":[["player2","point_as_werewolf","player6"]]',
        "Canonical pre-speech public_events:",
        '{"suspected_werewolves":[...]}',
        f"prompt_version: {LABEL_PROMPT_VERSION}",
    ):
        assert required in prompt
    for forbidden in (
        "belief_mode",
        "no_extra_narrowing",
        "Mode B",
        "strictly reduce",
        "pair support",
        "complete pair",
    ):
        assert forbidden not in prompt


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"suspected_werewolves":[]}', []),
        ('{"suspected_werewolves":["player3"]}', ["player3"]),
        (
            '{"suspected_werewolves":["player6","player2","player3"]}',
            ["player2", "player3", "player6"],
        ),
        (
            '{"suspected_werewolves":'
            '["player2","player3","player4","player5","player6","player7"]}',
            ["player2", "player3", "player4", "player5", "player6", "player7"],
        ),
    ],
)
def test_parser_accepts_player_level_sets_of_any_legal_size(raw, expected):
    assert PlayingAgentBeliefReporter.parse_response(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        '{"suspected_werewolves":[],"confidence":1}',
        '{"belief_mode":"no_extra_narrowing"}',
        '{"belief_mode":"narrowed","believed_werewolves":["player3"]}',
        '{"believed_werewolves":["player3"]}',
        '```json\n{"suspected_werewolves":[]}\n```',
        '{"suspected_werewolves":[]}{"suspected_werewolves":[]}',
        '{"suspected_werewolves":"player3"}',
        '{"suspected_werewolves":[3]}',
        '{"suspected_werewolves":[1,6]}',
        '{"suspected_werewolves":["1","6"]}',
        '{"suspected_werewolves":["player3","player3"]}',
        '{"suspected_werewolves":["player8"]}',
        '{"suspected_werewolves":["Player_3"]}',
        '{"suspected_werewolves":[],"extra":true}',
    ],
)
def test_parser_rejects_noncanonical_or_old_structures(raw):
    with pytest.raises((TypeError, ValueError)):
        PlayingAgentBeliefReporter.parse_response(raw)
    result, _, _ = _report(raw)
    assert result["status"] == STATUS_PARSE_ERROR
    assert result["suspected_werewolves"] is None


@pytest.mark.parametrize(
    (
        "response",
        "player_id",
        "identity",
        "known_wolves",
        "known_non_wolves",
        "status",
        "expected",
    ),
    [
        (
            '{"suspected_werewolves":[]}',
            1,
            "Villager",
            [],
            ["player1"],
            STATUS_OK,
            [],
        ),
        (
            '{"suspected_werewolves":["player3"]}',
            1,
            "Villager",
            [],
            ["player1"],
            STATUS_OK,
            ["player3"],
        ),
        (
            '{"suspected_werewolves":["player2","player3","player6"]}',
            1,
            "Villager",
            [],
            ["player1"],
            STATUS_OK,
            ["player2", "player3", "player6"],
        ),
        (
            '{"suspected_werewolves":'
            '["player2","player3","player4","player5","player6","player7"]}',
            1,
            "Villager",
            [],
            ["player1"],
            STATUS_OK,
            ["player2", "player3", "player4", "player5", "player6", "player7"],
        ),
        (
            '{"suspected_werewolves":'
            '["player1","player3","player4","player5"]}',
            3,
            "Seer",
            ["player3"],
            ["player2", "player6", "player7"],
            STATUS_OK,
            ["player1", "player3", "player4", "player5"],
        ),
        (
            '{"suspected_werewolves":[]}',
            3,
            "Seer",
            ["player2"],
            ["player3"],
            STATUS_SEMANTIC_ERROR,
            None,
        ),
        (
            '{"suspected_werewolves":["player2"]}',
            3,
            "Seer",
            ["player2"],
            ["player3"],
            STATUS_OK,
            ["player2"],
        ),
        (
            '{"suspected_werewolves":["player2"]}',
            3,
            "Seer",
            [],
            ["player2", "player3"],
            STATUS_SEMANTIC_ERROR,
            None,
        ),
        (
            '{"suspected_werewolves":["player3"]}',
            1,
            "Villager",
            [],
            ["player1", "player3"],
            STATUS_SEMANTIC_ERROR,
            None,
        ),
        (
            '{"suspected_werewolves":["player5"]}',
            4,
            "Witch",
            [],
            ["player4", "player5"],
            STATUS_SEMANTIC_ERROR,
            None,
        ),
        (
            '{"suspected_werewolves":["player2","player5"]}',
            2,
            "Werewolf",
            ["player2", "player5"],
            ["player1", "player3", "player4", "player6", "player7"],
            STATUS_OK,
            ["player2", "player5"],
        ),
        (
            '{"suspected_werewolves":["player2"]}',
            2,
            "Werewolf",
            ["player2", "player5"],
            ["player1", "player3", "player4", "player6", "player7"],
            STATUS_SEMANTIC_ERROR,
            None,
        ),
    ],
)
def test_suspicion_semantics_use_only_hard_knowledge(
    response,
    player_id,
    identity,
    known_wolves,
    known_non_wolves,
    status,
    expected,
):
    result, _, _ = _report(
        response,
        player_id=player_id,
        identity=identity,
        known_werewolves=known_wolves,
        known_non_werewolves=known_non_wolves,
    )
    assert result["status"] == status
    assert result["suspected_werewolves"] == expected


def test_report_does_not_call_offline_pair_projector(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("pair projector must remain offline")

    monkeypatch.setattr(
        "werewolf.models.twd_tom.belief_labels.suspicion_set_to_pair_target",
        forbidden,
    )
    result, backend, agent = _report(
        '{"suspected_werewolves":["player3"]}',
    )
    assert result["status"] == STATUS_OK
    assert result["suspected_werewolves"] == ["player3"]
    assert len(backend.calls) == 1
    assert not hasattr(agent, "messages")


@pytest.mark.parametrize(
    ("supports_json_schema", "format_type"),
    [
        (False, "json_object"),
        (True, "json_schema"),
    ],
)
def test_belief_request_uses_capability_format_and_fixed_budget(
    supports_json_schema,
    format_type,
):
    result, backend, _ = _report(
        '{"suspected_werewolves":[]}',
        supports_json_schema=supports_json_schema,
    )

    assert result["status"] == STATUS_OK
    assert len(backend.calls) == 1
    request = backend.calls[0]
    assert request["max_tokens"] == PRIVATE_BELIEF_MAX_TOKENS
    assert request["response_format"]["type"] == format_type
    assert request["extra_body"] == {
        "thinking": {"type": "disabled"}
    }
    if supports_json_schema:
        assert request["response_format"]["json_schema"] == {
            "name": "private_belief_report",
            "strict": True,
            "schema": PRIVATE_BELIEF_JSON_SCHEMA,
        }
        transport_schema = request["response_format"][
            "json_schema"
        ]["schema"]
        array_schema = transport_schema["properties"][
            "suspected_werewolves"
        ]
        for unsupported in (
            "uniqueItems",
            "contains",
            "minContains",
            "maxContains",
        ):
            assert unsupported not in array_schema
        assert transport_schema["additionalProperties"] is False
        assert transport_schema["required"] == [
            "suspected_werewolves"
        ]
        assert array_schema["minItems"] == 0
        assert array_schema["maxItems"] == 7
        assert array_schema["items"] == {
            "type": "string",
            "enum": [f"player{i}" for i in range(1, 8)],
        }
    else:
        assert request["response_format"] == {
            "type": "json_object"
        }


def test_reporter_uses_public_history_once_and_only_role_private_logs():
    observation = _observation(
        player_id=1,
        identity="Werewolf",
    )
    observation["game_log"] = [
        Log(
            viewer=1,
            source=2,
            target=0,
            content={
                "speech_content": "earlier public speech",
            },
            day=1,
            time="day",
            event="speech",
        ),
        Log(
            viewer=1,
            source=1,
            target=0,
            content={"wolf_team": [1, 5]},
            day=1,
            time="night",
            event="werewolf_team_info",
        ),
        Log(
            viewer=1,
            source=3,
            target=7,
            content={"cheked_identity": "bad"},
            day=1,
            time="night",
            event="skill_seer",
        ),
    ]

    result, backend, _ = _report(
        '{"suspected_werewolves":["player1","player5"]}',
        player_id=1,
        identity="Werewolf",
        known_werewolves=["player1", "player5"],
        known_non_werewolves=[
            "player2",
            "player3",
            "player4",
            "player6",
            "player7",
        ],
        observation=observation,
    )

    assert result["status"] == STATUS_OK
    request_messages = backend.calls[0]["messages"]
    assert len(request_messages) == 2
    private_context = json.loads(
        request_messages[0]["content"].split("\n", 1)[1]
    )
    assert "legally_visible_history" not in private_context
    assert "public_events" not in private_context
    assert (
        "Canonical pre-speech public_events:"
        not in request_messages[0]["content"]
    )
    assert request_messages[1]["content"].count(
        "Canonical pre-speech public_events:"
    ) == 1
    public_history_text = (
        request_messages[1]["content"]
        .split(
            "Canonical pre-speech public_events:\n",
            1,
        )[1]
        .split("\n\n", 1)[0]
    )
    public_history = json.loads(public_history_text)
    assert public_history == copy_public_events(
        _snapshot().public_events
    )

    messages = json.dumps(
        request_messages,
        ensure_ascii=False,
    )
    assert messages.count("earlier public speech") == 1
    assert "狼人队伍的成员是1号、5号" in messages
    assert "查验了7号" not in messages


@pytest.mark.parametrize(
    (
        "identity",
        "game_log",
        "required",
        "forbidden",
    ),
    [
        (
            "Seer",
            [
                Log(
                    1,
                    1,
                    6,
                    {"cheked_identity": "bad"},
                    1,
                    "night",
                    "skill_seer",
                ),
                Log(
                    1,
                    4,
                    7,
                    {"cheked_identity": "good"},
                    1,
                    "night",
                    "skill_seer",
                ),
            ],
            "查验了6号的身份是狼人",
            "查验了7号",
        ),
        (
            "Witch",
            [
                Log(
                    1,
                    0,
                    5,
                    {},
                    1,
                    "night",
                    "kill_decision",
                ),
                Log(
                    1,
                    1,
                    5,
                    {"heal": True},
                    1,
                    "night",
                    "skill_witch",
                ),
                Log(
                    1,
                    4,
                    7,
                    {"poison": True},
                    1,
                    "night",
                    "skill_witch",
                ),
            ],
            "使用解药治疗了5号",
            "使用毒药毒害了7号",
        ),
    ],
)
def test_reporter_keeps_only_observer_role_private_knowledge(
    identity,
    game_log,
    required,
    forbidden,
):
    agent = LLMAgent(
        backend=CapturingBackend(
            '{"suspected_werewolves":[]}'
        ),
        model_name="fake",
    )
    observation = _observation(
        player_id=1,
        identity=identity,
    )
    observation["game_log"] = game_log

    messages = json.dumps(
        agent._build_readonly_belief_context(
            observation
        ),
        ensure_ascii=False,
    )

    assert required in messages
    assert forbidden not in messages


def test_full_candidate_report_succeeds_once_without_retry():
    result, backend, _ = _report(
        '{"suspected_werewolves":'
        '["player2","player3","player4","player5","player6","player7"]}',
    )
    assert result["status"] == STATUS_OK
    assert result["suspected_werewolves"] == [
        "player2", "player3", "player4", "player5", "player6", "player7"
    ]
    assert result["error"] is None
    assert len(backend.calls) == 1


def test_current_prompt_version_is_player_suspicion_v2():
    assert LABEL_PROMPT_VERSION == (
        "classic7_pre_speech_player_suspicion_prompt_v2"
    )


def test_reporter_distinguishes_parse_backend_and_context_failures():
    parse_result, _, _ = _report("not json")
    assert parse_result["status"] == STATUS_PARSE_ERROR

    class BrokenAgent:
        def report_suspected_werewolves_readonly(self, **kwargs):
            raise RuntimeError("backend unavailable")

    broken = PlayingAgentBeliefReporter().report(
        agent=BrokenAgent(),
        observation=_observation(),
        observer_id=1,
        public_snapshot=_snapshot(),
        agent_backend_id="backend_a",
        known_werewolves=[],
        known_non_werewolves=["player1"],
    )
    assert broken["status"] == STATUS_REPORTER_ERROR

    mismatch = deepcopy(_observation())
    mismatch["current_act_idx"] = 4
    context = PlayingAgentBeliefReporter().report(
        agent=BrokenAgent(),
        observation=mismatch,
        observer_id=1,
        public_snapshot=_snapshot(),
        agent_backend_id="backend_a",
        known_werewolves=[],
        known_non_werewolves=["player1"],
    )
    assert context["status"] == STATUS_REPORTER_ERROR
    assert "speaker mismatch" in context["error"]
