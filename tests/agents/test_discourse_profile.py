import hashlib
import json
from types import SimpleNamespace

import pytest

from run_random import eval as run_game
from werewolf.agents import prompt_template_v0
from werewolf.agents.gpt_agent import GPTAgent
from werewolf.agents.llm_agent import (
    GameplaySpeechQualityError,
    PublicSpeechPlanValidationError,
    discourse_public_speech_plan_json_schema,
    public_evidence_player_ids,
    validate_discourse_public_speech_plan,
    validate_gameplay_public_speech,
)
from werewolf.agents.prompt_template_v0 import (
    STRICT_CLASSIC7_DISCOURSE_GAMEPLAY_PROMPT_PROFILE,
    _render_discourse_public_evidence,
    build_strict_classic7_discourse_speech_render_prompt,
    build_strict_classic7_speech_plan_prompt,
    build_strict_classic7_speech_render_prompt,
)
from werewolf.helper.log_utils import Log


def _public_events():
    return [
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
            "raw_text": "我认为player3是村民。",
            "sp_actions": [["player2", "point_as_villager", "player3"]],
        },
        {
            "event_idx": 3,
            "event_type": "turn_start",
            "speaker": "player4",
        },
        {
            "event_idx": 4,
            "event_type": "public_speech",
            "speaker": "player4",
            "raw_text": "我认为player3是狼人。",
            "sp_actions": [["player4", "point_as_werewolf", "player3"]],
        },
        {
            "event_idx": 5,
            "event_type": "turn_start",
            "speaker": "player1",
        },
    ]


def _observation(profile=STRICT_CLASSIC7_DISCOURSE_GAMEPLAY_PROMPT_PROFILE):
    observation = {
        "phase": "1_day_speech",
        "identity": "Werewolf",
        "current_act_idx": 1,
        "game_log": [
            Log(
                viewer=[0],
                source=-1,
                target=[0, 6],
                content={"wolf_team": [7], "private_canary": "WOLF-SECRET"},
                day=0,
                time="0_night_skill_wolf",
                event="werewolf_team_info",
            ),
            Log(
                viewer=[0],
                source=0,
                target=6,
                content={"kill_target": 6, "private_canary": "KILL-SECRET"},
                day=0,
                time="0_night_skill_wolf",
                event="kill_decision",
            ),
        ],
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
    if profile == STRICT_CLASSIC7_DISCOURSE_GAMEPLAY_PROMPT_PROFILE:
        observation["canonical_public_events"] = _public_events()
    return observation


class MetadataBackend:
    supports_json_schema = True

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.session = SimpleNamespace(game_id="discourse_game_001_seed_881")

    def chat_with_metadata(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0), {"finish_reason": "stop"}


def _valid_payload(refs=None):
    return {
        "public_actions": [{"action": "oppose", "target": 4}],
        "public_evidence_refs": [0, 2, 4] if refs is None else refs,
    }


def _validate(payload):
    return validate_discourse_public_speech_plan(
        payload,
        suggestible_player_ids=(2, 3, 4, 5, 6, 7),
        player_id=1,
        speaker_role="Werewolf",
        phase="1_day_speech",
        public_event_indices=(0, 1, 2, 3, 4, 5),
        game_context="discourse_game_001_seed_881",
    )


def test_discourse_plan_schema_and_validator_require_enriched_contract():
    plan = _validate(_valid_payload())
    assert plan.actions_as_list() == [{"action": "oppose", "target": 4}]
    assert plan.public_evidence_refs == (0, 2, 4)

    schema = discourse_public_speech_plan_json_schema(
        suggestible_player_ids=(2, 3, 4, 5, 6, 7),
        speaker_id=1,
        speaker_role="Werewolf",
        public_event_indices=(0, 1, 2, 3, 4, 5),
    )
    assert schema["required"] == [
        "public_actions",
        "public_evidence_refs",
    ]
    assert set(schema["properties"]) == {
        "public_actions",
        "public_evidence_refs",
    }
    assert "speaking_strategy" not in json.dumps(schema)
    assert schema["properties"]["public_evidence_refs"]["maxItems"] == 3
    assert "uniqueItems" not in schema["properties"]["public_evidence_refs"]


def test_discourse_plan_fails_closed_for_invalid_evidence_refs_and_fields():
    for refs in ([6], [99], [0, 1, 2, 3]):
        with pytest.raises(PublicSpeechPlanValidationError):
            _validate(_valid_payload(refs))

    with pytest.raises(
        PublicSpeechPlanValidationError,
        match="public_evidence_refs cannot contain duplicates",
    ):
        _validate(_valid_payload([0, 0]))

    invalid_fields = _valid_payload()
    invalid_fields["strategy"] = "extra"
    with pytest.raises(PublicSpeechPlanValidationError, match="fields"):
        _validate(invalid_fields)

    assert _validate(_valid_payload([])).public_evidence_refs == ()


def test_discourse_renderer_receives_only_selected_public_evidence():
    selected = [_public_events()[0], _public_events()[2], _public_events()[4]]
    renderer = build_strict_classic7_discourse_speech_render_prompt(
        phase_text="第1天白天公开发言",
        actor=1,
        public_actions=[{"action": "oppose", "target": 4}],
        selected_public_evidence=selected,
    )

    assert "oppose(player4)" in renderer
    assert "speaking_strategy" not in renderer
    assert '"epistemic_type":"observable_public_fact"' in renderer
    assert renderer.count('"epistemic_type":"public_claim"') == 2
    assert renderer.count('"truth_status":"not_authoritative_truth"') == 2
    assert "public_claim 及其中的 sp_actions 只能表述为对应玩家曾公开说过或声称过" in renderer
    assert '"event_idx":2' in renderer
    assert '"event_idx":4' in renderer
    assert "我认为player3是村民" in renderer
    assert "我认为player3是狼人" in renderer
    assert '"event_idx":0' in renderer
    assert '"event_idx":1' not in renderer
    assert '"event_idx":3' not in renderer
    assert '"event_idx":5' not in renderer
    assert "player6" not in renderer
    assert "player7" not in renderer
    assert "WOLF-SECRET" not in renderer
    assert "KILL-SECRET" not in renderer
    assert "不得升级为 player4 是狼人" in renderer
    assert "不得新增计划外角色判断、查验结果、救/毒/守声明或投票目标" in renderer
    assert "不得新增计划外玩家、角色判断、技能结果或投票目标" not in renderer


def test_speech_action_token_is_a_non_authoritative_public_claim():
    speech_action = {
        "token_type": "speech_action",
        "subject": "player2",
        "action": "point_as_werewolf",
        "object": "player4",
        "phase": None,
        "day": 0,
    }

    represented = json.loads(
        _render_discourse_public_evidence([speech_action])
    )

    assert represented == [
        {
            "epistemic_type": "public_claim",
            "claimant": "player2",
            "truth_status": "not_authoritative_truth",
            "event": speech_action,
        }
    ]
    assert all(
        item["epistemic_type"] != "observable_public_fact"
        for item in represented
    )


def test_evidence_player_boundary_is_exact_and_baseline_stays_strict():
    selected = [_public_events()[0], _public_events()[2], _public_events()[4]]
    assert public_evidence_player_ids(selected) == frozenset({2, 3, 4})

    validate_gameplay_public_speech(
        "player4值得怀疑。",
        player_id=1,
        phase="1_day_speech",
        planned_player_ids={4},
    )
    with pytest.raises(GameplaySpeechQualityError, match="missing"):
        validate_gameplay_public_speech(
            "我暂时保持观望。",
            player_id=1,
            phase="1_day_speech",
            planned_player_ids={4},
        )
    with pytest.raises(GameplaySpeechQualityError, match="unplanned player"):
        validate_gameplay_public_speech(
            "player4值得怀疑，player6也一样。",
            player_id=1,
            phase="1_day_speech",
            planned_player_ids={4},
        )

    validate_gameplay_public_speech(
        "player2此前谈到player3，因此我暂不信任player4。",
        player_id=1,
        phase="1_day_speech",
        planned_player_ids={4},
        additional_allowed_player_ids={2, 3},
    )
    validate_gameplay_public_speech(
        "player4值得怀疑。",
        player_id=1,
        phase="1_day_speech",
        planned_player_ids={4},
        additional_allowed_player_ids={2, 3},
    )
    with pytest.raises(GameplaySpeechQualityError, match="unplanned player"):
        validate_gameplay_public_speech(
            "player4值得怀疑，player6也一样。",
            player_id=1,
            phase="1_day_speech",
            planned_player_ids={4},
            additional_allowed_player_ids={2, 3},
        )
    with pytest.raises(GameplaySpeechQualityError, match="missing"):
        validate_gameplay_public_speech(
            "player2此前谈到player3。",
            player_id=1,
            phase="1_day_speech",
            planned_player_ids={4},
            additional_allowed_player_ids={2, 3},
        )

    validate_gameplay_public_speech(
        "我自己暂时不信任player4。",
        player_id=1,
        phase="1_day_speech",
        planned_player_ids={4},
    )


def test_baseline_renderer_prompt_remains_byte_exact():
    assert build_strict_classic7_speech_render_prompt(
        phase_text="第1天白天公开发言",
        actor=1,
        public_actions=[{"action": "oppose", "target": 4}],
    ) == """【当前发言者】player1
【当前阶段】第1天白天公开发言
【必须逐项表达的公开计划】
下面共有 1 项。
每一项都是独立且必须表达的原子命题。不得省略任何一项。
不得把某一项的 predicate 转移给另一项的 target。
最终正文必须逐个、独立、显式写出每个 target 的 playerN 或 N号，使每个 target 都能被单独识别；target 是当前发言者自己时也不例外。
即使同一 predicate 涉及 3 个或更多 target，也必须逐个列出。本计划必须逐个显式出现：player4/4号。
禁止用连续编号范围或集合/聚合指代替代任何 target，例如“N号至M号”“N-M号”“所有玩家”“全部玩家”“大家”“其他所有人”。
可以将多项自然合并为 2–4 句，但所有原子语义必须保留，且不能在合并时省略任何 target identity。
1. oppose(player4)
   必须明确表达质疑、反对、不认可 player4 或认为其发言可疑；不得因此自动产生狼人判断或投票意图。
【输出合同】
- 只输出 2–4 句中文公开发言正文，建议不超过 200 个汉字。
- 只将上面的公开表达义务表述成自然语言，不补充计划之外的游戏事实、玩家或判断。
- 不得新增计划外玩家、角色判断、技能结果或投票目标。
- 不输出 JSON、Markdown、标题或分析。"""


def test_discourse_planner_preserves_baseline_contract_byte_exactly():
    observation = _observation(profile="strict_classic7")
    baseline = build_strict_classic7_speech_plan_prompt(
        observation,
        suggestible_player_ids=(2, 3, 4, 5, 6, 7),
    )
    assert hashlib.sha256(baseline.encode()).hexdigest() == (
        "f67c003ba7ff48fd71a7e299fc275767310f5c89b91d732f2263c7693ae2adba"
    )

    discourse = prompt_template_v0.build_strict_classic7_discourse_speech_plan_prompt(
        observation,
        suggestible_player_ids=(2, 3, 4, 5, 6, 7),
        public_events=_public_events(),
    )
    marker = "【计划合同】"
    baseline_clause = (
        "- 只输出 public_actions；禁止 reasoning、strategy、notes、summary "
        "或其他自由文本字段。"
    )
    discourse_clause = (
        "- 只输出 public_actions 和 public_evidence_refs；禁止 reason、"
        "rationale、analysis、thought、notes、summary、evidence_text、"
        "strategy 或其他自由文本字段。"
    )
    assert discourse[discourse.index(marker):] == baseline[
        baseline.index(marker):
    ].replace(baseline_clause, discourse_clause, 1)
    assert "speaking_strategy" not in discourse


@pytest.mark.parametrize("replacement", ["missing", "duplicate"])
def test_discourse_prompt_composition_fails_closed(monkeypatch, replacement):
    plan_marker = "【计划合同】"
    output_clause = (
        "- 只输出 public_actions；禁止 reasoning、strategy、notes、summary "
        "或其他自由文本字段。"
    )
    render_rule = "- 不得新增计划外玩家、角色判断、技能结果或投票目标。"
    if replacement == "missing":
        plan_base = "baseline without marker"
        render_base = "baseline without rule"
    else:
        plan_base = (
            f"{plan_marker} first {plan_marker} second\n{output_clause}"
        )
        render_base = f"{render_rule}\n{render_rule}"

    monkeypatch.setattr(
        prompt_template_v0,
        "build_strict_classic7_speech_plan_prompt",
        lambda *_args, **_kwargs: plan_base,
    )
    with pytest.raises(ValueError, match="exactly once"):
        prompt_template_v0.build_strict_classic7_discourse_speech_plan_prompt(
            _observation(),
            suggestible_player_ids=(2, 3, 4, 5, 6, 7),
            public_events=_public_events(),
        )

    monkeypatch.setattr(
        prompt_template_v0,
        "build_strict_classic7_speech_render_prompt",
        lambda **_kwargs: render_base,
    )
    with pytest.raises(ValueError, match="exactly once"):
        prompt_template_v0.build_strict_classic7_discourse_speech_render_prompt(
            phase_text="第1天白天公开发言",
            actor=1,
            public_actions=[{"action": "oppose", "target": 4}],
            selected_public_evidence=[_public_events()[2]],
        )


@pytest.mark.parametrize("replacement", ["missing", "duplicate"])
def test_discourse_plan_output_clause_composition_fails_closed(
    monkeypatch,
    replacement,
):
    marker = "【计划合同】"
    clause = (
        "- 只输出 public_actions；禁止 reasoning、strategy、notes、summary "
        "或其他自由文本字段。"
    )
    plan_base = marker if replacement == "missing" else marker + clause + clause
    monkeypatch.setattr(
        prompt_template_v0,
        "build_strict_classic7_speech_plan_prompt",
        lambda *_args, **_kwargs: plan_base,
    )

    with pytest.raises(ValueError, match="exactly once"):
        prompt_template_v0.build_strict_classic7_discourse_speech_plan_prompt(
            _observation(),
            suggestible_player_ids=(2, 3, 4, 5, 6, 7),
            public_events=_public_events(),
        )


def test_discourse_agent_uses_planner_then_selected_evidence_renderer_only(
    tmp_path,
):
    backend = MetadataBackend(
        [
            json.dumps(_valid_payload(), ensure_ascii=False),
            "player2此前称player3是村民，而player4称player3是狼人；两者冲突，因此我暂不信任player4。",
        ]
    )
    agent = GPTAgent(
        backend=backend,
        model_name="agent-model",
        gameplay_prompt_profile=STRICT_CLASSIC7_DISCOURSE_GAMEPLAY_PROMPT_PROFILE,
        log_file=tmp_path / "player.jsonl",
    )
    agent.rate_limit = 0

    action = agent.act(_observation())

    assert action == (
        "speech",
        "player2此前称player3是村民，而player4称player3是狼人；两者冲突，因此我暂不信任player4。",
    )
    assert len(backend.calls) == 2
    planner = backend.calls[0]["messages"][0]["content"]
    renderer = backend.calls[1]["messages"][0]["content"]
    assert "真实狼队信息" in planner
    assert "真实夜间刀人决策" in planner
    assert "player6" in planner
    assert "player7" in planner
    assert "public_evidence_refs" in planner
    assert '"event_idx":2' in renderer
    assert '"event_idx":4' in renderer
    assert '"event_idx":0' in renderer
    assert '"event_idx":5' not in renderer
    assert "player6" not in renderer
    assert "player7" not in renderer
    assert "WOLF-SECRET" not in renderer
    assert "KILL-SECRET" not in renderer
    assert "public_evidence_refs" not in renderer
    assert backend.calls[0]["response_format"]["json_schema"]["name"] == (
        "discourse_public_speech_plan"
    )
    assert [call["temperature"] for call in backend.calls] == [1.0, 0.0]
    agent.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "player.jsonl").read_text().splitlines()
    ]
    validated = next(
        record
        for record in records
        if record["message"] == "speech_discourse_plan_validated"
    )
    assert validated["public_actions"] == [
        {"action": "oppose", "target": 4}
    ]
    assert "speaking_strategy" not in json.dumps(records)
    assert validated["public_evidence_refs"] == [0, 2, 4]
    assert [
        event["event_idx"]
        for event in validated["selected_public_evidence"]
    ] == [0, 2, 4]


def test_discourse_agent_regenerates_invalid_plan_before_renderer():
    invalid_payload = {
        "public_actions": [
            {"action": "point_as_villager", "target": 4},
            {"action": "point_as_witch", "target": 4},
        ],
        "public_evidence_refs": [],
    }
    backend = MetadataBackend([
        json.dumps(invalid_payload),
        json.dumps(_valid_payload([2, 4])),
        "player4的说法让我不太信任。",
    ])
    agent = GPTAgent(
        backend=backend,
        model_name="agent-model",
        gameplay_prompt_profile=STRICT_CLASSIC7_DISCOURSE_GAMEPLAY_PROMPT_PROFILE,
    )
    agent.rate_limit = 0

    assert agent.act(_observation()) == (
        "speech",
        "player4的说法让我不太信任。",
    )
    assert len(backend.calls) == 3
    assert backend.calls[0]["messages"] is backend.calls[1]["messages"]
    assert (
        backend.calls[0]["response_format"]
        is backend.calls[1]["response_format"]
    )
    assert "response_format" not in backend.calls[2]


@pytest.mark.parametrize(
    ("refs", "speech"),
    [
        ([2, 4], "player4值得怀疑，player6也一样。"),
        ([], "player4值得怀疑，player2也一样。"),
    ],
)
def test_discourse_agent_rejects_players_outside_actions_and_evidence(
    refs,
    speech,
):
    backend = MetadataBackend(
        [
            json.dumps(_valid_payload(refs), ensure_ascii=False),
            speech,
        ]
    )
    agent = GPTAgent(
        backend=backend,
        model_name="agent-model",
        gameplay_prompt_profile=STRICT_CLASSIC7_DISCOURSE_GAMEPLAY_PROMPT_PROFILE,
    )
    agent.rate_limit = 0

    with pytest.raises(GameplaySpeechQualityError, match="unplanned player"):
        agent.act(_observation())

    assert len(backend.calls) == 2


def test_discourse_agent_does_not_require_selected_evidence_players():
    backend = MetadataBackend(
        [
            json.dumps(_valid_payload([2]), ensure_ascii=False),
            "player4现在的说法让我不太信任。",
        ]
    )
    agent = GPTAgent(
        backend=backend,
        model_name="agent-model",
        gameplay_prompt_profile=STRICT_CLASSIC7_DISCOURSE_GAMEPLAY_PROMPT_PROFILE,
    )
    agent.rate_limit = 0

    assert agent.act(_observation()) == (
        "speech",
        "player4现在的说法让我不太信任。",
    )
    assert len(backend.calls) == 2


def test_strict_classic7_baseline_prompt_and_schema_remain_ordinary():
    observation = _observation(profile="strict_classic7")
    agent = GPTAgent(
        backend=MetadataBackend([]),
        model_name="agent-model",
        gameplay_prompt_profile="strict_classic7",
    )
    prompt = agent.format_observation(
        observation,
        suggestible_player_ids=(2, 3, 4, 5, 6, 7),
    )
    assert prompt == build_strict_classic7_speech_plan_prompt(
        observation,
        suggestible_player_ids=(2, 3, 4, 5, 6, 7),
    )
    assert "public_evidence_refs" not in prompt
    assert "speaking_strategy" not in prompt


def test_runtime_injects_public_events_only_for_discourse_profile():
    class Agent:
        def __init__(self, profile):
            self.gameplay_prompt_profile = profile
            self.observations = []

        def reset(self):
            pass

        def act(self, observation):
            self.observations.append(observation)
            return ("speech", "发言")

    class Env:
        phase = "speech"
        public_events = _public_events()

        def reset(self, *, roles):
            return {"current_act_idx": 1, "phase": "1_day_speech"}

        def step(self, action):
            return {}, None, True, {"Werewolf": 1}

    discourse = Agent(STRICT_CLASSIC7_DISCOURSE_GAMEPLAY_PROMPT_PROFILE)
    run_game(Env(), [discourse], ["Werewolf"])
    assert discourse.observations[0]["canonical_public_events"] == _public_events()

    baseline = Agent("strict_classic7")
    run_game(Env(), [baseline], ["Werewolf"])
    assert "canonical_public_events" not in baseline.observations[0]
