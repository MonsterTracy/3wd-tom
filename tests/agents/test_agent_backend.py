import inspect
import json
import re
from pathlib import Path
import tempfile
import unittest

from werewolf.agents import agent_registry
from werewolf.agents.gpt_agent import GPTAgent
from werewolf.agents.llm_agent import (
    BELIEF_ROLES,
    BeliefValidationError,
    DayCognitionReportV2,
    DayCognitionReportV3,
    GameplayActionValidationError,
    GameplayGenerationExhausted,
    GameplaySpeechQualityError,
    RoleReportValidationError,
    belief_response_format,
    day_cognition_response_format_v2,
    day_cognition_response_format_v3,
    decode_evidence_selection,
    decode_public_content_selection,
    parse_belief_response,
    parse_day_cognition_response_v2,
    parse_day_cognition_response_v3,
    parse_vote_response,
    validate_gameplay_public_speech,
    validate_role_report,
    vote_response_format,
)
from werewolf.agents.prompt_template_v0 import (
    DiscussionAct,
    NO_STANCE,
    STRICT_CLASSIC7_GAME_DESCRIPTION,
    build_public_claim_catalog,
    compile_discussion_intent_v2,
    derive_belief_constraints,
    freeze_discussion_candidates,
    project_discussion_content_indices,
    project_discussion_vote_stances,
)
from werewolf.backends import BackendError
from werewolf.helper.log_utils import Log
from werewolf.models.twd_tom.samples import SpeakerPreSpeechBelief
from werewolf.registry import Registry


_TEST_REALIZATIONS = {
    "point_as_werewolf": "我认为 player{target} 是狼人。",
    "point_as_non_werewolf": "我认为 player{target} 不是狼人，属于好人阵营。",
    "point_as_villager": "我认为 player{target} 是普通村民。",
    "point_as_seer": "我认为 player{target} 是预言家。",
    "point_as_witch": "我认为 player{target} 是女巫。",
    "support": "我支持 player{target}。",
    "oppose": "我质疑 player{target}。",
    "check_as_non_werewolf": "我查验过 player{target}，结果不是狼人。",
    "check_as_werewolf": "我查验过 player{target}，结果是狼人。",
    "save": "我用解药救了 player{target}。",
    "poison": "我对 player{target} 使用了毒药。",
    "vote_intent": "这一轮我建议投票放逐 player{target}。",
    "abstain_intent": "这一轮我选择弃票。",
    "no_commitment": "这一轮我暂不作明确的身份、查验、技能或投票表态。",
}
_TEST_SELF_CLAIMS = {
    "point_as_villager": "我是普通村民。",
    "point_as_seer": "我是预言家。",
    "point_as_witch": "我是女巫。",
}


def _render_test_public_speech(speaker, acts):
    return "\n".join(
        _TEST_SELF_CLAIMS[act.action]
        if act.target == speaker and act.action in _TEST_SELF_CLAIMS
        else _TEST_REALIZATIONS[act.action].format(target=act.target)
        for act in acts
    )


class MetadataBackend:
    supports_json_schema = True

    def __init__(self, responses, metadata=None):
        self.responses = list(responses)
        self.metadata = list(
            metadata or [{"finish_reason": "stop"}] * len(self.responses)
        )
        self.calls = []

    def chat_with_metadata(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        if (
            "response_format" not in kwargs
            and prompt.startswith("你现在只负责把已经冻结的公开表达意图")
        ):
            speaker = int(re.search(r"当前发言者：player([1-7])", prompt).group(1))
            intent_block = prompt.split(
                "本轮必须完整表达、且只能表达以下冻结意图：\n", 1
            )[1].split("\n\n写作约束：", 1)[0]
            acts = []
            for line in intent_block.splitlines():
                match = re.match(
                    r"- (?P<action>[a-z_]+)(?:\(player(?P<target>[1-7])\))?:",
                    line,
                )
                if match is None:
                    continue
                target = match.group("target")
                acts.append(
                    DiscussionAct(
                        match.group("action"),
                        None if target is None else int(target),
                    )
                )
            return (
                _render_test_public_speech(speaker, tuple(acts)),
                {"finish_reason": "stop"},
            )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response, self.metadata.pop(0)

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class ScriptedRealizationBackend(MetadataBackend):
    def __init__(self, cognition_response, realization_responses):
        super().__init__([cognition_response])
        self.realization_responses = list(realization_responses)

    def chat_with_metadata(self, **kwargs):
        prompt = kwargs["messages"][0]["content"]
        if (
            "response_format" not in kwargs
            and prompt.startswith("你现在只负责把已经冻结的公开表达意图")
        ):
            self.calls.append(kwargs)
            return self.realization_responses.pop(0), {"finish_reason": "stop"}
        return super().chat_with_metadata(**kwargs)


def _belief(player_id=1, *, role="unknown"):
    return json.dumps(
        {
            "belief": "当前信息有限。",
            "concise": "继续观察。",
            "roles": {
                f"player{candidate}": role
                for candidate in range(1, 8)
                if candidate != player_id
            },
        },
        ensure_ascii=False,
    )


def _structured_speech(speaker, raw_text, *actions, kind="speech"):
    del speaker, actions
    return kind, raw_text


def _belief_for(role_options, assignments=None):
    assignments = assignments or {}
    return json.dumps(
        {
            "belief": "只推断尚未确定的玩家。",
            "concise": "保留未知。",
            "roles": {
                player: assignments.get(player, "unknown")
                for player in role_options
            },
        },
        ensure_ascii=False,
    )


def _ranked_selection(selected, candidates, *, first_field):
    if not selected:
        return {"mode": "none"}
    if len(selected) == 1:
        return {"mode": "one", first_field: selected[0]}
    first = selected[0]
    remaining = [candidate for candidate in candidates if candidate != first]
    return {
        "mode": "two",
        first_field: first,
        "second_rank": remaining.index(selected[1]),
    }


def _day_cognition_v2_from_belief(
    belief_response,
    observation,
    *,
    content_actions=(),
    vote_stance=NO_STANCE,
    evidence_claim_ids=(),
):
    payload = json.loads(belief_response)
    snapshot = freeze_discussion_candidates(observation)
    payload["public_content_action_indices"] = [
        snapshot.index(action) for action in content_actions
    ]
    payload["public_vote_stance_index"] = (
        project_discussion_vote_stances(snapshot).index(vote_stance)
    )
    payload["evidence_claim_ids"] = list(evidence_claim_ids)
    return json.dumps(payload, ensure_ascii=False)


def _day_cognition_from_belief(
    belief_response,
    observation,
    *,
    content_actions=(),
    vote_stance=NO_STANCE,
    evidence_claim_ids=(),
):
    payload = json.loads(belief_response)
    snapshot = freeze_discussion_candidates(observation)
    content_indices = project_discussion_content_indices(snapshot)
    selected_indices = tuple(snapshot.index(action) for action in content_actions)
    payload["public_content_selection"] = _ranked_selection(
        selected_indices,
        content_indices,
        first_field="first_index",
    )
    payload["public_vote_stance_index"] = (
        project_discussion_vote_stances(snapshot).index(vote_stance)
    )
    claim_ids = tuple(
        claim.claim_id for claim in build_public_claim_catalog(observation)
    )
    payload["evidence_selection"] = _ranked_selection(
        evidence_claim_ids,
        claim_ids,
        first_field="first_claim_id",
    )
    return json.dumps(payload, ensure_ascii=False)


def _day_cognition(
    observation=None,
    *,
    role="unknown",
    content_actions=(),
    vote_stance=NO_STANCE,
    evidence=(),
):
    observation = observation or _observation()
    return _day_cognition_from_belief(
        _belief(observation["current_act_idx"], role=role),
        observation,
        content_actions=content_actions,
        vote_stance=vote_stance,
        evidence_claim_ids=evidence,
    )


def _day_cognition_v2(
    observation=None,
    *,
    role="unknown",
    content_actions=(),
    vote_stance=NO_STANCE,
    evidence=(),
):
    observation = observation or _observation()
    return _day_cognition_v2_from_belief(
        _belief(observation["current_act_idx"], role=role),
        observation,
        content_actions=content_actions,
        vote_stance=vote_stance,
        evidence_claim_ids=evidence,
    )


def _observation(phase="1_day_speech"):
    public_phase = "vote_pk" if "vote_pk" in phase else (
        "vote" if "vote" in phase else "speech"
    )
    return {
        "phase": phase,
        "identity": "Villager",
        "current_act_idx": 1,
        "game_log": [
            Log(
                viewer=[1, 2, 3, 4, 5, 6, 7],
                source=2,
                target=[1, 2, 3, 4, 5, 6, 7],
                content={"speech_content": "我觉得3号可疑。", "sp_actions": []},
                day=1,
                time="第1天白天",
                event="speech",
            )
        ],
        "valid_action": (
            [("vote_pk", 0), ("vote_pk", 3), ("vote_pk", 5)]
            if "vote_pk" in phase
            else [("vote", 0), ("vote", 2), ("vote", 4)]
            if "vote" in phase
            else []
        ),
        "authoritative_public_state": {
            "day": 1,
            "day_or_night": "day",
            "phase": public_phase,
            "last_night_result": {"day": 0, "dead_players": []},
            "prior_exiles": [],
            "alive_players": [1, 2, 3, 4, 5, 6, 7],
            "suggestible_exile_targets": [2, 3, 4, 5, 6, 7],
        },
    }


def _role_observation(identity, player_id, logs=(), phase="1_day_speech"):
    observation = _observation(phase)
    observation["identity"] = identity
    observation["current_act_idx"] = player_id
    observation["game_log"] = list(logs)
    return observation


def _parse_belief(raw_response, observation):
    exact_roles, role_options = derive_belief_constraints(observation)
    return parse_belief_response(
        raw_response,
        player_id=observation["current_act_idx"],
        self_role=observation["identity"],
        phase=observation["phase"],
        exact_roles=exact_roles,
        role_options=role_options,
    )


def _parse_day_cognition(raw_response, observation):
    exact_roles, role_options = derive_belief_constraints(observation)
    candidate_snapshot = freeze_discussion_candidates(observation)
    claim_ids = tuple(
        claim.claim_id for claim in build_public_claim_catalog(observation)
    )
    return parse_day_cognition_response_v3(
        raw_response,
        player_id=observation["current_act_idx"],
        self_role=observation["identity"],
        phase=observation["phase"],
        exact_roles=exact_roles,
        role_options=role_options,
        candidate_snapshot=candidate_snapshot,
        claim_ids=claim_ids,
    )


def _parse_day_cognition_v2(raw_response, observation):
    exact_roles, role_options = derive_belief_constraints(observation)
    candidate_snapshot = freeze_discussion_candidates(observation)
    claim_ids = tuple(
        claim.claim_id for claim in build_public_claim_catalog(observation)
    )
    return parse_day_cognition_response_v2(
        raw_response,
        player_id=observation["current_act_idx"],
        self_role=observation["identity"],
        phase=observation["phase"],
        exact_roles=exact_roles,
        role_options=role_options,
        candidate_snapshot=candidate_snapshot,
        claim_ids=claim_ids,
    )


def _validate_role_report(report, observation):
    exact_roles, role_options = derive_belief_constraints(observation)
    return validate_role_report(
        report,
        player_id=observation["current_act_idx"],
        self_role=observation["identity"],
        phase=observation["phase"],
        exact_roles=exact_roles,
        role_options=role_options,
    )


class GameplayCognitionTest(unittest.TestCase):
    def _agent(self, backend, **kwargs):
        agent = GPTAgent(
            backend=backend,
            model_name="agent-model",
            gameplay_prompt_profile="strict_classic7",
            **kwargs,
        )
        agent.rate_limit = 0
        return agent

    def test_belief_schema_has_exact_minimal_fields_and_accepts_unknown(self):
        observation = _observation()
        _exact_roles, role_options = derive_belief_constraints(observation)
        response_format = belief_response_format(
            supports_json_schema=True,
            role_options=role_options,
        )
        schema = response_format["json_schema"]["schema"]

        self.assertEqual(set(schema["properties"]), {"belief", "concise", "roles"})
        self.assertEqual(schema["required"], ["belief", "concise", "roles"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            schema["properties"]["belief"],
            {"type": "string", "minLength": 1, "maxLength": 256},
        )
        self.assertEqual(
            schema["properties"]["concise"],
            {"type": "string", "minLength": 1, "maxLength": 96},
        )
        role_schema = schema["properties"]["roles"]
        self.assertNotIn("player1", role_schema["properties"])
        self.assertEqual(set(role_schema["required"]), set(role_schema["properties"]))
        for player_schema in role_schema["properties"].values():
            self.assertEqual(tuple(player_schema["enum"]), BELIEF_ROLES)
            self.assertIn("unknown", player_schema["enum"])

        report = _parse_belief(
            _belief(role="unknown"),
            observation,
        )
        self.assertEqual(set(report.roles.values()), {"unknown"})

    def test_belief_parser_enforces_unicode_character_bounds(self):
        observation = _observation()
        accepted = json.loads(_belief())
        accepted["belief"] = "信" * 256
        accepted["concise"] = "结" * 96
        report = _parse_belief(
            json.dumps(accepted, ensure_ascii=False),
            observation,
        )
        self.assertEqual(len(report.belief), 256)
        self.assertEqual(len(report.concise), 96)

        for field, value in (
            ("belief", "信" * 257),
            ("concise", "结" * 97),
        ):
            with self.subTest(field=field):
                rejected = dict(accepted)
                rejected[field] = value
                with self.assertRaises(BeliefValidationError):
                    _parse_belief(
                        json.dumps(rejected, ensure_ascii=False),
                        observation,
                    )

    def test_day_cognition_v2_schema_has_exact_transport_fields(self):
        observation = _observation()
        _exact_roles, role_options = derive_belief_constraints(observation)
        snapshot = freeze_discussion_candidates(observation)
        claim_ids = tuple(
            claim.claim_id for claim in build_public_claim_catalog(observation)
        )
        response_format = day_cognition_response_format_v2(
            supports_json_schema=True,
            role_options=role_options,
            candidate_snapshot=snapshot,
            claim_ids=claim_ids,
        )
        schema = response_format["json_schema"]["schema"]

        self.assertEqual(
            response_format["json_schema"]["name"],
            "day_cognition_report_v2",
        )
        self.assertEqual(
            set(schema["properties"]),
            {
                "belief",
                "concise",
                "roles",
                "public_content_action_indices",
                "public_vote_stance_index",
                "evidence_claim_ids",
            },
        )
        self.assertNotIn("public_action_indices", schema["properties"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            schema["properties"]["belief"],
            {"type": "string", "minLength": 1, "maxLength": 256},
        )
        self.assertEqual(
            schema["properties"]["concise"],
            {"type": "string", "minLength": 1, "maxLength": 96},
        )
        action_schema = schema["properties"]["public_content_action_indices"]
        self.assertEqual(action_schema["type"], "array")
        self.assertNotIn("uniqueItems", action_schema)
        self.assertEqual(action_schema["minItems"], 0)
        self.assertEqual(action_schema["maxItems"], 2)
        self.assertEqual(action_schema["items"]["type"], "integer")
        self.assertEqual(
            action_schema["items"]["enum"],
            list(project_discussion_content_indices(snapshot)),
        )
        stance_schema = schema["properties"]["public_vote_stance_index"]
        self.assertEqual(stance_schema["type"], "integer")
        self.assertEqual(
            stance_schema["enum"],
            list(range(len(project_discussion_vote_stances(snapshot)))),
        )
        evidence_schema = schema["properties"]["evidence_claim_ids"]
        self.assertEqual(evidence_schema["type"], "array")
        self.assertNotIn("uniqueItems", evidence_schema)
        self.assertEqual(evidence_schema["minItems"], 0)
        self.assertEqual(evidence_schema["maxItems"], min(2, len(claim_ids)))
        self.assertEqual(evidence_schema["items"]["type"], "string")
        self.assertEqual(
            evidence_schema["items"]["enum"],
            ["claim_000"],
        )

    def test_day_cognition_v3_schema_uses_ranked_selection_objects(self):
        observation = _observation()
        _exact_roles, role_options = derive_belief_constraints(observation)
        snapshot = freeze_discussion_candidates(observation)
        content_indices = project_discussion_content_indices(snapshot)
        claim_ids = ("claim_000", "claim_001", "claim_002")
        response_format = day_cognition_response_format_v3(
            supports_json_schema=True,
            role_options=role_options,
            candidate_snapshot=snapshot,
            claim_ids=claim_ids,
        )
        schema = response_format["json_schema"]["schema"]

        self.assertEqual(
            response_format["json_schema"]["name"],
            "day_cognition_report_v3",
        )
        self.assertEqual(
            set(schema["properties"]),
            {
                "belief",
                "concise",
                "roles",
                "public_content_selection",
                "public_vote_stance_index",
                "evidence_selection",
            },
        )
        self.assertNotIn("public_content_action_indices", schema["properties"])
        self.assertNotIn("evidence_claim_ids", schema["properties"])

        content_variants = {
            variant["properties"]["mode"]["enum"][0]: variant
            for variant in schema["properties"][
                "public_content_selection"
            ]["oneOf"]
        }
        self.assertEqual(set(content_variants), {"none", "one", "two"})
        self.assertEqual(content_variants["none"]["required"], ["mode"])
        self.assertEqual(
            content_variants["one"]["properties"]["first_index"]["enum"],
            list(content_indices),
        )
        self.assertEqual(
            content_variants["two"]["properties"]["second_rank"]["enum"],
            list(range(len(content_indices) - 1)),
        )

        evidence_variants = {
            variant["properties"]["mode"]["enum"][0]: variant
            for variant in schema["properties"]["evidence_selection"][
                "oneOf"
            ]
        }
        self.assertEqual(set(evidence_variants), {"none", "one", "two"})
        self.assertEqual(
            evidence_variants["one"]["properties"]["first_claim_id"][
                "enum"
            ],
            list(claim_ids),
        )
        self.assertEqual(
            evidence_variants["two"]["properties"]["second_rank"]["enum"],
            [0, 1],
        )
        stance_schema = schema["properties"]["public_vote_stance_index"]
        self.assertEqual(
            stance_schema["enum"],
            list(range(len(project_discussion_vote_stances(snapshot)))),
        )

    def test_v3_public_content_decoder_is_strict_and_duplicate_free(self):
        snapshot = freeze_discussion_candidates(_observation())
        candidates = project_discussion_content_indices(snapshot)
        first = candidates[0]
        remaining = tuple(index for index in candidates if index != first)

        self.assertEqual(
            decode_public_content_selection(
                {"mode": "none"},
                candidate_snapshot=snapshot,
            ),
            (),
        )
        self.assertEqual(
            decode_public_content_selection(
                {"mode": "one", "first_index": first},
                candidate_snapshot=snapshot,
            ),
            (first,),
        )
        self.assertEqual(
            decode_public_content_selection(
                {"mode": "two", "first_index": first, "second_rank": 3},
                candidate_snapshot=snapshot,
            ),
            (first, remaining[3]),
        )
        for first_index in candidates:
            for second_rank in range(len(candidates) - 1):
                decoded = decode_public_content_selection(
                    {
                        "mode": "two",
                        "first_index": first_index,
                        "second_rank": second_rank,
                    },
                    candidate_snapshot=snapshot,
                )
                self.assertNotEqual(decoded[0], decoded[1])

        invalid = (
            {"mode": "one"},
            {"mode": "two", "first_index": first, "second_rank": -1},
            {
                "mode": "two",
                "first_index": first,
                "second_rank": len(candidates) - 1,
            },
            {"mode": "many", "first_index": first},
        )
        for selection in invalid:
            with self.subTest(selection=selection), self.assertRaises(
                BeliefValidationError
            ):
                decode_public_content_selection(
                    selection,
                    candidate_snapshot=snapshot,
                )

    def test_v3_evidence_decoder_is_strict_and_duplicate_free(self):
        claim_ids = ("claim_a", "claim_b", "claim_c")
        self.assertEqual(
            decode_evidence_selection(
                {"mode": "none"},
                claim_ids=claim_ids,
            ),
            (),
        )
        self.assertEqual(
            decode_evidence_selection(
                {"mode": "one", "first_claim_id": "claim_b"},
                claim_ids=claim_ids,
            ),
            ("claim_b",),
        )
        self.assertEqual(
            decode_evidence_selection(
                {
                    "mode": "two",
                    "first_claim_id": "claim_b",
                    "second_rank": 1,
                },
                claim_ids=claim_ids,
            ),
            ("claim_b", "claim_c"),
        )
        for first_claim_id in claim_ids:
            for second_rank in range(len(claim_ids) - 1):
                decoded = decode_evidence_selection(
                    {
                        "mode": "two",
                        "first_claim_id": first_claim_id,
                        "second_rank": second_rank,
                    },
                    claim_ids=claim_ids,
                )
                self.assertNotEqual(decoded[0], decoded[1])

        invalid = (
            {"mode": "one"},
            {
                "mode": "two",
                "first_claim_id": "claim_a",
                "second_rank": -1,
            },
            {
                "mode": "two",
                "first_claim_id": "claim_a",
                "second_rank": 2,
            },
            {"mode": "many", "first_claim_id": "claim_a"},
        )
        for selection in invalid:
            with self.subTest(selection=selection), self.assertRaises(
                BeliefValidationError
            ):
                decode_evidence_selection(selection, claim_ids=claim_ids)

    def test_day_cognition_v3_decodes_to_existing_semantic_transport(self):
        observation = _observation()
        observation["game_log"].append(
            Log(
                viewer=[1, 2, 3, 4, 5, 6, 7],
                source=3,
                target=[1, 2, 3, 4, 5, 6, 7],
                content={"speech_content": "我支持player4。", "sp_actions": []},
                day=1,
                time="第1天白天",
                event="speech",
            )
        )
        snapshot = freeze_discussion_candidates(observation)
        actions = (
            DiscussionAct("support", 4),
            DiscussionAct("oppose", 6),
        )
        report = _parse_day_cognition(
            _day_cognition(
                observation,
                content_actions=actions,
                evidence=("claim_000", "claim_001"),
            ),
            observation,
        )

        self.assertIsInstance(report, DayCognitionReportV3)
        self.assertEqual(
            tuple(snapshot[index] for index in report.public_content_action_indices),
            actions,
        )
        self.assertEqual(
            report.evidence_claim_ids,
            ("claim_000", "claim_001"),
        )
        self.assertEqual(
            compile_discussion_intent_v2(
                snapshot,
                public_content_action_indices=(
                    report.public_content_action_indices
                ),
                public_vote_stance_index=report.public_vote_stance_index,
            ),
            actions,
        )

    def test_day_cognition_parser_enforces_unicode_character_bounds(self):
        observation = _observation()
        accepted = json.loads(_day_cognition(observation))
        accepted["belief"] = "信" * 256
        accepted["concise"] = "结" * 96
        report = _parse_day_cognition(
            json.dumps(accepted, ensure_ascii=False),
            observation,
        )
        self.assertEqual(len(report.belief), 256)
        self.assertEqual(len(report.concise), 96)

        for field, value in (
            ("belief", "信" * 257),
            ("concise", "结" * 97),
        ):
            with self.subTest(field=field):
                rejected = dict(accepted)
                rejected[field] = value
                with self.assertRaises(BeliefValidationError):
                    _parse_day_cognition(
                        json.dumps(rejected, ensure_ascii=False),
                        observation,
                    )

    def test_invalid_day_cognition_json_reports_sanitized_truncated_raw_text(self):
        observation = _observation()
        raw_response = "invalid\n" + ("x" * 1100) + "TRUNCATED-SUFFIX"

        with self.assertRaises(BeliefValidationError) as raised:
            _parse_day_cognition(raw_response, observation)

        message = str(raised.exception)
        self.assertIn("player=1, phase=1_day_speech", message)
        self.assertIn(f"raw_response={raw_response[:1000]!r}", message)
        self.assertNotIn("TRUNCATED-SUFFIX", message)
        self.assertNotIn("invalid\n", message)
        self.assertIn(r"invalid\n", message)

    def test_day_cognition_v2_compiles_all_representation_cases(self):
        observation = _observation()
        snapshot = freeze_discussion_candidates(observation)

        def compile_response(content_actions=(), vote_stance=NO_STANCE):
            report = _parse_day_cognition_v2(
                _day_cognition_v2(
                    observation,
                    content_actions=content_actions,
                    vote_stance=vote_stance,
                ),
                observation,
            )
            self.assertIsInstance(report, DayCognitionReportV2)
            return compile_discussion_intent_v2(
                snapshot,
                public_content_action_indices=(
                    report.public_content_action_indices
                ),
                public_vote_stance_index=report.public_vote_stance_index,
            )

        oppose_seven = DiscussionAct("oppose", 7)
        abstain = DiscussionAct("abstain_intent", None)
        vote_six = DiscussionAct("vote_intent", 6)
        seer_claim = DiscussionAct("point_as_seer", 1)
        check_six = DiscussionAct("check_as_werewolf", 6)

        self.assertEqual(
            compile_response(),
            (DiscussionAct("no_commitment", None),),
        )
        self.assertEqual(compile_response((oppose_seven,)), (oppose_seven,))
        self.assertEqual(compile_response(vote_stance=abstain), (abstain,))
        self.assertEqual(
            compile_response((oppose_seven,), abstain),
            (oppose_seven, abstain),
        )
        self.assertEqual(
            compile_response((seer_claim, check_six), vote_six),
            (seer_claim, check_six, vote_six),
        )
        self.assertEqual(
            compile_response((check_six, seer_claim)),
            (check_six, seer_claim),
        )
        contradictory = (
            DiscussionAct("support", 3),
            DiscussionAct("oppose", 3),
        )
        self.assertEqual(compile_response(contradictory), contradictory)

    def test_day_cognition_v2_parser_rejects_residual_transport_errors(self):
        observation = _observation()
        snapshot = freeze_discussion_candidates(observation)
        base = json.loads(_day_cognition_v2(observation))
        index = lambda action, target=None: snapshot.index(
            DiscussionAct(action, target)
        )
        invalid_payloads = {
            "duplicate indices": {
                **base,
                "public_content_action_indices": [index("support", 2)] * 2,
            },
            "out of range": {
                **base,
                "public_content_action_indices": [len(snapshot)],
            },
            "too many actions": {
                **base,
                "public_content_action_indices": [
                    index("support", 2),
                    index("oppose", 3),
                    index("point_as_seer", 1),
                ],
            },
            "vote intent as content": {
                **base,
                "public_content_action_indices": [index("vote_intent", 2)],
            },
            "abstain as content": {
                **base,
                "public_content_action_indices": [index("abstain_intent")],
            },
            "no commitment as content": {
                **base,
                "public_content_action_indices": [index("no_commitment")],
            },
            "invalid stance": {
                **base,
                "public_vote_stance_index": len(
                    project_discussion_vote_stances(snapshot)
                ),
            },
            "stance is not scalar": {
                **base,
                "public_vote_stance_index": [0],
            },
            "unknown evidence": {
                **base,
                "evidence_claim_ids": ["claim_999"],
            },
            "duplicate evidence": {
                **base,
                "evidence_claim_ids": ["claim_000", "claim_000"],
            },
            "too much evidence": {
                **base,
                "evidence_claim_ids": ["claim_000", "claim_001", "claim_002"],
            },
            "old V1 field": {
                **base,
                "public_action_indices": [index("support", 2)],
            },
        }

        for case, payload in invalid_payloads.items():
            with self.subTest(case=case), self.assertRaises(BeliefValidationError):
                _parse_day_cognition_v2(json.dumps(payload), observation)

    def test_belief_rejects_extra_cognitive_fields_and_self_role(self):
        observation = _observation()
        payload = json.loads(_belief())
        for field in ("confidence", "alignment", "probability"):
            with self.subTest(field=field):
                candidate = dict(payload)
                candidate[field] = 0.5
                with self.assertRaises(BeliefValidationError):
                    _parse_belief(
                        json.dumps(candidate),
                        observation,
                    )

        payload["roles"]["player1"] = "Villager"
        with self.assertRaisesRegex(BeliefValidationError, "unresolved players"):
            _parse_belief(
                json.dumps(payload),
                observation,
            )

    def test_self_role_consumes_fixed_inventory(self):
        for identity in ("Witch", "Seer"):
            with self.subTest(identity=identity):
                observation = _role_observation(identity, 1)
                exact_roles, role_options = derive_belief_constraints(
                    observation
                )

                self.assertEqual(exact_roles, {})
                self.assertTrue(role_options)
                self.assertTrue(
                    all(
                        identity not in options
                        for options in role_options.values()
                    )
                )

        villager = _role_observation("Villager", 1)
        _exact_roles, role_options = derive_belief_constraints(villager)
        report = _parse_belief(
            _belief_for(
                role_options,
                {
                    "player2": "Villager",
                    "player3": "Villager",
                    "player4": "Villager",
                },
            ),
            villager,
        )
        with self.assertRaisesRegex(RoleReportValidationError, "Villager"):
            _validate_role_report(report, villager)

    def test_werewolf_teammate_is_fixed_outside_inference_and_merged(self):
        observation = _role_observation(
            "Werewolf",
            2,
            [
                Log(
                    viewer=[2, 7],
                    source=0,
                    target=[2, 7],
                    content={"wolf_team": [2, 7]},
                    day=0,
                    time="第0天夜晚",
                    event="werewolf_team_info",
                ),
                Log(
                    viewer=[2, 7],
                    source=0,
                    target=6,
                    content={"kill_decision": 6},
                    day=0,
                    time="第0天夜晚",
                    event="kill_decision",
                ),
                Log(
                    viewer=list(range(1, 8)),
                    source=4,
                    target=list(range(1, 8)),
                    content={"speech_content": "PUBLIC-EVIDENCE-CANARY", "sp_actions": []},
                    day=1,
                    time="第1天白天",
                    event="speech",
                ),
            ],
        )
        exact_roles, role_options = derive_belief_constraints(observation)

        self.assertEqual(exact_roles, {"player7": "Werewolf"})
        self.assertEqual(
            set(role_options),
            {"player1", "player3", "player4", "player5", "player6"},
        )
        self.assertTrue(
            all("Werewolf" not in options for options in role_options.values())
        )
        role_schema = belief_response_format(
            supports_json_schema=True,
            role_options=role_options,
        )["json_schema"]["schema"]["properties"]["roles"]
        self.assertNotIn("player7", role_schema["properties"])

        report = _parse_belief(_belief_for(role_options), observation)
        self.assertEqual(report.roles["player7"], "Werewolf")
        self.assertEqual(
            set(report.roles),
            {f"player{player}" for player in (1, 3, 4, 5, 6, 7)},
        )

        leaked_known = json.loads(_belief_for(role_options))
        leaked_known["roles"]["player7"] = "Villager"
        with self.assertRaises(BeliefValidationError):
            _parse_belief(json.dumps(leaked_known), observation)

        backend = MetadataBackend([
            _day_cognition_from_belief(
                _belief_for(role_options),
                observation,
                evidence_claim_ids=("claim_000",),
            ),
        ])
        agent = self._agent(backend)
        self.assertEqual(
            agent.act(observation),
            _structured_speech(
                2,
                "这一轮我暂不作明确的身份、查验、技能或投票表态。",
                ("no_commitment", None),
            ),
        )
        self.assertEqual(len(backend.calls), 2)
        cognition_prompt = backend.calls[0]["messages"][0]["content"]
        private_canaries = (
            "Actual role supplied by the Environment: Werewolf",
            "真实狼队信息（仅用于内部策略）：player2, player7",
            "第0天夜晚：击杀 player6",
        )
        for canary in private_canaries:
            self.assertIn(canary, cognition_prompt)
        self.assertIn("PUBLIC-EVIDENCE-CANARY", cognition_prompt)

    def test_seer_good_excludes_werewolf_without_fixing_role(self):
        observation = _role_observation(
            "Seer",
            3,
            [
                Log(
                    viewer=[3],
                    source=3,
                    target=1,
                    content={"cheked_identity": "good"},
                    day=1,
                    time="第1天夜晚",
                    event="skill_seer",
                )
            ],
        )
        exact_roles, role_options = derive_belief_constraints(observation)

        self.assertEqual(exact_roles, {})
        self.assertEqual(
            role_options["player1"],
            ("Witch", "Villager", "unknown"),
        )
        self.assertIn("Werewolf", role_options["player2"])
        self.assertEqual(
            _parse_belief(_belief_for(role_options), observation).roles[
                "player1"
            ],
            "unknown",
        )
        self.assertEqual(
            _parse_belief(
                _belief_for(role_options, {"player1": "Witch"}),
                observation,
            ).roles["player1"],
            "Witch",
        )

        invalid_response = _belief_for(
            role_options,
            {"player1": "Werewolf"},
        )
        invalid_report = _parse_belief(invalid_response, observation)
        with self.assertRaisesRegex(
            RoleReportValidationError,
            "player1",
        ):
            _validate_role_report(invalid_report, observation)
        self.assertEqual(invalid_report.roles["player1"], "Werewolf")

        backend = MetadataBackend([
            _day_cognition_from_belief(invalid_response, observation),
        ])
        agent = self._agent(backend)
        self.assertEqual(
            agent.act(observation),
            _structured_speech(
                3,
                "这一轮我暂不作明确的身份、查验、技能或投票表态。",
                ("no_commitment", None),
            ),
        )
        self.assertEqual(len(backend.calls), 2)

    def test_seer_bad_is_fixed_outside_inference_and_merged(self):
        observation = _role_observation(
            "Seer",
            3,
            [
                Log(
                    viewer=[3],
                    source=3,
                    target=1,
                    content={"cheked_identity": "bad"},
                    day=1,
                    time="第1天夜晚",
                    event="skill_seer",
                )
            ],
        )
        exact_roles, role_options = derive_belief_constraints(observation)

        self.assertEqual(exact_roles, {"player1": "Werewolf"})
        self.assertNotIn("player1", role_options)
        report = _parse_belief(_belief_for(role_options), observation)
        self.assertEqual(report.roles["player1"], "Werewolf")
        self.assertEqual(len(report.roles), 6)

    def test_inventory_invalid_report_is_detected_without_gating_gameplay(self):
        observation = _observation()
        _exact_roles, role_options = derive_belief_constraints(observation)
        invalid_response = _belief_for(
            role_options,
            {
                "player2": "Villager",
                "player3": "Villager",
                "player4": "Villager",
            },
        )
        report = _parse_belief(invalid_response, observation)
        with self.assertRaisesRegex(RoleReportValidationError, "Villager"):
            _validate_role_report(report, observation)
        self.assertEqual(
            [report.roles[f"player{player}"] for player in (2, 3, 4)],
            ["Villager", "Villager", "Villager"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Player_1.jsonl"
            day_response = _day_cognition_from_belief(
                invalid_response,
                observation,
            )
            backend = MetadataBackend([day_response])
            agent = self._agent(backend, log_file=str(path))
            try:
                self.assertEqual(
                    agent.act(observation),
                    _structured_speech(
                        1,
                        "这一轮我暂不作明确的身份、查验、技能或投票表态。",
                        ("no_commitment", None),
                    ),
                )
            finally:
                agent.close()

            self.assertEqual(len(backend.calls), 2)
            belief_calls = [
                call
                for call in backend.calls
                if "response_format" in call
                and call["response_format"]["json_schema"]["name"]
                == "day_cognition_report_v3"
            ]
            self.assertEqual(len(belief_calls), 1)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(records[0]["response"], day_response)

    def test_subjective_concrete_role_report_remains_valid(self):
        observation = _observation()
        _exact_roles, role_options = derive_belief_constraints(observation)
        response = _belief_for(role_options, {"player2": "Seer"})
        report = _parse_belief(response, observation)

        self.assertIs(_validate_role_report(report, observation), report)
        self.assertEqual(report.roles["player2"], "Seer")

        backend = MetadataBackend([
            _day_cognition_from_belief(response, observation),
        ])
        agent = self._agent(backend)
        self.assertEqual(
            agent.act(observation),
            _structured_speech(
                1,
                "这一轮我暂不作明确的身份、查验、技能或投票表态。",
                ("no_commitment", None),
            ),
        )
        self.assertEqual(len(backend.calls), 2)

    def test_inventory_invalid_report_does_not_gate_vote(self):
        observation = _observation("1_day_vote")
        _exact_roles, role_options = derive_belief_constraints(observation)
        invalid_response = _belief_for(
            role_options,
            {
                "player2": "Villager",
                "player3": "Villager",
                "player4": "Villager",
            },
        )
        backend = MetadataBackend([invalid_response, '{"target":0}'])
        agent = self._agent(backend)

        self.assertEqual(agent.act(observation), ("vote", 0))
        self.assertEqual(len(backend.calls), 2)
        vote_prompt = backend.calls[1]["messages"][0]["content"]
        gameplay_belief = vote_prompt.split(
            "FRESH PRIVATE BELIEF\n", 1
        )[1].split("\n\nVOTE", 1)[0]
        self.assertNotIn('"roles"', gameplay_belief)

    def test_malformed_belief_response_exhausts_three_generations(self):
        malformed = (
            "not-json",
            json.dumps({"belief": "判断", "roles": {}}),
            json.dumps({"belief": "判断", "concise": "结论", "roles": []}),
        )
        for response in malformed:
            with self.subTest(response=response):
                backend = MetadataBackend([response] * 3)
                agent = self._agent(backend)

                with self.assertRaises(GameplayGenerationExhausted):
                    agent.act(_observation())
                self.assertEqual(len(backend.calls), 3)

    def test_invalid_day_intent_or_evidence_exhausts_three_generations(self):
        observation = _observation()
        snapshot = freeze_discussion_candidates(observation)
        invalid_index = json.loads(_day_cognition(observation))
        invalid_index["public_content_selection"] = {
            "mode": "one",
            "first_index": len(snapshot),
        }
        invalid_evidence = json.loads(_day_cognition(observation))
        invalid_evidence["evidence_selection"] = {
            "mode": "one",
            "first_claim_id": "claim_999",
        }
        old_v1_payload = json.loads(_day_cognition(observation))
        old_v1_payload["public_action_indices"] = [0]
        del old_v1_payload["public_content_selection"]
        del old_v1_payload["public_vote_stance_index"]

        for payload in (invalid_index, invalid_evidence, old_v1_payload):
            with self.subTest(payload=payload):
                backend = MetadataBackend([json.dumps(payload)] * 3)
                agent = self._agent(backend)
                with self.assertRaises(GameplayGenerationExhausted):
                    agent.act(observation)
                self.assertEqual(len(backend.calls), 3)

    def test_dead_claims_and_witch_kill_target_constraints(self):
        claimed = _observation()
        exact_roles, _role_options = derive_belief_constraints(claimed)
        self.assertEqual(exact_roles, {})

        dead = _observation()
        dead["authoritative_public_state"]["alive_players"] = [
            1, 2, 3, 4, 5, 7
        ]
        _exact_roles, dead_options = derive_belief_constraints(dead)
        self.assertIn("player6", dead_options)

        witch = _role_observation(
            "Witch",
            4,
            [
                Log(
                    viewer=[4],
                    source=0,
                    target=5,
                    content={"kill_decision": 5},
                    day=1,
                    time="第1天夜晚",
                    event="kill_decision",
                ),
                Log(
                    viewer=[4],
                    source=4,
                    target=6,
                    content={"poison": 6},
                    day=1,
                    time="第1天夜晚",
                    event="skill_witch",
                ),
            ],
        )
        exact_roles, witch_options = derive_belief_constraints(witch)
        self.assertEqual(exact_roles, {})
        self.assertEqual(
            witch_options["player5"],
            ("Seer", "Villager", "unknown"),
        )
        self.assertNotIn("Werewolf", witch_options["player5"])
        self.assertIn("Werewolf", witch_options["player1"])
        self.assertEqual(witch_options["player6"], witch_options["player1"])

    def test_day_path_is_structured_cognition_then_public_only_speech(self):
        observation = _observation()
        observation["game_log"].append(
            Log(
                viewer=list(range(1, 8)),
                source=7,
                target=list(range(1, 8)),
                content={
                    "speech_content": "UNSELECTED-PUBLIC-CLAIM",
                    "sp_actions": [],
                },
                day=1,
                time="第1天白天",
                event="speech",
            )
        )
        backend = MetadataBackend([
            _day_cognition(
                observation,
                content_actions=(DiscussionAct("oppose", 2),),
                evidence=("claim_000",),
            ),
        ])
        agent = self._agent(backend)

        result = agent.act(observation)
        self.assertEqual(
            result,
            _structured_speech(
                1,
                "我质疑 player2。",
                ("oppose", 2),
            ),
        )
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(
            backend.calls[0]["response_format"]["json_schema"]["name"],
            "day_cognition_report_v3",
        )
        day_schema = backend.calls[0]["response_format"]["json_schema"]["schema"]
        self.assertEqual(
            set(day_schema["properties"]),
            {
                "belief",
                "concise",
                "roles",
                "public_content_selection",
                "public_vote_stance_index",
                "evidence_selection",
            },
        )
        cognition_prompt = backend.calls[0]["messages"][0]["content"]
        realization_call = backend.calls[1]
        self.assertNotIn("response_format", realization_call)
        realization_prompt = realization_call["messages"][0]["content"]
        self.assertIn("当前天数：Day 1", realization_prompt)
        self.assertIn(
            "- oppose(player2): publicly oppose / question player2",
            realization_prompt,
        )
        self.assertNotIn("playerX", realization_prompt)
        self.assertIn("只能表达以下冻结意图", realization_prompt)
        self.assertIn("BELIEF OUTPUT", cognition_prompt)
        self.assertIn("PUBLIC CONVERSATION", cognition_prompt)
        self.assertIn("claim_000", cognition_prompt)
        self.assertIn("claim_001", cognition_prompt)
        self.assertNotIn("UNSELECTED-PUBLIC-CLAIM", result[1])
        for private_cognition in ("当前信息有限。", "继续观察。", '"roles"'):
            self.assertNotIn(private_cognition, result[1])

    def test_day_cognition_receives_exact_pre_speech_suspicion_support(self):
        observation = _observation()
        backend = MetadataBackend([_day_cognition(observation)])
        agent = self._agent(backend)
        pre_speech_belief = SpeakerPreSpeechBelief(
            observer_id="player1",
            suspected_werewolves=("player3", "player7"),
            known_werewolves=(),
            known_non_werewolves=("player1",),
            source_schema_version=(
                "classic7_pre_speech_player_suspicion_v5"
            ),
            label_prompt_version=(
                "classic7_pre_speech_player_suspicion_prompt_v5"
            ),
            label_provenance=(
                "alive_observer_readonly_pre_speech_report_v3"
            ),
            step_idx=4,
            structured_input_digest="digest-4",
        )

        agent.act_with_pre_speech_belief(
            observation,
            pre_speech_belief=pre_speech_belief,
        )

        prompt = backend.calls[0]["messages"][0]["content"]
        self.assertIn("PRE-SPEECH PRIVATE BELIEF (IMMUTABLE INPUT)", prompt)
        self.assertIn(
            '{"suspected_werewolves":["player3","player7"]}',
            prompt,
        )
        self.assertIn("communication may strategically differ", prompt)

    def test_evidence_linkage_cannot_republish_nested_raw_claims(self):
        observation = _observation()
        raw_claim = "建议投 player2。"
        nested_claim = "player2 说过“建议投 player2。”，我再次转述。"
        observation["game_log"] = [
            Log(
                viewer=list(range(1, 8)),
                source=2,
                target=list(range(1, 8)),
                content={"speech_content": raw_claim, "sp_actions": []},
                day=1,
                time="第1天白天",
                event="speech",
            ),
            Log(
                viewer=list(range(1, 8)),
                source=3,
                target=list(range(1, 8)),
                content={"speech_content": nested_claim, "sp_actions": []},
                day=1,
                time="第1天白天",
                event="speech",
            ),
        ]
        evidence_selections = (
            (),
            ("claim_000",),
            ("claim_001",),
            ("claim_000", "claim_001"),
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Player_1.jsonl"
            backend = MetadataBackend([
                _day_cognition(
                    observation,
                    content_actions=(DiscussionAct("oppose", 6),),
                    evidence=evidence,
                )
                for evidence in evidence_selections
            ])
            agent = self._agent(backend, log_file=str(path))
            try:
                speeches = tuple(
                    agent.act(observation)
                    for _evidence in evidence_selections
                )
            finally:
                agent.close()

            records = [
                json.loads(line)
                for line in path.read_text().splitlines()
            ]

        self.assertEqual(
            speeches,
            (
                _structured_speech(
                    1,
                    "我质疑 player6。",
                    ("oppose", 6),
                ),
            ) * len(evidence_selections),
        )
        self.assertEqual(len(backend.calls), 2 * len(evidence_selections))
        for call in backend.calls:
            prompt = call["messages"][0]["content"]
            self.assertIn("claim_000", prompt)
            self.assertIn("claim_001", prompt)
            self.assertIn(raw_claim, prompt)
            self.assertIn(nested_claim, prompt)
        cognition_records = [
            record for record in records if record["message"] == "belief"
        ]
        self.assertEqual(
            tuple(
                decode_evidence_selection(
                    json.loads(record["response"])["evidence_selection"],
                    claim_ids=("claim_000", "claim_001"),
                )
                for record in cognition_records
            ),
            evidence_selections,
        )
        for _action, speech in speeches:
            self.assertNotIn(raw_claim, speech)
            self.assertNotIn(nested_claim, speech)

    def test_strategic_self_role_bluff_survives_deterministic_realization(self):
        observation = _role_observation("Werewolf", 3)
        _exact_roles, role_options = derive_belief_constraints(observation)
        backend = MetadataBackend([
            _day_cognition_from_belief(
                _belief_for(role_options),
                observation,
                content_actions=(
                    DiscussionAct("point_as_seer", 3),
                    DiscussionAct("check_as_werewolf", 6),
                ),
                vote_stance=DiscussionAct("vote_intent", 6),
            ),
        ])
        agent = self._agent(backend)

        self.assertEqual(
            agent.act(observation),
            _structured_speech(
                3,
                "我是预言家。\n"
                "我查验过 player6，结果是狼人。\n"
                "这一轮我建议投票放逐 player6。",
                ("point_as_seer", 3),
                ("check_as_werewolf", 6),
                ("vote_intent", 6),
            ),
        )
        self.assertEqual(len(backend.calls), 2)
        self.assertIn(
            "Actual role supplied by the Environment: Werewolf",
            backend.calls[0]["messages"][0]["content"],
        )

    def test_content_then_abstain_stance_uses_one_cognition_call(self):
        observation = _observation()
        backend = MetadataBackend([
            _day_cognition(
                observation,
                content_actions=(DiscussionAct("oppose", 7),),
                vote_stance=DiscussionAct("abstain_intent", None),
            ),
        ])
        agent = self._agent(backend)

        self.assertEqual(
            agent.act(observation),
            _structured_speech(
                1,
                "我质疑 player7。\n这一轮我选择弃票。",
                ("oppose", 7),
                ("abstain_intent", None),
            ),
        )
        self.assertEqual(len(backend.calls), 2)

    def test_no_commitment_is_deterministic_with_one_cognition_call(self):
        backend = MetadataBackend([_day_cognition()])
        agent = self._agent(backend)

        self.assertEqual(
            agent.act(_observation()),
            _structured_speech(
                1,
                "这一轮我暂不作明确的身份、查验、技能或投票表态。",
                ("no_commitment", None),
            ),
        )
        self.assertEqual(len(backend.calls), 2)

    def test_pk_speech_uses_one_cognition_call_and_preserves_action_order(self):
        observation = _observation("2_day_speech_pk")
        observation["game_log"].append(
            Log(
                viewer=list(range(1, 8)),
                source=0,
                target=0,
                content={
                    "vote_outcome": "draw",
                    "speech_queue": [2, 3, 5],
                },
                day=2,
                time="第2天白天",
                event="end_vote",
            )
        )
        backend = MetadataBackend([
            _day_cognition(
                observation,
                content_actions=(DiscussionAct("oppose", 3),),
                vote_stance=DiscussionAct("vote_intent", 5),
            ),
        ])
        agent = self._agent(backend)

        self.assertEqual(
            agent.act(observation),
            _structured_speech(
                1,
                "我质疑 player3。\n这一轮我建议投票放逐 player5。",
                ("oppose", 3),
                ("vote_intent", 5),
                kind="speech_pk",
            ),
        )
        self.assertEqual(len(backend.calls), 2)
        snapshot = freeze_discussion_candidates(observation)
        self.assertEqual(
            project_discussion_vote_stances(snapshot),
            (
                NO_STANCE,
                DiscussionAct("vote_intent", 2),
                DiscussionAct("vote_intent", 3),
                DiscussionAct("vote_intent", 5),
                DiscussionAct("abstain_intent", None),
            ),
        )
        stance_schema = backend.calls[0]["response_format"]["json_schema"][
            "schema"
        ]["properties"]["public_vote_stance_index"]
        self.assertEqual(stance_schema["enum"], [0, 1, 2, 3, 4])

    def test_legacy_pk_speech_preserves_environment_action_kind(self):
        backend = MetadataBackend(["公开发言"])
        agent = GPTAgent(
            backend=backend,
            model_name="agent-model",
            gameplay_prompt_profile="legacy",
        )
        agent.rate_limit = 0

        self.assertEqual(
            agent.act(_observation("2_day_speech_pk")),
            ("speech_pk", "公开发言"),
        )
        self.assertEqual(len(backend.calls), 1)

    def test_day_logs_only_cognition_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Player_1.jsonl"
            backend = MetadataBackend([_day_cognition()])
            agent = self._agent(backend, log_file=str(path))
            try:
                agent.act(_observation())
            finally:
                agent.close()

            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(
                [record["message"] for record in records],
                ["belief", "speech_realization"],
            )
            self.assertEqual(
                [record["finish_reason"] for record in records],
                ["stop", "stop"],
            )

    def test_vote_generates_fresh_belief_then_one_legal_target(self):
        backend = MetadataBackend([_belief(), '{"target":4}'])
        agent = self._agent(backend)

        self.assertEqual(agent.act(_observation("1_day_vote")), ("vote", 4))
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(
            backend.calls[0]["response_format"]["json_schema"]["name"],
            "belief_report",
        )
        vote_format = backend.calls[1]["response_format"]["json_schema"]
        self.assertEqual(vote_format["name"], "vote")
        self.assertEqual(
            vote_format["schema"]["properties"]["target"]["enum"],
            [0, 2, 4],
        )
        vote_prompt = backend.calls[1]["messages"][0]["content"]
        self.assertIn("FRESH PRIVATE BELIEF", vote_prompt)
        gameplay_belief = json.loads(
            vote_prompt.split("FRESH PRIVATE BELIEF\n", 1)[1].split(
                "\n\nVOTE", 1
            )[0]
        )
        self.assertEqual(
            gameplay_belief,
            {"belief": "当前信息有限。", "concise": "继续观察。"},
        )
        self.assertNotIn("roles", gameplay_belief)
        self.assertIn("Do not preserve", vote_prompt)

    def test_vote_abstention_maps_to_exact_environment_action(self):
        for phase, expected in (
            ("1_day_vote", ("vote", 0)),
            ("1_day_vote_pk", ("vote_pk", 0)),
        ):
            with self.subTest(phase=phase):
                backend = MetadataBackend([_belief(), '{"target":0}'])
                agent = self._agent(backend)

                self.assertEqual(agent.act(_observation(phase)), expected)
                self.assertEqual(len(backend.calls), 2)

    def test_vote_schema_and_parser_use_exact_frozen_targets(self):
        legal_targets = (0, 1, 3, 4)
        schema = vote_response_format(
            supports_json_schema=True,
            legal_targets=legal_targets,
        )["json_schema"]["schema"]

        self.assertEqual(
            schema["properties"]["target"]["enum"],
            [0, 1, 3, 4],
        )
        self.assertEqual(
            parse_vote_response(
                '{"target":0}',
                legal_targets=legal_targets,
                phase="1_day_vote",
            ),
            0,
        )
        with self.assertRaises(GameplayActionValidationError):
            parse_vote_response(
                '{"target":0}',
                legal_targets=(1, 3, 4),
                phase="1_day_vote",
            )

    def test_malformed_or_illegal_vote_exhausts_three_generations(self):
        for response in ("not-json", '{"target":3}', '{"target":2,"extra":1}'):
            with self.subTest(response=response):
                backend = MetadataBackend([_belief()] + [response] * 3)
                agent = self._agent(backend)
                with self.assertRaises(GameplayGenerationExhausted):
                    agent.act(_observation("1_day_vote"))
                self.assertEqual(len(backend.calls), 4)

    def test_truncated_vote_exhausts_three_vote_generations(self):
        backend = MetadataBackend(
            [_belief()] + ['{"target":2}'] * 3,
            metadata=[{"finish_reason": "stop"}] + [
                {"finish_reason": "length"}
            ] * 3,
        )
        agent = self._agent(backend)

        with self.assertRaisesRegex(GameplayGenerationExhausted, "truncated"):
            agent.act(_observation("1_day_vote"))
        self.assertEqual(len(backend.calls), 4)

    def test_strict_night_prompt_uses_same_chinese_common_rule_contract(self):
        observation = {
            "phase": "1_night_skill_seer",
            "identity": "Seer",
            "current_act_idx": 3,
            "game_log": [],
            "valid_action": [("check", 2), ("check", 4)],
        }
        backend = MetadataBackend(['{"action_index":0}'])
        agent = self._agent(backend)

        self.assertEqual(agent.act(observation), ("check", 2))
        prompt = backend.calls[0]["messages"][0]["content"]
        self.assertIn(STRICT_CLASSIC7_GAME_DESCRIPTION, prompt)
        self.assertIn("不能主动放弃查验", prompt)
        self.assertIn("不能对自己使用解药", prompt)
        self.assertIn("所有存活玩家都参加PK投票", prompt)
        self.assertNotIn("Exactly 2 Werewolves", prompt)

    def test_invalid_night_action_exhausts_three_generations(self):
        observation = {
            "phase": "1_night_skill_seer",
            "identity": "Seer",
            "current_act_idx": 3,
            "game_log": [],
            "valid_action": [("check", 2), ("check", 4)],
        }
        for response in ('{"action_index":99}', "{'action_index':0}", "not-json"):
            with self.subTest(response=response):
                backend = MetadataBackend([response] * 3)
                agent = self._agent(backend)
                with self.assertRaisesRegex(
                    GameplayGenerationExhausted,
                    "invalid night",
                ):
                    agent.act(observation)
                self.assertEqual(len(backend.calls), 3)
                self.assertEqual(
                    backend.calls[0]["response_format"]["json_schema"]["name"],
                    "night_action_selection",
                )

    def test_night_snapshot_schema_index_maps_to_exact_environment_action(self):
        cases = (
            (
                "Werewolf",
                "1_night_skill_wolf",
                [("kill", 3), ("kill", 7)],
                1,
                ("kill", 7),
            ),
            (
                "Seer",
                "1_night_skill_seer",
                [("check", 1), ("check", 2), ("check", 4)],
                2,
                ("check", 4),
            ),
            (
                "Witch",
                "1_night_skill_witch",
                [("witch_pass", 0), ("witch_poison", 2), ("witch_heal", 5)],
                0,
                ("witch_pass", 0),
            ),
            (
                "Witch",
                "1_night_skill_witch",
                [("witch_pass", 0), ("witch_poison", 2), ("witch_heal", 5)],
                1,
                ("witch_poison", 2),
            ),
            (
                "Witch",
                "1_night_skill_witch",
                [("witch_pass", 0), ("witch_poison", 2), ("witch_heal", 5)],
                2,
                ("witch_heal", 5),
            ),
        )
        for identity, phase, valid_actions, index, expected in cases:
            with self.subTest(identity=identity, expected=expected):
                backend = MetadataBackend([json.dumps({"action_index": index})])
                agent = self._agent(backend, gameplay_max_tokens=512)
                action = agent.act({
                    "phase": phase,
                    "identity": identity,
                    "current_act_idx": 1,
                    "game_log": [],
                    "valid_action": valid_actions,
                })

                self.assertEqual(action, expected)
                self.assertEqual(len(backend.calls), 1)
                request = backend.calls[0]
                schema = request["response_format"]["json_schema"]["schema"]
                self.assertEqual(
                    schema["properties"]["action_index"]["enum"],
                    list(range(len(valid_actions))),
                )
                self.assertEqual(request["max_tokens"], 512)
                self.assertNotIn("Guard", request["messages"][0]["content"])

    def test_truncated_day_cognition_recovers_on_third_generation(self):
        backend = MetadataBackend(
            [_day_cognition()] * 3,
            [
                {"finish_reason": "length"},
                {"finish_reason": "length"},
                {"finish_reason": "stop"},
            ],
        )
        agent = self._agent(backend)

        self.assertEqual(
            agent.act(_observation()),
            _structured_speech(
                1,
                "这一轮我暂不作明确的身份、查验、技能或投票表态。",
                DiscussionAct("no_commitment", None),
            ),
        )
        self.assertEqual(len(backend.calls), 4)

    def test_agent_has_no_persistent_belief_state(self):
        agent = self._agent(MetadataBackend([]))
        self.assertFalse(any("belief" in name for name in vars(agent)))

    def test_speech_integrity_checks_are_deterministic(self):
        for content, finish_reason in (("", "stop"), ("发言", "length"), ('{"speech":"发言"}', "stop")):
            with self.subTest(content=content):
                with self.assertRaises(GameplaySpeechQualityError):
                    validate_gameplay_public_speech(
                        content,
                        finish_reason=finish_reason,
                        player_id=1,
                        phase="1_day_speech",
                    )

    def test_noncanonical_player_like_references_are_rejected(self):
        invalid_speeches = (
            "我是 player7，我直接公开站队，playerX 绝对是好人阵营的一员。",
            "我认为 player0 是狼人。",
            "我认为 player8 是狼人。",
            "我认为 player12 是狼人。",
            "我支持 player_2。",
            "我暂时观察 player。",
        )
        for content in invalid_speeches:
            with self.subTest(content=content), self.assertRaisesRegex(
                GameplaySpeechQualityError,
                "invalid player reference",
            ):
                validate_gameplay_public_speech(
                    content,
                    finish_reason="stop",
                    player_id=1,
                    phase="1_day_speech",
                )

        valid = "我支持 player1，也质疑 PLAYER7。"
        self.assertEqual(
            validate_gameplay_public_speech(
                valid,
                finish_reason="stop",
                player_id=1,
                phase="1_day_speech",
            ),
            valid,
        )

    def test_player_placeholder_retries_public_realization(self):
        observation = _observation()
        backend = ScriptedRealizationBackend(
            _day_cognition(
                observation,
                content_actions=(DiscussionAct("support", 2),),
            ),
            (
                "我支持 playerX。",
                "我支持 player2。",
            ),
        )
        agent = self._agent(backend)

        self.assertEqual(agent.act(observation), ("speech", "我支持 player2。"))
        self.assertEqual(len(backend.calls), 3)

    def test_player_placeholder_exhausts_public_realization(self):
        observation = _observation()
        backend = ScriptedRealizationBackend(
            _day_cognition(
                observation,
                content_actions=(DiscussionAct("support", 2),),
            ),
            ("我支持 playerX。",) * 3,
        )
        agent = self._agent(backend)

        with self.assertRaisesRegex(
            GameplayGenerationExhausted,
            "speech_realization",
        ):
            agent.act(observation)
        self.assertEqual(len(backend.calls), 4)

    def test_active_prompt_control_markers_cannot_be_public_speech(self):
        for marker in (
            "GAME / ROLE",
            "KNOWN INFORMATION",
            "AUTHORITATIVE INFORMATION",
            "PUBLIC CONVERSATION",
            "CURRENT PRIVATE BELIEF",
            "FRESH PRIVATE BELIEF",
            "BELIEF OUTPUT",
            "COMMON PUBLIC RULES",
            "PUBLIC AUTHORITATIVE INFORMATION",
            "DISCUSSION INTENT",
            "SELECTED PUBLIC EVIDENCE",
            "Environment authoritative public state",
            "Authoritative public history (chronological)",
            "Private facts legally visible to this player",
        ):
            with self.subTest(marker=marker), self.assertRaises(
                GameplaySpeechQualityError
            ):
                validate_gameplay_public_speech(
                    f"正常发言\n{marker}",
                    finish_reason="stop",
                    player_id=1,
                    phase="1_day_speech",
                )

    def test_gameplay_max_tokens_forward_to_belief_speech_and_night(self):
        speech_backend = MetadataBackend([_day_cognition()])
        self._agent(speech_backend, gameplay_max_tokens=512).act(_observation())
        self.assertEqual(
            [call["max_tokens"] for call in speech_backend.calls],
            [512, 512],
        )

        night_backend = MetadataBackend(['{"action_index":0}'])
        self._agent(night_backend, gameplay_max_tokens=512).act({
            "phase": "1_night_skill_seer",
            "identity": "Seer",
            "current_act_idx": 3,
            "game_log": [],
            "valid_action": [("check", 2)],
        })
        self.assertEqual(night_backend.calls[0]["max_tokens"], 512)

    def test_o1_unconfigured_limit_and_temperature_are_preserved(self):
        backend = MetadataBackend(["公开发言"])
        agent = GPTAgent(backend=backend, model_name="o1-test-model")
        agent.rate_limit = 0

        self.assertEqual(
            agent.act({
                "phase": "1_day_speech",
                "identity": "Villager",
                "current_act_idx": 1,
                "game_log": [],
                "valid_action": ("speech", -1),
            }),
            ("speech", "公开发言"),
        )
        self.assertEqual(backend.calls[0]["max_tokens"], 32000)
        self.assertIsNone(backend.calls[0]["temperature"])

    def test_invalid_gameplay_max_tokens_are_rejected(self):
        for invalid in (True, False, 0, -1, 1.5, "512"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "gameplay_max_tokens"
            ):
                GPTAgent(
                    backend=MetadataBackend([]),
                    model_name="agent-model",
                    gameplay_max_tokens=invalid,
                )

    def test_backend_model_and_schema_support_are_required(self):
        backend = MetadataBackend([_belief()])
        backend.supports_json_schema = False
        agent = self._agent(backend)
        with self.assertRaisesRegex(BackendError, "JSON Schema"):
            agent.act(_observation())
        self.assertEqual(backend.calls, [])

    def test_registry_injects_backend_and_model(self):
        backend = MetadataBackend([])
        agent_type, params = agent_registry.build(
            "gpt",
            backend=backend,
            default_model="agent-model",
            gameplay_prompt_profile="strict_classic7",
        )
        agent = agent_registry.build_agent(
            agent_type,
            player_idx=0,
            agent_param=params,
            env_param={"n_player": 7, "n_role": 4},
            log_file=None,
        )
        self.assertIs(agent.backend, backend)
        self.assertEqual(agent.model_name, "agent-model")

    def test_registry_preserves_model_override_and_llm_alias(self):
        backend = MetadataBackend([])
        _, explicit = agent_registry.build(
            "gpt",
            backend=backend,
            default_model="default",
            model_name="explicit",
        )
        _, alias = agent_registry.build(
            "gpt",
            backend=backend,
            default_model="default",
            llm="model-alias",
        )

        self.assertEqual(explicit["model_name"], "explicit")
        self.assertEqual(alias["model_name"], "model-alias")

    def test_registry_has_no_provider_or_credential_responsibility(self):
        source = inspect.getsource(Registry)
        for forbidden in ("openai.OpenAI", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
