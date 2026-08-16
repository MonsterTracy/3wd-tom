import inspect
import json
from pathlib import Path
import tempfile
import unittest

from werewolf.agents import agent_registry
from werewolf.agents.gpt_agent import GPTAgent
from werewolf.agents.twdm_agent import TWDMStrategyAgent
from werewolf.agents.llm_agent import (
    BELIEF_ROLES,
    BeliefValidationError,
    GameplayActionValidationError,
    GameplaySpeechQualityError,
    belief_response_format,
    parse_belief_response,
    parse_vote_response,
    validate_gameplay_public_speech,
    vote_response_format,
)
from werewolf.agents.prompt_template_v0 import derive_belief_constraints
from werewolf.backends import BackendError
from werewolf.helper.log_utils import Log
from werewolf.registry import Registry


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
        with self.assertRaisesRegex(BeliefValidationError, "Villager"):
            _parse_belief(
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
                )
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

    def test_inventory_violation_fails_once_without_repair(self):
        observation = _observation()
        _exact_roles, role_options = derive_belief_constraints(observation)
        backend = MetadataBackend([
            _belief_for(
                role_options,
                {
                    "player2": "Villager",
                    "player3": "Villager",
                    "player4": "Villager",
                },
            )
        ])
        agent = self._agent(backend)

        with self.assertRaisesRegex(BeliefValidationError, "Villager"):
            agent.act(observation)
        self.assertEqual(len(backend.calls), 1)

    def test_dead_claims_and_witch_target_do_not_fix_roles(self):
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
                )
            ],
        )
        exact_roles, witch_options = derive_belief_constraints(witch)
        self.assertEqual(exact_roles, {})
        self.assertEqual(witch_options["player5"], witch_options["player1"])

    def test_day_path_is_belief_then_direct_speech(self):
        backend = MetadataBackend([_belief(), "我会重点听2号和4号的后续发言。"])
        agent = self._agent(backend)

        self.assertEqual(
            agent.act(_observation()),
            ("speech", "我会重点听2号和4号的后续发言。"),
        )
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(
            backend.calls[0]["response_format"]["json_schema"]["name"],
            "belief_report",
        )
        self.assertNotIn("response_format", backend.calls[1])
        self.assertIn("CURRENT PRIVATE BELIEF", backend.calls[1]["messages"][0]["content"])
        self.assertNotIn("public_actions", backend.calls[1]["messages"][0]["content"])

    def test_day_logs_only_belief_and_speech_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Player_1.jsonl"
            backend = MetadataBackend([_belief(), "我继续观察。"])
            agent = self._agent(backend, log_file=str(path))
            try:
                agent.act(_observation())
            finally:
                agent.close()

            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([record["message"] for record in records], ["belief", "speech"])
            self.assertEqual([record["finish_reason"] for record in records], ["stop", "stop"])

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

    def test_malformed_or_illegal_vote_fails_without_retry_or_fallback(self):
        for response in ("not-json", '{"target":3}', '{"target":2,"extra":1}'):
            with self.subTest(response=response):
                backend = MetadataBackend([_belief(), response])
                agent = self._agent(backend)
                with self.assertRaises(GameplayActionValidationError):
                    agent.act(_observation("1_day_vote"))
                self.assertEqual(len(backend.calls), 2)

    def test_truncated_vote_fails_after_one_vote_generation(self):
        backend = MetadataBackend(
            [_belief(), '{"target":2}'],
            metadata=[{"finish_reason": "stop"}, {"finish_reason": "length"}],
        )
        agent = self._agent(backend)

        with self.assertRaisesRegex(GameplayActionValidationError, "truncated"):
            agent.act(_observation("1_day_vote"))
        self.assertEqual(len(backend.calls), 2)

    def test_invalid_night_action_fails_after_one_generation(self):
        observation = {
            "phase": "1_night_skill_seer",
            "identity": "Seer",
            "current_act_idx": 3,
            "game_log": [],
            "valid_action": [("check", 2), ("check", 4)],
        }
        for response in ('{"action_index":99}', "{'action_index':0}", "not-json"):
            with self.subTest(response=response):
                backend = MetadataBackend([response])
                agent = self._agent(backend)
                with self.assertRaisesRegex(GameplayActionValidationError, "invalid night"):
                    agent.act(observation)
                self.assertEqual(len(backend.calls), 1)
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

    def test_truncated_speech_and_belief_fail_without_regeneration(self):
        cases = (
            ([_belief()], [{"finish_reason": "length"}], BeliefValidationError, 1),
            (
                [_belief(), "未完成发言"],
                [{"finish_reason": "stop"}, {"finish_reason": "length"}],
                GameplaySpeechQualityError,
                2,
            ),
        )
        for responses, metadata, error, calls in cases:
            with self.subTest(error=error.__name__):
                backend = MetadataBackend(responses, metadata)
                agent = self._agent(backend)
                with self.assertRaises(error):
                    agent.act(_observation())
                self.assertEqual(len(backend.calls), calls)

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

    def test_active_prompt_control_markers_cannot_be_public_speech(self):
        for marker in (
            "GAME / ROLE",
            "KNOWN INFORMATION",
            "AUTHORITATIVE INFORMATION",
            "PUBLIC CONVERSATION",
            "CURRENT PRIVATE BELIEF",
            "FRESH PRIVATE BELIEF",
            "BELIEF OUTPUT",
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
        speech_backend = MetadataBackend([_belief(), "我继续观察。"])
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

    def test_twdm_generation_preserves_backend_model_and_token_forwarding(self):
        backend = MetadataBackend(["  structured response  "])
        agent = TWDMStrategyAgent(
            backend=backend,
            model_name="twdm-model",
            temperature=0.1,
            gameplay_max_tokens=512,
        )

        response = agent._TWDMStrategyAgent__api_generate(
            [{"role": "user", "content": " prompt "}]
        )

        self.assertEqual(response, "structured response")
        self.assertEqual(backend.calls[0]["model"], "twdm-model")
        self.assertEqual(backend.calls[0]["temperature"], 0.1)
        self.assertEqual(backend.calls[0]["max_tokens"], 512)
        self.assertEqual(
            backend.calls[0]["messages"],
            [{"role": "user", "content": "prompt"}],
        )

    def test_registry_has_no_provider_or_credential_responsibility(self):
        source = inspect.getsource(Registry)
        for forbidden in ("openai.OpenAI", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
