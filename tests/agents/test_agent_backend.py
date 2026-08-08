import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from run_random import eval as run_game
from werewolf.agents import agent_registry
from werewolf.agents.gpt_agent import GPTAgent
from werewolf.agents.llm_agent import (
    GameplaySpeechQualityError,
    PublicSpeechPlanValidationError,
    canonical_suggestible_player_ids,
    public_speech_plan_json_schema,
    validate_gameplay_public_speech,
    validate_public_speech_plan,
)
from werewolf.agents.twdm_agent import TWDMStrategyAgent
from werewolf.backends import BackendError
from werewolf.models.twd_tom.schema import ACTION_NAMES
from werewolf.registry import Registry


def _schema_accepts_plan(schema, payload):
    if not isinstance(payload, dict) or set(payload) != {"public_actions"}:
        return False
    actions = payload["public_actions"]
    if not isinstance(actions, list):
        return False
    branches = schema["properties"]["public_actions"]["items"]["oneOf"]
    for item in actions:
        if not isinstance(item, dict) or set(item) != {"action", "target"}:
            return False
        matches = 0
        for branch in branches:
            properties = branch["properties"]
            action_rule = properties["action"]
            action_matches = (
                item["action"] == action_rule["const"]
                if "const" in action_rule
                else item["action"] in action_rule["enum"]
            )
            target_matches = (
                isinstance(item["target"], int)
                and not isinstance(item["target"], bool)
                and item["target"] in properties["target"]["enum"]
            )
            matches += action_matches and target_matches
        if matches != 1:
            return False
    return True


class RecordingBackend:
    def __init__(self, responses=None):
        self.responses = list(responses or ["response"])
        self.calls = []

    def chat(
        self,
        messages,
        model=None,
        temperature=0.7,
        max_tokens=None,
        response_format=None,
        **kwargs,
    ):
        self._record(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            **kwargs,
        )
        return self.responses.pop(0)

    def _record(
        self,
        messages,
        model=None,
        temperature=0.7,
        max_tokens=None,
        response_format=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": response_format,
                **kwargs,
            }
        )


class MetadataBackend(RecordingBackend):
    def __init__(self, responses=None, metadata=None):
        super().__init__(responses)
        self.metadata = list(
            metadata
            or [{"finish_reason": "stop"}] * len(self.responses)
        )

    def chat_with_metadata(self, **kwargs):
        self._record(**kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response, self.metadata.pop(0)


class AgentBackendTest(unittest.TestCase):
    @staticmethod
    def _strict_observation():
        return {
            "phase": "1_day_speech",
            "identity": "Villager",
            "current_act_idx": 1,
            "game_log": [],
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

    @staticmethod
    def _dead_player_observation():
        observation = AgentBackendTest._strict_observation()
        observation["current_act_idx"] = 2
        observation["authoritative_public_state"].update({
            "last_night_result": {"day": 0, "dead_players": [3]},
            "alive_players": [1, 2, 4, 5, 6, 7],
            "suggestible_exile_targets": [1, 4, 5, 6, 7],
        })
        return observation

    @staticmethod
    def _player_records(path):
        return [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line
        ]

    def test_strict_player_log_records_each_returned_call_before_validation(self):
        cases = (
            (
                ['{"public_actions":[{"action":"vote_intent","target":1}]}'],
                None,
                PublicSpeechPlanValidationError,
                1,
            ),
            (
                ['{"public_actions":[{"action":"oppose","target":3}]}', "我反对4号。"],
                None,
                GameplaySpeechQualityError,
                2,
            ),
            (
                ['{"public_actions":[]}', "我继续观察。"],
                [{"finish_reason": "stop"}, {"finish_reason": "length"}],
                GameplaySpeechQualityError,
                2,
            ),
            (
                ['{"public_actions":[{"action":"oppose","target":3}]}', "我反对3号。"],
                None,
                None,
                2,
            ),
        )
        for responses, metadata, error_type, expected_count in cases:
            with self.subTest(responses=responses), tempfile.TemporaryDirectory() as tmp:
                log_path = Path(tmp) / "Player_1.jsonl"
                backend = MetadataBackend(responses, metadata=metadata)
                backend.supports_json_schema = True
                backend.session = SimpleNamespace(game_id="game_001_seed_1")
                agent = GPTAgent(
                    backend=backend,
                    model_name="agent-model",
                    log_file=str(log_path),
                    gameplay_prompt_profile="strict_classic7",
                )
                agent.backend_id = "fake-backend"
                agent.rate_limit = 0
                try:
                    if error_type is None:
                        self.assertEqual(
                            agent.act(self._strict_observation()),
                            ("speech", "我反对3号。"),
                        )
                    else:
                        with self.assertRaises(error_type):
                            agent.act(self._strict_observation())
                finally:
                    agent.close()

                records = self._player_records(log_path)
                self.assertEqual(len(backend.calls), expected_count)
                self.assertEqual(len(records), expected_count)
                self.assertEqual(
                    [call["temperature"] for call in backend.calls],
                    [1.0, 0.0][:expected_count],
                )
                self.assertEqual(
                    [record["response"] for record in records],
                    responses[:expected_count],
                )
                self.assertEqual(
                    [record["message"] for record in records],
                    ["speech_plan", "speech_render"][:expected_count],
                )
                expected_metadata = metadata or [
                    {"finish_reason": "stop"}
                ] * expected_count
                self.assertEqual(
                    [record["finish_reason"] for record in records],
                    [
                        item["finish_reason"]
                        for item in expected_metadata[:expected_count]
                    ],
                )
                self.assertEqual(records[0]["response_format"]["type"], "json_schema")
                self.assertEqual(records[0]["messages"], backend.calls[0]["messages"])
                self.assertEqual(records[0]["model"], "agent-model")
                self.assertEqual(records[0]["backend_id"], "fake-backend")
                self.assertEqual(records[0]["game_id"], "game_001_seed_1")
                if expected_count == 2:
                    self.assertIsNone(records[1]["response_format"])

    def test_legacy_speech_still_writes_one_success_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "Player_1.jsonl"
            agent = GPTAgent(
                backend=MetadataBackend(["这是发言"]),
                model_name="agent-model",
                log_file=str(log_path),
            )
            agent.rate_limit = 0
            try:
                self.assertEqual(
                    agent.act({
                        "phase": "1_day_speech",
                        "identity": "Villager",
                        "current_act_idx": 1,
                        "game_log": [],
                        "valid_action": ("speech", -1),
                    }),
                    ("speech", "这是发言"),
                )
            finally:
                agent.close()

            records = self._player_records(log_path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["response"], "这是发言")
            self.assertEqual(records[0]["message"], "1_day_speech")

    def test_renderer_backend_error_keeps_real_failed_call_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "Player_1.jsonl"
            backend = MetadataBackend([
                '{"public_actions":[]}',
                BackendError("renderer failed"),
            ])
            backend.supports_json_schema = True
            agent = GPTAgent(
                backend=backend,
                model_name="agent-model",
                log_file=str(log_path),
                gameplay_prompt_profile="strict_classic7",
            )
            agent.rate_limit = 0
            try:
                with self.assertRaisesRegex(BackendError, "renderer failed"):
                    agent.act(self._strict_observation())
            finally:
                agent.close()

            records = self._player_records(log_path)
            self.assertEqual(len(backend.calls), 2)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["dispatch_status"], "ok")
            self.assertEqual(records[1]["message"], "speech_render")
            self.assertEqual(records[1]["dispatch_status"], "error")
            self.assertEqual(records[1]["error_type"], "BackendError")
            self.assertEqual(records[1]["error_message"], "renderer failed")
            self.assertIsNone(records[1]["response"])

    def test_strict_validation_failure_prevents_environment_step(self):
        class Env:
            phase = "speech"
            public_events = []

            def __init__(self, observation):
                self.observation = observation
                self.step_calls = 0

            def reset(self, *, roles):
                return self.observation

            def step(self, action):
                self.step_calls += 1
                raise AssertionError("validation failure reached env.step")

        for responses, error_type in (
            (
                ['{"public_actions":[{"action":"vote_intent","target":1}]}'],
                PublicSpeechPlanValidationError,
            ),
            (
                ['{"public_actions":[{"action":"oppose","target":3}]}', "我反对4号。"],
                GameplaySpeechQualityError,
            ),
        ):
            with self.subTest(responses=responses):
                backend = MetadataBackend(responses)
                backend.supports_json_schema = True
                agent = GPTAgent(
                    backend=backend,
                    model_name="agent-model",
                    gameplay_prompt_profile="strict_classic7",
                )
                agent.rate_limit = 0
                env = Env(self._strict_observation())

                with self.assertRaises(error_type):
                    run_game(env, [agent], ["Villager"])
                self.assertEqual(env.step_calls, 0)

    def test_strict_speech_uses_plan_then_isolated_renderer(self):
        backend = MetadataBackend([
            '{"public_actions":[{"action":"oppose","target":3}]}',
            "我不认可3号的说法。",
        ])
        backend.supports_json_schema = True
        agent = GPTAgent(
            backend=backend,
            model_name="agent-model",
            gameplay_prompt_profile="strict_classic7",
            gameplay_max_tokens=512,
        )
        agent.rate_limit = 0

        self.assertEqual(
            agent.act(self._strict_observation()),
            ("speech", "我不认可3号的说法。"),
        )
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(backend.calls[0]["response_format"]["type"], "json_schema")
        self.assertIsNone(backend.calls[1]["response_format"])
        self.assertEqual(
            [call["temperature"] for call in backend.calls],
            [1.0, 0.0],
        )
        self.assertEqual([call["max_tokens"] for call in backend.calls], [512, 512])
        renderer = backend.calls[1]["messages"][0]["content"]
        self.assertIn("oppose(player3)", renderer)
        self.assertNotIn("【你合法知道的私有信息】", renderer)
        self.assertNotIn("【权威公共状态】", renderer)
        self.assertNotIn("【当前存活】", renderer)

    def test_strict_renderer_temperature_is_independent_of_gameplay_temperature(self):
        backend = MetadataBackend([
            '{"public_actions":[{"action":"oppose","target":3}]}',
            "我不认可3号的说法。",
        ])
        backend.supports_json_schema = True
        agent = GPTAgent(
            backend=backend,
            model_name="agent-model",
            temperature=0.7,
            gameplay_prompt_profile="strict_classic7",
        )
        agent.rate_limit = 0

        self.assertEqual(
            agent.act(self._strict_observation()),
            ("speech", "我不认可3号的说法。"),
        )
        self.assertEqual(
            [call["temperature"] for call in backend.calls],
            [0.7, 0.0],
        )

    def test_renderer_does_not_receive_r9_state_players_outside_plan(self):
        observation = self._strict_observation()
        observation["current_act_idx"] = 2
        observation["authoritative_public_state"].update({
            "last_night_result": {"day": 0, "dead_players": [1, 3]},
            "alive_players": [2, 4, 5, 6, 7],
            "suggestible_exile_targets": [4, 5, 6, 7],
        })
        backend = MetadataBackend([
            '{"public_actions":[{"action":"point_as_villager","target":4}]}',
            "我目前更倾向认为player4是村民。",
        ])
        backend.supports_json_schema = True
        agent = GPTAgent(
            backend=backend,
            model_name="agent-model",
            gameplay_prompt_profile="strict_classic7",
        )
        agent.rate_limit = 0

        self.assertEqual(
            agent.act(observation),
            ("speech", "我目前更倾向认为player4是村民。"),
        )
        renderer = backend.calls[1]["messages"][0]["content"]
        self.assertIn("player2", renderer)
        self.assertIn("player4", renderer)
        self.assertIn("第1天白天公开发言", renderer)
        for player_id in (1, 3, 5, 6, 7):
            self.assertNotIn(f"player{player_id}", renderer)
        for state_label in (
            "昨夜公开结果",
            "此前放逐",
            "当前存活",
            "当前可公开建议放逐",
        ):
            self.assertNotIn(state_label, renderer)

    def test_public_speech_plan_validation_contract(self):
        state = self._strict_observation()["authoritative_public_state"]
        valid_cases = (
            [],
            [
                {"action": "point_as_seer", "target": 1},
                {"action": "check_as_werewolf", "target": 3},
                {"action": "vote_intent", "target": 3},
            ],
            [
                {"action": "point_as_werewolf", "target": 3},
                {"action": "oppose", "target": 3},
                {"action": "vote_intent", "target": 3},
            ],
            [
                {"action": "check_as_werewolf", "target": 1},
                {"action": "point_as_werewolf", "target": 1},
            ],
            [
                {"action": "check_as_good", "target": 1},
                {"action": "point_as_villager", "target": 1},
            ],
            [
                {"action": "point_as_villager", "target": 1},
                {"action": "check_as_good", "target": 1},
                {"action": "vote_intent", "target": 7},
            ],
        )
        for actions in valid_cases:
            with self.subTest(valid=actions):
                plan = validate_public_speech_plan(
                    {"public_actions": actions},
                    suggestible_player_ids=tuple(
                        state["suggestible_exile_targets"]
                    ),
                    player_id=1,
                    speaker_role="Villager",
                    phase="1_day_speech",
                )
                self.assertEqual(plan.as_list(), actions)

        invalid_cases = (
            {"public_actions": [], "notes": "free text"},
            {"public_actions": [{"action": "invented", "target": 3}]},
            {"public_actions": [{"action": "oppose", "target": 3, "why": "x"}]},
            {"public_actions": [{"action": "vote_intent", "target": 1}]},
            {"public_actions": [
                {"action": "oppose", "target": 3},
                {"action": "oppose", "target": 3},
            ]},
        )
        for payload in invalid_cases:
            with self.subTest(invalid=payload), self.assertRaises(
                PublicSpeechPlanValidationError
            ):
                validate_public_speech_plan(
                    payload,
                    suggestible_player_ids=tuple(
                        state["suggestible_exile_targets"]
                    ),
                    player_id=1,
                    speaker_role="Villager",
                    phase="1_day_speech",
                )

    def test_public_speech_plan_rejects_oppose_self(self):
        candidates = (1, 2, 3, 4, 5, 6)
        with self.assertRaises(PublicSpeechPlanValidationError):
            validate_public_speech_plan(
                {"public_actions": [
                    {"action": "oppose", "target": 7},
                ]},
                suggestible_player_ids=candidates,
                player_id=7,
                speaker_role="Villager",
                phase="1_day_speech",
            )

    def test_public_speech_plan_keeps_soft_consistency_combinations(self):
        candidates = (1, 2, 3, 4, 5, 6)
        good_judgment_actions = (
            "point_as_villager",
            "point_as_seer",
            "point_as_witch",
            "point_as_guard",
            "check_as_good",
        )
        allowed_plans = tuple(
            [
                {"action": action, "target": 3},
                {"action": "vote_intent", "target": 3},
            ]
            for action in good_judgment_actions
        ) + (
            [
                {"action": "point_as_werewolf", "target": 3},
                {"action": "vote_intent", "target": 3},
            ],
            [
                {"action": "check_as_werewolf", "target": 3},
                {"action": "vote_intent", "target": 3},
            ],
            [
                {"action": "oppose", "target": 3},
                {"action": "vote_intent", "target": 3},
            ],
            [
                {"action": "support", "target": 3},
                {"action": "vote_intent", "target": 3},
            ],
            [
                {"action": "point_as_villager", "target": 3},
                {"action": "vote_intent", "target": 4},
            ],
            [{"action": "oppose", "target": 3}],
            [{"action": "support", "target": 7}],
        )
        for actions in allowed_plans:
            with self.subTest(actions=actions):
                plan = validate_public_speech_plan(
                    {"public_actions": actions},
                    suggestible_player_ids=candidates,
                    player_id=7,
                    speaker_role="Villager",
                    phase="1_day_speech",
                )
                self.assertEqual(plan.as_list(), actions)

    def test_dynamic_plan_schema_separates_vote_target_domain(self):
        candidates = (1, 2, 4, 5)
        schema = public_speech_plan_json_schema(
            suggestible_player_ids=candidates,
            speaker_id=7,
            speaker_role="Villager",
        )
        branches = schema["properties"]["public_actions"]["items"]["oneOf"]
        vote_branch, oppose_branch, self_wolf_claim_branch, other_branch = branches

        self.assertEqual(vote_branch["properties"]["action"], {"const": "vote_intent"})
        self.assertEqual(vote_branch["properties"]["target"]["enum"], list(candidates))
        self.assertEqual(oppose_branch["properties"]["action"], {"const": "oppose"})
        self.assertEqual(
            oppose_branch["properties"]["target"]["enum"],
            [1, 2, 3, 4, 5, 6],
        )
        self.assertEqual(
            set(self_wolf_claim_branch["properties"]["action"]["enum"]),
            {"point_as_werewolf", "check_as_werewolf"},
        )
        self.assertEqual(
            self_wolf_claim_branch["properties"]["target"]["enum"],
            list(range(1, 8)),
        )
        self.assertEqual(
            set(other_branch["properties"]["action"]["enum"]),
            set(ACTION_NAMES) - {
                "vote_intent",
                "oppose",
                "point_as_werewolf",
                "check_as_werewolf",
            },
        )
        self.assertEqual(other_branch["properties"]["target"]["enum"], list(range(1, 8)))
        self.assertFalse(_schema_accepts_plan(schema, {
            "public_actions": [{"action": "vote_intent", "target": 3}]
        }))
        for action, target in (
            ("vote_intent", 1),
            ("oppose", 1),
            ("oppose", 6),
            ("check_as_good", 3),
            ("support", 7),
            ("point_as_villager", 7),
            ("point_as_seer", 7),
        ):
            with self.subTest(action=action, target=target):
                self.assertTrue(_schema_accepts_plan(schema, {
                    "public_actions": [{"action": action, "target": target}]
                }))
        self.assertFalse(_schema_accepts_plan(schema, {
            "public_actions": [{"action": "oppose", "target": 7}]
        }))
        branch_actions = [
            {branch["properties"]["action"]["const"]}
            if "const" in branch["properties"]["action"]
            else set(branch["properties"]["action"]["enum"])
            for branch in branches
        ]
        self.assertEqual(set().union(*branch_actions), set(ACTION_NAMES))
        self.assertEqual(
            sum(len(actions) for actions in branch_actions),
            len(ACTION_NAMES),
        )

    def test_dynamic_plan_schema_omits_vote_branch_for_empty_candidates(self):
        schema = public_speech_plan_json_schema(
            suggestible_player_ids=(),
            speaker_id=7,
            speaker_role="Villager",
        )
        branches = schema["properties"]["public_actions"]["items"]["oneOf"]

        self.assertEqual(len(branches), 3)
        self.assertEqual(branches[0]["properties"]["action"], {"const": "oppose"})
        self.assertNotIn("vote_intent", branches[2]["properties"]["action"]["enum"])
        self.assertTrue(_schema_accepts_plan(schema, {"public_actions": []}))
        self.assertTrue(_schema_accepts_plan(schema, {
            "public_actions": [{"action": "check_as_good", "target": 3}]
        }))
        self.assertFalse(_schema_accepts_plan(schema, {
            "public_actions": [{"action": "vote_intent", "target": 1}]
        }))

    def test_true_werewolf_self_identity_disclosure_is_not_representable(self):
        observation = self._strict_observation()
        observation["identity"] = "Werewolf"
        observation["current_act_idx"] = 6
        observation["authoritative_public_state"][
            "suggestible_exile_targets"
        ] = [1, 2, 3, 4, 5, 7]
        candidates = canonical_suggestible_player_ids(
            observation["authoritative_public_state"]
        )
        backend = MetadataBackend([
            '{"public_actions":['
            '{"action":"point_as_werewolf","target":6}]}'
        ])
        backend.supports_json_schema = True
        backend.session = SimpleNamespace(game_id="game_010_seed_464")
        agent = GPTAgent(
            backend=backend,
            model_name="agent-model",
            gameplay_prompt_profile="strict_classic7",
        )
        agent.rate_limit = 0
        with self.assertRaises(PublicSpeechPlanValidationError):
            agent.act(observation)
        self.assertEqual(len(backend.calls), 1)
        schema = backend.calls[0]["response_format"]["json_schema"]["schema"]

        for action in ("point_as_werewolf", "check_as_werewolf"):
            self.assertFalse(_schema_accepts_plan(schema, {
                "public_actions": [{"action": action, "target": 6}],
            }))
            with self.assertRaises(PublicSpeechPlanValidationError):
                validate_public_speech_plan(
                    {"public_actions": [{"action": action, "target": 6}]},
                    suggestible_player_ids=candidates,
                    player_id=6,
                    speaker_role="Werewolf",
                    phase="1_day_speech",
                    game_context="game_010_seed_464",
                )
            for target in candidates:
                with self.subTest(action=action, target=target):
                    self.assertTrue(_schema_accepts_plan(schema, {
                        "public_actions": [{"action": action, "target": target}],
                    }))

        for role in ("Villager", "Seer", "Witch"):
            role_schema = public_speech_plan_json_schema(
                suggestible_player_ids=candidates,
                speaker_id=6,
                speaker_role=role,
            )
            for action in ("point_as_werewolf", "check_as_werewolf"):
                with self.subTest(role=role, action=action):
                    payload = {
                        "public_actions": [{"action": action, "target": 6}],
                    }
                    self.assertTrue(_schema_accepts_plan(role_schema, payload))
                    self.assertEqual(
                        validate_public_speech_plan(
                            payload,
                            suggestible_player_ids=candidates,
                            player_id=6,
                            speaker_role=role,
                            phase="1_day_speech",
                        ).as_list(),
                        payload["public_actions"],
                    )

    def test_strict_flow_shares_candidates_across_prompt_schema_and_validator(self):
        observation = self._dead_player_observation()
        candidates = canonical_suggestible_player_ids(
            observation["authoritative_public_state"]
        )
        backend = MetadataBackend([
            '{"public_actions":[{"action":"check_as_good","target":3},'
            '{"action":"vote_intent","target":4}]}',
            "我查验玩家3为好人，这一轮我会投玩家4。",
        ])
        backend.supports_json_schema = True
        agent = GPTAgent(
            backend=backend,
            model_name="agent-model",
            gameplay_prompt_profile="strict_classic7",
        )
        agent.rate_limit = 0

        self.assertEqual(
            agent.act(observation),
            ("speech", "我查验玩家3为好人，这一轮我会投玩家4。"),
        )
        planner_call = backend.calls[0]
        self.assertIn(
            "【当前可公开建议放逐】player1, player4, player5, player6, player7",
            planner_call["messages"][0]["content"],
        )
        schema = planner_call["response_format"]["json_schema"]["schema"]
        vote_branch = schema["properties"]["public_actions"]["items"]["oneOf"][0]
        self.assertEqual(vote_branch["properties"]["target"]["enum"], list(candidates))
        self.assertEqual(len(backend.calls), 2)
        renderer_prompt = backend.calls[1]["messages"][0]["content"]
        self.assertIn("player2", renderer_prompt)
        self.assertIn("player3", renderer_prompt)
        self.assertIn("player4", renderer_prompt)
        for player_id in (1, 5, 6, 7):
            self.assertNotIn(f"player{player_id}", renderer_prompt)

    def test_dead_vote_target_fails_schema_and_post_validator(self):
        observation = self._dead_player_observation()
        candidates = canonical_suggestible_player_ids(
            observation["authoritative_public_state"]
        )
        payload = {"public_actions": [
            {"action": "check_as_good", "target": 3},
            {"action": "vote_intent", "target": 3},
        ]}
        schema = public_speech_plan_json_schema(
            suggestible_player_ids=candidates,
            speaker_id=2,
            speaker_role="Villager",
        )
        self.assertFalse(_schema_accepts_plan(schema, payload))
        with self.assertRaisesRegex(
            PublicSpeechPlanValidationError,
            "not currently suggestible",
        ):
            validate_public_speech_plan(
                payload,
                suggestible_player_ids=candidates,
                player_id=2,
                speaker_role="Villager",
                phase="1_day_speech",
            )

        backend = MetadataBackend([json.dumps(payload)])
        backend.supports_json_schema = True
        agent = GPTAgent(
            backend=backend,
            model_name="agent-model",
            gameplay_prompt_profile="strict_classic7",
        )
        agent.rate_limit = 0
        with self.assertRaises(PublicSpeechPlanValidationError):
            agent.act(observation)
        self.assertEqual(len(backend.calls), 1)

    def test_final_speech_must_realize_exact_plan_player_scope(self):
        self.assertEqual(
            validate_gameplay_public_speech(
                "我反对3号。",
                finish_reason="stop",
                player_id=1,
                phase="1_day_speech",
                planned_player_ids={3},
            ),
            "我反对3号。",
        )
        for speech, planned in (("我反对4号。", {3}), ("我继续听。", {3}), ("我观察2号。", set())):
            with self.subTest(speech=speech), self.assertRaises(
                GameplaySpeechQualityError
            ):
                validate_gameplay_public_speech(
                    speech,
                    finish_reason="stop",
                    player_id=1,
                    phase="1_day_speech",
                    planned_player_ids=planned,
                )

    def test_player_prefix_references_satisfy_real_plan_coverage(self):
        speech = "我认为玩家3是值得信赖的村民，而玩家4的发言存在可疑之处。"
        self.assertEqual(
            validate_gameplay_public_speech(
                speech,
                finish_reason="stop",
                player_id=2,
                phase="1_day_speech",
                planned_player_ids={3, 4},
            ),
            speech,
        )

    def test_explicit_player_reference_formats_are_supported(self):
        for reference in (
            "player3", "player 3", "Player3", "玩家3", "玩家 3",
            "3号", "3号玩家", "3号位", "玩家三", "三号", "三号玩家", "三号位",
        ):
            with self.subTest(reference=reference):
                self.assertEqual(
                    validate_gameplay_public_speech(
                        f"我关注{reference}。",
                        finish_reason="stop",
                        player_id=1,
                        phase="1_day_speech",
                        planned_player_ids={3},
                    ),
                    f"我关注{reference}。",
                )

    def test_non_player_numbers_are_not_player_references(self):
        for text in (
            "第3天", "第三天", "第3点", "3票", "3人死亡",
            "第3轮", "第3项计划", "计划中的第3项", "3个目标",
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    validate_gameplay_public_speech(
                        text,
                        finish_reason="stop",
                        player_id=1,
                        phase="1_day_speech",
                        planned_player_ids=set(),
                    ),
                    text,
                )

    def test_invalid_explicit_player_references_are_rejected(self):
        for reference in (
            "player0", "player8", "玩家0", "玩家8",
            "0号玩家", "8号位", "玩家八",
        ):
            with self.subTest(reference=reference), self.assertRaisesRegex(
                GameplaySpeechQualityError,
                "invalid player reference",
            ):
                validate_gameplay_public_speech(
                    f"我关注{reference}。",
                    finish_reason="stop",
                    player_id=1,
                    phase="1_day_speech",
                )

    def test_strict_speech_stops_at_failed_stage_without_retry_or_fallback(self):
        cases = (
            ([BackendError("planner failed")], None, 1),
            (["not json"], None, 1),
            (['{"public_actions":[]}'], [{"finish_reason": "length"}], 1),
            (['{"public_actions":[{"action":"vote_intent","target":1}]}'], None, 1),
            (['{"public_actions":[]}'], [None], 1),
            (['{"public_actions":[]}', BackendError("renderer failed")], None, 2),
            (['{"public_actions":[]}', "观望"], [{"finish_reason": "stop"}, None], 2),
            (['{"public_actions":[]}', "观望"], [{"finish_reason": "stop"}, {"finish_reason": "length"}], 2),
            (['{"public_actions":[]}', "提到2号"], None, 2),
        )
        for responses, metadata, call_count in cases:
            with self.subTest(responses=responses):
                backend = MetadataBackend(responses, metadata=metadata)
                backend.supports_json_schema = True
                agent = GPTAgent(
                    backend=backend,
                    model_name="agent-model",
                    gameplay_prompt_profile="strict_classic7",
                )
                agent.rate_limit = 0
                with self.assertRaises((BackendError, PublicSpeechPlanValidationError, GameplaySpeechQualityError)):
                    agent.act(self._strict_observation())
                self.assertEqual(len(backend.calls), call_count)

        backend = MetadataBackend(['{"public_actions":[]}'])
        agent = GPTAgent(
            backend=backend,
            model_name="agent-model",
            gameplay_prompt_profile="strict_classic7",
        )
        agent.rate_limit = 0
        with self.assertRaisesRegex(BackendError, "JSON Schema support"):
            agent.act(self._strict_observation())
        self.assertEqual(backend.calls, [])

    def test_gameplay_public_speech_quality_accepts_safe_numeric_context(self):
        for speech in (
            "第2天我会关注player1到player7的发言。",
            "昨夜1人死亡，目前有2票需要重新判断。",
            "我认为3号玩家的逻辑更可信。",
        ):
            with self.subTest(speech=speech):
                self.assertEqual(
                    validate_gameplay_public_speech(
                        speech,
                        finish_reason="stop",
                        player_id=1,
                        phase="2_day_speech",
                    ),
                    speech,
                )

    def test_gameplay_public_speech_quality_rejects_deterministic_failures(self):
        cases = (
            ("", None),
            ("   ", None),
            ("正常发言", "length"),
            ("我怀疑player0", "stop"),
            ("我怀疑player8", "stop"),
            ("我怀疑player12", "stop"),
            ("我怀疑0号玩家", "stop"),
            ("我怀疑12号位", "stop"),
            ('{"speech": "我怀疑3号"}', "stop"),
            ("【权威公共状态】存活玩家如下", "stop"),
            ("current_act_idx=3", "stop"),
        )
        for speech, finish_reason in cases:
            with self.subTest(speech=speech, finish_reason=finish_reason):
                with self.assertRaises(GameplaySpeechQualityError):
                    validate_gameplay_public_speech(
                        speech,
                        finish_reason=finish_reason,
                        player_id=1,
                        phase="2_day_speech",
                    )

    def test_gpt_agent_speech_uses_backend_chat_and_agent_model(self):
        backend = MetadataBackend(["这是发言"])
        agent = GPTAgent(
            backend=backend,
            model_name="agent-model",
            temperature=0.2,
        )
        agent.rate_limit = 0
        observation = {
            "phase": "1_day_speech",
            "identity": "Villager",
            "current_act_idx": 1,
            "game_log": [],
            "valid_action": ("speech", -1),
        }

        action = agent.act(observation)

        self.assertEqual(action, ("speech", "这是发言"))
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(backend.calls[0]["model"], "agent-model")
        self.assertEqual(backend.calls[0]["temperature"], 0.2)
        self.assertIsNone(backend.calls[0]["max_tokens"])

    def test_gameplay_speech_requires_explicit_fresh_metadata(self):
        no_metadata_backend = RecordingBackend(["不会被旧路径读取"])
        agent = GPTAgent(
            backend=no_metadata_backend,
            model_name="agent-model",
        )
        agent.rate_limit = 0
        observation = {
            "phase": "1_day_speech",
            "identity": "Villager",
            "current_act_idx": 1,
            "game_log": [],
            "valid_action": ("speech", -1),
        }

        with self.assertRaisesRegex(
            BackendError,
            "must support chat_with_metadata",
        ):
            agent.act(observation)
        self.assertEqual(no_metadata_backend.calls, [])

        backend = MetadataBackend(
            ["第一次发言", "第二次发言"],
            metadata=[{"finish_reason": "stop"}, None],
        )
        agent.backend = backend
        self.assertEqual(agent.act(observation), ("speech", "第一次发言"))
        with self.assertRaisesRegex(
            BackendError,
            "requires finish_reason metadata",
        ):
            agent.act(observation)
        self.assertEqual(len(backend.calls), 2)

    def test_gpt_agent_applies_gameplay_max_tokens_to_speech_and_action(self):
        observations = (
            (
                "speech",
                "公开发言",
                {
                    "phase": "1_day_speech",
                    "identity": "Villager",
                    "current_act_idx": 1,
                    "game_log": [],
                    "valid_action": ("speech", -1),
                },
                ("speech", "公开发言"),
            ),
            (
                "vote",
                "{'投票': '2'}",
                {
                    "phase": "1_day_vote",
                    "identity": "Villager",
                    "current_act_idx": 1,
                    "game_log": [],
                    "valid_action": [("vote", 2)],
                },
                ("vote", 2),
            ),
        )

        for name, response, observation, expected in observations:
            with self.subTest(name=name):
                backend = (
                    MetadataBackend([response])
                    if name == "speech"
                    else RecordingBackend([response])
                )
                agent = GPTAgent(
                    backend=backend,
                    model_name="agent-model",
                    gameplay_max_tokens=512,
                )
                agent.rate_limit = 0

                self.assertEqual(agent.act(observation), expected)
                self.assertEqual(
                    backend.calls[0]["max_tokens"],
                    512,
                )

    def test_gpt_agent_preserves_legacy_o1_limit_when_unconfigured(self):
        backend = MetadataBackend(["公开发言"])
        agent = GPTAgent(
            backend=backend,
            model_name="o1-test-model",
        )
        agent.rate_limit = 0

        agent.act(
            {
                "phase": "1_day_speech",
                "identity": "Villager",
                "current_act_idx": 1,
                "game_log": [],
                "valid_action": ("speech", -1),
            }
        )

        self.assertEqual(backend.calls[0]["max_tokens"], 32000)
        self.assertIsNone(backend.calls[0]["temperature"])

    def test_agent_rejects_invalid_gameplay_max_tokens(self):
        for invalid in (True, False, 0, -1, 1.5, "512"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    ValueError,
                    "gameplay_max_tokens",
                ):
                    GPTAgent(
                        backend=RecordingBackend(),
                        model_name="agent-model",
                        gameplay_max_tokens=invalid,
                    )

    def test_twdm_generation_uses_backend_chat_and_agent_model(self):
        backend = RecordingBackend(["  structured response  "])
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

    def test_registry_injects_backend_and_resolves_model_name(self):
        backend = RecordingBackend()

        agent_type, params = agent_registry.build(
            "gpt",
            backend=backend,
            default_model="default-agent-model",
            temperature=0.3,
            gameplay_prompt_profile="strict_classic7",
            gameplay_max_tokens=512,
        )
        agent = agent_registry.build_agent(
            agent_type,
            player_idx=0,
            agent_param=params,
            env_param={"n_player": 7, "n_role": 4},
            log_file=None,
        )

        self.assertIs(agent.backend, backend)
        self.assertEqual(agent.model_name, "default-agent-model")
        self.assertEqual(agent.temperature, 0.3)
        self.assertEqual(
            agent.gameplay_prompt_profile,
            "strict_classic7",
        )
        self.assertEqual(agent.gameplay_max_tokens, 512)

    def test_registry_supports_per_agent_model_override_and_llm_alias(self):
        backend = RecordingBackend()

        _, explicit_params = agent_registry.build(
            "gpt",
            backend=backend,
            default_model="default",
            model_name="explicit",
            temperature=0.3,
        )
        _, alias_params = agent_registry.build(
            "gpt",
            backend=backend,
            default_model="default",
            llm="legacy-alias",
            temperature=0.3,
        )

        self.assertEqual(explicit_params["model_name"], "explicit")
        self.assertEqual(alias_params["model_name"], "legacy-alias")

    def test_registry_has_no_provider_or_credential_responsibility(self):
        source = inspect.getsource(Registry)

        for forbidden in (
            "openai.OpenAI",
            "openai.AzureOpenAI",
            "OPENAI_API_KEY",
            "DEEPSEEK_API_KEY",
            "AZURE_OPENAI_API_KEY",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
