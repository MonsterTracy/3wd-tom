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
    GameplayActionValidationError,
    GameplaySpeechQualityError,
    PublicSpeechPlanValidationError,
    _extract_explicit_player_references,
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
                ['{"public_actions":[{"action":"vote_intent","target":1}]}'] * 2,
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
        self.assertNotIn("extra_body", backend.calls[0])
        self.assertEqual(
            backend.calls[1]["extra_body"],
            {
                "chat_template_kwargs": {
                    "enable_thinking": False,
                }
            },
        )
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
                {"action": "check_as_werewolf", "target": 2},
                {"action": "point_as_werewolf", "target": 2},
            ],
            [
                {"action": "check_as_good", "target": 2},
                {"action": "point_as_villager", "target": 2},
            ],
            [
                {"action": "point_as_villager", "target": 2},
                {"action": "check_as_good", "target": 2},
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
            {"public_actions": [{"action": ["bad"], "target": 1}]},
            {"public_actions": [{"action": "oppose", "target": 3, "why": "x"}]},
            {"public_actions": [{"action": "vote_intent", "target": 1}]},
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

    def test_public_speech_plan_stably_deduplicates_exact_pairs(self):
        actions = [
            {"action": "support", "target": 2},
            {"action": "check_as_good", "target": 1},
            {"action": "oppose", "target": 5},
            {"action": "check_as_good", "target": 1},
            {"action": "vote_intent", "target": 7},
        ]
        plan = validate_public_speech_plan(
            {"public_actions": actions},
            suggestible_player_ids=(2, 3, 4, 5, 6, 7),
            player_id=1,
            speaker_role="Villager",
            phase="1_day_speech",
        )
        self.assertEqual(plan.as_list(), [
            {"action": "support", "target": 2},
            {"action": "check_as_good", "target": 1},
            {"action": "oppose", "target": 5},
            {"action": "vote_intent", "target": 7},
        ])

        distinct_pairs = [
            {"action": "check_as_good", "target": 4},
            {"action": "point_as_villager", "target": 4},
            {"action": "check_as_werewolf", "target": 4},
            {"action": "support", "target": 1},
            {"action": "support", "target": 2},
            {"action": "vote_intent", "target": 2},
            {"action": "vote_intent", "target": 3},
        ]
        plan = validate_public_speech_plan(
            {"public_actions": distinct_pairs},
            suggestible_player_ids=(2, 3, 4, 5, 6, 7),
            player_id=1,
            speaker_role="Villager",
            phase="1_day_speech",
        )
        self.assertEqual(plan.as_list(), distinct_pairs)

    def test_public_speech_plan_rejects_observed_claim_contradictions(self):
        common = {
            "suggestible_player_ids": (1, 2, 4, 5, 6, 7),
            "player_id": 3,
            "speaker_role": "Witch",
            "phase": "1_day_speech",
        }
        invalid_plans = (
            [
                {"action": "point_as_villager", "target": 3},
                {"action": "point_as_witch", "target": 3},
            ],
            [
                {"action": "point_as_witch", "target": 3},
                {"action": "check_as_good", "target": 1},
            ],
        )
        for actions in invalid_plans:
            with self.subTest(actions=actions), self.assertRaises(
                PublicSpeechPlanValidationError
            ):
                validate_public_speech_plan(
                    {"public_actions": actions},
                    **common,
                )

        bluff = [
            {"action": "check_as_good", "target": 1},
            {"action": "vote_intent", "target": 2},
        ]
        self.assertEqual(
            validate_public_speech_plan(
                {"public_actions": bluff},
                **common,
            ).as_list(),
            bluff,
        )

    def test_seed_510_duplicate_plan_reaches_renderer_once(self):
        observation = self._strict_observation()
        observation["phase"] = "2_day_speech_pk"
        observation["current_act_idx"] = 2
        observation["authoritative_public_state"].update({
            "day": 2,
            "phase": "speech_pk",
            "suggestible_exile_targets": [1, 3, 4, 5, 6, 7],
        })
        backend = MetadataBackend([
            '{"public_actions":['
            '{"action":"check_as_good","target":1},'
            '{"action":"check_as_good","target":1}]}',
            "我查验1号是好人。",
        ])
        backend.supports_json_schema = True
        backend.session = SimpleNamespace(game_id="game_006_seed_510")
        agent = GPTAgent(
            backend=backend,
            model_name="agent-model",
            gameplay_prompt_profile="strict_classic7",
        )
        agent.rate_limit = 0

        self.assertEqual(
            agent.act(observation),
            ("speech", "我查验1号是好人。"),
        )
        self.assertEqual(len(backend.calls), 2)
        renderer_prompt = backend.calls[1]["messages"][0]["content"]
        self.assertEqual(renderer_prompt.count("check_as_good(player1)"), 1)

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
        ] * 2)
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
        self.assertEqual(len(backend.calls), 2)
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

        backend = MetadataBackend([json.dumps(payload)] * 2)
        backend.supports_json_schema = True
        agent = GPTAgent(
            backend=backend,
            model_name="agent-model",
            gameplay_prompt_profile="strict_classic7",
        )
        agent.rate_limit = 0
        with self.assertRaises(PublicSpeechPlanValidationError):
            agent.act(observation)
        self.assertEqual(len(backend.calls), 2)

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

    def test_range_reference_does_not_satisfy_explicit_plan_coverage(self):
        with self.assertRaisesRegex(
            GameplaySpeechQualityError,
            r"planned player reference\(s\) missing \[3, 4, 5, 6\]",
        ):
            validate_gameplay_public_speech(
                "我查验了1号至7号，结果均为好人，且未获知具体身份。",
                finish_reason="stop",
                player_id=2,
                phase="1_day_speech",
                planned_player_ids=set(range(1, 8)),
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

    def test_seed_539_self_and_grouped_player_references_cover_plan(self):
        speech = (
            "我认为1号是预言家，同时支持7号。"
            "但我对3、4、5号的发言均存疑，认为他们可疑。"
            "经查验，我确认自己是好人。"
            "基于以上判断，我准备投票放逐4号。"
        )
        self.assertEqual(
            _extract_explicit_player_references(
                speech,
                speaker_id=2,
                context="player=2, phase=1_day_speech",
            ),
            {1, 2, 3, 4, 5, 7},
        )
        self.assertEqual(
            validate_gameplay_public_speech(
                speech,
                finish_reason="stop",
                player_id=2,
                phase="1_day_speech",
                planned_player_ids={1, 2, 3, 4, 5, 7},
            ),
            speech,
        )

    def test_grouped_player_references_require_explicit_player_suffix(self):
        for separator in ("、", "，", ","):
            with self.subTest(separator=separator):
                content = f"我质疑3{separator}4{separator}5号。"
                self.assertEqual(
                    _extract_explicit_player_references(
                        content,
                        speaker_id=2,
                        context="player=2, phase=1_day_speech",
                    ),
                    {2, 3, 4, 5},
                )
        for content in ("2025年", "3票", "3人", "第3项", "3、4、5人"):
            with self.subTest(content=content):
                self.assertEqual(
                    _extract_explicit_player_references(
                        content,
                        speaker_id=2,
                        context="player=2, phase=1_day_speech",
                    ),
                    set(),
                )

    def test_self_reference_maps_only_to_current_speaker(self):
        for content in ("我会说明。", "我自己会说明。", "自己会说明。"):
            with self.subTest(content=content):
                self.assertEqual(
                    _extract_explicit_player_references(
                        content,
                        speaker_id=2,
                        context="player=2, phase=1_day_speech",
                    ),
                    {2},
                )
        for content in ("他自己会说明。", "他们自己会说明。", "他会说明。"):
            with self.subTest(content=content):
                self.assertEqual(
                    _extract_explicit_player_references(
                        content,
                        speaker_id=2,
                        context="player=2, phase=1_day_speech",
                    ),
                    set(),
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

    def test_strict_speech_uses_bounded_plan_retry_and_no_fallback(self):
        cases = (
            ([BackendError("planner failed")], None, 1),
            (["not json"] * 2, None, 2),
            (
                ['{"public_actions":[]}'] * 2,
                [{"finish_reason": "length"}] * 2,
                2,
            ),
            (
                ['{"public_actions":[{"action":"vote_intent","target":1}]}'] * 2,
                None,
                2,
            ),
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

    def test_strict_speech_regenerates_complete_plan_with_same_input(self):
        invalid_plan = json.dumps({"public_actions": [
            {"action": "point_as_villager", "target": 1},
            {"action": "point_as_witch", "target": 1},
        ]})
        valid_plan = json.dumps({"public_actions": [
            {"action": "oppose", "target": 3},
        ]})

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "Player_1.jsonl"
            backend = MetadataBackend([
                invalid_plan,
                valid_plan,
                "我不认可3号。",
            ])
            backend.supports_json_schema = True
            agent = GPTAgent(
                backend=backend,
                model_name="agent-model",
                log_file=str(log_path),
                gameplay_prompt_profile="strict_classic7",
                gameplay_max_tokens=512,
            )
            agent.rate_limit = 0
            try:
                self.assertEqual(
                    agent.act(self._strict_observation()),
                    ("speech", "我不认可3号。"),
                )
            finally:
                agent.close()

            self.assertEqual(len(backend.calls), 3)
            first, second, renderer = backend.calls
            self.assertIs(first["messages"], second["messages"])
            self.assertIs(
                first["response_format"],
                second["response_format"],
            )
            for field in ("messages", "response_format", "temperature", "max_tokens"):
                self.assertEqual(first[field], second[field])
            self.assertEqual(first["temperature"], 1.0)
            self.assertEqual(first["max_tokens"], 512)
            self.assertNotIn(invalid_plan, second["messages"][0]["content"])
            self.assertNotIn("correction", second["messages"][0]["content"])
            self.assertIsNone(renderer["response_format"])

            plan_records = [
                record
                for record in self._player_records(log_path)
                if record["message"] == "speech_plan"
            ]
            self.assertEqual(
                [record["gen_times"] for record in plan_records],
                [0, 1],
            )

    def test_strict_speech_aborts_after_two_invalid_plans(self):
        invalid_plan = json.dumps({"public_actions": [
            {"action": "point_as_villager", "target": 1},
            {"action": "point_as_witch", "target": 1},
        ]})
        backend = MetadataBackend([invalid_plan, invalid_plan])
        backend.supports_json_schema = True
        agent = GPTAgent(
            backend=backend,
            model_name="agent-model",
            gameplay_prompt_profile="strict_classic7",
        )
        agent.rate_limit = 0

        with self.assertRaisesRegex(
            PublicSpeechPlanValidationError,
            "exhausted 2 attempts.*both Villager and Witch",
        ) as captured:
            agent.act(self._strict_observation())
        self.assertIn(
            "game=unavailable, player=1, phase=1_day_speech",
            str(captured.exception),
        )
        self.assertIsInstance(
            captured.exception.__cause__,
            PublicSpeechPlanValidationError,
        )
        self.assertEqual(len(backend.calls), 2)
        self.assertTrue(all(
            call["response_format"] is not None
            for call in backend.calls
        ))

    def test_strict_speech_does_not_retry_backend_or_renderer_failure(self):
        backend = MetadataBackend([BackendError("planner failed")])
        backend.supports_json_schema = True
        agent = GPTAgent(
            backend=backend,
            model_name="agent-model",
            gameplay_prompt_profile="strict_classic7",
        )
        agent.rate_limit = 0
        with self.assertRaisesRegex(BackendError, "planner failed"):
            agent.act(self._strict_observation())
        self.assertEqual(len(backend.calls), 1)

        backend = MetadataBackend([
            '{"public_actions":[{"action":"oppose","target":3}]}',
            "我不认可4号。",
        ])
        backend.supports_json_schema = True
        agent = GPTAgent(
            backend=backend,
            model_name="agent-model",
            gameplay_prompt_profile="strict_classic7",
        )
        agent.rate_limit = 0
        with self.assertRaises(GameplaySpeechQualityError):
            agent.act(self._strict_observation())
        self.assertEqual(len(backend.calls), 2)
        self.assertIsNotNone(backend.calls[0]["response_format"])
        self.assertIsNone(backend.calls[1]["response_format"])

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
        self.assertNotIn("extra_body", backend.calls[0])

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
                if name == "vote":
                    self.assertEqual(
                        backend.calls[0]["temperature"],
                        1.0,
                    )
                    self.assertEqual(
                        backend.calls[0]["extra_body"],
                        {
                            "chat_template_kwargs": {
                                "enable_thinking": False,
                            }
                        },
                    )
                else:
                    self.assertNotIn("extra_body", backend.calls[0])

    def test_vote_retries_three_times_then_keeps_existing_fallback(self):
        backend = RecordingBackend(["bad", "still bad", "also bad"])
        agent = GPTAgent(
            backend=backend,
            model_name="agent-model",
            temperature=0.6,
            gameplay_max_tokens=512,
        )
        agent.rate_limit = 0
        fallback_calls = []
        agent.choose_fallback_vote_action = lambda *_args: (
            fallback_calls.append(True) or "{'投票': '2'}"
        )
        observation = {
            "phase": "1_day_vote",
            "identity": "Villager",
            "current_act_idx": 1,
            "game_log": [],
            "valid_action": [("vote", 2), ("vote", 3)],
        }

        self.assertEqual(agent.act(observation), ("vote", 2))
        self.assertEqual(len(backend.calls), 3)
        self.assertEqual(fallback_calls, [True])
        for call in backend.calls:
            self.assertEqual(call["temperature"], 0.6)
            self.assertEqual(call["max_tokens"], 512)
            self.assertEqual(
                call["extra_body"],
                {
                    "chat_template_kwargs": {
                        "enable_thinking": False,
                    }
                },
            )

    def test_invalid_vote_response_retries_then_accepts_valid_response(self):
        backend = RecordingBackend(["bad", "{'投票': '3'}"])
        agent = GPTAgent(
            backend=backend,
            model_name="agent-model",
            gameplay_max_tokens=512,
        )
        agent.rate_limit = 0
        observation = {
            "phase": "1_day_vote_pk",
            "identity": "Villager",
            "current_act_idx": 1,
            "game_log": [],
            "valid_action": [("vote_pk", 2), ("vote_pk", 3)],
        }

        self.assertEqual(agent.act(observation), ("vote_pk", 3))
        self.assertEqual(len(backend.calls), 2)

    def test_v25_structural_matcher_remains_the_membership_boundary(self):
        agent = GPTAgent()
        candidates = (
            "{'查验':'7'}",
            "{'解药': '否', '毒药': '4'}",
        )
        cases = (
            ("{'查验': '7'}", "{'查验':'7'}"),
            ('```json\n{"查验":"7"}\n```', "{'查验':'7'}"),
            (
                "{'毒药': '4', '解药': '否'}",
                "{'解药': '否', '毒药': '4'}",
            ),
        )
        for response, expected in cases:
            with self.subTest(response=response):
                self.assertEqual(
                    agent.match_authoritative_action_response(
                        response,
                        candidates,
                    ),
                    expected,
                )

    def test_v26_night_snapshot_drives_prompt_schema_and_mapping(self):
        cases = (
            (
                "seer",
                "1_night_skill_seer",
                [("check", 1), ("check", 2), ("check", 4), ("check", 5)],
                '{"action_index": 2}',
                ("check", 4),
                "{'查验':'4'}",
            ),
            (
                "werewolf",
                "0_night_skill_wolf",
                [("kill", 3), ("kill", 7)],
                '```json\n{"action_index":0}\n```',
                ("kill", 3),
                "{'杀害':'3'}",
            ),
            (
                "guard",
                "0_night_skill_guard",
                [("guard", 2), ("guard", 4), ("guard", 6)],
                '{"action_index":1}',
                ("guard", 4),
                "{'守卫':'4'}",
            ),
            (
                "witch",
                "0_night_skill_witch",
                [
                    ("witch_pass", 0),
                    ("witch_poison", 1),
                    ("witch_poison", 2),
                    ("witch_heal", 4),
                ],
                '{"action_index":3}',
                ("witch_heal", 4),
                "{'解药': '4', '毒药': '否'}",
            ),
        )
        for name, phase, valid_actions, response, expected, display in cases:
            with self.subTest(name=name):
                backend = MetadataBackend([response])
                backend.supports_json_schema = True
                agent = GPTAgent(
                    backend=backend,
                    model_name="agent-model",
                    gameplay_max_tokens=512,
                )
                agent.rate_limit = 0

                action = agent.act({
                    "phase": phase,
                    "identity": "Villager",
                    "current_act_idx": 1,
                    "game_log": [],
                    "valid_action": valid_actions,
                })

                self.assertEqual(action, expected)
                self.assertEqual(len(backend.calls), 1)
                request = backend.calls[0]
                schema = request["response_format"]["json_schema"]["schema"]
                self.assertEqual(request["response_format"]["type"], "json_schema")
                self.assertEqual(
                    request["response_format"]["json_schema"]["name"],
                    "night_action_selection",
                )
                self.assertTrue(
                    request["response_format"]["json_schema"]["strict"]
                )
                self.assertEqual(request["max_tokens"], 512)
                self.assertEqual(
                    request["extra_body"],
                    {
                        "chat_template_kwargs": {
                            "enable_thinking": False,
                        }
                    },
                )
                self.assertEqual(
                    schema["properties"]["action_index"]["enum"],
                    list(range(len(valid_actions))),
                )
                prompt = request["messages"][0]["content"]
                snapshot = agent.freeze_authoritative_action_candidates(
                    valid_actions
                )
                for index, (candidate_display, env_action) in enumerate(snapshot):
                    self.assertIn(f"{index}: {candidate_display}", prompt)
                    self.assertEqual(valid_actions[index], env_action)
                self.assertEqual(
                    agent.nlp_action_to_env_action[display],
                    expected,
                )
                if name == "seer":
                    self.assertNotIn("{'查验':'3'}", prompt)
                elif name == "werewolf":
                    self.assertNotIn("{'杀害':'2'}", prompt)
                elif name == "guard":
                    self.assertNotIn("{'守卫':'3'}", prompt)
                else:
                    self.assertNotIn(
                        "{'解药': '4', '毒药': '1'}",
                        prompt,
                    )

    def test_v26_seed520_witch_has_no_index_for_poisoned_player3(self):
        valid_actions = [
            ("witch_pass", 0),
            ("witch_poison", 1),
            ("witch_poison", 2),
            ("witch_poison", 4),
            ("witch_poison", 5),
            ("witch_poison", 6),
            ("witch_poison", 7),
        ]
        backend = MetadataBackend(['{"action_index":3}'])
        backend.supports_json_schema = True
        agent = GPTAgent(backend=backend, model_name="agent-model")
        agent.rate_limit = 0

        self.assertEqual(
            agent.act({
                "phase": "1_night_skill_witch",
                "identity": "Witch",
                "current_act_idx": 6,
                "game_log": [],
                "valid_action": valid_actions,
            }),
            ("witch_poison", 4),
        )
        prompt = backend.calls[0]["messages"][0]["content"]
        self.assertNotIn("'毒药': '3'", prompt)
        self.assertEqual(
            backend.calls[0]["response_format"]["json_schema"]["schema"]
            ["properties"]["action_index"]["enum"],
            list(range(len(valid_actions))),
        )

    def test_v26_logs_raw_index_and_canonical_action_without_regeneration(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "Player_5.jsonl"
            backend = MetadataBackend(['{"action_index":1}'])
            backend.supports_json_schema = True
            agent = GPTAgent(
                backend=backend,
                model_name="agent-model",
                log_file=str(log_path),
            )
            agent.rate_limit = 0
            try:
                self.assertEqual(
                    agent.act({
                        "phase": "1_night_skill_seer",
                        "identity": "Seer",
                        "current_act_idx": 5,
                        "game_log": [],
                        "valid_action": [("check", 3), ("check", 7)],
                    }),
                    ("check", 7),
                )
            finally:
                agent.close()

            records = self._player_records(log_path)
            self.assertEqual(len(backend.calls), 1)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["response"], '{"action_index":1}')
            self.assertEqual(records[0]["action"], "{'查验':'7'}")
            self.assertEqual(records[0]["gen_times"], 0)

    def test_v26_rejects_invalid_structured_night_response_once(self):
        invalid_responses = (
            "{}",
            '{"action_index":-1}',
            '{"action_index":999}',
            '{"action_index":true}',
            '{"action_index":1.0}',
            '{"action_index":"1"}',
            '{"action_index":null}',
            '{"action_index":1,"extra":2}',
            "[]",
            "null",
            "not json",
            'Thinking about the choice.\n{"action_index":0}',
        )

        for response in invalid_responses:
            with self.subTest(response=response):
                backend = MetadataBackend([response])
                backend.supports_json_schema = True
                agent = GPTAgent(
                    backend=backend,
                    model_name="agent-model",
                )
                agent.rate_limit = 0
                with self.assertRaisesRegex(
                    GameplayActionValidationError,
                    "invalid night action selection.*phase=.*response=",
                ):
                    agent.act({
                        "phase": "0_night_skill_witch",
                        "identity": "Witch",
                        "current_act_idx": 7,
                        "game_log": [],
                        "valid_action": [
                            ("witch_pass", 0),
                            ("witch_heal", 3),
                            ("witch_poison", 4),
                        ],
                    })
                self.assertEqual(len(backend.calls), 1)

    def test_v26_truncated_night_response_is_fatal_without_retry(self):
        backend = MetadataBackend(
            ['{"action_index":0}'],
            metadata=[{"finish_reason": "length"}],
        )
        backend.supports_json_schema = True
        agent = GPTAgent(
            backend=backend,
            model_name="agent-model",
        )
        agent.rate_limit = 0

        with self.assertRaisesRegex(
            GameplayActionValidationError,
            "phase=.*finish_reason='length'",
        ):
            agent.act({
                "phase": "1_night_skill_seer",
                "identity": "Seer",
                "current_act_idx": 5,
                "game_log": [],
                "valid_action": [("check", 7)],
            })
        self.assertEqual(len(backend.calls), 1)

    def test_v26_night_action_requires_backend_model_and_schema_support(self):
        cases = (
            (None, "agent-model", "backend and model_name are required"),
            (
                MetadataBackend(['{"action_index":0}']),
                None,
                "backend and model_name are required",
            ),
            (
                MetadataBackend(['{"action_index":0}']),
                "agent-model",
                "require backend JSON Schema support",
            ),
        )

        for backend, model_name, error in cases:
            with self.subTest(
                has_backend=backend is not None,
                model_name=model_name,
            ):
                agent = GPTAgent(
                    backend=backend,
                    model_name=model_name,
                )
                agent.rate_limit = 0
                with self.assertRaisesRegex(
                    BackendError,
                    error,
                ):
                    agent.act({
                        "phase": "1_night_skill_seer",
                        "identity": "Seer",
                        "current_act_idx": 5,
                        "game_log": [],
                        "valid_action": [("check", 7)],
                    })
                if backend is not None:
                    self.assertEqual(backend.calls, [])

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
