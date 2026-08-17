import re
import time

from werewolf.agents.llm_agent import (
    BeliefValidationError,
    GameplayActionValidationError,
    LLMAgent,
    RoleReportValidationError,
    belief_response_format,
    day_cognition_response_format,
    night_action_response_format,
    parse_belief_response,
    parse_day_cognition_response,
    parse_vote_response,
    validate_gameplay_public_speech,
    validate_role_report,
    vote_response_format,
)
from werewolf.agents.prompt_template_v0 import (
    STRICT_CLASSIC7_GAMEPLAY_PROMPT_PROFILE,
    build_belief_prompt,
    build_day_cognition_prompt,
    build_public_claim_catalog,
    build_public_speech_prompt,
    build_vote_prompt,
    derive_belief_constraints,
    freeze_discussion_candidates,
)
from werewolf.backends import BackendError
from . import agent_registry as AgentRegistry


_CONSTRAINED_NIGHT_PHASES = (
    "skill_wolf",
    "skill_seer",
    "skill_witch",
)


@AgentRegistry.register(["gpt", "gpt-4", "GPT-4", "gpt4", "o1", "gpt4o", "gpt4o-mini", "deepseek"])
class GPTAgent(LLMAgent):
    def __init__(
        self,
        backend=None,
        model_name=None,
        tokenizer=None,
        temperature=1.0,
        log_file=None,
        gameplay_prompt_profile="legacy",
        gameplay_max_tokens=None,
    ):
        super().__init__(
            backend=backend,
            model_name=model_name,
            tokenizer=tokenizer,
            temperature=temperature,
            log_file=log_file,
            gameplay_prompt_profile=gameplay_prompt_profile,
            gameplay_max_tokens=gameplay_max_tokens,
        )
        self.rate_limit = 6
        self.temperature = temperature

    def act(self, observation):
        phase = observation["phase"]
        is_speech = "speech" in phase
        is_vote = "vote" in phase
        is_night = any(name in phase for name in _CONSTRAINED_NIGHT_PHASES)
        is_strict = self.gameplay_prompt_profile == STRICT_CLASSIC7_GAMEPLAY_PROMPT_PROFILE

        time.sleep(self.rate_limit)
        temperature, max_tokens = self._request_limits()

        if is_speech and is_strict:
            day_cognition, candidate_snapshot, claim_catalog = (
                self._generate_day_cognition(
                    observation,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )
            return self._generate_public_speech(
                observation,
                day_cognition=day_cognition,
                candidate_snapshot=candidate_snapshot,
                claim_catalog=claim_catalog,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        if is_vote and is_strict:
            belief = self._generate_belief(
                observation,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return self._generate_vote(
                observation,
                belief=belief,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        if is_night:
            return self._generate_night_action(
                observation,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        if is_speech:
            prompt = self.format_observation(observation)
            content, metadata = self._chat_with_metadata(
                [{"role": "user", "content": prompt}],
                player_log_context={"stage": "speech", "observation": observation},
                temperature=temperature,
                max_tokens=max_tokens,
            )
            validate_gameplay_public_speech(
                content,
                finish_reason=metadata["finish_reason"],
                player_id=observation.get("current_act_idx"),
                phase=phase,
            )
            return ("speech", self.extract_answer(content.strip()))

        if is_vote:
            raise BackendError(
                "vote cognition requires gameplay_prompt_profile='strict_classic7'"
            )
        raise ValueError(f"unsupported gameplay phase: {phase!r}")

    def _request_limits(self):
        is_o1 = self.model_name is not None and "o1" in self.model_name
        temperature = None if is_o1 else self.temperature
        max_tokens = self.gameplay_max_tokens
        if max_tokens is None and is_o1:
            max_tokens = 32000
        return temperature, max_tokens

    def _generate_belief(self, observation, *, temperature, max_tokens):
        player_id = observation.get("current_act_idx")
        phase = observation.get("phase")
        exact_roles, role_options = derive_belief_constraints(observation)
        prompt = build_belief_prompt(
            observation,
            exact_roles=exact_roles,
            role_options=role_options,
        )
        content, metadata = self._chat_with_metadata(
            [{"role": "user", "content": prompt}],
            player_log_context={"stage": "belief", "observation": observation},
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=belief_response_format(
                supports_json_schema=getattr(self.backend, "supports_json_schema", False),
                role_options=role_options,
            ),
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        if metadata["finish_reason"] == "length":
            raise BeliefValidationError(
                f"belief response was truncated (player={player_id}, phase={phase!r})"
            )
        report = parse_belief_response(
            content,
            player_id=player_id,
            self_role=observation.get("identity"),
            phase=phase,
            exact_roles=exact_roles,
            role_options=role_options,
        )
        try:
            validate_role_report(
                report,
                player_id=player_id,
                self_role=observation.get("identity"),
                phase=phase,
                exact_roles=exact_roles,
                role_options=role_options,
            )
        except RoleReportValidationError:
            # The raw response is already retained by the existing call audit.
            pass
        return report

    def _generate_day_cognition(
        self,
        observation,
        *,
        temperature,
        max_tokens,
    ):
        player_id = observation.get("current_act_idx")
        phase = observation.get("phase")
        exact_roles, role_options = derive_belief_constraints(observation)
        candidate_snapshot = freeze_discussion_candidates(observation)
        claim_catalog = build_public_claim_catalog(observation)
        claim_ids = tuple(claim.claim_id for claim in claim_catalog)
        prompt = build_day_cognition_prompt(
            observation,
            exact_roles=exact_roles,
            role_options=role_options,
            candidate_snapshot=candidate_snapshot,
            claim_catalog=claim_catalog,
        )
        content, metadata = self._chat_with_metadata(
            [{"role": "user", "content": prompt}],
            player_log_context={"stage": "belief", "observation": observation},
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=day_cognition_response_format(
                supports_json_schema=getattr(
                    self.backend,
                    "supports_json_schema",
                    False,
                ),
                role_options=role_options,
                candidate_snapshot=candidate_snapshot,
                claim_ids=claim_ids,
            ),
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        if metadata["finish_reason"] == "length":
            raise BeliefValidationError(
                f"Day cognition response was truncated "
                f"(player={player_id}, phase={phase!r})"
            )
        report = parse_day_cognition_response(
            content,
            player_id=player_id,
            self_role=observation.get("identity"),
            phase=phase,
            exact_roles=exact_roles,
            role_options=role_options,
            candidate_snapshot=candidate_snapshot,
            claim_ids=claim_ids,
        )
        try:
            validate_role_report(
                report,
                player_id=player_id,
                self_role=observation.get("identity"),
                phase=phase,
                exact_roles=exact_roles,
                role_options=role_options,
            )
        except RoleReportValidationError:
            # The raw response is already retained by the existing call audit.
            pass
        return report, candidate_snapshot, claim_catalog

    def _generate_public_speech(
        self,
        observation,
        *,
        day_cognition,
        candidate_snapshot,
        claim_catalog,
        temperature,
        max_tokens,
    ):
        phase = observation.get("phase")
        player_id = observation.get("current_act_idx")
        discussion_acts = tuple(
            candidate_snapshot[index]
            for index in day_cognition.public_action_indices
        )
        claim_by_id = {
            claim.claim_id: claim for claim in claim_catalog
        }
        selected_claims = tuple(
            claim_by_id[claim_id]
            for claim_id in day_cognition.evidence_claim_ids
        )
        prompt = build_public_speech_prompt(
            observation,
            discussion_acts=discussion_acts,
            selected_claims=selected_claims,
        )
        content, metadata = self._chat_with_metadata(
            [{"role": "user", "content": prompt}],
            player_log_context={"stage": "speech", "observation": observation},
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        validate_gameplay_public_speech(
            content,
            finish_reason=metadata["finish_reason"],
            player_id=player_id,
            phase=phase,
        )
        return ("speech", content.strip())

    def _generate_vote(
        self,
        observation,
        *,
        belief,
        temperature,
        max_tokens,
    ):
        phase = observation.get("phase")
        candidates = self.freeze_legal_vote_candidates(
            observation["valid_action"],
            phase=phase,
        )
        legal_targets = tuple(target for target, _action in candidates)
        action_by_target = dict(candidates)
        prompt = build_vote_prompt(
            observation,
            belief.gameplay_dict(),
            legal_targets,
        )
        content, metadata = self._chat_with_metadata(
            [{"role": "user", "content": prompt}],
            player_log_context={"stage": "vote", "observation": observation},
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=vote_response_format(
                supports_json_schema=getattr(self.backend, "supports_json_schema", False),
                legal_targets=legal_targets,
            ),
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        if metadata["finish_reason"] == "length":
            raise GameplayActionValidationError(
                f"vote response was truncated (phase={phase!r})"
            )
        target = parse_vote_response(content, legal_targets=legal_targets, phase=phase)
        return action_by_target[target]

    def _generate_night_action(
        self,
        observation,
        *,
        temperature,
        max_tokens,
    ):
        phase = observation.get("phase")
        candidate_snapshot = self.freeze_authoritative_action_candidates(
            observation["valid_action"]
        )
        prompt = self.format_observation(observation, action_candidates=candidate_snapshot)
        content, metadata = self._chat_with_metadata(
            [{"role": "user", "content": prompt}],
            player_log_context={"stage": "night", "observation": observation},
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=night_action_response_format(
                supports_json_schema=getattr(self.backend, "supports_json_schema", False),
                candidate_snapshot=candidate_snapshot,
            ),
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        if metadata["finish_reason"] == "length":
            raise GameplayActionValidationError(
                f"night action response was truncated (phase={phase!r}, finish_reason='length')"
            )
        _action, env_action = self.parse_night_action_selection(
            content,
            candidate_snapshot,
            phase=phase,
        )
        return env_action

    def extract_answer(self, response):
        pattern = r'\n\n\"(.*?)\"'
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            response = matches[0]
        return response
