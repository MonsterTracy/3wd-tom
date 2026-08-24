import re
import time

from werewolf.agents.llm_agent import (
    BeliefValidationError,
    GameplayActionValidationError,
    GameplayGenerationExhausted,
    GameplaySpeechQualityError,
    LLMAgent,
    RoleReportValidationError,
    belief_response_format,
    day_cognition_response_format_v3,
    night_action_response_format,
    parse_belief_response,
    parse_day_cognition_response_v3,
    parse_vote_response,
    validate_gameplay_public_speech,
    validate_role_report,
    vote_response_format,
)
from werewolf.agents.prompt_template_v0 import (
    STRICT_CLASSIC7_GAMEPLAY_PROMPT_PROFILE,
    build_belief_prompt,
    build_day_cognition_prompt,
    build_public_speech_realization_prompt,
    build_public_claim_catalog,
    build_vote_prompt,
    compile_discussion_intent_v2,
    derive_belief_constraints,
    freeze_discussion_candidates,
)
from werewolf.backends import BackendError
from werewolf.models.twd_tom.schema import normalize_player
from werewolf.models.twd_tom.samples import SpeakerPreSpeechBelief
from . import agent_registry as AgentRegistry


_CONSTRAINED_NIGHT_PHASES = (
    "skill_wolf",
    "skill_seer",
    "skill_witch",
)
GAMEPLAY_GENERATION_MAX_ATTEMPTS = 3
_GAMEPLAY_GENERATION_ERRORS = (
    BeliefValidationError,
    GameplayActionValidationError,
    GameplaySpeechQualityError,
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
        return self._act(observation, pre_speech_belief=None)

    def act_with_pre_speech_belief(
        self,
        observation,
        *,
        pre_speech_belief,
    ):
        """Generate speech from the exact immutable PRE self-report."""

        if not isinstance(pre_speech_belief, SpeakerPreSpeechBelief):
            raise TypeError(
                "pre_speech_belief must be SpeakerPreSpeechBelief"
            )
        if self.gameplay_prompt_profile != STRICT_CLASSIC7_GAMEPLAY_PROMPT_PROFILE:
            raise ValueError(
                "PRE-belief cognition handoff requires strict_classic7 gameplay"
            )
        if normalize_player(observation.get("current_act_idx")) != (
            pre_speech_belief.observer_id
        ):
            raise ValueError("PRE-belief observer does not match current speaker")
        if "speech" not in observation.get("phase", ""):
            raise ValueError("PRE-belief handoff is only valid for speech")
        return self._act(
            observation,
            pre_speech_belief=pre_speech_belief,
        )

    def _act(self, observation, *, pre_speech_belief):
        phase = observation["phase"]
        is_speech = "speech" in phase
        speech_kind = "speech_pk" if "speech_pk" in phase else "speech"
        is_vote = "vote" in phase
        is_night = any(name in phase for name in _CONSTRAINED_NIGHT_PHASES)
        is_strict = self.gameplay_prompt_profile == STRICT_CLASSIC7_GAMEPLAY_PROMPT_PROFILE

        time.sleep(self.rate_limit)
        temperature, max_tokens = self._request_limits()

        if is_speech and is_strict:
            day_cognition, candidate_snapshot, claim_catalog = (
                self._generate_day_cognition(
                    observation,
                    pre_speech_belief=pre_speech_belief,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )
            discussion_acts = compile_discussion_intent_v2(
                candidate_snapshot,
                public_content_action_indices=(
                    day_cognition.public_content_action_indices
                ),
                public_vote_stance_index=(
                    day_cognition.public_vote_stance_index
                ),
            )
            raw_text = self._generate_public_speech(
                observation,
                discussion_acts=discussion_acts,
                claim_catalog=claim_catalog,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return speech_kind, raw_text

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
            def generate_speech(attempt):
                content, metadata = self._chat_with_metadata(
                    [{"role": "user", "content": prompt}],
                    player_log_context={
                        "stage": "speech",
                        "observation": observation,
                        "gen_times": attempt - 1,
                    },
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                validate_gameplay_public_speech(
                    content,
                    finish_reason=metadata["finish_reason"],
                    player_id=observation.get("current_act_idx"),
                    phase=phase,
                )
                return speech_kind, self.extract_answer(content.strip())

            return self._retry_validated_generation(
                stage="speech",
                generate=generate_speech,
            )

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

    def _retry_validated_generation(self, *, stage, generate):
        last_error = None
        for attempt in range(1, GAMEPLAY_GENERATION_MAX_ATTEMPTS + 1):
            try:
                return generate(attempt)
            except _GAMEPLAY_GENERATION_ERRORS as exc:
                last_error = exc
        raise GameplayGenerationExhausted(
            stage=stage,
            attempts=GAMEPLAY_GENERATION_MAX_ATTEMPTS,
            last_error=last_error,
        ) from last_error

    def _generate_belief(self, observation, *, temperature, max_tokens):
        return self._retry_validated_generation(
            stage="belief",
            generate=lambda attempt: self._generate_belief_once(
                observation,
                temperature=temperature,
                max_tokens=max_tokens,
                attempt=attempt,
            ),
        )

    def _generate_belief_once(
        self,
        observation,
        *,
        temperature,
        max_tokens,
        attempt,
    ):
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
            player_log_context={
                "stage": "belief",
                "observation": observation,
                "gen_times": attempt - 1,
            },
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
        pre_speech_belief,
        temperature,
        max_tokens,
    ):
        return self._retry_validated_generation(
            stage="day_cognition",
            generate=lambda attempt: self._generate_day_cognition_once(
                observation,
                pre_speech_belief=pre_speech_belief,
                temperature=temperature,
                max_tokens=max_tokens,
                attempt=attempt,
            ),
        )

    def _generate_day_cognition_once(
        self,
        observation,
        *,
        pre_speech_belief,
        temperature,
        max_tokens,
        attempt,
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
            pre_speech_belief=pre_speech_belief,
        )
        content, metadata = self._chat_with_metadata(
            [{"role": "user", "content": prompt}],
            player_log_context={
                "stage": "belief",
                "observation": observation,
                "gen_times": attempt - 1,
            },
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=day_cognition_response_format_v3(
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
        report = parse_day_cognition_response_v3(
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
        discussion_acts,
        claim_catalog,
        temperature,
        max_tokens,
    ):
        """Generate public natural language from a frozen communication intent."""

        return self._retry_validated_generation(
            stage="speech_realization",
            generate=lambda attempt: self._generate_public_speech_once(
                observation,
                discussion_acts=discussion_acts,
                claim_catalog=claim_catalog,
                temperature=temperature,
                max_tokens=max_tokens,
                attempt=attempt,
            ),
        )

    def _generate_public_speech_once(
        self,
        observation,
        *,
        discussion_acts,
        claim_catalog,
        temperature,
        max_tokens,
        attempt,
    ):
        """Generate one public realization from a frozen communication intent."""

        phase = observation.get("phase")
        player_id = observation.get("current_act_idx")
        prompt = build_public_speech_realization_prompt(
            observation,
            discussion_acts=discussion_acts,
            claim_catalog=claim_catalog,
        )
        content, metadata = self._chat_with_metadata(
            [{"role": "user", "content": prompt}],
            player_log_context={
                "stage": "speech_realization",
                "observation": observation,
                "gen_times": attempt - 1,
            },
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        validate_gameplay_public_speech(
            content,
            finish_reason=metadata["finish_reason"],
            player_id=player_id,
            phase=phase,
            strict_chinese=True,
            required_player_ids=(
                act.target
                for act in discussion_acts
                if act.target is not None
            ),
        )
        return content.strip()

    def _generate_vote(
        self,
        observation,
        *,
        belief,
        temperature,
        max_tokens,
    ):
        return self._retry_validated_generation(
            stage="vote",
            generate=lambda attempt: self._generate_vote_once(
                observation,
                belief=belief,
                temperature=temperature,
                max_tokens=max_tokens,
                attempt=attempt,
            ),
        )

    def _generate_vote_once(
        self,
        observation,
        *,
        belief,
        temperature,
        max_tokens,
        attempt,
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
            player_log_context={
                "stage": "vote",
                "observation": observation,
                "gen_times": attempt - 1,
            },
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
        return self._retry_validated_generation(
            stage="night_action",
            generate=lambda attempt: self._generate_night_action_once(
                observation,
                temperature=temperature,
                max_tokens=max_tokens,
                attempt=attempt,
            ),
        )

    def _generate_night_action_once(
        self,
        observation,
        *,
        temperature,
        max_tokens,
        attempt,
    ):
        phase = observation.get("phase")
        candidate_snapshot = self.freeze_authoritative_action_candidates(
            observation["valid_action"]
        )
        prompt = self.format_observation(observation, action_candidates=candidate_snapshot)
        content, metadata = self._chat_with_metadata(
            [{"role": "user", "content": prompt}],
            player_log_context={
                "stage": "night",
                "observation": observation,
                "gen_times": attempt - 1,
            },
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
