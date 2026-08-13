import json
import time
import re
from werewolf.agents.llm_agent import (
    GameplayActionValidationError,
    LLMAgent,
    PublicSpeechPlanValidationError,
    canonical_suggestible_player_ids,
    discourse_public_speech_plan_response_format,
    night_action_response_format,
    public_evidence_player_ids,
    public_speech_plan_response_format,
    validate_gameplay_public_speech,
    validate_discourse_public_speech_plan,
    validate_public_speech_plan,
)
from werewolf.backends import BackendError
from werewolf.agents.prompt_template_v0 import (
    CON,
    STRICT_CLASSIC7_GAMEPLAY_PROMPT_PROFILE,
    STRICT_CLASSIC7_DISCOURSE_GAMEPLAY_PROMPT_PROFILE,
    _render_authoritative_public_phase,
    build_strict_classic7_discourse_speech_render_prompt,
    build_strict_classic7_speech_render_prompt,
)
from werewolf.models.twd_tom.public_events import normalize_public_events
from . import agent_registry as AgentRegistry


_CONSTRAINED_NIGHT_PHASES = (
    "skill_wolf",
    "skill_seer",
    "skill_witch",
)


@AgentRegistry.register(["gpt", "gpt-4", "GPT-4", "gpt4", "o1", "gpt4o", "gpt4o-mini", 'deepseek'])
class GPTAgent(LLMAgent):
    def __init__(self,
                 backend=None,
                 model_name=None,
                 tokenizer=None,
                 temperature=1.0,
                 log_file=None,
                 gameplay_prompt_profile="legacy",
                 gameplay_max_tokens=None):
        super().__init__(backend=backend, model_name=model_name, tokenizer=tokenizer,
                         temperature=temperature, log_file=log_file,
                         gameplay_prompt_profile=gameplay_prompt_profile,
                         gameplay_max_tokens=gameplay_max_tokens)
        self.rate_limit = 6
        self.temperature = temperature

    def act(self, observation):
        phase = observation['phase']
        is_constrained_night_action = any(
            night_phase in phase
            for night_phase in _CONSTRAINED_NIGHT_PHASES
        )
        is_strict_speech = (
            'speech' in phase
            and self.gameplay_prompt_profile in {
                STRICT_CLASSIC7_GAMEPLAY_PROMPT_PROFILE,
                STRICT_CLASSIC7_DISCOURSE_GAMEPLAY_PROMPT_PROFILE,
            }
        )
        suggestible_player_ids = None
        if is_strict_speech:
            suggestible_player_ids = canonical_suggestible_player_ids(
                observation.get("authoritative_public_state")
            )
        if (
            is_strict_speech
            and self.gameplay_prompt_profile
            == STRICT_CLASSIC7_DISCOURSE_GAMEPLAY_PROMPT_PROFILE
        ):
            public_events = normalize_public_events(
                observation.get("canonical_public_events")
            )
            observation = dict(observation)
            observation["canonical_public_events"] = public_events
        night_candidate_snapshot = None
        if is_constrained_night_action:
            night_candidate_snapshot = (
                self.freeze_authoritative_action_candidates(
                    observation["valid_action"]
                )
            )
        prompt = self.format_observation(
            observation,
            suggestible_player_ids=suggestible_player_ids,
            action_candidates=night_candidate_snapshot,
        )
        valid_action = list(self.nlp_action_to_env_action.keys())  
        time.sleep(self.rate_limit)
        is_o1 = self.model_name is not None and "o1" in self.model_name
        request_temperature = None if is_o1 else self.temperature
        request_max_tokens = self.gameplay_max_tokens
        if request_max_tokens is None and is_o1:
            request_max_tokens = 32000
        if 'speech' in phase:
            if is_strict_speech:
                raw_action, checked_action, prompt = (
                    self._generate_strict_public_speech(
                        observation=observation,
                        planner_prompt=prompt,
                        suggestible_player_ids=suggestible_player_ids,
                        temperature=request_temperature,
                        max_tokens=request_max_tokens,
                    )
                )
            else:
                messages = [{'role': 'user', 'content': prompt}]
                raw_action, metadata = self._chat_with_metadata(
                    messages,
                    temperature=request_temperature,
                    max_tokens=request_max_tokens,
                )
                validate_gameplay_public_speech(
                    raw_action,
                    finish_reason=metadata["finish_reason"],
                    player_id=observation.get("current_act_idx"),
                    phase=phase,
                )
                raw_action = raw_action.strip()
                checked_action = self.extract_answer(raw_action)
            gen_times = 0
            env_action = ('speech', checked_action)

            if self.has_log and not is_strict_speech:
                self.logger.info(phase,
                                 extra={"prompt": prompt,
                                        "response": checked_action,
                                        "action": raw_action,
                                        "player_id": observation['current_act_idx'],
                                        "role": observation['identity'],
                                        "phase": phase,
                                        "gen_times": gen_times})
        else: 
            retry_count = 0
            raw_action = None
            selected_env_action = None
            if self.backend is not None and self.model_name:
                action = ''
                while action not in valid_action:
                    retry_count += 1
                    if retry_count > 3:
                        if "vote" in phase:
                            raw_action = self.choose_fallback_vote_action(observation, valid_action)
                        else:
                            raise GameplayActionValidationError(
                                "night action response was not resolved"
                            )
                        action = raw_action
                        break
                    messages = [{'role': 'user', 'content': prompt}]
                    if is_constrained_night_action:
                        raw_action, metadata = self._chat_with_metadata(
                            messages,
                            temperature=request_temperature,
                            max_tokens=request_max_tokens,
                            response_format=night_action_response_format(
                                supports_json_schema=getattr(
                                    self.backend,
                                    "supports_json_schema",
                                    False,
                                ),
                                candidate_snapshot=night_candidate_snapshot,
                            ),
                        )
                        if metadata["finish_reason"] == "length":
                            raise GameplayActionValidationError(
                                "night action response was truncated "
                                f"(phase={phase!r}, finish_reason='length')"
                            )
                        raw_action = raw_action.strip().strip("- ")
                    else:
                        request_kwargs = {}
                        if "vote" in phase:
                            request_kwargs["extra_body"] = {
                                "chat_template_kwargs": {
                                    "enable_thinking": False,
                                }
                            }
                        raw_action = self._chat(
                            messages,
                            temperature=request_temperature,
                            max_tokens=request_max_tokens,
                            **request_kwargs,
                        ).strip().strip("- ")
                    if "vote" in phase:
                        parsed_vote_action = self.parse_vote_action(raw_action, observation, valid_action)
                        if parsed_vote_action is not None:
                            action = parsed_vote_action
                    else:
                        if is_constrained_night_action:
                            action, selected_env_action = (
                                self.parse_night_action_selection(
                                    raw_action,
                                    night_candidate_snapshot,
                                    phase=phase,
                                )
                            )
                        else:
                            action = self.match_authoritative_action_response(
                                raw_action,
                                valid_action,
                            )
                            if action is None:
                                raise GameplayActionValidationError(
                                    "invalid gameplay action response "
                                    f"(phase={phase!r}, response={raw_action!r}, "
                                    f"authoritative_candidates={valid_action!r})"
                                )
            else:
                if "vote" in phase:
                    action = self.choose_fallback_vote_action(observation, valid_action)
                else:
                    raise BackendError(
                        "Agent backend and model_name are required."
                    )
            env_action = (
                selected_env_action
                if selected_env_action is not None
                else self.nlp_action_to_env_action[action]
            )
            if raw_action is None:
                raw_action = action
            if self.has_log:
                self.logger.info(phase,
                                 extra={"prompt": prompt,
                                        "response": raw_action,
                                        "action": action,
                                        "player_id": observation['current_act_idx'],
                                        "role": observation['identity'],
                                        "phase": phase,
                                        "gen_times": retry_count - 1})
        return env_action

    def _generate_strict_public_speech(
        self,
        *,
        observation,
        planner_prompt,
        suggestible_player_ids,
        temperature,
        max_tokens,
    ):
        player_id = observation.get("current_act_idx")
        speaker_role = observation.get("identity")
        phase = observation.get("phase")
        game_context = getattr(
            getattr(self.backend, "session", None),
            "game_id",
            None,
        )
        plan_content, plan_metadata = self._chat_with_metadata(
            [{"role": "user", "content": planner_prompt}],
            player_log_context={
                "stage": "speech_plan",
                "observation": observation,
            },
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=(
                discourse_public_speech_plan_response_format(
                    supports_json_schema=getattr(
                        self.backend,
                        "supports_json_schema",
                        False,
                    ),
                    suggestible_player_ids=suggestible_player_ids,
                    speaker_id=player_id,
                    speaker_role=speaker_role,
                    public_event_indices=tuple(
                        event["event_idx"]
                        for event in observation["canonical_public_events"]
                    ),
                )
                if self.gameplay_prompt_profile
                == STRICT_CLASSIC7_DISCOURSE_GAMEPLAY_PROMPT_PROFILE
                else public_speech_plan_response_format(
                    supports_json_schema=getattr(
                        self.backend,
                        "supports_json_schema",
                        False,
                    ),
                    suggestible_player_ids=suggestible_player_ids,
                    speaker_id=player_id,
                    speaker_role=speaker_role,
                )
            ),
        )
        context = (
            f"game={game_context or 'unavailable'}, "
            f"player={player_id}, phase={phase}"
        )
        if plan_metadata["finish_reason"] == "length":
            raise PublicSpeechPlanValidationError(
                f"planner response was truncated ({context})"
            )
        try:
            plan_payload = json.loads(plan_content)
        except (TypeError, json.JSONDecodeError) as exc:
            raise PublicSpeechPlanValidationError(
                f"planner response is not valid JSON ({context})"
            ) from exc
        if self.gameplay_prompt_profile == (
            STRICT_CLASSIC7_DISCOURSE_GAMEPLAY_PROMPT_PROFILE
        ):
            public_events = normalize_public_events(
                observation["canonical_public_events"]
            )
            event_by_index = {
                event["event_idx"]: event for event in public_events
            }
            plan = validate_discourse_public_speech_plan(
                plan_payload,
                suggestible_player_ids=suggestible_player_ids,
                player_id=player_id,
                speaker_role=speaker_role,
                phase=phase,
                public_event_indices=tuple(event_by_index),
                game_context=game_context,
            )
            selected_public_evidence = [
                event_by_index[index]
                for index in plan.public_evidence_refs
            ]
            allowed_speech_player_ids = set(plan.targets) | set(
                public_evidence_player_ids(selected_public_evidence)
            )
            renderer_prompt = build_strict_classic7_discourse_speech_render_prompt(
                phase_text=_render_authoritative_public_phase(
                    observation["authoritative_public_state"]
                ),
                actor=player_id,
                public_actions=plan.actions_as_list(),
                selected_public_evidence=selected_public_evidence,
            )
            if self.has_log:
                self.logger.info(
                    "speech_discourse_plan_validated",
                    extra={
                        "game_id": game_context,
                        "player_id": player_id,
                        "phase": phase,
                        "public_actions": plan.actions_as_list(),
                        "public_evidence_refs": list(
                            plan.public_evidence_refs
                        ),
                        "selected_public_evidence": (
                            selected_public_evidence
                        ),
                    },
                )
        else:
            plan = validate_public_speech_plan(
                plan_payload,
                suggestible_player_ids=suggestible_player_ids,
                player_id=player_id,
                speaker_role=speaker_role,
                phase=phase,
                game_context=game_context,
            )
            renderer_prompt = build_strict_classic7_speech_render_prompt(
                phase_text=_render_authoritative_public_phase(
                    observation["authoritative_public_state"]
                ),
                actor=player_id,
                public_actions=plan.as_list(),
            )
            allowed_speech_player_ids = plan.targets
        rendered_content, render_metadata = self._chat_with_metadata(
            [{"role": "user", "content": renderer_prompt}],
            player_log_context={
                "stage": "speech_render",
                "observation": observation,
            },
            temperature=0.0,
            max_tokens=max_tokens,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False,
                }
            },
        )
        validate_gameplay_public_speech(
            rendered_content,
            finish_reason=render_metadata["finish_reason"],
            player_id=player_id,
            phase=phase,
            planned_player_ids=allowed_speech_player_ids,
        )
        final_speech = rendered_content.strip()
        return final_speech, final_speech, renderer_prompt

    def extract_answer(self, response):
        pattern = r'\n\n\"(.*?)\"'
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            response = matches[0]
        return response
