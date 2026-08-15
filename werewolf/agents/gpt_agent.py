import time
from werewolf.agents.llm_agent import (
    GameplayActionValidationError,
    LLMAgent,
    night_action_response_format,
    validate_gameplay_public_speech,
)
from werewolf.backends import BackendError
from . import agent_registry as AgentRegistry


_CONSTRAINED_NIGHT_PHASES = (
    "skill_wolf",
    "skill_seer",
    "skill_guard",
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
                 gameplay_max_tokens=None):
        super().__init__(backend=backend, model_name=model_name, tokenizer=tokenizer,
                         temperature=temperature, log_file=log_file,
                         gameplay_max_tokens=gameplay_max_tokens)
        self.rate_limit = 6
        self.temperature = temperature

    def act(self, observation):
        phase = observation['phase']
        is_constrained_night_action = any(
            night_phase in phase
            for night_phase in _CONSTRAINED_NIGHT_PHASES
        )
        is_vote = "vote" in phase
        night_candidate_snapshot = None
        if is_constrained_night_action:
            night_candidate_snapshot = (
                self.freeze_authoritative_action_candidates(
                    observation["valid_action"]
                )
            )
        prompt = None
        valid_action = []
        if not is_vote:
            prompt = self.format_observation(
                observation,
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
            messages = [{'role': 'user', 'content': prompt}]
            raw_action, metadata = self._chat_with_metadata(
                messages,
                player_log_context={
                    "stage": "public_speech",
                    "observation": observation,
                },
                temperature=request_temperature,
                max_tokens=request_max_tokens,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": False,
                    }
                },
            )
            validate_gameplay_public_speech(
                raw_action,
                finish_reason=metadata["finish_reason"],
                player_id=observation.get("current_act_idx"),
                phase=phase,
            )
            checked_action = raw_action
            env_action = ('speech', checked_action)
        elif is_vote:
            env_action, generation_info = self.generate_indexed_vote_action(
                observation,
                temperature=request_temperature,
                max_tokens=request_max_tokens,
            )
            if self.has_log:
                self.logger.info(
                    phase,
                    extra={
                        "prompt": generation_info["prompt"],
                        "response": generation_info["response"],
                        "action": generation_info["action"],
                        "player_id": observation['current_act_idx'],
                        "role": observation['identity'],
                        "phase": phase,
                        "gen_times": 0,
                    },
                )
        else: 
            retry_count = 0
            raw_action = None
            selected_env_action = None
            if self.backend is not None and self.model_name:
                action = ''
                while action not in valid_action:
                    retry_count += 1
                    if retry_count > 3:
                        raise GameplayActionValidationError(
                            "night action response was not resolved"
                        )
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
                            extra_body={
                                "chat_template_kwargs": {
                                    "enable_thinking": False,
                                }
                            },
                        )
                        if metadata["finish_reason"] == "length":
                            raise GameplayActionValidationError(
                                "night action response was truncated "
                                f"(phase={phase!r}, finish_reason='length')"
                            )
                        raw_action = raw_action.strip().strip("- ")
                    else:
                        raw_action = self._chat(
                            messages,
                            temperature=request_temperature,
                            max_tokens=request_max_tokens,
                        ).strip().strip("- ")
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
