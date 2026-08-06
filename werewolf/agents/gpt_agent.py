import json
import time
import re
import random
from werewolf.agents.llm_agent import (
    LLMAgent,
    PublicSpeechPlanValidationError,
    public_speech_plan_response_format,
    validate_gameplay_public_speech,
    validate_public_speech_plan,
)
from werewolf.agents.prompt_template_v0 import (
    CON,
    STRICT_CLASSIC7_GAMEPLAY_PROMPT_PROFILE,
    build_strict_classic7_speech_render_prompt,
)
from . import agent_registry as AgentRegistry


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
        prompt = self.format_observation(observation)
        phase = observation['phase']
        valid_action = list(self.nlp_action_to_env_action.keys())  
        time.sleep(self.rate_limit)
        is_o1 = self.model_name is not None and "o1" in self.model_name
        request_temperature = None if is_o1 else self.temperature
        request_max_tokens = self.gameplay_max_tokens
        if request_max_tokens is None and is_o1:
            request_max_tokens = 32000
        if 'speech' in phase:
            is_strict_speech = self.gameplay_prompt_profile == (
                STRICT_CLASSIC7_GAMEPLAY_PROMPT_PROFILE
            )
            if is_strict_speech:
                raw_action, checked_action, prompt = (
                    self._generate_strict_public_speech(
                        observation=observation,
                        planner_prompt=prompt,
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
            if self.backend is not None and self.model_name:
                action = ''
                while action not in valid_action:
                    retry_count += 1
                    if retry_count > 3:
                        if "vote" in phase:
                            raw_action = self.choose_fallback_vote_action(observation, valid_action)
                        else:
                            raw_action = valid_action[random.randint(0, len(valid_action) - 1)]
                        action = raw_action
                        break
                    messages = [{'role': 'user', 'content': prompt}]
                    raw_action = self._chat(
                        messages,
                        temperature=request_temperature,
                        max_tokens=request_max_tokens,
                    ).strip().strip("- ")
                    if "vote" in phase:
                        parsed_vote_action = self.parse_vote_action(raw_action, observation, valid_action)
                        if parsed_vote_action is not None:
                            action = parsed_vote_action
                    else:
                        try:
                            assert raw_action in valid_action
                            action = raw_action
                        except:
                            action = valid_action[random.randint(0, len(valid_action) - 1)]
            else:
                if "vote" in phase:
                    action = self.choose_fallback_vote_action(observation, valid_action)
                else:
                    action = valid_action[random.randint(0, len(valid_action) - 1)]
                print("random choose a valid action, action: {} valid_action: {}".format(action, valid_action))
            env_action = self.nlp_action_to_env_action[action]
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
        temperature,
        max_tokens,
    ):
        player_id = observation.get("current_act_idx")
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
            response_format=public_speech_plan_response_format(
                supports_json_schema=getattr(
                    self.backend,
                    "supports_json_schema",
                    False,
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
        plan = validate_public_speech_plan(
            plan_payload,
            authoritative_public_state=observation.get(
                "authoritative_public_state"
            ),
            player_id=player_id,
            phase=phase,
            game_context=game_context,
        )
        renderer_prompt = build_strict_classic7_speech_render_prompt(
            authoritative_public_state=observation[
                "authoritative_public_state"
            ],
            actor=player_id,
            public_actions=plan.as_list(),
        )
        rendered_content, render_metadata = self._chat_with_metadata(
            [{"role": "user", "content": renderer_prompt}],
            player_log_context={
                "stage": "speech_render",
                "observation": observation,
            },
            temperature=temperature,
            max_tokens=max_tokens,
        )
        validate_gameplay_public_speech(
            rendered_content,
            finish_reason=render_metadata["finish_reason"],
            player_id=player_id,
            phase=phase,
            planned_player_ids=plan.targets,
        )
        final_speech = rendered_content.strip()
        return final_speech, final_speech, renderer_prompt

    def extract_answer(self, response):
        pattern = r'\n\n\"(.*?)\"'
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            response = matches[0]
        return response
