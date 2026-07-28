import time
import re
import random
from werewolf.agents.llm_agent import LLMAgent
from werewolf.agents.prompt_template_v0 import CON
from . import agent_registry as AgentRegistry


@AgentRegistry.register(["gpt", "gpt-4", "GPT-4", "gpt4", "o1", "gpt4o", "gpt4o-mini", 'deepseek'])
class GPTAgent(LLMAgent):
    def __init__(self,
                 backend=None,
                 model_name=None,
                 tokenizer=None,
                 temperature=1.0,
                 log_file=None):
        super().__init__(backend=backend, model_name=model_name, tokenizer=tokenizer,
                         temperature=temperature, log_file=log_file)
        self.rate_limit = 6
        self.temperature = temperature

    def act(self, observation):
        prompt = self.format_observation(observation)
        phase = observation['phase']
        valid_action = list(self.nlp_action_to_env_action.keys())  
        time.sleep(self.rate_limit)
        if 'speech' in phase:
            if self.backend is not None and self.model_name:
                messages = [{'role': 'user', 'content': prompt}]
                if "o1" in self.model_name:
                    raw_action = self._chat(
                        messages, temperature=None, max_tokens=32000
                    ).strip()
                else:
                    raw_action = self._chat(
                        messages, temperature=self.temperature
                    ).strip()
                checked_action = self.extract_answer(raw_action)
                gen_times = 0
            else:
                raw_action = "aaa"
                gen_times = -1
                checked_action = 'bbb'
            env_action = ('speech', checked_action)

            if self.has_log:
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
                    if "o1" in self.model_name:
                        raw_action = self._chat(
                            messages, temperature=None, max_tokens=32000
                        ).strip().strip("- ")
                    else:
                        raw_action = self._chat(
                            messages, temperature=self.temperature
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

    def extract_answer(self, response):
        pattern = r'\n\n\"(.*?)\"'
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            response = matches[0]
        return response
