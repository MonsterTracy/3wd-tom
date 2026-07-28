from typing import Dict
from pydantic import BaseModel


class Registry(BaseModel):
    """Registry for storing and building classes."""

    name: str
    entries: Dict = {}
    translator_entries: Dict = {} 

    def register(self, keys: list):
        def decorator(cls):
            for key in keys:
                if key in self.entries:
                    raise ValueError(f"Key {key} is already registered with a different class.")
                self.entries[key] = cls
            return cls
        return decorator


    def build(self, type: str, backend=None, default_model=None, **kwargs):
        if type not in self.entries:
            raise ValueError(
                f'{type} is not registered. Please register with the .register("{type}") method provided in {self.name} registry'
            )

        model_name = (
            kwargs.get("model_name")
            or kwargs.get("llm")
            or default_model
        )
        if not model_name:
            raise ValueError(f"model_name is required for agent type: {type}")

        agent_params = {
            "backend": backend,
            "model_name": model_name,
            "tokenizer": kwargs.get("tokenizer"),
            "temperature": kwargs.get("temperature", 1.0),
        }
        if type.lower() == "twdm_agent":
            agent_params["twdm_config"] = kwargs.get("twdm_config", {})

        return type, agent_params

    def build_agent(self, type: str,
                    player_idx,
                    agent_param,
                    env_param,
                    log_file):
        
        if type not in self.entries:
            raise ValueError(
                f'{type} is not registered. Please register with the .register("{type}") method provided in {self.name} registry'
            )
        if type == "twdm_agent":
            return self.entries[type](backend=agent_param["backend"],
                                      model_name=agent_param["model_name"],
                                      tokenizer=agent_param.get("tokenizer"),
                                      temperature=agent_param["temperature"],
                                      log_file=log_file,
                                      twdm_config=agent_param.get("twdm_config", {}))

        return self.entries[type](backend=agent_param["backend"],
                                  model_name=agent_param["model_name"],
                                  tokenizer=agent_param.get("tokenizer"),
                                  temperature=agent_param["temperature"],
                                  log_file=log_file)

    def get_all_entries(self):
        return self.entries
