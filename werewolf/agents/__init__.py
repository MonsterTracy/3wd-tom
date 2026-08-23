from werewolf.registry import Registry

agent_registry = Registry(name="agent")

from werewolf.agents.gpt_agent import GPTAgent

__all__ = [
    "agent_registry",
    "GPTAgent",
]
