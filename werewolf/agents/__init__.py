from werewolf.registry import Registry

agent_registry = Registry(name="agent")

from werewolf.agents.gpt_agent import GPTAgent
from werewolf.agents.twdm_agent import TWDMStrategyAgent

__all__ = [
    "agent_registry",
    "GPTAgent",
    "TWDMStrategyAgent",
]
