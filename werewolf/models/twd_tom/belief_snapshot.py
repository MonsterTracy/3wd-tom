"""Collect readonly playing-agent reports on one frozen public history."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from werewolf.models.twd_tom.samples import PublicSnapshot
from werewolf.models.twd_tom.schema import normalize_player


_EXTERNAL_AGENT_FIELDS = {
    "backend",
    "handler",
    "logger",
    "tokenizer",
    "strategy",
    "matcher",
}


def _snapshot_agent_state(agent) -> dict[str, Any]:
    """Copy agent-owned state while excluding external service handles."""

    return {
        field_name: deepcopy(value)
        for field_name, value in vars(agent).items()
        if field_name not in _EXTERNAL_AGENT_FIELDS
    }


class PlayingAgentBeliefSnapshotCollector:
    """Collect one direct pair-belief self-report from each alive player."""

    def __init__(self, reporter, agents) -> None:
        if reporter is None or not hasattr(reporter, "report"):
            raise TypeError("reporter must provide report()")
        if not isinstance(agents, (list, tuple)) or len(agents) != 7:
            raise ValueError("agents must contain exactly seven playing agents")
        self.reporter = reporter
        self.agents = tuple(agents)

    def collect(self, public_snapshot: PublicSnapshot, *, env) -> dict[str, Any]:
        if not isinstance(public_snapshot, PublicSnapshot):
            raise TypeError("public_snapshot must be a PublicSnapshot")
        if not hasattr(env, "get_observation_for"):
            raise TypeError("environment must provide get_observation_for()")
        if not hasattr(env, "get_twd_tom_hard_knowledge_for"):
            raise TypeError(
                "environment must provide get_twd_tom_hard_knowledge_for()"
            )

        reports: dict[str, dict[str, Any]] = {}
        for player_id in public_snapshot.observer_ids:
            belief_owner = normalize_player(player_id)
            agent = self.agents[player_id - 1]
            backend_id = getattr(agent, "backend_id", None)
            if not isinstance(backend_id, str) or not backend_id.strip():
                raise ValueError(f"{belief_owner} has no backend_alias")

            observation = env.get_observation_for(player_id)
            known_werewolves, known_non_werewolves = (
                env.get_twd_tom_hard_knowledge_for(player_id)
            )
            state_before = _snapshot_agent_state(agent)
            result = self.reporter.report(
                agent=agent,
                observation=observation,
                belief_owner_id=belief_owner,
                public_snapshot=public_snapshot,
                backend_alias=backend_id,
                known_werewolves=known_werewolves,
                known_non_werewolves=known_non_werewolves,
            )
            state_after = _snapshot_agent_state(agent)
            record_agent_state = getattr(
                self.reporter,
                "record_agent_state",
                None,
            )
            if record_agent_state is not None:
                record_agent_state(
                    observer_id=belief_owner,
                    state_before=state_before,
                    state_after=state_after,
                )
            if state_after != state_before:
                raise RuntimeError(
                    f"readonly belief report mutated {belief_owner} agent state"
                )
            if not isinstance(result, dict):
                raise TypeError("reporter result must be a dictionary")
            if result.get("player_id") != belief_owner:
                raise ValueError("reporter returned an unexpected belief_owner")
            reports[belief_owner] = result

        return reports


__all__ = ["PlayingAgentBeliefSnapshotCollector"]
