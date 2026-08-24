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


class BeliefSnapshotCollectionError(RuntimeError):
    """One observer row exhausted the readonly label protocol."""

    def __init__(
        self,
        *,
        observer_id: str,
        status: str,
        error: str,
        generation_attempt_count: int,
    ) -> None:
        self.observer_id = observer_id
        self.status = status
        self.error = error
        self.generation_attempt_count = generation_attempt_count
        super().__init__(
            "readonly belief report failed: "
            f"observer={observer_id} status={status!r} error={error!r} "
            f"generation_attempt_count={generation_attempt_count}"
        )


def _snapshot_agent_state(agent) -> dict[str, Any]:
    """Copy agent-owned state while excluding external service handles."""

    return {
        field_name: deepcopy(value)
        for field_name, value in vars(agent).items()
        if field_name not in _EXTERNAL_AGENT_FIELDS
    }


class PlayingAgentBeliefSnapshotCollector:
    """Call each alive playing agent once through a detached context."""

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
            observer = normalize_player(player_id)
            agent = self.agents[player_id - 1]
            backend_id = getattr(agent, "backend_id", None)
            if not isinstance(backend_id, str) or not backend_id.strip():
                raise ValueError(f"{observer} has no agent_backend_id")

            observation = env.get_observation_for(player_id)
            known_werewolves, known_non_werewolves = (
                env.get_twd_tom_hard_knowledge_for(player_id)
            )
            state_before = _snapshot_agent_state(agent)
            result = self.reporter.report(
                agent=agent,
                observation=observation,
                observer_id=observer,
                public_snapshot=public_snapshot,
                agent_backend_id=backend_id,
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
                    observer_id=observer,
                    state_before=state_before,
                    state_after=state_after,
                )
            if state_after != state_before:
                raise RuntimeError(
                    f"readonly belief report mutated {observer} agent state"
                )
            if not isinstance(result, dict):
                raise TypeError("reporter result must be a dictionary")
            if result.get("observer") != observer:
                raise ValueError("reporter returned an unexpected observer")
            if result.get("status") != "ok":
                raise BeliefSnapshotCollectionError(
                    observer_id=observer,
                    status=result.get("status"),
                    error=result.get("error"),
                    generation_attempt_count=result.get(
                        "generation_attempt_count",
                        0,
                    ),
                )
            reports[observer] = result

        return reports


__all__ = [
    "BeliefSnapshotCollectionError",
    "PlayingAgentBeliefSnapshotCollector",
]
