"""Tests for random-runtime agent profile assignment."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import run_random


SHARED_PROFILE = {
    "profile_name": "shared_deepseek",
    "agent_type": "deepseek",
    "backend": "deepseek",
    "model": "test-model",
    "model_params": {
        "temperature": 0.0,
    },
    "sample_ratio": 1.0,
}


FULL_ROLES = [
    "Werewolf",
    "Villager",
    "Seer",
    "Werewolf",
    "Witch",
    "Villager",
    "Villager",
]


ENV_CONFIG = {
    "n_player": 7,
    "n_role": 4,
    "n_werewolf": 2,
    "n_villager": 3,
    "n_seer": 1,
    "n_witch": 1,
    "n_guard": 0,
    "n_hunter": 0,
}


def patch_agent_registry(
    monkeypatch,
):
    """Replace the registry reference with deterministic test doubles."""

    def fake_resolve_backend(
        backend_name,
        backends,
    ):
        return backends[
            backend_name
        ]

    def fake_build(
        agent_type,
        **kwargs,
    ):
        return (
            "fake_agent_type",
            {
                "requested_agent_type": (
                    agent_type
                ),
                "backend": kwargs[
                    "backend"
                ],
                "model_name": kwargs[
                    "model_name"
                ],
                "temperature": kwargs.get(
                    "temperature",
                    1.0,
                ),
            },
        )

    def fake_build_agent(
        agent_type,
        index,
        agent_param,
        env_param,
        log_file,
    ):
        class FakeAgent(dict):
            pass

        return FakeAgent({
            "agent_type": agent_type,
            "index": index,
            "agent_param": agent_param,
            "env_param": env_param,
            "log_file": log_file,
        })

    fake_registry = SimpleNamespace(
        build=fake_build,
        build_agent=(
            fake_build_agent
        ),
    )

    monkeypatch.setattr(
        run_random,
        "resolve_backend",
        fake_resolve_backend,
    )

    monkeypatch.setattr(
        run_random,
        "agent_registry",
        fake_registry,
    )


def test_shared_profile_can_control_both_teams(
    monkeypatch,
):
    patch_agent_registry(
        monkeypatch
    )

    (
        role_to_profile,
        agents,
    ) = run_random.assign_agents(
        candidate_profiles=[
            SHARED_PROFILE,
        ],
        env_config=ENV_CONFIG,
        log_save_path=None,
        assigined_roles=FULL_ROLES,
        must_include=[],
        backends={
            "deepseek": object(),
        },
        allow_cross_team_profiles=True,
    )

    assert role_to_profile == [
        "shared_deepseek",
    ] * 7

    assert len(agents) == 7
    assert {agent.backend_id for agent in agents} == {"deepseek"}

    assert [
        agent["index"]
        for agent in agents
    ] == list(range(7))

    assert {
        agent[
            "agent_param"
        ][
            "requested_agent_type"
        ]
        for agent in agents
    } == {
        "deepseek",
    }

    assert {
        agent[
            "agent_param"
        ][
            "model_name"
        ]
        for agent in agents
    } == {
        "test-model",
    }


def test_default_mode_preserves_team_exclusivity(
    monkeypatch,
):
    patch_agent_registry(
        monkeypatch
    )

    with pytest.raises(
        ValueError,
        match=(
            "no eligible agent profiles "
            "for role: Villager"
        ),
    ):
        run_random.assign_agents(
            candidate_profiles=[
                SHARED_PROFILE,
            ],
            env_config=ENV_CONFIG,
            log_save_path=None,
            assigined_roles=[
                "Werewolf",
                "Villager",
            ],
            must_include=[],
            backends={
                "deepseek": object(),
            },
        )


def test_invalid_cross_team_setting_is_rejected(
    monkeypatch,
):
    patch_agent_registry(
        monkeypatch
    )

    with pytest.raises(
        TypeError,
        match=(
            "allow_cross_team_profiles "
            "must be boolean"
        ),
    ):
        run_random.assign_agents(
            candidate_profiles=[
                SHARED_PROFILE,
            ],
            env_config=ENV_CONFIG,
            log_save_path=None,
            assigined_roles=[
                "Werewolf",
            ],
            must_include=[],
            backends={
                "deepseek": object(),
            },
            allow_cross_team_profiles=1,
        )
