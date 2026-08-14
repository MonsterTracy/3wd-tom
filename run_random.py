"""Run one seven-player Werewolf rollout.

Legacy ToM samples are collected immediately before each public speech. Formal
ToM and Public Belief Matrix samples are collected after a speech completes and
before the next action. Collectors never receive role assignments.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import random
import time
from copy import deepcopy
from typing import Any

import yaml

from werewolf.agents import agent_registry
from werewolf.backends import (
    OpenAICompatibleBackend,
    create_backend,
    load_named_backends,
    resolve_backend,
)
from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
from werewolf.models import SpeechPerceiver
from werewolf.models.tom.collection import Collector as TomCollector
from werewolf.models.tom.reporter import (
    BeliefReporter,
    FORMAL_REPORTER_BASE_URL,
    FORMAL_REPORTER_MODEL,
)
from werewolf.models.public_belief_matrix.collection import (
    PUBLIC_BELIEF_MATRIX_COLLECTION_MODE,
    PUBLIC_BELIEF_MATRIX_SUPERVISION_BOUNDARY,
    PublicBeliefMatrixSampleCollector,
)
from werewolf.models.public_belief_matrix.reporter import PublicBeliefMatrixReporter
from werewolf.models.twd_tom.belief_snapshot import (
    PlayingAgentBeliefSnapshotCollector,
    PublicOnlyBeliefSnapshotCollector,
)
from werewolf.models.twd_tom.collector import (
    TWDToMSampleCollector,
)
from werewolf.models.twd_tom.samples import (
    PUBLIC_SPEECH_EVENTS,
    make_public_only_twd_tom_sample,
)
from werewolf.models.twd_tom.shadow import (
    SecondOrderToMShadow,
)
from werewolf.runtime_config import normalize_runtime_config
from werewolf.speech.private_belief_perceiver import (
    PlayingAgentBeliefReporter,
)
from werewolf.speech.public_belief_perceiver import PublicOnlyBeliefReporter


PRIVATE_CONDITIONED_COLLECTION_MODE = "private_conditioned"
PUBLIC_ONLY_COLLECTION_MODE = "public_only"
COLLECTION_MODES = (
    PRIVATE_CONDITIONED_COLLECTION_MODE,
    PUBLIC_ONLY_COLLECTION_MODE,
    PUBLIC_BELIEF_MATRIX_COLLECTION_MODE,
)


def _alive_observer_ids(env) -> list[int]:
    """Return public 1-based IDs for currently alive players."""

    alive = getattr(env, "alive", None)

    if not isinstance(alive, (list, tuple)):
        raise TypeError(
            "environment must provide an alive sequence"
        )

    observer_ids = [
        index + 1
        for index, is_alive in enumerate(alive)
        if is_alive == 1
    ]

    if not observer_ids:
        raise RuntimeError(
            "cannot collect a belief snapshot "
            "with no alive players"
        )

    return observer_ids


def eval(
    env,
    agent_list,
    roles_,
    sample_collector=None,
    call_audit=None,
    tom2_shadow=None,
    tom_collector=None,
):
    """Run one game and optionally collect subjective ToM samples.

    The environment needs the hidden role assignment to simulate the
    game, but that assignment is never passed to the ToM collector.

    Legacy ToM routes collect before each public ``speech`` or ``speech_pk``.
    The formal ToM and PBM routes collect after that speech completes and
    before the next action. Formal labels use each alive observer's legal
    post-speech observation.
    """

    for agent in agent_list:
        agent.reset()

    done = False
    obs = env.reset(
        roles=roles_,
    )
    step_idx = 0
    info = None

    while not done:
        current_act_idx = obs[
            "current_act_idx"
        ]
        action_phase = obs["phase"]
        trigger = getattr(env, "phase", None)
        action_round = getattr(env, "day", None)

        post_speech_collection = (
            sample_collector is not None
            and getattr(sample_collector, "collection_timing", None)
            == PUBLIC_BELIEF_MATRIX_SUPERVISION_BOUNDARY
        )
        if (
            trigger in PUBLIC_SPEECH_EVENTS
        ):
            if tom2_shadow is not None:
                tom2_shadow.record(
                    step_idx=step_idx,
                    phase=action_phase,
                    speaker_id=current_act_idx,
                    public_events=env.public_events,
                )
            if sample_collector is not None and not post_speech_collection:
                sample_collector.record(
                    env,
                    step_idx=step_idx,
                    trigger=trigger,
                    phase=action_phase,
                    speaker_id=current_act_idx,
                    observer_ids=(
                        _alive_observer_ids(env)
                    ),
                )

        audit_context = (
            call_audit.gameplay_context(
                acting_player_id=current_act_idx,
                observation=obs,
                public_events=env.public_events,
            )
            if call_audit is not None
            else nullcontext()
        )
        with audit_context:
            action = agent_list[
                current_act_idx - 1
            ].act(obs)

        obs, _, done, info = env.step(
            action
        )

        if trigger in PUBLIC_SPEECH_EVENTS and tom_collector is not None:
            tom_collector.record(
                env,
                step_idx=step_idx,
                round_number=action_round,
                phase=trigger,
                speaker_id=current_act_idx,
            )

        if trigger in PUBLIC_SPEECH_EVENTS and post_speech_collection:
            sample_collector.record(
                env,
                step_idx=step_idx,
                trigger=trigger,
                phase=action_phase,
                speaker_id=current_act_idx,
            )

        step_idx += 1

    if not isinstance(info, dict):
        raise RuntimeError(
            "finished game has no result information"
        )

    if info.get("Werewolf") == 1:
        return "Werewolf win"

    if info.get("Werewolf") == -1:
        return "Villager win"

    raise RuntimeError(
        "finished game has no recognized winner"
    )


def _weighted_profile_choice(profiles):
    if not profiles:
        raise ValueError(
            "no eligible agent profiles"
        )

    weights = [
        profile["sample_ratio"]
        for profile in profiles
    ]

    if (
        any(
            weight < 0
            for weight in weights
        )
        or not any(
            weight > 0
            for weight in weights
        )
    ):
        raise ValueError(
            "eligible agent profiles must have "
            "a positive sample_ratio"
        )

    return random.choices(
        profiles,
        weights=weights,
        k=1,
    )[0]


def assign_agents(
    candidate_profiles,
    env_config,
    log_save_path,
    assigined_roles,
    must_include,
    backends,
    allow_cross_team_profiles=False,
):
    """Assign configured agent profiles to the fixed role list.

    By default, a profile name remains exclusive to one faction. This
    preserves the random-competition behavior used to compare agent
    profiles across Werewolf and village teams.

    Collection-only configurations may explicitly set
    ``allow_cross_team_profiles=True`` so one API agent profile can
    control all seven players without duplicating equivalent profiles.
    """

    if not isinstance(
        allow_cross_team_profiles,
        bool,
    ):
        raise TypeError(
            "allow_cross_team_profiles must be boolean"
        )

    werewolf_team = {
        "Werewolf",
    }

    village_team = {
        "Villager",
        "Seer",
        "Witch",
        "Guard",
    }

    env_param = {
        "n_player": env_config[
            "n_player"
        ],
        "n_role": env_config[
            "n_role"
        ],
    }

    all_agent_profiles = {}
    profile_backend_ids = {}
    role2agent_list = []

    village_profiles = set()
    werewolf_profiles = set()

    forced_profile = None

    if must_include:
        required_profiles = [
            profile
            for profile in candidate_profiles
            if profile["profile_name"]
            in must_include
        ]

        forced_profile = (
            _weighted_profile_choice(
                required_profiles
            )
        )

    for index, role in enumerate(
        assigined_roles
    ):
        if (
            index == 0
            and forced_profile is not None
        ):
            profile = forced_profile
        else:
            if (
                role not in werewolf_team
                and role not in village_team
            ):
                raise ValueError(
                    f"unsupported role: {role}"
                )

            if allow_cross_team_profiles:
                eligible_profiles = list(
                    candidate_profiles
                )
            else:
                if role in werewolf_team:
                    opposite_profiles = (
                        village_profiles
                    )
                else:
                    opposite_profiles = (
                        werewolf_profiles
                    )

                eligible_profiles = [
                    profile
                    for profile
                    in candidate_profiles
                    if profile["profile_name"]
                    not in opposite_profiles
                ]

            if not eligible_profiles:
                raise ValueError(
                    "no eligible agent profiles "
                    f"for role: {role}"
                )

            profile = (
                _weighted_profile_choice(
                    eligible_profiles
                )
            )

        profile_name = profile[
            "profile_name"
        ]
        profile_backend_ids[profile_name] = profile["backend"]

        if role in werewolf_team:
            werewolf_profiles.add(
                profile_name
            )
        else:
            village_profiles.add(
                profile_name
            )

        if (
            profile_name
            not in all_agent_profiles
        ):
            model_params = dict(
                profile["model_params"]
            )

            all_agent_profiles[
                profile_name
            ] = agent_registry.build(
                profile["agent_type"],
                backend=resolve_backend(
                    profile["backend"],
                    backends,
                ),
                model_name=profile[
                    "model"
                ],
                **model_params,
            )

        role2agent_list.append(
            profile_name
        )

    agent_list = []

    for index, _role in enumerate(
        assigined_roles
    ):
        log_file = (
            os.path.join(
                log_save_path,
                f"Player_{index + 1}.jsonl",
            )
            if log_save_path is not None
            else None
        )

        profile_name = (
            role2agent_list[index]
        )

        agent_type, agent_param = (
            all_agent_profiles[
                profile_name
            ]
        )

        agent = (
            agent_registry.build_agent(
                agent_type,
                index,
                agent_param,
                env_param,
                log_file,
            )
        )
        agent.backend_id = profile_backend_ids[profile_name]

        agent_list.append(
            agent
        )

    return (
        role2agent_list,
        agent_list,
    )


def _resolve_backend_map(
    normalized,
    *,
    backend=None,
    backend_settings=None,
    backends=None,
):
    """Resolve injected, legacy, or configured backend instances."""

    if backends is not None:
        return dict(
            backends
        )

    backend_names = list(
        normalized["backends"]
    )

    if backend is not None:
        if len(backend_names) != 1:
            raise ValueError(
                "a single injected backend "
                "requires exactly one "
                "configured backend"
            )

        return {
            backend_names[0]: backend,
        }

    if backend_settings is not None:
        if len(backend_names) != 1:
            raise ValueError(
                "legacy backend_settings "
                "requires exactly one "
                "configured backend"
            )

        return {
            backend_names[0]: (
                create_backend(
                    backend_settings
                )
            ),
        }

    return load_named_backends(
        normalized,
        env_file=".env",
    )


def build_runtime(
    parsed_yaml,
    log_save_path,
    backend=None,
    backend_settings=None,
    roles=None,
    random_seed=None,
    backends=None,
):
    """Build the game environment and action agents."""

    config_for_normalization = (
        deepcopy(parsed_yaml)
    )

    if (
        backend_settings is not None
        and "backend"
        in config_for_normalization
        and "backends"
        not in config_for_normalization
    ):
        legacy_backend = (
            config_for_normalization[
                "backend"
            ]
        )

        for field in (
            "default_model",
            "agent_model",
            "parser_model",
        ):
            value = getattr(
                backend_settings,
                field,
                None,
            )

            if value is not None:
                legacy_backend.setdefault(
                    field,
                    value,
                )

    normalized = (
        normalize_runtime_config(
            config_for_normalization
        )
    )

    agent_config = normalized[
        "agent_config"
    ]

    all_candidate_agents = (
        agent_config[
            "all_candidates"
        ]
    )

    env_config = dict(
        normalized["env_config"]
    )

    backend_map = (
        _resolve_backend_map(
            normalized,
            backend=backend,
            backend_settings=(
                backend_settings
            ),
            backends=backends,
        )
    )

    parser_config = normalized[
        "parser"
    ]

    speech_perceiver = (
        SpeechPerceiver(
            backend=resolve_backend(
                parser_config["backend"],
                backend_map,
            ),
            model_name=parser_config[
                "model"
            ],
        )
    )

    env_config[
        "log_save_path"
    ] = log_save_path

    env = WerewolfTextEnvV0(
        **env_config,
        speech_perceiver=(
            speech_perceiver
        ),
    )

    if env_config.get(
        "n_hunter",
        0,
    ) != 0:
        raise ValueError(
            "TWDM 7-player environment "
            "does not support Hunter."
        )

    if random_seed is not None:
        random.seed(
            random_seed
        )

    if roles is None:
        roles = (
            ["Werewolf"]
            * env_config["n_werewolf"]
            + ["Villager"]
            * env_config["n_villager"]
            + ["Seer"]
            * env_config["n_seer"]
            + ["Witch"]
            * env_config["n_witch"]
            + ["Guard"]
            * env_config["n_guard"]
        )

        random.shuffle(
            roles
        )
    else:
        roles = list(
            roles
        )

    must_include = agent_config.get(
        "must_include",
        [],
    )

    allow_cross_team_profiles = (
        agent_config.get(
            "allow_cross_team_profiles",
            False,
        )
    )

    if not isinstance(
        allow_cross_team_profiles,
        bool,
    ):
        raise TypeError(
            "agent_config.allow_cross_team_profiles "
            "must be boolean"
        )

    role2agent_list, agent_list = (
        assign_agents(
            all_candidate_agents,
            env_config,
            log_save_path,
            roles,
            must_include=(
                must_include
            ),
            backends=backend_map,
            allow_cross_team_profiles=(
                allow_cross_team_profiles
            ),
        )
    )

    return (
        env,
        agent_list,
        roles,
        role2agent_list,
    )


def build_twd_tom_sample_collector(
    *,
    agent_list,
    output_path,
    game_id,
    report_audit=None,
    collection_mode=PRIVATE_CONDITIONED_COLLECTION_MODE,
    reporter_dispatch=None,
):
    """Build the selected readonly belief collection stack."""

    if collection_mode == PRIVATE_CONDITIONED_COLLECTION_MODE:
        snapshot_collector = (
            PlayingAgentBeliefSnapshotCollector(
                PlayingAgentBeliefReporter(
                    audit_hook=report_audit,
                ),
                agent_list,
            )
        )
        sample_builder = None
    elif collection_mode == PUBLIC_ONLY_COLLECTION_MODE:
        dispatches = []
        for agent in agent_list:
            backend_id = getattr(agent, "backend_id", None)
            model_name = getattr(agent, "model_name", None)
            backend = getattr(agent, "backend", None)
            if not isinstance(backend_id, str) or not backend_id.strip():
                raise ValueError("every public-only dispatch requires backend_id")
            if not isinstance(model_name, str) or not model_name.strip():
                raise ValueError("every public-only dispatch requires model_name")
            if backend is None or not hasattr(backend, "chat"):
                raise TypeError("every public-only dispatch requires backend.chat()")
            dispatches.append(
                {
                    "backend": backend,
                    "backend_id": backend_id,
                    "model_name": model_name,
                }
            )
        snapshot_collector = PublicOnlyBeliefSnapshotCollector(
            PublicOnlyBeliefReporter(audit_hook=report_audit),
            dispatches,
        )
        sample_builder = make_public_only_twd_tom_sample
    elif collection_mode == PUBLIC_BELIEF_MATRIX_COLLECTION_MODE:
        if not isinstance(reporter_dispatch, dict):
            raise TypeError("PBM collection requires one shared reporter_dispatch")
        return PublicBeliefMatrixSampleCollector(
            output_path=output_path,
            game_id=game_id,
            reporter=PublicBeliefMatrixReporter(audit_hook=report_audit),
            reporter_dispatch=reporter_dispatch,
        )
    else:
        raise ValueError(f"collection_mode must be one of {COLLECTION_MODES}")

    arguments = {
        "output_path": output_path,
        "snapshot_collector": snapshot_collector,
        "game_id": game_id,
    }
    if sample_builder is not None:
        arguments["sample_builder"] = sample_builder
    return TWDToMSampleCollector(
        **arguments,
    )


def build_tom_collector(
    *,
    env,
    reporter_backend,
    output_path,
    game_id,
    seed,
):
    """Build the single formal post-speech collection path."""

    if env.n_guard == 1 and env.n_witch == 0:
        episode_context = "seer_guard"
    elif env.n_guard == 0 and env.n_witch == 1:
        episode_context = "seer_witch"
    else:
        raise ValueError("formal ToM requires seer_guard or seer_witch")
    return TomCollector(
        output_path,
        game_id=game_id,
        seed=seed,
        episode_context=episode_context,
        reporter=BeliefReporter(reporter_backend),
    )


def _build_formal_tom_reporter_backend(tom_sample_path):
    if tom_sample_path is None:
        return None
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError(
            "DEEPSEEK_API_KEY is required for --tom_sample_path"
        )
    return OpenAICompatibleBackend(
        api_key=api_key.strip(),
        base_url=FORMAL_REPORTER_BASE_URL,
        default_model=FORMAL_REPORTER_MODEL,
        max_retries=0,
        supports_json_schema=False,
    )


def _write_role_assignment(
    *,
    log_save_path,
    roles,
    role2agent_list,
) -> None:
    """Write game-audit roles outside the subjective ToM JSONL."""

    records = [
        {
            "id": index + 1,
            "role": role,
            "model": (
                role2agent_list[index]
            ),
        }
        for index, role in enumerate(
            roles
        )
    ]

    output_file = os.path.join(
        log_save_path,
        "roles_model_assignment.json",
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            records,
            json_file,
            ensure_ascii=False,
            indent=4,
        )


def main_cli(args):
    """CLI implementation."""

    tom_sample_path = getattr(args, "tom_sample_path", None)
    formal_reporter_backend = _build_formal_tom_reporter_backend(
        tom_sample_path
    )

    if args.log_save_path is None:
        run_name = time.strftime(
            "%Y%m%d_%H%M%S"
        )

        args.log_save_path = os.path.join(
            "logs",
            run_name,
        )

    os.makedirs(
        args.log_save_path,
        exist_ok=True,
    )

    with open(
        args.config,
        "r",
        encoding="utf-8",
    ) as config_file:
        parsed_yaml = yaml.safe_load(
            config_file
        )

    if not isinstance(
        parsed_yaml,
        dict,
    ):
        raise ValueError(
            "runtime config must be a mapping"
        )

    config_save_path = os.path.join(
        args.log_save_path,
        "config.yaml",
    )

    with open(
        config_save_path,
        "w",
        encoding="utf-8",
    ) as config_file:
        yaml.dump(
            parsed_yaml,
            config_file,
            allow_unicode=True,
            sort_keys=False,
        )

    normalized = (
        normalize_runtime_config(
            deepcopy(parsed_yaml)
        )
    )

    backend_map = load_named_backends(
        normalized,
        env_file=".env",
    )

    (
        env,
        agent_list,
        roles,
        role2agent_list,
    ) = build_runtime(
        parsed_yaml,
        log_save_path=(
            args.log_save_path
        ),
        random_seed=getattr(
            args,
            "random_seed",
            None,
        ),
        backends=backend_map,
    )

    print(
        "New rollout: ",
        roles,
    )
    print()

    for role, profile_name in zip(
        roles,
        role2agent_list,
    ):
        print(
            role,
            "\t",
            profile_name,
        )

    if len(roles) != len(
        role2agent_list
    ):
        raise RuntimeError(
            "roles and role2agent_list "
            "must have equal lengths"
        )

    _write_role_assignment(
        log_save_path=(
            args.log_save_path
        ),
        roles=roles,
        role2agent_list=(
            role2agent_list
        ),
    )

    begin = time.time()
    sample_collector = None
    tom_collector = None
    tom2_shadow = None

    sample_path = getattr(
        args,
        "twd_tom_sample_path",
        None,
    )

    game_id = os.path.basename(
        os.path.normpath(
            args.log_save_path
        )
    )
    shadow_options = _resolve_tom2_shadow_options(args)

    try:
        if sample_path is not None:
            sample_collector = (
                build_twd_tom_sample_collector(
                    agent_list=agent_list,
                    output_path=sample_path,
                    game_id=game_id,
                )
            )
        if tom_sample_path is not None:
            tom_collector = build_tom_collector(
                env=env,
                reporter_backend=formal_reporter_backend,
                output_path=tom_sample_path,
                game_id=game_id,
                seed=getattr(args, "random_seed", None),
            )
        if shadow_options is not None:
            checkpoint_path, device, output_path = shadow_options
            tom2_shadow = SecondOrderToMShadow(
                checkpoint_path=checkpoint_path,
                device=device,
                output_path=output_path,
                game_id=game_id,
            )
        result = eval(
            env,
            agent_list,
            roles,
            sample_collector=(
                sample_collector
            ),
            tom2_shadow=tom2_shadow,
            tom_collector=tom_collector,
        )
    finally:
        if sample_collector is not None:
            sample_collector.close()
        if tom_collector is not None:
            tom_collector.close()
        if tom2_shadow is not None:
            tom2_shadow.close()

    print(
        time.time() - begin,
        result,
    )

    return result


def build_arg_parser() -> (
    argparse.ArgumentParser
):
    """Build the command-line parser."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default=(
            "configs/random_models.yaml"
        ),
        help=(
            "path to the game runtime config"
        ),
    )

    parser.add_argument(
        "--log_save_path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--tom_sample_path",
        type=str,
        default=None,
        help="optional JSONL path for formal post-speech ToM samples",
    )

    parser.add_argument(
        "--twd_tom_sample_path",
        type=str,
        default=None,
        help=(
            "optional JSONL path for "
            "playing-agent ToM samples"
        ),
    )

    parser.add_argument(
        "--twd_tom2_shadow_checkpoint",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--twd_tom2_shadow_device",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--twd_tom2_shadow_output_path",
        type=str,
        default=None,
    )

    return parser


def _resolve_tom2_shadow_options(args):
    values = (
        getattr(args, "twd_tom2_shadow_checkpoint", None),
        getattr(args, "twd_tom2_shadow_device", None),
        getattr(args, "twd_tom2_shadow_output_path", None),
    )
    if values == (None, None, None):
        return None
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(
            "second-order shadow checkpoint, device, and output path "
            "must be provided together"
        )
    if values[1] == "auto":
        raise ValueError("second-order shadow device must be explicit")
    return values


if __name__ == "__main__":
    arguments = (
        build_arg_parser().parse_args()
    )

    main_cli(
        arguments
    )
