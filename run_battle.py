import random
from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
import time
import argparse
import os
from copy import deepcopy
from werewolf.agents import agent_registry
from werewolf.backends import (
    create_backend,
    load_named_backends,
    resolve_backend,
)
from werewolf.models import SpeechPerceiver
from werewolf.models.twd_tom.collector import TWDToMSampleCollector
from werewolf.runtime_config import normalize_runtime_config
import yaml

def eval(env, agent_list, roles_, sample_collector=None):
    print(agent_list)
    for agent in agent_list:
        agent.reset()
    done = False
    obs = env.reset(roles=roles_)
    step_idx = 0
    while not done:
        if sample_collector is not None:
            sample_collector.record(
                obs,
                roles_,
                step_idx=step_idx,
            )
        current_act_idx = obs['current_act_idx']
        action = agent_list[current_act_idx - 1].act(obs)
        obs, reward, done, info = env.step(action)
        step_idx += 1
    if done:
        if info['Werewolf'] == 1:
            return 'Werewolf win'
        elif info['Werewolf'] == -1:
            return 'Villager win'

def get_replaced_wolf_id(replace_players, assgined_roles):
    replace_type = replace_players.split("_")[1]
    if replace_type == "last":
        reversed_lst = assgined_roles[::-1]
        index_in_reversed = reversed_lst.index("Werewolf")
        replace_id = len(assgined_roles) - 1 - index_in_reversed
    elif replace_type == "random":
        indexes = [i for i, x in enumerate(assgined_roles) if x == "Werewolf"]
        replace_id = random.choice(indexes)
    else:
        raise NotImplementedError
    return replace_id


def get_replaced_ordinary_villager_ids(assgined_roles, replace_number):
    # ordinary_villager means only the Villager role.
    indexes = [i for i, x in enumerate(assgined_roles) if x == "Villager"]
    replace_ids = random.sample(indexes, replace_number)
    return replace_ids


def get_replaced_village_team_ids(assgined_roles, replace_number):
    # village_team means non-werewolf roles: Villager, Seer, Witch, Guard.
    village_team_roles = ("Villager", "Seer", "Witch", "Guard")
    indexes = [i for i, x in enumerate(assgined_roles) if x in village_team_roles]
    replace_ids = random.sample(indexes, replace_number)
    return replace_ids


def parse_replace_players(replace_players):
    if replace_players.startswith("ordinary_villager_"):
        return "ordinary_villager", replace_players.split("ordinary_villager_", 1)[1]
    if replace_players.startswith("village_team_"):
        return "village_team", replace_players.split("village_team_", 1)[1]

    replace_role, replace_arg = replace_players.split("_", 1)
    if replace_role not in ("werewolf", "seer", "guard", "witch", "gods"):
        raise ValueError(
            f"Unsupported replace_role '{replace_role}'. Use 'ordinary_villager' or 'village_team'."
        )
    return replace_role, replace_arg


def assign_agents_and_roles(assgined_roles, all_agent_models, env_param,
                            agent_config, log_save_path):
    agent_list = []
    village_team_model = all_agent_models["village_team"]
    def get_log_file(player_idx):
        if log_save_path is None:
            return None
        return os.path.join(log_save_path, f"Player_{player_idx + 1}.jsonl")

    if "replace" not in agent_config:
        for i, role in enumerate(assgined_roles):
            log_file = get_log_file(i)
            if role.lower() == "werewolf":
                type, agent_param = all_agent_models["werewolf"]
            else:
                type, agent_param = village_team_model
            agent = agent_registry.build_agent(type, i, agent_param, env_param, log_file)
            agent_list.append(agent)
        return agent_list
    replace_players = agent_config["replace"]["replace_player"]
    replace_role, replace_arg = parse_replace_players(replace_players)
    normalized_replace_role = replace_role.lower()
    if normalized_replace_role == "werewolf":
        repalce_id = get_replaced_wolf_id(replace_players, assgined_roles)
        for i, role in enumerate(assgined_roles):
            log_file = get_log_file(i)
            if role.lower() == "werewolf" and i != repalce_id:
                type, agent_param = all_agent_models["werewolf"]
            elif role.lower() == "werewolf" and i == repalce_id:
                type, agent_param = all_agent_models["replace"]
            else:
                type, agent_param = village_team_model
            agent = agent_registry.build_agent(type, i, agent_param, env_param, log_file)
            agent_list.append(agent)
        return agent_list
    elif normalized_replace_role in ["seer", "guard", "witch"]:
        for i, role in enumerate(assgined_roles):
            log_file = get_log_file(i)
            if role.lower() == "werewolf":
                type, agent_param = all_agent_models["werewolf"]
            elif role.lower() == normalized_replace_role:
                type, agent_param = all_agent_models["replace"]
            else:
                type, agent_param = village_team_model
            agent = agent_registry.build_agent(type, i, agent_param, env_param, log_file)
            agent_list.append(agent)
        return agent_list
    elif normalized_replace_role == "gods":
        replace_gods = replace_arg.split("-")
        for i, role in enumerate(assgined_roles):
            log_file = get_log_file(i)
            if role.lower() == "werewolf":
                type, agent_param = all_agent_models["werewolf"]
            elif role.lower() in replace_gods:
                type, agent_param = all_agent_models["replace"]
            else:
                type, agent_param = village_team_model
            agent = agent_registry.build_agent(type, i, agent_param, env_param, log_file)
            agent_list.append(agent)
        return agent_list
    elif normalized_replace_role == "ordinary_villager":
        replace_number = int(replace_arg)
        replace_ids = get_replaced_ordinary_villager_ids(assgined_roles, replace_number)
        for i, role in enumerate(assgined_roles):
            log_file = get_log_file(i)
            if role.lower() == "werewolf":
                type, agent_param = all_agent_models["werewolf"]
            elif i in replace_ids:
                type, agent_param = all_agent_models["replace"]
            else:
                type, agent_param = village_team_model
            agent = agent_registry.build_agent(type, i, agent_param, env_param, log_file)
            agent_list.append(agent)
        return agent_list
    elif normalized_replace_role == "village_team":
        replace_number = int(replace_arg.replace("random", ""))
        replace_ids = get_replaced_village_team_ids(assgined_roles, replace_number)
        for i, role in enumerate(assgined_roles):
            log_file = get_log_file(i)
            if role.lower() == "werewolf":
                type, agent_param = all_agent_models["werewolf"]
            elif i in replace_ids:
                type, agent_param = all_agent_models["replace"]
            else:
                type, agent_param = village_team_model
            agent = agent_registry.build_agent(type, i, agent_param, env_param, log_file)
            agent_list.append(agent)
        return agent_list
    else:
        raise ValueError(
            f"Unsupported replace_role '{replace_role}'. Use 'ordinary_villager' or 'village_team'."
        )


def define_agents(agent_config, env_config, log_save_path, assgined_roles,
                  backends):
    env_param = {
        "n_player": env_config["n_player"],
        "n_role": env_config["n_role"]
    }
    all_agent_models = {}
    profile_models = {}
    for group, profile in agent_config.items():
        if (
            not isinstance(profile, dict)
            or "profile_name" not in profile
        ):
            continue
        profile_name = profile["profile_name"]
        if profile_name not in profile_models:
            model_params = dict(profile["model_params"])
            profile_models[profile_name] = agent_registry.build(
                profile["agent_type"],
                backend=resolve_backend(
                    profile["backend"],
                    backends,
                ),
                model_name=profile["model"],
                **model_params,
            )
        all_agent_models[group] = profile_models[profile_name]
    return assign_agents_and_roles(
        assgined_roles,
        all_agent_models,
        env_param,
        agent_config,
        log_save_path,
    )


def check_agent_config(agent_config):
    assert "werewolf" in agent_config, "agent_config.werewolf is required."
    assert "village_team" in agent_config, "agent_config.village_team is required."
    assert "villager" not in agent_config, (
        "agent_config.villager is no longer supported; use agent_config.village_team."
    )

def build_runtime(parsed_yaml, log_save_path, backend=None,
                  backend_settings=None, roles=None, backends=None):
    config_for_normalization = deepcopy(parsed_yaml)
    if (
        backend_settings is not None
        and "backend" in config_for_normalization
        and "backends" not in config_for_normalization
    ):
        legacy_backend = config_for_normalization["backend"]
        for field in (
            "default_model",
            "agent_model",
            "parser_model",
        ):
            value = getattr(backend_settings, field, None)
            if value is not None:
                legacy_backend.setdefault(field, value)

    normalized = normalize_runtime_config(
        config_for_normalization
    )
    agent_config = normalized["agent_config"]
    env_config = normalized["env_config"]

    if backends is not None:
        backend_map = dict(backends)
    elif backend is not None:
        backend_names = list(normalized["backends"])
        if len(backend_names) != 1:
            raise ValueError(
                "a single injected backend requires exactly "
                "one configured backend"
            )
        backend_map = {backend_names[0]: backend}
    elif backend_settings is not None:
        backend_names = list(normalized["backends"])
        if len(backend_names) != 1:
            raise ValueError(
                "legacy backend_settings requires exactly "
                "one configured backend"
            )
        backend_map = {
            backend_names[0]: create_backend(backend_settings)
        }
    else:
        backend_map = load_named_backends(normalized)

    parser_config = normalized["parser"]
    speech_perceiver = SpeechPerceiver(
        backend=resolve_backend(
            parser_config["backend"],
            backend_map,
        ),
        model_name=parser_config["model"],
    )

    env_config["log_save_path"] = log_save_path
    env = WerewolfTextEnvV0(
        **env_config,
        speech_perceiver=speech_perceiver,
    )
    if env_config.get("n_hunter", 0) != 0:
        raise ValueError("TWDM 7-player environment does not support Hunter.")

    if roles is None:
        roles = (
            ["Werewolf"] * env_config["n_werewolf"]
            + ["Villager"] * env_config["n_villager"]
            + ["Seer"] * env_config["n_seer"]
            + ["Witch"] * env_config["n_witch"]
            + ["Guard"] * env_config["n_guard"]
        )
        random.shuffle(roles)
    else:
        roles = list(roles)

    check_agent_config(agent_config)
    agent_list = define_agents(
        agent_config,
        env_config,
        log_save_path,
        roles,
        backends=backend_map,
    )
    return env, agent_list, roles

def main_cli(args):
    if args.log_save_path is None:
        run_name = time.strftime("%Y%m%d_%H%M%S")
        args.log_save_path = os.path.join("logs", run_name)

    os.makedirs(args.log_save_path, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        parsed_yaml = yaml.safe_load(f)

    config_save_path = os.path.join(args.log_save_path, "config.yaml")
    with open(config_save_path, "w", encoding="utf-8") as f:
        yaml.dump(parsed_yaml, f, allow_unicode=True, sort_keys=False)

    env, agent_list, roles = build_runtime(
        parsed_yaml,
        log_save_path=args.log_save_path,
    )
    print("New rollout: ", roles)

    begin = time.time()
    sample_collector = None
    sample_path = getattr(args, "twd_tom_sample_path", None)
    if sample_path is not None:
        game_id = os.path.basename(
            os.path.normpath(args.log_save_path)
        )
        sample_collector = TWDToMSampleCollector(
            output_path=sample_path,
            game_id=game_id,
        )
    try:
        result = eval(
            env,
            agent_list,
            roles,
            sample_collector=sample_collector,
        )
    finally:
        if sample_collector is not None:
            sample_collector.close()
    print(time.time() - begin, result)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config',
                           type=str, default="configs/deepseek_vs_twdm.yaml",
                           help="path to the config file of the game")
    argparser.add_argument('--log_save_path', type=str, default=None)
    argparser.add_argument(
        '--twd_tom_sample_path',
        type=str,
        default=None,
    )
    args = argparser.parse_args()
    main_cli(args)
