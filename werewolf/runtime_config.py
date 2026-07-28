from collections.abc import Mapping
from copy import deepcopy


_SUPPORTED_BACKEND_TYPE = "openai_compatible"

_PROFILE_FIELDS = {
    "profile_name",
    "agent_type",
    "model_type",
    "backend",
    "model",
    "model_params",
    "sample_ratio",
}

def _as_mapping(value, field_name):
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{field_name} must be a mapping"
        )

    return value


def _required_string(value, field_name):
    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise ValueError(
            f"{field_name} is required"
        )

    return value


def normalize_backend_config(
    config: Mapping,
    *,
    legacy=False,
) -> dict:
    config = _as_mapping(
        config,
        "backend config",
    )

    backend_type = (
        config.get("type")
        or config.get("backend_type")
        or _SUPPORTED_BACKEND_TYPE
    )

    if backend_type != _SUPPORTED_BACKEND_TYPE:
        raise ValueError(
            f"unsupported backend type: "
            f"{backend_type}"
        )

    return {
        "type": backend_type,
        "base_url": deepcopy(
            config.get("base_url")
        ),
        "api_key_env": (
            "OPENAI_API_KEY"
            if legacy
            else deepcopy(
                config.get("api_key_env")
            )
        ),
        "default_model": deepcopy(
            config.get("default_model")
        ),
    }


def normalize_parser_config(
    config,
    backends: Mapping,
    *,
    legacy_backend: Mapping | None = None,
) -> dict:
    if legacy_backend is not None:
        legacy_backend = _as_mapping(
            legacy_backend,
            "legacy backend config",
        )

        backend_name = "default"

        model = (
            legacy_backend.get("parser_model")
            or legacy_backend.get(
                "default_model"
            )
            or legacy_backend.get(
                "agent_model"
            )
        )

        model_params = {
            "temperature": 0.0,
        }

    else:
        config = _as_mapping(
            config,
            "parser config",
        )

        backend_name = config.get(
            "backend"
        )

        model = config.get("model")

        raw_model_params = config.get(
            "model_params",
            {},
        )

        model_params = deepcopy(
            _as_mapping(
                raw_model_params,
                "parser.model_params",
            )
        )

    backend_name = _required_string(
        backend_name,
        "parser.backend",
    )

    if backend_name not in backends:
        raise ValueError(
            "parser backend does not exist: "
            f"{backend_name}"
        )

    model = _required_string(
        model,
        "parser.model",
    )

    return {
        "backend": backend_name,
        "model": model,
        "model_params": model_params,
    }


def normalize_agent_profile(
    profile: Mapping,
    backends: Mapping,
    *,
    legacy_agent_model=None,
    legacy=False,
) -> dict:
    profile = _as_mapping(
        profile,
        "agent profile",
    )

    legacy = (
        legacy
        or "model_type" in profile
    )

    raw_model_params = profile.get(
        "model_params",
        {},
    )

    model_params = deepcopy(
        _as_mapping(
            raw_model_params,
            "agent profile model_params",
        )
    )

    model_name_alias = model_params.pop(
        "model_name",
        None,
    )

    llm_alias = model_params.pop(
        "llm",
        None,
    )

    if legacy:
        profile_name = (
            profile.get("profile_name")
            or profile.get("model_type")
        )

        agent_type = (
            profile.get("agent_type")
            or profile.get("model_type")
        )

        backend_name = (
            profile.get("backend")
            or "default"
        )

        model = (
            profile.get("model")
            or model_name_alias
            or llm_alias
            or legacy_agent_model
        )

    else:
        profile_name = profile.get(
            "profile_name"
        )

        agent_type = profile.get(
            "agent_type"
        )

        backend_name = profile.get(
            "backend"
        )

        model = profile.get("model")

    profile_name = _required_string(
        profile_name,
        "agent profile profile_name",
    )

    agent_type = _required_string(
        agent_type,
        (
            f"agent profile "
            f"{profile_name}.agent_type"
        ),
    )

    backend_name = _required_string(
        backend_name,
        (
            f"agent profile "
            f"{profile_name}.backend"
        ),
    )

    if backend_name not in backends:
        raise ValueError(
            "agent profile backend "
            "does not exist: "
            f"{backend_name}"
        )

    model = _required_string(
        model,
        (
            f"agent profile "
            f"{profile_name}.model"
        ),
    )

    sample_ratio = profile.get(
        "sample_ratio",
        1.0,
    )

    if sample_ratio is None:
        sample_ratio = 1.0

    normalized = {
        key: deepcopy(value)
        for key, value in profile.items()
        if key not in _PROFILE_FIELDS
    }

    normalized.update(
        {
            "profile_name": profile_name,
            "agent_type": agent_type,
            "backend": backend_name,
            "model": model,
            "model_params": model_params,
            "sample_ratio": sample_ratio,
        }
    )

    return normalized


def _normalize_agent_config(
    config,
    backends,
    *,
    legacy_schema,
    legacy_agent_model,
):
    config = _as_mapping(
        config,
        "agent_config",
    )

    normalized = {}
    profile_names = set()
    legacy_aliases = {}

    def register_profile(
        raw_profile,
        normalized_profile,
    ):
        profile_name = normalized_profile[
            "profile_name"
        ]

        if profile_name in profile_names:
            raise ValueError(
                "duplicate profile_name: "
                f"{profile_name}"
            )

        profile_names.add(profile_name)

        model_type = raw_profile.get(
            "model_type"
        )

        if (
            isinstance(model_type, str)
            and model_type
        ):
            legacy_aliases.setdefault(
                model_type,
                set(),
            ).add(profile_name)

    if "all_candidates" in config:
        raw_candidates = config[
            "all_candidates"
        ]

        if not isinstance(
            raw_candidates,
            list,
        ):
            raise ValueError(
                "agent_config.all_candidates "
                "must be a list"
            )

        candidates = []

        for candidate in raw_candidates:
            raw_candidate = _as_mapping(
                candidate,
                "agent candidate",
            )

            normalized_candidate = (
                normalize_agent_profile(
                    raw_candidate,
                    backends,
                    legacy_agent_model=(
                        legacy_agent_model
                    ),
                    legacy=legacy_schema,
                )
            )

            register_profile(
                raw_candidate,
                normalized_candidate,
            )

            candidates.append(
                normalized_candidate
            )

        normalized[
            "all_candidates"
        ] = candidates

    for key, value in config.items():
        if key in {
            "all_candidates",
            "must_include",
        }:
            continue

        if (
            isinstance(value, Mapping)
            and _PROFILE_FIELDS.intersection(
                value
            )
        ):
            normalized_profile = (
                normalize_agent_profile(
                    value,
                    backends,
                    legacy_agent_model=(
                        legacy_agent_model
                    ),
                    legacy=legacy_schema,
                )
            )

            register_profile(
                value,
                normalized_profile,
            )

            normalized[key] = (
                normalized_profile
            )

        else:
            normalized[key] = deepcopy(
                value
            )

    raw_must_include = config.get(
        "must_include",
        [],
    )

    if not isinstance(
        raw_must_include,
        list,
    ):
        raise ValueError(
            "agent_config.must_include "
            "must be a list"
        )

    must_include = []

    for requested_name in raw_must_include:
        if requested_name in profile_names:
            resolved_name = requested_name

        elif legacy_schema:
            aliases = legacy_aliases.get(
                requested_name,
                set(),
            )

            if len(aliases) != 1:
                raise ValueError(
                    "must_include profile "
                    "does not exist: "
                    f"{requested_name}"
                )

            resolved_name = next(
                iter(aliases)
            )

        else:
            raise ValueError(
                "must_include profile "
                "does not exist: "
                f"{requested_name}"
            )

        must_include.append(
            resolved_name
        )

    normalized[
        "must_include"
    ] = must_include

    return normalized


def normalize_runtime_config(
    config: Mapping,
) -> dict:
    config = _as_mapping(
        config,
        "runtime config",
    )

    normalized = deepcopy(
        dict(config)
    )

    if "belief" in config:
        raise ValueError(
            "runtime belief reporter config is unsupported; "
            "readonly reports use each playing agent backend"
        )

    has_legacy_backend = (
        "backend" in config
    )

    has_named_backends = (
        "backends" in config
    )

    if (
        has_legacy_backend
        and has_named_backends
    ):
        raise ValueError(
            "runtime config cannot contain "
            "both backend and backends"
        )

    if (
        not has_legacy_backend
        and not has_named_backends
    ):
        raise ValueError(
            "runtime config must contain "
            "backend or backends"
        )

    if has_legacy_backend:
        legacy_backend = _as_mapping(
            config["backend"],
            "backend",
        )

        backends = {
            "default": (
                normalize_backend_config(
                    legacy_backend,
                    legacy=True,
                )
            )
        }

        parser = normalize_parser_config(
            None,
            backends,
            legacy_backend=legacy_backend,
        )

        legacy_agent_model = (
            legacy_backend.get(
                "agent_model"
            )
            or legacy_backend.get(
                "default_model"
            )
        )

        legacy_schema = True

        normalized.pop(
            "backend",
            None,
        )

    else:
        raw_backends = _as_mapping(
            config["backends"],
            "backends",
        )

        backends = {
            _required_string(
                name,
                "backend name",
            ): normalize_backend_config(
                backend_config
            )
            for name, backend_config
            in raw_backends.items()
        }

        parser = normalize_parser_config(
            config.get("parser"),
            backends,
        )

        legacy_agent_model = None
        legacy_schema = False

    normalized["backends"] = backends
    normalized["parser"] = parser

    normalized["agent_config"] = (
        _normalize_agent_config(
            config.get(
                "agent_config",
                {},
            ),
            backends,
            legacy_schema=legacy_schema,
            legacy_agent_model=(
                legacy_agent_model
            ),
        )
    )

    return normalized
