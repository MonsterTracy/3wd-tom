"""Archived collection of one pre-speech subjective ToM game.

This is the only dedicated real-data collection entry point.

It reuses the existing game runtime but wraps the subjective sample
collector with the frozen contract:

- collect immediately before every public speech;
- collect synchronized beliefs from all currently alive players;
- keep role assignments in a separate audit-only manifest.

The training JSONL never receives the true role assignment.
"""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from run_random import (
    build_runtime,
    build_twd_tom_sample_collector,
    eval as run_game,
)
from werewolf.backends import (
    load_named_backends,
)
from werewolf.models.twd_tom.samples import SAMPLE_SCHEMA_VERSION
from werewolf.models.twd_tom.public_events import PUBLIC_EVENT_SCHEMA_VERSION
from werewolf.models.twd_tom.schema import (
    GLOBAL_TRUTH_INJECTED,
    LABEL_CONTEXT_SCOPE,
    LABEL_PROMPT_VERSION,
    LABEL_PROVENANCE,
    LABEL_SOURCE,
    MODEL_INPUT_SCOPE,
    NUMERIC_ANNOTATION_PRESENT,
    OBSERVER_SELECTION,
    OTHER_PLAYERS_PRIVATE_INFORMATION_VISIBLE,
    PRIVATE_CONTEXT_SERIALIZED,
    RAW_LABEL_FIELD,
    RAW_LABEL_SEMANTICS,
    RAW_LABEL_TYPE,
    REPORT_CONTEXT_MODE,
    REPORT_SIDE_EFFECT_FREE,
    REPORT_TIMING,
)
from werewolf.runtime_config import (
    normalize_runtime_config,
)


@dataclass(frozen=True)
class CollectionRunConfig:
    """Configuration for one real collection game."""

    runtime_config_path: str
    sample_path: str

    log_save_path: str | None = None
    random_seed: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(
                self.runtime_config_path,
                str,
            )
            or not self.runtime_config_path.strip()
        ):
            raise ValueError(
                "runtime_config_path is required"
            )

        if (
            not isinstance(
                self.sample_path,
                str,
            )
            or not self.sample_path.strip()
        ):
            raise ValueError(
                "sample_path is required"
            )

        if (
            self.log_save_path is not None
            and (
                not isinstance(
                    self.log_save_path,
                    str,
                )
                or not self.log_save_path.strip()
            )
        ):
            raise ValueError(
                "log_save_path must be non-empty "
                "text or None"
            )

        if self.random_seed is not None:
            if (
                isinstance(self.random_seed, bool)
                or not isinstance(
                    self.random_seed,
                    int,
                )
                or self.random_seed < 0
            ):
                raise ValueError(
                    "random_seed must be a "
                    "non-negative integer or None"
                )

def _atomic_json_dump(
    value: Any,
    path: Path,
) -> None:
    """Write a JSON object through a temporary file."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(
        path
    )


def _write_runtime_config_copy(
    parsed_yaml: dict[str, Any],
    log_save_path: Path,
) -> Path:
    """Save the exact runtime YAML used by the game."""

    output_path = (
        log_save_path
        / "config.yaml"
    )

    output_path.write_text(
        yaml.safe_dump(
            parsed_yaml,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return output_path


def _write_audit_manifest(
    *,
    log_save_path: Path,
    game_id: str,
    roles: list[str],
    role2agent_list: list[str],
    sample_path: Path,
    random_seed: int | None,
) -> Path:
    """Write truth data separately for simulation auditing.

    This file must never be used as a training target source.
    """

    if len(roles) != len(
        role2agent_list
    ):
        raise ValueError(
            "roles and role2agent_list must "
            "have equal lengths"
        )

    manifest = {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "public_event_schema_version": PUBLIC_EVENT_SCHEMA_VERSION,
        "label_prompt_version": LABEL_PROMPT_VERSION,
        "raw_label_field": RAW_LABEL_FIELD,
        "raw_label_type": RAW_LABEL_TYPE,
        "raw_label_semantics": RAW_LABEL_SEMANTICS,
        "label_provenance": LABEL_PROVENANCE,
        "label_source": LABEL_SOURCE,
        "label_context_scope": LABEL_CONTEXT_SCOPE,
        "model_input_scope": MODEL_INPUT_SCOPE,
        "report_context_mode": REPORT_CONTEXT_MODE,
        "report_side_effect_free": REPORT_SIDE_EFFECT_FREE,
        "global_truth_injected": GLOBAL_TRUTH_INJECTED,
        "other_players_private_information_visible": (
            OTHER_PLAYERS_PRIVATE_INFORMATION_VISIBLE
        ),
        "private_context_serialized": PRIVATE_CONTEXT_SERIALIZED,
        "numeric_annotation_present": NUMERIC_ANNOTATION_PRESENT,
        "online_pair_projection": False,
        "observer_selection": OBSERVER_SELECTION,
        "report_timing": REPORT_TIMING,
        "purpose": (
            "simulation_audit_only_not_training_labels"
        ),
        "game_id": game_id,
        "sample_path": str(
            sample_path
        ),
        "random_seed": random_seed,
        "collection_mode": "every_pre_public_speech",
        "players": [
            {
                "player": (
                    f"player{index + 1}"
                ),
                "role": role,
                "agent_profile": (
                    role2agent_list[index]
                ),
            }
            for index, role in enumerate(
                roles
            )
        ],
    }

    output_path = (
        log_save_path
        / "collection_audit_manifest.json"
    )

    _atomic_json_dump(
        manifest,
        output_path,
    )

    return output_path


def run_collection(
    config: CollectionRunConfig,
) -> dict[str, Any]:
    """Run one real game and collect controlled belief samples."""

    runtime_config_path = Path(
        config.runtime_config_path
    ).resolve()

    if not runtime_config_path.is_file():
        raise FileNotFoundError(
            "runtime config not found: "
            f"{runtime_config_path}"
        )

    sample_path = Path(
        config.sample_path
    ).resolve()

    sample_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if config.log_save_path is None:
        run_name = time.strftime(
            "twd_tom_collection_%Y%m%d_%H%M%S"
        )

        log_save_path = (
            Path("logs")
            / run_name
        ).resolve()
    else:
        log_save_path = Path(
            config.log_save_path
        ).resolve()

    log_save_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    parsed_yaml = yaml.safe_load(
        runtime_config_path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        parsed_yaml,
        dict,
    ):
        raise ValueError(
            "runtime config must be a mapping"
        )

    normalized_config = (
        normalize_runtime_config(
            deepcopy(
                parsed_yaml
            )
        )
    )

    backend_map = load_named_backends(
        normalized_config,
        env_file=".env",
    )

    (
        env,
        agent_list,
        roles,
        role2agent_list,
    ) = build_runtime(
        parsed_yaml,
        log_save_path=str(
            log_save_path
        ),
        random_seed=(
            config.random_seed
        ),
        backends=backend_map,
    )

    game_id = log_save_path.name

    runtime_copy_path = (
        _write_runtime_config_copy(
            parsed_yaml,
            log_save_path,
        )
    )

    audit_manifest_path = (
        _write_audit_manifest(
            log_save_path=(
                log_save_path
            ),
            game_id=game_id,
            roles=list(roles),
            role2agent_list=list(
                role2agent_list
            ),
            sample_path=sample_path,
            random_seed=(
                config.random_seed
            ),
        )
    )

    base_collector = (
        build_twd_tom_sample_collector(
            agent_list=agent_list,
            output_path=str(
                sample_path
            ),
            game_id=game_id,
        )
    )

    started_at = time.time()

    try:
        game_result = run_game(
            env,
            agent_list,
            roles,
            sample_collector=(
                base_collector
            ),
        )
    finally:
        base_collector.close()

    elapsed_seconds = (
        time.time()
        - started_at
    )

    collection_statistics = {
        "snapshots_written": base_collector.samples_written,
    }

    status = (
        "ok"
        if collection_statistics[
            "snapshots_written"
        ]
        > 0
        else "no_samples"
    )

    summary = {
        "status": status,
        "game_id": game_id,
        "game_result": game_result,
        "elapsed_seconds": (
            elapsed_seconds
        ),
        "runtime_config_path": str(
            runtime_config_path
        ),
        "runtime_config_copy": str(
            runtime_copy_path
        ),
        "sample_path": str(
            sample_path
        ),
        "log_save_path": str(
            log_save_path
        ),
        "audit_manifest_path": str(
            audit_manifest_path
        ),
        "collection": (
            collection_statistics
        ),
    }

    summary_path = (
        log_save_path
        / "collection_summary.json"
    )

    _atomic_json_dump(
        summary,
        summary_path,
    )

    summary[
        "summary_path"
    ] = str(
        summary_path
    )

    return summary


def build_arg_parser() -> (
    argparse.ArgumentParser
):
    """Build the real collection CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Collect one quality-controlled "
            "subjective ToM game."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Runtime YAML containing agent and parser sections."
        ),
    )

    parser.add_argument(
        "--sample-path",
        required=True,
        help=(
            "Destination subjective JSONL path."
        ),
    )

    parser.add_argument(
        "--log-save-path",
        default=None,
        help=(
            "Directory for game logs and "
            "audit-only metadata."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    return parser


def main() -> int:
    """CLI entry point."""

    args = (
        build_arg_parser()
        .parse_args()
    )

    summary = run_collection(
        CollectionRunConfig(
            runtime_config_path=(
                args.config
            ),
            sample_path=(
                args.sample_path
            ),
            log_save_path=(
                args.log_save_path
            ),
            random_seed=args.seed,
        )
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )

    return (
        0
        if summary["status"] == "ok"
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
