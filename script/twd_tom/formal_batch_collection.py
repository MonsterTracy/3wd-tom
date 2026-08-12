"""Run one monitored formal belief-label collection batch."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from script.twd_tom.monitored_collection import (
    MonitoredCollectionConfig,
    run_monitored_collection,
)
from run_random import (
    COLLECTION_MODES,
    PRIVATE_CONDITIONED_COLLECTION_MODE,
    PUBLIC_ONLY_COLLECTION_MODE,
)
from werewolf.models.public_belief_matrix.collection import (
    PUBLIC_BELIEF_MATRIX_COLLECTION_MODE,
    PUBLIC_BELIEF_MATRIX_PROVENANCE,
    PUBLIC_BELIEF_MATRIX_SAMPLE_SCHEMA_VERSION,
    PUBLIC_BELIEF_MATRIX_VISIBLE_PREFIX_SCHEMA_VERSION,
)
from werewolf.models.public_belief_matrix.reporter import PUBLIC_BELIEF_MATRIX_PROMPT_VERSION
from werewolf.models.twd_tom.samples import (
    PUBLIC_ONLY_SAMPLE_SCHEMA_VERSION,
    SAMPLE_SCHEMA_VERSION,
)
from werewolf.models.twd_tom.schema import PUBLIC_ONLY_LABEL_PROMPT_VERSION


@dataclass(frozen=True)
class FormalBatchConfig:
    batch_id: str
    monitored: MonitoredCollectionConfig
    collection_mode: str = PRIVATE_CONDITIONED_COLLECTION_MODE

    def __post_init__(self) -> None:
        if not isinstance(self.batch_id, str) or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", self.batch_id
        ) is None:
            raise ValueError("batch_id must be a filesystem-safe identifier")
        output_name = Path(self.monitored.output_dir).name.lower()
        if self.batch_id.lower() not in output_name:
            raise ValueError("output directory name must contain batch_id")
        if self.collection_mode not in COLLECTION_MODES:
            raise ValueError(f"collection_mode must be one of {COLLECTION_MODES}")
        if self.collection_mode != PUBLIC_BELIEF_MATRIX_COLLECTION_MODE and self.monitored.game_count != 10:
            raise ValueError("existing collection modes require exactly ten games")


def run_formal_batch(config: FormalBatchConfig):
    """Delegate one formal raw batch to the frozen monitored runner."""

    mode_metadata = {
        "formal_batch_only": True,
        "batch_id": config.batch_id,
        "schema_version": (
            PUBLIC_BELIEF_MATRIX_SAMPLE_SCHEMA_VERSION
            if config.collection_mode == PUBLIC_BELIEF_MATRIX_COLLECTION_MODE
            else
            PUBLIC_ONLY_SAMPLE_SCHEMA_VERSION
            if config.collection_mode == PUBLIC_ONLY_COLLECTION_MODE
            else SAMPLE_SCHEMA_VERSION
        ),
    }
    if config.collection_mode == PUBLIC_ONLY_COLLECTION_MODE:
        mode_metadata.update(
            {
                "belief_information_scope": "public_events_only",
                "playing_agent_context_reused": False,
                "true_role_visible": False,
                "private_memory_visible": False,
                "prompt_version": PUBLIC_ONLY_LABEL_PROMPT_VERSION,
            }
        )
    elif config.collection_mode == PUBLIC_BELIEF_MATRIX_COLLECTION_MODE:
        mode_metadata.update(
            {
                **PUBLIC_BELIEF_MATRIX_PROVENANCE,
                "visible_prefix_schema_version": PUBLIC_BELIEF_MATRIX_VISIBLE_PREFIX_SCHEMA_VERSION,
                "prompt_version": PUBLIC_BELIEF_MATRIX_PROMPT_VERSION,
                "playing_agent_context_reused": False,
                "true_role_visible": False,
                "private_memory_visible": False,
            }
        )
    return run_monitored_collection(
        config.monitored,
        artifact_prefix="formal_batch",
        required_name_token="formal_batch",
        game_id_prefix=f"formal_{config.batch_id}_game",
        mode_metadata=mode_metadata,
        collection_mode=config.collection_mode,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--game-count", required=True, type=int)
    parser.add_argument("--seeds", required=True, type=int, nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--logs-root")
    parser.add_argument("--max-gameplay-calls-per-game", required=True, type=int)
    parser.add_argument("--max-belief-calls-per-game", required=True, type=int)
    parser.add_argument("--max-total-calls-per-game", required=True, type=int)
    parser.add_argument("--max-wall-seconds-per-game", required=True, type=float)
    parser.add_argument("--max-total-calls", required=True, type=int)
    parser.add_argument("--max-wall-seconds", required=True, type=float)
    parser.add_argument("--privacy-safe-logging", required=True, action="store_true")
    parser.add_argument("--audit-only-metadata", required=True, action="store_true")
    parser.add_argument(
        "--collection-mode",
        choices=COLLECTION_MODES,
        default=PRIVATE_CONDITIONED_COLLECTION_MODE,
    )
    return parser


def main(argv: Sequence[str] | None = None):
    args = build_arg_parser().parse_args(argv)
    monitored = MonitoredCollectionConfig(
        runtime_config_path=args.config,
        output_dir=args.output_dir,
        game_count=args.game_count,
        seeds=tuple(args.seeds),
        max_gameplay_calls_per_game=args.max_gameplay_calls_per_game,
        max_belief_calls_per_game=args.max_belief_calls_per_game,
        max_total_calls_per_game=args.max_total_calls_per_game,
        max_wall_seconds_per_game=args.max_wall_seconds_per_game,
        max_total_calls=args.max_total_calls,
        max_wall_seconds=args.max_wall_seconds,
        privacy_safe_logging=args.privacy_safe_logging,
        audit_only_metadata=args.audit_only_metadata,
        logs_root=args.logs_root,
    )
    summary = run_formal_batch(
        FormalBatchConfig(
            batch_id=args.batch_id,
            monitored=monitored,
            collection_mode=args.collection_mode,
        )
    )
    if summary["status"] != "ok":
        raise SystemExit(1)
    return summary


if __name__ == "__main__":
    main()
