"""Run one frozen ten-game playing-agent belief-set collection batch."""

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
from werewolf.models.twd_tom.collector import require_clean_collection_worktree
from werewolf.models.twd_tom.samples import (
    ACTOR_PAIR_BELIEF_ANNOTATION_VERSION,
    ACTOR_PAIR_BELIEF_SCHEMA_VERSION,
)


@dataclass(frozen=True)
class FormalBatchConfig:
    batch_id: str
    monitored: MonitoredCollectionConfig

    def __post_init__(self) -> None:
        if not isinstance(self.batch_id, str) or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", self.batch_id
        ) is None:
            raise ValueError("batch_id must be a filesystem-safe identifier")
        output_name = Path(self.monitored.output_dir).name.lower()
        if self.batch_id.lower() not in output_name:
            raise ValueError("output directory name must contain batch_id")


def run_formal_batch(config: FormalBatchConfig):
    """Delegate one formal raw batch to the frozen monitored runner."""

    require_clean_collection_worktree()
    return run_monitored_collection(
        config.monitored,
        artifact_prefix="formal_batch",
        required_name_token="formal_batch",
        game_id_prefix=f"formal_{config.batch_id}_game",
        mode_metadata={
            "formal_batch_only": True,
            "batch_id": config.batch_id,
            "schema_version": ACTOR_PAIR_BELIEF_SCHEMA_VERSION,
            "annotation_version": ACTOR_PAIR_BELIEF_ANNOTATION_VERSION,
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--game-count", required=True, type=int)
    parser.add_argument("--seeds", required=True, type=int, nargs=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-gameplay-calls-per-game", required=True, type=int)
    parser.add_argument("--max-belief-calls-per-game", required=True, type=int)
    parser.add_argument("--max-total-calls-per-game", required=True, type=int)
    parser.add_argument("--max-wall-seconds-per-game", required=True, type=float)
    parser.add_argument("--max-total-calls", required=True, type=int)
    parser.add_argument("--max-wall-seconds", required=True, type=float)
    parser.add_argument("--privacy-safe-logging", required=True, action="store_true")
    parser.add_argument("--audit-only-metadata", required=True, action="store_true")
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
    )
    summary = run_formal_batch(
        FormalBatchConfig(batch_id=args.batch_id, monitored=monitored)
    )
    if summary["status"] != "ok":
        raise SystemExit(1)
    return summary


if __name__ == "__main__":
    main()
