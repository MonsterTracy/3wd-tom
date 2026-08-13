"""Run one or two explicit raw-content speech semantic fidelity audits."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from run_random import build_runtime, eval as run_game
from werewolf.runtime_config import normalize_runtime_config
from werewolf.speech.speech_fidelity_audit import SpeechFidelityAuditSidecar


def _source_commit() -> str:
    repository_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot resolve speech audit source commit") from exc
    commit = result.stdout.strip()
    if re.fullmatch(r"[0-9a-fA-F]{40}", commit) is None:
        raise RuntimeError("source commit must be a 40-character SHA")
    return commit


def _validated_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    values = tuple(seeds)
    if not 1 <= len(values) <= 2:
        raise ValueError("speech fidelity audit requires one or two seeds")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in values):
        raise TypeError("audit seeds must be integers")
    if len(set(values)) != len(values):
        raise ValueError("audit seeds must be distinct")
    return values


def run_speech_fidelity_audit(
    *,
    config_path: str | Path,
    seeds: Sequence[int],
    audit_file: str | Path,
) -> dict[str, Any]:
    """Run an isolated raw-content audit without creating formal artifacts."""

    seed_values = _validated_seeds(seeds)
    runtime_path = Path(config_path).resolve()
    if not runtime_path.is_file():
        raise FileNotFoundError(f"runtime config not found: {runtime_path}")
    parsed_yaml = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
    if not isinstance(parsed_yaml, dict):
        raise ValueError("runtime config must be a mapping")
    normalized = normalize_runtime_config(parsed_yaml)
    parser_config = normalized["parser"]
    source_commit = _source_commit()
    game_summaries = []

    with SpeechFidelityAuditSidecar(
        audit_file,
        source_commit=source_commit,
        backend_id=parser_config["backend"],
        model_id=parser_config["model"],
    ) as sidecar:
        for game_index, seed in enumerate(seed_values, start=1):
            game_id = f"speech_fidelity_game_{game_index:03d}_seed_{seed}"
            env, agents, roles, _profiles = build_runtime(
                parsed_yaml,
                log_save_path=None,
                random_seed=seed,
            )
            audited_perceiver = sidecar.audited_perceiver(
                env.speech_perceiver,
                game_id=game_id,
                seed=seed,
            )
            env.speech_perceiver = audited_perceiver
            try:
                run_game(
                    env,
                    agents,
                    roles,
                    sample_collector=audited_perceiver,
                )
            finally:
                for agent in agents:
                    close = getattr(agent, "close", None)
                    if callable(close):
                        close()
            game_summaries.append({"game_id": game_id, "seed": seed})

    summary = {
        "audit_schema": "speech_semantic_fidelity_audit_v1",
        "audit_file": str(Path(audit_file).resolve()),
        "source_commit": source_commit,
        "game_count": len(game_summaries),
        "games": game_summaries,
        "formal_artifacts_created": False,
        "raw_content_audit": True,
    }
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", required=True, type=int, nargs="+")
    parser.add_argument("--audit-file", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_argument_parser().parse_args(argv)
    summary = run_speech_fidelity_audit(
        config_path=args.config,
        seeds=args.seeds,
        audit_file=args.audit_file,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
