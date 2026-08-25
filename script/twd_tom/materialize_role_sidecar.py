"""Materialize supervision-only observer roles from canonical trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from script.twd_tom.collect_canonical_trajectories import (
    validate_canonical_belief_batch,
)
from script.twd_tom.materialize_canonical_belief_dataset import (
    validate_split_manifest,
)
from werewolf.models.twd_tom.supervision import (
    ROLE_SIDECAR_SCHEMA_VERSION,
    normalize_observer_roles,
)
from werewolf.trajectory import canonical_digest, canonical_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON file must contain one object: {path}")
    return value


def _trajectory_roles(trajectory: Mapping[str, Any]) -> dict[str, str]:
    players = trajectory.get("players")
    if not isinstance(players, list) or len(players) != 7:
        raise ValueError("canonical trajectory must contain seven players")
    roles: dict[str, str] = {}
    for expected_id, player in enumerate(players, start=1):
        if not isinstance(player, Mapping) or player.get("player_id") != expected_id:
            raise ValueError("trajectory players must use ordered IDs 1...7")
        roles[f"player{expected_id}"] = player.get("role")
    return normalize_observer_roles(roles)


def materialize_role_sidecar(
    *,
    canonical_root: str | Path,
    split_manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Write an immutable role sidecar without changing canonical samples."""

    root = Path(canonical_root).resolve()
    split_manifest_path = Path(split_manifest_path).resolve()
    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError(f"role sidecar output already exists: {destination}")
    verified_batch = validate_canonical_belief_batch(root)
    split_manifest = validate_split_manifest(split_manifest_path)
    if split_manifest["canonical_batch_summary_digest"] != verified_batch[
        "batch_summary_digest"
    ]:
        raise ValueError("split manifest and canonical batch digests differ")
    split_game_ids = {
        game_id
        for game_ids in split_manifest["game_ids"].values()
        for game_id in game_ids
    }
    verified_games = {game["game_id"]: game for game in verified_batch["games"]}
    if split_game_ids != set(verified_games):
        raise ValueError("split manifest and canonical batch game IDs differ")

    games: dict[str, Any] = {}
    for game_id in sorted(verified_games):
        game_dir = (root / verified_games[game_id]["relative_path"]).parent
        summary_path = game_dir / "summary.json"
        trajectory_path = game_dir / "trajectory.json"
        summary = _load_json(summary_path)
        trajectory = _load_json(trajectory_path)
        if summary.get("game_id") != game_id or trajectory.get("game_id") != game_id:
            raise ValueError(f"role source game_id mismatch: {game_id}")
        if summary.get("summary_digest") != verified_games[game_id][
            "game_summary_digest"
        ]:
            raise ValueError(f"role source summary digest mismatch: {game_id}")
        trajectory_payload = dict(trajectory)
        trajectory_digest = trajectory_payload.pop("trajectory_digest", None)
        if trajectory_digest != canonical_digest(trajectory_payload):
            raise ValueError(f"trajectory digest mismatch: {game_id}")
        trajectory_sha256 = _sha256(trajectory_path)
        if summary.get("trajectory_digest") != trajectory_digest:
            raise ValueError(f"summary trajectory digest mismatch: {game_id}")
        if summary.get("trajectory_sha256") != trajectory_sha256:
            raise ValueError(f"summary trajectory SHA-256 mismatch: {game_id}")
        games[game_id] = {
            "game_summary_digest": summary["summary_digest"],
            "trajectory_digest": trajectory_digest,
            "trajectory_sha256": trajectory_sha256,
            "observer_roles": _trajectory_roles(trajectory),
        }

    report = {
        "schema_version": ROLE_SIDECAR_SCHEMA_VERSION,
        "canonical_batch_summary_digest": verified_batch["batch_summary_digest"],
        "canonical_batch_summary_sha256": verified_batch["batch_summary_sha256"],
        "split_manifest_digest": split_manifest["manifest_digest"],
        "games": games,
    }
    report["sidecar_digest"] = canonical_digest(report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(report) + "\n")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize supervision-only observer role metadata."
    )
    parser.add_argument("--canonical-root", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = materialize_role_sidecar(
        canonical_root=args.canonical_root,
        split_manifest_path=args.split_manifest,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
