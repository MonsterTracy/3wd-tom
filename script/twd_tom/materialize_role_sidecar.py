"""Materialize development-only supervision roles from canonical trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from script.twd_tom.collect_canonical_trajectories import (
    BATCH_SUMMARY_SCHEMA_VERSION,
    CANONICAL_COLLECTION_MODE,
    GAME_SUMMARY_SCHEMA_VERSION,
)
from script.twd_tom.materialize_canonical_belief_dataset import (
    validate_split_manifest,
)
from script.twd_tom.materialize_development_folds import (
    DEVELOPMENT_FOLD_MANIFEST_SCHEMA_VERSION,
    DEVELOPMENT_FOLD_POLICY_VERSION,
)
from werewolf.models.twd_tom.supervision import (
    ROLE_SIDECAR_SCHEMA_VERSION,
    load_role_sidecar_report,
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


def _load_development_fold_manifest(path: Path) -> dict[str, Any]:
    manifest = _load_json(path)
    if manifest.get("schema_version") != DEVELOPMENT_FOLD_MANIFEST_SCHEMA_VERSION:
        raise ValueError("development fold manifest schema mismatch")
    if manifest.get("policy_version") != DEVELOPMENT_FOLD_POLICY_VERSION:
        raise ValueError("development fold policy mismatch")
    payload = dict(manifest)
    digest = payload.pop("manifest_digest", None)
    if digest != canonical_digest(payload):
        raise ValueError("development fold manifest digest mismatch")

    development_ids = manifest.get("development_game_ids")
    sealed_ids = manifest.get("sealed_test_game_ids")
    if (
        not isinstance(development_ids, list)
        or len(development_ids) != 54
        or len(set(development_ids)) != 54
        or any(not isinstance(game_id, str) or not game_id for game_id in development_ids)
    ):
        raise ValueError("development fold manifest must contain exactly 54 game IDs")
    if (
        not isinstance(sealed_ids, list)
        or len(sealed_ids) != 6
        or len(set(sealed_ids)) != 6
        or any(not isinstance(game_id, str) or not game_id for game_id in sealed_ids)
    ):
        raise ValueError("development fold manifest must contain exactly 6 sealed IDs")
    if set(development_ids) & set(sealed_ids):
        raise ValueError("development and sealed game IDs overlap")
    if manifest.get("fold_count") != 5:
        raise ValueError("development role sidecar requires exactly 5 folds")
    folds = manifest.get("folds")
    if not isinstance(folds, Mapping) or len(folds) != 5:
        raise ValueError("development fold descriptor count mismatch")
    for fold_name, descriptor in folds.items():
        if not isinstance(fold_name, str) or not isinstance(descriptor, Mapping):
            raise TypeError("development fold descriptors must be mappings")
        train_ids = descriptor.get("train_game_ids")
        validation_ids = descriptor.get("validation_game_ids")
        if not isinstance(train_ids, list) or not isinstance(validation_ids, list):
            raise TypeError("development fold game IDs must be lists")
        if set(train_ids) & set(validation_ids):
            raise ValueError(f"development fold train/validation overlap: {fold_name}")
        if set(train_ids) | set(validation_ids) != set(development_ids):
            raise ValueError(f"development fold game IDs differ: {fold_name}")
    return manifest


def _validated_development_lineage(
    development_fold_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fold_manifest = _load_development_fold_manifest(
        development_fold_manifest_path
    )
    source_manifest_path = (
        development_fold_manifest_path.parent
        / fold_manifest["source_split_manifest_relative_path"]
    ).resolve()
    if _sha256(source_manifest_path) != fold_manifest[
        "source_split_manifest_sha256"
    ]:
        raise ValueError("source split manifest SHA-256 mismatch")
    source_manifest = validate_split_manifest(
        source_manifest_path,
        verify_split_files=("train", "validation"),
    )
    if source_manifest["manifest_digest"] != fold_manifest[
        "source_split_manifest_digest"
    ]:
        raise ValueError("source split manifest digest mismatch")
    if source_manifest["canonical_batch_summary_digest"] != fold_manifest[
        "canonical_batch_summary_digest"
    ]:
        raise ValueError("canonical batch summary digest mismatch")
    source_development_ids = set(source_manifest["game_ids"]["train"]) | set(
        source_manifest["game_ids"]["validation"]
    )
    source_sealed_ids = set(source_manifest["game_ids"]["test"])
    if source_development_ids != set(fold_manifest["development_game_ids"]):
        raise ValueError("development game IDs differ from source split")
    if source_sealed_ids != set(fold_manifest["sealed_test_game_ids"]):
        raise ValueError("sealed game IDs differ from source split")
    return fold_manifest, source_manifest


def validate_development_role_sidecar(
    *,
    role_sidecar_path: str | Path,
    development_fold_manifest_path: str | Path,
) -> dict[str, Any]:
    """Validate one role sidecar against its exact development-fold lineage."""

    fold_manifest, source_manifest = _validated_development_lineage(
        Path(development_fold_manifest_path).resolve()
    )
    report = load_role_sidecar_report(Path(role_sidecar_path).resolve())
    if report["development_fold_manifest_digest"] != fold_manifest[
        "manifest_digest"
    ]:
        raise ValueError("role sidecar and development fold digests differ")
    if report["split_manifest_digest"] != source_manifest["manifest_digest"]:
        raise ValueError("role sidecar and source split digests differ")
    if report["canonical_batch_summary_digest"] != fold_manifest[
        "canonical_batch_summary_digest"
    ]:
        raise ValueError("role sidecar and canonical batch digests differ")
    game_ids = set(report["games"])
    if game_ids != set(fold_manifest["development_game_ids"]):
        raise ValueError("role sidecar game IDs differ from development games")
    if game_ids & set(fold_manifest["sealed_test_game_ids"]):
        raise ValueError("role sidecar contains sealed game IDs")
    return report


def materialize_role_sidecar(
    *,
    canonical_root: str | Path,
    development_fold_manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Write an immutable 54-game development-only role sidecar."""

    root = Path(canonical_root).resolve()
    development_fold_manifest_path = Path(
        development_fold_manifest_path
    ).resolve()
    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError(f"role sidecar output already exists: {destination}")
    fold_manifest, source_manifest = _validated_development_lineage(
        development_fold_manifest_path
    )

    batch_summary_path = root / "summary.json"
    batch_summary = _load_json(batch_summary_path)
    if batch_summary.get("schema_version") != BATCH_SUMMARY_SCHEMA_VERSION:
        raise ValueError("canonical batch summary schema mismatch")
    batch_payload = dict(batch_summary)
    batch_digest = batch_payload.pop("summary_digest", None)
    if batch_digest != canonical_digest(batch_payload):
        raise ValueError("canonical batch summary digest mismatch")
    if batch_summary.get("collection_mode") != CANONICAL_COLLECTION_MODE:
        raise ValueError("canonical batch is not in canonical mode")
    if (
        batch_summary.get("canonical_eligible") is not True
        or batch_summary.get("target_reached") is not True
    ):
        raise ValueError("canonical batch is not canonical-eligible")
    if batch_digest != fold_manifest["canonical_batch_summary_digest"]:
        raise ValueError("canonical batch and development fold digests differ")
    batch_sha256 = _sha256(batch_summary_path)
    if batch_sha256 != source_manifest["canonical_batch_summary_sha256"]:
        raise ValueError("canonical batch summary SHA-256 mismatch")

    batch_game_ids = batch_summary.get("game_ids")
    completed_seeds = batch_summary.get("completed_seeds")
    planned_seeds = batch_summary.get("seeds")
    game_summary_digests = batch_summary.get("game_summary_digests")
    source_game_ids = {
        game_id
        for split_ids in source_manifest["game_ids"].values()
        for game_id in split_ids
    }
    if (
        not isinstance(batch_game_ids, list)
        or not isinstance(completed_seeds, list)
        or len(batch_game_ids) != len(completed_seeds)
        or len(set(batch_game_ids)) != len(batch_game_ids)
        or set(batch_game_ids) != source_game_ids
    ):
        raise ValueError("canonical batch game IDs differ from source split")
    if (
        not isinstance(planned_seeds, list)
        or len(set(planned_seeds)) != len(planned_seeds)
        or any(seed not in planned_seeds for seed in completed_seeds)
    ):
        raise ValueError("canonical batch seed mapping is invalid")
    if not isinstance(game_summary_digests, Mapping) or set(
        game_summary_digests
    ) != set(batch_game_ids):
        raise ValueError("canonical batch game summary digests differ")
    if game_summary_digests != source_manifest["game_summary_digests"]:
        raise ValueError("canonical and split game summary digests differ")

    seed_positions = {
        seed: position for position, seed in enumerate(planned_seeds, start=1)
    }
    game_seeds = dict(zip(batch_game_ids, completed_seeds))

    games: dict[str, Any] = {}
    for game_id in sorted(fold_manifest["development_game_ids"]):
        seed = game_seeds[game_id]
        game_dir = root / "games" / f"game_{seed_positions[seed]:04d}_seed_{seed}"
        summary_path = game_dir / "summary.json"
        trajectory_path = game_dir / "trajectory.json"
        summary = _load_json(summary_path)
        trajectory = _load_json(trajectory_path)
        if summary.get("game_id") != game_id or trajectory.get("game_id") != game_id:
            raise ValueError(f"role source game_id mismatch: {game_id}")
        if summary.get("schema_version") != GAME_SUMMARY_SCHEMA_VERSION:
            raise ValueError(f"role source game summary schema mismatch: {game_id}")
        summary_payload = dict(summary)
        summary_digest = summary_payload.pop("summary_digest", None)
        if summary_digest != canonical_digest(summary_payload):
            raise ValueError(f"role source summary digest mismatch: {game_id}")
        if summary_digest != game_summary_digests[game_id]:
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
            "game_summary_digest": summary_digest,
            "trajectory_digest": trajectory_digest,
            "trajectory_sha256": trajectory_sha256,
            "observer_roles": _trajectory_roles(trajectory),
        }

    report = {
        "schema_version": ROLE_SIDECAR_SCHEMA_VERSION,
        "canonical_batch_summary_digest": batch_digest,
        "canonical_batch_summary_sha256": batch_sha256,
        "split_manifest_digest": source_manifest["manifest_digest"],
        "development_fold_manifest_digest": fold_manifest["manifest_digest"],
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
    parser.add_argument("--development-fold-manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = materialize_role_sidecar(
        canonical_root=args.canonical_root,
        development_fold_manifest_path=args.development_fold_manifest,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
