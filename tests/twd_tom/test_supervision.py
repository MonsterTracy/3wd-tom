from copy import deepcopy
import json

import pytest
import torch

import script.twd_tom.materialize_role_sidecar as role_sidecar_module
from script.twd_tom.materialize_canonical_belief_dataset import (
    materialize_canonical_belief_dataset,
)
from script.twd_tom.materialize_development_folds import (
    DEVELOPMENT_FOLD_MANIFEST_FILENAME,
    materialize_development_folds,
)
from script.twd_tom.materialize_role_sidecar import (
    materialize_role_sidecar,
    validate_development_role_sidecar,
)
from werewolf.models.twd_tom.dataset import TWDToMDataset
from werewolf.models.twd_tom.supervision import (
    ALL_ALIVE_SCOPE,
    NON_WOLF_ALIVE_SCOPE,
    SPEAKER_ALIVE_SCOPE,
    VILLAGER_ALIVE_SCOPE,
    build_observer_supervision_mask,
    load_role_sidecar,
    rotate_observer_roles,
)
from werewolf.trajectory import canonical_digest, canonical_json


ROLES = {
    "player1": "Werewolf",
    "player2": "Werewolf",
    "player3": "Villager",
    "player4": "Villager",
    "player5": "Villager",
    "player6": "Seer",
    "player7": "Witch",
}


def test_supervision_scope_is_one_pure_alive_and_scope_mask():
    alive = torch.tensor([True, False, True, True, False, True, True])

    assert torch.equal(
        build_observer_supervision_mask(
            alive_mask=alive,
            observer_roles=ROLES,
            speaker_id=4,
            scope=ALL_ALIVE_SCOPE,
        ),
        alive,
    )
    assert build_observer_supervision_mask(
        alive_mask=alive,
        observer_roles=ROLES,
        speaker_id=4,
        scope=NON_WOLF_ALIVE_SCOPE,
    ).tolist() == [False, False, True, True, False, True, True]
    assert build_observer_supervision_mask(
        alive_mask=alive,
        observer_roles=ROLES,
        speaker_id=4,
        scope=VILLAGER_ALIVE_SCOPE,
    ).tolist() == [False, False, True, True, False, False, False]
    assert build_observer_supervision_mask(
        alive_mask=alive,
        observer_roles=None,
        speaker_id=4,
        scope=SPEAKER_ALIVE_SCOPE,
    ).tolist() == [False, False, False, True, False, False, False]


def test_scope_mask_allows_a_legitimate_zero_villager_boundary():
    alive = torch.tensor([True, True, False, False, False, True, True])

    result = build_observer_supervision_mask(
        alive_mask=alive,
        observer_roles=ROLES,
        speaker_id=6,
        scope=VILLAGER_ALIVE_SCOPE,
    )

    assert not result.any()


def test_role_rotation_matches_cyclic_player_rotation():
    rotated = rotate_observer_roles(ROLES, shift=2)

    assert rotated == {
        "player1": "Seer",
        "player2": "Witch",
        "player3": "Werewolf",
        "player4": "Werewolf",
        "player5": "Villager",
        "player6": "Villager",
        "player7": "Villager",
    }


def _development_role_inputs(
    tmp_path,
    suspicion_sample_factory,
    canonical_belief_batch_factory,
):
    canonical_root = tmp_path / "canonical"
    samples = {
        f"game_{index:03d}": [
            suspicion_sample_factory(game_id=f"game_{index:03d}")
        ]
        for index in range(60)
    }
    batch_summary = canonical_belief_batch_factory(canonical_root, samples)
    split_root = tmp_path / "dataset"
    split_manifest = materialize_canonical_belief_dataset(
        canonical_root=canonical_root,
        output_dir=split_root,
        split_seed=42,
        train_game_count=48,
        validation_game_count=6,
        test_game_count=6,
    )
    fold_root = tmp_path / "development_folds"
    fold_manifest = materialize_development_folds(
        train_path=split_root / "train.jsonl",
        validation_path=split_root / "validation.jsonl",
        output_dir=fold_root,
        fold_count=5,
        fold_seed=42,
    )
    return {
        "canonical_root": canonical_root,
        "batch_summary": batch_summary,
        "split_root": split_root,
        "split_manifest": split_manifest,
        "fold_root": fold_root,
        "fold_manifest": fold_manifest,
        "fold_manifest_path": (
            fold_root / DEVELOPMENT_FOLD_MANIFEST_FILENAME
        ),
    }


def _write_digest_bound_json(path, value):
    payload = dict(value)
    payload.pop("manifest_digest", None)
    payload["manifest_digest"] = canonical_digest(payload)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def test_development_role_sidecar_reads_exactly_54_and_never_opens_sealed(
    tmp_path,
    monkeypatch,
    suspicion_sample_factory,
    canonical_belief_batch_factory,
):
    inputs = _development_role_inputs(
        tmp_path,
        suspicion_sample_factory,
        canonical_belief_batch_factory,
    )
    canonical_root = inputs["canonical_root"]
    fold_manifest = inputs["fold_manifest"]
    batch_summary = inputs["batch_summary"]
    seed_positions = {
        seed: position
        for position, seed in enumerate(batch_summary["seeds"], start=1)
    }
    trajectory_paths = {
        game_id: (
            canonical_root
            / "games"
            / f"game_{seed_positions[seed]:04d}_seed_{seed}"
            / "trajectory.json"
        ).resolve()
        for game_id, seed in zip(
            batch_summary["game_ids"], batch_summary["completed_seeds"]
        )
    }
    development_ids = set(fold_manifest["development_game_ids"])
    sealed_ids = set(fold_manifest["sealed_test_game_ids"])
    development_trajectories = {
        trajectory_paths[game_id] for game_id in development_ids
    }
    sealed_trajectories = {trajectory_paths[game_id] for game_id in sealed_ids}
    opened_trajectories = []
    hashed_trajectories = []
    original_load_json = role_sidecar_module._load_json
    original_sha256 = role_sidecar_module._sha256

    def guarded_load_json(path):
        resolved = path.resolve()
        if resolved in sealed_trajectories:
            pytest.fail(f"opened sealed trajectory: {resolved}")
        if resolved.name == "trajectory.json":
            opened_trajectories.append(resolved)
        return original_load_json(path)

    def guarded_sha256(path):
        resolved = path.resolve()
        if resolved in sealed_trajectories:
            pytest.fail(f"hashed sealed trajectory: {resolved}")
        if resolved.name == "trajectory.json":
            hashed_trajectories.append(resolved)
        return original_sha256(path)

    monkeypatch.setattr(role_sidecar_module, "_load_json", guarded_load_json)
    monkeypatch.setattr(role_sidecar_module, "_sha256", guarded_sha256)
    # Absence proves lineage validation does not inspect the sealed JSONL.
    (inputs["split_root"] / "test.jsonl").unlink()
    sidecar_path = tmp_path / "role_sidecar.json"

    report = materialize_role_sidecar(
        canonical_root=canonical_root,
        development_fold_manifest_path=inputs["fold_manifest_path"],
        output_path=sidecar_path,
    )

    assert len(report["games"]) == 54
    assert set(report["games"]) == development_ids
    assert set(report["games"]).isdisjoint(sealed_ids)
    assert development_ids.isdisjoint(sealed_ids)
    assert set(opened_trajectories) == development_trajectories
    assert len(opened_trajectories) == 54
    assert set(hashed_trajectories) == development_trajectories
    assert len(hashed_trajectories) == 54
    assert report["split_manifest_digest"] == inputs["split_manifest"][
        "manifest_digest"
    ]
    assert report["canonical_batch_summary_digest"] == fold_manifest[
        "canonical_batch_summary_digest"
    ]
    assert report["development_fold_manifest_digest"] == fold_manifest[
        "manifest_digest"
    ]
    assert validate_development_role_sidecar(
        role_sidecar_path=sidecar_path,
        development_fold_manifest_path=inputs["fold_manifest_path"],
    ) == report
    roles = load_role_sidecar(sidecar_path)
    assert set(roles) == development_ids
    assert all(game_roles == ROLES for game_roles in roles.values())

    tampered = deepcopy(report)
    first_game = next(iter(tampered["games"]))
    tampered["games"][first_game]["observer_roles"]["player1"] = "Villager"
    tampered_path = tmp_path / "tampered_role_sidecar.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_role_sidecar(tampered_path)

    for mutation in ("missing", "extra"):
        changed = deepcopy(report)
        changed.pop("sidecar_digest")
        if mutation == "missing":
            changed["games"].pop(next(iter(changed["games"])))
        else:
            changed["games"]["game_extra"] = deepcopy(
                next(iter(changed["games"].values()))
            )
        changed["sidecar_digest"] = canonical_digest(changed)
        changed_path = tmp_path / f"role_sidecar_{mutation}.json"
        changed_path.write_text(
            canonical_json(changed) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="game IDs differ"):
            validate_development_role_sidecar(
                role_sidecar_path=changed_path,
                development_fold_manifest_path=inputs["fold_manifest_path"],
            )

    changed_fold = deepcopy(fold_manifest)
    changed_fold["fold_seed"] += 1
    changed_fold_path = inputs["fold_root"] / "changed_fold_manifest.json"
    _write_digest_bound_json(changed_fold_path, changed_fold)
    with pytest.raises(ValueError, match="development fold digests differ"):
        validate_development_role_sidecar(
            role_sidecar_path=sidecar_path,
            development_fold_manifest_path=changed_fold_path,
        )


def test_development_role_sidecar_lineage_failures_are_closed(
    tmp_path,
    suspicion_sample_factory,
    canonical_belief_batch_factory,
):
    inputs = _development_role_inputs(
        tmp_path,
        suspicion_sample_factory,
        canonical_belief_batch_factory,
    )
    original = inputs["fold_manifest"]
    development_ids = original["development_game_ids"]
    sealed_ids = original["sealed_test_game_ids"]
    cases = []

    canonical_mismatch = deepcopy(original)
    canonical_mismatch["canonical_batch_summary_digest"] = "0" * 64
    cases.append(("canonical_mismatch", canonical_mismatch, "canonical batch"))

    source_mismatch = deepcopy(original)
    source_mismatch["source_split_manifest_digest"] = "0" * 64
    cases.append(("source_mismatch", source_mismatch, "source split"))

    missing = deepcopy(original)
    missing["development_game_ids"] = development_ids[:-1]
    cases.append(("missing", missing, "exactly 54"))

    extra = deepcopy(original)
    extra["development_game_ids"] = development_ids + ["game_extra"]
    cases.append(("extra", extra, "exactly 54"))

    overlapping = deepcopy(original)
    overlapping["development_game_ids"] = [
        sealed_ids[0],
        *development_ids[1:],
    ]
    cases.append(("overlap", overlapping, "overlap"))

    for name, manifest, error in cases:
        manifest_path = inputs["fold_root"] / f"{name}.json"
        _write_digest_bound_json(manifest_path, manifest)
        with pytest.raises(ValueError, match=error):
            materialize_role_sidecar(
                canonical_root=inputs["canonical_root"],
                development_fold_manifest_path=manifest_path,
                output_path=tmp_path / f"roles_{name}.json",
            )


def test_non_wolf_supervision_still_intersects_label_observation(
    suspicion_sample_factory,
):
    sample = suspicion_sample_factory(observers=(1, 2, 3, 5))
    item = TWDToMDataset(
        [sample],
        observer_roles_by_game={sample["game_id"]: ROLES},
        supervision_scope=NON_WOLF_ALIVE_SCOPE,
    )[0]

    assert item["observer_alive_mask"].tolist() == [
        True,
        True,
        True,
        False,
        True,
        False,
        False,
    ]
    assert item["observer_scope_mask"].tolist() == [
        False,
        False,
        True,
        False,
        True,
        False,
        False,
    ]
    assert item["label_observed_mask"].tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert item["observer_supervision_mask"].tolist() == [
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
