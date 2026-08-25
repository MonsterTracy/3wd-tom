import json

import pytest
import torch

from script.twd_tom.materialize_canonical_belief_dataset import (
    materialize_canonical_belief_dataset,
)
from script.twd_tom.materialize_role_sidecar import materialize_role_sidecar
from werewolf.models.twd_tom.supervision import (
    ALL_ALIVE_SCOPE,
    NON_WOLF_ALIVE_SCOPE,
    SPEAKER_ALIVE_SCOPE,
    VILLAGER_ALIVE_SCOPE,
    build_observer_supervision_mask,
    load_role_sidecar,
    rotate_observer_roles,
)


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


def test_role_sidecar_is_digest_bound_and_complete(
    tmp_path,
    suspicion_sample_factory,
    canonical_belief_batch_factory,
):
    canonical_root = tmp_path / "canonical"
    samples = {
        game_id: [suspicion_sample_factory(game_id=game_id)]
        for game_id in ("game_a", "game_b", "game_c")
    }
    canonical_belief_batch_factory(canonical_root, samples)
    split_root = tmp_path / "dataset"
    manifest = materialize_canonical_belief_dataset(
        canonical_root=canonical_root,
        output_dir=split_root,
        split_seed=42,
        train_game_count=1,
        validation_game_count=1,
        test_game_count=1,
    )
    sidecar_path = tmp_path / "role_sidecar.json"

    report = materialize_role_sidecar(
        canonical_root=canonical_root,
        split_manifest_path=split_root / "split_manifest.json",
        output_path=sidecar_path,
    )

    assert report["split_manifest_digest"] == manifest["manifest_digest"]
    roles = load_role_sidecar(sidecar_path)
    assert set(roles) == set(samples)
    assert all(game_roles == ROLES for game_roles in roles.values())

    tampered = json.loads(sidecar_path.read_text(encoding="utf-8"))
    tampered["games"]["game_a"]["observer_roles"]["player1"] = "Villager"
    sidecar_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_role_sidecar(sidecar_path)
