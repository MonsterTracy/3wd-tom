from copy import deepcopy

import pytest
import torch

from werewolf.models.public_belief_matrix.collection import (
    PUBLIC_BELIEF_MATRIX_SAMPLE_SCHEMA_VERSION,
)
from werewolf.models.public_belief_matrix.dataset import (
    PUBLIC_BELIEF_MATRIX_MATERIALIZATION_VERSION,
    PublicBeliefMatrixDataset,
    collate_public_belief_matrix_batch,
)
from werewolf.models.public_belief_matrix.targets import (
    suspicion_reports_to_matrix_target,
)
from werewolf.models.twd_tom.action_features import PublicEventFeatureBuilder


def _record(sample, seed=876):
    target = suspicion_reports_to_matrix_target(sample["observer_reports"])
    return {
        "materialization_version": PUBLIC_BELIEF_MATRIX_MATERIALIZATION_VERSION,
        "source_schema_version": PUBLIC_BELIEF_MATRIX_SAMPLE_SCHEMA_VERSION,
        "seed": seed,
        "sample": sample,
        "matrix_target": [list(row) for row in target.matrix_target],
        "observer_row_mask": list(target.observer_row_mask),
    }


def test_dataset_returns_exact_prefix_target_mask_and_audit_metadata(pbm_sample_factory):
    sample = pbm_sample_factory()
    sample["observer_reports"][0]["suspected_werewolves"] = ["player3", "player5"]
    sample["observer_reports"][1].update(
        status="parse_error", suspected_werewolves=None, error="bad"
    )
    dataset = PublicBeliefMatrixDataset([_record(sample)])
    item = dataset[0]

    assert set(PublicEventFeatureBuilder.FEATURE_FIELDS).issubset(item)
    assert all(item[field].ndim == 1 for field in PublicEventFeatureBuilder.FEATURE_FIELDS)
    assert item["matrix_target"].shape == (7, 7)
    assert item["matrix_target"].dtype == torch.float32
    assert item["observer_row_mask"].shape == (7,)
    assert item["observer_row_mask"].dtype == torch.bool
    assert item["observer_row_mask"].tolist() == [True, False, True, True, True, True, True]
    assert item["matrix_target"][0].tolist() == [0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0]
    assert item["matrix_target"][1].count_nonzero().item() == 0
    torch.testing.assert_close(item["matrix_target"][2], torch.full((7,), 1.0 / 7.0))
    assert item["metadata"] == {
        "game_id": sample["game_id"],
        "snapshot_id": sample["snapshot_id"],
        "seed": 876,
    }
    expected = suspicion_reports_to_matrix_target(sample["observer_reports"])
    torch.testing.assert_close(
        item["matrix_target"], torch.tensor(expected.matrix_target, dtype=torch.float32)
    )


@pytest.mark.parametrize("mutation", ["missing", "unknown", "unequal"])
def test_dataset_rejects_prefix_contract_drift(pbm_sample_factory, mutation):
    sample = pbm_sample_factory()
    if mutation == "missing":
        del sample["structured_prefix"]["action_ids"]
    elif mutation == "unknown":
        sample["structured_prefix"]["extra"] = []
    else:
        sample["structured_prefix"]["action_ids"].append(0)
    with pytest.raises((ValueError, TypeError)):
        PublicBeliefMatrixDataset([_record(sample)])


def test_dataset_rejects_raw_truth_and_all_invalid_rows(pbm_sample_factory):
    private = pbm_sample_factory()
    private["true_roles"] = ["werewolf"]
    with pytest.raises(ValueError, match="forbidden"):
        PublicBeliefMatrixDataset([_record(private)])

    invalid = pbm_sample_factory()
    for report in invalid["observer_reports"]:
        report.update(status="reporter_error", suspected_werewolves=None, error="down")
    with pytest.raises(ValueError, match="at least one valid"):
        PublicBeliefMatrixDataset([_record(invalid)])


def test_materialized_target_tampering_fails_closed(pbm_sample_factory):
    record = _record(pbm_sample_factory())
    record["matrix_target"][0][0] = 1.0
    with pytest.raises(ValueError, match="does not match"):
        PublicBeliefMatrixDataset([record])

    record = _record(pbm_sample_factory())
    record["seed"] = 877
    with pytest.raises(ValueError, match="seed does not match"):
        PublicBeliefMatrixDataset([record])


def test_collate_pads_prefix_only_and_keeps_metadata_separate(pbm_sample_factory):
    first = PublicBeliefMatrixDataset([_record(pbm_sample_factory())])[0]
    second_record = _record(pbm_sample_factory(snapshot_number=2))
    second_record["sample"] = deepcopy(second_record["sample"])
    dataset = PublicBeliefMatrixDataset([second_record])
    batch = collate_public_belief_matrix_batch([first, dataset[0]])
    assert batch["subject_ids"].ndim == 2
    assert batch["matrix_target"].shape == (2, 7, 7)
    assert batch["observer_row_mask"].shape == (2, 7)
    assert isinstance(batch["metadata"], list)
