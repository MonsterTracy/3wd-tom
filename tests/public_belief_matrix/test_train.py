from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader

from script.public_belief_matrix.train import (
    PublicBeliefMatrixTrainingConfig,
    build_model_from_checkpoint,
    run_epoch,
    save_checkpoint,
)
from werewolf.models.public_belief_matrix.backbone import (
    PublicBeliefMatrixBackbone,
    PublicBeliefMatrixBackboneConfig,
)
from werewolf.models.public_belief_matrix.collection import (
    PUBLIC_BELIEF_MATRIX_SAMPLE_SCHEMA_VERSION,
)
from werewolf.models.public_belief_matrix.dataset import (
    PUBLIC_BELIEF_MATRIX_MATERIALIZATION_VERSION,
    PublicBeliefMatrixDataset,
    collate_public_belief_matrix_batch,
)
from werewolf.models.public_belief_matrix.losses import (
    masked_row_soft_target_cross_entropy,
)
from werewolf.models.public_belief_matrix.targets import (
    suspicion_reports_to_matrix_target,
)


def _record(sample):
    target = suspicion_reports_to_matrix_target(sample["observer_reports"])
    return {
        "materialization_version": PUBLIC_BELIEF_MATRIX_MATERIALIZATION_VERSION,
        "source_schema_version": PUBLIC_BELIEF_MATRIX_SAMPLE_SCHEMA_VERSION,
        "seed": 876,
        "sample": sample,
        "matrix_target": [list(row) for row in target.matrix_target],
        "observer_row_mask": list(target.observer_row_mask),
    }


def _loader(sample):
    return DataLoader(
        PublicBeliefMatrixDataset([_record(sample)]),
        batch_size=1,
        collate_fn=collate_public_belief_matrix_batch,
    )


def test_one_batch_forward_loss_and_backward_are_finite(pbm_sample_factory):
    torch.manual_seed(7)
    model = PublicBeliefMatrixBackbone(PublicBeliefMatrixBackboneConfig(max_seq_len=256))
    batch = next(iter(_loader(pbm_sample_factory())))
    output = model(**{key: batch[key] for key in (
        "subject_ids", "action_ids", "object_ids", "event_type_ids",
        "phase_ids", "day_values", "attention_mask",
    )})
    loss = masked_row_soft_target_cross_entropy(
        output["matrix_logits"], batch["matrix_target"], batch["observer_row_mask"]
    )
    assert output["matrix_logits"].shape == (1, 7, 7)
    assert output["matrix_probabilities"].shape == (1, 7, 7)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_all_masked_loss_fails_closed():
    with pytest.raises(ValueError, match="at least one valid"):
        masked_row_soft_target_cross_entropy(
            torch.zeros(1, 7, 7), torch.zeros(1, 7, 7), torch.zeros(1, 7, dtype=torch.bool)
        )


def test_validation_epoch_does_not_update_parameters(pbm_sample_factory):
    torch.manual_seed(8)
    model = PublicBeliefMatrixBackbone(PublicBeliefMatrixBackboneConfig(max_seq_len=256))
    before = deepcopy(model.state_dict())
    metrics = run_epoch(model, _loader(pbm_sample_factory()), device=torch.device("cpu"), optimizer=None)
    assert metrics["loss"] >= 0
    for name, value in before.items():
        torch.testing.assert_close(value, model.state_dict()[name])


def test_checkpoint_round_trip_is_strict(tmp_path):
    torch.manual_seed(9)
    model = PublicBeliefMatrixBackbone(PublicBeliefMatrixBackboneConfig(max_seq_len=256))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    config = PublicBeliefMatrixTrainingConfig(epochs=1)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        training_config=config,
        split_manifest_path=tmp_path / "manifest.json",
    )
    restored, checkpoint = build_model_from_checkpoint(checkpoint_path)
    assert restored.config == model.config
    assert checkpoint["epoch"] == 1
    assert restored.state_dict().keys() == model.state_dict().keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name])
