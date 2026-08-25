import pytest
import torch
from torch.optim import AdamW

from script.twd_tom.train import (
    TrainingConfig,
    build_model,
    checkpoint_payload,
)
from werewolf.models.twd_tom.action_features import PublicEventFeatureBuilder
from werewolf.models.twd_tom.dataset import TWDToMDataset
from werewolf.models.twd_tom.inference import PrefixBeliefPredictor


def make_predictor(*, max_seq_len=32, private_conditioning=False):
    config = TrainingConfig(
        output_dir="run",
        dataset_path="train.jsonl",
        validation_dataset_path="validation.jsonl",
        epochs=1,
        batch_size=1,
        max_seq_len=max_seq_len,
        backbone="gpt2_block",
        device="cpu",
        private_conditioning=private_conditioning,
    )
    return PrefixBeliefPredictor(build_model(config))


def test_predictor_and_dataset_encode_the_same_strict_pre_prefix(
    suspicion_sample_factory,
):
    sample = suspicion_sample_factory()
    predictor = make_predictor()
    encoded = predictor.encode_prefix(
        sample["public_events"],
        sample["speech_annotations"],
        speaker_id=sample["speaker_id"],
    )
    dataset_item = TWDToMDataset(
        [sample],
        feature_builder=PublicEventFeatureBuilder(max_seq_len=32),
    )[0]

    for field_name in PublicEventFeatureBuilder.FEATURE_FIELDS:
        assert torch.equal(encoded[field_name], dataset_item[field_name])


def test_predictor_returns_stateless_relative_suspicion_matrix(
    suspicion_sample_factory,
):
    sample = suspicion_sample_factory()
    predictor = make_predictor()
    first = predictor.predict(
        sample["public_events"],
        sample["speech_annotations"],
        speaker_id=sample["speaker_id"],
    )
    second = predictor.predict(
        sample["public_events"],
        sample["speech_annotations"],
        speaker_id=sample["speaker_id"],
    )

    assert set(first) == {"belief_logits", "belief_matrix"}
    assert first["belief_logits"].shape == (7, 7)
    assert first["belief_matrix"].shape == (7, 7)
    assert torch.isfinite(first["belief_logits"]).all()
    assert torch.equal(first["belief_matrix"].diagonal(), torch.zeros(7))
    torch.testing.assert_close(
        first["belief_matrix"].sum(dim=-1),
        torch.ones(7),
    )
    torch.testing.assert_close(first["belief_logits"], second["belief_logits"])
    torch.testing.assert_close(first["belief_matrix"], second["belief_matrix"])
    assert not hasattr(predictor, "belief_state")


def test_predictor_rejects_a_non_matching_pre_boundary(
    suspicion_sample_factory,
):
    sample = suspicion_sample_factory()
    predictor = make_predictor()
    with pytest.raises(ValueError, match="matching turn_start"):
        predictor.predict(
            sample["public_events"],
            sample["speech_annotations"],
            speaker_id=3,
        )


def test_private_predictor_returns_only_the_requested_observer_row(
    suspicion_sample_factory,
):
    sample = suspicion_sample_factory()
    predictor = make_predictor(private_conditioning=True)

    result = predictor.predict_observer(
        sample["public_events"],
        sample["speech_annotations"],
        speaker_id=sample["speaker_id"],
        observer_id=1,
        known_werewolves=[],
        known_non_werewolves=["player1"],
    )

    assert set(result) == {"belief_logits", "belief_vector"}
    assert result["belief_logits"].shape == (7,)
    assert result["belief_vector"].shape == (7,)
    assert result["belief_vector"][0] == 0
    torch.testing.assert_close(result["belief_vector"].sum(), torch.tensor(1.0))
    with pytest.raises(ValueError, match="predict_observer"):
        predictor.predict(
            sample["public_events"],
            sample["speech_annotations"],
            speaker_id=sample["speaker_id"],
        )


def test_predictor_restores_checkpoint_and_runs(
    tmp_path,
    suspicion_sample_factory,
):
    config = TrainingConfig(
        output_dir="run",
        dataset_path="train.jsonl",
        validation_dataset_path="validation.jsonl",
        epochs=1,
        batch_size=1,
        max_seq_len=32,
        backbone="gpt2_block",
        device="cpu",
    )
    model = build_model(config)
    metrics = {"mean_loss": 1.0, "valid_observer_count": 1}
    payload = checkpoint_payload(
        model=model,
        optimizer=AdamW(model.parameters()),
        config=config,
        epoch=1,
        train_metrics=metrics,
        validation_metrics=metrics,
        best_epoch=1,
        best_validation_mean_loss=1.0,
        run_provenance={
            "train_dataset_path": "train.jsonl",
            "validation_dataset_path": "validation.jsonl",
            "output_dir": "run",
        },
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(payload, checkpoint_path)

    predictor = PrefixBeliefPredictor.from_checkpoint(
        checkpoint_path,
        device="cpu",
    )
    sample = suspicion_sample_factory()
    result = predictor.predict(
        sample["public_events"],
        sample["speech_annotations"],
        speaker_id=sample["speaker_id"],
    )
    assert result["belief_matrix"].shape == (7, 7)


def test_private_predictor_restores_its_explicit_checkpoint_scope(
    tmp_path,
    suspicion_sample_factory,
):
    config = TrainingConfig(
        output_dir="run",
        dataset_path="train.jsonl",
        validation_dataset_path="validation.jsonl",
        epochs=1,
        batch_size=1,
        max_seq_len=32,
        backbone="gpt2_block",
        device="cpu",
        private_conditioning=True,
    )
    model = build_model(config)
    metrics = {"mean_loss": 1.0, "valid_observer_count": 1}
    payload = checkpoint_payload(
        model=model,
        optimizer=AdamW(model.parameters()),
        config=config,
        epoch=1,
        train_metrics=metrics,
        validation_metrics=metrics,
        best_epoch=1,
        best_validation_mean_loss=1.0,
        run_provenance={
            "train_dataset_path": "train.jsonl",
            "validation_dataset_path": "validation.jsonl",
            "output_dir": "run",
        },
    )
    checkpoint_path = tmp_path / "private.pt"
    torch.save(payload, checkpoint_path)

    predictor = PrefixBeliefPredictor.from_checkpoint(
        checkpoint_path,
        device="cpu",
    )
    sample = suspicion_sample_factory()
    result = predictor.predict_observer(
        sample["public_events"],
        sample["speech_annotations"],
        speaker_id=sample["speaker_id"],
        observer_id=1,
        known_werewolves=[],
        known_non_werewolves=["player1"],
    )
    assert result["belief_vector"].shape == (7,)
