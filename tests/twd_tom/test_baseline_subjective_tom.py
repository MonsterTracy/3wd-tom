import json

import pytest
import torch

from script.twd_tom.baseline import (
    PROBABILITY_FLOOR,
    BaselineConfig,
    build_arg_parser,
    build_uniform_pair_probabilities,
    fit_observer_empirical_pair_prior,
    probabilities_to_finite_logits,
    run_baselines,
)
from script.twd_tom.project_suspicion_to_pairs import project_suspicion_sample
from werewolf.models.twd_tom.belief_labels import pair_probabilities_to_belief_marginals
from werewolf.models.twd_tom.dataset import TWDToMDataset, collate_twd_tom_samples
from torch.utils.data import DataLoader


def write_jsonl(path, samples):
    path.write_text("".join(json.dumps(sample) + "\n" for sample in samples), encoding="utf-8")


def test_uniform_pair_distribution_and_marginals():
    probabilities = build_uniform_pair_probabilities()
    assert probabilities.shape == (7, 21)
    assert probabilities.dtype == torch.float64
    assert torch.equal(probabilities, torch.full_like(probabilities, 1 / 21))
    marginals = pair_probabilities_to_belief_marginals(probabilities)
    assert torch.all(marginals.diagonal() > 0)
    assert torch.allclose(marginals.sum(-1), torch.full((7,), 2.0, dtype=torch.float64))


def test_probability_conversion_is_finite_and_exact():
    probabilities = build_uniform_pair_probabilities()
    logits = probabilities_to_finite_logits(probabilities)
    assert PROBABILITY_FLOOR > 0
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(torch.softmax(logits, -1), probabilities, rtol=1e-12, atol=1e-12)


def test_empirical_prior_fits_only_valid_training_rows(projected_sample_factory):
    samples = [projected_sample_factory(game_id=f"train_{i}") for i in range(2)]
    loader = DataLoader(
        TWDToMDataset(samples, target_dtype=torch.float64), batch_size=2,
        collate_fn=collate_twd_tom_samples,
    )
    with pytest.raises(ValueError, match="missing observers"):
        fit_observer_empirical_pair_prior(loader)


def test_run_baselines_reports_all_valid_rows(tmp_path, suspicion_sample_factory):
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    output = tmp_path / "summary.json"
    raw_train = suspicion_sample_factory(game_id="train", observers=tuple(range(1, 8)))
    raw_validation = suspicion_sample_factory(
        game_id="validation", observers=tuple(range(1, 8))
    )
    for sample in (raw_train, raw_validation):
        sample["belief_status"]["player1"] = "ok"
        sample["suspected_werewolves"]["player1"] = []
        sample["belief_errors"]["player1"] = None
    train_sample = project_suspicion_sample(raw_train)
    validation_sample = project_suspicion_sample(raw_validation)
    write_jsonl(train, [train_sample])
    write_jsonl(validation, [validation_sample])
    summary = run_baselines(BaselineConfig(str(train), str(validation), str(output), batch_size=1))
    assert output.is_file()
    assert summary["train_data_audit"]["raw_record_count"] == 1
    assert summary["validation_data_audit"]["valid_subject_count"] == 7
    for baseline in summary["baselines"].values():
        metrics = baseline["validation_metrics"]
        assert metrics["valid_subject_count"] == 7


def test_baseline_cli_has_no_test_or_truth_inputs():
    parser = build_arg_parser()
    destinations = {action.dest for action in parser._actions}
    assert not destinations & {"test", "test_dataset", "roles", "truth", "actual_wolves"}
