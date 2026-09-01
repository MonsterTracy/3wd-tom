import json

import torch
from torch.optim import AdamW

from script.twd_tom.export_belief_worst_cases import (
    WORST_CASE_SCHEMA_VERSION,
    aggregate_worst_case_exports,
    export_belief_worst_cases,
)
from script.twd_tom.train import (
    TrainingConfig,
    build_model,
    checkpoint_payload,
)


def test_worst_case_export_uses_the_bound_validation_targets(
    tmp_path,
    training_sample_factory,
):
    dataset_path = tmp_path / "validation.jsonl"
    dataset_path.write_text(
        json.dumps(training_sample_factory(game_id="validation_game")) + "\n",
        encoding="utf-8",
    )
    config = TrainingConfig(
        output_dir=str(tmp_path / "run"),
        dataset_path=str(dataset_path),
        validation_dataset_path=str(dataset_path),
        epochs=1,
        batch_size=1,
        backbone="gpt2_block",
        dense_supervision=True,
        device="cpu",
    )
    model = build_model(config)
    optimizer = AdamW(model.parameters())
    metrics = {"mean_loss": 1.0, "valid_observer_count": 7}
    payload = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=1,
        train_metrics=metrics,
        validation_metrics=metrics,
        best_epoch=1,
        best_validation_mean_loss=1.0,
        run_provenance={
            "train_dataset_path": "validation.jsonl",
            "validation_dataset_path": "validation.jsonl",
            "output_dir": "run",
        },
    )
    checkpoint_path = tmp_path / "best.pt"
    torch.save(payload, checkpoint_path)
    report = export_belief_worst_cases(
        config=config,
        checkpoint_path=checkpoint_path,
        output_jsonl=tmp_path / "worst.jsonl",
        output_csv=tmp_path / "worst.csv",
        limit=3,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "worst.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert report["exported_row_count"] == 3
    assert rows[0]["schema_version"] == WORST_CASE_SCHEMA_VERSION
    assert rows[0]["public_history"]
    assert rows[0]["legacy_v1_target"]
    assert rows[0]["v1_empty_uniform_nonself_target"]
    assert rows[0]["v2_target"] is None
    assert rows[0]["model_prediction"]


def test_worst_case_aggregation_reranks_across_folds(tmp_path):
    rows = []
    paths = []
    for fold, error in enumerate((0.2, 0.8)):
        row = {
            "schema_version": WORST_CASE_SCHEMA_VERSION,
            "game_id": f"game_{fold}",
            "step_idx": 1,
            "observer": "player1",
            "max_probability_error": error,
            "error_rank": 1,
        }
        path = tmp_path / f"fold_{fold}.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        paths.append(path)
        rows.append(row)
    aggregate_worst_case_exports(
        input_jsonl_paths=paths,
        output_jsonl=tmp_path / "oof.jsonl",
        output_csv=tmp_path / "oof.csv",
        limit=2,
    )
    output = [
        json.loads(line)
        for line in (tmp_path / "oof.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert output[0]["game_id"] == "game_1"
    assert [row["error_rank"] for row in output] == [1, 2]
