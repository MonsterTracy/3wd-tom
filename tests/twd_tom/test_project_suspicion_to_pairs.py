from __future__ import annotations

import json
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader

from script.twd_tom.project_suspicion_to_pairs import (
    PROJECTED_SAMPLE_FIELDS,
    build_argument_parser,
    main,
    project_jsonl,
    project_suspicion_sample,
    validate_raw_suspicion_sample,
)
from werewolf.models.twd_tom.belief_backbone import (
    ToMBeliefBackbone,
    ToMBeliefBackboneConfig,
)
from werewolf.models.twd_tom.dataset import (
    TWDToMDataset,
    collate_twd_tom_samples,
)
from werewolf.models.twd_tom.losses import masked_pair_kl_divergence
from werewolf.models.twd_tom.samples import (
    SAMPLE_SCHEMA_VERSION as PLAYER_SUSPICION_SCHEMA_VERSION,
)
from tests.twd_tom.public_event_fixtures import public_history_fields
from werewolf.models.twd_tom.schema import (
    LABEL_PROMPT_VERSION,
    PAIR_ORDERING,
    PROJECTED_SCHEMA_VERSION,
    PROJECTION_VERSION,
    canonical_wolf_pairs,
)


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _all_ok(raw):
    raw = deepcopy(raw)
    for subject, status in raw["belief_status"].items():
        if status != "ok":
            raw["belief_status"][subject] = "ok"
            raw["belief_errors"][subject] = None
            raw["suspected_werewolves"][subject] = []
    return raw


def test_projected_row_preserves_raw_metadata_and_adds_exact_contract(
    suspicion_sample_factory,
):
    raw = suspicion_sample_factory()
    projected = project_suspicion_sample(raw)
    assert set(projected) == PROJECTED_SAMPLE_FIELDS
    assert projected["schema_version"] == PROJECTED_SCHEMA_VERSION
    assert projected["source_schema_version"] == PLAYER_SUSPICION_SCHEMA_VERSION
    assert projected["projection_version"] == PROJECTION_VERSION
    assert projected["pair_ordering"] == PAIR_ORDERING
    assert projected["label_prompt_version"] == (
        "classic7_pre_speech_player_suspicion_prompt_v2"
    )
    assert projected["label_prompt_version"] == LABEL_PROMPT_VERSION
    assert projected["target_distribution_is_reporter_probability"] is False
    assert projected["target_distribution_is_deterministic_encoding"] is True
    for key, value in raw.items():
        if key != "schema_version":
            assert projected[key] == value
    assert projected["pair_targets"]["player1"] is None
    assert len(projected["pair_targets"]["player3"]) == 21
    assert sum(projected["pair_targets"]["player3"]) == pytest.approx(1.0)
    assert raw == suspicion_sample_factory()


def test_projector_supports_empty_single_two_three_and_known_wolves(
    suspicion_sample_factory,
):
    raw = _all_ok(
        suspicion_sample_factory(observers=(1, 2, 3, 4, 5, 6, 7))
    )
    raw["suspected_werewolves"].update(
        {
            "player1": [],
            "player2": ["player3"],
            "player3": ["player1", "player2"],
            "player4": ["player1", "player2", "player3"],
            "player5": ["player1", "player2"],
            "player6": ["player1", "player2"],
            "player7": [],
        }
    )
    raw["known_werewolves"]["player5"] = ["player1"]
    raw["known_non_werewolves"]["player5"] = ["player5"]
    raw["known_werewolves"]["player6"] = ["player1", "player2"]
    raw["known_non_werewolves"]["player6"] = [
        "player3",
        "player4",
        "player5",
        "player6",
        "player7",
    ]
    projected = project_suspicion_sample(raw)
    assert set(projected["pair_targets"]) == {
        f"player{i}" for i in range(1, 8)
    }
    for target in projected["pair_targets"].values():
        assert len(target) == 21
        assert sum(target) == pytest.approx(1.0)


def test_projector_rejects_prompt_v1_and_noncanonical_full_candidates(
    suspicion_sample_factory,
):
    old_prompt = suspicion_sample_factory()
    old_prompt["label_prompt_version"] = (
        "classic7_pre_speech_player_suspicion_prompt_v1"
    )
    with pytest.raises(ValueError, match="label_prompt_version"):
        project_suspicion_sample(old_prompt)

    full_candidates = suspicion_sample_factory()
    full_candidates["suspected_werewolves"]["player3"] = [
        "player1",
        "player2",
        "player4",
        "player5",
        "player6",
        "player7",
    ]
    with pytest.raises(ValueError, match="cannot equal all legal candidates"):
        project_suspicion_sample(full_candidates)


def test_no_extra_hard_knowledge_and_wolf_reports_project_canonically(
    suspicion_sample_factory,
):
    known_one = suspicion_sample_factory(observers=(3,))
    known_one["belief_status"]["player3"] = "ok"
    known_one["belief_errors"]["player3"] = None
    known_one["known_werewolves"]["player3"] = ["player1"]
    known_one["known_non_werewolves"]["player3"] = ["player3"]
    known_one["suspected_werewolves"]["player3"] = ["player1"]
    target = project_suspicion_sample(known_one)["pair_targets"]["player3"]
    positive = [
        probability
        for probability, pair in zip(target, canonical_wolf_pairs(), strict=True)
        if "player1" in pair and "player3" not in pair
    ]
    assert len(positive) == 5
    assert positive == pytest.approx([1 / 5] * 5)
    assert sum(target) == pytest.approx(1.0)

    wolf = suspicion_sample_factory(observers=(2,))
    wolf["belief_status"]["player2"] = "ok"
    wolf["belief_errors"]["player2"] = None
    wolf["known_werewolves"]["player2"] = ["player2", "player6"]
    wolf["known_non_werewolves"]["player2"] = [
        "player1",
        "player3",
        "player4",
        "player5",
        "player7",
    ]
    wolf["suspected_werewolves"]["player2"] = ["player2", "player6"]
    wolf_target = project_suspicion_sample(wolf)["pair_targets"]["player2"]
    pair_index = canonical_wolf_pairs().index(("player2", "player6"))
    assert wolf_target[pair_index] == pytest.approx(1.0)
    assert sum(value > 0.0 for value in wolf_target) == 1


def test_project_jsonl_is_one_to_one_ordered_atomic_and_refuses_overwrite(
    tmp_path,
    suspicion_sample_factory,
):
    first = suspicion_sample_factory(game_id="first", step_idx=1)
    second = suspicion_sample_factory(game_id="second", step_idx=2)
    input_path = tmp_path / "raw.jsonl"
    output_path = tmp_path / "nested" / "projected.jsonl"
    _write_jsonl(input_path, [first, second])
    assert project_jsonl(input_path, output_path) == 2
    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["game_id"] for row in rows] == ["first", "second"]
    assert [row["step_idx"] for row in rows] == [1, 2]
    assert not (output_path.parent / f".{output_path.name}.tmp").exists()
    with pytest.raises(FileExistsError, match="already exists"):
        project_jsonl(input_path, output_path)


def test_cli_accepts_only_explicit_input_and_output(
    tmp_path,
    suspicion_sample_factory,
    capsys,
):
    input_path = tmp_path / "raw.jsonl"
    output_path = tmp_path / "projected.jsonl"
    _write_jsonl(input_path, [suspicion_sample_factory()])
    parser = build_argument_parser()
    destinations = {action.dest for action in parser._actions}
    assert destinations == {"help", "input", "output"}
    assert main(["--input", str(input_path), "--output", str(output_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["record_count"] == 1
    assert result["schema_version"] == PROJECTED_SCHEMA_VERSION
    assert output_path.is_file()


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda row: row.update(
                schema_version="classic7_pre_speech_player_suspicion_v1"
            ),
            "schema",
        ),
        (lambda row: row.update(schema_version="classic7_pre_speech_pair_belief_v1"), "schema"),
        (lambda row: row.update(schema_version="unknown"), "schema"),
        (lambda row: row.update(label_prompt_version="wrong"), "prompt"),
        (lambda row: row.update(public_event_digest="0" * 64), "digest"),
        (lambda row: row.pop("phase"), "field set"),
        (lambda row: row.update(extra_field=True), "field set"),
        (
            lambda row: row["belief_status"].pop("player3"),
            "observer keys",
        ),
        (
            lambda row: row["suspected_werewolves"].update(player3=[]),
            "contain all known",
        ),
        (
            lambda row: row["suspected_werewolves"].update(player3=["player3"]),
            "known_non_werewolves",
        ),
        (
            lambda row: row["belief_status"].update(player3="unknown"),
            "status",
        ),
        (
            lambda row: row["suspected_werewolves"].update(player3=None),
            "requires a list",
        ),
        (
            lambda row: row["suspected_werewolves"].update(player1=[]),
            "must have no suspicion",
        ),
    ],
)
def test_raw_validation_rejects_malformed_rows(
    suspicion_sample_factory,
    mutation,
    match,
):
    raw = suspicion_sample_factory()
    if "contain all known" in match:
        raw["known_werewolves"]["player3"] = ["player7"]
        raw["known_non_werewolves"]["player3"] = ["player3"]
    mutation(raw)
    with pytest.raises((TypeError, ValueError), match=match):
        validate_raw_suspicion_sample(raw)


def test_invalid_input_creates_no_output_or_partial_file(
    tmp_path,
    suspicion_sample_factory,
):
    raw = suspicion_sample_factory()
    raw["public_event_digest"] = "bad"
    input_path = tmp_path / "raw.jsonl"
    output_path = tmp_path / "new" / "projected.jsonl"
    _write_jsonl(input_path, [raw])
    with pytest.raises(ValueError, match="digest"):
        project_jsonl(input_path, output_path)
    assert not output_path.parent.exists()
    assert not output_path.exists()


def test_synthetic_end_to_end_projection_dataset_model_and_loss(
    tmp_path,
    suspicion_sample_factory,
):
    def make_row(step_idx):
        row = suspicion_sample_factory(
            game_id="synthetic",
            step_idx=step_idx,
            observers=(1, 2, 3, 4, 5),
        )
        for subject in row["belief_status"]:
            row["belief_status"][subject] = "ok"
            row["belief_errors"][subject] = None
            row["suspected_werewolves"][subject] = []
        row["known_werewolves"]["player2"] = ["player3"]
        row["suspected_werewolves"]["player2"] = ["player3"]
        row["known_werewolves"]["player3"] = ["player3", "player6"]
        row["known_non_werewolves"]["player3"] = [
            "player1",
            "player2",
            "player4",
            "player5",
            "player7",
        ]
        row["suspected_werewolves"]["player3"] = ["player3", "player6"]
        row["known_werewolves"]["player4"] = ["player3"]
        row["suspected_werewolves"]["player4"] = ["player3", "player5"]
        row["belief_status"]["player5"] = "semantic_error"
        row["belief_errors"]["player5"] = (
            "suspected_werewolves cannot equal all legal candidates unless "
            "hard knowledge already determines the full candidate set"
        )
        row["suspected_werewolves"]["player5"] = None
        return row

    first = make_row(1)
    for key, value in public_history_fields(
        [], speaker_id=first["speaker_id"]
    ).items():
        first[key] = value
    second = make_row(2)
    input_path = tmp_path / "raw.jsonl"
    output_path = tmp_path / "projected.jsonl"
    _write_jsonl(input_path, [first, second])
    assert project_jsonl(input_path, output_path) == 2

    dataset = TWDToMDataset.from_jsonl(output_path)
    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collate_twd_tom_samples,
    )
    batch = next(iter(loader))
    assert batch["pair_targets"].shape == (2, 7, 21)
    assert batch["subject_mask"].shape == (2, 7)
    assert torch.allclose(
        batch["pair_targets"][batch["subject_mask"]].sum(-1),
        torch.ones(8),
    )
    assert not batch["subject_mask"][:, 4].any()
    assert torch.count_nonzero(batch["pair_targets"][:, 4]).item() == 0
    for private_field in (
        "suspected_werewolves",
        "known_werewolves",
        "known_non_werewolves",
    ):
        assert private_field not in {
            key for key in batch if key != "metadata"
        }

    model = ToMBeliefBackbone(
        ToMBeliefBackboneConfig(
            d_model=16,
            n_head=4,
            n_layer=1,
            dropout=0.0,
            max_seq_len=8,
        )
    )
    output = model(
        batch["subject_ids"],
        batch["action_ids"],
        batch["object_ids"],
        batch["attention_mask"],
        event_type_ids=batch["event_type_ids"],
        phase_ids=batch["phase_ids"],
        day_values=batch["day_values"],
    )
    assert output["pair_logits"].shape == (2, 7, 21)
    assert output["belief_matrix"].shape == (2, 7, 7)
    assert torch.allclose(
        output["belief_matrix"].sum(-1),
        torch.full((2, 7), 2.0),
        atol=1e-6,
    )
    loss = masked_pair_kl_divergence(
        output["pair_logits"],
        batch["pair_targets"],
        batch["subject_mask"],
    )
    assert torch.isfinite(loss)
