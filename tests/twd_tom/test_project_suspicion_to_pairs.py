from __future__ import annotations

import json
from copy import deepcopy

import pytest

from archive.legacy_tom.script.twd_tom.project_suspicion_to_pairs import (
    PROJECTED_SAMPLE_FIELDS,
    build_argument_parser,
    main,
    project_jsonl,
    project_suspicion_sample,
    validate_raw_suspicion_sample,
)
from werewolf.models.twd_tom.samples import (
    PUBLIC_ONLY_SAMPLE_SCHEMA_VERSION,
    SAMPLE_SCHEMA_VERSION as PLAYER_SUSPICION_SCHEMA_VERSION,
)
from werewolf.models.twd_tom.schema import (
    LABEL_PROMPT_VERSION,
    PAIR_ORDERING,
    PROJECTED_SCHEMA_VERSION,
    PROJECTION_VERSION,
    PUBLIC_ONLY_LABEL_PROMPT_VERSION,
    PUBLIC_ONLY_LABEL_PROVENANCE,
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


def _public_only(raw):
    raw = _all_ok(raw)
    raw["schema_version"] = PUBLIC_ONLY_SAMPLE_SCHEMA_VERSION
    raw["label_prompt_version"] = PUBLIC_ONLY_LABEL_PROMPT_VERSION
    raw["label_provenance"] = PUBLIC_ONLY_LABEL_PROVENANCE
    raw["known_werewolves"] = {
        subject: [] for subject in raw["known_werewolves"]
    }
    raw["known_non_werewolves"] = {
        subject: [] for subject in raw["known_non_werewolves"]
    }
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


def test_public_only_exact_contract_projects_without_hard_knowledge(
    suspicion_sample_factory,
):
    raw = _public_only(suspicion_sample_factory(observers=(3,)))
    raw["suspected_werewolves"]["player3"] = ["player2", "player5"]
    projected = project_suspicion_sample(raw)
    target = projected["pair_targets"]["player3"]
    assert projected["source_schema_version"] == (
        PUBLIC_ONLY_SAMPLE_SCHEMA_VERSION
    )
    assert len(target) == 21
    assert sum(target) == pytest.approx(1.0)
    pair_mass = dict(zip(canonical_wolf_pairs(), target, strict=True))
    assert pair_mass[("player1", "player2")] / pair_mass[
        ("player1", "player3")
    ] == pytest.approx(2.0)
    assert pair_mass[("player2", "player5")] / pair_mass[
        ("player1", "player3")
    ] == pytest.approx(4.0)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update(label_provenance=PUBLIC_ONLY_LABEL_PROVENANCE),
        lambda row: row.update(
            schema_version=PUBLIC_ONLY_SAMPLE_SCHEMA_VERSION,
            label_prompt_version=PUBLIC_ONLY_LABEL_PROMPT_VERSION,
        ),
    ],
)
def test_private_public_cross_contracts_are_rejected(
    suspicion_sample_factory,
    mutation,
):
    raw = suspicion_sample_factory()
    mutation(raw)
    with pytest.raises(ValueError, match="source contract tuple"):
        validate_raw_suspicion_sample(raw)


def test_project_jsonl_rejects_mixed_source_lineages(
    tmp_path,
    suspicion_sample_factory,
):
    private = suspicion_sample_factory(game_id="private")
    public = _public_only(suspicion_sample_factory(game_id="public"))
    input_path = tmp_path / "mixed.jsonl"
    output_path = tmp_path / "projected.jsonl"
    _write_jsonl(input_path, [private, public])
    with pytest.raises(ValueError, match="cannot mix"):
        project_jsonl(input_path, output_path)
    assert not output_path.exists()


def test_projector_rejects_prompt_v1_and_projects_full_candidates(
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
    projected = project_suspicion_sample(full_candidates)
    target = projected["pair_targets"]["player3"]
    assert len(target) == 21
    assert all(probability >= 0.0 for probability in target)
    assert sum(target) == pytest.approx(1.0)


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
