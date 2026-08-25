"""Strict Annotation V2 sidecar and Dataset integration tests."""

import hashlib
import json
from copy import deepcopy

import pytest
import torch

from script.twd_tom.audit_belief_label_repeatability import (
    audit_belief_label_repeatability,
)
from werewolf.models.twd_tom.annotation_v2 import (
    BELIEF_V2_INFORMATION_BOUNDARY,
    BELIEF_V2_SCHEMA_VERSION,
    SPEECH_V2_INFORMATION_BOUNDARY,
    SPEECH_V2_SCHEMA_VERSION,
    load_belief_v2_annotations,
    load_speech_v2_annotations,
    normalize_belief_v2_annotation,
    normalize_speech_v2_annotation,
)
from werewolf.models.twd_tom.dataset import (
    TWDToMDataset,
    V2_TARGET_CONVERSION,
)
from werewolf.models.twd_tom.dense_dataset import DenseTWDToMDataset
from werewolf.models.twd_tom.public_events import (
    public_event_digest,
    structured_input_digest,
)
from werewolf.models.twd_tom.speech_annotations import (
    make_speech_annotation,
    speech_annotation_digest,
)


ORDINAL_SCALE = {
    "0": "strongly_good",
    "1": "lean_good",
    "2": "unresolved",
    "3": "suspicious",
    "4": "strongly_wolf",
    "null": "self_or_hard_knowledge_separate",
}


def _with_digest(value):
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return {
        **value,
        "annotation_digest": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def _speech_v2(sample, event=None):
    event = event or next(
        item for item in sample["public_events"]
        if item["event_type"] == "public_speech"
    )
    return normalize_speech_v2_annotation(_with_digest({
        "schema_version": SPEECH_V2_SCHEMA_VERSION,
        "game_id": sample["game_id"],
        "source_game_dir": "synthetic_game",
        "speech_event_idx": event["event_idx"],
        "step_idx": 0,
        "phase": sample["phase"],
        "speaker": event["speaker"],
        "raw_text": event["raw_text"],
        "information_boundary": SPEECH_V2_INFORMATION_BOUNDARY,
        "annotation_method": "synthetic_manual_v2",
        "annotation_confidence": "high",
        "manual_reviewed": True,
        "review_required": False,
        "manual_review_note": "synthetic strict fixture",
        "review_provenance": {},
        "auto_candidate_claims_before_full_review": [],
        "auto_candidate_compat_actions_before_full_review": [],
        "claims": [],
        "compat_actions": [[event["speaker"], "support", "player4"]],
        "integrity_flags": [],
    }))


def _later_snapshot(sample):
    later = deepcopy(sample)
    later["step_idx"] += 1
    later["label_cutoff_step_idx"] = later["step_idx"]
    prior_speaker = f"player{later['speaker_id']}"
    event_idx = len(later["public_events"])
    later["public_events"].extend([
        {
            "event_idx": event_idx,
            "event_type": "public_speech",
            "speaker": prior_speaker,
            "raw_text": "second frozen speech",
        },
        {
            "event_idx": event_idx + 1,
            "event_type": "turn_start",
            "speaker": "player3",
        },
    ])
    later["speech_annotations"].append(make_speech_annotation(
        event_idx=event_idx,
        speaker=prior_speaker,
        raw_text="second frozen speech",
        parser_model_id="synthetic_parser",
        parser_call_id="synthetic_second",
        annotation_source="llm_parser",
        status="ok",
        actions=[[prior_speaker, "oppose", "player4"]],
        raw_response=None,
        error_type=None,
        error_message=None,
    ))
    later["speaker_id"] = 3
    later["public_action_count"] += 1
    later["public_event_digest"] = public_event_digest(later["public_events"])
    later["speech_annotation_digest"] = speech_annotation_digest(
        later["speech_annotations"]
    )
    later["structured_input_digest"] = structured_input_digest(
        later["public_events"], later["speech_annotations"]
    )
    return later


def _belief_v2(sample, observer, *, observed=True, role="Villager"):
    candidates = [f"player{i}" for i in range(1, 8) if f"player{i}" != observer]
    target = candidates[0]
    distribution = {f"player{i}": 0.0 for i in range(1, 8)}
    if observed:
        distribution[target] = 1.0
    return normalize_belief_v2_annotation(_with_digest({
        "schema_version": BELIEF_V2_SCHEMA_VERSION,
        "game_id": sample["game_id"],
        "source_game_dir": "synthetic_game",
        "step_idx": sample["step_idx"],
        "phase": sample["phase"],
        "day": 1,
        "current_speaker": f"player{sample['speaker_id']}",
        "observer": observer,
        "observer_role": role,
        "is_current_speaker": observer == f"player{sample['speaker_id']}",
        "public_action_count": sample["public_action_count"],
        "public_event_digest": sample["public_event_digest"],
        "hard_knowledge": {
            "known_werewolves_other": [],
            "known_non_werewolves_other": [],
        },
        "information_boundary": BELIEF_V2_INFORMATION_BOUNDARY,
        "annotation_method": "synthetic_manual_v2",
        "annotation_confidence": "high",
        "review_required": False,
        "constraint_violations": [],
        "training_recommendation": {
            "compat_relative_suspicion_distribution": distribution,
            "compat_suspected_werewolves": [target] if observed else [],
            "distribution_loss_mask": observed,
            "ordinal_or_pairwise_preferred": True,
        },
        "v2_label": {
            "evidence_event_ids": [],
            "insufficient_evidence_or_abstain": not observed,
            "ordinal_derivation_reasons": {
                player: "synthetic fixture" for player in candidates
            },
            "ordinal_scale": ORDINAL_SCALE,
            "pairwise_suspicion_relations": [],
            "subjective_suspicion_ordinal": {
                player: (4 if observed and player == target else 2)
                for player in candidates
            },
        },
    }))


def test_v2_jsonl_loaders_reject_digest_tampering(
    tmp_path,
    suspicion_sample_factory,
):
    sample = suspicion_sample_factory()
    speech = _speech_v2(sample)
    belief = _belief_v2(sample, "player1")
    speech_path = tmp_path / "speech.jsonl"
    belief_path = tmp_path / "belief.jsonl"
    speech_path.write_text(json.dumps(speech) + "\n", encoding="utf-8")
    belief_path.write_text(json.dumps(belief) + "\n", encoding="utf-8")
    assert len(load_speech_v2_annotations(speech_path)) == 1
    assert len(load_belief_v2_annotations(belief_path)) == 1

    belief["phase"] = "2_day_speech"
    belief_path.write_text(json.dumps(belief) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="day differs from phase"):
        load_belief_v2_annotations(belief_path)


def test_dataset_selects_v2_speech_and_loss_mask_without_uniform_imputation(
    suspicion_sample_factory,
):
    sample = suspicion_sample_factory()
    game_roles = {
        "player1": "Werewolf",
        "player2": "Werewolf",
        "player3": "Villager",
        "player4": "Villager",
        "player5": "Villager",
        "player6": "Seer",
        "player7": "Witch",
    }
    roles = {sample["game_id"]: game_roles}
    speech = _speech_v2(sample)
    beliefs = {
        (sample["game_id"], sample["step_idx"], observer): _belief_v2(
            sample,
            observer,
            observed=observer != "player5",
            role=game_roles[observer],
        )
        for observer in sample["suspected_werewolves"]
    }
    baseline = TWDToMDataset([sample])[0]
    item = TWDToMDataset(
        [sample],
        observer_roles_by_game=roles,
        speech_annotation_source="v2",
        belief_annotation_source="v2",
        speech_v2_annotations={
            (sample["game_id"], speech["speech_event_idx"]): speech
        },
        belief_v2_annotations=beliefs,
    )[0]

    assert item["metadata"]["target_conversion"] == V2_TARGET_CONVERSION
    assert item["metadata"]["speech_annotation_source"] == "v2"
    assert not torch.equal(item["object_ids"], baseline["object_ids"])
    assert not item["label_observed_mask"][4]
    assert torch.count_nonzero(item["belief_targets"][4]) == 0
    assert not item["observer_supervision_mask"][4]
    torch.testing.assert_close(item["belief_targets"], item["v2_belief_targets"])
    torch.testing.assert_close(
        item["v1_empty_unobserved_belief_targets"],
        baseline["belief_targets"],
    )


def test_v2_dataset_fails_closed_when_sidecar_row_is_missing(
    suspicion_sample_factory,
):
    sample = suspicion_sample_factory()
    with pytest.raises(ValueError, match="has no record"):
        TWDToMDataset(
            [sample],
            belief_annotation_source="v2",
            belief_v2_annotations={},
        )


def test_v2_speech_fails_closed_when_phase_differs_from_public_history(
    suspicion_sample_factory,
):
    sample = suspicion_sample_factory()
    speech = _speech_v2(sample)
    speech["phase"] = "2_day_speech"

    with pytest.raises(ValueError, match="phase differs"):
        TWDToMDataset(
            [sample],
            speech_annotation_source="v2",
            speech_v2_annotations={
                (sample["game_id"], speech["speech_event_idx"]): speech
            },
        )


def test_dense_v2_rotation_keeps_all_boundaries_and_targets_aligned(
    suspicion_sample_factory,
):
    first = suspicion_sample_factory()
    second = _later_snapshot(first)
    speech_records = {}
    for event in second["public_events"]:
        if event["event_type"] == "public_speech":
            record = _speech_v2(second, event)
            speech_records[(second["game_id"], event["event_idx"])] = record
    belief_records = {}
    for sample in (first, second):
        for observer in sample["suspected_werewolves"]:
            belief_records[(sample["game_id"], sample["step_idx"], observer)] = (
                _belief_v2(sample, observer)
            )
    baseline = DenseTWDToMDataset(
        [first, second],
        speech_annotation_source="v2",
        belief_annotation_source="v2",
        speech_v2_annotations=speech_records,
        belief_v2_annotations=belief_records,
    )[0]
    rotated = DenseTWDToMDataset(
        [first, second],
        enable_cyclic_rotation=True,
        augmentation_seed=2,
        speech_annotation_source="v2",
        belief_annotation_source="v2",
        speech_v2_annotations=speech_records,
        belief_v2_annotations=belief_records,
    )[0]

    torch.testing.assert_close(
        rotated["belief_targets"],
        torch.roll(baseline["belief_targets"], shifts=(2, 2), dims=(1, 2)),
    )
    assert torch.equal(
        rotated["label_observed_mask"],
        torch.roll(baseline["label_observed_mask"], shifts=2, dims=1),
    )


def test_repeatability_audit_keeps_abstention_out_of_distribution_metrics(
    tmp_path,
    suspicion_sample_factory,
):
    sample = suspicion_sample_factory(observers=(1, 2))
    roles = {"player1": "Werewolf", "player2": "Werewolf"}
    records = [
        _belief_v2(
            sample,
            observer,
            observed=observer == "player1",
            role=roles[observer],
        )
        for observer in sample["suspected_werewolves"]
    ]
    paths = []
    for replicate_index in range(3):
        path = tmp_path / f"replicate_{replicate_index}.jsonl"
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        paths.append(path)
    result = audit_belief_label_repeatability(
        replicate_paths=paths,
        output_path=tmp_path / "summary.json",
        per_state_jsonl_path=tmp_path / "states.jsonl",
        per_state_csv_path=tmp_path / "states.csv",
    )

    assert result["state_count"] == 2
    assert result["overall"]["pair_count"] == 6
    assert result["overall"]["observed_pair_count"] == 3
    assert result["overall"][
        "all_replicates_exact_support_agreement_rate"
    ] == 1.0
