"""Strict loaders and adapters for immutable Annotation V2 sidecars."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from werewolf.models.twd_tom.public_events import (
    parse_public_phase,
    public_speech_actions,
    structured_input_digest,
)
from werewolf.models.twd_tom.schema import (
    NUM_PLAYERS,
    PLAYER_NAMES,
    PLAYER_TO_ID,
    parse_speech_action,
)
from werewolf.models.twd_tom.speech_annotations import (
    SPEECH_ANNOTATION_SCHEMA_VERSION,
    STATUS_NO_ACTION,
    STATUS_OK,
    make_speech_annotation,
    speech_annotation_digest,
)


V1_ANNOTATION_SOURCE = "v1"
V2_ANNOTATION_SOURCE = "v2"
LEGACY_V1_BELIEF_SOURCE = "legacy_v1"
V1_EMPTY_UNIFORM_NONSELF_BELIEF_SOURCE = "v1_empty_uniform_nonself"
SPEECH_ANNOTATION_SOURCES = (V1_ANNOTATION_SOURCE, V2_ANNOTATION_SOURCE)
BELIEF_ANNOTATION_SOURCES = (
    LEGACY_V1_BELIEF_SOURCE,
    V1_EMPTY_UNIFORM_NONSELF_BELIEF_SOURCE,
    V2_ANNOTATION_SOURCE,
)

SPEECH_V2_SCHEMA_VERSION = "classic7_speech_annotation_v2_public_only_v1"
BELIEF_V2_SCHEMA_VERSION = "classic7_belief_annotation_v2_all_observers_v1"
SPEECH_V2_INFORMATION_BOUNDARY = (
    "actual_public_speech_text_plus_public_phase_and_speaker_id_only"
)
BELIEF_V2_INFORMATION_BOUNDARY = (
    "observer_legal_private_state_plus_public_history_strictly_before_target_speech"
)

_SPEECH_V2_FIELDS = {
    "annotation_confidence",
    "annotation_digest",
    "annotation_method",
    "auto_candidate_claims_before_full_review",
    "auto_candidate_compat_actions_before_full_review",
    "claims",
    "compat_actions",
    "game_id",
    "information_boundary",
    "integrity_flags",
    "manual_review_note",
    "manual_reviewed",
    "phase",
    "raw_text",
    "review_provenance",
    "review_required",
    "schema_version",
    "source_game_dir",
    "speaker",
    "speech_event_idx",
    "step_idx",
}
_BELIEF_V2_FIELDS = {
    "annotation_confidence",
    "annotation_digest",
    "annotation_method",
    "constraint_violations",
    "current_speaker",
    "day",
    "game_id",
    "hard_knowledge",
    "information_boundary",
    "is_current_speaker",
    "observer",
    "observer_role",
    "phase",
    "public_action_count",
    "public_event_digest",
    "review_required",
    "schema_version",
    "source_game_dir",
    "step_idx",
    "training_recommendation",
    "v2_label",
}
_TRAINING_RECOMMENDATION_FIELDS = {
    "compat_relative_suspicion_distribution",
    "compat_suspected_werewolves",
    "distribution_loss_mask",
    "ordinal_or_pairwise_preferred",
}
_V2_LABEL_FIELDS = {
    "evidence_event_ids",
    "insufficient_evidence_or_abstain",
    "ordinal_derivation_reasons",
    "ordinal_scale",
    "pairwise_suspicion_relations",
    "subjective_suspicion_ordinal",
}
_ORDINAL_SCALE = {
    "0": "strongly_good",
    "1": "lean_good",
    "2": "unresolved",
    "3": "suspicious",
    "4": "strongly_wolf",
    "null": "self_or_hard_knowledge_separate",
}
_ROLE_NAMES = {"Werewolf", "Villager", "Seer", "Witch"}
_CONFIDENCE_NAMES = {"low", "medium", "high"}


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    field_name: str,
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(
            f"{field_name} field set mismatch; missing={missing}, extra={extra}"
        )


def _require_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return value


def _require_non_negative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _require_sequence(value: Any, *, field_name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence")
    return value


def _validate_annotation_digest(value: Mapping[str, Any]) -> None:
    digest = value.get("annotation_digest")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("annotation_digest must be a lowercase SHA-256 digest")
    payload = dict(value)
    payload.pop("annotation_digest")
    if digest != _canonical_digest(payload):
        raise ValueError("annotation_digest does not match annotation payload")


def _normalize_actions(value: Any, *, speaker: str) -> list[list[str | None]]:
    actions = []
    for raw_action in _require_sequence(value, field_name="compat_actions"):
        action = parse_speech_action(raw_action)
        if action.subject != speaker:
            raise ValueError("V2 speech action subject must equal speaker")
        actions.append(action.to_list())
    if len({tuple(action) for action in actions}) != len(actions):
        raise ValueError("V2 speech compat_actions cannot contain duplicates")
    return actions


def normalize_speech_v2_annotation(value: Any) -> dict[str, Any]:
    """Validate one public-only semantic speech annotation."""

    if not isinstance(value, Mapping):
        raise TypeError("V2 speech annotation must be a mapping")
    _require_exact_fields(value, _SPEECH_V2_FIELDS, field_name="V2 speech")
    if value["schema_version"] != SPEECH_V2_SCHEMA_VERSION:
        raise ValueError("unsupported V2 speech schema_version")
    if value["information_boundary"] != SPEECH_V2_INFORMATION_BOUNDARY:
        raise ValueError("unsupported V2 speech information boundary")
    game_id = _require_text(value["game_id"], field_name="game_id")
    _require_text(value["source_game_dir"], field_name="source_game_dir")
    _require_text(value["annotation_method"], field_name="annotation_method")
    _require_text(value["manual_review_note"], field_name="manual_review_note")
    phase = _require_text(value["phase"], field_name="phase")
    parse_public_phase(phase)
    speaker = value["speaker"]
    if speaker not in PLAYER_NAMES:
        raise ValueError("V2 speech speaker must be canonical")
    _require_non_negative_int(value["speech_event_idx"], field_name="speech_event_idx")
    _require_non_negative_int(value["step_idx"], field_name="step_idx")
    if not isinstance(value["raw_text"], str):
        raise TypeError("V2 speech raw_text must be text")
    if value["annotation_confidence"] not in _CONFIDENCE_NAMES:
        raise ValueError("unsupported V2 speech annotation_confidence")
    for field_name in ("manual_reviewed", "review_required"):
        if not isinstance(value[field_name], bool):
            raise TypeError(f"V2 speech {field_name} must be bool")
    if not isinstance(value["review_provenance"], Mapping):
        raise TypeError("V2 speech review_provenance must be a mapping")
    for field_name in (
        "auto_candidate_claims_before_full_review",
        "auto_candidate_compat_actions_before_full_review",
        "claims",
        "integrity_flags",
    ):
        _require_sequence(value[field_name], field_name=field_name)
    normalized = deepcopy(dict(value))
    normalized["game_id"] = game_id
    normalized["compat_actions"] = _normalize_actions(
        value["compat_actions"], speaker=speaker
    )
    _validate_annotation_digest(value)
    return normalized


def _normalize_player_list(value: Any, *, field_name: str) -> list[str]:
    players = list(_require_sequence(value, field_name=field_name))
    if any(player not in PLAYER_NAMES for player in players):
        raise ValueError(f"{field_name} must contain canonical player IDs")
    if len(set(players)) != len(players):
        raise ValueError(f"{field_name} cannot contain duplicates")
    expected = sorted(players, key=PLAYER_TO_ID.__getitem__)
    if players != expected:
        raise ValueError(f"{field_name} must use canonical player order")
    return players


def normalize_belief_v2_annotation(value: Any) -> dict[str, Any]:
    """Validate one all-observer V2 belief annotation and loss mask."""

    if not isinstance(value, Mapping):
        raise TypeError("V2 belief annotation must be a mapping")
    _require_exact_fields(value, _BELIEF_V2_FIELDS, field_name="V2 belief")
    if value["schema_version"] != BELIEF_V2_SCHEMA_VERSION:
        raise ValueError("unsupported V2 belief schema_version")
    if value["information_boundary"] != BELIEF_V2_INFORMATION_BOUNDARY:
        raise ValueError("unsupported V2 belief information boundary")
    _require_text(value["game_id"], field_name="game_id")
    _require_text(value["source_game_dir"], field_name="source_game_dir")
    _require_text(value["annotation_method"], field_name="annotation_method")
    step_idx = _require_non_negative_int(value["step_idx"], field_name="step_idx")
    observer = value["observer"]
    speaker = value["current_speaker"]
    if observer not in PLAYER_NAMES or speaker not in PLAYER_NAMES:
        raise ValueError("V2 belief observer and speaker must be canonical")
    if value["observer_role"] not in _ROLE_NAMES:
        raise ValueError("unsupported V2 belief observer_role")
    if value["annotation_confidence"] not in _CONFIDENCE_NAMES:
        raise ValueError("unsupported V2 belief annotation_confidence")
    for field_name in ("review_required", "is_current_speaker"):
        if not isinstance(value[field_name], bool):
            raise TypeError(f"V2 belief {field_name} must be bool")
    if value["is_current_speaker"] != (observer == speaker):
        raise ValueError("V2 belief is_current_speaker is inconsistent")
    phase = _require_text(value["phase"], field_name="phase")
    day, _ = parse_public_phase(phase)
    if value["day"] != day:
        raise ValueError("V2 belief day differs from phase")
    _require_non_negative_int(
        value["public_action_count"], field_name="public_action_count"
    )
    digest = value["public_event_digest"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("public_event_digest must be a lowercase SHA-256 digest")
    violations = _require_sequence(
        value["constraint_violations"], field_name="constraint_violations"
    )
    if any(not isinstance(item, str) for item in violations):
        raise TypeError("constraint_violations must contain text")

    hard = value["hard_knowledge"]
    if not isinstance(hard, Mapping):
        raise TypeError("V2 belief hard_knowledge must be a mapping")
    _require_exact_fields(
        hard,
        {"known_werewolves_other", "known_non_werewolves_other"},
        field_name="V2 belief hard_knowledge",
    )
    known_wolves = _normalize_player_list(
        hard["known_werewolves_other"],
        field_name="known_werewolves_other",
    )
    known_non_wolves = _normalize_player_list(
        hard["known_non_werewolves_other"],
        field_name="known_non_werewolves_other",
    )
    if observer in set(known_wolves) | set(known_non_wolves):
        raise ValueError("V2 belief hard_knowledge must exclude the observer")
    if set(known_wolves) & set(known_non_wolves):
        raise ValueError("V2 belief hard_knowledge sets must be disjoint")

    recommendation = value["training_recommendation"]
    if not isinstance(recommendation, Mapping):
        raise TypeError("V2 belief training_recommendation must be a mapping")
    _require_exact_fields(
        recommendation,
        _TRAINING_RECOMMENDATION_FIELDS,
        field_name="V2 belief training_recommendation",
    )
    loss_mask = recommendation["distribution_loss_mask"]
    if not isinstance(loss_mask, bool):
        raise TypeError("distribution_loss_mask must be bool")
    if not isinstance(recommendation["ordinal_or_pairwise_preferred"], bool):
        raise TypeError("ordinal_or_pairwise_preferred must be bool")
    distribution = recommendation["compat_relative_suspicion_distribution"]
    if not isinstance(distribution, Mapping) or set(distribution) != set(PLAYER_NAMES):
        raise ValueError("V2 compat distribution must map all seven players")
    normalized_distribution: dict[str, float] = {}
    for player in PLAYER_NAMES:
        probability = distribution[player]
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(float(probability))
            or probability < 0
        ):
            raise ValueError("V2 compat probabilities must be finite and non-negative")
        normalized_distribution[player] = float(probability)
    if normalized_distribution[observer] != 0.0:
        raise ValueError("V2 compat distribution diagonal must be zero")
    support = _normalize_player_list(
        recommendation["compat_suspected_werewolves"],
        field_name="compat_suspected_werewolves",
    )
    positive_support = [
        player for player in PLAYER_NAMES if normalized_distribution[player] > 0.0
    ]
    total = sum(normalized_distribution.values())
    if loss_mask:
        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("observed V2 compat distribution must sum to one")
        if support != positive_support:
            raise ValueError("V2 compat support differs from positive probabilities")
    elif total != 0.0 or support:
        raise ValueError("unobserved V2 belief must keep a zero distribution")

    label = value["v2_label"]
    if not isinstance(label, Mapping):
        raise TypeError("V2 label must be a mapping")
    _require_exact_fields(label, _V2_LABEL_FIELDS, field_name="V2 label")
    if not isinstance(label["insufficient_evidence_or_abstain"], bool):
        raise TypeError("V2 insufficient_evidence_or_abstain must be bool")
    if label["ordinal_scale"] != _ORDINAL_SCALE:
        raise ValueError("unsupported V2 ordinal scale")
    expected_players = set(PLAYER_NAMES) - {observer}
    scores = label["subjective_suspicion_ordinal"]
    if not isinstance(scores, Mapping) or set(scores) != expected_players:
        raise ValueError(
            "V2 subjective_suspicion_ordinal must map every non-self player"
        )
    reasons = label["ordinal_derivation_reasons"]
    if not isinstance(reasons, Mapping) or set(reasons) not in (
        expected_players,
        set(PLAYER_NAMES),
    ):
        raise ValueError(
            "V2 ordinal_derivation_reasons must map every non-self player, "
            "with an optional explicit self-exclusion reason"
        )
    if observer in reasons and reasons[observer] != "hard_or_self_excluded":
        raise ValueError("V2 self derivation reason must be hard_or_self_excluded")
    for score in label["subjective_suspicion_ordinal"].values():
        if score is not None and (
            isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 4
        ):
            raise ValueError("V2 ordinal scores must be null or integers in [0, 4]")
    for reason in label["ordinal_derivation_reasons"].values():
        _require_text(reason, field_name="ordinal_derivation_reason")
    _require_sequence(label["evidence_event_ids"], field_name="evidence_event_ids")
    _require_sequence(
        label["pairwise_suspicion_relations"],
        field_name="pairwise_suspicion_relations",
    )

    normalized = deepcopy(dict(value))
    normalized["hard_knowledge"] = {
        "known_werewolves_other": known_wolves,
        "known_non_werewolves_other": known_non_wolves,
    }
    normalized["training_recommendation"] = {
        **dict(recommendation),
        "compat_relative_suspicion_distribution": normalized_distribution,
        "compat_suspected_werewolves": support,
    }
    _validate_annotation_digest(value)
    return normalized


def _load_jsonl(path: str | Path) -> list[Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"annotation sidecar not found: {source}")
    records = []
    with source.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid annotation JSON on line {line_number}: {exc}"
                ) from exc
    if not records:
        raise ValueError("annotation sidecar cannot be empty")
    return records


def load_speech_v2_annotations(
    path: str | Path,
) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for raw in _load_jsonl(path):
        record = normalize_speech_v2_annotation(raw)
        key = (record["game_id"], record["speech_event_idx"])
        if key in result:
            raise ValueError(f"duplicate V2 speech key: {key}")
        result[key] = record
    return result


def load_belief_v2_annotations(
    path: str | Path,
) -> dict[tuple[str, int, str], dict[str, Any]]:
    result: dict[tuple[str, int, str], dict[str, Any]] = {}
    for raw in _load_jsonl(path):
        record = normalize_belief_v2_annotation(raw)
        key = (record["game_id"], record["step_idx"], record["observer"])
        if key in result:
            raise ValueError(f"duplicate V2 belief key: {key}")
        result[key] = record
    return result


def annotation_set_digest(records: Mapping[Any, Mapping[str, Any]]) -> str:
    """Bind an annotation set independently of JSONL line order."""

    return _canonical_digest(sorted(record["annotation_digest"] for record in records.values()))


def apply_speech_v2_to_sample(
    sample: Mapping[str, Any],
    records: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a derived V1-shaped sample whose speech actions come from V2."""

    derived = deepcopy(dict(sample))
    game_id = _require_text(derived.get("game_id"), field_name="sample.game_id")
    annotations = []
    current_phase = None
    for event in derived.get("public_events", []):
        if event.get("event_type") == "phase_change":
            current_phase = event.get("phase")
            continue
        if event.get("event_type") != "public_speech":
            continue
        event_idx = event.get("event_idx")
        record = records.get((game_id, event_idx))
        if record is None:
            raise ValueError(
                f"V2 speech sidecar has no record for {(game_id, event_idx)}"
            )
        if record["speaker"] != event.get("speaker"):
            raise ValueError("V2 speech speaker differs from public event")
        if record["raw_text"] != event.get("raw_text"):
            raise ValueError("V2 speech raw_text differs from public event")
        if record["phase"] != current_phase:
            raise ValueError("V2 speech phase differs from public event history")
        actions = record["compat_actions"]
        annotations.append(make_speech_annotation(
            event_idx=event_idx,
            speaker=record["speaker"],
            raw_text=record["raw_text"],
            parser_model_id="annotation_v2_public_only",
            parser_call_id=record["annotation_digest"],
            annotation_source="llm_parser",
            status=STATUS_OK if actions else STATUS_NO_ACTION,
            actions=actions,
            raw_response=None,
            error_type=None,
            error_message=None,
        ))
    derived["speech_annotations"] = annotations
    derived["speech_annotation_schema_version"] = SPEECH_ANNOTATION_SCHEMA_VERSION
    derived["public_action_count"] = len(
        public_speech_actions(derived["public_events"], annotations)
    )
    derived["speech_annotation_digest"] = speech_annotation_digest(annotations)
    derived["structured_input_digest"] = structured_input_digest(
        derived["public_events"], annotations
    )
    return derived


def belief_v2_targets_for_sample(
    sample: Mapping[str, Any],
    records: Mapping[tuple[str, int, str], Mapping[str, Any]],
    *,
    observer_roles: Mapping[str, str] | None,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bind V2 belief rows to one immutable PRE sample without imputation."""

    game_id = _require_text(sample.get("game_id"), field_name="sample.game_id")
    step_idx = _require_non_negative_int(sample.get("step_idx"), field_name="sample.step_idx")
    speaker = f"player{sample['speaker_id']}"
    targets = torch.zeros((NUM_PLAYERS, NUM_PLAYERS), dtype=dtype)
    observed = torch.zeros(NUM_PLAYERS, dtype=torch.bool)
    for observer_id in sample["observer_ids"]:
        observer = f"player{observer_id}"
        key = (game_id, step_idx, observer)
        record = records.get(key)
        if record is None:
            raise ValueError(f"V2 belief sidecar has no record for {key}")
        if record["current_speaker"] != speaker:
            raise ValueError("V2 belief speaker differs from PRE sample")
        if record["phase"] != sample["phase"]:
            raise ValueError("V2 belief phase differs from PRE sample")
        if record["public_event_digest"] != sample["public_event_digest"]:
            raise ValueError("V2 belief public_event_digest differs from PRE sample")
        if record["public_action_count"] != sample["public_action_count"]:
            raise ValueError("V2 belief public_action_count differs from PRE sample")
        if observer_roles is not None and record["observer_role"] != observer_roles[observer]:
            raise ValueError("V2 belief observer_role differs from role sidecar")
        expected_wolves = sorted(
            set(sample["known_werewolves"][observer]) - {observer},
            key=PLAYER_TO_ID.__getitem__,
        )
        expected_non_wolves = sorted(
            set(sample["known_non_werewolves"][observer]) - {observer},
            key=PLAYER_TO_ID.__getitem__,
        )
        if record["hard_knowledge"] != {
            "known_werewolves_other": expected_wolves,
            "known_non_werewolves_other": expected_non_wolves,
        }:
            raise ValueError("V2 belief hard_knowledge differs from PRE sample")
        recommendation = record["training_recommendation"]
        row_index = observer_id - 1
        observed[row_index] = recommendation["distribution_loss_mask"]
        if observed[row_index]:
            targets[row_index] = torch.tensor(
                [
                    recommendation["compat_relative_suspicion_distribution"][player]
                    for player in PLAYER_NAMES
                ],
                dtype=dtype,
            )
    return targets, observed


__all__ = [
    "BELIEF_ANNOTATION_SOURCES",
    "BELIEF_V2_SCHEMA_VERSION",
    "LEGACY_V1_BELIEF_SOURCE",
    "SPEECH_ANNOTATION_SOURCES",
    "SPEECH_V2_SCHEMA_VERSION",
    "V1_ANNOTATION_SOURCE",
    "V1_EMPTY_UNIFORM_NONSELF_BELIEF_SOURCE",
    "V2_ANNOTATION_SOURCE",
    "annotation_set_digest",
    "apply_speech_v2_to_sample",
    "belief_v2_targets_for_sample",
    "load_belief_v2_annotations",
    "load_speech_v2_annotations",
    "normalize_belief_v2_annotation",
    "normalize_speech_v2_annotation",
]
