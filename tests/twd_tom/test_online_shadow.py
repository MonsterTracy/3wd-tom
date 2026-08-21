"""Tests for public-only online second-order ToM shadow inference."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch.optim import AdamW

from run_random import (
    _resolve_tom2_shadow_options,
    build_arg_parser,
    eval as run_game,
)
from script.twd_tom.train import (
    TrainingConfig,
    build_model,
    checkpoint_payload,
)
from tests.twd_tom.public_event_fixtures import (
    make_public_events,
    make_training_sample,
)
from werewolf.models.twd_tom.action_features import PublicEventFeatureBuilder
from werewolf.models.twd_tom.dataset import TWDToMDataset
from werewolf.models.twd_tom.schema import (
    NUM_WOLF_PAIR_CLASSES,
    PAIR_ORDERING,
    SECOND_ORDER_TARGET_ENCODING,
)
from werewolf.models.twd_tom.shadow import SecondOrderToMShadow


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_checkpoint(tmp_path, tom_order):
    config = TrainingConfig(
        tom_order=tom_order,
        output_dir=str(tmp_path / "training-output"),
        dataset_path=str(tmp_path / "train.jsonl"),
        validation_dataset_path=str(tmp_path / "val.jsonl"),
        batch_size=1,
        max_seq_len=32,
    )
    model = build_model(config)
    optimizer = AdamW(model.parameters())
    checkpoint = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=1,
        train_metrics={"mean_loss": 1.0, "valid_subject_count": 1},
        validation_metrics={"mean_loss": 0.5, "valid_subject_count": 1},
        best_epoch=1,
        best_validation_mean_loss=0.5,
        run_provenance={
            "git_commit_sha": "1" * 40,
            "git_worktree_clean": True,
            "train_dataset_path": "data/synthetic/train.jsonl",
            "train_dataset_sha256": "0" * 64,
            "validation_dataset_path": "data/synthetic/val.jsonl",
            "validation_dataset_sha256": "0" * 64,
            "output_dir": "outputs/synthetic",
            "python_version": "test",
            "torch_version": str(torch.__version__),
            "transformers_version": "test",
            "platform": "test",
            "requested_device": "cpu",
            "resolved_device": "cpu",
            "deterministic_algorithms_enabled": True,
            "seed": 42,
        },
    )
    path = tmp_path / f"tom{tom_order}.pt"
    torch.save(checkpoint, path)
    return path


@pytest.fixture
def second_checkpoint(tmp_path):
    return _write_checkpoint(tmp_path, 2)


def _new_shadow(tmp_path, checkpoint_path, *, name="shadow.jsonl"):
    return SecondOrderToMShadow(
        checkpoint_path=str(checkpoint_path),
        device="cpu",
        output_path=str(tmp_path / name),
        game_id="game_shadow_001",
    )


def test_strict_load_and_public_only_probability_matrix(
    tmp_path,
    second_checkpoint,
):
    public_events = make_public_events([], speaker_id=1)
    original_events = deepcopy(public_events)
    shadow = _new_shadow(tmp_path, second_checkpoint)
    forwarded = {}

    def capture_inputs(_module, _args, kwargs):
        forwarded.update(kwargs)

    hook = shadow.model.register_forward_pre_hook(capture_inputs, with_kwargs=True)
    try:
        record = shadow.record(
            step_idx=0,
            phase="1_day_speech",
            speaker_id=1,
            public_events=public_events,
        )
    finally:
        hook.remove()
        shadow.close()

    assert set(forwarded) == set(PublicEventFeatureBuilder.FEATURE_FIELDS)
    assert public_events == original_events
    pair_matrix = torch.tensor(record["pair_probability_matrix"])
    assert pair_matrix.shape == (7, NUM_WOLF_PAIR_CLASSES)
    assert torch.isfinite(pair_matrix).all()
    assert torch.all(pair_matrix >= 0)
    torch.testing.assert_close(pair_matrix.sum(dim=-1), torch.ones(7))
    marginal_matrix = torch.tensor(record["wolf_marginal_matrix"])
    assert marginal_matrix.shape == (7, 7)
    assert torch.isfinite(marginal_matrix).all()
    assert torch.all((marginal_matrix >= 0) & (marginal_matrix <= 1))
    torch.testing.assert_close(
        marginal_matrix.sum(dim=-1),
        torch.full((7,), 2.0),
    )
    assert record["target_encoding"] == SECOND_ORDER_TARGET_ENCODING
    assert record["pair_class_count"] == NUM_WOLF_PAIR_CLASSES
    assert record["pair_ordering"] == PAIR_ORDERING
    assert "suspicion_matrix" not in record
    assert record["event_idx"] == public_events[-1]["event_idx"]
    assert record["public_event_count"] == len(public_events)
    assert record["observer_supervision_mask"] == [False] * 7
    assert record["supervision_boundary"] is None
    assert "observer_evidence_mask" not in record
    assert "observer_public_action_count" not in record

    saved = json.loads((tmp_path / "shadow.jsonl").read_text(encoding="utf-8"))
    assert saved == record
    serialized = json.dumps(saved)
    for forbidden in (
        "logits",
        "roles",
        "known_werewolves",
        "known_non_werewolves",
        "suspected_werewolves",
        "targets",
    ):
        assert forbidden not in serialized


def test_shadow_records_all_other_players_after_a_completed_speech(
    tmp_path,
    second_checkpoint,
):
    public_events = make_public_events(
        [["player2", "support", "player4"]],
        speaker_id=2,
    )
    with _new_shadow(tmp_path, second_checkpoint) as shadow:
        record = shadow.record(
            step_idx=0,
            phase="1_day_speech",
            speaker_id=2,
            public_events=public_events,
        )
    assert record["observer_supervision_mask"] == [
        True,
        False,
        True,
        True,
        True,
        True,
        True,
    ]
    assert record["supervision_boundary"] == (
        "post_completed_public_speech_pre_next_action_v1"
    )
    assert len(record["pair_probability_matrix"]) == 7
    assert len(record["wolf_marginal_matrix"]) == 7


def test_shadow_supervision_mask_matches_dataset_reasoning_player(
    tmp_path, second_checkpoint
):
    sample = make_training_sample(2, with_latest_action=True)
    dataset = TWDToMDataset([sample], tom_order=2)
    item = dataset[0]
    with _new_shadow(tmp_path, second_checkpoint) as shadow:
        record = shadow.record(
            step_idx=sample["step_idx"],
            phase=sample["phase"],
            speaker_id=sample["speaker_id"],
            public_events=sample["public_events"],
        )
    expected = [
        player_id != item["reasoning_player_id"].item()
        for player_id in range(1, 8)
    ]
    assert record["observer_supervision_mask"] == expected
    assert item["post_completed_public_speech_pre_next_action"]


def test_old_seven_class_and_first_order_checkpoints_are_rejected(
    tmp_path,
    second_checkpoint,
):
    first_path = _write_checkpoint(tmp_path, 1)
    with pytest.raises(ValueError, match="tom_order=2"):
        _new_shadow(tmp_path, first_path, name="first.jsonl")

    old_second = torch.load(
        second_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    old_second["target_encoding"] = (
        "classic7_player_suspicion_distribution_v1"
    )
    old_second["output_class_count"] = 7
    old_second["model_config"]["pair_class_count"] = 7
    old_second_path = tmp_path / "old-second.pt"
    torch.save(old_second, old_second_path)
    with pytest.raises(ValueError, match="target_encoding"):
        _new_shadow(tmp_path, old_second_path, name="old-second.jsonl")


def test_existing_output_and_duplicate_snapshot_fail(
    tmp_path,
    second_checkpoint,
):
    existing = tmp_path / "existing.jsonl"
    existing.write_text("do not overwrite\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        _new_shadow(tmp_path, second_checkpoint, name=existing.name)
    assert existing.read_text(encoding="utf-8") == "do not overwrite\n"
    with pytest.raises(FileNotFoundError, match="parent does not exist"):
        SecondOrderToMShadow(
            checkpoint_path=str(second_checkpoint),
            device="cpu",
            output_path=str(tmp_path / "missing" / "shadow.jsonl"),
            game_id="game_shadow_001",
        )

    shadow = _new_shadow(tmp_path, second_checkpoint)
    arguments = {
        "step_idx": 0,
        "phase": "1_day_speech",
        "speaker_id": 1,
        "public_events": make_public_events([], speaker_id=1),
    }
    try:
        shadow.record(**arguments)
        with pytest.raises(RuntimeError, match="already recorded"):
            shadow.record(**arguments)
    finally:
        shadow.close()


def test_non_finite_model_output_is_rejected(tmp_path, second_checkpoint):
    shadow = _new_shadow(tmp_path, second_checkpoint)
    shadow.model.forward = lambda **_kwargs: {
        "pair_probabilities": torch.full(
            (1, 7, NUM_WOLF_PAIR_CLASSES),
            float("nan"),
        ),
        "wolf_marginals": torch.zeros((1, 7, 7)),
    }
    try:
        with pytest.raises(ValueError, match="finite"):
            shadow.record(
                step_idx=0,
                phase="1_day_speech",
                speaker_id=1,
                public_events=make_public_events([], speaker_id=1),
            )
    finally:
        shadow.close()


def test_shadow_consumes_backbone_derived_outputs_without_recomputation(
    tmp_path,
    second_checkpoint,
):
    shadow = _new_shadow(tmp_path, second_checkpoint)
    probabilities = torch.zeros((1, 7, NUM_WOLF_PAIR_CLASSES))
    probabilities[..., 0] = 1.0
    marginals = torch.zeros((1, 7, 7))
    marginals[..., 5:] = 1.0
    shadow.model.forward = lambda **_kwargs: {
        "observer_pair_logits": torch.full_like(probabilities, float("nan")),
        "pair_probabilities": probabilities,
        "wolf_marginals": marginals,
    }
    try:
        record = shadow.record(
            step_idx=0,
            phase="1_day_speech",
            speaker_id=1,
            public_events=make_public_events([], speaker_id=1),
        )
    finally:
        shadow.close()
    assert torch.equal(
        torch.tensor(record["pair_probability_matrix"]),
        probabilities[0],
    )
    assert torch.equal(
        torch.tensor(record["wolf_marginal_matrix"]),
        marginals[0],
    )


class OneSpeechEnvironment:
    def __init__(self):
        self.phase = "speech"
        self.alive = [1] * 7
        self.public_events = []
        self.step_actions = []

    def reset(self, roles):
        self.public_events = make_public_events([], speaker_id=1)
        return {"current_act_idx": 1, "phase": "1_day_speech"}

    def step(self, action):
        self.step_actions.append(action)
        self.public_events.append(
            {
                "event_idx": len(self.public_events),
                "event_type": "public_speech",
                "speaker": "player1",
                "raw_text": action[1],
                "sp_actions": [],
            }
        )
        return (
            {"current_act_idx": 1, "phase": "1_day_vote"},
            0,
            True,
            {"Werewolf": -1},
        )


class SpeechAgent:
    def __init__(self, before_act=None):
        self.before_act = before_act
        self.returned_action = ("speech", "unchanged speech")

    def reset(self):
        pass

    def act(self, observation):
        if self.before_act is not None:
            self.before_act()
        return self.returned_action


class RecordingShadow:
    def __init__(self, *, error=None):
        self.calls = []
        self.error = error

    def record(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        if self.error is not None:
            raise self.error


def test_shadow_runs_after_turn_start_before_agent_without_changing_game():
    env = OneSpeechEnvironment()
    shadow = RecordingShadow()

    def verify_shadow_precedes_action():
        assert len(shadow.calls) == 1
        assert shadow.calls[0]["public_events"][-1]["event_type"] == "turn_start"

    agent = SpeechAgent(before_act=verify_shadow_precedes_action)
    result = run_game(
        env,
        [agent],
        roles_=["Villager"] * 7,
        tom2_shadow=shadow,
    )
    assert result == "Villager win"
    assert env.step_actions == [agent.returned_action]
    assert len(shadow.calls) == 1
    assert shadow.calls[0]["public_events"][-1]["event_type"] == "turn_start"
    assert env.public_events[-1]["event_type"] == "public_speech"


def test_shadow_failure_aborts_before_agent_action():
    env = OneSpeechEnvironment()
    acted = []
    agent = SpeechAgent(before_act=lambda: acted.append(True))
    with pytest.raises(RuntimeError, match="inference failed"):
        run_game(
            env,
            [agent],
            roles_=["Villager"] * 7,
            tom2_shadow=RecordingShadow(error=RuntimeError("inference failed")),
        )
    assert acted == []


def test_minimal_game_fixture_writes_one_real_shadow_record(
    tmp_path,
    second_checkpoint,
):
    output_path = tmp_path / "game-shadow.jsonl"
    with SecondOrderToMShadow(
        checkpoint_path=str(second_checkpoint),
        device="cpu",
        output_path=str(output_path),
        game_id="game_fixture",
    ) as shadow:
        env = OneSpeechEnvironment()
        agent = SpeechAgent()
        result = run_game(
            env,
            [agent],
            roles_=["Villager"] * 7,
            tom2_shadow=shadow,
        )
    assert result == "Villager win"
    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(records) == 1
    assert len(records[0]["pair_probability_matrix"]) == 7
    assert len(records[0]["pair_probability_matrix"][0]) == 21
    assert len(records[0]["wolf_marginal_matrix"]) == 7
    assert len(records[0]["wolf_marginal_matrix"][0]) == 7
    assert records[0]["observer_supervision_mask"] == [False] * 7
    assert records[0]["supervision_boundary"] is None
    assert "suspicion_matrix" not in records[0]


def test_cli_shadow_arguments_are_all_or_none():
    parser = build_arg_parser()
    plain = parser.parse_args([])
    assert _resolve_tom2_shadow_options(plain) is None
    assert plain.random_seed is None

    args = parser.parse_args(
        [
            "--twd_tom2_shadow_checkpoint",
            "best.pt",
            "--twd_tom2_shadow_device",
            "cpu",
            "--twd_tom2_shadow_output_path",
            "shadow.jsonl",
            "--random_seed",
            "42",
        ]
    )
    assert _resolve_tom2_shadow_options(args) == (
        "best.pt",
        "cpu",
        "shadow.jsonl",
    )
    assert args.random_seed == 42
    incomplete = parser.parse_args(
        ["--twd_tom2_shadow_checkpoint", "best.pt"]
    )
    with pytest.raises(ValueError, match="provided together"):
        _resolve_tom2_shadow_options(incomplete)
    automatic = parser.parse_args(
        [
            "--twd_tom2_shadow_checkpoint",
            "best.pt",
            "--twd_tom2_shadow_device",
            "auto",
            "--twd_tom2_shadow_output_path",
            "shadow.jsonl",
        ]
    )
    with pytest.raises(ValueError, match="explicit"):
        _resolve_tom2_shadow_options(automatic)
