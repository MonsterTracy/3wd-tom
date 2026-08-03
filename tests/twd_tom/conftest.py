from copy import deepcopy
import hashlib
import json
import os

import pytest

from script.twd_tom.project_suspicion_to_pairs import project_suspicion_sample
from werewolf.models.twd_tom.public_events import (
    PUBLIC_EVENT_SCHEMA_VERSION,
    public_event_digest,
    structured_input_digest,
)
from werewolf.models.twd_tom.samples import (
    SAMPLE_SCHEMA_VERSION,
    freeze_public_snapshot,
    make_twd_tom_sample,
)
from werewolf.models.twd_tom.schema import (
    LABEL_PROVENANCE,
    LABEL_PROMPT_VERSION,
    canonical_wolf_pairs,
)
from werewolf.speech.pair_belief_self_reporter import (
    PAIR_BELIEF_PROMPT_VERSION,
    canonical_json_sha256,
)
from tests.twd_tom.public_event_fixtures import make_public_events
from tests.twd_tom.public_event_fixtures import make_training_sample


@pytest.fixture
def suspicion_sample_factory():
    def make(*, game_id="game_001", step_idx=1, observers=(1, 3, 5)):
        actions = [["player2", "point_as_werewolf", "player7"]]
        public_events = make_public_events(actions, speaker_id=2)
        suspicions = {}
        statuses = {}
        errors = {}
        backend_ids = {}
        known_werewolves = {}
        known_non_werewolves = {}
        for index, observer_id in enumerate(observers):
            subject = f"player{observer_id}"
            if index == 0:
                suspicions[subject] = None
                statuses[subject] = "parse_error"
            elif index == 1:
                suspicions[subject] = ["player7"]
                statuses[subject] = "ok"
            else:
                suspicions[subject] = []
                statuses[subject] = "ok"
            errors[subject] = (
                "synthetic invalid report"
                if statuses[subject] != "ok"
                else None
            )
            backend_ids[subject] = "fake_backend"
            known_werewolves[subject] = []
            known_non_werewolves[subject] = [subject]
        return {
            "schema_version": SAMPLE_SCHEMA_VERSION,
            "game_id": game_id,
            "step_idx": step_idx,
            "report_trigger": "pre_public_speech",
            "phase": "1_day_speech",
            "speaker_id": 2,
            "observer_ids": list(observers),
            "public_event_schema_version": PUBLIC_EVENT_SCHEMA_VERSION,
            "public_events": deepcopy(public_events),
            "public_event_digest": public_event_digest(public_events),
            "structured_input_digest": structured_input_digest(public_events),
            "suspected_werewolves": suspicions,
            "known_werewolves": known_werewolves,
            "known_non_werewolves": known_non_werewolves,
            "belief_status": statuses,
            "belief_errors": errors,
            "label_provenance": LABEL_PROVENANCE,
            "agent_backend_ids": backend_ids,
            "label_cutoff_step_idx": step_idx,
            "public_action_count": len(actions),
            "label_prompt_version": LABEL_PROMPT_VERSION,
        }
    return make


@pytest.fixture
def projected_sample_factory(suspicion_sample_factory):
    def make(**kwargs):
        return project_suspicion_sample(suspicion_sample_factory(**kwargs))

    return make


@pytest.fixture
def training_sample_factory():
    return make_training_sample


@pytest.fixture
def actor_pair_sample_factory():
    def make(*, game_id="synthetic_actor_game", speaker_id=1, alive=(1,)):
        events = make_public_events([], speaker_id=speaker_id)
        snapshot = freeze_public_snapshot(
            game_id=game_id,
            step_idx=1,
            phase="1_day_speech",
            speaker_id=speaker_id,
            report_trigger="pre_public_speech",
            observer_ids=alive,
            public_events=events,
        )
        reports = {}
        for player_id in alive:
            player = f"player{player_id}"
            target = [
                1.0 / 15 if player not in pair else 0.0
                for pair in canonical_wolf_pairs()
            ]
            payload = {
                "payload_version": "readonly_pair_belief_self_report_payload_v1",
                "messages": [{"role": "user", "content": f"private-{player}"}],
                "request": {"temperature": 0.0, "max_tokens": 256},
            }
            reports[player] = {
                "player_id": player,
                "report_status": "ok",
                "report_error": None,
                "pair_probabilities": target,
                "known_werewolves": [],
                "known_non_werewolves": [player],
                "reporter_input_payload": payload,
                "reporter_input_payload_sha256": canonical_json_sha256(payload),
                "raw_reporter_output": json.dumps({"pair_probabilities": target}),
                "parsed_output": {"pair_probabilities": target},
                "hard_knowledge_validation": {"status": "valid"},
                "report_provenance": "playing_agent_readonly_direct_pair_belief_self_report_v1",
                "backend_alias": "synthetic_backend",
                "resolved_model_name": "synthetic-model",
                "prompt_version": PAIR_BELIEF_PROMPT_VERSION,
                "prompt_sha256": hashlib.sha256(
                    payload["messages"][-1]["content"].encode("utf-8")
                ).hexdigest(),
                "parser_version": "strict_pair_probability_json_v1",
                "sampling_parameters": {"temperature": 0.0, "max_tokens": 256},
                "reporter_seed": None,
            }
        return make_twd_tom_sample(
            public_snapshot=snapshot,
            reports=reports,
            collection_provenance={
                "generator_name": "twd_tom_actor_pair_belief_collector",
                "generator_version": "1",
                "git_commit_sha": "a" * 40,
                "git_worktree_clean": True,
                "collection_timestamp_utc": "2026-08-03T00:00:00+00:00",
                "game_seed": 42,
                "source_config_path": "configs/synthetic.yaml",
                "source_config_sha256": "b" * 64,
                "resolved_runtime_config_sha256": "c" * 64,
                "resolved_backend_config_sha256": {
                    "synthetic_backend": "e" * 64
                },
            },
        )

    return make


@pytest.fixture
def require_real_twd_tom_data():
    def require(*paths):
        if os.environ.get("RUN_TWD_TOM_REAL_DATA_TESTS") != "1":
            pytest.skip(
                "set RUN_TWD_TOM_REAL_DATA_TESTS=1 to run formal-data smoke tests"
            )
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            pytest.skip(f"formal ToM data is unavailable: {missing}")

    return require
