from copy import deepcopy

import pytest

from script.twd_tom.project_suspicion_to_pairs import project_suspicion_sample
from werewolf.models.twd_tom.public_events import (
    PUBLIC_EVENT_SCHEMA_VERSION,
    public_event_digest,
    structured_input_digest,
)
from werewolf.models.twd_tom.samples import SAMPLE_SCHEMA_VERSION
from werewolf.models.twd_tom.schema import (
    LABEL_PROVENANCE,
    LABEL_PROMPT_VERSION,
)
from tests.twd_tom.public_event_fixtures import make_public_events


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
