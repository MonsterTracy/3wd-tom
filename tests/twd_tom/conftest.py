from copy import deepcopy
import hashlib
import json
import os

import pytest

from werewolf.models.twd_tom.samples import SAMPLE_SCHEMA_VERSION
from werewolf.models.twd_tom.schema import (
    LABEL_PROVENANCE,
    LABEL_PROMPT_VERSION,
)
from tests.twd_tom.public_event_fixtures import public_history_fields
from tests.twd_tom.public_event_fixtures import make_training_sample
from werewolf.trajectory import canonical_digest, canonical_json


def _sha256(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json(path, value):
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


@pytest.fixture
def suspicion_sample_factory():
    def make(
        *,
        game_id="game_001",
        step_idx=1,
        observers=(1, 2, 3, 5),
        suspicions_by_observer=None,
        failed_observer=None,
    ):
        actions = [["player2", "point_as_werewolf", "player7"]]
        public_history = public_history_fields(actions, speaker_id=2)
        suspicions = {}
        statuses = {}
        errors = {}
        backend_ids = {}
        known_werewolves = {}
        known_non_werewolves = {}
        for index, observer_id in enumerate(observers):
            subject = f"player{observer_id}"
            if observer_id == failed_observer:
                suspicions[subject] = None
                statuses[subject] = "parse_error"
            else:
                default = ["player7"] if observer_id != 7 and index < 3 else []
                suspicions[subject] = list(
                    (suspicions_by_observer or {}).get(observer_id, default)
                )
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
            **deepcopy(public_history),
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
def canonical_belief_batch_factory():
    def make(root, samples_by_game, *, reverse=False):
        from script.twd_tom.collect_canonical_trajectories import (
            BATCH_PLAN_SCHEMA_VERSION,
            BATCH_SUMMARY_SCHEMA_VERSION,
            GAME_SUMMARY_SCHEMA_VERSION,
        )

        items = list(samples_by_game.items())
        if reverse:
            items.reverse()
        game_summaries = {}
        for index, (game_id, records) in enumerate(items, start=1):
            game_dir = root / "games" / f"game_{index:04d}"
            game_dir.mkdir(parents=True)
            ordered_records = list(reversed(records)) if reverse else list(records)
            belief_path = game_dir / "belief_snapshots.jsonl"
            belief_path.write_text(
                "".join(
                    json.dumps(record, ensure_ascii=False) + "\n"
                    for record in ordered_records
                ),
                encoding="utf-8",
            )
            game_summary = {
                "schema_version": GAME_SUMMARY_SCHEMA_VERSION,
                "collection_mode": "canonical",
                "canonical_eligible": True,
                "game_id": game_id,
                "belief_snapshot_count": len(records),
                "belief_report_count": sum(
                    len(record["observer_ids"]) for record in records
                ),
                "belief_snapshots_sha256": _sha256(belief_path),
                "call_audit": {"gameplay_fallback_count": 0},
            }
            game_summary["summary_digest"] = canonical_digest(game_summary)
            _write_json(game_dir / "summary.json", game_summary)
            game_summaries[game_id] = game_summary

        game_ids = sorted(game_summaries)
        plan = {
            "schema_version": BATCH_PLAN_SCHEMA_VERSION,
            "collection_mode": "canonical",
            "canonical_eligible": True,
            "planned_game_count": len(game_ids),
        }
        plan["plan_digest"] = canonical_digest(plan)
        _write_json(root / "plan.json", plan)
        summary = {
            "schema_version": BATCH_SUMMARY_SCHEMA_VERSION,
            "collection_mode": "canonical",
            "canonical_eligible": True,
            "total_gameplay_fallback_count": 0,
            "plan_digest": plan["plan_digest"],
            "planned_game_count": len(game_ids),
            "completed_game_count": len(game_ids),
            "game_ids": game_ids,
            "game_summary_digests": {
                game_id: game_summaries[game_id]["summary_digest"]
                for game_id in game_ids
            },
            "total_belief_snapshot_count": sum(
                game_summaries[game_id]["belief_snapshot_count"]
                for game_id in game_ids
            ),
            "total_belief_report_count": sum(
                game_summaries[game_id]["belief_report_count"]
                for game_id in game_ids
            ),
        }
        summary["summary_digest"] = canonical_digest(summary)
        _write_json(root / "summary.json", summary)
        return summary

    return make


@pytest.fixture
def training_sample_factory():
    return make_training_sample


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
