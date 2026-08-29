import pytest

from werewolf.models.twd_tom.onuw_parity_audit import (
    action_only_information_loss_audit,
    sequence_capacity_audit,
)
from werewolf.models.twd_tom.onuw_parity_synthetic import synthetic_parity_games


def test_action_only_information_loss_audit_records_frozen_metrics():
    audit = action_only_information_loss_audit(synthetic_parity_games()[0])
    assert audit["zero_action_speech_rate"] == pytest.approx(0.5)
    assert audit["consecutive_queries_sharing_token_cutoff_rate"] == pytest.approx(0.5)
    assert audit["same_context_different_target_rate"] == pytest.approx(1.0)
    assert audit["shared_context_adjacent_target_tv_mean"] > 0
    assert audit["shared_context_adjacent_target_js_mean"] > 0


def test_sequence_capacity_audit_reports_required_percentiles_and_longest_prefix():
    audit = sequence_capacity_audit(synthetic_parity_games())
    for percentile in ("p50", "p90", "p95", "p99"):
        assert f"sequence_length_{percentile}" in audit
    assert audit["sequence_length_max"] == 3
    assert audit["longest_query_prefix"] == 3
    assert audit["capacity_decision"] == "unfrozen"
