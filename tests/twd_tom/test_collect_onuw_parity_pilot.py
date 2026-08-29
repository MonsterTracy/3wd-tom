from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from script.twd_tom.collect_canonical_trajectories import (
    _validate_classic7_config,
)
from script.twd_tom.collect_onuw_parity_pilot import (
    PARITY_PILOT_NAMESPACE,
    _pilot_contract,
    _validate_agent_profiles,
)
from werewolf.runtime_config import normalize_runtime_config


CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "onuw_parity_pilot_qwen35_9b.yaml"
)


def _config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_checked_in_pilot_config_freezes_new_fail_closed_namespace():
    parsed = _config()
    normalized = normalize_runtime_config(parsed)
    _validate_classic7_config(normalized)
    _validate_agent_profiles(normalized)
    contract = _pilot_contract(parsed, seeds=[5101, 5102, 5103])
    assert contract["data_namespace"] == PARITY_PILOT_NAMESPACE
    assert contract["allow_gameplay_fallback"] is False
    assert contract["formal_training_eligible"] is False
    assert contract["content_profile"] == "onuw_action_only"
    assert contract["modality_profile"] == "onuw_agent_declared_multimodal"


def test_pilot_config_rejects_old_seed_range_and_text_only_agent():
    parsed = _config()
    with pytest.raises(ValueError, match="seed range"):
        _pilot_contract(parsed, seeds=[4101, 4102, 4103])

    broken = deepcopy(parsed)
    broken["agent_config"]["all_candidates"][0]["model_params"][
        "speech_modality_profile"
    ] = "onuw_no_face_no_tone"
    with pytest.raises(ValueError, match="agent-declared face/tone"):
        _validate_agent_profiles(normalize_runtime_config(broken))


def test_pilot_contract_rejects_fallback_or_training_eligibility():
    for field, value in (
        ("allow_gameplay_fallback", True),
        ("formal_training_eligible", True),
    ):
        parsed = _config()
        parsed["pipeline"]["onuw_parity_pilot"][field] = value
        with pytest.raises(ValueError, match="contract mismatch"):
            _pilot_contract(parsed, seeds=[5101, 5102, 5103])
