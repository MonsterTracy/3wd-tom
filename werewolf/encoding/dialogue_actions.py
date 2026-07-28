PREDICATE2ID = {
    "none": 0,
    "claim_role": 1,
    "claim_camp": 2,
    "counter_claim": 3,
    "report_check_result": 4,
    "suspect": 5,
    "accuse_as_werewolf": 6,
    "support": 7,
    "oppose": 8,
    "defend_self": 9,
    "defend_other": 10,
    "attack_logic": 11,
    "question": 12,
    "vote_intention": 13,
    "follow_vote": 14,
    "hedge": 15,
    "retract": 16,
    "vote": 17,
    "death": 18,
    "exile": 19,
    "report_witch_save": 20,
    "report_witch_poison": 21,
}

ROLE2ID = {
    None: 0,
    "None": 0,
    "Werewolf": 1,
    "Seer": 2,
    "Witch": 3,
    "Guard": 4,
    "Villager": 5,
    "Unknown": 6,
}

CAMP2ID = {
    None: 0,
    "None": 0,
    "Village": 1,
    "Werewolf": 2,
    "Unknown": 3,
}

POLARITY2ID = {
    None: 0,
    "None": 0,
    "positive": 1,
    "negative": 2,
    "neutral": 3,
}

CERTAINTY2ID = {
    None: 0,
    "None": 0,
    "explicit": 1,
    "implicit": 2,
    "hedge": 3,
}

PHASE2ID = {
    "none": 0,
    "night": 1,
    "night_result": 2,
    "day_speech": 3,
    "day_vote": 4,
    "speech": 5,
    "speech_pk": 6,
    "vote": 7,
    "vote_pk": 8,
    "exile": 9,
}

EVENT_TYPE2ID = {
    "pad": 0,
    "dialogue_action": 1,
    "vote": 2,
    "pk_vote": 3,
    "death": 4,
    "exile": 5,
    "private_role_info": 6,
    "private_check_result": 7,
    "private_wolf_team": 8,
    "private_witch_info": 9,
}


def safe_id(mapping: dict, key, default: str) -> int:
    try:
        return mapping[key]
    except (KeyError, TypeError):
        return mapping[default]


def id_to_name(mapping: dict) -> dict:
    return {value: key for key, value in mapping.items()}


def _normalized_player_id(value) -> int:
    if type(value) is int and value > 0:
        return value
    return 0


def _normalized_category(mapping: dict, value, default):
    try:
        return value if value in mapping else default
    except TypeError:
        return default


def normalize_claim(claim: dict) -> dict:
    certainty = claim.get("certainty")
    if certainty not in ("explicit", "implicit", "hedge"):
        certainty = "implicit"

    return {
        "speaker": _normalized_player_id(claim.get("speaker")),
        "predicate": _normalized_category(
            PREDICATE2ID,
            claim.get("predicate"),
            "none",
        ),
        "target": _normalized_player_id(claim.get("target")),
        "role": _normalized_category(ROLE2ID, claim.get("role"), None),
        "camp": _normalized_category(CAMP2ID, claim.get("camp"), None),
        "polarity": _normalized_category(
            POLARITY2ID,
            claim.get("polarity"),
            None,
        ),
        "certainty": certainty,
        "condition": claim.get("condition"),
        "source_text": claim.get("source_text"),
    }


__all__ = [
    "PREDICATE2ID",
    "ROLE2ID",
    "CAMP2ID",
    "POLARITY2ID",
    "CERTAINTY2ID",
    "PHASE2ID",
    "EVENT_TYPE2ID",
    "safe_id",
    "id_to_name",
    "normalize_claim",
]
