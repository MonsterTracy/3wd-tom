import copy
import unittest

from werewolf.encoding.dialogue_actions import (
    CAMP2ID,
    CERTAINTY2ID,
    EVENT_TYPE2ID,
    PHASE2ID,
    POLARITY2ID,
    PREDICATE2ID,
    ROLE2ID,
    id_to_name,
    normalize_claim,
    safe_id,
)


class DialogueActionsTest(unittest.TestCase):
    def test_safe_id_returns_known_mapping_value(self):
        self.assertEqual(
            safe_id(PREDICATE2ID, "suspect", "none"),
            PREDICATE2ID["suspect"],
        )

    def test_safe_id_falls_back_for_invalid_value(self):
        self.assertEqual(
            safe_id(PREDICATE2ID, "invalid", "none"),
            PREDICATE2ID["none"],
        )

    def test_id_to_name_reverses_mapping(self):
        reversed_mapping = id_to_name(PREDICATE2ID)

        self.assertEqual(reversed_mapping[PREDICATE2ID["claim_role"]], "claim_role")

    def test_normalize_claim_handles_missing_fields_without_mutating_input(self):
        claim = {"condition": "如果2号继续跳预言家", "source_text": "我再考虑"}
        original = copy.deepcopy(claim)

        normalized = normalize_claim(claim)

        self.assertEqual(claim, original)
        self.assertEqual(
            normalized,
            {
                "speaker": 0,
                "predicate": "none",
                "target": 0,
                "role": None,
                "camp": None,
                "polarity": None,
                "certainty": "implicit",
                "condition": "如果2号继续跳预言家",
                "source_text": "我再考虑",
            },
        )

    def test_normalize_claim_falls_back_for_invalid_categories(self):
        normalized = normalize_claim(
            {
                "speaker": 2,
                "predicate": "invented",
                "target": None,
                "role": "Hunter",
                "camp": "Neutral",
                "polarity": "maybe",
                "certainty": "certain",
            }
        )

        self.assertEqual(normalized["predicate"], "none")
        self.assertIsNone(normalized["role"])
        self.assertIsNone(normalized["camp"])
        self.assertIsNone(normalized["polarity"])
        self.assertEqual(normalized["certainty"], "implicit")
        self.assertEqual(normalized["target"], 0)

    def test_required_mapping_values_exist(self):
        for predicate in (
            "none",
            "claim_role",
            "claim_camp",
            "counter_claim",
            "report_check_result",
            "suspect",
            "accuse_as_werewolf",
            "support",
            "oppose",
            "defend_self",
            "defend_other",
            "attack_logic",
            "question",
            "vote_intention",
            "follow_vote",
            "hedge",
            "retract",
            "vote",
            "death",
            "exile",
            "report_witch_save",
            "report_witch_poison",
        ):
            self.assertIn(predicate, PREDICATE2ID)

        for role in (None, "Werewolf", "Seer", "Witch", "Guard", "Villager", "Unknown"):
            self.assertIn(role, ROLE2ID)

        for camp in (None, "Village", "Werewolf", "Unknown"):
            self.assertIn(camp, CAMP2ID)

        for polarity in (None, "positive", "negative", "neutral"):
            self.assertIn(polarity, POLARITY2ID)

        for certainty in (None, "explicit", "implicit", "hedge"):
            self.assertIn(certainty, CERTAINTY2ID)

        for phase in (
            "none",
            "night",
            "night_result",
            "day_speech",
            "day_vote",
            "speech",
            "speech_pk",
            "vote",
            "vote_pk",
            "exile",
        ):
            self.assertIn(phase, PHASE2ID)

        for event_type in (
            "pad",
            "dialogue_action",
            "vote",
            "pk_vote",
            "death",
            "exile",
            "private_role_info",
            "private_check_result",
            "private_wolf_team",
            "private_witch_info",
        ):
            self.assertIn(event_type, EVENT_TYPE2ID)

    def test_witch_report_predicates_are_appended_after_existing_ids(self):
        self.assertEqual(PREDICATE2ID["exile"], 19)
        self.assertEqual(PREDICATE2ID["report_witch_save"], 20)
        self.assertEqual(PREDICATE2ID["report_witch_poison"], 21)

    def test_normalize_claim_accepts_witch_report_predicates(self):
        for predicate in ("report_witch_save", "report_witch_poison"):
            with self.subTest(predicate=predicate):
                normalized = normalize_claim(
                    {
                        "speaker": 4,
                        "predicate": predicate,
                        "target": 6,
                    }
                )

                self.assertEqual(normalized["predicate"], predicate)
                self.assertNotEqual(normalized["predicate"], "none")

    def test_speaker_and_target_remain_one_based(self):
        normalized = normalize_claim(
            {
                "speaker": 3,
                "predicate": "support",
                "target": 7,
                "role": None,
                "camp": "Village",
                "polarity": "positive",
                "certainty": "explicit",
                "condition": None,
                "source_text": "我支持7号",
            }
        )

        self.assertEqual(normalized["speaker"], 3)
        self.assertEqual(normalized["target"], 7)


if __name__ == "__main__":
    unittest.main()
