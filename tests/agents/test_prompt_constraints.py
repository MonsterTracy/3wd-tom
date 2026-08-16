import unittest

from werewolf.agents.prompt_template_v0 import (
    _render_authoritative_public_history,
    build_belief_prompt,
    build_speech_prompt,
    build_vote_prompt,
)
from werewolf.helper.log_utils import Log


def _observation(*, identity="Villager", phase="1_day_speech"):
    return {
        "identity": identity,
        "phase": phase,
        "current_act_idx": 3,
        "valid_action": [],
        "game_log": [
            Log(
                viewer=[1, 2, 3, 4, 5, 6, 7],
                source=2,
                target=[1, 2, 3, 4, 5, 6, 7],
                content={
                    "speech_content": "我声称player5是狼人。",
                    "sp_actions": [["player2", "point_as_werewolf", "player5"]],
                },
                day=1,
                time="第1天白天",
                event="speech",
            )
        ],
        "authoritative_public_state": {
            "day": 1,
            "day_or_night": "day",
            "phase": "vote" if "vote" in phase else "speech",
            "last_night_result": {"day": 0, "dead_players": []},
            "prior_exiles": [],
            "alive_players": [1, 2, 3, 4, 5, 6, 7],
            "suggestible_exile_targets": [1, 2, 4, 5, 6, 7],
        },
    }


BELIEF = {
    "belief": "player5当前更像狼人。",
    "concise": "重点观察player5。",
    "roles": {
        "player1": "unknown",
        "player2": "unknown",
        "player4": "unknown",
        "player5": "Werewolf",
        "player6": "unknown",
        "player7": "unknown",
    },
}


class GameplayPromptTest(unittest.TestCase):
    def test_fixed_classic7_role_contract_has_no_other_roles(self):
        prompt = build_belief_prompt(_observation())

        for rule in (
            "Exactly 2 Werewolves",
            "Exactly 1 Seer",
            "Exactly 1 Witch",
            "Exactly 3 Villagers",
            "No other roles exist in the current game",
        ):
            self.assertIn(rule, prompt)
        self.assertNotIn("Guard", prompt)

    def test_strict_context_rejects_non_witch_role_setting(self):
        observation = _observation()
        observation["game_log"] = [
            Log(
                viewer=list(range(1, 8)),
                source=0,
                target=0,
                content={"Werewolf": 2, "Seer": 1, "Guard": 1, "Villager": 3},
                day=0,
                time="第0天夜晚",
                event="game_setting",
            )
        ]

        with self.assertRaisesRegex(ValueError, "1 Witch"):
            build_belief_prompt(observation)

    def test_belief_prompt_uses_fixed_premises_and_unresolved_domain(self):
        prompt = build_belief_prompt(_observation(identity="Seer"))

        self.assertIn("roles as fixed premises", prompt)
        self.assertIn("Do not reinterpret or re-guess them", prompt)
        self.assertIn("compact step-by-step role deduction", prompt)
        self.assertIn("Infer only these unresolved players", prompt)
        self.assertIn("roles object must contain exactly those unresolved", prompt)
        self.assertIn("Do not restate the game rules", prompt)
        self.assertIn("recount or recompute the fixed 7-player role composition", prompt)
        self.assertIn("Do not repeat the observation or history", prompt)
        self.assertIn("Do not mechanically discuss every unresolved player", prompt)
        self.assertIn('Use "unknown" when the available', prompt)
        self.assertIn("information is insufficient", prompt)
        self.assertIn("belief field must reason only about unresolved", prompt)
        self.assertIn("as concise as possible", prompt)
        self.assertIn("about 50 words", prompt)
        self.assertIn("concise field must be a short derived conclusion", prompt)
        self.assertIn("no more than 2 short sentences", prompt)

    def test_belief_prompt_separates_authority_private_facts_and_raw_claims(self):
        prompt = build_belief_prompt(_observation())

        for section in ("GAME / ROLE", "KNOWN INFORMATION", "PUBLIC CONVERSATION"):
            self.assertEqual(prompt.count(section), 1)
        self.assertIn("Environment authoritative public state", prompt)
        self.assertIn("Private facts legally visible to this player", prompt)
        self.assertIn("raw chronological public speech", prompt)
        self.assertIn("truthful, deceptive, mistaken or strategic", prompt)
        self.assertIn("not an authoritative fact", prompt)
        self.assertIn("我声称player5是狼人。", prompt)
        self.assertNotIn("sp_actions", prompt)

    def test_multiday_public_results_stay_authoritative_and_chronological(self):
        logs = [
            Log(
                viewer=list(range(1, 8)),
                source=0,
                target=0,
                content={"Werewolf": 2, "Seer": 1, "Witch": 1, "Villager": 3},
                day=0,
                time="第0天夜晚",
                event="game_setting",
            ),
            Log(
                viewer=list(range(1, 8)),
                source=2,
                target=list(range(1, 8)),
                content={"speech_content": "我觉得player4发言可疑。", "sp_actions": []},
                day=1,
                time="第1天白天",
                event="speech",
            ),
            Log(
                viewer=list(range(1, 8)),
                source=2,
                target=4,
                content={"vote_target": 4},
                day=1,
                time="第1天白天",
                event="vote",
            ),
            Log(
                viewer=list(range(1, 8)),
                source=3,
                target=4,
                content={"vote_target": 4},
                day=1,
                time="第1天白天",
                event="vote_pk",
            ),
            Log(
                viewer=list(range(1, 8)),
                source=0,
                target=4,
                content={"vote_outcome": 4, "expelled": 4},
                day=1,
                time="第1天白天",
                event="end_vote",
            ),
            Log(
                viewer=list(range(1, 8)),
                source=0,
                target=[6],
                content={"dead_list": [6]},
                day=1,
                time="第1天夜晚",
                event="end_night",
            ),
        ]
        observation = _observation(phase="2_day_speech")
        observation["game_log"] = logs
        observation["authoritative_public_state"].update({
            "day": 2,
            "last_night_result": {"day": 1, "dead_players": [6]},
            "prior_exiles": [{"player_id": 4, "day": 1}],
            "alive_players": [1, 2, 3, 5, 7],
            "suggestible_exile_targets": [1, 2, 5, 7],
        })

        prompt = build_belief_prompt(observation)
        authoritative, conversation = prompt.split("PUBLIC CONVERSATION", 1)
        vote = "completed vote: player2 voted for player4"
        pk_vote = "completed PK vote: player3 voted for player4"
        exile = "player4 was exiled"
        night = "completed night result: player6 died"
        for item in (vote, pk_vote, exile, night):
            self.assertIn(item, authoritative)
            self.assertNotIn(item, conversation)
        self.assertLess(authoritative.index(vote), authoritative.index(pk_vote))
        self.assertLess(authoritative.index(pk_vote), authoritative.index(exile))
        self.assertLess(authoritative.index(exile), authoritative.index(night))
        self.assertNotIn("我觉得player4发言可疑。", authoritative)
        self.assertIn("我觉得player4发言可疑。", conversation)
        self.assertNotIn("sp_actions", prompt)

    def test_private_role_events_never_enter_authoritative_public_history(self):
        private_events = (
            Log([], 1, [1, 2], {"wolf_team": [1, 2]}, 1, "第1天夜晚", "werewolf_team_info"),
            Log([], 1, 5, {"kill_target": 5}, 1, "第1天夜晚", "skill_wolf"),
            Log([], 0, 5, {"kill_decision": 5}, 1, "第1天夜晚", "kill_decision"),
            Log([], 3, 5, {"cheked_identity": "bad"}, 1, "第1天夜晚", "skill_seer"),
            Log([], 4, 5, {"poison": 5}, 1, "第1天夜晚", "skill_witch"),
        )

        for private_event in private_events:
            with self.subTest(event=private_event.event):
                self.assertEqual(
                    _render_authoritative_public_history([private_event]),
                    "- (no completed public history yet)",
                )

    def test_belief_prompt_supplies_actual_role_without_requesting_self_guess(self):
        prompt = build_belief_prompt(_observation(identity="Seer"))

        self.assertIn("Actual role supplied by the Environment: Seer", prompt)
        self.assertIn("Environment-supplied self role", prompt)
        self.assertIn("Infer only these unresolved players", prompt)

    def test_speech_prompt_requests_direct_natural_language(self):
        prompt = build_speech_prompt(_observation(), BELIEF)

        self.assertIn("CURRENT PRIVATE BELIEF", prompt)
        self.assertIn("Directly produce", prompt)
        self.assertIn("reveal, hide, bluff or", prompt)
        self.assertIn("deceive strategically", prompt)
        self.assertIn("Do not output JSON", prompt)
        self.assertNotIn("public_actions", prompt)

    def test_vote_prompt_uses_fresh_belief_and_rejects_intent_inheritance(self):
        prompt = build_vote_prompt(
            _observation(phase="1_day_vote"),
            BELIEF,
            (0, 1, 4, 5),
        )

        self.assertIn("FRESH PRIVATE BELIEF", prompt)
        self.assertIn("[0, 1, 4, 5]", prompt)
        self.assertIn("target 0 = abstain", prompt)
        self.assertIn("target 1..7 = vote for that player", prompt)
        self.assertIn("Do not preserve or", prompt)
        self.assertIn("inherit a target merely because", prompt)


if __name__ == "__main__":
    unittest.main()
