import inspect
import unittest

from werewolf.agents.prompt_template_v0 import (
    DISCUSSION_ACTIONS,
    DiscussionAct,
    NO_STANCE,
    _render_authoritative_public_history,
    build_belief_prompt,
    build_day_cognition_prompt,
    build_public_claim_catalog,
    build_vote_prompt,
    derive_belief_constraints,
    derive_discussion_vote_targets,
    freeze_discussion_candidates,
    project_discussion_content_indices,
    project_discussion_vote_stances,
    render_deterministic_public_speech,
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
            "2名狼人、1名预言家、1名女巫、3名普通村民",
            "本局没有守卫、猎人或其他任何角色",
            "所有玩家的身份整局固定",
        ):
            self.assertIn(rule, prompt)
        self.assertNotIn("Guard", prompt)

    def test_strict_classic7_common_rules_cover_frozen_mechanics(self):
        prompt = build_belief_prompt(_observation())

        required_rules = (
            "可以主动选择当晚不进行击杀",
            "不能主动放弃查验",
            "已经查验过的玩家不能再次查验",
            "查验只提供信息",
            "没有任何玩家死亡",
            "1瓶解药和1瓶毒药",
            "不能对自己使用解药",
            "不能毒杀自己",
            "毒药目标可以与当晚狼队的击杀目标是同一名玩家",
            "使用毒药也不保证当晚一定出现两名不同的死亡玩家",
            "公开发言只是玩家自己的陈述",
            "已经投出的具体票不会提前公开",
            "所有存活玩家都参加PK投票，包括PK候选人本人",
            "如果最高票再次平票",
            "没有任何PK候选人获得有效票数",
            "不再有任何存活狼人",
            "存活狼人数达到或超过其他存活玩家总数",
        )
        for rule in required_rules:
            with self.subTest(rule=rule):
                self.assertIn(rule, prompt)

        self.assertNotIn("女巫应该", prompt)
        self.assertNotIn("预言家应该", prompt)
        self.assertNotIn("通常", prompt)
        self.assertNotIn("一般", prompt)

    def test_common_rules_keep_no_target_edge_cases_explicit(self):
        prompt = build_belief_prompt(_observation(identity="Seer"))

        self.assertIn(
            "不存在任何合法且未查验的目标，则该夜不执行查验",
            prompt,
        )
        self.assertIn(
            "没有任何玩家获得有效票数，则无人被放逐，也不进入PK阶段",
            prompt,
        )
        self.assertIn(
            "没有任何PK候选人获得有效票数，则本轮无人被放逐",
            prompt,
        )

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

    def test_belief_prompt_separates_gameplay_cognition_from_role_report(self):
        prompt = build_belief_prompt(_observation(identity="Seer"))

        self.assertIn("roles as fixed premises", prompt)
        self.assertIn("Do not reinterpret or re-guess them", prompt)
        self.assertIn("Infer only these unresolved players", prompt)
        self.assertIn("roles object must contain exactly those unresolved", prompt)
        self.assertIn("roles-object inference rules above apply to the roles field", prompt)
        self.assertIn("not to which", prompt)
        self.assertIn("players the gameplay belief may discuss", prompt)
        self.assertIn("belief field is gameplay cognition", prompt)
        self.assertIn("Environment authoritative", prompt)
        self.assertIn("exact-known private facts", prompt)
        self.assertIn("Use both exact-known facts and relevant unresolved-player", prompt)
        self.assertIn("You may mention exact-known", prompt)
        self.assertIn("do not reinterpret or re-guess fixed facts", prompt)
        self.assertIn("do not try to explain or enumerate the roles object", prompt)
        self.assertIn("Do not restate the game rules", prompt)
        self.assertIn("recount or recompute the fixed 7-player role composition", prompt)
        self.assertIn("Do not repeat the observation or history", prompt)
        self.assertIn("Do not mechanically discuss every unresolved player", prompt)
        self.assertIn('Use "unknown" when the available', prompt)
        self.assertIn("information is insufficient", prompt)
        self.assertNotIn("belief field must reason only about unresolved", prompt)
        self.assertIn("as concise as possible", prompt)
        self.assertIn("about 50 words", prompt)
        self.assertIn("concise field must be a short gameplay conclusion", prompt)
        self.assertIn("not a\nsummary of the roles object", prompt)
        self.assertIn("no more than 2 short sentences", prompt)

    def test_belief_prompt_renders_remaining_global_role_inventory(self):
        wolf_team = Log(
            viewer=[3, 7],
            source=0,
            target=[3, 7],
            content={"wolf_team": [3, 7]},
            day=0,
            time="第0天夜晚",
            event="werewolf_team_info",
        )
        cases = (
            ("Villager", [], (2, 1, 1, 2)),
            ("Witch", [], (2, 1, 0, 3)),
            ("Seer", [], (2, 0, 1, 3)),
            ("Werewolf", [wolf_team], (0, 1, 1, 3)),
        )

        for identity, extra_logs, counts in cases:
            with self.subTest(identity=identity):
                observation = _observation(identity=identity)
                observation["game_log"].extend(extra_logs)
                prompt = build_belief_prompt(observation)
                expected_inventory = """Remaining concrete role inventory for unresolved players:
Werewolf: {0}
Seer: {1}
Witch: {2}
Villager: {3}""".format(*counts)

                self.assertIn(expected_inventory, prompt)
                self.assertIn(
                    "Across all unresolved players, concrete role guesses must not",
                    prompt,
                )
                self.assertIn("exceed this remaining inventory", prompt)
                self.assertIn("global constraint across the complete roles object", prompt)
                self.assertIn('"unknown" consumes no role slot', prompt)

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

    def test_private_night_history_preserves_role_authoritative_actions(self):
        wolf = _observation(identity="Werewolf")
        wolf["game_log"].extend([
            Log([3, 7], 0, [3, 7], {"wolf_team": [3, 7]}, 0, "第0天夜晚", "werewolf_team_info"),
            Log([3, 7], 3, 1, {"kill_target": 1}, 0, "第0天夜晚", "skill_wolf"),
            Log([3, 7], 7, 0, {"kill_target": 0}, 0, "第0天夜晚", "skill_wolf"),
            Log([3, 7], 0, 0, {"kill_decision": 0}, 0, "第0天夜晚", "kill_decision"),
        ])
        wolf_prompt = build_belief_prompt(wolf)
        self.assertIn("第0天夜晚：player3 提交击杀 player1", wolf_prompt)
        self.assertIn("第0天夜晚：player7 提交主动空刀", wolf_prompt)
        self.assertIn("第0天夜晚：主动空刀", wolf_prompt)

        seer = _observation(identity="Seer")
        seer["game_log"].append(
            Log([3], 3, 5, {"cheked_identity": "good"}, 0, "第0天夜晚", "skill_seer")
        )
        seer_prompt = build_belief_prompt(seer)
        self.assertIn("第0天夜晚:player5=不是狼人", seer_prompt)

        witch = _observation(identity="Witch")
        witch["game_log"].extend([
            Log([3], 0, 5, {"kill_decision": 5}, 0, "第0天夜晚", "kill_decision"),
            Log([3], 3, 5, {"heal": 5}, 0, "第0天夜晚", "skill_witch"),
            Log([3], 3, 6, {"poison": 6}, 1, "第1天夜晚", "skill_witch"),
        ])
        witch_prompt = build_belief_prompt(witch)
        self.assertIn("第0天夜晚：对 player5 使用解药", witch_prompt)
        self.assertIn("第1天夜晚：对 player6 使用毒药", witch_prompt)
        self.assertIn("第0天夜晚：player5", witch_prompt)
        self.assertIn("解药真实状态：已使用", witch_prompt)
        self.assertIn("毒药真实状态：已使用", witch_prompt)

    def test_public_speech_preserves_authoritative_temporal_provenance(self):
        raw_speech = "我声称这是第9天夜晚；昨夜平安夜？！  原文不变。"
        observation = _observation(phase="2_day_speech")
        observation["game_log"] = [
            Log(
                viewer=list(range(1, 8)),
                source=4,
                target=list(range(1, 8)),
                content={"speech_content": raw_speech, "sp_actions": []},
                day=1,
                time="第1天白天",
                event="speech",
            ),
            Log(
                viewer=list(range(1, 8)),
                source=4,
                target=list(range(1, 8)),
                content={"speech_content": raw_speech, "sp_actions": []},
                day=2,
                time="第2天白天",
                event="speech_pk",
            ),
        ]

        prompt = build_belief_prompt(observation)
        authoritative, conversation = prompt.split("PUBLIC CONVERSATION", 1)
        day_one = f"- [第1天白天 / speech] player4：{raw_speech}"
        day_two = f"- [第2天白天 / speech_pk] player4：{raw_speech}"

        self.assertIn(day_one, conversation)
        self.assertIn(day_two, conversation)
        self.assertEqual(conversation.count(raw_speech), 2)
        self.assertNotIn(raw_speech, authoritative)
        self.assertIn("truthful, deceptive, mistaken or strategic", conversation)
        self.assertIn("not an authoritative fact", conversation)
        self.assertNotIn("- [第9天夜晚 /", conversation)

        catalog = build_public_claim_catalog(observation)
        self.assertEqual(
            tuple(claim.claim_id for claim in catalog),
            ("claim_000", "claim_001"),
        )
        exact_roles, role_options = derive_belief_constraints(observation)
        day_prompt = build_day_cognition_prompt(
            observation,
            exact_roles=exact_roles,
            role_options=role_options,
            candidate_snapshot=freeze_discussion_candidates(observation),
            claim_catalog=catalog,
        )
        self.assertLess(day_prompt.index("claim_000"), day_prompt.index("claim_001"))
        self.assertIn(f"[claim_000] [第1天白天 / speech] player4：{raw_speech}", day_prompt)
        self.assertIn(f"[claim_001] [第2天白天 / speech_pk] player4：{raw_speech}", day_prompt)

    def test_day_claim_ids_do_not_change_default_belief_or_vote_history(self):
        observation = _observation()
        gameplay_belief = {
            "belief": BELIEF["belief"],
            "concise": BELIEF["concise"],
        }
        default_prompts = (
            build_belief_prompt(observation),
            build_vote_prompt(observation, gameplay_belief, (0, 1, 4, 5)),
        )
        rendered_speech = (
            "- [第1天白天 / speech] "
            "player2：我声称player5是狼人。"
        )

        for prompt in default_prompts:
            with self.subTest(prompt=prompt):
                self.assertIn(rendered_speech, prompt)
                self.assertNotIn("claim_000", prompt)

        exact_roles, role_options = derive_belief_constraints(observation)
        catalog = build_public_claim_catalog(observation)
        day_prompt = build_day_cognition_prompt(
            observation,
            exact_roles=exact_roles,
            role_options=role_options,
            candidate_snapshot=freeze_discussion_candidates(observation),
            claim_catalog=catalog,
        )
        self.assertIn("[claim_000] [第1天白天 / speech]", day_prompt)
        self.assertIn("DISCUSSION INTENT OUTPUT", day_prompt)
        self.assertNotIn("DISCUSSION ACTION COMPATIBILITY", day_prompt)
        self.assertNotIn("public_action_indices", day_prompt)
        self.assertIn(
            "Set public_content_selection.mode to none, one or two",
            day_prompt,
        )
        self.assertIn(
            "second_rank is the zero-based rank in the same list after "
            "removing first_index",
            day_prompt,
        )
        self.assertIn(
            "Choose exactly one public_vote_stance_index",
            day_prompt,
        )
        self.assertIn(
            "NO_STANCE means no publicly stated voting tendency in this speech",
            day_prompt,
        )
        self.assertIn(
            "speech intent, not the authoritative later Vote-phase ballot",
            day_prompt,
        )
        self.assertIn("no_commitment is not selectable", day_prompt)
        self.assertIn(
            "Set evidence_selection.mode to none, one or two",
            day_prompt,
        )
        self.assertIn(
            "second_rank is the zero-based rank in the catalog after "
            "removing that claim",
            day_prompt,
        )
        self.assertIn(
            "public content is empty and NO_STANCE is\n"
            "selected, the program represents the empty discussion intent "
            "canonically",
            day_prompt,
        )
        content_candidates = day_prompt.split(
            "second_rank is the zero-based rank in the same list after "
            "removing first_index:\n",
            1,
        )[1].split(
            "\nChoose exactly one public_vote_stance_index",
            1,
        )[0]
        for excluded_action in (
            "vote_intent",
            "abstain_intent",
            "no_commitment",
        ):
            self.assertNotIn(excluded_action, content_candidates)
        stance_candidates = day_prompt.split(
            "Choose exactly one public_vote_stance_index from this "
            "deterministic projection:\n",
            1,
        )[1].split("\nNO_STANCE means", 1)[0]
        self.assertIn(f"0: {NO_STANCE}", stance_candidates)
        self.assertIn("vote_intent(player1)", stance_candidates)
        self.assertIn("abstain_intent", stance_candidates)
        for excluded_action in ("support", "oppose", "no_commitment"):
            self.assertNotIn(excluded_action, stance_candidates)
        snapshot = freeze_discussion_candidates(observation)
        self.assertTrue(project_discussion_content_indices(snapshot))
        self.assertEqual(
            project_discussion_vote_stances(snapshot)[0],
            NO_STANCE,
        )
        self.assertIn(
            "Strategic deception and bluff remain allowed within this frozen "
            "candidate space",
            day_prompt,
        )
        self.assertIn(
            "relevant evidence for the current cognition and\n"
            "discussion-intent decision",
            day_prompt,
        )
        self.assertIn("internal linkage record for\naudit only", day_prompt)
        self.assertIn(
            "do not authorize or require quoting, paraphrasing or mentioning",
            day_prompt,
        )
        self.assertIn("does not assert truth or falsity", day_prompt)
        self.assertIn("not prove causal influence", day_prompt)
        self.assertNotIn(
            "the only prior raw public utterances that the final public\n"
            "speech may explicitly reference",
            day_prompt,
        )
        self.assertIn(
            "point_as_villager(playerX): publicly claim playerX is "
            "specifically an ordinary Villager; not generic good / non-wolf",
            day_prompt,
        )
        self.assertIn(
            'check_as_non_werewolf(playerX): speaker publicly claims a Seer-style '
            'check on playerX returned "not Werewolf"; not necessarily Villager',
            day_prompt,
        )
        self.assertIn("1瓶解药和1瓶毒药", day_prompt)
        self.assertIn("point_as_witch(playerX)", day_prompt)
        self.assertIn("save(playerX)", day_prompt)
        self.assertIn("poison(playerX)", day_prompt)
        self.assertIn("no_commitment", day_prompt)
        self.assertNotIn("sp_actions", day_prompt)

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

    def test_deterministic_renderer_covers_all_discussion_act_meanings(self):
        cases = (
            ("point_as_werewolf", 2, "我认为 player2 是狼人。"),
            (
                "point_as_non_werewolf",
                2,
                "我认为 player2 不是狼人，属于好人阵营。",
            ),
            ("point_as_villager", 2, "我认为 player2 是普通村民。"),
            ("point_as_seer", 2, "我认为 player2 是预言家。"),
            ("point_as_witch", 2, "我认为 player2 是女巫。"),
            ("support", 2, "我支持 player2。"),
            ("oppose", 2, "我质疑 player2。"),
            ("check_as_non_werewolf", 2, "我查验过 player2，结果不是狼人。"),
            ("check_as_werewolf", 2, "我查验过 player2，结果是狼人。"),
            ("save", 2, "我用解药救了 player2。"),
            ("poison", 2, "我对 player2 使用了毒药。"),
            ("vote_intent", 2, "这一轮我建议投票放逐 player2。"),
            ("abstain_intent", None, "这一轮我选择弃票。"),
            (
                "no_commitment",
                None,
                "这一轮我暂不作明确的身份、查验、技能或投票表态。",
            ),
        )

        self.assertEqual(tuple(action for action, _target, _text in cases), DISCUSSION_ACTIONS)
        for action, target, expected in cases:
            with self.subTest(action=action):
                self.assertEqual(
                    render_deterministic_public_speech(
                        1,
                        discussion_acts=(DiscussionAct(action, target),),
                    ),
                    expected,
                )

        realized = {action: text for action, _target, text in cases}
        self.assertNotIn("好人", realized["point_as_villager"])
        self.assertNotIn("普通村民", realized["check_as_non_werewolf"])

    def test_deterministic_renderer_naturalizes_self_role_claims(self):
        cases = (
            ("point_as_villager", "我是普通村民。"),
            ("point_as_seer", "我是预言家。"),
            ("point_as_witch", "我是女巫。"),
        )

        for action, expected in cases:
            with self.subTest(action=action):
                self.assertEqual(
                    render_deterministic_public_speech(
                        3,
                        discussion_acts=(DiscussionAct(action, 3),),
                    ),
                    expected,
                )

        with self.assertRaisesRegex(ValueError, "cannot accuse itself"):
            render_deterministic_public_speech(
                3,
                discussion_acts=(DiscussionAct("point_as_werewolf", 3),),
            )

    def test_deterministic_renderer_preserves_order_and_is_atemporal(self):
        speech = render_deterministic_public_speech(
            3,
            discussion_acts=(
                DiscussionAct("check_as_werewolf", 6),
                DiscussionAct("save", 5),
                DiscussionAct("poison", 4),
            ),
        )

        self.assertEqual(
            speech.splitlines(),
            [
                "我查验过 player6，结果是狼人。",
                "我用解药救了 player5。",
                "我对 player4 使用了毒药。",
            ],
        )
        for temporal_anchor in (
            "昨晚",
            "前晚",
            "第一夜",
            "第1夜",
            "今天查验",
            "Night0",
            "Night1",
        ):
            self.assertNotIn(temporal_anchor, speech)

    def test_deterministic_renderer_has_no_evidence_interface(self):
        signature = inspect.signature(render_deterministic_public_speech)

        self.assertEqual(
            tuple(signature.parameters),
            ("speaker_id", "discussion_acts"),
        )
        self.assertNotIn("selected_claims", signature.parameters)
        self.assertEqual(
            render_deterministic_public_speech(
                3,
                discussion_acts=(DiscussionAct("oppose", 6),),
            ),
            "我质疑 player6。",
        )

    def test_deterministic_renderer_unknown_action_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unknown DiscussionAct action"):
            render_deterministic_public_speech(
                1,
                discussion_acts=(DiscussionAct("unknown_action", 2),),
            )

    def test_frozen_discussion_candidates_enforce_only_hard_boundaries(self):
        self.assertEqual(
            DISCUSSION_ACTIONS,
            (
                "point_as_werewolf",
                "point_as_non_werewolf",
                "point_as_villager",
                "point_as_seer",
                "point_as_witch",
                "support",
                "oppose",
                "check_as_non_werewolf",
                "check_as_werewolf",
                "save",
                "poison",
                "vote_intent",
                "abstain_intent",
                "no_commitment",
            ),
        )
        normal = _observation(identity="Werewolf")
        normal["authoritative_public_state"]["alive_players"] = [1, 2, 3, 5, 6, 7]
        normal["authoritative_public_state"]["suggestible_exile_targets"] = [1, 2, 5, 6, 7]
        normal["game_log"].append(
            Log([3, 7], 0, [3, 7], {"wolf_team": [3, 7]}, 0, "第0天夜晚", "werewolf_team_info")
        )
        snapshot = freeze_discussion_candidates(normal)

        vote_targets = tuple(
            act.target for act in snapshot if act.action == "vote_intent"
        )
        self.assertEqual(vote_targets, (1, 2, 5, 6, 7))
        self.assertNotIn(DiscussionAct("vote_intent", 3), snapshot)
        self.assertNotIn(DiscussionAct("vote_intent", 4), snapshot)
        for action in ("check_as_non_werewolf", "check_as_werewolf", "save", "poison"):
            self.assertNotIn(DiscussionAct(action, 3), snapshot)
        self.assertNotIn(DiscussionAct("point_as_werewolf", 3), snapshot)
        self.assertIn(DiscussionAct("point_as_werewolf", 7), snapshot)
        self.assertEqual(snapshot, freeze_discussion_candidates(normal))

        for identity in ("Villager", "Seer", "Witch"):
            with self.subTest(identity=identity):
                candidate_set = freeze_discussion_candidates(
                    _observation(identity=identity)
                )
                self.assertNotIn(
                    DiscussionAct("point_as_werewolf", 3),
                    candidate_set,
                )
                self.assertIn(
                    DiscussionAct("point_as_villager", 3),
                    candidate_set,
                )

    def test_day_prompt_encourages_existing_interaction_actions(self):
        observation = _observation()
        exact_roles, role_options = derive_belief_constraints(observation)
        prompt = build_day_cognition_prompt(
            observation,
            exact_roles=exact_roles,
            role_options=role_options,
            candidate_snapshot=freeze_discussion_candidates(observation),
            claim_catalog=build_public_claim_catalog(observation),
        )

        self.assertIn("INTERACTION GUIDANCE", prompt)
        self.assertIn("support(...) or oppose(...)", prompt)
        self.assertIn("current table judgment", prompt)
        self.assertIn("No new action type", prompt)

    def test_pk_vote_intent_uses_only_latest_draw_speech_queue(self):
        observation = _observation(identity="Villager", phase="2_day_speech_pk")
        observation["authoritative_public_state"]["phase"] = "speech_pk"
        observation["game_log"].append(
            Log(
                list(range(1, 8)),
                0,
                0,
                {"vote_outcome": "draw", "speech_queue": [2, 3, 5]},
                2,
                "第2天白天",
                "end_vote",
            )
        )

        self.assertEqual(derive_discussion_vote_targets(observation), (2, 5))
        vote_targets = tuple(
            act.target
            for act in freeze_discussion_candidates(observation)
            if act.action == "vote_intent"
        )
        self.assertEqual(vote_targets, (2, 5))
        self.assertNotIn(1, vote_targets)

        observation["game_log"] = observation["game_log"][:-1]
        with self.assertRaisesRegex(ValueError, "draw speech_queue"):
            freeze_discussion_candidates(observation)

    def test_vote_prompt_uses_fresh_belief_and_rejects_intent_inheritance(self):
        contradictory_belief = {
            "belief": "我是女巫，应投已死且不在合法集合中的player7。",
            "concise": "投player7。",
        }
        prompt = build_vote_prompt(
            _observation(phase="1_day_vote"),
            contradictory_belief,
            (0, 1, 4, 5),
        )

        self.assertIn("FRESH PRIVATE BELIEF", prompt)
        self.assertIn(contradictory_belief["belief"], prompt)
        self.assertIn("[0, 1, 4, 5]", prompt)
        self.assertIn("target 0 = abstain", prompt)
        self.assertIn("target 1..7 = vote for that player", prompt)
        self.assertIn("Environment authoritative information", prompt)
        self.assertIn("frozen legal candidate set", prompt)
        self.assertIn("fallible subjective assessment", prompt)
        self.assertIn("current alive/dead/exiled state", prompt)
        self.assertIn("actual self role", prompt)
        self.assertIn("exact-known private facts", prompt)
        self.assertIn("exact-known other-player roles", prompt)
        self.assertIn("authoritative premises and candidates control", prompt)
        self.assertIn("faction objective", prompt)
        self.assertNotIn(
            "Base the choice on the fresh belief and faction objective",
            prompt,
        )
        self.assertIn("Do not preserve or", prompt)
        self.assertIn("inherit a target merely because", prompt)


if __name__ == "__main__":
    unittest.main()
