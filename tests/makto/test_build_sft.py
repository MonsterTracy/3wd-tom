import json

import pytest

from script.makto.build_sft import (
    DatasetBuildError,
    SOURCE_REVISION,
    SOURCE_SETTING,
    build_dataset,
    replay_game,
    seer_candidates,
    vote_candidates,
    witch_candidates,
    wolf_candidates,
)
from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0


ROLES = {
    1: "werewolf",
    2: "seer",
    3: "witch",
    4: "werewolf",
    5: "simple_villager",
    6: "simple_villager",
    7: "simple_villager",
}


def _roles():
    return [
        {
            "event": "roles",
            "content": {"player": player, "role": role, "uid": player * 10},
        }
        for player, role in ROLES.items()
    ]


def _cycle(round_number, status):
    return {
        "event": "cycle_round",
        "content": {"round": round_number, "status": status},
    }


def _speech(player, text, day="1-1"):
    return {
        "event": "speech",
        "content": {"player": player, "context": text, "day": day},
    }


def _vote(player, target, day="1-1", marker="ANNOTATION-SECRET"):
    return {
        "event": "voted",
        "content": {
            "player": player,
            "voted_to_player": target,
            "day": day,
            "reason": marker,
            "role_prediction": {"marker": marker},
        },
    }


def _record(records, task, actor=None, occurrence=0):
    matches = [
        record
        for record in records
        if record["task"] == task
        and (actor is None or record["actor"] == actor)
    ]
    return matches[occurrence]


def test_causal_speech_keeps_only_prior_public_speech_and_own_history():
    events = _roles() + [
        _cycle(1, "day"),
        _speech(5, "FIRST-OWN-SPEECH"),
        {
            "event": "speech_summary",
            "content": {
                "player": 5,
                "day": "1-1",
                "identity_label": "SUMMARY-SECRET",
                "call_for_vote": "CALL-FOR-VOTE-SECRET",
                "self_present": "SELF-PRESENT-SECRET",
            },
        },
        _speech(6, "SECOND-SPEECH"),
        {"event": "review", "content": {"context": "REVIEW-SECRET"}},
        _speech(5, "FUTURE-OWN-SPEECH"),
    ]

    records, _excluded = replay_game(events, game_id="synthetic_speech")
    first = _record(records, "speech", actor=5, occurrence=0)
    second = _record(records, "speech", actor=6)
    third = _record(records, "speech", actor=5, occurrence=1)

    assert "SECOND-SPEECH" not in first["messages"][0]["content"]
    assert "FUTURE-OWN-SPEECH" not in first["messages"][0]["content"]
    assert "FIRST-OWN-SPEECH" in second["messages"][0]["content"]
    assert "FUTURE-OWN-SPEECH" not in second["messages"][0]["content"]
    assert "FIRST-OWN-SPEECH" in third["messages"][0]["content"]
    assert "SECOND-SPEECH" in third["messages"][0]["content"]
    assert first["messages"][1]["content"] == "FIRST-OWN-SPEECH"
    assert all(
        marker not in record["messages"][0]["content"]
        for record in records
        for marker in (
            "SUMMARY-SECRET",
            "CALL-FOR-VOTE-SECRET",
            "SELF-PRESENT-SECRET",
            "REVIEW-SECRET",
        )
    )


def test_role_private_information_is_causal_and_actor_scoped():
    events = _roles() + [
        _cycle(1, "night"),
        {
            "event": "inquired",
            "content": {"night": 1, "player": 5, "is_werewolf": False},
        },
        {
            "event": "werewolf_night_discuss",
            "content": {
                "night": 1,
                "context": [],
                "decision_kill": {"1": 5, "4": 5},
            },
        },
        {
            "event": "werewolf_kill",
            "content": {"night": 1, "target_player": 5},
        },
        {"event": "healed", "content": {"night": 1, "player": None}},
        {"event": "poison", "content": {"night": 1, "player": None}},
        _cycle(1, "day"),
        _speech(6, "VILLAGER"),
        _speech(1, "WOLF"),
        _speech(2, "SEER"),
        _speech(3, "WITCH"),
    ]

    records, _excluded = replay_game(events, game_id="synthetic_roles")
    villager_prompt = _record(records, "speech", actor=6)["messages"][0]["content"]
    wolf_prompt = _record(records, "speech", actor=1)["messages"][0]["content"]
    seer_prompt = _record(records, "speech", actor=2)["messages"][0]["content"]
    witch_prompt = _record(records, "witch", actor=3)["messages"][0]["content"]
    first_wolf = _record(records, "wolf", actor=1)["messages"][0]["content"]
    second_wolf = _record(records, "wolf", actor=4)["messages"][0]["content"]

    assert "你知道的狼人队伍" not in villager_prompt
    assert "你已完成的查验" not in villager_prompt
    assert "你知道的狼人队伍：1号、4号" in wolf_prompt
    assert "2号是预言家" not in wolf_prompt
    assert "5号是非狼人" in seer_prompt
    assert "本夜狼人最终目标：5号" in witch_prompt
    assert "查验" not in witch_prompt.split("你合法知道的私有信息：", 1)[1].split(
        "按时间顺序公开的历史：", 1
    )[0]
    assert "本夜在你之前的狼人选择" not in first_wolf
    assert "本夜在你之前的狼人选择：1号选择5号" in second_wolf


def test_completed_wolf_nights_persist_only_in_wolf_private_history():
    events = _roles() + [
        _cycle(1, "night"),
        {
            "event": "werewolf_night_discuss",
            "content": {
                "night": 1,
                "context": [],
                "decision_kill": {"1": 5, "4": 6},
            },
        },
        {
            "event": "werewolf_kill",
            "content": {"night": 1, "target_player": 6},
        },
        {"event": "healed", "content": {"night": 1, "player": None}},
        {"event": "poison", "content": {"night": 1, "player": None}},
        _cycle(1, "day"),
        _speech(1, "WOLF-DAY-ONE"),
        _speech(2, "SEER-DAY-ONE"),
        _speech(3, "WITCH-DAY-ONE"),
        _speech(5, "VILLAGER-DAY-ONE"),
        _cycle(2, "night"),
        {
            "event": "werewolf_night_discuss",
            "content": {
                "night": 2,
                "context": [],
                "decision_kill": {"1": 2, "4": None},
            },
        },
        {
            "event": "werewolf_kill",
            "content": {"night": 2, "target_player": 2},
        },
        {"event": "healed", "content": {"night": 2, "player": None}},
        {"event": "poison", "content": {"night": 2, "player": None}},
        _cycle(2, "day"),
        _speech(1, "WOLF-DAY-TWO", day="2-1"),
    ]

    records, _excluded = replay_game(events, game_id="wolf_private_history")
    night_one = (
        "第1夜已完成狼人行动：1号选择5号；4号选择6号；"
        "狼队最终目标为6号。"
    )
    night_two = (
        "第2夜已完成狼人行动：1号选择2号；4号选择pass；"
        "狼队最终目标为2号。"
    )
    first_night_two_wolf = _record(
        records, "wolf", actor=1, occurrence=1
    )["messages"][0]["content"]
    second_night_two_wolf = _record(
        records, "wolf", actor=4, occurrence=1
    )["messages"][0]["content"]
    later_wolf_speech = _record(
        records, "speech", actor=1, occurrence=1
    )["messages"][0]["content"]

    assert night_one in first_night_two_wolf
    assert night_two not in first_night_two_wolf
    assert "本夜在你之前的狼人选择" not in first_night_two_wolf
    assert night_one in second_night_two_wolf
    assert night_two not in second_night_two_wolf
    assert "本夜在你之前的狼人选择：1号选择2号" in second_night_two_wolf
    assert "4号选择pass" not in second_night_two_wolf
    assert night_one in later_wolf_speech
    assert night_two in later_wolf_speech
    assert "选择0号" not in "\n".join(
        record["messages"][0]["content"] for record in records
    )

    for actor in (2, 3, 5):
        non_wolf_prompt = _record(
            records, "speech", actor=actor
        )["messages"][0]["content"]
        assert "已完成狼人行动" not in non_wolf_prompt
        assert "狼人选择" not in non_wolf_prompt


def test_completed_wolf_night_requires_consistent_individual_decisions():
    missing_decisions = _roles() + [
        _cycle(1, "night"),
        {
            "event": "werewolf_kill",
            "content": {"night": 1, "target_player": 5},
        },
    ]
    with pytest.raises(
        DatasetBuildError,
        match="without current-night individual decisions",
    ):
        replay_game(missing_decisions, game_id="missing_wolf_decisions")

    inconsistent_final = _roles() + [
        _cycle(1, "night"),
        {
            "event": "werewolf_night_discuss",
            "content": {
                "night": 1,
                "context": [],
                "decision_kill": {"1": 5, "4": 6},
            },
        },
        {
            "event": "werewolf_kill",
            "content": {"night": 1, "target_player": 7},
        },
    ]
    with pytest.raises(DatasetBuildError, match="absent from individual choices"):
        replay_game(inconsistent_final, game_id="inconsistent_wolf_final")


def test_all_task_prompts_state_exact_win_conditions():
    events = _roles() + [
        _cycle(1, "night"),
        {
            "event": "inquired",
            "content": {"night": 1, "player": 5, "is_werewolf": False},
        },
        {
            "event": "werewolf_night_discuss",
            "content": {
                "night": 1,
                "context": [],
                "decision_kill": {"1": 7, "4": 7},
            },
        },
        {
            "event": "werewolf_kill",
            "content": {"night": 1, "target_player": 7},
        },
        {"event": "healed", "content": {"night": 1, "player": None}},
        {"event": "poison", "content": {"night": 1, "player": None}},
        _cycle(1, "day"),
        _speech(5, "SPEECH"),
        {"event": "vote_start", "content": None},
        _vote(5, None),
    ]

    records, _excluded = replay_game(events, game_id="win_conditions")
    by_task = {
        task: _record(records, task)["messages"][0]["content"]
        for task in ("speech", "vote", "wolf", "seer", "witch")
    }
    assert set(by_task) == {"speech", "vote", "wolf", "seer", "witch"}
    for prompt in by_task.values():
        assert "所有狼人出局 → 村民阵营获胜" in prompt
        assert "3名普通村民全部出局（即所有普通村民出局）" in prompt
        assert "预言家与女巫全部出局（即所有神职出局）" in prompt
        assert "狼人阵营获胜" in prompt
        assert "你的目标是使自己所属阵营达到其胜利条件。" in prompt


def test_builder_candidate_order_matches_frozen_environment_semantics():
    runtime_roles = [
        "Werewolf",
        "Seer",
        "Witch",
        "Werewolf",
        "Villager",
        "Villager",
        "Villager",
    ]
    env = WerewolfTextEnvV0(log_save_path=None)
    env.reset(roles=runtime_roles)
    env.alive = [1, 1, 0, 1, 1, 0, 1]
    alive = {1, 2, 4, 5, 7}

    env.phase = "skill_wolf"
    env.current_act_idx = 0
    assert wolf_candidates(alive) == [
        list(action) for action in env.get_observation()["valid_action"]
    ]

    env.phase = "skill_seer"
    env.current_act_idx = 1
    env.seer_check_target = {"prior": 4}
    assert seer_candidates(alive, 2, {5}) == [
        list(action) for action in env.get_observation()["valid_action"]
    ]

    env.phase = "skill_witch"
    env.current_act_idx = 2
    phase_id = env.get_phase(env.day, env.day_or_night, "skill_wolf")
    env.werewolf_kill_decision[phase_id] = 4
    assert witch_candidates(
        alive,
        antidote_available=True,
        poison_available=True,
        wolf_target=5,
    ) == [list(action) for action in env.get_observation()["valid_action"]]

    env.phase = "vote"
    env.current_act_idx = 1
    assert vote_candidates(alive, 2) == [
        list(action) for action in env.get_observation()["valid_action"]
    ]


def test_vote_snapshot_maps_abstain_and_rejects_illegal_observed_target():
    events = _roles() + [
        _cycle(1, "day"),
        {"event": "vote_start", "content": None},
        _vote(5, None),
        {"event": "vote_results", "content": {}},
    ]
    records, _excluded = replay_game(events, game_id="synthetic_vote")
    vote = _record(records, "vote")
    assert vote["candidate_snapshot"] == vote_candidates(set(range(1, 8)), 5)
    assert vote["messages"][1]["content"] == '{"action_index":0}'
    assert "ANNOTATION-SECRET" not in vote["messages"][0]["content"]

    illegal = _roles() + [
        _cycle(1, "day"),
        {"event": "vote_start", "content": None},
        _vote(5, 5),
    ]
    with pytest.raises(DatasetBuildError, match="outside candidate snapshot"):
        replay_game(illegal, game_id="illegal_vote")


def test_wolf_self_and_teammate_targets_map_and_dead_target_rejects():
    legal = _roles() + [
        _cycle(1, "night"),
        {
            "event": "werewolf_night_discuss",
            "content": {
                "night": 1,
                "context": [],
                "decision_kill": {"1": 1, "4": 1},
            },
        },
    ]
    records, _excluded = replay_game(legal, game_id="legal_wolf_targets")
    wolf_records = [record for record in records if record["task"] == "wolf"]
    assert [record["messages"][1]["content"] for record in wolf_records] == [
        '{"action_index":1}',
        '{"action_index":1}',
    ]
    assert "1号选择1号" in wolf_records[1]["messages"][0]["content"]

    dead_target = _roles() + [
        _cycle(1, "night"),
        {
            "event": "werewolf_night_discuss",
            "content": {"night": 1, "context": [], "decision_kill": {"1": 5}},
        },
        {
            "event": "werewolf_kill",
            "content": {"night": 1, "target_player": 5},
        },
        {"event": "healed", "content": {"night": 1, "player": None}},
        {"event": "poison", "content": {"night": 1, "player": None}},
        _cycle(1, "day"),
        _cycle(2, "night"),
        {
            "event": "werewolf_night_discuss",
            "content": {"night": 2, "context": [], "decision_kill": {"1": 5}},
        },
    ]
    with pytest.raises(DatasetBuildError, match="outside candidate snapshot"):
        replay_game(dead_target, game_id="dead_wolf_target")


def test_seer_result_enters_only_later_private_state_and_repeat_rejects():
    legal = _roles() + [
        _cycle(1, "night"),
        {
            "event": "inquired",
            "content": {"night": 1, "player": 5, "is_werewolf": False},
        },
        _cycle(1, "day"),
        _cycle(2, "night"),
        {
            "event": "inquired",
            "content": {"night": 2, "player": 6, "is_werewolf": False},
        },
    ]
    records, _excluded = replay_game(legal, game_id="causal_seer")
    first = _record(records, "seer", occurrence=0)
    second = _record(records, "seer", occurrence=1)
    first_private = first["messages"][0]["content"].split(
        "你合法知道的私有信息：", 1
    )[1].split("按时间顺序公开的历史：", 1)[0]
    assert "5号是非狼人" not in first_private
    assert "5号是非狼人" in second["messages"][0]["content"]
    assert "6号是非狼人" not in second["messages"][0]["content"]
    assert ["check", 2] not in first["candidate_snapshot"]
    assert ["check", 5] not in second["candidate_snapshot"]

    repeated = legal[:-1] + [
        {
            "event": "inquired",
            "content": {"night": 2, "player": 5, "is_werewolf": False},
        }
    ]
    with pytest.raises(DatasetBuildError, match="outside candidate snapshot"):
        replay_game(repeated, game_id="repeat_seer")

    self_check = _roles() + [
        _cycle(1, "night"),
        {
            "event": "inquired",
            "content": {"night": 1, "player": 2, "is_werewolf": False},
        },
    ]
    with pytest.raises(DatasetBuildError, match="outside candidate snapshot"):
        replay_game(self_check, game_id="self_check_seer")


def test_witch_contamination_excludes_only_that_witch_later_behavior():
    events = _roles() + [
        _cycle(1, "night"),
        {
            "event": "werewolf_night_discuss",
            "content": {
                "night": 1,
                "context": [],
                "decision_kill": {"1": 5, "4": 5},
            },
        },
        {
            "event": "werewolf_kill",
            "content": {"night": 1, "target_player": 5},
        },
        {"event": "healed", "content": {"night": 1, "player": 5}},
        {"event": "poison", "content": {"night": 1, "player": None}},
        _cycle(1, "day"),
        _cycle(2, "night"),
        {
            "event": "werewolf_night_discuss",
            "content": {
                "night": 2,
                "context": [],
                "decision_kill": {"1": 6, "4": 6},
            },
        },
        {
            "event": "werewolf_kill",
            "content": {"night": 2, "target_player": 6},
        },
        {"event": "poison", "content": {"night": 2, "player": 7}},
        _cycle(2, "day"),
        _speech(3, "CONTAMINATED-WITCH-SPEECH", day="2-1"),
        _speech(1, "CLEAN-WOLF-SPEECH", day="2-1"),
        {"event": "vote_start", "content": None},
        _vote(3, 1, day="2-1"),
        _vote(1, None, day="2-1"),
        {"event": "vote_results", "content": {"3": 1}},
    ]

    records, excluded = replay_game(events, game_id="witch_contamination")
    assert len([record for record in records if record["task"] == "witch"]) == 1
    assert excluded == {
        "witch_after_antidote_spent": 1,
        "later_contaminated_witch_speech": 1,
        "later_contaminated_witch_vote": 1,
    }
    assert not any(
        record["actor"] == 3 and record["task"] in {"speech", "vote"}
        for record in records
    )
    assert _record(records, "speech", actor=1)["messages"][1]["content"] == (
        "CLEAN-WOLF-SPEECH"
    )
    assert _record(records, "vote", actor=1)["messages"][1]["content"] == (
        '{"action_index":0}'
    )


def test_output_schema_annotation_exclusion_and_bytes_are_deterministic(tmp_path):
    events = _roles() + [
        {"event": "end_rule", "content": "RULE-ANNOTATION"},
        _cycle(1, "day"),
        _speech(5, "RAW-HUMAN-SPEECH"),
        {
            "event": "speech_summary",
            "content": {
                "player": 5,
                "day": "1-1",
                "identity_label": "SUMMARY-SECRET",
                "call_for_vote": "CALL-FOR-VOTE-SECRET",
                "self_present": "SELF-PRESENT-SECRET",
            },
        },
        {
            "event": "bad_player",
            "content": {"player_ids": ["BAD-PLAYER-SECRET"], "uids": []},
        },
        {"event": "vote_start", "content": None},
        _vote(5, None, marker="REASON-SECRET"),
        {"event": "vote_results", "content": {}},
        {"event": "review", "content": {"context": "REVIEW-SECRET"}},
        {"event": "end", "content": None},
    ]
    source_root = tmp_path / "source"
    game_dir = source_root / "raw/train/7_player_game/seer_witch/game_1"
    game_dir.mkdir(parents=True)
    (game_dir / "event_zh.json").write_text(
        json.dumps(events, ensure_ascii=False),
        encoding="utf-8",
    )

    output_a = tmp_path / "a.jsonl"
    output_b = tmp_path / "b.jsonl"
    _, manifest_a, _ = build_dataset(
        source_root=source_root,
        output=output_a,
        enforce_expected=False,
    )
    _, manifest_b, _ = build_dataset(
        source_root=source_root,
        output=output_b,
        enforce_expected=False,
    )
    assert output_a.read_bytes() == output_b.read_bytes()
    assert manifest_a.read_bytes() == manifest_b.read_bytes()

    records = [json.loads(line) for line in output_a.read_text().splitlines()]
    assert set(records[0]) == {
        "source",
        "task",
        "actor",
        "role",
        "candidate_snapshot",
        "messages",
    }
    assert records[0]["source"] == {
        "dataset": "makto",
        "revision": SOURCE_REVISION,
        "split": "train",
        "setting": SOURCE_SETTING,
        "game_id": "game_1",
        "event_index": 9,
    }
    visible = "\n".join(
        message["content"]
        for record in records
        for message in record["messages"]
    )
    for forbidden in (
        "RULE-ANNOTATION",
        "SUMMARY-SECRET",
        "CALL-FOR-VOTE-SECRET",
        "SELF-PRESENT-SECRET",
        "BAD-PLAYER-SECRET",
        "REASON-SECRET",
        "REVIEW-SECRET",
    ):
        assert forbidden not in visible
    assert SOURCE_REVISION not in visible
    assert "game_1" not in visible
    assert records[0]["candidate_snapshot"] is None
    assert records[0]["messages"][1]["content"] == "RAW-HUMAN-SPEECH"

    rejected_output = tmp_path / "rejected.jsonl"
    with pytest.raises(DatasetBuildError, match="expected 20 games, got 1"):
        build_dataset(source_root=source_root, output=rejected_output)
    assert not rejected_output.exists()
