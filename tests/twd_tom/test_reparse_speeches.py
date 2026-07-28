"""Tests for offline comparison of old and new speech actions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import script.twd_tom.reparse_speeches as reparse_module
from script.twd_tom.reparse_speeches import (
    ReparseConfig,
    build_reparse_report,
    load_game_log,
    run_reparse,
)


def make_speech_record(
    *,
    source,
    speech,
    old_actions,
    event="speech",
    day=1,
):
    return {
        "viewer": [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
        ],
        "source": source,
        "target": [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
        ],
        "content": {
            "speech_content": speech,
            "sp_actions": (
                old_actions
            ),
        },
        "day": day,
        "time": "day",
        "event": event,
    }


class FakeParser:
    def __init__(self):
        self.calls = []

    def parse(
        self,
        speaker,
        speech,
        day,
        phase,
    ):
        self.calls.append(
            {
                "speaker": speaker,
                "speech": speech,
                "day": day,
                "phase": phase,
            }
        )

        if speech == "我还是村民。":
            return [
                [
                    f"player{speaker}",
                    "point_as_villager",
                    f"player{speaker}",
                ]
            ]

        if speech == "我比较关注2号。":
            return [
                [
                    f"player{speaker}",
                    "oppose",
                    "player2",
                ]
            ]

        if speech == "解析失败":
            raise RuntimeError(
                "fake parser failure"
            )

        return []


def test_build_report_compares_old_and_new_actions():
    parser = FakeParser()

    records = [
        {
            "event": "vote",
            "source": 1,
            "content": {
                "vote_target": 2,
            },
            "day": 1,
        },
        make_speech_record(
            source=4,
            speech="我还是村民。",
            old_actions=[
                [
                    "player4",
                    "point_as_villager",
                    "player4",
                ]
            ],
        ),
        make_speech_record(
            source=5,
            speech="我比较关注2号。",
            old_actions=[],
        ),
        make_speech_record(
            source=6,
            speech="解析失败",
            old_actions=[
                [
                    "player6",
                    "support",
                    "player3",
                ]
            ],
            event="speech_pk",
        ),
    ]

    report = build_reparse_report(
        records,
        parser=parser,
        source_path=(
            "/tmp/game_log.json"
        ),
        source_sha256="abc123",
        parser_backend_name=(
            "fake_backend"
        ),
        parser_model_name=(
            "fake_model"
        ),
    )

    assert report[
        "status"
    ] == "ok"

    assert report[
        "purpose"
    ] == (
        "speech_parser_comparison_only"
    )

    summary = report[
        "summary"
    ]

    assert summary[
        "speech_event_count"
    ] == 3

    assert summary[
        "parser_call_count"
    ] == 3

    assert summary[
        "parser_error_count"
    ] == 1

    assert summary[
        "old_nonempty_event_count"
    ] == 2

    assert summary[
        "new_nonempty_event_count"
    ] == 2

    assert summary[
        "old_action_count"
    ] == 2

    assert summary[
        "new_action_count"
    ] == 2

    assert summary[
        "changed_event_count"
    ] == 2

    assert summary[
        "identical_event_count"
    ] == 1

    assert summary[
        "gained_nonempty_event_count"
    ] == 1

    assert summary[
        "lost_nonempty_event_count"
    ] == 1

    assert summary[
        "old_action_name_counts"
    ] == {
        "point_as_villager": 1,
        "support": 1,
    }

    assert summary[
        "new_action_name_counts"
    ] == {
        "oppose": 1,
        "point_as_villager": 1,
    }

    events = report[
        "events"
    ]

    assert events[0][
        "changed"
    ] is False

    assert events[1][
        "added_actions"
    ] == [
        [
            "player5",
            "oppose",
            "player2",
        ]
    ]

    assert events[2][
        "parse_status"
    ] == "error"

    assert (
        "fake parser failure"
        in events[2][
            "parse_error"
        ]
    )

    assert len(
        parser.calls
    ) == 3


def test_load_game_log_reads_list(
    tmp_path,
):
    path = (
        tmp_path
        / "game_log.json"
    )

    expected = [
        make_speech_record(
            source=1,
            speech="测试",
            old_actions=[],
        )
    ]

    path.write_text(
        json.dumps(
            expected,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert load_game_log(
        path
    ) == expected


def test_load_game_log_rejects_non_list(
    tmp_path,
):
    path = (
        tmp_path
        / "game_log.json"
    )

    path.write_text(
        json.dumps(
            {
                "event": "speech",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        TypeError,
        match="root must be a list",
    ):
        load_game_log(
            path
        )


def test_invalid_saved_speaker_is_rejected():
    parser = FakeParser()

    records = [
        make_speech_record(
            source=0,
            speech="测试",
            old_actions=[],
        )
    ]

    with pytest.raises(
        ValueError,
        match="one-based source",
    ):
        build_reparse_report(
            records,
            parser=parser,
            source_path="game.json",
            source_sha256="abc",
            parser_backend_name="fake",
            parser_model_name="fake",
        )


def test_log_without_public_speech_is_rejected():
    parser = FakeParser()

    with pytest.raises(
        ValueError,
        match="contains no speech",
    ):
        build_reparse_report(
            [
                {
                    "event": "vote",
                    "source": 1,
                    "content": {
                        "vote_target": 2,
                    },
                    "day": 1,
                }
            ],
            parser=parser,
            source_path="game.json",
            source_sha256="abc",
            parser_backend_name="fake",
            parser_model_name="fake",
        )


def test_run_reparse_writes_report_without_modifying_source(
    tmp_path,
    monkeypatch,
):
    game_log_path = (
        tmp_path
        / "game_log.json"
    )

    runtime_config_path = (
        tmp_path
        / "runtime.yaml"
    )

    output_path = (
        tmp_path
        / "reparse_report.json"
    )

    records = [
        make_speech_record(
            source=5,
            speech="我比较关注2号。",
            old_actions=[],
        )
    ]

    game_log_path.write_text(
        json.dumps(
            records,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    original_bytes = (
        game_log_path.read_bytes()
    )

    original_sha256 = (
        hashlib.sha256(
            original_bytes
        ).hexdigest()
    )

    runtime_config_path.write_text(
        """
backends:
  fake:
    type: openai_compatible
    api_key_env: FAKE_API_KEY
parser:
  backend: fake
  model: fake-model
agent_config:
  must_include: []
  all_candidates: []
env_config:
  n_player: 7
""".strip()
        + "\n",
        encoding="utf-8",
    )

    fake_backend = object()
    fake_parser = FakeParser()

    monkeypatch.setattr(
        reparse_module,
        "normalize_runtime_config",
        lambda config: {
            **dict(config),
            "parser": {
                "backend": "fake",
                "model": "fake-model",
                "model_params": {
                    "temperature": 0.0,
                },
            },
        },
    )

    monkeypatch.setattr(
        reparse_module,
        "load_named_backends",
        lambda config, env_file=None: {
            "fake": fake_backend,
        },
    )

    monkeypatch.setattr(
        reparse_module,
        "resolve_backend",
        lambda name, backends: (
            backends[name]
        ),
    )

    monkeypatch.setattr(
        reparse_module,
        "SpeechPerceiver",
        lambda backend, model_name: (
            fake_parser
        ),
    )

    result = run_reparse(
        ReparseConfig(
            game_log_path=str(
                game_log_path
            ),
            runtime_config_path=str(
                runtime_config_path
            ),
            output_path=str(
                output_path
            ),
            env_file=None,
        )
    )

    assert result[
        "status"
    ] == "ok"

    assert result[
        "speech_event_count"
    ] == 1

    assert result[
        "old_nonempty_event_count"
    ] == 0

    assert result[
        "new_nonempty_event_count"
    ] == 1

    assert output_path.is_file()

    stored_report = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    assert stored_report[
        "source_game_log"
    ][
        "sha256"
    ] == original_sha256

    assert stored_report[
        "events"
    ][0][
        "new_sp_actions"
    ] == [
        [
            "player5",
            "oppose",
            "player2",
        ]
    ]

    assert (
        game_log_path.read_bytes()
        == original_bytes
    )


def test_report_contains_no_truth_keys():
    parser = FakeParser()

    report = build_reparse_report(
        [
            make_speech_record(
                source=5,
                speech=(
                    "我比较关注2号。"
                ),
                old_actions=[],
            )
        ],
        parser=parser,
        source_path="game.json",
        source_sha256="abc",
        parser_backend_name="fake",
        parser_model_name="fake",
    )

    forbidden_keys = {
        "roles",
        "true_roles",
        "actual_wolves",
        "wolf_labels",
        "truth",
        "role_assignment",
        "god_view",
    }

    def visit(
        value,
    ):
        if isinstance(
            value,
            dict,
        ):
            for key, child in (
                value.items()
            ):
                assert (
                    key
                    not in forbidden_keys
                )

                visit(
                    child
                )

        elif isinstance(
            value,
            list,
        ):
            for child in value:
                visit(
                    child
                )

    visit(
        report
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "game_log_path": "",
        },
        {
            "runtime_config_path": "",
        },
        {
            "output_path": "",
        },
        {
            "parser_model": "",
        },
    ],
)
def test_invalid_config_is_rejected(
    kwargs,
):
    arguments = {
        "game_log_path": (
            "game_log.json"
        ),
        "runtime_config_path": (
            "runtime.yaml"
        ),
        "output_path": (
            "report.json"
        ),
    }

    arguments.update(
        kwargs
    )

    with pytest.raises(
        ValueError,
    ):
        ReparseConfig(
            **arguments
        )
