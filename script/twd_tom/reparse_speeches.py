"""Reparse public speeches from an existing Werewolf game log.

This tool performs an offline comparison between:

    old_sp_actions stored in game_log.json
    new_sp_actions produced by the current SpeechPerceiver

It does not:

- rerun the Werewolf game;
- collect subjective belief labels;
- modify the source game log;
- read or export hidden role assignments.

Only ``speech`` and ``speech_pk`` records are processed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from werewolf.backends import (
    load_named_backends,
    resolve_backend,
)
from werewolf.runtime_config import (
    normalize_runtime_config,
)
from werewolf.speech.speech_perceiver import (
    SpeechActionValidationError,
    SpeechPerceiver,
)


PUBLIC_SPEECH_EVENTS = {
    "speech",
    "speech_pk",
}

FORBIDDEN_TRUTH_KEYS = {
    "roles",
    "true_roles",
    "actual_wolves",
    "wolf_labels",
    "truth",
    "role_assignment",
    "god_view",
}


@dataclass(frozen=True)
class ReparseConfig:
    """Configuration for one speech reparse job."""

    game_log_path: str
    runtime_config_path: str
    output_path: str

    env_file: str | None = ".env"
    parser_model: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty_string(
            self.game_log_path,
            field_name="game_log_path",
        )

        _require_non_empty_string(
            self.runtime_config_path,
            field_name="runtime_config_path",
        )

        _require_non_empty_string(
            self.output_path,
            field_name="output_path",
        )

        if self.env_file is not None:
            _require_non_empty_string(
                self.env_file,
                field_name="env_file",
            )

        if self.parser_model is not None:
            _require_non_empty_string(
                self.parser_model,
                field_name="parser_model",
            )


def _require_non_empty_string(
    value: Any,
    *,
    field_name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise ValueError(
            f"{field_name} must be a non-empty string"
        )

    return value


def _sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def load_game_log(
    path: str | Path,
) -> list[dict[str, Any]]:
    """Load and validate one saved game_log.json file."""

    resolved_path = Path(
        path
    ).resolve()

    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"game log not found: {resolved_path}"
        )

    try:
        value = json.loads(
            resolved_path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            "game log is not valid JSON"
        ) from exc

    if not isinstance(value, list):
        raise TypeError(
            "game log root must be a list"
        )

    records: list[
        dict[str, Any]
    ] = []

    for record_index, record in enumerate(
        value,
        start=1,
    ):
        if not isinstance(
            record,
            Mapping,
        ):
            raise TypeError(
                "game log record "
                f"{record_index} must be a mapping"
            )

        records.append(
            dict(record)
        )

    return records


def _normalize_action_rows(
    value: Any,
) -> tuple[
    list[list[str]],
    int,
]:
    """Normalize valid action triplets and count malformed entries."""

    if not isinstance(value, list):
        return [], 1

    normalized: list[
        list[str]
    ] = []

    invalid_count = 0

    for item in value:
        if (
            not isinstance(
                item,
                Sequence,
            )
            or isinstance(
                item,
                (str, bytes),
            )
            or len(item) != 3
        ):
            invalid_count += 1
            continue

        subject, action, object_player = (
            item
        )

        if not all(
            isinstance(part, str)
            and part.strip()
            for part in (
                subject,
                action,
                object_player,
            )
        ):
            invalid_count += 1
            continue

        normalized.append(
            [
                subject.strip(),
                action.strip(),
                object_player.strip(),
            ]
        )

    return normalized, invalid_count


def _action_keys(
    actions: Sequence[
        Sequence[str]
    ],
) -> set[
    tuple[str, str, str]
]:
    return {
        (
            action[0],
            action[1],
            action[2],
        )
        for action in actions
        if len(action) == 3
    }


def _action_name_counts(
    events: Sequence[
        Mapping[str, Any]
    ],
    *,
    field_name: str,
) -> dict[str, int]:
    counter: Counter[str] = (
        Counter()
    )

    for event in events:
        actions = event.get(
            field_name,
            [],
        )

        if not isinstance(
            actions,
            list,
        ):
            continue

        for action in actions:
            if (
                isinstance(action, list)
                and len(action) == 3
                and isinstance(
                    action[1],
                    str,
                )
            ):
                counter[
                    action[1]
                ] += 1

    return {
        action_name: int(count)
        for action_name, count
        in sorted(
            counter.items()
        )
    }


def _validate_saved_speaker(
    value: Any,
    *,
    log_index: int,
) -> int:
    """Validate the one-based speaker ID in a saved game log."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= 7
    ):
        raise ValueError(
            "saved speech record must contain "
            "a one-based source in [1, 7]; "
            f"log record {log_index} has "
            f"source={value!r}"
        )

    return value


def _validate_day(
    value: Any,
    *,
    log_index: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            "speech day must be a "
            "non-negative integer; "
            f"log record {log_index} has "
            f"day={value!r}"
        )

    return value


def build_reparse_report(
    game_log_records: Sequence[
        Mapping[str, Any]
    ],
    *,
    parser: Any,
    source_path: str,
    source_sha256: str,
    parser_backend_name: str,
    parser_model_name: str,
) -> dict[str, Any]:
    """Reparse all public speeches and build a comparison report."""

    if (
        isinstance(
            game_log_records,
            (str, bytes),
        )
        or not isinstance(
            game_log_records,
            Sequence,
        )
    ):
        raise TypeError(
            "game_log_records must be a sequence"
        )

    if parser is None or not hasattr(
        parser,
        "parse_strict",
    ):
        raise TypeError(
            "parser must provide parse_strict()"
        )

    events: list[
        dict[str, Any]
    ] = []

    speech_index = 0
    parser_call_count = 0
    parser_error_count = 0
    invalid_old_action_count = 0
    invalid_new_action_count = 0

    for log_index, raw_record in enumerate(
        game_log_records,
        start=1,
    ):
        if not isinstance(
            raw_record,
            Mapping,
        ):
            raise TypeError(
                f"log record {log_index} must be a mapping"
            )

        phase = raw_record.get(
            "event"
        )

        if phase not in PUBLIC_SPEECH_EVENTS:
            continue

        speech_index += 1

        speaker_id = (
            _validate_saved_speaker(
                raw_record.get(
                    "source"
                ),
                log_index=log_index,
            )
        )

        day = _validate_day(
            raw_record.get(
                "day",
                0,
            ),
            log_index=log_index,
        )

        content = raw_record.get(
            "content"
        )

        if not isinstance(
            content,
            Mapping,
        ):
            raise TypeError(
                "speech record content must "
                "be a mapping; "
                f"log record {log_index} is invalid"
            )

        speech = content.get(
            "speech_content"
        )

        if not isinstance(
            speech,
            str,
        ):
            raise TypeError(
                "speech_content must be text; "
                f"log record {log_index} is invalid"
            )

        (
            old_actions,
            old_invalid_count,
        ) = _normalize_action_rows(
            content.get(
                "sp_actions",
                [],
            )
        )

        invalid_old_action_count += (
            old_invalid_count
        )

        parse_status = "ok"
        parse_error = None
        invalid_new_actions: list[
            dict[str, Any]
        ] = []
        new_actions: list[
            list[str]
        ] = []

        if not speech.strip():
            parse_status = (
                "skipped_empty_speech"
            )
        else:
            parser_call_count += 1

            try:
                raw_new_actions = (
                    parser.parse_strict(
                        speaker=speaker_id,
                        speech=speech,
                        day=day,
                        phase=str(phase),
                    )
                )

                (
                    new_actions,
                    new_invalid_count,
                ) = _normalize_action_rows(
                    raw_new_actions
                )

                invalid_new_action_count += (
                    new_invalid_count
                )

                if new_invalid_count:
                    parse_status = (
                        "invalid_parser_output"
                    )
            except SpeechActionValidationError as exc:
                invalid_new_action_count += (
                    exc.invalid_count
                )
                parse_status = (
                    "invalid_parser_output"
                )
                parse_error = (
                    f"{type(exc).__name__}: {exc}"
                )
                invalid_new_actions = (
                    exc.failures
                )
                new_actions = []
            except Exception as exc:
                parser_error_count += 1
                parse_status = "error"
                parse_error = (
                    f"{type(exc).__name__}: {exc}"
                )
                new_actions = []

        old_keys = _action_keys(
            old_actions
        )

        new_keys = _action_keys(
            new_actions
        )

        added_keys = sorted(
            new_keys - old_keys
        )

        removed_keys = sorted(
            old_keys - new_keys
        )

        event_record = {
            "speech_index": (
                speech_index
            ),
            "log_index": log_index,
            "speaker_id": speaker_id,
            "speaker": (
                f"player{speaker_id}"
            ),
            "day": day,
            "phase": phase,
            "speech": speech,
            "parse_status": (
                parse_status
            ),
            "parse_error": parse_error,
            "invalid_new_actions": (
                invalid_new_actions
            ),
            "old_sp_actions": (
                old_actions
            ),
            "new_sp_actions": (
                new_actions
            ),
            "old_action_count": len(
                old_actions
            ),
            "new_action_count": len(
                new_actions
            ),
            "old_nonempty": bool(
                old_actions
            ),
            "new_nonempty": bool(
                new_actions
            ),
            "changed": (
                old_actions
                != new_actions
            ),
            "added_actions": [
                list(action)
                for action in added_keys
            ],
            "removed_actions": [
                list(action)
                for action in removed_keys
            ],
            "invalid_old_action_count": (
                old_invalid_count
            ),
        }

        events.append(
            event_record
        )

    if not events:
        raise ValueError(
            "game log contains no speech "
            "or speech_pk records"
        )

    speech_event_count = len(
        events
    )

    old_nonempty_event_count = sum(
        bool(
            event[
                "old_nonempty"
            ]
        )
        for event in events
    )

    new_nonempty_event_count = sum(
        bool(
            event[
                "new_nonempty"
            ]
        )
        for event in events
    )

    old_action_count = sum(
        int(
            event[
                "old_action_count"
            ]
        )
        for event in events
    )

    new_action_count = sum(
        int(
            event[
                "new_action_count"
            ]
        )
        for event in events
    )

    changed_event_count = sum(
        bool(
            event["changed"]
        )
        for event in events
    )

    gained_nonempty_event_count = sum(
        (
            not event[
                "old_nonempty"
            ]
            and event[
                "new_nonempty"
            ]
        )
        for event in events
    )

    lost_nonempty_event_count = sum(
        (
            event[
                "old_nonempty"
            ]
            and not event[
                "new_nonempty"
            ]
        )
        for event in events
    )

    summary = {
        "speech_event_count": (
            speech_event_count
        ),
        "parser_call_count": (
            parser_call_count
        ),
        "parser_error_count": (
            parser_error_count
        ),
        "old_nonempty_event_count": (
            old_nonempty_event_count
        ),
        "new_nonempty_event_count": (
            new_nonempty_event_count
        ),
        "old_nonempty_coverage": (
            old_nonempty_event_count
            / speech_event_count
        ),
        "new_nonempty_coverage": (
            new_nonempty_event_count
            / speech_event_count
        ),
        "old_action_count": (
            old_action_count
        ),
        "new_action_count": (
            new_action_count
        ),
        "changed_event_count": (
            changed_event_count
        ),
        "identical_event_count": (
            speech_event_count
            - changed_event_count
        ),
        "gained_nonempty_event_count": (
            gained_nonempty_event_count
        ),
        "lost_nonempty_event_count": (
            lost_nonempty_event_count
        ),
        "invalid_old_action_count": (
            invalid_old_action_count
        ),
        "invalid_new_action_count": (
            invalid_new_action_count
        ),
        "old_action_name_counts": (
            _action_name_counts(
                events,
                field_name=(
                    "old_sp_actions"
                ),
            )
        ),
        "new_action_name_counts": (
            _action_name_counts(
                events,
                field_name=(
                    "new_sp_actions"
                ),
            )
        ),
    }

    return {
        "status": "ok",
        "purpose": (
            "speech_parser_comparison_only"
        ),
        "source_game_log": {
            "path": source_path,
            "sha256": source_sha256,
        },
        "parser": {
            "backend": (
                parser_backend_name
            ),
            "model": (
                parser_model_name
            ),
        },
        "summary": summary,
        "events": events,
    }


def _atomic_json_dump(
    value: Any,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(
        path
    )


def _find_forbidden_truth_keys(
    value: Any,
    *,
    path: str = "root",
) -> list[str]:
    """Verify that the comparison report contains no truth fields."""

    found: list[str] = []

    if isinstance(
        value,
        Mapping,
    ):
        for key, child in (
            value.items()
        ):
            child_path = (
                f"{path}.{key}"
            )

            if key in FORBIDDEN_TRUTH_KEYS:
                found.append(
                    child_path
                )

            found.extend(
                _find_forbidden_truth_keys(
                    child,
                    path=child_path,
                )
            )

    elif isinstance(
        value,
        list,
    ):
        for index, child in enumerate(
            value
        ):
            found.extend(
                _find_forbidden_truth_keys(
                    child,
                    path=(
                        f"{path}[{index}]"
                    ),
                )
            )

    return found


def run_reparse(
    config: ReparseConfig,
) -> dict[str, Any]:
    """Run the configured parser over an existing game log."""

    game_log_path = Path(
        config.game_log_path
    ).resolve()

    runtime_config_path = Path(
        config.runtime_config_path
    ).resolve()

    output_path = Path(
        config.output_path
    ).resolve()

    if not runtime_config_path.is_file():
        raise FileNotFoundError(
            "runtime config not found: "
            f"{runtime_config_path}"
        )

    try:
        raw_runtime_config = (
            yaml.safe_load(
                runtime_config_path.read_text(
                    encoding="utf-8"
                )
            )
        )
    except yaml.YAMLError as exc:
        raise ValueError(
            "runtime config is not valid YAML"
        ) from exc

    if not isinstance(
        raw_runtime_config,
        Mapping,
    ):
        raise TypeError(
            "runtime config root must be a mapping"
        )

    normalized_runtime_config = (
        normalize_runtime_config(
            raw_runtime_config
        )
    )

    parser_config = (
        normalized_runtime_config[
            "parser"
        ]
    )

    parser_backend_name = (
        parser_config[
            "backend"
        ]
    )

    parser_model_name = (
        config.parser_model
        or parser_config[
            "model"
        ]
    )

    backend_map = (
        load_named_backends(
            raw_runtime_config,
            env_file=(
                config.env_file
            ),
        )
    )

    parser_backend = (
        resolve_backend(
            parser_backend_name,
            backend_map,
        )
    )

    parser = SpeechPerceiver(
        backend=parser_backend,
        model_name=(
            parser_model_name
        ),
    )

    source_sha256 = _sha256_file(
        game_log_path
    )

    game_log_records = (
        load_game_log(
            game_log_path
        )
    )

    report = build_reparse_report(
        game_log_records,
        parser=parser,
        source_path=str(
            game_log_path
        ),
        source_sha256=(
            source_sha256
        ),
        parser_backend_name=(
            parser_backend_name
        ),
        parser_model_name=(
            parser_model_name
        ),
    )

    forbidden_paths = (
        _find_forbidden_truth_keys(
            report
        )
    )

    if forbidden_paths:
        raise RuntimeError(
            "truth-related fields unexpectedly "
            "entered reparse report: "
            f"{forbidden_paths}"
        )

    _atomic_json_dump(
        report,
        output_path,
    )

    result = {
        "status": "ok",
        "output_path": str(
            output_path
        ),
        **report["summary"],
    }

    return result


def build_arg_parser() -> (
    argparse.ArgumentParser
):
    parser = argparse.ArgumentParser(
        description=(
            "Reparse existing public speeches "
            "with the current ONUW-style parser."
        )
    )

    parser.add_argument(
        "--game-log",
        required=True,
        help=(
            "Path to an existing game_log.json."
        ),
    )

    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Runtime YAML containing parser "
            "and backend settings."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Destination comparison report JSON."
        ),
    )

    parser.add_argument(
        "--env-file",
        default=".env",
        help=(
            "Environment file containing API keys."
        ),
    )

    parser.add_argument(
        "--parser-model",
        default=None,
        help=(
            "Optional parser model override."
        ),
    )

    return parser


def main() -> int:
    args = (
        build_arg_parser()
        .parse_args()
    )

    result = run_reparse(
        ReparseConfig(
            game_log_path=(
                args.game_log
            ),
            runtime_config_path=(
                args.config
            ),
            output_path=(
                args.output
            ),
            env_file=(
                args.env_file
            ),
            parser_model=(
                args.parser_model
            ),
        )
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
