"""Summarize an audit-only same-observation Reporter A/B sidecar."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


REPORTERS = ("qwen", "deepseek")
PRIMARY_ROLES = ("Werewolf", "Seer", "Witch", "Villager")


def _empty_counts() -> dict:
    return {
        reporter: {"valid": 0, "attempts": 0}
        for reporter in REPORTERS
    }


def summarize_reporter_ab(path: str | Path) -> dict:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at line {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL row at line {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise TypeError(f"row {line_number} must be a mapping")
            rows.append(row)

    overall = _empty_counts()
    by_role = {role: _empty_counts() for role in PRIMARY_ROLES}
    errors = {reporter: Counter() for reporter in REPORTERS}
    set_counts = {
        reporter: {"empty": 0, "nonempty": 0}
        for reporter in REPORTERS
    }
    paired = Counter()

    for row in rows:
        role = row.get("observer_role")
        if not isinstance(role, str) or not role:
            raise ValueError("observer_role must be non-empty text")
        role_counts = by_role.setdefault(role, _empty_counts())
        validity = {}
        for reporter in REPORTERS:
            result = row.get(reporter)
            if not isinstance(result, dict):
                raise TypeError(f"{reporter} result must be a mapping")
            valid = result.get("valid") is True
            validity[reporter] = valid
            overall[reporter]["attempts"] += 1
            role_counts[reporter]["attempts"] += 1
            if valid:
                overall[reporter]["valid"] += 1
                role_counts[reporter]["valid"] += 1
                suspected = result.get("suspected_werewolves")
                if not isinstance(suspected, list):
                    raise TypeError(
                        f"valid {reporter} result requires a suspicion list"
                    )
                label = "empty" if not suspected else "nonempty"
                set_counts[reporter][label] += 1
            else:
                error = result.get("error")
                errors[reporter][str(error)] += 1

        if validity == {"qwen": True, "deepseek": True}:
            paired["both_valid"] += 1
        elif validity == {"qwen": True, "deepseek": False}:
            paired["qwen_only_valid"] += 1
        elif validity == {"qwen": False, "deepseek": True}:
            paired["deepseek_only_valid"] += 1
        else:
            paired["both_invalid"] += 1

    return {
        "overall": overall,
        "by_observer_role": by_role,
        "error_type_counts": {
            reporter: dict(sorted(counter.items()))
            for reporter, counter in errors.items()
        },
        "valid_set_counts": set_counts,
        "paired_checkpoint_counts": {
            name: paired[name]
            for name in (
                "both_valid",
                "qwen_only_valid",
                "deepseek_only_valid",
                "both_invalid",
            )
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("sidecar_path", type=Path)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(
        json.dumps(
            summarize_reporter_ab(args.sidecar_path),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
