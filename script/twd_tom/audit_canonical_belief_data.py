"""Audit canonical tom-v2 belief data before dataset materialization."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from werewolf.models.twd_tom.action_features import PublicEventFeatureBuilder
from werewolf.models.twd_tom.dataset import TWDToMDataset
from werewolf.models.twd_tom.public_events import structured_event_tokens
from werewolf.models.twd_tom.samples import SAMPLE_SCHEMA_VERSION
from werewolf.models.twd_tom.schema import LABEL_PROMPT_VERSION, LABEL_PROVENANCE
from werewolf.trajectory import canonical_digest, canonical_json


AUDIT_SCHEMA_VERSION = "classic7_canonical_belief_data_audit_v1"
BELIEF_SNAPSHOTS_FILENAME = "belief_snapshots.jsonl"


def _positive_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL line at {path}:{line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise TypeError(
                    f"JSONL record must be an object: {path}:{line_number}"
                )
            records.append(record)
    if not records:
        raise ValueError(f"canonical belief file cannot be empty: {path}")
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(canonical_json(value) + "\n")


def audit_canonical_belief_data(
    *,
    canonical_root: str | Path,
    max_seq_len: int = 256,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fail closed on invalid labels and report sequence/support statistics."""

    max_seq_len = _positive_integer(max_seq_len, field_name="max_seq_len")
    canonical_root = Path(canonical_root).resolve()
    games_root = canonical_root / "games"
    if not games_root.is_dir():
        raise FileNotFoundError(f"canonical games directory not found: {games_root}")
    paths = sorted(games_root.glob(f"*/{BELIEF_SNAPSHOTS_FILENAME}"))
    if not paths:
        raise FileNotFoundError(
            f"no canonical belief snapshots found under: {games_root}"
        )

    records_by_game: dict[str, list[dict[str, Any]]] = {}
    source_files: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    support_size_counts: Counter[int] = Counter()
    failed_reports: list[dict[str, Any]] = []

    for path in paths:
        records = _load_jsonl(path)
        game_ids = {record.get("game_id") for record in records}
        if len(game_ids) != 1:
            raise ValueError(
                "one canonical belief file must contain exactly one game_id: "
                f"{path}"
            )
        game_id = next(iter(game_ids))
        if not isinstance(game_id, str) or not game_id.strip():
            raise ValueError(f"canonical game_id must be non-empty text: {path}")
        if game_id in records_by_game:
            raise ValueError(f"duplicate canonical game_id: {game_id}")
        step_indices = [record.get("step_idx") for record in records]
        if any(
            isinstance(step, bool) or not isinstance(step, int)
            for step in step_indices
        ):
            raise TypeError(f"every canonical step_idx must be an integer: {path}")
        if len(step_indices) != len(set(step_indices)):
            raise ValueError(f"duplicate canonical (game_id, step_idx): {game_id}")
        records.sort(key=lambda record: record["step_idx"])

        for record in records:
            if record.get("schema_version") != SAMPLE_SCHEMA_VERSION:
                raise ValueError("canonical raw sample schema version mismatch")
            if record.get("label_prompt_version") != LABEL_PROMPT_VERSION:
                raise ValueError("canonical label prompt version mismatch")
            if record.get("label_provenance") != LABEL_PROVENANCE:
                raise ValueError("canonical label provenance mismatch")
            observer_ids = record.get("observer_ids")
            statuses = record.get("belief_status")
            suspicions = record.get("suspected_werewolves")
            errors = record.get("belief_errors")
            if not isinstance(observer_ids, list):
                raise TypeError("canonical observer_ids must be a list")
            if not all(
                isinstance(value, Mapping)
                for value in (statuses, suspicions, errors)
            ):
                raise TypeError("canonical belief row fields must be mappings")
            subjects = {f"player{observer_id}" for observer_id in observer_ids}
            if any(set(value) != subjects for value in (statuses, suspicions, errors)):
                raise ValueError("canonical belief row observer sets differ")
            for subject in sorted(subjects):
                status = statuses[subject]
                status_counts[str(status)] += 1
                if status != "ok":
                    failed_reports.append(
                        {
                            "game_id": game_id,
                            "step_idx": record["step_idx"],
                            "observer": subject,
                            "status": status,
                            "error": errors[subject],
                        }
                    )
                    continue
                if errors[subject] is not None:
                    raise ValueError("status=ok belief report must have a null error")
                suspected = suspicions[subject]
                if not isinstance(suspected, list):
                    raise TypeError(
                        "status=ok suspected_werewolves row must be a list"
                    )
                support_size_counts[len(suspected)] += 1
        records_by_game[game_id] = records
        source_files.append(
            {
                "game_id": game_id,
                "relative_path": str(path.relative_to(canonical_root)),
                "sha256": _sha256(path),
                "snapshot_count": len(records),
            }
        )

    if failed_reports:
        raise ValueError(
            "canonical audit requires status=ok for every alive observer; "
            f"failed_report_count={len(failed_reports)} first={failed_reports[0]}"
        )

    all_records = [
        record
        for game_id in sorted(records_by_game)
        for record in records_by_game[game_id]
    ]
    raw_token_counts = [
        len(
            structured_event_tokens(
                record.get("public_events"),
                record.get("speech_annotations"),
            )
        )
        for record in all_records
    ]
    feature_builder = PublicEventFeatureBuilder(max_seq_len=max_seq_len)
    dataset = TWDToMDataset(all_records, feature_builder=feature_builder)
    retained_token_counts = [
        int(dataset[index]["attention_mask"].sum().item())
        for index in range(len(dataset))
    ]
    truncated_sample_count = sum(
        retained < raw
        for retained, raw in zip(retained_token_counts, raw_token_counts)
    )
    sample_count = len(all_records)
    observer_report_count = sum(status_counts.values())
    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "PASS",
        "canonical_root": str(canonical_root),
        "raw_schema_version": SAMPLE_SCHEMA_VERSION,
        "label_prompt_version": LABEL_PROMPT_VERSION,
        "label_provenance": LABEL_PROVENANCE,
        "max_seq_len": max_seq_len,
        "game_count": len(records_by_game),
        "sample_count": sample_count,
        "observer_report_count": observer_report_count,
        "status_counts": dict(sorted(status_counts.items())),
        "suspicion_support_size_counts": {
            str(size): count
            for size, count in sorted(support_size_counts.items())
        },
        "raw_structured_token_count": {
            "min": min(raw_token_counts),
            "max": max(raw_token_counts),
            "mean": sum(raw_token_counts) / sample_count,
        },
        "retained_structured_token_count": {
            "min": min(retained_token_counts),
            "max": max(retained_token_counts),
            "mean": sum(retained_token_counts) / sample_count,
        },
        "truncated_sample_count": truncated_sample_count,
        "truncated_sample_fraction": truncated_sample_count / sample_count,
        "source_files": source_files,
    }
    report["audit_digest"] = canonical_digest(report)
    if output_path is not None:
        _write_json_new(Path(output_path).resolve(), report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit canonical tom-v2 belief snapshots before materialization."
    )
    parser.add_argument("--canonical-root", required=True)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--output")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    report = audit_canonical_belief_data(
        canonical_root=args.canonical_root,
        max_seq_len=args.max_seq_len,
        output_path=args.output,
    )
    print(canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
