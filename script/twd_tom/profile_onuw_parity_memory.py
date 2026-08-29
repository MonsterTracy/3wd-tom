"""Measure one real parity training step after pilot length review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from werewolf.models.twd_tom.onuw_parity_audit import sequence_capacity_audit
from werewolf.models.twd_tom.onuw_parity_dataset import (
    OnuwParityGameDataset,
    collate_onuw_parity_games,
)
from werewolf.models.twd_tom.onuw_parity_model import (
    OnuwParityBeliefModel,
    OnuwParityModelConfig,
)
from werewolf.models.twd_tom.onuw_parity_objective import (
    onuw_parity_belief_objective,
)


def _load_games(path: Path) -> list[dict]:
    games = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not a game object")
            games.append(value)
    if not games:
        raise ValueError("pilot JSONL contains no games")
    return games


def profile(*, games: list[dict], batch_size: int) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for an actual VRAM measurement")
    if isinstance(batch_size, bool) or not 1 <= batch_size <= len(games):
        raise ValueError("batch_size must be in [1, number of pilot games]")
    selected = games[:batch_size]
    if len({game["content_profile"] for game in selected}) != 1:
        raise ValueError("one memory probe requires one content profile")
    if len({game["modality_profile"] for game in selected}) != 1:
        raise ValueError("one memory probe requires one modality profile")
    capacity = sequence_capacity_audit(selected)
    dataset = OnuwParityGameDataset(selected)
    batch = collate_onuw_parity_games(
        [dataset[index] for index in range(len(dataset))]
    )
    device = torch.device("cuda")
    model = OnuwParityBeliefModel(
        OnuwParityModelConfig(
            max_positions=capacity["sequence_length_max"],
            content_profile=selected[0]["content_profile"],
            modality_profile=selected[0]["modality_profile"],
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    tensor_batch = {
        key: value.to(device)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    logits = model(
        **{
            key: tensor_batch[key]
            for key in (
                "subject_ids",
                "action_ids",
                "object_ids",
                "token_type_ids",
                "face_ids",
                "tone_ids",
                "phase_ids",
                "day_values",
                "token_attention_mask",
                "query_positions",
                "query_valid_mask",
                "observer_alive_mask",
            )
        }
    )
    objective = onuw_parity_belief_objective(
        logits,
        tensor_batch["belief_targets"],
        tensor_batch["query_valid_mask"],
        tensor_batch["observer_alive_mask"],
    )
    objective["loss"].backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    return {
        "batch_size_games": batch_size,
        "token_shape": list(tensor_batch["token_attention_mask"].shape),
        "query_shape": list(tensor_batch["query_valid_mask"].shape),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "loss": float(objective["loss"].detach()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "device_name": torch.cuda.get_device_name(device),
        "sequence_capacity": capacity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("games_jsonl", type=Path)
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            profile(
                games=_load_games(args.games_jsonl),
                batch_size=args.batch_size,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
