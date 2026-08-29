"""Run one deterministic forward/backward pass over synthetic parity games."""

from __future__ import annotations

import argparse
import json

import torch

from werewolf.models.twd_tom.onuw_parity_audit import (
    action_only_information_loss_audit,
    sequence_capacity_audit,
)
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
from werewolf.models.twd_tom.onuw_parity_protocol import (
    ONUW_ACTION_ONLY,
    ONUW_AGENT_DECLARED_MULTIMODAL,
)
from werewolf.models.twd_tom.onuw_parity_synthetic import synthetic_parity_games


def run_smoke(*, seed: int = 17) -> dict:
    torch.manual_seed(seed)
    games = synthetic_parity_games()
    dataset = OnuwParityGameDataset(games)
    batch = collate_onuw_parity_games([dataset[index] for index in range(len(dataset))])
    model = OnuwParityBeliefModel(
        OnuwParityModelConfig(
            max_positions=batch["token_attention_mask"].shape[1],
            content_profile=ONUW_ACTION_ONLY,
            modality_profile=ONUW_AGENT_DECLARED_MULTIMODAL,
        )
    )
    model.train()
    feature_names = (
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
    logits = model(**{name: batch[name] for name in feature_names})
    objective = onuw_parity_belief_objective(
        logits,
        batch["belief_targets"],
        batch["query_valid_mask"],
        batch["observer_alive_mask"],
    )
    objective["loss"].backward()
    gradient_is_finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    return {
        "status": "ok" if gradient_is_finite else "invalid_gradient",
        "seed": seed,
        "logits_shape": list(logits.shape),
        "loss": float(objective["loss"].detach()),
        "row_micro_kl": float(objective["row_micro_kl"].detach()),
        "valid_row_count": int(objective["valid_row_count"]),
        "valid_query_count": int(objective["valid_query_count"]),
        "gradient_is_finite": gradient_is_finite,
        "sequence_capacity": sequence_capacity_audit(games),
        "information_loss": [
            action_only_information_loss_audit(game) for game in games
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    print(json.dumps(run_smoke(seed=args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
