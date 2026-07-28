import torch
from torch import nn


class FixedTwdCostProvider(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("lambda_pp", torch.tensor(0.0))
        self.register_buffer("lambda_pn", torch.tensor(1.0))
        self.register_buffer("lambda_bp", torch.tensor(0.25))
        self.register_buffer("lambda_bn", torch.tensor(0.25))
        self.register_buffer("lambda_np", torch.tensor(1.0))
        self.register_buffer("lambda_nn", torch.tensor(0.0))

    def forward(self, wolf_prob, context=None):
        return {
            name: getattr(self, name).to(
                device=wolf_prob.device,
                dtype=wolf_prob.dtype,
            )
            for name in (
                "lambda_pp",
                "lambda_pn",
                "lambda_bp",
                "lambda_bn",
                "lambda_np",
                "lambda_nn",
            )
        }


class TWDRiskLayer(nn.Module):
    def __init__(self, tau: float = 0.7, cost_provider: nn.Module | None = None):
        super().__init__()
        self.tau = tau
        self.cost_provider = (
            FixedTwdCostProvider()
            if cost_provider is None
            else cost_provider
        )

    def forward(self, wolf_prob, context=None):
        if self.tau <= 0:
            raise ValueError("tau must be greater than zero")

        r = wolf_prob.clamp(1e-5, 1 - 1e-5)
        costs = self.cost_provider(wolf_prob, context=context)

        risk_pos = costs["lambda_pp"] * r + costs["lambda_pn"] * (1 - r)
        risk_bnd = costs["lambda_bp"] * r + costs["lambda_bn"] * (1 - r)
        risk_neg = costs["lambda_np"] * r + costs["lambda_nn"] * (1 - r)

        risks = torch.stack([risk_pos, risk_bnd, risk_neg], dim=-1)
        region_probs = torch.softmax(-risks / self.tau, dim=-1)
        hard_region = risks.argmin(dim=-1)

        return {
            "region_probs": region_probs,
            "risks": risks,
            "hard_region": hard_region,
            "costs": costs,
        }
