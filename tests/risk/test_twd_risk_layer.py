import unittest

import torch
from torch import nn

from werewolf.models.risk.twd_risk_layer import (
    FixedTwdCostProvider,
    TWDRiskLayer,
)


POS = 0
BND = 1
NEG = 2
COST_NAMES = (
    "lambda_pp",
    "lambda_pn",
    "lambda_bp",
    "lambda_bn",
    "lambda_np",
    "lambda_nn",
)


class DummyCostProvider(nn.Module):
    def __init__(self):
        super().__init__()
        self.called = False
        self.received_context = None

    def forward(self, wolf_prob, context=None):
        self.called = True
        self.received_context = context
        zeros = torch.zeros_like(wolf_prob)
        ones = torch.ones_like(wolf_prob)
        return {
            "lambda_pp": zeros,
            "lambda_pn": ones,
            "lambda_bp": torch.full_like(wolf_prob, 0.25),
            "lambda_bn": torch.full_like(wolf_prob, 0.25),
            "lambda_np": ones,
            "lambda_nn": zeros,
        }


class TWDRiskLayerTest(unittest.TestCase):
    def test_low_probability_selects_neg(self):
        output = TWDRiskLayer()(torch.tensor(0.1))

        self.assertEqual(output["risks"].argmin().item(), NEG)
        self.assertEqual(output["region_probs"].argmax().item(), NEG)
        self.assertEqual(output["hard_region"].item(), NEG)

    def test_middle_probability_selects_bnd(self):
        output = TWDRiskLayer()(torch.tensor(0.5))

        self.assertEqual(output["risks"].argmin().item(), BND)
        self.assertEqual(output["region_probs"].argmax().item(), BND)
        self.assertEqual(output["hard_region"].item(), BND)

    def test_high_probability_selects_pos(self):
        output = TWDRiskLayer()(torch.tensor(0.9))

        self.assertEqual(output["risks"].argmin().item(), POS)
        self.assertEqual(output["region_probs"].argmax().item(), POS)
        self.assertEqual(output["hard_region"].item(), POS)

    def test_two_dimensional_input_shapes(self):
        output = TWDRiskLayer()(torch.full((2, 7), 0.5))

        self.assertEqual(output["region_probs"].shape, (2, 7, 3))
        self.assertEqual(output["risks"].shape, (2, 7, 3))
        self.assertEqual(output["hard_region"].shape, (2, 7))

    def test_three_dimensional_input_shapes(self):
        output = TWDRiskLayer()(torch.full((2, 3, 7), 0.5))

        self.assertEqual(output["region_probs"].shape, (2, 3, 7, 3))
        self.assertEqual(output["risks"].shape, (2, 3, 7, 3))
        self.assertEqual(output["hard_region"].shape, (2, 3, 7))

    def test_region_probabilities_sum_to_one(self):
        output = TWDRiskLayer()(torch.tensor([[0.1, 0.5, 0.9]]))

        torch.testing.assert_close(
            output["region_probs"].sum(dim=-1),
            torch.ones(1, 3),
        )

    def test_boundary_probabilities_produce_finite_outputs(self):
        output = TWDRiskLayer()(torch.tensor([0.0, 1.0]))

        self.assertTrue(torch.isfinite(output["risks"]).all())
        self.assertTrue(torch.isfinite(output["region_probs"]).all())

    def test_fixed_costs_are_scalar_buffers_not_parameters(self):
        provider = FixedTwdCostProvider()
        buffers = dict(provider.named_buffers())

        self.assertEqual(list(provider.parameters()), [])
        self.assertEqual(set(buffers), set(COST_NAMES))
        self.assertTrue(all(value.ndim == 0 for value in buffers.values()))

    def test_fixed_cost_values(self):
        costs = FixedTwdCostProvider()(torch.tensor(0.5))

        self.assertEqual(costs["lambda_pp"].item(), 0.0)
        self.assertEqual(costs["lambda_pn"].item(), 1.0)
        self.assertEqual(costs["lambda_bp"].item(), 0.25)
        self.assertEqual(costs["lambda_bn"].item(), 0.25)
        self.assertEqual(costs["lambda_np"].item(), 1.0)
        self.assertEqual(costs["lambda_nn"].item(), 0.0)

    def test_fixed_costs_follow_input_dtype_and_device(self):
        wolf_prob = torch.tensor([0.1, 0.9], dtype=torch.float64)

        costs = FixedTwdCostProvider()(wolf_prob)

        for value in costs.values():
            self.assertEqual(value.dtype, torch.float64)
            self.assertEqual(value.device, wolf_prob.device)

    def test_non_positive_initial_tau_raises(self):
        for tau in (0.0, -0.1):
            with self.subTest(tau=tau):
                with self.assertRaises(ValueError):
                    TWDRiskLayer(tau=tau)(torch.tensor(0.5))

    def test_dynamically_assigned_zero_tau_raises(self):
        layer = TWDRiskLayer()
        layer.tau = 0

        with self.assertRaises(ValueError):
            layer(torch.tensor(0.5))

    def test_forward_supports_context_none(self):
        output = TWDRiskLayer()(torch.tensor([0.5]), context=None)

        self.assertEqual(output["hard_region"].shape, (1,))

    def test_custom_cost_provider_is_called(self):
        provider = DummyCostProvider()
        context = {"phase": "speech"}
        wolf_prob = torch.full((2, 7), 0.5)

        output = TWDRiskLayer(cost_provider=provider)(
            wolf_prob,
            context=context,
        )

        self.assertTrue(provider.called)
        self.assertIs(provider.received_context, context)
        self.assertTrue(
            all(value.shape == wolf_prob.shape for value in output["costs"].values())
        )

    def test_probability_argmax_matches_risk_argmin(self):
        wolf_prob = torch.tensor([[0.1, 0.5, 0.9]])
        output = TWDRiskLayer()(wolf_prob)

        self.assertTrue(
            torch.equal(
                output["region_probs"].argmax(dim=-1),
                output["risks"].argmin(dim=-1),
            )
        )


if __name__ == "__main__":
    unittest.main()
