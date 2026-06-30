import unittest

import torch
import torch.nn as nn

from models.flow_matching import FlowMatchingChunkHead
from models.qwen_football import QwenFootballModel


class FlowMatchingTests(unittest.TestCase):
    def test_flow_head_returns_velocity_field(self):
        head = FlowMatchingChunkHead(state_dim=16 + 3 * 12, horizon=4)
        visual_proj = torch.randn(2, 16)
        bbox = torch.randn(2, 3, 4)
        residual_state = torch.randn(2, 4, 2)
        t = torch.tensor([0.25, 0.75])

        velocity = head(visual_proj, bbox, residual_state, t)

        self.assertEqual(tuple(velocity.shape), (2, 4, 2))
        self.assertTrue(torch.isfinite(velocity).all())

    def test_qwen_flow_matching_outputs_build_target_velocity(self):
        model = QwenFootballModel.__new__(QwenFootballModel)
        nn.Module.__init__(model)
        model.horizon = 4
        model.flow_steps = 2
        model.visual_proj = nn.Identity()
        model.flow_head = FlowMatchingChunkHead(state_dim=16 + 3 * 12, horizon=4)

        visual_features = torch.randn(1, 16)
        bbox = torch.tensor([[
            [0.10, 0.20, 0.05, 0.10],
            [0.12, 0.21, 0.05, 0.10],
            [0.14, 0.22, 0.05, 0.10],
        ]])
        future = torch.tensor([[
            [0.17, 0.25],
            [0.20, 0.28],
            [0.23, 0.32],
            [0.27, 0.36],
        ]])
        noise = torch.zeros_like(future)
        t = torch.tensor([0.5])

        outputs = QwenFootballModel.flow_matching_outputs(
            model,
            visual_features=visual_features,
            player_bbox=bbox,
            future_xy=future,
            noise=noise,
            t=t,
        )

        target_residual = future - QwenFootballModel._constant_velocity_future(model, bbox)
        self.assertTrue(torch.allclose(outputs["target_velocity"], target_residual))
        self.assertTrue(torch.allclose(outputs["residual_state"], target_residual * 0.5))
        self.assertEqual(tuple(outputs["pred_velocity"].shape), (1, 4, 2))

    def test_qwen_flow_rollout_returns_future_xy(self):
        model = QwenFootballModel.__new__(QwenFootballModel)
        nn.Module.__init__(model)
        model.horizon = 4
        model.flow_steps = 2
        model.visual_proj = nn.Identity()
        model.flow_head = FlowMatchingChunkHead(state_dim=16 + 3 * 12, horizon=4)

        visual_features = torch.randn(1, 16)
        bbox = torch.tensor([[
            [0.10, 0.20, 0.05, 0.10],
            [0.12, 0.21, 0.05, 0.10],
            [0.14, 0.22, 0.05, 0.10],
        ]])

        outputs = QwenFootballModel.predict_flow_from_visual_features(
            model,
            visual_features=visual_features,
            player_bbox=bbox,
            return_outputs=True,
        )

        self.assertEqual(tuple(outputs["future_xy"].shape), (1, 4, 2))
        self.assertEqual(tuple(outputs["residual_xy"].shape), (1, 4, 2))
        self.assertTrue(torch.isfinite(outputs["future_xy"]).all())


if __name__ == "__main__":
    unittest.main()
