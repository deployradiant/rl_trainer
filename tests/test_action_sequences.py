import unittest
from math import ceil

import torch

from models.qwen_football import QwenFootballModel, build_relative_action_codebook
from training.mcts import SearchNode, TrajectoryMCTS


def make_codebook_model(horizon: int = 10, action_stride: int = 5) -> QwenFootballModel:
    model = QwenFootballModel.__new__(QwenFootballModel)
    model.horizon = horizon
    model.action_stride = action_stride
    model.rollout_steps = ceil(horizon / action_stride)
    model.speed_floor = 0.0015
    model.action_codebook = build_relative_action_codebook()
    return model


class ActionSequenceTests(unittest.TestCase):
    def test_constant_velocity_labels_stay_consistent_across_chunks(self):
        model = make_codebook_model()
        bbox = torch.tensor([[
            [0.10, 0.20, 0.05, 0.10],
            [0.12, 0.20, 0.05, 0.10],
            [0.14, 0.20, 0.05, 0.10],
        ]])
        future = torch.tensor([[
            [0.16, 0.20],
            [0.18, 0.20],
            [0.20, 0.20],
            [0.22, 0.20],
            [0.24, 0.20],
            [0.26, 0.20],
            [0.28, 0.20],
            [0.30, 0.20],
            [0.32, 0.20],
            [0.34, 0.20],
        ]])

        labels = QwenFootballModel.action_labels(model, bbox, future)
        expected_action = 4 * 5 + 2  # heading 0 degrees, speed multiplier 1.0

        self.assertEqual(labels.tolist(), [[expected_action, expected_action]])

    def test_value_targets_match_constant_velocity_labels_across_chunks(self):
        model = make_codebook_model()
        bbox = torch.tensor([[
            [0.10, 0.20, 0.05, 0.10],
            [0.12, 0.20, 0.05, 0.10],
            [0.14, 0.20, 0.05, 0.10],
        ]])
        future = torch.tensor([[
            [0.16, 0.20],
            [0.18, 0.20],
            [0.20, 0.20],
            [0.22, 0.20],
            [0.24, 0.20],
            [0.26, 0.20],
            [0.28, 0.20],
            [0.30, 0.20],
            [0.32, 0.20],
            [0.34, 0.20],
        ]])

        labels = QwenFootballModel.action_labels(model, bbox, future)
        targets = QwenFootballModel.action_value_targets(model, bbox, future)

        self.assertTrue(torch.allclose(targets.max(dim=-1).values, torch.zeros(1, 2), atol=1e-6))
        self.assertEqual(targets.argmax(dim=-1).tolist(), labels.tolist())

    def test_mcts_transition_preserves_per_frame_velocity_history(self):
        model = make_codebook_model()
        bbox = torch.tensor([[
            [0.10, 0.20, 0.05, 0.10],
            [0.12, 0.20, 0.05, 0.10],
            [0.14, 0.20, 0.05, 0.10],
        ]])

        class DummyModel:
            horizon = model.horizon
            action_stride = model.action_stride
            rollout_steps = model.rollout_steps
            speed_floor = model.speed_floor
            action_codebook = model.action_codebook
            _recent_velocity = QwenFootballModel._recent_velocity
            _candidate_velocities = QwenFootballModel._candidate_velocities
            _append_xy_history = QwenFootballModel._append_xy_history

            def visual_proj(self, visual_features):
                return visual_features

            def _policy_logits(self, visual_proj, bbox_state, step):
                return torch.zeros((1, self.action_codebook.shape[0]))

            def _action_values(self, visual_proj, bbox_state, step):
                return torch.zeros((1, self.action_codebook.shape[0]))

        search = TrajectoryMCTS(DummyModel(), torch.zeros((1, 1)), bbox, simulations=1)
        node = SearchNode(
            step=0,
            bbox_state=bbox,
            current_xy=bbox[:, -1, :2],
            future_points=[],
            actions=[],
        )
        after_two_chunks = search._transition(
            search._transition(node, 22),
            22,
        )

        recent_velocity = QwenFootballModel._recent_velocity(model, after_two_chunks.bbox_state[:, :, :2])
        self.assertTrue(torch.allclose(recent_velocity, torch.tensor([[0.02, 0.00]]), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
