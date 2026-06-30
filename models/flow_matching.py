"""Conditional flow-matching head for trajectory residual chunks."""

import torch
import torch.nn as nn

from models.kinematics import kinematic_features


class FlowMatchingChunkHead(nn.Module):
    """Predict the flow velocity for a noised future residual trajectory."""

    def __init__(
        self,
        state_dim: int,
        horizon: int,
        step_dim: int = 32,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.horizon = horizon
        self.step_embedding = nn.Embedding(horizon, step_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim + step_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        visual_proj: torch.Tensor,
        player_bbox: torch.Tensor,
        residual_state: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Return flow velocity shaped [B, horizon, 2].

        Args:
            visual_proj: [B, D] projected Qwen visual features.
            player_bbox: [B, T, 4] tracking history.
            residual_state: [B, horizon, 2] interpolated residual trajectory.
            t: [B] or [B, 1, 1] interpolation time in [0, 1].
        """
        batch_size = visual_proj.shape[0]
        player_features = kinematic_features(player_bbox).reshape(batch_size, -1)
        state = torch.cat([visual_proj, player_features], dim=-1)

        if t.ndim == 1:
            t = t.view(batch_size, 1, 1)
        elif t.ndim == 2:
            t = t.view(batch_size, 1, 1)
        t_features = t.expand(-1, self.horizon, 1)

        step_ids = torch.arange(self.horizon, dtype=torch.long, device=visual_proj.device)
        step_features = self.step_embedding(step_ids).unsqueeze(0).expand(batch_size, -1, -1)
        state_features = state.unsqueeze(1).expand(-1, self.horizon, -1)
        return self.net(torch.cat([state_features, step_features, residual_state, t_features], dim=-1))
