"""Action chunk head for bounded trajectory-residual prediction."""

import torch
import torch.nn as nn


class ActionChunkHead(nn.Module):
    """Predict a full future xy residual chunk from visual and player state."""

    def __init__(
        self,
        state_dim: int,
        horizon: int,
        step_dim: int = 32,
        hidden_dim: int = 256,
        max_relative_xy: float = 0.25,
    ):
        super().__init__()
        self.horizon = horizon
        self.max_relative_xy = max_relative_xy
        self.step_embedding = nn.Embedding(horizon, step_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim + step_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, visual_proj: torch.Tensor, player_bbox: torch.Tensor) -> torch.Tensor:
        """Return bounded future xy residuals shaped [B, horizon, 2]."""
        batch_size = visual_proj.shape[0]
        bbox_flat = player_bbox.reshape(batch_size, -1)
        state = torch.cat([visual_proj, bbox_flat], dim=-1)

        step_ids = torch.arange(self.horizon, dtype=torch.long, device=visual_proj.device)
        step_features = self.step_embedding(step_ids).unsqueeze(0).expand(batch_size, -1, -1)
        state_features = state.unsqueeze(1).expand(-1, self.horizon, -1)
        raw_relative_xy = self.net(torch.cat([state_features, step_features], dim=-1))
        return torch.tanh(raw_relative_xy) * self.max_relative_xy
