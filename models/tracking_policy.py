"""Tracking-only action chunk baseline."""

import torch
import torch.nn as nn

from models.kinematics import KINEMATIC_FEATURE_DIM, constant_velocity_future, kinematic_features


class TrackingChunkBaseline(nn.Module):
    """Predict future trajectory chunks from bbox history only."""

    def __init__(
        self,
        horizon: int,
        hidden_dim: int = 128,
        step_dim: int = 32,
        max_relative_xy: float = 0.25,
    ):
        super().__init__()
        self.horizon = horizon
        self.max_relative_xy = max_relative_xy
        self.encoder = nn.GRU(
            input_size=KINEMATIC_FEATURE_DIM,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.step_embedding = nn.Embedding(horizon, step_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + step_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        final = self.decoder[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        player_bbox: torch.Tensor,
        return_outputs: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        features = kinematic_features(player_bbox)
        _, hidden = self.encoder(features)
        state = hidden[-1]
        batch_size = player_bbox.shape[0]
        step_ids = torch.arange(self.horizon, dtype=torch.long, device=player_bbox.device)
        step_features = self.step_embedding(step_ids).unsqueeze(0).expand(batch_size, -1, -1)
        state_features = state.unsqueeze(1).expand(-1, self.horizon, -1)
        raw_residual = self.decoder(torch.cat([state_features, step_features], dim=-1))
        residual_xy = torch.tanh(raw_residual) * self.max_relative_xy
        base_xy = constant_velocity_future(player_bbox, self.horizon)
        future_xy = base_xy + residual_xy
        if return_outputs:
            return {"future_xy": future_xy, "residual_xy": residual_xy, "base_xy": base_xy}
        return future_xy
