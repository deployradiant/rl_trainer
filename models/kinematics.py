"""Kinematic features shared by Qwen and tracking-only trajectory heads."""

import torch

KINEMATIC_FEATURE_DIM = 12


def temporal_deltas(values: torch.Tensor) -> torch.Tensor:
    """Return first-order temporal deltas with a zero first step."""
    first = torch.zeros_like(values[:, :1])
    return torch.cat([first, values[:, 1:] - values[:, :-1]], dim=1)


def kinematic_features(player_bbox: torch.Tensor) -> torch.Tensor:
    """Build per-frame player state features from bbox history.

    Features are absolute xy, current-relative xy, bbox size, velocity,
    acceleration, speed, and acceleration magnitude.
    """
    xy = player_bbox[:, :, :2]
    wh = player_bbox[:, :, 2:]
    velocity = temporal_deltas(xy)
    acceleration = temporal_deltas(velocity)
    speed = velocity.norm(dim=-1, keepdim=True)
    acceleration_mag = acceleration.norm(dim=-1, keepdim=True)
    relative_xy = xy - xy[:, -1:]
    return torch.cat([
        xy,
        relative_xy,
        wh,
        velocity,
        acceleration,
        speed,
        acceleration_mag,
    ], dim=-1)


def constant_velocity_future(player_bbox: torch.Tensor, horizon: int) -> torch.Tensor:
    """Extrapolate future xy from the average observed velocity."""
    xy = player_bbox[:, :, :2]
    velocity = (xy[:, -1] - xy[:, 0]) / max(1, xy.shape[1] - 1)
    steps = torch.arange(1, horizon + 1, dtype=xy.dtype, device=xy.device).view(1, horizon, 1)
    return xy[:, -1:].expand(-1, horizon, -1) + velocity.unsqueeze(1) * steps
