"""Simple trajectory baselines for player movement forecasting."""

import torch

from models.kinematics import constant_velocity_future


def constant_position(player_bbox: torch.Tensor, horizon: int) -> torch.Tensor:
    """Repeat the last observed center for every future frame."""
    current_xy = player_bbox[:, -1, :2]
    return current_xy.unsqueeze(1).expand(-1, horizon, -1)


def constant_velocity(player_bbox: torch.Tensor, horizon: int) -> torch.Tensor:
    """Extrapolate from the average input-window velocity."""
    return constant_velocity_future(player_bbox, horizon)


def batch_ade(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample average displacement error."""
    return (pred - target).norm(dim=-1).mean(dim=-1)


def batch_fde(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample final displacement error."""
    return (pred[:, -1] - target[:, -1]).norm(dim=-1)
