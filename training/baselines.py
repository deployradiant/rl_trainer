"""Simple trajectory baselines for player movement forecasting."""

import torch


def constant_position(player_bbox: torch.Tensor, horizon: int) -> torch.Tensor:
    """Repeat the last observed center for every future frame."""
    current_xy = player_bbox[:, -1, :2]
    return current_xy.unsqueeze(1).expand(-1, horizon, -1)


def constant_velocity(player_bbox: torch.Tensor, horizon: int) -> torch.Tensor:
    """Extrapolate from the average input-window velocity."""
    xy = player_bbox[:, :, :2]
    velocity = (xy[:, -1] - xy[:, 0]) / max(1, xy.shape[1] - 1)
    steps = torch.arange(1, horizon + 1, dtype=xy.dtype, device=xy.device).view(1, horizon, 1)
    return xy[:, -1:].expand(-1, horizon, -1) + velocity.unsqueeze(1) * steps


def batch_ade(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample average displacement error."""
    return (pred - target).norm(dim=-1).mean(dim=-1)


def batch_fde(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample final displacement error."""
    return (pred[:, -1] - target[:, -1]).norm(dim=-1)
