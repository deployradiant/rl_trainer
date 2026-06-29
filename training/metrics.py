"""Trajectory metric helpers."""

import numpy as np


def ade(pred: np.ndarray, target: np.ndarray) -> float:
    """Average Displacement Error over the future window."""
    return float(np.mean(np.linalg.norm(pred - target, axis=-1)))


def fde(pred: np.ndarray, target: np.ndarray) -> float:
    """Final Displacement Error at the last timestep."""
    return float(np.linalg.norm(pred[-1] - target[-1]))
