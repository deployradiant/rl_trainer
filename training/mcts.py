"""MCTS over the discrete football action codebook."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


@dataclass
class SearchNode:
    step: int
    bbox_state: torch.Tensor
    current_xy: torch.Tensor
    future_points: list[torch.Tensor]
    actions: list[int]
    priors: dict[int, float] = field(default_factory=dict)
    children: dict[int, "SearchNode"] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def expanded(self) -> bool:
        return bool(self.priors)

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def trajectory_error(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    diff = pred - target
    distances = diff.norm(dim=-1)
    return float(distances.mean().item()), float(distances[-1].item())


class TrajectoryMCTS:
    """PUCT search using policy logits as priors and target trajectory as reward.

    The reward is intentionally oracle-supervised for the current overfit gate.
    This proves the search/dynamics contract before we introduce a learned value
    or task reward.
    """

    def __init__(
        self,
        model: nn.Module,
        visual_features: torch.Tensor,
        player_bbox: torch.Tensor,
        target_xy: torch.Tensor,
        simulations: int = 128,
        top_k: int = 8,
        c_puct: float = 1.5,
        fde_weight: float = 0.5,
    ):
        if player_bbox.shape[0] != 1:
            raise ValueError("TrajectoryMCTS currently expects batch size 1")
        self.model = model
        self.player_bbox = player_bbox
        self.target_xy = target_xy.squeeze(0)
        self.simulations = simulations
        self.top_k = top_k
        self.c_puct = c_puct
        self.fde_weight = fde_weight
        self.horizon = int(model.horizon)
        self.action_stride = int(model.action_stride)
        self.rollout_steps = int(model.rollout_steps)
        self.visual_proj = model.visual_proj(visual_features.reshape(1, -1))

    def run(self) -> dict:
        root = SearchNode(
            step=0,
            bbox_state=self.player_bbox,
            current_xy=self.player_bbox[:, -1, :2],
            future_points=[],
            actions=[],
        )
        self._expand(root)

        for _ in range(self.simulations):
            leaf, path = self._select_leaf(root)
            if not self._is_terminal(leaf):
                self._expand(leaf)
            value = self._evaluate(leaf)
            for node in path:
                node.visits += 1
                node.value_sum += value

        actions, future_xy = self._best_rollout(root)
        ade, fde = trajectory_error(future_xy, self.target_xy)
        return {
            "future_xy": future_xy,
            "actions": actions,
            "ADE": ade,
            "FDE": fde,
            "root_visits": root.visits,
            "root_children": {
                str(action): {
                    "visits": child.visits,
                    "value": child.value,
                    "prior": root.priors.get(action, 0.0),
                }
                for action, child in sorted(root.children.items())
            },
        }

    def _select_leaf(self, root: SearchNode) -> tuple[SearchNode, list[SearchNode]]:
        node = root
        path = [node]
        while node.expanded and not self._is_terminal(node):
            action = self._select_action(node)
            if action not in node.children:
                node.children[action] = self._transition(node, action)
                node = node.children[action]
                path.append(node)
                break
            node = node.children[action]
            path.append(node)
        return node, path

    def _select_action(self, node: SearchNode) -> int:
        total_visits = max(1, node.visits)
        best_action = None
        best_score = -float("inf")
        for action, prior in node.priors.items():
            child = node.children.get(action)
            child_visits = child.visits if child else 0
            child_value = child.value if child else 0.0
            exploration = self.c_puct * prior * np.sqrt(total_visits) / (1 + child_visits)
            score = child_value + exploration
            if score > best_score:
                best_score = score
                best_action = action
        if best_action is None:
            raise RuntimeError("Cannot select from an unexpanded node")
        return best_action

    def _expand(self, node: SearchNode) -> None:
        if self._is_terminal(node):
            return
        logits = self.model._policy_logits(self.visual_proj, node.bbox_state, node.step)
        probs = logits.softmax(dim=-1).squeeze(0)
        k = min(self.top_k, probs.numel())
        values, indices = torch.topk(probs, k=k)
        prior_sum = values.sum().clamp_min(1e-8)
        node.priors = {
            int(action.item()): float((prior / prior_sum).item())
            for action, prior in zip(indices, values)
        }

    def _transition(self, node: SearchNode, action: int) -> SearchNode:
        velocity = self.model._recent_velocity(node.bbox_state[:, :, :2])
        candidates = self.model._candidate_velocities(velocity)
        next_velocity = candidates[:, action]
        current_xy = node.current_xy
        future_points = list(node.future_points)
        frames_this_step = min(self.action_stride, self.horizon - len(future_points))
        for _ in range(frames_this_step):
            current_xy = current_xy + next_velocity
            future_points.append(current_xy.squeeze(0))

        next_bbox = torch.cat([current_xy, node.bbox_state[:, -1, 2:]], dim=-1).unsqueeze(1)
        bbox_state = torch.cat([node.bbox_state[:, 1:], next_bbox], dim=1)
        return SearchNode(
            step=node.step + 1,
            bbox_state=bbox_state,
            current_xy=current_xy,
            future_points=future_points,
            actions=node.actions + [action],
        )

    def _evaluate(self, node: SearchNode) -> float:
        _, future_xy = self._complete_with_policy(node)
        ade, fde = trajectory_error(future_xy, self.target_xy)
        return -(ade + self.fde_weight * fde)

    def _complete_with_policy(self, node: SearchNode) -> tuple[list[int], torch.Tensor]:
        actions = list(node.actions)
        bbox_state = node.bbox_state
        current_xy = node.current_xy
        future_points = list(node.future_points)
        step = node.step
        while len(future_points) < self.horizon:
            logits = self.model._policy_logits(self.visual_proj, bbox_state, step)
            action = int(logits.argmax(dim=-1).item())
            actions.append(action)
            velocity = self.model._recent_velocity(bbox_state[:, :, :2])
            candidates = self.model._candidate_velocities(velocity)
            next_velocity = candidates[:, action]
            frames_this_step = min(self.action_stride, self.horizon - len(future_points))
            for _ in range(frames_this_step):
                current_xy = current_xy + next_velocity
                future_points.append(current_xy.squeeze(0))

            next_bbox = torch.cat([current_xy, bbox_state[:, -1, 2:]], dim=-1).unsqueeze(1)
            bbox_state = torch.cat([bbox_state[:, 1:], next_bbox], dim=1)
            step += 1
        return actions, torch.stack(future_points)

    def _best_rollout(self, root: SearchNode) -> tuple[list[int], torch.Tensor]:
        node = root
        while node.children and not self._is_terminal(node):
            node = max(node.children.values(), key=lambda child: (child.visits, child.value))
        return self._complete_with_policy(node)

    def _is_terminal(self, node: SearchNode) -> bool:
        return node.step >= self.rollout_steps or len(node.future_points) >= self.horizon


def run_trajectory_mcts(
    model: nn.Module,
    visual_features: torch.Tensor,
    player_bbox: torch.Tensor,
    target_xy: torch.Tensor,
    simulations: int,
    top_k: int,
    c_puct: float,
    fde_weight: float,
) -> dict:
    search = TrajectoryMCTS(
        model=model,
        visual_features=visual_features,
        player_bbox=player_bbox,
        target_xy=target_xy,
        simulations=simulations,
        top_k=top_k,
        c_puct=c_puct,
        fde_weight=fde_weight,
    )
    return search.run()
